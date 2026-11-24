//! `openrag` command-line interface.
//!
//! ## Who a command runs as
//!
//! * **API key** (`OPENRAG_API_KEY`): the command runs as that key's principal with the
//!   same IAM/RLS/Guard checks as the HTTP API (fail-closed).
//! * **Remote** (`OPENRAG_API_URL` + `OPENRAG_API_KEY`): HTTP-backed commands are sent to
//!   that server instead of the local database.
//! * **Local operator** (`DATABASE_URL` + `operator = true` in config.toml, or
//!   `OPENRAG_OPERATOR=1`): off by default. When enabled, the CLI acts as a local operator:
//!   - user-level operations (`memory patch|promote`, `ingest`, `wiki import|export`) run as
//!     the owner of the target resource, so ledger rules (e.g. retracted entries) still apply;
//!   - admin operations (`redact`, `index`, `bench`, `pipeline`, `concept`) run with org-admin
//!     authority, attributed to `OPENRAG_PRINCIPAL_ID` if set, else the org's earliest active
//!     admin, else the resource owner. The attribution is printed in the output;
//!   - org-wide operator commands apply to `OPENRAG_ORG_ID`, or to every org when unset.
//!
//! ## Configuration file
//!
//! `$OPENRAG_CONFIG`, else `$XDG_CONFIG_HOME/openrag/config.toml`, else
//! `~/.config/openrag/config.toml`. Keys: `api_url`, `api_key`, `org_id`, `principal_id`,
//! `database_url`, `mode`, `operator`. Environment variables always win over the file.
//! A file holding `api_key` or `database_url` must not be readable by group/others.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

use axum::http::{Method, StatusCode};
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use openrag_core::{
    AccessScope, AuthLevel, ExternalMirrorBackend, OpenRagError, PgPineconeEngine, PipelineConfig,
    PrincipalContext,
};
use openrag_eval::{PipelinePromoter, PromotePipelineInput};
use openrag_memory::MarkdownSync;
use openrag_pii::PiiEngine;
use openrag_store::{DefaultOutboxExecutor, OutboxWorker, PgPineconeManager, StoreMode, VectorTable};
use serde_json::{json, Map, Value};
use sqlx::Row;
use tokio::net::TcpListener;
use uuid::Uuid;

use crate::api_core::{resolve_bucket_rls, resolve_org_and_dept_paths};
use crate::auth::{authenticate_bearer_token, issue_cli_code, load_principal_context, CliCodeCaps};
use crate::llm_router::neutralize_literal_pii_tokens;
use crate::mcp::{build_multipart, dispatch_as, pct_encode, DispatchBody, StdioOptions};
use crate::state::{non_empty_env, AppState, OpenRagMode};

// ============================================================================
// Command tree
// ============================================================================

#[derive(Debug, Parser)]
#[command(
    name = "openrag",
    version,
    about = "OpenRAG — secure, local-first RAG: server, MCP (16 tools), ingestion and operations",
    long_about = None,
    propagate_version = true,
    after_help = "Identity: api_key (config.toml or OPENRAG_API_KEY) runs as that key's principal; api_url sends HTTP-backed commands to a remote server; local operator mode needs `operator = true` in config.toml (or OPENRAG_OPERATOR=1).\nConfig: $OPENRAG_CONFIG, else ~/.config/openrag/config.toml (api_url, api_key, org_id, principal_id, database_url, mode, operator). Environment variables override the file."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start the local stack (PostgreSQL 17 + OpenRAG) with docker compose
    Up(UpArgs),
    /// Run the HTTP API and Streamable HTTP MCP endpoint (POST /mcp)
    Serve(ServeArgs),
    /// Ingest a file or a folder (HWP/HWPX/PDF/DOCX/PPTX/HTML/MD/TXT/PNG/JPG) into a bucket
    Ingest(IngestArgs),
    /// Vector index engines, external mirrors and the indexing outbox
    #[command(subcommand)]
    Index(IndexCmd),
    /// Memory ledger: append patches and promote entries
    #[command(subcommand)]
    Memory(MemoryCmd),
    /// Cascade PII redaction across chunks, pages, thumbnails, memories, wiki and vault
    #[command(subcommand)]
    Redact(RedactCmd),
    /// Obsidian-compatible wiki import/export
    #[command(subcommand)]
    Wiki(WikiCmd),
    /// Canonical concepts and cross-lingual aliases
    #[command(subcommand)]
    Concept(ConceptCmd),
    /// Golden-set benchmarks and AutoRAG optimization
    #[command(subcommand)]
    Bench(BenchCmd),
    /// Pipeline configuration promotion
    #[command(subcommand)]
    Pipeline(PipelineCmd),
    /// Run the local MCP server (16 tools) over stdio
    Mcp(McpArgs),
    /// One-time CLI authorization codes (exchanged for API keys)
    #[command(subcommand)]
    Auth(AuthCmd),
    /// First organization, department and admin on a fresh database (local operator)
    #[command(subcommand)]
    Org(OrgCmd),
    /// Create and list buckets
    #[command(subcommand)]
    Bucket(BucketCmd),
    /// Register this server's MCP endpoint in Claude Code or Codex
    Connect(ConnectArgs),
    /// Set up a fresh install in one step: org and admin key, CLI config, a bucket, a folder of
    /// documents and MCP clients
    Init(InitArgs),
    /// Documents in a bucket and the documents-folder watcher (status, scan now)
    #[command(subcommand)]
    Docs(DocsCmd),
    /// API keys: list and revoke
    #[command(subcommand)]
    Key(KeyCmd),
    /// Team members: add one (account, key and connect commands) or list them
    #[command(subcommand)]
    Member(MemberCmd),
}

#[derive(Debug, Subcommand)]
pub enum DocsCmd {
    /// Documents in a bucket (in force only, unless --all)
    List {
        #[arg(long, default_value = "docs")]
        bucket: String,
        /// Part of the title or path
        #[arg(long)]
        query: Option<String>,
        /// Also list superseded and removed documents
        #[arg(long)]
        all: bool,
        #[arg(long, default_value_t = 50)]
        limit: u32,
        #[arg(long, default_value_t = 0)]
        offset: u32,
    },
    /// What the watcher is doing: folders, documents, failed files and why (org admins)
    Status,
    /// Scan the documents folders now instead of at the next interval (org admins)
    Scan,
}

#[derive(Debug, Subcommand)]
pub enum KeyCmd {
    /// API keys (prefix only): every key for org admins, your own otherwise
    List {
        /// Only this member's keys
        #[arg(long)]
        principal: Option<Uuid>,
        /// Include revoked keys
        #[arg(long)]
        all: bool,
    },
    /// Stop a key at once (its id from `openrag key list`)
    Revoke { id: Uuid },
}

#[derive(Debug, Subcommand)]
pub enum MemberCmd {
    /// Add a member with an API key and print the commands that connect their Claude Code / Codex
    Add {
        #[arg(long)]
        email: String,
        /// Display name
        #[arg(long)]
        name: String,
        /// Permission level (1..10)
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(i16).range(1..=10))]
        level: i16,
        /// Department id (default: the first admin's department)
        #[arg(long)]
        dept: Option<Uuid>,
        /// Name of the key, e.g. the member's laptop (default: the e-mail)
        #[arg(long)]
        key_name: Option<String>,
        /// URL the member's machine uses to reach this server
        #[arg(long, env = "OPENRAG_PUBLIC_URL")]
        public_url: Option<String>,
    },
    /// Members of the organization (org admins)
    List,
}

/// The first organization of a fresh install.
#[derive(Debug, Clone, Args)]
pub struct OrgSpec {
    /// Organization slug (a-z, 0-9, '-', '_'; used in ARNs)
    #[arg(long, default_value = "default")]
    pub slug: String,
    /// Organization display name
    #[arg(long, default_value = "Default")]
    pub name: String,
    /// First department's ARN path (a-z, 0-9, '-', '_')
    #[arg(long, default_value = "general")]
    pub dept: String,
    /// First department's display name (default: --dept)
    #[arg(long)]
    pub dept_name: Option<String>,
    /// Admin e-mail
    #[arg(long, default_value = "admin@localhost")]
    pub admin_email: String,
    /// Admin display name
    #[arg(long, default_value = "Admin")]
    pub admin_name: String,
}

#[derive(Debug, Subcommand)]
pub enum OrgCmd {
    /// Create the organization, its first department and an admin (permission level 10, org
    /// admin), then mint an API key for the admin. Needs DATABASE_URL and operator mode.
    Init {
        #[command(flatten)]
        spec: OrgSpec,
        /// Write the API key (and --api-url) to the CLI config file
        #[arg(long)]
        save: bool,
        /// Server URL stored with --save
        #[arg(long)]
        api_url: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum BucketCmd {
    /// Create a bucket
    Create {
        /// Bucket slug
        slug: String,
        /// Display name (default: the slug)
        #[arg(long)]
        name: Option<String>,
        /// Who may read it
        #[arg(long, value_enum, default_value_t = ScopeArg::Org)]
        scope: ScopeArg,
        /// Minimum permission level (1..10)
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(i16).range(1..=10))]
        auth_level: i16,
        /// Owning department (default: the caller's first department)
        #[arg(long)]
        dept: Option<Uuid>,
    },
    /// List the buckets the caller can see
    List,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum McpClient {
    /// Claude Code (`claude mcp add`)
    Claude,
    /// Codex (`codex mcp add`)
    Codex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ClaudeScope {
    /// Every project of this user
    User,
    /// `.mcp.json` in the current directory, shared with the repository
    Project,
    /// This project, this user only
    Local,
}

impl ClaudeScope {
    const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
            Self::Local => "local",
        }
    }
}

#[derive(Debug, Args)]
pub struct ConnectArgs {
    /// MCP client to register the server with
    #[arg(value_enum)]
    pub client: McpClient,
    /// Name of the MCP server entry
    #[arg(long, default_value = "openrag")]
    pub name: String,
    /// Claude Code configuration scope
    #[arg(long, value_enum, default_value_t = ClaudeScope::User)]
    pub scope: ClaudeScope,
    /// Print the command instead of running it (the key is masked)
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct InitArgs {
    /// Server to set up; it must be running (`openrag up` or `openrag serve`)
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub api_url: String,
    /// URL MCP clients use to reach the server (default: --api-url); set it when the clients run
    /// on other machines. Compose sets OPENRAG_PUBLIC_URL to the published host port.
    #[arg(long, env = "OPENRAG_PUBLIC_URL")]
    pub public_url: Option<String>,
    #[command(flatten)]
    pub org: OrgSpec,
    /// Bucket to create for the documents
    #[arg(long, default_value = "docs")]
    pub bucket: String,
    /// Folder to ingest into the bucket (recursively)
    #[arg(long)]
    pub docs: Option<PathBuf>,
    /// Register the server in these MCP clients (repeatable)
    #[arg(long = "connect", value_enum)]
    pub connect: Vec<McpClient>,
    /// Claude Code configuration scope for --connect claude
    #[arg(long, value_enum, default_value_t = ClaudeScope::User)]
    pub mcp_scope: ClaudeScope,
}

#[derive(Debug, Subcommand)]
pub enum AuthCmd {
    /// Issue a single-use code for a principal; exchange it at POST /v1/auth/cli/code-exchange.
    /// With OPENRAG_API_KEY the issuing admin's key bounds the code; with only DATABASE_URL
    /// the local operator issues it (bootstrap for the first admin).
    Code {
        /// Principal the code (and the resulting API key) belongs to
        #[arg(long)]
        principal: Uuid,
        /// Org of the principal (operator mode; defaults to OPENRAG_ORG_ID or the principal's org)
        #[arg(long)]
        org: Option<Uuid>,
        /// Code lifetime in seconds (default 600, max 86400)
        #[arg(long)]
        ttl: Option<i64>,
        /// Permission-level ceiling for the minted key
        #[arg(long, value_parser = clap::value_parser!(i16).range(1..=10))]
        max_auth_level: Option<i16>,
        /// ARN pattern the minted key is limited to (repeatable)
        #[arg(long = "scoped-arn")]
        scoped_arns: Vec<String>,
        /// Permission group the minted key is limited to (repeatable)
        #[arg(long = "scoped-group")]
        scoped_groups: Vec<Uuid>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum UpProfile {
    /// PostgreSQL + OpenRAG services
    Standard,
    /// Same stack on an internal-only network (no egress)
    Airgap,
}

#[derive(Debug, Args)]
pub struct UpArgs {
    /// Deployment profile
    #[arg(long, value_enum, default_value_t = UpProfile::Standard)]
    pub profile: UpProfile,
    /// Directory containing docker/ (defaults to OPENRAG_HOME or the current directory)
    #[arg(long)]
    pub root: Option<PathBuf>,
    /// Run in the foreground instead of detached
    #[arg(long)]
    pub foreground: bool,
    /// Print the docker compose command without running it
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Bind address (default OPENRAG_BIND, else HOST:PORT, else 127.0.0.1:8080)
    #[arg(long)]
    pub bind: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ScopeArg {
    Private,
    Dept,
    Group,
    Org,
}

impl ScopeArg {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Private => "private",
            Self::Dept => "dept",
            Self::Group => "group",
            Self::Org => "org",
        }
    }

    const fn to_access(self) -> AccessScope {
        match self {
            Self::Private => AccessScope::Private,
            Self::Dept => AccessScope::Dept,
            Self::Group => AccessScope::Group,
            Self::Org => AccessScope::Org,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OcrModeArg {
    Share,
    Private,
    Zdr,
    Local,
}

impl OcrModeArg {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Share => "share",
            Self::Private => "private",
            Self::Zdr => "zdr",
            Self::Local => "local",
        }
    }
}

#[derive(Debug, Args)]
pub struct IngestArgs {
    /// File to ingest, or a folder (every supported file in it, recursively)
    pub path: PathBuf,
    /// Bucket slug or id
    #[arg(long)]
    pub bucket: String,
    /// Access scope
    #[arg(long, value_enum)]
    pub scope: Option<ScopeArg>,
    /// Minimum permission level (1..10)
    #[arg(long, value_parser = clap::value_parser!(i16).range(1..=10))]
    pub auth_level: Option<i16>,
    /// Allowed permission group ids (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// Tags (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub tags: Vec<String>,
    /// OCR/processing sensitivity (L8+ content is never processed as share)
    #[arg(long, value_enum)]
    pub mode: Option<OcrModeArg>,
    /// Document title (defaults to the file name)
    #[arg(long)]
    pub title: Option<String>,
    /// Pipeline YAML applied to this ingestion (local mode)
    #[arg(long)]
    pub pipeline: Option<PathBuf>,
    /// When the document takes effect, e.g. its authoring/enactment date
    /// (RFC 3339 or YYYY-MM-DD; default: ingestion time)
    #[arg(long, value_parser = parse_datetime_arg)]
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// When the document stops being valid (RFC 3339 or YYYY-MM-DD)
    #[arg(long, value_parser = parse_datetime_arg)]
    pub valid_to: Option<chrono::DateTime<chrono::Utc>>,
    /// Revision number (newer revisions of the same title are preferred in search)
    #[arg(long, value_parser = clap::value_parser!(i32).range(1..))]
    pub revision: Option<i32>,
}

/// Accepts RFC 3339 (`2026-03-01T09:00:00+09:00`) or a plain date (`2026-03-01`, UTC midnight).
pub fn parse_datetime_arg(raw: &str) -> Result<chrono::DateTime<chrono::Utc>, String> {
    let t = raw.trim();
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(t) {
        return Ok(dt.with_timezone(&chrono::Utc));
    }
    chrono::NaiveDate::parse_from_str(t, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|naive| naive.and_utc())
        .ok_or_else(|| format!("'{t}' is not an RFC 3339 timestamp or YYYY-MM-DD date"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EngineArg {
    Native,
    Pgvector,
    Remote,
}

impl EngineArg {
    const fn to_engine(self) -> PgPineconeEngine {
        match self {
            Self::Native => PgPineconeEngine::Native,
            Self::Pgvector => PgPineconeEngine::Pgvector,
            Self::Remote => PgPineconeEngine::Remote,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Pgvector => "pgvector",
            Self::Remote => "remote",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MirrorArg {
    #[value(name = "s3-vectors", alias = "s3_vectors")]
    S3Vectors,
    Qdrant,
    Faiss,
    Usearch,
    None,
}

impl MirrorArg {
    const fn to_backend(self) -> ExternalMirrorBackend {
        match self {
            Self::S3Vectors => ExternalMirrorBackend::S3Vectors,
            Self::Qdrant => ExternalMirrorBackend::Qdrant,
            Self::Faiss => ExternalMirrorBackend::Faiss,
            Self::Usearch => ExternalMirrorBackend::Usearch,
            Self::None => ExternalMirrorBackend::None,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum IndexCmd {
    /// Switch the pg_pinecone engine of a vector table
    Engine {
        /// Vector table (corpus.chunks, corpus.pages, corpus.page_figures, wiki.pages, memory.entries)
        table: String,
        /// Engine to use
        #[arg(long = "set", value_enum)]
        engine: EngineArg,
    },
    /// Mirror a vector table to an external backend
    Mirror {
        /// Vector table
        table: String,
        /// External backend
        #[arg(long = "to", value_enum)]
        to: MirrorArg,
    },
    /// Show outbox, watermark and quarantine status
    Status,
    /// Drain pending outbox items (priority order)
    Drain {
        /// Maximum items to claim
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Retry quarantined items
    #[command(group(ArgGroup::new("target").required(true).args(["id", "all"])))]
    Retry {
        /// Quarantine item id
        #[arg(long)]
        id: Option<Uuid>,
        /// Retry every unresolved quarantine item
        #[arg(long)]
        all: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PatchOpArg {
    Amend,
    Supersede,
    Retract,
    Redact,
}

impl PatchOpArg {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Amend => "amend",
            Self::Supersede => "supersede",
            Self::Retract => "retract",
            Self::Redact => "redact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PromoteToArg {
    Dept,
    Group,
    Org,
    Wiki,
}

impl PromoteToArg {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Dept => "dept",
            Self::Group => "group",
            Self::Org => "org",
            Self::Wiki => "wiki",
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum MemoryCmd {
    /// Append an immutable patch to a memory entry
    Patch {
        memory_id: Uuid,
        #[arg(long, value_enum)]
        op: PatchOpArg,
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        reason: Option<String>,
        /// With --op redact: also clear prev_text in the patch ledger
        #[arg(long)]
        hard_purge: bool,
    },
    /// Promote a memory entry up the scope ladder or into the wiki
    Promote {
        memory_id: Uuid,
        #[arg(long = "to", value_enum)]
        to: PromoteToArg,
        /// Target bucket (required for --to wiki)
        #[arg(long)]
        bucket: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RedactScopeArg {
    Org,
    Dept,
    Document,
    Chunk,
    Memory,
    Wiki,
    #[value(name = "vault_entry", alias = "vault-entry")]
    VaultEntry,
    #[value(name = "pattern_org", alias = "pattern-org")]
    PatternOrg,
    #[value(name = "pattern_dept", alias = "pattern-dept")]
    PatternDept,
}

#[derive(Debug, Args)]
pub struct RedactRunArgs {
    /// Redaction scope; target flags pick the resource, --scope org|dept picks the pattern range
    #[arg(long, value_enum, default_value_t = RedactScopeArg::Org)]
    pub scope: RedactScopeArg,
    /// Reason recorded in the redaction ledger
    #[arg(long)]
    pub reason: String,
    #[arg(long = "doc")]
    pub doc: Option<Uuid>,
    #[arg(long = "chunk")]
    pub chunk: Option<Uuid>,
    #[arg(long = "memory")]
    pub memory: Option<Uuid>,
    #[arg(long = "wiki")]
    pub wiki: Option<Uuid>,
    /// Literal pattern to redact everywhere in scope
    #[arg(long)]
    pub pattern: Option<String>,
    /// Department id for --scope dept pattern redaction
    #[arg(long)]
    pub dept: Option<Uuid>,
    #[arg(long = "vault-entry")]
    pub vault_entry: Option<Uuid>,
    /// Black out matching thumbnail regions and re-embed
    #[arg(long)]
    pub redact_thumbnails: bool,
    /// Destroy vault ciphertext and clear patch prev_text
    #[arg(long, conflicts_with = "tombstone_delete")]
    pub hard_purge: bool,
    /// Tombstone and delete the target rows
    #[arg(long)]
    pub tombstone_delete: bool,
}

#[derive(Debug, Subcommand)]
pub enum RedactCmd {
    /// Run a cascade redaction job
    Run(RedactRunArgs),
}

#[derive(Debug, Subcommand)]
pub enum WikiCmd {
    /// Import an Obsidian-style markdown directory into a bucket's wiki
    Import {
        dir: PathBuf,
        #[arg(long)]
        bucket: String,
        #[arg(long, value_enum)]
        scope: Option<ScopeArg>,
        #[arg(long, value_parser = clap::value_parser!(i16).range(1..=10))]
        auth_level: Option<i16>,
    },
    /// Export a bucket's wiki pages as markdown with YAML frontmatter
    Export {
        bucket: String,
        #[arg(long = "out")]
        out: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConceptCmd {
    /// Link two terms as aliases of one canonical concept
    Link {
        source: String,
        target: String,
        #[arg(long = "canonical")]
        canonical: Option<String>,
        #[arg(long)]
        lang: Option<String>,
        #[arg(long)]
        relation: Option<String>,
        #[arg(long)]
        concept_id: Option<Uuid>,
    },
    /// Remove an alias (or a whole concept)
    Unlink {
        source: Option<String>,
        target: Option<String>,
        #[arg(long)]
        concept_id: Option<Uuid>,
        #[arg(long)]
        alias_id: Option<Uuid>,
    },
    /// List concepts and aliases
    List {
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        concept_id: Option<Uuid>,
        #[arg(long)]
        lang: Option<String>,
        /// Print JSON only
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum BenchCmd {
    /// Run all three benchmark tracks on a golden set
    Run {
        #[arg(long)]
        golden_set: String,
        #[arg(long)]
        pipeline: Option<PathBuf>,
        #[arg(long)]
        bucket: Option<Uuid>,
    },
    /// AutoRAG optimization with hard security gates
    Optimize {
        #[arg(long)]
        golden_set: String,
        #[arg(long)]
        search_space: PathBuf,
        #[arg(long)]
        pipeline: Option<PathBuf>,
        #[arg(long)]
        strategy: Option<String>,
        #[arg(long)]
        bucket: Option<Uuid>,
        #[arg(long)]
        require_zero_pii_leak: bool,
        #[arg(long)]
        require_zero_arn_violation: bool,
        /// Write the best pipeline YAML here (only when every security gate passes)
        #[arg(long)]
        export_best: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
pub enum PipelineCmd {
    /// Register a pipeline YAML and promote it (safety-checked)
    Promote {
        yaml_path: PathBuf,
        #[arg(long)]
        bucket: String,
    },
}

#[derive(Debug, Args)]
pub struct McpArgs {
    /// Serve MCP over stdin/stdout (NDJSON or Content-Length framing)
    #[arg(long)]
    pub stdio: bool,
    /// API key identifying the caller (default OPENRAG_API_KEY)
    #[arg(long, env = "OPENRAG_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Expected principal id; the call fails if the key belongs to someone else
    #[arg(long)]
    pub principal: Option<Uuid>,
}

// ============================================================================
// Errors & output
// ============================================================================

#[derive(Debug)]
pub struct CliError(pub String);

impl CliError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl From<OpenRagError> for CliError {
    fn from(e: OpenRagError) -> Self {
        Self(e.to_string())
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

fn http_error(status: StatusCode, body: &Value) -> CliError {
    let code = body.get("error").and_then(Value::as_str).unwrap_or("error");
    let msg = body
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_else(|| status.canonical_reason().unwrap_or("request failed"));
    CliError(format!("HTTP {} {code}: {msg}", status.as_u16()))
}

fn print_json(v: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
    );
}

// ============================================================================
// Configuration file
// ============================================================================

#[derive(Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CliConfigFile {
    api_url: Option<String>,
    api_key: Option<String>,
    org_id: Option<Uuid>,
    principal_id: Option<Uuid>,
    database_url: Option<String>,
    mode: Option<String>,
    operator: Option<bool>,
}

fn default_config_path() -> Option<PathBuf> {
    if let Some(xdg) = non_empty_env("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("openrag").join("config.toml"));
    }
    non_empty_env("HOME").map(|h| PathBuf::from(h).join(".config").join("openrag").join("config.toml"))
}

#[cfg(unix)]
fn readable_by_others(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o077 != 0
}

#[cfg(not(unix))]
fn readable_by_others(_meta: &std::fs::Metadata) -> bool {
    false
}

/// Loads the CLI config file and exports its values as defaults for unset environment
/// variables. Call before the async runtime starts and before argument parsing (clap reads
/// `OPENRAG_API_KEY` from the environment). Returns the file used, if any.
pub fn apply_config_file() -> Result<Option<PathBuf>, String> {
    apply_config_file_inner(false)
}

/// `openrag init` and `openrag org init` write the config file, so it may not exist yet.
fn creates_config(args: &[String]) -> bool {
    let words: Vec<&str> = args.iter().skip(1).map(String::as_str).filter(|a| !a.starts_with('-')).take(2).collect();
    matches!(words.as_slice(), ["init", ..] | ["org", "init"])
}

fn apply_config_file_inner(allow_missing: bool) -> Result<Option<PathBuf>, String> {
    let explicit = non_empty_env("OPENRAG_CONFIG").map(PathBuf::from);
    let Some(path) = explicit.clone().or_else(default_config_path) else {
        return Ok(None);
    };
    let meta = match std::fs::metadata(&path) {
        Ok(m) => m,
        Err(e) if (explicit.is_none() || allow_missing) && e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None)
        }
        Err(e) => return Err(format!("cannot read config {}: {e}", path.display())),
    };
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("cannot read config {}: {e}", path.display()))?;
    let cfg: CliConfigFile =
        toml::from_str(&raw).map_err(|e| format!("invalid config {}: {e}", path.display()))?;
    if (cfg.api_key.is_some() || cfg.database_url.is_some()) && readable_by_others(&meta) {
        return Err(format!(
            "{} holds credentials but is readable by group/others; run `chmod 600 {}`",
            path.display(),
            path.display()
        ));
    }
    let pairs: [(&str, Option<String>); 7] = [
        ("OPENRAG_API_URL", cfg.api_url),
        ("OPENRAG_API_KEY", cfg.api_key),
        ("OPENRAG_ORG_ID", cfg.org_id.map(|u| u.to_string())),
        ("OPENRAG_PRINCIPAL_ID", cfg.principal_id.map(|u| u.to_string())),
        ("DATABASE_URL", cfg.database_url),
        ("OPENRAG_MODE", cfg.mode),
        ("OPENRAG_OPERATOR", cfg.operator.map(|b| if b { "1" } else { "0" }.to_string())),
    ];
    for (name, value) in pairs {
        if let Some(v) = value.filter(|v| !v.trim().is_empty()) {
            if non_empty_env(name).is_none() {
                // Edition 2021: safe; runs before any other thread exists.
                std::env::set_var(name, v);
            }
        }
    }
    Ok(Some(path))
}

fn operator_enabled() -> bool {
    non_empty_env("OPENRAG_OPERATOR")
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

// ============================================================================
// Entry points
// ============================================================================

/// Binary entrypoint for `openrag`.
pub fn main_entry() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = apply_config_file_inner(creates_config(&args)) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let cli = Cli::parse();
    let default_level = if matches!(cli.command, Command::Serve(_)) { "info" } else { "warn" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to start async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError(msg)) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

pub async fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Up(a) => cmd_up(a).await,
        Command::Serve(a) => cmd_serve(a).await,
        Command::Ingest(a) => cmd_ingest(a).await,
        Command::Index(c) => cmd_index(c).await,
        Command::Memory(c) => cmd_memory(c).await,
        Command::Redact(RedactCmd::Run(a)) => cmd_redact(a).await,
        Command::Wiki(c) => cmd_wiki(c).await,
        Command::Concept(c) => cmd_concept(c).await,
        Command::Bench(c) => cmd_bench(c).await,
        Command::Pipeline(c) => cmd_pipeline(c).await,
        Command::Mcp(a) => cmd_mcp(a).await,
        Command::Auth(c) => cmd_auth(c).await,
        Command::Org(OrgCmd::Init { spec, save, api_url }) => cmd_org_init(spec, save, api_url).await,
        Command::Bucket(c) => cmd_bucket(c).await,
        Command::Connect(a) => cmd_connect(a).await,
        Command::Init(a) => cmd_init(a).await,
        Command::Docs(c) => cmd_docs(c).await,
        Command::Key(c) => cmd_key(c).await,
        Command::Member(c) => cmd_member(c).await,
    }
}

// ============================================================================
// Execution contexts
// ============================================================================

struct Local {
    state: Arc<AppState>,
    key_principal: Option<PrincipalContext>,
}

struct Remote {
    base: String,
    key: String,
    http: reqwest::Client,
}

enum Exec {
    Local(Local),
    Remote(Remote),
}

fn env_uuid(name: &str) -> Result<Option<Uuid>, CliError> {
    match non_empty_env(name) {
        None => Ok(None),
        Some(raw) => Uuid::parse_str(raw.trim())
            .map(Some)
            .map_err(|_| CliError(format!("{name} must be a UUID"))),
    }
}

async fn local() -> Result<Local, CliError> {
    if non_empty_env("DATABASE_URL").is_none() {
        return Err(CliError::new(
            "DATABASE_URL is not set (or set OPENRAG_API_URL + OPENRAG_API_KEY for a remote server)",
        ));
    }
    let state = AppState::from_env().await?;
    let key_principal = match non_empty_env("OPENRAG_API_KEY") {
        Some(key) => {
            let worker = state.require_worker_pool()?;
            Some(authenticate_bearer_token(worker, key.trim()).await?)
        }
        None => None,
    };
    if key_principal.is_none() && !operator_enabled() {
        return Err(CliError::new(
            "no identity: set api_key in config.toml (or OPENRAG_API_KEY), or enable local operator mode with `operator = true` in config.toml (or OPENRAG_OPERATOR=1)",
        ));
    }
    Ok(Local { state, key_principal })
}

async fn exec() -> Result<Exec, CliError> {
    if let Some(base) = non_empty_env("OPENRAG_API_URL") {
        let key = non_empty_env("OPENRAG_API_KEY")
            .ok_or_else(|| CliError::new("OPENRAG_API_URL requires OPENRAG_API_KEY"))?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| CliError(format!("http client: {e}")))?;
        return Ok(Exec::Remote(Remote {
            base: base.trim_end_matches('/').to_string(),
            key,
            http,
        }));
    }
    Ok(Exec::Local(local().await?))
}

impl Remote {
    async fn call(&self, method: Method, path: &str, body: DispatchBody) -> Result<Value, CliError> {
        let url = format!("{}{path}", self.base);
        let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
            .map_err(|e| CliError(format!("method: {e}")))?;
        let mut rb = self
            .http
            .request(reqwest_method, &url)
            .bearer_auth(&self.key);
        rb = match body {
            DispatchBody::Empty => rb,
            DispatchBody::Json(v) => rb.json(&v),
            DispatchBody::Multipart { boundary, bytes } => rb
                .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                .body(bytes),
        };
        let resp = rb
            .send()
            .await
            .map_err(|e| CliError(format!("request to {url} failed: {e}")))?;
        let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let text = resp.text().await.unwrap_or_default();
        let decoded = if text.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({ "text": text }))
        };
        if status.is_success() {
            Ok(decoded)
        } else {
            Err(http_error(status, &decoded))
        }
    }
}

async fn local_call(
    state: &Arc<AppState>,
    principal: &PrincipalContext,
    method: Method,
    path: &str,
    body: DispatchBody,
) -> Result<Value, CliError> {
    dispatch_as(state, principal, method, path, body)
        .await
        .map(|(_, v)| v)
        .map_err(|(status, v)| http_error(status, &v))
}

impl Local {
    fn worker(&self) -> Result<&openrag_db::DbPool, CliError> {
        Ok(self.state.require_worker_pool()?)
    }

    fn is_operator(&self) -> bool {
        self.key_principal.is_none()
    }

    async fn load(&self, org_id: Uuid, principal_id: Uuid) -> Result<PrincipalContext, CliError> {
        Ok(load_principal_context(self.worker()?, org_id, principal_id, None, "principal not found").await?)
    }

    /// Principal for a user-level operation on a resource owned by `owner_id`.
    async fn user_principal(&self, org_id: Uuid, owner_id: Uuid) -> Result<PrincipalContext, CliError> {
        match &self.key_principal {
            Some(p) => Ok(p.clone()),
            None => self.load(org_id, owner_id).await,
        }
    }

    /// Principal for an admin operation in `org_id`. Operator mode elevates the
    /// attributed principal to org-admin authority.
    async fn admin_principal(
        &self,
        org_id: Uuid,
        fallback_owner: Option<Uuid>,
    ) -> Result<PrincipalContext, CliError> {
        if let Some(p) = &self.key_principal {
            return Ok(p.clone());
        }
        let worker = self.worker()?;
        let attributed = if let Some(pid) = env_uuid("OPENRAG_PRINCIPAL_ID")? {
            Some(pid)
        } else {
            let admin: Option<Uuid> = sqlx::query_scalar(
                "SELECT id FROM iam.principals
                 WHERE org_id = $1 AND is_org_admin = true AND status = 'active'
                 ORDER BY created_at, id LIMIT 1",
            )
            .bind(org_id)
            .fetch_optional(worker.inner())
            .await
            .map_err(|e| CliError(format!("lookup org admin: {e}")))?;
            admin.or(fallback_owner)
        };
        let pid = attributed.ok_or_else(|| {
            CliError(format!(
                "org {org_id} has no active admin to attribute this operation to; set OPENRAG_PRINCIPAL_ID or OPENRAG_API_KEY"
            ))
        })?;
        let mut ctx = self.load(org_id, pid).await?;
        ctx.is_org_admin = true;
        Ok(ctx)
    }

    /// Orgs an operator command applies to.
    async fn operator_orgs(&self) -> Result<Vec<Uuid>, CliError> {
        if let Some(p) = &self.key_principal {
            return Ok(vec![p.org_id]);
        }
        if let Some(org) = env_uuid("OPENRAG_ORG_ID")? {
            return Ok(vec![org]);
        }
        let orgs: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM iam.orgs ORDER BY created_at, id")
            .fetch_all(self.worker()?.inner())
            .await
            .map_err(|e| CliError(format!("list orgs: {e}")))?;
        if orgs.is_empty() {
            return Err(CliError::new("no organizations found"));
        }
        Ok(orgs)
    }

    fn require_admin_key(&self) -> Result<(), CliError> {
        match &self.key_principal {
            Some(p) if !p.is_org_admin => Err(CliError::new(
                "HTTP 403 forbidden: org_admin privilege required",
            )),
            _ => Ok(()),
        }
    }

    /// Resolves a bucket (slug or id) to `(org_id, bucket_id, owner_principal_id)`.
    async fn find_bucket(&self, bucket_ref: &str) -> Result<(Uuid, Uuid, Uuid), CliError> {
        let worker = self.worker()?;
        let org_filter = match &self.key_principal {
            Some(p) => Some(p.org_id),
            None => env_uuid("OPENRAG_ORG_ID")?,
        };
        let parsed = Uuid::parse_str(bucket_ref.trim()).ok();
        let rows = sqlx::query(
            "SELECT org_id, id, owner_principal_id FROM corpus.buckets
             WHERE ($1::uuid IS NULL OR org_id = $1)
               AND (($2::uuid IS NOT NULL AND id = $2) OR slug = $3)
             LIMIT 2",
        )
        .bind(org_filter)
        .bind(parsed)
        .bind(bucket_ref.trim())
        .fetch_all(worker.inner())
        .await
        .map_err(|e| CliError(format!("lookup bucket: {e}")))?;
        match rows.as_slice() {
            [] => Err(CliError(format!("bucket '{bucket_ref}' not found"))),
            [row] => {
                let get = |c: &str| {
                    row.try_get::<Uuid, _>(c)
                        .map_err(|e| CliError(format!("bucket column {c}: {e}")))
                };
                Ok((get("org_id")?, get("id")?, get("owner_principal_id")?))
            }
            _ => Err(CliError(format!(
                "bucket '{bucket_ref}' exists in several orgs; set OPENRAG_ORG_ID or OPENRAG_API_KEY"
            ))),
        }
    }

    /// Org and owner of a row in `table` (worker pool, operator lookups only).
    async fn owner_of(&self, table: &str, id: Uuid) -> Result<(Uuid, Uuid), CliError> {
        let sql = format!("SELECT org_id, owner_principal_id FROM {table} WHERE id = $1");
        let row = sqlx::query(&sql)
            .bind(id)
            .fetch_optional(self.worker()?.inner())
            .await
            .map_err(|e| CliError(format!("lookup {table}: {e}")))?
            .ok_or_else(|| CliError(format!("{table} {id} not found")))?;
        let org: Uuid = row.try_get("org_id").map_err(|e| CliError(format!("{table}.org_id: {e}")))?;
        let owner: Uuid = row
            .try_get("owner_principal_id")
            .map_err(|e| CliError(format!("{table}.owner_principal_id: {e}")))?;
        Ok((org, owner))
    }
}

fn require_local(exec: Exec, what: &str) -> Result<Local, CliError> {
    match exec {
        Exec::Local(l) => Ok(l),
        Exec::Remote(_) => Err(CliError(format!(
            "`{what}` needs direct database access; unset OPENRAG_API_URL and set DATABASE_URL"
        ))),
    }
}

// ============================================================================
// up / serve / mcp
// ============================================================================

async fn cmd_up(a: UpArgs) -> Result<(), CliError> {
    let root = a
        .root
        .or_else(|| non_empty_env("OPENRAG_HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    let base = root.join("docker").join("docker-compose.yml");
    if !base.is_file() {
        return Err(CliError(format!(
            "compose file not found: {} (run from the OpenRAG checkout or pass --root)",
            base.display()
        )));
    }
    let mut args: Vec<String> = vec!["compose".into(), "-f".into(), base.display().to_string()];
    if a.profile == UpProfile::Airgap {
        let airgap = root.join("docker").join("docker-compose.airgap.yml");
        if !airgap.is_file() {
            return Err(CliError(format!("compose file not found: {}", airgap.display())));
        }
        args.extend(["-f".into(), airgap.display().to_string()]);
    }
    // postgres, local-embed (embeddings) and openrag-api; the airgap overlay keeps local-embed
    // offline and points the API at the site's own servers.
    args.extend(["--profile".into(), "app".into()]);
    args.push("up".into());
    if !a.foreground {
        args.push("-d".into());
    }
    if a.dry_run {
        let mut cmd = vec!["docker".to_string()];
        cmd.extend(args);
        print_json(&json!({ "command": cmd }));
        return Ok(());
    }
    let status = tokio::process::Command::new("docker")
        .args(&args)
        .status()
        .await
        .map_err(|e| CliError(format!("failed to run docker: {e}")))?;
    if status.success() {
        Ok(())
    } else {
        Err(CliError(format!("docker compose exited with {status}")))
    }
}

fn resolve_bind_addr(cli_bind: Option<String>) -> Result<SocketAddr, CliError> {
    let raw = cli_bind
        .or_else(|| non_empty_env("OPENRAG_BIND"))
        .unwrap_or_else(|| {
            let host = non_empty_env("HOST").unwrap_or_else(|| "127.0.0.1".to_string());
            let port = non_empty_env("PORT").unwrap_or_else(|| "8080".to_string());
            format!("{host}:{port}")
        });
    raw.parse::<SocketAddr>()
        .map_err(|e| CliError(format!("invalid bind address '{raw}': {e}")))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            let _ = sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

async fn cmd_serve(a: ServeArgs) -> Result<(), CliError> {
    let addr = resolve_bind_addr(a.bind)?;
    let state = AppState::from_env().await?;
    crate::watch::start_from_env(&state)?;
    let router = crate::build_router(state);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(addr = %addr, "openrag serve listening (HTTP API + POST /mcp)");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn cmd_mcp(a: McpArgs) -> Result<(), CliError> {
    if !a.stdio {
        return Err(CliError::new(
            "pass --stdio to serve MCP over stdin/stdout (Streamable HTTP MCP is served by `openrag serve` at POST /mcp)",
        ));
    }
    crate::mcp::run_stdio(StdioOptions {
        api_key: a.api_key,
        principal_id: a.principal,
    })
    .await?;
    Ok(())
}

// ============================================================================
// ingest
// ============================================================================

fn parse_uuid_list(raw: &[String], what: &str) -> Result<Vec<Uuid>, CliError> {
    raw.iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| Uuid::parse_str(s).map_err(|_| CliError(format!("{what}: '{s}' is not a UUID"))))
        .collect()
}

async fn apply_pipeline_file(state: &AppState, path: &Path) -> Result<(), CliError> {
    if !path.is_file() {
        return Err(CliError(format!("pipeline file not found: {}", path.display())));
    }
    let cfg = PipelineConfig::from_yaml_file(path)?;
    *state.pii_engine.write().await = PiiEngine::from_config(&cfg.pii);
    *state.pipeline_config.write().await = cfg;
    Ok(())
}

/// File extensions `openrag ingest` picks up from a folder.
const INGEST_EXTENSIONS: &[&str] = crate::watch::SUPPORTED_TYPES;

/// The file itself, or every supported file under a folder (sorted, hidden entries skipped).
fn ingest_targets(path: &Path) -> Result<Vec<PathBuf>, CliError> {
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).map_err(|e| CliError(format!("cannot read {}: {e}", dir.display())))? {
            let p = entry?.path();
            let hidden = p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.'));
            if hidden {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| INGEST_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            {
                out.push(p);
            }
        }
    }
    out.sort();
    if out.is_empty() {
        return Err(CliError(format!("no supported files under {}", path.display())));
    }
    Ok(out)
}

/// The document key of a file: its path below the folder being ingested (`/` separated), or the
/// file name for a single file. Ingesting the same key again with new content replaces the
/// in-force copy (the server closes it and returns `supersedes`) instead of adding a second one.
fn document_key(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).ok().filter(|r| !r.as_os_str().is_empty());
    let rel = rel.unwrap_or_else(|| Path::new(file.file_name().unwrap_or(file.as_os_str())));
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

async fn ingest_one(ex: &Exec, a: &IngestArgs, path: &Path) -> Result<Value, CliError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| CliError(format!("cannot read {}: {e}", path.display())))?;
    if bytes.is_empty() {
        return Err(CliError(format!("{} is empty", path.display())));
    }
    let groups = parse_uuid_list(&a.groups, "--groups")?;
    let tags: Vec<String> = a.tags.iter().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("document")
        .to_string();

    let mut fields: Vec<(&str, String)> = Vec::new();
    fields.push(("title", a.title.clone().unwrap_or_else(|| file_name.clone())));
    if let Some(s) = a.scope {
        fields.push(("access_scope", s.as_str().to_string()));
    }
    if let Some(l) = a.auth_level {
        fields.push(("min_auth_level", l.to_string()));
    }
    fields.push(("allowed_group_ids", json!(groups).to_string()));
    if !tags.is_empty() {
        fields.push(("tags", json!(tags).to_string()));
    }
    if let Some(m) = a.mode {
        fields.push(("ocr_mode", m.as_str().to_string()));
    }
    if let (Some(f), Some(t)) = (a.valid_from, a.valid_to) {
        if t <= f {
            return Err(CliError::new("--valid-to must be later than --valid-from"));
        }
    }
    if let Some(d) = a.valid_from {
        fields.push(("valid_from", d.to_rfc3339()));
    }
    if let Some(d) = a.valid_to {
        fields.push(("valid_to", d.to_rfc3339()));
    }
    if let Some(r) = a.revision {
        fields.push(("revision", r.to_string()));
    }
    fields.push(("external_key", document_key(&a.path, path)));
    let (boundary, body) = build_multipart(&file_name, &bytes, &fields);

    match ex {
        Exec::Remote(r) => {
            let path = format!("/v2/buckets/{}/documents", pct_encode(a.bucket.trim()));
            r.call(Method::POST, &path, DispatchBody::Multipart { boundary, bytes: body }).await
        }
        Exec::Local(l) => {
            let (org, bucket_id, owner) = l.find_bucket(&a.bucket).await?;
            let principal = l.user_principal(org, owner).await?;
            let path = format!("/v2/buckets/{bucket_id}/documents");
            local_call(&l.state, &principal, Method::POST, &path, DispatchBody::Multipart { boundary, bytes: body }).await
        }
    }
}

/// Ingests every file under `a.path`; one failure does not stop the rest.
async fn ingest_many(ex: &Exec, a: &IngestArgs) -> Result<Value, CliError> {
    let files = ingest_targets(&a.path)?;
    let mut results = Vec::with_capacity(files.len());
    let (mut failed, mut unchanged) = (0usize, 0usize);
    for f in &files {
        match ingest_one(ex, a, f).await {
            Ok(v) => {
                let same = v.get("unchanged").and_then(Value::as_bool) == Some(true);
                if same {
                    unchanged += 1;
                }
                let mut r = json!({
                    "file": f.display().to_string(),
                    "document_id": v.get("document_id").or_else(|| v.get("id")).cloned().unwrap_or(Value::Null),
                    "revision": v.get("revision").cloned().unwrap_or(Value::Null),
                });
                if same {
                    r["unchanged"] = json!(true);
                }
                if let Some(old) = v.get("supersedes").filter(|o| !o.is_null()) {
                    r["supersedes"] = old.clone();
                }
                results.push(r);
            }
            Err(e) => {
                failed += 1;
                results.push(json!({ "file": f.display().to_string(), "error": e.0 }));
            }
        }
    }
    Ok(json!({
        "ingested": files.len() - failed - unchanged,
        "unchanged": unchanged,
        "failed": failed,
        "results": results,
    }))
}

async fn cmd_ingest(a: IngestArgs) -> Result<(), CliError> {
    let ex = exec().await?;
    match &ex {
        Exec::Remote(_) if a.pipeline.is_some() => {
            return Err(CliError::new("--pipeline applies to local ingestion only"));
        }
        Exec::Local(l) => {
            if let Some(p) = a.pipeline.as_deref() {
                apply_pipeline_file(&l.state, p).await?;
            }
        }
        Exec::Remote(_) => {}
    }
    if !a.path.is_dir() {
        print_json(&ingest_one(&ex, &a, &a.path).await?);
        return Ok(());
    }
    if a.title.is_some() {
        return Err(CliError::new("--title applies to a single file, not a folder"));
    }
    let out = ingest_many(&ex, &a).await?;
    print_json(&out);
    match out.get("failed").and_then(Value::as_u64) {
        Some(n) if n > 0 => Err(CliError(format!("{n} file(s) failed"))),
        _ => Ok(()),
    }
}

// ============================================================================
// index
// ============================================================================

async fn cmd_index(c: IndexCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    match c {
        IndexCmd::Engine { table, engine } => {
            let vt = VectorTable::from_str(table.trim())?;
            match ex {
                Exec::Remote(r) => {
                    let out = r
                        .call(
                            Method::PUT,
                            "/api/studio/index/engines",
                            DispatchBody::Json(json!({ "table_name": vt.as_str(), "engine": engine.as_str() })),
                        )
                        .await?;
                    print_json(&out);
                }
                Exec::Local(l) => {
                    l.require_admin_key()?;
                    let store_mode = if l.state.mode == OpenRagMode::Live {
                        StoreMode::Live
                    } else {
                        StoreMode::Offline
                    };
                    let pinecone_key = non_empty_env("PINECONE_API_KEY");
                    let mut bindings = Vec::new();
                    for org in l.operator_orgs().await? {
                        let b = PgPineconeManager::set_engine(
                            l.worker()?,
                            org,
                            vt,
                            engine.to_engine(),
                            store_mode,
                            pinecone_key.as_deref(),
                        )
                        .await?;
                        bindings.push(b);
                    }
                    let fallback = engine == EngineArg::Remote && pinecone_key.is_none();
                    print_json(&json!({
                        "table_name": vt.as_str(),
                        "engine": engine.as_str(),
                        "mode": if fallback { "fallback" } else { engine.as_str() },
                        "bindings": bindings,
                    }));
                }
            }
        }
        IndexCmd::Mirror { table, to } => {
            let l = require_local(ex, "index mirror")?;
            l.require_admin_key()?;
            let vt = VectorTable::from_str(table.trim())?;
            let mut bindings = Vec::new();
            for org in l.operator_orgs().await? {
                let b = PgPineconeManager::set_external_mirror(l.worker()?, org, vt, to.to_backend(), json!({})).await?;
                bindings.push(b);
            }
            print_json(&json!({ "table_name": vt.as_str(), "bindings": bindings }));
        }
        IndexCmd::Status => {
            let l = require_local(ex, "index status")?;
            l.require_admin_key()?;
            let org = match &l.key_principal {
                Some(p) => Some(p.org_id),
                None => env_uuid("OPENRAG_ORG_ID")?,
            };
            let summary = OutboxWorker::default().status(l.worker()?, org).await?;
            print_json(&serde_json::to_value(&summary).unwrap_or_else(|_| json!({})));
        }
        IndexCmd::Drain { limit } => {
            let l = require_local(ex, "index drain")?;
            l.require_admin_key()?;
            let org = match &l.key_principal {
                Some(p) => Some(p.org_id),
                None => env_uuid("OPENRAG_ORG_ID")?,
            };
            let report = OutboxWorker::default()
                .drain_for_org(l.worker()?, org, limit, &DefaultOutboxExecutor::default())
                .await?;
            print_json(&serde_json::to_value(&report).unwrap_or_else(|_| json!({})));
        }
        IndexCmd::Retry { id, all } => {
            let l = require_local(ex, "index retry")?;
            l.require_admin_key()?;
            let org = match &l.key_principal {
                Some(p) => Some(p.org_id),
                None => env_uuid("OPENRAG_ORG_ID")?,
            };
            let worker = OutboxWorker::default();
            let retried = worker.retry_quarantine(l.worker()?, id, all, org).await?;
            if let Some(qid) = id {
                if retried == 0 {
                    return Err(CliError(format!("quarantine item {qid} not found or already resolved")));
                }
            }
            let drained = worker
                .drain_for_org(l.worker()?, org, 64, &DefaultOutboxExecutor::default())
                .await?;
            print_json(&json!({
                "retried": retried,
                "drained": serde_json::to_value(&drained).unwrap_or_else(|_| json!({})),
            }));
        }
    }
    Ok(())
}

// ============================================================================
// memory
// ============================================================================

async fn cmd_memory(c: MemoryCmd) -> Result<(), CliError> {
    let (memory_id, path, body) = match c {
        MemoryCmd::Patch {
            memory_id,
            op,
            text,
            reason,
            hard_purge,
        } => {
            let mut b = Map::new();
            b.insert("memory_id".into(), json!(memory_id));
            b.insert("op".into(), json!(op.as_str()));
            if let Some(t) = text {
                b.insert("text".into(), json!(t));
            }
            if let Some(r) = reason {
                b.insert("reason".into(), json!(r));
            }
            b.insert("hard_purge".into(), json!(hard_purge));
            (memory_id, "/api/app/memory/patch", Value::Object(b))
        }
        MemoryCmd::Promote { memory_id, to, bucket } => {
            if to == PromoteToArg::Wiki && bucket.as_deref().is_none_or(|b| b.trim().is_empty()) {
                return Err(CliError::new("--to wiki requires --bucket"));
            }
            let mut b = Map::new();
            b.insert("memory_id".into(), json!(memory_id));
            b.insert("to".into(), json!(to.as_str()));
            if let Some(bk) = bucket {
                b.insert("bucket".into(), json!(bk));
            }
            (memory_id, "/api/app/memory/promote", Value::Object(b))
        }
    };

    let out = match exec().await? {
        Exec::Remote(r) => r.call(Method::POST, path, DispatchBody::Json(body)).await?,
        Exec::Local(l) => {
            let principal = match &l.key_principal {
                Some(p) => p.clone(),
                None => {
                    let (org, owner) = l.owner_of("memory.entries", memory_id).await?;
                    l.user_principal(org, owner).await?
                }
            };
            local_call(&l.state, &principal, Method::POST, path, DispatchBody::Json(body)).await?
        }
    };
    print_json(&out);
    Ok(())
}

// ============================================================================
// redact
// ============================================================================

struct RedactTarget {
    scope: &'static str,
    target_ref: Value,
    /// (table, id) used to locate org/owner in operator mode.
    locate: Option<(&'static str, Uuid)>,
}

fn redact_target(a: &RedactRunArgs) -> Result<RedactTarget, CliError> {
    let mut picks: Vec<RedactTarget> = Vec::new();
    if let Some(id) = a.doc {
        picks.push(RedactTarget {
            scope: "document",
            target_ref: json!({ "document_id": id }),
            locate: Some(("corpus.documents", id)),
        });
    }
    if let Some(id) = a.chunk {
        picks.push(RedactTarget {
            scope: "chunk",
            target_ref: json!({ "chunk_id": id }),
            locate: Some(("corpus.chunks", id)),
        });
    }
    if let Some(id) = a.memory {
        picks.push(RedactTarget {
            scope: "memory",
            target_ref: json!({ "memory_entry_id": id }),
            locate: Some(("memory.entries", id)),
        });
    }
    if let Some(id) = a.wiki {
        picks.push(RedactTarget {
            scope: "wiki",
            target_ref: json!({ "wiki_page_id": id }),
            locate: Some(("wiki.pages", id)),
        });
    }
    if let Some(id) = a.vault_entry {
        picks.push(RedactTarget {
            scope: "vault_entry",
            target_ref: json!({ "vault_entry_id": id }),
            locate: None,
        });
    }
    if let Some(p) = a.pattern.as_deref() {
        if p.trim().is_empty() {
            return Err(CliError::new("--pattern must not be empty"));
        }
        let dept_scoped = matches!(a.scope, RedactScopeArg::Dept | RedactScopeArg::PatternDept);
        if dept_scoped {
            let dept = a
                .dept
                .ok_or_else(|| CliError::new("--scope dept pattern redaction requires --dept"))?;
            picks.push(RedactTarget {
                scope: "pattern_dept",
                target_ref: json!({ "pattern": p, "dept_id": dept }),
                locate: None,
            });
        } else {
            picks.push(RedactTarget {
                scope: "pattern_org",
                target_ref: json!({ "pattern": p }),
                locate: None,
            });
        }
    }
    let target = match picks.len() {
        0 => return Err(CliError::new(
            "pass exactly one target: --doc, --chunk, --memory, --wiki, --vault-entry or --pattern",
        )),
        1 => picks.remove(0),
        _ => return Err(CliError::new("pass only one redaction target per job")),
    };
    let consistent = match a.scope {
        RedactScopeArg::Org | RedactScopeArg::Dept => true,
        RedactScopeArg::Document => target.scope == "document",
        RedactScopeArg::Chunk => target.scope == "chunk",
        RedactScopeArg::Memory => target.scope == "memory",
        RedactScopeArg::Wiki => target.scope == "wiki",
        RedactScopeArg::VaultEntry => target.scope == "vault_entry",
        RedactScopeArg::PatternOrg => target.scope == "pattern_org",
        RedactScopeArg::PatternDept => target.scope == "pattern_dept",
    };
    if !consistent {
        return Err(CliError(format!(
            "--scope does not match the target flag (target is {})",
            target.scope
        )));
    }
    Ok(target)
}

async fn cmd_redact(a: RedactRunArgs) -> Result<(), CliError> {
    if a.reason.trim().is_empty() {
        return Err(CliError::new("--reason must not be empty"));
    }
    let target = redact_target(&a)?;
    let mode = if a.tombstone_delete {
        "tombstone_delete"
    } else if a.hard_purge {
        "hard_purge"
    } else {
        "mask_keep_vault"
    };
    let body = json!({
        "target_scope": target.scope,
        "target_ref": target.target_ref,
        "redaction_mode": mode,
        "redact_thumbnails": a.redact_thumbnails,
        "reason": a.reason,
    });

    match exec().await? {
        Exec::Remote(r) => {
            let out = r
                .call(Method::POST, "/api/studio/pii/redactions", DispatchBody::Json(body))
                .await?;
            print_json(&out);
        }
        Exec::Local(l) => {
            let jobs: Vec<(Uuid, Option<Uuid>)> = if let Some(p) = &l.key_principal {
                vec![(p.org_id, None)]
            } else if let Some((table, id)) = target.locate {
                let (org, owner) = l.owner_of(table, id).await?;
                vec![(org, Some(owner))]
            } else if target.scope == "vault_entry" {
                let org = env_uuid("OPENRAG_ORG_ID")?.ok_or_else(|| {
                    CliError::new("vault redaction in operator mode requires OPENRAG_ORG_ID")
                })?;
                vec![(org, None)]
            } else {
                l.operator_orgs().await?.into_iter().map(|o| (o, None)).collect()
            };

            let multi = jobs.len() > 1;
            let mut results = Vec::new();
            let mut failures = Vec::new();
            for (org, owner) in jobs {
                let principal = match l.admin_principal(org, owner).await {
                    Ok(p) => p,
                    Err(e) if multi => {
                        results.push(json!({ "org_id": org, "skipped": e.0 }));
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                match local_call(
                    &l.state,
                    &principal,
                    Method::POST,
                    "/api/studio/pii/redactions",
                    DispatchBody::Json(body.clone()),
                )
                .await
                {
                    Ok(v) => results.push(json!({
                        "org_id": org,
                        "attributed_to": principal.principal_id,
                        "operator_mode": l.is_operator(),
                        "job": v,
                    })),
                    Err(e) => {
                        failures.push(format!("org {org}: {}", e.0));
                        results.push(json!({ "org_id": org, "error": e.0 }));
                    }
                }
            }
            print_json(&json!({ "jobs": results }));
            if !failures.is_empty() {
                return Err(CliError(failures.join("; ")));
            }
        }
    }
    Ok(())
}

// ============================================================================
// wiki
// ============================================================================

/// Copies the vault's markdown into a private temp dir with literal `[PII:...]` tokens
/// neutralized (write-time laundering guard). Symlinks are followed only when they
/// resolve inside the vault; anything escaping it is skipped and reported.
fn neutralized_copy(src_root: &Path, dst_root: &Path) -> Result<Vec<String>, CliError> {
    let canon_root = src_root
        .canonicalize()
        .map_err(|e| CliError(format!("invalid wiki directory '{}': {e}", src_root.display())))?;
    let mut skipped = Vec::new();
    let mut stack: Vec<PathBuf> = vec![canon_root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let meta = std::fs::symlink_metadata(&path)?;
            let resolved = if meta.file_type().is_symlink() {
                match path.canonicalize() {
                    Ok(t) if t.starts_with(&canon_root) => t,
                    _ => {
                        skipped.push(path.display().to_string());
                        continue;
                    }
                }
            } else {
                path.clone()
            };
            let rmeta = std::fs::metadata(&resolved)?;
            let rel = path
                .strip_prefix(&canon_root)
                .map_err(|e| CliError(format!("path outside vault: {e}")))?;
            if rmeta.is_dir() {
                if meta.file_type().is_symlink() {
                    // Avoid cycles: do not descend through directory symlinks.
                    skipped.push(path.display().to_string());
                    continue;
                }
                std::fs::create_dir_all(dst_root.join(rel))?;
                stack.push(path);
            } else if rmeta.is_file()
                && resolved
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("md"))
            {
                let raw = std::fs::read_to_string(&resolved)?;
                let dst = dst_root.join(rel);
                if let Some(parent) = dst.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(dst, neutralize_literal_pii_tokens(&raw))?;
            }
        }
    }
    Ok(skipped)
}

async fn cmd_wiki(c: WikiCmd) -> Result<(), CliError> {
    let l = require_local(exec().await?, "wiki")?;
    match c {
        WikiCmd::Import {
            dir,
            bucket,
            scope,
            auth_level,
        } => {
            let (org, bucket_id, owner) = l.find_bucket(&bucket).await?;
            let principal = l.user_principal(org, owner).await?;
            let staging = std::env::temp_dir().join(format!("openrag-wiki-import-{}", Uuid::new_v4().simple()));
            std::fs::create_dir_all(&staging)?;
            let result = async {
                let skipped = neutralized_copy(&dir, &staging)?;
                let app_pool = l.state.require_app_pool()?;
                let mut tx = app_pool.begin_rls_tx(&principal).await?;
                let rec = resolve_bucket_rls(&mut tx, org, &bucket_id.to_string()).await?;
                let (org_slug, dept_path) = resolve_org_and_dept_paths(&l.state, org, rec.dept_id).await?;
                let default_scope = match scope {
                    Some(s) => s.to_access(),
                    None => AccessScope::from_str(&rec.access_scope)?,
                };
                let default_level = AuthLevel::try_from(auth_level.unwrap_or(rec.min_auth_level))?;
                let report = MarkdownSync::import_markdown_dir_rls(
                    &mut tx,
                    &principal,
                    &staging,
                    Some(bucket_id),
                    rec.dept_id,
                    &org_slug,
                    &dept_path,
                    default_scope,
                    default_level,
                    l.state.text_embedder.as_ref(),
                )
                .await?;
                tx.commit().await?;
                Ok::<Value, CliError>(json!({
                    "bucket_id": bucket_id,
                    "report": serde_json::to_value(&report).unwrap_or_else(|_| json!({})),
                    "skipped_unsafe_paths": skipped,
                }))
            }
            .await;
            let _ = std::fs::remove_dir_all(&staging);
            print_json(&result?);
        }
        WikiCmd::Export { bucket, out } => {
            let (org, bucket_id, owner) = l.find_bucket(&bucket).await?;
            let principal = l.user_principal(org, owner).await?;
            let app_pool = l.state.require_app_pool()?;
            let mut tx = app_pool.begin_rls_tx(&principal).await?;
            let files = MarkdownSync::export_markdown_dir_rls(&mut tx, org, Some(bucket_id), &out).await?;
            tx.commit().await?;
            let listed: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
            print_json(&json!({ "bucket_id": bucket_id, "exported": listed.len(), "files": listed }));
        }
    }
    Ok(())
}

// ============================================================================
// concept
// ============================================================================

async fn concept_call(ex: &Exec, method: Method, path: &str, body: DispatchBody) -> Result<Value, CliError> {
    match ex {
        Exec::Remote(r) => r.call(method, path, body).await,
        Exec::Local(l) => {
            let principal = match &l.key_principal {
                Some(p) => p.clone(),
                None => {
                    let org = env_uuid("OPENRAG_ORG_ID")?.ok_or_else(|| {
                        CliError::new("concept commands need OPENRAG_API_KEY, or OPENRAG_ORG_ID in operator mode")
                    })?;
                    l.admin_principal(org, None).await?
                }
            };
            local_call(&l.state, &principal, method, path, body).await
        }
    }
}

fn find_alias_id(resolved: &Value, alias_text: &str) -> Option<Uuid> {
    let want = alias_text.trim().to_lowercase();
    let aliases = resolved
        .get("aliases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    aliases.iter().find_map(|a| {
        let text = a
            .get("alias_text")
            .or_else(|| a.get("alias"))
            .and_then(Value::as_str)?;
        if text.trim().to_lowercase() == want {
            a.get("id").and_then(Value::as_str).and_then(|s| Uuid::parse_str(s).ok())
        } else {
            None
        }
    })
}

async fn cmd_concept(c: ConceptCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    match c {
        ConceptCmd::Link {
            source,
            target,
            canonical,
            lang,
            relation,
            concept_id,
        } => {
            if source.trim().is_empty() || target.trim().is_empty() {
                return Err(CliError::new("source and target must not be empty"));
            }
            if source.trim().to_lowercase() == target.trim().to_lowercase() {
                return Err(CliError::new("cannot link a term to itself"));
            }
            let mut b = Map::new();
            b.insert("source".into(), json!(source.trim()));
            b.insert("target".into(), json!(target.trim()));
            if let Some(v) = canonical {
                b.insert("canonical_name".into(), json!(v));
            }
            if let Some(v) = lang {
                b.insert("lang".into(), json!(v));
            }
            if let Some(v) = relation {
                b.insert("relation".into(), json!(v));
            }
            if let Some(v) = concept_id {
                b.insert("concept_id".into(), json!(v));
            }
            let out = concept_call(&ex, Method::POST, "/v1/concepts/link", DispatchBody::Json(Value::Object(b))).await?;
            print_json(&out);
        }
        ConceptCmd::Unlink {
            source,
            target,
            concept_id,
            alias_id,
        } => {
            let out = match (concept_id, alias_id) {
                (Some(cid), Some(aid)) => {
                    concept_call(&ex, Method::DELETE, &format!("/v1/concepts/{cid}/aliases/{aid}"), DispatchBody::Empty).await?
                }
                (Some(cid), None) if source.is_none() => {
                    concept_call(&ex, Method::DELETE, &format!("/v1/concepts/{cid}"), DispatchBody::Empty).await?
                }
                _ => {
                    let source = source
                        .filter(|s| !s.trim().is_empty())
                        .ok_or_else(|| CliError::new("pass <source> [<target>], or --concept-id [--alias-id]"))?;
                    let resolved = concept_call(
                        &ex,
                        Method::POST,
                        "/v1/concepts/resolve",
                        DispatchBody::Json(json!({ "term": source.trim(), "include_aliases": true })),
                    )
                    .await?;
                    let cid = concept_id
                        .or_else(|| {
                            resolved
                                .get("concept_id")
                                .and_then(Value::as_str)
                                .and_then(|s| Uuid::parse_str(s).ok())
                        })
                        .ok_or_else(|| CliError(format!("no concept found for '{}'", source.trim())))?;
                    let alias_text = target.as_deref().unwrap_or(&source);
                    let aid = find_alias_id(&resolved, alias_text)
                        .ok_or_else(|| CliError(format!("alias '{}' not found on concept {cid}", alias_text.trim())))?;
                    concept_call(&ex, Method::DELETE, &format!("/v1/concepts/{cid}/aliases/{aid}"), DispatchBody::Empty).await?
                }
            };
            print_json(&out);
        }
        ConceptCmd::List {
            query,
            concept_id,
            lang,
            json: _,
        } => {
            let out = if let Some(cid) = concept_id {
                concept_call(&ex, Method::GET, &format!("/v1/concepts/{cid}"), DispatchBody::Empty).await?
            } else {
                let mut qs = Vec::new();
                if let Some(q) = query.filter(|q| !q.trim().is_empty()) {
                    qs.push(format!("query={}", pct_encode(q.trim())));
                }
                if let Some(l) = lang.filter(|l| !l.trim().is_empty()) {
                    qs.push(format!("lang={}", pct_encode(l.trim())));
                }
                let path = if qs.is_empty() {
                    "/v1/concepts".to_string()
                } else {
                    format!("/v1/concepts?{}", qs.join("&"))
                };
                concept_call(&ex, Method::GET, &path, DispatchBody::Empty).await?
            };
            print_json(&out);
        }
    }
    Ok(())
}

// ============================================================================
// bench / pipeline
// ============================================================================

async fn admin_exec_principal(l: &Local) -> Result<PrincipalContext, CliError> {
    match &l.key_principal {
        Some(p) => Ok(p.clone()),
        None => {
            let org = env_uuid("OPENRAG_ORG_ID")?.ok_or_else(|| {
                CliError::new("operator mode needs OPENRAG_ORG_ID (or pass OPENRAG_API_KEY)")
            })?;
            l.admin_principal(org, None).await
        }
    }
}

fn golden_set_fields(raw: &str, body: &mut Map<String, Value>) {
    match Uuid::parse_str(raw.trim()) {
        Ok(id) => {
            body.insert("golden_set_id".into(), json!(id));
        }
        Err(_) => {
            body.insert("golden_set".into(), json!(raw.trim()));
        }
    }
}

async fn read_text(path: &Path, what: &str) -> Result<String, CliError> {
    tokio::fs::read_to_string(path)
        .await
        .map_err(|e| CliError(format!("cannot read {what} {}: {e}", path.display())))
}

async fn studio_call(ex: &Exec, path: &str, body: Value) -> Result<Value, CliError> {
    match ex {
        Exec::Remote(r) => r.call(Method::POST, path, DispatchBody::Json(body)).await,
        Exec::Local(l) => {
            let principal = admin_exec_principal(l).await?;
            local_call(&l.state, &principal, Method::POST, path, DispatchBody::Json(body)).await
        }
    }
}

async fn cmd_bench(c: BenchCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    match c {
        BenchCmd::Run {
            golden_set,
            pipeline,
            bucket,
        } => {
            let mut b = Map::new();
            golden_set_fields(&golden_set, &mut b);
            if let Some(p) = pipeline {
                b.insert("pipeline_yaml".into(), json!(read_text(&p, "pipeline").await?));
            }
            if let Some(bid) = bucket {
                b.insert("bucket_id".into(), json!(bid));
            }
            let out = studio_call(&ex, "/api/studio/eval/bench", Value::Object(b)).await?;
            print_json(&out);
        }
        BenchCmd::Optimize {
            golden_set,
            search_space,
            pipeline,
            strategy,
            bucket,
            require_zero_pii_leak,
            require_zero_arn_violation,
            export_best,
        } => {
            let mut b = Map::new();
            golden_set_fields(&golden_set, &mut b);
            b.insert("search_space_yaml".into(), json!(read_text(&search_space, "search space").await?));
            if let Some(p) = pipeline {
                b.insert("base_pipeline_yaml".into(), json!(read_text(&p, "pipeline").await?));
            }
            if let Some(s) = strategy {
                b.insert("strategy".into(), json!(s));
            }
            if let Some(bid) = bucket {
                b.insert("bucket_id".into(), json!(bid));
            }
            b.insert("require_zero_pii_leak".into(), json!(require_zero_pii_leak));
            b.insert("require_zero_arn_violation".into(), json!(require_zero_arn_violation));
            let report = studio_call(&ex, "/api/studio/eval/optimize", Value::Object(b)).await?;
            print_json(&report);

            let gates = report.get("security_gates").cloned().unwrap_or(Value::Null);
            let pii = gates.get("pii_leak_count").and_then(Value::as_i64).unwrap_or(0);
            let arn = gates.get("arn_violation_count").and_then(Value::as_i64).unwrap_or(0);
            if pii > 0 || arn > 0 {
                return Err(CliError(format!(
                    "security gate failed (pii_leak_count={pii}, arn_violation_count={arn}); best config not exported"
                )));
            }
            if let Some(out) = export_best {
                let yaml = report
                    .get("best_config_yaml")
                    .and_then(Value::as_str)
                    .ok_or_else(|| CliError::new("optimizer produced no best config to export"))?;
                tokio::fs::write(&out, yaml)
                    .await
                    .map_err(|e| CliError(format!("cannot write {}: {e}", out.display())))?;
                eprintln!("exported best pipeline to {}", out.display());
            }
        }
    }
    Ok(())
}

async fn cmd_pipeline(c: PipelineCmd) -> Result<(), CliError> {
    let PipelineCmd::Promote { yaml_path, bucket } = c;
    let l = require_local(exec().await?, "pipeline promote")?;
    let yaml = read_text(&yaml_path, "pipeline").await?;
    let principal = match &l.key_principal {
        Some(p) => p.clone(),
        None => match env_uuid("OPENRAG_ORG_ID")? {
            Some(org) => l.admin_principal(org, None).await?,
            None => {
                let (org, _, owner) = l.find_bucket(&bucket).await?;
                l.admin_principal(org, Some(owner)).await?
            }
        },
    };
    let name = yaml_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("pipeline")
        .to_string();
    let store = Arc::new(l.state.pg_eval_store()?);
    let promoter = PipelinePromoter::new(store);
    let record = promoter
        .register_config(principal.org_id, &name, &yaml, Some(principal.principal_id))
        .await?;
    let promoted = promoter
        .promote_config(
            principal.org_id,
            &principal,
            PromotePipelineInput {
                pipeline_config_id: Some(record.id),
                bench_run_id: None,
                bucket_ref: Some(bucket),
            },
        )
        .await?;
    print_json(&serde_json::to_value(&promoted).unwrap_or_else(|_| json!({})));
    Ok(())
}

// ============================================================================
// auth
// ============================================================================

async fn cmd_auth(c: AuthCmd) -> Result<(), CliError> {
    let AuthCmd::Code {
        principal,
        org,
        ttl,
        max_auth_level,
        scoped_arns,
        scoped_groups,
    } = c;
    let mut body = Map::new();
    body.insert("principal_id".into(), json!(principal));
    if let Some(t) = ttl {
        body.insert("ttl_seconds".into(), json!(t));
    }
    if let Some(l) = max_auth_level {
        body.insert("max_auth_level".into(), json!(l));
    }
    body.insert("scoped_arns".into(), json!(scoped_arns));
    body.insert("scoped_group_ids".into(), json!(scoped_groups));
    let body = Value::Object(body);

    let out = match exec().await? {
        Exec::Remote(r) => r.call(Method::POST, "/api/studio/iam/cli-codes", DispatchBody::Json(body)).await?,
        Exec::Local(l) => match &l.key_principal {
            Some(p) => {
                local_call(&l.state, p, Method::POST, "/api/studio/iam/cli-codes", DispatchBody::Json(body)).await?
            }
            None => {
                let org_id = match org.or(env_uuid("OPENRAG_ORG_ID")?) {
                    Some(o) => o,
                    None => sqlx::query_scalar("SELECT org_id FROM iam.principals WHERE id = $1")
                        .bind(principal)
                        .fetch_optional(l.worker()?.inner())
                        .await
                        .map_err(|e| CliError(format!("lookup principal: {e}")))?
                        .ok_or_else(|| CliError(format!("principal {principal} not found")))?,
                };
                let issued = issue_cli_code(
                    l.worker()?,
                    org_id,
                    principal,
                    None,
                    CliCodeCaps {
                        max_auth_level,
                        scoped_arns,
                        scoped_group_ids: scoped_groups,
                    },
                    ttl,
                )
                .await?;
                serde_json::to_value(&issued).unwrap_or_else(|_| json!({}))
            }
        },
    };
    print_json(&out);
    Ok(())
}

// ============================================================================
// org / bucket / connect / init
// ============================================================================

fn valid_path_segment(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

fn db_err(what: &'static str) -> impl Fn(sqlx::Error) -> CliError {
    move |e| CliError(format!("{what}: {e}"))
}

/// Server state for operator-only bootstrap work; ignores any API key in the environment.
async fn operator_state(what: &str) -> Result<Arc<AppState>, CliError> {
    if non_empty_env("DATABASE_URL").is_none() {
        return Err(CliError(format!(
            "`{what}` needs DATABASE_URL (the server's database, e.g. postgres://openrag:…@127.0.0.1:5432/openrag)"
        )));
    }
    if !operator_enabled() {
        return Err(CliError(format!(
            "`{what}` runs as the local operator: set OPENRAG_OPERATOR=1 (or `operator = true` in config.toml)"
        )));
    }
    Ok(AppState::from_env().await?)
}

struct Bootstrapped {
    org_id: Uuid,
    dept_id: Option<Uuid>,
    principal_id: Uuid,
    api_key: String,
    created: bool,
}

/// Creates the organization, its first department and an admin (or, with `reuse`, takes the
/// existing organization's first admin), then mints an API key for that admin.
async fn bootstrap_org(spec: &OrgSpec, reuse: bool, what: &str) -> Result<Bootstrapped, CliError> {
    for (flag, v) in [("--slug", &spec.slug), ("--dept", &spec.dept)] {
        if !valid_path_segment(v) {
            return Err(CliError(format!("{flag} must use a-z, 0-9, '-' or '_' (got '{v}')")));
        }
    }
    let state = operator_state(what).await?;
    let worker = state.require_worker_pool()?;
    let existing: Option<Uuid> = sqlx::query_scalar("SELECT id FROM iam.orgs WHERE slug = $1")
        .bind(&spec.slug)
        .fetch_optional(worker.inner())
        .await
        .map_err(db_err("lookup iam.orgs"))?;
    let (org_id, dept_id, principal_id, created) = match existing {
        Some(org_id) => {
            if !reuse {
                return Err(CliError(format!(
                    "organization '{}' already exists; mint a key for one of its admins with `openrag auth code --principal <id>`",
                    spec.slug
                )));
            }
            let admin: Option<Uuid> = sqlx::query_scalar(
                "SELECT id FROM iam.principals
                 WHERE org_id = $1 AND is_org_admin AND status = 'active'
                 ORDER BY created_at, id LIMIT 1",
            )
            .bind(org_id)
            .fetch_optional(worker.inner())
            .await
            .map_err(db_err("lookup org admin"))?;
            let admin = admin.ok_or_else(|| {
                CliError(format!("organization '{}' has no active admin", spec.slug))
            })?;
            (org_id, None, admin, false)
        }
        None => {
            let (org_id, dept_id, admin_id) = (Uuid::now_v7(), Uuid::now_v7(), Uuid::now_v7());
            let mut tx = worker.inner().begin().await.map_err(db_err("begin bootstrap"))?;
            sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, $3)")
                .bind(org_id)
                .bind(&spec.slug)
                .bind(spec.name.trim())
                .execute(&mut *tx)
                .await
                .map_err(db_err("insert iam.orgs"))?;
            sqlx::query("INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, $3, $4)")
                .bind(dept_id)
                .bind(org_id)
                .bind(spec.dept_name.as_deref().unwrap_or(&spec.dept).trim())
                .bind(&spec.dept)
                .execute(&mut *tx)
                .await
                .map_err(db_err("insert iam.departments"))?;
            sqlx::query(
                "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, is_org_admin)
                 VALUES ($1, $2, $3, $4, 10, true)",
            )
            .bind(admin_id)
            .bind(org_id)
            .bind(spec.admin_email.trim())
            .bind(spec.admin_name.trim())
            .execute(&mut *tx)
            .await
            .map_err(db_err("insert iam.principals"))?;
            sqlx::query(
                "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
                 VALUES ($1, $2, $3, 'dept_admin')",
            )
            .bind(org_id)
            .bind(dept_id)
            .bind(admin_id)
            .execute(&mut *tx)
            .await
            .map_err(db_err("insert iam.dept_memberships"))?;
            tx.commit().await.map_err(db_err("commit bootstrap"))?;
            (org_id, Some(dept_id), admin_id, true)
        }
    };
    let issued = issue_cli_code(worker, org_id, principal_id, None, CliCodeCaps::default(), None).await?;
    let (_, axum::Json(minted)) = crate::auth::cli_code_exchange(
        axum::extract::State(Arc::clone(&state)),
        axum::Json(crate::auth::CliCodeExchangeRequest {
            code: issued.code,
            org_id: None,
            principal_id: None,
            name: Some("openrag-init".into()),
            max_auth_level: None,
            scoped_arns: Vec::new(),
            scoped_group_ids: Vec::new(),
            ttl_seconds: None,
        }),
    )
    .await
    .map_err(|e| CliError(format!("mint API key: {} {}", e.code, e.message)))?;
    Ok(Bootstrapped {
        org_id,
        dept_id,
        principal_id,
        api_key: minted.api_key,
        created,
    })
}

/// Sets keys in the CLI config file (OPENRAG_CONFIG, else ~/.config/openrag/config.toml),
/// keeping the others, and leaves it readable by the owner only.
fn save_cli_config(values: &[(&str, String)]) -> Result<PathBuf, CliError> {
    let path = non_empty_env("OPENRAG_CONFIG")
        .map(PathBuf::from)
        .or_else(default_config_path)
        .ok_or_else(|| CliError::new("no config path: set OPENRAG_CONFIG or HOME"))?;
    let mut table: toml::Table = match std::fs::read_to_string(&path) {
        Ok(raw) => toml::from_str(&raw).map_err(|e| CliError(format!("invalid config {}: {e}", path.display())))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => toml::Table::new(),
        Err(e) => return Err(CliError(format!("cannot read config {}: {e}", path.display()))),
    };
    for (k, v) in values {
        table.insert((*k).to_string(), toml::Value::String(v.clone()));
    }
    let text = toml::to_string(&table).map_err(|e| CliError(format!("encode config: {e}")))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)?;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        f.write_all(text.as_bytes())?;
    }
    #[cfg(not(unix))]
    std::fs::write(&path, text)?;
    Ok(path)
}

fn mask_key(key: &str) -> String {
    format!("{}…", key.chars().take(10).collect::<String>())
}

/// Calls the API as the caller: the API key's principal, or (operator mode) the only org's admin.
async fn call_as_caller(ex: &Exec, method: Method, path: &str, body: DispatchBody) -> Result<Value, CliError> {
    match ex {
        Exec::Remote(r) => r.call(method, path, body).await,
        Exec::Local(l) => {
            let principal = match &l.key_principal {
                Some(p) => p.clone(),
                None => {
                    let orgs = l.operator_orgs().await?;
                    let [org] = orgs.as_slice() else {
                        return Err(CliError::new("several organizations: set OPENRAG_ORG_ID or use an API key"));
                    };
                    l.admin_principal(*org, None).await?
                }
            };
            local_call(&l.state, &principal, method, path, body).await
        }
    }
}

async fn cmd_org_init(spec: OrgSpec, save: bool, api_url: Option<String>) -> Result<(), CliError> {
    let b = bootstrap_org(&spec, false, "org init").await?;
    let mut out = json!({
        "org_id": b.org_id,
        "slug": spec.slug,
        "dept_id": b.dept_id,
        "admin_principal_id": b.principal_id,
        "admin_email": spec.admin_email,
        "api_key": b.api_key,
    });
    if save {
        let mut values = vec![
            ("api_key", b.api_key.clone()),
            ("org_id", b.org_id.to_string()),
            ("principal_id", b.principal_id.to_string()),
        ];
        if let Some(u) = api_url {
            values.push(("api_url", u.trim_end_matches('/').to_string()));
        }
        let path = save_cli_config(&values)?;
        out["config"] = json!(path.display().to_string());
    }
    print_json(&out);
    Ok(())
}

async fn cmd_bucket(c: BucketCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    let out = match c {
        BucketCmd::Create { slug, name, scope, auth_level, dept } => {
            let body = json!({
                "name": name.unwrap_or_else(|| slug.clone()),
                "slug": slug,
                "access_scope": scope.as_str(),
                "min_auth_level": auth_level,
                "dept_id": dept,
            });
            call_as_caller(&ex, Method::POST, "/v2/buckets", DispatchBody::Json(body)).await?
        }
        BucketCmd::List => call_as_caller(&ex, Method::GET, "/v2/buckets", DispatchBody::Empty).await?,
    };
    print_json(&out);
    Ok(())
}

async fn cmd_docs(c: DocsCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    let out = match c {
        DocsCmd::List { bucket, query, all, limit, offset } => {
            let mut path = format!("/v2/buckets/{}/documents?limit={limit}&offset={offset}", pct_encode(bucket.trim()));
            if let Some(q) = query.filter(|q| !q.trim().is_empty()) {
                path.push_str(&format!("&q={}", pct_encode(q.trim())));
            }
            if all {
                path.push_str("&include_closed=true");
            }
            call_as_caller(&ex, Method::GET, &path, DispatchBody::Empty).await?
        }
        DocsCmd::Status => call_as_caller(&ex, Method::GET, "/v2/watch", DispatchBody::Empty).await?,
        DocsCmd::Scan => call_as_caller(&ex, Method::POST, "/v2/watch/scan", DispatchBody::Empty).await?,
    };
    print_json(&out);
    Ok(())
}

async fn cmd_key(c: KeyCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    let out = match c {
        KeyCmd::List { principal, all } => {
            let mut path = String::from("/api/studio/iam/api-keys?include_revoked=");
            path.push_str(if all { "true" } else { "false" });
            if let Some(p) = principal {
                path.push_str(&format!("&principal_id={p}"));
            }
            call_as_caller(&ex, Method::GET, &path, DispatchBody::Empty).await?
        }
        KeyCmd::Revoke { id } => {
            call_as_caller(&ex, Method::POST, &format!("/api/studio/iam/api-keys/{id}/revoke"), DispatchBody::Empty).await?
        }
    };
    print_json(&out);
    Ok(())
}

async fn cmd_member(c: MemberCmd) -> Result<(), CliError> {
    let ex = exec().await?;
    match c {
        MemberCmd::List => {
            let out = call_as_caller(&ex, Method::GET, "/api/studio/iam/principals", DispatchBody::Empty).await?;
            print_json(&out);
        }
        MemberCmd::Add { email, name, level, dept, key_name, public_url } => {
            let dept = match dept {
                Some(d) => Some(d),
                None => {
                    let members = call_as_caller(&ex, Method::GET, "/api/studio/iam/principals", DispatchBody::Empty).await?;
                    members
                        .get("principals")
                        .and_then(Value::as_array)
                        .and_then(|ps| ps.iter().find(|p| p.get("is_org_admin").and_then(Value::as_bool) == Some(true)))
                        .and_then(|p| p.get("dept_ids").and_then(Value::as_array))
                        .and_then(|ids| ids.first())
                        .and_then(Value::as_str)
                        .and_then(|s| Uuid::parse_str(s).ok())
                }
            };
            let body = json!({
                "email": email.trim(),
                "display_name": name.trim(),
                "auth_level": level,
                "dept_ids": dept.into_iter().collect::<Vec<_>>(),
            });
            let created = call_as_caller(&ex, Method::POST, "/api/studio/iam/principals", DispatchBody::Json(body)).await?;
            let principal_id = created
                .get("id")
                .or_else(|| created.get("principal_id"))
                .and_then(Value::as_str)
                .ok_or_else(|| CliError::new("the server did not return the new member's id"))?
                .to_string();
            let key_body = json!({ "principal_id": principal_id, "name": key_name.unwrap_or_else(|| email.trim().to_string()) });
            let key = call_as_caller(&ex, Method::POST, "/api/studio/iam/api-keys", DispatchBody::Json(key_body)).await?;
            let api_key = key
                .get("api_key")
                .and_then(Value::as_str)
                .ok_or_else(|| CliError::new("the server did not return the new key"))?
                .to_string();
            let public = public_url
                .map(|u| u.trim_end_matches('/').to_string())
                .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
            print_json(&json!({
                "member": { "id": principal_id, "email": email.trim(), "name": name.trim(), "auth_level": level, "dept_id": dept },
                "api_key": api_key,
                "key_id": key.get("id").or_else(|| key.get("key_id")).cloned().unwrap_or(Value::Null),
                "connect": connect_commands(&public, &api_key),
                "next": "Send the member one connect line (it holds their key) to run on their machine. They see only documents their level and departments allow. Stop the key with `openrag key revoke <key_id>`.",
            }));
        }
    }
    Ok(())
}

/// Registers `base`/mcp in an MCP client under `name`.
async fn connect_client(
    client: McpClient,
    name: &str,
    scope: ClaudeScope,
    base: &str,
    key: &str,
    dry_run: bool,
) -> Result<Value, CliError> {
    let mcp_url = format!("{}/mcp", base.trim_end_matches('/'));
    let (program, args, note): (&str, Vec<String>, Option<&str>) = match client {
        McpClient::Claude => (
            "claude",
            vec![
                "mcp".into(),
                "add".into(),
                "--transport".into(),
                "http".into(),
                "--scope".into(),
                scope.as_str().into(),
                name.into(),
                mcp_url.clone(),
                "--header".into(),
                format!("Authorization: Bearer {key}"),
            ],
            None,
        ),
        McpClient::Codex => (
            "codex",
            vec![
                "mcp".into(),
                "add".into(),
                name.into(),
                "--url".into(),
                mcp_url.clone(),
                "--bearer-token-env-var".into(),
                "OPENRAG_API_KEY".into(),
            ],
            Some("Codex reads the key from $OPENRAG_API_KEY: export it (api_key in the CLI config) in the shell that starts codex"),
        ),
    };
    let client_name = match client {
        McpClient::Claude => "claude",
        McpClient::Codex => "codex",
    };
    let shown: Vec<String> = std::iter::once(program.to_string())
        .chain(args.iter().map(|a| a.replace(key, &mask_key(key))))
        .collect();
    if dry_run {
        return Ok(json!({ "client": client_name, "command": shown, "note": note }));
    }
    let out = tokio::process::Command::new(program)
        .args(&args)
        .output()
        .await
        .map_err(|e| CliError(format!("cannot run {program}: {e} (is it installed and on PATH?)")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let msg = if err.trim().is_empty() { String::from_utf8_lossy(&out.stdout) } else { err };
        return Err(CliError(format!("{program} mcp add failed: {}", msg.trim().replace(key, &mask_key(key)))));
    }
    Ok(json!({ "client": client_name, "registered": name, "url": mcp_url, "command": shown, "note": note }))
}

async fn cmd_connect(a: ConnectArgs) -> Result<(), CliError> {
    let base = non_empty_env("OPENRAG_API_URL")
        .ok_or_else(|| CliError::new("set api_url in config.toml (or OPENRAG_API_URL); `openrag init` writes it"))?;
    let key = non_empty_env("OPENRAG_API_KEY")
        .ok_or_else(|| CliError::new("set api_key in config.toml (or OPENRAG_API_KEY); `openrag init` writes it"))?;
    print_json(&connect_client(a.client, &a.name, a.scope, &base, key.trim(), a.dry_run).await?);
    Ok(())
}

async fn cmd_init(a: InitArgs) -> Result<(), CliError> {
    let base = a.api_url.trim_end_matches('/').to_string();
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| CliError(format!("http client: {e}")))?;
    // 1. The server must answer.
    let mut up = false;
    for _ in 0..30 {
        if let Ok(r) = http.get(format!("{base}/health")).send().await {
            if r.status().is_success() {
                up = true;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
    if !up {
        return Err(CliError(format!(
            "no OpenRAG server answers at {base}/health; start it with `openrag up` (or `openrag serve`) first"
        )));
    }
    // 2. Organization, department, admin and the admin's API key (reused on a second run).
    let b = bootstrap_org(&a.org, true, "init").await?;
    let public = a
        .public_url
        .as_deref()
        .map(|u| u.trim_end_matches('/').to_string())
        .unwrap_or_else(|| base.clone());
    // 3. CLI config, so later commands (ingest, bucket, connect) need no flags. Inside the server
    // container there may be no writable home; the setup goes on without it.
    let config = match save_cli_config(&[
        ("api_url", base.clone()),
        ("api_key", b.api_key.clone()),
        ("org_id", b.org_id.to_string()),
        ("principal_id", b.principal_id.to_string()),
    ]) {
        Ok(path) => json!(path.display().to_string()),
        Err(e) => json!({ "not_saved": e.0 }),
    };
    let remote = Exec::Remote(Remote {
        base: base.clone(),
        key: b.api_key.clone(),
        http,
    });
    // 4. A bucket for the documents.
    let body = json!({ "slug": a.bucket, "name": a.bucket, "access_scope": "org", "min_auth_level": 1 });
    let bucket = match call_as_caller(&remote, Method::POST, "/v2/buckets", DispatchBody::Json(body)).await {
        Ok(v) => json!({ "slug": a.bucket, "id": v.get("id").cloned().unwrap_or(Value::Null), "created": true }),
        Err(e) if e.0.contains("HTTP 409") => json!({ "slug": a.bucket, "created": false }),
        Err(e) => return Err(e),
    };
    // 5. Documents.
    let documents = match &a.docs {
        Some(dir) => {
            let args = IngestArgs {
                path: dir.clone(),
                bucket: a.bucket.clone(),
                scope: Some(ScopeArg::Org),
                auth_level: Some(1),
                groups: Vec::new(),
                tags: Vec::new(),
                mode: None,
                title: None,
                pipeline: None,
                valid_from: None,
                valid_to: None,
                revision: None,
            };
            ingest_many(&remote, &args).await?
        }
        None => Value::Null,
    };
    // 6. MCP clients: registered here with --connect, otherwise the commands to run where they are.
    let mut clients = Vec::new();
    for c in &a.connect {
        clients.push(match connect_client(*c, "openrag", a.mcp_scope, &public, &b.api_key, false).await {
            Ok(v) => v,
            Err(e) => json!({ "client": format!("{c:?}").to_lowercase(), "error": e.0 }),
        });
    }
    let (api_key, connect, next) = if a.connect.is_empty() {
        (
            json!(b.api_key),
            connect_commands(&public, &b.api_key),
            "Run connect.claude (or connect.codex) on the machine with Claude Code (or Codex), then ask it about the documents. Keep the key: running init again mints a new one.",
        )
    } else {
        (
            json!(mask_key(&b.api_key)),
            json!(clients),
            "Ask your MCP client about the documents; it searches them with rag_search. More files: openrag ingest <file or folder> --bucket <bucket>",
        )
    };
    print_json(&json!({
        "server": public,
        "org": { "id": b.org_id, "slug": a.org.slug, "created": b.created, "dept_id": b.dept_id },
        "admin": { "principal_id": b.principal_id, "email": a.org.admin_email },
        "api_key": api_key,
        "config": config,
        "bucket": bucket,
        "documents": documents,
        "connect": connect,
        "next": next,
    }));
    Ok(())
}

/// Shell commands that register the server in Claude Code and Codex, for the machine they run on.
fn connect_commands(public: &str, key: &str) -> Value {
    let mcp = format!("{}/mcp", public.trim_end_matches('/'));
    json!({
        "claude": format!("claude mcp add --transport http --scope user openrag {mcp} --header 'Authorization: Bearer {key}'"),
        "codex": format!("export OPENRAG_API_KEY='{key}' && codex mcp add openrag --url {mcp} --bearer-token-env-var OPENRAG_API_KEY"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn auth_level_bounds_are_enforced_by_parser() {
        for bad in ["0", "11"] {
            let r = Cli::try_parse_from(["openrag", "ingest", "d.txt", "--bucket", "b", "--auth-level", bad]);
            assert!(r.is_err(), "auth level {bad} must be rejected");
        }
        assert!(Cli::try_parse_from(["openrag", "ingest", "d.txt", "--bucket", "b", "--auth-level", "10"]).is_ok());
    }

    #[test]
    fn datetime_args_accept_rfc3339_and_plain_dates() {
        let d = parse_datetime_arg("2026-03-01T09:00:00+09:00").map(|d| d.to_rfc3339());
        assert_eq!(d.ok().as_deref(), Some("2026-03-01T00:00:00+00:00"));
        let d = parse_datetime_arg("2026-03-01").map(|d| d.to_rfc3339());
        assert_eq!(d.ok().as_deref(), Some("2026-03-01T00:00:00+00:00"));
        assert!(parse_datetime_arg("03/01/2026").is_err());
        assert!(Cli::try_parse_from(["openrag", "ingest", "d.txt", "--bucket", "b", "--revision", "0"]).is_err());
    }

    #[test]
    fn unknown_patch_op_and_engine_are_rejected() {
        let id = Uuid::new_v4().to_string();
        assert!(Cli::try_parse_from(["openrag", "memory", "patch", &id, "--op", "overwrite"]).is_err());
        assert!(Cli::try_parse_from(["openrag", "index", "engine", "corpus.chunks", "--set", "elastic"]).is_err());
        assert!(Cli::try_parse_from(["openrag", "up", "--profile", "cloud"]).is_err());
        assert!(Cli::try_parse_from(["openrag", "index", "mirror", "corpus.chunks", "--to", "s3-vectors"]).is_ok());
    }

    #[test]
    fn setup_commands_parse_with_defaults() {
        match Cli::try_parse_from(["openrag", "init", "--docs", "./docs", "--connect", "claude", "--connect", "codex"]) {
            Ok(Cli { command: Command::Init(a) }) => {
                assert_eq!(a.api_url, "http://127.0.0.1:8080");
                assert_eq!(a.org.slug, "default");
                assert_eq!(a.bucket, "docs");
                assert_eq!(a.connect, vec![McpClient::Claude, McpClient::Codex]);
                assert_eq!(a.mcp_scope, ClaudeScope::User);
            }
            other => assert!(other.is_ok(), "init parse failed: {other:?}"),
        }
        assert!(Cli::try_parse_from(["openrag", "org", "init", "--slug", "acme", "--save"]).is_ok());
        assert!(Cli::try_parse_from(["openrag", "bucket", "create", "handbook", "--scope", "dept", "--auth-level", "3"]).is_ok());
        assert!(Cli::try_parse_from(["openrag", "bucket", "create", "x", "--auth-level", "11"]).is_err());
        assert!(Cli::try_parse_from(["openrag", "connect", "claude", "--scope", "project", "--dry-run"]).is_ok());
        assert!(Cli::try_parse_from(["openrag", "connect", "cursor"]).is_err());
        let argv = |a: &[&str]| a.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(creates_config(&argv(&["openrag", "init", "--docs", "d"])));
        assert!(creates_config(&argv(&["openrag", "org", "init", "--save"])));
        assert!(!creates_config(&argv(&["openrag", "concept", "list"])));
        assert!(!creates_config(&argv(&["openrag", "--help"])));
    }

    #[test]
    fn org_slugs_and_ingest_folders() -> Result<(), Box<dyn std::error::Error>> {
        assert!(valid_path_segment("acme-01_hq"));
        for bad in ["", "ACME", "a:b", "경영지원", "a b"] {
            assert!(!valid_path_segment(bad), "{bad:?}");
        }
        let dir = tempfile::tempdir()?;
        std::fs::create_dir_all(dir.path().join("sub"))?;
        for f in ["b.md", "sub/c.PDF", "skip.exe", ".hidden.md"] {
            std::fs::write(dir.path().join(f), "x")?;
        }
        let found: Vec<String> = ingest_targets(dir.path())
            .map_err(|e| e.0)?
            .iter()
            .filter_map(|p| p.strip_prefix(dir.path()).ok().map(|r| r.display().to_string()))
            .collect();
        assert_eq!(found, ["b.md", "sub/c.PDF"]);
        let empty = tempfile::tempdir()?;
        assert!(ingest_targets(empty.path()).is_err());
        // Document keys: the path below the ingested folder, or the file name for one file.
        assert_eq!(document_key(dir.path(), &dir.path().join("sub/c.PDF")), "sub/c.PDF");
        assert_eq!(document_key(dir.path(), &dir.path().join("b.md")), "b.md");
        let one = dir.path().join("b.md");
        assert_eq!(document_key(&one, &one), "b.md");
        Ok(())
    }

    #[test]
    fn redact_target_selection() {
        let parse = |args: &[&str]| -> RedactRunArgs {
            let mut full = vec!["openrag", "redact", "run"];
            full.extend_from_slice(args);
            match Cli::try_parse_from(full) {
                Ok(Cli {
                    command: Command::Redact(RedactCmd::Run(a)),
                }) => a,
                other => {
                    assert!(other.is_ok(), "parse failed: {other:?}");
                    RedactRunArgs {
                        scope: RedactScopeArg::Org,
                        reason: String::new(),
                        doc: None,
                        chunk: None,
                        memory: None,
                        wiki: None,
                        pattern: None,
                        dept: None,
                        vault_entry: None,
                        redact_thumbnails: false,
                        hard_purge: false,
                        tombstone_delete: false,
                    }
                }
            }
        };
        let id = Uuid::new_v4().to_string();
        let t = redact_target(&parse(&["--reason", "r", "--memory", &id])).map(|t| t.scope);
        assert_eq!(t.ok(), Some("memory"));
        let t = redact_target(&parse(&["--reason", "r", "--pattern", "900101-1234567"])).map(|t| t.scope);
        assert_eq!(t.ok(), Some("pattern_org"));
        assert!(redact_target(&parse(&["--reason", "r"])).is_err());
        assert!(redact_target(&parse(&["--scope", "document", "--reason", "r", "--memory", &id])).is_err());
    }
}
