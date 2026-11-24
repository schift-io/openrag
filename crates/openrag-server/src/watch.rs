//! Watched document folders. The server keeps buckets in step with folders on disk: a new file is
//! ingested, a changed file replaces its in-force copy, a moved or renamed file keeps its document
//! (matched by the file's SHA-256, so nothing is parsed or embedded again), and a deleted file goes
//! out of force (`valid_to`) while its record stays.
//!
//! Configuration (environment):
//! - `OPENRAG_DOCS_DIRS`: folders to watch, a JSON list such as `["/docs", "/mnt/hr"]` (entries may
//!   be objects: `{"path": "/mnt/hr", "bucket": "hr", "types": ["pdf", "hwp"]}`) or a comma-separated
//!   list. Unset: no watching.
//! - `OPENRAG_DOCS_TYPES`: file types to pick up (default `hwp, hwpx, md, pdf, docx, pptx, html`).
//! - `OPENRAG_DOCS_BUCKET`: bucket for entries without one (default `docs`).
//! - `OPENRAG_DOCS_SCAN_SECS`: seconds between scans (default 30).
//! - `OPENRAG_DOCS_ORG`: organization slug, when the database holds more than one.
//!
//! Folders are scanned, not subscribed to, so network shares and Docker bind mounts work the same.
//! A scan only reads files whose size or modification time changed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use openrag_core::{OpenRagError, PrincipalContext};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::api_core::{ingest_document_into_bucket, row_to_bucket, BucketRecord, IngestPayload};
use crate::auth::load_principal_context;
use crate::state::{non_empty_env, AppState};

/// File types picked up when `OPENRAG_DOCS_TYPES` is unset.
pub const DEFAULT_TYPES: &[&str] = &["hwp", "hwpx", "md", "pdf", "docx", "pptx", "html"];

/// Every type the parser reads.
pub const SUPPORTED_TYPES: &[&str] = &[
    "hwp", "hwpx", "hml", "pdf", "md", "markdown", "txt", "docx", "pptx", "html", "htm", "png",
    "jpg", "jpeg", "webp",
];

/// Files larger than the HTTP upload limit are skipped, so a folder and an upload accept the same.
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// A file modified more recently than this may still be being written; it waits for the next scan.
const SETTLE: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchFolder {
    pub path: String,
    pub bucket: String,
    pub types: Vec<String>,
}

impl WatchFolder {
    fn accepts(&self, ext: &str) -> bool {
        let ext = ext.to_ascii_lowercase();
        self.types.iter().any(|t| t == &ext || alias(t) == Some(ext.as_str()))
    }
}

fn alias(t: &str) -> Option<&'static str> {
    match t {
        "md" => Some("markdown"),
        "markdown" => Some("md"),
        "html" => Some("htm"),
        "htm" => Some("html"),
        "jpg" => Some("jpeg"),
        "jpeg" => Some("jpg"),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocsWatchConfig {
    pub folders: Vec<WatchFolder>,
    pub interval: Duration,
    pub org: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DirEntry {
    Path(String),
    Full {
        path: String,
        #[serde(default)]
        bucket: Option<String>,
        #[serde(default)]
        types: Option<TypesValue>,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TypesValue {
    List(Vec<String>),
    Text(String),
}

fn invalid(msg: String) -> OpenRagError {
    OpenRagError::Validation(msg)
}

fn split_list(raw: &str) -> Vec<String> {
    raw.split([',', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_types(items: Vec<String>, what: &str) -> Result<Vec<String>, OpenRagError> {
    let mut out: Vec<String> = Vec::new();
    for raw in items {
        for t in raw.split_whitespace() {
            let t = t.trim_start_matches('.').to_ascii_lowercase();
            if t.is_empty() {
                continue;
            }
            if !SUPPORTED_TYPES.contains(&t.as_str()) {
                return Err(invalid(format!(
                    "{what}: '{t}' is not a file type OpenRAG reads (supported: {})",
                    SUPPORTED_TYPES.join(", ")
                )));
            }
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
    if out.is_empty() {
        return Err(invalid(format!("{what} lists no file types")));
    }
    Ok(out)
}

fn types_from_text(raw: &str, what: &str) -> Result<Vec<String>, OpenRagError> {
    let t = raw.trim();
    let items = if t.starts_with('[') {
        serde_json::from_str::<Vec<String>>(t).map_err(|e| invalid(format!("{what} is not a JSON list of strings: {e}")))?
    } else {
        split_list(t)
    };
    parse_types(items, what)
}

impl DocsWatchConfig {
    /// The configuration from the environment; `None` when `OPENRAG_DOCS_DIRS` is unset or empty.
    pub fn from_env() -> Result<Option<Self>, OpenRagError> {
        let Some(dirs) = non_empty_env("OPENRAG_DOCS_DIRS") else {
            return Ok(None);
        };
        Self::parse(
            &dirs,
            non_empty_env("OPENRAG_DOCS_TYPES").as_deref(),
            non_empty_env("OPENRAG_DOCS_BUCKET").as_deref(),
            non_empty_env("OPENRAG_DOCS_SCAN_SECS").as_deref(),
            non_empty_env("OPENRAG_DOCS_ORG").as_deref(),
        )
    }

    pub fn parse(
        dirs: &str,
        types: Option<&str>,
        bucket: Option<&str>,
        scan_secs: Option<&str>,
        org: Option<&str>,
    ) -> Result<Option<Self>, OpenRagError> {
        let default_types = match types {
            Some(t) => types_from_text(t, "OPENRAG_DOCS_TYPES")?,
            None => DEFAULT_TYPES.iter().map(ToString::to_string).collect(),
        };
        let default_bucket = bucket.map(str::trim).filter(|b| !b.is_empty()).unwrap_or("docs").to_string();
        let interval = match scan_secs {
            Some(s) => {
                let secs: u64 = s
                    .trim()
                    .parse()
                    .map_err(|_| invalid(format!("OPENRAG_DOCS_SCAN_SECS must be a whole number of seconds, got '{s}'")))?;
                if secs == 0 {
                    return Err(invalid("OPENRAG_DOCS_SCAN_SECS must be at least 1".into()));
                }
                Duration::from_secs(secs)
            }
            None => Duration::from_secs(30),
        };

        let trimmed = dirs.trim();
        let entries: Vec<DirEntry> = if trimmed.starts_with('[') {
            serde_json::from_str(trimmed).map_err(|e| {
                invalid(format!(
                    "OPENRAG_DOCS_DIRS is not a JSON list of folders (\"/docs\" or {{\"path\": …, \"bucket\": …, \"types\": […]}}): {e}"
                ))
            })?
        } else {
            split_list(trimmed).into_iter().map(DirEntry::Path).collect()
        };

        let mut folders: Vec<WatchFolder> = Vec::new();
        for entry in entries {
            let (path, bucket, types) = match entry {
                DirEntry::Path(p) => (p, None, None),
                DirEntry::Full { path, bucket, types } => (path, bucket, types),
            };
            let path = path.trim().trim_end_matches('/').to_string();
            let path = if path.is_empty() { "/".to_string() } else { path };
            if !path.starts_with('/') {
                return Err(invalid(format!("OPENRAG_DOCS_DIRS: '{path}' must be an absolute path")));
            }
            if folders.iter().any(|f| f.path == path) {
                return Err(invalid(format!("OPENRAG_DOCS_DIRS lists '{path}' twice")));
            }
            let types = match types {
                Some(TypesValue::List(l)) => parse_types(l, &format!("types of '{path}'"))?,
                Some(TypesValue::Text(t)) => types_from_text(&t, &format!("types of '{path}'"))?,
                None => default_types.clone(),
            };
            let bucket = bucket
                .map(|b| b.trim().to_string())
                .filter(|b| !b.is_empty())
                .unwrap_or_else(|| default_bucket.clone());
            folders.push(WatchFolder { path, bucket, types });
        }
        if folders.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            folders,
            interval,
            org: org.map(str::trim).filter(|o| !o.is_empty()).map(ToString::to_string),
        }))
    }
}

/// Starts the watcher when `OPENRAG_DOCS_DIRS` is set. A bad setting stops the server instead of
/// being ignored.
pub fn start_from_env(state: &Arc<AppState>) -> Result<(), OpenRagError> {
    let Some(cfg) = DocsWatchConfig::from_env()? else {
        return Ok(());
    };
    for f in &cfg.folders {
        tracing::info!(folder = %f.path, bucket = %f.bucket, types = %f.types.join(","), "watching documents folder");
    }
    let watcher = Watcher::new(Arc::clone(state), cfg);
    tokio::spawn(watcher.run());
    Ok(())
}

// ---------------------------------------------------------------------------
// Planning (pure)
// ---------------------------------------------------------------------------

/// A document the watcher manages, as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedDoc {
    pub id: Uuid,
    pub key: String,
    pub sha: String,
    pub root: String,
}

/// A file on disk. `sha` is `None` while the file may still be being written (or failed to
/// ingest and has not changed since): its document, if any, is left as it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskFile {
    pub key: String,
    pub sha: Option<String>,
    pub root: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Same content at the same path; refresh the stored size/mtime.
    Keep { doc: Uuid, file: usize },
    /// Same content at another path: the document follows the file.
    Move { doc: Uuid, file: usize },
    /// New or changed content; `replaces` is the document at that path.
    Ingest { file: usize, replaces: Option<Uuid> },
    /// The file is gone: the document goes out of force.
    Retire { doc: Uuid },
}

/// What to do so the documents match the files. Content (SHA-256) decides identity: a file whose
/// content a document already has is a move (or a copy, which needs nothing), never a new
/// ingestion. Documents under `unavailable_roots` are never retired, and with `hold_retirements`
/// (a file is still being written, possibly the new place of a document) none are.
pub fn plan(
    files: &[DiskFile],
    docs: &[WatchedDoc],
    unavailable_roots: &HashSet<String>,
    hold_retirements: bool,
) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut seen: HashSet<Uuid> = HashSet::new();
    let mut done: HashSet<usize> = HashSet::new();
    let on_disk: HashSet<&str> = files.iter().map(|f| f.key.as_str()).collect();

    let mut order: Vec<usize> = (0..files.len()).collect();
    order.sort_by(|a, b| files[*a].key.cmp(&files[*b].key));

    // Files being written (or failing) hold on to the documents at their path.
    for &i in &order {
        if files[i].sha.is_none() {
            for d in docs.iter().filter(|d| d.key == files[i].key) {
                seen.insert(d.id);
            }
            done.insert(i);
        }
    }
    // Same path, same content.
    for &i in &order {
        if done.contains(&i) {
            continue;
        }
        let sha = files[i].sha.as_deref();
        if let Some(d) = docs.iter().find(|d| d.key == files[i].key && Some(d.sha.as_str()) == sha && !seen.contains(&d.id)) {
            seen.insert(d.id);
            done.insert(i);
            steps.push(Step::Keep { doc: d.id, file: i });
        }
    }
    // Same content elsewhere: a move. Documents whose old path is gone go first.
    for &i in &order {
        if done.contains(&i) {
            continue;
        }
        let sha = files[i].sha.as_deref();
        let candidate = docs
            .iter()
            .filter(|d| Some(d.sha.as_str()) == sha && !seen.contains(&d.id))
            .min_by_key(|d| (on_disk.contains(d.key.as_str()), d.key.clone()));
        if let Some(d) = candidate {
            seen.insert(d.id);
            done.insert(i);
            steps.push(Step::Move { doc: d.id, file: i });
        }
    }
    // A copy of content a document already has needs nothing.
    for &i in &order {
        if done.contains(&i) {
            continue;
        }
        let sha = files[i].sha.as_deref();
        if docs.iter().any(|d| Some(d.sha.as_str()) == sha) {
            done.insert(i);
        }
    }
    // New or changed content.
    for &i in &order {
        if done.contains(&i) {
            continue;
        }
        let replaces = docs.iter().find(|d| d.key == files[i].key && !seen.contains(&d.id)).map(|d| d.id);
        if let Some(id) = replaces {
            seen.insert(id);
        }
        steps.push(Step::Ingest { file: i, replaces });
    }
    if hold_retirements {
        return steps;
    }
    for d in docs {
        if !seen.contains(&d.id) && !unavailable_roots.contains(&d.root) {
            steps.push(Step::Retire { doc: d.id });
        }
    }
    steps
}

// ---------------------------------------------------------------------------
// Scanning and applying
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Found {
    key: String,
    path: PathBuf,
    root: String,
    size: u64,
    mtime: String,
    settled: bool,
    /// Zero bytes: an editor between truncating and writing, or an emptied file. Either way
    /// its document stays as it is.
    empty: bool,
}

struct Scan {
    files: Vec<Found>,
    unavailable: HashSet<String>,
    empty_roots: HashSet<String>,
    too_large: Vec<(String, u64)>,
}

fn mtime_string(t: SystemTime) -> String {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_nanos().to_string()).unwrap_or_default()
}

fn skip_name(name: &str) -> bool {
    // Hidden files, Office lock files (~$Report.docx) and editor backups.
    name.starts_with('.') || name.starts_with("~$") || name.ends_with('~')
}

fn scan_folders(folders: &[WatchFolder], all_roots: &[String]) -> Scan {
    let now = SystemTime::now();
    let mut scan = Scan {
        files: Vec::new(),
        unavailable: HashSet::new(),
        empty_roots: HashSet::new(),
        too_large: Vec::new(),
    };
    for folder in folders {
        let root = Path::new(&folder.path);
        if !root.is_dir() {
            scan.unavailable.insert(folder.path.clone());
            continue;
        }
        let before = scan.files.len();
        let mut stack = vec![root.to_path_buf()];
        let mut failed = false;
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(folder = %dir.display(), error = %e, "cannot read folder; nothing under it is retired this scan");
                    failed = true;
                    continue;
                }
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
                if skip_name(name) {
                    continue;
                }
                let Ok(meta) = std::fs::metadata(&path) else { continue };
                if meta.is_dir() {
                    let is_link = entry.file_type().map(|t| t.is_symlink()).unwrap_or(false);
                    let p = path.to_string_lossy();
                    // Another watched folder below this one is scanned with its own settings.
                    if !is_link && !all_roots.iter().any(|r| r.as_str() == p) {
                        stack.push(path);
                    }
                    continue;
                }
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if !meta.is_file() || !folder.accepts(ext) {
                    continue;
                }
                let key = path.to_string_lossy().into_owned();
                if meta.len() > MAX_FILE_BYTES {
                    scan.too_large.push((key, meta.len()));
                    continue;
                }
                let modified = meta.modified().unwrap_or(UNIX_EPOCH);
                let settled = now.duration_since(modified).map(|age| age >= SETTLE).unwrap_or(false);
                scan.files.push(Found {
                    key,
                    path,
                    root: folder.path.clone(),
                    size: meta.len(),
                    mtime: mtime_string(modified),
                    settled,
                    empty: meta.len() == 0,
                });
            }
        }
        if failed {
            scan.unavailable.insert(folder.path.clone());
        } else if scan.files.len() == before {
            scan.empty_roots.insert(folder.path.clone());
        }
    }
    scan
}

#[derive(Debug, Clone)]
struct StoredDoc {
    doc: WatchedDoc,
    title: String,
    size: Option<String>,
    mtime: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Stamp {
    size: u64,
    mtime: String,
}

#[derive(Default, Debug)]
struct Tally {
    added: usize,
    replaced: usize,
    moved: usize,
    removed: usize,
    failed: usize,
}

struct Watcher {
    state: Arc<AppState>,
    cfg: DocsWatchConfig,
    /// key -> (size/mtime, sha) of files already hashed.
    hashes: HashMap<String, (Stamp, String)>,
    /// key -> size/mtime of files that failed to ingest; retried once they change.
    failures: HashMap<String, Stamp>,
    /// Messages already logged, so a waiting or broken state is reported once, not every scan.
    reported: HashSet<String>,
}

fn db(e: impl std::fmt::Display, what: &str) -> OpenRagError {
    OpenRagError::Database(format!("{what}: {e}"))
}

fn file_name(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

impl Watcher {
    fn new(state: Arc<AppState>, cfg: DocsWatchConfig) -> Self {
        Self { state, cfg, hashes: HashMap::new(), failures: HashMap::new(), reported: HashSet::new() }
    }

    fn report_once(&mut self, msg: String) -> bool {
        self.reported.insert(msg)
    }

    async fn run(mut self) {
        let mut groups: BTreeMap<String, Vec<WatchFolder>> = BTreeMap::new();
        for f in &self.cfg.folders {
            groups.entry(f.bucket.clone()).or_default().push(f.clone());
        }
        let all_roots: Vec<String> = self.cfg.folders.iter().map(|f| f.path.clone()).collect();
        loop {
            for (bucket, folders) in &groups {
                if let Err(e) = self.sync_bucket(bucket, folders, &all_roots).await {
                    let msg = format!("{bucket}: {e}");
                    if self.report_once(msg) {
                        tracing::warn!(bucket = %bucket, error = %e, "documents folder sync failed; retrying every scan");
                    }
                }
            }
            tokio::time::sleep(self.cfg.interval).await;
        }
    }

    async fn org_id(&mut self) -> Result<Option<Uuid>, OpenRagError> {
        let worker = self.state.require_worker_pool()?;
        let rows: Vec<Uuid> = match self.cfg.org.as_deref() {
            Some(slug) => sqlx::query_scalar("SELECT id FROM iam.orgs WHERE slug = $1")
                .bind(slug)
                .fetch_all(worker.inner())
                .await
                .map_err(|e| db(e, "lookup organization"))?,
            None => sqlx::query_scalar("SELECT id FROM iam.orgs ORDER BY created_at LIMIT 2")
                .fetch_all(worker.inner())
                .await
                .map_err(|e| db(e, "lookup organization"))?,
        };
        match rows.as_slice() {
            [] => {
                if self.report_once("no-org".into()) {
                    tracing::info!("documents folders wait for an organization (run `openrag init`)");
                }
                Ok(None)
            }
            [one] => Ok(Some(*one)),
            _ => Err(OpenRagError::Validation(
                "the database holds more than one organization: set OPENRAG_DOCS_ORG to the slug to watch for".into(),
            )),
        }
    }

    async fn bucket(&mut self, org: Uuid, slug: &str) -> Result<Option<BucketRecord>, OpenRagError> {
        let worker = self.state.require_worker_pool()?;
        let row = sqlx::query(
            "SELECT id, org_id, dept_id, owner_principal_id, slug, name, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, created_at
             FROM corpus.buckets WHERE org_id = $1 AND slug = $2 LIMIT 1",
        )
        .bind(org)
        .bind(slug)
        .fetch_optional(worker.inner())
        .await
        .map_err(|e| db(e, "lookup bucket"))?;
        match row {
            Some(r) => Ok(Some(row_to_bucket(&r)?)),
            None => {
                if self.report_once(format!("no-bucket:{slug}")) {
                    tracing::info!(bucket = %slug, "documents folders wait for this bucket (`openrag init` creates `docs`)");
                }
                Ok(None)
            }
        }
    }

    async fn stored_docs(&self, org: Uuid, bucket: Uuid, roots: &[String]) -> Result<Vec<StoredDoc>, OpenRagError> {
        let worker = self.state.require_worker_pool()?;
        let rows = sqlx::query(
            "SELECT id, title, content_sha256,
                    metadata->>'external_key' AS key, metadata->>'watch_root' AS root,
                    metadata->>'watch_size' AS size, metadata->>'watch_mtime' AS mtime
             FROM corpus.documents
             WHERE org_id = $1 AND bucket_id = $2
               AND metadata->>'source' = 'watch' AND metadata->>'watch_root' = ANY($3)
               AND status IN ('indexed', 'pending')
               AND (valid_to IS NULL OR valid_to > now())
             ORDER BY created_at",
        )
        .bind(org)
        .bind(bucket)
        .bind(roots)
        .fetch_all(worker.inner())
        .await
        .map_err(|e| db(e, "list watched documents"))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let get = |c: &str| r.try_get::<Option<String>, _>(c).map_err(|e| db(e, c));
            let (Some(key), Some(root), Some(sha)) = (get("key")?, get("root")?, get("content_sha256")?) else {
                continue;
            };
            out.push(StoredDoc {
                doc: WatchedDoc { id: r.try_get("id").map_err(|e| db(e, "id"))?, key, sha, root },
                title: r.try_get("title").map_err(|e| db(e, "title"))?,
                size: get("size")?,
                mtime: get("mtime")?,
            });
        }
        Ok(out)
    }

    /// SHA-256 of a file, reusing the stored or cached value while size and mtime are unchanged.
    async fn sha_of(&mut self, f: &Found, stored: &HashMap<&str, &StoredDoc>) -> Result<String, OpenRagError> {
        let stamp = Stamp { size: f.size, mtime: f.mtime.clone() };
        if let Some(d) = stored.get(f.key.as_str()) {
            if d.size.as_deref() == Some(f.size.to_string().as_str()) && d.mtime.as_deref() == Some(f.mtime.as_str()) {
                return Ok(d.doc.sha.clone());
            }
        }
        if let Some((s, sha)) = self.hashes.get(&f.key) {
            if *s == stamp {
                return Ok(sha.clone());
            }
        }
        let bytes = tokio::fs::read(&f.path)
            .await
            .map_err(|e| OpenRagError::Validation(format!("cannot read {}: {e}", f.key)))?;
        let sha = openrag_parse::sha256_hex_bytes(&bytes);
        self.hashes.insert(f.key.clone(), (stamp, sha.clone()));
        Ok(sha)
    }

    async fn sync_bucket(&mut self, slug: &str, folders: &[WatchFolder], all_roots: &[String]) -> Result<(), OpenRagError> {
        let Some(org) = self.org_id().await? else { return Ok(()) };
        let Some(bucket) = self.bucket(org, slug).await? else { return Ok(()) };
        let state = Arc::clone(&self.state);
        let principal = load_principal_context(
            state.require_worker_pool()?,
            org,
            bucket.owner_principal_id,
            None,
            "the bucket owner no longer exists",
        )
        .await?;

        let (folders_owned, roots_owned) = (folders.to_vec(), all_roots.to_vec());
        let scan = tokio::task::spawn_blocking(move || scan_folders(&folders_owned, &roots_owned))
            .await
            .map_err(|e| OpenRagError::Internal(format!("folder scan: {e}")))?;
        for root in &scan.unavailable {
            if self.report_once(format!("unavailable:{root}")) {
                tracing::warn!(folder = %root, "documents folder is missing or unreadable; its documents stay as they are");
            }
        }
        for (key, size) in &scan.too_large {
            if self.report_once(format!("large:{key}:{size}")) {
                tracing::warn!(file = %key, bytes = size, "file is larger than 64 MiB; skipped");
            }
        }

        let roots: Vec<String> = folders.iter().map(|f| f.path.clone()).collect();
        let stored = self.stored_docs(org, bucket.id, &roots).await?;
        let by_key: HashMap<&str, &StoredDoc> = stored.iter().map(|d| (d.doc.key.as_str(), d)).collect();

        // A folder that loses every file at once is more likely an unmounted share than a
        // cleared library; its documents are kept until files show up there again. Deleting the
        // last single file of a folder still retires it.
        let mut keep_roots = scan.unavailable.clone();
        for root in &scan.empty_roots {
            let held = stored.iter().filter(|d| &d.doc.root == root).count();
            if held >= 2 {
                keep_roots.insert(root.clone());
                if self.report_once(format!("empty:{root}")) {
                    tracing::warn!(
                        folder = %root,
                        documents = held,
                        "documents folder lost all its files at once; its documents are kept (to retire them on purpose, leave one file until the others are gone)"
                    );
                }
            }
        }

        let mut disk = Vec::with_capacity(scan.files.len());
        for f in &scan.files {
            let stamp = Stamp { size: f.size, mtime: f.mtime.clone() };
            let failing = self.failures.get(&f.key) == Some(&stamp);
            let sha = if !f.settled || failing || f.empty {
                None
            } else {
                match self.sha_of(f, &by_key).await {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::warn!(file = %f.key, error = %e, "cannot read file");
                        None
                    }
                }
            };
            disk.push(DiskFile { key: f.key.clone(), sha, root: f.root.clone() });
        }
        let docs: Vec<WatchedDoc> = stored.iter().map(|d| d.doc.clone()).collect();
        let writing = scan.files.iter().any(|f| !f.settled);
        let steps = plan(&disk, &docs, &keep_roots, writing);

        let mut tally = Tally::default();
        for step in steps {
            match step {
                Step::Keep { doc, file } => {
                    let f = &scan.files[file];
                    let d = stored.iter().find(|d| d.doc.id == doc);
                    let fresh = d.is_some_and(|d| {
                        d.size.as_deref() == Some(f.size.to_string().as_str()) && d.mtime.as_deref() == Some(f.mtime.as_str())
                    });
                    if !fresh {
                        self.mark(org, doc, f, None).await?;
                    }
                }
                Step::Move { doc, file } => {
                    let f = &scan.files[file];
                    let d = stored.iter().find(|d| d.doc.id == doc);
                    let old_name = d.map(|d| file_name(&d.doc.key).to_string()).unwrap_or_default();
                    let new_title = d.filter(|d| d.title == old_name).map(|_| file_name(&f.key).to_string());
                    self.mark(org, doc, f, new_title.as_deref()).await?;
                    tally.moved += 1;
                    tracing::info!(from = %d.map(|d| d.doc.key.as_str()).unwrap_or(""), to = %f.key, document_id = %doc, "document moved");
                }
                Step::Ingest { file, replaces } => {
                    let f = scan.files[file].clone();
                    let Some(sha) = disk[file].sha.clone() else { continue };
                    match self.ingest(&principal, &bucket, &f, &sha, replaces, &docs).await {
                        Ok(Some(true)) => tally.replaced += 1,
                        Ok(Some(false)) => tally.added += 1,
                        Ok(None) => {}
                        Err(e) => {
                            tally.failed += 1;
                            self.failures.insert(f.key.clone(), Stamp { size: f.size, mtime: f.mtime.clone() });
                            tracing::warn!(file = %f.key, error = %e, "cannot ingest file; retried when it changes");
                        }
                    }
                }
                Step::Retire { doc } => {
                    self.retire(org, doc).await?;
                    tally.removed += 1;
                    let key = stored.iter().find(|d| d.doc.id == doc).map(|d| d.doc.key.clone()).unwrap_or_default();
                    tracing::info!(file = %key, document_id = %doc, "file removed; document out of force (record kept)");
                }
            }
        }
        let present: HashSet<&str> = scan.files.iter().map(|f| f.key.as_str()).collect();
        self.hashes.retain(|k, _| present.contains(k.as_str()));
        self.failures.retain(|k, _| present.contains(k.as_str()));
        if tally.added + tally.replaced + tally.moved + tally.removed + tally.failed > 0 {
            tracing::info!(
                bucket = %slug,
                added = tally.added,
                replaced = tally.replaced,
                moved = tally.moved,
                removed = tally.removed,
                failed = tally.failed,
                "documents folder synced"
            );
        }
        Ok(())
    }

    /// Ingests one file. `Some(true)` when it replaced a document, `Some(false)` when it added
    /// one, `None` when the content was already in the bucket.
    async fn ingest(
        &mut self,
        principal: &PrincipalContext,
        bucket: &BucketRecord,
        f: &Found,
        sha: &str,
        replaces: Option<Uuid>,
        watched: &[WatchedDoc],
    ) -> Result<Option<bool>, OpenRagError> {
        let bytes = tokio::fs::read(&f.path)
            .await
            .map_err(|e| OpenRagError::Validation(format!("cannot read {}: {e}", f.key)))?;
        if openrag_parse::sha256_hex_bytes(&bytes) != sha {
            // Changed again since it was hashed; the next scan picks it up.
            self.hashes.remove(&f.key);
            return Ok(None);
        }
        let name = file_name(&f.key).to_string();
        let payload = IngestPayload {
            file_bytes: Some(bytes),
            filename: Some(name.clone()),
            mime_type: None,
            title: Some(name),
            text: None,
            access_scope: None,
            min_auth_level: None,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            sensitivity: None,
            ocr_mode: None,
            batch: false,
            tags: Vec::new(),
            external_backend: None,
            filterable_metadata: None,
            revision: None,
            valid_from: None,
            valid_to: None,
            source_uri: Some(f.key.clone()),
            slug: None,
            external_key: Some(f.key.clone()),
            bucket_ref: None,
        };
        let result = ingest_document_into_bucket(&self.state, principal, bucket, payload).await?;
        let id = result
            .get("document_id")
            .or_else(|| result.get("id"))
            .and_then(Value::as_str)
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| OpenRagError::Internal("ingest returned no document id".into()))?;
        let unchanged = result.get("unchanged").and_then(Value::as_bool) == Some(true);
        let superseded = result.get("supersedes").and_then(Value::as_str).and_then(|s| Uuid::parse_str(s).ok());

        if unchanged && watched.iter().any(|d| d.id == id) {
            // Same content as another watched file: a copy, nothing to store.
            if let Some(old) = replaces.filter(|old| *old != id) {
                self.retire(principal.org_id, old).await?;
            }
            return Ok(None);
        }
        // A new document, or content that was already in the bucket under another name (an
        // earlier `openrag ingest`, or a folder that moved): the watcher takes it over.
        self.mark(principal.org_id, id, f, None).await?;
        if let Some(old) = replaces.filter(|old| *old != id && superseded != Some(*old)) {
            self.retire(principal.org_id, old).await?;
        }
        tracing::info!(file = %f.key, document_id = %id, replaced = ?replaces, "file ingested");
        Ok(if unchanged { None } else { Some(replaces.is_some()) })
    }

    /// Records the file a document came from (path, folder, size, mtime); with `title`, renames it.
    async fn mark(&self, org: Uuid, doc: Uuid, f: &Found, title: Option<&str>) -> Result<(), OpenRagError> {
        let worker = self.state.require_worker_pool()?;
        let meta = json!({
            "source": "watch",
            "watch_root": f.root,
            "external_key": f.key,
            "watch_size": f.size.to_string(),
            "watch_mtime": f.mtime,
        });
        sqlx::query(
            "UPDATE corpus.documents
             SET metadata = metadata || $3,
                 source_uri = $4,
                 title = COALESCE($5, title),
                 updated_at = now()
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org)
        .bind(doc)
        .bind(meta)
        .bind(&f.key)
        .bind(title)
        .execute(worker.inner())
        .await
        .map_err(|e| db(e, "record watched file"))?;
        Ok(())
    }

    /// Takes a document out of force; the row, its pages and chunks stay.
    async fn retire(&self, org: Uuid, doc: Uuid) -> Result<(), OpenRagError> {
        let worker = self.state.require_worker_pool()?;
        sqlx::query(
            "UPDATE corpus.documents
             SET valid_to = now(), updated_at = now(),
                 metadata = metadata || jsonb_build_object('watch_removed_at', now())
             WHERE org_id = $1 AND id = $2 AND (valid_to IS NULL OR valid_to > now())",
        )
        .bind(org)
        .bind(doc)
        .execute(worker.inner())
        .await
        .map_err(|e| db(e, "retire document"))?;
        Ok(())
    }
}
