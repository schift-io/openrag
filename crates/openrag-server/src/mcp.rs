//! Unified 13-tool MCP server (JSON-RPC 2.0).
//!
//! Two transports share one tool registry and one security stack:
//!
//! * **stdio** (`openrag mcp --stdio`, `openrag-mcp`): auto-detects NDJSON and
//!   LSP-style `Content-Length` framing per message and answers in the same framing.
//!   The caller is identified by `--api-key` / `OPENRAG_API_KEY`; the key is
//!   re-authenticated on every tool call so revocation takes effect immediately.
//! * **Streamable HTTP** (`POST /mcp`): fail-closed Bearer auth (`401`), optional
//!   `Mcp-Session-Id` bound to the authenticated principal, JSON or SSE replies.
//!
//! Every tool runs as the authenticated principal. Wherever an HTTP handler already
//! exists, the tool is dispatched in-process to that handler so IAM/RLS, Guard and
//! `[PII:...]` neutralization stay identical to the REST surface. Tools without a REST
//! twin (`rag_get_page`, `wiki_search`, `wiki_get`, `memory_*` recall/reflect) run
//! their queries inside `begin_rls_tx` and apply the API-key ARN scope.

use std::collections::HashMap;
use std::path::Path as FsPath;
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::header::{ACCEPT, ALLOW, CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use base64::Engine as _;
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, MemoryBackendKind, OpenRagError,
    PrincipalContext,
};
use openrag_memory::{
    EngageRequest, MemoryEdgeType, MemoryEntry, MemoryStatus, ReflectRequest, WikiCompiler,
    WikiPage, WikiStatus,
};
use serde_json::{json, Map, Value};
use sqlx::Row;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tower::ServiceExt;
use uuid::Uuid;

use crate::api_core::{resolve_bucket_rls, resolve_org_and_dept_paths};
use crate::auth::{authenticate_bearer_token, enforce_api_key_arn_scope, AuthPrincipal, TrustedPrincipal};
use crate::llm_router::{neutralize_literal_pii_tokens, seal_text_for_llm};
use crate::mcp_skills;
use crate::mcp_tasks::{default_task_store, PrincipalKey};
#[doc(hidden)]
pub use crate::mcp_tasks::{TaskHookFn, TaskHookOutcome, TaskStore};
use crate::state::AppState;

/// The exact, ordered MCP tool registry.
pub const MCP_TOOL_NAMES: [&str; 16] = [
    "rag_search",
    "rag_ask",
    "rag_ingest",
    "rag_get_page",
    "rag_list_documents",
    "rag_read_sheet",
    "docs_status",
    "wiki_search",
    "wiki_get",
    "memory_search",
    "memory_patch",
    "memory_engage",
    "memory_reflect",
    "guard_check",
    "concept_link",
    "concept_resolve",
];

pub const MODERN_PROTOCOL_VERSIONS: [&str; 1] = ["2026-07-28"];
pub const SUPPORTED_PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];
pub const DEFAULT_PROTOCOL_VERSION: &str = "2025-03-26";
pub const SESSION_HEADER: &str = "mcp-session-id";
const PROTOCOL_VERSION_HEADER: &str = "mcp-protocol-version";
const METHOD_HEADER: &str = "mcp-method";
const NAME_HEADER: &str = "mcp-name";

const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CLIENT_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const META_SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";

const EXT_SKILLS: &str = "io.modelcontextprotocol/skills";
const EXT_TASKS: &str = "io.modelcontextprotocol/tasks";

const CACHE_TTL_MS: u64 = 300_000;
const CACHE_SCOPE_PRIVATE: &str = "private";

const BASE64_SENTINEL_PREFIX: &str = "=?base64?";
const BASE64_SENTINEL_SUFFIX: &str = "?=";

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_INVALID_REQUEST: i64 = -32600;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INVALID_PARAMS: i64 = -32602;
const JSONRPC_INTERNAL_ERROR: i64 = -32603;
pub const JSONRPC_HEADER_MISMATCH: i64 = -32020;
pub const JSONRPC_MISSING_REQUIRED_CLIENT_CAPABILITY: i64 = -32021;
pub const JSONRPC_UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

const LEGACY_INSTRUCTIONS: &str = "OpenRAG knowledge tools. Every call runs as the authenticated principal: results are filtered by ARN scope, permission groups, permission level and row-level security, and text passes the PII/Safe-Word guard. Search with rag_search; each hit's source.path names the original file. Search the way a careful researcher would: split a question with several parts or conditions into one rag_search per part; for vague or descriptive questions also pass hypothetical_answer (a short passage worded like the document you expect); when results are weak, retry with other wording, synonyms or the formal term; when a hit is relevant but cut off, read its whole page with rag_get_page. Excel and CSV files are indexed as catalog cards (sheets, columns, value ranges, first rows): when a hit has source.kind = spreadsheet, read the rows you need with rag_read_sheet instead of guessing from the card; to find a specific value in it, pass find (and column) rather than paging through every row. rag_list_documents shows what is in a bucket.";

const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
const MAX_DISPATCH_BODY_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_TOP_K: u64 = 10;
const MAX_TOP_K: u64 = 200;
const MAX_FANOUT_BUCKETS: usize = 32;
const SESSION_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MAX_SESSIONS: usize = 4096;

// ============================================================================
// Engine
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpTransport {
    Stdio,
    Http,
}

impl McpTransport {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Stdio => "stdio",
            Self::Http => "http",
        }
    }
}

type LazyState = Arc<tokio::sync::OnceCell<Result<Arc<AppState>, String>>>;

#[derive(Clone)]
enum StateSource {
    Ready(Arc<AppState>),
    Lazy(LazyState),
}

#[derive(Clone)]
enum CallerSource {
    Principal(Box<PrincipalContext>),
    ApiKey {
        key: Option<String>,
        expected_principal: Option<Uuid>,
    },
}

/// Options for the stdio transport.
#[derive(Debug, Clone, Default)]
pub struct StdioOptions {
    pub api_key: Option<String>,
    pub principal_id: Option<Uuid>,
}

/// Transport-independent JSON-RPC 2.0 MCP engine.
#[derive(Clone)]
pub struct McpEngine {
    transport: McpTransport,
    state: StateSource,
    caller: CallerSource,
    tasks: Arc<TaskStore>,
    task_hook: Option<TaskHookFn>,
}

struct RpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl RpcError {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    fn with_data(code: i64, message: impl Into<String>, data: Value) -> Self {
        Self {
            code,
            message: message.into(),
            data: Some(data),
        }
    }

    fn unsupported_version(requested: &str) -> Self {
        let mut supported: Vec<&str> = Vec::with_capacity(
            MODERN_PROTOCOL_VERSIONS.len() + SUPPORTED_PROTOCOL_VERSIONS.len(),
        );
        supported.extend_from_slice(&MODERN_PROTOCOL_VERSIONS);
        supported.extend_from_slice(&SUPPORTED_PROTOCOL_VERSIONS);
        Self::with_data(
            JSONRPC_UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version",
            json!({
                "supported": supported,
                "requested": requested,
            }),
        )
    }

    fn missing_tasks_capability() -> Self {
        Self::with_data(
            JSONRPC_MISSING_REQUIRED_CLIENT_CAPABILITY,
            "Missing required client capability",
            json!({
                "requiredCapabilities": {
                    "extensions": {
                        EXT_TASKS: {},
                    }
                }
            }),
        )
    }
}

struct ToolCtx {
    state: Arc<AppState>,
    principal: PrincipalContext,
    transport: McpTransport,
}

/// Tool-level failure rendered as `isError: true` content.
#[derive(Debug)]
struct ToolFailure {
    code: String,
    message: String,
    status: Option<u16>,
}

impl ToolFailure {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            status: None,
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_params", message)
    }

    fn not_found_or_denied(what: &str) -> Self {
        Self::new("not_found", format!("{what} not found or access denied"))
    }

    fn to_value(&self) -> Value {
        let mut v = json!({
            "error": self.code,
            "code": self.code,
            "message": self.message,
        });
        if let (Some(obj), Some(status)) = (v.as_object_mut(), self.status) {
            obj.insert("status".into(), json!(status));
        }
        v
    }
}

impl From<OpenRagError> for ToolFailure {
    fn from(err: OpenRagError) -> Self {
        let api = crate::error::ApiError::from(err);
        Self {
            code: api.code,
            message: api.message,
            status: Some(api.status.as_u16()),
        }
    }
}

impl McpEngine {
    /// HTTP transport: the caller was already authenticated by the `/mcp` handler.
    pub fn for_http(state: Arc<AppState>, principal: PrincipalContext) -> Self {
        Self {
            transport: McpTransport::Http,
            state: StateSource::Ready(state),
            caller: CallerSource::Principal(Box::new(principal)),
            tasks: default_task_store(),
            task_hook: None,
        }
    }

    /// Stdio transport: `AppState` is built lazily from the environment on the first
    /// tool call so `initialize`/`tools/list` and parse errors work without a database.
    pub fn for_stdio(opts: StdioOptions) -> Self {
        Self {
            transport: McpTransport::Stdio,
            state: StateSource::Lazy(Arc::new(tokio::sync::OnceCell::new())),
            caller: CallerSource::ApiKey {
                key: opts.api_key.filter(|k| !k.trim().is_empty()),
                expected_principal: opts.principal_id,
            },
            tasks: default_task_store(),
            task_hook: None,
        }
    }

    /// Stdio transport over an already-built state (used by tests and `openrag serve` embeds).
    pub fn for_stdio_with_state(state: Arc<AppState>, opts: StdioOptions) -> Self {
        Self {
            transport: McpTransport::Stdio,
            state: StateSource::Ready(state),
            caller: CallerSource::ApiKey {
                key: opts.api_key.filter(|k| !k.trim().is_empty()),
                expected_principal: opts.principal_id,
            },
            tasks: default_task_store(),
            task_hook: None,
        }
    }

    /// Replaces the task store used by this engine (for isolated tests).
    #[doc(hidden)]
    pub fn with_task_store(mut self, tasks: Arc<TaskStore>) -> Self {
        self.tasks = tasks;
        self
    }

    /// Injects a test hook future for matching `tools/call` requests.
    #[doc(hidden)]
    pub fn with_task_hook(mut self, hook: TaskHookFn) -> Self {
        self.task_hook = Some(hook);
        self
    }

    /// Handles one decoded JSON-RPC payload (single message or batch).
    /// Returns `None` when nothing must be sent back (notifications / responses only).
    pub async fn handle_message(&self, msg: Value) -> Option<Value> {
        match msg {
            Value::Array(items) => {
                if items.is_empty() {
                    return Some(rpc_error(Value::Null, JSONRPC_INVALID_REQUEST, "empty batch"));
                }
                let mut out = Vec::new();
                for item in items {
                    if let Some(resp) = self.handle_single(item).await {
                        out.push(resp);
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(Value::Array(out))
                }
            }
            other => self.handle_single(other).await,
        }
    }

    async fn handle_single(&self, msg: Value) -> Option<Value> {
        let Value::Object(obj) = msg else {
            return Some(rpc_error(
                Value::Null,
                JSONRPC_INVALID_REQUEST,
                "JSON-RPC message must be an object",
            ));
        };
        let id = obj.get("id").cloned();
        let is_notification = id.is_none();
        let id_val = id.unwrap_or(Value::Null);

        if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return if is_notification {
                None
            } else {
                Some(rpc_error(id_val, JSONRPC_INVALID_REQUEST, "jsonrpc must be \"2.0\""))
            };
        }

        let Some(method) = obj.get("method").and_then(Value::as_str) else {
            // A client-side response (result/error) to a server request: nothing to answer.
            if obj.contains_key("result") || obj.contains_key("error") {
                return None;
            }
            return Some(rpc_error(id_val, JSONRPC_INVALID_REQUEST, "missing method"));
        };

        if is_notification {
            // notifications/initialized, notifications/cancelled, ... need no reply.
            return None;
        }

        let params = obj.get("params").cloned().unwrap_or_else(|| json!({}));

        // Dual-era dispatch: `initialize` is always legacy; any other request carrying
        // `params._meta["io.modelcontextprotocol/protocolVersion"]` speaks modern MCP.
        if method != "initialize" && modern_protocol_version_in_params(&params).is_some() {
            let outcome = self.handle_modern(method, &params).await;
            return Some(match outcome {
                Ok(result) => json!({ "jsonrpc": "2.0", "id": id_val, "result": result }),
                Err(err) => rpc_error_response(id_val, err),
            });
        }

        let outcome = match method {
            "initialize" => Ok(initialize_result(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_definitions() })),
            "tools/call" => self.tools_call(&params).await,
            _ => Err((JSONRPC_METHOD_NOT_FOUND, format!("method not found: {method}"))),
        };

        Some(match outcome {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id_val, "result": result }),
            Err((code, message)) => rpc_error(id_val, code, &message),
        })
    }

    async fn handle_modern(&self, method: &str, params: &Value) -> Result<Value, RpcError> {
        let Some(version) = modern_protocol_version_in_params(params).and_then(Value::as_str) else {
            return Err(RpcError::new(
                JSONRPC_INVALID_PARAMS,
                format!("_meta['{META_PROTOCOL_VERSION}'] must be a string"),
            ));
        };
        if !MODERN_PROTOCOL_VERSIONS.contains(&version) {
            return Err(RpcError::unsupported_version(version));
        }

        // Capabilities are stateless and read per request; nothing is remembered across calls.
        let client_caps = extract_client_capabilities(params)?;

        match method {
            "server/discover" => Ok(finalize_modern_result(discover_result(), "complete", true)),
            "tools/list" => {
                validate_optional_cursor(params)?;
                Ok(finalize_modern_result(
                    json!({ "tools": tool_definitions() }),
                    "complete",
                    true,
                ))
            }
            "tools/call" => self.modern_tools_call(params, &client_caps).await,
            "resources/list" => {
                validate_optional_cursor(params)?;
                let payload = mcp_skills::list_resources()
                    .map_err(|e| RpcError::new(JSONRPC_INTERNAL_ERROR, e))?;
                Ok(finalize_modern_result(payload, "complete", true))
            }
            "resources/templates/list" => {
                validate_optional_cursor(params)?;
                Ok(finalize_modern_result(
                    mcp_skills::list_resource_templates(),
                    "complete",
                    true,
                ))
            }
            "resources/read" => {
                let uri = required_str_param(params, "uri", "resources/read")?;
                let payload = mcp_skills::read_resource(uri)
                    .map_err(|e| RpcError::new(JSONRPC_INTERNAL_ERROR, e))?
                    .ok_or_else(|| {
                        RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown resource URI: {uri}"))
                    })?;
                Ok(finalize_modern_result(payload, "complete", true))
            }
            "skills/list" => {
                validate_optional_cursor(params)?;
                let payload = mcp_skills::list_skills()
                    .map_err(|e| RpcError::new(JSONRPC_INTERNAL_ERROR, e))?;
                Ok(finalize_modern_result(payload, "complete", true))
            }
            "skills/get" => {
                let uri = required_str_param(params, "uri", "skills/get")?;
                let payload = mcp_skills::get_skill(uri)
                    .map_err(|e| RpcError::new(JSONRPC_INTERNAL_ERROR, e))?
                    .ok_or_else(|| {
                        RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown skill URI: {uri}"))
                    })?;
                Ok(finalize_modern_result(payload, "complete", true))
            }
            "tasks/get" => self.tasks_get(params, &client_caps).await,
            "tasks/update" => self.tasks_update(params, &client_caps).await,
            "tasks/cancel" => self.tasks_cancel(params, &client_caps).await,
            _ => Err(RpcError::new(
                JSONRPC_METHOD_NOT_FOUND,
                format!("method not found: {method}"),
            )),
        }
    }

    async fn modern_tools_call(
        &self,
        params: &Value,
        client_caps: &Map<String, Value>,
    ) -> Result<Value, RpcError> {
        let (name, args) = parse_tool_call_params(params)?;
        let supports_tasks = client_supports_tasks(client_caps);

        if let Some(hook) = &self.task_hook {
            if let Some(fut) = hook(name, &args) {
                if supports_tasks {
                    let owner = match self.resolve_principal_key().await {
                        Ok(owner) => owner,
                        Err(fail) => {
                            return Ok(finalize_modern_result(tool_error(&fail), "complete", false));
                        }
                    };
                    if let Some((task_id, create_payload)) =
                        self.tasks.try_create(owner, Some(format!("Running {name}...")))
                    {
                        let tasks = Arc::clone(&self.tasks);
                        let tid = task_id.clone();
                        let join = tokio::spawn(async move {
                            match fut.await {
                                TaskHookOutcome::ToolOk(v) => {
                                    let res =
                                        finalize_modern_result(tool_success(v), "complete", false);
                                    tasks.complete(&tid, res);
                                }
                                TaskHookOutcome::ToolErr { code, message } => {
                                    let res = finalize_modern_result(
                                        tool_error(&ToolFailure::new(code, message)),
                                        "complete",
                                        false,
                                    );
                                    tasks.complete(&tid, res);
                                }
                                TaskHookOutcome::RpcErr { code, message } => {
                                    tasks.fail(
                                        &tid,
                                        json!({ "code": code, "message": message }),
                                        Some(message),
                                    );
                                }
                            }
                        });
                        self.tasks.set_abort_handle(&task_id, join.abort_handle());
                        return Ok(finalize_modern_result(create_payload, "task", false));
                    }
                }
                return match fut.await {
                    TaskHookOutcome::ToolOk(v) => {
                        Ok(finalize_modern_result(tool_success(v), "complete", false))
                    }
                    TaskHookOutcome::ToolErr { code, message } => Ok(finalize_modern_result(
                        tool_error(&ToolFailure::new(code, message)),
                        "complete",
                        false,
                    )),
                    TaskHookOutcome::RpcErr { code, message } => Err(RpcError::new(code, message)),
                };
            }
        }

        let is_ingest_file = name == "rag_ingest"
            && args
                .get("path")
                .and_then(Value::as_str)
                .is_some_and(|p| !p.trim().is_empty());
        let is_watch_scan =
            name == "docs_status" && args.get("scan").and_then(Value::as_bool) == Some(true);

        if supports_tasks && (is_ingest_file || is_watch_scan) {
            let ctx = match self.resolve_ctx().await {
                Ok(ctx) => ctx,
                Err(fail) => {
                    return Ok(finalize_modern_result(tool_error(&fail), "complete", false));
                }
            };
            let owner = PrincipalKey {
                org_id: ctx.principal.org_id,
                principal_id: ctx.principal.principal_id,
            };
            let initial_msg = if is_ingest_file {
                "Ingesting file..."
            } else {
                "Scanning documents folders..."
            };
            if let Some((task_id, create_payload)) =
                self.tasks.try_create(owner, Some(initial_msg.to_string()))
            {
                let tasks = Arc::clone(&self.tasks);
                let tid = task_id.clone();
                let tool_name = name.to_string();
                let join = tokio::spawn(async move {
                    let raw_outcome = if is_watch_scan {
                        docs_status_scan_task(&ctx, &tasks, &tid).await
                    } else {
                        call_tool(&ctx, &tool_name, &args).await
                    };
                    let final_outcome = match raw_outcome {
                        Ok(value) => seal_for_ai(&ctx, value).await,
                        Err(fail) => Err(fail),
                    };
                    let tool_res = match final_outcome {
                        Ok(value) => tool_success(value),
                        Err(fail) => tool_error(&fail),
                    };
                    tasks.complete(&tid, finalize_modern_result(tool_res, "complete", false));
                });
                self.tasks.set_abort_handle(&task_id, join.abort_handle());
                return Ok(finalize_modern_result(create_payload, "task", false));
            }
        }

        let sync_result = self
            .tools_call_parsed(name, &args)
            .await;
        Ok(finalize_modern_result(sync_result, "complete", false))
    }

    async fn tasks_get(
        &self,
        params: &Value,
        client_caps: &Map<String, Value>,
    ) -> Result<Value, RpcError> {
        if !client_supports_tasks(client_caps) {
            return Err(RpcError::missing_tasks_capability());
        }
        let task_id = required_str_param(params, "taskId", "tasks/get")?;
        let owner = self.resolve_principal_key().await.map_err(|_| {
            RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown or expired taskId: {task_id}"))
        })?;
        let task = self.tasks.get(task_id, owner).ok_or_else(|| {
            RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown or expired taskId: {task_id}"))
        })?;
        Ok(finalize_modern_result(task, "complete", false))
    }

    async fn tasks_update(
        &self,
        params: &Value,
        client_caps: &Map<String, Value>,
    ) -> Result<Value, RpcError> {
        if !client_supports_tasks(client_caps) {
            return Err(RpcError::missing_tasks_capability());
        }
        let task_id = required_str_param(params, "taskId", "tasks/update")?;
        if !params.get("inputResponses").is_some_and(Value::is_object) {
            return Err(RpcError::new(
                JSONRPC_INVALID_PARAMS,
                "tasks/update requires an object 'inputResponses'",
            ));
        }
        let owner = self.resolve_principal_key().await.map_err(|_| {
            RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown or expired taskId: {task_id}"))
        })?;
        if !self.tasks.acknowledge_update(task_id, owner) {
            return Err(RpcError::new(
                JSONRPC_INVALID_PARAMS,
                format!("unknown or expired taskId: {task_id}"),
            ));
        }
        Ok(finalize_modern_result(json!({}), "complete", false))
    }

    async fn tasks_cancel(
        &self,
        params: &Value,
        client_caps: &Map<String, Value>,
    ) -> Result<Value, RpcError> {
        if !client_supports_tasks(client_caps) {
            return Err(RpcError::missing_tasks_capability());
        }
        let task_id = required_str_param(params, "taskId", "tasks/cancel")?;
        let owner = self.resolve_principal_key().await.map_err(|_| {
            RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown or expired taskId: {task_id}"))
        })?;
        if !self.tasks.cancel(task_id, owner) {
            return Err(RpcError::new(
                JSONRPC_INVALID_PARAMS,
                format!("unknown or expired taskId: {task_id}"),
            ));
        }
        Ok(finalize_modern_result(json!({}), "complete", false))
    }

    async fn resolve_principal_key(&self) -> Result<PrincipalKey, ToolFailure> {
        if let CallerSource::Principal(p) = &self.caller {
            return Ok(PrincipalKey {
                org_id: p.org_id,
                principal_id: p.principal_id,
            });
        }
        let ctx = self.resolve_ctx().await?;
        Ok(PrincipalKey {
            org_id: ctx.principal.org_id,
            principal_id: ctx.principal.principal_id,
        })
    }

    async fn tools_call(&self, params: &Value) -> Result<Value, (i64, String)> {
        let (name, args) = parse_tool_call_params(params).map_err(|e| (e.code, e.message))?;
        Ok(self.tools_call_parsed(name, &args).await)
    }

    async fn tools_call_parsed(&self, name: &str, args: &Map<String, Value>) -> Value {
        let outcome = match self.resolve_ctx().await {
            Ok(ctx) => match call_tool(&ctx, name, args).await {
                Ok(value) => seal_for_ai(&ctx, value).await,
                Err(fail) => Err(fail),
            },
            Err(fail) => Err(fail),
        };
        match outcome {
            Ok(value) => tool_success(value),
            Err(fail) => tool_error(&fail),
        }
    }

    /// Builds the lazily-initialized stdio state. Kept out of `handle_message` so the
    /// HTTP handler future stays `Send`; the stdio loop calls it before `tools/call`.
    pub async fn ensure_state(&self) {
        if let StateSource::Lazy(cell) = &self.state {
            if cell.get().is_none() {
                let res = AppState::from_env().await.map_err(|e| e.to_string());
                let _ = cell.set(res);
            }
        }
    }

    fn resolve_state(&self) -> Result<Arc<AppState>, ToolFailure> {
        match &self.state {
            StateSource::Ready(s) => Ok(Arc::clone(s)),
            StateSource::Lazy(cell) => match cell.get() {
                Some(Ok(s)) => Ok(Arc::clone(s)),
                Some(Err(e)) => Err(ToolFailure::new(
                    "server_unavailable",
                    format!("OpenRAG state could not be initialized (check DATABASE_URL): {e}"),
                )),
                None => Err(ToolFailure::new("server_unavailable", "OpenRAG state is not initialized")),
            },
        }
    }

    async fn resolve_ctx(&self) -> Result<ToolCtx, ToolFailure> {
        let state = self.resolve_state()?;
        let principal = match &self.caller {
            CallerSource::Principal(p) => (**p).clone(),
            CallerSource::ApiKey {
                key,
                expected_principal,
            } => {
                let Some(key) = key.as_deref() else {
                    return Err(ToolFailure {
                        code: "unauthorized".into(),
                        message: "missing API key: pass --api-key or set OPENRAG_API_KEY".into(),
                        status: Some(401),
                    });
                };
                let worker = state.worker_pool.as_ref().ok_or_else(|| ToolFailure {
                    code: "unauthorized".into(),
                    message: "authentication database not configured (set DATABASE_URL)".into(),
                    status: Some(401),
                })?;
                let ctx = authenticate_bearer_token(worker, key).await.map_err(|e| {
                    let mut f = ToolFailure::from(e);
                    if f.status == Some(500) {
                        f.message = "authentication failed".into();
                    }
                    f
                })?;
                if let Some(expected) = expected_principal {
                    if *expected != ctx.principal_id {
                        return Err(ToolFailure {
                            code: "unauthorized".into(),
                            message: "--principal does not match the API key's principal".into(),
                            status: Some(401),
                        });
                    }
                }
                ctx
            }
        };
        Ok(ToolCtx {
            state,
            principal,
            transport: self.transport,
        })
    }
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

fn rpc_error_response(id: Value, err: RpcError) -> Value {
    let mut error_obj = json!({
        "code": err.code,
        "message": err.message,
    });
    if let (Some(obj), Some(data)) = (error_obj.as_object_mut(), err.data) {
        obj.insert("data".into(), data);
    }
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": error_obj,
    })
}

fn modern_protocol_version_in_params(params: &Value) -> Option<&Value> {
    params
        .get("_meta")
        .and_then(Value::as_object)
        .and_then(|meta| meta.get(META_PROTOCOL_VERSION))
}

fn modern_protocol_version_in_msg(msg: &Value) -> Option<&Value> {
    if msg.get("method").and_then(Value::as_str) == Some("initialize") {
        return None;
    }
    msg.get("params").and_then(modern_protocol_version_in_params)
}

fn extract_client_capabilities(params: &Value) -> Result<Map<String, Value>, RpcError> {
    match params
        .get("_meta")
        .and_then(Value::as_object)
        .and_then(|meta| meta.get(META_CLIENT_CAPABILITIES))
    {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(_) => Err(RpcError::new(
            JSONRPC_INVALID_PARAMS,
            format!("_meta['{META_CLIENT_CAPABILITIES}'] must be an object"),
        )),
    }
}

fn client_supports_tasks(client_caps: &Map<String, Value>) -> bool {
    client_caps
        .get("extensions")
        .and_then(Value::as_object)
        .and_then(|ext| ext.get(EXT_TASKS))
        .is_some_and(Value::is_object)
}

fn validate_optional_cursor(params: &Value) -> Result<(), RpcError> {
    match params.get("cursor") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(s)) if s.is_empty() => Ok(()),
        _ => Err(RpcError::new(JSONRPC_INVALID_PARAMS, "invalid cursor")),
    }
}

fn required_str_param<'a>(params: &'a Value, field: &str, method: &str) -> Result<&'a str, RpcError> {
    params
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(JSONRPC_INVALID_PARAMS, format!("{method} requires a string '{field}'")))
}

fn parse_tool_call_params(params: &Value) -> Result<(&str, Map<String, Value>), RpcError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(JSONRPC_INVALID_PARAMS, "tools/call requires a string 'name'"))?;
    if !MCP_TOOL_NAMES.contains(&name) {
        return Err(RpcError::new(JSONRPC_INVALID_PARAMS, format!("unknown tool: {name}")));
    }
    let args = match params.get("arguments") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => {
            return Err(RpcError::new(
                JSONRPC_INVALID_PARAMS,
                "tools/call 'arguments' must be an object",
            ))
        }
    };
    Ok((name, args))
}

fn server_info() -> Value {
    json!({ "name": "openrag", "version": env!("CARGO_PKG_VERSION") })
}

fn finalize_modern_result(mut result: Value, result_type: &str, cacheable: bool) -> Value {
    if let Some(obj) = result.as_object_mut() {
        obj.insert("resultType".into(), json!(result_type));
        if cacheable {
            obj.insert("ttlMs".into(), json!(CACHE_TTL_MS));
            obj.insert("cacheScope".into(), json!(CACHE_SCOPE_PRIVATE));
        }
        let meta = obj
            .entry("_meta")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(meta_obj) = meta.as_object_mut() {
            meta_obj.insert(META_SERVER_INFO.into(), server_info());
        }
    }
    result
}

fn discover_result() -> Value {
    json!({
        "supportedVersions": MODERN_PROTOCOL_VERSIONS,
        "capabilities": {
            "tools": {},
            "resources": {},
            "extensions": {
                EXT_SKILLS: {},
                EXT_TASKS: {},
            },
        },
        "instructions": format!(
            "{LEGACY_INSTRUCTIONS} For a full search workflow guide, read the openrag-search skill at skill://openrag-search/SKILL.md."
        ),
    })
}

fn initialize_result(params: &Value) -> Value {
    let requested = params.get("protocolVersion").and_then(Value::as_str);
    let version = requested
        .filter(|v| SUPPORTED_PROTOCOL_VERSIONS.contains(v))
        .unwrap_or(DEFAULT_PROTOCOL_VERSION);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": server_info(),
        "instructions": LEGACY_INSTRUCTIONS,
    })
}

fn tool_success(value: Value) -> Value {
    let text = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
    let mut out = json!({
        "content": [{ "type": "text", "text": text }],
        "isError": false,
    });
    if value.is_object() {
        if let Some(obj) = out.as_object_mut() {
            obj.insert("structuredContent".into(), value);
        }
    }
    out
}

fn tool_error(fail: &ToolFailure) -> Value {
    let body = fail.to_value();
    let text = serde_json::to_string(&body).unwrap_or_else(|_| fail.message.clone());
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": true,
    })
}

// ============================================================================
// Tool registry (schemas)
// ============================================================================

fn str_prop(desc: &str) -> Value {
    json!({ "type": "string", "description": desc })
}

fn int_prop(desc: &str, min: i64, max: i64) -> Value {
    json!({ "type": "integer", "description": desc, "minimum": min, "maximum": max })
}

fn str_list_prop(desc: &str) -> Value {
    json!({ "type": "array", "items": { "type": "string" }, "description": desc })
}

fn scope_prop() -> Value {
    json!({ "type": "string", "enum": ["private", "dept", "group", "org"], "description": "Access scope" })
}

fn backend_prop() -> Value {
    json!({ "type": "string", "enum": ["openrag_pg", "kumiho"], "description": "Memory backend override" })
}

fn schema(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })
}

fn tool(name: &str, title: &str, description: &str, input_schema: Value, read_only: bool) -> Value {
    json!({
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": input_schema,
        "annotations": {
            "title": title,
            "readOnlyHint": read_only,
            "destructiveHint": false,
            "openWorldHint": false,
        },
    })
}

/// MCP `tools/list` payload — exactly [`MCP_TOOL_NAMES`], in order.
pub fn tool_definitions() -> Vec<Value> {
    let tags = str_list_prop("Tag names (leading '#' optional)");
    vec![
        tool(
            "rag_search",
            "Search documents",
            "Vector search (with concept expansion) over the documents you may read. By default only documents in force now are returned (expired, not-yet-effective, superseded and older revisions are dropped); pass as_of for another reference date. If nothing in force matches, non-current matches come back separately in not_in_force with their validity — never present those as current. Omit 'bucket' to search every bucket you can access. Inline #tag / tag:name in the query filters by tag. Split a question with several parts or conditions into one search per part, and retry weak results with other wording or synonyms.",
            schema(
                json!({
                    "query": str_prop("Search query"),
                    "bucket": str_prop("Bucket slug or id (optional)"),
                    "top_k": int_prop("Maximum results", 1, MAX_TOP_K as i64),
                    "spaces": { "type": "array", "items": { "type": "string", "enum": ["chunks", "thumb", "wiki", "memory"] }, "description": "Search spaces" },
                    "fusion": { "type": "string", "enum": ["rrf", "convex"], "description": "Score fusion" },
                    "tags_any": tags.clone(),
                    "tags_all": tags.clone(),
                    "tags_exclude": tags.clone(),
                    "expand_tag_aliases": { "type": "boolean", "description": "Expand tags through concept aliases" },
                    "latest_revision_only": { "type": "boolean", "description": "Default true: only documents in force at the reference time. false also returns expired/scheduled/superseded/older revisions, each labeled in 'validity'" },
                    "as_of": { "type": "string", "description": "Reference date (YYYY-MM-DD or RFC 3339), e.g. the date an event happened. Default: now. Future dates surface scheduled documents." },
                    "hypothetical_answer": str_prop("Optional, for vague or descriptive questions: two or three sentences that would answer the question, worded the way a company document would put it. Used only to find similar passages; it is never shown or stored, so a plausible guess is fine. Report facts only from the results."),
                }),
                &["query"],
            ),
            true,
        ),
        tool(
            "rag_ask",
            "Ask with citations",
            "Answer a question from your accessible documents with citations. By default only documents in force at the reference time ground the answer. The policy router picks the LLM (L8+ context stays on-premise); blocked safe-words return guard_blocked.",
            schema(
                json!({
                    "query": str_prop("Question"),
                    "bucket": str_prop("Bucket slug or id (optional)"),
                    "top_k": int_prop("Context passages", 1, 50),
                    "mode": str_prop("Model label passed to the policy router (default openrag-auto)"),
                    "latest_revision_only": { "type": "boolean", "description": "Default true: ground the answer only on documents in force at the reference time. false also uses expired/scheduled/superseded/older revisions, each labeled in citations[].validity" },
                    "as_of": { "type": "string", "description": "Reference date (YYYY-MM-DD or RFC 3339), e.g. the date the question is about. Default: now." },
                }),
                &["query"],
            ),
            true,
        ),
        tool(
            "rag_ingest",
            "Add a document",
            "Ingest text (or, over local stdio only, a file path) into a bucket. Content passes the ingestion guard and is stored as is; PII is masked whenever it is sent to an AI.",
            schema(
                json!({
                    "bucket": str_prop("Bucket slug or id"),
                    "path": str_prop("Local file path (stdio transport only)"),
                    "text": str_prop("Document text"),
                    "title": str_prop("Document title"),
                    "access_scope": scope_prop(),
                    "allowed_group_ids": str_list_prop("Allowed permission group ids"),
                    "denied_group_ids": str_list_prop("Denied permission group ids (explicit deny wins)"),
                    "min_auth_level": int_prop("Minimum permission level", 1, 10),
                    "tags": tags.clone(),
                    "valid_from": { "type": "string", "format": "date-time", "description": "When the document takes effect (RFC 3339, e.g. its authoring/enactment date). Defaults to ingestion time." },
                    "valid_to": { "type": "string", "format": "date-time", "description": "When the document stops being valid (RFC 3339). Expired documents are excluded from search." },
                    "revision": int_prop("Revision number; newer revisions of the same title are preferred", 1, 1_000_000),
                }),
                &["bucket"],
            ),
            false,
        ),
        tool(
            "rag_get_page",
            "Read a page",
            "Read one document page (redacted text, figure boxes, thumbnail link). Pass page_id, or document_id with page_no (default 1).",
            schema(
                json!({
                    "page_id": str_prop("Page id"),
                    "document_id": str_prop("Document id"),
                    "page_no": int_prop("1-based page number", 1, 100_000),
                }),
                &[],
            ),
            true,
        ),
        tool(
            "rag_list_documents",
            "List documents",
            "List the documents you may read in a bucket (title, original path, revision, whether in force, number of passages). Filter by part of the title or path with 'query'. By default only documents in force are listed; include_closed also lists superseded and removed ones. Omit 'bucket' to list every bucket you can access.",
            schema(
                json!({
                    "bucket": str_prop("Bucket slug or id (optional)"),
                    "query": str_prop("Part of the title or path"),
                    "include_closed": { "type": "boolean", "description": "Also list documents no longer in force" },
                    "limit": int_prop("Maximum documents per bucket", 1, 500),
                    "offset": int_prop("Skip this many documents", 0, 1_000_000),
                }),
                &[],
            ),
            true,
        ),
        tool(
            "rag_read_sheet",
            "Read spreadsheet rows",
            "Read rows of an Excel (.xlsx) or CSV document found by rag_search or rag_list_documents (source.kind = spreadsheet). Search only indexes the file's catalog card, so look up a specific value (an order number, a name, a date like 2024-03) here with 'find': only rows with a cell containing it come back, with total_matches and row_index (each row's position; pass it as 'offset' without 'find' to read the rows around it). Case and thousands separators are ignored; 'column' limits the search to one column, 'exact' requires the whole cell to match. Without 'find', returns the rows from 'offset' (below the header). Either way: column names, total_rows, next_offset for paging, and a Markdown table. Omit 'sheet' for the first sheet, or with 'find' to search every sheet (matches_by_sheet counts them; rows come from the first sheet with a match). Only files from a watched documents folder can be read this way.",
            schema(
                json!({
                    "document_id": str_prop("Document id"),
                    "sheet": str_prop("Sheet name (default: the first sheet; with find, every sheet)"),
                    "find": str_prop("Keep only rows with a cell containing this text"),
                    "column": str_prop("Look for 'find' in this column only (a name from columns)"),
                    "exact": { "type": "boolean", "description": "The whole cell must equal 'find'" },
                    "offset": int_prop("First data row (or first match, with find), 0-based", 0, 10_000_000),
                    "limit": int_prop("Rows to return (default 50)", 1, 500),
                }),
                &["document_id"],
            ),
            true,
        ),
        tool(
            "docs_status",
            "Documents folder status",
            "What the documents-folder watcher is doing (organization admins): watched folders, documents in force per bucket, files that failed to ingest and why, progress of a running scan. Pass scan=true to scan the folders now instead of waiting for the next interval.",
            schema(
                json!({
                    "scan": { "type": "boolean", "description": "Start a scan now" },
                }),
                &[],
            ),
            false,
        ),
        tool(
            "wiki_search",
            "Search the wiki",
            "Search wiki pages you may read (concept/alias aware).",
            schema(
                json!({
                    "query": str_prop("Search query"),
                    "bucket": str_prop("Bucket slug or id (optional)"),
                    "top_k": int_prop("Maximum results", 1, MAX_TOP_K as i64),
                }),
                &["query"],
            ),
            true,
        ),
        tool(
            "wiki_get",
            "Read a wiki page",
            "Read one wiki page by page_id or slug.",
            schema(
                json!({
                    "page_id": str_prop("Wiki page id"),
                    "slug": str_prop("Wiki page slug"),
                    "bucket": str_prop("Bucket slug or id to disambiguate a slug (optional)"),
                }),
                &[],
            ),
            true,
        ),
        tool(
            "memory_search",
            "Search memories",
            "Search the active memory view (retracted and superseded entries are excluded).",
            schema(
                json!({
                    "query": str_prop("Search query"),
                    "scope": scope_prop(),
                    "top_k": int_prop("Maximum results", 1, MAX_TOP_K as i64),
                }),
                &["query"],
            ),
            true,
        ),
        tool(
            "memory_patch",
            "Correct a memory",
            "Append an immutable patch to a memory entry: amend, supersede, retract or redact.",
            schema(
                json!({
                    "memory_id": str_prop("Memory entry id"),
                    "op": { "type": "string", "enum": ["amend", "supersede", "retract", "redact"], "description": "Patch operation" },
                    "text": str_prop("New text (amend/supersede)"),
                    "reason": str_prop("Reason recorded in the patch ledger"),
                }),
                &["memory_id", "op"],
            ),
            false,
        ),
        tool(
            "memory_engage",
            "Recall memories",
            "Cognitive recall: concept-expanded memory retrieval with belief-revision edges.",
            schema(
                json!({
                    "query": str_prop("What to recall"),
                    "scope": scope_prop(),
                    "top_k": int_prop("Maximum memories", 1, MAX_TOP_K as i64),
                    "backend": backend_prop(),
                }),
                &["query"],
            ),
            true,
        ),
        tool(
            "memory_reflect",
            "Record a memory",
            "Record an episode and distilled memory entry; optionally link it to an existing memory (derived_from, supports, contradicts, supersedes). Content passes the ingestion guard.",
            schema(
                json!({
                    "content": str_prop("Memory content"),
                    "title": str_prop("Short title"),
                    "episode_id": str_prop("Existing episode id"),
                    "scope": scope_prop(),
                    "dept_id": str_prop("Home department id (defaults to your first department)"),
                    "min_auth_level": int_prop("Minimum permission level (<= your level)", 1, 10),
                    "edge_type": { "type": "string", "enum": ["derived_from", "supports", "contradicts", "supersedes"], "description": "Edge to target_memory_id" },
                    "target_memory_id": str_prop("Memory to link with edge_type"),
                    "tags": tags,
                    "backend": backend_prop(),
                }),
                &["content"],
            ),
            false,
        ),
        tool(
            "guard_check",
            "Check text with the guard",
            "Evaluate text against the active PII/Safe-Word guard policy. Returns allow, warn, mask or block plus sanitized text.",
            schema(
                json!({
                    "text": str_prop("Text to check"),
                    "surface": { "type": "string", "enum": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"], "description": "Guard surface (default mcp_egress)" },
                }),
                &["text"],
            ),
            true,
        ),
        tool(
            "concept_link",
            "Link synonyms",
            "Link two terms (e.g. Korean and English) as aliases of one canonical concept.",
            schema(
                json!({
                    "source": str_prop("First term"),
                    "target": str_prop("Second term"),
                    "canonical_name": str_prop("Canonical concept name"),
                    "lang": str_prop("Language code of the alias"),
                    "relation": str_prop("Relation label"),
                    "concept_id": str_prop("Existing concept id to attach to"),
                }),
                &["source", "target"],
            ),
            false,
        ),
        tool(
            "concept_resolve",
            "Resolve a term",
            "Resolve a term to its canonical concept and aliases (exact, alias, optional 256d semantic match).",
            schema(
                json!({
                    "term": str_prop("Term to resolve"),
                    "include_aliases": { "type": "boolean", "description": "Include alias list" },
                    "use_semantic": { "type": "boolean", "description": "Use 256d semantic matching" },
                    "top_k": int_prop("Maximum matches", 1, 50),
                    "embedding": { "type": "array", "items": { "type": "number" }, "minItems": 256, "maxItems": 256, "description": "Optional 256d query embedding" },
                    "threshold": { "type": "number", "minimum": 0, "maximum": 1, "description": "Semantic match threshold" },
                }),
                &["term"],
            ),
            true,
        ),
    ]
}

// ============================================================================
// Argument helpers
// ============================================================================

fn arg_str_opt(args: &Map<String, Value>, key: &str) -> Result<Option<String>, ToolFailure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(ToolFailure::invalid(format!("'{key}' must be a string"))),
    }
}

fn arg_str_req(args: &Map<String, Value>, key: &str) -> Result<String, ToolFailure> {
    match arg_str_opt(args, key)? {
        Some(s) if !s.trim().is_empty() => Ok(s),
        Some(_) => Err(ToolFailure::invalid(format!("'{key}' must not be empty"))),
        None => Err(ToolFailure::invalid(format!("missing required argument '{key}'"))),
    }
}

fn arg_u64_opt(args: &Map<String, Value>, key: &str) -> Result<Option<u64>, ToolFailure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| ToolFailure::invalid(format!("'{key}' must be a non-negative integer"))),
    }
}

fn arg_bool_opt(args: &Map<String, Value>, key: &str) -> Result<Option<bool>, ToolFailure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(ToolFailure::invalid(format!("'{key}' must be a boolean"))),
    }
}

fn arg_str_list_opt(args: &Map<String, Value>, key: &str) -> Result<Option<Vec<String>>, ToolFailure> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| ToolFailure::invalid(format!("'{key}' must be an array of strings")))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(ToolFailure::invalid(format!("'{key}' must be an array of strings"))),
    }
}

fn arg_uuid_opt(args: &Map<String, Value>, key: &str) -> Result<Option<Uuid>, ToolFailure> {
    match arg_str_opt(args, key)? {
        None => Ok(None),
        Some(s) => Uuid::parse_str(s.trim())
            .map(Some)
            .map_err(|_| ToolFailure::invalid(format!("'{key}' must be a UUID"))),
    }
}

fn arg_uuid_list_opt(args: &Map<String, Value>, key: &str) -> Result<Option<Vec<Uuid>>, ToolFailure> {
    match arg_str_list_opt(args, key)? {
        None => Ok(None),
        Some(list) => list
            .iter()
            .map(|s| {
                Uuid::parse_str(s.trim())
                    .map_err(|_| ToolFailure::invalid(format!("'{key}' must contain UUIDs")))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
    }
}

fn arg_datetime_opt(
    args: &Map<String, Value>,
    key: &str,
) -> Result<Option<chrono::DateTime<chrono::Utc>>, ToolFailure> {
    match arg_str_opt(args, key)?.filter(|s| !s.trim().is_empty()) {
        None => Ok(None),
        Some(s) => crate::cli::parse_datetime_arg(&s)
            .map(Some)
            .map_err(|e| ToolFailure::invalid(format!("'{key}': {e}"))),
    }
}

fn arg_scope_opt(args: &Map<String, Value>, key: &str) -> Result<Option<AccessScope>, ToolFailure> {
    match arg_str_opt(args, key)? {
        None => Ok(None),
        Some(s) => AccessScope::from_str(s.trim())
            .map(Some)
            .map_err(|_| ToolFailure::invalid(format!("'{key}' must be one of private, dept, group, org"))),
    }
}

fn arg_backend_opt(args: &Map<String, Value>) -> Result<Option<MemoryBackendKind>, ToolFailure> {
    match arg_str_opt(args, "backend")? {
        None => Ok(None),
        Some(s) => serde_json::from_value::<MemoryBackendKind>(Value::String(s.trim().to_string()))
            .map(Some)
            .map_err(|_| ToolFailure::invalid("'backend' must be openrag_pg or kumiho")),
    }
}

fn top_k(args: &Map<String, Value>, default: u64) -> Result<u64, ToolFailure> {
    match arg_u64_opt(args, "top_k")? {
        None => Ok(default),
        Some(0) => Err(ToolFailure::invalid("'top_k' must be >= 1")),
        Some(n) => Ok(n.min(MAX_TOP_K)),
    }
}

/// Percent-encodes everything except RFC 3986 unreserved characters.
pub(crate) fn pct_encode(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for b in raw.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn insert_opt(obj: &mut Map<String, Value>, key: &str, val: Option<Value>) {
    if let Some(v) = val {
        obj.insert(key.to_string(), v);
    }
}

// ============================================================================
// In-process dispatch to the REST handlers
// ============================================================================

pub(crate) enum DispatchBody {
    Empty,
    Json(Value),
    Multipart { boundary: String, bytes: Vec<u8> },
}

/// Runs one request through the REST router as `principal` (no network, no re-auth).
/// Returns the decoded JSON body on 2xx, the handler's error body otherwise.
pub(crate) async fn dispatch_as(
    state: &Arc<AppState>,
    principal: &PrincipalContext,
    method: Method,
    uri: &str,
    body: DispatchBody,
) -> Result<(StatusCode, Value), (StatusCode, Value)> {
    let (content_type, bytes) = match body {
        DispatchBody::Empty => (None, Vec::new()),
        DispatchBody::Json(v) => (
            Some("application/json".to_string()),
            serde_json::to_vec(&v).unwrap_or_default(),
        ),
        DispatchBody::Multipart { boundary, bytes } => {
            (Some(format!("multipart/form-data; boundary={boundary}")), bytes)
        }
    };

    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(ct) = content_type {
        builder = builder.header(CONTENT_TYPE, ct);
    }
    let mut req = match builder.body(Body::from(bytes)) {
        Ok(r) => r,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "invalid_request", "message": format!("could not build request: {e}") }),
            ))
        }
    };
    req.extensions_mut().insert(TrustedPrincipal(principal.clone()));

    let router = crate::api_router().with_state(Arc::clone(state));
    let resp = match router.oneshot(req).await {
        Ok(r) => r,
        Err(never) => match never {},
    };
    let status = resp.status();
    let raw = match axum::body::to_bytes(resp.into_body(), MAX_DISPATCH_BODY_BYTES).await {
        Ok(b) => b,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": "internal_error", "message": format!("response body: {e}") }),
            ))
        }
    };
    let decoded = if raw.is_empty() {
        json!({})
    } else {
        serde_json::from_slice::<Value>(&raw)
            .unwrap_or_else(|_| json!({ "text": String::from_utf8_lossy(&raw).to_string() }))
    };
    if status.is_success() {
        Ok((status, decoded))
    } else {
        Err((status, decoded))
    }
}

fn failure_from_http(status: StatusCode, body: &Value) -> ToolFailure {
    let code = body
        .get("error")
        .and_then(Value::as_str)
        .map_or_else(|| format!("http_{}", status.as_u16()), ToString::to_string);
    let message = body
        .get("message")
        .and_then(Value::as_str)
        .map_or_else(|| status.canonical_reason().unwrap_or("request failed").to_string(), ToString::to_string);
    ToolFailure {
        code,
        message,
        status: Some(status.as_u16()),
    }
}

async fn dispatch(
    ctx: &ToolCtx,
    method: Method,
    uri: &str,
    body: DispatchBody,
) -> Result<Value, ToolFailure> {
    dispatch_as(&ctx.state, &ctx.principal, method, uri, body)
        .await
        .map(|(_, v)| v)
        .map_err(|(status, v)| failure_from_http(status, &v))
}

/// Minimal `multipart/form-data` encoder for file ingestion.
pub(crate) fn build_multipart(
    file_name: &str,
    file_bytes: &[u8],
    fields: &[(&str, String)],
) -> (String, Vec<u8>) {
    let boundary = format!("openrag-{}", Uuid::new_v4().simple());
    let safe_name: String = file_name
        .chars()
        .map(|c| if c == '"' || c == '\\' || c.is_control() { '_' } else { c })
        .collect();
    let mut out: Vec<u8> = Vec::with_capacity(file_bytes.len() + 512);
    for (name, value) in fields {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        out.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    out.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{safe_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    out.extend_from_slice(file_bytes);
    out.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (boundary, out)
}

// ============================================================================
// PII on the way out
// ============================================================================

/// Tool results go to the MCP client, which is an AI: mask the PII in every string value,
/// the same way as on the way to an LLM provider (`seal_text_for_llm`). Stored data keeps it.
async fn seal_for_ai(ctx: &ToolCtx, mut value: Value) -> Result<Value, ToolFailure> {
    let mut strings = Vec::new();
    collect_strings(&value, &mut strings);
    if strings.is_empty() {
        return Ok(value);
    }
    let pii = ctx.state.pii_engine.read().await.clone();
    let mut unused_restore = Vec::new();
    let mut sealed = Vec::with_capacity(strings.len());
    for s in &strings {
        sealed.push(
            seal_text_for_llm(&ctx.state, ctx.principal.org_id, &pii, s, &mut unused_restore)
                .await?,
        );
    }
    replace_strings(&mut value, &mut sealed.into_iter());
    Ok(value)
}

fn collect_strings(v: &Value, out: &mut Vec<String>) {
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Array(items) => items.iter().for_each(|x| collect_strings(x, out)),
        Value::Object(map) => map.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

/// Walks `v` in the same order as `collect_strings`.
fn replace_strings(v: &mut Value, next: &mut impl Iterator<Item = String>) {
    match v {
        Value::String(s) => {
            if let Some(new) = next.next() {
                *s = new;
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|x| replace_strings(x, next)),
        Value::Object(map) => map.values_mut().for_each(|x| replace_strings(x, next)),
        _ => {}
    }
}

// ============================================================================
// Tools
// ============================================================================

async fn call_tool(ctx: &ToolCtx, name: &str, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    match name {
        "rag_search" => rag_search(ctx, args).await,
        "rag_ask" => rag_ask(ctx, args).await,
        "rag_ingest" => rag_ingest(ctx, args).await,
        "rag_get_page" => rag_get_page(ctx, args).await,
        "rag_list_documents" => rag_list_documents(ctx, args).await,
        "rag_read_sheet" => rag_read_sheet(ctx, args).await,
        "docs_status" => docs_status(ctx, args).await,
        "wiki_search" => wiki_search(ctx, args).await,
        "wiki_get" => wiki_get(ctx, args).await,
        "memory_search" => memory_search(ctx, args).await,
        "memory_patch" => memory_patch(ctx, args).await,
        "memory_engage" => memory_engage(ctx, args).await,
        "memory_reflect" => memory_reflect(ctx, args).await,
        "guard_check" => guard_check(ctx, args).await,
        "concept_link" => concept_link(ctx, args).await,
        "concept_resolve" => concept_resolve(ctx, args).await,
        other => Err(ToolFailure::invalid(format!("unknown tool: {other}"))),
    }
}

async fn rag_search(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let query = arg_str_req(args, "query")?;
    let k = top_k(args, DEFAULT_TOP_K)?;
    let in_force_only = arg_bool_opt(args, "latest_revision_only")?.unwrap_or(true);
    let as_of = arg_datetime_opt(args, "as_of")?;
    let mut body = Map::new();
    body.insert("query".into(), json!(query));
    body.insert("top_k".into(), json!(k));
    insert_opt(&mut body, "spaces", arg_str_list_opt(args, "spaces")?.map(|v| json!(v)));
    insert_opt(&mut body, "fusion", arg_str_opt(args, "fusion")?.map(|v| json!(v)));
    for key in ["tags_any", "tags_all", "tags_exclude"] {
        insert_opt(&mut body, key, arg_str_list_opt(args, key)?.map(|v| json!(v)));
    }
    insert_opt(
        &mut body,
        "expand_tag_aliases",
        arg_bool_opt(args, "expand_tag_aliases")?.map(|v| json!(v)),
    );
    insert_opt(&mut body, "as_of", as_of.map(|d| json!(d.to_rfc3339())));
    insert_opt(
        &mut body,
        "hypothetical_answer",
        arg_str_opt(args, "hypothetical_answer")?
            .map(|h| h.trim().to_string())
            .filter(|h| !h.is_empty())
            .map(|h| json!(h)),
    );
    body.insert("latest_revision_only".into(), json!(in_force_only));
    let bucket = arg_str_opt(args, "bucket")?
        .map(|b| b.trim().to_string())
        .filter(|b| !b.is_empty());
    let reference_time = as_of.unwrap_or_else(chrono::Utc::now).to_rfc3339();

    let mut out = search_once(ctx, bucket.as_deref(), Value::Object(body.clone()), k).await?;
    let empty = out
        .get("results")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty);
    if in_force_only && empty {
        // Nothing in force matched: surface what did match, labeled, instead of an empty
        // answer that invites "no such document" — but keep it out of `results`.
        body.insert("latest_revision_only".into(), json!(false));
        let rest = search_once(ctx, bucket.as_deref(), Value::Object(body), k).await?;
        let not_in_force: Vec<Value> = rest
            .get("results")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter(|h| h.get("validity").and_then(Value::as_str) != Some("current"))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        if let Some(obj) = out.as_object_mut() {
            if !not_in_force.is_empty() {
                obj.insert(
                    "notice".into(),
                    json!(format!(
                        "No document in force at {reference_time} matched. The matches in not_in_force are expired, not yet effective, superseded or older revisions (see each 'validity'); do not present them as current."
                    )),
                );
            }
            obj.insert("not_in_force".into(), json!(not_in_force));
        }
    }
    if let Some(obj) = out.as_object_mut() {
        obj.insert("reference_time".into(), json!(reference_time));
        obj.insert("in_force_only".into(), json!(in_force_only));
    }
    Ok(out)
}

/// One search pass: a single bucket, or a fan-out over every bucket the caller can see.
async fn search_once(ctx: &ToolCtx, bucket: Option<&str>, body: Value, k: u64) -> Result<Value, ToolFailure> {
    if let Some(bucket) = bucket {
        let uri = format!("/v2/buckets/{}/search", pct_encode(bucket));
        let mut out = dispatch(ctx, Method::POST, &uri, DispatchBody::Json(body)).await?;
        if let Some(obj) = out.as_object_mut() {
            obj.insert("bucket".into(), json!(bucket));
        }
        return Ok(out);
    }

    // No bucket: fan out over every bucket the caller can see (RLS-filtered list).
    let listed = dispatch(ctx, Method::GET, "/v2/buckets", DispatchBody::Empty).await?;
    let bucket_ids: Vec<String> = listed
        .get("buckets")
        .or_else(|| listed.get("items"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|b| b.get("id").and_then(Value::as_str).map(ToString::to_string))
                .take(MAX_FANOUT_BUCKETS)
                .collect()
        })
        .unwrap_or_default();

    let mut results: Vec<Value> = Vec::new();
    let mut facets: Map<String, Value> = Map::new();
    let mut skipped = 0usize;
    for bid in &bucket_ids {
        let uri = format!("/v2/buckets/{}/search", pct_encode(bid));
        match dispatch(ctx, Method::POST, &uri, DispatchBody::Json(body.clone())).await {
            Ok(resp) => {
                if let Some(items) = resp.get("results").and_then(Value::as_array) {
                    for item in items {
                        let mut item = item.clone();
                        if let Some(obj) = item.as_object_mut() {
                            obj.entry("bucket_id").or_insert_with(|| json!(bid));
                        }
                        results.push(item);
                    }
                }
                if let Some(f) = resp.get("tag_facets").and_then(Value::as_object) {
                    for (tag, count) in f {
                        let add = count.as_i64().unwrap_or(0);
                        let cur = facets.get(tag).and_then(Value::as_i64).unwrap_or(0);
                        facets.insert(tag.clone(), json!(cur + add));
                    }
                }
            }
            Err(fail) if fail.code == "validation_error" || fail.code == "invalid_query" => {
                return Err(fail);
            }
            Err(_) => skipped += 1,
        }
    }
    results.sort_by(|a, b| {
        let sa = a.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        let sb = b.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(usize::try_from(k).unwrap_or(usize::MAX));
    Ok(json!({
        "results": results,
        "items": results,
        "tag_facets": facets,
        "buckets_searched": bucket_ids.len() - skipped,
    }))
}

async fn rag_list_documents(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let mut qs: Vec<String> = Vec::new();
    if let Some(q) = arg_str_opt(args, "query")?.filter(|q| !q.trim().is_empty()) {
        qs.push(format!("q={}", pct_encode(q.trim())));
    }
    if arg_bool_opt(args, "include_closed")? == Some(true) {
        qs.push("include_closed=true".into());
    }
    if let Some(l) = arg_u64_opt(args, "limit")? {
        qs.push(format!("limit={l}"));
    }
    if let Some(o) = arg_u64_opt(args, "offset")? {
        qs.push(format!("offset={o}"));
    }
    let query = if qs.is_empty() { String::new() } else { format!("?{}", qs.join("&")) };
    let bucket = arg_str_opt(args, "bucket")?.map(|b| b.trim().to_string()).filter(|b| !b.is_empty());
    if let Some(b) = bucket {
        let uri = format!("/v2/buckets/{}/documents{query}", pct_encode(&b));
        return dispatch(ctx, Method::GET, &uri, DispatchBody::Empty).await;
    }
    let listed = dispatch(ctx, Method::GET, "/v2/buckets", DispatchBody::Empty).await?;
    let mut buckets = Vec::new();
    for b in listed.get("buckets").and_then(Value::as_array).into_iter().flatten().take(MAX_FANOUT_BUCKETS) {
        let Some(id) = b.get("id").and_then(Value::as_str) else { continue };
        let uri = format!("/v2/buckets/{}/documents{query}", pct_encode(id));
        if let Ok(v) = dispatch(ctx, Method::GET, &uri, DispatchBody::Empty).await {
            buckets.push(v);
        }
    }
    Ok(json!({ "buckets": buckets }))
}

async fn rag_read_sheet(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let id = arg_uuid_opt(args, "document_id")?.ok_or_else(|| ToolFailure::invalid("document_id is required"))?;
    let mut qs: Vec<String> = Vec::new();
    if let Some(sh) = arg_str_opt(args, "sheet")?.filter(|s| !s.trim().is_empty()) {
        qs.push(format!("sheet={}", pct_encode(sh.trim())));
    }
    if let Some(f) = arg_str_opt(args, "find")?.filter(|s| !s.trim().is_empty()) {
        qs.push(format!("find={}", pct_encode(f.trim())));
    }
    if let Some(c) = arg_str_opt(args, "column")?.filter(|s| !s.trim().is_empty()) {
        qs.push(format!("column={}", pct_encode(c.trim())));
    }
    if arg_bool_opt(args, "exact")? == Some(true) {
        qs.push("exact=true".into());
    }
    if let Some(o) = arg_u64_opt(args, "offset")? {
        qs.push(format!("offset={o}"));
    }
    if let Some(l) = arg_u64_opt(args, "limit")? {
        qs.push(format!("limit={l}"));
    }
    let query = if qs.is_empty() { String::new() } else { format!("?{}", qs.join("&")) };
    dispatch(ctx, Method::GET, &format!("/v2/documents/{id}/sheet{query}"), DispatchBody::Empty).await
}

async fn docs_status(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let mut scan = Value::Null;
    if arg_bool_opt(args, "scan")? == Some(true) {
        scan = dispatch(ctx, Method::POST, "/v2/watch/scan", DispatchBody::Empty).await?;
    }
    let mut status = dispatch(ctx, Method::GET, "/v2/watch", DispatchBody::Empty).await?;
    if let (Some(obj), false) = (status.as_object_mut(), scan.is_null()) {
        obj.insert("scan".into(), scan);
    }
    Ok(status)
}

/// Asynchronous task runner for `docs_status` with `scan: true`: triggers a scan and
/// polls `/v2/watch` until no bucket is scanning, reporting progress in `statusMessage`.
async fn docs_status_scan_task(
    ctx: &ToolCtx,
    tasks: &TaskStore,
    task_id: &str,
) -> Result<Value, ToolFailure> {
    let started_at = chrono::Utc::now();
    let scan = dispatch(ctx, Method::POST, "/v2/watch/scan", DispatchBody::Empty).await?;

    let grace_start = Instant::now();
    let mut saw_scanning = false;
    let mut status = dispatch(ctx, Method::GET, "/v2/watch", DispatchBody::Empty).await?;

    loop {
        let (scanning_summary, finished_after) = inspect_watch_scan(&status, started_at);
        if let Some(summary) = scanning_summary {
            saw_scanning = true;
            tasks.set_status_message(task_id, summary);
        } else if saw_scanning
            || finished_after
            || grace_start.elapsed() >= Duration::from_secs(2)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
        status = dispatch(ctx, Method::GET, "/v2/watch", DispatchBody::Empty).await?;
    }

    if let Some(obj) = status.as_object_mut() {
        obj.insert("scan".into(), scan);
    }
    Ok(status)
}

fn inspect_watch_scan(
    status: &Value,
    started_at: chrono::DateTime<chrono::Utc>,
) -> (Option<String>, bool) {
    let Some(buckets) = status.get("buckets").and_then(Value::as_object) else {
        return (None, false);
    };
    let mut scanning_parts = Vec::new();
    let mut finished_after = false;
    for (slug, info) in buckets {
        let state = info.get("state").and_then(Value::as_str).unwrap_or_default();
        if state == "scanning" {
            if let Some(prog) = info.get("progress").and_then(Value::as_array) {
                if prog.len() == 2 {
                    let done = prog[0].as_u64().unwrap_or(0);
                    let total = prog[1].as_u64().unwrap_or(0);
                    scanning_parts.push(format!("{slug}: {done}/{total}"));
                    continue;
                }
            }
            scanning_parts.push(format!("{slug}: scanning"));
        }
        if let Some(fin) = info
            .get("last_scan_finished_at")
            .and_then(Value::as_str)
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        {
            if fin.with_timezone(&chrono::Utc) >= started_at {
                finished_after = true;
            }
        }
    }
    let summary = if scanning_parts.is_empty() {
        None
    } else {
        Some(format!("Scanning {}", scanning_parts.join(", ")))
    };
    (summary, finished_after)
}

async fn rag_ask(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let query = arg_str_req(args, "query")?;
    let model = arg_str_opt(args, "mode")?
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| "openrag-auto".to_string());
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), json!([{ "role": "user", "content": query }]));
    body.insert("include_citations".into(), json!(true));
    body.insert("stream".into(), json!(false));
    // The caller is an AI: keep the answer in the tokens the LLM saw.
    body.insert("keep_pii_tokens".into(), json!(true));
    if let Some(bucket) = arg_str_opt(args, "bucket")?.filter(|b| !b.trim().is_empty()) {
        body.insert("bucket".into(), json!(bucket.trim()));
    }
    if let Some(k) = arg_u64_opt(args, "top_k")? {
        if k == 0 {
            return Err(ToolFailure::invalid("'top_k' must be >= 1"));
        }
        body.insert("top_k".into(), json!(k.min(50)));
    }
    let in_force_only = arg_bool_opt(args, "latest_revision_only")?.unwrap_or(true);
    let as_of = arg_datetime_opt(args, "as_of")?;
    body.insert("latest_revision_only".into(), json!(in_force_only));
    insert_opt(&mut body, "as_of", as_of.map(|d| json!(d.to_rfc3339())));
    let reference_time = as_of.unwrap_or_else(chrono::Utc::now).to_rfc3339();
    let resp = dispatch(ctx, Method::POST, "/v1/chat/completions", DispatchBody::Json(Value::Object(body))).await?;
    let answer = resp
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let citations = resp.get("citations").cloned().unwrap_or_else(|| json!([]));
    let mut out = json!({
        "answer": answer,
        "citations": citations,
        "sources": resp.get("sources").cloned().unwrap_or_else(|| json!([])),
        "admission": resp.get("admission").cloned().unwrap_or(Value::Null),
        "model": resp.get("model").cloned().unwrap_or(Value::Null),
        "usage": resp.get("usage").cloned().unwrap_or(Value::Null),
        "reference_time": reference_time,
        "in_force_only": in_force_only,
    });
    if in_force_only && citations.as_array().is_none_or(Vec::is_empty) {
        // Never let an ungrounded answer pass as the current state of the documents.
        if let Some(obj) = out.as_object_mut() {
            obj.insert(
                "notice".into(),
                json!(format!(
                    "No document in force at {reference_time} grounded this answer. Do not present it as the current state; \
                     retry with as_of set to the relevant date, or latest_revision_only=false to include expired and superseded documents."
                )),
            );
        }
    }
    Ok(out)
}

async fn rag_ingest(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let bucket = arg_str_req(args, "bucket")?;
    let path = arg_str_opt(args, "path")?.filter(|p| !p.trim().is_empty());
    let text = arg_str_opt(args, "text")?;
    let title = arg_str_opt(args, "title")?;
    let scope = arg_scope_opt(args, "access_scope")?;
    let allowed = arg_uuid_list_opt(args, "allowed_group_ids")?.unwrap_or_default();
    let denied = arg_uuid_list_opt(args, "denied_group_ids")?.unwrap_or_default();
    let level = arg_u64_opt(args, "min_auth_level")?;
    if let Some(l) = level {
        if !(1..=10).contains(&l) {
            return Err(ToolFailure::invalid("'min_auth_level' must be between 1 and 10"));
        }
    }
    let tags = arg_str_list_opt(args, "tags")?.unwrap_or_default();
    let valid_from = arg_datetime_opt(args, "valid_from")?;
    let valid_to = arg_datetime_opt(args, "valid_to")?;
    if let (Some(f), Some(t)) = (valid_from, valid_to) {
        if t <= f {
            return Err(ToolFailure::invalid("'valid_to' must be later than 'valid_from'"));
        }
    }
    let revision = match arg_u64_opt(args, "revision")? {
        None => None,
        Some(0) => return Err(ToolFailure::invalid("'revision' must be >= 1")),
        Some(r) => Some(i32::try_from(r).map_err(|_| ToolFailure::invalid("'revision' is too large"))?),
    };
    let uri = format!("/v2/buckets/{}/documents", pct_encode(bucket.trim()));

    match (path, text) {
        (Some(_), Some(_)) => Err(ToolFailure::invalid("pass either 'path' or 'text', not both")),
        (None, None) => Err(ToolFailure::invalid("missing required argument 'text' (or 'path' over stdio)")),
        (None, Some(text)) => {
            if text.trim().is_empty() {
                return Err(ToolFailure::invalid("'text' must not be empty"));
            }
            let mut body = Map::new();
            body.insert("title".into(), json!(title.unwrap_or_else(|| default_title(&text))));
            body.insert("text".into(), json!(text));
            insert_opt(&mut body, "access_scope", scope.map(|s| json!(s.as_str())));
            insert_opt(&mut body, "min_auth_level", level.map(|l| json!(l)));
            body.insert("allowed_group_ids".into(), json!(allowed));
            body.insert("denied_group_ids".into(), json!(denied));
            body.insert("tags".into(), json!(tags));
            insert_opt(&mut body, "valid_from", valid_from.map(|d| json!(d.to_rfc3339())));
            insert_opt(&mut body, "valid_to", valid_to.map(|d| json!(d.to_rfc3339())));
            insert_opt(&mut body, "revision", revision.map(|r| json!(r)));
            dispatch(ctx, Method::POST, &uri, DispatchBody::Json(Value::Object(body))).await
        }
        (Some(path), None) => {
            if ctx.transport != McpTransport::Stdio {
                return Err(ToolFailure::invalid(
                    "'path' is only available over the local stdio transport; send 'text' instead",
                ));
            }
            let p = FsPath::new(path.trim());
            let meta = tokio::fs::metadata(p)
                .await
                .map_err(|e| ToolFailure::invalid(format!("cannot read '{}': {e}", p.display())))?;
            if !meta.is_file() {
                return Err(ToolFailure::invalid(format!("'{}' is not a regular file", p.display())));
            }
            if meta.len() > MAX_DISPATCH_BODY_BYTES as u64 {
                return Err(ToolFailure::invalid("file exceeds the 64 MiB ingestion limit"));
            }
            let bytes = tokio::fs::read(p)
                .await
                .map_err(|e| ToolFailure::invalid(format!("cannot read '{}': {e}", p.display())))?;
            let file_name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("document")
                .to_string();
            let mut fields: Vec<(&str, String)> = Vec::new();
            if let Some(t) = title {
                fields.push(("title", t));
            }
            if let Some(s) = scope {
                fields.push(("access_scope", s.as_str().to_string()));
            }
            if let Some(l) = level {
                fields.push(("min_auth_level", l.to_string()));
            }
            fields.push(("allowed_group_ids", json!(allowed).to_string()));
            fields.push(("denied_group_ids", json!(denied).to_string()));
            if !tags.is_empty() {
                fields.push(("tags", json!(tags).to_string()));
            }
            if let Some(d) = valid_from {
                fields.push(("valid_from", d.to_rfc3339()));
            }
            if let Some(d) = valid_to {
                fields.push(("valid_to", d.to_rfc3339()));
            }
            if let Some(r) = revision {
                fields.push(("revision", r.to_string()));
            }
            let (boundary, body) = build_multipart(&file_name, &bytes, &fields);
            dispatch(ctx, Method::POST, &uri, DispatchBody::Multipart { boundary, bytes: body }).await
        }
    }
}

fn default_title(text: &str) -> String {
    let first = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("Untitled");
    first.chars().take(80).collect()
}

/// Replaces detector spans with `[REDACTED:<category>]` (used only when a page has
/// no stored redacted markdown).
fn mask_with_detector(pii: &openrag_pii::PiiEngine, text: &str) -> String {
    let mut spans = pii.detect(text);
    spans.sort_by_key(|s| s.start_byte);
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for s in spans {
        if s.start_byte < cursor || s.end_byte > text.len() || s.start_byte >= s.end_byte {
            continue;
        }
        if !text.is_char_boundary(s.start_byte) || !text.is_char_boundary(s.end_byte) {
            continue;
        }
        out.push_str(&text[cursor..s.start_byte]);
        out.push_str(&format!("[REDACTED:{}]", s.category.as_canonical_str()));
        cursor = s.end_byte;
    }
    out.push_str(&text[cursor..]);
    out
}

async fn rag_get_page(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let page_id = arg_uuid_opt(args, "page_id")?;
    let document_id = arg_uuid_opt(args, "document_id")?;
    let page_no = arg_u64_opt(args, "page_no")?.unwrap_or(1);
    if page_id.is_none() && document_id.is_none() {
        return Err(ToolFailure::invalid("pass 'page_id' or 'document_id'"));
    }
    if page_no == 0 {
        return Err(ToolFailure::invalid("'page_no' is 1-based"));
    }
    let page_no_i32 = i32::try_from(page_no).map_err(|_| ToolFailure::invalid("'page_no' is too large"))?;

    let app_pool = ctx.state.require_app_pool()?;
    let mut tx = app_pool.begin_rls_tx(&ctx.principal).await?;
    let row = sqlx::query(
        "SELECT p.id, p.document_id, p.page_no, p.arn, p.access_scope, p.min_auth_level,
                p.redacted_markdown, p.raw_markdown, d.title AS document_title, d.bucket_id
         FROM corpus.pages p
         JOIN corpus.documents d ON d.org_id = p.org_id AND d.id = p.document_id
         WHERE p.org_id = $1
           AND d.status <> 'archived'
           AND (($2::uuid IS NOT NULL AND p.id = $2)
                OR ($2::uuid IS NULL AND p.document_id = $3 AND p.page_no = $4))
         LIMIT 1",
    )
    .bind(ctx.principal.org_id)
    .bind(page_id)
    .bind(document_id)
    .bind(page_no_i32)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("rag_get_page: {e}")))?;

    let Some(row) = row else {
        tx.commit().await?;
        return Err(ToolFailure::not_found_or_denied("page"));
    };
    let db = |e: sqlx::Error| OpenRagError::Database(format!("rag_get_page column: {e}"));
    let id: Uuid = row.try_get("id").map_err(db)?;
    let arn: String = row.try_get("arn").map_err(db)?;
    if enforce_api_key_arn_scope(&ctx.principal, &arn).is_err() {
        tx.commit().await?;
        return Err(ToolFailure::not_found_or_denied("page"));
    }
    let redacted: Option<String> = row.try_get("redacted_markdown").map_err(db)?;
    let raw: Option<String> = row.try_get("raw_markdown").map_err(db)?;

    let fig_rows = sqlx::query(
        "SELECT id, kind, caption, bbox_norm_1000
         FROM corpus.page_figures
         WHERE org_id = $1 AND page_id = $2
         ORDER BY id",
    )
    .bind(ctx.principal.org_id)
    .bind(id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("rag_get_page figures: {e}")))?;
    tx.commit().await?;

    let mut boxes = Vec::with_capacity(fig_rows.len());
    for f in &fig_rows {
        let bbox: Option<Vec<i32>> = f.try_get("bbox_norm_1000").map_err(db)?;
        boxes.push(json!({
            "id": f.try_get::<Uuid, _>("id").map_err(db)?,
            "kind": f.try_get::<String, _>("kind").map_err(db)?,
            "caption": f.try_get::<Option<String>, _>("caption").map_err(db)?,
            "bbox_norm_1000": bbox,
        }));
    }

    let text = match redacted {
        Some(t) => t,
        None => {
            let pii = ctx.state.pii_engine.read().await.clone();
            mask_with_detector(&pii, raw.as_deref().unwrap_or_default())
        }
    };
    let thumb = format!("/api/app/pages/{id}/thumbnail");
    Ok(json!({
        "id": id,
        "page_id": id,
        "document_id": row.try_get::<Uuid, _>("document_id").map_err(db)?,
        "bucket_id": row.try_get::<Uuid, _>("bucket_id").map_err(db)?,
        "document_title": row.try_get::<String, _>("document_title").map_err(db)?,
        "page_no": row.try_get::<i32, _>("page_no").map_err(db)?,
        "access_scope": row.try_get::<String, _>("access_scope").map_err(db)?,
        "min_auth_level": row.try_get::<i16, _>("min_auth_level").map_err(db)?,
        "text": text,
        "redacted_markdown": text,
        "boxes": boxes,
        "figures": boxes,
        "thumbnail_url": thumb,
        "thumb_uri": thumb,
    }))
}

fn wiki_page_json(page: &WikiPage, body: bool) -> Value {
    let mut v = json!({
        "id": page.id,
        "page_id": page.id,
        "bucket_id": page.bucket_id,
        "slug": page.slug,
        "title": page.title,
        "tags": page.tags,
        "access_scope": page.access_scope.as_str(),
        "min_auth_level": page.min_auth_level.as_i16(),
        "version": page.version,
        "updated_at": page.updated_at,
    });
    if let Some(obj) = v.as_object_mut() {
        if body {
            obj.insert("body_md".into(), json!(page.body_md));
        } else {
            let snippet: String = page.body_md.chars().take(400).collect();
            obj.insert("snippet".into(), json!(snippet));
        }
    }
    v
}

async fn wiki_search(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let query = arg_str_req(args, "query")?;
    let k = top_k(args, DEFAULT_TOP_K)?;
    let bucket = arg_str_opt(args, "bucket")?.filter(|b| !b.trim().is_empty());

    let app_pool = ctx.state.require_app_pool()?;
    let mut tx = app_pool.begin_rls_tx(&ctx.principal).await?;
    let bucket_id = match bucket.as_deref() {
        Some(b) => match resolve_bucket_rls(&mut tx, ctx.principal.org_id, b).await {
            Ok(rec) => Some(rec.id),
            Err(OpenRagError::NotFound(_)) => {
                tx.commit().await?;
                return Err(ToolFailure::not_found_or_denied("bucket"));
            }
            Err(e) => return Err(e.into()),
        },
        None => None,
    };
    let fetch = usize::try_from(k.saturating_mul(4)).unwrap_or(usize::MAX).min(800);
    let hits = WikiCompiler::search_pages_rls(
        &mut tx,
        ctx.principal.org_id,
        &query,
        None,
        fetch,
        ctx.state.text_embedder.as_ref(),
    )
    .await?;
    tx.commit().await?;

    let results: Vec<Value> = hits
        .into_iter()
        .filter(|h| h.page.status == WikiStatus::Active)
        .filter(|h| bucket_id.is_none() || h.page.bucket_id == bucket_id)
        .filter(|h| enforce_api_key_arn_scope(&ctx.principal, &h.page.arn).is_ok())
        .take(usize::try_from(k).unwrap_or(usize::MAX))
        .map(|h| {
            let mut v = wiki_page_json(&h.page, false);
            if let Some(obj) = v.as_object_mut() {
                obj.insert("score".into(), json!(h.score));
            }
            v
        })
        .collect();
    Ok(json!({ "results": results, "pages": results }))
}

async fn wiki_get(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let page_id = arg_uuid_opt(args, "page_id")?;
    let slug = arg_str_opt(args, "slug")?.filter(|s| !s.trim().is_empty());
    let bucket = arg_str_opt(args, "bucket")?.filter(|b| !b.trim().is_empty());
    if page_id.is_none() && slug.is_none() {
        return Err(ToolFailure::invalid("pass 'page_id' or 'slug'"));
    }

    let app_pool = ctx.state.require_app_pool()?;
    let mut tx = app_pool.begin_rls_tx(&ctx.principal).await?;
    let page = if let Some(id) = page_id {
        WikiCompiler::get_page_by_id_rls(&mut tx, ctx.principal.org_id, id).await?
    } else {
        let bucket_id = match bucket.as_deref() {
            Some(b) => match resolve_bucket_rls(&mut tx, ctx.principal.org_id, b).await {
                Ok(rec) => Some(rec.id),
                Err(OpenRagError::NotFound(_)) => {
                    tx.commit().await?;
                    return Err(ToolFailure::not_found_or_denied("bucket"));
                }
                Err(e) => return Err(e.into()),
            },
            None => None,
        };
        let slug = slug.unwrap_or_default();
        WikiCompiler::get_page_by_slug_rls(&mut tx, ctx.principal.org_id, bucket_id, slug.trim()).await?
    };
    tx.commit().await?;

    match page {
        Some(p)
            if p.status == WikiStatus::Active
                && enforce_api_key_arn_scope(&ctx.principal, &p.arn).is_ok() =>
        {
            Ok(wiki_page_json(&p, true))
        }
        _ => Err(ToolFailure::not_found_or_denied("wiki page")),
    }
}

fn memory_json(entry: &MemoryEntry) -> Value {
    json!({
        "id": entry.id,
        "memory_id": entry.id,
        "title": entry.title,
        "effective_text": entry.effective_text,
        "text": entry.effective_text,
        "access_scope": entry.access_scope,
        "min_auth_level": entry.min_auth_level,
        "memory_kind": entry.memory_kind,
        "tags": entry.tags,
        "status": entry.status,
        "version": entry.version,
        "source_episode_id": entry.source_episode_id,
        "promoted_wiki_id": entry.promoted_wiki_id,
        "updated_at": entry.updated_at,
    })
}

async fn memory_search(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let query = arg_str_req(args, "query")?;
    let scope = arg_scope_opt(args, "scope")?;
    let k = usize::try_from(top_k(args, DEFAULT_TOP_K)?).unwrap_or(usize::MAX);
    let router = ctx.state.active_memory_router().await?;
    let fetch = if scope.is_some() { k.saturating_mul(4).min(800) } else { k };
    let entries = router.search(&ctx.principal, &query, None, fetch, None).await?;
    let items: Vec<Value> = entries
        .iter()
        .filter(|e| e.status == MemoryStatus::Active)
        .filter(|e| scope.is_none_or(|s| e.access_scope == s))
        .filter(|e| enforce_api_key_arn_scope(&ctx.principal, &e.arn).is_ok())
        .take(k)
        .map(memory_json)
        .collect();
    Ok(json!({ "memories": items, "results": items, "backend": router.primary_kind().as_str() }))
}

async fn memory_patch(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let memory_id = arg_uuid_opt(args, "memory_id")?
        .ok_or_else(|| ToolFailure::invalid("missing required argument 'memory_id'"))?;
    let op = arg_str_req(args, "op")?;
    if !matches!(op.as_str(), "amend" | "supersede" | "retract" | "redact") {
        return Err(ToolFailure::invalid("'op' must be one of amend, supersede, retract, redact"));
    }
    let mut body = Map::new();
    body.insert("memory_id".into(), json!(memory_id));
    body.insert("op".into(), json!(op));
    insert_opt(&mut body, "text", arg_str_opt(args, "text")?.map(|v| json!(v)));
    insert_opt(&mut body, "reason", arg_str_opt(args, "reason")?.map(|v| json!(v)));
    dispatch(ctx, Method::POST, "/api/app/memory/patch", DispatchBody::Json(Value::Object(body))).await
}

async fn memory_engage(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let query = arg_str_req(args, "query")?;
    let scope = arg_scope_opt(args, "scope")?;
    let k = usize::try_from(top_k(args, 8)?).unwrap_or(usize::MAX);
    let backend = arg_backend_opt(args)?;
    let router = ctx.state.active_memory_router().await?;
    let resp = router
        .engage(
            &ctx.principal,
            EngageRequest {
                query,
                scope,
                top_k: k,
                tag_filter: None,
            },
            backend,
        )
        .await?;
    let memories: Vec<Value> = resp
        .memories
        .iter()
        .filter(|h| h.entry.status == MemoryStatus::Active)
        .filter(|h| enforce_api_key_arn_scope(&ctx.principal, &h.entry.arn).is_ok())
        .map(|h| {
            let mut v = memory_json(&h.entry);
            if let Some(obj) = v.as_object_mut() {
                obj.insert("score".into(), json!(h.score));
                obj.insert("edges".into(), serde_json::to_value(&h.edges).unwrap_or_else(|_| json!([])));
            }
            v
        })
        .collect();
    Ok(json!({
        "memories": memories,
        "results": memories,
        "expanded_terms": resp.expanded_terms,
        "tag_facets": serde_json::to_value(&resp.tag_facets).unwrap_or_else(|_| json!([])),
        "backend": backend.unwrap_or_else(|| router.primary_kind()).as_str(),
    }))
}

async fn memory_reflect(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let content = arg_str_req(args, "content")?;
    let title = arg_str_opt(args, "title")?.filter(|t| !t.trim().is_empty());
    let episode_id = arg_uuid_opt(args, "episode_id")?;
    let scope = arg_scope_opt(args, "scope")?.unwrap_or(AccessScope::Private);
    let edge_type = match arg_str_opt(args, "edge_type")? {
        None => None,
        Some(s) => Some(MemoryEdgeType::from_str(s.trim()).map_err(|_| {
            ToolFailure::invalid("'edge_type' must be one of derived_from, supports, contradicts, supersedes")
        })?),
    };
    let target_memory_id = arg_uuid_opt(args, "target_memory_id")?;
    if edge_type.is_some() != target_memory_id.is_some() {
        return Err(ToolFailure::invalid("'edge_type' and 'target_memory_id' must be passed together"));
    }
    let backend = arg_backend_opt(args)?;
    let tags = arg_str_list_opt(args, "tags")?.unwrap_or_default();

    let dept_id = match arg_uuid_opt(args, "dept_id")? {
        Some(d) if ctx.principal.dept_ids.contains(&d) => d,
        Some(_) => return Err(ToolFailure::new("forbidden", "you are not a member of that department")),
        None => *ctx.principal.dept_ids.first().ok_or_else(|| {
            ToolFailure::new(
                "no_department",
                "memory_reflect needs a home department, and you have no department membership",
            )
        })?,
    };

    let caller_level = ctx.principal.auth_level.as_i16();
    let level_i16 = match arg_u64_opt(args, "min_auth_level")? {
        Some(l) => i16::try_from(l).map_err(|_| ToolFailure::invalid("'min_auth_level' out of range"))?,
        None => caller_level.min(3),
    };
    if !(1..=10).contains(&level_i16) {
        return Err(ToolFailure::invalid("'min_auth_level' must be between 1 and 10"));
    }
    if level_i16 > caller_level {
        return Err(ToolFailure::new(
            "forbidden",
            "min_auth_level cannot exceed your own permission level",
        ));
    }
    let min_auth_level = AuthLevel::try_from(level_i16)?;

    let (org_slug, dept_path) = resolve_org_and_dept_paths(&ctx.state, ctx.principal.org_id, dept_id).await?;

    // API-key ARN ceiling: the would-be entry ARN must fall inside the key's scope.
    if !ctx.principal.scoped_arns.is_empty() {
        let probe = Arn::new(
            org_slug.clone(),
            dept_path.clone(),
            scope,
            ArnResourceType::Memory,
            min_auth_level,
            Uuid::new_v4().to_string(),
        )?
        .to_canonical();
        if enforce_api_key_arn_scope(&ctx.principal, &probe).is_err() {
            return Err(ToolFailure::new("forbidden", "API key scope does not permit writing this memory"));
        }
    }

    let router = ctx.state.active_memory_router().await?;
    let resp = router
        .reflect(
            &ctx.principal,
            ReflectRequest {
                dept_id,
                org_slug,
                dept_path,
                content: neutralize_literal_pii_tokens(&content),
                title: title.map(|t| neutralize_literal_pii_tokens(&t)),
                memory_kind: None,
                episode_id,
                access_scope: scope,
                allowed_group_ids: Vec::new(),
                denied_group_ids: Vec::new(),
                min_auth_level,
                tags,
                edge_type,
                target_memory_id,
                extract_concepts: true,
            },
            backend,
        )
        .await?;

    let mut out = memory_json(&resp.entry);
    if let Some(obj) = out.as_object_mut() {
        obj.insert("episode_id".into(), json!(resp.episode_id));
        obj.insert("edge".into(), serde_json::to_value(&resp.edge).unwrap_or(Value::Null));
        obj.insert(
            "extracted_concepts".into(),
            json!(resp
                .extracted_concepts
                .iter()
                .map(|(a, b)| json!([a, b]))
                .collect::<Vec<_>>()),
        );
        obj.insert(
            "backend".into(),
            json!(backend.unwrap_or_else(|| router.primary_kind()).as_str()),
        );
    }
    Ok(out)
}

async fn guard_check(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let text = match arg_str_opt(args, "text")? {
        Some(t) => t,
        None => return Err(ToolFailure::invalid("missing required argument 'text'")),
    };
    let surface = arg_str_opt(args, "surface")?
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "mcp_egress".to_string());
    // Only text + surface are forwarded: callers cannot override rules or the allowlist.
    let resp = dispatch(
        ctx,
        Method::POST,
        "/api/studio/pii/guard-test",
        DispatchBody::Json(json!({ "text": text, "surface": surface.trim() })),
    )
    .await?;
    Ok(json!({
        "decision": resp.get("decision").cloned().unwrap_or(Value::Null),
        "action": resp.get("action").cloned().unwrap_or(Value::Null),
        "sanitized_text": resp.get("sanitized_text").cloned().unwrap_or(Value::Null),
        "matched_categories": resp.get("matched_categories").cloned().unwrap_or_else(|| json!([])),
        "warnings": resp.get("warnings").cloned().unwrap_or_else(|| json!([])),
        "surface": surface.trim(),
    }))
}

async fn concept_link(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let source = arg_str_req(args, "source")?;
    let target = arg_str_req(args, "target")?;
    if source.trim().to_lowercase() == target.trim().to_lowercase() {
        return Err(ToolFailure::invalid("cannot link a term to itself"));
    }
    let mut body = Map::new();
    body.insert("source".into(), json!(source.trim()));
    body.insert("target".into(), json!(target.trim()));
    for key in ["canonical_name", "lang", "relation"] {
        insert_opt(&mut body, key, arg_str_opt(args, key)?.map(|v| json!(v)));
    }
    insert_opt(&mut body, "concept_id", arg_uuid_opt(args, "concept_id")?.map(|v| json!(v)));
    dispatch(ctx, Method::POST, "/v1/concepts/link", DispatchBody::Json(Value::Object(body))).await
}

async fn concept_resolve(ctx: &ToolCtx, args: &Map<String, Value>) -> Result<Value, ToolFailure> {
    let term = arg_str_req(args, "term")?;
    let mut body = Map::new();
    body.insert("term".into(), json!(term.trim()));
    for key in ["include_aliases", "use_semantic"] {
        insert_opt(&mut body, key, arg_bool_opt(args, key)?.map(|v| json!(v)));
    }
    insert_opt(&mut body, "top_k", arg_u64_opt(args, "top_k")?.map(|v| json!(v)));
    if let Some(v) = args.get("embedding").filter(|v| !v.is_null()) {
        if !v.is_array() {
            return Err(ToolFailure::invalid("'embedding' must be an array of numbers"));
        }
        body.insert("embedding".into(), v.clone());
    }
    if let Some(v) = args.get("threshold").filter(|v| !v.is_null()) {
        if !v.is_number() {
            return Err(ToolFailure::invalid("'threshold' must be a number"));
        }
        body.insert("threshold".into(), v.clone());
    }
    dispatch(ctx, Method::POST, "/v1/concepts/resolve", DispatchBody::Json(Value::Object(body))).await
}

// ============================================================================
// Streamable HTTP transport (POST/GET/DELETE /mcp)
// ============================================================================

struct McpSession {
    principal_id: Uuid,
    org_id: Uuid,
    last_seen: Instant,
}

fn sessions() -> &'static Mutex<HashMap<String, McpSession>> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, McpSession>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn create_session(principal: &PrincipalContext) -> String {
    let id = Uuid::new_v4().simple().to_string();
    let mut map = sessions().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let now = Instant::now();
    map.retain(|_, s| now.duration_since(s.last_seen) < SESSION_TTL);
    if map.len() >= MAX_SESSIONS {
        if let Some(oldest) = map
            .iter()
            .min_by_key(|(_, s)| s.last_seen)
            .map(|(k, _)| k.clone())
        {
            map.remove(&oldest);
        }
    }
    map.insert(
        id.clone(),
        McpSession {
            principal_id: principal.principal_id,
            org_id: principal.org_id,
            last_seen: now,
        },
    );
    id
}

/// `true` when the session exists, is live and belongs to `principal`.
fn touch_session(id: &str, principal: &PrincipalContext) -> bool {
    let mut map = sessions().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match map.get_mut(id) {
        Some(s)
            if s.principal_id == principal.principal_id
                && s.org_id == principal.org_id
                && s.last_seen.elapsed() < SESSION_TTL =>
        {
            s.last_seen = Instant::now();
            true
        }
        _ => false,
    }
}

fn end_session(id: &str, principal: &PrincipalContext) -> bool {
    let mut map = sessions().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let owned = map
        .get(id)
        .is_some_and(|s| s.principal_id == principal.principal_id && s.org_id == principal.org_id);
    if owned {
        map.remove(id);
    }
    owned
}

pub fn mcp_routes() -> Router<Arc<AppState>> {
    Router::new().route(
        "/mcp",
        post(mcp_post_handler).get(mcp_get_handler).delete(mcp_delete_handler),
    )
}

fn json_response(status: StatusCode, body: &Value) -> Response {
    let bytes = serde_json::to_vec(body).unwrap_or_default();
    let mut resp = (status, bytes).into_response();
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    resp
}

/// DNS-rebinding guard: a browser `Origin` must be loopback or the server's own host.
fn origin_allowed(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(ORIGIN).and_then(|v| v.to_str().ok()) else {
        return true;
    };
    let host_of = |s: &str| -> String {
        let without_scheme = s.split_once("://").map_or(s, |(_, rest)| rest);
        let authority = without_scheme.split('/').next().unwrap_or_default();
        if let Some(rest) = authority.strip_prefix('[') {
            return rest.split(']').next().unwrap_or_default().to_ascii_lowercase();
        }
        authority.split(':').next().unwrap_or_default().to_ascii_lowercase()
    };
    let origin_host = host_of(origin);
    if matches!(origin_host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return true;
    }
    headers
        .get(HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|h| host_of(h) == origin_host)
}

fn contains_method(msg: &Value, method: &str) -> bool {
    let is_m = |v: &Value| v.get("method").and_then(Value::as_str) == Some(method);
    match msg {
        Value::Array(items) => items.iter().any(is_m),
        other => is_m(other),
    }
}

fn is_valid_http_header_value(raw: &str) -> bool {
    raw.bytes().all(|b| b == b'\t' || (0x20..=0x7E).contains(&b))
}

fn decode_mcp_header_value(raw: &str) -> Result<String, &'static str> {
    if !is_valid_http_header_value(raw) {
        return Err("header contains invalid characters");
    }
    if let Some(rest) = raw.strip_prefix(BASE64_SENTINEL_PREFIX) {
        if let Some(b64) = rest.strip_suffix(BASE64_SENTINEL_SUFFIX) {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|_| "invalid Base64 sentinel encoding")?;
            return String::from_utf8(bytes)
                .map_err(|_| "Base64 sentinel decoded to invalid UTF-8");
        }
    }
    Ok(raw.to_string())
}

fn single_header<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, &'static str> {
    let mut iter = headers.get_all(name).iter();
    let Some(first) = iter.next() else {
        return Ok(None);
    };
    if iter.next().is_some() {
        return Err("duplicate header");
    }
    let text = first.to_str().map_err(|_| "header is not valid ASCII")?;
    Ok(Some(text))
}

/// Validates required Streamable HTTP headers for a modern (`2026-07-28`) request.
fn validate_modern_http_headers(
    headers: &HeaderMap,
    msg: &Value,
) -> Result<(), (StatusCode, Value)> {
    // Notifications have no request header requirements in 2026-07-28.
    if msg.get("id").is_none() {
        return Ok(());
    }
    let req_id = msg.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = msg.get("method").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(meta_ver) = modern_protocol_version_in_msg(msg).and_then(Value::as_str) else {
        return Err((
            StatusCode::BAD_REQUEST,
            rpc_error(
                req_id,
                JSONRPC_INVALID_PARAMS,
                &format!("_meta['{META_PROTOCOL_VERSION}'] must be a string"),
            ),
        ));
    };

    match single_header(headers, PROTOCOL_VERSION_HEADER) {
        Ok(Some(hdr_ver)) if is_valid_http_header_value(hdr_ver) && hdr_ver == meta_ver => {}
        Ok(Some(hdr_ver)) => {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    &format!(
                        "Header mismatch: MCP-Protocol-Version '{hdr_ver}' does not match body '{meta_ver}'"
                    ),
                ),
            ));
        }
        Ok(None) => {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    "Header mismatch: missing required MCP-Protocol-Version header",
                ),
            ));
        }
        Err(reason) => {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    &format!("Header mismatch: invalid MCP-Protocol-Version header ({reason})"),
                ),
            ));
        }
    }

    match single_header(headers, METHOD_HEADER) {
        Ok(Some(hdr_method)) if is_valid_http_header_value(hdr_method) && hdr_method == method => {}
        Ok(Some(hdr_method)) => {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    &format!(
                        "Header mismatch: Mcp-Method '{hdr_method}' does not match body '{method}'"
                    ),
                ),
            ));
        }
        Ok(None) => {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    "Header mismatch: missing required Mcp-Method header",
                ),
            ));
        }
        Err(reason) => {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    &format!("Header mismatch: invalid Mcp-Method header ({reason})"),
                ),
            ));
        }
    }

    if !MODERN_PROTOCOL_VERSIONS.contains(&meta_ver) {
        return Err((
            StatusCode::BAD_REQUEST,
            rpc_error_response(req_id, RpcError::unsupported_version(meta_ver)),
        ));
    }

    // Per Streamable HTTP (tools/call, resources/read) and Tasks extension (tasks/*).
    let expected_name_field = match method {
        "tools/call" => Some("name"),
        "resources/read" => Some("uri"),
        "tasks/get" | "tasks/update" | "tasks/cancel" => Some("taskId"),
        _ => None,
    };

    if let Some(field) = expected_name_field {
        let raw_hdr = match single_header(headers, NAME_HEADER) {
            Ok(Some(v)) => v,
            Ok(None) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    rpc_error(
                        req_id,
                        JSONRPC_HEADER_MISMATCH,
                        "Header mismatch: missing required Mcp-Name header",
                    ),
                ));
            }
            Err(reason) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    rpc_error(
                        req_id,
                        JSONRPC_HEADER_MISMATCH,
                        &format!("Header mismatch: invalid Mcp-Name header ({reason})"),
                    ),
                ));
            }
        };
        let decoded = decode_mcp_header_value(raw_hdr).map_err(|reason| {
            (
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id.clone(),
                    JSONRPC_HEADER_MISMATCH,
                    &format!("Header mismatch: invalid Mcp-Name header ({reason})"),
                ),
            )
        })?;
        let body_val = msg
            .get("params")
            .and_then(|p| p.get(field))
            .and_then(Value::as_str);
        if body_val != Some(decoded.as_str()) {
            return Err((
                StatusCode::BAD_REQUEST,
                rpc_error(
                    req_id,
                    JSONRPC_HEADER_MISMATCH,
                    "Header mismatch: Mcp-Name header does not match request body",
                ),
            ));
        }
    }

    Ok(())
}

fn modern_http_error_status(payload: &Value) -> Option<StatusCode> {
    let code = payload.pointer("/error/code").and_then(Value::as_i64)?;
    match code {
        JSONRPC_HEADER_MISMATCH
        | JSONRPC_MISSING_REQUIRED_CLIENT_CAPABILITY
        | JSONRPC_UNSUPPORTED_PROTOCOL_VERSION
        | JSONRPC_INVALID_REQUEST => Some(StatusCode::BAD_REQUEST),
        JSONRPC_METHOD_NOT_FOUND => Some(StatusCode::NOT_FOUND),
        _ => None,
    }
}

async fn mcp_post_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !origin_allowed(&headers) {
        return json_response(
            StatusCode::FORBIDDEN,
            &json!({ "error": "forbidden", "message": "Origin not allowed for /mcp" }),
        );
    }

    let msg: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return json_response(
                StatusCode::BAD_REQUEST,
                &rpc_error(Value::Null, JSONRPC_PARSE_ERROR, "Parse error"),
            )
        }
    };

    if modern_protocol_version_in_msg(&msg).is_some() {
        if let Err((status, err_body)) = validate_modern_http_headers(&headers, &msg) {
            return json_response(status, &err_body);
        }
        let engine = McpEngine::for_http(state, principal);
        let reply = engine.handle_message(msg).await;
        return match reply {
            None => StatusCode::ACCEPTED.into_response(),
            Some(payload) => {
                if let Some(status) = modern_http_error_status(&payload) {
                    return json_response(status, &payload);
                }
                let accept = headers
                    .get(ACCEPT)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let wants_sse_only =
                    accept.contains("text/event-stream") && !accept.contains("application/json");
                if wants_sse_only {
                    let data = serde_json::to_string(&payload).unwrap_or_default();
                    let mut r =
                        (StatusCode::OK, format!("event: message\ndata: {data}\n\n")).into_response();
                    r.headers_mut()
                        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                    r.headers_mut().insert(
                        HeaderName::from_static("x-accel-buffering"),
                        HeaderValue::from_static("no"),
                    );
                    r
                } else {
                    json_response(StatusCode::OK, &payload)
                }
            }
        };
    }

    // Legacy Streamable HTTP path (unchanged).
    if let Some(v) = headers.get(PROTOCOL_VERSION_HEADER).and_then(|v| v.to_str().ok()) {
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&v.trim()) {
            return json_response(
                StatusCode::BAD_REQUEST,
                &rpc_error(Value::Null, JSONRPC_INVALID_REQUEST, "unsupported MCP-Protocol-Version"),
            );
        }
    }

    let is_init = contains_method(&msg, "initialize");
    let incoming_session = headers
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string);
    if !is_init {
        if let Some(sid) = incoming_session.as_deref() {
            if !touch_session(sid, &principal) {
                return json_response(
                    StatusCode::NOT_FOUND,
                    &rpc_error(Value::Null, JSONRPC_INVALID_REQUEST, "unknown or expired Mcp-Session-Id"),
                );
            }
        }
    }

    let engine = McpEngine::for_http(state, principal.clone());
    let reply = engine.handle_message(msg).await;
    let session_out = if is_init {
        Some(create_session(&principal))
    } else {
        incoming_session
    };

    let mut resp = match reply {
        None => StatusCode::ACCEPTED.into_response(),
        Some(payload) => {
            let accept = headers
                .get(ACCEPT)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_ascii_lowercase();
            let wants_sse_only =
                accept.contains("text/event-stream") && !accept.contains("application/json");
            if wants_sse_only {
                let data = serde_json::to_string(&payload).unwrap_or_default();
                let mut r = (StatusCode::OK, format!("event: message\ndata: {data}\n\n")).into_response();
                r.headers_mut()
                    .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                r
            } else {
                json_response(StatusCode::OK, &payload)
            }
        }
    };
    if let Some(sid) = session_out {
        if let Ok(v) = HeaderValue::from_str(&sid) {
            resp.headers_mut().insert(SESSION_HEADER, v);
        }
    }
    resp
}

/// No server-initiated stream is offered; the spec allows `405` for GET.
async fn mcp_get_handler(AuthPrincipal(_principal): AuthPrincipal) -> Response {
    let mut resp = json_response(
        StatusCode::METHOD_NOT_ALLOWED,
        &json!({ "error": "method_not_allowed", "message": "this server does not offer a server-initiated SSE stream; use POST /mcp" }),
    );
    resp.headers_mut()
        .insert(ALLOW, HeaderValue::from_static("POST, DELETE"));
    resp
}

async fn mcp_delete_handler(AuthPrincipal(principal): AuthPrincipal, headers: HeaderMap) -> Response {
    let Some(sid) = headers.get(SESSION_HEADER).and_then(|v| v.to_str().ok()) else {
        return json_response(
            StatusCode::BAD_REQUEST,
            &json!({ "error": "missing_session", "message": "Mcp-Session-Id header required" }),
        );
    };
    if end_session(sid.trim(), &principal) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        json_response(
            StatusCode::NOT_FOUND,
            &json!({ "error": "not_found", "message": "unknown Mcp-Session-Id" }),
        )
    }
}

// ============================================================================
// stdio transport
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    Ndjson,
    ContentLength,
}

#[derive(Debug)]
pub enum StdioFrame {
    Message(Vec<u8>, Framing),
    /// A frame whose envelope itself is broken (e.g. bad `Content-Length`).
    Malformed(Framing),
    Eof,
}

/// Reads one frame, auto-detecting NDJSON vs `Content-Length` framing.
pub async fn read_frame<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> std::io::Result<StdioFrame> {
    loop {
        let mut line = Vec::new();
        let n = reader.read_until(b'\n', &mut line).await?;
        if n == 0 {
            return Ok(StdioFrame::Eof);
        }
        let trimmed = String::from_utf8_lossy(&line).trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            let mut length = rest.trim().parse::<usize>().ok();
            // Remaining headers until the blank separator line.
            loop {
                let mut hdr = Vec::new();
                let hn = reader.read_until(b'\n', &mut hdr).await?;
                if hn == 0 {
                    return Ok(StdioFrame::Eof);
                }
                let h = String::from_utf8_lossy(&hdr).trim().to_ascii_lowercase();
                if h.is_empty() {
                    break;
                }
                if let Some(v) = h.strip_prefix("content-length:") {
                    length = v.trim().parse::<usize>().ok();
                }
            }
            let Some(len) = length.filter(|l| *l <= MAX_FRAME_BYTES) else {
                return Ok(StdioFrame::Malformed(Framing::ContentLength));
            };
            let mut body = vec![0u8; len];
            reader.read_exact(&mut body).await?;
            return Ok(StdioFrame::Message(body, Framing::ContentLength));
        }
        if line.len() > MAX_FRAME_BYTES {
            return Ok(StdioFrame::Malformed(Framing::Ndjson));
        }
        return Ok(StdioFrame::Message(trimmed.into_bytes(), Framing::Ndjson));
    }
}

/// Writes one JSON-RPC payload in the given framing.
pub async fn write_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    payload: &Value,
    framing: Framing,
) -> std::io::Result<()> {
    let body = serde_json::to_vec(payload).unwrap_or_default();
    match framing {
        Framing::Ndjson => {
            writer.write_all(&body).await?;
            writer.write_all(b"\n").await?;
        }
        Framing::ContentLength => {
            writer
                .write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
                .await?;
            writer.write_all(&body).await?;
        }
    }
    writer.flush().await
}

/// Serves MCP over arbitrary byte streams until EOF.
pub async fn serve_stream<R, W>(engine: &McpEngine, reader: R, writer: &mut W) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut reader = BufReader::new(reader);
    loop {
        match read_frame(&mut reader).await? {
            StdioFrame::Eof => return Ok(()),
            StdioFrame::Malformed(framing) => {
                write_frame(writer, &rpc_error(Value::Null, JSONRPC_PARSE_ERROR, "Parse error"), framing).await?;
            }
            StdioFrame::Message(bytes, framing) => {
                let reply = match serde_json::from_slice::<Value>(&bytes) {
                    Ok(msg) => {
                        if contains_method(&msg, "tools/call")
                            || contains_method(&msg, "tasks/get")
                            || contains_method(&msg, "tasks/update")
                            || contains_method(&msg, "tasks/cancel")
                        {
                            engine.ensure_state().await;
                        }
                        engine.handle_message(msg).await
                    }
                    Err(_) => Some(rpc_error(Value::Null, JSONRPC_PARSE_ERROR, "Parse error")),
                };
                if let Some(payload) = reply {
                    write_frame(writer, &payload, framing).await?;
                }
            }
        }
    }
}

/// `openrag mcp --stdio` / `openrag-mcp` entrypoint. Logs go to stderr only.
pub async fn run_stdio(opts: StdioOptions) -> std::io::Result<()> {
    let engine = McpEngine::for_stdio(opts);
    let mut stdout = tokio::io::stdout();
    tracing::info!(transport = McpTransport::Stdio.as_str(), "openrag MCP server ready");
    serve_stream(&engine, tokio::io::stdin(), &mut stdout).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_exactly_13_unique_tools_in_order() {
        let defs = tool_definitions();
        let names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, MCP_TOOL_NAMES.to_vec());
        for d in &defs {
            assert_eq!(d.pointer("/inputSchema/type").and_then(Value::as_str), Some("object"));
        }
    }

    #[test]
    fn pct_encode_escapes_reserved() {
        assert_eq!(pct_encode("a b/한"), "a%20b%2F%ED%95%9C");
        assert_eq!(pct_encode("slug-1_x.y~"), "slug-1_x.y~");
    }

    #[test]
    fn origin_guard_allows_loopback_and_same_host_only() {
        let mut h = HeaderMap::new();
        assert!(origin_allowed(&h));
        h.insert(ORIGIN, HeaderValue::from_static("http://localhost:3000"));
        assert!(origin_allowed(&h));
        h.insert(ORIGIN, HeaderValue::from_static("https://evil.example"));
        h.insert(HOST, HeaderValue::from_static("rag.internal:8080"));
        assert!(!origin_allowed(&h));
        h.insert(ORIGIN, HeaderValue::from_static("https://rag.internal"));
        assert!(origin_allowed(&h));
    }

    #[test]
    fn base64_sentinel_header_decoding() {
        assert_eq!(decode_mcp_header_value("rag_search"), Ok("rag_search".to_string()));
        assert_eq!(
            decode_mcp_header_value("=?base64?SGVsbG8sIOS4lueVjA==?="),
            Ok("Hello, 世界".to_string())
        );
        assert_eq!(
            decode_mcp_header_value("=?base64?IHBhZGRlZCA=?="),
            Ok(" padded ".to_string())
        );
        assert_eq!(
            decode_mcp_header_value("=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),
            Ok("=?base64?literal?=".to_string())
        );
        assert!(decode_mcp_header_value("=?base64?***not-base64***?=").is_err());
        assert!(decode_mcp_header_value("=?base64?/w==?=").is_err());
    }

    fn test_principal() -> Result<PrincipalContext, OpenRagError> {
        Ok(PrincipalContext::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            AuthLevel::new(5)?,
        ))
    }

    async fn send_mcp_http(
        app: &Router,
        principal: Option<&PrincipalContext>,
        method: Method,
        headers: &[(&str, &str)],
        body: Option<Value>,
    ) -> Result<(StatusCode, HeaderMap, Value), String> {
        let mut builder = Request::builder().method(method).uri("/mcp");
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        let raw_body = match body {
            Some(v) => {
                builder = builder.header(CONTENT_TYPE, "application/json");
                serde_json::to_vec(&v).map_err(|e| e.to_string())?
            }
            None => Vec::new(),
        };
        let mut req = builder
            .body(Body::from(raw_body))
            .map_err(|e| e.to_string())?;
        if let Some(p) = principal {
            req.extensions_mut().insert(TrustedPrincipal(p.clone()));
        }
        let resp = match app.clone().oneshot(req).await {
            Ok(r) => r,
            Err(never) => match never {},
        };
        let status = resp.status();
        let resp_headers = resp.headers().clone();
        let bytes = axum::body::to_bytes(resp.into_body(), MAX_DISPATCH_BODY_BYTES)
            .await
            .map_err(|e| e.to_string())?;
        let json = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or_else(|_| {
                Value::String(String::from_utf8_lossy(&bytes).to_string())
            })
        };
        Ok((status, resp_headers, json))
    }

    #[tokio::test]
    async fn http_legacy_session_works_unchanged() -> Result<(), String> {
        let state = AppState::new_offline_for_test().map_err(|e| e.to_string())?;
        let app = crate::build_router(state);
        let p1 = test_principal().map_err(|e| e.to_string())?;
        let p2 = test_principal().map_err(|e| e.to_string())?;

        // Unauthenticated fails closed with 401.
        let (st, _, _) = send_mcp_http(
            &app,
            None,
            Method::POST,
            &[],
            Some(json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})),
        )
        .await?;
        assert_eq!(st, StatusCode::UNAUTHORIZED);

        // Legacy initialize mints Mcp-Session-Id.
        let (st, hdrs, body) = send_mcp_http(
            &app,
            Some(&p1),
            Method::POST,
            &[],
            Some(json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "legacy", "version": "1"}}})),
        )
        .await?;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(body["result"]["protocolVersion"], "2025-03-26");
        assert!(body["result"].get("resultType").is_none());
        let sid = hdrs
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| "missing Mcp-Session-Id".to_string())?
            .to_string();

        // Subsequent legacy call echoes Mcp-Session-Id.
        let (st, hdrs, body) = send_mcp_http(
            &app,
            Some(&p1),
            Method::POST,
            &[(SESSION_HEADER, &sid)],
            Some(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})),
        )
        .await?;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(
            hdrs.get(SESSION_HEADER).and_then(|v| v.to_str().ok()),
            Some(sid.as_str())
        );
        assert_eq!(body["result"]["tools"].as_array().map_or(0, Vec::len), 16);
        assert!(body["result"].get("resultType").is_none());

        // Another principal cannot ride the legacy session.
        let (st, _, _) = send_mcp_http(
            &app,
            Some(&p2),
            Method::POST,
            &[(SESSION_HEADER, &sid)],
            Some(json!({"jsonrpc": "2.0", "id": 3, "method": "ping"})),
        )
        .await?;
        assert_eq!(st, StatusCode::NOT_FOUND);

        // GET -> 405, DELETE -> 204.
        let (st, _, _) = send_mcp_http(&app, Some(&p1), Method::GET, &[], None).await?;
        assert_eq!(st, StatusCode::METHOD_NOT_ALLOWED);
        let (st, _, _) =
            send_mcp_http(&app, Some(&p1), Method::DELETE, &[(SESSION_HEADER, &sid)], None).await?;
        assert_eq!(st, StatusCode::NO_CONTENT);
        Ok(())
    }

    #[tokio::test]
    async fn http_modern_header_validation_and_statelessness() -> Result<(), String> {
        let state = AppState::new_offline_for_test().map_err(|e| e.to_string())?;
        let app = crate::build_router(state);
        let p = test_principal().map_err(|e| e.to_string())?;

        let modern_meta = |ver: &str| {
            json!({
                META_PROTOCOL_VERSION: ver,
                META_CLIENT_CAPABILITIES: {},
            })
        };

        let discover_body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": { "_meta": modern_meta("2026-07-28") }
        });

        // Missing MCP-Protocol-Version -> 400 -32020.
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[(METHOD_HEADER, "server/discover")],
            Some(discover_body.clone()),
        )
        .await?;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], JSONRPC_HEADER_MISMATCH);

        // Missing Mcp-Method -> 400 -32020.
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[(PROTOCOL_VERSION_HEADER, "2026-07-28")],
            Some(discover_body.clone()),
        )
        .await?;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], JSONRPC_HEADER_MISMATCH);

        // Mismatched MCP-Protocol-Version -> 400 -32020.
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2025-06-18"),
                (METHOD_HEADER, "server/discover"),
            ],
            Some(discover_body.clone()),
        )
        .await?;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], JSONRPC_HEADER_MISMATCH);

        // Valid discover ignores Mcp-Session-Id and never emits one.
        let (st, hdrs, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2026-07-28"),
                (METHOD_HEADER, "server/discover"),
                (SESSION_HEADER, "bogus-session-must-be-ignored"),
            ],
            Some(discover_body),
        )
        .await?;
        assert_eq!(st, StatusCode::OK);
        assert!(hdrs.get(SESSION_HEADER).is_none());
        assert_eq!(body["result"]["resultType"], "complete");

        // Unsupported version (matching header and _meta) -> 400 -32022 with data.
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2099-01-01"),
                (METHOD_HEADER, "server/discover"),
            ],
            Some(json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "server/discover",
                "params": { "_meta": modern_meta("2099-01-01") }
            })),
        )
        .await?;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], JSONRPC_UNSUPPORTED_PROTOCOL_VERSION);
        assert_eq!(
            body["error"]["data"]["supported"],
            json!(["2026-07-28", "2025-06-18", "2025-03-26", "2024-11-05"])
        );
        assert_eq!(body["error"]["data"]["requested"], "2099-01-01");

        // Missing Mcp-Name on tools/call -> 400 -32020.
        let call_body = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "rag_search",
                "arguments": { "query": "test" },
                "_meta": modern_meta("2026-07-28")
            }
        });
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2026-07-28"),
                (METHOD_HEADER, "tools/call"),
            ],
            Some(call_body.clone()),
        )
        .await?;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], JSONRPC_HEADER_MISMATCH);

        // Mismatched Mcp-Name on tools/call -> 400 -32020.
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2026-07-28"),
                (METHOD_HEADER, "tools/call"),
                (NAME_HEADER, "other_tool"),
            ],
            Some(call_body),
        )
        .await?;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], JSONRPC_HEADER_MISMATCH);

        // Base64 sentinel Mcp-Name accepted on resources/read.
        let skill_uri = "skill://openrag-search/SKILL.md";
        let b64_uri = format!(
            "=?base64?{}?=",
            base64::engine::general_purpose::STANDARD.encode(skill_uri.as_bytes())
        );
        let (st, hdrs, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2026-07-28"),
                (METHOD_HEADER, "resources/read"),
                (NAME_HEADER, &b64_uri),
            ],
            Some(json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {
                    "uri": skill_uri,
                    "_meta": modern_meta("2026-07-28")
                }
            })),
        )
        .await?;
        assert_eq!(st, StatusCode::OK);
        assert!(hdrs.get(SESSION_HEADER).is_none());
        assert_eq!(body["result"]["resultType"], "complete");
        assert_eq!(body["result"]["contents"][0]["uri"], skill_uri);

        // skills/get does NOT require Mcp-Name header.
        let (st, _, body) = send_mcp_http(
            &app,
            Some(&p),
            Method::POST,
            &[
                (PROTOCOL_VERSION_HEADER, "2026-07-28"),
                (METHOD_HEADER, "skills/get"),
            ],
            Some(json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "skills/get",
                "params": {
                    "uri": skill_uri,
                    "_meta": modern_meta("2026-07-28")
                }
            })),
        )
        .await?;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(body["result"]["skill"]["uri"], skill_uri);

        // Unknown modern method -> 404 -32601.
        for unknown in ["ping", "logging/setLevel", "resources/subscribe", "resources/directory/read"] {
            let (st, _, body) = send_mcp_http(
                &app,
                Some(&p),
                Method::POST,
                &[
                    (PROTOCOL_VERSION_HEADER, "2026-07-28"),
                    (METHOD_HEADER, unknown),
                ],
                Some(json!({
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": unknown,
                    "params": { "_meta": modern_meta("2026-07-28") }
                })),
            )
            .await?;
            assert_eq!(st, StatusCode::NOT_FOUND, "method={unknown}");
            assert_eq!(body["error"]["code"], JSONRPC_METHOD_NOT_FOUND);
        }
        Ok(())
    }

    #[tokio::test]
    async fn task_store_prunes_finished_tasks_after_ttl() -> Result<(), String> {
        let store = TaskStore::with_limits(25, 10, 32);
        let owner = PrincipalKey {
            org_id: Uuid::new_v4(),
            principal_id: Uuid::new_v4(),
        };
        let (tid, _) = store
            .try_create(owner, None)
            .ok_or_else(|| "expected task creation".to_string())?;
        store.complete(&tid, json!({"resultType": "complete", "content": []}));
        assert!(store.get(&tid, owner).is_some());
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(store.get(&tid, owner).is_none());
        Ok(())
    }
}

