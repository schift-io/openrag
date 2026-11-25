//! In-process store for the MCP Tasks extension (`io.modelcontextprotocol/tasks`).
//!
//! Tasks live in memory and are bound to the `(org_id, principal_id)` that created them.
//! Tasks do not survive a server restart: a subsequent `tasks/get`, `tasks/update`, or
//! `tasks/cancel` for a task created before restart returns JSON-RPC `-32602`.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::{json, Map, Value};
use tokio::sync::oneshot;
use tokio::task::AbortHandle;
use uuid::Uuid;

pub(crate) const DEFAULT_TASK_TTL_MS: u64 = 3_600_000;
pub(crate) const DEFAULT_POLL_INTERVAL_MS: u64 = 2_000;
pub(crate) const MAX_LIVE_TASKS_PER_PRINCIPAL: usize = 32;

/// Outcome returned by an injected test hook future.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum TaskHookOutcome {
    ToolOk(Value),
    ToolErr { code: String, message: String },
    RpcErr { code: i64, message: String },
}

/// Optional test hook that intercepts selected `tools/call` invocations with a custom future.
#[doc(hidden)]
pub type TaskHookFn = Arc<
    dyn Fn(&str, &Map<String, Value>) -> Option<Pin<Box<dyn Future<Output = TaskHookOutcome> + Send>>>
        + Send
        + Sync,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PrincipalKey {
    pub(crate) org_id: Uuid,
    pub(crate) principal_id: Uuid,
}

enum TaskState {
    Working,
    InputRequired {
        pending_key: String,
        input_requests: Map<String, Value>,
        responder: Option<oneshot::Sender<Value>>,
    },
    Completed(Value),
    Failed(Value),
    Cancelled,
}

impl TaskState {
    const fn status_str(&self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::InputRequired { .. } => "input_required",
            Self::Completed(_) => "completed",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    const fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed(_) | Self::Failed(_) | Self::Cancelled)
    }
}

struct TaskRecord {
    task_id: String,
    owner: PrincipalKey,
    state: TaskState,
    status_message: Option<String>,
    created_at: String,
    last_updated_at: String,
    ttl_ms: u64,
    poll_interval_ms: u64,
    finished_at: Option<Instant>,
    abort_handle: Option<AbortHandle>,
    next_input_seq: u64,
}

impl TaskRecord {
    fn is_expired(&self, now: Instant) -> bool {
        self.finished_at
            .is_some_and(|fin| now.saturating_duration_since(fin) >= Duration::from_millis(self.ttl_ms))
    }

    /// Flat `Task` fields shared by `CreateTaskResult` and `GetTaskResult` (per `ext-tasks_schema_v2.ts`).
    fn to_task_json(&self, include_detail: bool) -> Value {
        let mut obj = Map::new();
        obj.insert("taskId".into(), json!(self.task_id));
        obj.insert("status".into(), json!(self.state.status_str()));
        if let Some(msg) = &self.status_message {
            obj.insert("statusMessage".into(), json!(msg));
        }
        obj.insert("createdAt".into(), json!(self.created_at));
        obj.insert("lastUpdatedAt".into(), json!(self.last_updated_at));
        obj.insert("ttlMs".into(), json!(self.ttl_ms));
        if !self.state.is_terminal() {
            obj.insert("pollIntervalMs".into(), json!(self.poll_interval_ms));
        }
        if include_detail {
            match &self.state {
                TaskState::InputRequired { input_requests, .. } => {
                    obj.insert("inputRequests".into(), Value::Object(input_requests.clone()));
                }
                TaskState::Completed(res) => {
                    obj.insert("result".into(), res.clone());
                }
                TaskState::Failed(err) => {
                    obj.insert("error".into(), err.clone());
                }
                TaskState::Working | TaskState::Cancelled => {}
            }
        }
        Value::Object(obj)
    }
}

/// In-process task store shared across requests on one server process.
/// Tasks are not persisted across server restarts.
#[doc(hidden)]
pub struct TaskStore {
    inner: Mutex<HashMap<String, TaskRecord>>,
    ttl_ms: u64,
    poll_interval_ms: u64,
    max_live_per_principal: usize,
}

impl TaskStore {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_TASK_TTL_MS,
            DEFAULT_POLL_INTERVAL_MS,
            MAX_LIVE_TASKS_PER_PRINCIPAL,
        )
    }

    #[doc(hidden)]
    pub fn with_limits(ttl_ms: u64, poll_interval_ms: u64, max_live_per_principal: usize) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            ttl_ms,
            poll_interval_ms,
            max_live_per_principal,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, TaskRecord>> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn prune_locked(map: &mut HashMap<String, TaskRecord>, now: Instant) {
        map.retain(|_, rec| !rec.is_expired(now));
    }

    /// Attempts to reserve a live task slot for `owner`.
    /// Returns `None` if `owner` already has `max_live_per_principal` live tasks.
    pub(crate) fn try_create(
        &self,
        owner: PrincipalKey,
        status_message: Option<String>,
    ) -> Option<(String, Value)> {
        let mut map = self.lock();
        let now = Instant::now();
        Self::prune_locked(&mut map, now);

        let live_count = map
            .values()
            .filter(|rec| rec.owner == owner && !rec.state.is_terminal())
            .count();
        if live_count >= self.max_live_per_principal {
            return None;
        }

        let task_id = Uuid::new_v4().to_string();
        let ts = Utc::now().to_rfc3339();
        let rec = TaskRecord {
            task_id: task_id.clone(),
            owner,
            state: TaskState::Working,
            status_message,
            created_at: ts.clone(),
            last_updated_at: ts,
            ttl_ms: self.ttl_ms,
            poll_interval_ms: self.poll_interval_ms,
            finished_at: None,
            abort_handle: None,
            next_input_seq: 1,
        };
        let create_payload = rec.to_task_json(false);
        map.insert(task_id.clone(), rec);
        Some((task_id, create_payload))
    }

    /// Registers the spawned background task's `AbortHandle`.
    /// If the task was already cancelled before the handle arrived, aborts immediately.
    pub(crate) fn set_abort_handle(&self, task_id: &str, handle: AbortHandle) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(task_id) {
            if rec.state.is_terminal() {
                handle.abort();
            } else {
                rec.abort_handle = Some(handle);
            }
        } else {
            handle.abort();
        }
    }

    pub(crate) fn set_status_message(&self, task_id: &str, message: impl Into<String>) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(task_id) {
            if !rec.state.is_terminal() {
                rec.status_message = Some(message.into());
                rec.last_updated_at = Utc::now().to_rfc3339();
            }
        }
    }

    pub(crate) fn complete(&self, task_id: &str, result: Value) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(task_id) {
            if !rec.state.is_terminal() {
                rec.state = TaskState::Completed(result);
                rec.status_message = None;
                rec.last_updated_at = Utc::now().to_rfc3339();
                rec.finished_at = Some(Instant::now());
                rec.abort_handle = None;
            }
        }
    }

    pub(crate) fn fail(&self, task_id: &str, error_obj: Value, status_message: Option<String>) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(task_id) {
            if !rec.state.is_terminal() {
                rec.state = TaskState::Failed(error_obj);
                rec.status_message = status_message;
                rec.last_updated_at = Utc::now().to_rfc3339();
                rec.finished_at = Some(Instant::now());
                rec.abort_handle = None;
            }
        }
    }

    pub(crate) fn get(&self, task_id: &str, caller: PrincipalKey) -> Option<Value> {
        let mut map = self.lock();
        let now = Instant::now();
        Self::prune_locked(&mut map, now);
        let rec = map.get(task_id)?;
        if rec.owner != caller {
            return None;
        }
        Some(rec.to_task_json(true))
    }

    /// Moves a live task into `input_required` with a task-lifetime-unique request key
    /// (`<key_prefix>-<seq>`), returning a receiver that resolves when `tasks/update` answers it.
    pub(crate) fn request_input(
        &self,
        task_id: &str,
        key_prefix: &str,
        request_obj: Value,
        status_message: Option<String>,
    ) -> Option<oneshot::Receiver<Value>> {
        let mut map = self.lock();
        let rec = map.get_mut(task_id)?;
        if rec.state.is_terminal() {
            return None;
        }
        let seq = rec.next_input_seq;
        rec.next_input_seq = rec.next_input_seq.saturating_add(1);
        let key = format!("{key_prefix}-{seq}");
        let mut input_requests = Map::new();
        input_requests.insert(key.clone(), request_obj);
        let (tx, rx) = oneshot::channel();
        rec.state = TaskState::InputRequired {
            pending_key: key,
            input_requests,
            responder: Some(tx),
        };
        rec.status_message = status_message;
        rec.last_updated_at = Utc::now().to_rfc3339();
        Some(rx)
    }

    /// Validates ownership and liveness for `tasks/update`, delivering a matching outstanding
    /// input response and returning the task to `working`. Unknown or non-outstanding keys are
    /// ignored per the Tasks extension specification.
    pub(crate) fn deliver_update(
        &self,
        task_id: &str,
        caller: PrincipalKey,
        input_responses: &Map<String, Value>,
    ) -> bool {
        let mut map = self.lock();
        let now = Instant::now();
        Self::prune_locked(&mut map, now);
        let Some(rec) = map.get_mut(task_id) else {
            return false;
        };
        if rec.owner != caller {
            return false;
        }
        if let TaskState::InputRequired {
            pending_key,
            responder,
            ..
        } = &mut rec.state
        {
            if let Some(answer) = input_responses.get(pending_key.as_str()).cloned() {
                let tx = responder.take();
                rec.state = TaskState::Working;
                rec.status_message = Some("Resuming after user input...".to_string());
                rec.last_updated_at = Utc::now().to_rfc3339();
                if let Some(tx) = tx {
                    let _ = tx.send(answer);
                }
            }
        }
        true
    }

    /// Cancels a task owned by `caller`. If the task already reached a terminal state,
    /// its terminal state is preserved. Returns `false` if unknown, expired, or unowned.
    pub(crate) fn cancel(&self, task_id: &str, caller: PrincipalKey) -> bool {
        let mut map = self.lock();
        let now = Instant::now();
        Self::prune_locked(&mut map, now);
        let Some(rec) = map.get_mut(task_id) else {
            return false;
        };
        if rec.owner != caller {
            return false;
        }
        if !rec.state.is_terminal() {
            if let Some(handle) = rec.abort_handle.take() {
                handle.abort();
            }
            rec.state = TaskState::Cancelled;
            rec.status_message = Some("Task cancelled by client".to_string());
            rec.last_updated_at = Utc::now().to_rfc3339();
            rec.finished_at = Some(now);
        }
        true
    }
}

impl Default for TaskStore {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn default_task_store() -> Arc<TaskStore> {
    static STORE: OnceLock<Arc<TaskStore>> = OnceLock::new();
    Arc::clone(STORE.get_or_init(|| Arc::new(TaskStore::new())))
}
