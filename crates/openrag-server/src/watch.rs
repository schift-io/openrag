//! Watched document folders. The server keeps buckets in step with folders on disk: a new file is
//! ingested, a changed file replaces its in-force copy, a moved or renamed file keeps its document
//! (matched by the file's SHA-256, so nothing is parsed or embedded again), and a deleted file goes
//! out of force (`valid_to`) while its record stays. When a file with that content shows up again
//! (restored from the trash, a remounted share), the same document comes back into force.
//!
//! Configuration (environment):
//! - `OPENRAG_DOCS_DIRS`: folders to watch, a JSON list such as `["/docs", "/mnt/hr"]` (entries may
//!   be objects: `{"path": "/mnt/hr", "bucket": "hr", "types": ["pdf", "hwp"]}`) or a comma-separated
//!   list. Unset: no watching.
//! - `OPENRAG_DOCS_TYPES`: file types to pick up (default `hwp, hwpx, md, pdf, docx, pptx, html`).
//! - `OPENRAG_DOCS_BUCKET`: bucket for entries without one (default `docs`).
//! - `OPENRAG_DOCS_SCAN_SECS`: seconds between scans (default 30).
//! - `OPENRAG_DOCS_ORG`: organization slug, when the database holds more than one.
//! - `OPENRAG_DOCS_HOST_DIR`: where `/docs` lives on the host, so search results can name the
//!   original file as the user sees it (compose sets it from `OPENRAG_DOCS_DIR`). Entries may set
//!   their own `"host"`.
//!
//! Folders are scanned, not subscribed to, so network shares and Docker bind mounts work the same.
//! A scan only reads files whose size or modification time changed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use openrag_core::{OpenRagError, PrincipalContext};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::api_core::{ingest_document_into_bucket, row_to_bucket, BucketRecord, IngestPayload};
use crate::auth::load_principal_context;
use crate::state::{non_empty_env, AppState};

/// File types picked up when `OPENRAG_DOCS_TYPES` is unset.
pub const DEFAULT_TYPES: &[&str] = &["hwp", "hwpx", "md", "pdf", "docx", "pptx", "html", "xlsx", "csv"];

/// Every type the parser reads.
pub const SUPPORTED_TYPES: &[&str] = &[
    "hwp", "hwpx", "hml", "pdf", "md", "markdown", "txt", "docx", "pptx", "html", "htm", "xlsx",
    "xlsm", "csv", "tsv", "png", "jpg", "jpeg", "webp",
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
    /// The same folder as seen on the host, for showing original paths.
    pub host: Option<String>,
}

fn host_dir(raw: Option<&str>) -> Option<String> {
    let h = raw?.trim().trim_end_matches(['/', '\\']);
    // Only absolute paths (`/srv/docs`, `C:\\Docs`) can name a file for the user.
    let absolute = h.starts_with('/') || h.as_bytes().get(1) == Some(&b':');
    (absolute && !h.is_empty()).then(|| h.to_string())
}

impl WatchFolder {
    /// The host path of a file under this folder, when the host location is known.
    pub fn host_path(&self, key: &str) -> Option<String> {
        let host = self.host.as_ref()?;
        let rest = key.strip_prefix(&self.path)?;
        let sep = if host.contains('\\') && !host.contains('/') { "\\" } else { "/" };
        Some(format!("{host}{}", rest.replace('/', sep)))
    }

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
        #[serde(default)]
        host: Option<String>,
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
        Ok(Self::parse(
            &dirs,
            non_empty_env("OPENRAG_DOCS_TYPES").as_deref(),
            non_empty_env("OPENRAG_DOCS_BUCKET").as_deref(),
            non_empty_env("OPENRAG_DOCS_SCAN_SECS").as_deref(),
            non_empty_env("OPENRAG_DOCS_ORG").as_deref(),
        )?
        .map(|c| c.with_docs_host(non_empty_env("OPENRAG_DOCS_HOST_DIR").as_deref())))
    }

    /// Gives the `/docs` entry (when it has no `host` of its own) the host folder it is mounted from.
    pub fn with_docs_host(mut self, host: Option<&str>) -> Self {
        if let Some(h) = host_dir(host) {
            for f in self.folders.iter_mut().filter(|f| f.path == "/docs" && f.host.is_none()) {
                f.host = Some(h.clone());
            }
        }
        self
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
            let (path, bucket, types, host) = match entry {
                DirEntry::Path(p) => (p, None, None, None),
                DirEntry::Full { path, bucket, types, host } => (path, bucket, types, host),
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
            folders.push(WatchFolder { path, bucket, types, host: host_dir(host.as_deref()) });
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
    let _ = CONFIG.set(cfg.clone());
    if let Ok(mut r) = report().write() {
        r.running = true;
        r.interval_secs = cfg.interval.as_secs();
        r.folders = cfg
            .folders
            .iter()
            .map(|f| FolderStatus {
                path: f.path.clone(),
                host_path: f.host.clone(),
                bucket: f.bucket.clone(),
                types: f.types.clone(),
                available: true,
                files: 0,
            })
            .collect();
    }
    let watcher = Watcher::new(Arc::clone(state), cfg);
    tokio::spawn(watcher.run());
    Ok(())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

static CONFIG: OnceLock<DocsWatchConfig> = OnceLock::new();
static REPORT: OnceLock<RwLock<WatchReport>> = OnceLock::new();
static SCAN_NOW: OnceLock<Notify> = OnceLock::new();

fn report() -> &'static RwLock<WatchReport> {
    REPORT.get_or_init(|| RwLock::new(WatchReport::default()))
}

fn scan_now() -> &'static Notify {
    SCAN_NOW.get_or_init(Notify::new)
}

/// What the watcher is doing, for `GET /v2/watch`, `docs_status` and `openrag docs status`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WatchReport {
    pub running: bool,
    pub interval_secs: u64,
    pub folders: Vec<FolderStatus>,
    pub buckets: BTreeMap<String, BucketStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FolderStatus {
    pub path: String,
    pub host_path: Option<String>,
    pub bucket: String,
    pub types: Vec<String>,
    /// False while the folder is missing or unreadable (its documents are kept).
    pub available: bool,
    /// Files of the watched types found in the last scan.
    pub files: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BucketStatus {
    /// `waiting` (no organization or bucket yet), `scanning`, `idle` or `error`.
    pub state: String,
    pub message: Option<String>,
    pub last_scan_started_at: Option<DateTime<Utc>>,
    pub last_scan_finished_at: Option<DateTime<Utc>>,
    /// Documents in force that came from the folders.
    pub documents: usize,
    /// Files being ingested in the current scan: (done, total).
    pub progress: Option<(usize, usize)>,
    /// What the last scan that changed something did.
    pub last_changes: Option<Changes>,
    pub last_changed_at: Option<DateTime<Utc>>,
    pub failed: Vec<FailedFile>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Changes {
    pub added: usize,
    pub replaced: usize,
    pub moved: usize,
    pub removed: usize,
    pub restored: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailedFile {
    pub path: String,
    pub host_path: Option<String>,
    pub reason: String,
    pub since: DateTime<Utc>,
}

/// The current report; `None` when no folder is watched.
pub fn status() -> Option<WatchReport> {
    let r = report().read().ok()?.clone();
    r.running.then_some(r)
}

/// Asks the watcher to scan now instead of at the next interval. False when nothing is watched.
pub fn request_scan() -> bool {
    if CONFIG.get().is_none() {
        return false;
    }
    scan_now().notify_one();
    true
}

/// The file behind a watched document's key, when the key lies inside a watched folder.
pub fn watched_file(key: &str) -> Option<PathBuf> {
    let cfg = CONFIG.get()?;
    let path = Path::new(key);
    if !path.is_absolute() || path.components().any(|c| matches!(c, Component::ParentDir)) {
        return None;
    }
    cfg.folders
        .iter()
        .any(|f| key.strip_prefix(&f.path).is_some_and(|rest| rest.starts_with('/')))
        .then(|| path.to_path_buf())
}

/// The host path of a watched file, when its folder's host location is known.
pub fn host_path_of(key: &str) -> Option<String> {
    let cfg = CONFIG.get()?;
    cfg.folders
        .iter()
        .filter(|f| key.strip_prefix(&f.path).is_some_and(|rest| rest.starts_with('/')))
        .max_by_key(|f| f.path.len())
        .and_then(|f| f.host_path(key))
}

fn update_bucket(slug: &str, f: impl FnOnce(&mut BucketStatus)) {
    if let Ok(mut r) = report().write() {
        f(r.buckets.entry(slug.to_string()).or_default());
    }
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
        too_large: Vec::new(),
    };
    for folder in folders {
        let root = Path::new(&folder.path);
        if !root.is_dir() {
            scan.unavailable.insert(folder.path.clone());
            continue;
        }
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
    host: Option<String>,
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
    restored: usize,
    failed: usize,
}

struct Watcher {
    state: Arc<AppState>,
    cfg: DocsWatchConfig,
    /// key -> (size/mtime, sha) of files already hashed.
    hashes: HashMap<String, (Stamp, String)>,
    /// key -> size/mtime, reason and time of files that failed to ingest; retried once they change.
    failures: HashMap<String, (Stamp, String, DateTime<Utc>)>,
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

    fn host_of(&self, f: &Found) -> Option<String> {
        self.cfg.folders.iter().find(|w| w.path == f.root).and_then(|w| w.host_path(&f.key))
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
                    update_bucket(bucket, |b| {
                        b.state = "error".into();
                        b.message = Some(e.to_string());
                        b.progress = None;
                    });
                    let msg = format!("{bucket}: {e}");
                    if self.report_once(msg) {
                        tracing::warn!(bucket = %bucket, error = %e, "documents folder sync failed; retrying every scan");
                    }
                }
            }
            tokio::select! {
                () = tokio::time::sleep(self.cfg.interval) => {}
                () = scan_now().notified() => {}
            }
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
                for f in &self.cfg.folders {
                    update_bucket(&f.bucket, |b| {
                        b.state = "waiting".into();
                        b.message = Some("no organization yet: run `openrag init`".into());
                    });
                }
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
                update_bucket(slug, |b| {
                    b.state = "waiting".into();
                    b.message = Some(format!("bucket '{slug}' does not exist yet (`openrag init` creates `docs`)"));
                });
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
                    metadata->>'watch_size' AS size, metadata->>'watch_mtime' AS mtime,
                    metadata->>'watch_host_path' AS host
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
                host: get("host")?,
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

        update_bucket(slug, |b| {
            b.state = "scanning".into();
            b.message = None;
            b.last_scan_started_at = Some(Utc::now());
        });
        let (folders_owned, roots_owned) = (folders.to_vec(), all_roots.to_vec());
        let scan = tokio::task::spawn_blocking(move || scan_folders(&folders_owned, &roots_owned))
            .await
            .map_err(|e| OpenRagError::Internal(format!("folder scan: {e}")))?;
        if let Ok(mut r) = report().write() {
            for fs in r.folders.iter_mut().filter(|fs| fs.bucket == slug) {
                fs.available = !scan.unavailable.contains(&fs.path);
                fs.files = scan.files.iter().filter(|f| f.root == fs.path).count();
            }
        }
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

        // A folder that cannot be read keeps its documents. An empty one does not: its files
        // were removed, and if it was an unmounted share they come back as the same documents.
        let keep_roots = scan.unavailable.clone();

        let mut disk = Vec::with_capacity(scan.files.len());
        for f in &scan.files {
            let stamp = Stamp { size: f.size, mtime: f.mtime.clone() };
            let failing = self.failures.get(&f.key).is_some_and(|(s, _, _)| *s == stamp);
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
        let total = steps.iter().filter(|s| matches!(s, Step::Ingest { .. })).count();
        let (mut done, mut last_log) = (0usize, Instant::now());
        if total > 0 {
            update_bucket(slug, |b| b.progress = Some((0, total)));
            if total >= 20 {
                tracing::info!(bucket = %slug, files = total, "documents folder: ingesting new and changed files");
            }
        }
        for step in steps {
            match step {
                Step::Keep { doc, file } => {
                    let f = &scan.files[file];
                    let d = stored.iter().find(|d| d.doc.id == doc);
                    let fresh = d.is_some_and(|d| {
                        d.size.as_deref() == Some(f.size.to_string().as_str()) && d.mtime.as_deref() == Some(f.mtime.as_str())
                    });
                    let host_known = self.host_of(f).is_none() || d.is_some_and(|d| d.host.is_some());
                    if !fresh || !host_known {
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
                    if replaces.is_none() {
                        if let Some(doc) = self.restore(org, bucket.id, &f, &sha).await? {
                            tally.restored += 1;
                            tracing::info!(file = %f.key, document_id = %doc, "file is back; document in force again (not re-embedded)");
                            continue;
                        }
                    }
                    match self.ingest(&principal, &bucket, &f, &sha, replaces, &docs).await {
                        Ok(Some(true)) => tally.replaced += 1,
                        Ok(Some(false)) => tally.added += 1,
                        Ok(None) => {}
                        Err(e) => {
                            tally.failed += 1;
                            self.failures.insert(f.key.clone(), (Stamp { size: f.size, mtime: f.mtime.clone() }, e.to_string(), Utc::now()));
                            tracing::warn!(file = %f.key, error = %e, "cannot ingest file; retried when it changes");
                        }
                    }
                    done += 1;
                    update_bucket(slug, |b| b.progress = Some((done, total)));
                    if total >= 20 && (last_log.elapsed() >= Duration::from_secs(30) || done % 100 == 0) && done < total {
                        last_log = Instant::now();
                        tracing::info!(bucket = %slug, done, total, "documents folder: still ingesting");
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
        let in_force = self.stored_docs(org, bucket.id, &roots).await.map(|d| d.len()).unwrap_or(0);
        let mut failed: Vec<FailedFile> = self
            .failures
            .iter()
            .filter(|(k, _)| folders.iter().any(|f| k.starts_with(&format!("{}/", f.path))))
            .map(|(k, (_, reason, since))| FailedFile {
                path: k.clone(),
                host_path: host_path_of(k),
                reason: reason.clone(),
                since: *since,
            })
            .collect();
        failed.sort_by(|a, b| a.path.cmp(&b.path));
        let changed = tally.added + tally.replaced + tally.moved + tally.removed + tally.restored + tally.failed > 0;
        update_bucket(slug, |b| {
            b.state = "idle".into();
            b.progress = None;
            b.last_scan_finished_at = Some(Utc::now());
            b.documents = in_force;
            b.failed = failed;
            if changed {
                b.last_changes = Some(Changes {
                    added: tally.added,
                    replaced: tally.replaced,
                    moved: tally.moved,
                    removed: tally.removed,
                    restored: tally.restored,
                    failed: tally.failed,
                });
                b.last_changed_at = Some(Utc::now());
            }
        });
        if tally.added + tally.replaced + tally.moved + tally.removed + tally.restored + tally.failed > 0 {
            tracing::info!(
                bucket = %slug,
                added = tally.added,
                replaced = tally.replaced,
                moved = tally.moved,
                removed = tally.removed,
                restored = tally.restored,
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
            remote_ocr: openrag_parse::RemoteOcrConsent::Default,
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
        let rel = f.key.strip_prefix(&f.root).unwrap_or(&f.key).trim_start_matches('/');
        let meta = json!({
            "source": "watch",
            "watch_root": f.root,
            "external_key": f.key,
            "watch_rel_path": rel,
            "watch_host_path": self.host_of(f),
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

    /// Puts back in force the latest document the watcher retired with this content, at the
    /// file's current path. Nothing is parsed or embedded again.
    async fn restore(&self, org: Uuid, bucket: Uuid, f: &Found, sha: &str) -> Result<Option<Uuid>, OpenRagError> {
        let worker = self.state.require_worker_pool()?;
        let row = sqlx::query(
            "SELECT id, title, metadata->>'external_key' AS key FROM corpus.documents
             WHERE org_id = $1 AND bucket_id = $2 AND content_sha256 = $3
               AND metadata->>'source' = 'watch' AND metadata ? 'watch_removed_at'
               AND valid_to IS NOT NULL AND valid_to <= now()
               AND status IN ('indexed', 'pending')
             ORDER BY valid_to DESC
             LIMIT 1",
        )
        .bind(org)
        .bind(bucket)
        .bind(sha)
        .fetch_optional(worker.inner())
        .await
        .map_err(|e| db(e, "lookup retired document"))?;
        let Some(row) = row else { return Ok(None) };
        let id: Uuid = row.try_get("id").map_err(|e| db(e, "id"))?;
        let title: String = row.try_get("title").map_err(|e| db(e, "title"))?;
        let old_key: Option<String> = row.try_get("key").map_err(|e| db(e, "key"))?;
        sqlx::query(
            "UPDATE corpus.documents
             SET valid_to = NULL, updated_at = now(),
                 metadata = (metadata - 'watch_removed_at')
                     || jsonb_build_object('watch_restored_at', now(), 'watch_last_removed_at', metadata->'watch_removed_at')
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org)
        .bind(id)
        .execute(worker.inner())
        .await
        .map_err(|e| db(e, "restore document"))?;
        let renamed = old_key.as_deref().map(file_name) == Some(title.as_str()) && file_name(&f.key) != title;
        self.mark(org, id, f, renamed.then(|| file_name(&f.key))).await?;
        Ok(Some(id))
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
