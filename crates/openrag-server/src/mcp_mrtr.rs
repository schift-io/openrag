//! Signed MRTR `requestState` tokens, single-use nonce tracking, and staged file upload storage
//! for MCP elicitation (Part B) and the draft File Uploads extension (`io.modelcontextprotocol/files`).
//!
//! Pending uploads and consumed nonces live in memory (like MCP tasks) and do not survive a server
//! restart. Staged file bytes are written under `<blob_dir>/mcp-staging/` and removed after the
//! retry finishes or once the latest `requestState` bound to them expires.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use chrono::Utc;
use openrag_pii::MasterKek;
use ring::rand::{SecureRandom, SystemRandom};
use ring::{hkdf, hmac};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::mcp_tasks::PrincipalKey;

pub(crate) const REQUEST_STATE_VERSION: u8 = 1;
pub(crate) const REQUEST_STATE_HKDF_INFO: &[u8] = b"openrag/mcp-request-state/v1";
pub(crate) const REQUEST_STATE_TTL_SECS: i64 = 15 * 60;
pub(crate) const UPLOAD_TARGET_TTL_SECS: i64 = 10 * 60;
pub(crate) const MAX_PENDING_UPLOADS_PER_PRINCIPAL: usize = 8;
pub(crate) const MAX_INLINE_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_FILE_UPLOAD_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestStateKind {
    Upload,
    RemoteOcr,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StagedUploadBinding {
    pub(crate) file_id: String,
    pub(crate) upload_id: String,
    pub(crate) confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RequestStatePayload {
    pub(crate) v: u8,
    pub(crate) kind: RequestStateKind,
    pub(crate) org_id: Uuid,
    pub(crate) principal_id: Uuid,
    pub(crate) tool: String,
    pub(crate) args_sha256: String,
    pub(crate) exp: i64,
    pub(crate) nonce: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) uploads: Vec<StagedUploadBinding>,
    /// True once an upload request was issued in this logical call so no retry can emit a second.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(crate) upload_issued: bool,
    /// Carried for `RemoteOcr` so missing/malformed `inputResponses` can re-ask without re-parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) file_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) pages_needing_ocr: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) total_pages: Option<u64>,
}

struct HmacKeyLen;

impl hkdf::KeyType for HmacKeyLen {
    fn len(&self) -> usize {
        32
    }
}

/// Derives the `requestState` HMAC-SHA256 key from `MasterKek` via HKDF-SHA256.
fn derive_request_state_key(kek: &MasterKek) -> Result<hmac::Key, &'static str> {
    let salt = hkdf::Salt::new(hkdf::HKDF_SHA256, b"");
    let prk = salt.extract(kek.as_bytes());
    let info = [REQUEST_STATE_HKDF_INFO];
    let okm = prk
        .expand(&info, HmacKeyLen)
        .map_err(|_| "HKDF expand failed for requestState key")?;
    let mut key_bytes = Zeroizing::new([0u8; 32]);
    okm.fill(&mut *key_bytes)
        .map_err(|_| "HKDF fill failed for requestState key")?;
    Ok(hmac::Key::new(hmac::HMAC_SHA256, &*key_bytes))
}

/// Generates 128 random bits encoded as unpadded base64url (22 ASCII chars, valid URL path segment).
pub(crate) fn random_token_128() -> Result<String, &'static str> {
    let mut bytes = [0u8; 16];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "CSPRNG failure")?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// Encodes JSON canonically: sorted object keys at every depth, integral floats written as
/// integers (so `1` and `1.0` hash identically), compact.
pub(crate) fn canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::String(s) => {
            if let Ok(encoded) = serde_json::to_string(s) {
                out.push_str(&encoded);
            }
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    out.push_str("null");
                } else if f == 0.0 {
                    out.push('0');
                } else if f.fract() == 0.0 {
                    // Preserve JSON Schema instance equality between `1` and `1.0`.
                    if ((i64::MIN as f64)..9_223_372_036_854_775_808.0).contains(&f) {
                        out.push_str(&(f as i64).to_string());
                    } else if (0.0..18_446_744_073_709_551_616.0).contains(&f) {
                        out.push_str(&(f as u64).to_string());
                    } else if let Ok(s) = serde_json::to_string(&f) {
                        out.push_str(&s);
                    }
                } else if let Ok(s) = serde_json::to_string(&f) {
                    out.push_str(&s);
                }
            }
        }
        Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                canonical_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (idx, key) in keys.into_iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                if let Ok(encoded_key) = serde_json::to_string(key) {
                    out.push_str(&encoded_key);
                }
                out.push(':');
                if let Some(val) = map.get(key) {
                    canonical_json(val, out);
                }
            }
            out.push('}');
        }
    }
}

pub(crate) fn canonical_args_sha256(args: &Map<String, Value>) -> String {
    let mut canonical = String::new();
    canonical_json(&Value::Object(args.clone()), &mut canonical);
    let digest = Sha256::digest(canonical.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

pub(crate) fn sign_request_state(
    kek: &MasterKek,
    payload: &RequestStatePayload,
) -> Result<String, &'static str> {
    let key = derive_request_state_key(kek)?;
    let payload_bytes =
        serde_json::to_vec(payload).map_err(|_| "failed serializing requestState")?;
    let tag = hmac::sign(&key, &payload_bytes);
    Ok(format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(&payload_bytes),
        URL_SAFE_NO_PAD.encode(tag.as_ref()),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestStateVerifyError {
    Malformed,
    BadSignature,
    UnsupportedVersion,
    Expired,
    Consumed,
    PrincipalMismatch,
    ToolMismatch,
    ArgumentsMismatch,
    KeyDerivation,
}

/// Verifies `requestState` MAC (constant time), version, expiry, non-consumption, principal,
/// tool name, and canonical arguments digest. Does not consume the nonce yet.
pub(crate) fn verify_request_state(
    kek: &MasterKek,
    nonces: &RequestStateNonceStore,
    token: &str,
    principal: PrincipalKey,
    tool: &str,
    args: &Map<String, Value>,
    now_unix: i64,
) -> Result<RequestStatePayload, RequestStateVerifyError> {
    let Some((payload_b64, sig_b64)) = token.split_once('.') else {
        return Err(RequestStateVerifyError::Malformed);
    };
    if sig_b64.contains('.') || payload_b64.is_empty() || sig_b64.is_empty() {
        return Err(RequestStateVerifyError::Malformed);
    }
    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| RequestStateVerifyError::Malformed)?;
    let sig_bytes = URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|_| RequestStateVerifyError::Malformed)?;

    let key = derive_request_state_key(kek).map_err(|_| RequestStateVerifyError::KeyDerivation)?;
    hmac::verify(&key, &payload_bytes, &sig_bytes)
        .map_err(|_| RequestStateVerifyError::BadSignature)?;

    let payload: RequestStatePayload = serde_json::from_slice(&payload_bytes)
        .map_err(|_| RequestStateVerifyError::Malformed)?;
    if payload.v != REQUEST_STATE_VERSION {
        return Err(RequestStateVerifyError::UnsupportedVersion);
    }
    if now_unix >= payload.exp {
        return Err(RequestStateVerifyError::Expired);
    }
    if nonces.is_consumed(&payload.nonce, now_unix) {
        return Err(RequestStateVerifyError::Consumed);
    }
    if payload.org_id != principal.org_id || payload.principal_id != principal.principal_id {
        return Err(RequestStateVerifyError::PrincipalMismatch);
    }
    if payload.tool != tool {
        return Err(RequestStateVerifyError::ToolMismatch);
    }
    if payload.args_sha256 != canonical_args_sha256(args) {
        return Err(RequestStateVerifyError::ArgumentsMismatch);
    }
    Ok(payload)
}

/// In-memory set of consumed `requestState` nonces kept until each nonce's expiry.
/// Does not survive a server restart.
#[derive(Debug, Default)]
pub struct RequestStateNonceStore {
    consumed: Mutex<HashMap<String, i64>>,
}

impl RequestStateNonceStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, i64>> {
        self.consumed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn is_consumed(&self, nonce: &str, now_unix: i64) -> bool {
        let mut map = self.lock();
        map.retain(|_, exp| *exp > now_unix);
        map.contains_key(nonce)
    }

    /// Atomically marks `nonce` consumed until `exp`. Returns `false` if already consumed.
    pub(crate) fn try_consume(&self, nonce: &str, exp: i64, now_unix: i64) -> bool {
        let mut map = self.lock();
        map.retain(|_, e| *e > now_unix);
        if map.contains_key(nonce) {
            return false;
        }
        map.insert(nonce.to_string(), exp);
        true
    }
}

// ============================================================================
// Staged file upload store (Part C)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UploadSlotStatus {
    AwaitingPut,
    Writing,
    Uploaded,
}

#[derive(Debug, Clone)]
struct PendingUploadRecord {
    owner: PrincipalKey,
    expected_size: u64,
    target_expires_at: i64,
    state_expires_at: i64,
    part_path: PathBuf,
    final_path: PathBuf,
    status: UploadSlotStatus,
}

#[derive(Debug, Clone)]
pub(crate) struct PutSlotClaim {
    pub(crate) expected_size: u64,
    pub(crate) part_path: PathBuf,
    pub(crate) final_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PutClaimError {
    NotFoundOrExpired,
    Forbidden,
    AlreadyWritten,
    MissingContentLength,
    InvalidContentLength,
    PayloadTooLarge,
    ContentLengthMismatch,
}

/// In-memory store for pending MCP file uploads.
///
/// Pending uploads live in memory (capped at 8 per principal) and do not survive a server
/// restart. Staged files live under `<blob_dir>/mcp-staging/` and are deleted once consumed
/// or after their latest bound `requestState` expires.
pub struct McpUploadStore {
    staging_dir: PathBuf,
    inner: Mutex<HashMap<String, PendingUploadRecord>>,
    sweeper_started: AtomicBool,
}

impl McpUploadStore {
    pub fn new(staging_dir: PathBuf) -> Self {
        Self {
            staging_dir,
            inner: Mutex::new(HashMap::new()),
            sweeper_started: AtomicBool::new(false),
        }
    }

    pub fn staging_dir(&self) -> &Path {
        &self.staging_dir
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, PendingUploadRecord>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn prune_locked(map: &mut HashMap<String, PendingUploadRecord>, now_unix: i64) -> Vec<PathBuf> {
        let mut to_delete = Vec::new();
        map.retain(|_, rec| {
            if now_unix >= rec.state_expires_at {
                to_delete.push(rec.part_path.clone());
                to_delete.push(rec.final_path.clone());
                false
            } else {
                true
            }
        });
        to_delete
    }

    /// Spawns at most one background sweeper task per store when running inside a Tokio runtime.
    pub(crate) fn ensure_sweeper(self: &Arc<Self>) {
        if self
            .sweeper_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let store = Arc::clone(self);
                handle.spawn(async move {
                    let mut interval = tokio::time::interval(Duration::from_secs(60));
                    interval.tick().await;
                    loop {
                        interval.tick().await;
                        store.sweep_expired().await;
                    }
                });
            } else {
                self.sweeper_started.store(false, Ordering::Release);
            }
        }
    }

    /// Removes expired pending upload entries and deletes any orphaned staged files.
    pub async fn sweep_expired(&self) {
        self.sweep_expired_at(Utc::now().timestamp()).await;
    }

    #[doc(hidden)]
    pub async fn sweep_expired_at(&self, now_unix: i64) {
        let paths = {
            let mut map = self.lock();
            Self::prune_locked(&mut map, now_unix)
        };
        remove_files_best_effort(&paths).await;
    }

    /// Reserves a pending upload slot for `owner` (enforcing the per-principal cap of 8).
    pub(crate) async fn register(
        self: &Arc<Self>,
        owner: PrincipalKey,
        expected_size: u64,
        target_expires_at: i64,
        state_expires_at: i64,
    ) -> Result<String, &'static str> {
        self.ensure_sweeper();
        let upload_id = random_token_128()?;
        let now_unix = Utc::now().timestamp();
        let part_path = self.staging_dir.join(format!("{upload_id}.part"));
        let final_path = self.staging_dir.join(&upload_id);

        let (inserted, pruned) = {
            let mut map = self.lock();
            let pruned = Self::prune_locked(&mut map, now_unix);
            let live_for_owner = map
                .values()
                .filter(|rec| rec.owner == owner && now_unix < rec.state_expires_at)
                .count();
            if live_for_owner >= MAX_PENDING_UPLOADS_PER_PRINCIPAL {
                (false, pruned)
            } else {
                map.insert(
                    upload_id.clone(),
                    PendingUploadRecord {
                        owner,
                        expected_size,
                        target_expires_at,
                        state_expires_at,
                        part_path,
                        final_path,
                        status: UploadSlotStatus::AwaitingPut,
                    },
                );
                (true, pruned)
            }
        };
        remove_files_best_effort(&pruned).await;
        if !inserted {
            return Err("too many pending file uploads for this principal (limit 8)");
        }
        Ok(upload_id)
    }

    /// Validates ownership, expiry, one-write constraint, and `Content-Length`, transitioning the
    /// slot from `AwaitingPut` to `Writing`.
    pub(crate) async fn claim_for_put(
        &self,
        upload_id: &str,
        caller: PrincipalKey,
        content_length_hdr: Option<Result<u64, ()>>,
    ) -> Result<PutSlotClaim, PutClaimError> {
        if !is_safe_upload_id(upload_id) {
            return Err(PutClaimError::NotFoundOrExpired);
        }
        let now_unix = Utc::now().timestamp();
        let (result, pruned) = {
            let mut map = self.lock();
            let pruned = Self::prune_locked(&mut map, now_unix);
            let res = match map.get_mut(upload_id) {
                None => Err(PutClaimError::NotFoundOrExpired),
                Some(rec) => {
                    if now_unix >= rec.target_expires_at || now_unix >= rec.state_expires_at {
                        Err(PutClaimError::NotFoundOrExpired)
                    } else if rec.owner != caller {
                        Err(PutClaimError::Forbidden)
                    } else if rec.status != UploadSlotStatus::AwaitingPut {
                        Err(PutClaimError::AlreadyWritten)
                    } else {
                        match content_length_hdr {
                            None => Err(PutClaimError::MissingContentLength),
                            Some(Err(())) => Err(PutClaimError::InvalidContentLength),
                            Some(Ok(len)) if len > MAX_FILE_UPLOAD_BYTES => {
                                Err(PutClaimError::PayloadTooLarge)
                            }
                            Some(Ok(len)) if len != rec.expected_size => {
                                Err(PutClaimError::ContentLengthMismatch)
                            }
                            Some(Ok(_)) => {
                                rec.status = UploadSlotStatus::Writing;
                                Ok(PutSlotClaim {
                                    expected_size: rec.expected_size,
                                    part_path: rec.part_path.clone(),
                                    final_path: rec.final_path.clone(),
                                })
                            }
                        }
                    }
                }
            };
            (res, pruned)
        };
        remove_files_best_effort(&pruned).await;
        result
    }

    pub(crate) fn mark_uploaded(&self, upload_id: &str) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(upload_id) {
            rec.status = UploadSlotStatus::Uploaded;
        }
    }

    /// Reverts a failed/aborted in-progress PUT before completion so the `.part` file is cleaned up.
    pub(crate) fn abort_put(&self, upload_id: &str) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(upload_id) {
            if rec.status == UploadSlotStatus::Writing {
                rec.status = UploadSlotStatus::AwaitingPut;
            }
        }
    }

    /// Returns the staged final file path if the upload belongs to `caller`, has completed its
    /// PUT, and has not expired.
    pub(crate) fn get_uploaded_path(&self, upload_id: &str, caller: PrincipalKey) -> Option<PathBuf> {
        let now_unix = Utc::now().timestamp();
        let map = self.lock();
        let rec = map.get(upload_id)?;
        if rec.owner == caller
            && rec.status == UploadSlotStatus::Uploaded
            && now_unix < rec.state_expires_at
        {
            Some(rec.final_path.clone())
        } else {
            None
        }
    }

    /// Keeps a staged file alive until a follow-up `requestState` (e.g. Part B consent) expires.
    pub(crate) fn extend_state_expiry(&self, upload_id: &str, new_state_exp: i64) {
        let mut map = self.lock();
        if let Some(rec) = map.get_mut(upload_id) {
            if new_state_exp > rec.state_expires_at {
                rec.state_expires_at = new_state_exp;
            }
        }
    }

    /// Removes the upload entry and deletes its staged files from disk.
    pub(crate) async fn remove_and_delete(&self, upload_id: &str) {
        let paths = {
            let mut map = self.lock();
            map.remove(upload_id)
                .map(|rec| vec![rec.part_path, rec.final_path])
                .unwrap_or_default()
        };
        remove_files_best_effort(&paths).await;
    }

    /// Synchronous drop-safe deletion helper (used when a background task is aborted/cancelled).
    pub(crate) fn remove_and_delete_sync(&self, upload_id: &str) {
        let paths = {
            let mut map = self.lock();
            map.remove(upload_id)
                .map(|rec| vec![rec.part_path, rec.final_path])
                .unwrap_or_default()
        };
        for path in paths {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Drop guard ensuring staged upload files are deleted even if a background task is cancelled.
pub(crate) struct StagedUploadGuard {
    store: Arc<McpUploadStore>,
    upload_ids: Vec<String>,
}

impl StagedUploadGuard {
    pub(crate) fn new(store: Arc<McpUploadStore>, upload_ids: Vec<String>) -> Self {
        Self { store, upload_ids }
    }

    pub(crate) fn disarm(&mut self) {
        self.upload_ids.clear();
    }
}

impl Drop for StagedUploadGuard {
    fn drop(&mut self) {
        for id in self.upload_ids.drain(..) {
            self.store.remove_and_delete_sync(&id);
        }
    }
}

fn is_safe_upload_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

async fn remove_files_best_effort(paths: &[PathBuf]) {
    for path in paths {
        let _ = tokio::fs::remove_file(path).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_normalizes_key_order_and_integral_floats() {
        let a: Map<String, Value> = serde_json::from_value(json!({
            "b": 1,
            "a": { "z": 2.0, "y": [1.0, 2, -0.0, 1.5] }
        }))
        .unwrap_or_default();
        let b: Map<String, Value> = serde_json::from_value(json!({
            "a": { "y": [1, 2.0, 0, 1.50], "z": 2 },
            "b": 1.0
        }))
        .unwrap_or_default();
        assert_eq!(canonical_args_sha256(&a), canonical_args_sha256(&b));

        let c: Map<String, Value> = serde_json::from_value(json!({
            "b": 2,
            "a": { "z": 2, "y": [1, 2, 0, 1.5] }
        }))
        .unwrap_or_default();
        assert_ne!(canonical_args_sha256(&a), canonical_args_sha256(&c));
    }

    #[test]
    fn request_state_sign_verify_tamper_expiry_replay_and_binding() -> Result<(), String> {
        let kek = crate::state::resolve_master_kek_dev_default();
        let nonces = RequestStateNonceStore::new();
        let alice = PrincipalKey {
            org_id: Uuid::new_v4(),
            principal_id: Uuid::new_v4(),
        };
        let bob = PrincipalKey {
            org_id: alice.org_id,
            principal_id: Uuid::new_v4(),
        };
        let args: Map<String, Value> = serde_json::from_value(json!({
            "bucket": "docs",
            "revision": 1
        }))
        .map_err(|e| e.to_string())?;
        let equiv_args: Map<String, Value> = serde_json::from_value(json!({
            "revision": 1.0,
            "bucket": "docs"
        }))
        .map_err(|e| e.to_string())?;
        let other_args: Map<String, Value> = serde_json::from_value(json!({
            "bucket": "other",
            "revision": 1
        }))
        .map_err(|e| e.to_string())?;

        let now = 1_700_000_000i64;
        let payload = RequestStatePayload {
            v: REQUEST_STATE_VERSION,
            kind: RequestStateKind::RemoteOcr,
            org_id: alice.org_id,
            principal_id: alice.principal_id,
            tool: "rag_ingest".into(),
            args_sha256: canonical_args_sha256(&args),
            exp: now + 900,
            nonce: random_token_128().map_err(str::to_string)?,
            uploads: Vec::new(),
            upload_issued: false,
            file_name: Some("scan.pdf".into()),
            pages_needing_ocr: Some(2),
            total_pages: Some(3),
        };
        let token = sign_request_state(&kek, &payload).map_err(str::to_string)?;

        // Valid with equivalent canonical arguments (`1` vs `1.0` and reordered keys).
        let verified = verify_request_state(&kek, &nonces, &token, alice, "rag_ingest", &equiv_args, now)
            .map_err(|e| format!("{e:?}"))?;
        assert_eq!(verified.nonce, payload.nonce);

        // Malformed token fails.
        assert_eq!(
            verify_request_state(&kek, &nonces, "not-a-token", alice, "rag_ingest", &args, now).err(),
            Some(RequestStateVerifyError::Malformed)
        );
        // Tampered payload or signature fails with BadSignature.
        let mut tampered = token.clone();
        tampered.push('A');
        assert_eq!(
            verify_request_state(&kek, &nonces, &tampered, alice, "rag_ingest", &args, now).err(),
            Some(RequestStateVerifyError::BadSignature)
        );
        let (p_b64, s_b64) = token.split_once('.').ok_or("missing dot")?;
        let mut s_bytes = URL_SAFE_NO_PAD.decode(s_b64).map_err(|e| e.to_string())?;
        s_bytes[0] ^= 0xff;
        let bad_sig_token = format!("{p_b64}.{}", URL_SAFE_NO_PAD.encode(s_bytes));
        assert_eq!(
            verify_request_state(&kek, &nonces, &bad_sig_token, alice, "rag_ingest", &args, now).err(),
            Some(RequestStateVerifyError::BadSignature)
        );

        // Expired fails.
        assert_eq!(
            verify_request_state(&kek, &nonces, &token, alice, "rag_ingest", &args, now + 900).err(),
            Some(RequestStateVerifyError::Expired)
        );

        // Wrong principal fails.
        assert_eq!(
            verify_request_state(&kek, &nonces, &token, bob, "rag_ingest", &args, now).err(),
            Some(RequestStateVerifyError::PrincipalMismatch)
        );

        // Wrong tool fails.
        assert_eq!(
            verify_request_state(&kek, &nonces, &token, alice, "rag_search", &args, now).err(),
            Some(RequestStateVerifyError::ToolMismatch)
        );

        // Changed arguments fail.
        assert_eq!(
            verify_request_state(&kek, &nonces, &token, alice, "rag_ingest", &other_args, now).err(),
            Some(RequestStateVerifyError::ArgumentsMismatch)
        );

        // Consumed nonce fails replay.
        assert!(nonces.try_consume(&payload.nonce, payload.exp, now));
        assert!(!nonces.try_consume(&payload.nonce, payload.exp, now));
        assert_eq!(
            verify_request_state(&kek, &nonces, &token, alice, "rag_ingest", &args, now).err(),
            Some(RequestStateVerifyError::Consumed)
        );

        Ok(())
    }
}
