use base64::Engine as _;
use chrono::{DateTime, Utc};
use openrag_core::{BBoxNorm1000, OpenRagError};
use openrag_db::RlsTransaction;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};
use sqlx::{PgPool, Row};
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::category::PiiCategory;

#[derive(Clone)]
pub struct MasterKek {
    key_bytes: Zeroizing<[u8; 32]>,
}

impl std::fmt::Debug for MasterKek {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MasterKek([REDACTED])")
    }
}

impl MasterKek {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self {
            key_bytes: Zeroizing::new(bytes),
        }
    }

    pub fn from_env() -> Result<Self, OpenRagError> {
        if let Ok(b64) = std::env::var("OPENRAG_PII_KEK_B64") {
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(b64.trim())
                .map_err(|e| OpenRagError::Config(format!("invalid OPENRAG_PII_KEK_B64: {e}")))?;
            let arr: [u8; 32] = decoded.try_into().map_err(|_| {
                OpenRagError::Config("OPENRAG_PII_KEK_B64 must decode to 32 bytes".into())
            })?;
            return Ok(Self::from_bytes(arr));
        }
        if let Ok(hex_str) = std::env::var("OPENRAG_PII_KEK_HEX") {
            let decoded = hex::decode(hex_str.trim())
                .map_err(|e| OpenRagError::Config(format!("invalid OPENRAG_PII_KEK_HEX: {e}")))?;
            let arr: [u8; 32] = decoded.try_into().map_err(|_| {
                OpenRagError::Config("OPENRAG_PII_KEK_HEX must decode to 32 bytes".into())
            })?;
            return Ok(Self::from_bytes(arr));
        }
        Err(OpenRagError::Config(
            "OPENRAG_PII_KEK_B64 or OPENRAG_PII_KEK_HEX environment variable is required".into(),
        ))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key_bytes
    }
}

pub struct VaultCrypto;

impl VaultCrypto {
    pub fn generate_dek() -> Result<Zeroizing<[u8; 32]>, OpenRagError> {
        let rng = SystemRandom::new();
        let mut dek = Zeroizing::new([0u8; 32]);
        rng.fill(&mut *dek)
            .map_err(|_| OpenRagError::Internal("CSPRNG failed generating DEK".into()))?;
        Ok(dek)
    }

    pub fn wrap_dek(
        kek: &MasterKek,
        org_id: Uuid,
        dek: &[u8; 32],
    ) -> Result<Vec<u8>, OpenRagError> {
        let mut nonce_bytes = [0u8; 12];
        SystemRandom::new()
            .fill(&mut nonce_bytes)
            .map_err(|_| OpenRagError::Internal("CSPRNG failed generating KEK nonce".into()))?;

        let unbound = UnboundKey::new(&AES_256_GCM, kek.as_bytes())
            .map_err(|_| OpenRagError::Internal("invalid AES-256-GCM KEK".into()))?;
        let key = LessSafeKey::new(unbound);

        let mut in_out = dek.to_vec();
        key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce_bytes),
            Aad::from(org_id.as_bytes()),
            &mut in_out,
        )
        .map_err(|_| OpenRagError::Internal("failed wrapping DEK with KEK".into()))?;

        let mut wrapped = Vec::with_capacity(12 + in_out.len());
        wrapped.extend_from_slice(&nonce_bytes);
        wrapped.extend_from_slice(&in_out);
        Ok(wrapped)
    }

    pub fn unwrap_dek(
        kek: &MasterKek,
        org_id: Uuid,
        wrapped: &[u8],
    ) -> Result<Zeroizing<[u8; 32]>, OpenRagError> {
        if wrapped.len() != 12 + 32 + 16 {
            return Err(OpenRagError::Validation(format!(
                "invalid wrapped DEK length: expected 60, got {}",
                wrapped.len()
            )));
        }
        let nonce_bytes: [u8; 12] = wrapped[..12]
            .try_into()
            .map_err(|_| OpenRagError::Internal("nonce slice mismatch".into()))?;
        let mut ct = Zeroizing::new(wrapped[12..].to_vec());

        let unbound = UnboundKey::new(&AES_256_GCM, kek.as_bytes())
            .map_err(|_| OpenRagError::Internal("invalid AES-256-GCM KEK".into()))?;
        let key = LessSafeKey::new(unbound);

        let plain = key
            .open_in_place(
                Nonce::assume_unique_for_key(nonce_bytes),
                Aad::from(org_id.as_bytes()),
                &mut ct,
            )
            .map_err(|_| {
                OpenRagError::Forbidden("DEK authentication failed (wrong KEK/org or tampered)".into())
            })?;

        let arr: [u8; 32] = plain
            .try_into()
            .map_err(|_| OpenRagError::Internal("unwrapped DEK was not 32 bytes".into()))?;
        Ok(Zeroizing::new(arr))
    }

    pub fn encrypt_value(
        dek: &[u8; 32],
        org_id: Uuid,
        token: &str,
        plaintext: &str,
    ) -> Result<(Vec<u8>, Vec<u8>), OpenRagError> {
        let mut nonce_bytes = [0u8; 12];
        SystemRandom::new()
            .fill(&mut nonce_bytes)
            .map_err(|_| OpenRagError::Internal("CSPRNG failed generating value nonce".into()))?;

        let unbound = UnboundKey::new(&AES_256_GCM, dek)
            .map_err(|_| OpenRagError::Internal("invalid AES-256-GCM DEK".into()))?;
        let key = LessSafeKey::new(unbound);

        let aad_str = format!("{org_id}:{token}");
        let mut in_out = plaintext.as_bytes().to_vec();
        key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce_bytes),
            Aad::from(aad_str.as_bytes()),
            &mut in_out,
        )
        .map_err(|_| OpenRagError::Internal("failed encrypting PII value".into()))?;

        Ok((in_out, nonce_bytes.to_vec()))
    }

    pub fn decrypt_value(
        dek: &[u8; 32],
        org_id: Uuid,
        token: &str,
        ciphertext: &[u8],
        nonce: &[u8],
    ) -> Result<String, OpenRagError> {
        if nonce.len() != 12 || ciphertext.len() < 16 {
            return Err(OpenRagError::Forbidden(
                "invalid or wiped PII vault ciphertext/nonce".into(),
            ));
        }
        let nonce_arr: [u8; 12] = nonce
            .try_into()
            .map_err(|_| OpenRagError::Forbidden("invalid AES-GCM nonce length".into()))?;

        let unbound = UnboundKey::new(&AES_256_GCM, dek)
            .map_err(|_| OpenRagError::Internal("invalid AES-256-GCM DEK".into()))?;
        let key = LessSafeKey::new(unbound);

        let aad_str = format!("{org_id}:{token}");
        let mut buf = Zeroizing::new(ciphertext.to_vec());
        let plain = key
            .open_in_place(
                Nonce::assume_unique_for_key(nonce_arr),
                Aad::from(aad_str.as_bytes()),
                &mut buf,
            )
            .map_err(|_| {
                OpenRagError::Forbidden(
                    "vault entry AES-256-GCM authentication failed (tampered or wrong org)".into(),
                )
            })?;

        let s = std::str::from_utf8(plain)
            .map_err(|_| OpenRagError::Internal("decrypted PII payload is not UTF-8".into()))?;
        Ok(s.to_string())
    }

    /// Generates an opaque deterministic token `[PII:<TAG>:<8hex>]` without leaking plaintext.
    pub fn make_token(dek: &[u8; 32], org_id: Uuid, category: PiiCategory, plaintext: &str) -> String {
        let key = hmac::Key::new(hmac::HMAC_SHA256, dek);
        let msg = format!(
            "{org_id}:{}:{plaintext}",
            category.as_canonical_str()
        );
        let tag = hmac::sign(&key, msg.as_bytes());
        let short_hex = hex::encode(&tag.as_ref()[..4]);
        format!("[PII:{}:{short_hex}]", category.token_tag())
    }
}

#[derive(Debug, Clone)]
pub struct VaultEntryRow {
    pub id: Uuid,
    pub org_id: Uuid,
    pub vault_key_id: Uuid,
    pub token: String,
    pub entity_type: String,
    pub ciphertext: Vec<u8>,
    pub nonce: Vec<u8>,
    pub detector_source: String,
    pub purged_at: Option<DateTime<Utc>>,
    pub wrapped_dek: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct StoredChunkSpanInput {
    pub vault_entry_id: Option<Uuid>,
    pub entity_type: String,
    pub start_offset: i32,
    pub end_offset: i32,
    pub bbox_norm_1000: Option<BBoxNorm1000>,
}

/// Repository for `pii.vault_keys`, `pii.vault_entries`, `pii.chunk_spans` (worker pool),
/// and `pii.audit_logs` (RLS-safe via `RlsTransaction` or worker pool).
pub struct PiiVaultRepo;

impl PiiVaultRepo {
    pub async fn get_or_create_active_key(
        worker_pool: &PgPool,
        org_id: Uuid,
        kek: &MasterKek,
    ) -> Result<(Uuid, Zeroizing<[u8; 32]>), OpenRagError> {
        let existing = sqlx::query(
            "SELECT id, wrapped_dek
             FROM pii.vault_keys
             WHERE org_id = $1 AND status = 'active'
             ORDER BY key_version DESC
             LIMIT 1",
        )
        .bind(org_id)
        .fetch_optional(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed querying pii.vault_keys: {e}")))?;

        if let Some(row) = existing {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid vault_keys.id: {e}")))?;
            let wrapped: Vec<u8> = row
                .try_get("wrapped_dek")
                .map_err(|e| OpenRagError::Database(format!("invalid wrapped_dek: {e}")))?;
            let dek = VaultCrypto::unwrap_dek(kek, org_id, &wrapped)?;
            return Ok((id, dek));
        }

        let mut tx = worker_pool.begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed beginning vault_keys tx: {e}"))
        })?;

        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(org_id.to_string())
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed acquiring vault_keys advisory lock: {e}"))
            })?;

        let existing_locked = sqlx::query(
            "SELECT id, wrapped_dek
             FROM pii.vault_keys
             WHERE org_id = $1 AND status = 'active'
             ORDER BY key_version DESC
             LIMIT 1",
        )
        .bind(org_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed re-querying pii.vault_keys: {e}")))?;

        if let Some(row) = existing_locked {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid vault_keys.id: {e}")))?;
            let wrapped: Vec<u8> = row
                .try_get("wrapped_dek")
                .map_err(|e| OpenRagError::Database(format!("invalid wrapped_dek: {e}")))?;
            let dek = VaultCrypto::unwrap_dek(kek, org_id, &wrapped)?;
            tx.commit().await.map_err(|e| {
                OpenRagError::Database(format!("failed committing vault_keys tx: {e}"))
            })?;
            return Ok((id, dek));
        }

        let next_ver: i32 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(key_version), 0) + 1 FROM pii.vault_keys WHERE org_id = $1",
        )
        .bind(org_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed computing next key_version: {e}")))?;

        let dek = VaultCrypto::generate_dek()?;
        let wrapped = VaultCrypto::wrap_dek(kek, org_id, &dek)?;
        let inserted_id: Option<Uuid> = sqlx::query_scalar(
            "INSERT INTO pii.vault_keys (org_id, key_version, wrapped_dek, algorithm, status)
             VALUES ($1, $2, $3, 'AES-256-GCM', 'active')
             ON CONFLICT (org_id, key_version) DO NOTHING
             RETURNING id",
        )
        .bind(org_id)
        .bind(next_ver)
        .bind(&wrapped)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed inserting pii.vault_keys: {e}")))?;

        let (key_id, final_dek) = if let Some(id) = inserted_id {
            (id, dek)
        } else {
            let row = sqlx::query(
                "SELECT id, wrapped_dek
                 FROM pii.vault_keys
                 WHERE org_id = $1 AND status = 'active'
                 ORDER BY key_version DESC
                 LIMIT 1",
            )
            .bind(org_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| OpenRagError::Database(format!("failed fetching active vault key: {e}")))?;
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid vault_keys.id: {e}")))?;
            let wrapped: Vec<u8> = row
                .try_get("wrapped_dek")
                .map_err(|e| OpenRagError::Database(format!("invalid wrapped_dek: {e}")))?;
            let unwrapped = VaultCrypto::unwrap_dek(kek, org_id, &wrapped)?;
            (id, unwrapped)
        };

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed committing vault_keys tx: {e}"))
        })?;

        Ok((key_id, final_dek))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn store_vault_entry(
        worker_pool: &PgPool,
        org_id: Uuid,
        vault_key_id: Uuid,
        token: &str,
        entity_type: &str,
        ciphertext: &[u8],
        nonce: &[u8],
        detector_source: &str,
        detector_trace: &serde_json::Value,
    ) -> Result<Uuid, OpenRagError> {
        let entry_id: Uuid = sqlx::query_scalar(
            "INSERT INTO pii.vault_entries (
                org_id, vault_key_id, token, entity_type, ciphertext, nonce,
                detector_source, detector_trace
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             ON CONFLICT (org_id, token) DO UPDATE SET
                vault_key_id = EXCLUDED.vault_key_id,
                entity_type = EXCLUDED.entity_type,
                ciphertext = EXCLUDED.ciphertext,
                nonce = EXCLUDED.nonce,
                detector_source = EXCLUDED.detector_source,
                detector_trace = EXCLUDED.detector_trace,
                purged_at = NULL
             RETURNING id",
        )
        .bind(org_id)
        .bind(vault_key_id)
        .bind(token)
        .bind(entity_type)
        .bind(ciphertext)
        .bind(nonce)
        .bind(detector_source)
        .bind(detector_trace)
        .fetch_one(worker_pool)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("failed inserting pii.vault_entries: {e}"))
        })?;
        Ok(entry_id)
    }

    pub async fn record_chunk_spans(
        worker_pool: &PgPool,
        org_id: Uuid,
        chunk_id: Uuid,
        page_id: Option<Uuid>,
        spans: &[StoredChunkSpanInput],
    ) -> Result<(), OpenRagError> {
        for s in spans {
            let bbox_arr: Option<Vec<i32>> = s.bbox_norm_1000.map(|b| b.to_array().to_vec());
            sqlx::query(
                "INSERT INTO pii.chunk_spans (
                    org_id, chunk_id, page_id, vault_entry_id, entity_type,
                    start_offset, end_offset, bbox_norm_1000
                 ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            )
            .bind(org_id)
            .bind(chunk_id)
            .bind(page_id)
            .bind(s.vault_entry_id)
            .bind(&s.entity_type)
            .bind(s.start_offset)
            .bind(s.end_offset)
            .bind(bbox_arr)
            .execute(worker_pool)
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed inserting pii.chunk_spans: {e}"))
            })?;
        }
        Ok(())
    }

    pub async fn fetch_vault_entry_for_unmask(
        worker_pool: &PgPool,
        org_id: Uuid,
        token: &str,
    ) -> Result<Option<VaultEntryRow>, OpenRagError> {
        let row = sqlx::query(
            "SELECT ve.id, ve.org_id, ve.vault_key_id, ve.token, ve.entity_type,
                    ve.ciphertext, ve.nonce, ve.detector_source, ve.purged_at,
                    vk.wrapped_dek
             FROM pii.vault_entries ve
             JOIN pii.vault_keys vk
               ON vk.id = ve.vault_key_id AND vk.org_id = ve.org_id
             WHERE ve.org_id = $1 AND ve.token = $2",
        )
        .bind(org_id)
        .bind(token)
        .fetch_optional(worker_pool)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("failed fetching pii.vault_entries: {e}"))
        })?;

        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(VaultEntryRow {
            id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
            org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
            vault_key_id: row
                .try_get("vault_key_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            token: row.try_get("token").map_err(|e| OpenRagError::Database(e.to_string()))?,
            entity_type: row
                .try_get("entity_type")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            ciphertext: row
                .try_get("ciphertext")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            nonce: row.try_get("nonce").map_err(|e| OpenRagError::Database(e.to_string()))?,
            detector_source: row
                .try_get("detector_source")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            purged_at: row
                .try_get("purged_at")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            wrapped_dek: row
                .try_get("wrapped_dek")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn write_audit_log_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        principal_id: Uuid,
        action: &str,
        resource_arn: Option<&str>,
        vault_entry_id: Option<Uuid>,
        detector_source: Option<&str>,
        allowed: bool,
        reason: Option<&str>,
    ) -> Result<Uuid, OpenRagError> {
        let id: Uuid = sqlx::query_scalar(
            "INSERT INTO pii.audit_logs (
                org_id, principal_id, action, resource_arn, vault_entry_id,
                detector_source, allowed, reason
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             RETURNING id",
        )
        .bind(org_id)
        .bind(principal_id)
        .bind(action)
        .bind(resource_arn)
        .bind(vault_entry_id)
        .bind(detector_source)
        .bind(allowed)
        .bind(reason)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed writing pii.audit_logs: {e}")))?;
        Ok(id)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn write_audit_log_worker(
        worker_pool: &PgPool,
        org_id: Uuid,
        principal_id: Option<Uuid>,
        action: &str,
        resource_arn: Option<&str>,
        vault_entry_id: Option<Uuid>,
        detector_source: Option<&str>,
        allowed: bool,
        reason: Option<&str>,
    ) -> Result<Uuid, OpenRagError> {
        let id: Uuid = sqlx::query_scalar(
            "INSERT INTO pii.audit_logs (
                org_id, principal_id, action, resource_arn, vault_entry_id,
                detector_source, allowed, reason
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             RETURNING id",
        )
        .bind(org_id)
        .bind(principal_id)
        .bind(action)
        .bind(resource_arn)
        .bind(vault_entry_id)
        .bind(detector_source)
        .bind(allowed)
        .bind(reason)
        .fetch_one(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed writing pii.audit_logs: {e}")))?;
        Ok(id)
    }
}
