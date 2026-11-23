use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use openrag_core::{IamAction, OpenRagError, PrincipalContext};
use openrag_iam::{
    GroupDefinition, GroupMembership, IamEvaluator, IamPolicy, ResourceDescriptor,
    ResourceGroupBinding,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::detector::PiiSpan;
use crate::vault::{MasterKek, PiiVaultRepo, VaultCrypto};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedTokenRecord {
    pub token: String,
    pub entity_type: String,
    pub vault_entry_id: Option<Uuid>,
    pub start_offset: usize,
    pub end_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedText {
    pub masked_text: String,
    pub tokens: Vec<SealedTokenRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnmaskResult {
    pub text: String,
    pub unmasked_count: usize,
    pub denied_count: usize,
    pub allowed: bool,
    pub reason: String,
}

#[derive(Debug, Clone)]
struct InMemoryVaultItem {
    org_id: Uuid,
    entity_type: String,
    wrapped_dek: Vec<u8>,
    ciphertext: Vec<u8>,
    nonce: Vec<u8>,
    purged: bool,
}

/// Pure offline in-memory vault store implementing genuine AES-256-GCM DEK/KEK envelope encryption.
#[derive(Debug, Default)]
pub struct InMemoryVaultStore {
    org_deks: Mutex<HashMap<Uuid, Vec<u8>>>,
    entries: Mutex<HashMap<(Uuid, String), InMemoryVaultItem>>,
}

impl InMemoryVaultStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_create_dek(
        &self,
        org_id: Uuid,
        kek: &MasterKek,
    ) -> Result<(Vec<u8>, zeroize::Zeroizing<[u8; 32]>), OpenRagError> {
        let mut guard = self
            .org_deks
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryVaultStore mutex poisoned".into()))?;
        if let Some(wrapped) = guard.get(&org_id) {
            let dek = VaultCrypto::unwrap_dek(kek, org_id, wrapped)?;
            return Ok((wrapped.clone(), dek));
        }
        let dek = VaultCrypto::generate_dek()?;
        let wrapped = VaultCrypto::wrap_dek(kek, org_id, &dek)?;
        guard.insert(org_id, wrapped.clone());
        Ok((wrapped, dek))
    }

    pub fn seal_entry(
        &self,
        org_id: Uuid,
        token: String,
        entity_type: String,
        wrapped_dek: Vec<u8>,
        ciphertext: Vec<u8>,
        nonce: Vec<u8>,
    ) -> Result<(), OpenRagError> {
        let mut guard = self
            .entries
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryVaultStore mutex poisoned".into()))?;
        guard.insert(
            (org_id, token),
            InMemoryVaultItem {
                org_id,
                entity_type,
                wrapped_dek,
                ciphertext,
                nonce,
                purged: false,
            },
        );
        Ok(())
    }

    pub fn tamper_ciphertext_for_test(
        &self,
        org_id: Uuid,
        token: &str,
    ) -> Result<(), OpenRagError> {
        let mut guard = self
            .entries
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryVaultStore mutex poisoned".into()))?;
        let item = guard
            .get_mut(&(org_id, token.to_string()))
            .ok_or_else(|| OpenRagError::NotFound(format!("token {token} not found")))?;
        if let Some(first) = item.ciphertext.first_mut() {
            *first ^= 0xFF;
        }
        Ok(())
    }

    pub fn purge_entry(&self, org_id: Uuid, token: &str) -> Result<(), OpenRagError> {
        let mut guard = self
            .entries
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryVaultStore mutex poisoned".into()))?;
        let item = guard
            .get_mut(&(org_id, token.to_string()))
            .ok_or_else(|| OpenRagError::NotFound(format!("token {token} not found")))?;
        item.purged = true;
        item.ciphertext.clear();
        item.nonce.clear();
        Ok(())
    }

    fn fetch_entry(&self, org_id: Uuid, token: &str) -> Result<Option<InMemoryVaultItem>, OpenRagError> {
        let guard = self
            .entries
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryVaultStore mutex poisoned".into()))?;
        Ok(guard.get(&(org_id, token.to_string())).cloned())
    }
}

pub struct CapsuleIamCtx<'a> {
    pub resource: &'a ResourceDescriptor,
    pub groups: &'a [GroupDefinition],
    pub memberships: &'a [GroupMembership],
    pub bindings: &'a [ResourceGroupBinding],
    pub policies: &'a [IamPolicy],
}

pub struct Capsule;

impl Capsule {
    pub fn mask_and_seal_in_memory(
        text: &str,
        spans: &[PiiSpan],
        org_id: Uuid,
        kek: &MasterKek,
        store: &InMemoryVaultStore,
    ) -> Result<SealedText, OpenRagError> {
        if spans.is_empty() {
            return Ok(SealedText {
                masked_text: text.to_string(),
                tokens: Vec::new(),
            });
        }

        let (wrapped_dek, dek) = store.get_or_create_dek(org_id, kek)?;
        let mut sorted = spans.to_vec();
        sorted.sort_by_key(|s| s.start_byte);

        let mut masked = String::with_capacity(text.len());
        let mut cursor = 0usize;
        let mut records = Vec::with_capacity(sorted.len());

        for span in sorted {
            if span.start_byte < cursor || span.end_byte > text.len() {
                continue;
            }
            if !text.is_char_boundary(span.start_byte) || !text.is_char_boundary(span.end_byte) {
                return Err(OpenRagError::Internal(
                    "PII span byte offsets do not align with UTF-8 char boundaries".into(),
                ));
            }
            masked.push_str(&text[cursor..span.start_byte]);

            let token = VaultCrypto::make_token(&dek, org_id, span.category, &span.matched_text);
            let (ciphertext, nonce) =
                VaultCrypto::encrypt_value(&dek, org_id, &token, &span.matched_text)?;
            store.seal_entry(
                org_id,
                token.clone(),
                span.category.as_canonical_str().to_string(),
                wrapped_dek.clone(),
                ciphertext,
                nonce,
            )?;

            let tok_start = masked.len();
            masked.push_str(&token);
            let tok_end = masked.len();
            records.push(SealedTokenRecord {
                token,
                entity_type: span.category.as_canonical_str().to_string(),
                vault_entry_id: None,
                start_offset: tok_start,
                end_offset: tok_end,
            });
            cursor = span.end_byte;
        }
        masked.push_str(&text[cursor..]);

        Ok(SealedText {
            masked_text: masked,
            tokens: records,
        })
    }

    pub async fn mask_and_seal_pg(
        worker_pool: &PgPool,
        text: &str,
        spans: &[PiiSpan],
        org_id: Uuid,
        kek: &MasterKek,
    ) -> Result<SealedText, OpenRagError> {
        if spans.is_empty() {
            return Ok(SealedText {
                masked_text: text.to_string(),
                tokens: Vec::new(),
            });
        }

        let (vault_key_id, dek) =
            PiiVaultRepo::get_or_create_active_key(worker_pool, org_id, kek).await?;
        let mut sorted = spans.to_vec();
        sorted.sort_by_key(|s| s.start_byte);

        let mut masked = String::with_capacity(text.len());
        let mut cursor = 0usize;
        let mut records = Vec::with_capacity(sorted.len());

        for span in sorted {
            if span.start_byte < cursor || span.end_byte > text.len() {
                continue;
            }
            if !text.is_char_boundary(span.start_byte) || !text.is_char_boundary(span.end_byte) {
                return Err(OpenRagError::Internal(
                    "PII span byte offsets do not align with UTF-8 char boundaries".into(),
                ));
            }
            masked.push_str(&text[cursor..span.start_byte]);

            let token = VaultCrypto::make_token(&dek, org_id, span.category, &span.matched_text);
            let (ciphertext, nonce) =
                VaultCrypto::encrypt_value(&dek, org_id, &token, &span.matched_text)?;
            let entry_id = PiiVaultRepo::store_vault_entry(
                worker_pool,
                org_id,
                vault_key_id,
                &token,
                span.category.as_canonical_str(),
                &ciphertext,
                &nonce,
                &span.detector_source,
                &span.detector_trace,
            )
            .await?;

            let tok_start = masked.len();
            masked.push_str(&token);
            let tok_end = masked.len();
            records.push(SealedTokenRecord {
                token,
                entity_type: span.category.as_canonical_str().to_string(),
                vault_entry_id: Some(entry_id),
                start_offset: tok_start,
                end_offset: tok_end,
            });
            cursor = span.end_byte;
        }
        masked.push_str(&text[cursor..]);

        Ok(SealedText {
            masked_text: masked,
            tokens: records,
        })
    }

    pub fn authorize_unmask(
        principal: &PrincipalContext,
        iam_ctx: Option<&CapsuleIamCtx<'_>>,
    ) -> (bool, String) {
        if !principal.can_unmask_pii {
            return (false, "principal_cannot_unmask_pii".to_string());
        }
        if let Some(ctx) = iam_ctx {
            let decision = IamEvaluator::evaluate(
                principal,
                ctx.resource,
                IamAction::PiiUnmask,
                ctx.groups,
                ctx.memberships,
                ctx.bindings,
                ctx.policies,
            );
            if !decision.allowed {
                return (false, decision.reason);
            }
        }
        (true, "authorized_pii_unmask".to_string())
    }

    pub fn unmask_in_memory(
        masked_text: &str,
        principal: &PrincipalContext,
        iam_ctx: Option<&CapsuleIamCtx<'_>>,
        kek: &MasterKek,
        store: &InMemoryVaultStore,
    ) -> Result<UnmaskResult, OpenRagError> {
        let (allowed, reason) = Self::authorize_unmask(principal, iam_ctx);
        if !allowed {
            return Ok(UnmaskResult {
                text: masked_text.to_string(),
                unmasked_count: 0,
                denied_count: count_pii_tokens(masked_text),
                allowed: false,
                reason,
            });
        }

        let mut out = String::with_capacity(masked_text.len());
        let mut last = 0usize;
        let mut unmasked_count = 0usize;
        let mut denied_count = 0usize;

        for (start, end, tok) in find_pii_tokens(masked_text) {
            out.push_str(&masked_text[last..start]);
            match store.fetch_entry(principal.org_id, tok)? {
                Some(item) if !item.purged && item.org_id == principal.org_id => {
                    match VaultCrypto::unwrap_dek(kek, item.org_id, &item.wrapped_dek).and_then(
                        |dek| {
                            let _ = &item.entity_type;
                            VaultCrypto::decrypt_value(
                                &dek,
                                item.org_id,
                                tok,
                                &item.ciphertext,
                                &item.nonce,
                            )
                        },
                    ) {
                        Ok(plain) => {
                            out.push_str(&plain);
                            unmasked_count += 1;
                        }
                        Err(_) => {
                            out.push_str(tok);
                            denied_count += 1;
                        }
                    }
                }
                _ => {
                    out.push_str(tok);
                    denied_count += 1;
                }
            }
            last = end;
        }
        out.push_str(&masked_text[last..]);

        Ok(UnmaskResult {
            text: out,
            unmasked_count,
            denied_count,
            allowed: denied_count == 0,
            reason: if denied_count == 0 {
                reason
            } else {
                "partial_or_failed_vault_decrypt".into()
            },
        })
    }

    pub async fn unmask_pg(
        worker_pool: &PgPool,
        masked_text: &str,
        principal: &PrincipalContext,
        iam_ctx: Option<&CapsuleIamCtx<'_>>,
        resource_arn: Option<&str>,
        kek: &MasterKek,
    ) -> Result<UnmaskResult, OpenRagError> {
        let (allowed, reason) = Self::authorize_unmask(principal, iam_ctx);
        if !allowed {
            let _ = PiiVaultRepo::write_audit_log_worker(
                worker_pool,
                principal.org_id,
                Some(principal.principal_id),
                "pii:unmask",
                resource_arn,
                None,
                None,
                false,
                Some(&reason),
            )
            .await;
            return Ok(UnmaskResult {
                text: masked_text.to_string(),
                unmasked_count: 0,
                denied_count: count_pii_tokens(masked_text),
                allowed: false,
                reason,
            });
        }

        let mut out = String::with_capacity(masked_text.len());
        let mut last = 0usize;
        let mut unmasked_count = 0usize;
        let mut denied_count = 0usize;

        for (start, end, tok) in find_pii_tokens(masked_text) {
            out.push_str(&masked_text[last..start]);
            let row =
                PiiVaultRepo::fetch_vault_entry_for_unmask(worker_pool, principal.org_id, tok)
                    .await?;
            match row {
                Some(entry) if entry.purged_at.is_none() => {
                    let decrypted = VaultCrypto::unwrap_dek(kek, entry.org_id, &entry.wrapped_dek)
                        .and_then(|dek| {
                            VaultCrypto::decrypt_value(
                                &dek,
                                entry.org_id,
                                tok,
                                &entry.ciphertext,
                                &entry.nonce,
                            )
                        });
                    match decrypted {
                        Ok(plain) => {
                            out.push_str(&plain);
                            unmasked_count += 1;
                            let _ = PiiVaultRepo::write_audit_log_worker(
                                worker_pool,
                                principal.org_id,
                                Some(principal.principal_id),
                                "pii:unmask",
                                resource_arn,
                                Some(entry.id),
                                Some(&entry.detector_source),
                                true,
                                Some("authorized_pii_unmask"),
                            )
                            .await;
                        }
                        Err(e) => {
                            out.push_str(tok);
                            denied_count += 1;
                            let _ = PiiVaultRepo::write_audit_log_worker(
                                worker_pool,
                                principal.org_id,
                                Some(principal.principal_id),
                                "pii:unmask",
                                resource_arn,
                                Some(entry.id),
                                Some(&entry.detector_source),
                                false,
                                Some(&e.to_string()),
                            )
                            .await;
                        }
                    }
                }
                Some(purged_entry) => {
                    out.push_str(tok);
                    denied_count += 1;
                    let _ = PiiVaultRepo::write_audit_log_worker(
                        worker_pool,
                        principal.org_id,
                        Some(principal.principal_id),
                        "pii:unmask",
                        resource_arn,
                        Some(purged_entry.id),
                        Some(&purged_entry.detector_source),
                        false,
                        Some("vault_entry_purged"),
                    )
                    .await;
                }
                None => {
                    out.push_str(tok);
                    denied_count += 1;
                }
            }
            last = end;
        }
        out.push_str(&masked_text[last..]);

        Ok(UnmaskResult {
            text: out,
            unmasked_count,
            denied_count,
            allowed: denied_count == 0,
            reason: if denied_count == 0 {
                reason
            } else {
                "vault_entry_purged_or_tampered".into()
            },
        })
    }
}

fn find_pii_tokens(text: &str) -> Vec<(usize, usize, &str)> {
    static RE: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"\[PII:[A-Z0-9_]+:[0-9a-fA-F]{4,32}\]").ok());
    let Some(re) = RE.as_ref() else {
        return Vec::new();
    };
    re.find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str()))
        .collect()
}

fn count_pii_tokens(text: &str) -> usize {
    find_pii_tokens(text).len()
}
