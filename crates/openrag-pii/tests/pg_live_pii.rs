use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use openrag_core::{
    AccessScope, AuthLevel, BBoxNorm1000, GuardAction, GuardRuleConfig, GuardRuleKind,
    GuardSurface, OpenRagError, PrincipalContext,
};
use openrag_db::DbPool;
use openrag_pii::{
    Capsule, CascadeRedactor, DetectorProfileInput, GuardEngine,
    GuardPolicyRuntime, GuardRepo, GuardRuleRuntime, LocalThumbnailStore, MasterKek, PiiEngine,
    PiiProfileRepo, PiiVaultRepo, RedactionJobRequest, RedactionMode, RedactionTargetScope,
    StoredChunkSpanInput,
};
use pgvector::Vector;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

fn normalized_256d_seed(seed: f32) -> Vector {
    let mut v = vec![0.01f32; 256];
    v[0] = seed;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    Vector::from(v)
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_pii_profile_vault_guard_and_cascade_redaction() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;

    let org_id = Uuid::new_v4();
    let org_slug = format!("org-pii-{}", org_id.simple());
    let dept_id = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let admin_id = Uuid::new_v4();
    let member_id = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'PII Test Org')")
        .bind(org_id)
        .bind(&org_slug)
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Security', 'sec')",
    )
    .bind(dept_id)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $3, $4, 'OrgAdmin', 9, true, true),
         ($2, $3, $5, 'Member', 5, false, false)",
    )
    .bind(admin_id)
    .bind(member_id)
    .bind(org_id)
    .bind(format!("admin-{}@pii.test", org_id.simple()))
    .bind(format!("member-{}@pii.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'dept_admin'),
         ($1, $2, $4, 'member')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(member_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, name, slug, access_scope, min_auth_level, arn)
         VALUES ($1, $2, $3, $4, 'Sec Bucket', 'sec-bucket', 'dept', 1, $5)",
    )
    .bind(bucket_id)
    .bind(org_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(format!("arn:openrag:{org_slug}:sec:dept:bucket:L1/{bucket_id}"))
    .execute(admin.inner())
    .await?;

    // 1. F14 PiiProfileRepo CRUD
    let profile = PiiProfileRepo::upsert_profile(
        worker.inner(),
        org_id,
        &DetectorProfileInput {
            name: "default".into(),
            use_schift_v7: true,
            use_marker_ko_pii: true,
            use_schift_v6_org: false,
            custom_dict_terms: vec!["PROJECT-ALPHA-99".into()],
            allowlist_terms: vec!["02-120".into()],
            is_default: true,
        },
    )
    .await?;
    assert!(profile.use_marker_ko_pii);
    let loaded_prof = PiiProfileRepo::get_active_profile(worker.inner(), org_id)
        .await?
        .ok_or("missing active detector profile")?;
    assert_eq!(loaded_prof.id, profile.id);

    // 2. Seed document, page (with real WebP thumbnail), chunk, memory entry + patch, wiki page + revision
    let tmp_dir = tempfile::tempdir()?;
    let thumb_path_1 = tmp_dir.path().join("page-001.webp");
    let thumb_path_2 = tmp_dir.path().join("page-002.webp");

    let white_img = RgbaImage::from_pixel(400, 400, Rgba([255, 255, 255, 255]));
    let mut webp_cur = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(white_img).write_to(&mut webp_cur, ImageFormat::WebP)?;
    let orig_webp_bytes = webp_cur.into_inner();
    std::fs::write(&thumb_path_1, &orig_webp_bytes)?;
    std::fs::write(&thumb_path_2, &orig_webp_bytes)?;

    let doc_id = Uuid::new_v4();
    let page_1_id = Uuid::new_v4();
    let page_2_id = Uuid::new_v4();
    let chunk_id = Uuid::new_v4();
    let mem_id = Uuid::new_v4();
    let wiki_id = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, sensitivity)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'k', 'Secret Doc', 'sha1', 'private')",
    )
    .bind(doc_id)
    .bind(org_id)
    .bind(bucket_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(format!("arn:openrag:{org_slug}:sec:dept:document:L1/{doc_id}"))
    .execute(worker.inner())
    .await?;

    for (pid, pno, uri) in [
        (page_1_id, 1, thumb_path_1.to_string_lossy().into_owned()),
        (page_2_id, 2, thumb_path_2.to_string_lossy().into_owned()),
    ] {
        sqlx::query(
            "INSERT INTO corpus.pages (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, raw_markdown, redacted_markdown, thumb_uri, content_sha256)
             VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'k', $7, 'SECRET_NAME 900101-1234567', 'SECRET_NAME 900101-1234567', $8, 'sha_p')",
        )
        .bind(pid)
        .bind(org_id)
        .bind(doc_id)
        .bind(dept_id)
        .bind(admin_id)
        .bind(format!("arn:openrag:{org_slug}:sec:dept:page:L1/{pid}"))
        .bind(pno)
        .bind(uri)
        .execute(worker.inner())
        .await?;
    }

    // F15 Seal chunk PII into pii.vault_entries & pii.chunk_spans
    let kek = MasterKek::from_bytes([0x55u8; 32]);
    let pii_engine = PiiEngine::from_profile(&profile);
    let raw_chunk = "성명:홍길동 주민번호 900101-1234567";
    let spans = pii_engine.detect(raw_chunk);
    let sealed = Capsule::mask_and_seal_pg(worker.inner(), raw_chunk, &spans, org_id, &kek).await?;
    assert!(!sealed.masked_text.contains("900101-1234567"));

    sqlx::query(
        "INSERT INTO corpus.chunks (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, bbox_norm_1000, text_hash, embedding)
         VALUES ($1, $2, $3, $4, $5, $6, 'dept', 1, $7, 'k', 0, $8, $8, ARRAY[100,100,700,400], 'shac', $9)",
    )
    .bind(chunk_id)
    .bind(org_id)
    .bind(doc_id)
    .bind(page_1_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(format!("arn:openrag:{org_slug}:sec:dept:chunk:L1/{chunk_id}"))
    .bind(&sealed.masked_text)
    .bind(normalized_256d_seed(1.0))
    .execute(worker.inner())
    .await?;

    let span_inputs: Vec<StoredChunkSpanInput> = sealed
        .tokens
        .iter()
        .map(|t| StoredChunkSpanInput {
            vault_entry_id: t.vault_entry_id,
            entity_type: t.entity_type.clone(),
            start_offset: t.start_offset as i32,
            end_offset: t.end_offset as i32,
            bbox_norm_1000: BBoxNorm1000::new(100, 100, 700, 400).ok(),
        })
        .collect();
    PiiVaultRepo::record_chunk_spans(worker.inner(), org_id, chunk_id, Some(page_1_id), &span_inputs)
        .await?;

    let mut admin_ctx = PrincipalContext::new(org_id, admin_id, AuthLevel::new(9)?);
    admin_ctx.can_unmask_pii = true;
    admin_ctx.is_org_admin = true;
    admin_ctx.dept_ids = vec![dept_id];
    admin_ctx.dept_admin_ids = vec![dept_id];

    let mut member_ctx = PrincipalContext::new(org_id, member_id, AuthLevel::new(5)?);
    member_ctx.dept_ids = vec![dept_id];

    let denied_unmask = Capsule::unmask_pg(
        worker.inner(),
        &sealed.masked_text,
        &member_ctx,
        None,
        None,
        &kek,
    )
    .await?;
    assert!(!denied_unmask.allowed);

    let ok_unmask = Capsule::unmask_pg(
        worker.inner(),
        &sealed.masked_text,
        &admin_ctx,
        None,
        None,
        &kek,
    )
    .await?;
    assert!(ok_unmask.allowed);
    assert_eq!(ok_unmask.text, raw_chunk);

    // 3. F16 GuardRepo + invalid regex rejection + guard_events RLS (zero raw secret)
    let gp = GuardRepo::get_or_create_active_policy(worker.inner(), org_id, "default").await?;
    let invalid_rule = GuardRuleConfig {
        kind: GuardRuleKind::Regex,
        pattern_or_category: "[unclosed(".into(),
        action: GuardAction::Block,
        surfaces: vec![GuardSurface::InboundQuery],
        replacement_text: None,
        enabled: true,
        priority: 10,
    };
    assert!(matches!(
        GuardRepo::create_rule(worker.inner(), org_id, gp.id, &invalid_rule).await,
        Err(OpenRagError::Validation(_))
    ));
    let valid_rule = GuardRepo::create_rule(
        worker.inner(),
        org_id,
        gp.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "PROJECT-ORION".into(),
            action: GuardAction::Block,
            surfaces: vec![GuardSurface::InboundQuery],
            replacement_text: None,
            enabled: true,
            priority: 10,
        },
    )
    .await?;

    let rt_policy = GuardPolicyRuntime {
        id: Some(gp.id),
        org_id,
        master_enabled: true,
        default_action: GuardAction::Mask,
        allowlist_terms: Vec::new(),
        semantic_threshold: None,
        rules: vec![GuardRuleRuntime {
            id: Some(valid_rule.id),
            kind: valid_rule.kind,
            pattern_or_category: valid_rule.pattern_or_category.clone(),
            action: valid_rule.action,
            surfaces: valid_rule.surfaces.clone(),
            replacement_text: valid_rule.replacement_text.clone(),
            enabled: true,
            priority: valid_rule.priority,
        }],
    };
    let secret_query = "Reveal PROJECT-ORION codename";
    let g_dec = GuardEngine::evaluate(
        secret_query,
        GuardSurface::InboundQuery,
        &rt_policy,
        &pii_engine,
        None,
        None,
    )?;
    assert_eq!(g_dec.action, GuardAction::Block);

    let mut rls_tx = app.begin_rls_tx(&member_ctx).await?;
    let event_id = GuardRepo::log_guard_event_rls(
        &mut rls_tx,
        org_id,
        Some(gp.id),
        member_id,
        GuardSurface::InboundQuery,
        secret_query,
        &g_dec,
    )
    .await?;
    rls_tx.commit().await?;

    let ev_meta: serde_json::Value =
        sqlx::query_scalar("SELECT metadata FROM pii.guard_events WHERE id = $1")
            .bind(event_id)
            .fetch_one(worker.inner())
            .await?;
    assert!(!ev_meta.to_string().contains("PROJECT-ORION"));

    // 4. Seed linked memory + patch + wiki + revision for CascadeRedactor
    sqlx::query(
        "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, title, effective_text, bm25_text, content_sha256, source_document_id, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'k', 'fact', 'Mem', 'Memo with 900101-1234567', 'Memo with 900101-1234567', 'sham', $7, $8)",
    )
    .bind(mem_id)
    .bind(org_id)
    .bind(bucket_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(format!("arn:openrag:{org_slug}:sec:dept:memory:L1/{mem_id}"))
    .bind(doc_id)
    .bind(normalized_256d_seed(2.0))
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO memory.patches (org_id, memory_entry_id, patch_op, prev_version, new_version, prev_text, new_text, actor_type, actor_id)
         VALUES ($1, $2, 'amend', 1, 2, 'Old secret 900101-1234567', 'Memo with 900101-1234567', 'human', 'admin')",
    )
    .bind(org_id)
    .bind(mem_id)
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'k', 'sec-wiki', 'Sec Wiki', 'Wiki secret 900101-1234567', 'Wiki secret', 'shaw', $7)",
    )
    .bind(wiki_id)
    .bind(org_id)
    .bind(bucket_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(format!("arn:openrag:{org_slug}:sec:dept:wiki:L1/{wiki_id}"))
    .bind(normalized_256d_seed(3.0))
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.page_sources (org_id, page_id, source_type, source_id)
         VALUES ($1, $2, 'document', $3)",
    )
    .bind(org_id)
    .bind(wiki_id)
    .bind(doc_id)
    .execute(worker.inner())
    .await?;

    // 5. F17 Non-admin member forbidden from CascadeRedactor
    let thumb_store = LocalThumbnailStore::new();
    let forbidden_job = CascadeRedactor::run_job(
        worker.inner(),
        &member_ctx,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::Document,
            target_ref: serde_json::json!({ "document_id": doc_id }),
            redaction_mode: RedactionMode::MaskKeepVault,
            redact_thumbnails: true,
            reason: "unauthorized attempt".into(),
        },
        &thumb_store,
        Some(&kek),
    )
    .await;
    assert!(matches!(forbidden_job, Err(OpenRagError::Forbidden(_))));

    // 6. F17 Run with redact_thumbnails=false first on page_2 via chunk? Run on Document with redact_thumbnails=false?
    // Test MaskKeepVault with redact_thumbnails=false leaves the thumbnails untouched
    let no_thumb_res = CascadeRedactor::run_job(
        worker.inner(),
        &admin_ctx,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::Document,
            target_ref: serde_json::json!({ "document_id": doc_id }),
            redaction_mode: RedactionMode::MaskKeepVault,
            redact_thumbnails: false,
            reason: "text only redaction".into(),
        },
        &thumb_store,
        Some(&kek),
    )
    .await?;
    assert_eq!(no_thumb_res.affected_counts.thumbnails_masked, 0);
    let p2_redacted_uri: Option<String> =
        sqlx::query_scalar("SELECT redacted_thumb_uri FROM corpus.pages WHERE id = $1")
            .bind(page_2_id)
            .fetch_one(worker.inner())
            .await?;
    assert!(p2_redacted_uri.is_none());

    // 7. F17 HardPurge + redact_thumbnails=true masks thumbnails, purges vault & memory.patches.prev_text
    let purge_res = CascadeRedactor::run_job(
        worker.inner(),
        &admin_ctx,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::Document,
            target_ref: serde_json::json!({
                "document_id": doc_id,
                "bbox_norm_1000": [100, 100, 700, 400]
            }),
            redaction_mode: RedactionMode::HardPurge,
            redact_thumbnails: true,
            reason: "GDPR / PIPA hard purge".into(),
        },
        &thumb_store,
        Some(&kek),
    )
    .await?;

    assert_eq!(purge_res.affected_counts.thumbnails_masked, 2);
    assert!(purge_res.affected_counts.vault_entries_purged >= 1);
    assert!(purge_res.affected_counts.memory_patches_cleared >= 1);
    // chunk, memory entry and wiki page (pages carry no vectors, so nothing is enqueued for them)
    assert!(purge_res.affected_counts.outbox_enqueued >= 3);

    // Check page_1 redacted_thumb_uri points at the masked image
    let red_uri: String =
        sqlx::query_scalar("SELECT redacted_thumb_uri FROM corpus.pages WHERE id = $1")
            .bind(page_1_id)
            .fetch_one(worker.inner())
            .await?;
    let masked_file_bytes = std::fs::read(&red_uri)?;
    let decoded = image::load_from_memory(&masked_file_bytes)?.to_rgba8();
    let px = decoded.get_pixel(120, 100);
    assert!(px[0] <= 16 && px[1] <= 16 && px[2] <= 16);

    // Check memory.patches prev_text cleared to NULL
    let non_null_prev: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM memory.patches WHERE memory_entry_id = $1 AND prev_text IS NOT NULL",
    )
    .bind(mem_id)
    .fetch_one(worker.inner())
    .await?;
    assert_eq!(non_null_prev, 0);

    // Check purged vault entry fails unmask
    let purged_unmask = Capsule::unmask_pg(
        worker.inner(),
        &sealed.masked_text,
        &admin_ctx,
        None,
        None,
        &kek,
    )
    .await?;
    assert!(!purged_unmask.allowed);
    assert_eq!(purged_unmask.unmasked_count, 0);

    // Check immutable redaction_events recorded
    let ev_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.redaction_events WHERE redaction_job_id = $1",
    )
    .bind(purge_res.job_id)
    .fetch_one(worker.inner())
    .await?;
    assert!(ev_count >= 5);

    let _ = AccessScope::Dept;
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_cascade_redactor_cross_dept_rejection_and_post_purge_reingest() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let org_id = Uuid::new_v4();
    let org_slug = format!("org-xdept-{}", org_id.simple());
    let dept_a_id = Uuid::new_v4();
    let dept_b_id = Uuid::new_v4();
    let bucket_a_id = Uuid::new_v4();
    let bucket_b_id = Uuid::new_v4();
    let dept_a_admin_id = Uuid::new_v4();
    let dept_b_owner_id = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Cross-Dept Test Org')")
        .bind(org_id)
        .bind(&org_slug)
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $3, 'Dept A', 'dept-a'),
         ($2, $3, 'Dept B', 'dept-b')",
    )
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $3, $4, 'DeptAAdmin', 8, true, false),
         ($2, $3, $5, 'DeptBOwner', 8, true, false)",
    )
    .bind(dept_a_admin_id)
    .bind(dept_b_owner_id)
    .bind(org_id)
    .bind(format!("adm-a-{}@xdept.test", org_id.simple()))
    .bind(format!("own-b-{}@xdept.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $4, 'dept_admin'),
         ($1, $3, $5, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(dept_a_admin_id)
    .bind(dept_b_owner_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, name, slug, access_scope, min_auth_level, arn) VALUES
         ($1, $3, $4, $6, 'Bucket A', 'bucket-a', 'dept', 1, $8),
         ($2, $3, $5, $7, 'Bucket B', 'bucket-b', 'dept', 1, $9)",
    )
    .bind(bucket_a_id)
    .bind(bucket_b_id)
    .bind(org_id)
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(dept_a_admin_id)
    .bind(dept_b_owner_id)
    .bind(format!("arn:openrag:{org_slug}:dept-a:dept:bucket:L1/{bucket_a_id}"))
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:bucket:L1/{bucket_b_id}"))
    .execute(admin.inner())
    .await?;

    // Seed Dept A doc + Dept B chunk, memory, and wiki (also link Dept B memory/wiki to Dept A doc to verify Document cascade isolation)
    let doc_a_id = Uuid::new_v4();
    let doc_b_id = Uuid::new_v4();
    let chunk_b_id = Uuid::new_v4();
    let mem_b_id = Uuid::new_v4();
    let wiki_b_id = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, sensitivity) VALUES
         ($1, $3, $4, $6, $8, 'dept', 1, $10, 'ka', 'Doc A', 'sha_a', 'private'),
         ($2, $3, $5, $7, $9, 'dept', 1, $11, 'kb', 'Doc B', 'sha_b', 'private')",
    )
    .bind(doc_a_id)
    .bind(doc_b_id)
    .bind(org_id)
    .bind(bucket_a_id)
    .bind(bucket_b_id)
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(dept_a_admin_id)
    .bind(dept_b_owner_id)
    .bind(format!("arn:openrag:{org_slug}:dept-a:dept:document:L1/{doc_a_id}"))
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:document:L1/{doc_b_id}"))
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunks (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'kb', 0, 'Dept B chunk text', 'Dept B chunk text', 'shacb', $7)",
    )
    .bind(chunk_b_id)
    .bind(org_id)
    .bind(doc_b_id)
    .bind(dept_b_id)
    .bind(dept_b_owner_id)
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:chunk:L1/{chunk_b_id}"))
    .bind(normalized_256d_seed(1.0))
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, title, effective_text, bm25_text, content_sha256, source_document_id, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'kb', 'fact', 'Mem B', 'Dept B memory text', 'Dept B memory text', 'shamb', $7, $8)",
    )
    .bind(mem_b_id)
    .bind(org_id)
    .bind(bucket_b_id)
    .bind(dept_b_id)
    .bind(dept_b_owner_id)
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:memory:L1/{mem_b_id}"))
    .bind(doc_a_id)
    .bind(normalized_256d_seed(2.0))
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'kb', 'dept-b-wiki', 'Wiki B', 'Dept B wiki text', 'Dept B wiki text', 'shawb', $7)",
    )
    .bind(wiki_b_id)
    .bind(org_id)
    .bind(bucket_b_id)
    .bind(dept_b_id)
    .bind(dept_b_owner_id)
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:wiki:L1/{wiki_b_id}"))
    .bind(normalized_256d_seed(3.0))
    .execute(worker.inner())
    .await?;

    sqlx::query(
        "INSERT INTO wiki.page_sources (org_id, page_id, source_type, source_id)
         VALUES ($1, $2, 'document', $3)",
    )
    .bind(org_id)
    .bind(wiki_b_id)
    .bind(doc_a_id)
    .execute(worker.inner())
    .await?;

    let mut dept_a_admin = PrincipalContext::new(org_id, dept_a_admin_id, AuthLevel::new(8)?);
    dept_a_admin.can_unmask_pii = true;
    dept_a_admin.dept_ids = vec![dept_a_id];
    dept_a_admin.dept_admin_ids = vec![dept_a_id];

    let mut dept_b_admin = PrincipalContext::new(org_id, dept_b_owner_id, AuthLevel::new(8)?);
    dept_b_admin.can_unmask_pii = true;
    dept_b_admin.dept_ids = vec![dept_b_id];
    dept_b_admin.dept_admin_ids = vec![dept_b_id];

    let thumb_store = LocalThumbnailStore::new();
    let kek = MasterKek::from_bytes([0x66u8; 32]);

    // (a) Dept A admin MUST be rejected (Forbidden) on Dept B's Chunk, Memory, and Wiki scopes
    for (scope, ref_json) in [
        (RedactionTargetScope::Chunk, serde_json::json!({ "chunk_id": chunk_b_id })),
        (RedactionTargetScope::Memory, serde_json::json!({ "memory_entry_id": mem_b_id })),
        (RedactionTargetScope::Wiki, serde_json::json!({ "wiki_page_id": wiki_b_id })),
    ] {
        let res = CascadeRedactor::run_job(
            worker.inner(),
            &dept_a_admin,
            RedactionJobRequest {
                org_id,
                target_scope: scope,
                target_ref: ref_json,
                redaction_mode: RedactionMode::HardPurge,
                redact_thumbnails: false,
                reason: "cross-dept test".into(),
            },
            &thumb_store,
                Some(&kek),
        )
        .await;
        assert!(
            matches!(res, Err(OpenRagError::Forbidden(_))),
            "expected Forbidden for {scope:?}, got {res:?}"
        );
    }

    // Also Document cascade by Dept A admin on Doc A MUST NOT mutate Dept B's memory/wiki
    let doc_a_res = CascadeRedactor::run_job(
        worker.inner(),
        &dept_a_admin,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::Document,
            target_ref: serde_json::json!({ "document_id": doc_a_id }),
            redaction_mode: RedactionMode::MaskKeepVault,
            redact_thumbnails: false,
            reason: "dept A doc redaction".into(),
        },
        &thumb_store,
        Some(&kek),
    )
    .await?;
    assert_eq!(doc_a_res.affected_counts.memories, 0);
    assert_eq!(doc_a_res.affected_counts.wiki_pages, 0);

    let chunk_b_text: String =
        sqlx::query_scalar("SELECT redacted_text FROM corpus.chunks WHERE id = $1")
            .bind(chunk_b_id)
            .fetch_one(worker.inner())
            .await?;
    assert_eq!(chunk_b_text, "Dept B chunk text");

    let mem_b_text: String =
        sqlx::query_scalar("SELECT effective_text FROM memory.entries WHERE id = $1")
            .bind(mem_b_id)
            .fetch_one(worker.inner())
            .await?;
    assert_eq!(mem_b_text, "Dept B memory text");

    let wiki_b_text: String =
        sqlx::query_scalar("SELECT body_md FROM wiki.pages WHERE id = $1")
            .bind(wiki_b_id)
            .fetch_one(worker.inner())
            .await?;
    assert_eq!(wiki_b_text, "Dept B wiki text");

    // (b) Concurrent first-time get_or_create_active_key for fresh org does not collide
    let (k1, k2) = tokio::join!(
        PiiVaultRepo::get_or_create_active_key(worker.inner(), org_id, &kek),
        PiiVaultRepo::get_or_create_active_key(worker.inner(), org_id, &kek),
    );
    assert_eq!(k1?.0, k2?.0);

    // (c) Post-HardPurge re-ingest + authorized unmask roundtrip
    let pii_engine = PiiEngine::default();
    let raw_pii = "긴급 연락처 010-5555-7777 확인";
    let spans = pii_engine.detect(raw_pii);
    let sealed1 = Capsule::mask_and_seal_pg(worker.inner(), raw_pii, &spans, org_id, &kek).await?;
    assert!(!sealed1.masked_text.contains("010-5555-7777"));

    let span_inputs: Vec<StoredChunkSpanInput> = sealed1
        .tokens
        .iter()
        .map(|t| StoredChunkSpanInput {
            vault_entry_id: t.vault_entry_id,
            entity_type: t.entity_type.clone(),
            start_offset: t.start_offset as i32,
            end_offset: t.end_offset as i32,
            bbox_norm_1000: None,
        })
        .collect();
    PiiVaultRepo::record_chunk_spans(worker.inner(), org_id, chunk_b_id, None, &span_inputs).await?;

    // HardPurge via Chunk scope by Dept B admin (and kek = None to verify chunk_spans collection too!)
    let purge_res = CascadeRedactor::run_job(
        worker.inner(),
        &dept_b_admin,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::Chunk,
            target_ref: serde_json::json!({ "chunk_id": chunk_b_id }),
            redaction_mode: RedactionMode::HardPurge,
            redact_thumbnails: false,
            reason: "purge before re-ingest".into(),
        },
        &thumb_store,
        None,
    )
    .await?;
    assert!(purge_res.affected_counts.vault_entries_purged >= 1);

    let after_purge_unmask = Capsule::unmask_pg(
        worker.inner(),
        &sealed1.masked_text,
        &dept_b_admin,
        None,
        None,
        &kek,
    )
    .await?;
    assert!(!after_purge_unmask.allowed);
    assert_eq!(after_purge_unmask.unmasked_count, 0);

    // Re-ingest the exact same PII value in the same org -> must refresh ciphertext & clear purged_at
    let sealed2 = Capsule::mask_and_seal_pg(worker.inner(), raw_pii, &spans, org_id, &kek).await?;
    let after_reingest_unmask = Capsule::unmask_pg(
        worker.inner(),
        &sealed2.masked_text,
        &dept_b_admin,
        None,
        None,
        &kek,
    )
    .await?;
    assert!(after_reingest_unmask.allowed);
    assert_eq!(after_reingest_unmask.unmasked_count, sealed2.tokens.len());
    assert_eq!(after_reingest_unmask.text, raw_pii);

    // (d) PatternDept with kek=None still collects vault_entry_ids from pii.chunk_spans
    let chunk_b2_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'kb', 1, 'Dept B pattern target', 'Dept B pattern target', 'shacb2', $7)",
    )
    .bind(chunk_b2_id)
    .bind(org_id)
    .bind(doc_b_id)
    .bind(dept_b_id)
    .bind(dept_b_owner_id)
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:chunk:L1/{chunk_b2_id}"))
    .bind(normalized_256d_seed(1.5))
    .execute(worker.inner())
    .await?;

    let span_inputs_2: Vec<StoredChunkSpanInput> = sealed2
        .tokens
        .iter()
        .map(|t| StoredChunkSpanInput {
            vault_entry_id: t.vault_entry_id,
            entity_type: t.entity_type.clone(),
            start_offset: t.start_offset as i32,
            end_offset: t.end_offset as i32,
            bbox_norm_1000: None,
        })
        .collect();
    PiiVaultRepo::record_chunk_spans(worker.inner(), org_id, chunk_b2_id, None, &span_inputs_2).await?;

    let pat_purge = CascadeRedactor::run_job(
        worker.inner(),
        &dept_b_admin,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::PatternDept,
            target_ref: serde_json::json!({
                "dept_id": dept_b_id,
                "pattern": "Dept B pattern target"
            }),
            redaction_mode: RedactionMode::HardPurge,
            redact_thumbnails: false,
            reason: "pattern_dept chunk_spans purge without kek".into(),
        },
        &thumb_store,
        None,
    )
    .await?;
    assert!(pat_purge.affected_counts.chunks >= 1);
    assert!(pat_purge.affected_counts.vault_entries_purged >= 1);
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_pattern_dept_kek_vault_isolation_and_empty_pattern_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let org_id = Uuid::new_v4();
    let org_slug = format!("org-pdept-{}", org_id.simple());
    let dept_a_id = Uuid::new_v4();
    let dept_b_id = Uuid::new_v4();
    let bucket_a_id = Uuid::new_v4();
    let bucket_b_id = Uuid::new_v4();
    let dept_a_admin_id = Uuid::new_v4();
    let dept_b_admin_id = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'PatternDept Isolation Org')")
        .bind(org_id)
        .bind(&org_slug)
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $3, 'Dept A', 'dept-a'),
         ($2, $3, 'Dept B', 'dept-b')",
    )
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $3, $4, 'DeptAAdmin', 8, true, false),
         ($2, $3, $5, 'DeptBAdmin', 8, true, false)",
    )
    .bind(dept_a_admin_id)
    .bind(dept_b_admin_id)
    .bind(org_id)
    .bind(format!("pdept-a-{}@test.local", org_id.simple()))
    .bind(format!("pdept-b-{}@test.local", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $4, 'dept_admin'),
         ($1, $3, $5, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(dept_a_admin_id)
    .bind(dept_b_admin_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, name, slug, access_scope, min_auth_level, arn) VALUES
         ($1, $3, $4, $6, 'Bucket A', 'bucket-a', 'dept', 1, $8),
         ($2, $3, $5, $7, 'Bucket B', 'bucket-b', 'dept', 1, $9)",
    )
    .bind(bucket_a_id)
    .bind(bucket_b_id)
    .bind(org_id)
    .bind(dept_a_id)
    .bind(dept_b_id)
    .bind(dept_a_admin_id)
    .bind(dept_b_admin_id)
    .bind(format!("arn:openrag:{org_slug}:dept-a:dept:bucket:L1/{bucket_a_id}"))
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:bucket:L1/{bucket_b_id}"))
    .execute(admin.inner())
    .await?;

    let doc_b_id = Uuid::new_v4();
    let chunk_b_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, sensitivity)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'kb', 'Doc B', 'sha_b_only', 'private')",
    )
    .bind(doc_b_id)
    .bind(org_id)
    .bind(bucket_b_id)
    .bind(dept_b_id)
    .bind(dept_b_admin_id)
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:document:L1/{doc_b_id}"))
    .execute(worker.inner())
    .await?;

    let kek = MasterKek::from_bytes([0x77u8; 32]);
    let pii_engine = PiiEngine::default();
    let raw_b_pii = "부서B 기밀 연락처 010-9999-8888";
    let spans = pii_engine.detect(raw_b_pii);
    let sealed_b = Capsule::mask_and_seal_pg(worker.inner(), raw_b_pii, &spans, org_id, &kek).await?;
    assert!(!sealed_b.tokens.is_empty());

    sqlx::query(
        "INSERT INTO corpus.chunks (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
         VALUES ($1, $2, $3, $4, $5, 'dept', 1, $6, 'kb', 0, $7, $7, 'sha_cb', $8)",
    )
    .bind(chunk_b_id)
    .bind(org_id)
    .bind(doc_b_id)
    .bind(dept_b_id)
    .bind(dept_b_admin_id)
    .bind(format!("arn:openrag:{org_slug}:dept-b:dept:chunk:L1/{chunk_b_id}"))
    .bind(&sealed_b.masked_text)
    .bind(normalized_256d_seed(1.0))
    .execute(worker.inner())
    .await?;

    let span_inputs: Vec<StoredChunkSpanInput> = sealed_b
        .tokens
        .iter()
        .map(|t| StoredChunkSpanInput {
            vault_entry_id: t.vault_entry_id,
            entity_type: t.entity_type.clone(),
            start_offset: t.start_offset as i32,
            end_offset: t.end_offset as i32,
            bbox_norm_1000: None,
        })
        .collect();
    PiiVaultRepo::record_chunk_spans(worker.inner(), org_id, chunk_b_id, None, &span_inputs).await?;

    let mut dept_a_admin = PrincipalContext::new(org_id, dept_a_admin_id, AuthLevel::new(8)?);
    dept_a_admin.can_unmask_pii = true;
    dept_a_admin.dept_ids = vec![dept_a_id];
    dept_a_admin.dept_admin_ids = vec![dept_a_id];

    let mut dept_b_admin = PrincipalContext::new(org_id, dept_b_admin_id, AuthLevel::new(8)?);
    dept_b_admin.can_unmask_pii = true;
    dept_b_admin.dept_ids = vec![dept_b_id];
    dept_b_admin.dept_admin_ids = vec![dept_b_id];

    let thumb_store = LocalThumbnailStore::new();

    // (a) Empty pattern MUST be rejected with Validation error
    let empty_pat_res = CascadeRedactor::run_job(
        worker.inner(),
        &dept_a_admin,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::PatternDept,
            target_ref: serde_json::json!({
                "dept_id": dept_a_id,
                "pattern": ""
            }),
            redaction_mode: RedactionMode::HardPurge,
            redact_thumbnails: false,
            reason: "empty pattern should fail".into(),
        },
        &thumb_store,
        Some(&kek),
    )
    .await;
    assert!(
        matches!(empty_pat_res, Err(OpenRagError::Validation(_))),
        "expected Validation error on empty pattern, got {empty_pat_res:?}"
    );

    // (b) PatternDept by Dept A admin with kek = Some(&kek) and HardPurge MUST NOT purge Dept B's exclusive vault entries
    let dept_a_purge = CascadeRedactor::run_job(
        worker.inner(),
        &dept_a_admin,
        RedactionJobRequest {
            org_id,
            target_scope: RedactionTargetScope::PatternDept,
            target_ref: serde_json::json!({
                "dept_id": dept_a_id,
                "pattern": "010-9999-8888"
            }),
            redaction_mode: RedactionMode::HardPurge,
            redact_thumbnails: false,
            reason: "dept A pattern purge must not touch dept B vault".into(),
        },
        &thumb_store,
        Some(&kek),
    )
    .await?;
    assert_eq!(dept_a_purge.affected_counts.vault_entries_purged, 0);

    let purged_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.vault_entries WHERE id = $1 AND purged_at IS NOT NULL",
    )
    .bind(sealed_b.tokens[0].vault_entry_id)
    .fetch_one(worker.inner())
    .await?;
    assert_eq!(purged_count, 0);

    let still_unmaskable = Capsule::unmask_pg(
        worker.inner(),
        &sealed_b.masked_text,
        &dept_b_admin,
        None,
        None,
        &kek,
    )
    .await?;
    assert!(still_unmaskable.allowed);
    assert_eq!(still_unmaskable.text, raw_b_pii);
    Ok(())
}
