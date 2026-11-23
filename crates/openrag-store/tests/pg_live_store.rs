use std::collections::HashMap;
use std::sync::Arc;

use openrag_core::{
    AuthLevel, ExternalMirrorBackend, OpenRagError, PgPineconeEngine, PrincipalContext,
};
use openrag_db::DbPool;
use openrag_store::{
    add_tags_rls, compute_tag_facets_rls, expand_tag_filter_rls, l2_normalize, remove_tags_rls,
    Bm25Query, Bm25Store, DefaultOutboxExecutor, EnqueueOutboxItem, OssBackendKind,
    OssVectorAdapter, OutboxEnqueuer, OutboxOp, OutboxWorker, PgPineconeManager, S3VectorsAdapter,
    StoreMode, TagFilter, TagTable, VectorSearchQuery, VectorStoreAdapter, VectorTable,
    VectorUpsertItem, VECTOR_DIM,
};
use pgvector::Vector;
use serde_json::{json, Map};
use sqlx::Row;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

fn norm_vec(axis: usize, val: f32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut raw = vec![0.02_f32; VECTOR_DIM];
    raw[axis % VECTOR_DIM] = val;
    Ok(l2_normalize(&raw)?)
}

struct TestOrgSeed {
    org_id: Uuid,
    dept_id: Uuid,
    alice: Uuid,
    bucket_id: Uuid,
    alice_ctx: PrincipalContext,
    bob_ctx: PrincipalContext,
}

async fn seed_org(admin: &DbPool) -> Result<TestOrgSeed, Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'StoreTest Org')")
        .bind(org_id)
        .bind(format!("store-{}", org_id.simple()))
        .execute(admin.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Fin', 'fin')",
    )
    .bind(dept_id)
    .bind(org_id)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level) VALUES
         ($1, $3, $4, 'Alice', 8),
         ($2, $3, $5, 'Bob', 8)",
    )
    .bind(alice)
    .bind(bob)
    .bind(org_id)
    .bind(format!("alice-{}@store.test", org_id.simple()))
    .bind(format!("bob-{}@store.test", org_id.simple()))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'member'),
         ($1, $2, $4, 'member')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(alice)
    .bind(bob)
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
         VALUES ($1, $2, $3, $4, 'fin-b', 'Fin Bucket', 'dept', 1, $5)",
    )
    .bind(bucket_id)
    .bind(org_id)
    .bind(dept_id)
    .bind(alice)
    .bind(format!("arn:openrag:store:fin:dept:bucket:L1/{bucket_id}"))
    .execute(admin.inner())
    .await?;

    let mut alice_ctx = PrincipalContext::new(org_id, alice, AuthLevel::new(8)?);
    alice_ctx.dept_ids = vec![dept_id];
    let mut bob_ctx = PrincipalContext::new(org_id, bob, AuthLevel::new(8)?);
    bob_ctx.dept_ids = vec![dept_id];

    Ok(TestOrgSeed {
        org_id,
        dept_id,
        alice,
        bucket_id,
        alice_ctx,
        bob_ctx,
    })
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_pg_pinecone_3_engine_switching_and_rls_invariance() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;

    let s = seed_org(&admin).await?;

    // Seed 2 docs + chunks (1 private, 1 dept)
    let doc_priv = Uuid::new_v4();
    let doc_dept = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $3, $4, $5, $6, 'private', 3, $7, 'priv:k', 'Priv Doc', 'sp1'),
         ($2, $3, $4, $5, $6, 'dept', 3, $8, 'dept:k', 'Dept Doc', 'sd1')",
    )
    .bind(doc_priv)
    .bind(doc_dept)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:private:document:L3/{doc_priv}"))
    .bind(format!("arn:openrag:store:fin:dept:document:L3/{doc_dept}"))
    .execute(admin.inner())
    .await?;

    let c_priv = Uuid::new_v4();
    let c_dept = Uuid::new_v4();
    let q_vec = norm_vec(0, 0.99)?;
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
         VALUES
         ($1, $3, $5, $7, $8, 'private', 3, $9, 'priv:k', 0, '제3조 출장 여비 비밀', '제3조 출장 여비 비밀', 'h1', $11),
         ($2, $4, $6, $7, $8, 'dept', 3, $10, 'dept:k', 0, '제3조(출장 여비 지급 기준) 공무 출장 시 일비를 지급한다.', '제3조(출장 여비 지급 기준) 공무 출장 시 일비를 지급한다.', 'h2', $12)",
    )
    .bind(c_priv)
    .bind(c_dept)
    .bind(s.org_id)
    .bind(s.org_id)
    .bind(doc_priv)
    .bind(doc_dept)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:private:chunk:L3/{c_priv}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_dept}"))
    .bind(Vector::from(q_vec.clone()))
    .bind(Vector::from(norm_vec(0, 0.95)?))
    .execute(admin.inner())
    .await?;

    // Live remote without PINECONE_API_KEY -> ExternalAdapter (503)
    let live_err = PgPineconeManager::set_engine(
        &worker,
        s.org_id,
        VectorTable::CorpusChunks,
        PgPineconeEngine::Remote,
        StoreMode::Live,
        Some(""),
    )
    .await;
    assert!(matches!(live_err, Err(OpenRagError::ExternalAdapter { .. })));

    let mgr = PgPineconeManager::new();
    for engine in [
        PgPineconeEngine::Native,
        PgPineconeEngine::Pgvector,
        PgPineconeEngine::Remote,
    ] {
        let binding = PgPineconeManager::set_engine(
            &worker,
            s.org_id,
            VectorTable::CorpusChunks,
            engine,
            StoreMode::Offline,
            None,
        )
        .await?;
        assert_eq!(binding.pg_pinecone_engine, engine);

        let mut tx_bob = app.begin_rls_tx(&s.bob_ctx).await?;
        let hits = mgr
            .search_rls(
                &mut tx_bob,
                Some(&binding),
                &VectorSearchQuery {
                    org_id: s.org_id,
                    table: VectorTable::CorpusChunks,
                    query_vector: q_vec.clone(),
                    acl_scope_key: None,
                    bucket_id: Some(s.bucket_id),
                    top_k: 5,
                    tag_filter: None,
                },
            )
            .await?;
        tx_bob.rollback().await?;
        // Bob only sees dept chunk, NEVER Alice's private chunk across all 3 engines
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, c_dept);
    }

    // Idempotency check: count(*) == 1 for (org_id, 'corpus.chunks')
    let cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM vectors.index_bindings WHERE org_id = $1 AND table_name = 'corpus.chunks'",
    )
    .bind(s.org_id)
    .fetch_one(worker.inner())
    .await?;
    assert_eq!(cnt, 1);

    // S3 mirror binding populates sv-prod-<sha256(org_id)[:24]>
    let mirror = PgPineconeManager::set_external_mirror(
        &worker,
        s.org_id,
        VectorTable::CorpusChunks,
        ExternalMirrorBackend::S3Vectors,
        json!({}),
    )
    .await?;
    assert_eq!(mirror.external_backend, ExternalMirrorBackend::S3Vectors);
    let expected_bucket = openrag_store::s3_bucket_name("sv-prod", s.org_id);
    assert_eq!(
        mirror.external_config.get("bucket_name").and_then(|v| v.as_str()),
        Some(expected_bucket.as_str())
    );

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_bm25_tags_and_facets_under_rls() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;
    let caps = app.capabilities().await?;

    let doc_priv = Uuid::new_v4();
    let doc_dept = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $3, $4, $5, $6, 'private', 3, $7, 'priv:k', 'Priv', 'sp'),
         ($2, $3, $4, $5, $6, 'dept', 3, $8, 'dept:k', 'Dept', 'sd')",
    )
    .bind(doc_priv)
    .bind(doc_dept)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:private:document:L3/{doc_priv}"))
    .bind(format!("arn:openrag:store:fin:dept:document:L3/{doc_dept}"))
    .execute(admin.inner())
    .await?;

    let c1 = Uuid::new_v4();
    let c2 = Uuid::new_v4();
    let c_priv = Uuid::new_v4();
    let pad = "일반행정문서 ".repeat(40);
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES
         ($1, $4, $6, $8, $9, 'dept', 3, $10, 'dept:k', 0, '감사원특수활동비 단일', '감사원특수활동비 단일', 't1'),
         ($2, $4, $6, $8, $9, 'dept', 3, $11, 'dept:k', 1, $13, $13, 't2'),
         ($3, $4, $5, $8, $9, 'private', 3, $12, 'priv:k', 0, '감사원특수활동비 비밀', '감사원특수활동비 비밀', 't3')",
    )
    .bind(c1)
    .bind(c2)
    .bind(c_priv)
    .bind(s.org_id)
    .bind(doc_priv)
    .bind(doc_dept)
    .bind(s.dept_id)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c1}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c2}"))
    .bind(format!("arn:openrag:store:fin:private:chunk:L3/{c_priv}"))
    .bind(format!("제3조(감사원특수활동비) {pad} 감사원특수활동비"))
    .execute(admin.inner())
    .await?;

    // Concept alias: llmwiki = 세컨드 브레인
    let concept_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.concepts (id, org_id, canonical_key, display_name, category)
         VALUES ($1, $2, 'llmwiki', 'LLM Wiki', 'domain')",
    )
    .bind(concept_id)
    .bind(s.org_id)
    .execute(admin.inner())
    .await?;
    sqlx::query(
        "INSERT INTO wiki.concept_aliases (org_id, concept_id, alias_text, normalized_alias, compact_alias, source)
         VALUES
         ($1, $2, 'llmwiki', 'llmwiki', 'llmwiki', 'manual'),
         ($1, $2, '세컨드 브레인', '세컨드 브레인', '세컨드브레인', 'manual')",
    )
    .bind(s.org_id)
    .bind(concept_id)
    .execute(admin.inner())
    .await?;

    // Alice adds tags to doc_dept (propagating to c1 & c2) and secret tag to doc_priv
    let mut tx_alice = app.begin_rls_tx(&s.alice_ctx).await?;
    add_tags_rls(
        &mut tx_alice,
        s.org_id,
        TagTable::CorpusDocuments,
        doc_dept,
        &["llmwiki".into(), "budget".into()],
        true,
    )
    .await?;
    add_tags_rls(
        &mut tx_alice,
        s.org_id,
        TagTable::CorpusDocuments,
        doc_priv,
        &["secret-only".into()],
        true,
    )
    .await?;
    remove_tags_rls(
        &mut tx_alice,
        s.org_id,
        TagTable::CorpusChunks,
        c1,
        &["budget".into()],
        false,
    )
    .await?;
    tx_alice.commit().await?;

    // Bob searches with alias-expanded tag #세컨드브레인 AND BM25
    let mut tx_bob = app.begin_rls_tx(&s.bob_ctx).await?;
    let bm25_hits = Bm25Store::search_rls(
        &mut tx_bob,
        &caps,
        &Bm25Query {
            org_id: s.org_id,
            table: VectorTable::CorpusChunks,
            query_text: "감사원특수활동비".into(),
            expanded_terms: vec![],
            bucket_id: Some(s.bucket_id),
            tag_filter: Some(TagFilter {
                tags_any: vec!["#세컨드브레인".into()],
                tags_all: vec![],
                tags_exclude: vec![],
                expand_tag_aliases: true,
            }),
            limit: 10,
        },
    )
    .await?;
    assert_eq!(bm25_hits.len(), 2);
    assert_eq!(bm25_hits[0].id, c2);
    assert_eq!(bm25_hits[1].id, c1);

    // Korean clause search
    let clause_hits = Bm25Store::search_rls(
        &mut tx_bob,
        &caps,
        &Bm25Query {
            org_id: s.org_id,
            table: VectorTable::CorpusChunks,
            query_text: "3조".into(),
            expanded_terms: vec![],
            bucket_id: Some(s.bucket_id),
            tag_filter: None,
            limit: 10,
        },
    )
    .await?;
    assert_eq!(clause_hits.len(), 1);
    assert_eq!(clause_hits[0].id, c2);

    // Bob's tag facets MUST NOT include Alice's private 'secret-only' tag
    let facets = compute_tag_facets_rls(
        &mut tx_bob,
        s.org_id,
        &[TagTable::CorpusDocuments, TagTable::CorpusChunks],
        Some(s.bucket_id),
        20,
    )
    .await?;
    assert!(facets.iter().any(|f| f.tag == "llmwiki"));
    assert!(!facets.iter().any(|f| f.tag == "secret-only"));

    let exp = expand_tag_filter_rls(
        &mut tx_bob,
        s.org_id,
        &TagFilter {
            tags_any: vec!["세컨드브레인".into()],
            tags_all: vec![],
            tags_exclude: vec![],
            expand_tag_aliases: true,
        },
    )
    .await?;
    assert!(exp.tags_any.contains(&"llmwiki".to_string()));
    tx_bob.rollback().await?;

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_outbox_priority_noop_watermarks_and_quarantine() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let worker_pool = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let doc_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'dept:k', 'Outbox Doc', 'dsha')",
    )
    .bind(doc_id)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:dept:document:L3/{doc_id}"))
    .execute(admin.inner())
    .await?;

    let c_low = Uuid::new_v4();
    let c_hi = Uuid::new_v4();
    let c_trans = Uuid::new_v4();
    let c_422 = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash)
         VALUES
         ($1, $5, $6, $7, $8, 'dept', 3, $9, 'dept:k', 0, 'a', 'a', 'h1'),
         ($2, $5, $6, $7, $8, 'dept', 3, $10, 'dept:k', 1, 'b', 'b', 'h2'),
         ($3, $5, $6, $7, $8, 'dept', 3, $11, 'dept:k', 2, 'c', 'c', 'h3'),
         ($4, $5, $6, $7, $8, 'dept', 3, $12, 'dept:k', 3, 'd', 'd', 'h4')",
    )
    .bind(c_low)
    .bind(c_hi)
    .bind(c_trans)
    .bind(c_422)
    .bind(s.org_id)
    .bind(doc_id)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_low}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_hi}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_trans}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_422}"))
    .execute(admin.inner())
    .await?;

    // Bob (peer without write access on Alice's dept doc) cannot enqueue outbox
    let mut tx_bob = app.begin_rls_tx(&s.bob_ctx).await?;
    let denied = OutboxEnqueuer::enqueue_rls(
        &mut tx_bob,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_low,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-low".into()),
            priority: 10,
            payload: json!({}),
            idempotency_key: None,
            max_attempts: None,
        },
        false,
    )
    .await;
    assert!(matches!(denied, Err(OpenRagError::Forbidden(_))));
    tx_bob.rollback().await?;

    // Alice enqueues priority 10 first, then priority 1; plus duplicate idempotency_key -> Conflict
    let mut tx_alice = app.begin_rls_tx(&s.alice_ctx).await?;
    let id_low = OutboxEnqueuer::enqueue_rls(
        &mut tx_alice,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_low,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-low".into()),
            priority: 10,
            payload: json!({}),
            idempotency_key: Some("k-low".into()),
            max_attempts: None,
        },
        false,
    )
    .await?;
    assert!(id_low.is_some());

    let dup = OutboxEnqueuer::enqueue_rls(
        &mut tx_alice,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_low,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-low".into()),
            priority: 10,
            payload: json!({}),
            idempotency_key: Some("k-low".into()),
            max_attempts: None,
        },
        false,
    )
    .await;
    assert!(matches!(dup, Err(OpenRagError::Conflict(_))));
    tx_alice.rollback().await?;

    let mut tx_alice = app.begin_rls_tx(&s.alice_ctx).await?;
    let oid_low = OutboxEnqueuer::enqueue_rls(
        &mut tx_alice,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_low,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-low".into()),
            priority: 10,
            payload: json!({}),
            idempotency_key: None,
            max_attempts: None,
        },
        false,
    )
    .await?
    .ok_or("missing oid_low")?;

    let oid_hi = OutboxEnqueuer::enqueue_rls(
        &mut tx_alice,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_hi,
            op: OutboxOp::RedactReindex,
            content_sha256: Some("sha-hi".into()),
            priority: 1,
            payload: json!({}),
            idempotency_key: None,
            max_attempts: None,
        },
        false,
    )
    .await?
    .ok_or("missing oid_hi")?;
    tx_alice.commit().await?;

    let worker = OutboxWorker::default();
    let exec = DefaultOutboxExecutor::new("pg_pinecone");

    // Priority 1 preempts priority 10
    let rep1 = worker.drain_for_org(&worker_pool, Some(s.org_id), 1, &exec).await?;
    assert_eq!(rep1.claimed_ids, vec![oid_hi]);
    assert_eq!(rep1.processed, 1);

    let rep2 = worker.drain_for_org(&worker_pool, Some(s.org_id), 1, &exec).await?;
    assert_eq!(rep2.claimed_ids, vec![oid_low]);
    assert_eq!(rep2.processed, 1);

    // SHA-256 No-Op skip does not increment indexed_count
    let mut tx_noop = app.begin_rls_tx(&s.alice_ctx).await?;
    OutboxEnqueuer::enqueue_rls(
        &mut tx_noop,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_low,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-low".into()),
            priority: 5,
            payload: json!({}),
            idempotency_key: None,
            max_attempts: None,
        },
        false,
    )
    .await?;
    tx_noop.commit().await?;

    let rep_noop = worker.drain_for_org(&worker_pool, Some(s.org_id), 10, &exec).await?;
    assert_eq!(rep_noop.skipped_noop, 1);
    assert_eq!(rep_noop.processed, 0);

    let st_after_noop = worker.status(&worker_pool, Some(s.org_id)).await?;
    let wm = st_after_noop
        .watermarks
        .iter()
        .find(|w| w.target_table == "corpus.chunks")
        .ok_or("missing corpus.chunks watermark")?;
    assert_eq!(wm.indexed_count, 2);

    // 2 transient failures -> still pending; 3rd -> quarantined; 422 -> immediate quarantine
    let mut tx_fail = app.begin_rls_tx(&s.alice_ctx).await?;
    let oid_trans = OutboxEnqueuer::enqueue_rls(
        &mut tx_fail,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_trans,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-t".into()),
            priority: 5,
            payload: json!({ "simulate_transient_error": true }),
            idempotency_key: None,
            max_attempts: Some(3),
        },
        false,
    )
    .await?
    .ok_or("missing oid_trans")?;
    let oid_422 = OutboxEnqueuer::enqueue_rls(
        &mut tx_fail,
        &EnqueueOutboxItem {
            org_id: s.org_id,
            target_table: VectorTable::CorpusChunks,
            target_id: c_422,
            op: OutboxOp::UpsertEmbed,
            content_sha256: Some("sha-422".into()),
            priority: 6,
            payload: json!({ "simulate_error_code": 422 }),
            idempotency_key: None,
            max_attempts: Some(3),
        },
        false,
    )
    .await?
    .ok_or("missing oid_422")?;
    tx_fail.commit().await?;

    for expected_attempts in [1_i32, 2_i32] {
        let _ = worker.drain_for_org(&worker_pool, Some(s.org_id), 1, &exec).await?;
        let row = sqlx::query("SELECT status, attempts FROM indexing.outbox WHERE id = $1")
            .bind(oid_trans)
            .fetch_one(worker_pool.inner())
            .await?;
        assert_eq!(row.try_get::<String, _>("status")?, "pending");
        assert_eq!(row.try_get::<i32, _>("attempts")?, expected_attempts);
    }
    let _ = worker.drain_for_org(&worker_pool, Some(s.org_id), 1, &exec).await?;
    let row_q = sqlx::query("SELECT status, attempts FROM indexing.outbox WHERE id = $1")
        .bind(oid_trans)
        .fetch_one(worker_pool.inner())
        .await?;
    assert_eq!(row_q.try_get::<String, _>("status")?, "quarantined");
    assert_eq!(row_q.try_get::<i32, _>("attempts")?, 3);

    // Next drain claims oid_422 and immediately quarantines on attempt 1
    let _ = worker.drain_for_org(&worker_pool, Some(s.org_id), 1, &exec).await?;
    let row_422 = sqlx::query("SELECT status, attempts FROM indexing.outbox WHERE id = $1")
        .bind(oid_422)
        .fetch_one(worker_pool.inner())
        .await?;
    assert_eq!(row_422.try_get::<String, _>("status")?, "quarantined");
    assert_eq!(row_422.try_get::<i32, _>("attempts")?, 1);

    let st_q = worker.status(&worker_pool, Some(s.org_id)).await?;
    assert_eq!(st_q.unresolved_quarantine.len(), 2);

    // Retry nonexistent quarantine ID -> NotFound
    let not_found = worker
        .retry_quarantine(&worker_pool, Some(Uuid::new_v4()), false, Some(s.org_id))
        .await;
    assert!(matches!(not_found, Err(OpenRagError::NotFound(_))));

    // Retry first quarantine ID -> resolves & re-drives cleanly
    let qid = st_q.unresolved_quarantine[0].id;
    let retried = worker
        .retry_quarantine(&worker_pool, Some(qid), false, Some(s.org_id))
        .await?;
    assert_eq!(retried, 1);

    let rep_rec = worker.drain_for_org(&worker_pool, Some(s.org_id), 10, &exec).await?;
    assert_eq!(rep_rec.processed, 1);

    // Retrying an already-resolved quarantine ID must fail with Conflict and NOT reset the 'done' outbox row
    let resolved_q_err = worker
        .retry_quarantine(&worker_pool, Some(qid), false, Some(s.org_id))
        .await;
    assert!(matches!(resolved_q_err, Err(OpenRagError::Conflict(_))));
    let outbox_id_opt: Option<i64> =
        sqlx::query_scalar("SELECT outbox_id FROM indexing.quarantine WHERE id = $1")
            .bind(qid)
            .fetch_one(worker_pool.inner())
            .await?;
    if let Some(oid) = outbox_id_opt {
        let status_after: String =
            sqlx::query_scalar("SELECT status FROM indexing.outbox WHERE id = $1")
                .bind(oid)
                .fetch_one(worker_pool.inner())
                .await?;
        assert_eq!(status_after, "done");
    }

    let _unused: HashMap<Uuid, String> = HashMap::new();
    let _unused_arc: Arc<()> = Arc::new(());
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn live_s3_and_oss_adapters_search_rls_enforce_tags_aliases_bucket_and_overfetch() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let s = seed_org(&admin).await?;

    let bucket_b = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
         VALUES ($1, $2, $3, $4, 'fin-b2', 'Fin Bucket 2', 'dept', 1, $5)",
    )
    .bind(bucket_b)
    .bind(s.org_id)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:dept:bucket:L1/{bucket_b}"))
    .execute(admin.inner())
    .await?;

    let doc_a = Uuid::new_v4();
    let doc_b = Uuid::new_v4();
    let doc_priv = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
         VALUES
         ($1, $4, $5, $7, $8, 'dept', 3, $9, 'dept:k', 'Doc A', 'sa'),
         ($2, $4, $6, $7, $8, 'dept', 3, $10, 'dept:k', 'Doc B', 'sb'),
         ($3, $4, $5, $7, $8, 'private', 3, $11, 'priv:k', 'Doc Priv', 'sp')",
    )
    .bind(doc_a)
    .bind(doc_b)
    .bind(doc_priv)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(bucket_b)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:dept:document:L3/{doc_a}"))
    .bind(format!("arn:openrag:store:fin:dept:document:L3/{doc_b}"))
    .bind(format!("arn:openrag:store:fin:private:document:L3/{doc_priv}"))
    .execute(admin.inner())
    .await?;

    let c_match = Uuid::new_v4();
    let c_draft = Uuid::new_v4();
    let c_bucket_b = Uuid::new_v4();
    let c_priv = Uuid::new_v4();

    let v_hi = norm_vec(0, 0.99)?;
    let v_mid = norm_vec(0, 0.95)?;
    let v_match = norm_vec(0, 0.70)?;

    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, tags, embedding)
         VALUES
         ($1, $5, $6, $9, $10, 'dept', 3, $11, 'dept:k', 0, 'match', 'match', 'hm', ARRAY['llmwiki']::text[], $15),
         ($2, $5, $6, $9, $10, 'dept', 3, $12, 'dept:k', 1, 'draft', 'draft', 'hd', ARRAY['llmwiki','draft']::text[], $16),
         ($3, $5, $7, $9, $10, 'dept', 3, $13, 'dept:k', 0, 'other_bucket', 'other_bucket', 'hb', ARRAY['llmwiki']::text[], $16),
         ($4, $5, $8, $9, $10, 'private', 3, $14, 'priv:k', 0, 'priv', 'priv', 'hp', ARRAY['llmwiki']::text[], $16)",
    )
    .bind(c_match)
    .bind(c_draft)
    .bind(c_bucket_b)
    .bind(c_priv)
    .bind(s.org_id)
    .bind(doc_a)
    .bind(doc_b)
    .bind(doc_priv)
    .bind(s.dept_id)
    .bind(s.alice)
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_match}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_draft}"))
    .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{c_bucket_b}"))
    .bind(format!("arn:openrag:store:fin:private:chunk:L3/{c_priv}"))
    .bind(Vector::from(v_match.clone()))
    .bind(Vector::from(v_hi.clone()))
    .execute(admin.inner())
    .await?;

    // Concept alias: llmwiki = 세컨드 브레인
    let concept_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO wiki.concepts (id, org_id, canonical_key, display_name, category)
         VALUES ($1, $2, 'llmwiki', 'LLM Wiki', 'domain')",
    )
    .bind(concept_id)
    .bind(s.org_id)
    .execute(admin.inner())
    .await?;
    sqlx::query(
        "INSERT INTO wiki.concept_aliases (org_id, concept_id, alias_text, normalized_alias, compact_alias, source)
         VALUES
         ($1, $2, 'llmwiki', 'llmwiki', 'llmwiki', 'manual'),
         ($1, $2, '세컨드 브레인', '세컨드 브레인', '세컨드브레인', 'manual')",
    )
    .bind(s.org_id)
    .bind(concept_id)
    .execute(admin.inner())
    .await?;

    // Also add 5 higher-scoring decoy chunks in bucket_b to verify overfetch keeps top_k full
    let mut upsert_items = vec![
        VectorUpsertItem {
            id: c_match,
            document_id: Some(doc_a),
            acl_scope_key: "dept:k".into(),
            vector: v_match.clone(),
            text: Some("match".into()),
            tags: vec!["llmwiki".into()],
            metadata: Map::new(),
        },
        VectorUpsertItem {
            id: c_draft,
            document_id: Some(doc_a),
            acl_scope_key: "dept:k".into(),
            vector: v_hi.clone(),
            text: Some("draft".into()),
            tags: vec!["llmwiki".into(), "draft".into()],
            metadata: Map::new(),
        },
        VectorUpsertItem {
            id: c_bucket_b,
            document_id: Some(doc_b),
            acl_scope_key: "dept:k".into(),
            vector: v_hi.clone(),
            text: Some("other_bucket".into()),
            tags: vec!["llmwiki".into()],
            metadata: Map::new(),
        },
        VectorUpsertItem {
            id: c_priv,
            document_id: Some(doc_priv),
            acl_scope_key: "priv:k".into(),
            vector: v_hi.clone(),
            text: Some("priv".into()),
            tags: vec!["llmwiki".into()],
            metadata: Map::new(),
        },
    ];
    for idx in 0..5 {
        let decoy_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO corpus.chunks
             (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, tags, embedding)
             VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'dept:k', $7, 'decoy', 'decoy', $8, ARRAY['llmwiki']::text[], $9)",
        )
        .bind(decoy_id)
        .bind(s.org_id)
        .bind(doc_b)
        .bind(s.dept_id)
        .bind(s.alice)
        .bind(format!("arn:openrag:store:fin:dept:chunk:L3/{decoy_id}"))
        .bind(idx + 10)
        .bind(format!("hd-{idx}"))
        .bind(Vector::from(v_mid.clone()))
        .execute(admin.inner())
        .await?;
        upsert_items.push(VectorUpsertItem {
            id: decoy_id,
            document_id: Some(doc_b),
            acl_scope_key: "dept:k".into(),
            vector: v_mid.clone(),
            text: Some("decoy".into()),
            tags: vec!["llmwiki".into()],
            metadata: Map::new(),
        });
    }

    let s3_adapter = S3VectorsAdapter::new_offline();
    s3_adapter
        .upsert(s.org_id, VectorTable::CorpusChunks, &upsert_items)
        .await?;

    let oss_adapter = OssVectorAdapter::new_offline(OssBackendKind::Qdrant);
    oss_adapter
        .upsert(s.org_id, VectorTable::CorpusChunks, &upsert_items)
        .await?;

    let query = VectorSearchQuery {
        org_id: s.org_id,
        table: VectorTable::CorpusChunks,
        query_vector: v_hi,
        acl_scope_key: None,
        bucket_id: Some(s.bucket_id),
        top_k: 1,
        tag_filter: Some(TagFilter {
            tags_any: vec!["#세컨드브레인".into()],
            tags_all: vec![],
            tags_exclude: vec!["draft".into()],
            expand_tag_aliases: true,
        }),
    };

    let mut tx_bob = app.begin_rls_tx(&s.bob_ctx).await?;
    let s3_hits = s3_adapter.search_rls(&mut tx_bob, &query).await?;
    let oss_hits = oss_adapter.search_rls(&mut tx_bob, &query).await?;
    tx_bob.rollback().await?;

    assert_eq!(s3_hits.len(), 1);
    assert_eq!(s3_hits[0].id, c_match);
    assert_eq!(oss_hits.len(), 1);
    assert_eq!(oss_hits[0].id, c_match);

    Ok(())
}
