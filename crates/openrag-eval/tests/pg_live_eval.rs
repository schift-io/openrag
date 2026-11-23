use std::collections::BTreeMap;
use std::sync::Arc;

use openrag_core::{AuthLevel, FusionMode, OpenRagError, PrincipalContext};
use openrag_db::DbPool;
use openrag_eval::{
    AbQueryInput, AbQueryItem, AbSearchEvaluator, AbSearchRequest, AbVariantConfig,
    AutoRagOptimizer, BenchRunRequest, BenchRunner, BenchTrack, CreateGoldenItemInput,
    CreateGoldenSetInput, EvalStore, FeedbackManager, FeedbackVerdict, GoldenAxis,
    GoldenSetManager, OptimizationStatus, OptimizeRequest, PgEvalStore, PipelinePromoter,
    PromoteFeedbackInput, PromotePipelineInput, ReviewStatus, SubmitFeedbackInput,
};
use openrag_memory::DeterministicTextEmbedder;
use openrag_retrieve::PgHybridRetriever;
use pgvector::Vector;
use sqlx::Row;
use tempfile::TempDir;
use uuid::Uuid;

fn env_url(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|e| format!("missing env var {key}: {e}").into())
}

struct SeededOrg {
    org_id: Uuid,
    dept_id: Uuid,
    slug: String,
    member: Uuid,
    admin: Uuid,
    bucket_id: Uuid,
    member_ctx: PrincipalContext,
    admin_ctx: PrincipalContext,
}

async fn seed_org(admin_pool: &DbPool) -> Result<SeededOrg, Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let member = Uuid::new_v4();
    let admin = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let slug = format!("eval-{}", &org_id.simple().to_string()[..8]);

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Eval Test Org')")
        .bind(org_id)
        .bind(&slug)
        .execute(admin_pool.inner())
        .await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Audit', 'audit')",
    )
    .bind(dept_id)
    .bind(org_id)
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $3, $4, 'Member', 3, false, false),
         ($2, $3, $5, 'Admin', 10, true, true)",
    )
    .bind(member)
    .bind(admin)
    .bind(org_id)
    .bind(format!("member-{}@eval.test", org_id.simple()))
    .bind(format!("admin-{}@eval.test", org_id.simple()))
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'member'),
         ($1, $2, $4, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(member)
    .bind(admin)
    .execute(admin_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES
         ($1, $2, $3, $4, $5, 'Eval Bucket', 'dept', 1, $6)",
    )
    .bind(bucket_id)
    .bind(org_id)
    .bind(dept_id)
    .bind(admin)
    .bind(format!("b-{slug}"))
    .bind(format!("arn:openrag:{slug}:audit:dept:bucket:L1/{bucket_id}"))
    .execute(admin_pool.inner())
    .await?;

    let mut member_ctx = PrincipalContext::new(org_id, member, AuthLevel::new(3)?);
    member_ctx.dept_ids = vec![dept_id];

    let mut admin_ctx = PrincipalContext::new(org_id, admin, AuthLevel::MAX);
    admin_ctx.dept_ids = vec![dept_id];
    admin_ctx.dept_admin_ids = vec![dept_id];
    admin_ctx.is_org_admin = true;
    admin_ctx.can_unmask_pii = true;

    Ok(SeededOrg {
        org_id,
        dept_id,
        slug,
        member,
        admin,
        bucket_id,
        member_ctx,
        admin_ctx,
    })
}

async fn seed_dept_and_private_docs(
    admin_pool: &DbPool,
    s: &SeededOrg,
    topic: &str,
) -> Result<(Uuid, Uuid), Box<dyn std::error::Error>> {
    let doc_dept = Uuid::new_v4();
    let doc_priv = Uuid::new_v4();
    let emb = Vector::from(DeterministicTextEmbedder::embed_sync(topic)?);

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
         VALUES
         ($1, $3, $4, $5, $6, 'dept', 3, $7, 'd:3', 'Dept Benchmark Doc', $9, ARRAY['eval']::text[]),
         ($2, $3, $4, $5, $6, 'private', 9, $8, 'p:9', 'Private L9 Doc', $10, ARRAY['secret']::text[])",
    )
    .bind(doc_dept)
    .bind(doc_priv)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:audit:dept:doc:L3/{doc_dept}", s.slug))
    .bind(format!("arn:openrag:{}:audit:private:doc:L9/{doc_priv}", s.slug))
    .bind(format!("sha-dept-{doc_dept}"))
    .bind(format!("sha-priv-{doc_priv}"))
    .execute(admin_pool.inner())
    .await?;

    let c_dept = Uuid::new_v4();
    let c_priv = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
         VALUES
         ($1, $3, $5, $7, $8, 'dept', 3, $9, 'd:3', 0, $11, $11, $13, $15, ARRAY['eval']::text[]),
         ($2, $3, $6, $7, $8, 'private', 9, $10, 'p:9', 0, $12, $12, $14, $15, ARRAY['secret']::text[])",
    )
    .bind(c_dept)
    .bind(c_priv)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(doc_dept)
    .bind(doc_priv)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:audit:dept:chunk:L3/{c_dept}", s.slug))
    .bind(format!("arn:openrag:{}:audit:private:chunk:L9/{c_priv}", s.slug))
    .bind(format!("{topic} 출장 여비 70000원 공식 기준"))
    .bind(format!("{topic} SECRET_L9_PRIVATE_ONLY"))
    .bind(format!("hc-dept-{c_dept}"))
    .bind(format!("hc-priv-{c_priv}"))
    .bind(emb)
    .execute(admin_pool.inner())
    .await?;

    Ok((doc_dept, doc_priv))
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_feedback_rls_and_idempotent_double_promote() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let s1 = seed_org(&admin).await?;
    let s2 = seed_org(&admin).await?;

    let store = Arc::new(PgEvalStore::new(app, worker.clone()));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let fb_mgr = FeedbackManager::new(Arc::clone(&store));

    // Member submits feedback via app_pool (openrag_app NOBYPASSRLS)
    let fb = fb_mgr
        .submit_feedback_rls(
            &s1.member_ctx,
            SubmitFeedbackInput {
                query_text: "이중 승격 방지 검색어".into(),
                response_text: Some("출장 여비 7만원".into()),
                cited_chunk_ids: Vec::new(),
                expected_doc_ids: Vec::new(),
                verdict: FeedbackVerdict::Down,
                comment: Some("missing latest revision".into()),
            },
        )
        .await?;
    assert!(fb.promoted_golden_item_id.is_none());

    // Member sees own feedback under RLS; org2 admin sees ZERO org1 feedback
    let member_list = fb_mgr.list_feedback_rls(&s1.member_ctx).await?;
    assert!(member_list.iter().any(|f| f.id == fb.id));
    let org2_list = fb_mgr.list_feedback_rls(&s2.admin_ctx).await?;
    assert!(!org2_list.iter().any(|f| f.id == fb.id));

    let gs = gs_mgr
        .create_golden_set(
            s1.org_id,
            CreateGoldenSetInput {
                name: format!("fb-gs-{}", Uuid::new_v4().simple()),
                axis: GoldenAxis::Retrieval,
                description: None,
                version: Some(1),
            },
        )
        .await?;
    let expected_doc = Uuid::new_v4();

    // Non-admin & cross-org admin rejected
    assert!(matches!(
        fb_mgr
            .promote_feedback(
                s1.org_id,
                &s1.member_ctx,
                fb.id,
                PromoteFeedbackInput {
                    golden_set_id: gs.id,
                    axis: Some(GoldenAxis::Retrieval),
                    test_principal_id: None,
                    expected_doc_ids: vec![expected_doc],
                    forbidden_doc_ids: Vec::new(),
                    expected_answer: None,
                    expected_pii_tokens: Vec::new(),
                    expected_edges: None,
                    review_status: None,
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));
    assert!(matches!(
        fb_mgr
            .promote_feedback(
                s1.org_id,
                &s2.admin_ctx,
                fb.id,
                PromoteFeedbackInput {
                    golden_set_id: gs.id,
                    axis: Some(GoldenAxis::Retrieval),
                    test_principal_id: None,
                    expected_doc_ids: vec![expected_doc],
                    forbidden_doc_ids: Vec::new(),
                    expected_answer: None,
                    expected_pii_tokens: Vec::new(),
                    expected_edges: None,
                    review_status: None,
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));

    // Double promote is idempotent and inserts exactly 1 row into eval.golden_items
    let p1 = fb_mgr
        .promote_feedback(
            s1.org_id,
            &s1.admin_ctx,
            fb.id,
            PromoteFeedbackInput {
                golden_set_id: gs.id,
                axis: Some(GoldenAxis::Retrieval),
                test_principal_id: None,
                expected_doc_ids: vec![expected_doc],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;
    assert!(!p1.already_promoted);
    assert_eq!(p1.feedback.promoted_golden_item_id, Some(p1.item.id));

    let p2 = fb_mgr
        .promote_feedback(
            s1.org_id,
            &s1.admin_ctx,
            fb.id,
            PromoteFeedbackInput {
                golden_set_id: gs.id,
                axis: Some(GoldenAxis::Retrieval),
                test_principal_id: None,
                expected_doc_ids: vec![Uuid::new_v4()],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;
    assert!(p2.already_promoted);
    assert_eq!(p2.item.id, p1.item.id);

    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM eval.golden_items WHERE golden_set_id = $1")
            .bind(gs.id)
            .fetch_one(worker.inner())
            .await?;
    assert_eq!(count, 1);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_four_axis_golden_set_review_status_and_three_track_bench(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let s = seed_org(&admin).await?;
    let topic = "출장여비다축평가";
    let (doc_dept, doc_priv) = seed_dept_and_private_docs(&admin, &s, topic).await?;

    let store = Arc::new(PgEvalStore::new(app.clone(), worker.clone()));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let retriever = PgHybridRetriever::new_pg(app);

    let gs = gs_mgr
        .create_golden_set(
            s.org_id,
            CreateGoldenSetInput {
                name: format!("4axis-{}", Uuid::new_v4().simple()),
                axis: GoldenAxis::Multi,
                description: Some("4-axis live PG bench".into()),
                version: Some(1),
            },
        )
        .await?;

    let it_ret = gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: topic.into(),
                test_principal_id: Some(s.member),
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: vec![doc_priv],
                expected_answer: Some("70000원 공식 기준".into()),
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: Some(ReviewStatus::Pending),
            },
        )
        .await?;
    let it_ret = gs_mgr
        .update_review_status(s.org_id, it_ret.id, ReviewStatus::Approved)
        .await?;
    assert_eq!(it_ret.review_status, ReviewStatus::Approved);

    let _it_pii = gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pii,
                query_text: format!("{topic} 880101-1234567"),
                test_principal_id: Some(s.member),
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: vec!["880101-1234567".into()],
                expected_edges: None,
                review_status: Some(ReviewStatus::Approved),
            },
        )
        .await?;

    let _it_pararel = gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pararel,
                query_text: topic.into(),
                test_principal_id: Some(s.member),
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: Some(serde_json::json!([{"src": topic, "dst": "70000원"}])),
                review_status: Some(ReviewStatus::Approved),
            },
        )
        .await?;

    let it_rej = gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Ocr,
                query_text: "거절될 OCR 항목".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: Some(ReviewStatus::Rejected),
            },
        )
        .await?;

    let summary = BenchRunner::run_all_tracks(
        store.as_ref(),
        &retriever,
        &s.admin_ctx,
        &BenchRunRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(s.bucket_id),
            ..BenchRunRequest::default()
        },
        None,
    )
    .await?;

    assert_eq!(summary.runs.len(), 3);
    assert!(summary.security_gates.pass);
    assert!(summary.runs.iter().all(|r| r.security_gate_pass));

    // Verify PG GENERATED ALWAYS security_gate_pass and rejected item exclusion directly in SQL
    let rej_cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM eval.bench_item_results WHERE golden_item_id = $1",
    )
    .bind(it_rej.id)
    .fetch_one(worker.inner())
    .await?;
    assert_eq!(rej_cnt, 0);

    let ok_cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM eval.bench_item_results WHERE golden_item_id = $1",
    )
    .bind(it_ret.id)
    .fetch_one(worker.inner())
    .await?;
    assert_eq!(ok_cnt, 3);

    // Verify GENERATED ALWAYS flips to false when pii_leak_count > 0
    let mut bad_pipeline = openrag_eval::default_pipeline_config();
    bad_pipeline.pii.capsule_enabled = false;
    let bad_summary = BenchRunner::run_all_tracks(
        store.as_ref(),
        &retriever,
        &s.admin_ctx,
        &BenchRunRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(s.bucket_id),
            ..BenchRunRequest::default()
        },
        Some(&bad_pipeline),
    )
    .await?;
    let t3 = bad_summary
        .runs
        .iter()
        .find(|r| r.track == BenchTrack::Track3SecurityOcr)
        .ok_or("missing track3_security_ocr")?;
    assert!(t3.pii_leak_count >= 1);
    assert!(!t3.security_gate_pass);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_ab_search_with_golden_set_and_forbidden_gate(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let s = seed_org(&admin).await?;
    let topic = "에이비검색실시간검증";
    let (doc_dept, _doc_priv) = seed_dept_and_private_docs(&admin, &s, topic).await?;

    let store = Arc::new(PgEvalStore::new(app.clone(), worker));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let retriever = PgHybridRetriever::new_pg(app);

    let gs = gs_mgr
        .create_golden_set(
            s.org_id,
            CreateGoldenSetInput {
                name: format!("ab-pg-{}", Uuid::new_v4().simple()),
                axis: GoldenAxis::Retrieval,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: topic.into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;

    let mut variants = BTreeMap::new();
    variants.insert(
        "a".into(),
        AbVariantConfig {
            fusion: Some(FusionMode::Rrf),
            alpha: Some(0.2),
            ..AbVariantConfig::default()
        },
    );
    variants.insert(
        "b".into(),
        AbVariantConfig {
            fusion: Some(FusionMode::Convex),
            alpha: Some(0.5),
            ..AbVariantConfig::default()
        },
    );

    let clean = AbSearchEvaluator::evaluate(
        &retriever,
        Some(store.as_ref()),
        &s.member_ctx,
        &AbSearchRequest {
            bucket_id: Some(s.bucket_id),
            golden_set_id: Some(gs.id),
            k: Some(5),
            queries: vec![AbQueryInput::QueryText(topic.into())],
            variants: variants.clone(),
        },
        None,
    )
    .await?;
    assert!(clean.security_gates.pass);
    assert!(clean.winner.is_some());
    assert!(clean.variants["a"].mrr > 0.0);

    // Forbidden doc returned to admin trips arn_violation_count
    let gate_trip = AbSearchEvaluator::evaluate(
        &retriever,
        None,
        &s.admin_ctx,
        &AbSearchRequest {
            bucket_id: Some(s.bucket_id),
            golden_set_id: None,
            k: Some(5),
            queries: vec![AbQueryInput::Inline(AbQueryItem {
                query: topic.into(),
                relevant_ids: Vec::new(),
                forbidden_ids: vec![doc_dept],
                expected_pii_tokens: Vec::new(),
            })],
            variants,
        },
        None,
    )
    .await?;
    assert!(!gate_trip.security_gates.pass);
    assert!(gate_trip.security_gates.arn_violation_count >= 1);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_autorag_optimizer_security_gates_and_pipeline_promoter(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let s = seed_org(&admin).await?;
    let topic = "오토라그최적화실시간";
    let (doc_dept, _doc_priv) = seed_dept_and_private_docs(&admin, &s, topic).await?;

    let store = Arc::new(PgEvalStore::new(app.clone(), worker.clone()));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let promoter = PipelinePromoter::new(Arc::clone(&store));
    let retriever = PgHybridRetriever::new_pg(app);

    let gs_name = format!("opt-pg-{}", Uuid::new_v4().simple());
    let gs = gs_mgr
        .create_golden_set(
            s.org_id,
            CreateGoldenSetInput {
                name: gs_name.clone(),
                axis: GoldenAxis::Multi,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: topic.into(),
                test_principal_id: Some(s.member),
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pii,
                query_text: format!("{topic} 900101-1234567"),
                test_principal_id: Some(s.member),
                expected_doc_ids: vec![doc_dept],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: vec!["900101-1234567".into()],
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;

    let tmp = TempDir::new()?;
    let best_path = tmp.path().join("best_pg.yaml");
    let clean_space = format!(
        r#"
version: 1
golden_set: {gs_name}
strategy: greedy_node
nodes:
  retrieve.fusion: [rrf, convex]
  retrieve.alpha: [0.2, 0.5]
"#
    );
    let report = AutoRagOptimizer::optimize(
        store.as_ref(),
        &retriever,
        &s.admin_ctx,
        &OptimizeRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(s.bucket_id),
            search_space_yaml: clean_space,
            export_best_path: Some(best_path.to_string_lossy().into_owned()),
            ..OptimizeRequest::default()
        },
    )
    .await?;
    assert_eq!(report.trial.status, OptimizationStatus::Completed);
    assert!(report.trial.security_gate_pass);
    assert!(best_path.is_file());

    // Blocked security gate (grid with pii.capsule_enabled: [false]) refuses export
    let blocked_path = tmp.path().join("blocked_pg.yaml");
    let blocked_space = format!(
        r#"
version: 1
golden_set: {gs_name}
strategy: grid
nodes:
  pii.capsule_enabled: [false]
  retrieve.fusion: [rrf]
"#
    );
    let blocked_res = AutoRagOptimizer::optimize(
        store.as_ref(),
        &retriever,
        &s.admin_ctx,
        &OptimizeRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(s.bucket_id),
            search_space_yaml: blocked_space,
            export_best_path: Some(blocked_path.to_string_lossy().into_owned()),
            ..OptimizeRequest::default()
        },
    )
    .await;
    assert!(matches!(blocked_res, Err(OpenRagError::Forbidden(_))));
    assert!(!blocked_path.exists());

    let trials = store.list_optimization_trials(s.org_id, Some(gs.id)).await?;
    assert!(trials
        .iter()
        .any(|t| t.status == OptimizationStatus::BlockedSecurityGate && !t.security_gate_pass));

    // Promote clean exported YAML & bind corpus.buckets.pipeline_config_id
    let best_yaml = std::fs::read_to_string(&best_path)?;
    let bucket_id_str = s.bucket_id.to_string();
    let promoted = promoter
        .promote_from_yaml(
            s.org_id,
            &s.admin_ctx,
            &format!("prod-{}", Uuid::new_v4().simple()),
            &best_yaml,
            Some(&bucket_id_str),
        )
        .await?;
    assert!(promoted.is_promoted);

    let bound_cfg_id: Option<Uuid> =
        sqlx::query_scalar("SELECT pipeline_config_id FROM corpus.buckets WHERE id = $1")
            .bind(s.bucket_id)
            .fetch_one(worker.inner())
            .await?;
    assert_eq!(bound_cfg_id, Some(promoted.id));

    // Refuse promotion when capsule_enabled=false OR when bench_run failed security gate
    let unsafe_yaml = best_yaml.replace("capsule_enabled: true", "capsule_enabled: false");
    let unsafe_reg = promoter
        .register_config(
            s.org_id,
            &format!("unsafe-{}", Uuid::new_v4().simple()),
            &unsafe_yaml,
            Some(s.admin),
        )
        .await?;
    assert!(matches!(
        promoter
            .promote_config(
                s.org_id,
                &s.admin_ctx,
                PromotePipelineInput {
                    pipeline_config_id: Some(unsafe_reg.id),
                    ..PromotePipelineInput::default()
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));
    let still_false: bool =
        sqlx::query_scalar("SELECT is_promoted FROM eval.pipeline_configs WHERE id = $1")
            .bind(unsafe_reg.id)
            .fetch_one(worker.inner())
            .await?;
    assert!(!still_false);

    // Register clean config, attach failed bench_run, verify promotion blocked
    let clean_reg = promoter
        .register_config(
            s.org_id,
            &format!("clean-gated-{}", Uuid::new_v4().simple()),
            &best_yaml,
            Some(s.admin),
        )
        .await?;
    let fail_row = sqlx::query(
        "INSERT INTO eval.bench_runs (org_id, golden_set_id, pipeline_config_id, track, pii_leak_count, arn_violation_count)
         VALUES ($1, $2, $3, 'track3_security_ocr', 0, 1)
         RETURNING id, security_gate_pass",
    )
    .bind(s.org_id)
    .bind(gs.id)
    .bind(clean_reg.id)
    .fetch_one(worker.inner())
    .await?;
    let fail_run_id: Uuid = fail_row.try_get("id")?;
    let fail_gate: bool = fail_row.try_get("security_gate_pass")?;
    assert!(!fail_gate);

    assert!(matches!(
        promoter
            .promote_config(
                s.org_id,
                &s.admin_ctx,
                PromotePipelineInput {
                    pipeline_config_id: Some(clean_reg.id),
                    bench_run_id: Some(fail_run_id),
                    ..PromotePipelineInput::default()
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_org_scoped_leaderboard_and_poc_report() -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let s1 = seed_org(&admin).await?;
    let s2 = seed_org(&admin).await?;
    let topic = "리더보드조직격리";
    let (doc1, _) = seed_dept_and_private_docs(&admin, &s1, topic).await?;

    let store = Arc::new(PgEvalStore::new(app.clone(), worker));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let retriever = PgHybridRetriever::new_pg(app);

    let gs1 = gs_mgr
        .create_golden_set(
            s1.org_id,
            CreateGoldenSetInput {
                name: format!("lb1-{}", Uuid::new_v4().simple()),
                axis: GoldenAxis::Retrieval,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            s1.org_id,
            gs1.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: topic.into(),
                test_principal_id: Some(s1.member),
                expected_doc_ids: vec![doc1],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;

    BenchRunner::run_all_tracks(
        store.as_ref(),
        &retriever,
        &s1.admin_ctx,
        &BenchRunRequest {
            golden_set_id: Some(gs1.id),
            bucket_id: Some(s1.bucket_id),
            ..BenchRunRequest::default()
        },
        None,
    )
    .await?;

    let lb1 = BenchRunner::leaderboard(store.as_ref(), s1.org_id).await?;
    let lb2 = BenchRunner::leaderboard(store.as_ref(), s2.org_id).await?;
    assert_eq!(lb1.len(), 3);
    assert!(lb2.is_empty(), "org2 leaderboard must not see org1 runs");

    let rep1 = BenchRunner::generate_poc_report(store.as_ref(), s1.org_id, Some(gs1.id)).await?;
    assert!(rep1.overall_security_gate_pass);
    assert!(rep1.markdown.contains(&gs1.name));

    let rep2 = BenchRunner::generate_poc_report(store.as_ref(), s2.org_id, None).await?;
    assert!(rep2.entries.is_empty());

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL 17 instance via scripts/test-pg-local.sh"]
async fn pg_remediation_s15_ab_pair_and_multitrack_promoter_gates(
) -> Result<(), Box<dyn std::error::Error>> {
    let admin = DbPool::connect(&env_url("ADMIN_DATABASE_URL")?, 4).await?;
    let app = DbPool::connect(&env_url("DATABASE_URL")?, 4).await?;
    let worker = DbPool::connect(&env_url("WORKER_DATABASE_URL")?, 4).await?;

    let s = seed_org(&admin).await?;
    let doc_id = Uuid::new_v4();
    let chunk_id = Uuid::new_v4();
    let doc_text = "AutoRAG 보안 게이트 검증용 본문 S15";
    let emb = Vector::from(DeterministicTextEmbedder::embed_sync(doc_text)?);

    sqlx::query(
        "INSERT INTO corpus.documents
         (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 'S15 Doc', $7, ARRAY['eval']::text[])",
    )
    .bind(doc_id)
    .bind(s.org_id)
    .bind(s.bucket_id)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:audit:dept:doc:L3/{doc_id}", s.slug))
    .bind(format!("sha-s15-{doc_id}"))
    .execute(admin.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunks
         (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding, tags)
         VALUES ($1, $2, $3, $4, $5, 'dept', 3, $6, 'd:3', 0, $7, $7, $8, $9, ARRAY['eval']::text[])",
    )
    .bind(chunk_id)
    .bind(s.org_id)
    .bind(doc_id)
    .bind(s.dept_id)
    .bind(s.admin)
    .bind(format!("arn:openrag:{}:audit:dept:chunk:L3/{chunk_id}", s.slug))
    .bind(doc_text)
    .bind(format!("hc-s15-{chunk_id}"))
    .bind(emb)
    .execute(admin.inner())
    .await?;

    let store = Arc::new(PgEvalStore::new(app.clone(), worker.clone()));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let promoter = PipelinePromoter::new(Arc::clone(&store));
    let retriever = PgHybridRetriever::new_pg(app);

    let gs_name = format!("s15-pg-{}", Uuid::new_v4().simple());
    let gs = gs_mgr
        .create_golden_set(
            s.org_id,
            CreateGoldenSetInput {
                name: gs_name.clone(),
                axis: GoldenAxis::Multi,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            s.org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pii,
                query_text: "S15 pii 점검".into(),
                test_principal_id: Some(s.member),
                expected_doc_ids: vec![doc_id],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;

    // 1. Track3SecurityOcr fails on tokenless PII item when capsule_enabled = false
    let mut bad_pipeline = openrag_eval::default_pipeline_config();
    bad_pipeline.pii.capsule_enabled = false;
    let bench_summary = BenchRunner::run_all_tracks(
        store.as_ref(),
        &retriever,
        &s.admin_ctx,
        &BenchRunRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(s.bucket_id),
            ..BenchRunRequest::default()
        },
        Some(&bad_pipeline),
    )
    .await?;
    let t3 = bench_summary
        .runs
        .iter()
        .find(|r| r.track == BenchTrack::Track3SecurityOcr)
        .ok_or("missing track3_security_ocr")?;
    assert!(t3.pii_leak_count >= 1);
    assert!(!t3.security_gate_pass);

    // 2. Optimizer fails closed across greedy_node, grid, ab_pair on tokenless S15 pii.capsule_enabled: [false]
    let tmp = TempDir::new()?;
    for (idx, strat) in ["greedy_node", "grid", "ab_pair"].iter().enumerate() {
        let export_path = tmp.path().join(format!("pg_s15_blocked_{idx}.yaml"));
        let yaml = format!(
            r#"
version: 1
golden_set: {gs_name}
strategy: {strat}
security_gates:
  require_zero_pii_leak: true
  require_zero_arn_violation: true
nodes:
  pii.capsule_enabled: [false]
  retrieve.fusion: [rrf]
"#
        );
        let res = AutoRagOptimizer::optimize(
            store.as_ref(),
            &retriever,
            &s.admin_ctx,
            &OptimizeRequest {
                golden_set_id: Some(gs.id),
                bucket_id: Some(s.bucket_id),
                search_space_yaml: yaml,
                require_zero_pii_leak: Some(true),
                require_zero_arn_violation: Some(true),
                export_best_path: Some(export_path.to_string_lossy().into_owned()),
                ..OptimizeRequest::default()
            },
        )
        .await;
        assert!(matches!(res, Err(OpenRagError::Forbidden(_))));
        assert!(!export_path.exists());
    }

    // 3. ab_pair fails closed on iam.bypass_rls_test_only: [true]
    let ab_arn_path = tmp.path().join("pg_ab_arn_blocked.yaml");
    let ab_arn_yaml = format!(
        r#"
version: 1
golden_set: {gs_name}
strategy: ab_pair
nodes:
  iam.bypass_rls_test_only: [true]
  retrieve.fusion: [rrf]
"#
    );
    let ab_arn_res = AutoRagOptimizer::optimize(
        store.as_ref(),
        &retriever,
        &s.admin_ctx,
        &OptimizeRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(s.bucket_id),
            search_space_yaml: ab_arn_yaml,
            export_best_path: Some(ab_arn_path.to_string_lossy().into_owned()),
            ..OptimizeRequest::default()
        },
    )
    .await;
    assert!(matches!(ab_arn_res, Err(OpenRagError::Forbidden(_))));
    assert!(!ab_arn_path.exists());

    // 4. PipelinePromoter: failed track3_security_ocr followed by passing track1_retrieval MUST block promotion
    let clean_yaml = include_str!("../../../configs/pipeline.default.yaml");
    let cfg_a = promoter
        .register_config(
            s.org_id,
            &format!("mt-a-{}", Uuid::new_v4().simple()),
            clean_yaml,
            Some(s.admin),
        )
        .await?;
    let cfg_b = promoter
        .register_config(
            s.org_id,
            &format!("mt-b-{}", Uuid::new_v4().simple()),
            clean_yaml,
            Some(s.admin),
        )
        .await?;

    sqlx::query(
        "INSERT INTO eval.bench_runs (org_id, golden_set_id, pipeline_config_id, track, pii_leak_count, arn_violation_count)
         VALUES ($1, $2, $3, 'track3_security_ocr', 1, 0)",
    )
    .bind(s.org_id)
    .bind(gs.id)
    .bind(cfg_a.id)
    .execute(worker.inner())
    .await?;

    let pass_t1_id: Uuid = sqlx::query_scalar(
        "INSERT INTO eval.bench_runs (org_id, golden_set_id, pipeline_config_id, track, pii_leak_count, arn_violation_count)
         VALUES ($1, $2, $3, 'track1_retrieval', 0, 0)
         RETURNING id",
    )
    .bind(s.org_id)
    .bind(gs.id)
    .bind(cfg_a.id)
    .fetch_one(worker.inner())
    .await?;

    assert!(matches!(
        promoter
            .promote_config(
                s.org_id,
                &s.admin_ctx,
                PromotePipelineInput {
                    pipeline_config_id: Some(cfg_a.id),
                    bench_run_id: None,
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));
    let is_promoted: bool =
        sqlx::query_scalar("SELECT is_promoted FROM eval.pipeline_configs WHERE id = $1")
            .bind(cfg_a.id)
            .fetch_one(worker.inner())
            .await?;
    assert!(!is_promoted);

    // Mismatched bench_run.pipeline_config_id rejected for cfg_b
    assert!(matches!(
        promoter
            .promote_config(
                s.org_id,
                &s.admin_ctx,
                PromotePipelineInput {
                    pipeline_config_id: Some(cfg_b.id),
                    bench_run_id: Some(pass_t1_id),
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));

    // thumbnail.dim != 256 rejected
    let bad_thumb_cfg = store
        .register_pipeline_config(
            s.org_id,
            &openrag_eval::RegisterPipelineInput {
                name: format!("bad-thumb-{}", Uuid::new_v4().simple()),
                profile: openrag_eval::PipelineProfileKind::Standard,
                pipeline_yaml: clean_yaml.replace("dim: 256", "dim: 128"),
                config_json: serde_json::json!({
                    "pii": { "capsule_enabled": true },
                    "store": { "dim": 256 },
                    "thumbnail": { "dim": 256 }
                }),
                created_by: Some(s.admin),
            },
        )
        .await?;
    assert!(matches!(
        promoter
            .promote_config(
                s.org_id,
                &s.admin_ctx,
                PromotePipelineInput {
                    pipeline_config_id: Some(bad_thumb_cfg.id),
                    bench_run_id: None,
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));

    Ok(())
}
