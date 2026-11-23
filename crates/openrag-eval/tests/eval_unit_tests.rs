use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::Utc;
use openrag_core::{
    AccessScope, AuthLevel, FusionMode, OpenRagError, OptimizationStrategy, PrincipalContext,
};
use openrag_eval::{
    check_arn_violation, check_pii_leak, default_pipeline_config, mrr, ndcg_at_k,
    normalize_search_space_node_key, parse_pipeline_yaml_tolerant, parse_search_space_yaml,
    recall_at_k, AbQueryInput, AbQueryItem, AbSearchEvaluator, AbSearchRequest, AbVariantConfig,
    AutoRagOptimizer, BenchRunRequest, BenchRunner, BenchTrack, CreateGoldenItemInput,
    CreateGoldenSetInput, EvalStore, FeedbackManager, FeedbackVerdict, GoldenAxis,
    GoldenSetManager, InMemoryEvalStore, OptimizationStatus, OptimizeRequest, PipelinePromoter,
    PromoteFeedbackInput, PromotePipelineInput, ReviewStatus, SubmitFeedbackInput,
};
use openrag_memory::DeterministicTextEmbedder;
use openrag_pii::PiiEngine;
use openrag_retrieve::{
    InMemoryDocRecord, InMemoryHybridRetriever, InMemoryRetrieveBackend, RetrieveSpace,
};
use tempfile::TempDir;
use uuid::Uuid;

fn make_admin_ctx(org_id: Uuid, dept_id: Uuid) -> Result<PrincipalContext, OpenRagError> {
    let mut ctx = PrincipalContext::new(org_id, Uuid::new_v4(), AuthLevel::MAX);
    ctx.dept_ids = vec![dept_id];
    ctx.dept_admin_ids = vec![dept_id];
    ctx.is_org_admin = true;
    ctx.can_unmask_pii = true;
    Ok(ctx)
}

fn make_member_ctx(org_id: Uuid, dept_id: Uuid) -> Result<PrincipalContext, OpenRagError> {
    let mut ctx = PrincipalContext::new(org_id, Uuid::new_v4(), AuthLevel::new(3)?);
    ctx.dept_ids = vec![dept_id];
    Ok(ctx)
}

fn seed_doc_record(
    org_id: Uuid,
    bucket_id: Uuid,
    dept_id: Uuid,
    owner_id: Uuid,
    doc_id: Uuid,
    text: &str,
) -> Result<InMemoryDocRecord, OpenRagError> {
    let emb = DeterministicTextEmbedder::embed_sync(text)?;
    Ok(InMemoryDocRecord {
        id: Uuid::new_v4(),
        org_id,
        bucket_id: Some(bucket_id),
        dept_id,
        owner_principal_id: owner_id,
        access_scope: AccessScope::Dept,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        min_auth_level: 3,
        space: RetrieveSpace::Chunks,
        document_id: Some(doc_id),
        page_id: None,
        title: Some("Test Doc".into()),
        text: text.to_string(),
        tags: vec!["eval".into()],
        document_tags: Some(vec!["eval".into()]),
        revision: 1,
        valid_from: Utc::now(),
        valid_to: None,
        is_archived: false,
        is_superseded: false,
        thumb_uri: None,
        bbox_norm_1000: None,
        acl_scope_key: "dept:3".into(),
        embedding: Some(emb),
    })
}

#[test]
fn metrics_math_exactness_and_edge_cases() {
    let d1 = Uuid::new_v4();
    let d2 = Uuid::new_v4();
    let d3 = Uuid::new_v4();
    let d4 = Uuid::new_v4();

    // Empty relevant / zero k -> 0.0, finite
    assert_eq!(mrr(&[d1, d2], &[]), 0.0);
    assert_eq!(recall_at_k(&[d1, d2], &[], 10), 0.0);
    assert_eq!(recall_at_k(&[d1, d2], &[d1], 0), 0.0);
    assert_eq!(ndcg_at_k(&[d1, d2], &[], 10), 0.0);
    assert_eq!(ndcg_at_k(&[d1, d2], &[d1], 0), 0.0);

    // Duplicate doc IDs across multiple chunks are deduplicated before ranking
    let ranked_with_dups = vec![d4, d4, d2, d1];
    let rel = vec![d2, d1];
    // Deduped order: [d4 (rank 1), d2 (rank 2), d1 (rank 3)]
    assert!((mrr(&ranked_with_dups, &rel) - 0.5).abs() < 1e-12);
    assert!((recall_at_k(&ranked_with_dups, &rel, 1) - 0.0).abs() < 1e-12);
    assert!((recall_at_k(&ranked_with_dups, &rel, 2) - 0.5).abs() < 1e-12);
    assert!((recall_at_k(&ranked_with_dups, &rel, 10) - 1.0).abs() < 1e-12);

    // Exact NDCG@3: hits at rank 2 and rank 3 vs ideal ranks 1 and 2
    let dcg = (1.0 / 3.0_f64.log2()) + (1.0 / 4.0_f64.log2());
    let idcg = (1.0 / 2.0_f64.log2()) + (1.0 / 3.0_f64.log2());
    assert!((ndcg_at_k(&ranked_with_dups, &rel, 3) - (dcg / idcg)).abs() < 1e-12);
    assert!((ndcg_at_k(&[d1, d2, d3], &[d1, d2], 10) - 1.0).abs() < 1e-12);
}

#[test]
fn security_checkers_detect_arn_and_pii_leaks() -> Result<(), Box<dyn std::error::Error>> {
    let d1 = Uuid::new_v4();
    let d2 = Uuid::new_v4();
    assert!(!check_arn_violation(&[d1], &[d2], false));
    assert!(check_arn_violation(&[d1, d2], &[d2], false));
    assert!(check_arn_violation(&[d1], &[], true));

    let mut pipeline = default_pipeline_config();
    let engine = PiiEngine::from_config(&pipeline.pii);

    // Sealed capsule token passes with 0 leak
    assert!(!check_pii_leak(
        "담당자 [PII:RRN_KR:deadbeef] 연락처 [PII:PHONE_KR:11223344]",
        &["880101-1234567".into(), "010-1234-5678".into()],
        &pipeline,
        &engine
    )?);

    // Unmasked PII in egress text triggers leak even with capsule_enabled=true
    assert!(check_pii_leak(
        "주민번호 880101-1234567 노출",
        &["880101-1234567".into()],
        &pipeline,
        &engine
    )?);

    // capsule_enabled=false triggers leak on raw PII
    pipeline.pii.capsule_enabled = false;
    assert!(check_pii_leak(
        "연락처 010-1234-5678",
        &["010-1234-5678".into()],
        &pipeline,
        &engine
    )?);

    Ok(())
}

#[test]
fn search_space_and_pipeline_yaml_tolerant_parsing_and_bva(
) -> Result<(), Box<dyn std::error::Error>> {
    let example_space = include_str!("../../../configs/search_space.example.yaml");
    let parsed_example = parse_search_space_yaml(example_space)?;
    assert_eq!(parsed_example.strategy, OptimizationStrategy::GreedyNode);
    assert_eq!(parsed_example.metric, "ndcg_at_10");
    assert!(parsed_example.nodes.contains_key("retrieve.fusion"));
    assert!(parsed_example.nodes.contains_key("retrieve.alpha"));

    // E2E dotted YAML omitting metric & security_gates
    let e2e_dotted = r#"
version: 1
golden_set: gov_v1
strategy: grid
nodes:
  retrieve.fusion: [rrf, convex]
  retrieve.alpha: [0.2, 0.5]
  pii.capsule_enabled: [true]
  iam.bypass_rls_test_only: [false]
"#;
    let parsed_dotted = parse_search_space_yaml(e2e_dotted)?;
    assert_eq!(parsed_dotted.metric, "ndcg_at_10");
    assert!(parsed_dotted.security_gates.require_zero_pii_leak);
    assert!(parsed_dotted.security_gates.require_zero_arn_violation);
    assert_eq!(
        normalize_search_space_node_key("retrieve_fusion"),
        "retrieve.fusion"
    );

    // Reject empty nodes and unknown strategy
    let empty_nodes = r#"
version: 1
golden_set: gov_v1
strategy: greedy_node
nodes: {}
"#;
    assert!(matches!(
        parse_search_space_yaml(empty_nodes),
        Err(OpenRagError::Validation(_))
    ));

    let bad_strategy = r#"
version: 1
golden_set: gov_v1
strategy: random_walk
nodes:
  retrieve.fusion: [rrf]
"#;
    assert!(matches!(
        parse_search_space_yaml(bad_strategy),
        Err(OpenRagError::Validation(_))
    ));

    // Pipeline default & airgap + FixtureFactory tolerant shape
    let def_yaml = include_str!("../../../configs/pipeline.default.yaml");
    let airgap_yaml = include_str!("../../../configs/pipeline.airgap.yaml");
    assert_eq!(parse_pipeline_yaml_tolerant(def_yaml)?.store.dim, 256);
    assert_eq!(parse_pipeline_yaml_tolerant(airgap_yaml)?.store.dim, 256);

    let fixture_yaml = r#"
version: 1
profile: custom
parse:
  hwp: kordoc
  ocr: schift_ocr
  dewarp: true
  ocr_mode_by_auth_level:
    share_max_level: 3
    private_min_level: 4
    zdr_min_level: 7
thumbnail:
  format: webp
  dim: 256
  max_edge_px: 1280
pii:
  schift_v7: true
  custom_dict: []
  capsule_enabled: true
store:
  pg_pinecone_engine: native
  dim: 256
  bm25: pg_search
retrieve:
  spaces: [chunks, thumb, wiki, memory]
  fusion: rrf
  alpha: 0.3
  rerank: true
  latest_revision_boost: true
  admission_threshold: 0.25
"#;
    let parsed_fixture = parse_pipeline_yaml_tolerant(fixture_yaml)?;
    assert_eq!(parsed_fixture.store.dim, 256);
    assert!((parsed_fixture.retrieve.latest_revision_boost - 0.15).abs() < 1e-6);

    // Reject dim != 256
    let bad_dim = fixture_yaml.replace("dim: 256", "dim: 128");
    assert!(matches!(
        parse_pipeline_yaml_tolerant(&bad_dim),
        Err(OpenRagError::Config(_))
    ));

    Ok(())
}

#[tokio::test]
async fn golden_set_validation_and_idempotent_feedback_promotion(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let other_org = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;
    let member = make_member_ctx(org_id, dept_id)?;
    let other_admin = make_admin_ctx(other_org, Uuid::new_v4())?;

    let store = Arc::new(InMemoryEvalStore::new());
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let fb_mgr = FeedbackManager::new(Arc::clone(&store));

    // Reject empty golden set name
    assert!(matches!(
        gs_mgr
            .create_golden_set(
                org_id,
                CreateGoldenSetInput {
                    name: "   ".into(),
                    axis: GoldenAxis::Retrieval,
                    description: None,
                    version: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));

    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "gov_v1".into(),
                axis: GoldenAxis::Multi,
                description: Some("4-axis eval".into()),
                version: Some(1),
            },
        )
        .await?;

    // Duplicate name in same org -> Conflict
    assert!(matches!(
        gs_mgr
            .create_golden_set(
                org_id,
                CreateGoldenSetInput {
                    name: "gov_v1".into(),
                    axis: GoldenAxis::Retrieval,
                    description: None,
                    version: None,
                },
            )
            .await,
        Err(OpenRagError::Conflict(_))
    ));

    // Item validation: reject Multi, empty query, all-empty expected
    let doc_id = Uuid::new_v4();
    assert!(matches!(
        gs_mgr
            .add_golden_item(
                org_id,
                gs.id,
                CreateGoldenItemInput {
                    axis: GoldenAxis::Multi,
                    query_text: "valid".into(),
                    test_principal_id: None,
                    expected_doc_ids: vec![doc_id],
                    forbidden_doc_ids: Vec::new(),
                    expected_answer: None,
                    expected_pii_tokens: Vec::new(),
                    expected_edges: None,
                    review_status: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));
    assert!(matches!(
        gs_mgr
            .add_golden_item(
                org_id,
                gs.id,
                CreateGoldenItemInput {
                    axis: GoldenAxis::Retrieval,
                    query_text: "   ".into(),
                    test_principal_id: None,
                    expected_doc_ids: vec![doc_id],
                    forbidden_doc_ids: Vec::new(),
                    expected_answer: None,
                    expected_pii_tokens: Vec::new(),
                    expected_edges: None,
                    review_status: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));
    assert!(matches!(
        gs_mgr
            .add_golden_item(
                org_id,
                gs.id,
                CreateGoldenItemInput {
                    axis: GoldenAxis::Retrieval,
                    query_text: "missing targets".into(),
                    test_principal_id: None,
                    expected_doc_ids: Vec::new(),
                    forbidden_doc_ids: Vec::new(),
                    expected_answer: None,
                    expected_pii_tokens: Vec::new(),
                    expected_edges: None,
                    review_status: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));

    // Allow expected_doc_ids=[] when forbidden_doc_ids is non-empty!
    let forb_only = gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: "forbidden leak check".into(),
                test_principal_id: Some(member.principal_id),
                expected_doc_ids: Vec::new(),
                forbidden_doc_ids: vec![doc_id],
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;
    assert_eq!(forb_only.review_status, ReviewStatus::Approved);

    // Submit feedback & test non-admin / cross-org / idempotent double promote
    let fb = fb_mgr
        .submit_feedback_rls(
            &member,
            SubmitFeedbackInput {
                query_text: "출장 여비 규정".into(),
                response_text: Some("70,000원 기준입니다.".into()),
                cited_chunk_ids: vec![doc_id],
                expected_doc_ids: Vec::new(),
                verdict: FeedbackVerdict::WrongEvidence,
                comment: Some("need latest rev".into()),
            },
        )
        .await?;

    // Non-admin blocked
    assert!(matches!(
        fb_mgr
            .promote_feedback(
                org_id,
                &member,
                fb.id,
                PromoteFeedbackInput {
                    golden_set_id: gs.id,
                    axis: Some(GoldenAxis::Retrieval),
                    test_principal_id: None,
                    expected_doc_ids: vec![doc_id],
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

    // Cross-org admin blocked
    assert!(matches!(
        fb_mgr
            .promote_feedback(
                org_id,
                &other_admin,
                fb.id,
                PromoteFeedbackInput {
                    golden_set_id: gs.id,
                    axis: Some(GoldenAxis::Retrieval),
                    test_principal_id: None,
                    expected_doc_ids: vec![doc_id],
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

    let p1 = fb_mgr
        .promote_feedback(
            org_id,
            &admin,
            fb.id,
            PromoteFeedbackInput {
                golden_set_id: gs.id,
                axis: Some(GoldenAxis::Retrieval),
                test_principal_id: None,
                expected_doc_ids: vec![doc_id],
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

    // Second promote to same golden set is idempotent & creates zero duplicates
    let p2 = fb_mgr
        .promote_feedback(
            org_id,
            &admin,
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
    assert_eq!(gs_mgr.list_golden_items(org_id, gs.id, true).await?.len(), 2);

    // Promoting to a DIFFERENT golden set returns Conflict
    let gs2 = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "gov_v2".into(),
                axis: GoldenAxis::Retrieval,
                description: None,
                version: None,
            },
        )
        .await?;
    assert!(matches!(
        fb_mgr
            .promote_feedback(
                org_id,
                &admin,
                fb.id,
                PromoteFeedbackInput {
                    golden_set_id: gs2.id,
                    axis: Some(GoldenAxis::Retrieval),
                    test_principal_id: None,
                    expected_doc_ids: vec![doc_id],
                    forbidden_doc_ids: Vec::new(),
                    expected_answer: None,
                    expected_pii_tokens: Vec::new(),
                    expected_edges: None,
                    review_status: None,
                },
            )
            .await,
        Err(OpenRagError::Conflict(_))
    ));

    Ok(())
}

#[tokio::test]
async fn bench_runner_3_tracks_excludes_rejected_items_and_reports(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;
    let member = make_member_ctx(org_id, dept_id)?;

    let doc_ok = Uuid::new_v4();
    let doc_rej = Uuid::new_v4();
    let doc_forb = Uuid::new_v4();

    let backend = InMemoryRetrieveBackend::new();
    backend
        .insert(seed_doc_record(
            org_id,
            bucket_id,
            dept_id,
            admin.principal_id,
            doc_ok,
            "출장 여비 개정 규정 70000원 지원",
        )?)
        .await;
    // Private L9 doc blocked for L3 member
    let mut priv_rec = seed_doc_record(
        org_id,
        bucket_id,
        dept_id,
        admin.principal_id,
        doc_forb,
        "기밀 프로젝트 출장 문서",
    )?;
    priv_rec.access_scope = AccessScope::Private;
    priv_rec.min_auth_level = 9;
    backend.insert(priv_rec).await;

    let retriever = InMemoryHybridRetriever::new(backend);
    let store = Arc::new(InMemoryEvalStore::new());
    store.insert_principal(member.clone())?;
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));

    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "bench_multi".into(),
                axis: GoldenAxis::Multi,
                description: None,
                version: None,
            },
        )
        .await?;

    let approved = gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: "출장 여비 개정".into(),
                test_principal_id: Some(member.principal_id),
                expected_doc_ids: vec![doc_ok],
                forbidden_doc_ids: vec![doc_forb],
                expected_answer: Some("70000원 지원".into()),
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: Some(ReviewStatus::Approved),
            },
        )
        .await?;

    let pii_item = gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pii,
                query_text: "주민번호 880101-1234567 출장".into(),
                test_principal_id: Some(member.principal_id),
                expected_doc_ids: vec![doc_ok],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: vec!["880101-1234567".into()],
                expected_edges: None,
                review_status: Some(ReviewStatus::Approved),
            },
        )
        .await?;

    let pararel_item = gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pararel,
                query_text: "출장 여비 개정".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_ok],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: Some(serde_json::json!([{"src": "출장", "dst": "70000원"}])),
                review_status: Some(ReviewStatus::Approved),
            },
        )
        .await?;

    let rejected = gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Ocr,
                query_text: "거절된 OCR 항목".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_rej],
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
        &admin,
        &BenchRunRequest {
            golden_set: Some("bench_multi".into()),
            bucket_id: Some(bucket_id),
            ..BenchRunRequest::default()
        },
        None,
    )
    .await?;

    assert_eq!(summary.runs.len(), 3);
    assert!(summary.runs.iter().any(|r| r.track == BenchTrack::Track1Retrieval));
    assert!(summary.runs.iter().any(|r| r.track == BenchTrack::Track2E2eRag));
    assert!(summary.runs.iter().any(|r| r.track == BenchTrack::Track3SecurityOcr));
    assert!(summary.security_gates.pass);

    // Rejected item strictly excluded; 3 approved items evaluated across 3 tracks = 9 results
    assert!(!summary
        .item_results
        .iter()
        .any(|ir| ir.golden_item_id == rejected.id));
    assert_eq!(summary.item_results.len(), 9);
    assert!(summary
        .item_results
        .iter()
        .any(|ir| ir.golden_item_id == approved.id));
    assert!(summary
        .item_results
        .iter()
        .any(|ir| ir.golden_item_id == pii_item.id));
    assert!(summary
        .item_results
        .iter()
        .any(|ir| ir.golden_item_id == pararel_item.id));

    let lb = BenchRunner::leaderboard(store.as_ref(), org_id).await?;
    assert_eq!(lb.len(), 3);
    let report = BenchRunner::generate_poc_report(store.as_ref(), org_id, Some(gs.id)).await?;
    assert!(report.overall_security_gate_pass);
    assert!(report.markdown.contains("bench_multi"));

    Ok(())
}

#[tokio::test]
async fn ab_search_evaluator_inline_golden_ref_and_security_gates(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let bucket_pii_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;

    let doc_rel = Uuid::new_v4();
    let doc_forb = Uuid::new_v4();
    let doc_pii = Uuid::new_v4();

    let backend = InMemoryRetrieveBackend::new();
    backend
        .insert(seed_doc_record(
            org_id,
            bucket_id,
            dept_id,
            admin.principal_id,
            doc_rel,
            "경조사 지원금 기준 안내",
        )?)
        .await;
    backend
        .insert(seed_doc_record(
            org_id,
            bucket_id,
            dept_id,
            admin.principal_id,
            doc_forb,
            "금지 문서 보안 검증",
        )?)
        .await;
    backend
        .insert(seed_doc_record(
            org_id,
            bucket_pii_id,
            dept_id,
            admin.principal_id,
            doc_pii,
            "유출 테스트 880101-1234567 포함",
        )?)
        .await;

    let retriever = InMemoryHybridRetriever::new(backend);
    let store = Arc::new(InMemoryEvalStore::new());
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "ab_gs".into(),
                axis: GoldenAxis::Retrieval,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: "경조사 지원금 기준".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_rel],
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
        "variant_a".into(),
        AbVariantConfig {
            fusion: Some(FusionMode::Rrf),
            alpha: Some(0.2),
            ..AbVariantConfig::default()
        },
    );
    variants.insert(
        "variant_b".into(),
        AbVariantConfig {
            fusion: Some(FusionMode::Convex),
            alpha: Some(0.5),
            ..AbVariantConfig::default()
        },
    );

    // 1. Clean string query resolved via golden_set_id + inline query
    let clean_report = AbSearchEvaluator::evaluate(
        &retriever,
        Some(store.as_ref()),
        &admin,
        &AbSearchRequest {
            bucket_id: Some(bucket_id),
            golden_set_id: Some(gs.id),
            k: Some(5),
            queries: vec![AbQueryInput::QueryText("경조사 지원금 기준".into())],
            variants: variants.clone(),
        },
        None,
    )
    .await?;
    assert!(clean_report.security_gates.pass);
    assert!(clean_report.winner.is_some());
    let json_val = serde_json::to_value(&clean_report)?;
    assert!(json_val.get("variant_a").is_some());
    assert!(json_val.get("variant_b").is_some());
    assert!(json_val["variant_a"]["mrr"].as_f64().unwrap_or(0.0) > 0.0);

    // 2. Forbidden doc hit trips arn_violation_count
    let arn_fail = AbSearchEvaluator::evaluate(
        &retriever,
        None,
        &admin,
        &AbSearchRequest {
            bucket_id: Some(bucket_id),
            golden_set_id: None,
            k: Some(5),
            queries: vec![AbQueryInput::Inline(AbQueryItem {
                query: "금지 문서 보안 검증".into(),
                relevant_ids: Vec::new(),
                forbidden_ids: vec![doc_forb],
                expected_pii_tokens: Vec::new(),
            })],
            variants: variants.clone(),
        },
        None,
    )
    .await?;
    assert!(!arn_fail.security_gates.pass);
    assert!(arn_fail.security_gates.arn_violation_count >= 1);
    assert!(arn_fail.winner.is_none());

    // 3. Unmasked PII in hit text trips pii_leak_count
    let pii_fail = AbSearchEvaluator::evaluate(
        &retriever,
        None,
        &admin,
        &AbSearchRequest {
            bucket_id: Some(bucket_pii_id),
            golden_set_id: None,
            k: Some(5),
            queries: vec![AbQueryInput::Inline(AbQueryItem {
                query: "유출 테스트".into(),
                relevant_ids: vec![doc_pii],
                forbidden_ids: Vec::new(),
                expected_pii_tokens: vec!["880101-1234567".into()],
            })],
            variants,
        },
        None,
    )
    .await?;
    assert!(!pii_fail.security_gates.pass);
    assert!(pii_fail.security_gates.pii_leak_count >= 1);

    Ok(())
}

#[tokio::test]
async fn autorag_optimizer_strategies_and_hard_security_gates(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;

    let doc_rel = Uuid::new_v4();
    let doc_forb = Uuid::new_v4();

    let backend = InMemoryRetrieveBackend::new();
    backend
        .insert(seed_doc_record(
            org_id,
            bucket_id,
            dept_id,
            admin.principal_id,
            doc_rel,
            "자동 최적화 벤치마크 정답 문서",
        )?)
        .await;
    let retriever = InMemoryHybridRetriever::new(backend);

    let store = Arc::new(InMemoryEvalStore::new());
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "opt_gs".into(),
                axis: GoldenAxis::Multi,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Retrieval,
                query_text: "자동 최적화 벤치마크 정답".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_rel],
                forbidden_doc_ids: vec![doc_forb],
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pii,
                query_text: "주민번호 900101-1234567 확인".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_rel],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: vec!["900101-1234567".into()],
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;

    let tmp = TempDir::new()?;

    // 1. GreedyNode, Grid, AbPair all succeed and export clean YAML
    for (idx, strat) in ["greedy_node", "grid", "ab_pair"].iter().enumerate() {
        let export_path = tmp.path().join(format!("best_{idx}.yaml"));
        let yaml = format!(
            r#"
version: 1
golden_set: opt_gs
strategy: {strat}
nodes:
  retrieve.fusion: [rrf, convex]
  retrieve.alpha: [0.2, 0.5]
  retrieve.rerank: [true, false]
"#
        );
        let report = AutoRagOptimizer::optimize(
            store.as_ref(),
            &retriever,
            &admin,
            &OptimizeRequest {
                golden_set_id: Some(gs.id),
                bucket_id: Some(bucket_id),
                search_space_yaml: yaml,
                export_best_path: Some(export_path.to_string_lossy().into_owned()),
                ..OptimizeRequest::default()
            },
        )
        .await?;

        assert_eq!(report.trial.status, OptimizationStatus::Completed);
        assert!(report.trial.security_gate_pass);
        assert!(report.security_gates.pass);
        assert!(export_path.is_file());
        let exported = std::fs::read_to_string(&export_path)?;
        let parsed = parse_pipeline_yaml_tolerant(&exported)?;
        assert!(parsed.pii.capsule_enabled);
    }

    // 2. Hard PII gate block: pii.capsule_enabled: [false] -> Forbidden & no export file
    let blocked_pii_path = tmp.path().join("blocked_pii.yaml");
    let bad_pii_yaml = r#"
version: 1
golden_set: opt_gs
strategy: greedy_node
security_gates:
  require_zero_pii_leak: true
  require_zero_arn_violation: true
nodes:
  pii.capsule_enabled: [false]
  retrieve.fusion: [rrf]
"#;
    let res_pii = AutoRagOptimizer::optimize(
        store.as_ref(),
        &retriever,
        &admin,
        &OptimizeRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(bucket_id),
            search_space_yaml: bad_pii_yaml.into(),
            export_best_path: Some(blocked_pii_path.to_string_lossy().into_owned()),
            ..OptimizeRequest::default()
        },
    )
    .await;
    assert!(matches!(res_pii, Err(OpenRagError::Forbidden(_))));
    assert!(!blocked_pii_path.exists());

    // 3. Hard ARN gate block: iam.bypass_rls_test_only: [true] -> Forbidden & no export file
    let blocked_arn_path = tmp.path().join("blocked_arn.yaml");
    let bad_arn_yaml = r#"
version: 1
golden_set: opt_gs
strategy: greedy_node
nodes:
  iam.bypass_rls_test_only: [true]
  retrieve.fusion: [rrf]
"#;
    let res_arn = AutoRagOptimizer::optimize(
        store.as_ref(),
        &retriever,
        &admin,
        &OptimizeRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(bucket_id),
            search_space_yaml: bad_arn_yaml.into(),
            export_best_path: Some(blocked_arn_path.to_string_lossy().into_owned()),
            ..OptimizeRequest::default()
        },
    )
    .await;
    assert!(matches!(res_arn, Err(OpenRagError::Forbidden(_))));
    assert!(!blocked_arn_path.exists());

    let trials = store.list_optimization_trials(org_id, Some(gs.id)).await?;
    let blocked_count = trials
        .iter()
        .filter(|t| t.status == OptimizationStatus::BlockedSecurityGate)
        .count();
    assert_eq!(blocked_count, 2);

    Ok(())
}

#[tokio::test]
async fn pipeline_promoter_blocks_unsafe_configs_and_failed_bench_runs(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;
    let member = make_member_ctx(org_id, dept_id)?;

    let store = Arc::new(InMemoryEvalStore::new());
    let promoter = PipelinePromoter::new(Arc::clone(&store));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));

    let clean_yaml = include_str!("../../../configs/pipeline.default.yaml");
    let unsafe_capsule_yaml = clean_yaml.replace("capsule_enabled: true", "capsule_enabled: false");

    // 1. promote_from_yaml blocks capsule_enabled: false
    assert!(matches!(
        promoter
            .promote_from_yaml(org_id, &admin, "unsafe-yaml", &unsafe_capsule_yaml, Some("b1"))
            .await,
        Err(OpenRagError::Forbidden(_))
    ));

    // 2. Non-admin blocked
    assert!(matches!(
        promoter
            .promote_from_yaml(org_id, &member, "member-try", clean_yaml, Some("b1"))
            .await,
        Err(OpenRagError::Forbidden(_))
    ));

    // 3. Config with failing bench_run blocked from promotion
    let cfg = promoter
        .register_config(org_id, "gated-cfg", clean_yaml, Some(admin.principal_id))
        .await?;
    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "gate_gs".into(),
                axis: GoldenAxis::Retrieval,
                description: None,
                version: None,
            },
        )
        .await?;
    let failed_run = store
        .insert_bench_run(
            &openrag_eval::NewBenchRun {
                org_id,
                golden_set_id: gs.id,
                pipeline_config_id: Some(cfg.id),
                trial_id: None,
                track: BenchTrack::Track3SecurityOcr,
                metrics: serde_json::json!({}),
                pii_leak_count: 1,
                arn_violation_count: 0,
            },
            &[],
        )
        .await?;
    assert!(!failed_run.security_gate_pass);

    assert!(matches!(
        promoter
            .promote_config(
                org_id,
                &admin,
                PromotePipelineInput {
                    pipeline_config_id: Some(cfg.id),
                    bench_run_id: Some(failed_run.id),
                    bucket_ref: Some("bucket-1".into()),
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));
    assert!(!store.get_pipeline_config(org_id, cfg.id).await?.is_promoted);

    // 4. Clean config promotes and binds bucket
    let promoted = promoter
        .promote_from_yaml(org_id, &admin, "clean-prod", clean_yaml, Some("bucket-1"))
        .await?;
    assert!(promoted.is_promoted);
    assert!(promoted.promoted_at.is_some());
    assert_eq!(
        store.get_bucket_binding(org_id, "bucket-1")?,
        Some(promoted.id)
    );

    Ok(())
}

#[tokio::test]
async fn tokenless_pii_capsule_disabled_fails_all_strategies_and_track3(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;

    let doc_id = Uuid::new_v4();
    let backend = InMemoryRetrieveBackend::new();
    backend
        .insert(seed_doc_record(
            org_id,
            bucket_id,
            dept_id,
            admin.principal_id,
            doc_id,
            "AutoRAG 보안 게이트 검증용 본문 S15",
        )?)
        .await;
    let retriever = InMemoryHybridRetriever::new(backend);

    let store = Arc::new(InMemoryEvalStore::new());
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));
    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "s15_tokenless_gs".into(),
                axis: GoldenAxis::Multi,
                description: None,
                version: None,
            },
        )
        .await?;
    gs_mgr
        .add_golden_item(
            org_id,
            gs.id,
            CreateGoldenItemInput {
                axis: GoldenAxis::Pii,
                query_text: "S15 pii 점검".into(),
                test_principal_id: None,
                expected_doc_ids: vec![doc_id],
                forbidden_doc_ids: Vec::new(),
                expected_answer: None,
                expected_pii_tokens: Vec::new(),
                expected_edges: None,
                review_status: None,
            },
        )
        .await?;

    // 1. Track3SecurityOcr fails when pii.capsule_enabled = false even without synthetic PII tokens
    let mut bad_pipeline = default_pipeline_config();
    bad_pipeline.pii.capsule_enabled = false;
    let bench_summary = BenchRunner::run_all_tracks(
        store.as_ref(),
        &retriever,
        &admin,
        &BenchRunRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(bucket_id),
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

    // 2. AutoRagOptimizer blocks pii.capsule_enabled: [false] across all 3 strategies
    let tmp = TempDir::new()?;
    for (idx, strat) in ["greedy_node", "grid", "ab_pair"].iter().enumerate() {
        let export_path = tmp.path().join(format!("s15_blocked_{idx}.yaml"));
        let yaml = format!(
            r#"
version: 1
golden_set: s15_tokenless_gs
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
            &admin,
            &OptimizeRequest {
                golden_set_id: Some(gs.id),
                bucket_id: Some(bucket_id),
                search_space_yaml: yaml,
                require_zero_pii_leak: Some(true),
                require_zero_arn_violation: Some(true),
                export_best_path: Some(export_path.to_string_lossy().into_owned()),
                ..OptimizeRequest::default()
            },
        )
        .await;
        assert!(
            matches!(res, Err(OpenRagError::Forbidden(_))),
            "strategy {strat} should fail closed on pii.capsule_enabled: [false]"
        );
        assert!(
            !export_path.exists(),
            "strategy {strat} must not export unsafe YAML"
        );
    }

    // 3. ab_pair also blocks iam.bypass_rls_test_only: [true]
    let ab_arn_path = tmp.path().join("ab_pair_arn_blocked.yaml");
    let ab_arn_yaml = r#"
version: 1
golden_set: s15_tokenless_gs
strategy: ab_pair
nodes:
  iam.bypass_rls_test_only: [true]
  retrieve.fusion: [rrf]
"#;
    let ab_arn_res = AutoRagOptimizer::optimize(
        store.as_ref(),
        &retriever,
        &admin,
        &OptimizeRequest {
            golden_set_id: Some(gs.id),
            bucket_id: Some(bucket_id),
            search_space_yaml: ab_arn_yaml.into(),
            export_best_path: Some(ab_arn_path.to_string_lossy().into_owned()),
            ..OptimizeRequest::default()
        },
    )
    .await;
    assert!(matches!(ab_arn_res, Err(OpenRagError::Forbidden(_))));
    assert!(!ab_arn_path.exists());

    let trials = store.list_optimization_trials(org_id, Some(gs.id)).await?;
    let blocked_cnt = trials
        .iter()
        .filter(|t| t.status == OptimizationStatus::BlockedSecurityGate && !t.security_gate_pass)
        .count();
    assert_eq!(blocked_cnt, 4);

    Ok(())
}

#[tokio::test]
async fn pipeline_promoter_multi_track_gate_mismatch_and_thumbnail_dim(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;

    let store = Arc::new(InMemoryEvalStore::new());
    let promoter = PipelinePromoter::new(Arc::clone(&store));
    let gs_mgr = GoldenSetManager::new(Arc::clone(&store));

    let clean_yaml = include_str!("../../../configs/pipeline.default.yaml");
    let cfg_a = promoter
        .register_config(org_id, "cfg-a", clean_yaml, Some(admin.principal_id))
        .await?;
    let cfg_b = promoter
        .register_config(org_id, "cfg-b", clean_yaml, Some(admin.principal_id))
        .await?;

    let gs = gs_mgr
        .create_golden_set(
            org_id,
            CreateGoldenSetInput {
                name: "multitrack_gate_gs".into(),
                axis: GoldenAxis::Multi,
                description: None,
                version: None,
            },
        )
        .await?;

    // Failed track3_security_ocr followed by passing track1_retrieval for cfg_a
    let fail_t3 = store
        .insert_bench_run(
            &openrag_eval::NewBenchRun {
                org_id,
                golden_set_id: gs.id,
                pipeline_config_id: Some(cfg_a.id),
                trial_id: None,
                track: BenchTrack::Track3SecurityOcr,
                metrics: serde_json::json!({}),
                pii_leak_count: 1,
                arn_violation_count: 0,
            },
            &[],
        )
        .await?;
    assert!(!fail_t3.security_gate_pass);

    let pass_t1 = store
        .insert_bench_run(
            &openrag_eval::NewBenchRun {
                org_id,
                golden_set_id: gs.id,
                pipeline_config_id: Some(cfg_a.id),
                trial_id: None,
                track: BenchTrack::Track1Retrieval,
                metrics: serde_json::json!({}),
                pii_leak_count: 0,
                arn_violation_count: 0,
            },
            &[],
        )
        .await?;
    assert!(pass_t1.security_gate_pass);

    // 1. promote_config without bench_run_id must fail because latest track3_security_ocr failed
    assert!(matches!(
        promoter
            .promote_config(
                org_id,
                &admin,
                PromotePipelineInput {
                    pipeline_config_id: Some(cfg_a.id),
                    bench_run_id: None,
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));
    assert!(!store.get_pipeline_config(org_id, cfg_a.id).await?.is_promoted);

    // Even explicitly supplying the passing track1 run_id must still fail due to failed track3
    assert!(matches!(
        promoter
            .promote_config(
                org_id,
                &admin,
                PromotePipelineInput {
                    pipeline_config_id: Some(cfg_a.id),
                    bench_run_id: Some(pass_t1.id),
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Forbidden(_))
    ));
    assert!(!store.get_pipeline_config(org_id, cfg_a.id).await?.is_promoted);

    // 2. Mismatched bench_run.pipeline_config_id rejected
    assert!(matches!(
        promoter
            .promote_config(
                org_id,
                &admin,
                PromotePipelineInput {
                    pipeline_config_id: Some(cfg_b.id),
                    bench_run_id: Some(pass_t1.id),
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));
    assert!(!store.get_pipeline_config(org_id, cfg_b.id).await?.is_promoted);

    // 3. Re-running track3_security_ocr cleanly allows cfg_a promotion
    let pass_t3 = store
        .insert_bench_run(
            &openrag_eval::NewBenchRun {
                org_id,
                golden_set_id: gs.id,
                pipeline_config_id: Some(cfg_a.id),
                trial_id: None,
                track: BenchTrack::Track3SecurityOcr,
                metrics: serde_json::json!({}),
                pii_leak_count: 0,
                arn_violation_count: 0,
            },
            &[],
        )
        .await?;
    let promoted_a = promoter
        .promote_config(
            org_id,
            &admin,
            PromotePipelineInput {
                pipeline_config_id: Some(cfg_a.id),
                bench_run_id: Some(pass_t3.id),
                bucket_ref: None,
            },
        )
        .await?;
    assert!(promoted_a.is_promoted);

    // 4. thumbnail.dim != 256 rejected
    let bad_thumb_yaml = clean_yaml.replace("dim: 256", "dim: 128");
    let bad_thumb_rec = store
        .register_pipeline_config(
            org_id,
            &openrag_eval::RegisterPipelineInput {
                name: "bad-thumb-dim".into(),
                profile: openrag_eval::PipelineProfileKind::Standard,
                pipeline_yaml: bad_thumb_yaml,
                config_json: serde_json::json!({
                    "pii": { "capsule_enabled": true },
                    "store": { "dim": 256 },
                    "thumbnail": { "dim": 256 }
                }),
                created_by: Some(admin.principal_id),
            },
        )
        .await?;
    assert!(matches!(
        promoter
            .promote_config(
                org_id,
                &admin,
                PromotePipelineInput {
                    pipeline_config_id: Some(bad_thumb_rec.id),
                    bench_run_id: None,
                    bucket_ref: None,
                },
            )
            .await,
        Err(OpenRagError::Validation(_))
    ));
    assert!(
        !store
            .get_pipeline_config(org_id, bad_thumb_rec.id)
            .await?
            .is_promoted
    );

    Ok(())
}

#[tokio::test]
async fn ab_search_request_rejects_reserved_variant_keys(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let admin = make_admin_ctx(org_id, dept_id)?;
    let retriever = InMemoryHybridRetriever::new(InMemoryRetrieveBackend::new());

    for reserved in ["security_gates", "winner", "query_count"] {
        let mut variants = BTreeMap::new();
        variants.insert(reserved.to_string(), AbVariantConfig::default());
        let req = AbSearchRequest {
            bucket_id: None,
            golden_set_id: None,
            k: Some(5),
            queries: vec![AbQueryInput::QueryText("test query".into())],
            variants,
        };
        assert!(
            matches!(req.validate(), Err(OpenRagError::Validation(_))),
            "expected Validation error for reserved key '{reserved}'"
        );
        let eval_res = AbSearchEvaluator::evaluate(&retriever, None, &admin, &req, None).await;
        assert!(matches!(eval_res, Err(OpenRagError::Validation(_))));
    }

    Ok(())
}
