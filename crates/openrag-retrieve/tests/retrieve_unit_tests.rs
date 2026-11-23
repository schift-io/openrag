use std::collections::{BTreeMap, HashMap};

use chrono::{Duration, Utc};
use openrag_core::{
    AccessScope, AuthLevel, BBoxNorm1000, FusionMode, OpenRagError, PipelineConfig,
    PrincipalContext, SearchSpaceConfig,
};
use openrag_memory::{ConceptCategory, ConceptSource, DeterministicTextEmbedder};
use openrag_retrieve::{
    build_concept_lookup_terms, fuse_and_rank, has_conflicting_tags, parse_spaces,
    sanitize_and_parse_inline_tags, BackendExecutionOutput, EvidenceAdmission, FusionContext,
    HybridRetriever, HydratedCandidate, InMemoryDocRecord, InMemoryHybridRetriever,
    InMemoryRetrieveBackend, ResolvedRetrieveParams, RetrieveOverrides, RetrieveSpace,
    SearchRequest,
};
use openrag_store::{l2_normalize, Bm25Hit, TagFilter, VectorHit, VectorTable, VECTOR_DIM};
use uuid::Uuid;

fn make_ctx(
    org_id: Uuid,
    principal_id: Uuid,
    dept_id: Uuid,
    level: u8,
) -> Result<PrincipalContext, Box<dyn std::error::Error>> {
    let mut ctx = PrincipalContext::new(org_id, principal_id, AuthLevel::new(level)?);
    ctx.dept_ids = vec![dept_id];
    Ok(ctx)
}

fn basis_vec(
    axis: usize,
    primary: f32,
    secondary_axis: Option<usize>,
    secondary: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut raw = vec![0.001_f32; VECTOR_DIM];
    raw[axis % VECTOR_DIM] = primary;
    if let Some(sec) = secondary_axis {
        raw[sec % VECTOR_DIM] = secondary;
    }
    Ok(l2_normalize(&raw)?)
}

fn sample_record(
    org_id: Uuid,
    dept_id: Uuid,
    owner: Uuid,
    space: RetrieveSpace,
    title: &str,
    text: &str,
    tags: &[&str],
) -> Result<InMemoryDocRecord, Box<dyn std::error::Error>> {
    let emb = DeterministicTextEmbedder::embed_sync(&format!("{title} {text}"))?;
    let id = Uuid::new_v4();
    let doc_id = match space {
        RetrieveSpace::Chunks | RetrieveSpace::Thumb => Some(Uuid::new_v4()),
        RetrieveSpace::Wiki | RetrieveSpace::Memory => None,
    };
    let page_id = if space == RetrieveSpace::Thumb {
        Some(id)
    } else {
        None
    };
    Ok(InMemoryDocRecord {
        id,
        org_id,
        bucket_id: None,
        dept_id,
        owner_principal_id: owner,
        access_scope: AccessScope::Dept,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        min_auth_level: 3,
        space,
        document_id: doc_id,
        page_id,
        title: Some(title.into()),
        text: text.into(),
        tags: tags.iter().map(|s| (*s).to_string()).collect(),
        document_tags: None,
        revision: 1,
        valid_from: Utc::now() - Duration::days(1),
        valid_to: None,
        is_archived: false,
        is_superseded: false,
        thumb_uri: if space == RetrieveSpace::Thumb {
            Some(format!("s3://thumbs/{id}.webp"))
        } else {
            None
        },
        bbox_norm_1000: None,
        acl_scope_key: format!("{org_id}:dept"),
        embedding: Some(emb),
    })
}

#[tokio::test]
async fn test_validation_rejects_invalid_alpha_top_k_spaces_and_vectors(
) -> Result<(), Box<dyn std::error::Error>> {
    let backend = InMemoryRetrieveBackend::new();
    let retriever = InMemoryHybridRetriever::new(backend);
    let ctx = make_ctx(Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4(), 3)?;

    // Alpha boundaries: 0.0 and 1.0 valid; -0.01, 1.01, NaN, Inf invalid
    for bad_alpha in [-0.01_f32, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut req = SearchRequest::new("테스트");
        req.alpha = Some(bad_alpha);
        let err = retriever
            .search(&ctx, &req)
            .await
            .err()
            .ok_or("expected Validation error for alpha")?;
        assert!(
            matches!(err, OpenRagError::Validation(_)),
            "unexpected error: {err:?}"
        );
    }
    for ok_alpha in [0.0_f32, 1.0] {
        let mut req = SearchRequest::new("테스트");
        req.alpha = Some(ok_alpha);
        assert!(retriever.search(&ctx, &req).await.is_ok());
    }

    // top_k boundaries: 0 and 201 invalid; 1 and 200 valid
    for bad_k in [0_usize, 201] {
        let mut req = SearchRequest::new("테스트");
        req.top_k = Some(bad_k);
        let err = retriever
            .search(&ctx, &req)
            .await
            .err()
            .ok_or("expected Validation error for top_k")?;
        assert!(matches!(err, OpenRagError::Validation(_)));
    }
    for ok_k in [1_usize, 200] {
        let mut req = SearchRequest::new("테스트");
        req.top_k = Some(ok_k);
        assert!(retriever.search(&ctx, &req).await.is_ok());
    }

    // Empty / unknown spaces rejected
    assert!(matches!(
        parse_spaces(&[]),
        Err(OpenRagError::Validation(_))
    ));
    assert!(matches!(
        parse_spaces(&["unknown_space".into()]),
        Err(OpenRagError::Validation(_))
    ));
    let parsed_spaces = parse_spaces(&[
        "chunks".into(),
        "thumb".into(),
        "wiki".into(),
        "memory".into(),
    ])?;
    assert_eq!(parsed_spaces.len(), 4);

    let mut empty_sp_req = SearchRequest::new("테스트");
    empty_sp_req.spaces = Some(Vec::new());
    assert!(matches!(
        retriever.search(&ctx, &empty_sp_req).await,
        Err(OpenRagError::Validation(_))
    ));

    // Invalid rrf_k, admission_threshold, latest_revision_boost
    let mut bad_rrf = SearchRequest::new("테스트");
    bad_rrf.rrf_k = Some(0);
    assert!(matches!(
        retriever.search(&ctx, &bad_rrf).await,
        Err(OpenRagError::Validation(_))
    ));

    let mut bad_adm = SearchRequest::new("테스트");
    bad_adm.admission_threshold = Some(1.05);
    assert!(matches!(
        retriever.search(&ctx, &bad_adm).await,
        Err(OpenRagError::Validation(_))
    ));

    let mut bad_boost = SearchRequest::new("테스트");
    bad_boost.latest_revision_boost = Some(-0.1);
    assert!(matches!(
        retriever.search(&ctx, &bad_boost).await,
        Err(OpenRagError::Validation(_))
    ));

    // Invalid query_vector (wrong dim, all-zero, NaN, Inf)
    for bad_vec in [
        vec![1.0_f32; 127],
        vec![0.0_f32; 256],
        {
            let mut v = vec![0.1_f32; 256];
            v[0] = f32::NAN;
            v
        },
        {
            let mut v = vec![0.1_f32; 256];
            v[10] = f32::INFINITY;
            v
        },
    ] {
        let mut req = SearchRequest::new("테스트");
        req.query_vector = Some(bad_vec);
        let err = retriever
            .search(&ctx, &req)
            .await
            .err()
            .ok_or("expected vector validation error")?;
        assert!(matches!(err, OpenRagError::Validation(_)));
    }

    Ok(())
}

#[test]
fn test_inline_tag_parsing_punctuation_negation_and_url_fragments() {
    let p1 = sanitize_and_parse_inline_tags("규정 #alpha, 확인.");
    assert_eq!(p1.inline_tags, vec!["alpha"]);
    assert_eq!(p1.clean_query, "규정 확인.");

    let p2 = sanitize_and_parse_inline_tags("규정 tag:alpha;");
    assert_eq!(p2.inline_tags, vec!["alpha"]);
    assert_eq!(p2.clean_query, "규정");

    let p3 = sanitize_and_parse_inline_tags("#alpha #beta 규정 -#draft -tag:obsolete");
    assert_eq!(p3.inline_tags, vec!["alpha", "beta"]);
    assert_eq!(p3.excluded_tags, vec!["draft", "obsolete"]);
    assert_eq!(p3.clean_query, "규정");

    let p4 = sanitize_and_parse_inline_tags("https://example.org/docs#section1 참고");
    assert!(p4.inline_tags.is_empty());
    assert_eq!(p4.clean_query, "https://example.org/docs#section1 참고");

    let p5 = sanitize_and_parse_inline_tags("# tag: 공백");
    assert!(p5.inline_tags.is_empty());
    assert_eq!(p5.clean_query, "공백");

    let p6 = sanitize_and_parse_inline_tags("#한글_태그 #Policy-2026");
    assert_eq!(p6.inline_tags, vec!["한글_태그", "policy-2026"]);
}

#[tokio::test]
async fn test_empty_punctuation_and_tag_only_queries() -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let bucket_id = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    let mut rec = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "보안 규정",
        "비밀번호 관리 정책",
        &["security"],
    )?;
    rec.bucket_id = Some(bucket_id);
    // Even with no vector embedding, tag-only queries must still find tagged rows!
    rec.embedding = None;
    backend.insert(rec).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    for empty_q in ["", "   ", "!!!???..."] {
        let resp = retriever.search(&ctx, &SearchRequest::new(empty_q)).await?;
        assert!(resp.hits.is_empty());
        assert_eq!(resp.tag_facets, BTreeMap::new());
        assert_eq!(resp.admission, EvidenceAdmission::Abstain);
        assert_eq!(resp.max_score, 0.0);
        assert_eq!(resp.top_score, 0.0);
        assert_eq!(resp.top_auth_level, None);
    }

    // Tag-only query "#security" must NOT short-circuit to empty!
    let mut tag_only_req = SearchRequest::new("#security");
    tag_only_req.bucket_id = Some(bucket_id);
    let tag_resp = retriever.search(&ctx, &tag_only_req).await?;
    assert_eq!(tag_resp.hits.len(), 1);
    assert_eq!(tag_resp.parsed_inline_tags, vec!["security"]);
    assert_eq!(tag_resp.tag_facets.get("security"), Some(&1));

    Ok(())
}

#[tokio::test]
async fn test_rrf_k60_and_convex_alpha_fusion_and_score_breakdown(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let q_vec = basis_vec(0, 1.0, None, 0.0)?;
    let vec_doc_emb = basis_vec(0, 1.0, None, 0.0)?; // cos ~ 1.0
    let bm25_doc_emb = basis_vec(5, 1.0, None, 0.0)?; // cos ~ 0.0

    let backend = InMemoryRetrieveBackend::new();
    let mut rec_vec = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Vector Match",
        "일반 배경 설명",
        &["vec"],
    )?;
    rec_vec.embedding = Some(vec_doc_emb);
    let vec_id = rec_vec.id;

    let mut rec_kw = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Wiki,
        "Keyword Match",
        "전자결재 표준 지침 전자결재 표준 지침",
        &["kw"],
    )?;
    rec_kw.embedding = Some(bm25_doc_emb);
    let kw_id = rec_kw.id;

    backend.insert(rec_vec).await;
    backend.insert(rec_kw).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // Pure vector (convex alpha=0.0, rerank=false)
    let mut req_v = SearchRequest::new("전자결재 표준 지침");
    req_v.query_vector = Some(q_vec.clone());
    req_v.fusion = Some(FusionMode::Convex);
    req_v.alpha = Some(0.0);
    req_v.rerank = Some(false);
    let res_v = retriever.search(&ctx, &req_v).await?;
    assert_eq!(res_v.hits[0].id, vec_id);
    assert!(
        (res_v.hits[0].score_breakdown.convex_score - res_v.hits[0].score_breakdown.vector_score)
            .abs()
            < 1e-5
    );
    // alpha 0 does not query BM25: the keyword-only document carries no BM25 score.
    let kw_v = res_v
        .hits
        .iter()
        .find(|h| h.id == kw_id)
        .ok_or("keyword doc missing from the vector pool")?;
    assert!(kw_v.score_breakdown.bm25_score.abs() < 1e-6);

    // Pure BM25 (convex alpha=1.0, rerank=false)
    let mut req_b = SearchRequest::new("전자결재 표준 지침");
    req_b.query_vector = Some(q_vec.clone());
    req_b.fusion = Some(FusionMode::Convex);
    req_b.alpha = Some(1.0);
    req_b.rerank = Some(false);
    let res_b = retriever.search(&ctx, &req_b).await?;
    assert_eq!(res_b.hits[0].id, kw_id);
    assert!(
        (res_b.hits[0].score_breakdown.convex_score - res_b.hits[0].score_breakdown.bm25_score)
            .abs()
            < 1e-5
    );

    // RRF k=60
    let mut req_rrf = SearchRequest::new("전자결재 표준 지침");
    req_rrf.query_vector = Some(q_vec);
    req_rrf.fusion = Some(FusionMode::Rrf);
    req_rrf.rrf_k = Some(60);
    let res_rrf = retriever.search(&ctx, &req_rrf).await?;
    assert!(!res_rrf.hits.is_empty());
    let bd = &res_rrf.hits[0].score_breakdown;
    assert!(bd.rrf_score > 0.0 && bd.rrf_score <= 1.0);
    assert!(bd.fused_score > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_two_stage_rerank_promotes_exact_match_and_populates_bbox(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    let generic = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "일반 안내",
        "비밀번호 보안 관련 일반 소개 문서입니다.",
        &["general"],
    )?;
    let mut exact = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "비밀번호 만료 주기",
        "비밀번호 만료 주기는 90일이며 대소문자 및 숫자를 포함해야 한다.",
        &["security"],
    )?;
    exact.bbox_norm_1000 = Some(BBoxNorm1000::new(10, 20, 400, 120)?);
    let exact_id = exact.id;

    backend.insert(generic).await;
    backend.insert(exact).await;

    let retriever = InMemoryHybridRetriever::new(backend);
    let mut req = SearchRequest::new("비밀번호 만료 주기는 몇 일인가요?");
    req.rerank = Some(true);
    req.top_k = Some(5);
    let resp = retriever.search(&ctx, &req).await?;
    assert_eq!(resp.hits[0].id, exact_id);
    assert!(resp.hits[0].score_breakdown.rerank_score > 0.0);
    assert_eq!(
        resp.hits[0].bbox_norm_1000,
        Some(BBoxNorm1000::new(10, 20, 400, 120)?)
    );

    Ok(())
}

#[tokio::test]
async fn test_latest_revision_boost_and_valid_to_and_wiki_supersedes(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    let mut v1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "2025년 출장 여비 규정",
        "출장 여비 5만원 지급 기준",
        &["travel"],
    )?;
    v1.revision = 1;
    v1.valid_to = Some(Utc::now() - Duration::days(30));
    let v1_id = v1.id;

    let mut v2 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "2026년 출장 여비 개정 규정",
        "출장 여비 7만원 지급 기준",
        &["travel"],
    )?;
    v2.revision = 2;
    v2.valid_from = Utc::now() - Duration::days(5);
    let v2_id = v2.id;

    let mut w1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Wiki,
        "출장 여비 위키 v1",
        "출장 여비 구버전 위키 안내",
        &["travel"],
    )?;
    w1.revision = 1;
    w1.is_superseded = true;
    let w1_id = w1.id;

    let mut w2 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Wiki,
        "출장 여비 위키 v2",
        "출장 여비 개정 위키 안내",
        &["travel"],
    )?;
    w2.revision = 2;
    let w2_id = w2.id;

    backend.insert(v1).await;
    backend.insert(v2).await;
    backend.insert(w1).await;
    backend.insert(w2).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // Without latest_revision_only: v2 > v1 and w2 > w1
    let mut req = SearchRequest::new("출장 여비");
    req.latest_revision_only = false;
    let resp = retriever.search(&ctx, &req).await?;
    let ids: Vec<Uuid> = resp.hits.iter().map(|h| h.id).collect();
    let pos_v2 = ids.iter().position(|x| *x == v2_id).ok_or("missing v2")?;
    let pos_v1 = ids.iter().position(|x| *x == v1_id).ok_or("missing v1")?;
    assert!(pos_v2 < pos_v1);
    let pos_w2 = ids.iter().position(|x| *x == w2_id).ok_or("missing w2")?;
    let pos_w1 = ids.iter().position(|x| *x == w1_id).ok_or("missing w1")?;
    assert!(pos_w2 < pos_w1);
    assert!(resp.hits[pos_v2].score_breakdown.revision_boost > 0.0);

    // With latest_revision_only: expired v1 & superseded w1 excluded
    let mut req_only = SearchRequest::new("출장 여비");
    req_only.latest_revision_only = true;
    let resp_only = retriever.search(&ctx, &req_only).await?;
    let ids_only: Vec<Uuid> = resp_only.hits.iter().map(|h| h.id).collect();
    assert!(ids_only.contains(&v2_id));
    assert!(ids_only.contains(&w2_id));
    assert!(!ids_only.contains(&v1_id));
    assert!(!ids_only.contains(&w1_id));

    Ok(())
}

#[tokio::test]
async fn test_evidence_admission_and_top_auth_level_and_rls_isolation(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();
    let alice_ctx = make_ctx(org_id, alice, dept_id, 9)?;
    let bob_ctx = make_ctx(org_id, bob, dept_id, 3)?;

    let backend = InMemoryRetrieveBackend::new();
    let mut priv_l9 = sample_record(
        org_id,
        dept_id,
        alice,
        RetrieveSpace::Chunks,
        "기밀 규정",
        "신입 직원 연차 유급 휴가 비밀 부칙",
        &["secret-tag"],
    )?;
    priv_l9.access_scope = AccessScope::Private;
    priv_l9.min_auth_level = 9;

    let dept_l3 = sample_record(
        org_id,
        dept_id,
        alice,
        RetrieveSpace::Chunks,
        "연차 휴가 규정",
        "신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.",
        &["hr"],
    )?;

    backend.insert(priv_l9).await;
    backend.insert(dept_l3).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // Bob (L3) sees only L3 doc, no secret-tag in facets, top_auth_level = 3, Admit at 0.35
    let bob_resp = retriever
        .search(
            &bob_ctx,
            &SearchRequest::new("신입 직원 연차 휴가는 며칠인가요?"),
        )
        .await?;
    assert_eq!(bob_resp.admission, EvidenceAdmission::Admit);
    assert_eq!(bob_resp.top_auth_level, Some(3));
    assert!(!bob_resp.tag_facets.contains_key("secret-tag"));

    // Alice (L9) sees her L9 private doc too -> top_auth_level = 9
    let alice_resp = retriever
        .search(&alice_ctx, &SearchRequest::new("신입 직원 연차 휴가"))
        .await?;
    assert_eq!(alice_resp.admission, EvidenceAdmission::Admit);
    assert_eq!(alice_resp.top_auth_level, Some(9));
    assert_eq!(alice_resp.tag_facets.get("secret-tag"), Some(&1));

    // Off-topic query at admission_threshold = 0.95 -> Abstain
    let mut off_req = SearchRequest::new("화성 탐사선 로켓 연료 배합 비율은?");
    off_req.admission_threshold = Some(0.95);
    let off_resp = retriever.search(&bob_ctx, &off_req).await?;
    assert_eq!(off_resp.admission, EvidenceAdmission::Abstain);

    Ok(())
}

#[tokio::test]
async fn test_concept_alias_expansion_and_corpus_pages_doc_tag_filtering_and_facets(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    backend
        .concepts()
        .link_terms(
            org_id,
            Some(owner),
            "apple",
            "사과",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    backend
        .concepts()
        .link_terms(
            org_id,
            Some(owner),
            "llmwiki",
            "세컨드 브레인",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    backend
        .concepts()
        .link_terms(
            org_id,
            Some(owner),
            "llmwiki",
            "세컨드브레인",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;

    // Thumb (CorpusPages) hit with tags on parent document_tags (NOT rec.tags), plus matching chunk sharing document_id
    let doc_wiki_id = Uuid::new_v4();
    let mut chunk1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Wiki Guide",
        "지식관리 llmwiki 아키텍처 설명",
        &["llmwiki"],
    )?;
    chunk1.document_id = Some(doc_wiki_id);
    let mut chunk2 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Wiki Guide",
        "apple 프로젝트 사례",
        &["llmwiki"],
    )?;
    chunk2.document_id = Some(doc_wiki_id);
    let mut thumb1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Thumb,
        "Wiki Guide",
        "지식관리 llmwiki 다이어그램",
        &[], // CorpusPages has no tags column!
    )?;
    thumb1.document_id = Some(doc_wiki_id);
    thumb1.document_tags = Some(vec!["llmwiki".into()]);

    let mut thumb_other = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Thumb,
        "Other Thumb",
        "지식관리 일반 다이어그램",
        &[],
    )?;
    thumb_other.document_tags = Some(vec!["general".into()]);
    let thumb_other_id = thumb_other.id;

    backend.insert(chunk1).await;
    backend.insert(chunk2).await;
    backend.insert(thumb1).await;
    backend.insert(thumb_other).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // 1. Multi-word concept expansion ("세컨드 브레인" -> finds "llmwiki")
    let resp_multi = retriever
        .search(&ctx, &SearchRequest::new("세컨드 브레인 아키텍처"))
        .await?;
    assert!(resp_multi.expanded_terms.iter().any(|t| t == "llmwiki"));
    assert!(!resp_multi.hits.is_empty());

    // 2. Single-word cross-lingual expansion ("사과" -> finds "apple")
    let resp_single = retriever
        .search(&ctx, &SearchRequest::new("사과 프로젝트"))
        .await?;
    assert!(resp_single.expanded_terms.iter().any(|t| t == "apple"));
    assert!(resp_single.hits.iter().any(|h| h.text.contains("apple")));

    // 3. Tag alias expansion + CorpusPages filtering + deduplicated facet count (doc_wiki counted once, not 3x!)
    let mut tag_req = SearchRequest::new("지식관리 #세컨드브레인");
    tag_req.tag_filter = Some(TagFilter {
        expand_tag_aliases: true,
        ..TagFilter::default()
    });
    // Thumbnails are not a default space; this case checks their document-tag filtering.
    tag_req.spaces = Some(vec![RetrieveSpace::Chunks, RetrieveSpace::Thumb]);
    let resp_tag = retriever.search(&ctx, &tag_req).await?;
    assert!(resp_tag
        .hits
        .iter()
        .any(|h| h.space == RetrieveSpace::Thumb && h.document_id == Some(doc_wiki_id)));
    assert!(resp_tag.hits.iter().all(|h| h.id != thumb_other_id));
    assert_eq!(resp_tag.tag_facets.get("llmwiki"), Some(&1));

    // 4. Conflicting tags_all + tags_exclude -> empty hits + {} facets
    let mut conflict_req = SearchRequest::new("지식관리");
    conflict_req.tag_filter = Some(TagFilter {
        tags_all: vec!["llmwiki".into()],
        tags_exclude: vec!["llmwiki".into()],
        ..TagFilter::default()
    });
    let resp_conflict = retriever.search(&ctx, &conflict_req).await?;
    assert!(resp_conflict.hits.is_empty());
    assert_eq!(resp_conflict.tag_facets, BTreeMap::new());

    Ok(())
}

#[test]
fn test_config_overrides_from_pipeline_default_and_airgap_and_search_space(
) -> Result<(), Box<dyn std::error::Error>> {
    let default_yaml = std::fs::read_to_string("../../configs/pipeline.default.yaml")?;
    let airgap_yaml = std::fs::read_to_string("../../configs/pipeline.airgap.yaml")?;
    let search_space_yaml = std::fs::read_to_string("../../configs/search_space.example.yaml")?;

    let default_cfg = PipelineConfig::from_yaml_str(&default_yaml)?;
    let airgap_cfg = PipelineConfig::from_yaml_str(&airgap_yaml)?;
    let space_cfg = SearchSpaceConfig::from_yaml_str(&search_space_yaml)?;

    let r_def =
        HybridRetriever::from_pipeline_config(InMemoryRetrieveBackend::new(), &default_cfg)?;
    assert_eq!(r_def.default_config().fusion, FusionMode::Convex);
    assert_eq!(r_def.default_config().rrf_k, 60);
    assert!(r_def.default_config().alpha.abs() < 1e-6);
    assert_eq!(r_def.default_config().spaces, vec!["chunks", "wiki", "memory"]);
    assert_eq!(r_def.default_config().top_k, 12);
    assert!(r_def.default_config().rerank);
    assert!((r_def.default_config().latest_revision_boost - 0.15).abs() < 1e-6);
    assert!((r_def.default_config().admission_threshold - 0.35).abs() < 1e-6);

    let r_air = HybridRetriever::from_pipeline_config(InMemoryRetrieveBackend::new(), &airgap_cfg)?;
    assert_eq!(r_air.default_config().fusion, FusionMode::Convex);
    assert!(r_air.default_config().alpha.abs() < 1e-6);
    assert!((r_air.default_config().latest_revision_boost - 0.2).abs() < 1e-6);
    assert!((r_air.default_config().admission_threshold - 0.4).abs() < 1e-6);

    assert!(space_cfg.nodes.contains_key("retrieve_fusion"));
    let overrides = RetrieveOverrides {
        fusion: Some(FusionMode::Convex),
        alpha: Some(0.5),
        top_k: Some(20),
        rerank: Some(false),
        latest_revision_boost: Some(0.25),
        ..RetrieveOverrides::default()
    };
    let merged = overrides.apply_to_config(r_def.default_config())?;
    assert_eq!(merged.fusion, FusionMode::Convex);
    assert!((merged.alpha - 0.5).abs() < 1e-6);
    assert_eq!(merged.top_k, 20);
    assert!(!merged.rerank);
    assert!((merged.latest_revision_boost - 0.25).abs() < 1e-6);

    let req_from_cfg = SearchRequest::from_retrieve_config("테스트", &merged)?;
    assert_eq!(req_from_cfg.fusion, Some(FusionMode::Convex));
    assert_eq!(req_from_cfg.top_k, Some(20));

    Ok(())
}

#[test]
fn test_bug1_filter_before_max_bm25_and_rrf_ranks_and_nan_sanitization(
) -> Result<(), Box<dyn std::error::Error>> {
    let archived_id = Uuid::new_v4();
    let active_id = Uuid::new_v4();
    let nan_id = Uuid::new_v4();
    let doc_arch = Uuid::new_v4();
    let doc_act = Uuid::new_v4();
    let doc_nan = Uuid::new_v4();
    let now = Utc::now();
    let qvec = basis_vec(0, 1.0, None, 0.0)?;

    let mut hydrated = HashMap::new();
    hydrated.insert(
        (RetrieveSpace::Chunks, archived_id),
        HydratedCandidate {
            id: archived_id,
            document_id: Some(doc_arch),
            page_id: None,
            space: RetrieveSpace::Chunks,
            title: Some("Archived".into()),
            text: "예산 집행 안내".into(),
            tags: vec!["archived-tag".into()],
            min_auth_level: Some(3),
            revision: Some(1),
            valid_from: Some(now - Duration::days(2)),
            valid_to: None,
            is_archived: true,
            is_superseded: false,
            thumb_uri: None,
            bbox_norm_1000: None,
            acl_scope_key: "org:dept".into(),
            embedding: Some(qvec.clone()),
        },
    );
    hydrated.insert(
        (RetrieveSpace::Chunks, active_id),
        HydratedCandidate {
            id: active_id,
            document_id: Some(doc_act),
            page_id: None,
            space: RetrieveSpace::Chunks,
            title: Some("Active".into()),
            text: "예산 집행 안내".into(),
            tags: vec!["active-tag".into()],
            min_auth_level: Some(3),
            revision: Some(1),
            valid_from: Some(now - Duration::days(2)),
            valid_to: None,
            is_archived: false,
            is_superseded: false,
            thumb_uri: None,
            bbox_norm_1000: None,
            acl_scope_key: "org:dept".into(),
            embedding: Some(qvec.clone()),
        },
    );
    hydrated.insert(
        (RetrieveSpace::Chunks, nan_id),
        HydratedCandidate {
            id: nan_id,
            document_id: Some(doc_nan),
            page_id: None,
            space: RetrieveSpace::Chunks,
            title: Some("NaN Hit".into()),
            text: "기타 안내".into(),
            tags: vec!["active-tag".into()],
            min_auth_level: Some(3),
            revision: Some(1),
            valid_from: Some(now - Duration::days(2)),
            valid_to: None,
            is_archived: false,
            is_superseded: false,
            thumb_uri: None,
            bbox_norm_1000: None,
            acl_scope_key: "org:dept".into(),
            embedding: None,
        },
    );

    let exec = BackendExecutionOutput {
        vector_hits: vec![
            (
                RetrieveSpace::Chunks,
                VectorHit {
                    id: archived_id,
                    document_id: Some(doc_arch),
                    table: VectorTable::CorpusChunks,
                    cosine_sim: 0.99,
                    distance: 0.01,
                    acl_scope_key: "org:dept".into(),
                    tags: vec![],
                    metadata: serde_json::Map::new(),
                },
            ),
            (
                RetrieveSpace::Chunks,
                VectorHit {
                    id: nan_id,
                    document_id: Some(doc_nan),
                    table: VectorTable::CorpusChunks,
                    cosine_sim: f32::NAN,
                    distance: f32::NAN,
                    acl_scope_key: "org:dept".into(),
                    tags: vec![],
                    metadata: serde_json::Map::new(),
                },
            ),
            (
                RetrieveSpace::Chunks,
                VectorHit {
                    id: active_id,
                    document_id: Some(doc_act),
                    table: VectorTable::CorpusChunks,
                    cosine_sim: 0.50,
                    distance: 0.50,
                    acl_scope_key: "org:dept".into(),
                    tags: vec![],
                    metadata: serde_json::Map::new(),
                },
            ),
        ],
        bm25_hits: vec![
            (
                RetrieveSpace::Chunks,
                Bm25Hit {
                    id: archived_id,
                    document_id: Some(doc_arch),
                    table: VectorTable::CorpusChunks,
                    score: 100.0,
                    snippet: String::new(),
                    tags: vec![],
                },
            ),
            (
                RetrieveSpace::Chunks,
                Bm25Hit {
                    id: nan_id,
                    document_id: Some(doc_nan),
                    table: VectorTable::CorpusChunks,
                    score: f64::NAN,
                    snippet: String::new(),
                    tags: vec![],
                },
            ),
            (
                RetrieveSpace::Chunks,
                Bm25Hit {
                    id: active_id,
                    document_id: Some(doc_act),
                    table: VectorTable::CorpusChunks,
                    score: 2.0,
                    snippet: String::new(),
                    tags: vec![],
                },
            ),
        ],
        tag_only_hits: Vec::new(),
        expanded_terms: Vec::new(),
        expanded_tag_filter: None,
        query_vector: qvec,
        hydrated,
    };

    let params = ResolvedRetrieveParams {
        spaces: vec![RetrieveSpace::Chunks],
        fusion: FusionMode::Convex,
        rrf_k: 60,
        alpha: 1.0,
        top_k: 10,
        rerank: false,
        latest_revision_boost: 0.15,
        latest_revision_only: false,
        admission_threshold: 0.35,
    };
    let resp = fuse_and_rank(
        FusionContext {
            clean_query: "예산 집행 안내".into(),
            parsed_inline_tags: Vec::new(),
            parsed_excluded_tags: Vec::new(),
            tag_only_query: false,
            facet_limit: 20,
            now,
        },
        &params,
        exec,
    );

    assert_eq!(resp.hits.len(), 2);
    assert_eq!(resp.hits[0].id, active_id);
    // Archived hit (bm25=100.0) must NOT pollute max_bm25 or RRF rank-1!
    assert!((resp.hits[0].score_breakdown.bm25_score - 1.0).abs() < 1e-5);
    assert!((resp.hits[0].score_breakdown.rrf_score - 1.0).abs() < 1e-5);
    assert_eq!(resp.admission, EvidenceAdmission::Admit);

    // Non-finite NaN hit must produce only finite bounded numbers in [0.0, 1.0]
    for h in &resp.hits {
        assert!(h.score.is_finite() && (0.0..=1.0).contains(&h.score));
        let bd = &h.score_breakdown;
        for val in [
            bd.vector_score,
            bd.bm25_score,
            bd.rrf_score,
            bd.convex_score,
            bd.revision_boost,
            bd.rerank_score,
            bd.fused_score,
        ] {
            assert!(val.is_finite(), "non-finite score breakdown field: {bd:?}");
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_bug2_off_topic_abstains_at_default_035_while_grounded_admits(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    let mut doc1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "사무용품 구매",
        "사무용품 구매 신청 절차 안내입니다.",
        &["ops"],
    )?;
    doc1.revision = 2;
    let doc2 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "연차 휴가 규정",
        "신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.",
        &["hr"],
    )?;
    let doc3 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "비밀번호 정책",
        "비밀번호 만료 주기는 90일이며 정기 변경해야 한다.",
        &["security"],
    )?;
    backend.insert(doc1).await;
    backend.insert(doc2).await;
    backend.insert(doc3).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // Off-topic query at BOTH default 0.35 and 0.95 under Rrf (rerank=true/false) & Convex must Abstain!
    for (fusion, rerank, thresh) in [
        (FusionMode::Rrf, true, 0.35_f32),
        (FusionMode::Rrf, false, 0.35),
        (FusionMode::Convex, true, 0.35),
        (FusionMode::Rrf, true, 0.95),
    ] {
        let mut req = SearchRequest::new("화성 탐사선 로켓 연료 배합 비율은?");
        req.fusion = Some(fusion);
        req.rerank = Some(rerank);
        req.admission_threshold = Some(thresh);
        let resp = retriever.search(&ctx, &req).await?;
        assert_eq!(
            resp.admission,
            EvidenceAdmission::Abstain,
            "off-topic query falsely admitted for fusion={fusion:?}, rerank={rerank}, thresh={thresh}, top_score={}",
            resp.top_score
        );
    }

    // Grounded queries at default 0.35 must cleanly Admit
    for q in [
        "신입 직원 연차 휴가는 며칠인가요?",
        "비밀번호 만료 주기는 90일",
    ] {
        let resp = retriever.search(&ctx, &SearchRequest::new(q)).await?;
        assert_eq!(
            resp.admission,
            EvidenceAdmission::Admit,
            "grounded query failed to admit: {q}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_bug3_future_valid_from_not_boosted_and_excluded_when_latest_only(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    let mut curr = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "사내 보안 규정 v1",
        "사내 보안 규정 현행 기준 안내",
        &["policy"],
    )?;
    curr.revision = 1;
    curr.valid_from = Utc::now() - Duration::days(10);
    let curr_id = curr.id;

    let mut future = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "사내 보안 규정 v2 예정",
        "사내 보안 규정 현행 기준 안내",
        &["policy"],
    )?;
    future.revision = 2;
    future.valid_from = Utc::now() + Duration::days(30);
    let future_id = future.id;

    backend.insert(curr).await;
    backend.insert(future).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // 1. latest_revision_only = false: curr ranks strictly above future, future gets 0 boost
    let resp = retriever
        .search(&ctx, &SearchRequest::new("사내 보안 규정"))
        .await?;
    let ids: Vec<Uuid> = resp.hits.iter().map(|h| h.id).collect();
    let pos_curr = ids
        .iter()
        .position(|x| *x == curr_id)
        .ok_or("missing curr")?;
    let pos_future = ids
        .iter()
        .position(|x| *x == future_id)
        .ok_or("missing future")?;
    assert!(pos_curr < pos_future);
    assert!(resp.hits[pos_curr].score_breakdown.revision_boost > 0.0);
    assert_eq!(resp.hits[pos_future].score_breakdown.revision_boost, 0.0);

    // 2. latest_revision_only = true: future valid_from > now is excluded
    let mut req_only = SearchRequest::new("사내 보안 규정");
    req_only.latest_revision_only = true;
    let resp_only = retriever.search(&ctx, &req_only).await?;
    let ids_only: Vec<Uuid> = resp_only.hits.iter().map(|h| h.id).collect();
    assert!(ids_only.contains(&curr_id));
    assert!(!ids_only.contains(&future_id));

    Ok(())
}

#[tokio::test]
async fn test_bug4_and_bug6_archived_and_thumb_tag_starvation_in_memory(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();

    // 50 archived Chunks + 1 active Chunk
    for i in 0..50 {
        let mut arch = sample_record(
            org_id,
            dept_id,
            owner,
            RetrieveSpace::Chunks,
            &format!("Archived {i}"),
            "아키텍처 표준 규격 문서",
            &["archived-only-tag"],
        )?;
        arch.is_archived = true;
        backend.insert(arch).await;
    }
    let active_chunk = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Active Spec",
        "아키텍처 표준 규격 문서",
        &["active-tag"],
    )?;
    let active_chunk_id = active_chunk.id;
    backend.insert(active_chunk).await;

    // 120 noise Thumb pages + 1 target-tag Thumb page
    for i in 0..120 {
        let mut noise = sample_record(
            org_id,
            dept_id,
            owner,
            RetrieveSpace::Thumb,
            &format!("Noise Thumb {i}"),
            "아키텍처 다이어그램 상세 설명",
            &[],
        )?;
        noise.document_tags = Some(vec!["noise".into()]);
        backend.insert(noise).await;
    }
    let mut target_thumb = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Thumb,
        "Target Thumb",
        "아키텍처 다이어그램 상세 설명",
        &[],
    )?;
    target_thumb.document_tags = Some(vec!["target-tag".into()]);
    let target_thumb_id = target_thumb.id;
    backend.insert(target_thumb).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    let mut req_arch = SearchRequest::new("아키텍처 표준 규격");
    req_arch.spaces = Some(vec![RetrieveSpace::Chunks]);
    req_arch.top_k = Some(5);
    let resp_arch = retriever.search(&ctx, &req_arch).await?;
    assert_eq!(resp_arch.hits.len(), 1);
    assert_eq!(resp_arch.hits[0].id, active_chunk_id);

    let facets = retriever.bucket_tag_facets(&ctx, None, None, 20).await?;
    assert!(!facets.contains_key("archived-only-tag"));
    assert_eq!(facets.get("active-tag"), Some(&1));

    let mut req_thumb = SearchRequest::new("아키텍처 다이어그램 상세 설명 #target-tag");
    req_thumb.spaces = Some(vec![RetrieveSpace::Thumb]);
    req_thumb.top_k = Some(10);
    let resp_thumb = retriever.search(&ctx, &req_thumb).await?;
    assert_eq!(resp_thumb.hits.len(), 1);
    assert_eq!(resp_thumb.hits[0].id, target_thumb_id);
    assert_eq!(resp_thumb.admission, EvidenceAdmission::Admit);

    Ok(())
}

#[tokio::test]
async fn test_bug5_tag_only_convex_admits_and_punct_tag_triggers_null_embedding(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    // Record WITH embedding whose text does NOT contain "payroll"
    let payroll_rec = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "급여 지침",
        "매월 25일 임직원 급여 계좌 이체 기준 안내",
        &["payroll"],
    )?;
    let payroll_id = payroll_rec.id;

    // Record WITHOUT embedding (e.g. wiki page with embedding IS NULL)
    let mut sec_wiki = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Wiki,
        "보안 위키",
        "내부 망분리 보안 점검 체크리스트",
        &["security"],
    )?;
    sec_wiki.embedding = None;
    let sec_wiki_id = sec_wiki.id;

    backend.insert(payroll_rec).await;
    backend.insert(sec_wiki).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    // #payroll under Convex (alpha = 0.0, 0.3, 1.0) & rerank=true/false MUST Admit
    for (alpha, rerank) in [(0.3_f32, true), (0.3, false), (0.0, true), (1.0, false)] {
        let mut req = SearchRequest::new("#payroll");
        req.fusion = Some(FusionMode::Convex);
        req.alpha = Some(alpha);
        req.rerank = Some(rerank);
        let resp = retriever.search(&ctx, &req).await?;
        assert!(resp.hits.iter().any(|h| h.id == payroll_id));
        assert_eq!(
            resp.admission,
            EvidenceAdmission::Admit,
            "tag-only Convex failed to admit at alpha={alpha}, rerank={rerank}, top_score={}",
            resp.top_score
        );
    }

    // Reviewer minor note: "#security !!!" (clean_query="!!!" non-empty, 0 searchable tokens)
    // must still execute tag_only_space lookup for embedding=None rows.
    let resp_punct = retriever
        .search(&ctx, &SearchRequest::new("#security !!!"))
        .await?;
    assert!(resp_punct.hits.iter().any(|h| h.id == sec_wiki_id));
    assert_eq!(resp_punct.admission, EvidenceAdmission::Admit);

    Ok(())
}

#[tokio::test]
async fn test_edge_cases_conflicting_tags_and_4word_and_particle_concept_lookup(
) -> Result<(), Box<dyn std::error::Error>> {
    // FINDING-4a: unnormalized tags_all vs tags_exclude
    let unnorm_conflict = TagFilter {
        tags_all: vec!["#Security".into()],
        tags_exclude: vec!["security".into()],
        ..TagFilter::default()
    };
    assert!(has_conflicting_tags(&unnorm_conflict));

    // FINDING-6a & 6b: 4-word shingle & Korean particle suffix stripping
    let terms_4w = build_concept_lookup_terms("신규 인공 지능 검색 엔진 보고");
    assert!(terms_4w.iter().any(|t| t == "인공 지능 검색 엔진"));

    let terms_particle = build_concept_lookup_terms("사과를 프로젝트 보고");
    assert!(terms_particle.iter().any(|t| t == "사과"));

    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();
    backend
        .concepts()
        .link_terms(
            org_id,
            Some(owner),
            "ai search engine",
            "인공 지능 검색 엔진",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;
    backend
        .concepts()
        .link_terms(
            org_id,
            Some(owner),
            "apple",
            "사과",
            ConceptCategory::Domain,
            ConceptSource::Manual,
        )
        .await?;

    let doc_ai = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "AI Search Engine",
        "Next-gen ai search engine benchmark results",
        &["ai"],
    )?;
    let doc_apple = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Apple Project",
        "apple roadmap overview",
        &["fruit"],
    )?;
    backend.insert(doc_ai).await;
    backend.insert(doc_apple).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    let r4 = retriever
        .search(&ctx, &SearchRequest::new("신규 인공 지능 검색 엔진 보고"))
        .await?;
    assert!(r4.expanded_terms.iter().any(|t| t == "ai search engine"));
    assert!(r4.hits.iter().any(|h| h.text.contains("ai search engine")));

    let rp = retriever
        .search(&ctx, &SearchRequest::new("사과를 프로젝트 보고"))
        .await?;
    assert!(rp.expanded_terms.iter().any(|t| t == "apple"));
    assert!(rp.hits.iter().any(|h| h.text.contains("apple")));

    Ok(())
}

#[tokio::test]
async fn test_latest_revision_only_does_not_suppress_corpus_with_same_title_wiki_version(
) -> Result<(), Box<dyn std::error::Error>> {
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;

    let backend = InMemoryRetrieveBackend::new();

    let mut wiki_v5 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Wiki,
        "Leave Policy",
        "Leave Policy wiki summary with updated FAQ",
        &["hr"],
    )?;
    wiki_v5.revision = 5;
    let wiki_v5_id = wiki_v5.id;

    let mut chunk_r2 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Leave Policy",
        "Leave Policy official document revision two",
        &["hr"],
    )?;
    chunk_r2.revision = 2;
    let chunk_r2_id = chunk_r2.id;

    let mut chunk_r1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "Leave Policy",
        "Leave Policy official document revision one",
        &["hr"],
    )?;
    chunk_r1.revision = 1;
    let chunk_r1_id = chunk_r1.id;

    let mut thumb_r1 = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Thumb,
        "Leave Policy",
        "Leave Policy scanned page revision one",
        &["hr"],
    )?;
    thumb_r1.revision = 1;
    let thumb_r1_id = thumb_r1.id;

    backend.insert(wiki_v5).await;
    backend.insert(chunk_r2).await;
    backend.insert(chunk_r1).await;
    backend.insert(thumb_r1).await;

    let retriever = InMemoryHybridRetriever::new(backend);

    let mut req = SearchRequest::new("Leave Policy");
    req.latest_revision_only = true;
    req.top_k = Some(10);
    let resp = retriever.search(&ctx, &req).await?;
    let hit_ids: Vec<Uuid> = resp.hits.iter().map(|h| h.id).collect();

    assert!(hit_ids.contains(&wiki_v5_id), "Wiki v5 should survive");
    assert!(
        hit_ids.contains(&chunk_r2_id),
        "Chunks r2 should NOT be suppressed by Wiki v5 sharing the same title"
    );
    assert!(
        !hit_ids.contains(&chunk_r1_id),
        "Chunks r1 should still be suppressed by Chunks r2"
    );
    assert!(
        !hit_ids.contains(&thumb_r1_id),
        "Thumb r1 shares Corpus space group and should still be suppressed by Chunks r2"
    );

    Ok(())
}

#[tokio::test]
async fn test_as_of_reference_time_and_validity_labels() -> Result<(), Box<dyn std::error::Error>> {
    use chrono::TimeZone;
    use openrag_retrieve::ValidityState;

    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let owner = Uuid::new_v4();
    let ctx = make_ctx(org_id, owner, dept_id, 5)?;
    let date = |y: i32, m: u32, d: u32| {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0)
            .single()
            .ok_or("bad date")
    };

    let backend = InMemoryRetrieveBackend::new();
    let text = "국내 출장 일비 기준 안내";
    let mut old = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "출장 규정",
        text,
        &[],
    )?;
    old.revision = 1;
    old.valid_from = date(2020, 1, 1)?;
    old.valid_to = Some(date(2024, 1, 1)?);
    let mut cur = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "출장 규정",
        text,
        &[],
    )?;
    cur.revision = 2;
    cur.valid_from = date(2024, 1, 1)?;
    let mut next = sample_record(
        org_id,
        dept_id,
        owner,
        RetrieveSpace::Chunks,
        "출장 규정",
        text,
        &[],
    )?;
    next.revision = 3;
    next.valid_from = Utc::now() + Duration::days(30);
    let (old_id, cur_id, next_id) = (old.id, cur.id, next.id);
    backend.insert(old).await;
    backend.insert(cur).await;
    backend.insert(next).await;
    let retriever = InMemoryHybridRetriever::new(backend);

    let search = |as_of: Option<chrono::DateTime<Utc>>, only: bool| {
        let mut req = SearchRequest::new("출장 일비 기준");
        req.as_of = as_of;
        req.latest_revision_only = only;
        req
    };
    let labels = |hits: &[openrag_retrieve::SearchHit]| -> HashMap<Uuid, ValidityState> {
        hits.iter().map(|h| (h.id, h.validity)).collect()
    };

    // Now, everything returned and labeled honestly.
    let all_now = labels(&retriever.search(&ctx, &search(None, false)).await?.hits);
    assert_eq!(all_now.get(&old_id), Some(&ValidityState::Expired));
    assert_eq!(all_now.get(&cur_id), Some(&ValidityState::Current));
    assert_eq!(all_now.get(&next_id), Some(&ValidityState::Scheduled));

    // Now, in force only.
    let in_force_now = labels(&retriever.search(&ctx, &search(None, true)).await?.hits);
    assert_eq!(
        in_force_now.keys().copied().collect::<Vec<_>>(),
        vec![cur_id]
    );

    // As of mid-2022 the 2020 revision was the one in force.
    let in_2022 = labels(
        &retriever
            .search(&ctx, &search(Some(date(2022, 6, 1)?), true))
            .await?
            .hits,
    );
    assert_eq!(in_2022.keys().copied().collect::<Vec<_>>(), vec![old_id]);
    assert_eq!(in_2022.get(&old_id), Some(&ValidityState::Current));

    // After the scheduled revision takes effect, rev 2 becomes an older revision.
    let later = Some(Utc::now() + Duration::days(60));
    let in_force_later = labels(&retriever.search(&ctx, &search(later, true)).await?.hits);
    assert_eq!(
        in_force_later.keys().copied().collect::<Vec<_>>(),
        vec![next_id]
    );
    let all_later = labels(&retriever.search(&ctx, &search(later, false)).await?.hits);
    assert_eq!(all_later.get(&cur_id), Some(&ValidityState::OlderRevision));
    assert_eq!(all_later.get(&next_id), Some(&ValidityState::Current));
    Ok(())
}
