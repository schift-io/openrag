#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header::AUTHORIZATION, Method, Request, StatusCode};
use http_body_util::BodyExt;
use openrag_core::{GuardAction, GuardRuleConfig, GuardRuleKind, GuardSurface, OpenRagError};
use openrag_memory::MemoryGuardHook;
use openrag_server::error::ApiError;
use openrag_server::state::{
    default_pipeline_config, resolve_master_kek_dev_default, PiiMemoryGuardHook,
};
use openrag_server::{build_router, AppState, OpenRagMode};
use serde_json::Value;
use tower::ServiceExt;
use uuid::Uuid;

#[tokio::test]
async fn health_endpoint_succeeds_unauthenticated_offline() -> anyhow::Result<()> {
    let state = AppState::new_offline_for_test()?;
    let app = build_router(state);

    let req = Request::builder()
        .method(Method::GET)
        .uri("/health")
        .body(Body::empty())?;
    let resp = app.oneshot(req).await?;
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await?.to_bytes();
    let payload: Value = serde_json::from_slice(&body)?;
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["mode"], "offline");
    assert_eq!(payload["db_connected"], false);
    Ok(())
}

#[tokio::test]
async fn protected_routes_fail_closed_401_without_or_with_bad_bearer_token() -> anyhow::Result<()> {
    let state = AppState::new_offline_for_test()?;
    let app = build_router(state);
    let doc_id = Uuid::new_v4();

    let protected_cases = [
        (Method::GET, "/v2/buckets"),
        (Method::POST, "/v2/buckets"),
        (Method::GET, "/v2/buckets/default"),
        (Method::GET, "/v2/buckets/default/documents"),
        (Method::POST, "/v2/buckets/default/documents"),
        (Method::POST, "/v2/buckets/default/search"),
        (Method::PUT, "/v1/buckets/default/documents/sync"),
        (Method::POST, "/v1/chat/completions"),
        (Method::POST, "/v1/eval/ab-search"),
        (Method::GET, "/v1/concepts"),
        (Method::POST, "/v1/concepts"),
        (Method::POST, "/v1/concepts/link"),
        (Method::POST, "/v1/concepts/resolve"),
    ];

    for (method, uri) in protected_cases {
        let req_no_auth = Request::builder()
            .method(method.clone())
            .uri(uri)
            .body(Body::empty())?;
        let resp_no_auth = app.clone().oneshot(req_no_auth).await?;
        assert_eq!(
            resp_no_auth.status(),
            StatusCode::UNAUTHORIZED,
            "expected 401 without Authorization on {method} {uri}"
        );

        let req_bad_auth = Request::builder()
            .method(method.clone())
            .uri(uri)
            .header(AUTHORIZATION, "Bearer orag_malformed_invalid_token")
            .body(Body::empty())?;
        let resp_bad_auth = app.clone().oneshot(req_bad_auth).await?;
        assert_eq!(
            resp_bad_auth.status(),
            StatusCode::UNAUTHORIZED,
            "expected 401 with invalid Bearer token on {method} {uri}"
        );
    }

    let tag_uri = format!("/v2/buckets/default/documents/{doc_id}/tags");
    let tag_req = Request::builder()
        .method(Method::GET)
        .uri(&tag_uri)
        .body(Body::empty())?;
    assert_eq!(app.clone().oneshot(tag_req).await?.status(), StatusCode::UNAUTHORIZED);

    for (method, uri) in [
        (Method::POST, format!("/v1/wiki/pages/{doc_id}/tags")),
        (Method::DELETE, format!("/v1/wiki/pages/{doc_id}/tags")),
        (Method::POST, format!("/v1/memories/{doc_id}/tags")),
        (Method::DELETE, format!("/v1/memories/{doc_id}/tags")),
    ] {
        let req = Request::builder().method(method).uri(&uri).body(Body::empty())?;
        assert_eq!(app.clone().oneshot(req).await?.status(), StatusCode::UNAUTHORIZED);
    }

    Ok(())
}

#[tokio::test]
async fn cli_code_exchange_rejects_blank_or_invalid_code() -> anyhow::Result<()> {
    let state = AppState::new_offline_for_test()?;
    let app = build_router(state);

    let blank_req = Request::builder()
        .method(Method::POST)
        .uri("/v1/auth/cli/code-exchange")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"code":"   "}"#))?;
    let blank_resp = app.clone().oneshot(blank_req).await?;
    assert_eq!(blank_resp.status(), StatusCode::BAD_REQUEST);

    let invalid_req = Request::builder()
        .method(Method::POST)
        .uri("/v1/auth/cli/code-exchange")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"code":"invalid-expired-code-999999"}"#))?;
    let invalid_resp = app.oneshot(invalid_req).await?;
    assert_eq!(invalid_resp.status(), StatusCode::UNAUTHORIZED);

    Ok(())
}

#[tokio::test]
async fn pii_memory_guard_hook_blocks_concept_linked_alias_and_obfuscation() -> anyhow::Result<()> {
    let state = AppState::new_offline_for_test()?;
    let org_id = Uuid::new_v4();

    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.pii.guard.rules.push(GuardRuleConfig {
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "사과".into(),
            action: GuardAction::Block,
            surfaces: vec![GuardSurface::InboundQuery, GuardSurface::IngestionGate],
            replacement_text: None,
            enabled: true,
            priority: 1,
        });
    }

    state
        .in_memory_concepts
        .link_terms(
            org_id,
            None,
            "사과",
            "apple",
            openrag_memory::ConceptCategory::SafewordAlias,
            openrag_memory::ConceptSource::Manual,
        )
        .await?;

    let hook = PiiMemoryGuardHook::new(
        None,
        Arc::clone(&state.pipeline_config),
        Arc::clone(&state.pii_engine),
        state.in_memory_concepts.clone(),
    );

    let blocked = hook
        .inspect(org_id, "Confidential ａｐｐｌｅ project", GuardSurface::IngestionGate)
        .await;
    assert!(matches!(
        blocked,
        Err(openrag_core::OpenRagError::GuardBlocked { .. })
    ));

    let allowed = hook
        .inspect(org_id, "application architecture review", GuardSurface::IngestionGate)
        .await?;
    assert_eq!(allowed.action, GuardAction::Allow);
    Ok(())
}

#[test]
fn mode_and_dev_kek_are_deterministic() -> anyhow::Result<()> {
    assert_eq!("offline".parse::<OpenRagMode>()?, OpenRagMode::Offline);
    assert_eq!("LIVE".parse::<OpenRagMode>()?, OpenRagMode::Live);
    assert_eq!("".parse::<OpenRagMode>()?, OpenRagMode::Auto);
    assert!("invalid-mode".parse::<OpenRagMode>().is_err());

    let kek1 = resolve_master_kek_dev_default();
    let kek2 = resolve_master_kek_dev_default();
    assert_eq!(kek1.as_bytes(), kek2.as_bytes());

    let cfg = default_pipeline_config();
    cfg.validate()?;
    Ok(())
}

#[test]
fn database_rls_and_unique_violation_map_to_403_and_409() {
    let rls_err = ApiError::from(OpenRagError::Database(
        "failed inserting corpus.documents: error returned from database: new row violates row-level security policy for table \"documents\"".into(),
    ));
    assert_eq!(rls_err.status, StatusCode::FORBIDDEN);
    assert_eq!(rls_err.code, "rls_denied");

    let code_err = ApiError::from(OpenRagError::Database("SQLSTATE 42501".into()));
    assert_eq!(code_err.status, StatusCode::FORBIDDEN);
    assert_eq!(code_err.code, "rls_denied");

    let dup_err = ApiError::from(OpenRagError::Database(
        "duplicate key value violates unique constraint \"uniq\"".into(),
    ));
    assert_eq!(dup_err.status, StatusCode::CONFLICT);
    assert_eq!(dup_err.code, "conflict");
}

#[tokio::test]
async fn app_and_studio_routes_fail_closed_401_without_or_with_bad_bearer_token() -> anyhow::Result<()>
{
    let state = AppState::new_offline_for_test()?;
    let app = build_router(state);
    let sample_id = Uuid::new_v4();
    let sample_alias_id = Uuid::new_v4();

    let protected_cases = [
        (Method::POST, "/api/app/chat".to_string()),
        (Method::POST, "/api/app/upload".to_string()),
        (
            Method::GET,
            format!("/api/app/pages/{sample_id}/thumbnail"),
        ),
        (Method::POST, "/api/app/feedback".to_string()),
        (Method::GET, "/api/app/wiki".to_string()),
        (Method::GET, format!("/api/app/wiki/{sample_id}")),
        (Method::GET, "/api/app/memory".to_string()),
        (Method::POST, "/api/app/memory/patch".to_string()),
        (Method::POST, "/api/app/memory/retract".to_string()),
        (Method::POST, "/api/app/memory/promote".to_string()),
        (Method::GET, "/api/app/connect".to_string()),
        (Method::GET, "/api/app/concepts".to_string()),
        (Method::POST, "/api/app/concepts".to_string()),
        (Method::POST, "/api/app/concepts/link".to_string()),
        (Method::GET, format!("/api/app/concepts/{sample_id}")),
        (Method::POST, "/api/app/concepts/resolve".to_string()),
        (Method::GET, "/api/studio/pipeline".to_string()),
        (Method::PUT, "/api/studio/pipeline".to_string()),
        (Method::POST, "/api/studio/pipeline/test".to_string()),
        (Method::GET, "/api/studio/index/engines".to_string()),
        (Method::PUT, "/api/studio/index/engines".to_string()),
        (Method::GET, "/api/studio/index/outbox".to_string()),
        (
            Method::POST,
            format!("/api/studio/index/quarantine/{sample_id}/retry"),
        ),
        (Method::POST, "/api/studio/iam/orgs".to_string()),
        (Method::POST, "/api/studio/iam/principals".to_string()),
        (Method::POST, "/api/studio/iam/api-keys".to_string()),
        (Method::GET, "/api/studio/iam/groups".to_string()),
        (Method::POST, "/api/studio/iam/groups".to_string()),
        (
            Method::POST,
            format!("/api/studio/iam/groups/{sample_id}/members"),
        ),
        (
            Method::DELETE,
            format!("/api/studio/iam/groups/{sample_id}/members/{sample_alias_id}"),
        ),
        (
            Method::POST,
            "/api/studio/iam/resource-bindings".to_string(),
        ),
        (Method::POST, "/api/studio/iam/policies".to_string()),
        (Method::POST, "/api/studio/iam/simulate".to_string()),
        (Method::GET, "/api/studio/pii/profiles".to_string()),
        (Method::PUT, "/api/studio/pii/profiles".to_string()),
        (Method::GET, "/api/studio/pii/guard-rules".to_string()),
        (Method::POST, "/api/studio/pii/guard-rules".to_string()),
        (
            Method::PUT,
            format!("/api/studio/pii/guard-rules/{sample_id}"),
        ),
        (
            Method::DELETE,
            format!("/api/studio/pii/guard-rules/{sample_id}"),
        ),
        (Method::GET, "/api/studio/pii/guard-policies".to_string()),
        (Method::PUT, "/api/studio/pii/guard-policies".to_string()),
        (Method::PUT, "/api/studio/pii/guard-policy".to_string()),
        (Method::GET, "/api/studio/pii/guard-events".to_string()),
        (Method::POST, "/api/studio/pii/guard-test".to_string()),
        (Method::GET, "/api/studio/pii/redactions".to_string()),
        (Method::POST, "/api/studio/pii/redactions".to_string()),
        (Method::GET, "/api/studio/pii/audit-logs".to_string()),
        (Method::GET, "/api/studio/gateway/providers".to_string()),
        (Method::POST, "/api/studio/gateway/routing-rules".to_string()),
        (Method::GET, "/api/studio/gateway/logs".to_string()),
        (Method::GET, "/api/studio/eval/golden-sets".to_string()),
        (Method::POST, "/api/studio/eval/golden-sets".to_string()),
        (
            Method::GET,
            format!("/api/studio/eval/golden-sets/{sample_id}/items"),
        ),
        (
            Method::POST,
            format!("/api/studio/eval/golden-sets/{sample_id}/items"),
        ),
        (
            Method::PATCH,
            format!("/api/studio/eval/golden-items/{sample_id}"),
        ),
        (
            Method::POST,
            format!("/api/studio/eval/feedback/{sample_id}/promote"),
        ),
        (Method::POST, "/api/studio/eval/bench".to_string()),
        (Method::POST, "/api/studio/eval/optimize".to_string()),
        (Method::GET, "/api/studio/eval/leaderboard".to_string()),
        (Method::GET, "/api/studio/eval/report".to_string()),
        (Method::POST, "/api/studio/eval/promote".to_string()),
        (Method::GET, "/api/studio/concepts".to_string()),
        (Method::POST, "/api/studio/concepts".to_string()),
        (Method::POST, "/api/studio/concepts/link".to_string()),
        (
            Method::DELETE,
            format!("/api/studio/concepts/{sample_id}"),
        ),
        (
            Method::DELETE,
            format!("/api/studio/concepts/{sample_id}/aliases/{sample_alias_id}"),
        ),
        (Method::GET, "/api/studio/wiki/conflicts".to_string()),
        (Method::GET, "/api/studio/plugins/connectors".to_string()),
        (Method::GET, "/api/studio/knowledge/tags".to_string()),
    ];

    for (method, uri) in protected_cases {
        let req_no_auth = Request::builder()
            .method(method.clone())
            .uri(&uri)
            .body(Body::empty())?;
        let resp_no_auth = app.clone().oneshot(req_no_auth).await?;
        assert_eq!(
            resp_no_auth.status(),
            StatusCode::UNAUTHORIZED,
            "expected 401 without Authorization on {method} {uri}"
        );

        let req_bad_auth = Request::builder()
            .method(method.clone())
            .uri(&uri)
            .header(AUTHORIZATION, "Bearer orag_invalid_token_m4c2")
            .body(Body::empty())?;
        let resp_bad_auth = app.clone().oneshot(req_bad_auth).await?;
        assert_eq!(
            resp_bad_auth.status(),
            StatusCode::UNAUTHORIZED,
            "expected 401 with bad Bearer token on {method} {uri}"
        );
    }

    Ok(())
}

#[tokio::test]
async fn active_memory_router_tracks_runtime_pipeline_backend_updates() -> anyhow::Result<()> {
    let state = AppState::new_offline_for_test()?;
    let router_default = state.active_memory_router().await?;
    assert_eq!(
        router_default.primary_kind(),
        openrag_core::MemoryBackendKind::Kumiho
    );

    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.memory.backend = openrag_core::MemoryBackendKind::OpenragPg;
        cfg.memory.offline_fallback = true;
    }

    let router_fallback = state.active_memory_router().await?;
    assert_eq!(
        router_fallback.primary_kind(),
        openrag_core::MemoryBackendKind::Kumiho
    );

    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.memory.offline_fallback = false;
    }
    assert!(
        state.active_memory_router().await.is_err(),
        "openrag_pg without PG pool and offline_fallback=false must fail closed"
    );
    Ok(())
}

#[tokio::test]
async fn thumbnail_store_adapter_roundtrip_and_mask() -> anyhow::Result<()> {
    use openrag_core::BBoxNorm1000;
    use openrag_parse::{
        DocumentParser, InMemoryBlobStore, ParsePolicyCtx, ThumbnailBlobStore, Thumbnailer,
    };
    use openrag_pii::{mask_image_bboxes, LocalThumbnailStore, ThumbnailStore};
    use openrag_server::api_studio::BlobThumbnailStoreAdapter;

    let blob_store: Arc<dyn ThumbnailBlobStore> = Arc::new(InMemoryBlobStore::default());
    let adapter = BlobThumbnailStoreAdapter(Arc::clone(&blob_store));
    let cfg = default_pipeline_config();

    let parsed = DocumentParser::new()
        .parse(
            b"Confidential Report Body\nTop secret metric 12345",
            Some("text/plain"),
            Some("report.txt"),
            &cfg.parse,
            &ParsePolicyCtx::default(),
        )
        .await?;
    let thumbs = Thumbnailer::new().render(&parsed, None, &cfg.thumbnail)?;
    let orig_bytes = thumbs
        .first()
        .map(|t| t.webp_bytes.clone())
        .ok_or_else(|| anyhow::anyhow!("missing rendered thumbnail"))?;

    let orig_uri = blob_store.put_bytes("org/doc/page_1.webp", &orig_bytes).await?;
    let loaded = adapter.read_bytes(&orig_uri).await?;
    assert_eq!(loaded, orig_bytes);

    let masked = mask_image_bboxes(
        &loaded,
        &[BBoxNorm1000 {
            x_min: 0,
            y_min: 0,
            x_max: 1000,
            y_max: 1000,
        }],
    )?;
    let redacted_uri = LocalThumbnailStore::derive_redacted_uri(&orig_uri);
    assert!(redacted_uri.ends_with(".redacted.webp"));
    adapter.write_bytes(&redacted_uri, &masked).await?;
    assert_eq!(blob_store.get_bytes(&redacted_uri).await?, masked);
    assert_ne!(masked, orig_bytes, "full blackout must change the image");

    Ok(())
}

#[test]
fn studio_pipeline_yaml_rejects_invalid_fields_even_with_memory_section() -> anyhow::Result<()> {
    use openrag_core::MemoryBackendKind;
    use openrag_server::api_studio::parse_studio_pipeline_yaml;

    let valid_minimal = "version: 1\nprofile: standard\nmemory:\n  backend: kumiho\n";
    let (parsed_min, _) = parse_studio_pipeline_yaml(valid_minimal)?;
    assert_eq!(parsed_min.memory.backend, MemoryBackendKind::Kumiho);

    let bad_dim = "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nstore:\n  dim: 128\n";
    assert!(
        parse_studio_pipeline_yaml(bad_dim).is_err(),
        "store.dim=128 alongside memory: must be rejected"
    );

    let bad_alpha_high =
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nretrieve:\n  alpha: 1.5\n";
    assert!(
        parse_studio_pipeline_yaml(bad_alpha_high).is_err(),
        "retrieve.alpha=1.5 alongside memory: must be rejected"
    );

    let bad_alpha_neg =
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nretrieve:\n  alpha: -0.1\n";
    assert!(
        parse_studio_pipeline_yaml(bad_alpha_neg).is_err(),
        "retrieve.alpha=-0.1 alongside memory: must be rejected"
    );

    let bad_auth_low = "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nllm:\n  local_only_min_auth_level: 0\n";
    assert!(
        parse_studio_pipeline_yaml(bad_auth_low).is_err(),
        "llm.local_only_min_auth_level=0 alongside memory: must be rejected"
    );

    let bad_auth_high = "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nllm:\n  local_only_min_auth_level: 11\n";
    assert!(
        parse_studio_pipeline_yaml(bad_auth_high).is_err(),
        "llm.local_only_min_auth_level=11 alongside memory: must be rejected"
    );

    let bad_guard_action = "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\npii:\n  guard:\n    default_action: nuke\n";
    assert!(
        parse_studio_pipeline_yaml(bad_guard_action).is_err(),
        "pii.guard.default_action=nuke alongside memory: must be rejected"
    );

    Ok(())
}

#[test]
fn neutralize_literal_pii_tokens_is_case_insensitive() {
    use openrag_server::llm_router::neutralize_literal_pii_tokens;

    let input = "upper=[PII:RRN_KR:1234abcd] lower_rrn=[PII:rrn_kr:1234abcd] lower_phone=[PII:phone_kr:abcd1234] mixed=[pii:Email:DeAdBeEf]";
    let out = neutralize_literal_pii_tokens(input);
    assert_eq!(
        out,
        "upper=[PII-LITERAL:RRN_KR:1234abcd] lower_rrn=[PII-LITERAL:rrn_kr:1234abcd] lower_phone=[PII-LITERAL:phone_kr:abcd1234] mixed=[PII-LITERAL:Email:DeAdBeEf]"
    );
    assert!(!out.contains("[PII:"));
    assert!(!out.contains("[pii:"));
}

#[tokio::test]
async fn llm_completion_respects_requested_spaces() -> anyhow::Result<()> {
    use openrag_core::{AccessScope, AuthLevel, PrincipalContext};
    use openrag_retrieve::{InMemoryDocRecord, RetrieveSpace};
    use openrag_server::llm_router::{run_chat_completion, LlmCompletionInput};

    let state = AppState::new_offline_for_test()?;
    let org_id = Uuid::new_v4();
    let dept_id = Uuid::new_v4();
    let principal_id = Uuid::new_v4();
    let mut principal = PrincipalContext::new(org_id, principal_id, AuthLevel::new(5)?);
    principal.dept_ids.push(dept_id);

    let chunk_id = Uuid::new_v4();
    let wiki_id = Uuid::new_v4();

    state
        .in_memory_retrieve
        .backend()
        .insert(InMemoryDocRecord {
            id: chunk_id,
            document_id: Some(Uuid::new_v4()),
            page_id: None,
            org_id,
            bucket_id: None,
            dept_id,
            owner_principal_id: principal_id,
            access_scope: AccessScope::Dept,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: 1,
            space: RetrieveSpace::Chunks,
            title: Some("Chunk Hit".into()),
            text: "SpacesFilterToken chunk evidence alpha".into(),
            tags: Vec::new(),
            document_tags: None,
            revision: 1,
            valid_from: chrono::Utc::now(),
            valid_to: None,
            is_archived: false,
            is_superseded: false,
            thumb_uri: None,
            bbox_norm_1000: None,
            acl_scope_key: "scope".into(),
            embedding: None,
        })
        .await;

    state
        .in_memory_retrieve
        .backend()
        .insert(InMemoryDocRecord {
            id: wiki_id,
            document_id: None,
            page_id: None,
            org_id,
            bucket_id: None,
            dept_id,
            owner_principal_id: principal_id,
            access_scope: AccessScope::Dept,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: 1,
            space: RetrieveSpace::Wiki,
            title: Some("Wiki Hit".into()),
            text: "SpacesFilterToken wiki evidence beta".into(),
            tags: Vec::new(),
            document_tags: None,
            revision: 1,
            valid_from: chrono::Utc::now(),
            valid_to: None,
            is_archived: false,
            is_superseded: false,
            thumb_uri: None,
            bbox_norm_1000: None,
            acl_scope_key: "scope".into(),
            embedding: None,
        })
        .await;

    let wiki_only = run_chat_completion(LlmCompletionInput {
        state: &state,
        principal: &principal,
        requested_model: "openrag-auto",
        user_prompt: "SpacesFilterToken",
        bucket_ref: None,
        spaces: Some(vec![RetrieveSpace::Wiki]),
        include_citations: true,
        unmask_pii: false,
        allow_unmasked_pii: false,
        keep_pii_tokens: false,
        admission_threshold: Some(0.0),
        top_k: Some(5),
        latest_revision_only: false,
        as_of: None,
    })
    .await?;
    assert!(!wiki_only.citations.is_empty());
    assert!(wiki_only
        .citations
        .iter()
        .all(|c| c.space == "wiki"));

    let chunks_only = run_chat_completion(LlmCompletionInput {
        state: &state,
        principal: &principal,
        requested_model: "openrag-auto",
        user_prompt: "SpacesFilterToken",
        bucket_ref: None,
        spaces: Some(vec![RetrieveSpace::Chunks]),
        include_citations: true,
        unmask_pii: false,
        allow_unmasked_pii: false,
        keep_pii_tokens: false,
        admission_threshold: Some(0.0),
        top_k: Some(5),
        latest_revision_only: false,
        as_of: None,
    })
    .await?;
    assert!(!chunks_only.citations.is_empty());
    assert!(chunks_only
        .citations
        .iter()
        .all(|c| c.space == "chunks"));

    Ok(())
}
