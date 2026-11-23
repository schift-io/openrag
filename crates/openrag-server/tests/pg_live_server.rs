#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

use std::io::{Cursor, Write};
use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header::AUTHORIZATION, header::CONTENT_TYPE, Method, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, ExternalMirrorBackend, GuardAction,
    GuardRuleConfig, GuardRuleKind, GuardSurface, OpenRagError, PipelineProfile, PrincipalContext,
};
use openrag_db::DbPool;
use openrag_memory::{DeterministicTextEmbedder, MemoryKind, ReflectRequest};
use openrag_parse::sha256_hex_str;
use openrag_pii::GuardRepo;
use openrag_server::llm_router::LlmHttpTransport;
use openrag_server::state::{ensure_default_guard_policy, ensure_migrations_if_needed};
use openrag_server::{build_router, AppState, OpenRagMode};
use openrag_store::{PgPineconeManager, VectorTable};
use pgvector::Vector;
use regex::Regex;
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

struct EchoVllmTransport;

#[async_trait]
impl LlmHttpTransport for EchoVllmTransport {
    async fn post_json(
        &self,
        _url: &str,
        _bearer: Option<&str>,
        body: &Value,
    ) -> Result<(u16, String), OpenRagError> {
        let model = body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("Qwen/Qwen2.5-32B-Instruct");
        let sys = body
            .get("messages")
            .and_then(|m| m.get(0))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let user = body
            .get("messages")
            .and_then(|m| m.get(1))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let content = format!("Mock answer ({model}): {sys} || USER: {user}");
        Ok((
            200,
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }]
            })
            .to_string(),
        ))
    }
}

struct SeededOrg {
    org_id: Uuid,
    slug: String,
    dept_id: Uuid,
    dept2_id: Uuid,
    admin_id: Uuid,
    owner_id: Uuid,
    peer_id: Uuid,
    l7_id: Uuid,
    l8_id: Uuid,
    other_org_id: Uuid,
    other_admin_id: Uuid,
}

/// The two gateway providers these tests route between. OpenRAG seeds none on its own: providers
/// come from the environment (`OPENRAG_LOCAL_LLM_*`, `OPENRAG_LLM_CLI`).
async fn seed_test_providers(worker: &DbPool, org_id: Uuid) -> anyhow::Result<()> {
    use openrag_server::llm_router::{seed_providers, ProviderSeed};
    seed_providers(
        worker,
        org_id,
        &[
            ProviderSeed {
                name: "openrouter".into(),
                provider_type: "openrouter".into(),
                base_url: "https://openrouter.ai/api/v1".into(),
                model_id: "anthropic/claude-3.5-sonnet".into(),
                is_on_premise: false,
                max_auth_level: 7,
                allow_unmasked_pii: false,
                config: json!({ "api_key_env": "OPENROUTER_API_KEY" }),
            },
            ProviderSeed {
                name: "vllm_local".into(),
                provider_type: "vllm_local".into(),
                base_url: "http://127.0.0.1:8000/v1".into(),
                model_id: "Qwen/Qwen2.5-32B-Instruct".into(),
                is_on_premise: true,
                max_auth_level: 10,
                allow_unmasked_pii: true,
                config: json!({ "api_key_env": "VLLM_API_KEY" }),
            },
        ],
    )
    .await?;
    Ok(())
}

async fn seed_org(worker_pool: &DbPool) -> anyhow::Result<SeededOrg> {
    let org_id = Uuid::new_v4();
    let other_org_id = Uuid::new_v4();
    let slug = format!("m4c1-{}", &org_id.simple().to_string()[..8]);
    let other_slug = format!("m4c1-other-{}", &other_org_id.simple().to_string()[..8]);
    let dept_id = Uuid::new_v4();
    let dept2_id = Uuid::new_v4();
    let other_dept_id = Uuid::new_v4();
    let admin_id = Uuid::new_v4();
    let owner_id = Uuid::new_v4();
    let peer_id = Uuid::new_v4();
    let l7_id = Uuid::new_v4();
    let l8_id = Uuid::new_v4();
    let other_admin_id = Uuid::new_v4();

    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, $3), ($4, $5, $6)")
        .bind(org_id)
        .bind(&slug)
        .bind(format!("Org {slug}"))
        .bind(other_org_id)
        .bind(&other_slug)
        .bind(format!("Org {other_slug}"))
        .execute(worker_pool.inner())
        .await?;
    seed_test_providers(worker_pool, org_id).await?;
    seed_test_providers(worker_pool, other_org_id).await?;

    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
         ($1, $2, 'Dept1', 'dept1'),
         ($3, $2, 'Dept2', 'dept2'),
         ($4, $5, 'OtherDept', 'other')",
    )
    .bind(dept_id)
    .bind(org_id)
    .bind(dept2_id)
    .bind(other_dept_id)
    .bind(other_org_id)
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $2, $3, 'Admin', 10, true, true),
         ($4, $2, $5, 'OwnerL3', 3, false, false),
         ($6, $2, $7, 'PeerL3', 3, false, false),
         ($8, $2, $9, 'UserL7', 7, false, false),
         ($10, $2, $11, 'UserL8', 8, true, false),
         ($12, $13, $14, 'OtherAdmin', 10, true, true)",
    )
    .bind(admin_id)
    .bind(org_id)
    .bind(format!("admin-{admin_id}@test.local"))
    .bind(owner_id)
    .bind(format!("owner-{owner_id}@test.local"))
    .bind(peer_id)
    .bind(format!("peer-{peer_id}@test.local"))
    .bind(l7_id)
    .bind(format!("l7-{l7_id}@test.local"))
    .bind(l8_id)
    .bind(format!("l8-{l8_id}@test.local"))
    .bind(other_admin_id)
    .bind(other_org_id)
    .bind(format!("other-{other_admin_id}@test.local"))
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'dept_admin'),
         ($1, $2, $4, 'member'),
         ($1, $2, $5, 'member'),
         ($1, $2, $6, 'member'),
         ($1, $2, $7, 'member'),
         ($1, $8, $3, 'dept_admin'),
         ($9, $10, $11, 'dept_admin')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(owner_id)
    .bind(peer_id)
    .bind(l7_id)
    .bind(l8_id)
    .bind(dept2_id)
    .bind(other_org_id)
    .bind(other_dept_id)
    .bind(other_admin_id)
    .execute(worker_pool.inner())
    .await?;

    Ok(SeededOrg {
        org_id,
        slug,
        dept_id,
        dept2_id,
        admin_id,
        owner_id,
        peer_id,
        l7_id,
        l8_id,
        other_org_id,
        other_admin_id,
    })
}

async fn send_json(
    app: &Router,
    method: Method,
    uri: &str,
    token: Option<&str>,
    body: Option<Value>,
) -> anyhow::Result<(StatusCode, Value, String)> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(tok) = token {
        builder = builder.header(AUTHORIZATION, format!("Bearer {tok}"));
    }
    let req = if let Some(b) = body {
        builder = builder.header(CONTENT_TYPE, "application/json");
        builder.body(Body::from(serde_json::to_vec(&b)?))?
    } else {
        builder.body(Body::empty())?
    };
    let resp = app.clone().oneshot(req).await?;
    let status = resp.status();
    let bytes = resp.into_body().collect().await?.to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let parsed = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
    Ok((status, parsed, text))
}

async fn mint_key(
    app: &Router,
    worker_pool: &DbPool,
    org_id: Uuid,
    principal_id: Uuid,
    max_auth_level: Option<i16>,
    scoped_arns: Vec<String>,
) -> anyhow::Result<String> {
    let issued = openrag_server::auth::issue_cli_code(
        worker_pool,
        org_id,
        principal_id,
        None,
        openrag_server::auth::CliCodeCaps::default(),
        None,
    )
    .await?;
    let mut payload = json!({
        "code": issued.code,
        "org_id": org_id,
        "principal_id": principal_id,
        "scoped_arns": scoped_arns,
    });
    if let Some(lvl) = max_auth_level {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("max_auth_level".into(), json!(lvl));
        }
    }
    let (status, body, text) = send_json(
        app,
        Method::POST,
        "/v1/auth/cli/code-exchange",
        None,
        Some(payload),
    )
    .await?;
    assert_eq!(status, StatusCode::CREATED, "code-exchange failed: {text}");
    let tok = body["api_key"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing api_key in {body}"))?;
    Ok(tok.to_string())
}

fn build_hwpx_bytes(title: &str, paragraph: &str) -> anyhow::Result<Vec<u8>> {
    let buf = Cursor::new(Vec::new());
    let mut zw = zip::ZipWriter::new(buf);
    let opts_stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let opts_deflate = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zw.start_file("mimetype", opts_stored)?;
    zw.write_all(b"application/hwp+zip")?;

    let section = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <hs:sec xmlns:hs=\"http://www.hancom.co.kr/hwpml/2011/section\" \
         xmlns:hp=\"http://www.hancom.co.kr/hwpml/2011/paragraph\">\
         <hp:p><hp:run><hp:t>{title}</hp:t></hp:run></hp:p>\
         <hp:p><hp:run><hp:t>{paragraph}</hp:t></hp:run></hp:p>\
         </hs:sec>"
    );
    zw.start_file("Contents/section0.xml", opts_deflate)?;
    zw.write_all(section.as_bytes())?;
    let finished = zw.finish()?;
    Ok(finished.into_inner())
}

fn build_multipart_body(
    boundary: &str,
    fields: &[(&str, &str)],
    file: Option<(&str, &str, &[u8])>,
) -> Vec<u8> {
    let mut body = Vec::new();
    for (k, v) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes(),
        );
    }
    if let Some((filename, ct, bytes)) = file {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {ct}\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c1_end_to_end_core_http_router() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();

    // 1. Idempotent startup migration probe on psql-migrated DB returns false (skipped cleanly)
    let migrated_again = ensure_migrations_if_needed(&worker_pool).await?;
    assert!(!migrated_again, "psql-migrated DB should not re-run migrations");

    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;

    // 2. GET /health -> 200 OK
    let (h_status, h_json, _) = send_json(&app, Method::GET, "/health", None, None).await?;
    assert_eq!(h_status, StatusCode::OK);
    assert_eq!(h_json["status"], "ok");
    assert_eq!(h_json["db_connected"], true);

    // 3. Code exchange + revoked/expired/malformed 401 checks
    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;
    let peer_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;
    let l7_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.l7_id, None, vec![]).await?;
    let l8_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.l8_id, None, vec![]).await?;
    let other_key = mint_key(&app, &worker_pool, seeded.other_org_id, seeded.other_admin_id, None, vec![]).await?;

    // Revoked key -> 401
    let revoked_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;
    sqlx::query("UPDATE iam.api_keys SET revoked_at = now() WHERE key_hash = $1")
        .bind(openrag_db::sha256_hex(&revoked_key))
        .execute(worker_pool.inner())
        .await?;
    let (rev_status, _, _) =
        send_json(&app, Method::GET, "/v2/buckets", Some(&revoked_key), None).await?;
    assert_eq!(rev_status, StatusCode::UNAUTHORIZED);

    // Expired key -> 401
    let expired_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;
    sqlx::query("UPDATE iam.api_keys SET expires_at = now() - interval '1 hour' WHERE key_hash = $1")
        .bind(openrag_db::sha256_hex(&expired_key))
        .execute(worker_pool.inner())
        .await?;
    let (exp_status, _, _) =
        send_json(&app, Method::GET, "/v2/buckets", Some(&expired_key), None).await?;
    assert_eq!(exp_status, StatusCode::UNAUTHORIZED);

    // Scoped ARN & max_auth_level ceiling on API key
    let scoped_key = mint_key(
        &app, &worker_pool,
        seeded.org_id,
        seeded.l8_id,
        Some(4),
        vec![format!("arn:openrag:{}:dept1:*", seeded.slug)],
    )
    .await?;
    let (dept2_bkt_status, dept2_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("d2-{}", Uuid::new_v4().simple()),
            "name": "Dept2",
            "dept_id": seeded.dept2_id,
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(dept2_bkt_status, StatusCode::CREATED);
    let dept2_bid = dept2_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing dept2 bucket id"))?;
    let (scoped_get_status, _, _) = send_json(
        &app,
        Method::GET,
        &format!("/v2/buckets/{dept2_bid}"),
        Some(&scoped_key),
        None,
    )
    .await?;
    assert!(
        scoped_get_status == StatusCode::FORBIDDEN || scoped_get_status == StatusCode::NOT_FOUND
    );

    // 4. Bucket CRUD + Multipart HWPX ingest + Parenthetical Concept extraction + Search + Delete
    let bkt_slug = format!("core-{}", &Uuid::new_v4().simple().to_string()[..8]);
    let (b_status, b_json, b_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": bkt_slug,
            "name": "Core Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(b_status, StatusCode::CREATED, "create bucket failed: {b_txt}");
    let bid = b_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket id"))?;

    // Empty slug -> 400
    let (es_status, _, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({"slug": "   ", "name": "Empty"})),
    )
    .await?;
    assert_eq!(es_status, StatusCode::BAD_REQUEST);

    // Multipart HWPX upload with parenthetical 한글(English) + tags
    let hwpx = build_hwpx_bytes("계약서", "사과(apple) 공급망 AlphaToken987 본문")?;
    let boundary = "----OpenRagBoundaryM4c1";
    let mp_body = build_multipart_body(
        boundary,
        &[
            ("access_scope", "dept"),
            ("min_auth_level", "3"),
            ("tags", "[\"procurement\",\"2026\"]"),
        ],
        Some(("contract.hwpx", "application/hwp+zip", &hwpx)),
    );
    let mp_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/v2/buckets/{bid}/documents"))
        .header(AUTHORIZATION, format!("Bearer {admin_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(mp_body))?;
    let mp_resp = app.clone().oneshot(mp_req).await?;
    assert_eq!(mp_resp.status(), StatusCode::CREATED);
    let mp_bytes = mp_resp.into_body().collect().await?.to_bytes();
    let mp_json: Value = serde_json::from_slice(&mp_bytes)?;
    let hwpx_doc_id = mp_json["document_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing document_id"))?;

    // Verify zdr + batch=true -> 400
    let zdr_body = build_multipart_body(
        boundary,
        &[("access_scope", "dept"), ("mode", "zdr"), ("batch", "true")],
        Some(("test.txt", "text/plain", b"hello")),
    );
    let zdr_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/v2/buckets/{bid}/documents"))
        .header(AUTHORIZATION, format!("Bearer {admin_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(zdr_body))?;
    let zdr_resp = app.clone().oneshot(zdr_req).await?;
    assert_eq!(zdr_resp.status(), StatusCode::BAD_REQUEST);

    // Verify L8 + mode=share never persists/returns share
    let l8_body = build_multipart_body(
        boundary,
        &[
            ("access_scope", "dept"),
            ("min_auth_level", "8"),
            ("sensitivity", "confidential"),
            ("mode", "share"),
        ],
        Some(("l8.txt", "text/plain", b"classified L8")),
    );
    let l8_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/v2/buckets/{bid}/documents"))
        .header(AUTHORIZATION, format!("Bearer {admin_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(l8_body))?;
    let l8_resp = app.clone().oneshot(l8_req).await?;
    assert_eq!(l8_resp.status(), StatusCode::CREATED);
    let l8_json: Value = serde_json::from_slice(&l8_resp.into_body().collect().await?.to_bytes())?;
    assert_ne!(l8_json["ocr_mode"], "share");

    // Verify corrupt HWPX -> 400
    let bad_body = build_multipart_body(
        boundary,
        &[("access_scope", "dept")],
        Some(("broken.hwpx", "application/hwp+zip", b"PK\x03\x04corrupt")),
    );
    let bad_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/v2/buckets/{bid}/documents"))
        .header(AUTHORIZATION, format!("Bearer {admin_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(bad_body))?;
    let bad_resp = app.clone().oneshot(bad_req).await?;
    assert!(
        bad_resp.status() == StatusCode::BAD_REQUEST
            || bad_resp.status() == StatusCode::UNPROCESSABLE_ENTITY
    );

    // Verify S3 Vectors 2KB filterable cap -> 400
    let big_val = "x".repeat(2500);
    let (s3_status, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "S3 Overflow",
            "text": "small text",
            "external_backend": "s3_vectors",
            "filterable_metadata": { "oversized": big_val },
        })),
    )
    .await?;
    assert_eq!(s3_status, StatusCode::BAD_REQUEST);

    // Search via slug with inline #tag and concept-expanded query, verify both results+hits and facets.tags
    let (sr_status, sr_json, sr_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bkt_slug}/search"),
        Some(&admin_key),
        Some(json!({
            "query": "apple #procurement",
            "top_k": 5,
        })),
    )
    .await?;
    assert_eq!(sr_status, StatusCode::OK, "search failed: {sr_txt}");
    assert!(sr_json["results"].is_array());
    assert!(sr_json["hits"].is_array());
    assert!(sr_json["facets"]["tags"].is_object());
    assert!(sr_txt.contains("AlphaToken987"));

    // Invalid alpha -> 400
    let (bad_sr_status, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/search"),
        Some(&admin_key),
        Some(json!({"query": "test", "alpha": 1.5})),
    )
    .await?;
    assert_eq!(bad_sr_status, StatusCode::BAD_REQUEST);

    // Delete HWPX doc -> subsequent search must not return hwpx_doc_id
    let (del_doc_status, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v2/buckets/{bid}/documents/{hwpx_doc_id}"),
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(del_doc_status, StatusCode::OK);
    let (sr_after_status, _, sr_after_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/search"),
        Some(&admin_key),
        Some(json!({"query": "AlphaToken987", "top_k": 5})),
    )
    .await?;
    assert_eq!(sr_after_status, StatusCode::OK);
    assert!(!sr_after_txt.contains(hwpx_doc_id));

    // 5. Sync endpoint: Wiki SHA256 No-Op + Corpus page-delta + 409 conflict + empty body 400
    let wiki_body = "# Sync Wiki\n본문 내용";
    let wiki_sha = openrag_store::sha256_hex(wiki_body.as_bytes());
    let sync_wiki_payload = json!({
        "documents": [{
            "path": "sync-noop.md",
            "slug": "sync-noop",
            "title": "Sync NoOp",
            "body_md": wiki_body,
            "body_sha256": wiki_sha,
        }]
    });
    let (sw1_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(sync_wiki_payload.clone()),
    )
    .await?;
    assert_eq!(sw1_status, StatusCode::OK);

    let wm_before: i64 = sqlx::query_scalar(
        "SELECT COALESCE(SUM(indexed_count), 0)::bigint FROM indexing.watermarks WHERE org_id = $1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;

    let (sw2_status, sw2_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(sync_wiki_payload),
    )
    .await?;
    assert_eq!(sw2_status, StatusCode::OK);
    assert!(sw2_json["skipped"].as_u64().unwrap_or(0) >= 1);

    let wm_after: i64 = sqlx::query_scalar(
        "SELECT COALESCE(SUM(indexed_count), 0)::bigint FROM indexing.watermarks WHERE org_id = $1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(wm_before, wm_after);

    // Stale base_sha256 -> 409 Conflict
    let (conflict_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "sync-noop",
                "body_md": "# Changed",
                "base_sha256": "0000dead",
            }]
        })),
    )
    .await?;
    assert_eq!(conflict_status, StatusCode::CONFLICT);

    // Empty body_md -> 400 (never 500)
    let (empty_sync_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({"documents": [{"slug": "empty-doc", "body_md": ""}]})),
    )
    .await?;
    assert_eq!(empty_sync_status, StatusCode::BAD_REQUEST);

    // Corpus page-level delta update (< 3 pages updated when 1 of 3 changes)
    let (cp1_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "path": "delta.pdf",
                "slug": "delta-doc",
                "pages": ["Page 1 same", "Page 2 old", "Page 3 same"],
                "content_sha256": "sha-v1",
            }]
        })),
    )
    .await?;
    assert_eq!(cp1_status, StatusCode::OK);

    let (cp2_status, cp2_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "path": "delta.pdf",
                "slug": "delta-doc",
                "pages": ["Page 1 same", "Page 2 updated", "Page 3 same"],
                "content_sha256": "sha-v2",
            }]
        })),
    )
    .await?;
    assert_eq!(cp2_status, StatusCode::OK);
    assert_eq!(cp2_json["delta_updated"], 1);
    assert_eq!(cp2_json["upserted_chunks"], 1);

    // 6. M7 Policy LLM Router + Chat Completions (JSON + SSE + L7 vs L8 + Guard block zero-log)
    let (_, b7_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&l7_key),
        Some(json!({"slug": format!("b7-{}", Uuid::new_v4().simple()), "name": "B7", "min_auth_level": 7})),
    )
    .await?;
    let bid7 = b7_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing b7 id"))?;
    let (_, b8_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&l8_key),
        Some(json!({"slug": format!("b8-{}", Uuid::new_v4().simple()), "name": "B8", "min_auth_level": 8})),
    )
    .await?;
    let bid8 = b8_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing b8 id"))?;

    let _ = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid7}/documents"),
        Some(&l7_key),
        Some(json!({"title": "L7 Doc", "text": "연락처 010-1234-5678 안내", "min_auth_level": 7})),
    )
    .await?;
    let _ = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid8}/documents"),
        Some(&l8_key),
        Some(json!({"title": "L8 Doc", "text": "L8 기밀 감사 계획", "min_auth_level": 8})),
    )
    .await?;

    let (c7_status, c7_json, c7_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l7_key),
        Some(json!({
            "model": "anthropic/claude-3.5-sonnet",
            "bucket_id": bid7,
            "messages": [{"role": "user", "content": "연락처 안내"}],
        })),
    )
    .await?;
    assert_eq!(c7_status, StatusCode::OK, "L7 chat failed: {c7_txt}");
    assert_eq!(c7_json["provider"], "openrouter");
    assert_eq!(c7_json["pii_masked"], true);
    assert!(!c7_txt.contains("010-1234-5678"));

    let (c8_status, c8_json, c8_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l8_key),
        Some(json!({
            "model": "anthropic/claude-3.5-sonnet",
            "bucket_id": bid8,
            "messages": [{"role": "user", "content": "기밀 감사 계획"}],
        })),
    )
    .await?;
    assert_eq!(c8_status, StatusCode::OK, "L8 chat failed: {c8_txt}");
    assert_eq!(c8_json["provider"], "vllm_local");
    assert_eq!(c8_json["is_on_premise"], true);
    assert_eq!(c8_json["max_context_auth_level"], 8);

    // Streaming SSE
    let sse_req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(AUTHORIZATION, format!("Bearer {l7_key}"))
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&json!({
            "model": "openrag-anything",
            "stream": true,
            "messages": [{"role": "user", "content": "안내 요약"}],
        }))?))?;
    let sse_resp = app.clone().oneshot(sse_req).await?;
    assert_eq!(sse_resp.status(), StatusCode::OK);
    let sse_ct = sse_resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(sse_ct.contains("text/event-stream"));
    let sse_text = String::from_utf8_lossy(&sse_resp.into_body().collect().await?.to_bytes()).to_string();
    assert!(sse_text.contains("data: [DONE]"));

    // Empty messages -> 400
    let (empty_msg_status, _, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l7_key),
        Some(json!({"model": "openrag-auto", "messages": []})),
    )
    .await?;
    assert_eq!(empty_msg_status, StatusCode::BAD_REQUEST);

    // Guard block must insert ZERO rows in gateway.request_logs
    let pol = GuardRepo::get_or_create_active_policy(worker_pool.inner(), seeded.org_id, "default").await?;
    let _ = GuardRepo::create_rule(
        worker_pool.inner(),
        seeded.org_id,
        pol.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "PROJECT-ORION".into(),
            action: GuardAction::Block,
            surfaces: vec![GuardSurface::InboundQuery, GuardSurface::IngestionGate],
            replacement_text: None,
            enabled: true,
            priority: 1,
        },
    )
    .await?;

    let logs_before: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM gateway.request_logs WHERE org_id = $1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;

    let (blk_status, _, blk_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&admin_key),
        Some(json!({
            "model": "openrag-auto",
            "messages": [{"role": "user", "content": "PROJECT-ORION 요약"}],
        })),
    )
    .await?;
    assert_eq!(blk_status, StatusCode::FORBIDDEN);
    assert!(blk_txt.contains("guard_blocked"));

    let logs_after: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM gateway.request_logs WHERE org_id = $1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(logs_before, logs_after, "guard block must insert 0 rows into gateway.request_logs");

    // IngestionGate block before DB insert
    let (ing_blk_status, _, ing_blk_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({"title": "Blocked", "text": "Secret PROJECT-ORION leak"})),
    )
    .await?;
    assert_eq!(ing_blk_status, StatusCode::FORBIDDEN);
    assert!(ing_blk_txt.contains("guard_blocked"));

    // Airgap profile routes L7 to vllm_local
    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.profile = PipelineProfile::Airgap;
    }
    let (ag_status, ag_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l7_key),
        Some(json!({"model": "openrag-auto", "bucket_id": bid7, "messages": [{"role": "user", "content": "망분리"}]})),
    )
    .await?;
    assert_eq!(ag_status, StatusCode::OK);
    assert_eq!(ag_json["provider"], "vllm_local");
    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.profile = PipelineProfile::Standard;
    }

    // 7. POST /v1/eval/ab-search
    let (ab_status, ab_json, ab_txt) = send_json(
        &app,
        Method::POST,
        "/v1/eval/ab-search",
        Some(&admin_key),
        Some(json!({
            "bucket": bkt_slug,
            "k": 5,
            "queries": [{"query": "Sync", "relevant_ids": [], "forbidden_ids": []}],
            "variants": {
                "var_a": {"fusion": "rrf", "alpha": 0.2},
                "var_b": {"fusion": "convex", "alpha": 0.5}
            }
        })),
    )
    .await?;
    assert_eq!(ab_status, StatusCode::OK, "ab_search failed: {ab_txt}");
    assert!(ab_json["var_a"]["mrr"].is_number());
    assert_eq!(ab_json["security_gates"]["pass"], true);

    // 8. Concepts CRUD, self-link 400, 256d validation, creator/admin 403 delete vs 404 cross-org
    let (self_link_status, _, _) = send_json(
        &app,
        Method::POST,
        "/v1/concepts/link",
        Some(&owner_key),
        Some(json!({"source": "Apple", "target": " apple "})),
    )
    .await?;
    assert_eq!(self_link_status, StatusCode::BAD_REQUEST);

    let mut base_vec = vec![0.0_f32; 256];
    base_vec[0] = 1.0;
    let (cc_status, cc_json, cc_txt) = send_json(
        &app,
        Method::POST,
        "/v1/concepts",
        Some(&owner_key),
        Some(json!({
            "canonical_name": "ORG-CODENAME",
            "aliases": ["사내코드명"],
            "embedding": base_vec,
        })),
    )
    .await?;
    assert_eq!(cc_status, StatusCode::CREATED, "create concept failed: {cc_txt}");
    let cid = cc_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing concept id"))?;
    let alias_id = cc_json["aliases"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing alias id"))?;

    // Semantic resolve threshold + bad dim
    let (bad_dim_status, _, _) = send_json(
        &app,
        Method::POST,
        "/v1/concepts/resolve",
        Some(&owner_key),
        Some(json!({"term": "x", "use_semantic": true, "embedding": vec![0.0_f32; 127]})),
    )
    .await?;
    assert_eq!(bad_dim_status, StatusCode::BAD_REQUEST);

    let mut cos_vec = vec![0.0_f32; 256];
    cos_vec[0] = 0.9;
    cos_vec[1] = (1.0_f32 - 0.81_f32).sqrt();
    let (res_ok_status, res_ok_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/concepts/resolve",
        Some(&owner_key),
        Some(json!({"term": "unseen", "use_semantic": true, "embedding": cos_vec, "threshold": 0.85})),
    )
    .await?;
    assert_eq!(res_ok_status, StatusCode::OK);
    assert_eq!(res_ok_json["matched"], true);

    // Same-org peer delete -> 403; cross-org outsider -> 404
    let (peer_del_status, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/concepts/{cid}"),
        Some(&peer_key),
        None,
    )
    .await?;
    assert_eq!(peer_del_status, StatusCode::FORBIDDEN);

    let (other_del_status, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/concepts/{cid}"),
        Some(&other_key),
        None,
    )
    .await?;
    assert_eq!(other_del_status, StatusCode::NOT_FOUND);

    // Creator deletes alias then concept -> 200
    let (al_del_status, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/concepts/{cid}/aliases/{alias_id}"),
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(al_del_status, StatusCode::OK);

    let (own_del_status, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/concepts/{cid}"),
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(own_del_status, StatusCode::OK);

    // 9. Document tag CRUD propagates to chunks
    let (_, tag_doc_json, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({"title": "TagDoc", "text": "Tag propagation test", "access_scope": "dept"})),
    )
    .await?;
    let tag_doc_id = tag_doc_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing tag doc id"))?;

    let (at_status, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents/{tag_doc_id}/tags"),
        Some(&admin_key),
        Some(json!({"tags": ["Finance", "Policy-2026"]})),
    )
    .await?;
    assert_eq!(at_status, StatusCode::OK);

    let chunk_tags: Vec<String> = sqlx::query_scalar(
        "SELECT tags FROM corpus.chunks WHERE org_id = $1 AND document_id = $2::uuid LIMIT 1",
    )
    .bind(seeded.org_id)
    .bind(tag_doc_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(chunk_tags.contains(&"finance".to_string()));
    assert!(chunk_tags.contains(&"policy-2026".to_string()));

    let (gt_status, gt_json, _) = send_json(
        &app,
        Method::GET,
        &format!("/v2/buckets/{bid}/documents/{tag_doc_id}/tags"),
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(gt_status, StatusCode::OK);
    assert!(gt_json["tags"].to_string().contains("finance"));

    let (rt_status, rt_json, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v2/buckets/{bid}/documents/{tag_doc_id}/tags"),
        Some(&admin_key),
        Some(json!({"tags": ["finance"]})),
    )
    .await?;
    assert_eq!(rt_status, StatusCode::OK);
    assert!(!rt_json["tags"].to_string().contains("finance"));

    Ok(())
}

fn build_two_page_hwpx_bytes(page1_text: &str, page2_text: &str) -> anyhow::Result<Vec<u8>> {
    let buf = Cursor::new(Vec::new());
    let mut zw = zip::ZipWriter::new(buf);
    let opts_stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let opts_deflate = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zw.start_file("mimetype", opts_stored)?;
    zw.write_all(b"application/hwp+zip")?;

    for (idx, text) in [(0, page1_text), (1, page2_text)] {
        let section = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
             <hs:sec xmlns:hs=\"http://www.hancom.co.kr/hwpml/2011/section\" \
             xmlns:hp=\"http://www.hancom.co.kr/hwpml/2011/paragraph\">\
             <hp:p><hp:run><hp:t>{text}</hp:t></hp:run></hp:p>\
             </hs:sec>"
        );
        zw.start_file(format!("Contents/section{idx}.xml"), opts_deflate)?;
        zw.write_all(section.as_bytes())?;
    }
    let finished = zw.finish()?;
    Ok(finished.into_inner())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_unmask_pii_and_llm_egress_guard_and_allow_unmasked() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;

    // Seed Dave (L8, can_unmask_pii=true, explicit IAM policy Deny pii:unmask) and Eve (L8, can_unmask_pii=true, group member)
    let dave_id = Uuid::new_v4();
    let eve_id = Uuid::new_v4();
    let deny_group_id = Uuid::new_v4();
    let allow_group_id = Uuid::new_v4();
    let deny_policy_id = Uuid::new_v4();

    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $2, $3, 'Dave', 8, true, false),
         ($4, $2, $5, 'Eve', 8, true, false)",
    )
    .bind(dave_id)
    .bind(seeded.org_id)
    .bind(format!("dave-{dave_id}@test.local"))
    .bind(eve_id)
    .bind(format!("eve-{eve_id}@test.local"))
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'member'),
         ($1, $2, $4, 'member')",
    )
    .bind(seeded.org_id)
    .bind(seeded.dept_id)
    .bind(dave_id)
    .bind(eve_id)
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, min_auth_level, max_auth_level, allowed_actions) VALUES
         ($1, $2, 'Deny Unmask', 1, 10, ARRAY['read','search','pii:unmask']::text[]),
         ($3, $2, 'Reader Group', 1, 10, ARRAY['read','search','pii:unmask']::text[])",
    )
    .bind(deny_group_id)
    .bind(seeded.org_id)
    .bind(allow_group_id)
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
         ($1, $2, $3),
         ($1, $4, $5)",
    )
    .bind(seeded.org_id)
    .bind(deny_group_id)
    .bind(dave_id)
    .bind(allow_group_id)
    .bind(eve_id)
    .execute(worker_pool.inner())
    .await?;

    let deny_doc = json!({
        "statements": [{
            "sid": "deny-unmask-all",
            "effect": "Deny",
            "actions": ["pii:unmask"],
            "resources": [format!("arn:openrag:{}:*", seeded.slug)]
        }]
    });
    sqlx::query("INSERT INTO iam.policies (id, org_id, name, document) VALUES ($1, $2, 'DenyPiiUnmask', $3)")
        .bind(deny_policy_id)
        .bind(seeded.org_id)
        .bind(deny_doc)
        .execute(worker_pool.inner())
        .await?;
    sqlx::query(
        "INSERT INTO iam.policy_attachments (org_id, policy_id, target_type, target_id)
         VALUES ($1, $2, 'group', $3)",
    )
    .bind(seeded.org_id)
    .bind(deny_policy_id)
    .bind(deny_group_id)
    .execute(worker_pool.inner())
    .await?;

    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let l7_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.l7_id, None, vec![]).await?;
    let l8_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.l8_id, None, vec![]).await?;
    let dave_key = mint_key(&app, &worker_pool, seeded.org_id, dave_id, None, vec![]).await?;
    let eve_key = mint_key(&app, &worker_pool, seeded.org_id, eve_id, None, vec![]).await?;

    // Configure LlmEgress rules: rrn_kr Block, email Mask, phone_kr Mask
    let pol = GuardRepo::get_or_create_active_policy(worker_pool.inner(), seeded.org_id, "default")
        .await?;
    let _ = GuardRepo::create_rule(
        worker_pool.inner(),
        seeded.org_id,
        pol.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::PiiCategory,
            pattern_or_category: "rrn_kr".into(),
            action: GuardAction::Block,
            surfaces: vec![GuardSurface::LlmEgress, GuardSurface::McpEgress],
            replacement_text: None,
            enabled: true,
            priority: 5,
        },
    )
    .await?;
    let _ = GuardRepo::create_rule(
        worker_pool.inner(),
        seeded.org_id,
        pol.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::PiiCategory,
            pattern_or_category: "email".into(),
            action: GuardAction::Mask,
            surfaces: vec![GuardSurface::LlmEgress],
            replacement_text: None,
            enabled: true,
            priority: 15,
        },
    )
    .await?;
    let _ = GuardRepo::create_rule(
        worker_pool.inner(),
        seeded.org_id,
        pol.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::PiiCategory,
            pattern_or_category: "phone_kr".into(),
            action: GuardAction::Mask,
            surfaces: vec![GuardSurface::LlmEgress],
            replacement_text: None,
            enabled: true,
            priority: 16,
        },
    )
    .await?;

    let (_, b_json, b_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&l8_key),
        Some(json!({
            "slug": format!("unmask-bkt-{}", Uuid::new_v4().simple()),
            "name": "Unmask Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let bid = b_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket id: {b_txt}"))?;

    // 1a. L8 owner uploads dept doc -> unmask_pii: true succeeds and writes allowed=true audit log
    let (up8_status, _, up8_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&l8_key),
        Some(json!({
            "title": "Contact Sheet",
            "text": "EmergencyContact999 email is secret.ceo@acme.test and phone 010-5555-1212",
            "access_scope": "dept",
            "min_auth_level": 3,
        })),
    )
    .await?;
    assert_eq!(up8_status, StatusCode::CREATED, "upload l8 doc: {up8_txt}");

    let (u8_status, u8_json, u8_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l8_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "EmergencyContact999"}],
        })),
    )
    .await?;
    assert_eq!(u8_status, StatusCode::OK, "l8 unmask failed: {u8_txt}");
    let u8_ans = u8_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(u8_ans.contains("secret.ceo@acme.test"), "expected unmasked email in {u8_ans}");
    assert!(u8_ans.contains("010-5555-1212"), "expected unmasked phone in {u8_ans}");

    let allowed_audits: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = true AND resource_arn IS NOT NULL",
    )
    .bind(seeded.org_id)
    .bind(seeded.l8_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(allowed_audits >= 1);

    // 1b. Dave (can_unmask_pii=true, IAM policy Deny pii:unmask) -> PII stays masked + allowed=false audit log
    let (dave_status, dave_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&dave_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "EmergencyContact999"}],
        })),
    )
    .await?;
    assert_eq!(dave_status, StatusCode::OK);
    let dave_ans = dave_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!dave_ans.contains("secret.ceo@acme.test"));
    assert!(dave_ans.contains("[PII:EMAIL:"));

    let dave_denied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false AND reason = 'explicit_deny'",
    )
    .bind(seeded.org_id)
    .bind(dave_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(dave_denied, 1);

    // 1c. L7 (can_unmask_pii=false) -> PII stays masked + principal_cannot_unmask_pii audit log
    let (l7u_status, l7u_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l7_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "EmergencyContact999"}],
        })),
    )
    .await?;
    assert_eq!(l7u_status, StatusCode::OK);
    let l7u_ans = l7u_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!l7u_ans.contains("secret.ceo@acme.test"));
    let l7_denied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false AND reason = 'principal_cannot_unmask_pii'",
    )
    .bind(seeded.org_id)
    .bind(seeded.l7_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(l7_denied, 1);

    // 1d. Eve (can_unmask_pii=true) on group-scoped doc in its own bucket where resource_group_bindings lacks pii:unmask -> stays masked
    let (_, gb_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("grp-bkt-{}", Uuid::new_v4().simple()),
            "name": "Group Unmask Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let group_bid = gb_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing group bucket id"))?;

    let (_, gdoc_json, gdoc_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{group_bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "Group Contact",
            "text": "GroupTag777 contact group.owner@acme.test",
            "access_scope": "group",
            "allowed_group_ids": [],
            "min_auth_level": 3,
        })),
    )
    .await?;
    assert_eq!(gdoc_json["access_scope"], "group", "group upload failed: {gdoc_txt}");
    let gdoc_id = Uuid::parse_str(
        gdoc_json["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing group doc id"))?,
    )?;
    let gchunk_id = Uuid::parse_str(
        gdoc_json["chunks"][0]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing group chunk id"))?,
    )?;

    // Grant Eve's group read+search ONLY (no pii:unmask) on document + chunk
    sqlx::query(
        "INSERT INTO iam.resource_group_bindings (org_id, resource_type, resource_id, group_id, effect, actions) VALUES
         ($1, 'document', $2, $4, 'allow', ARRAY['read','search']::text[]),
         ($1, 'chunk', $3, $4, 'allow', ARRAY['read','search']::text[])",
    )
    .bind(seeded.org_id)
    .bind(gdoc_id)
    .bind(gchunk_id)
    .bind(allow_group_id)
    .execute(worker_pool.inner())
    .await?;

    let (eve1_status, eve1_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&eve_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": group_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "GroupTag777"}],
        })),
    )
    .await?;
    assert_eq!(eve1_status, StatusCode::OK);
    let eve1_ans = eve1_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!eve1_ans.contains("group.owner@acme.test"));
    assert!(eve1_ans.contains("[PII:EMAIL:"));

    let eve_denied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false",
    )
    .bind(seeded.org_id)
    .bind(eve_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(eve_denied, 1);

    // Now add pii:unmask on document binding -> Eve unmasks
    sqlx::query(
        "UPDATE iam.resource_group_bindings
         SET actions = ARRAY['read','search','pii:unmask']::text[]
         WHERE org_id = $1 AND resource_type = 'document' AND resource_id = $2 AND group_id = $3",
    )
    .bind(seeded.org_id)
    .bind(gdoc_id)
    .bind(allow_group_id)
    .execute(worker_pool.inner())
    .await?;

    let (eve2_status, eve2_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&eve_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": group_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "GroupTag777"}],
        })),
    )
    .await?;
    assert_eq!(eve2_status, StatusCode::OK);
    let eve2_ans = eve2_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        eve2_ans.contains("group.owner@acme.test"),
        "expected unmasked group.owner@acme.test in eve2_json: {eve2_json}"
    );

    // 1e. org_admin (is_org_admin=true, can_unmask_pii=true, NOT owner) in ['read','search'] group
    // on a Group-scoped doc inside an Org-scoped bucket -> must NOT unmask via bucket scope_org_admin
    let org_admin_only_id = Uuid::new_v4();
    let ro_group_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin)
         VALUES ($1, $2, $3, 'OrgAdminOnly', 8, true, true)",
    )
    .bind(org_admin_only_id)
    .bind(seeded.org_id)
    .bind(format!("orgadmin-{org_admin_only_id}@test.local"))
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
         VALUES ($1, $2, $3, 'member')",
    )
    .bind(seeded.org_id)
    .bind(seeded.dept_id)
    .bind(org_admin_only_id)
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.groups (id, org_id, name, min_auth_level, max_auth_level, allowed_actions)
         VALUES ($1, $2, 'Group ReadSearch Only', 1, 10, ARRAY['read','search']::text[])",
    )
    .bind(ro_group_id)
    .bind(seeded.org_id)
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.group_members (org_id, group_id, principal_id)
         VALUES ($1, $2, $3)",
    )
    .bind(seeded.org_id)
    .bind(ro_group_id)
    .bind(org_admin_only_id)
    .execute(worker_pool.inner())
    .await?;
    let org_admin_only_key =
        mint_key(&app, &worker_pool, seeded.org_id, org_admin_only_id, None, vec![]).await?;

    let (ob_status, ob_json, ob_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&l8_key),
        Some(json!({
            "slug": format!("org-bkt-{}", Uuid::new_v4().simple()),
            "name": "Org Bucket For Group Doc",
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(ob_status, StatusCode::CREATED, "create org bucket: {ob_txt}");
    let org_bid = ob_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing org bucket id"))?;

    let (og_status, og_json, og_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{org_bid}/documents"),
        Some(&l8_key),
        Some(json!({
            "title": "OrgGroupDoc",
            "text": "OrgBucketGroupTag888 secret email is org.group@acme.test",
            "access_scope": "group",
            "allowed_group_ids": [ro_group_id],
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(og_status, StatusCode::CREATED, "create group doc in org bucket: {og_txt}");
    assert_eq!(og_json["access_scope"], "group");

    for maybe_bid in [Some(org_bid), None] {
        let mut req_body = json!({
            "model": "openrag-auto",
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "OrgBucketGroupTag888"}],
        });
        if let Some(b) = maybe_bid {
            if let Some(obj) = req_body.as_object_mut() {
                obj.insert("bucket_id".into(), json!(b));
            }
        }
        let (oa_status, oa_json, oa_txt) = send_json(
            &app,
            Method::POST,
            "/v1/chat/completions",
            Some(&org_admin_only_key),
            Some(req_body),
        )
        .await?;
        assert_eq!(oa_status, StatusCode::OK, "org_admin chat failed: {oa_txt}");
        let oa_ans = oa_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            !oa_ans.contains("org.group@acme.test"),
            "org_admin leaked PII on group-scoped doc in org bucket: {oa_ans}"
        );
        assert!(
            oa_ans.contains("[PII:EMAIL:"),
            "expected masked email token for org_admin in {oa_ans}"
        );
    }

    let org_admin_denied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false",
    )
    .bind(seeded.org_id)
    .bind(org_admin_only_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(org_admin_denied, 2);

    // 1f. org_admin (is_org_admin=true, can_unmask_pii=true, dept_role='member', NOT dept_admin, NOT owner)
    // on Dept-scoped document AND Dept-scoped wiki page inside an Org-scoped bucket -> must NOT unmask via bucket scope_org_admin
    let org_admin_dept_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin)
         VALUES ($1, $2, $3, 'OrgAdminDeptMember', 8, true, true)",
    )
    .bind(org_admin_dept_id)
    .bind(seeded.org_id)
    .bind(format!("orgadmin-dept-{org_admin_dept_id}@test.local"))
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
         VALUES ($1, $2, $3, 'member')",
    )
    .bind(seeded.org_id)
    .bind(seeded.dept_id)
    .bind(org_admin_dept_id)
    .execute(worker_pool.inner())
    .await?;
    let org_admin_dept_key =
        mint_key(&app, &worker_pool, seeded.org_id, org_admin_dept_id, None, vec![]).await?;

    let (ob2_status, ob2_json, ob2_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&l8_key),
        Some(json!({
            "slug": format!("org-bkt-dept-{}", Uuid::new_v4().simple()),
            "name": "Org Bucket For Dept Resources",
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(ob2_status, StatusCode::CREATED, "create org bucket for dept: {ob2_txt}");
    let org_dept_bid = ob2_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing org_dept_bid"))?;

    let (dd_status, dd_json, dd_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{org_dept_bid}/documents"),
        Some(&l8_key),
        Some(json!({
            "title": "DeptDocInOrgBucket",
            "text": "DeptInOrgDocTag991 confidential email is dept.in.org.doc@acme.test",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(dd_status, StatusCode::CREATED, "create dept doc in org bucket: {dd_txt}");
    assert_eq!(dd_json["access_scope"], "dept");

    let (dw_status, _, dw_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{org_dept_bid}/documents/sync"),
        Some(&l8_key),
        Some(json!({
            "documents": [{
                "slug": "dept-wiki-in-org-bucket",
                "title": "DeptWikiInOrgBucket",
                "body_md": "# Dept Wiki\nDeptInOrgWikiTag992 secret email is dept.in.org.wiki@acme.test",
                "access_scope": "dept",
                "min_auth_level": 1,
            }]
        })),
    )
    .await?;
    assert_eq!(dw_status, StatusCode::OK, "sync dept wiki in org bucket: {dw_txt}");

    for (tag, secret_email) in [
        ("DeptInOrgDocTag991", "dept.in.org.doc@acme.test"),
        ("DeptInOrgWikiTag992", "dept.in.org.wiki@acme.test"),
    ] {
        let (oa_status, oa_json, oa_txt) = send_json(
            &app,
            Method::POST,
            "/v1/chat/completions",
            Some(&org_admin_dept_key),
            Some(json!({
                "model": "openrag-auto",
                "bucket_id": org_dept_bid,
                "unmask_pii": true,
                "messages": [{"role": "user", "content": tag}],
            })),
        )
        .await?;
        assert_eq!(oa_status, StatusCode::OK, "org_admin dept chat failed: {oa_txt}");
        let oa_ans = oa_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            !oa_ans.contains(secret_email),
            "org_admin leaked PII on dept-scoped resource in org bucket ({tag}): {oa_ans}"
        );
        assert!(
            oa_ans.contains("[PII:EMAIL:"),
            "expected masked email token for org_admin on dept-scoped resource ({tag}) in {oa_ans}"
        );
    }

    let org_admin_dept_denied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false",
    )
    .bind(seeded.org_id)
    .bind(org_admin_dept_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(org_admin_dept_denied, 2);

    let org_admin_dept_allowed: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = true",
    )
    .bind(seeded.org_id)
    .bind(org_admin_dept_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(org_admin_dept_allowed, 0);

    // 1g. Prompt-Echo / Token-Injection Vault Unmasking Oracle regression probes (3a-3d):
    //     Unmasking must be bound strictly to [PII:...] tokens originating from authorized retrieved hits.
    let bob_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin)
         VALUES ($1, $2, $3, 'BobMember', 8, true, false)",
    )
    .bind(bob_id)
    .bind(seeded.org_id)
    .bind(format!("bob-{bob_id}@test.local"))
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
         VALUES ($1, $2, $3, 'member')",
    )
    .bind(seeded.org_id)
    .bind(seeded.dept_id)
    .bind(bob_id)
    .execute(worker_pool.inner())
    .await?;
    let bob_key = mint_key(&app, &worker_pool, seeded.org_id, bob_id, None, vec![]).await?;

    let (pay_up_status, _, pay_up_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&l8_key),
        Some(json!({
            "title": "AlicePayrollDeptDoc",
            "text": "PAYROLLORACLE001 ceo.payroll.secret@acme.test",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(pay_up_status, StatusCode::CREATED, "upload payroll doc: {pay_up_txt}");

    let (bob_read_status, bob_read_json, bob_read_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "PAYROLLORACLE001"}],
        })),
    )
    .await?;
    assert_eq!(bob_read_status, StatusCode::OK, "bob dept read failed: {bob_read_txt}");
    let bob_read_ans = bob_read_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!bob_read_ans.contains("ceo.payroll.secret@acme.test"));
    let pii_re = Regex::new(r"\[PII:EMAIL:[0-9a-fA-F]{4,32}\]")?;
    let payroll_tok = pii_re
        .find(bob_read_ans)
        .map(|m| m.as_str().to_string())
        .ok_or_else(|| anyhow::anyhow!("missing sealed email token in {bob_read_ans}"))?;

    let (bob_bkt_status, bob_bkt_json, bob_bkt_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&bob_key),
        Some(json!({
            "slug": format!("bob-empty-oracle-{}", Uuid::new_v4().simple()),
            "name": "Bob Empty Private Oracle Bucket",
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(bob_bkt_status, StatusCode::CREATED, "create bob private bucket: {bob_bkt_txt}");
    let bob_bid = bob_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bob_bid"))?;

    let echo_transport = Arc::new(EchoVllmTransport);
    let offline_app = build_router(state.with_mode_and_transport(OpenRagMode::Offline, Arc::clone(&echo_transport) as Arc<dyn LlmHttpTransport>));
    let live_vllm_app = build_router(state.with_mode_and_transport(OpenRagMode::Live, echo_transport));

    // Vector 1 (3a & 3b): bob_member injects sealed token into own empty private bucket in BOTH offline & live vLLM
    for (mode_label, router_ref) in [("offline", &offline_app), ("live_vllm", &live_vllm_app)] {
        let (st, j, txt) = send_json(
            router_ref,
            Method::POST,
            "/v1/chat/completions",
            Some(&bob_key),
            Some(json!({
                "model": "openrag-auto",
                "bucket_id": bob_bid,
                "unmask_pii": true,
                "messages": [{"role": "user", "content": format!("Explain {payroll_tok}")}],
            })),
        )
        .await?;
        assert_eq!(st, StatusCode::OK, "bob empty oracle ({mode_label}) failed: {txt}");
        let ans = j["choices"][0]["message"]["content"].as_str().unwrap_or("");
        assert!(
            !ans.contains("ceo.payroll.secret@acme.test"),
            "oracle leaked plaintext email in {mode_label} mode: {ans}"
        );
        assert!(
            ans.contains(&payroll_tok),
            "expected sealed token to remain in {mode_label} mode: {ans}"
        );
        if mode_label == "offline" {
            assert!(ans.contains("No matching indexed documents were found"));
        } else {
            assert!(ans.starts_with("Mock answer ("));
        }
    }

    // Vector 2 (3d): bob_member uploads harmless private dummy doc BOBDUMMY-123 and injects foreign token
    let (dum_status, _, dum_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bob_bid}/documents"),
        Some(&bob_key),
        Some(json!({
            "title": "BobDummyHarmless",
            "text": "BOBDUMMY-123 harmless text",
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(dum_status, StatusCode::CREATED, "upload bob dummy doc: {dum_txt}");

    let (d_st, d_json, d_txt) = send_json(
        &live_vllm_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bob_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": format!("BOBDUMMY-123 {payroll_tok}")}],
        })),
    )
    .await?;
    assert_eq!(d_st, StatusCode::OK, "bob dummy doc oracle failed: {d_txt}");
    let d_ans = d_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(d_ans.contains("BOBDUMMY-123"), "expected dummy hit in context: {d_ans}");
    assert!(
        !d_ans.contains("ceo.payroll.secret@acme.test"),
        "dummy-hit oracle leaked foreign email: {d_ans}"
    );
    assert!(d_ans.contains(&payroll_tok));

    // Also verify mixed authorized own-PII doc + foreign injected token: only own PII unmasks
    let (_, bob_own_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&bob_key),
        Some(json!({
            "slug": format!("bob-own-pii-{}", Uuid::new_v4().simple()),
            "name": "Bob Own PII Bucket",
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let bob_own_bid = bob_own_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bob_own_bid"))?;
    let _ = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bob_own_bid}/documents"),
        Some(&bob_key),
        Some(json!({
            "title": "BobOwnPiiDoc",
            "text": "BOBOWNPII456 reach bob.own@acme.test",
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let (mix_st, mix_json, mix_txt) = send_json(
        &live_vllm_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bob_own_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": format!("BOBOWNPII456 {payroll_tok}")}],
        })),
    )
    .await?;
    assert_eq!(mix_st, StatusCode::OK, "bob mixed unmask failed: {mix_txt}");
    let mix_ans = mix_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(mix_ans.contains("bob.own@acme.test"), "expected own PII unmasked: {mix_ans}");
    assert!(
        !mix_ans.contains("ceo.payroll.secret@acme.test"),
        "mixed probe leaked foreign email: {mix_ans}"
    );
    assert!(mix_ans.contains(&payroll_tok));

    // Vector 3 (3c): org_admin in grp-ro (['read','search']) reads sealed [PII:EMAIL:...] from Group doc,
    // injects it into his own empty Org bucket -> plaintext email MUST NOT appear
    let (oa_grp_st, oa_grp_json, oa_grp_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&org_admin_only_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": org_bid,
            "unmask_pii": false,
            "messages": [{"role": "user", "content": "OrgBucketGroupTag888"}],
        })),
    )
    .await?;
    assert_eq!(oa_grp_st, StatusCode::OK, "org_admin group read failed: {oa_grp_txt}");
    let oa_grp_ans = oa_grp_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    let grp_tok = pii_re
        .find(oa_grp_ans)
        .map(|m| m.as_str().to_string())
        .ok_or_else(|| anyhow::anyhow!("missing sealed group email token in {oa_grp_ans}"))?;

    let (oa_bkt_st, oa_bkt_json, oa_bkt_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&org_admin_only_key),
        Some(json!({
            "slug": format!("oa-empty-oracle-{}", Uuid::new_v4().simple()),
            "name": "OrgAdmin Empty Org Oracle Bucket",
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(oa_bkt_st, StatusCode::CREATED, "create org_admin empty org bucket: {oa_bkt_txt}");
    let oa_empty_bid = oa_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing oa_empty_bid"))?;

    for (mode_label, router_ref) in [("offline", &offline_app), ("live_vllm", &live_vllm_app)] {
        let (st, j, txt) = send_json(
            router_ref,
            Method::POST,
            "/v1/chat/completions",
            Some(&org_admin_only_key),
            Some(json!({
                "model": "openrag-auto",
                "bucket_id": oa_empty_bid,
                "unmask_pii": true,
                "messages": [{"role": "user", "content": format!("Explain {grp_tok}")}],
            })),
        )
        .await?;
        assert_eq!(st, StatusCode::OK, "org_admin empty oracle ({mode_label}) failed: {txt}");
        let ans = j["choices"][0]["message"]["content"].as_str().unwrap_or("");
        assert!(
            !ans.contains("org.group@acme.test"),
            "org_admin empty org oracle leaked group email in {mode_label}: {ans}"
        );
        assert!(ans.contains(&grp_tok));
    }

    let bob_denied_audits: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false",
    )
    .bind(seeded.org_id)
    .bind(bob_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(bob_denied_audits >= 4, "expected denial audit logs for bob oracle attempts, got {bob_denied_audits}");

    let oa_ro_denied_audits: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false",
    )
    .bind(seeded.org_id)
    .bind(org_admin_only_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(oa_ro_denied_audits, 4, "expected 2 from 1e + 2 from 1g oracle attempts");

    let oa_ro_allowed_audits: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = true",
    )
    .bind(seeded.org_id)
    .bind(org_admin_only_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(oa_ro_allowed_audits, 0);

    // 1h. Stored-Token Injection / Cross-Document PII Token Laundering regression probes (4a, 4b, + raw-SQL decoy):
    //     Layer 1 neutralizes [PII:<TAG>:<HEX>] literals on doc ingest, wiki sync, and corpus sync;
    //     Layer 2 blocks unmasking of pre-existing chunk/page tokens lacking pii.chunk_spans provenance.
    let bob_launder_id = Uuid::new_v4();
    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin)
         VALUES ($1, $2, $3, 'BobLaunderMember', 8, true, false)",
    )
    .bind(bob_launder_id)
    .bind(seeded.org_id)
    .bind(format!("bob-launder-{bob_launder_id}@test.local"))
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
         VALUES ($1, $2, $3, 'member')",
    )
    .bind(seeded.org_id)
    .bind(seeded.dept_id)
    .bind(bob_launder_id)
    .execute(worker_pool.inner())
    .await?;
    let bob_launder_key = mint_key(&app, &worker_pool, seeded.org_id, bob_launder_id, None, vec![]).await?;

    let (bl_bkt_st, bl_bkt_json, bl_bkt_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&bob_launder_key),
        Some(json!({
            "slug": format!("bob-launder-{}", Uuid::new_v4().simple()),
            "name": "Bob Launder Private Bucket",
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(bl_bkt_st, StatusCode::CREATED, "create bob launder bucket: {bl_bkt_txt}");
    let bob_launder_bid = bl_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bob_launder_bid"))?;

    // 4a: Document ingest laundering attempt
    let (l4a_up_st, _, l4a_up_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bob_launder_bid}/documents"),
        Some(&bob_launder_key),
        Some(json!({
            "title": "LaunderDoc4a",
            "text": format!("LAUNDER-DOC-999 my note {payroll_tok}"),
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(l4a_up_st, StatusCode::CREATED, "4a launder doc ingest: {l4a_up_txt}");

    let (l4a_st, l4a_json, l4a_txt) = send_json(
        &live_vllm_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_launder_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bob_launder_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "LAUNDER-DOC-999"}],
        })),
    )
    .await?;
    assert_eq!(l4a_st, StatusCode::OK, "4a launder doc query: {l4a_txt}");
    let l4a_ans = l4a_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(l4a_ans.contains("LAUNDER-DOC-999"), "expected 4a doc hit: {l4a_ans}");
    assert!(
        !l4a_ans.contains("ceo.payroll.secret@acme.test"),
        "4a laundered doc leaked foreign email: {l4a_ans}"
    );
    assert!(
        l4a_ans.contains("[PII-LITERAL:EMAIL:"),
        "expected Layer 1 neutralized [PII-LITERAL:EMAIL:...] in 4a: {l4a_ans}"
    );

    // 4b: Wiki sync laundering attempt (+ repeat sync SHA-256 No-Op preservation)
    let wiki_sync_payload = json!({
        "documents": [{
            "kind": "wiki",
            "slug": "launder-wiki",
            "title": "LaunderWiki4b",
            "body_md": format!("LAUNDER-WIKI-999 {payroll_tok}"),
            "access_scope": "private",
            "min_auth_level": 1,
        }]
    });
    let (l4b_up_st, l4b_up_json, l4b_up_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bob_launder_bid}/documents/sync"),
        Some(&bob_launder_key),
        Some(wiki_sync_payload.clone()),
    )
    .await?;
    assert_eq!(l4b_up_st, StatusCode::OK, "4b launder wiki sync: {l4b_up_txt}");
    assert_eq!(l4b_up_json["documents"][0]["status"], "updated");

    let (l4b_noop_st, l4b_noop_json, l4b_noop_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bob_launder_bid}/documents/sync"),
        Some(&bob_launder_key),
        Some(wiki_sync_payload),
    )
    .await?;
    assert_eq!(l4b_noop_st, StatusCode::OK, "4b repeat wiki sync: {l4b_noop_txt}");
    assert_eq!(
        l4b_noop_json["documents"][0]["status"], "skipped",
        "repeat wiki sync must preserve SHA-256 No-Op skip: {l4b_noop_json}"
    );

    let (l4b_st, l4b_json, l4b_txt) = send_json(
        &live_vllm_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_launder_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bob_launder_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "LAUNDER-WIKI-999"}],
        })),
    )
    .await?;
    assert_eq!(l4b_st, StatusCode::OK, "4b launder wiki query: {l4b_txt}");
    let l4b_ans = l4b_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(l4b_ans.contains("LAUNDER-WIKI-999"), "expected 4b wiki hit: {l4b_ans}");
    assert!(
        !l4b_ans.contains("ceo.payroll.secret@acme.test"),
        "4b laundered wiki leaked foreign email: {l4b_ans}"
    );
    assert!(
        l4b_ans.contains("[PII-LITERAL:EMAIL:"),
        "expected Layer 1 neutralized [PII-LITERAL:EMAIL:...] in 4b: {l4b_ans}"
    );

    // Corpus sync laundering attempt (+ repeat sync SHA-256 No-Op preservation)
    let corpus_sync_payload = json!({
        "documents": [{
            "kind": "corpus",
            "external_key": "launder-corpus-sync",
            "title": "LaunderCorpusSync",
            "text": format!("LAUNDER-CORPUS-SYNC-999 {payroll_tok}"),
            "access_scope": "private",
            "min_auth_level": 1,
        }]
    });
    let (lcs_up_st, lcs_up_json, lcs_up_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bob_launder_bid}/documents/sync"),
        Some(&bob_launder_key),
        Some(corpus_sync_payload.clone()),
    )
    .await?;
    assert_eq!(lcs_up_st, StatusCode::OK, "corpus sync launder: {lcs_up_txt}");
    assert_eq!(lcs_up_json["documents"][0]["status"], "created");

    let (lcs_noop_st, lcs_noop_json, lcs_noop_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bob_launder_bid}/documents/sync"),
        Some(&bob_launder_key),
        Some(corpus_sync_payload),
    )
    .await?;
    assert_eq!(lcs_noop_st, StatusCode::OK, "repeat corpus sync: {lcs_noop_txt}");
    assert_eq!(
        lcs_noop_json["documents"][0]["status"], "skipped",
        "repeat corpus sync must preserve SHA-256 No-Op skip: {lcs_noop_json}"
    );

    let (lcs_st, lcs_json, lcs_txt) = send_json(
        &live_vllm_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_launder_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bob_launder_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "LAUNDER-CORPUS-SYNC-999"}],
        })),
    )
    .await?;
    assert_eq!(lcs_st, StatusCode::OK, "corpus sync launder query: {lcs_txt}");
    let lcs_ans = lcs_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(lcs_ans.contains("LAUNDER-CORPUS-SYNC-999"), "expected corpus sync hit: {lcs_ans}");
    assert!(!lcs_ans.contains("ceo.payroll.secret@acme.test"));
    assert!(lcs_ans.contains("[PII-LITERAL:EMAIL:"));

    // Layer 2 verification: raw-SQL-inserted corpus.documents / corpus.pages / corpus.chunks decoy
    // containing verbatim [PII:EMAIL:<hex>] with ZERO pii.chunk_spans rows
    let bob_launder_bid_uuid = Uuid::parse_str(bob_launder_bid)?;
    let decoy_doc_id = Uuid::now_v7();
    let decoy_page_id = Uuid::now_v7();
    let decoy_chunk_id = Uuid::now_v7();
    let decoy_text = format!("LAUNDER-SQL-999 {payroll_tok}");
    let decoy_sha = sha256_hex_str(&decoy_text);
    let decoy_emb = DeterministicTextEmbedder::embed_sync(&decoy_text)?;
    let decoy_acl_scope = format!("{}:private:{}:L1", seeded.org_id, seeded.dept_id);
    let decoy_doc_arn = Arn::new(
        &seeded.slug,
        "dept_a",
        AccessScope::Private,
        ArnResourceType::Document,
        AuthLevel::new(1)?,
        decoy_doc_id.to_string(),
    )?
    .to_canonical();
    let decoy_page_arn = Arn::new(
        &seeded.slug,
        "dept_a",
        AccessScope::Private,
        ArnResourceType::Page,
        AuthLevel::new(1)?,
        decoy_page_id.to_string(),
    )?
    .to_canonical();
    let decoy_chunk_arn = Arn::new(
        &seeded.slug,
        "dept_a",
        AccessScope::Private,
        ArnResourceType::Chunk,
        AuthLevel::new(1)?,
        decoy_chunk_id.to_string(),
    )?
    .to_canonical();

    sqlx::query(
        "INSERT INTO corpus.documents (
            id, org_id, bucket_id, dept_id, owner_principal_id,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
            arn, acl_scope_key, title, mime_type, content_sha256,
            sensitivity, status
         ) VALUES (
            $1, $2, $3, $4, $5,
            'private', ARRAY[]::uuid[], ARRAY[]::uuid[], 1,
            $6, $7, 'RawSqlDecoyDoc', 'text/plain', $8,
            'normal', 'indexed'
         )",
    )
    .bind(decoy_doc_id)
    .bind(seeded.org_id)
    .bind(bob_launder_bid_uuid)
    .bind(seeded.dept_id)
    .bind(bob_launder_id)
    .bind(&decoy_doc_arn)
    .bind(&decoy_acl_scope)
    .bind(&decoy_sha)
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.pages (
            id, org_id, document_id, dept_id, owner_principal_id,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
            arn, acl_scope_key, page_no, raw_markdown, redacted_markdown,
            content_sha256
         ) VALUES (
            $1, $2, $3, $4, $5,
            'private', ARRAY[]::uuid[], ARRAY[]::uuid[], 1,
            $6, $7, 1, $8, $8,
            $9
         )",
    )
    .bind(decoy_page_id)
    .bind(seeded.org_id)
    .bind(decoy_doc_id)
    .bind(seeded.dept_id)
    .bind(bob_launder_id)
    .bind(&decoy_page_arn)
    .bind(&decoy_acl_scope)
    .bind(&decoy_text)
    .bind(&decoy_sha)
    .execute(worker_pool.inner())
    .await?;

    sqlx::query(
        "INSERT INTO corpus.chunks (
            id, org_id, document_id, page_id, dept_id, owner_principal_id,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
            arn, acl_scope_key, chunk_idx, redacted_text, bm25_text,
            text_hash, embedding, embedding_model
         ) VALUES (
            $1, $2, $3, $4, $5, $6,
            'private', ARRAY[]::uuid[], ARRAY[]::uuid[], 1,
            $7, $8, 0, $9, $9,
            $10, $11, 'deterministic-256d'
         )",
    )
    .bind(decoy_chunk_id)
    .bind(seeded.org_id)
    .bind(decoy_doc_id)
    .bind(decoy_page_id)
    .bind(seeded.dept_id)
    .bind(bob_launder_id)
    .bind(&decoy_chunk_arn)
    .bind(&decoy_acl_scope)
    .bind(&decoy_text)
    .bind(&decoy_sha)
    .bind(Vector::from(decoy_emb))
    .execute(worker_pool.inner())
    .await?;

    let (sql_st, sql_json, sql_txt) = send_json(
        &live_vllm_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&bob_launder_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bob_launder_bid,
            "unmask_pii": true,
            "messages": [{"role": "user", "content": "LAUNDER-SQL-999"}],
        })),
    )
    .await?;
    assert_eq!(sql_st, StatusCode::OK, "raw-SQL decoy query failed: {sql_txt}");
    let sql_ans = sql_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(sql_ans.contains("LAUNDER-SQL-999"), "expected raw-SQL decoy hit: {sql_ans}");
    assert!(
        !sql_ans.contains("ceo.payroll.secret@acme.test"),
        "Layer 2 failed: raw-SQL decoy chunk leaked foreign email: {sql_ans}"
    );
    assert!(
        sql_ans.contains(&payroll_tok),
        "expected sealed token to remain un-unmasked on raw-SQL decoy: {sql_ans}"
    );

    let bob_launder_allowed: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = true",
    )
    .bind(seeded.org_id)
    .bind(bob_launder_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(bob_launder_allowed, 0, "bob_launder_member must have ZERO allowed=true audit rows");

    let bob_launder_denied: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.audit_logs
         WHERE org_id = $1 AND principal_id = $2 AND allowed = false",
    )
    .bind(seeded.org_id)
    .bind(bob_launder_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(bob_launder_denied >= 1, "expected denied audit row for raw-SQL decoy attempt");

    // 2. LlmEgress rrn_kr Block catches raw RRN BEFORE Capsule sealing, 0 request_logs written
    let logs_before: i64 =
        sqlx::query_scalar("SELECT count(*) FROM gateway.request_logs WHERE org_id = $1")
            .bind(seeded.org_id)
            .fetch_one(worker_pool.inner())
            .await?;

    let (rrn_status, rrn_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&admin_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "Check citizen RRN 900101-1234567 status"}],
        })),
    )
    .await?;
    assert_eq!(rrn_status, StatusCode::FORBIDDEN);
    assert_eq!(rrn_json["code"], "guard_blocked");

    let logs_after: i64 =
        sqlx::query_scalar("SELECT count(*) FROM gateway.request_logs WHERE org_id = $1")
            .bind(seeded.org_id)
            .fetch_one(worker_pool.inner())
            .await?;
    assert_eq!(logs_before, logs_after);

    // 3. allow_unmasked_pii: true (L3 vs L8 route to vllm_local, LlmEgress Mask applied, disabled vllm_local -> 403)
    let (_, b3_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&l8_key),
        Some(json!({
            "slug": format!("b3-unmasked-{}", Uuid::new_v4().simple()),
            "name": "B3 Unmasked",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let b3_id = b3_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing b3_id"))?;
    let _ = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{b3_id}/documents"),
        Some(&l8_key),
        Some(json!({
            "title": "L3 Unmasked Doc",
            "text": "UnmaskedQueryL3 reach plain.l3@acme.test",
            "access_scope": "dept",
            "min_auth_level": 3,
        })),
    )
    .await?;
    let _ = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&l8_key),
        Some(json!({
            "title": "L8 Unmasked Doc",
            "text": "UnmaskedQueryL8 reach plain.l8@acme.test",
            "access_scope": "dept",
            "min_auth_level": 8,
        })),
    )
    .await?;

    for (target_bid, q, expected_lvl) in [
        (b3_id, "UnmaskedQueryL3", 3),
        (bid, "UnmaskedQueryL8", 8),
    ] {
        let (au_status, au_json, au_txt) = send_json(
            &app,
            Method::POST,
            "/v1/chat/completions",
            Some(&l8_key),
            Some(json!({
                "model": "openrag-auto",
                "bucket_id": target_bid,
                "allow_unmasked_pii": true,
                "messages": [{"role": "user", "content": format!("{q} for caller@acme.test")}],
            })),
        )
        .await?;
        assert_eq!(au_status, StatusCode::OK, "allow_unmasked_pii failed: {au_txt}");
        assert_eq!(au_json["provider"], "vllm_local");
        assert_eq!(au_json["is_on_premise"], true);
        assert_eq!(au_json["pii_masked"], false);
        assert_eq!(au_json["max_context_auth_level"], expected_lvl);
        let ans = au_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!ans.contains("plain.l3@acme.test"));
        assert!(!ans.contains("plain.l8@acme.test"));
        assert!(!ans.contains("caller@acme.test"));
    }

    // Disabling vllm_local causes allow_unmasked_pii: true -> 403
    sqlx::query("UPDATE gateway.providers SET enabled = false WHERE org_id = $1 AND name = 'vllm_local'")
        .bind(seeded.org_id)
        .execute(worker_pool.inner())
        .await?;
    let (dis_status, _, _) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&l8_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "allow_unmasked_pii": true,
            "messages": [{"role": "user", "content": "UnmaskedQueryL3"}],
        })),
    )
    .await?;
    assert_eq!(dis_status, StatusCode::FORBIDDEN);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_multipage_mask_sync_guard_rls_concepts_and_concurrency() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;

    // 8. Concurrent ensure_default_guard_policy on a fresh org produces zero duplicate rules
    let conc_org_id = Uuid::new_v4();
    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, 'Conc Org')")
        .bind(conc_org_id)
        .bind(format!("conc-{}", Uuid::new_v4().simple()))
        .execute(worker_pool.inner())
        .await?;
    let pipeline_snap = state.pipeline_config.read().await.clone();
    let expected_rule_count = pipeline_snap.pii.guard.rules.len() as i64;
    let mut handles = Vec::new();
    for _ in 0..8 {
        let wp = worker_pool.clone();
        let pipe = pipeline_snap.clone();
        handles.push(tokio::spawn(async move {
            ensure_default_guard_policy(&wp, conc_org_id, &pipe).await
        }));
    }
    for h in handles {
        let _ = h.await??;
    }
    let pol_cnt: i64 =
        sqlx::query_scalar("SELECT count(*) FROM pii.guard_policies WHERE org_id = $1")
            .bind(conc_org_id)
            .fetch_one(worker_pool.inner())
            .await?;
    let rule_cnt: i64 =
        sqlx::query_scalar("SELECT count(*) FROM pii.guard_rules WHERE org_id = $1")
            .bind(conc_org_id)
            .fetch_one(worker_pool.inner())
            .await?;
    assert_eq!(pol_cnt, 1);
    assert_eq!(rule_cnt, expected_rule_count);

    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;
    let other_key = mint_key(&app, &worker_pool, seeded.other_org_id, seeded.other_admin_id, None, vec![]).await?;
    let l9_clamped_l3_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, Some(3), vec![]).await?;

    // Seed IngestionGate rules: PROJECT-BLACKBOX Block, MASK-PAGE2-SECRET Mask
    let pol = GuardRepo::get_or_create_active_policy(worker_pool.inner(), seeded.org_id, "default")
        .await?;
    let _ = GuardRepo::create_rule(
        worker_pool.inner(),
        seeded.org_id,
        pol.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "PROJECT-BLACKBOX".into(),
            action: GuardAction::Block,
            surfaces: vec![GuardSurface::IngestionGate, GuardSurface::LlmEgress],
            replacement_text: None,
            enabled: true,
            priority: 1,
        },
    )
    .await?;
    let _ = GuardRepo::create_rule(
        worker_pool.inner(),
        seeded.org_id,
        pol.id,
        &GuardRuleConfig {
            kind: GuardRuleKind::SafeWord,
            pattern_or_category: "MASK-PAGE2-SECRET".into(),
            action: GuardAction::Mask,
            surfaces: vec![GuardSurface::IngestionGate],
            replacement_text: Some("[MASKED-SW]".into()),
            enabled: true,
            priority: 10,
        },
    )
    .await?;

    let bkt_slug = format!("sync-m4c2-{}", &Uuid::new_v4().simple().to_string()[..8]);
    let (b_status, b_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": bkt_slug,
            "name": "Sync Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(b_status, StatusCode::CREATED);
    let bid = b_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket id"))?;

    // 4. 2-page HWPX with IngestionGate GuardAction::Mask masks page 2 in corpus.pages and corpus.chunks
    let hwpx2 = build_two_page_hwpx_bytes(
        "Page one safe text",
        "Page two contains MASK-PAGE2-SECRET confidential",
    )?;
    let boundary = "----OpenRagMultiPageBoundary";
    let mp_body = build_multipart_body(
        boundary,
        &[("title", "TwoPageHWPX"), ("access_scope", "dept")],
        Some(("two_page.hwpx", "application/hwp+zip", &hwpx2)),
    );
    let mp_req = Request::builder()
        .method(Method::POST)
        .uri(format!("/v2/buckets/{bid}/documents"))
        .header(AUTHORIZATION, format!("Bearer {admin_key}"))
        .header(CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(mp_body))?;
    let mp_resp = app.clone().oneshot(mp_req).await?;
    assert_eq!(mp_resp.status(), StatusCode::CREATED);
    let mp_json: Value = serde_json::from_slice(&mp_resp.into_body().collect().await?.to_bytes())?;
    let mp_doc_id = Uuid::parse_str(
        mp_json["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing mp doc id"))?,
    )?;
    let page2_redacted: String = sqlx::query_scalar(
        "SELECT redacted_markdown FROM corpus.pages WHERE org_id = $1 AND document_id = $2 AND page_no = 2",
    )
    .bind(seeded.org_id)
    .bind(mp_doc_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(!page2_redacted.contains("MASK-PAGE2-SECRET"));
    assert!(page2_redacted.contains("[MASKED-SW]"));

    let chunk2_texts: Vec<String> = sqlx::query_scalar(
        "SELECT redacted_text FROM corpus.chunks WHERE org_id = $1 AND document_id = $2",
    )
    .bind(seeded.org_id)
    .bind(mp_doc_id)
    .fetch_all(worker_pool.inner())
    .await?;
    assert!(chunk2_texts.iter().all(|t| !t.contains("MASK-PAGE2-SECRET")));
    assert!(chunk2_texts.iter().any(|t| t.contains("[MASKED-SW]")));

    // 5a. PUT /v1/buckets/{bucket}/documents/sync: PROJECT-BLACKBOX -> 403 guard_blocked + guard_events logged
    let (sb_status, sb_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "external_key": "sync-blk",
                "text": "Contains PROJECT-BLACKBOX secret",
            }]
        })),
    )
    .await?;
    assert_eq!(sb_status, StatusCode::FORBIDDEN);
    assert_eq!(sb_json["code"], "guard_blocked");
    let guard_ev_cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.guard_events WHERE org_id = $1 AND surface = 'ingestion_gate' AND decision = 'block'",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(guard_ev_cnt >= 1);

    // 5b. Sync PII + Mask SafeWord -> sealed in corpus.chunks & pii.chunk_spans populated + No-Op watermark invariant
    let sync_pii_payload = json!({
        "documents": [{
            "external_key": "sync-pii-1",
            "title": "Sync PII",
            "pages": [
                "Page 1 phone 010-9876-5432 and email sync.user@acme.test and MASK-PAGE2-SECRET",
                "Page 2 steady",
                "Page 3 trailing"
            ],
            "content_sha256": "sync-sha-v1"
        }]
    });
    let (sp1_status, sp1_json, sp1_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(sync_pii_payload.clone()),
    )
    .await?;
    assert_eq!(sp1_status, StatusCode::OK, "sync pii failed: {sp1_txt}");
    let synced_doc_id = Uuid::parse_str(
        sp1_json["documents"][0]["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing synced doc id"))?,
    )?;
    let p1_chunk_txt: String = sqlx::query_scalar(
        "SELECT redacted_text FROM corpus.chunks
         WHERE org_id = $1 AND document_id = $2
         ORDER BY chunk_idx ASC LIMIT 1",
    )
    .bind(seeded.org_id)
    .bind(synced_doc_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(!p1_chunk_txt.contains("010-9876-5432"));
    assert!(!p1_chunk_txt.contains("sync.user@acme.test"));
    assert!(!p1_chunk_txt.contains("MASK-PAGE2-SECRET"));
    assert!(p1_chunk_txt.contains("[PII:PHONE:"));
    assert!(p1_chunk_txt.contains("[PII:EMAIL:"));
    assert!(p1_chunk_txt.contains("[MASKED-SW]"));

    let chunk_spans_cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.chunk_spans cs
         JOIN corpus.chunks c ON c.id = cs.chunk_id
         WHERE cs.org_id = $1 AND c.document_id = $2",
    )
    .bind(seeded.org_id)
    .bind(synced_doc_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(chunk_spans_cnt >= 2);

    // Repeat No-Op preserves indexing.watermarks.indexed_count
    let wm_before: i64 = sqlx::query_scalar(
        "SELECT COALESCE(SUM(indexed_count), 0)::bigint FROM indexing.watermarks WHERE org_id = $1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    let (sp_noop_status, sp_noop_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(sync_pii_payload),
    )
    .await?;
    assert_eq!(sp_noop_status, StatusCode::OK);
    assert_eq!(sp_noop_json["skipped"], 1);
    let wm_after: i64 = sqlx::query_scalar(
        "SELECT COALESCE(SUM(indexed_count), 0)::bigint FROM indexing.watermarks WHERE org_id = $1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(wm_before, wm_after);

    // 5c. 1-of-3 delta update < 3 AND shrink (3 -> 1 page) + delete enqueues DeleteVector before DELETE
    let (delta_status, delta_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "external_key": "sync-pii-1",
                "pages": [
                    "Page 1 phone 010-9876-5432 and email sync.user@acme.test and MASK-PAGE2-SECRET",
                    "Page 2 updated delta",
                    "Page 3 trailing"
                ],
                "content_sha256": "sync-sha-v2"
            }]
        })),
    )
    .await?;
    assert_eq!(delta_status, StatusCode::OK);
    assert_eq!(delta_json["delta_updated"], 1);
    assert_eq!(delta_json["upserted_chunks"], 1);

    // Shrink 3 pages -> 1 page
    let (shrink_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "external_key": "sync-pii-1",
                "pages": ["Page 1 phone 010-9876-5432 and email sync.user@acme.test and MASK-PAGE2-SECRET"],
                "content_sha256": "sync-sha-v3"
            }]
        })),
    )
    .await?;
    assert_eq!(shrink_status, StatusCode::OK);
    let orphan_chunks: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM corpus.chunks WHERE org_id = $1 AND document_id = $2 AND page_id IS NULL",
    )
    .bind(seeded.org_id)
    .bind(synced_doc_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(orphan_chunks, 0);

    let del_vec_after_shrink: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM indexing.outbox
         WHERE org_id = $1 AND op = 'delete_vector' AND payload->>'document_id' = $2",
    )
    .bind(seeded.org_id)
    .bind(synced_doc_id.to_string())
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(del_vec_after_shrink >= 3);

    // Delete via sync -> enqueues DeleteVector before DELETE
    let (sdel_status, sdel_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bkt_slug}/documents/sync"),
        Some(&admin_key),
        Some(json!({"documents": [{"external_key": "sync-pii-1", "deleted": true}]})),
    )
    .await?;
    assert_eq!(sdel_status, StatusCode::OK);
    assert_eq!(sdel_json["deleted"], 1);

    // 6. L9 user with max_auth_level=3 key uploading min_auth_level=5 doc -> 403 rls_denied (not 500)
    let (rls_up_status, rls_up_json, rls_up_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&l9_clamped_l3_key),
        Some(json!({
            "title": "L5 Secret",
            "text": "Classified L5 material",
            "access_scope": "dept",
            "min_auth_level": 5,
        })),
    )
    .await?;
    assert_eq!(rls_up_status, StatusCode::FORBIDDEN, "expected 403 rls_denied: {rls_up_txt}");
    assert_eq!(rls_up_json["code"], "rls_denied");

    // 7. POST /v1/concepts/link with explicit concept_id preserves concept_id + GET /v1/concepts?query=&lang= + GET /v1/concepts/{id}
    let (cc_status, cc_json, _) = send_json(
        &app,
        Method::POST,
        "/v1/concepts",
        Some(&owner_key),
        Some(json!({
            "canonical_name": "FruitApple",
            "aliases": ["사과"],
        })),
    )
    .await?;
    assert_eq!(cc_status, StatusCode::CREATED);
    let cid = cc_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing concept id"))?;

    let (lk_status, lk_json, lk_txt) = send_json(
        &app,
        Method::POST,
        "/v1/concepts/link",
        Some(&owner_key),
        Some(json!({
            "concept_id": cid,
            "source": "Apple",
            "target": "풋사과",
        })),
    )
    .await?;
    assert_eq!(lk_status, StatusCode::CREATED, "concept_id link failed: {lk_txt}");
    assert_eq!(lk_json["concept_id"], cid);
    let aliases_str = lk_json["aliases"].to_string();
    assert!(aliases_str.contains("Apple") && aliases_str.contains("풋사과"));

    let (lq_ok, lq_ok_json, _) = send_json(
        &app,
        Method::GET,
        "/v1/concepts?query=apple&lang=ko",
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(lq_ok, StatusCode::OK);
    assert!(!lq_ok_json["concepts"].as_array().is_none_or(Vec::is_empty));

    let (lq_fr, lq_fr_json, _) = send_json(
        &app,
        Method::GET,
        "/v1/concepts?query=apple&lang=fr",
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(lq_fr, StatusCode::OK);
    assert!(lq_fr_json["concepts"].as_array().is_some_and(Vec::is_empty));

    let (gc_ok, gc_ok_json, _) = send_json(
        &app,
        Method::GET,
        &format!("/v1/concepts/{cid}"),
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(gc_ok, StatusCode::OK);
    assert_eq!(gc_ok_json["id"], cid);

    let (gc_cross, _, _) = send_json(
        &app,
        Method::GET,
        &format!("/v1/concepts/{cid}"),
        Some(&other_key),
        None,
    )
    .await?;
    assert_eq!(gc_cross, StatusCode::NOT_FOUND);

    let (gc_missing, _, _) = send_json(
        &app,
        Method::GET,
        &format!("/v1/concepts/{}", Uuid::new_v4()),
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(gc_missing, StatusCode::NOT_FOUND);

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_s3_caps_routing_rules_live_http_tags_and_list_filters() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;

    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;
    let peer_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;
    let other_key = mint_key(&app, &worker_pool, seeded.other_org_id, seeded.other_admin_id, None, vec![]).await?;

    let (_, b_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("area-bkt-{}", Uuid::new_v4().simple()),
            "name": "Area Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let bid = b_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket id"))?;

    // 9a. DB-seeded vectors.index_bindings (s3_vectors) enforces caps when external_backend is omitted
    PgPineconeManager::set_external_mirror(
        &worker_pool,
        seeded.org_id,
        VectorTable::CorpusChunks,
        ExternalMirrorBackend::S3Vectors,
        json!({}),
    )
    .await?;

    let big_meta_val = "x".repeat(2050);
    let (s3_meta_err, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "Too Big Meta",
            "text": "Normal text",
            "filterable_metadata": {"blob": big_meta_val},
        })),
    )
    .await?;
    assert_eq!(s3_meta_err, StatusCode::BAD_REQUEST);

    let big_text = "a".repeat(30721);
    let (s3_txt_err, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "Too Big Text",
            "text": big_text,
        })),
    )
    .await?;
    assert_eq!(s3_txt_err, StatusCode::BAD_REQUEST);

    let ok_text = "b".repeat(1000);
    let (s3_ok, _, s3_ok_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "Fits S3 Caps",
            "text": ok_text,
            "filterable_metadata": {"dept": "eng"},
        })),
    )
    .await?;
    assert_eq!(s3_ok, StatusCode::CREATED, "s3 ok failed: {s3_ok_txt}");

    // 9b. Reset DB binding to none, set pipeline.store.external_mirror = S3Vectors -> still enforces caps
    PgPineconeManager::set_external_mirror(
        &worker_pool,
        seeded.org_id,
        VectorTable::CorpusChunks,
        ExternalMirrorBackend::None,
        json!({}),
    )
    .await?;
    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.store.external_mirror = ExternalMirrorBackend::S3Vectors;
    }
    let (cfg_s3_err, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "Pipeline S3 Cap",
            "text": "Normal",
            "filterable_metadata": {"blob": "y".repeat(2050)},
        })),
    )
    .await?;
    assert_eq!(cfg_s3_err, StatusCode::BAD_REQUEST);
    {
        let mut cfg = state.pipeline_config.write().await;
        cfg.store.external_mirror = ExternalMirrorBackend::None;
    }

    // 10a. gateway.routing_rules priority & fallback provider selection in PG17
    openrag_server::llm_router::ensure_default_providers(&worker_pool, seeded.org_id).await?;
    let custom_ext_id = Uuid::new_v4();
    let vllm_id: Uuid = sqlx::query_scalar(
        "SELECT id FROM gateway.providers WHERE org_id = $1 AND name = 'vllm_local'",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO gateway.providers (
            id, org_id, name, provider_type, base_url, model_id,
            is_on_premise, max_auth_level, allow_unmasked_pii, config, enabled
         ) VALUES ($1, $2, 'l5-custom', 'openai', 'https://api.openai.com/v1', 'gpt-4o-mini', false, 5, false, '{}'::jsonb, false)",
    )
    .bind(custom_ext_id)
    .bind(seeded.org_id)
    .execute(worker_pool.inner())
    .await?;
    sqlx::query(
        "INSERT INTO gateway.routing_rules (
            org_id, priority, min_auth_level, max_auth_level,
            require_pii_masked, target_provider_id, fallback_provider_id, enabled
         ) VALUES ($1, 1, 1, 10, true, $2, $3, true)",
    )
    .bind(seeded.org_id)
    .bind(custom_ext_id)
    .bind(vllm_id)
    .execute(worker_pool.inner())
    .await?;

    let (rr_status, rr_json, rr_txt) = send_json(
        &app,
        Method::POST,
        "/v1/chat/completions",
        Some(&admin_key),
        Some(json!({
            "model": "openrag-auto",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "routing rule fallback check"}],
        })),
    )
    .await?;
    assert_eq!(rr_status, StatusCode::OK, "routing rule chat failed: {rr_txt}");
    assert_eq!(rr_json["provider"], "vllm_local");
    let logged_provider_id: Option<Uuid> = sqlx::query_scalar(
        "SELECT provider_id FROM gateway.request_logs WHERE org_id = $1 ORDER BY created_at DESC LIMIT 1",
    )
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(logged_provider_id, Some(vllm_id));

    // 10b. Live TcpListener + axum::serve mock OpenAI server under OpenRagMode::Live
    type CapturedBodyList = Vec<(Option<String>, Value)>;
    let captured_bodies: Arc<tokio::sync::Mutex<CapturedBodyList>> =
        Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let return_500 = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cap_clone = Arc::clone(&captured_bodies);
    let err_flag_clone = Arc::clone(&return_500);

    let mock_router = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(
            move |headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
                let cap = Arc::clone(&cap_clone);
                let flag = Arc::clone(&err_flag_clone);
                async move {
                    let auth = headers
                        .get(AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        .map(ToString::to_string);
                    cap.lock().await.push((auth, body));
                    if flag.load(std::sync::atomic::Ordering::SeqCst) {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            axum::Json(json!({"error": "upstream down"})),
                        );
                    }
                    (
                        StatusCode::OK,
                        axum::Json(json!({
                            "choices": [{"message": {"content": "MOCK_OPENAI_LIVE_OK"}}]
                        })),
                    )
                }
            },
        ),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let srv_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, mock_router).await;
    });

    let live_env_name = "OPENRAG_TEST_LIVE_MOCK_KEY";
    std::env::set_var(live_env_name, "sk-live-mock-999");
    sqlx::query(
        "UPDATE gateway.providers
         SET base_url = $2, config = $3
         WHERE org_id = $1 AND name = 'openrouter'",
    )
    .bind(seeded.org_id)
    .bind(format!("http://{addr}/v1"))
    .bind(json!({"api_key_env": live_env_name}))
    .execute(worker_pool.inner())
    .await?;
    // Remove routing rule so L3 masked request routes to openrouter
    sqlx::query("DELETE FROM gateway.routing_rules WHERE org_id = $1")
        .bind(seeded.org_id)
        .execute(worker_pool.inner())
        .await?;

    let _ = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({
            "title": "LivePiiDoc",
            "text": "LiveProbeTag888 email is live.leak.check@acme.test",
            "access_scope": "dept",
            "min_auth_level": 3,
        })),
    )
    .await?;

    let live_state = state.with_mode_and_transport(OpenRagMode::Live, Arc::clone(&state.llm_transport));
    let live_app = build_router(live_state);
    let (live_ok_status, live_ok_json, live_ok_txt) = send_json(
        &live_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&admin_key),
        Some(json!({
            "model": "anthropic/claude-3.5-sonnet",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "LiveProbeTag888"}],
        })),
    )
    .await?;
    assert_eq!(live_ok_status, StatusCode::OK, "live http mock failed: {live_ok_txt}");
    assert_eq!(
        live_ok_json["choices"][0]["message"]["content"],
        "MOCK_OPENAI_LIVE_OK"
    );
    {
        let cap = captured_bodies.lock().await;
        assert_eq!(cap.len(), 1);
        assert_eq!(cap[0].0.as_deref(), Some("Bearer sk-live-mock-999"));
        assert!(!cap[0].1.to_string().contains("live.leak.check@acme.test"));
    }

    return_500.store(true, std::sync::atomic::Ordering::SeqCst);
    let (live_503_status, _, _) = send_json(
        &live_app,
        Method::POST,
        "/v1/chat/completions",
        Some(&admin_key),
        Some(json!({
            "model": "anthropic/claude-3.5-sonnet",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "LiveProbeTag888"}],
        })),
    )
    .await?;
    assert_eq!(live_503_status, StatusCode::SERVICE_UNAVAILABLE);
    srv_handle.abort();

    // 11a. Authenticated Wiki/Memory Tag CRUD (/v1/wiki/pages/{id}/tags, /v1/memories/{id}/tags)
    let (_, ob_json, ob_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({
            "slug": format!("owner-wiki-bkt-{}", Uuid::new_v4().simple()),
            "name": "Owner Wiki Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let owner_bid = ob_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing owner bucket id: {ob_txt}"))?;

    let (wsync_status, wsync_json, wsync_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{owner_bid}/documents/sync"),
        Some(&owner_key),
        Some(json!({
            "documents": [{
                "slug": "owner-dept-wiki",
                "title": "Owner Dept Wiki",
                "body_md": "# Wiki Tag Test",
                "access_scope": "dept",
            }, {
                "slug": "owner-priv-wiki",
                "title": "Owner Private Wiki",
                "body_md": "# Private Wiki",
                "access_scope": "private",
            }]
        })),
    )
    .await?;
    assert_eq!(wsync_status, StatusCode::OK, "wiki sync failed: {wsync_txt}");
    let dept_wiki_id = wsync_json["documents"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing dept wiki id: {wsync_txt}"))?;
    let priv_wiki_id = wsync_json["documents"][1]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing priv wiki id: {wsync_txt}"))?;

    let (wt_add, wt_add_json, wt_add_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v1/wiki/pages/{dept_wiki_id}/tags"),
        Some(&owner_key),
        Some(json!({"tags": ["  #Tag-A ", "tag-a", "   ", "Tag-B"]})),
    )
    .await?;
    assert_eq!(wt_add, StatusCode::OK, "wiki tag add failed: {wt_add_txt}");
    assert_eq!(wt_add_json["tags"], json!(["tag-a", "tag-b"]));

    // Same-dept peer lacks write -> 403 without modifying tags
    let (wt_peer, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v1/wiki/pages/{dept_wiki_id}/tags"),
        Some(&peer_key),
        Some(json!({"tags": ["hacked"]})),
    )
    .await?;
    assert_eq!(wt_peer, StatusCode::FORBIDDEN);
    // Private wiki -> peer gets 404; cross-org -> 404; missing UUID -> 404
    let (wt_priv_peer, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v1/wiki/pages/{priv_wiki_id}/tags"),
        Some(&peer_key),
        Some(json!({"tags": ["hacked"]})),
    )
    .await?;
    assert_eq!(wt_priv_peer, StatusCode::NOT_FOUND);
    let (wt_other, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v1/wiki/pages/{dept_wiki_id}/tags"),
        Some(&other_key),
        Some(json!({"tags": ["hacked"]})),
    )
    .await?;
    assert_eq!(wt_other, StatusCode::NOT_FOUND);
    let (wt_miss, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/wiki/pages/{}/tags", Uuid::new_v4()),
        Some(&owner_key),
        Some(json!({"tags": ["tag-a"]})),
    )
    .await?;
    assert_eq!(wt_miss, StatusCode::NOT_FOUND);

    let (wt_del, wt_del_json, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/wiki/pages/{dept_wiki_id}/tags"),
        Some(&owner_key),
        Some(json!({"tags": ["tag-a"]})),
    )
    .await?;
    assert_eq!(wt_del, StatusCode::OK);
    assert_eq!(wt_del_json["tags"], json!(["tag-b"]));

    // Memory entries (dept + private)
    let mut owner_ctx = PrincipalContext::new(seeded.org_id, seeded.owner_id, AuthLevel::new(3)?);
    owner_ctx.dept_ids = vec![seeded.dept_id];
    let dept_mem = state
        .memory_router
        .reflect(
            &owner_ctx,
            ReflectRequest {
                dept_id: seeded.dept_id,
                org_slug: seeded.slug.clone(),
                dept_path: "dept1".into(),
                content: "Dept memory for tag CRUD".into(),
                title: Some("DeptMem".into()),
                memory_kind: Some(MemoryKind::Fact),
                episode_id: None,
                access_scope: AccessScope::Dept,
                allowed_group_ids: Vec::new(),
                denied_group_ids: Vec::new(),
                min_auth_level: AuthLevel::new(1)?,
                tags: Vec::new(),
                edge_type: None,
                target_memory_id: None,
                extract_concepts: false,
            },
            None,
        )
        .await?;
    let priv_mem = state
        .memory_router
        .reflect(
            &owner_ctx,
            ReflectRequest {
                dept_id: seeded.dept_id,
                org_slug: seeded.slug.clone(),
                dept_path: "dept1".into(),
                content: "Private memory for tag CRUD".into(),
                title: Some("PrivMem".into()),
                memory_kind: Some(MemoryKind::Fact),
                episode_id: None,
                access_scope: AccessScope::Private,
                allowed_group_ids: Vec::new(),
                denied_group_ids: Vec::new(),
                min_auth_level: AuthLevel::new(1)?,
                tags: Vec::new(),
                edge_type: None,
                target_memory_id: None,
                extract_concepts: false,
            },
            None,
        )
        .await?;
    let dept_mem_id = dept_mem.entry.id;
    let priv_mem_id = priv_mem.entry.id;

    let (mt_add, mt_add_json, mt_add_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v1/memories/{dept_mem_id}/tags"),
        Some(&owner_key),
        Some(json!({"tags": ["  #Mem-A ", "mem-a", "", "Mem-B"]})),
    )
    .await?;
    assert_eq!(mt_add, StatusCode::OK, "mem tag add failed: {mt_add_txt}");
    assert_eq!(mt_add_json["tags"], json!(["mem-a", "mem-b"]));

    let (mt_peer, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v1/memories/{dept_mem_id}/tags"),
        Some(&peer_key),
        Some(json!({"tags": ["hacked"]})),
    )
    .await?;
    assert_eq!(mt_peer, StatusCode::FORBIDDEN);

    let (mt_priv_peer, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v1/memories/{priv_mem_id}/tags"),
        Some(&peer_key),
        Some(json!({"tags": ["hacked"]})),
    )
    .await?;
    assert_eq!(mt_priv_peer, StatusCode::NOT_FOUND);

    let (mt_other, _, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/memories/{dept_mem_id}/tags"),
        Some(&other_key),
        Some(json!({"tags": ["mem-a"]})),
    )
    .await?;
    assert_eq!(mt_other, StatusCode::NOT_FOUND);

    let (mt_del, mt_del_json, _) = send_json(
        &app,
        Method::DELETE,
        &format!("/v1/memories/{dept_mem_id}/tags"),
        Some(&owner_key),
        Some(json!({"tags": ["mem-a"]})),
    )
    .await?;
    assert_eq!(mt_del, StatusCode::OK);
    assert_eq!(mt_del_json["tags"], json!(["mem-b"]));

    // 11b. GET /v2/buckets & GET /v2/buckets/{id}/documents filtering
    let (_, priv_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({
            "slug": format!("priv-bkt-{}", Uuid::new_v4().simple()),
            "name": "Owner Private Bucket",
            "access_scope": "private",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let priv_bid = priv_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing priv bucket id"))?;

    let (lb_peer, lb_peer_json, _) =
        send_json(&app, Method::GET, "/v2/buckets", Some(&peer_key), None).await?;
    assert_eq!(lb_peer, StatusCode::OK);
    assert!(!lb_peer_json.to_string().contains(priv_bid));
    assert!(lb_peer_json.to_string().contains(bid));

    // Archived wiki page tag mutation -> 404
    sqlx::query("UPDATE wiki.pages SET status = 'archived' WHERE org_id = $1 AND id = $2::uuid")
        .bind(seeded.org_id)
        .bind(dept_wiki_id)
        .execute(worker_pool.inner())
        .await?;
    let (wt_arch, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/v1/wiki/pages/{dept_wiki_id}/tags"),
        Some(&owner_key),
        Some(json!({"tags": ["stale"]})),
    )
    .await?;
    assert_eq!(wt_arch, StatusCode::NOT_FOUND);

    // Dept2 bucket excluded by dept1 scoped_arns on GET /v2/buckets
    let (_, d2_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("d2-list-{}", Uuid::new_v4().simple()),
            "name": "Dept2 List",
            "dept_id": seeded.dept2_id,
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    let d2_bid = d2_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing d2_bid"))?;
    let scoped_dept1_key = mint_key(
        &app, &worker_pool,
        seeded.org_id,
        seeded.admin_id,
        None,
        vec![format!("arn:openrag:{}:dept1:*", seeded.slug)],
    )
    .await?;
    let (lb_scoped, lb_scoped_json, _) =
        send_json(&app, Method::GET, "/v2/buckets", Some(&scoped_dept1_key), None).await?;
    assert_eq!(lb_scoped, StatusCode::OK);
    assert!(!lb_scoped_json.to_string().contains(d2_bid));
    assert!(lb_scoped_json.to_string().contains(bid));

    // L5 doc + deleted doc in bid
    let (_, l5_doc_json, l5_doc_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({"title": "L5 Hidden", "text": "L5 body", "min_auth_level": 5})),
    )
    .await?;
    let l5_doc_id = l5_doc_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing l5 doc id: {l5_doc_txt}"))?;

    let (_, del_doc_json, _) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&admin_key),
        Some(json!({"title": "To Delete", "text": "del body", "min_auth_level": 1})),
    )
    .await?;
    let del_doc_id = del_doc_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing del doc id"))?;
    let _ = send_json(
        &app,
        Method::DELETE,
        &format!("/v2/buckets/{bid}/documents/{del_doc_id}"),
        Some(&admin_key),
        None,
    )
    .await?;

    let l8_clamped_l3_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.l8_id, Some(3), vec![]).await?;
    let (ld_status, ld_json, _) = send_json(
        &app,
        Method::GET,
        &format!("/v2/buckets/{bid}/documents"),
        Some(&l8_clamped_l3_key),
        None,
    )
    .await?;
    assert_eq!(ld_status, StatusCode::OK);
    let ld_str = ld_json.to_string();
    assert!(!ld_str.contains(l5_doc_id));
    assert!(!ld_str.contains(del_doc_id));

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_wiki_sync_delete_by_slug_and_corpus_slug_delete() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;
    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let peer_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;

    let bkt_slug = format!("wiki-del-bkt-{}", &Uuid::new_v4().simple().to_string()[..8]);
    let (b_status, b_json, b_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": bkt_slug,
            "name": "Wiki Slug Delete Bucket",
            "access_scope": "dept",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(b_status, StatusCode::CREATED, "create bucket failed: {b_txt}");
    let bid = b_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket id"))?;

    // 1. Create wiki page via PUT /v1/buckets/{bucket}/documents/sync with slug + body_md
    let (cw_status, cw_json, cw_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "wiki-del-slug",
                "title": "Wiki Del Slug",
                "body_md": "# Wiki Del\nhello wiki delete",
            }]
        })),
    )
    .await?;
    assert_eq!(cw_status, StatusCode::OK, "create wiki via sync failed: {cw_txt}");
    assert_eq!(cw_json["updated"], 1);
    let wiki_page_id = Uuid::parse_str(
        cw_json["documents"][0]["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing wiki page id in {cw_json}"))?,
    )?;

    // 1b. Same-dept peer lacks write permission -> 403 Forbidden and wiki page remains active
    let (peer_dw_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&peer_key),
        Some(json!({
            "documents": [{
                "slug": "wiki-del-slug",
                "deleted": true,
            }]
        })),
    )
    .await?;
    assert_eq!(peer_dw_status, StatusCode::FORBIDDEN);
    let status_after_peer: String = sqlx::query_scalar(
        "SELECT status FROM wiki.pages WHERE org_id = $1 AND id = $2",
    )
    .bind(seeded.org_id)
    .bind(wiki_page_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(status_after_peer, "active");

    // 2. Delete wiki page omitting body_md/content/.md path -> must route to wiki, archive, and drain DeleteVector
    let (dw_status, dw_json, dw_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "wiki-del-slug",
                "deleted": true,
            }]
        })),
    )
    .await?;
    assert_eq!(dw_status, StatusCode::OK, "delete wiki by slug failed: {dw_txt}");
    assert_eq!(dw_json["deleted"], 1, "unexpected dw_json: {dw_json}");
    assert_eq!(dw_json["skipped"], 0);
    assert_eq!(dw_json["documents"][0]["kind"], "wiki");
    assert_eq!(dw_json["documents"][0]["status"], "deleted");

    let wiki_status: String = sqlx::query_scalar(
        "SELECT status FROM wiki.pages WHERE org_id = $1 AND id = $2",
    )
    .bind(seeded.org_id)
    .bind(wiki_page_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(wiki_status, "archived");

    let wiki_del_vec_done: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM indexing.outbox
         WHERE org_id = $1 AND target_table = 'wiki.pages' AND target_id = $2
           AND op = 'delete_vector' AND status = 'done'",
    )
    .bind(seeded.org_id)
    .bind(wiki_page_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(wiki_del_vec_done, 1);

    // 3. Repeat delete by slug is idempotent skip
    let (dw2_status, dw2_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "wiki-del-slug",
                "deleted": true,
            }]
        })),
    )
    .await?;
    assert_eq!(dw2_status, StatusCode::OK);
    assert_eq!(dw2_json["deleted"], 0);
    assert_eq!(dw2_json["skipped"], 1);

    // 4. Slug-keyed corpus document deletion still deletes from corpus.documents
    let (cc_status, cc_json, cc_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "corp-slug",
                "title": "Corp Slug Doc",
                "text": "corpus document keyed by slug",
            }]
        })),
    )
    .await?;
    assert_eq!(cc_status, StatusCode::OK, "create corpus by slug failed: {cc_txt}");
    assert_eq!(cc_json["updated"], 1);
    let corp_doc_id = Uuid::parse_str(
        cc_json["documents"][0]["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing corpus doc id in {cc_json}"))?,
    )?;

    let (dc_status, dc_json, dc_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "corp-slug",
                "deleted": true,
            }]
        })),
    )
    .await?;
    assert_eq!(dc_status, StatusCode::OK, "delete corpus by slug failed: {dc_txt}");
    assert_eq!(dc_json["deleted"], 1, "unexpected dc_json: {dc_json}");
    assert_eq!(dc_json["documents"][0]["kind"], "corpus");
    assert_eq!(dc_json["documents"][0]["status"], "deleted");

    let corp_remaining: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM corpus.documents WHERE org_id = $1 AND id = $2",
    )
    .bind(seeded.org_id)
    .bind(corp_doc_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(corp_remaining, 0);

    let corp_del_vec_done: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM indexing.outbox
         WHERE org_id = $1 AND target_table = 'corpus.chunks' AND op = 'delete_vector'
           AND payload->>'document_id' = $2 AND status = 'done'",
    )
    .bind(seeded.org_id)
    .bind(corp_doc_id.to_string())
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(corp_del_vec_done >= 1);

    // 5. Explicit kind = "wiki" deletion works and invalid kind returns 400
    let (cw2_status, cw2_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "wiki-explicit-kind",
                "title": "Wiki Explicit Kind",
                "body_md": "explicit kind body",
            }]
        })),
    )
    .await?;
    assert_eq!(cw2_status, StatusCode::OK);
    let wiki2_id = Uuid::parse_str(
        cw2_json["documents"][0]["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing wiki2 id"))?,
    )?;
    let (dwk_status, dwk_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "wiki-explicit-kind",
                "kind": "wiki",
                "deleted": true,
            }]
        })),
    )
    .await?;
    assert_eq!(dwk_status, StatusCode::OK);
    assert_eq!(dwk_json["deleted"], 1);
    assert_eq!(dwk_json["documents"][0]["kind"], "wiki");
    let wiki2_status: String = sqlx::query_scalar(
        "SELECT status FROM wiki.pages WHERE org_id = $1 AND id = $2",
    )
    .bind(seeded.org_id)
    .bind(wiki2_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(wiki2_status, "archived");

    let (bad_kind_status, _, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "bad-kind",
                "kind": "memory",
                "deleted": true,
            }]
        })),
    )
    .await?;
    assert_eq!(bad_kind_status, StatusCode::BAD_REQUEST);

    // 6. Same-batch [delete(slug), identical_upsert(slug)] leaves wiki.pages.status = 'active'
    //    with 0 delete_vector outbox rows created for that page
    let (bs1_status, bs1_json, bs1_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "batch-same-slug",
                "title": "Batch Same",
                "body_md": "# Batch Same\nidentical body content",
            }]
        })),
    )
    .await?;
    assert_eq!(bs1_status, StatusCode::OK, "create batch-same-slug failed: {bs1_txt}");
    assert_eq!(bs1_json["updated"], 1);
    let batch_same_id = Uuid::parse_str(
        bs1_json["documents"][0]["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing batch-same-slug id in {bs1_json}"))?,
    )?;

    let (bs2_status, bs2_json, bs2_txt) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [
                {
                    "slug": "batch-same-slug",
                    "deleted": true,
                },
                {
                    "slug": "batch-same-slug",
                    "title": "Batch Same",
                    "body_md": "# Batch Same\nidentical body content",
                }
            ]
        })),
    )
    .await?;
    assert_eq!(bs2_status, StatusCode::OK, "same-batch delete+recreate failed: {bs2_txt}");
    assert_eq!(bs2_json["skipped"], 1, "unexpected bs2_json: {bs2_json}");

    let batch_same_status: String = sqlx::query_scalar(
        "SELECT status FROM wiki.pages WHERE org_id = $1 AND id = $2",
    )
    .bind(seeded.org_id)
    .bind(batch_same_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(batch_same_status, "active");

    let batch_same_del_vec: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM indexing.outbox
         WHERE org_id = $1 AND target_table = 'wiki.pages' AND target_id = $2
           AND op = 'delete_vector'",
    )
    .bind(seeded.org_id)
    .bind(batch_same_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(
        batch_same_del_vec, 0,
        "expected 0 delete_vector outbox rows for same-batch delete + identical recreate"
    );

    Ok(())
}

async fn send_raw_bytes(
    app: &Router,
    method: Method,
    uri: &str,
    token: Option<&str>,
) -> anyhow::Result<(StatusCode, Option<String>, Vec<u8>)> {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(tok) = token {
        builder = builder.header(AUTHORIZATION, format!("Bearer {tok}"));
    }
    let resp = app.clone().oneshot(builder.body(Body::empty())?).await?;
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let bytes = resp.into_body().collect().await?.to_bytes().to_vec();
    Ok((status, content_type, bytes))
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c2_app_console_upload_chat_thumb_feedback_wiki_memory_and_connect(
) -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;
    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;
    let peer_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;

    // 1. /api/app/upload validation: missing/empty -> 400/422, invalid scope -> 400/422 without "arn"
    let boundary = "----OpenRagM4c2AppBoundary";
    let empty_req = Request::builder()
        .method(Method::POST)
        .uri("/api/app/upload")
        .header(AUTHORIZATION, format!("Bearer {owner_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(format!("--{boundary}--\r\n")))?;
    let empty_resp = app.clone().oneshot(empty_req).await?;
    assert!(
        empty_resp.status() == StatusCode::BAD_REQUEST
            || empty_resp.status() == StatusCode::UNPROCESSABLE_ENTITY
    );

    let bad_scope_body = build_multipart_body(
        boundary,
        &[("scope", "invalid_space_scope"), ("sensitivity", "normal")],
        Some(("notes.txt", "text/plain", b"hello")),
    );
    let bad_scope_req = Request::builder()
        .method(Method::POST)
        .uri("/api/app/upload")
        .header(AUTHORIZATION, format!("Bearer {owner_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(bad_scope_body))?;
    let bad_scope_resp = app.clone().oneshot(bad_scope_req).await?;
    assert!(
        bad_scope_resp.status() == StatusCode::BAD_REQUEST
            || bad_scope_resp.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
    let bad_scope_txt = String::from_utf8_lossy(
        &bad_scope_resp.into_body().collect().await?.to_bytes(),
    )
    .to_lowercase();
    assert!(
        !bad_scope_txt.contains("arn"),
        "app upload error leaked jargon 'arn': {bad_scope_txt}"
    );

    // 2. Owner creates a dept bucket and uploads a private HWPX with 한글(English) alias + #tag + PII
    let (bkt_st, bkt_json, bkt_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({
            "slug": format!("app-owner-priv-{}", Uuid::new_v4().simple()),
            "name": "Owner Private App Space",
            "access_scope": "private",
            "min_auth_level": 2,
        })),
    )
    .await?;
    assert_eq!(bkt_st, StatusCode::CREATED, "bucket create: {bkt_txt}");
    let priv_bid = bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing priv_bid"))?;

    let hwpx = build_hwpx_bytes(
        "M4c2 App Console Document",
        "세컨드 브레인(Second Brain) #m4c2tag owner contact secret.owner@corp.test",
    )?;
    let up_body = build_multipart_body(
        boundary,
        &[
            ("bucket", priv_bid),
            ("scope", "private"),
            ("sensitivity", "normal"),
            ("tags", "m4c2tag"),
        ],
        Some(("m4c2_note.hwpx", "application/hwp+zip", &hwpx)),
    );
    let up_req = Request::builder()
        .method(Method::POST)
        .uri("/api/app/upload")
        .header(AUTHORIZATION, format!("Bearer {owner_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(up_body))?;
    let up_resp = app.clone().oneshot(up_req).await?;
    let up_status = up_resp.status();
    let up_bytes = up_resp.into_body().collect().await?.to_bytes();
    let up_txt = String::from_utf8_lossy(&up_bytes).to_string();
    assert_eq!(up_status, StatusCode::CREATED, "upload failed: {up_txt}");
    let up_json: Value = serde_json::from_slice(&up_bytes)?;
    let doc_id = Uuid::parse_str(
        up_json["document_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing document_id in {up_json}"))?,
    )?;
    let page_id = Uuid::parse_str(
        up_json["page_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing page_id in {up_json}"))?,
    )?;

    // 3. Thumbnail retrieval: owner receives WebP/PNG bytes, peer receives 403/404 without image bytes, unknown -> 404
    let (th_st, th_ct, th_bytes) = send_raw_bytes(
        &app,
        Method::GET,
        &format!("/api/app/pages/{page_id}/thumbnail"),
        Some(&owner_key),
    )
    .await?;
    assert_eq!(th_st, StatusCode::OK, "owner thumbnail failed");
    let ct_str = th_ct.unwrap_or_default();
    assert!(
        ct_str.starts_with("image/webp") || ct_str.starts_with("image/png"),
        "unexpected thumbnail content-type: {ct_str}"
    );
    assert!(
        th_bytes.starts_with(b"RIFF") || th_bytes.starts_with(b"\x89PNG"),
        "expected WebP or PNG magic bytes"
    );

    let (peer_th_st, _, peer_th_bytes) = send_raw_bytes(
        &app,
        Method::GET,
        &format!("/api/app/pages/{page_id}/thumbnail"),
        Some(&peer_key),
    )
    .await?;
    assert!(
        peer_th_st == StatusCode::FORBIDDEN || peer_th_st == StatusCode::NOT_FOUND,
        "peer must not view owner's private thumbnail, got {peer_th_st}"
    );
    assert!(!peer_th_bytes.starts_with(b"RIFF") && !peer_th_bytes.starts_with(b"\x89PNG"));

    let (missing_th_st, _, _) = send_raw_bytes(
        &app,
        Method::GET,
        &format!("/api/app/pages/{}/thumbnail", Uuid::new_v4()),
        Some(&owner_key),
    )
    .await?;
    assert_eq!(missing_th_st, StatusCode::NOT_FOUND);

    // 4. Concept lookup reflects auto-extracted 한글(English) alias
    let (con_st, con_json, con_txt) = send_json(
        &app,
        Method::GET,
        "/api/app/concepts?query=Second%20Brain",
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(con_st, StatusCode::OK, "concepts failed: {con_txt}");
    assert!(
        serde_json::to_string(&con_json)?.contains("세컨드 브레인"),
        "expected extracted concept alias in {con_json}"
    );

    // 5. /api/app/chat returns answer + sources/citations with page_id, and /api/app/feedback persists & validates verdict
    let (chat_st, chat_json, chat_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/chat",
        Some(&owner_key),
        Some(json!({
            "message": "Second Brain #m4c2tag",
            "bucket": priv_bid,
            "top_k": 5,
        })),
    )
    .await?;
    assert_eq!(chat_st, StatusCode::OK, "app/chat failed: {chat_txt}");
    assert!(chat_json["answer"].is_string());
    assert!(chat_json["choices"].is_array());
    let citations = chat_json["citations"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing citations array in {chat_json}"))?;
    assert!(
        citations.iter().any(|c| c["page_id"].as_str() == Some(&page_id.to_string())),
        "expected citation page_id {page_id} in {chat_json}"
    );

    let (bad_fb_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/app/feedback",
        Some(&owner_key),
        Some(json!({
            "query": "Second Brain",
            "verdict": "nonsense_verdict",
        })),
    )
    .await?;
    assert!(
        bad_fb_st == StatusCode::BAD_REQUEST || bad_fb_st == StatusCode::UNPROCESSABLE_ENTITY
    );

    let (fb_st, fb_json, fb_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/feedback",
        Some(&owner_key),
        Some(json!({
            "query": "Second Brain #m4c2tag",
            "verdict": "up",
            "selected_doc_id": doc_id,
            "selected_page_id": page_id,
            "comment": "helpful citation",
        })),
    )
    .await?;
    assert!(
        fb_st == StatusCode::OK || fb_st == StatusCode::CREATED,
        "feedback failed: {fb_txt}"
    );
    let fb_id = Uuid::parse_str(
        fb_json["feedback_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing feedback_id in {fb_json}"))?,
    )?;

    // 6. Promote feedback -> golden set (idempotent on double promote!)
    let (gs_st, gs_json, gs_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/eval/golden-sets",
        Some(&admin_key),
        Some(json!({
            "name": format!("m4c2-gs-{}", Uuid::new_v4().simple()),
            "axis": "retrieval",
        })),
    )
    .await?;
    assert_eq!(gs_st, StatusCode::CREATED, "golden-set create: {gs_txt}");
    let gs_id = gs_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing gs_id"))?;

    let (pr1_st, _, pr1_txt) = send_json(
        &app,
        Method::POST,
        &format!("/api/studio/eval/feedback/{fb_id}/promote"),
        Some(&admin_key),
        Some(json!({
            "golden_set_id": gs_id,
            "axis": "retrieval",
            "expected_doc_ids": [doc_id],
        })),
    )
    .await?;
    assert!(
        pr1_st == StatusCode::OK || pr1_st == StatusCode::CREATED,
        "promote feedback 1 failed: {pr1_txt}"
    );
    let (pr2_st, _, pr2_txt) = send_json(
        &app,
        Method::POST,
        &format!("/api/studio/eval/feedback/{fb_id}/promote"),
        Some(&admin_key),
        Some(json!({
            "golden_set_id": gs_id,
            "axis": "retrieval",
            "expected_doc_ids": [doc_id],
        })),
    )
    .await?;
    assert!(
        pr2_st == StatusCode::OK || pr2_st == StatusCode::CONFLICT,
        "double promote should be idempotent 200 or 409: {pr2_txt}"
    );
    let gs_item_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM eval.golden_items WHERE golden_set_id = $1::uuid AND org_id = $2",
    )
    .bind(gs_id)
    .bind(seeded.org_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(gs_item_count, 1, "idempotent double promote must create exactly 1 item");

    // 7. /api/app/wiki + /api/app/wiki/{id}
    let (org_bkt_st, org_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("app-wiki-space-{}", Uuid::new_v4().simple()),
            "name": "App Wiki Bucket",
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(org_bkt_st, StatusCode::CREATED);
    let org_bid = org_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing org_bid"))?;

    let (ws_st, ws_json, _) = send_json(
        &app,
        Method::PUT,
        &format!("/v1/buckets/{org_bid}/documents/sync"),
        Some(&admin_key),
        Some(json!({
            "documents": [{
                "slug": "m4c2-app-wiki-page",
                "title": "M4c2 App Wiki Page",
                "body_md": "# App Wiki\nKnowledge summary #wiki-tag-m4c2",
                "tags": ["wiki-tag-m4c2"],
            }]
        })),
    )
    .await?;
    assert_eq!(ws_st, StatusCode::OK);
    let wiki_page_id = ws_json["documents"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing wiki_page_id"))?;

    let (wlist_st, wlist_json, _) =
        send_json(&app, Method::GET, "/api/app/wiki", Some(&owner_key), None).await?;
    assert_eq!(wlist_st, StatusCode::OK);
    assert!(serde_json::to_string(&wlist_json)?.contains("M4c2 App Wiki Page"));

    let (wget_st, wget_json, _) = send_json(
        &app,
        Method::GET,
        &format!("/api/app/wiki/{wiki_page_id}"),
        Some(&owner_key),
        None,
    )
    .await?;
    assert_eq!(wget_st, StatusCode::OK);
    assert_eq!(wget_json["title"], "M4c2 App Wiki Page");

    // 8. /api/app/connect contains 4 clients + stdio & streamable http snippets
    let (conn_st, conn_json, conn_txt) =
        send_json(&app, Method::GET, "/api/app/connect", Some(&owner_key), None).await?;
    assert_eq!(conn_st, StatusCode::OK, "connect failed: {conn_txt}");
    let conn_lower = conn_txt.to_lowercase();
    for needle in ["claude desktop", "claude code", "cursor", "codex", "stdio", "/mcp"] {
        assert!(
            conn_lower.contains(needle),
            "missing '{needle}' in /api/app/connect: {conn_json}"
        );
    }

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c2_studio_admin_gate_pipeline_index_iam_sim_pii_guard_and_cascade(
) -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;
    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;

    // 1. Non-admin member gets 403 on mutating /api/studio/* routes
    for (method, uri, body) in [
        (
            Method::POST,
            "/api/studio/iam/groups".to_string(),
            json!({ "name": "forbidden-grp" }),
        ),
        (
            Method::PUT,
            "/api/studio/index/engines".to_string(),
            json!({ "table": "corpus.chunks", "engine": "native" }),
        ),
        (
            Method::PUT,
            "/api/studio/pipeline".to_string(),
            json!({ "pipeline_yaml": "memory:\n  backend: kumiho\n" }),
        ),
        (
            Method::POST,
            "/api/studio/pii/redactions".to_string(),
            json!({
                "target_scope": "document",
                "target_ref": { "document_id": Uuid::new_v4() },
                "redaction_mode": "hard_purge",
                "reason": "unauthorized",
            }),
        ),
        (
            Method::POST,
            "/api/studio/eval/promote".to_string(),
            json!({ "pipeline_config_id": Uuid::new_v4() }),
        ),
    ] {
        let (st, _, txt) = send_json(
            &app,
            method.clone(),
            &uri,
            Some(&owner_key),
            Some(body),
        )
        .await?;
        assert_eq!(
            st,
            StatusCode::FORBIDDEN,
            "expected 403 for non-admin on {method} {uri}, got {st}: {txt}"
        );
    }

    // 2. Pipeline GET/PUT validation & runtime memory backend toggle + /pipeline/test
    let (pg_st, pg_json, _) =
        send_json(&app, Method::GET, "/api/studio/pipeline", Some(&admin_key), None).await?;
    assert_eq!(pg_st, StatusCode::OK);
    let orig_yaml = pg_json["pipeline_yaml"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing pipeline_yaml"))?
        .to_string();

    let (bad_pipe_st, _, _) = send_json(
        &app,
        Method::PUT,
        "/api/studio/pipeline",
        Some(&admin_key),
        Some(json!({ "pipeline_yaml": "memory:\n  backend: invalid_backend_xyz\n" })),
    )
    .await?;
    assert!(
        bad_pipe_st == StatusCode::BAD_REQUEST || bad_pipe_st == StatusCode::UNPROCESSABLE_ENTITY
    );

    let (pt_st, pt_json, pt_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/pipeline/test",
        Some(&admin_key),
        Some(json!({
            "pipeline_yaml": orig_yaml,
            "query": "test scoring query",
            "auth_level": 7,
        })),
    )
    .await?;
    assert_eq!(pt_st, StatusCode::OK, "pipeline/test failed: {pt_txt}");
    assert_eq!(pt_json["valid"], true);
    assert!(pt_json["vector"].is_number());
    assert!(pt_json["bm25"].is_number());
    assert!(pt_json["score"].is_number());
    assert_eq!(pt_json["ocr_mode"], "private");

    let (pt8_st, pt8_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/pipeline/test",
        Some(&admin_key),
        Some(json!({ "pipeline_yaml": orig_yaml, "auth_level": 8 })),
    )
    .await?;
    assert_eq!(pt8_st, StatusCode::OK);
    assert_eq!(pt8_json["ocr_mode"], "zdr");

    let (upd_st, upd_json, upd_txt) = send_json(
        &app,
        Method::PUT,
        "/api/studio/pipeline",
        Some(&admin_key),
        Some(json!({ "pipeline_yaml": "memory:\n  backend: openrag_pg\n" })),
    )
    .await?;
    assert_eq!(upd_st, StatusCode::OK, "pipeline update failed: {upd_txt}");
    assert_eq!(upd_json["memory"]["backend"], "openrag_pg");
    assert_eq!(
        state.active_memory_router().await?.primary_kind(),
        openrag_core::MemoryBackendKind::OpenragPg
    );
    // Restore original pipeline config
    let _ = send_json(
        &app,
        Method::PUT,
        "/api/studio/pipeline",
        Some(&admin_key),
        Some(json!({ "pipeline_yaml": orig_yaml })),
    )
    .await?;

    // 3. Index engines, outbox, and quarantine retry
    let (eng_st, eng_json, _) = send_json(
        &app,
        Method::GET,
        "/api/studio/index/engines",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(eng_st, StatusCode::OK);
    assert!(eng_json["engines"].is_array());

    let (eng_put_st, _, _) = send_json(
        &app,
        Method::PUT,
        "/api/studio/index/engines",
        Some(&admin_key),
        Some(json!({ "table": "corpus.chunks", "engine": "pgvector" })),
    )
    .await?;
    assert_eq!(eng_put_st, StatusCode::OK);
    let _ = send_json(
        &app,
        Method::PUT,
        "/api/studio/index/engines",
        Some(&admin_key),
        Some(json!({ "table": "corpus.chunks", "engine": "native" })),
    )
    .await?;

    let (ob_st, ob_json, _) = send_json(
        &app,
        Method::GET,
        "/api/studio/index/outbox",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(ob_st, StatusCode::OK);
    assert!(ob_json["outbox"].is_array() && ob_json["quarantine"].is_array());

    let (qr_miss_st, _, _) = send_json(
        &app,
        Method::POST,
        &format!("/api/studio/index/quarantine/{}/retry", Uuid::new_v4()),
        Some(&admin_key),
        Some(json!({})),
    )
    .await?;
    assert_eq!(qr_miss_st, StatusCode::NOT_FOUND);

    // 4. IAM studio: groups (omitted org_id/slug + inline member_ids), api-keys (token + api_key), simulate DB hydration
    let (grp_st, grp_json, grp_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/groups",
        Some(&admin_key),
        Some(json!({
            "name": format!("m4c2-grp-{}", Uuid::new_v4().simple()),
            "member_ids": [seeded.owner_id],
        })),
    )
    .await?;
    assert_eq!(grp_st, StatusCode::CREATED, "group create failed: {grp_txt}");
    let grp_id = Uuid::parse_str(
        grp_json["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing grp_id"))?,
    )?;

    let (bkt_st, bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("iam-sim-grp-{}", Uuid::new_v4().simple()),
            "name": "IAM Sim Group Bucket",
            "access_scope": "group",
            "allowed_group_ids": [grp_id],
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(bkt_st, StatusCode::CREATED);
    let grp_bid = bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing grp_bid"))?;

    let (sim_allow_st, sim_allow_json, sim_allow_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/simulate",
        Some(&admin_key),
        Some(json!({
            "principal_id": seeded.owner_id,
            "resource_type": "bucket",
            "resource_id": grp_bid,
            "action": "read",
        })),
    )
    .await?;
    assert_eq!(sim_allow_st, StatusCode::OK, "sim allow: {sim_allow_txt}");
    assert_eq!(sim_allow_json["allowed"], true, "owner in grp must pass sim");

    let (sim_deny_st, sim_deny_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/simulate",
        Some(&admin_key),
        Some(json!({
            "principal_id": seeded.peer_id,
            "resource_type": "bucket",
            "resource_id": grp_bid,
            "action": "read",
        })),
    )
    .await?;
    assert_eq!(sim_deny_st, StatusCode::OK);
    assert_eq!(sim_deny_json["allowed"], false, "peer not in grp must be denied");

    let (ak_st, ak_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/api-keys",
        Some(&admin_key),
        Some(json!({ "principal_id": seeded.owner_id })),
    )
    .await?;
    assert_eq!(ak_st, StatusCode::CREATED);
    assert!(ak_json["token"].is_string() && ak_json["api_key"].is_string());

    // 5. PII guard rules (`kind`/`rule_kind`, invalid regex -> 400), guard-policy, guard-test, Safe-Word 403 + zero leak in logs/events
    let (bad_rx_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/pii/guard-rules",
        Some(&admin_key),
        Some(json!({
            "kind": "regex",
            "pattern_or_category": "(unclosed[regex",
            "action": "block",
            "surfaces": ["inbound_query"],
        })),
    )
    .await?;
    assert!(
        bad_rx_st == StatusCode::BAD_REQUEST || bad_rx_st == StatusCode::UNPROCESSABLE_ENTITY
    );

    let (gp_st, _, gp_txt) = send_json(
        &app,
        Method::PUT,
        "/api/studio/pii/guard-policy",
        Some(&admin_key),
        Some(json!({
            "master_enabled": true,
            "default_action": "mask",
            "allowlist": ["allowed-safe-term"],
        })),
    )
    .await?;
    assert_eq!(gp_st, StatusCode::OK, "guard-policy put failed: {gp_txt}");

    let safeword = format!("FORBIDTOKEN_{}", Uuid::new_v4().simple());
    let (gr_st, _, gr_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/pii/guard-rules",
        Some(&admin_key),
        Some(json!({
            "kind": "safe_word",
            "pattern_or_category": safeword,
            "action": "block",
            "surfaces": ["inbound_query", "llm_egress", "ingestion_gate"],
        })),
    )
    .await?;
    assert_eq!(gr_st, StatusCode::CREATED, "guard-rule create failed: {gr_txt}");

    let (gt_st, gt_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/pii/guard-test",
        Some(&admin_key),
        Some(json!({
            "text": format!("testing {safeword} and 900101-1234567"),
            "surface": "inbound_query",
            "guard_enabled": true,
        })),
    )
    .await?;
    assert_eq!(gt_st, StatusCode::OK);
    assert_eq!(gt_json["decision"], "block");

    let (blk_st, blk_json, blk_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/chat",
        Some(&owner_key),
        Some(json!({
            "message": format!("please answer with {safeword} right now"),
            "bucket": grp_bid,
        })),
    )
    .await?;
    assert_eq!(blk_st, StatusCode::FORBIDDEN, "expected 403 guard_blocked: {blk_txt}");
    assert_eq!(blk_json["code"], "guard_blocked");

    let (gev_st, _, gev_txt) = send_json(
        &app,
        Method::GET,
        "/api/studio/pii/guard-events",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(gev_st, StatusCode::OK);
    assert!(
        !gev_txt.contains(&safeword),
        "guard-events leaked raw blocked safeword: {gev_txt}"
    );

    let (glogs_st, _, glogs_txt) = send_json(
        &app,
        Method::GET,
        "/api/studio/gateway/logs",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(glogs_st, StatusCode::OK);
    assert!(!glogs_txt.contains("900101-1234567"));

    // 6. CascadeRedactor hard_purge + redact_thumbnails reflects in GET /api/app/pages/{page_id}/thumbnail
    let boundary = "----OpenRagM4c2CascadeBoundary";
    let cascade_secret = "900101-1234567";
    let up_body = build_multipart_body(
        boundary,
        &[("bucket", grp_bid), ("scope", "group")],
        Some((
            "cascade.txt",
            "text/plain",
            format!("Employee RRN {cascade_secret} in group file").as_bytes(),
        )),
    );
    let up_req = Request::builder()
        .method(Method::POST)
        .uri("/api/app/upload")
        .header(AUTHORIZATION, format!("Bearer {admin_key}"))
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(up_body))?;
    let up_resp = app.clone().oneshot(up_req).await?;
    assert_eq!(up_resp.status(), StatusCode::CREATED);
    let up_json: Value = serde_json::from_slice(&up_resp.into_body().collect().await?.to_bytes())?;
    let red_doc_id = up_json["document_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing red_doc_id"))?;
    let red_page_id = up_json["page_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing red_page_id"))?;

    let (pre_st, _, orig_thumb_bytes) = send_raw_bytes(
        &app,
        Method::GET,
        &format!("/api/app/pages/{red_page_id}/thumbnail"),
        Some(&owner_key),
    )
    .await?;
    assert_eq!(pre_st, StatusCode::OK, "group member must view orig thumbnail");

    let (red_st, red_json, red_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/pii/redactions",
        Some(&admin_key),
        Some(json!({
            "target_scope": "document",
            "target_ref": { "document_id": red_doc_id },
            "redaction_mode": "hard_purge",
            "redact_thumbnails": true,
            "reason": "e2e cascade purge test",
        })),
    )
    .await?;
    assert_eq!(red_st, StatusCode::CREATED, "cascade redaction failed: {red_txt}");
    assert!(red_json["job_id"].is_string());

    let (post_st, _, post_thumb_bytes) = send_raw_bytes(
        &app,
        Method::GET,
        &format!("/api/app/pages/{red_page_id}/thumbnail"),
        Some(&owner_key),
    )
    .await?;
    assert_eq!(post_st, StatusCode::OK);
    assert_ne!(
        orig_thumb_bytes, post_thumb_bytes,
        "thumbnail endpoint must serve newly redacted bytes after CascadeRedactor"
    );

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c2_eval_wiki_conflicts_plugins_knowledge_tags_and_memory_security(
) -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;
    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;
    let peer_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.peer_id, None, vec![]).await?;

    // 1. Memory patch/retract/promote + PII token laundering prevention + RLS isolation
    let mut ctx_owner =
        PrincipalContext::new(seeded.org_id, seeded.owner_id, AuthLevel::new(2)?);
    ctx_owner.dept_ids = vec![seeded.dept_id];
    let mrouter = state.active_memory_router().await?;
    let entry = mrouter
        .upsert_entry(
            &ctx_owner,
            openrag_memory::UpsertMemoryInput {
                bucket_id: None,
                dept_id: seeded.dept_id,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(2)?,
                org_slug: seeded.slug.clone(),
                dept_path: "eng".into(),
                memory_kind: MemoryKind::Fact,
                title: Some("Initial Owner Fact".into()),
                text: "Original owner memory fact.".into(),
                tags: vec!["mem-tag-m4c2".into()],
                source_episode_id: None,
                source_document_id: None,
                confidence: Some(0.9),
            },
            None,
        )
        .await?;

    let (mlist_st, mlist_json, _) =
        send_json(&app, Method::GET, "/api/app/memory", Some(&owner_key), None).await?;
    assert_eq!(mlist_st, StatusCode::OK);
    assert!(serde_json::to_string(&mlist_json)?.contains(&entry.id.to_string()));

    // Peer cannot patch owner's private memory
    let (peer_patch_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/patch",
        Some(&peer_key),
        Some(json!({
            "entry_id": entry.id,
            "op": "amend",
            "new_body_text": "hijacked",
        })),
    )
    .await?;
    assert!(
        peer_patch_st == StatusCode::FORBIDDEN || peer_patch_st == StatusCode::NOT_FOUND
    );

    // Owner patching literal [PII:EMAIL:deadbeef] neutralizes token
    let (own_patch_st, own_patch_json, own_patch_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/patch",
        Some(&owner_key),
        Some(json!({
            "entry_id": entry.id,
            "op": "amend",
            "new_body_text": "Amended with forged [PII:EMAIL:deadbeef] token",
            "reason": "clarify",
        })),
    )
    .await?;
    assert_eq!(own_patch_st, StatusCode::OK, "patch failed: {own_patch_txt}");
    let patched_body = own_patch_json["effective_text"].as_str().unwrap_or("");
    assert!(!patched_body.contains("[PII:EMAIL:deadbeef]"));

    // Promote to wiki creates wiki.pages + wiki.page_sources
    let (org_bkt_st, org_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("mem-promote-wiki-{}", Uuid::new_v4().simple()),
            "name": "Memory Promote Target",
            "access_scope": "org",
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(org_bkt_st, StatusCode::CREATED);
    let org_bid = org_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing org_bid"))?;

    // Non-admin member cannot promote into admin-owned org bucket -> 403
    let (denied_pro_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({
            "entry_id": entry.id,
            "to": "wiki",
            "bucket": org_bid,
            "title": "Unauthorized Org Promote",
        })),
    )
    .await?;
    assert_eq!(denied_pro_st, StatusCode::FORBIDDEN);

    // Owner promotes into owner's own dept bucket -> succeeds & creates wiki.pages
    let (own_bkt_st, own_bkt_json, _) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({
            "slug": format!("own-promote-wiki-{}", Uuid::new_v4().simple()),
            "name": "Owner Memory Promote Target",
            "access_scope": "dept",
            "min_auth_level": 2,
        })),
    )
    .await?;
    assert_eq!(own_bkt_st, StatusCode::CREATED);
    let own_bid = own_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing own_bid"))?;

    let (mpro_st, mpro_json, mpro_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({
            "entry_id": entry.id,
            "to": "wiki",
            "bucket": own_bid,
            "title": "Promoted Memory Page",
        })),
    )
    .await?;
    assert!(
        mpro_st == StatusCode::OK || mpro_st == StatusCode::CREATED,
        "memory promote failed: {mpro_txt}"
    );
    assert!(mpro_json["page_id"].is_string());

    // Retract + second promote blocked with 409/400/422
    let (mret_st, _, mret_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/retract",
        Some(&owner_key),
        Some(json!({ "entry_id": entry.id, "reason": "no longer valid" })),
    )
    .await?;
    assert_eq!(mret_st, StatusCode::OK, "retract failed: {mret_txt}");

    let (mpro2_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({ "entry_id": entry.id, "to": "wiki", "bucket": org_bid })),
    )
    .await?;
    assert!(
        mpro2_st == StatusCode::CONFLICT
            || mpro2_st == StatusCode::BAD_REQUEST
            || mpro2_st == StatusCode::UNPROCESSABLE_ENTITY
    );

    // 2. Eval golden set -> bench -> optimize -> leaderboard -> report -> unsafe promote gate
    let (gs_st, gs_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/eval/golden-sets",
        Some(&admin_key),
        Some(json!({
            "name": format!("m4c2-bench-gs-{}", Uuid::new_v4().simple()),
            "axis": "retrieval",
        })),
    )
    .await?;
    assert_eq!(gs_st, StatusCode::CREATED);
    let gs_id = gs_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing gs_id"))?;

    let (gi_st, gi_json, gi_txt) = send_json(
        &app,
        Method::POST,
        &format!("/api/studio/eval/golden-sets/{gs_id}/items"),
        Some(&admin_key),
        Some(json!({
            "axis": "retrieval",
            "query_text": "Promoted Memory Page",
            "test_principal_id": seeded.owner_id,
            "expected_doc_ids": [Uuid::new_v4()],
        })),
    )
    .await?;
    assert_eq!(gi_st, StatusCode::CREATED, "add item failed: {gi_txt}");
    let item_id = gi_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing item_id"))?;

    let (patch_gi_st, _, _) = send_json(
        &app,
        Method::PATCH,
        &format!("/api/studio/eval/golden-items/{item_id}"),
        Some(&admin_key),
        Some(json!({ "review_status": "approved" })),
    )
    .await?;
    assert_eq!(patch_gi_st, StatusCode::OK);

    let (bench_st, _, bench_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/eval/bench",
        Some(&admin_key),
        Some(json!({ "golden_set_id": gs_id })),
    )
    .await?;
    assert_eq!(bench_st, StatusCode::OK, "bench failed: {bench_txt}");

    let (opt_st, opt_json, opt_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/eval/optimize",
        Some(&admin_key),
        Some(json!({ "golden_set_id": gs_id, "strategy": "greedy_node" })),
    )
    .await?;
    assert_eq!(opt_st, StatusCode::OK, "optimize failed: {opt_txt}");
    assert!(opt_json["best"].is_object());

    let (lb_st, lb_json, _) = send_json(
        &app,
        Method::GET,
        "/api/studio/eval/leaderboard",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(lb_st, StatusCode::OK);
    assert!(lb_json["security_gate"].is_boolean());

    let (rep_st, _, rep_txt) = send_json(
        &app,
        Method::GET,
        &format!("/api/studio/eval/report?golden_set_id={gs_id}"),
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(rep_st, StatusCode::OK);
    assert!(rep_txt.len() > 10);

    // Unsafe pipeline config blocked by security gate
    let unsafe_cfg_id = Uuid::new_v4();
    let unsafe_yaml = "version: 1\nprofile: standard\npii:\n  capsule_enabled: false\n";
    sqlx::query(
        "INSERT INTO eval.pipeline_configs (
            id, org_id, name, profile, pipeline_yaml, config_json, is_promoted, created_by
         ) VALUES ($1, $2, $3, 'standard', $4, $5::jsonb, false, $6)",
    )
    .bind(unsafe_cfg_id)
    .bind(seeded.org_id)
    .bind(format!("unsafe-cfg-{unsafe_cfg_id}"))
    .bind(unsafe_yaml)
    .bind(json!({"pii": {"capsule_enabled": false}}))
    .bind(seeded.admin_id)
    .execute(worker_pool.inner())
    .await?;

    let (un_prom_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/eval/promote",
        Some(&admin_key),
        Some(json!({ "pipeline_config_id": unsafe_cfg_id })),
    )
    .await?;
    assert!(
        un_prom_st == StatusCode::BAD_REQUEST
            || un_prom_st == StatusCode::FORBIDDEN
            || un_prom_st == StatusCode::CONFLICT
            || un_prom_st == StatusCode::UNPROCESSABLE_ENTITY
    );
    let still_promoted: bool = sqlx::query_scalar(
        "SELECT is_promoted FROM eval.pipeline_configs WHERE id = $1",
    )
    .bind(unsafe_cfg_id)
    .fetch_one(worker_pool.inner())
    .await?;
    assert!(!still_promoted);

    // 3. Wiki conflicts, plugins connectors, concepts, knowledge tags
    let (wc_st, wc_json, _) = send_json(
        &app,
        Method::GET,
        "/api/studio/wiki/conflicts",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(wc_st, StatusCode::OK);
    assert!(wc_json["conflicts"].is_array());

    let (pc_st, pc_json, pc_txt) = send_json(
        &app,
        Method::GET,
        "/api/studio/plugins/connectors",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(pc_st, StatusCode::OK, "connectors failed: {pc_txt}");
    assert!(pc_txt.contains("kordoc") || pc_json["connectors"].is_array());

    let (kt_st, kt_json, kt_txt) = send_json(
        &app,
        Method::GET,
        "/api/studio/knowledge/tags",
        Some(&admin_key),
        None,
    )
    .await?;
    assert_eq!(kt_st, StatusCode::OK, "knowledge tags failed: {kt_txt}");
    assert!(kt_json["tags"].is_array());

    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c2_remediation_regressions() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("WORKER_DATABASE_URL / DATABASE_URL required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let seeded = seed_org(&worker_pool).await?;
    let admin_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.admin_id, None, vec![]).await?;
    let owner_key = mint_key(&app, &worker_pool, seeded.org_id, seeded.owner_id, None, vec![]).await?;

    // 1. Cross-org IAM principal & API key creation must return 403
    let cross_email = format!("evil-{}@other.local", Uuid::new_v4().simple());
    let (cp_st, _, cp_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/principals",
        Some(&admin_key),
        Some(json!({
            "org_id": seeded.other_org_id,
            "email": cross_email,
            "display_name": "Evil Cross Org",
            "auth_level": 10,
            "is_org_admin": true,
            "can_unmask_pii": true,
        })),
    )
    .await?;
    assert_eq!(
        cp_st,
        StatusCode::FORBIDDEN,
        "cross-org principal creation must fail 403: {cp_txt}"
    );
    let evil_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM iam.principals WHERE org_id = $1 AND email = $2",
    )
    .bind(seeded.other_org_id)
    .bind(&cross_email)
    .fetch_one(worker_pool.inner())
    .await?;
    assert_eq!(evil_count, 0);

    let (ck_cross_org_st, _, ck_cross_org_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/api-keys",
        Some(&admin_key),
        Some(json!({
            "org_id": seeded.other_org_id,
            "principal_id": seeded.other_admin_id,
            "name": "stolen-other-org",
        })),
    )
    .await?;
    assert_eq!(
        ck_cross_org_st,
        StatusCode::FORBIDDEN,
        "cross-org api-key creation must fail 403: {ck_cross_org_txt}"
    );

    let (ck_cross_prin_st, _, ck_cross_prin_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/api-keys",
        Some(&admin_key),
        Some(json!({
            "org_id": seeded.org_id,
            "principal_id": seeded.other_admin_id,
            "name": "stolen-cross-principal",
        })),
    )
    .await?;
    assert_eq!(
        ck_cross_prin_st,
        StatusCode::FORBIDDEN,
        "minting api-key for foreign principal_id must fail 403: {ck_cross_prin_txt}"
    );

    // 2. Full pipeline YAML with memory: + invalid fields must return 400/422 and never clobber active config
    let (init_put_st, _, init_put_txt) = send_json(
        &app,
        Method::PUT,
        "/api/studio/pipeline",
        Some(&admin_key),
        Some(json!({ "pipeline_yaml": "version: 1\nprofile: standard\nmemory:\n  backend: kumiho\n" })),
    )
    .await?;
    assert_eq!(init_put_st, StatusCode::OK, "setup pipeline failed: {init_put_txt}");

    let (before_st, before_json, _) =
        send_json(&app, Method::GET, "/api/studio/pipeline", Some(&admin_key), None).await?;
    assert_eq!(before_st, StatusCode::OK);
    let before_yaml = before_json["pipeline_yaml"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing before pipeline_yaml"))?
        .to_string();
    assert_eq!(before_json["memory"]["backend"], "kumiho");

    for invalid_yaml in [
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nstore:\n  dim: 128\n",
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nretrieve:\n  alpha: 1.5\n",
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nretrieve:\n  alpha: -0.1\n",
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nllm:\n  local_only_min_auth_level: 0\n",
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\nllm:\n  local_only_min_auth_level: 11\n",
        "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\npii:\n  guard:\n    default_action: nuke\n",
    ] {
        let (put_st, _, put_txt) = send_json(
            &app,
            Method::PUT,
            "/api/studio/pipeline",
            Some(&admin_key),
            Some(json!({ "pipeline_yaml": invalid_yaml })),
        )
        .await?;
        assert!(
            put_st == StatusCode::BAD_REQUEST || put_st == StatusCode::UNPROCESSABLE_ENTITY,
            "PUT /api/studio/pipeline must reject invalid YAML, got {put_st}: {put_txt}"
        );

        let (test_st, _, test_txt) = send_json(
            &app,
            Method::POST,
            "/api/studio/pipeline/test",
            Some(&admin_key),
            Some(json!({ "pipeline_yaml": invalid_yaml, "query": "test" })),
        )
        .await?;
        assert!(
            test_st == StatusCode::BAD_REQUEST || test_st == StatusCode::UNPROCESSABLE_ENTITY,
            "POST /api/studio/pipeline/test must reject invalid YAML, got {test_st}: {test_txt}"
        );
    }

    let (after_st, after_json, _) =
        send_json(&app, Method::GET, "/api/studio/pipeline", Some(&admin_key), None).await?;
    assert_eq!(after_st, StatusCode::OK);
    assert_eq!(
        after_json["pipeline_yaml"].as_str(),
        Some(before_yaml.as_str()),
        "failed PUT /api/studio/pipeline must not clobber active pipeline"
    );
    assert_eq!(after_json["memory"]["backend"], "kumiho");

    let (restore_st, _, _) = send_json(
        &app,
        Method::PUT,
        "/api/studio/pipeline",
        Some(&admin_key),
        Some(json!({ "pipeline_yaml": "version: 1\nprofile: standard\nmemory:\n  backend: openrag_pg\n" })),
    )
    .await?;
    assert_eq!(restore_st, StatusCode::OK);

    // 3. POST /api/app/memory/promote returns 403 when target bucket denies caller via denied_group_ids or min_auth_level
    let mut ctx_owner =
        PrincipalContext::new(seeded.org_id, seeded.owner_id, AuthLevel::new(2)?);
    ctx_owner.dept_ids = vec![seeded.dept_id];
    let mrouter = state.active_memory_router().await?;
    let mem_for_bucket_test = mrouter
        .upsert_entry(
            &ctx_owner,
            openrag_memory::UpsertMemoryInput {
                bucket_id: None,
                dept_id: seeded.dept_id,
                access_scope: AccessScope::Private,
                allowed_group_ids: vec![],
                denied_group_ids: vec![],
                min_auth_level: AuthLevel::new(2)?,
                org_slug: seeded.slug.clone(),
                dept_path: "dept1".into(),
                memory_kind: MemoryKind::Fact,
                title: Some("Bucket Promote Auth Test".into()),
                text: "Memory text for bucket access check".into(),
                tags: vec![],
                source_episode_id: None,
                source_document_id: None,
                confidence: Some(0.9),
            },
            None,
        )
        .await?;

    let (grp_st, grp_json, grp_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/groups",
        Some(&admin_key),
        Some(json!({
            "name": format!("deny-promote-grp-{}", Uuid::new_v4().simple()),
            "member_ids": [seeded.owner_id],
        })),
    )
    .await?;
    assert_eq!(grp_st, StatusCode::CREATED, "group create failed: {grp_txt}");
    let deny_grp_id = Uuid::parse_str(
        grp_json["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing group id"))?,
    )?;

    let (b_deny_st, b_deny_json, b_deny_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("deny-grp-bkt-{}", Uuid::new_v4().simple()),
            "name": "Denied Group Bucket",
            "access_scope": "dept",
            "dept_id": seeded.dept_id,
            "denied_group_ids": [deny_grp_id],
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(b_deny_st, StatusCode::CREATED, "create deny bucket failed: {b_deny_txt}");
    let deny_bkt_id = b_deny_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing deny_bkt_id"))?;

    let (pro_deny_grp_st, _, pro_deny_grp_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({
            "memory_id": mem_for_bucket_test.id,
            "to": "wiki",
            "bucket": deny_bkt_id,
        })),
    )
    .await?;
    assert_eq!(
        pro_deny_grp_st,
        StatusCode::FORBIDDEN,
        "promote to denied_group_ids bucket must return 403, got {pro_deny_grp_st}: {pro_deny_grp_txt}"
    );

    let (b_lvl_st, b_lvl_json, b_lvl_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&admin_key),
        Some(json!({
            "slug": format!("high-lvl-bkt-{}", Uuid::new_v4().simple()),
            "name": "High Level Bucket",
            "access_scope": "dept",
            "dept_id": seeded.dept_id,
            "min_auth_level": 9,
        })),
    )
    .await?;
    assert_eq!(b_lvl_st, StatusCode::CREATED, "create high level bucket failed: {b_lvl_txt}");
    let high_lvl_bkt_id = b_lvl_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing high_lvl_bkt_id"))?;

    let (pro_low_lvl_st, _, pro_low_lvl_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({
            "memory_id": mem_for_bucket_test.id,
            "to": "wiki",
            "bucket": high_lvl_bkt_id,
        })),
    )
    .await?;
    assert_eq!(
        pro_low_lvl_st,
        StatusCode::FORBIDDEN,
        "promote to high min_auth_level bucket must return 403, got {pro_low_lvl_st}: {pro_low_lvl_txt}"
    );

    let (pro_miss_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({
            "memory_id": mem_for_bucket_test.id,
            "to": "wiki",
            "bucket": Uuid::new_v4(),
        })),
    )
    .await?;
    assert_eq!(pro_miss_st, StatusCode::NOT_FOUND);

    // 4 & 5. Pre-seeded memory with [PII:EMAIL:...] & lowercase [PII:rrn_kr:...] promoted to wiki
    //        neutralizes title fallback, wiki.pages, wiki.revisions, and indexing.outbox
    let (own_bkt_st, own_bkt_json, own_bkt_txt) = send_json(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({
            "slug": format!("owner-wiki-bkt-{}", Uuid::new_v4().simple()),
            "name": "Owner Wiki Bucket",
            "access_scope": "dept",
            "dept_id": seeded.dept_id,
            "min_auth_level": 1,
        })),
    )
    .await?;
    assert_eq!(own_bkt_st, StatusCode::CREATED, "owner bucket failed: {own_bkt_txt}");
    let own_bkt_id = own_bkt_json["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing own_bkt_id"))?;

    let preseed_id = Uuid::new_v4();
    let preseed_title = "Preseed Title [PII:EMAIL:aa11bb22] [PII:rrn_kr:1234abcd]";
    let preseed_body = "Preseed Body [PII:EMAIL:bfd2b25c] lower_rrn=[PII:rrn_kr:1234abcd] lower_phone=[PII:phone_kr:abcd1234]";
    let preseed_sha = sha256_hex_str(preseed_body);
    let preseed_arn = format!(
        "arn:openrag:{}:dept1:private:memory:L2/{}",
        seeded.slug, preseed_id
    );
    sqlx::query(
        "INSERT INTO memory.entries (
            id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level,
            arn, acl_scope_key, memory_kind, title, effective_text, bm25_text, content_sha256, status
         ) VALUES ($1, $2, $3, $4, 'private', 2, $5, 'dept1:private:L2', 'fact', $6, $7, $7, $8, 'active')",
    )
    .bind(preseed_id)
    .bind(seeded.org_id)
    .bind(seeded.dept_id)
    .bind(seeded.owner_id)
    .bind(&preseed_arn)
    .bind(preseed_title)
    .bind(preseed_body)
    .bind(&preseed_sha)
    .execute(worker_pool.inner())
    .await?;

    let (pro_pii_st, pro_pii_json, pro_pii_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/memory/promote",
        Some(&owner_key),
        Some(json!({
            "memory_id": preseed_id,
            "to": "wiki",
            "bucket": own_bkt_id,
        })),
    )
    .await?;
    assert_eq!(pro_pii_st, StatusCode::OK, "promote preseeded failed: {pro_pii_txt}");
    let wiki_page_id = Uuid::parse_str(
        pro_pii_json["promoted_wiki_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing promoted_wiki_id"))?,
    )?;

    let wiki_row = sqlx::query(
        "SELECT title, body_md, bm25_text, body_sha256 FROM wiki.pages WHERE org_id = $1 AND id = $2",
    )
    .bind(seeded.org_id)
    .bind(wiki_page_id)
    .fetch_one(worker_pool.inner())
    .await?;
    let w_title: String = sqlx::Row::try_get(&wiki_row, "title")?;
    let w_body: String = sqlx::Row::try_get(&wiki_row, "body_md")?;
    let w_bm25: String = sqlx::Row::try_get(&wiki_row, "bm25_text")?;
    let w_sha: String = sqlx::Row::try_get(&wiki_row, "body_sha256")?;
    assert!(!w_title.contains("[PII:"), "wiki.pages.title retained raw token: {w_title}");
    assert!(w_title.contains("[PII-LITERAL:EMAIL:aa11bb22]"));
    assert!(w_title.contains("[PII-LITERAL:rrn_kr:1234abcd]"));
    assert!(!w_body.contains("[PII:"), "wiki.pages.body_md retained raw token: {w_body}");
    assert!(!w_bm25.contains("[PII:"), "wiki.pages.bm25_text retained raw token: {w_bm25}");
    assert!(w_body.contains("[PII-LITERAL:EMAIL:bfd2b25c]"));
    assert!(w_body.contains("[PII-LITERAL:rrn_kr:1234abcd]"));
    assert!(w_body.contains("[PII-LITERAL:phone_kr:abcd1234]"));

    let rev_rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT body_md, body_sha256 FROM wiki.revisions WHERE org_id = $1 AND page_id = $2",
    )
    .bind(seeded.org_id)
    .bind(wiki_page_id)
    .fetch_all(worker_pool.inner())
    .await?;
    assert!(!rev_rows.is_empty());
    for (rev_body, rev_sha) in &rev_rows {
        assert!(!rev_body.contains("[PII:"), "wiki.revisions.body_md retained raw token: {rev_body}");
        assert_eq!(rev_body, &w_body);
        assert_eq!(rev_sha, &w_sha);
    }

    let outbox_rows: Vec<(Option<String>, Value)> = sqlx::query_as(
        "SELECT content_sha256, payload FROM indexing.outbox
         WHERE org_id = $1 AND target_table = 'wiki.pages' AND target_id = $2",
    )
    .bind(seeded.org_id)
    .bind(wiki_page_id)
    .fetch_all(worker_pool.inner())
    .await?;
    assert!(!outbox_rows.is_empty());
    for (ob_sha, ob_payload) in &outbox_rows {
        assert_eq!(ob_sha.as_deref(), Some(w_sha.as_str()));
        assert!(!ob_payload.to_string().contains("[PII:"));
    }

    // 6. POST /api/studio/iam/simulate returns 200 {allowed: false} for synthetic/cross-org/unknown principal
    let (sim_syn_st, sim_syn_json, sim_syn_txt) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/simulate",
        Some(&admin_key),
        Some(json!({
            "org_id": Uuid::new_v4(),
            "principal_id": Uuid::new_v4(),
            "auth_level": 1,
            "access_scope": "private",
            "min_auth_level": 9,
            "resource_arn": "arn:openrag:org1:dept1:private:document:L9/d1",
            "action": "read",
        })),
    )
    .await?;
    assert_eq!(sim_syn_st, StatusCode::OK, "synthetic simulate failed: {sim_syn_txt}");
    assert_eq!(sim_syn_json["allowed"], false);

    // Real cross-org principal ID must also fail closed without DB leak
    let (sim_cross_st, sim_cross_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/simulate",
        Some(&admin_key),
        Some(json!({
            "org_id": seeded.other_org_id,
            "principal_id": seeded.other_admin_id,
            "action": "read",
        })),
    )
    .await?;
    assert_eq!(sim_cross_st, StatusCode::OK);
    assert_eq!(sim_cross_json["allowed"], false);
    assert_eq!(sim_cross_json["reason"], "principal_not_found_or_cross_org");

    let (sim_unk_st, sim_unk_json, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/simulate",
        Some(&admin_key),
        Some(json!({
            "org_id": seeded.org_id,
            "principal_id": Uuid::new_v4(),
            "action": "read",
        })),
    )
    .await?;
    assert_eq!(sim_unk_st, StatusCode::OK);
    assert_eq!(sim_unk_json["allowed"], false);

    let (sim_bad_arn_st, _, _) = send_json(
        &app,
        Method::POST,
        "/api/studio/iam/simulate",
        Some(&admin_key),
        Some(json!({
            "org_id": seeded.org_id,
            "principal_id": seeded.owner_id,
            "resource_arn": "not:an:arn",
            "action": "read",
        })),
    )
    .await?;
    assert!(sim_bad_arn_st == StatusCode::BAD_REQUEST || sim_bad_arn_st == StatusCode::UNPROCESSABLE_ENTITY);

    // 7a. AppChatRequest::spaces filters retrieval spaces
    let spaces_token = format!("SpacesPgToken{}", Uuid::new_v4().simple());
    let (up_st, _, up_txt) = send_json(
        &app,
        Method::POST,
        &format!("/v2/buckets/{own_bkt_id}/documents"),
        Some(&owner_key),
        Some(json!({
            "title": "Spaces Chunk Doc",
            "text": format!("{spaces_token} chunk document evidence"),
            "access_scope": "dept",
        })),
    )
    .await?;
    assert_eq!(up_st, StatusCode::CREATED, "upload doc for spaces failed: {up_txt}");

    let (chat_wiki_st, chat_wiki_json, chat_wiki_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/chat",
        Some(&owner_key),
        Some(json!({
            "query": spaces_token,
            "spaces": ["wiki"],
            "admission_threshold": 0.0,
        })),
    )
    .await?;
    assert_eq!(chat_wiki_st, StatusCode::OK, "chat wiki spaces failed: {chat_wiki_txt}");
    let wiki_cits = chat_wiki_json["citations"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing wiki citations"))?;
    assert!(
        wiki_cits.iter().all(|c| c["space"] == "wiki"),
        "expected only wiki citations when spaces=['wiki'], got: {wiki_cits:?}"
    );
    assert!(
        !wiki_cits.iter().any(|c| c["space"] == "chunks"),
        "chunk citations must be excluded when spaces=['wiki']"
    );

    let (chat_chunks_st, chat_chunks_json, chat_chunks_txt) = send_json(
        &app,
        Method::POST,
        "/api/app/chat",
        Some(&owner_key),
        Some(json!({
            "query": spaces_token,
            "spaces": ["chunks"],
            "admission_threshold": 0.0,
        })),
    )
    .await?;
    assert_eq!(chat_chunks_st, StatusCode::OK, "chat chunks spaces failed: {chat_chunks_txt}");
    let chunk_cits = chat_chunks_json["citations"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("missing chunk citations"))?;
    assert!(
        !chunk_cits.is_empty() && chunk_cits.iter().all(|c| c["space"] == "chunks"),
        "expected non-empty chunk citations when spaces=['chunks'], got: {chunk_cits:?}"
    );

    // 7b. Non-admin member gets 403 on Studio operational GET endpoints
    for uri in [
        "/api/studio/index/outbox",
        "/api/studio/pii/redactions",
        "/api/studio/gateway/logs",
    ] {
        let (st, _, txt) = send_json(&app, Method::GET, uri, Some(&owner_key), None).await?;
        assert_eq!(
            st,
            StatusCode::FORBIDDEN,
            "expected 403 for non-admin on GET {uri}, got {st}: {txt}"
        );
    }

    Ok(())
}
