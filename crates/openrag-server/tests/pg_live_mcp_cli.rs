#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

//! Live PostgreSQL tests for the 13-tool MCP server (HTTP + stdio engine) and the
//! `openrag` CLI binary. Run through `./scripts/test-pg-local.sh`, which connects as the
//! NOSUPERUSER `openrag_app` role so row-level security is actually enforced.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header::AUTHORIZATION, header::CONTENT_TYPE, Method, Request, StatusCode};
use axum::Router;
use http_body_util::BodyExt;
use openrag_core::OpenRagError;
use openrag_db::DbPool;
use openrag_server::llm_router::LlmHttpTransport;
use openrag_server::mcp::{McpEngine, StdioOptions};
use openrag_server::{build_router, AppState, OpenRagMode};
use serde_json::{json, Value};
use tower::ServiceExt;
use uuid::Uuid;

struct EchoTransport;

#[async_trait]
impl LlmHttpTransport for EchoTransport {
    async fn post_json(
        &self,
        _url: &str,
        _bearer: Option<&str>,
        body: &Value,
    ) -> Result<(u16, String), OpenRagError> {
        let joined: String = body
            .get("messages")
            .and_then(Value::as_array)
            .map(|m| {
                m.iter()
                    .filter_map(|x| x.get("content").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(" || ")
            })
            .unwrap_or_default();
        Ok((
            200,
            json!({ "choices": [{ "message": { "role": "assistant", "content": format!("Mock: {joined}") } }] })
                .to_string(),
        ))
    }
}

struct Seed {
    org_id: Uuid,
    slug: String,
    dept_id: Uuid,
    admin_id: Uuid,
    owner_id: Uuid,
    l2_id: Uuid,
    denied_group_id: Uuid,
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

async fn seed(worker: &DbPool) -> anyhow::Result<Seed> {
    let org_id = Uuid::new_v4();
    let slug = format!("m4c3-{}", &org_id.simple().to_string()[..8]);
    let dept_id = Uuid::new_v4();
    let dept2_id = Uuid::new_v4();
    let admin_id = Uuid::new_v4();
    let owner_id = Uuid::new_v4();
    let l2_id = Uuid::new_v4();
    let denied_group_id = Uuid::new_v4();
    sqlx::query("INSERT INTO iam.orgs (id, slug, name) VALUES ($1, $2, $3)")
        .bind(org_id)
        .bind(&slug)
        .bind(format!("Org {slug}"))
        .execute(worker.inner())
        .await?;
    seed_test_providers(worker, org_id).await?;
    sqlx::query(
        "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ($1, $2, 'Dept1', 'dept1'), ($3, $2, 'Dept2', 'dept2')",
    )
    .bind(dept_id)
    .bind(org_id)
    .bind(dept2_id)
    .execute(worker.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
         ($1, $2, $3, 'Admin', 10, true, true),
         ($4, $2, $5, 'OwnerL5', 5, false, false),
         ($6, $2, $7, 'PeerL2', 2, false, false)",
    )
    .bind(admin_id)
    .bind(org_id)
    .bind(format!("admin-{admin_id}@test.local"))
    .bind(owner_id)
    .bind(format!("owner-{owner_id}@test.local"))
    .bind(l2_id)
    .bind(format!("l2-{l2_id}@test.local"))
    .execute(worker.inner())
    .await?;
    sqlx::query(
        "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
         ($1, $2, $3, 'dept_admin'), ($1, $2, $4, 'member'), ($1, $2, $5, 'member')",
    )
    .bind(org_id)
    .bind(dept_id)
    .bind(admin_id)
    .bind(owner_id)
    .bind(l2_id)
    .execute(worker.inner())
    .await?;
    sqlx::query("INSERT INTO iam.groups (id, org_id, dept_id, name) VALUES ($1, $2, NULL, $3)")
        .bind(denied_group_id)
        .bind(org_id)
        .bind(format!("deny-{}", &denied_group_id.simple().to_string()[..8]))
        .execute(worker.inner())
        .await?;
    sqlx::query("INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES ($1, $2, $3)")
        .bind(org_id)
        .bind(denied_group_id)
        .bind(l2_id)
        .execute(worker.inner())
        .await?;
    Ok(Seed {
        org_id,
        slug,
        dept_id,
        admin_id,
        owner_id,
        l2_id,
        denied_group_id,
    })
}

struct Resp {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    json: Value,
    text: String,
}

async fn send(
    app: &Router,
    method: Method,
    uri: &str,
    token: Option<&str>,
    body: Option<Value>,
    extra: &[(&str, &str)],
) -> anyhow::Result<Resp> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(t) = token {
        b = b.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    for (k, v) in extra {
        b = b.header(*k, *v);
    }
    let req = match body {
        Some(v) => b.header(CONTENT_TYPE, "application/json").body(Body::from(serde_json::to_vec(&v)?))?,
        None => b.body(Body::empty())?,
    };
    let resp = app.clone().oneshot(req).await?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await?.to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Ok(Resp {
        status,
        headers,
        json,
        text,
    })
}

async fn mint(app: &Router, worker: &DbPool, org: Uuid, principal: Uuid, extra: Value) -> anyhow::Result<String> {
    let issued = openrag_server::auth::issue_cli_code(
        worker,
        org,
        principal,
        None,
        openrag_server::auth::CliCodeCaps::default(),
        None,
    )
    .await?;
    let mut payload = json!({ "code": issued.code, "org_id": org, "principal_id": principal });
    if let (Some(p), Some(e)) = (payload.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            p.insert(k.clone(), v.clone());
        }
    }
    let r = send(app, Method::POST, "/v1/auth/cli/code-exchange", None, Some(payload), &[]).await?;
    assert_eq!(r.status, StatusCode::CREATED, "{}", r.text);
    Ok(r.json["api_key"].as_str().unwrap_or_default().to_string())
}

struct ToolOut {
    is_error: bool,
    text: String,
    data: Value,
}

async fn call(engine: &McpEngine, name: &str, args: Value) -> anyhow::Result<ToolOut> {
    let reply = engine
        .handle_message(json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": name, "arguments": args } }))
        .await
        .ok_or_else(|| anyhow::anyhow!("no reply for {name}"))?;
    let result = reply
        .get("result")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{name}: JSON-RPC error {reply}"))?;
    let text = result["content"][0]["text"].as_str().unwrap_or_default().to_string();
    let data = result
        .get("structuredContent")
        .cloned()
        .unwrap_or_else(|| serde_json::from_str(&text).unwrap_or(Value::Null));
    Ok(ToolOut {
        is_error: result["isError"].as_bool().unwrap_or(true),
        text,
        data,
    })
}

fn engine(state: &Arc<AppState>, key: &str) -> McpEngine {
    McpEngine::for_stdio_with_state(
        Arc::clone(state),
        StdioOptions {
            api_key: Some(key.to_string()),
            principal_id: None,
        },
    )
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c3_mcp_http_transport_and_13_tools_under_rls() -> anyhow::Result<()> {
    let base = AppState::from_env().await?;
    let state = base.with_mode_and_transport(OpenRagMode::Offline, Arc::new(EchoTransport));
    let worker = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("worker pool required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let s = seed(&worker).await?;
    let admin_key = mint(&app, &worker, s.org_id, s.admin_id, json!({})).await?;
    let owner_key = mint(&app, &worker, s.org_id, s.owner_id, json!({})).await?;
    let l2_key = mint(&app, &worker, s.org_id, s.l2_id, json!({})).await?;

    // ---- Streamable HTTP transport -------------------------------------------------
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}});
    let r = send(&app, Method::POST, "/mcp", None, Some(init.clone()), &[]).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED, "unauthenticated /mcp must be 401");
    let r = send(&app, Method::GET, "/mcp", None, None, &[]).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);

    let r = send(&app, Method::POST, "/mcp", Some(&admin_key), Some(init.clone()), &[]).await?;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    let sid = r
        .headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| anyhow::anyhow!("initialize must issue Mcp-Session-Id"))?
        .to_string();
    let r = send(
        &app,
        Method::POST,
        "/mcp",
        Some(&admin_key),
        Some(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
        &[("mcp-session-id", &sid)],
    )
    .await?;
    assert_eq!(r.status, StatusCode::ACCEPTED);
    let list = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
    let r = send(&app, Method::POST, "/mcp", Some(&admin_key), Some(list.clone()), &[("mcp-session-id", &sid)]).await?;
    assert_eq!(r.json["result"]["tools"].as_array().map_or(0, Vec::len), 13);
    // Session bound to its principal: another key cannot ride it; unknown ids are 404.
    let r = send(&app, Method::POST, "/mcp", Some(&owner_key), Some(list.clone()), &[("mcp-session-id", &sid)]).await?;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    let r = send(&app, Method::POST, "/mcp", Some(&admin_key), Some(list.clone()), &[("mcp-session-id", "nope")]).await?;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
    // SSE-only clients get an event stream; JSON-RPC errors keep their codes.
    let r = send(
        &app,
        Method::POST,
        "/mcp",
        Some(&admin_key),
        Some(json!({"jsonrpc": "2.0", "id": 3, "method": "unknown/method"})),
        &[("accept", "text/event-stream")],
    )
    .await?;
    assert!(r.text.starts_with("event: message\ndata: "), "{}", r.text);
    assert!(r.text.contains("-32601"));
    let r = send(&app, Method::POST, "/mcp", Some(&admin_key), Some(list.clone()), &[("origin", "https://evil.example"), ("host", "rag.internal")]).await?;
    assert_eq!(r.status, StatusCode::FORBIDDEN, "cross-origin browser calls are rejected");
    let r = send(&app, Method::GET, "/mcp", Some(&admin_key), None, &[]).await?;
    assert_eq!(r.status, StatusCode::METHOD_NOT_ALLOWED);
    // Server-local file paths are never readable over HTTP.
    let r = send(
        &app,
        Method::POST,
        "/mcp",
        Some(&admin_key),
        Some(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "rag_ingest", "arguments": {"bucket": "x", "path": "/etc/hosts"}}})),
        &[],
    )
    .await?;
    assert_eq!(r.json["result"]["isError"], true, "{}", r.text);
    assert!(!r.text.contains("localhost"));
    let r = send(&app, Method::DELETE, "/mcp", Some(&admin_key), None, &[("mcp-session-id", &sid)]).await?;
    assert_eq!(r.status, StatusCode::NO_CONTENT);

    // ---- Tools under IAM/RLS -------------------------------------------------------
    let admin = engine(&state, &admin_key);
    let owner = engine(&state, &owner_key);
    let l2 = engine(&state, &l2_key);

    let bslug = format!("mcp-{}", &Uuid::new_v4().simple().to_string()[..6]);
    let r = send(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({"slug": bslug, "name": "MCP", "access_scope": "dept", "min_auth_level": 1, "dept_id": s.dept_id})),
        &[],
    )
    .await?;
    assert!(r.status.is_success(), "{}", r.text);

    let ing = call(&owner, "rag_ingest", json!({"bucket": bslug, "title": "MCP 출장비", "text": "MCP 출장비 한도는 1일 75,000원이며 담당자 연락처 010-9876-5432 입니다.", "access_scope": "dept", "min_auth_level": 2})).await?;
    assert!(!ing.is_error, "{}", ing.text);
    let doc_id = ing.data["document_id"].as_str().or_else(|| ing.data["id"].as_str()).unwrap_or_default().to_string();

    let srch = call(&l2, "rag_search", json!({"query": "출장비 한도 75,000원", "bucket": bslug, "top_k": 5})).await?;
    assert!(!srch.is_error && srch.text.contains("75,000"), "{}", srch.text);
    let fan = call(&l2, "rag_search", json!({"query": "출장비 한도 75,000원"})).await?;
    assert!(!fan.is_error && fan.text.contains("75,000"), "bucket fan-out: {}", fan.text);
    let missing = call(&l2, "rag_search", json!({})).await?;
    assert!(missing.is_error);

    let ask = call(&l2, "rag_ask", json!({"query": "출장비 한도는 얼마인가요?", "bucket": bslug})).await?;
    assert!(!ask.is_error, "{}", ask.text);
    assert!(!ask.data["answer"].as_str().unwrap_or_default().is_empty());

    let page = call(&l2, "rag_get_page", json!({"document_id": doc_id, "page_no": 1})).await?;
    assert!(!page.is_error, "{}", page.text);
    assert!(page.text.contains("75,000"));
    assert!(!page.text.contains("010-9876-5432"), "page text must be redacted: {}", page.text);

    // L6 private document: invisible to the L2 peer and to the L5 owner-peer.
    let secret = call(&admin, "rag_ingest", json!({"bucket": bslug, "title": "L6", "text": "L6_RESTRICTED_PAGE_DATA 보고서", "access_scope": "private", "min_auth_level": 6})).await?;
    assert!(!secret.is_error, "{}", secret.text);
    let secret_doc = secret.data["document_id"].as_str().unwrap_or_default().to_string();
    for (who, eng) in [("l2", &l2), ("owner", &owner)] {
        let p = call(eng, "rag_get_page", json!({"document_id": secret_doc, "page_no": 1})).await?;
        assert!(p.is_error, "{who} must not read the L6 private page: {}", p.text);
        assert!(!p.text.contains("L6_RESTRICTED_PAGE_DATA"));
        let q = call(eng, "rag_search", json!({"query": "L6_RESTRICTED_PAGE_DATA 보고서", "bucket": bslug})).await?;
        assert!(!q.text.contains(&secret_doc), "{who} search leaked the L6 doc id");
    }
    let own = call(&admin, "rag_get_page", json!({"document_id": secret_doc})).await?;
    assert!(!own.is_error && own.text.contains("L6_RESTRICTED_PAGE_DATA"));

    // API-key ceiling: scoped to dept2 ARNs only -> dept1 page denied.
    let scoped = mint(&app, &worker, s.org_id, s.admin_id, json!({"max_auth_level": 4, "scoped_arns": [format!("arn:openrag:{}:dept2:*", s.slug)]})).await?;
    let p = call(&engine(&state, &scoped), "rag_get_page", json!({"document_id": doc_id})).await?;
    assert!(p.is_error, "scoped_arns must bound rag_get_page: {}", p.text);

    // Wiki: explicit-deny group beats org scope.
    let wiki_id = Uuid::new_v4();
    let wiki_slug = format!("deny-{}", &wiki_id.simple().to_string()[..6]);
    let ok_id = Uuid::new_v4();
    let ok_slug = format!("ok-{}", &ok_id.simple().to_string()[..6]);
    sqlx::query(
        "INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, denied_group_ids, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
         VALUES ($1, $2, $3, $4, 'org', ARRAY[$5]::uuid[], 1, $6, $7, $8, 'DeniedWiki', 'DENIED_WIKI_SECRET', 'DENIED_WIKI_SECRET', $9),
                ($10, $2, $3, $4, 'dept', ARRAY[]::uuid[], 1, $11, $12, $13, 'MCP 위키 제목', '# MCP 위키 본문\n위키 검색 테스트 내용', 'MCP 위키 검색 테스트 내용', $14)",
    )
    .bind(wiki_id)
    .bind(s.org_id)
    .bind(s.dept_id)
    .bind(s.admin_id)
    .bind(s.denied_group_id)
    .bind(format!("arn:openrag:{}:dept1:org:wiki:L1/{wiki_id}", s.slug))
    .bind(format!("{}:org", s.org_id))
    .bind(&wiki_slug)
    .bind(openrag_db::sha256_hex(&wiki_id.to_string()))
    .bind(ok_id)
    .bind(format!("arn:openrag:{}:dept1:dept:wiki:L1/{ok_id}", s.slug))
    .bind(format!("{}:dept1", s.org_id))
    .bind(&ok_slug)
    .bind(openrag_db::sha256_hex(&ok_id.to_string()))
    .execute(worker.inner())
    .await?;
    let wk = call(&l2, "wiki_get", json!({"slug": wiki_slug})).await?;
    assert!(wk.is_error && !wk.text.contains("DENIED_WIKI_SECRET"), "{}", wk.text);
    let wk = call(&admin, "wiki_get", json!({"slug": wiki_slug})).await?;
    assert!(!wk.is_error && wk.text.contains("DENIED_WIKI_SECRET"), "{}", wk.text);
    let ws = call(&owner, "wiki_search", json!({"query": "위키 검색 테스트"})).await?;
    assert!(!ws.is_error && ws.text.contains(&ok_slug), "{}", ws.text);
    let ws = call(&l2, "wiki_search", json!({"query": "DENIED_WIKI_SECRET"})).await?;
    assert!(!ws.text.contains("DENIED_WIKI_SECRET"), "denied wiki leaked via search");

    // Memory: reflect -> engage -> amend -> active search; literal vault tokens neutralized.
    let rf = call(&owner, "memory_reflect", json!({"content": "MCP 메모리 초기 기록 10만원", "scope": "dept"})).await?;
    assert!(!rf.is_error, "{}", rf.text);
    let mem_id = rf.data["memory_id"].as_str().unwrap_or_default().to_string();
    let eng = call(&owner, "memory_engage", json!({"query": "메모리 초기 기록", "scope": "dept"})).await?;
    assert!(!eng.is_error && eng.text.contains("10만원"), "{}", eng.text);
    let pt = call(&owner, "memory_patch", json!({"memory_id": mem_id, "op": "amend", "text": "MCP 메모리 개정 기록 12만원", "reason": "상향"})).await?;
    assert!(!pt.is_error, "{}", pt.text);
    let bad_op = call(&owner, "memory_patch", json!({"memory_id": mem_id, "op": "overwrite", "text": "x"})).await?;
    assert!(bad_op.is_error);
    let ms = call(&owner, "memory_search", json!({"query": "메모리 개정 기록", "scope": "dept"})).await?;
    assert!(!ms.is_error && ms.text.contains("12만원"), "{}", ms.text);
    let laundered = call(&owner, "memory_reflect", json!({"content": "기록 [PII:rrn:deadbeef] 세탁 시도", "scope": "private"})).await?;
    assert!(!laundered.is_error, "{}", laundered.text);
    let lid = Uuid::parse_str(laundered.data["memory_id"].as_str().unwrap_or_default())?;
    let stored: String = sqlx::query_scalar("SELECT effective_text FROM memory.entries WHERE id = $1")
        .bind(lid)
        .fetch_one(worker.inner())
        .await?;
    assert!(!stored.contains("[PII:rrn:deadbeef]"), "literal vault token must be neutralized: {stored}");
    let too_high = call(&l2, "memory_reflect", json!({"content": "x", "min_auth_level": 5})).await?;
    assert!(too_high.is_error, "cannot write above own level");

    // Guard: register the rules explicitly so the test does not depend on the cwd-relative
    // configs/pipeline.default.yaml being found.
    for rule in [
        json!({"kind": "keyword", "pattern_or_category": "CONFIDENTIAL-DRAFT", "action": "warn", "surfaces": ["mcp_egress", "llm_egress"], "enabled": true}),
        json!({"rule_kind": "safe_word", "pattern_or_category": "PROJECT-BLACKBOX", "action": "block", "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"], "enabled": true}),
        json!({"kind": "pii_category", "pattern_or_category": "phone_kr", "action": "mask", "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"], "enabled": true}),
    ] {
        let r = send(&app, Method::POST, "/api/studio/pii/guard-rules", Some(&admin_key), Some(rule), &[]).await?;
        assert!(r.status.is_success(), "guard rule: {}", r.text);
    }
    let g = call(&l2, "guard_check", json!({"text": "일반 업무 회의록 공유", "surface": "mcp_egress"})).await?;
    assert_eq!(g.data["decision"], "allow", "{}", g.text);
    let g = call(&l2, "guard_check", json!({"text": "CONFIDENTIAL-DRAFT 검토 요청", "surface": "mcp_egress"})).await?;
    assert!(matches!(g.data["decision"].as_str(), Some("warn" | "mask")), "{}", g.text);
    let g = call(&l2, "guard_check", json!({"text": "연락처 010-1234-5678 확인", "surface": "mcp_egress"})).await?;
    assert_eq!(g.data["decision"], "mask");
    assert!(!g.data["sanitized_text"].as_str().unwrap_or_default().contains("010-1234-5678"));
    let g = call(&l2, "guard_check", json!({"text": "PROJECT-BLACKBOX 일정", "surface": "mcp_egress"})).await?;
    assert_eq!(g.data["decision"], "block");
    let blocked = call(&owner, "rag_ingest", json!({"bucket": bslug, "title": "Blocked Gate", "text": "PROJECT-BLACKBOX 내부 일정"})).await?;
    assert!(blocked.is_error && blocked.text.to_lowercase().contains("block"), "{}", blocked.text);
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM corpus.documents WHERE org_id = $1 AND title = 'Blocked Gate'")
        .bind(s.org_id)
        .fetch_one(worker.inner())
        .await?;
    assert_eq!(n, 0, "blocked ingestion must not write");
    let ask_blocked = call(&l2, "rag_ask", json!({"query": "PROJECT-BLACKBOX 알려줘"})).await?;
    assert!(ask_blocked.is_error && ask_blocked.text.contains("guard_blocked"), "{}", ask_blocked.text);

    // Concepts.
    let selfl = call(&owner, "concept_link", json!({"source": "임베딩", "target": " 임베딩 "})).await?;
    assert!(selfl.is_error);
    let lk = call(&owner, "concept_link", json!({"source": "임베딩", "target": "embedding", "canonical_name": "embedding"})).await?;
    assert!(!lk.is_error, "{}", lk.text);
    let rs = call(&l2, "concept_resolve", json!({"term": "임베딩", "include_aliases": true})).await?;
    assert!(!rs.is_error && rs.text.contains("embedding"), "{}", rs.text);

    // Identity failures.
    let mismatch = McpEngine::for_stdio_with_state(
        Arc::clone(&state),
        StdioOptions {
            api_key: Some(l2_key.clone()),
            principal_id: Some(s.admin_id),
        },
    );
    assert!(call(&mismatch, "guard_check", json!({"text": "x"})).await?.is_error);
    sqlx::query("UPDATE iam.api_keys SET revoked_at = now() WHERE key_hash = $1")
        .bind(openrag_db::sha256_hex(&l2_key))
        .execute(worker.inner())
        .await?;
    let revoked = call(&l2, "rag_search", json!({"query": "출장비"})).await?;
    assert!(revoked.is_error && revoked.text.contains("unauthorized"), "{}", revoked.text);
    Ok(())
}

/// Config file enabling local operator mode, written once per test process.
fn operator_config() -> anyhow::Result<String> {
    let path = std::env::temp_dir().join(format!("openrag-test-operator-{}.toml", std::process::id()));
    if !path.exists() {
        std::fs::write(&path, "operator = true\n")?;
    }
    Ok(path.display().to_string())
}

/// Runs the real `openrag` binary isolated from the developer's own config and env;
/// operator mode is enabled through a config file unless `envs` overrides OPENRAG_CONFIG.
fn cli(args: &[&str], envs: &[(&str, &str)]) -> anyhow::Result<std::process::Output> {
    let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_openrag"));
    cmd.args(args)
        .env_remove("OPENRAG_API_KEY")
        .env_remove("OPENRAG_API_URL")
        .env_remove("OPENRAG_ORG_ID")
        .env_remove("OPENRAG_PRINCIPAL_ID")
        .env_remove("OPENRAG_OPERATOR")
        .env("OPENRAG_CONFIG", operator_config()?);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    Ok(cmd.output()?)
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c3_openrag_cli_binary_identity_and_commands() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("worker pool required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let s = seed(&worker).await?;
    let owner_key = mint(&app, &worker, s.org_id, s.owner_id, json!({})).await?;
    let l2_key = mint(&app, &worker, s.org_id, s.l2_id, json!({})).await?;
    let org = s.org_id.to_string();

    let bslug = format!("cli-{}", &Uuid::new_v4().simple().to_string()[..6]);
    let r = send(
        &app,
        Method::POST,
        "/v2/buckets",
        Some(&owner_key),
        Some(json!({"slug": bslug, "name": "CLI", "access_scope": "dept", "min_auth_level": 1, "dept_id": s.dept_id})),
        &[],
    )
    .await?;
    assert!(r.status.is_success(), "{}", r.text);
    let bucket_id = r.json["id"].as_str().or_else(|| r.json["bucket_id"].as_str()).unwrap_or_default().to_string();

    // ingest as an API-key principal, then find it through the REST search.
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("cli.md");
    std::fs::write(&file, "# CLI 적재\nCLI적재고유토큰Z991 검증 본문")?;
    let file_s = file.display().to_string();
    let out = cli(&["ingest", &file_s, "--bucket", &bslug, "--scope", "dept", "--auth-level", "3"], &[("OPENRAG_API_KEY", &owner_key)])?;
    assert!(out.status.success(), "ingest: {}", String::from_utf8_lossy(&out.stderr));
    let srch = send(
        &app,
        Method::POST,
        &format!("/v2/buckets/{bucket_id}/search"),
        Some(&owner_key),
        Some(json!({"query": "CLI적재고유토큰Z991", "top_k": 5})),
        &[],
    )
    .await?;
    let hit_text = srch.json["results"].to_string();
    assert!(hit_text.contains("CLI적재고유토큰Z991"), "{}", srch.text);
    let out = cli(&["ingest", &file_s, "--bucket", &bslug, "--auth-level", "11"], &[])?;
    assert!(!out.status.success());

    // memory patch in local-operator mode acts as the entry owner.
    let rf = call(&engine(&state, &owner_key), "memory_reflect", json!({"content": "일비 2만원", "scope": "dept"})).await?;
    let mem_id = rf.data["memory_id"].as_str().unwrap_or_default().to_string();
    let out = cli(&["memory", "patch", &mem_id, "--op", "amend", "--text", "일비 3만원", "--reason", "2026 개정"], &[])?;
    assert!(out.status.success(), "patch: {}", String::from_utf8_lossy(&out.stderr));
    let (text, version): (String, i32) = sqlx::query_as("SELECT effective_text, version FROM memory.entries WHERE id = $1")
        .bind(Uuid::parse_str(&mem_id)?)
        .fetch_one(worker.inner())
        .await?;
    assert_eq!((text.as_str(), version), ("일비 3만원", 2));
    let out = cli(&["memory", "patch", &mem_id, "--op", "retract", "--reason", "철회"], &[])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&["memory", "patch", &mem_id, "--op", "amend", "--text", "부활 시도"], &[])?;
    assert!(!out.status.success(), "a non-admin owner cannot patch a retracted entry");
    let out = cli(&["memory", "patch", &mem_id, "--op", "overwrite"], &[])?;
    assert!(!out.status.success());

    // index status / drain (operator, one org) and engine switch.
    let out = cli(&["index", "status"], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("pending"));
    let out = cli(&["index", "drain", "--limit", "10"], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&["index", "engine", "corpus.chunks", "--set", "pgvector"], &[("OPENRAG_ORG_ID", &org), ("OPENRAG_MODE", "offline")])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let engine_now: String = sqlx::query_scalar(
        "SELECT pg_pinecone_engine FROM vectors.index_bindings WHERE org_id = $1 AND table_name = 'corpus.chunks'",
    )
    .bind(s.org_id)
    .fetch_one(worker.inner())
    .await?;
    assert_eq!(engine_now, "pgvector");
    let out = cli(&["index", "engine", "iam.principals", "--set", "native"], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(!out.status.success(), "non-vector table must be rejected");
    let out = cli(&["index", "engine", "corpus.chunks", "--set", "native"], &[("OPENRAG_API_KEY", &l2_key)])?;
    assert!(!out.status.success(), "non-admin key cannot switch engines");
    let out = cli(&["index", "mirror", "corpus.chunks", "--to", "s3-vectors"], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let prefix = format!("sv-prod-{}", &openrag_db::sha256_hex(&org)[..24]);
    assert!(String::from_utf8_lossy(&out.stdout).contains(&prefix));
    let out = cli(&["index", "retry", "--id", &Uuid::new_v4().to_string()], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(!out.status.success(), "unknown quarantine id must fail");

    // redact: non-admin key refused; operator mode attributed to the org admin.
    let rf = call(&engine(&state, &owner_key), "memory_reflect", json!({"content": "주민번호 900101-1234567 보관", "scope": "dept"})).await?;
    let rid = rf.data["memory_id"].as_str().unwrap_or_default().to_string();
    let out = cli(&["redact", "run", "--memory", &rid, "--reason", "r", "--hard-purge"], &[("OPENRAG_API_KEY", &l2_key)])?;
    assert!(!out.status.success(), "non-admin key must not redact");
    let out = cli(&["redact", "run", "--memory", &rid, "--reason", "gdpr", "--hard-purge"], &[])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(stdout.contains(&s.admin_id.to_string()), "attribution must be printed: {stdout}");

    // concept: self-link rejected; link + list through the API key.
    let out = cli(&["concept", "link", "a", "A"], &[("OPENRAG_API_KEY", &owner_key)])?;
    assert!(!out.status.success());
    let out = cli(&["concept", "link", "세컨드 브레인", "second brain", "--canonical", "second-brain"], &[("OPENRAG_API_KEY", &owner_key)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&["concept", "list", "--query", "second", "--json"], &[("OPENRAG_API_KEY", &owner_key)])?;
    assert!(out.status.success());
    let listed: Value = serde_json::from_slice(&out.stdout)?;
    assert!(listed.to_string().contains("second"));
    let out = cli(&["concept", "unlink", "세컨드 브레인", "second brain"], &[("OPENRAG_API_KEY", &owner_key)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // wiki import (with an escaping symlink) / export.
    let vault = tempfile::tempdir()?;
    std::fs::write(vault.path().join("safe.md"), "---\ntitle: Safe\n---\n# Safe\n[[other]] 링크 [PII:rrn:cafebabe]")?;
    #[cfg(unix)]
    std::os::unix::fs::symlink("/etc/hosts", vault.path().join("escape.md"))?;
    let vault_s = vault.path().display().to_string();
    let out = cli(&["wiki", "import", &vault_s, "--bucket", &bslug, "--scope", "dept", "--auth-level", "3"], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let bodies: Vec<String> = sqlx::query_scalar("SELECT body_md FROM wiki.pages WHERE bucket_id = $1")
        .bind(Uuid::parse_str(&bucket_id)?)
        .fetch_all(worker.inner())
        .await?;
    assert!(!bodies.is_empty());
    assert!(bodies.iter().all(|b| !b.contains("localhost") && !b.contains("[PII:rrn:cafebabe]")), "{bodies:?}");
    let export = tempfile::tempdir()?;
    let export_s = export.path().display().to_string();
    let out = cli(&["wiki", "export", &bslug, "--out", &export_s], &[("OPENRAG_ORG_ID", &org)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(std::fs::read_dir(export.path())?.count() >= 1);

    // up: unknown profile rejected, dry-run prints the compose command.
    let out = cli(&["up", "--profile", "cloud"], &[])?;
    assert!(!out.status.success());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root_s = root.display().to_string();
    let out = cli(&["up", "--profile", "airgap", "--dry-run", "--root", &root_s], &[])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("docker-compose.airgap.yml"));
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c3_one_time_cli_codes_close_the_key_minting_bypass() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("worker pool required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let s = seed(&worker).await?;
    let exchange = |payload: Value| {
        let app = app.clone();
        async move { send(&app, Method::POST, "/v1/auth/cli/code-exchange", None, Some(payload), &[]).await }
    };

    // The old bypass: any string + org/principal ids no longer mints a key.
    let r = exchange(json!({"code": "anything-goes", "org_id": s.org_id, "principal_id": s.admin_id})).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{}", r.text);
    let keys: i64 = sqlx::query_scalar("SELECT count(*) FROM iam.api_keys WHERE org_id = $1")
        .bind(s.org_id)
        .fetch_one(worker.inner())
        .await?;
    assert_eq!(keys, 0);

    // Bootstrap: the local operator issues the first admin's code with the CLI.
    let out = cli(&["auth", "code", "--principal", &s.admin_id.to_string()], &[])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let issued: Value = serde_json::from_slice(&out.stdout)?;
    let code = issued["code"].as_str().unwrap_or_default().to_string();
    assert!(code.starts_with("oragc_"), "{issued}");
    let stored: i64 = sqlx::query_scalar("SELECT count(*) FROM iam.cli_auth_codes WHERE code_hash = $1")
        .bind(openrag_db::sha256_hex(&code))
        .fetch_one(worker.inner())
        .await?;
    assert_eq!(stored, 1, "only the hash is stored");

    // Wrong principal in the request: rejected and the code is NOT consumed.
    let r = exchange(json!({"code": code, "principal_id": s.owner_id})).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
    let r = exchange(json!({"code": code, "org_id": s.org_id, "principal_id": s.admin_id})).await?;
    assert_eq!(r.status, StatusCode::CREATED, "{}", r.text);
    let admin_key = r.json["api_key"].as_str().unwrap_or_default().to_string();
    // Single use.
    let r = exchange(json!({"code": code})).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);

    // Expiry.
    let short = openrag_server::auth::issue_cli_code(&worker, s.org_id, s.owner_id, None, Default::default(), Some(1)).await?;
    tokio::time::sleep(std::time::Duration::from_millis(1300)).await;
    let r = exchange(json!({"code": short.code})).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);

    // Admin-issued codes; non-admins and anonymous callers cannot issue.
    let r = send(&app, Method::POST, "/api/studio/iam/cli-codes", None, Some(json!({"principal_id": s.owner_id})), &[]).await?;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
    let r = send(
        &app,
        Method::POST,
        "/api/studio/iam/cli-codes",
        Some(&admin_key),
        Some(json!({"principal_id": s.owner_id, "max_auth_level": 4})),
        &[],
    )
    .await?;
    assert_eq!(r.status, StatusCode::CREATED, "{}", r.text);
    let owner_code = r.json["code"].as_str().unwrap_or_default().to_string();
    // Ceilings narrow, never widen.
    let r = exchange(json!({"code": owner_code, "max_auth_level": 9})).await?;
    assert_eq!(r.status, StatusCode::CREATED, "{}", r.text);
    assert_eq!(r.json["max_auth_level"], 4);
    let owner_key = r.json["api_key"].as_str().unwrap_or_default().to_string();
    let r = send(&app, Method::POST, "/api/studio/iam/cli-codes", Some(&owner_key), Some(json!({"principal_id": s.owner_id})), &[]).await?;
    assert_eq!(r.status, StatusCode::FORBIDDEN, "non-admin cannot issue codes");
    let out = cli(&["auth", "code", "--principal", &s.owner_id.to_string()], &[("OPENRAG_API_KEY", &owner_key)])?;
    assert!(!out.status.success());

    // An ARN-scoped admin key can only issue codes inside its own scope.
    let dept1 = format!("arn:openrag:{}:dept1:*", s.slug);
    let scoped_admin = {
        let c = openrag_server::auth::issue_cli_code(
            &worker,
            s.org_id,
            s.admin_id,
            None,
            openrag_server::auth::CliCodeCaps {
                max_auth_level: None,
                scoped_arns: vec![dept1.clone()],
                scoped_group_ids: Vec::new(),
            },
            None,
        )
        .await?;
        let r = exchange(json!({"code": c.code, "scoped_arns": [format!("arn:openrag:{}:dept2:*", s.slug)]})).await?;
        assert_eq!(r.status, StatusCode::FORBIDDEN, "exchange cannot widen the code's ARN scope");
        let r = exchange(json!({"code": c.code})).await?;
        assert_eq!(r.status, StatusCode::CREATED, "{}", r.text);
        r.json["api_key"].as_str().unwrap_or_default().to_string()
    };
    let r = send(&app, Method::POST, "/api/studio/iam/cli-codes", Some(&scoped_admin), Some(json!({"principal_id": s.l2_id})), &[]).await?;
    assert_eq!(r.status, StatusCode::CREATED, "{}", r.text);
    assert_eq!(r.json["scoped_arns"], json!([dept1]), "inherits the issuer scope");
    let r = send(
        &app,
        Method::POST,
        "/api/studio/iam/cli-codes",
        Some(&scoped_admin),
        Some(json!({"principal_id": s.l2_id, "scoped_arns": [format!("arn:openrag:{}:*", s.slug)]})),
        &[],
    )
    .await?;
    assert_eq!(r.status, StatusCode::FORBIDDEN);
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c3_cli_config_toml_identity_and_operator_opt_in() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let state = AppState::from_env().await?;
    let worker = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("worker pool required"))?
        .clone();
    let app = build_router(Arc::clone(&state));
    let s = seed(&worker).await?;
    let owner_key = mint(&app, &worker, s.org_id, s.owner_id, json!({})).await?;
    let rf = call(&engine(&state, &owner_key), "memory_reflect", json!({"content": "설정 파일 테스트 메모", "scope": "dept"})).await?;
    let mem_id = rf.data["memory_id"].as_str().unwrap_or_default().to_string();
    let dir = tempfile::tempdir()?;

    // Operator mode is opt-in: an empty config + DATABASE_URL alone is not an identity.
    let empty = dir.path().join("empty.toml");
    std::fs::write(&empty, "")?;
    let empty_s = empty.display().to_string();
    let out = cli(&["memory", "patch", &mem_id, "--op", "amend", "--text", "x"], &[("OPENRAG_CONFIG", &empty_s)])?;
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no identity"), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&["memory", "patch", &mem_id, "--op", "amend", "--text", "x"], &[("OPENRAG_CONFIG", &empty_s), ("OPENRAG_OPERATOR", "1")])?;
    assert!(out.status.success(), "env opt-in: {}", String::from_utf8_lossy(&out.stderr));

    // api_key from the file; a world-readable file holding credentials is refused.
    let with_key = dir.path().join("key.toml");
    std::fs::write(&with_key, format!("api_key = \"{owner_key}\"\norg_id = \"{}\"\n", s.org_id))?;
    std::fs::set_permissions(&with_key, std::fs::Permissions::from_mode(0o644))?;
    let key_s = with_key.display().to_string();
    let out = cli(&["concept", "list", "--json"], &[("OPENRAG_CONFIG", &key_s)])?;
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("chmod 600"));
    std::fs::set_permissions(&with_key, std::fs::Permissions::from_mode(0o600))?;
    let out = cli(&["concept", "list", "--json"], &[("OPENRAG_CONFIG", &key_s)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&["memory", "patch", &mem_id, "--op", "amend", "--text", "설정 키로 개정", "--reason", "cfg"], &[("OPENRAG_CONFIG", &key_s)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // Environment beats the file: a bogus key in the env must be used (and fail).
    let out = cli(&["concept", "list", "--json"], &[("OPENRAG_CONFIG", &key_s), ("OPENRAG_API_KEY", "orag_bogus")])?;
    assert!(!out.status.success(), "env OPENRAG_API_KEY must override config.toml");

    // Typos are rejected, explicit missing file is an error.
    let typo = dir.path().join("typo.toml");
    std::fs::write(&typo, "api_kye = \"x\"\n")?;
    let out = cli(&["concept", "list"], &[("OPENRAG_CONFIG", &typo.display().to_string())])?;
    assert!(!out.status.success() && String::from_utf8_lossy(&out.stderr).contains("invalid config"));
    let out = cli(&["--help"], &[("OPENRAG_CONFIG", "/nonexistent/openrag.toml")])?;
    assert!(!out.status.success());

    // openrag-mcp reads api_key from the same file.
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_openrag-mcp"))
        .env_remove("OPENRAG_API_KEY")
        .env("OPENRAG_CONFIG", &key_s)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let msg = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "guard_check", "arguments": {"text": "일반 문서"}}});
        stdin.write_all(format!("{msg}\n").as_bytes()).await?;
    }
    let out = child.wait_with_output().await?;
    let reply: Value = serde_json::from_slice(&out.stdout)?;
    assert_eq!(reply["result"]["isError"], false, "{reply}");

    // valid_from / valid_to / revision are honored: an expired document is not searchable.
    let bslug = format!("tm-{}", &Uuid::new_v4().simple().to_string()[..6]);
    let r = send(&app, Method::POST, "/v2/buckets", Some(&owner_key), Some(json!({"slug": bslug, "name": "Time", "access_scope": "dept", "min_auth_level": 1, "dept_id": s.dept_id})), &[]).await?;
    assert!(r.status.is_success(), "{}", r.text);
    let doc = dir.path().join("old.md");
    std::fs::write(&doc, "# 구 규정\n만료된규정토큰Q77 내용")?;
    let doc_s = doc.display().to_string();
    let out = cli(&["ingest", &doc_s, "--bucket", &bslug, "--valid-from", "2020-01-01", "--valid-to", "2021-01-01", "--revision", "3"], &[("OPENRAG_CONFIG", &key_s)])?;
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let ingested: Value = serde_json::from_slice(&out.stdout)?;
    let doc_id = Uuid::parse_str(ingested["document_id"].as_str().or_else(|| ingested["id"].as_str()).unwrap_or_default())?;
    let (vf, vt, rev): (chrono::DateTime<chrono::Utc>, Option<chrono::DateTime<chrono::Utc>>, i32) =
        sqlx::query_as("SELECT valid_from, valid_to, revision FROM corpus.documents WHERE id = $1")
            .bind(doc_id)
            .fetch_one(worker.inner())
            .await?;
    assert_eq!(vf.to_rfc3339(), "2020-01-01T00:00:00+00:00");
    assert_eq!(vt.map(|d| d.to_rfc3339()).as_deref(), Some("2021-01-01T00:00:00+00:00"));
    assert_eq!(rev, 3);
    let q = call(&engine(&state, &owner_key), "rag_search", json!({"query": "만료된규정토큰Q77", "bucket": bslug, "latest_revision_only": true})).await?;
    assert!(!q.is_error, "{}", q.text);
    assert!(!q.data["results"].to_string().contains(&doc_id.to_string()), "expired doc must be dropped with latest_revision_only: {}", q.text);
    // MCP default is in-force only; non-current matches come back separately, labeled.
    let d = call(&engine(&state, &owner_key), "rag_search", json!({"query": "만료된규정토큰Q77", "bucket": bslug})).await?;
    assert_eq!(d.data["in_force_only"], true);
    assert!(!d.data["results"].to_string().contains(&doc_id.to_string()), "{}", d.text);
    let nif = d.data["not_in_force"].as_array().cloned().unwrap_or_default();
    assert!(
        nif.iter().any(|h| h["document_id"] == json!(doc_id) && h["validity"] == "expired"),
        "expired match must be surfaced as not_in_force: {}",
        d.text
    );
    assert!(d.data["notice"].as_str().unwrap_or_default().contains("do not present them as current"));
    // As of a date inside its validity window, the same document is in force.
    let past = call(&engine(&state, &owner_key), "rag_search", json!({"query": "만료된규정토큰Q77", "bucket": bslug, "as_of": "2020-06-01"})).await?;
    let hits = past.data["results"].as_array().cloned().unwrap_or_default();
    assert!(
        hits.iter().any(|h| h["document_id"] == json!(doc_id) && h["validity"] == "current"),
        "as_of inside the window must return it as current: {}",
        past.text
    );
    // rag_ask grounds on in-force documents only by default, and says so when none matched.
    let ask_state = state.with_mode_and_transport(OpenRagMode::Offline, Arc::new(EchoTransport));
    let asker = engine(&ask_state, &owner_key);
    let now_ask = call(&asker, "rag_ask", json!({"query": "만료된규정토큰Q77", "bucket": bslug})).await?;
    assert!(!now_ask.is_error, "{}", now_ask.text);
    assert_eq!(now_ask.data["in_force_only"], true);
    assert!(!now_ask.data["citations"].to_string().contains(&doc_id.to_string()), "expired doc must not ground the answer: {}", now_ask.text);
    assert!(!now_ask.data["answer"].as_str().unwrap_or_default().contains("만료된규정토큰Q77 내용"), "{}", now_ask.text);
    assert!(now_ask.data["notice"].as_str().unwrap_or_default().contains("Do not present it as the current state"), "{}", now_ask.text);
    let past_ask = call(&asker, "rag_ask", json!({"query": "만료된규정토큰Q77", "bucket": bslug, "as_of": "2020-06-01"})).await?;
    let cites = past_ask.data["citations"].as_array().cloned().unwrap_or_default();
    assert!(
        cites.iter().any(|c| c["document_id"] == json!(doc_id) && c["validity"] == "current"),
        "as_of inside the window grounds on it, labeled current: {}",
        past_ask.text
    );
    assert!(past_ask.data.get("notice").is_none(), "{}", past_ask.text);
    let hist_ask = call(&asker, "rag_ask", json!({"query": "만료된규정토큰Q77", "bucket": bslug, "latest_revision_only": false})).await?;
    let cites = hist_ask.data["citations"].as_array().cloned().unwrap_or_default();
    assert!(
        cites.iter().any(|c| c["document_id"] == json!(doc_id) && c["validity"] == "expired"),
        "opt-in history keeps the expired label: {}",
        hist_ask.text
    );
    let bad = call(&engine(&state, &owner_key), "rag_ingest", json!({"bucket": bslug, "text": "x", "valid_from": "2026-02-01", "valid_to": "2026-01-01"})).await?;
    assert!(bad.is_error);
    let ok = call(&engine(&state, &owner_key), "rag_ingest", json!({"bucket": bslug, "title": "시행 문서", "text": "시행일 지정 문서 본문", "valid_from": "2026-01-01T09:00:00+09:00", "revision": 2})).await?;
    assert!(!ok.is_error, "{}", ok.text);
    Ok(())
}

#[tokio::test]
#[ignore = "requires live PostgreSQL"]
async fn pg_live_m4c3_migration_010_widens_seeded_sk_rule_only() -> anyhow::Result<()> {
    let state = AppState::from_env().await?;
    let worker = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("worker pool required"))?
        .clone();
    let s = seed(&worker).await?;
    // An org seeded before 010: the original default rule plus one rule an operator edited by hand.
    let policy_id: Uuid = sqlx::query_scalar(
        "INSERT INTO pii.guard_policies (org_id, name, master_enabled, default_action) VALUES ($1, 'default', true, 'mask') RETURNING id",
    )
    .bind(s.org_id)
    .fetch_one(worker.inner())
    .await?;
    let old_pattern = "(?i)sk-[a-z0-9]{20,}";
    let custom_pattern = "(?i)sk-[a-z0-9]{30,}";
    let mut ids = Vec::new();
    for pattern in [old_pattern, custom_pattern] {
        let id: Uuid = sqlx::query_scalar(
            "INSERT INTO pii.guard_rules (policy_id, org_id, rule_kind, pattern_or_category, action, surfaces, priority)
             VALUES ($1, $2, 'regex', $3, 'block', ARRAY['ingestion_gate','llm_egress','mcp_egress'], 1) RETURNING id",
        )
        .bind(policy_id)
        .bind(s.org_id)
        .bind(pattern)
        .fetch_one(worker.inner())
        .await?;
        ids.push(id);
    }
    let pattern_of = |id: Uuid| {
        let worker = worker.clone();
        async move {
            sqlx::query_scalar::<_, String>("SELECT pattern_or_category FROM pii.guard_rules WHERE id = $1")
                .bind(id)
                .fetch_one(worker.inner())
                .await
        }
    };

    let admin_key = mint(&build_router(Arc::clone(&state)), &worker, s.org_id, s.admin_id, json!({})).await?;
    let hyphenated = "배포 키 sk-proj-AbCdEf0123456789_ghIJ 공유";
    // The seeded pattern itself misses hyphenated keys. Checked on the pattern, not through the
    // DB: other tests in this binary start an AppState concurrently, and each start applies 010
    // to every org, so the row may already be rewritten by the time this test looks at it.
    assert!(!regex::Regex::new(old_pattern)?.is_match(hyphenated), "old rule misses hyphenated keys");

    // Whichever start rewrote it (this call or a concurrent one), the outcome must be the same.
    openrag_server::state::ensure_migrations_if_needed(&worker).await?;
    assert_eq!(pattern_of(ids[0]).await?, r"(?i)\bsk-[a-z0-9][a-z0-9_-]{15,}");
    assert_eq!(pattern_of(ids[1]).await?, custom_pattern, "hand-edited rule must be left alone");

    let after = call(&engine(&state, &admin_key), "guard_check", json!({"text": hyphenated, "surface": "mcp_egress"})).await?;
    assert_eq!(after.data["decision"], "block", "{}", after.text);

    // Idempotent: a second start leaves both rules as they are.
    openrag_server::state::ensure_migrations_if_needed(&worker).await?;
    assert_eq!(pattern_of(ids[0]).await?, r"(?i)\bsk-[a-z0-9][a-z0-9_-]{15,}");
    assert_eq!(pattern_of(ids[1]).await?, custom_pattern);
    Ok(())
}
