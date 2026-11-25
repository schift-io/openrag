#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

//! Offline MCP protocol tests: framing, lifecycle and fail-closed auth without a database.

use std::process::Stdio;
use std::sync::Arc;

use openrag_core::{AuthLevel, PrincipalContext};
use openrag_server::mcp::{
    serve_stream, tool_definitions, McpEngine, StdioOptions, TaskHookFn, TaskHookOutcome,
    TaskStore, JSONRPC_MISSING_REQUIRED_CLIENT_CAPABILITY, JSONRPC_UNSUPPORTED_PROTOCOL_VERSION,
    MCP_TOOL_NAMES,
};
use openrag_server::AppState;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::sync::watch;
use uuid::Uuid;

/// Empty config so binaries never pick up the developer's ~/.config/openrag/config.toml.
fn empty_config() -> anyhow::Result<String> {
    let path = std::env::temp_dir().join(format!("openrag-test-empty-{}.toml", std::process::id()));
    if !path.exists() {
        std::fs::write(&path, "")?;
    }
    Ok(path.display().to_string())
}

fn ndjson(msgs: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for m in msgs {
        out.extend_from_slice(m.to_string().as_bytes());
        out.push(b'\n');
    }
    out
}

fn content_length(msgs: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for m in msgs {
        let body = m.to_string();
        out.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
        out.extend_from_slice(body.as_bytes());
    }
    out
}

/// Parses server output that may mix NDJSON lines and Content-Length frames.
fn parse_output(raw: &[u8]) -> anyhow::Result<Vec<Value>> {
    let text = String::from_utf8(raw.to_vec())?;
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while !rest.trim().is_empty() {
        let trimmed = rest.trim_start();
        if let Some(after) = trimmed.strip_prefix("Content-Length: ") {
            let (len_str, tail) = after
                .split_once("\r\n\r\n")
                .ok_or_else(|| anyhow::anyhow!("bad header"))?;
            let len: usize = len_str.trim().parse()?;
            out.push(serde_json::from_str(&tail[..len])?);
            rest = &tail[len..];
        } else {
            let (line, tail) = trimmed.split_once('\n').unwrap_or((trimmed, ""));
            out.push(serde_json::from_str(line)?);
            rest = tail;
        }
    }
    Ok(out)
}

async fn run_engine(input: &[u8]) -> anyhow::Result<(Vec<Value>, Vec<u8>)> {
    let engine = McpEngine::for_stdio(StdioOptions::default());
    let mut output: Vec<u8> = Vec::new();
    serve_stream(&engine, input, &mut output).await?;
    Ok((parse_output(&output)?, output))
}

fn lifecycle() -> Vec<Value> {
    vec![
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "nonexistent_tool", "arguments": {}}}),
    ]
}

fn assert_lifecycle(replies: &[Value]) {
    // Notification produces no reply: 5 replies for 6 messages.
    assert_eq!(replies.len(), 5, "replies: {replies:?}");
    assert_eq!(replies[0]["id"], 1);
    assert_eq!(replies[0]["result"]["protocolVersion"], "2025-03-26");
    assert!(replies[0]["result"]["capabilities"]["tools"].is_object());
    assert_eq!(replies[1]["result"], json!({}));
    let names: Vec<&str> = replies[2]["result"]["tools"]
        .as_array()
        .map(|a| a.iter().filter_map(|t| t["name"].as_str()).collect())
        .unwrap_or_default();
    assert_eq!(names, MCP_TOOL_NAMES.to_vec());
    assert_eq!(replies[3]["error"]["code"], -32601);
    assert_eq!(replies[4]["error"]["code"], -32602);
}

#[tokio::test]
async fn stdio_ndjson_lifecycle_and_13_tools() -> anyhow::Result<()> {
    let (replies, raw) = run_engine(&ndjson(&lifecycle())).await?;
    assert!(!String::from_utf8_lossy(&raw).contains("Content-Length"));
    assert_lifecycle(&replies);
    Ok(())
}

#[tokio::test]
async fn stdio_content_length_lifecycle_answers_in_same_framing() -> anyhow::Result<()> {
    let (replies, raw) = run_engine(&content_length(&lifecycle())).await?;
    assert!(String::from_utf8_lossy(&raw).starts_with("Content-Length: "));
    assert_lifecycle(&replies);
    Ok(())
}

#[tokio::test]
async fn stdio_malformed_json_returns_32700_and_keeps_serving() -> anyhow::Result<()> {
    let mut input = b"{not-json\n".to_vec();
    input.extend(ndjson(&[json!({"jsonrpc": "2.0", "id": 9, "method": "ping"})]));
    let (replies, _) = run_engine(&input).await?;
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0]["error"]["code"], -32700);
    assert_eq!(replies[0]["id"], Value::Null);
    assert_eq!(replies[1]["id"], 9);
    Ok(())
}

#[tokio::test]
async fn stdio_batch_and_invalid_envelopes() -> anyhow::Result<()> {
    let batch = json!([
        {"jsonrpc": "2.0", "id": "a", "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "1.0", "id": "b", "method": "ping"}
    ]);
    let (replies, _) = run_engine(&ndjson(&[batch, json!([]), json!(42)])).await?;
    assert_eq!(replies.len(), 3);
    let first = replies[0].as_array().cloned().unwrap_or_default();
    assert_eq!(first.len(), 2, "notification inside batch gets no reply");
    assert_eq!(first[0]["id"], "a");
    assert_eq!(first[1]["error"]["code"], -32600);
    assert_eq!(replies[1]["error"]["code"], -32600);
    assert_eq!(replies[2]["error"]["code"], -32600);
    Ok(())
}

#[tokio::test]
async fn stdio_tool_call_without_api_key_fails_closed() -> anyhow::Result<()> {
    let call = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "rag_search", "arguments": {"query": "secret"}}});
    let (replies, _) = run_engine(&ndjson(&[call])).await?;
    assert_eq!(replies[0]["result"]["isError"], true);
    let text = replies[0]["result"]["content"][0]["text"].as_str().unwrap_or_default();
    assert!(
        text.contains("unauthorized") || text.contains("server_unavailable"),
        "unexpected: {text}"
    );
    Ok(())
}

#[tokio::test]
async fn openrag_binary_mcp_stdio_parse_error_without_database() -> anyhow::Result<()> {
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_openrag"))
        .args(["mcp", "--stdio"])
        .env_remove("DATABASE_URL")
        .env_remove("OPENRAG_API_KEY")
        .env("OPENRAG_CONFIG", empty_config()?)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"{not-json\n").await?;
    }
    let out = child.wait_with_output().await?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("-32700"), "stdout={stdout}");
    assert!(out.status.success());
    Ok(())
}

#[tokio::test]
async fn openrag_mcp_binary_lists_13_tools_with_content_length() -> anyhow::Result<()> {
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_openrag-mcp"))
        .env_remove("DATABASE_URL")
        .env_remove("OPENRAG_API_KEY")
        .env("OPENRAG_CONFIG", empty_config()?)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&content_length(&[json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})]))
            .await?;
    }
    let out = child.wait_with_output().await?;
    let replies = parse_output(&out.stdout)?;
    let count = replies[0]["result"]["tools"].as_array().map_or(0, Vec::len);
    assert_eq!(count, 16);
    Ok(())
}

#[test]
fn openrag_help_lists_every_subcommand() -> anyhow::Result<()> {
    let cfg = empty_config()?;
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_openrag"))
        .arg("--help")
        .env("OPENRAG_CONFIG", &cfg)
        .output()?;
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    for sub in [
        "up", "serve", "ingest", "index", "memory", "redact", "wiki", "concept", "bench", "pipeline", "mcp",
    ] {
        assert!(help.contains(sub), "--help missing {sub}");
        let sub_out = std::process::Command::new(env!("CARGO_BIN_EXE_openrag"))
            .args([sub, "--help"])
            .env("OPENRAG_CONFIG", &cfg)
            .output()?;
        assert!(sub_out.status.success(), "{sub} --help failed");
        assert!(!sub_out.stdout.is_empty());
    }
    let mcp = std::process::Command::new(env!("CARGO_BIN_EXE_openrag"))
        .args(["mcp", "--help"])
        .env("OPENRAG_CONFIG", &cfg)
        .output()?;
    assert!(String::from_utf8_lossy(&mcp.stdout).contains("--stdio"));
    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_openrag"))
        .arg("unknown-subcommand-xyz")
        .env("OPENRAG_CONFIG", &cfg)
        .output()?;
    assert!(!bad.status.success());
    Ok(())
}

// ============================================================================
// Dual-era MCP 2026-07-28, Skills, and Tasks offline tests
// ============================================================================

fn modern_meta(version: &str, client_caps: Value) -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": version,
        "io.modelcontextprotocol/clientInfo": { "name": "test-client", "version": "1.0.0" },
        "io.modelcontextprotocol/clientCapabilities": client_caps,
    })
}

fn assert_modern_envelope(result: &Value, expected_type: &str, expect_cacheable: bool) {
    assert_eq!(result["resultType"], expected_type);
    assert_eq!(
        result["_meta"]["io.modelcontextprotocol/serverInfo"],
        json!({ "name": "openrag", "version": env!("CARGO_PKG_VERSION") })
    );
    if expect_cacheable {
        assert_eq!(result["ttlMs"], 300_000);
        assert_eq!(result["cacheScope"], "private");
    } else {
        assert!(result.get("cacheScope").is_none());
    }
}

#[tokio::test]
async fn legacy_initialize_and_tools_list_output_is_pinned() -> anyhow::Result<()> {
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "pin", "version": "0" }
        }
    });
    let list_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });
    let (replies, _) = run_engine(&ndjson(&[init_req, list_req])).await?;
    assert_eq!(replies.len(), 2);

    let expected_init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "openrag", "version": env!("CARGO_PKG_VERSION") },
            "instructions": "OpenRAG knowledge tools. Every call runs as the authenticated principal: results are filtered by ARN scope, permission groups, permission level and row-level security, and text passes the PII/Safe-Word guard. Search with rag_search; each hit's source.path names the original file. Search the way a careful researcher would: split a question with several parts or conditions into one rag_search per part; for vague or descriptive questions also pass hypothetical_answer (a short passage worded like the document you expect); when results are weak, retry with other wording, synonyms or the formal term; when a hit is relevant but cut off, read its whole page with rag_get_page. Excel and CSV files are indexed as catalog cards (sheets, columns, value ranges, first rows): when a hit has source.kind = spreadsheet, read the rows you need with rag_read_sheet instead of guessing from the card; to find a specific value in it, pass find (and column) rather than paging through every row. rag_list_documents shows what is in a bucket."
        }
    });
    assert_eq!(replies[0], expected_init);

    let expected_list = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": tool_definitions()
        }
    });
    assert_eq!(replies[1], expected_list);
    Ok(())
}

#[tokio::test]
async fn modern_discover_versioning_unknown_methods_and_caching() -> anyhow::Result<()> {
    let engine = McpEngine::for_stdio(StdioOptions::default());

    // server/discover
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": "d1",
            "method": "server/discover",
            "params": { "_meta": modern_meta("2026-07-28", json!({})) }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    let res = &resp["result"];
    assert_modern_envelope(res, "complete", true);
    assert_eq!(res["supportedVersions"], json!(["2026-07-28"]));
    assert_eq!(
        res["capabilities"],
        json!({
            "tools": {},
            "resources": {},
            "extensions": {
                "io.modelcontextprotocol/skills": {},
                "io.modelcontextprotocol/tasks": {}
            }
        })
    );
    let instructions = res["instructions"].as_str().unwrap_or_default();
    assert!(instructions.contains("skill://openrag-search/SKILL.md"));
    assert!(instructions.contains("openrag-search"));

    // Modern tools/list preserves exact 16 tools in order and carries modern metadata + caching.
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": "t1",
            "method": "tools/list",
            "params": { "_meta": modern_meta("2026-07-28", json!({})) }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", true);
    assert_eq!(resp["result"]["tools"], json!(tool_definitions()));

    // Unsupported version -> -32022 with supported [modern, legacy...] and requested.
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": "v1",
            "method": "server/discover",
            "params": { "_meta": modern_meta("2099-01-01", json!({})) }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_eq!(resp["error"]["code"], JSONRPC_UNSUPPORTED_PROTOCOL_VERSION);
    assert_eq!(
        resp["error"]["data"],
        json!({
            "supported": ["2026-07-28", "2025-06-18", "2025-03-26", "2024-11-05"],
            "requested": "2099-01-01"
        })
    );

    // Removed/unimplemented methods -> -32601 in modern era.
    for method in ["ping", "logging/setLevel", "resources/subscribe", "resources/directory/read"] {
        let resp = engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": method,
                "method": method,
                "params": { "_meta": modern_meta("2026-07-28", json!({})) }
            }))
            .await
            .ok_or_else(|| anyhow::anyhow!("expected response"))?;
        assert_eq!(resp["error"]["code"], -32601, "method={method}");
    }

    // initialize is always legacy even if _meta is present.
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "_meta": modern_meta("2026-07-28", json!({}))
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_eq!(resp["result"]["protocolVersion"], "2025-03-26");
    assert!(resp["result"].get("resultType").is_none());

    Ok(())
}

#[tokio::test]
async fn modern_skills_and_resources_digest_size_and_frontmatter() -> anyhow::Result<()> {
    let engine = McpEngine::for_stdio(StdioOptions::default());
    let expected_raw = include_str!("../skills/openrag-search/SKILL.md");
    let expected_size = expected_raw.len();
    let expected_digest = format!("sha256:{}", hex::encode(Sha256::digest(expected_raw.as_bytes())));
    let skill_uri = "skill://openrag-search/SKILL.md";

    // skills/list
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "skills/list",
            "params": { "_meta": modern_meta("2026-07-28", json!({})) }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", true);
    let skills = resp["result"]["skills"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("skills must be array"))?;
    assert_eq!(skills.len(), 1);
    let entry = &skills[0];
    assert_eq!(entry["uri"], skill_uri);
    assert_eq!(entry["frontmatter"]["name"], "openrag-search");
    assert!(
        entry["frontmatter"]["description"]
            .as_str()
            .is_some_and(|d| !d.is_empty()),
        "frontmatter description must be non-empty"
    );
    assert_eq!(
        entry["resources"],
        json!([{
            "uri": skill_uri,
            "digest": expected_digest,
            "size": expected_size,
        }])
    );

    // skills/get returns the exact same entry.
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "skills/get",
            "params": {
                "uri": skill_uri,
                "_meta": modern_meta("2026-07-28", json!({}))
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", true);
    assert_eq!(&resp["result"]["skill"], entry);

    // resources/list
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": { "_meta": modern_meta("2026-07-28", json!({})) }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", true);
    let resources = resp["result"]["resources"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("resources must be array"))?;
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0]["uri"], skill_uri);
    assert_eq!(resources[0]["name"], "openrag-search");
    assert_eq!(resources[0]["mimeType"], "text/markdown");

    // resources/templates/list -> empty list.
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/templates/list",
            "params": { "_meta": modern_meta("2026-07-28", json!({})) }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", true);
    assert_eq!(resp["result"]["resourceTemplates"], json!([]));

    // resources/read returns exact embedded bytes.
    let resp = engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {
                "uri": skill_uri,
                "_meta": modern_meta("2026-07-28", json!({}))
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", true);
    assert_eq!(
        resp["result"]["contents"],
        json!([{
            "uri": skill_uri,
            "mimeType": "text/markdown",
            "text": expected_raw,
        }])
    );

    // Unknown URI -> -32602 for both skills/get and resources/read.
    for method in ["skills/get", "resources/read"] {
        let resp = engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": method,
                "params": {
                    "uri": "skill://does-not-exist/SKILL.md",
                    "_meta": modern_meta("2026-07-28", json!({}))
                }
            }))
            .await
            .ok_or_else(|| anyhow::anyhow!("expected response"))?;
        assert_eq!(resp["error"]["code"], -32602, "method={method}");
    }

    Ok(())
}

#[tokio::test]
async fn modern_tasks_lifecycle_isolation_cancel_and_cap() -> anyhow::Result<()> {
    let state = AppState::new_offline_for_test()?;
    let store = Arc::new(TaskStore::new());
    let org_id = Uuid::new_v4();
    let alice = PrincipalContext::new(org_id, Uuid::new_v4(), AuthLevel::new(5)?);
    let bob = PrincipalContext::new(org_id, Uuid::new_v4(), AuthLevel::new(5)?);

    let (gate_tx, gate_rx) = watch::channel(false);
    let hook: TaskHookFn = Arc::new(move |name, args| {
        if name != "rag_ingest" {
            return None;
        }
        let mode = args
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("ok")
            .to_string();
        let mut rx = gate_rx.clone();
        Some(Box::pin(async move {
            match mode.as_str() {
                "instant" => TaskHookOutcome::ToolOk(json!({ "ok": true, "mode": "instant" })),
                "tool_err" => TaskHookOutcome::ToolErr {
                    code: "bad_input".into(),
                    message: "simulated tool error".into(),
                },
                "rpc_err" => TaskHookOutcome::RpcErr {
                    code: -32603,
                    message: "simulated internal error".into(),
                },
                "slow" => {
                    let _ = rx.wait_for(|released| *released).await;
                    TaskHookOutcome::ToolOk(json!({ "ok": true, "mode": mode }))
                }
                _ => {
                    std::future::pending::<()>().await;
                    TaskHookOutcome::ToolOk(json!({ "ok": true, "mode": mode }))
                }
            }
        }))
    });

    let alice_engine = McpEngine::for_http(Arc::clone(&state), alice)
        .with_task_store(Arc::clone(&store))
        .with_task_hook(Arc::clone(&hook));
    let bob_engine = McpEngine::for_http(state, bob)
        .with_task_store(Arc::clone(&store))
        .with_task_hook(hook);

    let tasks_caps = json!({
        "extensions": { "io.modelcontextprotocol/tasks": {} }
    });

    // 1. Client without tasks capability always runs synchronously (never returns a task).
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rag_ingest",
                "arguments": { "bucket": "docs", "title": "instant", "path": "/tmp/a.pdf" },
                "_meta": modern_meta("2026-07-28", json!({}))
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", false);
    assert_eq!(resp["result"]["isError"], false);
    assert_eq!(resp["result"]["structuredContent"]["mode"], "instant");

    // 2. Client WITH tasks capability gets CreateTaskResult immediately while work is pending.
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "rag_ingest",
                "arguments": { "bucket": "docs", "title": "slow", "path": "/tmp/slow.pdf" },
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    let created = &resp["result"];
    assert_modern_envelope(created, "task", false);
    assert_eq!(created["status"], "working");
    assert_eq!(created["ttlMs"], 3_600_000);
    assert_eq!(created["pollIntervalMs"], 2_000);
    let task_id = created["taskId"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing taskId"))?
        .to_string();
    assert!(Uuid::parse_str(&task_id).is_ok());
    assert!(task_id.len() <= 128);

    // 3. tasks/get reports working before completion.
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tasks/get",
            "params": {
                "taskId": task_id,
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", false);
    assert_eq!(resp["result"]["taskId"], task_id);
    assert_eq!(resp["result"]["status"], "working");

    // 4. tasks/update acknowledges and ignores keys.
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tasks/update",
            "params": {
                "taskId": task_id,
                "inputResponses": { "unused_key": { "action": "accept" } },
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", false);

    // 5. Missing tasks capability on tasks/get -> -32021.
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tasks/get",
            "params": {
                "taskId": task_id,
                "_meta": modern_meta("2026-07-28", json!({}))
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_eq!(resp["error"]["code"], JSONRPC_MISSING_REQUIRED_CLIENT_CAPABILITY);

    // 6. Another principal (bob) gets -32602 for get/update/cancel.
    for (method, extra) in [
        ("tasks/get", json!({})),
        ("tasks/update", json!({ "inputResponses": {} })),
        ("tasks/cancel", json!({})),
    ] {
        let mut params = extra;
        if let Some(obj) = params.as_object_mut() {
            obj.insert("taskId".into(), json!(task_id));
            obj.insert("_meta".into(), modern_meta("2026-07-28", tasks_caps.clone()));
        }
        let resp = bob_engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": method,
                "params": params
            }))
            .await
            .ok_or_else(|| anyhow::anyhow!("expected response"))?;
        assert_eq!(resp["error"]["code"], -32602, "bob {method} must be -32602");
    }

    // 7. Release gate -> task completes with exact CallToolResult.
    let _ = gate_tx.send(true);
    let mut final_task = Value::Null;
    for _ in 0..40 {
        let resp = alice_engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tasks/get",
                "params": {
                    "taskId": task_id,
                    "_meta": modern_meta("2026-07-28", tasks_caps.clone())
                }
            }))
            .await
            .ok_or_else(|| anyhow::anyhow!("expected response"))?;
        if resp["result"]["status"] == "completed" {
            final_task = resp["result"].clone();
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(final_task["status"], "completed");
    assert_eq!(final_task["result"]["resultType"], "complete");
    assert_eq!(final_task["result"]["isError"], false);
    assert_eq!(final_task["result"]["structuredContent"]["mode"], "slow");

    // Cancelling an already-completed task leaves it completed.
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tasks/cancel",
            "params": {
                "taskId": task_id,
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", false);
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tasks/get",
            "params": {
                "taskId": task_id,
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_eq!(resp["result"]["status"], "completed");

    // 8. Tool error completes with status "completed" and isError: true; RPC error fails.
    for (mode, expected_status) in [("tool_err", "completed"), ("rpc_err", "failed")] {
        let resp = alice_engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "rag_ingest",
                    "arguments": { "bucket": "docs", "title": mode, "path": "/tmp/e.pdf" },
                    "_meta": modern_meta("2026-07-28", tasks_caps.clone())
                }
            }))
            .await
            .ok_or_else(|| anyhow::anyhow!("expected response"))?;
        let tid = resp["result"]["taskId"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing taskId"))?
            .to_string();
        let mut got = Value::Null;
        for _ in 0..40 {
            let r = alice_engine
                .handle_message(json!({
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tasks/get",
                    "params": {
                        "taskId": tid,
                        "_meta": modern_meta("2026-07-28", tasks_caps.clone())
                    }
                }))
                .await
                .ok_or_else(|| anyhow::anyhow!("expected response"))?;
            if r["result"]["status"] == expected_status {
                got = r["result"].clone();
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(got["status"], expected_status);
        if mode == "tool_err" {
            assert_eq!(got["result"]["isError"], true);
        } else {
            assert_eq!(got["error"]["code"], -32603);
        }
    }

    // 9. Cancel a working task -> status becomes "cancelled".
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "rag_ingest",
                "arguments": { "bucket": "docs", "title": "cancel_me", "path": "/tmp/c.pdf" },
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    let cancel_tid = resp["result"]["taskId"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing taskId"))?
        .to_string();
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tasks/cancel",
            "params": {
                "taskId": cancel_tid,
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", false);
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tasks/get",
            "params": {
                "taskId": cancel_tid,
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_eq!(resp["result"]["status"], "cancelled");

    // 10. Cap of 32 live tasks per principal: 33rd call falls back to synchronous.
    let mut live_ids = Vec::with_capacity(32);
    for i in 0..32 {
        let resp = alice_engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 100 + i,
                "method": "tools/call",
                "params": {
                    "name": "rag_ingest",
                    "arguments": { "bucket": "docs", "title": "hold", "path": "/tmp/h.pdf" },
                    "_meta": modern_meta("2026-07-28", tasks_caps.clone())
                }
            }))
            .await
            .ok_or_else(|| anyhow::anyhow!("expected response"))?;
        assert_eq!(resp["result"]["resultType"], "task", "task #{i}");
        if let Some(tid) = resp["result"]["taskId"].as_str() {
            live_ids.push(tid.to_string());
        }
    }
    assert_eq!(live_ids.len(), 32);

    // 33rd call exceeds cap -> runs synchronously ("instant" finishes without gate).
    let resp = alice_engine
        .handle_message(json!({
            "jsonrpc": "2.0",
            "id": 200,
            "method": "tools/call",
            "params": {
                "name": "rag_ingest",
                "arguments": { "bucket": "docs", "title": "instant", "path": "/tmp/33.pdf" },
                "_meta": modern_meta("2026-07-28", tasks_caps.clone())
            }
        }))
        .await
        .ok_or_else(|| anyhow::anyhow!("expected response"))?;
    assert_modern_envelope(&resp["result"], "complete", false);
    assert_eq!(resp["result"]["structuredContent"]["mode"], "instant");

    // Clean up the 32 live tasks.
    for tid in live_ids {
        let _ = alice_engine
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 300,
                "method": "tasks/cancel",
                "params": {
                    "taskId": tid,
                    "_meta": modern_meta("2026-07-28", tasks_caps.clone())
                }
            }))
            .await;
    }

    Ok(())
}
