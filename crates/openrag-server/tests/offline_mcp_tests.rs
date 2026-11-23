#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

//! Offline MCP protocol tests: framing, lifecycle and fail-closed auth without a database.

use std::process::Stdio;

use openrag_server::mcp::{serve_stream, McpEngine, StdioOptions, MCP_TOOL_NAMES};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

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
    assert_eq!(count, 13);
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
