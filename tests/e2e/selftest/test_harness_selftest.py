"""Offline self-tests verifying all E2E-M1 harness components without OpenRAG binaries."""

from __future__ import annotations

import configparser
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import tomllib
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import conftest as e2e_conftest
from lib.cli_client import (
    CliBinaryNotFoundError,
    CliExecutionError,
    OpenRagCli,
    resolve_binary,
)
from lib.fixtures import (
    FixtureFactory,
    decode_png_rgb,
    inspect_bbox_blackout,
    make_concept_alias_pairs,
    make_hwpx_bytes,
    make_korean_pii_doc,
    make_obfuscated_variants,
    make_obsidian_vault,
    make_parenthetical_alias_doc,
    make_pdf_bytes,
    make_pipeline_yaml,
    make_png_bytes,
    make_search_space_yaml,
    make_secret_payload,
    make_tagged_corpus_bundle,
    make_tagged_markdown,
    make_webp_bytes,
)
from lib.http_client import OpenRagHttpClient, extract_sse_text, extract_tag_facets
from lib.mcp_client import (
    OPENRAG_MCP_TOOLS,
    McpHttpClient,
    McpJsonRpcError,
    McpStdioClient,
)
from lib.pg_harness import (
    PgEphemeralCluster,
    PgPrincipalContext,
    PgSandboxError,
    build_rls_preamble_sql,
    discover_migrations,
)
from lib.plugin_client import PluginClient
from lib.web_client import (
    APP_ROUTES,
    FORBIDDEN_APP_JARGON,
    STUDIO_ROUTES,
    WebClient,
)

pytestmark = pytest.mark.selftest

_E2E_DIR = Path(__file__).resolve().parent.parent


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    os.chmod(path, 0o755)
    return path


class _DummyMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyItem:
    def __init__(self, nodeid: str, markers: set[str]) -> None:
        self.nodeid = nodeid
        self._markers = markers

    def get_closest_marker(self, name: str) -> _DummyMarker | None:
        return _DummyMarker(name) if name in self._markers else None


@pytest.mark.selftest
def test_pytest_markers_and_tier_filtering() -> None:
    pyproject_data = tomllib.loads((_E2E_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    ini_parser = configparser.ConfigParser()
    ini_parser.read(_E2E_DIR / "pytest.ini", encoding="utf-8")

    expected_names = (
        {"tier1", "tier2", "tier3", "tier4", "selftest"}
        | {f"f{i:02d}" for i in range(1, 35)}
        | {"f36", "f37"}
    )

    py_markers = pyproject_data["tool"]["pytest"]["ini_options"]["markers"]
    py_names = {entry.split(":", 1)[0].strip() for entry in py_markers}

    ini_markers_raw = ini_parser["pytest"]["markers"].strip().splitlines()
    ini_names = {entry.split(":", 1)[0].strip() for entry in ini_markers_raw if entry.strip()}

    assert len(py_markers) == 41
    assert len(ini_markers_raw) == 41
    assert py_names == expected_names
    assert ini_names == expected_names
    assert py_markers == [line.strip() for line in ini_markers_raw]

    items = [
        _DummyItem("t1", {"tier1", "f07"}),
        _DummyItem("t2", {"tier2", "f07"}),
        _DummyItem("t3", {"tier3"}),
        _DummyItem("t4", {"tier4"}),
        _DummyItem("st", {"selftest"}),
    ]

    kept_none, desel_none = e2e_conftest.filter_items_by_tier(items, None)  # type: ignore[arg-type]
    assert [i.nodeid for i in kept_none] == ["t1", "t2", "t3", "t4", "st"]
    assert desel_none == []

    kept_1, desel_1 = e2e_conftest.filter_items_by_tier(items, "1")  # type: ignore[arg-type]
    assert [i.nodeid for i in kept_1] == ["t1"]
    assert [i.nodeid for i in desel_1] == ["t2", "t3", "t4", "st"]

    kept_all, desel_all = e2e_conftest.filter_items_by_tier(items, "all")  # type: ignore[arg-type]
    assert [i.nodeid for i in kept_all] == ["t1", "t2", "t3", "t4"]
    assert [i.nodeid for i in desel_all] == ["st"]


@pytest.mark.selftest
def test_cli_client_resolution_timeout_and_json(tmp_path: Path) -> None:
    target_root = tmp_path / "repo"
    release_bin = _write_executable(
        target_root / "target" / "release" / "openrag",
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps({{'bin': 'release', 'args': sys.argv[1:]}}))\n",
    )
    debug_bin = _write_executable(
        target_root / "target" / "debug" / "openrag",
        f"#!{sys.executable}\nimport json, sys\nprint('INFO booting')\nprint(json.dumps({{'bin': 'debug', 'args': sys.argv[1:]}}))\n",
    )
    explicit_bin = _write_executable(
        tmp_path / "custom" / "openrag",
        f"#!{sys.executable}\nimport json, sys, time\n"
        "if 'sleep' in sys.argv:\n    time.sleep(2.0)\n"
        "if 'fail3' in sys.argv:\n    sys.stderr.write('boom\\n')\n    sys.exit(3)\n"
        "print(json.dumps({'bin': 'explicit', 'args': sys.argv[1:]}))\n",
    )

    assert resolve_binary("openrag", explicit_bin=explicit_bin, target_root=target_root) == explicit_bin.resolve()
    assert resolve_binary("openrag", target_root=target_root) == debug_bin.resolve()

    debug_bin.unlink()
    assert resolve_binary("openrag", target_root=target_root) == release_bin.resolve()

    with pytest.raises(CliBinaryNotFoundError):
        resolve_binary("nonexistent-openrag-bin-xyz", target_root=tmp_path / "empty")

    cli = OpenRagCli(target_root=target_root, explicit_bin=explicit_bin, default_timeout=5.0)
    res, data = cli.run_json(["ingest", "doc.pdf", "--bucket", "hr"])
    assert res.returncode == 0
    assert data == {"bin": "explicit", "args": ["ingest", "doc.pdf", "--bucket", "hr"]}

    # Also test trailing JSON extraction when stdout has preceding log lines
    debug_cli = OpenRagCli(
        target_root=target_root,
        explicit_bin=_write_executable(
            tmp_path / "custom" / "openrag_log",
            f"#!{sys.executable}\nprint('2026-11-22 INFO starting')\nprint('{{\"ok\": true}}')\n",
        ),
    )
    _, parsed = debug_cli.run_json(["index", "status"])
    assert parsed == {"ok": True}

    with pytest.raises(CliExecutionError) as exit_err:
        cli.run(["fail3"], expect_exit=0)
    assert exit_err.value.result.returncode == 3
    assert "boom" in exit_err.value.result.stderr

    with pytest.raises(CliExecutionError) as timeout_err:
        cli.run(["sleep"], timeout=0.2)
    assert "timed out" in str(timeout_err.value).lower()

    # Verify typed subcommand argument formatting
    _, redact_payload = cli.run_json(
        [
            "redact",
            "run",
            "--scope",
            "org",
            "--reason",
            "audit",
            "--doc",
            "d1",
            "--redact-thumbnails",
            "--hard-purge",
        ]
    )
    assert redact_payload["args"][0:2] == ["redact", "run"]
    res_redact = cli.redact_run(doc_id="d1", scope="org", hard_purge=True, reason="audit")
    assert res_redact.json()["args"] == redact_payload["args"]

    # Verify Follow-up 4 concept & tagged ingest CLI helpers
    res_link = cli.concept_link(
        "llmwiki",
        "세컨드 브레인",
        canonical_name="llmwiki",
        lang="ko",
        relation="synonym",
    )
    assert res_link.json()["args"] == [
        "concept",
        "link",
        "llmwiki",
        "세컨드 브레인",
        "--canonical",
        "llmwiki",
        "--lang",
        "ko",
        "--relation",
        "synonym",
    ]
    res_unlink = cli.concept_unlink("llmwiki", "세컨드 브레인", concept_id="c-1")
    assert res_unlink.json()["args"] == [
        "concept",
        "unlink",
        "llmwiki",
        "세컨드 브레인",
        "--concept-id",
        "c-1",
    ]
    res_clist = cli.concept_list(query="llmwiki", lang="ko", json_output=True)
    assert res_clist.json()["args"] == [
        "concept",
        "list",
        "--query",
        "llmwiki",
        "--lang",
        "ko",
        "--json",
    ]
    res_ingest_tags = cli.ingest(
        "doc.md",
        bucket="kb",
        scope="dept",
        tags=["llmwiki", "policy-2026"],
    )
    assert res_ingest_tags.json()["args"] == [
        "ingest",
        "doc.md",
        "--bucket",
        "kb",
        "--scope",
        "dept",
        "--tags",
        "llmwiki,policy-2026",
    ]


class _ThreadedHttpServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _start_test_http_server(
    handler_cls: type[http.server.BaseHTTPRequestHandler],
) -> tuple[_ThreadedHttpServer, str]:
    server = _ThreadedHttpServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


@pytest.mark.selftest
def test_http_client_rest_multipart_and_sse() -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/forbidden":
                body = json.dumps({"error": "arn_denied"}).encode("utf-8")
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/v1/concepts"):
                body = json.dumps(
                    {
                        "concepts": [
                            {"id": "c1", "canonical_name": "apple", "aliases": ["apple", "사과"]}
                        ]
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_DELETE(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            if self.path == "/v2/buckets/b1/documents/d1/tags":
                payload = json.loads(raw.decode("utf-8"))
                out = json.dumps({"doc_id": "d1", "removed": payload["tags"], "tags": ["llmwiki"]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if self.path == "/v1/concepts/link":
                payload = json.loads(raw.decode("utf-8"))
                out = json.dumps({"concept_id": "c1", "linked": payload}, ensure_ascii=False).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self.path == "/v2/buckets/b1/documents/d1/tags":
                payload = json.loads(raw.decode("utf-8"))
                out = json.dumps({"doc_id": "d1", "tags": payload["tags"]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self.path == "/v2/buckets/b1/search":
                payload = json.loads(raw.decode("utf-8"))
                out = json.dumps(
                    {
                        "echo": payload,
                        "results": [{"doc_id": "d1", "tags": ["llmwiki"]}],
                        "tag_facets": {"llmwiki": 2, "policy-2026": 1},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self.path == "/v2/buckets":
                payload = json.loads(raw.decode("utf-8"))
                out = json.dumps(
                    {
                        "id": "b1",
                        "slug": payload["slug"],
                        "auth": self.headers.get("Authorization"),
                    }
                ).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self.path == "/v2/buckets/b1/documents":
                ctype = self.headers.get("Content-Type", "")
                out = json.dumps(
                    {
                        "content_type": ctype,
                        "has_filename": 'filename="doc.pdf"' in raw.decode("latin-1"),
                        "has_scope": "dept" in raw.decode("utf-8", errors="ignore"),
                        "size": len(raw),
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self.path == "/v1/chat/completions":
                sse_body = (
                    'data: {"id":"1","choices":[{"delta":{"content":"안녕"}}]}\n\n'
                    'data: {"id":"1","choices":[{"delta":{"content":"하세요"}}]}\n\n'
                    "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse_body)))
                self.end_headers()
                self.wfile.write(sse_body)
                return
            self.send_response(404)
            self.end_headers()

    server, base_url = _start_test_http_server(Handler)
    try:
        client = OpenRagHttpClient(base_url, api_key="tok-123")
        assert client.health().assert_status(200).json() == {"status": "ok"}

        created = client.create_bucket({"slug": "hr"}).assert_status(201).json()
        assert created == {"id": "b1", "slug": "hr", "auth": "Bearer tok-123"}

        denied = client.request("GET", "/forbidden")
        assert denied.status == 403
        assert denied.assert_status(403).json()["error"] == "arn_denied"

        up = client.upload_document(
            "b1",
            filename="doc.pdf",
            content=make_pdf_bytes(["test"]),
            content_type="application/pdf",
            fields={"access_scope": "dept"},
        ).assert_status(200).json()
        assert up["content_type"].startswith("multipart/form-data; boundary=")
        assert up["has_filename"] is True
        assert up["has_scope"] is True

        status, _, events = client.chat_completions_stream(
            {"model": "openrag-default", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert status == 200
        assert len(events) == 3
        assert events[-1].data == "[DONE]"
        assert events[-1].json() is None
        assert extract_sse_text(events) == "안녕하세요"

        linked = client.link_concepts({"source": "apple", "target": "사과"}).assert_status(201).json()
        assert linked["concept_id"] == "c1"
        listed_concepts = client.list_concepts(query="apple", lang="ko").assert_status(200).json()
        assert listed_concepts["concepts"][0]["canonical_name"] == "apple"

        added_tags = client.add_document_tags("b1", "d1", ["llmwiki", "policy-2026"]).assert_status(200).json()
        assert added_tags["tags"] == ["llmwiki", "policy-2026"]
        rem_tags = client.remove_document_tags("b1", "d1", ["policy-2026"]).assert_status(200).json()
        assert rem_tags["tags"] == ["llmwiki"]

        search_resp = client.search_bucket(
            "b1",
            query="세컨드 브레인",
            top_k=5,
            tags_any=["llmwiki"],
            tags_all=["llmwiki"],
            tags_exclude=["draft"],
            expand_tag_aliases=True,
        ).assert_status(200)
        assert search_resp.json()["echo"]["expand_tag_aliases"] is True
        assert extract_tag_facets(search_resp) == {"llmwiki": 2, "policy-2026": 1}
    finally:
        server.shutdown()
        server.server_close()


_INLINE_MCP_SERVER = r"""
import json, sys

mode = sys.argv[1]
buf = sys.stdin.buffer

def write_msg(obj):
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if mode == "content-length":
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    else:
        sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()

def read_msg():
    while True:
        line = buf.readline()
        if not line:
            return None
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith(b"content-length:"):
            n = int(s.split(b":", 1)[1].strip())
            while True:
                h = buf.readline()
                if not h or h in (b"\r\n", b"\n"):
                    break
            return json.loads(buf.read(n).decode("utf-8"))
        return json.loads(s.decode("utf-8"))

while True:
    msg = read_msg()
    if msg is None:
        break
    if "id" not in msg:
        continue
    req_id = msg["id"]
    method = msg.get("method")
    params = msg.get("params") or {}
    if method == "initialize":
        write_msg({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-openrag-mcp", "version": "0.1.0"},
            },
        })
    elif method == "ping":
        write_msg({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "tools/list":
        names = [
            "rag_search", "rag_ask", "rag_ingest", "rag_get_page",
            "wiki_search", "wiki_get", "memory_search", "memory_patch",
            "memory_engage", "memory_reflect", "guard_check",
            "concept_link", "concept_resolve",
        ]
        write_msg({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": [{"name": n, "inputSchema": {"type": "object"}} for n in names]},
        })
    elif method == "tools/call":
        # Emit an unsolicited server notification before the response
        write_msg({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})
        tname = params.get("name")
        targs = params.get("arguments") or {}
        if tname == "guard_check" and "PROJECT-ORION" in targs.get("text", ""):
            write_msg({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": json.dumps({"decision": "block", "error": "guard_blocked"})}],
                    "structuredContent": {"decision": "block", "error": "guard_blocked"},
                },
            })
        else:
            write_msg({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": False,
                    "content": [{"type": "text", "text": json.dumps({"tool": tname, "args": targs})}],
                },
            })
    else:
        write_msg({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })
"""


@pytest.mark.selftest
@pytest.mark.parametrize("framing", ["ndjson", "content-length"])
def test_mcp_client_stdio_ndjson_and_content_length(framing: str) -> None:
    cmd = [sys.executable, "-u", "-c", _INLINE_MCP_SERVER, framing]
    with McpStdioClient(cmd, framing=framing, timeout=5.0) as mcp:  # type: ignore[arg-type]
        init_res = mcp.handshake()
        assert init_res["serverInfo"]["name"] == "mock-openrag-mcp"
        assert mcp.ping() == {}

        tools = mcp.tools_list()
        assert [t["name"] for t in tools] == list(OPENRAG_MCP_TOOLS)
        assert len(tools) == 13

        res = mcp.rag_search("출장 여비", bucket="hr", top_k=5)
        assert res.is_error is False
        assert res.json() == {
            "tool": "rag_search",
            "args": {"query": "출장 여비", "bucket": "hr", "top_k": 5},
        }
        assert len(mcp.notifications) == 1

        tagged_search = mcp.rag_search(
            "세컨드 브레인",
            bucket="kb",
            top_k=5,
            tags_any=["llmwiki"],
            tags_all=["policy-2026"],
            tags_exclude=["draft"],
            expand_tag_aliases=True,
        )
        assert tagged_search.json() == {
            "tool": "rag_search",
            "args": {
                "query": "세컨드 브레인",
                "bucket": "kb",
                "top_k": 5,
                "tags_any": ["llmwiki"],
                "tags_all": ["policy-2026"],
                "tags_exclude": ["draft"],
                "expand_tag_aliases": True,
            },
        }

        clink = mcp.concept_link("llmwiki", "세컨드 브레인", canonical_name="llmwiki", lang="ko")
        assert clink.is_error is False
        assert clink.json() == {
            "tool": "concept_link",
            "args": {
                "source": "llmwiki",
                "target": "세컨드 브레인",
                "canonical_name": "llmwiki",
                "lang": "ko",
            },
        }

        cresolve = mcp.concept_resolve(
            "세컨드 브레인", include_aliases=True, use_semantic=False, top_k=5
        )
        assert cresolve.is_error is False
        assert cresolve.json() == {
            "tool": "concept_resolve",
            "args": {
                "term": "세컨드 브레인",
                "include_aliases": True,
                "use_semantic": False,
                "top_k": 5,
            },
        }

        engage_res = mcp.memory_engage("출장 규정", scope="dept", top_k=3, backend="kumiho")
        assert engage_res.is_error is False
        assert engage_res.json() == {
            "tool": "memory_engage",
            "args": {"query": "출장 규정", "scope": "dept", "top_k": 3, "backend": "kumiho"},
        }

        reflect_res = mcp.memory_reflect(
            "2026년 개정 출장비 반영",
            episode_id="ep-1",
            scope="dept",
            edge_type="supersedes",
            target_memory_id="mem-0",
            backend="openrag_pg",
        )
        assert reflect_res.is_error is False
        assert reflect_res.json() == {
            "tool": "memory_reflect",
            "args": {
                "content": "2026년 개정 출장비 반영",
                "episode_id": "ep-1",
                "scope": "dept",
                "edge_type": "supersedes",
                "target_memory_id": "mem-0",
                "backend": "openrag_pg",
            },
        }

        blocked = mcp.guard_check("PROJECT-ORION 보고서", surface="mcp_egress")
        assert blocked.is_error is True
        assert blocked.json()["decision"] == "block"

        with pytest.raises(McpJsonRpcError) as rpc_err:
            mcp.raw_call("unknown/method", {})
        assert rpc_err.value.code == -32601


@pytest.mark.selftest
def test_mcp_client_http_json_and_sse() -> None:
    class McpHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/mcp":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            msg = json.loads(self.rfile.read(length).decode("utf-8"))
            method = msg.get("method")
            req_id = msg.get("id")

            if req_id is None:
                self.send_response(202)
                self.send_header("Mcp-Session-Id", "sess-e2e-42")
                self.end_headers()
                return

            if method == "initialize":
                body = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "mock-http-mcp", "version": "0.1.0"},
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Mcp-Session-Id", "sess-e2e-42")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            # Return SSE response verifying `Mcp-Session-Id` propagation
            seen_sid = self.headers.get("Mcp-Session-Id")
            if method == "tools/list":
                rpc_out = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": [{"name": "rag_search"}, {"name": "guard_check"}]},
                }
            else:
                args = (msg.get("params") or {}).get("arguments") or {}
                rpc_out = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": False,
                        "content": [{"type": "text", "text": json.dumps({"sid": seen_sid, "args": args})}],
                    },
                }

            sse_payload = f"event: message\ndata: {json.dumps(rpc_out)}\n\n".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse_payload)))
            self.end_headers()
            self.wfile.write(sse_payload)

    server, base_url = _start_test_http_server(McpHandler)
    try:
        mcp = McpHttpClient(OpenRagHttpClient(base_url))
        init = mcp.handshake()
        assert init["serverInfo"]["name"] == "mock-http-mcp"
        assert mcp.session_id == "sess-e2e-42"

        tools = mcp.tools_list()
        assert [t["name"] for t in tools] == ["rag_search", "guard_check"]

        res = mcp.rag_search("예산안", top_k=3)
        assert res.json() == {"sid": "sess-e2e-42", "args": {"query": "예산안", "top_k": 3}}
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.selftest
def test_pg_harness_principal_context_and_gucs() -> None:
    ctx = PgPrincipalContext(
        org_id="11111111-1111-1111-1111-111111111111",
        principal_id="22222222-2222-2222-2222-222222222222",
        auth_level=6,
        dept_ids=("33333333-3333-3333-3333-333333333333",),
        dept_admin_ids=("44444444-4444-4444-4444-444444444444",),
        group_ids=(
            "55555555-5555-5555-5555-555555555555",
            "66666666-6666-6666-6666-666666666666",
        ),
    )
    gucs = ctx.to_guc_map()
    assert set(gucs.keys()) == {
        "app.org_id",
        "app.principal_id",
        "app.auth_level",
        "app.dept_ids",
        "app.dept_admin_ids",
        "app.group_ids",
        "app.bypass_rls",
    }
    assert gucs["app.auth_level"] == "6"
    assert gucs["app.group_ids"] == (
        "55555555-5555-5555-5555-555555555555,66666666-6666-6666-6666-666666666666"
    )
    assert gucs["app.bypass_rls"] == "false"

    ctx_pg_arr = PgPrincipalContext(
        org_id=ctx.org_id,
        principal_id=ctx.principal_id,
        group_ids=ctx.group_ids,
        uuid_list_style="pg_array",
    )
    assert ctx_pg_arr.to_guc_map()["app.group_ids"] == (
        "{55555555-5555-5555-5555-555555555555,66666666-6666-6666-6666-666666666666}"
    )

    preamble = build_rls_preamble_sql(ctx)
    assert preamble.startswith("SET LOCAL ROLE openrag_app;")
    assert "set_config('app.bypass_rls', 'false', true)" in preamble
    assert "set_config('app.auth_level', '6', true)" in preamble

    with pytest.raises(ValueError):
        PgPrincipalContext(org_id=ctx.org_id, principal_id=ctx.principal_id, auth_level=0).to_guc_map()
    with pytest.raises(ValueError):
        PgPrincipalContext(org_id=ctx.org_id, principal_id=ctx.principal_id, auth_level=11).to_guc_map()
    with pytest.raises(ValueError):
        PgPrincipalContext(org_id=ctx.org_id, principal_id=ctx.principal_id, bypass_rls=True).to_guc_map()

    with pytest.raises(PgSandboxError):
        PgEphemeralCluster._check_sandbox_error(
            "FATAL: could not create shared memory segment: Operation not permitted (shmget)",
            "initdb",
        )


@pytest.mark.selftest
def test_web_client_zero_jargon_scanner(tmp_path: Path) -> None:
    assert len(APP_ROUTES) == 6 and "/app/connect" in APP_ROUTES
    assert len(STUDIO_ROUTES) == 6 and "/studio/iam" in STUDIO_ROUTES
    assert len(FORBIDDEN_APP_JARGON) == 8

    clean_html = """
    <html>
      <head><script>const vector = "internal"; const bm25 = 1;</script></head>
      <style>.outbox { color: red; }</style>
      <body>
        <h1>내 문서와 기억에게 질문하기</h1>
        <input placeholder="질문을 입력하세요" aria-label="문서 검색" />
        <p>부서 위키와 학습한 내용을 바탕으로 답변합니다. warning과 barn은 무관합니다.</p>
      </body>
    </html>
    """
    visible_clean = WebClient.extract_visible_html_text(clean_html)
    assert "internal" not in visible_clean
    assert WebClient.find_jargon_violations(visible_clean) == []

    leaky_html = """
    <html><body>
      <div>Search mode: BM25 + HNSW Vector over pg_pinecone RaBitQ</div>
      <input placeholder="Enter ARN or Outbox Quarantine ID" />
    </body></html>
    """
    found = [w.lower() for w in WebClient.find_jargon_violations(WebClient.extract_visible_html_text(leaky_html))]
    for expected in ("bm25", "hnsw", "vector", "pg_pinecone", "rabitq", "arn", "outbox", "quarantine"):
        assert expected in found

    # Static TSX scan on synthetic Next.js `/app` tree
    web_root = tmp_path / "web"
    web_root.mkdir(parents=True, exist_ok=True)
    (web_root / "package.json").write_text('{"name":"openrag-web"}', encoding="utf-8")
    app_dir = web_root / "src" / "app" / "app"
    (app_dir / "chat").mkdir(parents=True)
    (app_dir / "upload").mkdir(parents=True)

    (app_dir / "chat" / "page.tsx").write_text(
        'import { fetchApp } from "@/lib/api"; // vector bm25 comment ignored\n'
        "export default function ChatPage() {\n"
        '  return <main><input placeholder="무엇이든 물어보세요" /><p>근거 문서 보기</p></main>;\n'
        "}\n",
        encoding="utf-8",
    )
    (app_dir / "upload" / "page.tsx").write_text(
        "export default function UploadPage() {\n"
        '  return <main><input placeholder="ARN 입력" /><span>Outbox 상태</span></main>;\n'
        "}\n",
        encoding="utf-8",
    )

    wc = WebClient(tmp_path)
    scan_res = wc.scan_app_tsx_copy()
    assert "src/app/app/chat/page.tsx" not in scan_res
    assert [w.lower() for w in scan_res["src/app/app/upload/page.tsx"]] == ["arn", "outbox"]

    # Regression coverage for Challenger Probe 5 & Reviewer JSX interpolation scenarios
    probe_root = tmp_path / "probe_web"
    (probe_root / "web" / "src" / "app" / "app").mkdir(parents=True)
    (probe_root / "web" / "package.json").write_text('{"name":"openrag-probe"}', encoding="utf-8")
    probe_app = probe_root / "web" / "src" / "app" / "app"

    # 1. Arrow function + fetch('/api/studio/index/outbox') + import './vectorHelper' + TS generics -> ZERO false positives
    (probe_app / "safe_baseline.tsx").write_text(
        "import { fetchApi } from './vectorHelper';\n"
        "type Fn = <T>(x: T) => T;\n"
        "export default function SafePage() {\n"
        "  const [items, setItems] = useState<Array<string>>([]);\n"
        "  const load = () => fetch('/api/studio/index/outbox');\n"
        "  const check = (status: string) => status === 'quarantine' ? '대기' : '완료';\n"
        "  return (\n"
        "    <main>\n"
        "      {/* vector bm25 outbox comment ignored */}\n"
        "      <h1>문서 검색</h1>\n"
        "      <p>부서 위키</p>\n"
        "      <p>{status === 'outbox' ? '대기 중' : '완료'}</p>\n"
        "      <button onClick={load}>불러오기</button>\n"
        "    </main>\n"
        "  );\n"
        "}\n",
        encoding="utf-8",
    )

    # 2. Ternary > / < comparisons + arrow && JSX -> ZERO false positives
    (probe_app / "adversarial_fp.tsx").write_text(
        "export default function AdvFp({ count }: { count: number }) {\n"
        "  const x = count > 0 ? fetch('/api/studio/index/outbox') : count < 10;\n"
        "  const renderCard = () => fetch('/api/studio/index/quarantine') && (<div>문서 검색</div>);\n"
        "  const re = /http:\\/\\/vector.internal/;\n"
        "  return <>{renderCard()}</>;\n"
        "}\n",
        encoding="utf-8",
    )

    # 3. http:// URL in JSX text must NOT act as // comment stripping later jargon
    (probe_app / "adversarial_fn_url.tsx").write_text(
        "export default function UrlPage() {\n"
        "  return <main><p>연결 주소 http://localhost:8080/mcp 및 Vector Search 설정</p></main>;\n"
        "}\n",
        encoding="utf-8",
    )

    # 4. JSX {expr} interpolation must NOT hide adjacent visible text on either side
    (probe_app / "adversarial_fn_interp.tsx").write_text(
        "export default function InterpPage({ score }: { score: number }) {\n"
        "  return <main><p>점수 ({score}): BM25 + Vector 검색</p></main>;\n"
        "}\n",
        encoding="utf-8",
    )

    # 5. JSX expression string literals + user-facing props (including text=, template prop, etc.)
    (probe_app / "adversarial_fn_expr.tsx").write_text(
        "export default function ExprPage({ ready }: { ready: boolean }) {\n"
        "  return (\n"
        "    <main>\n"
        '      <span title="ok">{ready ? "Quarantine" : "HNSW"}</span>\n'
        '      <Badge text="RaBitQ" tooltip={`ARN 상태`} heading={"Outbox 목록"} />\n'
        "    </main>\n"
        "  );\n"
        "}\n",
        encoding="utf-8",
    )

    probe_scan = WebClient(probe_root).scan_app_tsx_copy()
    assert "src/app/app/safe_baseline.tsx" not in probe_scan
    assert "src/app/app/adversarial_fp.tsx" not in probe_scan
    assert probe_scan["src/app/app/adversarial_fn_url.tsx"] == ["Vector"]
    assert probe_scan["src/app/app/adversarial_fn_interp.tsx"] == ["BM25", "Vector"]
    assert probe_scan["src/app/app/adversarial_fn_expr.tsx"] == [
        "Quarantine",
        "HNSW",
        "RaBitQ",
        "ARN",
        "Outbox",
    ]

    connect_html = """
    <section>
      <h2>Claude Desktop</h2>
      <code>openrag mcp --stdio</code>
      <h2>Claude Code</h2>
      <h2>Cursor</h2>
      <h2>Codex</h2>
      <code>http://127.0.0.1:8080/mcp</code>
    </section>
    """
    snippets = wc.verify_connect_mcp_snippets(connect_html)
    assert all(snippets.values()), snippets


@pytest.mark.selftest
def test_fixtures_generators_and_bbox_pixel_check(tmp_path: Path) -> None:
    factory = FixtureFactory(tmp_path)

    # 1. PDF
    pdf = factory.pdf_bytes(["Page One", "Page Two"])
    assert pdf.startswith(b"%PDF-1.4")
    assert b"xref" in pdf
    assert b"%%EOF" in pdf
    assert b"/Count 2" in pdf
    if shutil.which("pdfinfo"):
        pdf_path = factory.pdf_file("test.pdf", ["Page One", "Page Two"])
        info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True)
        assert re.search(r"Pages:\s+2\b", info.stdout)

    # 2. HWPX
    hwpx = make_hwpx_bytes(
        title="보안 규정",
        paragraphs=["제1조 목적", "제2조 적용범위"],
        tables=[[["항목", "기준"], ["일비", "70,000원"]]],
    )
    with zipfile.ZipFile(factory.write_bytes("sample.hwpx", hwpx), "r") as zf:
        mimetype_info = zf.getinfo("mimetype")
        assert mimetype_info.compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"application/hwp+zip"
        sec_xml = zf.read("Contents/section0.xml").decode("utf-8")
        root = ET.fromstring(sec_xml)
        all_text = "".join(root.itertext())
        assert "보안 규정" in all_text
        assert "70,000원" in all_text

    # 3. PNG + pure-Python PNG decode + WebP + BBox blackout check
    bbox = (200, 300, 600, 700)
    white_png = make_png_bytes(100, 100, bg_rgb=(255, 255, 255), boxes=[(bbox, (240, 240, 240))])
    w, h, pixels = decode_png_rgb(white_png)
    assert (w, h) == (100, 100)
    assert len(pixels) == 10000
    assert inspect_bbox_blackout(white_png, bbox).is_blacked_out is False

    black_png = make_png_bytes(100, 100, bg_rgb=(255, 255, 255), boxes=[(bbox, (0, 0, 0))])
    rep_png = inspect_bbox_blackout(black_png, bbox)
    assert rep_png.is_blacked_out is True
    assert rep_png.black_ratio == 1.0

    white_webp = make_webp_bytes(100, 100, bg_rgb=(255, 255, 255), boxes=[(bbox, (200, 200, 200))])
    assert white_webp[:4] == b"RIFF" and white_webp[8:12] == b"WEBP"
    assert inspect_bbox_blackout(white_webp, bbox).is_blacked_out is False

    black_webp = make_webp_bytes(100, 100, bg_rgb=(255, 255, 255), boxes=[(bbox, (0, 0, 0))])
    rep_webp = inspect_bbox_blackout(black_webp, bbox)
    assert rep_webp.is_blacked_out is True
    assert rep_webp.black_ratio >= 0.99

    # 4. Korean PII + Secret payload
    pii_doc = factory.korean_pii_doc()
    assert "900101-1234567" in pii_doc["text"]
    assert "010-1234-5678" in pii_doc["text"]
    assert "PROJECT-ORION" in pii_doc["text"]

    secrets = make_secret_payload()
    assert "sk-test-openrag-" in secrets["transcript"]
    assert "AKIAIOSFODNN7EXAMPLE" in secrets["transcript"]
    assert "/Users/dev/secret/.env" in secrets["transcript"]

    # 5. Obsidian vault + YAML configs
    vault = factory.obsidian_vault()
    assert (vault / "index.md").is_file()
    assert "[[travel-policy-2026]]" in (vault / "index.md").read_text(encoding="utf-8")
    assert (vault / "travel-policy-2025.md").is_file()
    assert (vault / "travel-policy-2026.md").is_file()
    assert (vault / "security-guidelines.md").is_file()

    pipe_cfg = yaml.safe_load(make_pipeline_yaml(profile="airgap", engine="pgvector"))
    assert pipe_cfg["profile"] == "airgap"
    assert pipe_cfg["store"]["pg_pinecone_engine"] == "pgvector"
    assert pipe_cfg["pii"]["guard"]["enabled"] is True
    assert pipe_cfg["memory"]["backend"] == "openrag_pg"

    kumiho_cfg = yaml.safe_load(
        make_pipeline_yaml(memory_backend="kumiho", memory_mirror_to="kumiho")
    )
    assert kumiho_cfg["memory"] == {"backend": "kumiho", "mirror_to": "kumiho"}

    ss_cfg = yaml.safe_load(make_search_space_yaml())
    assert ss_cfg["strategy"] == "greedy_node"
    assert ss_cfg["security_gates"]["require_zero_pii_leak"] is True


@pytest.mark.selftest
def test_plugin_client_inspection_and_invocation(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"

    # P1 kordoc-mcp-bridge
    p1 = plugins_dir / "kordoc-mcp-bridge"
    p1.mkdir(parents=True)
    _write_executable(
        p1 / "server.py",
        f"#!{sys.executable}\nimport json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if 'id' in msg:\n"
        "        sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'protocolVersion':'2025-03-26','serverInfo':{'name':'kordoc-mock','version':'1'}}})+'\\n')\n"
        "        sys.stdout.flush()\n",
    )
    (p1 / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"kordoc": {"command": sys.executable, "args": ["-u", "server.py"]}}}),
        encoding="utf-8",
    )

    # P2 docufinder-connector
    p2 = plugins_dir / "docufinder-connector"
    p2.mkdir(parents=True)
    (p2 / "anything.config.json").write_text(
        json.dumps({"endpoint": "http://127.0.0.1:8080/v1/chat/completions", "model": "openrag"}),
        encoding="utf-8",
    )
    (p2 / "sync.py").write_text(
        "import json, os, sys\nprint(json.dumps({'argv': sys.argv[1:], 'bucket': os.environ.get('OPENRAG_BUCKET')}))\n",
        encoding="utf-8",
    )

    # P3 ai-memory-hooks
    p3 = plugins_dir / "ai-memory-hooks"
    (p3 / "hooks").mkdir(parents=True)
    (p3 / "scripts").mkdir(parents=True)
    (p3 / "scripts" / "hook.py").write_text(
        "import json, os, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "print(json.dumps({'event': os.environ['OPENRAG_HOOK_EVENT'], 'dry_run': os.environ.get('OPENRAG_HOOK_DRY_RUN'), 'keys': sorted(payload.keys())}))\n",
        encoding="utf-8",
    )
    (p3 / "hooks" / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"hooks": [{"type": "command", "command": f"{sys.executable} scripts/hook.py"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    pc = PluginClient(plugins_dir)
    assert ".mcp.json" in pc.inspect_kordoc_bridge()
    with pc.invoke_kordoc_mcp(api_url="http://127.0.0.1:8080", api_key="k1") as k_mcp:
        assert k_mcp.initialize()["serverInfo"]["name"] == "kordoc-mock"

    assert pc.get_docufinder_anything_config()["endpoint"].endswith("/v1/chat/completions")
    sync_res = pc.run_docufinder_sync(
        tmp_path, bucket="docs", api_url="http://127.0.0.1:8080", api_key="k1"
    )
    assert sync_res.json()["bucket"] == "docs"

    hook_res = pc.invoke_memory_hook(
        "UserPromptSubmit",
        {"prompt": "hello"},
        api_url="http://127.0.0.1:8080",
        api_key="k1",
        dry_run=True,
    )
    assert hook_res.json() == {"event": "UserPromptSubmit", "dry_run": "1", "keys": ["prompt"]}


@pytest.mark.selftest
def test_concepts_tags_fixtures_and_dynamic_migration_discovery(tmp_path: Path) -> None:
    import unicodedata

    # 1. Dynamic migration discovery: 000-007 vs 000-008 vs missing required prefixes
    mdir_8 = tmp_path / "mig8"
    mdir_8.mkdir()
    for i in range(8):
        (mdir_8 / f"{i:03d}_step.sql").write_text(f"-- {i}\n", encoding="utf-8")

    found_8 = discover_migrations(mdir_8)
    assert [p.name for p in found_8] == [f"{i:03d}_step.sql" for i in range(8)]

    with pytest.raises(FileNotFoundError, match="008"):
        discover_migrations(mdir_8, required_prefixes=[f"{i:03d}" for i in range(9)])

    mdir_9 = tmp_path / "mig9"
    mdir_9.mkdir()
    for i in range(8):
        (mdir_9 / f"{i:03d}_step.sql").write_text(f"-- {i}\n", encoding="utf-8")
    (mdir_9 / "008_concepts_tags.sql").write_text("-- concepts tags\n", encoding="utf-8")

    found_9 = discover_migrations(mdir_9)
    assert len(found_9) == 9
    assert found_9[-1].name == "008_concepts_tags.sql"
    assert discover_migrations(mdir_9, required_prefixes=[f"{i:03d}" for i in range(9)])[-1].name == "008_concepts_tags.sql"

    mdir_missing_7 = tmp_path / "mig_missing_7"
    mdir_missing_7.mkdir()
    for i in range(7):
        (mdir_missing_7 / f"{i:03d}_step.sql").write_text(f"-- {i}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="007"):
        discover_migrations(mdir_missing_7)

    # Also verify real repo migrations directory dynamically tolerates 000-007 OR 000-008
    repo_migrations = _E2E_DIR.parent.parent / "migrations"
    if repo_migrations.is_dir():
        repo_found = discover_migrations(repo_migrations)
        assert len(repo_found) >= 8
        assert repo_found[0].name.startswith("000_")

    # 2. Concept/Alias, Parenthetical HWPX, Obfuscation, and Tag fixtures
    factory = FixtureFactory(tmp_path / "fx")
    pairs = factory.concept_alias_pairs()
    assert any("사과" in g["aliases"] and "apple" in g["aliases"] for g in pairs["cross_lingual"])
    assert any("세컨드 브레인" in g["aliases"] and "llmwiki" in g["aliases"] for g in pairs["domain_synonyms"])

    pdoc = factory.parenthetical_alias_doc()
    assert "사과(apple)" in pdoc["text"]
    assert "세컨드 브레인(LLM Wiki)" in pdoc["text"]
    with zipfile.ZipFile(factory.write_bytes("parenthetical.hwpx", pdoc["hwpx_bytes"]), "r") as zf:
        sec_xml = zf.read("Contents/section0.xml").decode("utf-8")
        assert "사과(apple)" in sec_xml
        assert "세컨드 브레인(LLM Wiki)" in sec_xml

    obf = factory.obfuscated_variants("apple")
    assert unicodedata.normalize("NFKC", obf["fullwidth_nfkc"]) == "apple"
    assert "\u200b" in obf["zero_width"]
    assert obf["punct_spaced"] != "apple"
    assert obf["mixed"] != "apple"

    obf_ko = make_obfuscated_variants("사과")
    assert obf_ko["zero_width"] == "사\u200b과"

    md = factory.tagged_markdown(tags=["llmwiki", "세컨드브레인"])
    assert "#llmwiki" in md and "#세컨드브레인" in md
    parsed_fm = yaml.safe_load(md.split("---", 2)[1])
    assert parsed_fm["tags"] == ["llmwiki", "세컨드브레인"]

    bundle = factory.tagged_corpus_bundle()
    assert len(bundle) == 5
    assert bundle[0]["tags"] == ["hr", "2026", "draft"]

    # 3. extract_tag_facets helper (dict and list shapes)
    assert extract_tag_facets({"tag_facets": {"llmwiki": 3, "hr": 1}}) == {"llmwiki": 3, "hr": 1}
    assert extract_tag_facets(
        {"facets": {"tags": [{"tag": "llmwiki", "count": 2}, {"name": "policy-2026", "count": 4}]}}
    ) == {"llmwiki": 2, "policy-2026": 4}
    assert extract_tag_facets({"results": []}) == {}
