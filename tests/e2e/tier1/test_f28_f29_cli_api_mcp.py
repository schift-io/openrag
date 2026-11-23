"""Tier 1 E2E tests for F28-F29: openrag CLI, Axum HTTP API, Local Stdio & HTTP MCP (13 Tools)."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from conftest import fail_missing_target
from lib import (
    CliBinaryNotFoundError,
    FixtureFactory,
    McpHttpClient,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    WebClient,
    extract_sse_text,
)
from lib.mcp_client import OPENRAG_MCP_TOOLS

pytestmark = [pytest.mark.tier1]

EXPECTED_MCP_TOOLS: set[str] = set(OPENRAG_MCP_TOOLS)
assert len(EXPECTED_MCP_TOOLS) == 13


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> None:
    pg.ensure_started()
    out = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;")
    if out.strip() != "t":
        pg.apply_migrations(root / "migrations")


def _seed_org_and_user(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'MCP Org {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ('{dept_id}', '{org_id}', 'Core', 'core');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin)
        VALUES ('{user_id}', '{org_id}', 'u-{user_id[:6]}@example.org', 'User', 7, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
        VALUES ('{org_id}', '{dept_id}', '{user_id}', 'dept_admin');
        """
    )
    return {"org_id": org_id, "slug": slug, "dept_id": dept_id, "user_id": user_id}


def _mint_api_key(api_client: OpenRagHttpClient, ctx: dict[str, str]) -> str:
    resp = api_client.cli_code_exchange(
        {"code": issue_cli_code(ctx["org_id"], ctx["user_id"]), "org_id": ctx["org_id"], "principal_id": ctx["user_id"]}
    )
    resp.assert_status((200, 201))
    key = resp.json().get("api_key") or resp.json().get("token")
    assert isinstance(key, str) and key
    return key


# ============================================================================
# F28 — openrag CLI & Axum HTTP API Server (6 tests)
# ============================================================================


@pytest.mark.f28
def test_f28_cli_help_and_subcommand_surface(openrag_cli: OpenRagCli) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    root_help = openrag_cli.run(["--help"], check=True)
    for sub in ("up", "ingest", "index", "memory", "redact", "wiki", "bench", "pipeline", "mcp"):
        assert sub in root_help.stdout
        sub_help = openrag_cli.run([sub, "--help"], check=True)
        assert sub_help.returncode == 0
        assert len(sub_help.stdout.strip()) > 0

    mcp_help = openrag_cli.run(["mcp", "--help"], check=True)
    assert "--stdio" in mcp_help.stdout


@pytest.mark.f28
def test_f28_buckets_crud_documents_upload_list_delete_and_search(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)
    client = api_client.with_api_key(key)

    slug = f"crud-{uuid.uuid4().hex[:6]}"
    created = client.create_bucket(
        {"slug": slug, "name": "CRUD Bucket", "access_scope": "dept", "min_auth_level": 3}
    ).assert_status((200, 201)).json()
    bucket_id = str(created.get("id") or created.get("bucket_id"))

    assert slug in client.list_buckets().assert_status(200).text
    client.get_bucket(bucket_id).assert_status(200)

    hwpx = sample_factory.hwpx_file("crud.hwpx", title="CRUD 문서", paragraphs=["CRUD 검색 검증용 고유 문구 Alpha987"])
    up = client.upload_document(
        bucket_id,
        filename=hwpx.name,
        content=hwpx.read_bytes(),
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202)).json()
    doc_id = str(up.get("id") or up.get("document_id"))

    listed = client.list_documents(bucket_id).assert_status(200).text
    assert doc_id in listed or "crud.hwpx" in listed

    srch = client.search_bucket(bucket_id, {"query": "Alpha987", "top_k": 5}).assert_status(200)
    assert "Alpha987" in srch.text

    client.delete_document(bucket_id, doc_id).assert_status((200, 204))
    srch_after = client.search_bucket(bucket_id, {"query": "Alpha987", "top_k": 5}).assert_status(200)
    assert doc_id not in srch_after.text


@pytest.mark.f28
def test_f28_chat_completions_openai_json_and_sse_streaming(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)
    client = api_client.with_api_key(key)

    b = client.create_bucket({"slug": f"chat-{uuid.uuid4().hex[:6]}", "name": "Chat", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "근무 시간", "text": "코어 타임은 오전 10시부터 오후 4시까지입니다.", "access_scope": "dept"}).assert_status((200, 201))

    j_resp = client.chat_completions(
        {
            "model": "openrag-auto",
            "bucket_id": bid,
            "include_citations": True,
            "stream": False,
            "messages": [{"role": "user", "content": "코어 타임이 언제인가요?"}],
        }
    ).assert_status(200).json()
    assert j_resp["choices"][0]["message"]["content"]
    assert "citations" in j_resp or "sources" in j_resp

    status, _, events = client.chat_completions_stream(
        {
            "model": "openrag-auto",
            "bucket_id": bid,
            "include_citations": True,
            "messages": [{"role": "user", "content": "코어 타임이 언제인가요?"}],
        }
    )
    assert status == 200
    assert any(ev.data.strip() == "[DONE]" for ev in events)
    assert len(extract_sse_text(events)) > 0


@pytest.mark.f28
def test_f28_chat_completions_docufinder_anything_compatibility(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)
    client = api_client.with_api_key(key)

    payload = {
        "model": "openrag-anything",
        "messages": [{"role": "user", "content": "사내 보안 점검 안내를 요약해줘."}],
    }
    resp = client.chat_completions(payload).assert_status(200).json()
    assert resp.get("object") == "chat.completion"
    assert isinstance(resp.get("choices"), list) and len(resp["choices"]) >= 1
    assert "usage" in resp

    status, _, events = client.chat_completions_stream(payload)
    assert status == 200
    assert any(ev.data.strip() == "[DONE]" for ev in events)


@pytest.mark.f28
def test_f28_cli_code_exchange_mints_scoped_api_key(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    resp = api_client.cli_code_exchange(
        {
            "code": issue_cli_code(ctx["org_id"], ctx["user_id"]),
            "org_id": ctx["org_id"],
            "principal_id": ctx["user_id"],
            "max_auth_level": 6,
        }
    )
    resp.assert_status((200, 201))
    data = resp.json()
    minted = data.get("api_key") or data.get("token")
    assert isinstance(minted, str) and len(minted) >= 8

    api_client.with_api_key(minted).list_buckets().assert_status(200)
    mcp = McpHttpClient(api_client.with_api_key(minted))
    mcp.handshake()
    assert {t["name"] for t in mcp.tools_list()} == EXPECTED_MCP_TOOLS


@pytest.mark.f28
def test_f28_cli_ingest_and_api_search_roundtrip(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)
    client = api_client.with_api_key(key)

    slug = f"cli-rt-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": slug, "name": "CLI RT", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    hwpx = sample_factory.hwpx_file("cli_rt.hwpx", title="CLI 적재 테스트", paragraphs=["CLI적재고유토큰Z991 검증 본문"])
    openrag_cli.ingest(
        hwpx,
        bucket=slug,
        scope="dept",
        auth_level=3,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    srch = client.search_bucket(bid, {"query": "CLI적재고유토큰Z991", "top_k": 5}).assert_status(200)
    assert "CLI적재고유토큰Z991" in srch.text


# ============================================================================
# F29 — Local Stdio & Streamable HTTP MCP Server — 13 Tools + Snippets (7 tests)
# ============================================================================


@pytest.mark.f29
def test_f29_stdio_mcp_handshake_ndjson_and_content_length_13_tools(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)

    for framing, prefer_standalone in (("ndjson", False), ("content-length", True)):
        with mcp_stdio(
            api_key=key,
            principal_id=ctx["user_id"],
            framing=framing,
            prefer_standalone=prefer_standalone,
            env={"DATABASE_URL": pg_ephemeral.dsn()},
        ) as mcp:
            init_res = mcp.handshake()
            assert "capabilities" in init_res or "serverInfo" in init_res
            mcp.ping()
            tools = mcp.tools_list()
            names = {t["name"] for t in tools}
            assert names == EXPECTED_MCP_TOOLS


@pytest.mark.f29
def test_f29_http_mcp_streamable_session_and_13_tools_list(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)

    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()
    tools = mcp.tools_list()
    assert {t["name"] for t in tools} == EXPECTED_MCP_TOOLS


@pytest.mark.f29
def test_f29_mcp_rag_ingest_search_ask_get_page_lifecycle(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)
    slug = f"mcp-rag-{uuid.uuid4().hex[:6]}"
    api_client.with_api_key(key).create_bucket({"slug": slug, "name": "MCP RAG", "access_scope": "dept"}).assert_status((200, 201))

    with mcp_stdio(api_key=key, principal_id=ctx["user_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        ing = mcp.rag_ingest(
            bucket=slug,
            text="MCP 도구로 적재한 출장비 한도는 1일 75,000원이며 담당자 연락처 010-9876-5432 입니다.",
            access_scope="dept",
            min_auth_level=3,
            title="MCP 출장비 안내",
        )
        assert ing.is_error is False
        ing_data = ing.json()
        doc_id = str(ing_data.get("document_id") or ing_data.get("id"))

        srch = mcp.rag_search("출장비 한도 75,000원", bucket=slug, top_k=5)
        assert srch.is_error is False
        assert "75,000" in srch.text

        ask = mcp.rag_ask("출장비 한도는 얼마인가요?", bucket=slug)
        assert ask.is_error is False
        assert "75,000" in ask.text

        page = mcp.rag_get_page(document_id=doc_id, page_no=1)
        assert page.is_error is False
        assert "75,000" in page.text


@pytest.mark.f29
def test_f29_mcp_wiki_search_and_wiki_get_tools(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)
    slug = f"wiki-mcp-{uuid.uuid4().hex[:6]}"
    page_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{page_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['user_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:core:dept:wiki:L3/{page_id}', '{ctx['org_id']}:core', '{slug}', 'MCP 위키 제목', '# MCP 위키 본문\n위키 검색 테스트 내용', 'MCP 위키 검색 테스트 내용', '{hashlib.sha256(page_id.encode()).hexdigest()}');
        """
    )

    with mcp_stdio(api_key=key, principal_id=ctx["user_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        ws = mcp.wiki_search("위키 검색 테스트")
        assert ws.is_error is False
        assert slug in ws.text or "MCP 위키" in ws.text

        wg = mcp.wiki_get(slug=slug)
        assert wg.is_error is False
        assert "위키 검색 테스트 내용" in wg.text


@pytest.mark.f29
def test_f29_mcp_memory_search_patch_engage_reflect_tools(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)

    with mcp_stdio(api_key=key, principal_id=ctx["user_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        ref = mcp.memory_reflect("MCP 메모리 초기 기록 10만원", scope="dept")
        assert ref.is_error is False
        mem_id = str(ref.json().get("memory_id") or ref.json().get("id"))

        eng = mcp.memory_engage("메모리 초기 기록", scope="dept")
        assert eng.is_error is False
        assert "10만원" in eng.text

        patched = mcp.memory_patch(mem_id, "amend", text="MCP 메모리 개정 기록 12만원", reason="상향")
        assert patched.is_error is False

        ms = mcp.memory_search("메모리 개정 기록", scope="dept")
        assert ms.is_error is False
        assert "12만원" in ms.text


@pytest.mark.f29
def test_f29_mcp_guard_check_allow_warn_mask_statuses(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_user(pg_ephemeral)
    key = _mint_api_key(api_client, ctx)

    with mcp_stdio(api_key=key, principal_id=ctx["user_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        r_allow = mcp.guard_check("일반 업무 회의록 공유", surface="mcp_egress")
        assert r_allow.is_error is False
        assert r_allow.json().get("decision") == "allow"

        r_warn = mcp.guard_check("CONFIDENTIAL-DRAFT 검토 요청", surface="mcp_egress")
        assert r_warn.is_error is False
        assert r_warn.json().get("decision") in ("warn", "mask")

        r_mask = mcp.guard_check("연락처 010-1234-5678 확인", surface="mcp_egress")
        assert r_mask.is_error is False
        m_data = r_mask.json()
        assert m_data.get("decision") == "mask"
        assert "010-1234-5678" not in m_data.get("sanitized_text", "")


@pytest.mark.f29
def test_f29_readme_contains_all_four_mcp_client_snippets(
    openrag_root: Path,
    web_client: WebClient,
) -> None:
    readme_path = openrag_root / "README.md"
    if not readme_path.is_file():
        fail_missing_target(f"README.md not found at {readme_path}")
    text = readme_path.read_text(encoding="utf-8")
    checks = web_client.verify_connect_mcp_snippets(text)
    assert all(checks.values()), f"Missing MCP snippet checks in README.md: {checks}"
    for tool_name in EXPECTED_MCP_TOOLS:
        assert tool_name in text, f"README.md missing documentation for MCP tool {tool_name}"
