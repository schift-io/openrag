"""Tier 2 BVA & negative tests for F28-F29: CLI, HTTP API, Local Stdio & HTTP MCP."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from conftest import fail_missing_target
from lib import (
    CliBinaryNotFoundError,
    FixtureFactory,
    McpHttpClient,
    McpJsonRpcError,
    McpProtocolError,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
)

pytestmark = [pytest.mark.tier2]


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> None:
    pg.ensure_started()
    out = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;")
    if out.strip() != "t":
        pg.apply_migrations(root / "migrations")


def _seed_org_and_users(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept1_id = str(uuid.uuid4())
    dept2_id = str(uuid.uuid4())
    l2_id = str(uuid.uuid4())
    l9_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    denied_group_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'BVA MCP {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
            ('{dept1_id}', '{org_id}', 'Dept1', 'dept1'),
            ('{dept2_id}', '{org_id}', 'Dept2', 'dept2');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{l2_id}', '{org_id}', 'l2-{l2_id[:6]}@example.org', 'L2Peer', 2, false, false),
            ('{l9_id}', '{org_id}', 'l9-{l9_id[:6]}@example.org', 'L9User', 9, true, false),
            ('{admin_id}', '{org_id}', 'admin-{admin_id[:6]}@example.org', 'Admin', 10, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept1_id}', '{l2_id}', 'member'),
            ('{org_id}', '{dept1_id}', '{l9_id}', 'member'),
            ('{org_id}', '{dept2_id}', '{admin_id}', 'dept_admin');
        INSERT INTO iam.groups (id, org_id, dept_id, name) VALUES ('{denied_group_id}', '{org_id}', NULL, 'deny-{denied_group_id[:8]}');
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES ('{org_id}', '{denied_group_id}', '{l2_id}');
        """
    )
    return {
        "org_id": org_id,
        "slug": slug,
        "dept1_id": dept1_id,
        "dept2_id": dept2_id,
        "l2_id": l2_id,
        "l9_id": l9_id,
        "admin_id": admin_id,
        "denied_group_id": denied_group_id,
    }


def _mint_api_key(
    api_client: OpenRagHttpClient,
    ctx: dict[str, str],
    principal_id: str,
    *,
    max_auth_level: int | None = None,
    scoped_arns: list[str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "code": issue_cli_code(ctx["org_id"], principal_id),
        "org_id": ctx["org_id"],
        "principal_id": principal_id,
    }
    if max_auth_level is not None:
        payload["max_auth_level"] = max_auth_level
    if scoped_arns is not None:
        payload["scoped_arns"] = scoped_arns
    resp = api_client.cli_code_exchange(payload)
    resp.assert_status((200, 201))
    key = resp.json().get("api_key") or resp.json().get("token")
    assert isinstance(key, str) and key
    return key


# ============================================================================
# F28 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f28
def test_f28_bva_unauthenticated_and_revoked_api_key_rejected_401(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    revoked_key = _mint_api_key(api_client, ctx, ctx["l2_id"])
    pg_ephemeral.execute_sql(
        f"UPDATE iam.api_keys SET revoked_at = now() WHERE org_id = '{ctx['org_id']}' AND principal_id = '{ctx['l2_id']}';"
    )

    for client in (
        api_client,
        api_client.with_api_key("orag_malformed_invalid_token"),
        api_client.with_api_key(revoked_key),
    ):
        assert client.list_buckets().status == 401
        assert client.chat_completions({"model": "openrag-auto", "messages": [{"role": "user", "content": "hi"}]}).status == 401
        assert client.ab_search({"bucket_id": str(uuid.uuid4()), "queries": [], "variants": {}}).status == 401


@pytest.mark.f28
def test_f28_bva_api_key_scoped_arns_and_max_auth_level_ceiling(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    l9_full = _mint_api_key(api_client, ctx, ctx["l9_id"])

    b1 = api_client.with_api_key(l9_full).create_bucket(
        {"slug": f"d1-{uuid.uuid4().hex[:6]}", "name": "D1", "access_scope": "dept", "min_auth_level": 1}
    ).assert_status((200, 201)).json()
    bid1 = str(b1.get("id") or b1.get("bucket_id"))
    api_client.with_api_key(l9_full).create_document(
        bid1, {"title": "L6 Doc", "text": "L6_CEILING_SECRET", "access_scope": "dept", "min_auth_level": 6}
    ).assert_status((200, 201))

    b2 = api_client.with_api_key(admin_key).create_bucket(
        {"slug": f"d2-{uuid.uuid4().hex[:6]}", "name": "D2", "access_scope": "org", "min_auth_level": 1}
    ).assert_status((200, 201)).json()
    bid2 = str(b2.get("id") or b2.get("bucket_id"))
    api_client.with_api_key(admin_key).create_document(
        bid2, {"title": "Dept2 Doc", "text": "DEPT2_SCOPE_SECRET", "access_scope": "org", "min_auth_level": 1}
    ).assert_status((200, 201))

    scoped_key = _mint_api_key(
        api_client,
        ctx,
        ctx["l9_id"],
        max_auth_level=4,
        scoped_arns=[f"arn:openrag:{ctx['slug']}:dept1:*"],
    )
    s1 = api_client.with_api_key(scoped_key).search_bucket(bid1, {"query": "L6_CEILING_SECRET", "top_k": 5})
    assert s1.status == 403 or "L6_CEILING_SECRET" not in json.dumps(s1.json().get("results", []), ensure_ascii=False)

    s2 = api_client.with_api_key(scoped_key).search_bucket(bid2, {"query": "DEPT2_SCOPE_SECRET", "top_k": 5})
    assert s2.status == 403 or "DEPT2_SCOPE_SECRET" not in json.dumps(s2.json().get("results", []), ensure_ascii=False)


@pytest.mark.f28
def test_f28_bva_chat_completions_inbound_query_guard_blocks_safeword(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    resp = client.chat_completions(
        {"model": "openrag-auto", "messages": [{"role": "user", "content": "PROJECT-ORION 보고서 요약"}]}
    )
    assert resp.status == 403
    assert "guard_blocked" in resp.text.lower()


@pytest.mark.f28
def test_f28_bva_malformed_json_and_empty_messages_returns_400_422(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)

    r_bad_json = client.request(
        "POST", "/v1/chat/completions", data=b"{not-valid-json", headers={"Content-Type": "application/json"}
    )
    assert r_bad_json.status in (400, 422)

    r_empty_msgs = client.chat_completions({"model": "openrag-auto", "messages": []})
    assert r_empty_msgs.status in (400, 422)

    r_empty_slug = client.create_bucket({"slug": "", "name": "Empty"})
    assert r_empty_slug.status in (400, 422)


@pytest.mark.f28
def test_f28_bva_cli_unknown_subcommand_and_invalid_auth_level_0_11(
    openrag_cli: OpenRagCli,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    assert openrag_cli.run(["unknown-subcommand-xyz"]).returncode != 0
    doc = sample_factory.write_text("d.txt", "hello")
    assert openrag_cli.ingest(doc, bucket="b", auth_level=0).returncode != 0
    assert openrag_cli.ingest(doc, bucket="b", auth_level=11).returncode != 0


@pytest.mark.f28
def test_f28_bva_cli_code_exchange_invalid_or_expired_code_rejected_400_401(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    resp = api_client.cli_code_exchange({"code": "invalid-expired-code-999999"})
    assert resp.status in (400, 401, 403, 404)


# ============================================================================
# F29 BVA / Negative (7 tests)
# ============================================================================


@pytest.mark.f29
def test_f29_bva_mcp_unknown_method_returns_jsonrpc_32601(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])

    with mcp_stdio(api_key=key, principal_id=ctx["admin_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        with pytest.raises(McpJsonRpcError) as exc_info:
            mcp.raw_call("unknown/method", {})
        assert exc_info.value.code == -32601

    http_mcp = McpHttpClient(api_client.with_api_key(key))
    http_mcp.handshake()
    with pytest.raises(McpJsonRpcError) as h_exc:
        http_mcp.raw_call("unknown/method", {})
    assert h_exc.value.code == -32601


@pytest.mark.f29
def test_f29_bva_mcp_invalid_tool_params_returns_32602_or_error(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])

    with mcp_stdio(api_key=key, principal_id=ctx["admin_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        try:
            r1 = mcp.tools_call("rag_search", {})
            assert r1.is_error is True
        except McpJsonRpcError as exc:
            assert exc.code == -32602

        try:
            r2 = mcp.tools_call("nonexistent_tool", {})
            assert r2.is_error is True
        except McpJsonRpcError as exc:
            assert exc.code in (-32601, -32602)


@pytest.mark.f29
def test_f29_bva_mcp_unauthenticated_or_revoked_key_denied(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    revoked_key = _mint_api_key(api_client, ctx, ctx["l2_id"])
    pg_ephemeral.execute_sql(
        f"UPDATE iam.api_keys SET revoked_at = now() WHERE org_id = '{ctx['org_id']}' AND principal_id = '{ctx['l2_id']}';"
    )

    unauth_resp = api_client.mcp_raw({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert unauth_resp.status == 401

    with mcp_stdio(api_key=revoked_key, env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        try:
            mcp.handshake()
            res = mcp.rag_search("secret")
            assert res.is_error is True
        except McpJsonRpcError as exc:
            assert exc.code != 0 and bool(exc.message)
        except McpProtocolError as exc:
            assert str(exc)


@pytest.mark.f29
def test_f29_bva_mcp_guard_check_and_rag_ask_block_safeword_with_zero_leak(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    api_client.with_api_key(key).studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query", "mcp_egress"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    with mcp_stdio(api_key=key, principal_id=ctx["admin_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        g_res = mcp.guard_check("PROJECT-ORION", surface="mcp_egress")
        assert g_res.is_error is True or g_res.json().get("decision") == "block"

        ask_res = mcp.rag_ask("PROJECT-ORION 알려줘")
        assert ask_res.is_error is True
        assert "guard_blocked" in ask_res.text.lower() or "block" in ask_res.text.lower()


@pytest.mark.f29
def test_f29_bva_mcp_enforces_arn_group_and_level_on_get_page_and_wiki_get(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    l2_key = _mint_api_key(api_client, ctx, ctx["l2_id"])

    b = api_client.with_api_key(admin_key).create_bucket(
        {"slug": f"priv-b-{uuid.uuid4().hex[:6]}", "name": "PrivB", "access_scope": "dept", "min_auth_level": 6}
    ).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    doc = api_client.with_api_key(admin_key).create_document(
        bid, {"title": "L6 Page", "text": "L6_RESTRICTED_PAGE_DATA", "access_scope": "private", "min_auth_level": 6}
    ).assert_status((200, 201)).json()
    doc_id = str(doc.get("id") or doc.get("document_id"))

    wiki_id = str(uuid.uuid4())
    wiki_slug = f"deny-w-{wiki_id[:6]}"
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, denied_group_ids, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{wiki_id}', '{ctx['org_id']}', '{ctx['dept2_id']}', '{ctx['admin_id']}', 'org', ARRAY['{ctx['denied_group_id']}'::UUID], 1, 'arn:openrag:{ctx['slug']}:dept2:org:wiki:L1/{wiki_id}', '{ctx['org_id']}:org', '{wiki_slug}', 'DeniedWiki', 'DENIED_WIKI_SECRET', 'DENIED_WIKI_SECRET', '{uuid.uuid4().hex}');
        """
    )

    with mcp_stdio(api_key=l2_key, principal_id=ctx["l2_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        pg_res = mcp.rag_get_page(document_id=doc_id, page_no=1)
        assert pg_res.is_error is True
        assert "L6_RESTRICTED_PAGE_DATA" not in pg_res.text

        wk_res = mcp.wiki_get(slug=wiki_slug)
        assert wk_res.is_error is True
        assert "DENIED_WIKI_SECRET" not in wk_res.text


@pytest.mark.f29
def test_f29_bva_mcp_rag_ingest_blocked_by_ingestion_gate_guard(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    slug = f"gate-{uuid.uuid4().hex[:6]}"
    api_client.with_api_key(key).create_bucket({"slug": slug, "name": "Gate", "access_scope": "dept"}).assert_status((200, 201))

    with mcp_stdio(api_key=key, principal_id=ctx["admin_id"], env={"DATABASE_URL": pg_ephemeral.dsn()}) as mcp:
        mcp.handshake()
        res = mcp.rag_ingest(
            bucket=slug,
            text="sk-test-openrag-1234567890abcdef secret credential",
            access_scope="dept",
            title="Blocked Secret",
        )
        assert res.is_error is True
        assert "guard_blocked" in res.text.lower() or "block" in res.text.lower()

    cnt = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM corpus.documents WHERE org_id = '{ctx['org_id']}' AND title = 'Blocked Secret';"
    ).strip()
    assert int(cnt) == 0


@pytest.mark.f29
def test_f29_bva_mcp_malformed_json_over_stdio_returns_parse_error_32700(
    openrag_root: Path,
    openrag_cli: OpenRagCli,
) -> None:
    try:
        cmd = openrag_cli.mcp_stdio_command()
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=openrag_root,
    )
    try:
        out, _ = proc.communicate("{not-json\n", timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert "-32700" in out
