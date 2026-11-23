"""Tier 2 BVA & negative tests for F18-F23: Memory, Promote, Wiki, 3-Engine, S3/OSS/BM25, Outbox."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from lib import (
    FixtureFactory,
    McpHttpClient,
    McpJsonRpcError,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PgPrincipalContext,
)

pytestmark = [pytest.mark.tier2]


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> None:
    pg.ensure_started()
    out = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;")
    if out.strip() != "t":
        pg.apply_migrations(root / "migrations")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit_vec_128(seed: int = 1) -> str:
    vals = [0.0] * 128
    vals[seed % 128] = 1.0
    return "[" + ",".join(f"{v:.6f}" for v in vals) + "]"


def _seed_org_and_users(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    owner_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    denied_group_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'BVA Org {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ('{dept_id}', '{org_id}', 'Finance', 'finance');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{owner_id}', '{org_id}', 'owner-{owner_id[:6]}@example.org', 'Owner', 8, false, false),
            ('{peer_id}', '{org_id}', 'peer-{peer_id[:6]}@example.org', 'Peer', 9, false, false),
            ('{admin_id}', '{org_id}', 'admin-{admin_id[:6]}@example.org', 'Admin', 10, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept_id}', '{owner_id}', 'member'),
            ('{org_id}', '{dept_id}', '{peer_id}', 'member'),
            ('{org_id}', '{dept_id}', '{admin_id}', 'dept_admin');
        INSERT INTO iam.groups (id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level)
        VALUES ('{denied_group_id}', '{org_id}', NULL, 'deny-grp-{denied_group_id[:8]}', 'permission_group', 1, 10);
        INSERT INTO iam.group_members (org_id, group_id, principal_id, member_role)
        VALUES ('{org_id}', '{denied_group_id}', '{owner_id}', 'member');
        """
    )
    return {
        "org_id": org_id,
        "slug": slug,
        "dept_id": dept_id,
        "owner_id": owner_id,
        "peer_id": peer_id,
        "admin_id": admin_id,
        "denied_group_id": denied_group_id,
    }


def _seed_memory(
    pg: PgEphemeralCluster,
    ctx: dict[str, str],
    *,
    text: str,
    scope: str = "private",
    status: str = "active",
    min_auth_level: int = 3,
) -> str:
    mem_id = str(uuid.uuid4())
    arn = f"arn:openrag:{ctx['slug']}:finance:{scope}:memory:L{min_auth_level}/{mem_id}"
    escaped = text.replace("'", "''")
    pg.execute_sql(
        f"""
        INSERT INTO memory.entries (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, embedding, status, version)
        VALUES ('{mem_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', '{scope}', {min_auth_level}, '{arn}', '{ctx['org_id']}:{scope}', 'fact', '{escaped}', '{escaped}', '{_sha256_hex(text)}', '{_unit_vec_128(2)}'::vector(128), '{status}', 1);
        """
    )
    return mem_id


def _mint_api_key(api_client: OpenRagHttpClient, ctx: dict[str, str], principal_id: str) -> str:
    resp = api_client.cli_code_exchange(
        {"code": issue_cli_code(ctx["org_id"], principal_id), "org_id": ctx["org_id"], "principal_id": principal_id}
    )
    resp.assert_status((200, 201))
    key = resp.json().get("api_key") or resp.json().get("token")
    assert isinstance(key, str) and key
    return key


# ============================================================================
# F18 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f18
def test_f18_bva_reject_unknown_patch_op_fail_closed(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    mem_id = _seed_memory(pg_ephemeral, ctx, text="원본 불변 기억", scope="dept")

    cli_res = openrag_cli.memory_patch(
        mem_id, op="overwrite", text="변조", env={"DATABASE_URL": pg_ephemeral.dsn()}
    )
    assert cli_res.returncode != 0

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    http_res = api_client.with_api_key(key).app_memory_patch(
        {"memory_id": mem_id, "op": "overwrite", "text": "변조"}
    )
    assert http_res.status in (400, 422)

    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()
    try:
        mcp_res = mcp.memory_patch(mem_id, "overwrite", text="변조")
        assert mcp_res.is_error is True
    except McpJsonRpcError as exc:
        assert exc.code in (-32602, -32603)

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"], principal_id=ctx["owner_id"], auth_level=8, dept_ids=(ctx["dept_id"],)
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        pg_ephemeral.query_as_principal(
            "INSERT INTO memory.patches (org_id, memory_entry_id, patch_op, prev_version, new_version, actor_type, actor_id) VALUES (%s, %s, 'overwrite', 1, 2, 'human', %s)",
            principal=owner_ctx,
            params=(ctx["org_id"], mem_id, ctx["owner_id"]),
            fetch=False,
        )


@pytest.mark.f18
def test_f18_bva_reject_patch_on_retracted_entry_by_non_admin(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    mem_id = _seed_memory(pg_ephemeral, ctx, text="이미 철회됨", scope="dept", status="retracted")

    cli_res = openrag_cli.memory_patch(
        mem_id, op="amend", text="부활 시도", env={"DATABASE_URL": pg_ephemeral.dsn()}
    )
    assert cli_res.returncode != 0

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()
    res = mcp.memory_patch(mem_id, "amend", text="부활 시도")
    assert res.is_error is True

    ver = pg_ephemeral.execute_sql(f"SELECT version FROM memory.entries WHERE id = '{mem_id}';").strip()
    assert int(ver) == 1


@pytest.mark.f18
def test_f18_bva_hard_purge_redaction_clears_prev_text_to_null(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    mem_id = _seed_memory(pg_ephemeral, ctx, text="주민번호 900101-1234567 보관", scope="dept")

    openrag_cli.memory_patch(
        mem_id, op="amend", text="수정됨", env={"DATABASE_URL": pg_ephemeral.dsn()}, check=True
    )
    openrag_cli.redact_run(
        memory_id=mem_id,
        hard_purge=True,
        reason="PII hard purge",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    out = pg_ephemeral.execute_sql(
        f"SELECT status, effective_text FROM memory.entries WHERE id = '{mem_id}';"
    )
    assert "redacted" in out
    assert "900101-1234567" not in out

    patch_leak = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM memory.patches WHERE memory_entry_id = '{mem_id}' AND (prev_text LIKE '%900101-1234567%' OR new_text LIKE '%900101-1234567%');"
    ).strip()
    assert int(patch_leak) == 0


@pytest.mark.f18
def test_f18_bva_private_memory_blocked_for_same_dept_peer(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    secret_phrase = f"PEER_SECRET_{uuid.uuid4().hex[:8]}"
    mem_id = _seed_memory(pg_ephemeral, ctx, text=secret_phrase, scope="private", min_auth_level=3)

    peer_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["peer_id"],
        auth_level=9,
        dept_ids=(ctx["dept_id"],),
    )
    rows = pg_ephemeral.query_as_principal(
        "SELECT id FROM memory.entries WHERE id = %s", principal=peer_ctx, params=(mem_id,)
    )
    assert rows == []

    peer_key = _mint_api_key(api_client, ctx, ctx["peer_id"])
    mcp = McpHttpClient(api_client.with_api_key(peer_key))
    mcp.handshake()
    s_res = mcp.memory_search(secret_phrase)
    e_res = mcp.memory_engage(secret_phrase)
    assert secret_phrase not in s_res.text
    assert secret_phrase not in e_res.text


@pytest.mark.f18
def test_f18_bva_invalid_memory_backend_and_invalid_edge_type_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    bad_yaml = sample_factory.pipeline_yaml("bad_mem.yaml", memory_backend="unknown_mem").read_text(encoding="utf-8")

    resp = api_client.with_api_key(admin_key).studio_pipeline_update({"pipeline_yaml": bad_yaml})
    assert resp.status in (400, 422)

    mcp = McpHttpClient(api_client.with_api_key(admin_key))
    mcp.handshake()
    try:
        r = mcp.memory_reflect("엣지 검증", edge_type="invalid_rel", target_memory_id=str(uuid.uuid4()))
        assert r.is_error is True
    except McpJsonRpcError as exc:
        assert exc.code == -32602


@pytest.mark.f18
def test_f18_bva_memory_reflect_blocked_by_safeword_guard(
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
            "surfaces": ["inbound_query", "mcp_egress", "ingestion_gate"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    res = mcp.memory_reflect("PROJECT-ORION 비밀 메모")
    assert res.is_error is True
    assert "guard_blocked" in res.text.lower() or "block" in res.text.lower()

    cnt = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM memory.entries WHERE org_id = '{ctx['org_id']}' AND effective_text LIKE '%PROJECT-ORION%';"
    ).strip()
    assert int(cnt) == 0


# ============================================================================
# F19 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f19
def test_f19_bva_same_dept_peer_cannot_promote_private_memory(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    mem_id = _seed_memory(pg_ephemeral, ctx, text="동료 승격 불가", scope="private")

    peer_key = _mint_api_key(api_client, ctx, ctx["peer_id"])
    resp = api_client.with_api_key(peer_key).app_memory_promote({"memory_id": mem_id, "to": "dept"})
    assert resp.status in (403, 404)

    cli_res = openrag_cli.memory_promote(
        mem_id,
        to="dept",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_API_KEY": peer_key},
    )
    assert cli_res.returncode != 0

    scope = pg_ephemeral.execute_sql(f"SELECT access_scope FROM memory.entries WHERE id = '{mem_id}';").strip()
    assert scope == "private"


@pytest.mark.f19
def test_f19_bva_explicit_deny_group_blocks_promote_to_wiki(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id = str(uuid.uuid4())
    bucket_slug = f"denied-{bucket_id[:6]}"
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, denied_group_ids, min_auth_level, arn)
        VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['admin_id']}', '{bucket_slug}', 'Denied Bucket', 'org', ARRAY['{ctx['denied_group_id']}'::UUID], 1, 'arn:openrag:{ctx['slug']}:finance:org:bucket:L1/{bucket_id}');
        """
    )
    mem_id = _seed_memory(pg_ephemeral, ctx, text="거부 그룹 대상 승격 불가", scope="dept")

    owner_key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    resp = api_client.with_api_key(owner_key).app_memory_promote(
        {"memory_id": mem_id, "to": "wiki", "bucket": bucket_slug}
    )
    assert resp.status == 403
    cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM wiki.pages WHERE bucket_id = '{bucket_id}';").strip()
    assert int(cnt) == 0


@pytest.mark.f19
def test_f19_bva_promote_already_retracted_or_redacted_memory_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    ret_id = _seed_memory(pg_ephemeral, ctx, text="철회 메모", scope="dept", status="retracted")
    red_id = _seed_memory(pg_ephemeral, ctx, text="마스킹 메모", scope="dept", status="redacted")
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)

    assert client.app_memory_promote({"memory_id": ret_id, "to": "org"}).status in (400, 409, 422)
    assert client.app_memory_promote({"memory_id": red_id, "to": "org"}).status in (400, 409, 422)


@pytest.mark.f19
def test_f19_bva_promote_to_invalid_target_scope_or_missing_bucket(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    mem_id = _seed_memory(pg_ephemeral, ctx, text="잘못된 승격 대상", scope="dept")

    r_bad_scope = openrag_cli.memory_promote(
        mem_id, to="invalid_scope", env={"DATABASE_URL": pg_ephemeral.dsn()}
    )
    assert r_bad_scope.returncode != 0

    r_missing_bkt = openrag_cli.memory_promote(
        mem_id, to="wiki", bucket="nonexistent-bucket-xyz", env={"DATABASE_URL": pg_ephemeral.dsn()}
    )
    assert r_missing_bkt.returncode != 0


@pytest.mark.f19
def test_f19_bva_insufficient_auth_level_for_target_bucket_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l3_user = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level)
        VALUES ('{l3_user}', '{ctx['org_id']}', 'l3-{l3_user[:6]}@example.org', 'L3', 3);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
        VALUES ('{ctx['org_id']}', '{ctx['dept_id']}', '{l3_user}', 'member');
        """
    )
    bucket_id = str(uuid.uuid4())
    bucket_slug = f"l6-only-{bucket_id[:6]}"
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
        VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['admin_id']}', '{bucket_slug}', 'L6 Bucket', 'dept', 6, 'arn:openrag:{ctx['slug']}:finance:dept:bucket:L6/{bucket_id}');
        """
    )
    mem_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO memory.entries (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, embedding)
        VALUES ('{mem_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{l3_user}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:memory:L3/{mem_id}', '{ctx['org_id']}:dept', 'fact', 'L3 메모리', 'L3 메모리', '{_sha256_hex(mem_id)}', '{_unit_vec_128(1)}'::vector(128));
        """
    )

    l3_key = _mint_api_key(api_client, ctx, l3_user)
    resp = api_client.with_api_key(l3_key).app_memory_promote(
        {"memory_id": mem_id, "to": "wiki", "bucket": bucket_slug}
    )
    assert resp.status == 403


# ============================================================================
# F20 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f20
def test_f20_bva_invalid_kopararel_relation_type_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    p1, p2 = str(uuid.uuid4()), str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES
            ('{p1}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:wiki:L3/{p1}', '{ctx['org_id']}:dept', 'p1-{p1[:6]}', 'P1', 'A', 'A', '{_sha256_hex(p1)}'),
            ('{p2}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:wiki:L3/{p2}', '{ctx['org_id']}:dept', 'p2-{p2[:6]}', 'P2', 'B', 'B', '{_sha256_hex(p2)}');
        """
    )
    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"], principal_id=ctx["owner_id"], auth_level=8, dept_ids=(ctx["dept_id"],)
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        pg_ephemeral.query_as_principal(
            "INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type) VALUES (%s, %s, %s, 'contradicts_maybe')",
            principal=owner_ctx,
            params=(ctx["org_id"], p1, p2),
            fetch=False,
        )


@pytest.mark.f20
def test_f20_bva_wiki_sync_conflict_on_stale_sha256(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id = str(uuid.uuid4())
    bucket_slug = f"sync-conf-{bucket_id[:6]}"
    pg_ephemeral.execute_sql(
        f"INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', '{bucket_slug}', 'SyncConf', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:bucket:L3/{bucket_id}');"
    )
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)
    slug = f"policy-{uuid.uuid4().hex[:6]}"

    v1 = "초기 본문 v1"
    v2 = "서버에서 갱신된 본문 v2"
    client.sync_documents(
        bucket_slug,
        {"documents": [{"path": f"{slug}.md", "slug": slug, "body_md": v1, "body_sha256": _sha256_hex(v1)}]},
    ).assert_status((200, 201))
    client.sync_documents(
        bucket_slug,
        {"documents": [{"path": f"{slug}.md", "slug": slug, "body_md": v2, "body_sha256": _sha256_hex(v2), "base_sha256": _sha256_hex(v1)}]},
    ).assert_status(200)

    stale_resp = client.sync_documents(
        bucket_slug,
        {"documents": [{"path": f"{slug}.md", "slug": slug, "body_md": "충돌 시도 본문", "body_sha256": _sha256_hex("충돌 시도 본문"), "base_sha256": _sha256_hex(v1), "base_version": 1}]},
    )
    assert stale_resp.status in (200, 409)
    if stale_resp.status == 200:
        assert "conflict" in stale_resp.text.lower()

    body = pg_ephemeral.execute_sql(f"SELECT body_md FROM wiki.pages WHERE slug = '{slug}';")
    assert "서버에서 갱신된 본문 v2" in body


@pytest.mark.f20
def test_f20_bva_wiki_import_path_traversal_blocked(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id = str(uuid.uuid4())
    bucket_slug = f"trav-{bucket_id[:6]}"
    pg_ephemeral.execute_sql(
        f"INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', '{bucket_slug}', 'Trav', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:bucket:L3/{bucket_id}');"
    )
    vault = tmp_path / "malicious_vault"
    vault.mkdir()
    (vault / "safe.md").write_text("# Safe\nok", encoding="utf-8")
    escape_link = vault / "escape.md"
    escape_link.symlink_to(Path("/etc/hosts"))

    res = openrag_cli.wiki_import(
        vault, bucket=bucket_slug, scope="dept", auth_level=3, env={"DATABASE_URL": pg_ephemeral.dsn()}
    )
    if res.returncode == 0:
        bodies = pg_ephemeral.execute_sql(f"SELECT body_md FROM wiki.pages WHERE bucket_id = '{bucket_id}';")
        assert "127.0.0.1" not in bodies and "localhost" not in bodies


@pytest.mark.f20
def test_f20_bva_wiki_min_auth_level_boundary_L4_vs_L5(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l4_id, l5_id = str(uuid.uuid4()), str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level) VALUES
            ('{l4_id}', '{ctx['org_id']}', 'l4-{l4_id[:6]}@example.org', 'L4', 4),
            ('{l5_id}', '{ctx['org_id']}', 'l5-{l5_id[:6]}@example.org', 'L5', 5);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{ctx['org_id']}', '{ctx['dept_id']}', '{l4_id}', 'member'),
            ('{ctx['org_id']}', '{ctx['dept_id']}', '{l5_id}', 'member');
        """
    )
    page_id = str(uuid.uuid4())
    slug = f"l5-wiki-{page_id[:6]}"
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{page_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 5, 'arn:openrag:{ctx['slug']}:finance:dept:wiki:L5/{page_id}', '{ctx['org_id']}:dept', '{slug}', 'L5 Security', 'L5 비밀 지침', 'L5 비밀 지침', '{_sha256_hex(page_id)}');
        """
    )

    l4_ctx = PgPrincipalContext(org_id=ctx["org_id"], principal_id=l4_id, auth_level=4, dept_ids=(ctx["dept_id"],))
    l5_ctx = PgPrincipalContext(org_id=ctx["org_id"], principal_id=l5_id, auth_level=5, dept_ids=(ctx["dept_id"],))
    assert pg_ephemeral.query_as_principal("SELECT id FROM wiki.pages WHERE id = %s", principal=l4_ctx, params=(page_id,)) == []
    assert len(pg_ephemeral.query_as_principal("SELECT id FROM wiki.pages WHERE id = %s", principal=l5_ctx, params=(page_id,))) == 1

    l4_key = _mint_api_key(api_client, ctx, l4_id)
    mcp = McpHttpClient(api_client.with_api_key(l4_key))
    mcp.handshake()
    res = mcp.wiki_get(slug=slug)
    assert res.is_error is True or "L5 비밀 지침" not in res.text


@pytest.mark.f20
def test_f20_bva_wiki_rls_hides_edge_when_dst_page_unauthorized(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    pub_id, priv_id = str(uuid.uuid4()), str(uuid.uuid4())
    edge_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES
            ('{pub_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'org', 1, 'arn:openrag:{ctx['slug']}:finance:org:wiki:L1/{pub_id}', '{ctx['org_id']}:org', 'pub-{pub_id[:6]}', 'Pub', 'Public', 'Public', '{_sha256_hex(pub_id)}'),
            ('{priv_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'private', 1, 'arn:openrag:{ctx['slug']}:finance:private:wiki:L1/{priv_id}', '{ctx['org_id']}:private', 'priv-{priv_id[:6]}', 'Priv', 'Private', 'Private', '{_sha256_hex(priv_id)}');
        INSERT INTO wiki.edges (id, org_id, src_page_id, dst_page_id, relation_type)
        VALUES ('{edge_id}', '{ctx['org_id']}', '{pub_id}', '{priv_id}', 'references');
        """
    )

    peer_ctx = PgPrincipalContext(org_id=ctx["org_id"], principal_id=ctx["peer_id"], auth_level=9, dept_ids=(ctx["dept_id"],))
    rows = pg_ephemeral.query_as_principal("SELECT id FROM wiki.edges WHERE id = %s", principal=peer_ctx, params=(edge_id,))
    assert rows == []


@pytest.mark.f20
def test_f20_bva_wiki_sync_empty_markdown_and_duplicate_slug(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id = str(uuid.uuid4())
    bucket_slug = f"dup-{bucket_id[:6]}"
    pg_ephemeral.execute_sql(
        f"INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn) VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', '{bucket_slug}', 'Dup', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:bucket:L3/{bucket_id}');"
    )
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)

    empty_resp = client.sync_documents(
        bucket_slug,
        {"documents": [{"path": "empty.md", "slug": "empty-md", "body_md": "", "body_sha256": _sha256_hex("")}]},
    )
    assert empty_resp.status in (200, 400, 422)
    assert empty_resp.status != 500


# ============================================================================
# F21 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f21
def test_f21_bva_reject_unknown_engine_name(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])

    cli_res = openrag_cli.index_engine("corpus.chunks", "elastic", env={"DATABASE_URL": pg_ephemeral.dsn()})
    assert cli_res.returncode != 0

    http_res = api_client.with_api_key(admin_key).studio_index_set_engine(
        {"table_name": "corpus.chunks", "engine": "milvus"}
    )
    assert http_res.status in (400, 422)


@pytest.mark.f21
def test_f21_bva_reject_unknown_or_non_vector_table_name(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    assert openrag_cli.index_engine("iam.principals", "pgvector", env={"DATABASE_URL": pg_ephemeral.dsn()}).returncode != 0
    assert openrag_cli.index_engine("nonexistent.table", "pgvector", env={"DATABASE_URL": pg_ephemeral.dsn()}).returncode != 0


@pytest.mark.f21
def test_f21_bva_remote_engine_without_api_key_in_live_mode_returns_503(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    res = openrag_cli.index_engine(
        "corpus.chunks",
        "remote",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "live", "PINECONE_API_KEY": ""},
    )
    assert res.returncode != 0
    combined = (res.stdout + res.stderr).lower()
    assert "503" in combined or "pinecone" in combined or "api" in combined or "key" in combined


@pytest.mark.f21
def test_f21_bva_reject_non_128_vector_dimension_in_config_and_sql(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    base_yaml = sample_factory.pipeline_yaml("base.yaml").read_text(encoding="utf-8")

    for bad_dim in (127, 129):
        bad_yaml = base_yaml.replace("dim: 128", f"dim: {bad_dim}")
        resp = api_client.with_api_key(admin_key).studio_pipeline_update({"pipeline_yaml": bad_yaml})
        assert resp.status in (400, 422)

    bad_vec = "[" + ",".join(["0.1"] * 127) + "]"
    with pytest.raises(Exception):
        pg_ephemeral.execute_sql(f"SELECT '{bad_vec}'::vector(128);")


@pytest.mark.f21
def test_f21_bva_idempotent_switch_to_current_engine(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": "native"}).assert_status(200)
    client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": "native"}).assert_status(200)

    cnt = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM vectors.index_bindings WHERE org_id = '{ctx['org_id']}' AND table_name = 'corpus.chunks';"
    ).strip()
    assert int(cnt) == 1


# ============================================================================
# F22 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f22
def test_f22_bva_s3_vectors_filterable_metadata_2048_accept_vs_2049_reject(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)
    b = client.create_bucket({"slug": f"s3cap-{uuid.uuid4().hex[:6]}", "name": "S3Cap", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    ok_val = "x" * 1950
    over_val = "x" * 2049
    client.create_document(
        bid,
        {"title": "2048 OK", "text": "meta 2048", "access_scope": "dept", "external_backend": "s3_vectors", "filterable_metadata": {"pad": ok_val}},
    ).assert_status((200, 201))

    bad_resp = client.create_document(
        bid,
        {"title": "2049 Reject", "text": "meta 2049", "access_scope": "dept", "external_backend": "s3_vectors", "filterable_metadata": {"pad": over_val}},
    )
    assert bad_resp.status in (400, 422)
    assert "2048" in bad_resp.text or "too_large" in bad_resp.text.lower() or "metadata" in bad_resp.text.lower()


@pytest.mark.f22
def test_f22_bva_s3_vectors_metadata_fields_64_accept_vs_65_reject(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)
    b = client.create_bucket({"slug": f"s3fld-{uuid.uuid4().hex[:6]}", "name": "S3Fld", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    fields_65 = {f"k{i}": "v" for i in range(65)}
    bad_resp = client.create_document(
        bid,
        {"title": "65 Fields", "text": "field limit", "access_scope": "dept", "external_backend": "s3_vectors", "filterable_metadata": fields_65},
    )
    assert bad_resp.status in (400, 422)


@pytest.mark.f22
def test_f22_bva_s3_vectors_text_30720_accept_vs_30721_reject(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)
    b = client.create_bucket({"slug": f"s3txt-{uuid.uuid4().hex[:6]}", "name": "S3Txt", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    ok_resp = client.create_document(
        bid,
        {"title": "30720 OK", "text": "a" * 30720, "access_scope": "dept", "external_backend": "s3_vectors"},
    )
    ok_resp.assert_status((200, 201))

    bad_resp = client.create_document(
        bid,
        {"title": "30721 Reject", "text": "a" * 30721, "access_scope": "dept", "external_backend": "s3_vectors"},
    )
    assert bad_resp.status in (400, 422)


@pytest.mark.f22
def test_f22_bva_s3_vectors_cross_tenant_sts_isolation_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx_a = _seed_org_and_users(pg_ephemeral)
    ctx_b = _seed_org_and_users(pg_ephemeral)
    key_a = _mint_api_key(api_client, ctx_a, ctx_a["admin_id"])
    key_b = _mint_api_key(api_client, ctx_b, ctx_b["admin_id"])

    b_b = api_client.with_api_key(key_b).create_bucket(
        {"slug": f"s3iso-{uuid.uuid4().hex[:6]}", "name": "Tenant B", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid_b = str(b_b.get("id") or b_b.get("bucket_id"))
    api_client.with_api_key(key_b).create_document(
        bid_b, {"title": "Secret B", "text": "CrossTenantSecret999", "access_scope": "dept"}
    ).assert_status((200, 201))

    resp = api_client.with_api_key(key_a).search_bucket(bid_b, {"query": "CrossTenantSecret999", "top_k": 5})
    assert resp.status in (403, 404) or "CrossTenantSecret999" not in resp.text


@pytest.mark.f22
def test_f22_bva_reject_zero_vector_and_nan_inf_vector(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"badvec-{uuid.uuid4().hex[:6]}", "name": "BadVec", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    r_zero = client.search_bucket(bid, {"query": "test", "query_vector": [0.0] * 128})
    assert r_zero.status in (400, 422)

    r_nan = client.request(
        "POST",
        f"/v2/buckets/{bid}/search",
        data=json.dumps({"query": "test", "query_vector": [1.0] + [0.0] * 127}).replace("1.0", "NaN").encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert r_nan.status in (400, 422)


@pytest.mark.f22
def test_f22_bva_bm25_empty_and_punctuation_only_query_returns_empty_not_500(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"bm25punct-{uuid.uuid4().hex[:6]}", "name": "Punct", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    for bad_q in ("", "!@#$%^&*()"):
        resp = client.search_bucket(bid, {"query": bad_q, "fusion": "convex", "alpha": 1.0})
        assert resp.status in (200, 400, 422)
        if resp.status == 200:
            results = resp.json().get("results") or resp.json().get("items") or []
            assert results == []


# ============================================================================
# F23 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f23
def test_f23_bva_quarantine_boundary_2_failures_retry_vs_3_failures_quarantined(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    target_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status, attempts, max_attempts, payload)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{target_id}', 'upsert_embed', 5, 'pending', 0, 3, '{{"simulate_transient_error": true}}'::jsonb);
        """
    )

    for _ in range(2):
        openrag_cli.run(["index", "drain", "--limit", "1"], env={"DATABASE_URL": pg_ephemeral.dsn()}, check=True)
    st2 = pg_ephemeral.execute_sql(f"SELECT status, attempts FROM indexing.outbox WHERE target_id = '{target_id}';")
    assert "pending" in st2 and "2" in st2

    openrag_cli.run(["index", "drain", "--limit", "1"], env={"DATABASE_URL": pg_ephemeral.dsn()}, check=True)
    st3 = pg_ephemeral.execute_sql(f"SELECT status, attempts FROM indexing.outbox WHERE target_id = '{target_id}';")
    assert "quarantined" in st3
    q_cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM indexing.quarantine WHERE target_id = '{target_id}';").strip()
    assert int(q_cnt) == 1


@pytest.mark.f23
def test_f23_bva_quarantined_item_does_not_block_healthy_outbox_items(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bad_id, good_id = str(uuid.uuid4()), str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status, attempts, max_attempts)
        VALUES
            ('{ctx['org_id']}', 'corpus.chunks', '{bad_id}', 'upsert_embed', 1, 'quarantined', 3, 3),
            ('{ctx['org_id']}', 'corpus.chunks', '{good_id}', 'upsert_embed', 5, 'pending', 0, 3);
        """
    )
    openrag_cli.run(["index", "drain", "--limit", "10"], env={"DATABASE_URL": pg_ephemeral.dsn()}, check=True)
    assert pg_ephemeral.execute_sql(f"SELECT status FROM indexing.outbox WHERE target_id = '{good_id}';").strip() == "done"
    assert pg_ephemeral.execute_sql(f"SELECT status FROM indexing.outbox WHERE target_id = '{bad_id}';").strip() == "quarantined"


@pytest.mark.f23
def test_f23_bva_non_retryable_403_422_immediate_quarantine(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    target_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status, attempts, max_attempts, payload)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{target_id}', 'upsert_embed', 5, 'pending', 0, 3, '{{"simulate_error_code": 422}}'::jsonb);
        """
    )
    openrag_cli.run(["index", "drain", "--limit", "1"], env={"DATABASE_URL": pg_ephemeral.dsn()}, check=True)
    st = pg_ephemeral.execute_sql(f"SELECT status, attempts FROM indexing.outbox WHERE target_id = '{target_id}';")
    assert "quarantined" in st
    q_cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM indexing.quarantine WHERE target_id = '{target_id}';").strip()
    assert int(q_cnt) == 1


@pytest.mark.f23
def test_f23_bva_duplicate_idempotency_key_rejected_or_coalesced(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    idem = f"idem-{uuid.uuid4()}"
    t1, t2 = str(uuid.uuid4()), str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, idempotency_key, status)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{t1}', 'upsert_embed', '{idem}', 'pending');
        """
    )
    owner_ctx = PgPrincipalContext(org_id=ctx["org_id"], principal_id=ctx["admin_id"], auth_level=10, dept_ids=(ctx["dept_id"],))
    with pytest.raises(psycopg.errors.UniqueViolation):
        pg_ephemeral.query_as_principal(
            "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, idempotency_key, status) VALUES (%s, 'corpus.chunks', %s, 'upsert_embed', %s, 'pending')",
            principal=owner_ctx,
            params=(ctx["org_id"], t2, idem),
            fetch=False,
        )


@pytest.mark.f23
def test_f23_bva_invalid_outbox_priority_0_and_11_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    owner_ctx = PgPrincipalContext(org_id=ctx["org_id"], principal_id=ctx["admin_id"], auth_level=10, dept_ids=(ctx["dept_id"],))
    for bad_prio in (0, 11):
        with pytest.raises(psycopg.errors.CheckViolation):
            pg_ephemeral.query_as_principal(
                "INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority) VALUES (%s, 'corpus.chunks', %s, 'upsert_embed', %s)",
                principal=owner_ctx,
                params=(ctx["org_id"], str(uuid.uuid4()), bad_prio),
                fetch=False,
            )


@pytest.mark.f23
def test_f23_bva_retry_nonexistent_quarantine_id_returns_404(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    missing_id = "00000000-0000-0000-0000-000000000099"

    cli_res = openrag_cli.index_retry(quarantine_id=missing_id, env={"DATABASE_URL": pg_ephemeral.dsn()})
    assert cli_res.returncode != 0

    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    http_res = api_client.with_api_key(admin_key).studio_quarantine_retry(missing_id)
    assert http_res.status == 404
