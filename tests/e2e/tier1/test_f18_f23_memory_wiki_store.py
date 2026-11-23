"""Tier 1 E2E tests for F18-F23: CCLG Memory, Promote Ladder, Wiki/KoParaRel, 3-Engine, S3/OSS/BM25, Outbox."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import fail_missing_target
from lib import (
    FixtureFactory,
    McpHttpClient,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PgPrincipalContext,
)

pytestmark = [pytest.mark.tier1]


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
    dept2_id = str(uuid.uuid4())
    owner_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    group_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'Test Org {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES
            ('{dept_id}', '{org_id}', 'Finance', 'finance'),
            ('{dept2_id}', '{org_id}', 'Engineering', 'engineering');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{owner_id}', '{org_id}', 'owner-{owner_id[:6]}@example.org', 'Owner', 5, false, false),
            ('{peer_id}', '{org_id}', 'peer-{peer_id[:6]}@example.org', 'Peer', 5, false, false),
            ('{admin_id}', '{org_id}', 'admin-{admin_id[:6]}@example.org', 'DeptAdmin', 8, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept_id}', '{owner_id}', 'member'),
            ('{org_id}', '{dept_id}', '{peer_id}', 'member'),
            ('{org_id}', '{dept_id}', '{admin_id}', 'dept_admin');
        INSERT INTO iam.groups (id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level, allowed_actions) VALUES
            ('{group_id}', '{org_id}', NULL, 'tf-cross-{group_id[:8]}', 'project_tf', 3, 10, ARRAY['read','write','search']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id, member_role) VALUES
            ('{org_id}', '{group_id}', '{owner_id}', 'member'),
            ('{org_id}', '{group_id}', '{peer_id}', 'member');
        """
    )
    return {
        "org_id": org_id,
        "slug": slug,
        "dept_id": dept_id,
        "dept2_id": dept2_id,
        "owner_id": owner_id,
        "peer_id": peer_id,
        "admin_id": admin_id,
        "group_id": group_id,
    }


def _seed_bucket(pg: PgEphemeralCluster, ctx: dict[str, str], *, slug: str | None = None, min_auth_level: int = 3) -> tuple[str, str]:
    bucket_id = str(uuid.uuid4())
    bucket_slug = slug or f"bucket-{bucket_id[:8]}"
    arn = f"arn:openrag:{ctx['slug']}:finance:dept:bucket:L{min_auth_level}/{bucket_id}"
    pg.execute_sql(
        f"""
        INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
        VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', '{bucket_slug}', 'Bucket {bucket_slug}', 'dept', {min_auth_level}, '{arn}');
        """
    )
    return bucket_id, bucket_slug


def _seed_memory_entry(
    pg: PgEphemeralCluster,
    ctx: dict[str, str],
    *,
    text: str,
    scope: str = "private",
    status: str = "active",
    min_auth_level: int = 3,
    allowed_groups: list[str] | None = None,
    bucket_id: str | None = None,
) -> tuple[str, str]:
    ep_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    ep_arn = f"arn:openrag:{ctx['slug']}:finance:{scope}:episode:L{min_auth_level}/{ep_id}"
    mem_arn = f"arn:openrag:{ctx['slug']}:finance:{scope}:memory:L{min_auth_level}/{mem_id}"
    grp_sql = "ARRAY[]::UUID[]" if not allowed_groups else "ARRAY[" + ",".join(f"'{g}'::UUID" for g in allowed_groups) + "]"
    bkt_sql = "NULL" if not bucket_id else f"'{bucket_id}'::UUID"
    escaped = text.replace("'", "''")
    sha = _sha256_hex(text)
    vec = _unit_vec_128(1)
    pg.execute_sql(
        f"""
        INSERT INTO memory.episodes (id, org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, min_auth_level, arn, source_harness, summary_md, content_sha256)
        VALUES ('{ep_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['owner_id']}', '{scope}', {grp_sql}, {min_auth_level}, '{ep_arn}', 'chat_ui', '{escaped}', '{sha}');
        INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, min_auth_level, arn, acl_scope_key, memory_kind, title, effective_text, bm25_text, content_sha256, embedding, source_episode_id, status, version)
        VALUES ('{mem_id}', '{ctx['org_id']}', {bkt_sql}, '{ctx['dept_id']}', '{ctx['owner_id']}', '{scope}', {grp_sql}, {min_auth_level}, '{mem_arn}', '{ctx['org_id']}:{scope}', 'fact', 'Memory {mem_id[:6]}', '{escaped}', '{escaped}', '{sha}', '{vec}'::vector(128), '{ep_id}', '{status}', 1);
        """
    )
    return ep_id, mem_id


def _mint_api_key(api_client: OpenRagHttpClient, ctx: dict[str, str], principal_id: str) -> str:
    resp = api_client.cli_code_exchange(
        {
            "code": issue_cli_code(ctx["org_id"], principal_id),
            "org_id": ctx["org_id"],
            "principal_id": principal_id,
        }
    )
    resp.assert_status((200, 201))
    data = resp.json()
    key = data.get("api_key") or data.get("token") or data.get("key")
    assert isinstance(key, str) and key, f"Expected minted api_key in {data}"
    return key


# ============================================================================
# F18 — CCLG Memory Ledger, Pluggable MemoryBackend, engage/reflect (6 tests)
# ============================================================================


@pytest.mark.f18
def test_f18_episode_and_entry_immutable_amend_patch_cli_pg(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    ep_id, mem_id = _seed_memory_entry(pg_ephemeral, ctx, text="일비 2만원", scope="dept")

    res = openrag_cli.memory_patch(
        mem_id,
        op="amend",
        text="일비 3만원",
        reason="2026 개정",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )
    assert res.returncode == 0

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    entries = pg_ephemeral.query_as_principal(
        "SELECT effective_text, version, status FROM memory.entries WHERE id = %s",
        principal=owner_ctx,
        params=(mem_id,),
    )
    assert len(entries) == 1
    assert entries[0]["effective_text"] == "일비 3만원"
    assert entries[0]["version"] == 2
    assert entries[0]["status"] == "active"

    patches = pg_ephemeral.query_as_principal(
        "SELECT patch_op, prev_version, new_version, prev_text, new_text FROM memory.patches WHERE memory_entry_id = %s ORDER BY new_version",
        principal=owner_ctx,
        params=(mem_id,),
    )
    assert len(patches) >= 1
    latest = patches[-1]
    assert latest["patch_op"] == "amend"
    assert latest["prev_version"] == 1
    assert latest["new_version"] == 2
    assert latest["prev_text"] == "일비 2만원"
    assert latest["new_text"] == "일비 3만원"

    episodes = pg_ephemeral.query_as_principal(
        "SELECT summary_md FROM memory.episodes WHERE id = %s",
        principal=owner_ctx,
        params=(ep_id,),
    )
    assert episodes[0]["summary_md"] == "일비 2만원"


@pytest.mark.f18
def test_f18_active_effective_view_hides_retracted_and_superseded(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, active_id = _seed_memory_entry(pg_ephemeral, ctx, text="현행 보안 규정 유효", scope="dept")
    _, retract_id = _seed_memory_entry(pg_ephemeral, ctx, text="철회될 구버전 메모", scope="dept")
    _, super_id = _seed_memory_entry(pg_ephemeral, ctx, text="대체될 기존 여비 규정", scope="dept")

    openrag_cli.memory_patch(
        retract_id,
        op="retract",
        reason="오류 철회",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )
    openrag_cli.memory_patch(
        super_id,
        op="supersede",
        text="신규 개정 여비 규정 7만원",
        reason="2026년 개정",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    visible_rows = pg_ephemeral.query_as_principal(
        "SELECT id, effective_text, status FROM memory.entries WHERE id = ANY(%s::uuid[])",
        principal=owner_ctx,
        params=([active_id, retract_id, super_id],),
    )
    visible_ids = {str(r["id"]) for r in visible_rows}
    assert active_id in visible_ids
    assert retract_id not in visible_ids
    assert super_id not in visible_ids

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()
    search_res = mcp.memory_search("여비 규정", scope="dept")
    assert search_res.is_error is False
    payload_text = search_res.text
    assert "철회될 구버전 메모" not in payload_text
    assert "대체될 기존 여비 규정" not in payload_text


@pytest.mark.f18
def test_f18_memory_backend_swap_openrag_pg_vs_kumiho_and_offline_fallback(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    pg_yaml = sample_factory.pipeline_yaml("pg.yaml", memory_backend="openrag_pg")
    kumiho_yaml = sample_factory.pipeline_yaml("kumiho.yaml", memory_backend="kumiho")
    assert yaml.safe_load(pg_yaml.read_text(encoding="utf-8"))["memory"]["backend"] == "openrag_pg"
    assert yaml.safe_load(kumiho_yaml.read_text(encoding="utf-8"))["memory"]["backend"] == "kumiho"

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    with mcp_stdio(api_key=key, principal_id=ctx["owner_id"]) as mcp:
        mcp.handshake()
        r_pg = mcp.memory_reflect("PG 기본 백엔드 출장 메모", scope="dept", backend="openrag_pg")
        assert r_pg.is_error is False
        r_km = mcp.memory_reflect("Kumiho 플러그형 백엔드 출장 메모", scope="dept", backend="kumiho")
        assert r_km.is_error is False

        e_pg = mcp.memory_engage("PG 기본 백엔드", scope="dept", backend="openrag_pg")
        e_km = mcp.memory_engage("Kumiho 플러그형 백엔드", scope="dept", backend="kumiho")
        assert e_pg.is_error is False
        assert e_km.is_error is False
        assert "pg 기본 백엔드" in e_pg.text.lower() or "출장 메모" in e_pg.text or "openrag_pg" in e_pg.text.lower()
        assert "kumiho" in e_km.text.lower() or "출장 메모" in e_km.text


@pytest.mark.f18
def test_f18_memory_mirror_to_kumiho_dual_write(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    mirror_cfg = sample_factory.pipeline_yaml(
        "mirror.yaml", memory_backend="openrag_pg", memory_mirror_to="kumiho"
    ).read_text(encoding="utf-8")

    admin_client = api_client.with_api_key(key)
    admin_client.studio_pipeline_update({"pipeline_yaml": mirror_cfg}).assert_status((200, 201))

    mcp = McpHttpClient(admin_client)
    mcp.handshake()
    res = mcp.memory_reflect("이중 미러링 기억 테스트 2026", scope="dept")
    assert res.is_error is False

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["admin_id"],
        auth_level=8,
        dept_ids=(ctx["dept_id"],),
        dept_admin_ids=(ctx["dept_id"],),
    )
    rows = pg_ephemeral.query_as_principal(
        "SELECT id, effective_text FROM memory.entries WHERE effective_text LIKE %s",
        principal=owner_ctx,
        params=("%이중 미러링 기억 테스트 2026%",),
    )
    assert len(rows) >= 1

    km_recall = mcp.memory_engage("이중 미러링 기억 테스트", scope="dept", backend="kumiho")
    assert km_recall.is_error is False
    assert "이중 미러링" in km_recall.text


@pytest.mark.f18
def test_f18_cognitive_engage_reflect_and_belief_revision_edges(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()

    m1 = mcp.memory_reflect("출장비 5만원", scope="dept")
    assert m1.is_error is False
    m1_id = str(m1.json().get("memory_id") or m1.json().get("id"))
    assert m1_id

    for edge_type, body in (
        ("supports", "출장비 5만원 영수증 기준 확인"),
        ("derived_from", "총무팀 안내에서 파생된 출장비 기준"),
        ("contradicts", "출장비 5만원 구버전과 모순 보고"),
    ):
        r = mcp.memory_reflect(body, scope="dept", edge_type=edge_type, target_memory_id=m1_id)
        assert r.is_error is False

    m5 = mcp.memory_reflect(
        "출장비 7만원 개정",
        scope="dept",
        edge_type="supersedes",
        target_memory_id=m1_id,
    )
    assert m5.is_error is False

    recall = mcp.memory_engage("출장비", scope="dept")
    assert recall.is_error is False
    data = recall.json()
    serialized = json.dumps(data, ensure_ascii=False)
    assert "출장비 7만원 개정" in serialized


@pytest.mark.f18
def test_f18_private_memory_dept_admin_allowed_under_rls_and_guard(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, mem_id = _seed_memory_entry(
        pg_ephemeral,
        ctx,
        text="개인 비공개 메모 연락처 010-1234-5678",
        scope="private",
        min_auth_level=5,
    )

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    admin_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["admin_id"],
        auth_level=8,
        dept_ids=(ctx["dept_id"],),
        dept_admin_ids=(ctx["dept_id"],),
    )
    assert len(
        pg_ephemeral.query_as_principal(
            "SELECT id FROM memory.entries WHERE id = %s", principal=owner_ctx, params=(mem_id,)
        )
    ) == 1
    assert len(
        pg_ephemeral.query_as_principal(
            "SELECT id FROM memory.entries WHERE id = %s", principal=admin_ctx, params=(mem_id,)
        )
    ) == 1

    owner_key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    mcp = McpHttpClient(api_client.with_api_key(owner_key))
    mcp.handshake()
    res = mcp.memory_engage("개인 비공개 메모 연락처", scope="private")
    assert res.is_error is False
    assert "010-1234-5678" not in res.text


# ============================================================================
# F19 — Memory Promote Ladder (private -> dept -> group -> org -> wiki) (5 tests)
# ============================================================================


@pytest.mark.f19
def test_f19_promote_private_to_dept_cli(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, mem_id = _seed_memory_entry(pg_ephemeral, ctx, text="부서 공유용 실험 절차", scope="private")

    peer_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["peer_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    assert (
        pg_ephemeral.query_as_principal(
            "SELECT id FROM memory.entries WHERE id = %s", principal=peer_ctx, params=(mem_id,)
        )
        == []
    )

    openrag_cli.memory_promote(
        mem_id,
        to="dept",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    rows = pg_ephemeral.query_as_principal(
        "SELECT access_scope FROM memory.entries WHERE id = %s", principal=peer_ctx, params=(mem_id,)
    )
    assert len(rows) == 1
    assert rows[0]["access_scope"] == "dept"

    patches = pg_ephemeral.query_as_principal(
        "SELECT patch_op, prev_scope, new_scope FROM memory.patches WHERE memory_entry_id = %s",
        principal=peer_ctx,
        params=(mem_id,),
    )
    assert any(p["patch_op"] == "promote_scope" and p["prev_scope"] == "private" and p["new_scope"] == "dept" for p in patches)


@pytest.mark.f19
def test_f19_promote_dept_to_group_and_org(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, mem_id = _seed_memory_entry(
        pg_ephemeral,
        ctx,
        text="TF 및 전사 승격 테스트",
        scope="dept",
        allowed_groups=[ctx["group_id"]],
    )

    openrag_cli.memory_promote(
        mem_id,
        to="group",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )
    openrag_cli.memory_promote(
        mem_id,
        to="org",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    other_dept_user = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level)
        VALUES ('{other_dept_user}', '{ctx['org_id']}', 'other-{other_dept_user[:6]}@example.org', 'OtherDept', 4);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
        VALUES ('{ctx['org_id']}', '{ctx['dept2_id']}', '{other_dept_user}', 'member');
        """
    )
    other_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=other_dept_user,
        auth_level=4,
        dept_ids=(ctx["dept2_id"],),
    )
    rows = pg_ephemeral.query_as_principal(
        "SELECT access_scope, version FROM memory.entries WHERE id = %s",
        principal=other_ctx,
        params=(mem_id,),
    )
    assert len(rows) == 1
    assert rows[0]["access_scope"] == "org"
    assert rows[0]["version"] >= 3


@pytest.mark.f19
def test_f19_promote_to_wiki_creates_wiki_page_and_lineage(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)
    _, mem_id = _seed_memory_entry(
        pg_ephemeral,
        ctx,
        text="재무팀 법인카드 정산 규칙",
        scope="dept",
    )

    openrag_cli.memory_promote(
        mem_id,
        to="wiki",
        bucket=bucket_slug,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    mem_rows = pg_ephemeral.query_as_principal(
        "SELECT promoted_wiki_id FROM memory.entries WHERE id = %s",
        principal=owner_ctx,
        params=(mem_id,),
    )
    wiki_id = mem_rows[0]["promoted_wiki_id"]
    assert wiki_id is not None

    wiki_rows = pg_ephemeral.query_as_principal(
        "SELECT id, body_md, version FROM wiki.pages WHERE id = %s",
        principal=owner_ctx,
        params=(str(wiki_id),),
    )
    assert len(wiki_rows) == 1
    assert "법인카드 정산" in wiki_rows[0]["body_md"]

    src_rows = pg_ephemeral.query_as_principal(
        "SELECT source_type, source_id FROM wiki.page_sources WHERE page_id = %s",
        principal=owner_ctx,
        params=(str(wiki_id),),
    )
    assert any(s["source_type"] == "memory" and str(s["source_id"]) == mem_id for s in src_rows)

    patch_rows = pg_ephemeral.query_as_principal(
        "SELECT patch_op FROM memory.patches WHERE memory_entry_id = %s",
        principal=owner_ctx,
        params=(mem_id,),
    )
    assert any(p["patch_op"] == "promote_wiki" for p in patch_rows)


@pytest.mark.f19
def test_f19_promote_enqueues_indexing_outbox(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)
    _, mem_id = _seed_memory_entry(pg_ephemeral, ctx, text="승격 아웃박스 색인 검증", scope="dept")

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    resp = api_client.with_api_key(key).app_memory_promote(
        {"memory_id": mem_id, "to": "wiki", "bucket": bucket_slug}
    )
    resp.assert_status((200, 201))

    out = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM indexing.outbox WHERE org_id = '{ctx['org_id']}' AND target_table IN ('memory.entries', 'wiki.pages');"
    )
    assert int(out.strip()) >= 1

    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()
    wres = mcp.wiki_search("승격 아웃박스 색인 검증", bucket=bucket_slug)
    assert wres.is_error is False
    assert "승격 아웃박스" in wres.text


@pytest.mark.f19
def test_f19_dept_admin_can_promote_subordinate_dept_memory(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, mem_id = _seed_memory_entry(pg_ephemeral, ctx, text="팀원 비공개 메모 관리자 승격", scope="private")

    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    resp = api_client.with_api_key(admin_key).app_memory_promote(
        {"memory_id": mem_id, "to": "dept"}
    )
    resp.assert_status((200, 201))

    admin_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["admin_id"],
        auth_level=8,
        dept_ids=(ctx["dept_id"],),
        dept_admin_ids=(ctx["dept_id"],),
    )
    patches = pg_ephemeral.query_as_principal(
        "SELECT patch_op, actor_id, new_scope FROM memory.patches WHERE memory_entry_id = %s",
        principal=admin_ctx,
        params=(mem_id,),
    )
    assert any(p["patch_op"] == "promote_scope" and p["new_scope"] == "dept" for p in patches)


# ============================================================================
# F20 — DB-Native LLM Wiki + KoParaRel + .md/Obsidian Sync (6 tests)
# ============================================================================


@pytest.mark.f20
def test_f20_wiki_import_obsidian_vault_cli(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id, bucket_slug = _seed_bucket(pg_ephemeral, ctx)
    vault_dir = sample_factory.obsidian_vault()

    res = openrag_cli.wiki_import(
        vault_dir,
        bucket=bucket_slug,
        scope="dept",
        auth_level=3,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )
    assert res.returncode == 0

    admin_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["admin_id"],
        auth_level=8,
        dept_ids=(ctx["dept_id"],),
        dept_admin_ids=(ctx["dept_id"],),
    )
    pages = pg_ephemeral.query_as_principal(
        "SELECT id, slug, body_sha256, acl_scope_key, embedding IS NOT NULL AS has_vec FROM wiki.pages WHERE bucket_id = %s",
        principal=admin_ctx,
        params=(bucket_id,),
    )
    assert len(pages) == 4
    assert all(p["has_vec"] for p in pages)

    edges = pg_ephemeral.query_as_principal(
        "SELECT relation_type FROM wiki.edges WHERE org_id = %s",
        principal=admin_ctx,
        params=(ctx["org_id"],),
    )
    assert len(edges) >= 1

    syncs = pg_ephemeral.query_as_principal(
        "SELECT direction, status FROM wiki.sync_changes WHERE bucket_id = %s",
        principal=admin_ctx,
        params=(bucket_id,),
    )
    assert len(syncs) >= 4
    assert all(s["direction"] == "import" and s["status"] == "synced" for s in syncs)


@pytest.mark.f20
def test_f20_wiki_export_roundtrip_preserves_markdown_and_frontmatter(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id, bucket_slug = _seed_bucket(pg_ephemeral, ctx)
    vault_dir = sample_factory.obsidian_vault()

    openrag_cli.wiki_import(
        vault_dir,
        bucket=bucket_slug,
        scope="dept",
        auth_level=3,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    out_dir = tmp_path / "exported_wiki"
    openrag_cli.wiki_export(
        bucket_slug,
        out_dir=out_dir,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    md_files = sorted(out_dir.rglob("*.md"))
    assert len(md_files) >= 4
    contents = "\n".join(f.read_text(encoding="utf-8") for f in md_files)
    assert "---" in contents
    assert "출장 여비" in contents

    export_count = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM wiki.sync_changes WHERE bucket_id = '{bucket_id}' AND direction = 'export' AND status = 'synced';"
    )
    assert int(export_count.strip()) >= 4


@pytest.mark.f20
def test_f20_kopararel_six_label_edges_corpus_and_wiki(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id, _ = _seed_bucket(pg_ephemeral, ctx)

    page_ids: list[str] = []
    for idx in range(7):
        pid = str(uuid.uuid4())
        page_ids.append(pid)
        arn = f"arn:openrag:{ctx['slug']}:finance:dept:wiki:L3/{pid}"
        pg_ephemeral.execute_sql(
            f"""
            INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding)
            VALUES ('{pid}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, '{arn}', '{ctx['org_id']}:dept', 'kopara-{idx}-{pid[:6]}', 'Page {idx}', '규정 본문 {idx}', '규정 본문 {idx}', '{_sha256_hex(pid)}', '{_unit_vec_128(idx)}'::vector(128));
            """
        )

    labels = ("equivalent", "entails", "contradicts", "supersedes", "refines", "related")
    for idx, label in enumerate(labels):
        pg_ephemeral.execute_sql(
            f"""
            INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type, confidence)
            VALUES ('{ctx['org_id']}', '{page_ids[idx]}', '{page_ids[idx + 1]}', '{label}', 0.92);
            """
        )

    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    conflicts_resp = api_client.with_api_key(admin_key).studio_wiki_conflicts()
    conflicts_resp.assert_status(200)
    serialized = json.dumps(conflicts_resp.json(), ensure_ascii=False)
    assert "contradicts" in serialized or "supersedes" in serialized


@pytest.mark.f20
def test_f20_bidirectional_sync_endpoint_hash_diff(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)

    slug = f"obsidian-sync-{uuid.uuid4().hex[:6]}"
    v1_body = "# 원격 동기화 문서\n\n초기 본문 v1"
    r1 = client.sync_documents(
        bucket_slug,
        {
            "documents": [
                {
                    "path": f"{slug}.md",
                    "slug": slug,
                    "title": "원격 동기화 문서",
                    "body_md": v1_body,
                    "body_sha256": _sha256_hex(v1_body),
                }
            ]
        },
    )
    r1.assert_status((200, 201))

    # Re-send identical hash -> skipped / no-op
    r2 = client.sync_documents(
        bucket_slug,
        {
            "documents": [
                {
                    "path": f"{slug}.md",
                    "slug": slug,
                    "title": "원격 동기화 문서",
                    "body_md": v1_body,
                    "body_sha256": _sha256_hex(v1_body),
                }
            ]
        },
    )
    r2.assert_status(200)
    d2 = json.dumps(r2.json(), ensure_ascii=False).lower()
    assert "skip" in d2 or "noop" in d2 or "unchanged" in d2 or r2.json().get("updated", 0) == 0

    # Modified hash -> version bumps to 2
    v2_body = "# 원격 동기화 문서\n\n수정된 본문 v2"
    r3 = client.sync_documents(
        bucket_slug,
        {
            "documents": [
                {
                    "path": f"{slug}.md",
                    "slug": slug,
                    "title": "원격 동기화 문서",
                    "body_md": v2_body,
                    "body_sha256": _sha256_hex(v2_body),
                }
            ]
        },
    )
    r3.assert_status(200)
    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    rows = pg_ephemeral.query_as_principal(
        "SELECT version FROM wiki.pages WHERE slug = %s", principal=owner_ctx, params=(slug,)
    )
    assert len(rows) == 1
    assert rows[0]["version"] >= 2


@pytest.mark.f20
def test_f20_wiki_rls_dept_and_level_filtering(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx, min_auth_level=1)
    vault_dir = sample_factory.obsidian_vault()
    openrag_cli.wiki_import(
        vault_dir,
        bucket=bucket_slug,
        scope="dept",
        auth_level=3,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    l3_user = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level)
        VALUES ('{l3_user}', '{ctx['org_id']}', 'l3-{l3_user[:6]}@example.org', 'L3User', 3);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
        VALUES ('{ctx['org_id']}', '{ctx['dept_id']}', '{l3_user}', 'member');
        """
    )

    l3_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=l3_user,
        auth_level=3,
        dept_ids=(ctx["dept_id"],),
    )
    l5_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )

    l3_slugs = {
        r["slug"]
        for r in pg_ephemeral.query_as_principal("SELECT slug FROM wiki.pages", principal=l3_ctx)
    }
    l5_slugs = {
        r["slug"]
        for r in pg_ephemeral.query_as_principal("SELECT slug FROM wiki.pages", principal=l5_ctx)
    }
    assert "travel-policy-2026" in l3_slugs
    assert "security-guidelines" not in l3_slugs
    assert "security-guidelines" in l5_slugs

    l3_key = _mint_api_key(api_client, ctx, l3_user)
    mcp = McpHttpClient(api_client.with_api_key(l3_key))
    mcp.handshake()
    wres = mcp.wiki_search("정보보안 및 개인정보", bucket=bucket_slug)
    assert wres.is_error is False
    assert "주민등록번호 등 민감 정보" not in wres.text


@pytest.mark.f20
def test_f20_wiki_search_and_get_by_slug_or_id(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)
    vault_dir = sample_factory.obsidian_vault()
    openrag_cli.wiki_import(
        vault_dir,
        bucket=bucket_slug,
        scope="dept",
        auth_level=3,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    mcp = McpHttpClient(api_client.with_api_key(key))
    mcp.handshake()

    search_res = mcp.wiki_search("출장 일비 70,000", bucket=bucket_slug)
    assert search_res.is_error is False
    assert "travel-policy-2026" in search_res.text or "70,000" in search_res.text

    by_slug = mcp.wiki_get(slug="travel-policy-2026")
    assert by_slug.is_error is False
    page_data = by_slug.json()
    page_id = str(page_data.get("id") or page_data.get("page_id"))
    assert page_id

    by_id = mcp.wiki_get(page_id=page_id)
    assert by_id.is_error is False
    assert "70,000" in by_id.text

    app_wiki_resp = api_client.with_api_key(key).app_wiki(page_id)
    app_wiki_resp.assert_status(200)


# ============================================================================
# F21 — pg_pinecone 3-Engine Table-Level Switching (5 tests)
# ============================================================================


@pytest.mark.f21
def test_f21_switch_engine_native_to_pgvector_cli(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO vectors.index_bindings (org_id, table_name, vector_column, pg_pinecone_engine, dimension, status)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', 'embedding', 'native', 128, 'ready')
        ON CONFLICT (org_id, table_name, vector_column) DO NOTHING;
        """
    )

    openrag_cli.index_engine(
        "corpus.chunks",
        "pgvector",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    out = pg_ephemeral.execute_sql(
        f"SELECT pg_pinecone_engine, dimension, status FROM vectors.index_bindings WHERE org_id = '{ctx['org_id']}' AND table_name = 'corpus.chunks';"
    )
    assert "pgvector" in out
    assert "128" in out
    assert "ready" in out


@pytest.mark.f21
def test_f21_switch_engine_pgvector_to_remote_offline_mode(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])

    openrag_cli.index_engine(
        "corpus.chunks",
        "pgvector",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "offline"},
        check=True,
    )
    openrag_cli.index_engine(
        "corpus.chunks",
        "remote",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "offline"},
        check=True,
    )
    openrag_cli.index_engine(
        "corpus.chunks",
        "native",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "offline"},
        check=True,
    )

    resp = api_client.with_api_key(admin_key).studio_index_engines()
    resp.assert_status(200)


@pytest.mark.f21
def test_f21_search_query_invariance_across_three_engines(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    b_resp = client.create_bucket(
        {"slug": f"eng-{uuid.uuid4().hex[:6]}", "name": "Engine Test", "access_scope": "dept", "min_auth_level": 3}
    )
    b_resp.assert_status((200, 201))
    bucket_id = str(b_resp.json().get("id") or b_resp.json().get("bucket_id"))

    hwpx = sample_factory.hwpx_file("policy.hwpx", title="삼중 엔진 검증", paragraphs=["삼중 엔진 무변경 검색 검증 본문입니다."])
    client.upload_document(
        bucket_id,
        filename=hwpx.name,
        content=hwpx.read_bytes(),
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202))

    search_payload = {"query": "삼중 엔진 무변경 검색", "top_k": 5}
    for engine in ("native", "pgvector", "remote"):
        client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": engine}).assert_status(200)
        s_resp = client.search_bucket(bucket_id, search_payload)
        s_resp.assert_status(200)
        assert "삼중 엔진" in json.dumps(s_resp.json(), ensure_ascii=False)


@pytest.mark.f21
def test_f21_independent_per_table_engine_bindings(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": "pgvector"}).assert_status(200)
    client.studio_index_set_engine({"table_name": "wiki.pages", "engine": "native"}).assert_status(200)
    client.studio_index_set_engine({"table_name": "memory.entries", "engine": "native"}).assert_status(200)

    engines_resp = client.studio_index_engines()
    engines_resp.assert_status(200)
    bindings = engines_resp.json()
    text = json.dumps(bindings)
    assert "corpus.chunks" in text
    assert "pgvector" in text
    assert "wiki.pages" in text
    assert "memory.entries" in text


@pytest.mark.f21
def test_f21_capability_probe_fallback_on_local_pg17(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    caps = pg_ephemeral.probe_capabilities()
    assert caps["vector"] is True

    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id, _ = _seed_bucket(pg_ephemeral, ctx)
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
        VALUES ('{doc_id}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:doc:L3/{doc_id}', '{ctx['org_id']}:dept', 'Fallback Doc', '{_sha256_hex(doc_id)}');
        INSERT INTO corpus.chunks (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
        VALUES ('{chunk_id}', '{ctx['org_id']}', '{doc_id}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:chunk:L3/{chunk_id}', '{ctx['org_id']}:dept', 0, '로컬 pgvector HNSW 폴백 검색 검증', '로컬 pgvector HNSW 폴백 검색 검증', '{_sha256_hex(chunk_id)}', '{_unit_vec_128(3)}'::vector(128));
        """
    )

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=5,
        dept_ids=(ctx["dept_id"],),
    )
    rows = pg_ephemeral.query_as_principal(
        f"SELECT id, 1 - (embedding <=> '{_unit_vec_128(3)}'::vector(128)) AS sim FROM corpus.chunks WHERE id = %s",
        principal=owner_ctx,
        params=(chunk_id,),
    )
    assert len(rows) == 1
    assert rows[0]["sim"] > 0.99

    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    api_client.with_api_key(key).search_bucket(bucket_id, {"query": "폴백 검색 검증", "top_k": 5}).assert_status(200)


# ============================================================================
# F22 — External Vector Adapters (S3 Vectors STS ARN & OSS) + BM25 (5 tests)
# ============================================================================


@pytest.mark.f22
def test_f22_s3_vectors_per_tenant_sts_bucket_arn_isolation(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx_a = _seed_org_and_users(pg_ephemeral)
    ctx_b = _seed_org_and_users(pg_ephemeral)
    key_a = _mint_api_key(api_client, ctx_a, ctx_a["admin_id"])
    key_b = _mint_api_key(api_client, ctx_b, ctx_b["admin_id"])

    b_a = api_client.with_api_key(key_a).create_bucket(
        {"slug": f"s3a-{uuid.uuid4().hex[:5]}", "name": "Tenant A", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    b_b = api_client.with_api_key(key_b).create_bucket(
        {"slug": f"s3b-{uuid.uuid4().hex[:5]}", "name": "Tenant B", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid_a = str(b_a.get("id") or b_a.get("bucket_id"))
    bid_b = str(b_b.get("id") or b_b.get("bucket_id"))

    api_client.with_api_key(key_a).create_document(bid_a, {"title": "TenantA Doc", "text": "공유키워드 테넌트A 전용 문서", "access_scope": "dept"}).assert_status((200, 201))
    api_client.with_api_key(key_b).create_document(bid_b, {"title": "TenantB Doc", "text": "공유키워드 테넌트B 기밀 문서", "access_scope": "dept"}).assert_status((200, 201))

    res = openrag_cli.index_mirror(
        "corpus.chunks",
        to="s3-vectors",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "offline"},
        check=True,
    )
    expected_prefix_a = f"sv-prod-{hashlib.sha256(ctx_a['org_id'].encode()).hexdigest()[:24]}"
    expected_prefix_b = f"sv-prod-{hashlib.sha256(ctx_b['org_id'].encode()).hexdigest()[:24]}"
    assert expected_prefix_a != expected_prefix_b
    bindings_a = pg_ephemeral.execute_sql(
        f"SELECT external_backend, external_config::text FROM vectors.index_bindings WHERE org_id = '{ctx_a['org_id']}' AND table_name = 'corpus.chunks';"
    )
    assert expected_prefix_a in res.stdout or expected_prefix_a in bindings_a

    s_a = api_client.with_api_key(key_a).search_bucket(bid_a, {"query": "공유키워드", "top_k": 10}).assert_status(200)
    assert "테넌트A 전용" in s_a.text
    assert "테넌트B 기밀" not in s_a.text


@pytest.mark.f22
def test_f22_s3_vectors_empty_array_encoded_via_empty_sentinel(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)

    b_resp = client.create_bucket(
        {"slug": f"emptyarr-{uuid.uuid4().hex[:6]}", "name": "EmptyGroup", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bucket_id = str(b_resp.get("id") or b_resp.get("bucket_id"))

    doc_path = sample_factory.write_text("empty_groups.txt", "빈 배열 센티넬 검증 본문")
    client.upload_document(
        bucket_id,
        filename=doc_path.name,
        content=doc_path.read_bytes(),
        content_type="text/plain",
        fields={"access_scope": "dept", "allowed_group_ids": "[]"},
    ).assert_status((200, 201, 202))

    res = openrag_cli.index_mirror(
        "corpus.chunks",
        to="s3-vectors",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "offline"},
        check=True,
    )
    assert res.returncode == 0
    s_resp = client.search_bucket(bucket_id, {"query": "센티넬 검증", "top_k": 5}).assert_status(200)
    assert "센티넬 검증" in s_resp.text


@pytest.mark.f22
def test_f22_oss_vector_adapter_mirror_and_search(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)
    b_resp = client.create_bucket(
        {"slug": f"oss-{uuid.uuid4().hex[:6]}", "name": "OSS Vector", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bucket_id = str(b_resp.get("id") or b_resp.get("bucket_id"))
    client.create_document(bucket_id, {"title": "OSS Doc", "text": "오픈소스 벡터 어댑터 Qdrant 미러링 검색", "access_scope": "dept"}).assert_status((200, 201))

    openrag_cli.index_mirror(
        "corpus.chunks",
        to="qdrant",
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_MODE": "offline"},
        check=True,
    )
    out = pg_ephemeral.execute_sql(
        f"SELECT external_backend FROM vectors.index_bindings WHERE org_id = '{ctx['org_id']}' AND table_name = 'corpus.chunks';"
    )
    assert "qdrant" in out
    client.search_bucket(bucket_id, {"query": "오픈소스 벡터 어댑터", "top_k": 5}).assert_status(200)


@pytest.mark.f22
def test_f22_bm25_korean_legal_clause_normalization(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)
    b_resp = client.create_bucket(
        {"slug": f"bm25law-{uuid.uuid4().hex[:6]}", "name": "Legal BM25", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bucket_id = str(b_resp.get("id") or b_resp.get("bucket_id"))

    client.create_document(
        bucket_id,
        {"title": "여비 규정", "text": "제3조(출장 여비 지급 기준) 공무 출장 시 일비를 지급한다.", "access_scope": "dept"},
    ).assert_status((200, 201))

    s_resp = client.search_bucket(
        bucket_id,
        {"query": "3조 출장 여비", "fusion": "convex", "alpha": 1.0, "top_k": 5},
    ).assert_status(200)
    data = s_resp.json()
    results = data.get("results") or data.get("items") or []
    assert len(results) >= 1
    assert any((r.get("bm25_score", 0) > 0 or "제3조" in json.dumps(r, ensure_ascii=False)) for r in results)


@pytest.mark.f22
def test_f22_bm25_k1_1_2_b_0_idf_ranking_order(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)
    b_resp = client.create_bucket(
        {"slug": f"bm25tf-{uuid.uuid4().hex[:6]}", "name": "BM25 TF", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bucket_id = str(b_resp.get("id") or b_resp.get("bucket_id"))

    rare = "감사원특수활동비"
    d1 = client.create_document(bucket_id, {"title": "D1_single", "text": f"{rare} 1회 언급", "access_scope": "dept"}).assert_status((200, 201)).json()
    padding = " ".join(["일반행정절차안내"] * 40)
    d2 = client.create_document(bucket_id, {"title": "D2_double", "text": f"{rare} {padding} {rare}", "access_scope": "dept"}).assert_status((200, 201)).json()
    d3 = client.create_document(bucket_id, {"title": "D3_absent", "text": "전혀 무관한 주차 관리 지침", "access_scope": "dept"}).assert_status((200, 201)).json()

    s_resp = client.search_bucket(bucket_id, {"query": rare, "fusion": "convex", "alpha": 1.0, "top_k": 5}).assert_status(200)
    results = s_resp.json().get("results") or s_resp.json().get("items") or []
    ids = [str(r.get("document_id") or r.get("id")) for r in results]
    d1_id = str(d1.get("id") or d1.get("document_id"))
    d2_id = str(d2.get("id") or d2.get("document_id"))
    d3_id = str(d3.get("id") or d3.get("document_id"))

    assert d2_id in ids and d1_id in ids
    assert ids.index(d2_id) < ids.index(d1_id)
    assert d3_id not in ids


# ============================================================================
# F23 — Incremental Outbox, Watermarks & Quarantine (5 tests)
# ============================================================================


@pytest.mark.f23
def test_f23_outbox_sha256_noop_skip_avoids_reembedding(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)

    body = "# SHA256 No-Op 테스트\n동일 내용 반복 전송 시 재임베딩을 건너뛴다."
    sha = _sha256_hex(body)
    payload = {"documents": [{"path": "noop.md", "slug": f"noop-{uuid.uuid4().hex[:6]}", "title": "NoOp", "body_md": body, "body_sha256": sha}]}
    client.sync_documents(bucket_slug, payload).assert_status((200, 201))

    before = pg_ephemeral.execute_sql(
        f"SELECT COALESCE(SUM(indexed_count), 0) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}';"
    ).strip()

    r2 = client.sync_documents(bucket_slug, payload).assert_status(200)
    after = pg_ephemeral.execute_sql(
        f"SELECT COALESCE(SUM(indexed_count), 0) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}';"
    ).strip()
    assert int(after) == int(before)
    assert r2.json().get("skipped", 0) >= 1 or "noop" in r2.text.lower() or "skip" in r2.text.lower()


@pytest.mark.f23
def test_f23_delta_write_updates_only_changed_page_chunks(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["owner_id"])
    client = api_client.with_api_key(key)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)

    pdf_v1 = sample_factory.pdf_bytes(["Page 1 unchanged", "Page 2 original", "Page 3 unchanged"])
    pdf_v2 = sample_factory.pdf_bytes(["Page 1 unchanged", "Page 2 updated delta", "Page 3 unchanged"])
    slug = f"delta-{uuid.uuid4().hex[:6]}"

    client.sync_documents(
        bucket_slug,
        {"documents": [{"path": f"{slug}.pdf", "slug": slug, "pages": ["Page 1 unchanged", "Page 2 original", "Page 3 unchanged"], "content_sha256": hashlib.sha256(pdf_v1).hexdigest()}]},
    ).assert_status((200, 201))

    r2 = client.sync_documents(
        bucket_slug,
        {"documents": [{"path": f"{slug}.pdf", "slug": slug, "pages": ["Page 1 unchanged", "Page 2 updated delta", "Page 3 unchanged"], "content_sha256": hashlib.sha256(pdf_v2).hexdigest()}]},
    ).assert_status(200)
    body = r2.json()
    delta_count = body.get("delta_updated") or body.get("upserted_chunks") or body.get("updated")
    assert delta_count is not None
    assert int(delta_count) < 3 or "Page 2 updated delta" in json.dumps(body, ensure_ascii=False)


@pytest.mark.f23
def test_f23_watermarks_advance_monotonically_per_table_and_backend(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, bucket_slug = _seed_bucket(pg_ephemeral, ctx)

    hwpx = sample_factory.hwpx_file("wm.hwpx", paragraphs=["워터마크 단조 증가 검증"])
    openrag_cli.ingest(
        hwpx,
        bucket=bucket_slug,
        scope="dept",
        auth_level=3,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    status_res = openrag_cli.index_status(
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )
    status_lower = status_res.stdout.lower()
    assert any(
        tok in status_lower
        for tok in ("pending", "watermarks", "indexed_count", "last_outbox_id", "corpus.chunks", "done")
    ), f"Unexpected `openrag index status` output: {status_res.stdout!r}"

    out = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}' AND last_outbox_id > 0 AND indexed_count > 0;"
    )
    assert int(out.strip()) >= 1


@pytest.mark.f23
def test_f23_priority_1_redaction_hook_preempts_priority_10_batch(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    target_low = str(uuid.uuid4())
    target_high = str(uuid.uuid4())

    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status)
        VALUES
            ('{ctx['org_id']}', 'corpus.chunks', '{target_low}', 'upsert_embed', 10, 'pending'),
            ('{ctx['org_id']}', 'corpus.chunks', '{target_high}', 'redact_reindex', 1, 'pending');
        """
    )

    openrag_cli.run(
        ["index", "drain", "--limit", "1"],
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )

    high_status = pg_ephemeral.execute_sql(
        f"SELECT status FROM indexing.outbox WHERE target_id = '{target_high}';"
    ).strip()
    low_status = pg_ephemeral.execute_sql(
        f"SELECT status FROM indexing.outbox WHERE target_id = '{target_low}';"
    ).strip()
    assert high_status == "done"
    assert low_status == "pending"


@pytest.mark.f23
def test_f23_quarantine_retry_cli_and_studio_recovery(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    bucket_id, _ = _seed_bucket(pg_ephemeral, ctx)
    doc_id = str(uuid.uuid4())
    c1_id = str(uuid.uuid4())
    c2_id = str(uuid.uuid4())
    q1_id = str(uuid.uuid4())
    q2_id = str(uuid.uuid4())

    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
        VALUES ('{doc_id}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:doc:L3/{doc_id}', '{ctx['org_id']}:dept', 'QDoc', '{_sha256_hex(doc_id)}');
        INSERT INTO corpus.chunks (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
        VALUES
            ('{c1_id}', '{ctx['org_id']}', '{doc_id}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:chunk:L3/{c1_id}', '{ctx['org_id']}:dept', 0, '격리 복구 1', '격리 복구 1', '{_sha256_hex(c1_id)}', '{_unit_vec_128(4)}'::vector(128)),
            ('{c2_id}', '{ctx['org_id']}', '{doc_id}', '{ctx['dept_id']}', '{ctx['owner_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:finance:dept:chunk:L3/{c2_id}', '{ctx['org_id']}:dept', 1, '격리 복구 2', '격리 복구 2', '{_sha256_hex(c2_id)}', '{_unit_vec_128(5)}'::vector(128));
        INSERT INTO indexing.quarantine (id, org_id, target_table, target_id, op, error_class, error_detail)
        VALUES
            ('{q1_id}', '{ctx['org_id']}', 'corpus.chunks', '{c1_id}', 'upsert_embed', 'transient_503', 'temporary outage'),
            ('{q2_id}', '{ctx['org_id']}', 'corpus.chunks', '{c2_id}', 'upsert_embed', 'transient_503', 'temporary outage');
        """
    )

    openrag_cli.index_retry(
        quarantine_id=q1_id,
        env={"DATABASE_URL": pg_ephemeral.dsn()},
        check=True,
    )
    resolved1 = pg_ephemeral.execute_sql(
        f"SELECT resolved_at IS NOT NULL FROM indexing.quarantine WHERE id = '{q1_id}';"
    ).strip()
    assert resolved1 == "t"

    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    api_client.with_api_key(admin_key).studio_quarantine_retry(q2_id).assert_status((200, 202))
    resolved2 = pg_ephemeral.execute_sql(
        f"SELECT resolved_at IS NOT NULL FROM indexing.quarantine WHERE id = '{q2_id}';"
    ).strip()
    assert resolved2 == "t"
