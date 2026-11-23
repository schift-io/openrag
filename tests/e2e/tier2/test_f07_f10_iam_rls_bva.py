"""Tier 2 boundary, negative, and corner-case E2E tests for F07-F10 (IAM, ARN, Groups, Levels, Deny)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.pg_harness import PgEphemeralCluster, PgPrincipalContext

pytestmark = pytest.mark.tier2


def _ensure_migrated(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations directory: {migrations_dir}")
    pg_ephemeral.ensure_started()
    if getattr(pg_ephemeral, "_e2e_migrated_port", None) != pg_ephemeral.port:
        pg_ephemeral.apply_migrations(migrations_dir)
        setattr(pg_ephemeral, "_e2e_migrated_port", pg_ephemeral.port)
    return pg_ephemeral


@pytest.fixture(scope="module")
def migrated_pg(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    return _ensure_migrated(pg_ephemeral, openrag_root)


def _dynamic_insert(
    cur: psycopg.Cursor[Any],
    schema: str,
    table: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    cur.execute(
        """
        SELECT column_name, data_type, udt_name, is_nullable, column_default, is_generated
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    cols_meta = cur.fetchall()
    if not cols_meta:
        raise AssertionError(f"Table {schema}.{table} missing")
    available = {row[0]: row for row in cols_meta}
    insert_data = {k: v for k, v in values.items() if k in available}

    for col_name, data_type, udt_name, is_nullable, col_default, is_generated in cols_meta:
        if col_name in insert_data or is_generated == "ALWAYS" or is_nullable == "YES" or col_default is not None:
            continue
        if data_type == "uuid":
            insert_data[col_name] = str(uuid.uuid4())
        elif data_type in ("text", "character varying", "character"):
            insert_data[col_name] = f"e2e_{col_name}_{uuid.uuid4().hex[:8]}"
        elif data_type in ("integer", "smallint", "bigint"):
            insert_data[col_name] = 1
        elif data_type in ("double precision", "real", "numeric"):
            insert_data[col_name] = 0.0
        elif data_type == "boolean":
            insert_data[col_name] = False
        elif data_type in ("jsonb", "json"):
            insert_data[col_name] = "{}"
        elif data_type == "ARRAY":
            insert_data[col_name] = []
        elif data_type == "bytea":
            insert_data[col_name] = b"e2e"
        elif udt_name == "vector":
            insert_data[col_name] = "[" + ",".join(["1.0"] + ["0.0"] * 127) + "]"
        else:
            insert_data[col_name] = f"e2e_{uuid.uuid4().hex[:6]}"

    cols = list(insert_data.keys())
    phs: list[str] = []
    params: list[Any] = []
    for c in cols:
        val = insert_data[c]
        data_type, udt_name = available[c][1], available[c][2]
        if udt_name == "vector":
            phs.append("%s::vector(128)")
        elif data_type in ("jsonb", "json") and isinstance(val, str):
            phs.append(f"%s::{data_type}")
        else:
            phs.append("%s")
        params.append(val)

    sql = f'INSERT INTO "{schema}"."{table}" ({", ".join(cols)}) VALUES ({", ".join(phs)}) RETURNING *'
    cur.execute(sql, params)
    row = cur.fetchone()
    assert cur.description is not None and row is not None
    return {desc[0]: row[idx] for idx, desc in enumerate(cur.description)}


def _seed_iam_fixture(pg: PgEphemeralCluster) -> dict[str, str]:
    org_1 = str(uuid.uuid4())
    org_2 = str(uuid.uuid4())
    dept_a = str(uuid.uuid4())
    dept_b = str(uuid.uuid4())
    alice = str(uuid.uuid4())
    bob = str(uuid.uuid4())
    carol = str(uuid.uuid4())
    dave = str(uuid.uuid4())
    bkt_1 = str(uuid.uuid4())

    with psycopg.connect(pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for oid, slug in ((org_1, f"o1-{org_1[:6]}"), (org_2, f"o2-{org_2[:6]}")):
                    _dynamic_insert(cur, "iam", "orgs", {"id": oid, "slug": slug, "name": slug})
                for did, oid, slug in (
                    (dept_a, org_1, f"da-{dept_a[:6]}"),
                    (dept_b, org_1, f"db-{dept_b[:6]}"),
                ):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "departments",
                        {"id": did, "org_id": oid, "slug": slug, "name": slug, "arn_path": f"org/{slug}"},
                    )
                for pid, oid, lvl in (
                    (alice, org_1, 2),
                    (bob, org_1, 10),
                    (carol, org_1, 9),
                    (dave, org_1, 3),
                ):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "principals",
                        {
                            "id": pid,
                            "org_id": oid,
                            "email": f"{pid[:8]}@example.or.kr",
                            "display_name": pid[:8],
                            "auth_level": lvl,
                        },
                    )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "buckets",
                    {
                        "id": bkt_1,
                        "org_id": org_1,
                        "dept_id": dept_a,
                        "owner_principal_id": alice,
                        "slug": f"bva-bkt-{bkt_1[:6]}",
                        "name": "BVA Bucket",
                        "access_scope": "org",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_1}:da:org:bucket:L1/{bkt_1}",
                    },
                )
    return {
        "org_1": org_1,
        "org_2": org_2,
        "dept_a": dept_a,
        "dept_b": dept_b,
        "alice": alice,
        "bob": bob,
        "carol": carol,
        "dave": dave,
        "bkt_1": bkt_1,
    }


def _seed_doc_bundle(
    pg: PgEphemeralCluster,
    *,
    org_id: str,
    dept_id: str,
    bucket_id: str,
    owner_id: str,
    access_scope: str,
    min_auth_level: int = 1,
    allowed_group_ids: list[str] | None = None,
    denied_group_ids: list[str] | None = None,
) -> dict[str, str]:
    doc_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    ep_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    wiki_id = str(uuid.uuid4())
    ag = allowed_group_ids or []
    dg = denied_group_ids or []
    sk = f"{org_id}:{dept_id}:{access_scope}:L{min_auth_level}"

    with psycopg.connect(pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "corpus",
                    "documents",
                    {
                        "id": doc_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "owner_principal_id": owner_id,
                        "title": "BVA Doc",
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:da:{access_scope}:document:L{min_auth_level}/{doc_id}",
                        "acl_scope_key": sk,
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "pages",
                    {
                        "id": page_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "document_id": doc_id,
                        "owner_principal_id": owner_id,
                        "page_no": 1,
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:da:{access_scope}:page:L{min_auth_level}/{page_id}",
                        "acl_scope_key": sk,
                    },
                )
                _dynamic_insert(
                    cur,
                    "corpus",
                    "chunks",
                    {
                        "id": chunk_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "document_id": doc_id,
                        "page_id": page_id,
                        "owner_principal_id": owner_id,
                        "chunk_idx": 0,
                        "redacted_text": "BVA chunk",
                        "bm25_text": "BVA chunk",
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:da:{access_scope}:chunk:L{min_auth_level}/{chunk_id}",
                        "acl_scope_key": sk,
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "episodes",
                    {
                        "id": ep_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "owner_principal_id": owner_id,
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:da:{access_scope}:episode:L{min_auth_level}/{ep_id}",
                        "source_harness": "chat_ui",
                        "summary_md": "BVA ep",
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "entries",
                    {
                        "id": mem_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "source_episode_id": ep_id,
                        "owner_principal_id": owner_id,
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:da:{access_scope}:memory:L{min_auth_level}/{mem_id}",
                        "acl_scope_key": sk,
                        "memory_kind": "fact",
                        "effective_text": "BVA mem",
                        "bm25_text": "BVA mem",
                    },
                )
                _dynamic_insert(
                    cur,
                    "wiki",
                    "pages",
                    {
                        "id": wiki_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "owner_principal_id": owner_id,
                        "slug": f"w-{wiki_id[:6]}",
                        "title": "BVA Wiki",
                        "body_md": "BVA wiki",
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:da:{access_scope}:wiki:L{min_auth_level}/{wiki_id}",
                        "acl_scope_key": sk,
                    },
                )
    return {"doc_id": doc_id, "chunk_id": chunk_id, "mem_id": mem_id, "wiki_id": wiki_id}


# ---------------------------------------------------------------------------
# F07 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f07
def test_f07_bva_unset_or_empty_gucs_fail_closed_zero_rows(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="org",
        min_auth_level=1,
    )

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL ROLE {migrated_pg.app_role}")
                cur.execute("SELECT id FROM corpus.documents")
                assert cur.fetchall() == []

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL ROLE {migrated_pg.app_role}")
                cur.execute("SELECT set_config('app.org_id', '', true), set_config('app.principal_id', '', true)")
                cur.execute("SELECT id FROM corpus.documents")
                assert cur.fetchall() == []


@pytest.mark.tier2
@pytest.mark.f07
def test_f07_bva_dept_admin_of_dept_a_blocked_from_dept_b_private_doc(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_b"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["dave"],
        access_scope="private",
        min_auth_level=3,
    )
    carol_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["carol"],
        auth_level=9,
        dept_ids=(iam["dept_a"],),
        dept_admin_ids=(iam["dept_a"],),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=carol_ctx,
            params=(res["doc_id"],),
        )
        == []
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM memory.entries WHERE id = %s::uuid",
            principal=carol_ctx,
            params=(res["mem_id"],),
        )
        == []
    )


@pytest.mark.tier2
@pytest.mark.f07
def test_f07_bva_cross_org_spoofed_dept_admin_cannot_read_other_org(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=1,
    )
    spoof_ctx = PgPrincipalContext(
        org_id=iam["org_2"],
        principal_id=iam["carol"],
        auth_level=10,
        dept_ids=(iam["dept_a"],),
        dept_admin_ids=(iam["dept_a"],),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=spoof_ctx,
            params=(res["doc_id"],),
        )
        == []
    )


@pytest.mark.tier2
@pytest.mark.f07
def test_f07_bva_openrag_app_cannot_bypass_rls_even_if_guc_set_true(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=1,
    )
    with pytest.raises(ValueError):
        PgPrincipalContext(org_id=iam["org_1"], principal_id=iam["bob"], bypass_rls=True).to_guc_map()

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL ROLE {migrated_pg.app_role}")
                cur.execute(
                    """
                    SELECT set_config('app.org_id', %s, true),
                           set_config('app.principal_id', %s, true),
                           set_config('app.auth_level', '10', true),
                           set_config('app.dept_ids', %s, true),
                           set_config('app.dept_admin_ids', '', true),
                           set_config('app.group_ids', '', true),
                           set_config('app.bypass_rls', 'true', true)
                    """,
                    (iam["org_1"], iam["bob"], iam["dept_a"]),
                )
                cur.execute("SELECT id FROM corpus.documents WHERE id = %s::uuid", (res["doc_id"],))
                assert cur.fetchall() == []


@pytest.mark.tier2
@pytest.mark.f07
def test_f07_bva_high_level_peer_still_blocked_from_lower_level_private_resource(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=2,
    )
    l10_peer = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["bob"],
        auth_level=10,
        dept_ids=(iam["dept_a"],),
        dept_admin_ids=(),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=l10_peer,
            params=(res["doc_id"],),
        )
        == []
    )


# ---------------------------------------------------------------------------
# F08 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f08
def test_f08_bva_empty_allowed_group_ids_never_matches_empty_caller_groups(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="group",
        min_auth_level=1,
        allowed_group_ids=[],
    )
    caller = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["bob"],
        auth_level=10,
        dept_ids=(iam["dept_a"],),
        group_ids=(),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=caller,
            params=(res["doc_id"],),
        )
        == []
    )


@pytest.mark.tier2
@pytest.mark.f08
def test_f08_bva_group_binding_deny_overrides_group_binding_allow_for_same_user(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g1, g2 = str(uuid.uuid4()), str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    res_arn = f"arn:openrag:{iam['org_1']}:da:group:document:L1/{doc_id}"

    for gid in (g1, g2):
        api_client.studio_iam_request(
            "groups",
            method="POST",
            payload={"id": gid, "org_id": iam["org_1"], "slug": f"g-{gid[:6]}", "name": gid, "allowed_actions": ["read"]},
        ).assert_status((200, 201))
        api_client.studio_iam_request(
            f"groups/{gid}/members",
            method="POST",
            payload={"principal_id": iam["dave"]},
        ).assert_status((200, 201, 204))
    api_client.studio_iam_request(
        "resource-bindings",
        method="POST",
        payload={"org_id": iam["org_1"], "resource_type": "document", "resource_id": doc_id, "resource_arn": res_arn, "group_id": g1, "effect": "allow", "actions": ["read"]},
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        "resource-bindings",
        method="POST",
        payload={"org_id": iam["org_1"], "resource_type": "document", "resource_id": doc_id, "resource_arn": res_arn, "group_id": g2, "effect": "deny", "actions": ["read"]},
    ).assert_status((200, 201))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 3,
            "group_ids": [g1, g2],
            "allowed_group_ids": [g1],
            "denied_group_ids": [g2],
            "access_scope": "group",
            "min_auth_level": 1,
            "resource_type": "document",
            "resource_id": doc_id,
            "resource_arn": res_arn,
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is False


@pytest.mark.tier2
@pytest.mark.f08
def test_f08_bva_group_action_scoped_read_only_blocks_write_and_unmask(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g1 = str(uuid.uuid4())
    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={"id": g1, "org_id": iam["org_1"], "slug": f"srch-{g1[:6]}", "name": "SearchOnly", "allowed_actions": ["search"]},
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{g1}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    ).assert_status((200, 201, 204))

    base = {
        "org_id": iam["org_1"],
        "principal_id": iam["dave"],
        "auth_level": 3,
        "group_ids": [g1],
        "allowed_group_ids": [g1],
        "access_scope": "group",
        "min_auth_level": 1,
        "resource_arn": f"arn:openrag:{iam['org_1']}:da:group:document:L1/d1",
    }
    assert api_client.studio_iam_simulate(dict(base, action="search")).assert_status(200).json()["allowed"] is True
    assert api_client.studio_iam_simulate(dict(base, action="write")).assert_status(200).json()["allowed"] is False
    assert api_client.studio_iam_simulate(dict(base, action="pii:unmask")).assert_status(200).json()["allowed"] is False


@pytest.mark.tier2
@pytest.mark.f08
def test_f08_bva_unknown_or_other_org_group_uuid_ignored_fail_closed(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    valid_g = str(uuid.uuid4())
    other_org_g = str(uuid.uuid4())
    random_g = str(uuid.uuid4())
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="group",
        min_auth_level=1,
        allowed_group_ids=[valid_g],
    )
    ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["dave"],
        auth_level=3,
        dept_ids=(iam["dept_b"],),
        group_ids=(other_org_g, random_g),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=ctx,
            params=(res["doc_id"],),
        )
        == []
    )


@pytest.mark.tier2
@pytest.mark.f08
def test_f08_bva_removed_group_member_loses_access_immediately(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g1 = str(uuid.uuid4())
    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={"id": g1, "org_id": iam["org_1"], "slug": f"rm-{g1[:6]}", "name": "Removable", "allowed_actions": ["read"]},
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{g1}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    ).assert_status((200, 201, 204))

    payload = {
        "org_id": iam["org_1"],
        "principal_id": iam["dave"],
        "auth_level": 3,
        "allowed_group_ids": [g1],
        "access_scope": "group",
        "min_auth_level": 1,
        "resource_arn": f"arn:openrag:{iam['org_1']}:da:group:document:L1/doc-rm",
        "action": "read",
    }
    assert api_client.studio_iam_simulate(payload).assert_status(200).json()["allowed"] is True

    api_client.studio_iam_request(f"groups/{g1}/members/{iam['dave']}", method="DELETE").assert_status(
        (200, 204)
    )
    assert api_client.studio_iam_simulate(payload).assert_status(200).json()["allowed"] is False


# ---------------------------------------------------------------------------
# F09 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f09
def test_f09_bva_auth_level_off_by_one_below_min_auth_level_denied(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    l4_pid = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": l4_pid,
                        "org_id": iam["org_1"],
                        "email": f"l4-{l4_pid[:6]}@example.or.kr",
                        "display_name": "L4User",
                        "auth_level": 4,
                    },
                )
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="org",
        min_auth_level=5,
    )
    l4_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=l4_pid,
        auth_level=4,
        dept_ids=(iam["dept_b"],),
    )
    for tbl, key in (
        ("corpus.documents", "doc_id"),
        ("corpus.chunks", "chunk_id"),
        ("wiki.pages", "wiki_id"),
        ("memory.entries", "mem_id"),
    ):
        assert (
            migrated_pg.query_as_principal(
                f"SELECT id FROM {tbl} WHERE id = %s::uuid",
                principal=l4_ctx,
                params=(res[key],),
            )
            == []
        )


@pytest.mark.tier2
@pytest.mark.f09
def test_f09_bva_group_min_auth_level_higher_than_resource_denies_low_level_member(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g1 = str(uuid.uuid4())
    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={"id": g1, "org_id": iam["org_1"], "slug": f"gl7-{g1[:6]}", "name": "L7Group", "min_auth_level": 7, "allowed_actions": ["read"]},
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{g1}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    ).assert_status((200, 201, 204))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 3,
            "group_ids": [g1],
            "allowed_group_ids": [g1],
            "access_scope": "group",
            "min_auth_level": 2,
            "resource_arn": f"arn:openrag:{iam['org_1']}:da:group:document:L2/doc-low",
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is False


@pytest.mark.tier2
@pytest.mark.f09
def test_f09_bva_dept_admin_with_insufficient_auth_level_denied_on_l9_private_doc(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_doc_bundle(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=9,
    )
    low_admin = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["carol"],
        auth_level=4,
        dept_ids=(iam["dept_a"],),
        dept_admin_ids=(iam["dept_a"],),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=low_admin,
            params=(res["doc_id"],),
        )
        == []
    )


@pytest.mark.tier2
@pytest.mark.f09
def test_f09_bva_out_of_range_auth_level_0_and_11_rejected_db_and_api(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    for bad_lvl in (0, 11):
        with pytest.raises(ValueError):
            PgPrincipalContext(org_id=iam["org_1"], principal_id=iam["alice"], auth_level=bad_lvl).to_guc_map()

        with pytest.raises(psycopg.errors.CheckViolation):
            with psycopg.connect(migrated_pg.dsn()) as conn:
                with conn.cursor() as cur:
                    _dynamic_insert(
                        cur,
                        "iam",
                        "principals",
                        {
                            "id": str(uuid.uuid4()),
                            "org_id": iam["org_1"],
                            "email": f"badlvl-{bad_lvl}-{uuid.uuid4().hex[:4]}@example.or.kr",
                            "display_name": "BadLvl",
                            "auth_level": bad_lvl,
                        },
                    )

        resp = api_client.studio_iam_request(
            "principals",
            method="POST",
            payload={
                "org_id": iam["org_1"],
                "email": f"api-bad-{bad_lvl}-{uuid.uuid4().hex[:4]}@example.or.kr",
                "display_name": "BadApi",
                "auth_level": bad_lvl,
            },
        )
        assert resp.status in (400, 422)


@pytest.mark.tier2
@pytest.mark.f09
def test_f09_bva_api_key_max_auth_level_caps_principal_clearance(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    admin_pid = str(uuid.uuid4())
    api_client.studio_iam_request(
        "principals",
        method="POST",
        payload={
            "id": admin_pid,
            "org_id": iam["org_1"],
            "email": f"l9-{admin_pid[:6]}@example.or.kr",
            "display_name": "L9User",
            "auth_level": 9,
            "dept_ids": [iam["dept_a"]],
            "dept_admin_ids": [iam["dept_a"]],
        },
    ).assert_status((200, 201, 409))

    full_key = str(
        api_client.studio_iam_request(
            "api-keys",
            method="POST",
            payload={"org_id": iam["org_1"], "principal_id": admin_pid, "name": "full", "max_auth_level": 9},
        ).assert_status((200, 201)).json().get("token")
    )
    capped_key = str(
        api_client.studio_iam_request(
            "api-keys",
            method="POST",
            payload={"org_id": iam["org_1"], "principal_id": admin_pid, "name": "capped", "max_auth_level": 4},
        ).assert_status((200, 201)).json().get("token")
    )

    full_client = api_client.with_api_key(full_key)
    bkt = full_client.create_bucket(
        {"slug": f"f09-cap-{uuid.uuid4().hex[:6]}", "name": "Cap Bucket", "org_id": iam["org_1"], "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bkt_id = str(bkt.get("id") or bkt.get("bucket_id"))

    canary = f"CAP-L6-CANARY-{uuid.uuid4().hex[:6]}"
    full_client.create_document(
        bkt_id,
        {"title": "L6 Doc", "text": f"Classified {canary}", "access_scope": "org", "min_auth_level": 6},
    ).assert_status((200, 201))

    capped_res = (
        api_client.with_api_key(capped_key)
        .search_bucket(bkt_id, {"query": canary, "top_k": 5})
        .assert_status(200)
        .json()
    )
    assert canary not in str(capped_res)


# ---------------------------------------------------------------------------
# F10 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f10
def test_f10_bva_specific_arn_deny_overrides_wildcard_arn_allow(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    for effect, res_pat in (
        ("Allow", f"arn:openrag:{iam['org_1']}:*"),
        ("Deny", f"arn:openrag:{iam['org_1']}:hr:dept:document:*"),
    ):
        api_client.studio_iam_request(
            "policies",
            method="POST",
            payload={
                "org_id": iam["org_1"],
                "name": f"wc-{effect}-{uuid.uuid4().hex[:6]}",
                "document": {"statements": [{"effect": effect, "actions": ["read"], "resources": [res_pat]}]},
                "attachments": [{"target_type": "principal", "target_id": iam["dave"]}],
            },
        ).assert_status((200, 201))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 3,
            "access_scope": "org",
            "min_auth_level": 1,
            "resource_arn": f"arn:openrag:{iam['org_1']}:hr:dept:document:L5/doc1",
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is False
    assert len(sim.get("matched_denies") or []) >= 1


@pytest.mark.tier2
@pytest.mark.f10
def test_f10_bva_action_specific_deny_blocks_only_denied_action(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    unmask_pid = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": unmask_pid,
                        "org_id": iam["org_1"],
                        "email": f"unmask-{unmask_pid[:6]}@example.or.kr",
                        "display_name": "UnmaskUser",
                        "auth_level": 5,
                        "can_unmask_pii": True,
                    },
                )
    target_arn = f"arn:openrag:{iam['org_1']}:act:org:document:*"
    api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "name": f"act-allow-{uuid.uuid4().hex[:6]}",
            "document": {"statements": [{"effect": "Allow", "actions": ["read", "search", "pii:unmask"], "resources": [target_arn]}]},
            "attachments": [{"target_type": "principal", "target_id": unmask_pid}],
        },
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "name": f"act-deny-{uuid.uuid4().hex[:6]}",
            "document": {"statements": [{"effect": "Deny", "actions": ["pii:unmask"], "resources": [target_arn]}]},
            "attachments": [{"target_type": "principal", "target_id": unmask_pid}],
        },
    ).assert_status((200, 201))

    doc_arn = f"arn:openrag:{iam['org_1']}:act:org:document:L2/d1"
    base = {
        "org_id": iam["org_1"],
        "principal_id": unmask_pid,
        "auth_level": 5,
        "can_unmask_pii": True,
        "access_scope": "org",
        "min_auth_level": 1,
        "resource_arn": doc_arn,
    }
    assert api_client.studio_iam_simulate(dict(base, action="read")).assert_status(200).json()["allowed"] is True
    assert api_client.studio_iam_simulate(dict(base, action="pii:unmask")).assert_status(200).json()["allowed"] is False


@pytest.mark.tier2
@pytest.mark.f10
def test_f10_bva_dept_or_group_deny_overrides_direct_principal_allow(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g_deny = str(uuid.uuid4())
    doc_arn = f"arn:openrag:{iam['org_1']}:da:org:document:L2/{uuid.uuid4()}"

    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={"id": g_deny, "org_id": iam["org_1"], "slug": f"gd-{g_deny[:6]}", "name": "DenyGroup"},
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{g_deny}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    ).assert_status((200, 201, 204))
    api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "name": f"p-allow-{uuid.uuid4().hex[:6]}",
            "document": {"statements": [{"effect": "Allow", "actions": ["read"], "resources": [doc_arn]}]},
            "attachments": [{"target_type": "principal", "target_id": iam["dave"]}],
        },
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "name": f"g-deny-{uuid.uuid4().hex[:6]}",
            "document": {"statements": [{"effect": "Deny", "actions": ["read"], "resources": [doc_arn]}]},
            "attachments": [{"target_type": "group", "target_id": g_deny}],
        },
    ).assert_status((200, 201))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 3,
            "group_ids": [g_deny],
            "access_scope": "org",
            "min_auth_level": 1,
            "resource_arn": doc_arn,
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is False


@pytest.mark.tier2
@pytest.mark.f10
def test_f10_bva_simulator_fails_closed_on_unknown_principal_or_malformed_arn(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    r_unknown = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": "00000000-0000-0000-0000-000000009999",
            "auth_level": 5,
            "access_scope": "org",
            "resource_arn": f"arn:openrag:{iam['org_1']}:da:org:document:L1/d",
            "action": "read",
        }
    )
    assert r_unknown.status in (200, 400, 404)
    if r_unknown.status == 200:
        assert r_unknown.json()["allowed"] is False

    r_bad_arn = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 3,
            "access_scope": "org",
            "resource_arn": "not:an:arn",
            "action": "read",
        }
    )
    assert r_bad_arn.status in (200, 400, 422)
    if r_bad_arn.status == 200:
        assert r_bad_arn.json()["allowed"] is False


@pytest.mark.tier2
@pytest.mark.f10
def test_f10_bva_invalid_policy_document_or_missing_allow_fails_closed(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    bad_pol = api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "name": f"bad-pol-{uuid.uuid4().hex[:6]}",
            "document": {"statements": [{"effect": "Maybe", "actions": ["read"], "resources": ["*"]}]},
        },
    )
    assert bad_pol.status in (400, 422)

    sim_empty = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 3,
            "access_scope": "private",
            "min_auth_level": 1,
            "resource_arn": f"arn:openrag:{iam['org_1']}:da:private:document:L1/other",
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim_empty["allowed"] is False
