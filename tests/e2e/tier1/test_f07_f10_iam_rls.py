"""Tier 1 isolated happy-path E2E tests for F07-F10 (IAM, ARN, Permission Groups, Levels, Simulator)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.pg_harness import PgEphemeralCluster, PgPrincipalContext

pytestmark = pytest.mark.tier1


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
    dept_org2 = str(uuid.uuid4())
    alice = str(uuid.uuid4())  # owner in dept_a, L3
    bob = str(uuid.uuid4())    # peer in dept_a, L3
    carol = str(uuid.uuid4())  # dept_admin in dept_a, L3
    dave = str(uuid.uuid4())   # member in dept_b, L5
    eve = str(uuid.uuid4())    # member in org_2, L9
    bkt_1 = str(uuid.uuid4())

    with psycopg.connect(pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for oid, slug in ((org_1, f"org1-{org_1[:6]}"), (org_2, f"org2-{org_2[:6]}")):
                    _dynamic_insert(cur, "iam", "orgs", {"id": oid, "slug": slug, "name": slug})
                for did, oid, slug in (
                    (dept_a, org_1, f"da-{dept_a[:6]}"),
                    (dept_b, org_1, f"db-{dept_b[:6]}"),
                    (dept_org2, org_2, f"do2-{dept_org2[:6]}"),
                ):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "departments",
                        {"id": did, "org_id": oid, "slug": slug, "name": slug, "arn_path": f"org/{slug}"},
                    )
                for pid, oid, lvl, unmask in (
                    (alice, org_1, 3, False),
                    (bob, org_1, 3, False),
                    (carol, org_1, 3, True),
                    (dave, org_1, 5, False),
                    (eve, org_2, 9, True),
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
                            "can_unmask_pii": unmask,
                        },
                    )
                for pid, oid, did, role in (
                    (alice, org_1, dept_a, "member"),
                    (bob, org_1, dept_a, "member"),
                    (carol, org_1, dept_a, "dept_admin"),
                    (dave, org_1, dept_b, "member"),
                    (eve, org_2, dept_org2, "dept_admin"),
                ):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "dept_memberships",
                        {"org_id": oid, "principal_id": pid, "dept_id": did, "dept_role": role},
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
                        "slug": f"iam-bkt-{bkt_1[:6]}",
                        "name": "IAM Bucket",
                        "access_scope": "org",
                        "min_auth_level": 1,
                        "allowed_group_ids": [],
                        "arn": f"arn:openrag:{org_1}:dept-a:org:bucket:L1/{bkt_1}",
                    },
                )
    return {
        "org_1": org_1,
        "org_2": org_2,
        "dept_a": dept_a,
        "dept_b": dept_b,
        "dept_org2": dept_org2,
        "alice": alice,
        "bob": bob,
        "carol": carol,
        "dave": dave,
        "eve": eve,
        "bkt_1": bkt_1,
    }


def _seed_scoped_resources(
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
    scope_key = f"{org_id}:{dept_id}:{access_scope}:L{min_auth_level}"

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
                        "title": f"Doc {access_scope}",
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:dept-a:{access_scope}:document:L{min_auth_level}/{doc_id}",
                        "acl_scope_key": scope_key,
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
                        "arn": f"arn:openrag:{org_id}:dept-a:{access_scope}:page:L{min_auth_level}/{page_id}",
                        "acl_scope_key": scope_key,
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
                        "redacted_text": f"Secret {access_scope} chunk",
                        "bm25_text": f"Secret {access_scope} chunk",
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:dept-a:{access_scope}:chunk:L{min_auth_level}/{chunk_id}",
                        "acl_scope_key": scope_key,
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
                        "arn": f"arn:openrag:{org_id}:dept-a:{access_scope}:episode:L{min_auth_level}/{ep_id}",
                        "source_harness": "chat_ui",
                        "summary_md": f"Episode {access_scope}",
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
                        "arn": f"arn:openrag:{org_id}:dept-a:{access_scope}:memory:L{min_auth_level}/{mem_id}",
                        "acl_scope_key": scope_key,
                        "memory_kind": "fact",
                        "effective_text": f"Memory {access_scope}",
                        "bm25_text": f"Memory {access_scope}",
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
                        "slug": f"wiki-{wiki_id[:8]}",
                        "title": f"Wiki {access_scope}",
                        "body_md": f"# Wiki {access_scope}",
                        "access_scope": access_scope,
                        "min_auth_level": min_auth_level,
                        "allowed_group_ids": ag,
                        "denied_group_ids": dg,
                        "arn": f"arn:openrag:{org_id}:dept-a:{access_scope}:wiki:L{min_auth_level}/{wiki_id}",
                        "acl_scope_key": scope_key,
                    },
                )
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "ep_id": ep_id,
        "mem_id": mem_id,
        "wiki_id": wiki_id,
    }


def _create_api_key(
    api_client: Any,
    *,
    org_id: str,
    principal_id: str,
    auth_level: int = 5,
    max_auth_level: int | None = None,
    can_unmask_pii: bool = False,
    dept_ids: list[str] | None = None,
    dept_admin_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
) -> str:
    api_client.studio_iam_request(
        "principals",
        method="POST",
        payload={
            "id": principal_id,
            "org_id": org_id,
            "email": f"api-{principal_id[:8]}@example.or.kr",
            "display_name": f"User-{principal_id[:6]}",
            "auth_level": auth_level,
            "can_unmask_pii": can_unmask_pii,
            "dept_ids": dept_ids or [],
            "dept_admin_ids": dept_admin_ids or [],
            "group_ids": group_ids or [],
        },
    ).assert_status((200, 201, 409))

    key_resp = api_client.studio_iam_request(
        "api-keys",
        method="POST",
        payload={
            "org_id": org_id,
            "principal_id": principal_id,
            "name": f"key-{uuid.uuid4().hex[:6]}",
            "max_auth_level": max_auth_level or auth_level,
            "scoped_group_ids": group_ids or [],
        },
    ).assert_status((200, 201)).json()
    token = key_resp.get("token") or key_resp.get("api_key") or key_resp.get("key")
    assert isinstance(token, str) and token
    return token


# ---------------------------------------------------------------------------
# F07 — ARN & dept_admin vs peer Fail-Closed RLS
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f07
def test_f07_private_doc_and_memory_readable_by_owner(migrated_pg: PgEphemeralCluster) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=3,
    )
    alice_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["alice"],
        auth_level=3,
        dept_ids=(iam["dept_a"],),
    )
    for tbl, col in (
        ("corpus.documents", "doc_id"),
        ("corpus.chunks", "chunk_id"),
        ("memory.episodes", "ep_id"),
        ("memory.entries", "mem_id"),
    ):
        rows = migrated_pg.query_as_principal(
            f"SELECT id::text AS id FROM {tbl} WHERE id = %s::uuid",
            principal=alice_ctx,
            params=(res[col],),
        )
        assert len(rows) == 1, f"Owner Alice could not read private {tbl}"


@pytest.mark.tier1
@pytest.mark.f07
def test_f07_private_doc_and_memory_readable_by_same_dept_admin(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=3,
    )
    carol_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["carol"],
        auth_level=3,
        dept_ids=(iam["dept_a"],),
        dept_admin_ids=(iam["dept_a"],),
    )
    for tbl, col in (
        ("corpus.documents", "doc_id"),
        ("corpus.chunks", "chunk_id"),
        ("memory.episodes", "ep_id"),
        ("memory.entries", "mem_id"),
    ):
        rows = migrated_pg.query_as_principal(
            f"SELECT id::text AS id FROM {tbl} WHERE id = %s::uuid",
            principal=carol_ctx,
            params=(res[col],),
        )
        assert len(rows) == 1, f"dept_admin Carol could not read private {tbl}"


@pytest.mark.tier1
@pytest.mark.f07
def test_f07_private_doc_and_memory_strictly_blocked_for_same_dept_peer(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="private",
        min_auth_level=3,
    )
    bob_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["bob"],
        auth_level=3,
        dept_ids=(iam["dept_a"],),
        dept_admin_ids=(),
    )
    for tbl, col in (
        ("corpus.documents", "doc_id"),
        ("corpus.chunks", "chunk_id"),
        ("memory.episodes", "ep_id"),
        ("memory.entries", "mem_id"),
        ("wiki.pages", "wiki_id"),
    ):
        rows = migrated_pg.query_as_principal(
            f"SELECT id::text AS id FROM {tbl} WHERE id = %s::uuid",
            principal=bob_ctx,
            params=(res[col],),
        )
        assert rows == [], f"Peer Bob leaked private row from {tbl}: {rows}"


@pytest.mark.tier1
@pytest.mark.f07
def test_f07_dept_and_org_scope_isolation_across_depts_and_orgs(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    dept_res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="dept",
        min_auth_level=2,
    )
    org_res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="org",
        min_auth_level=2,
    )

    bob_ctx = PgPrincipalContext(
        org_id=iam["org_1"], principal_id=iam["bob"], auth_level=3, dept_ids=(iam["dept_a"],)
    )
    dave_ctx = PgPrincipalContext(
        org_id=iam["org_1"], principal_id=iam["dave"], auth_level=5, dept_ids=(iam["dept_b"],)
    )
    eve_ctx = PgPrincipalContext(
        org_id=iam["org_2"],
        principal_id=iam["eve"],
        auth_level=9,
        dept_ids=(iam["dept_org2"],),
        dept_admin_ids=(iam["dept_org2"],),
    )

    assert len(
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=bob_ctx,
            params=(dept_res["doc_id"],),
        )
    ) == 1
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=dave_ctx,
            params=(dept_res["doc_id"],),
        )
        == []
    )
    assert len(
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=dave_ctx,
            params=(org_res["doc_id"],),
        )
    ) == 1
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=eve_ctx,
            params=(org_res["doc_id"],),
        )
        == []
    )


@pytest.mark.tier1
@pytest.mark.f07
def test_f07_arn_format_and_http_bucket_search_enforces_dept_admin_vs_peer(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    alice_key = _create_api_key(
        api_client,
        org_id=iam["org_1"],
        principal_id=iam["alice"],
        auth_level=3,
        dept_ids=[iam["dept_a"]],
    )
    bob_key = _create_api_key(
        api_client,
        org_id=iam["org_1"],
        principal_id=iam["bob"],
        auth_level=3,
        dept_ids=[iam["dept_a"]],
    )
    carol_key = _create_api_key(
        api_client,
        org_id=iam["org_1"],
        principal_id=iam["carol"],
        auth_level=3,
        dept_ids=[iam["dept_a"]],
        dept_admin_ids=[iam["dept_a"]],
    )

    alice_client = api_client.with_api_key(alice_key)
    bkt = alice_client.create_bucket(
        {
            "slug": f"f07-priv-{uuid.uuid4().hex[:6]}",
            "name": "Private Search Bucket",
            "org_id": iam["org_1"],
            "dept_id": iam["dept_a"],
            "access_scope": "org",
        }
    ).assert_status((200, 201)).json()
    bucket_id = str(bkt.get("id") or bkt.get("bucket_id"))

    token = f"F07-PRIVATE-CANARY-{uuid.uuid4().hex[:8]}"
    doc = alice_client.create_document(
        bucket_id,
        {
            "title": "Alice Private Memo",
            "text": f"Confidential private memo {token}",
            "access_scope": "private",
            "min_auth_level": 3,
        },
    ).assert_status((200, 201)).json()
    arn = doc.get("arn")
    if isinstance(arn, str):
        assert arn.startswith("arn:openrag:")

    alice_hits = (
        alice_client.search_bucket(bucket_id, {"query": token, "top_k": 5})
        .assert_status(200)
        .json()
    )
    assert len(alice_hits.get("results") or alice_hits.get("hits") or []) >= 1

    carol_hits = (
        api_client.with_api_key(carol_key)
        .search_bucket(bucket_id, {"query": token, "top_k": 5})
        .assert_status(200)
        .json()
    )
    assert len(carol_hits.get("results") or carol_hits.get("hits") or []) >= 1

    bob_hits = (
        api_client.with_api_key(bob_key)
        .search_bucket(bucket_id, {"query": token, "top_k": 5})
        .assert_status(200)
        .json()
    )
    assert len(bob_hits.get("results") or bob_hits.get("hits") or []) == 0


# ---------------------------------------------------------------------------
# F08 — First-Class Permission Groups
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f08
def test_f08_cross_dept_tf_group_grants_read_on_group_scoped_resource(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    tf_group = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "groups",
                    {
                        "id": tf_group,
                        "org_id": iam["org_1"],
                        "dept_id": None,
                        "slug": f"tf-ai-{tf_group[:6]}",
                        "name": "Cross-Dept AI TF",
                        "min_auth_level": 1,
                        "max_auth_level": 10,
                        "allowed_actions": ["read", "search"],
                    },
                )
                for pid in (iam["alice"], iam["dave"]):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "group_members",
                        {"org_id": iam["org_1"], "group_id": tf_group, "principal_id": pid},
                    )

    res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="group",
        min_auth_level=2,
        allowed_group_ids=[tf_group],
    )

    dave_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["dave"],
        auth_level=5,
        dept_ids=(iam["dept_b"],),
        group_ids=(tf_group,),
    )
    bob_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["bob"],
        auth_level=3,
        dept_ids=(iam["dept_a"],),
        group_ids=(),
    )

    for tbl, col in (
        ("corpus.documents", "doc_id"),
        ("wiki.pages", "wiki_id"),
        ("memory.entries", "mem_id"),
    ):
        assert len(
            migrated_pg.query_as_principal(
                f"SELECT id FROM {tbl} WHERE id = %s::uuid",
                principal=dave_ctx,
                params=(res[col],),
            )
        ) == 1
        assert (
            migrated_pg.query_as_principal(
                f"SELECT id FROM {tbl} WHERE id = %s::uuid",
                principal=bob_ctx,
                params=(res[col],),
            )
            == []
        )


@pytest.mark.tier1
@pytest.mark.f08
def test_f08_studio_iam_group_crud_and_membership_management(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    grp_slug = f"tf-sec-{uuid.uuid4().hex[:6]}"
    created = api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "slug": grp_slug,
            "name": "Security TF",
            "min_auth_level": 3,
            "max_auth_level": 10,
            "allowed_actions": ["read", "search"],
        },
    ).assert_status((200, 201)).json()
    grp_id = str(created.get("id") or created.get("group_id"))
    assert grp_id

    listed = api_client.studio_iam_request("groups").assert_status(200).json()
    items = listed if isinstance(listed, list) else (listed.get("groups") or listed.get("items") or [])
    assert any(str(g.get("id") or g.get("group_id")) == grp_id for g in items)

    add_resp = api_client.studio_iam_request(
        f"groups/{grp_id}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    )
    add_resp.assert_status((200, 201, 204))


@pytest.mark.tier1
@pytest.mark.f08
def test_f08_resource_group_binding_allow_grants_access_with_specific_actions(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    grp_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    res_arn = f"arn:openrag:{iam['org_1']}:dept-a:group:document:L2/{doc_id}"

    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={
            "id": grp_id,
            "org_id": iam["org_1"],
            "slug": f"grp-bind-{grp_id[:6]}",
            "name": "Binding Group",
            "min_auth_level": 1,
            "allowed_actions": ["read", "search"],
        },
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{grp_id}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    ).assert_status((200, 201, 204))

    api_client.studio_iam_request(
        "resource-bindings",
        method="POST",
        payload={
            "org_id": iam["org_1"],
            "resource_type": "document",
            "resource_id": doc_id,
            "resource_arn": res_arn,
            "group_id": grp_id,
            "effect": "allow",
            "actions": ["read", "search"],
        },
    ).assert_status((200, 201))

    for act in ("read", "search"):
        sim = api_client.studio_iam_simulate(
            {
                "org_id": iam["org_1"],
                "principal_id": iam["dave"],
                "auth_level": 5,
                "dept_ids": [iam["dept_b"]],
                "group_ids": [grp_id],
                "resource_arn": res_arn,
                "resource_type": "document",
                "resource_id": doc_id,
                "access_scope": "group",
                "min_auth_level": 2,
                "allowed_group_ids": [grp_id],
                "action": act,
            }
        ).assert_status(200).json()
        assert sim["allowed"] is True
        assert act in (sim.get("effective_actions") or [])


@pytest.mark.tier1
@pytest.mark.f08
def test_f08_group_allows_pii_unmask_only_when_granted(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    grp_unmask = str(uuid.uuid4())
    grp_read_only = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    res_arn = f"arn:openrag:{iam['org_1']}:dept-a:group:document:L3/{doc_id}"

    for gid, actions in (
        (grp_unmask, ["read", "search", "pii:unmask"]),
        (grp_read_only, ["read", "search"]),
    ):
        api_client.studio_iam_request(
            "groups",
            method="POST",
            payload={
                "id": gid,
                "org_id": iam["org_1"],
                "slug": f"grp-{gid[:6]}",
                "name": f"Grp-{gid[:6]}",
                "min_auth_level": 1,
                "allowed_actions": actions,
            },
        ).assert_status((200, 201))
        api_client.studio_iam_request(
            f"groups/{gid}/members",
            method="POST",
            payload={"principal_id": iam["carol"]},
        ).assert_status((200, 201, 204))

    sim_ok = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["carol"],
            "auth_level": 3,
            "can_unmask_pii": True,
            "group_ids": [grp_unmask],
            "allowed_group_ids": [grp_unmask, grp_read_only],
            "resource_arn": res_arn,
            "access_scope": "group",
            "min_auth_level": 3,
            "action": "pii:unmask",
        }
    ).assert_status(200).json()
    assert sim_ok["allowed"] is True

    sim_denied = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["carol"],
            "auth_level": 3,
            "can_unmask_pii": True,
            "group_ids": [grp_read_only],
            "allowed_group_ids": [grp_unmask, grp_read_only],
            "resource_arn": res_arn,
            "access_scope": "group",
            "min_auth_level": 3,
            "action": "pii:unmask",
        }
    ).assert_status(200).json()
    assert sim_denied["allowed"] is False


@pytest.mark.tier1
@pytest.mark.f08
def test_f08_multi_group_principal_context_matches_any_allowed_group(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g1 = str(uuid.uuid4())
    g2 = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for gid, label in ((g1, "MultiGrp1"), (g2, "MultiGrp2")):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "groups",
                        {
                            "id": gid,
                            "org_id": iam["org_1"],
                            "dept_id": None,
                            "name": f"{label}-{gid[:6]}",
                            "min_auth_level": 1,
                            "max_auth_level": 10,
                            "allowed_actions": ["read", "search"],
                        },
                    )
                    _dynamic_insert(
                        cur,
                        "iam",
                        "group_members",
                        {"org_id": iam["org_1"], "group_id": gid, "principal_id": iam["dave"]},
                    )
    res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="group",
        min_auth_level=1,
        allowed_group_ids=[g2],
    )
    ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["dave"],
        auth_level=5,
        dept_ids=(iam["dept_b"],),
        group_ids=(g1, g2),
    )
    rows = migrated_pg.query_as_principal(
        "SELECT id FROM corpus.documents WHERE id = %s::uuid",
        principal=ctx,
        params=(res["doc_id"],),
    )
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# F09 — First-Class Permission Levels (auth_level 1..10)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f09
def test_f09_exact_min_auth_level_match_grants_access(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    for lvl in (1, 5, 10):
        pid = str(uuid.uuid4())
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    _dynamic_insert(
                        cur,
                        "iam",
                        "principals",
                        {
                            "id": pid,
                            "org_id": iam["org_1"],
                            "email": f"l{lvl}-{pid[:6]}@example.or.kr",
                            "display_name": f"L{lvl}",
                            "auth_level": lvl,
                        },
                    )
        res = _seed_scoped_resources(
            migrated_pg,
            org_id=iam["org_1"],
            dept_id=iam["dept_a"],
            bucket_id=iam["bkt_1"],
            owner_id=iam["alice"],
            access_scope="org",
            min_auth_level=lvl,
        )
        ctx = PgPrincipalContext(
            org_id=iam["org_1"],
            principal_id=pid,
            auth_level=lvl,
            dept_ids=(iam["dept_b"],),
        )
        rows = migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=ctx,
            params=(res["doc_id"],),
        )
        assert len(rows) == 1, f"Exact auth_level={lvl} should read min_auth_level={lvl}"


@pytest.mark.tier1
@pytest.mark.f09
def test_f09_higher_auth_level_clears_lower_min_auth_level(
    migrated_pg: PgEphemeralCluster,
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    l8_member = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": l8_member,
                        "org_id": iam["org_1"],
                        "email": f"l8-{l8_member[:6]}@example.or.kr",
                        "display_name": "L8Member",
                        "auth_level": 8,
                    },
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "dept_memberships",
                    {
                        "org_id": iam["org_1"],
                        "principal_id": l8_member,
                        "dept_id": iam["dept_a"],
                        "dept_role": "member",
                    },
                )
    high_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=l8_member,
        auth_level=8,
        dept_ids=(iam["dept_a"],),
    )
    for scope, min_lvl in (("org", 3), ("dept", 7)):
        res = _seed_scoped_resources(
            migrated_pg,
            org_id=iam["org_1"],
            dept_id=iam["dept_a"],
            bucket_id=iam["bkt_1"],
            owner_id=iam["alice"],
            access_scope=scope,
            min_auth_level=min_lvl,
        )
        rows = migrated_pg.query_as_principal(
            "SELECT id FROM corpus.chunks WHERE id = %s::uuid",
            principal=high_ctx,
            params=(res["chunk_id"],),
        )
        assert len(rows) == 1


@pytest.mark.tier1
@pytest.mark.f09
def test_f09_group_level_threshold_evaluated_with_membership(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    l8_member = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": l8_member,
                        "org_id": iam["org_1"],
                        "email": f"l8-grp-{l8_member[:6]}@example.or.kr",
                        "display_name": "L8GroupMember",
                        "auth_level": 8,
                    },
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "dept_memberships",
                    {
                        "org_id": iam["org_1"],
                        "principal_id": l8_member,
                        "dept_id": iam["dept_b"],
                        "dept_role": "member",
                    },
                )

    grp_id = str(uuid.uuid4())
    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={
            "id": grp_id,
            "org_id": iam["org_1"],
            "slug": f"grp-l6-{grp_id[:6]}",
            "name": "Level 6 TF",
            "min_auth_level": 6,
            "allowed_actions": ["read", "search"],
        },
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{grp_id}/members",
        method="POST",
        payload={"principal_id": l8_member},
    ).assert_status((200, 201, 204))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": l8_member,
            "auth_level": 6,
            "group_ids": [grp_id],
            "allowed_group_ids": [grp_id],
            "access_scope": "group",
            "min_auth_level": 4,
            "resource_arn": f"arn:openrag:{iam['org_1']}:dept-a:group:document:L4/doc1",
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is True
    eff_lvl = sim.get("effective_min_level", sim.get("effective_level"))
    assert int(eff_lvl) >= 6


@pytest.mark.tier1
@pytest.mark.f09
def test_f09_policy_attachment_level_override_enforced(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    l8_member = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": l8_member,
                        "org_id": iam["org_1"],
                        "email": f"l8-pol-{l8_member[:6]}@example.or.kr",
                        "display_name": "L8PolicyUser",
                        "auth_level": 8,
                    },
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "dept_memberships",
                    {
                        "org_id": iam["org_1"],
                        "principal_id": l8_member,
                        "dept_id": iam["dept_b"],
                        "dept_role": "member",
                    },
                )

    pol_id = str(uuid.uuid4())
    target_arn = f"arn:openrag:{iam['org_1']}:dept-a:dept:document:*"

    api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "id": pol_id,
            "org_id": iam["org_1"],
            "name": f"pol-l7-{pol_id[:6]}",
            "document": {
                "statements": [
                    {
                        "effect": "Allow",
                        "actions": ["read"],
                        "resources": [target_arn],
                    }
                ]
            },
            "attachments": [
                {
                    "target_type": "principal",
                    "target_id": l8_member,
                    "min_auth_level_override": 7,
                }
            ],
        },
    ).assert_status((200, 201))

    doc_arn = f"arn:openrag:{iam['org_1']}:dept-a:dept:document:L2/doc-x"
    allow_sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": l8_member,
            "auth_level": 7,
            "dept_ids": [iam["dept_b"]],
            "access_scope": "dept",
            "min_auth_level": 2,
            "resource_arn": doc_arn,
            "action": "read",
        }
    ).assert_status(200).json()
    assert allow_sim["allowed"] is True

    deny_sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": l8_member,
            "auth_level": 6,
            "dept_ids": [iam["dept_b"]],
            "access_scope": "dept",
            "min_auth_level": 2,
            "resource_arn": doc_arn,
            "action": "read",
        }
    ).assert_status(200).json()
    assert deny_sim["allowed"] is False


@pytest.mark.tier1
@pytest.mark.f09
def test_f09_http_bucket_search_filters_by_caller_auth_level(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    l10_admin = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": l10_admin,
                        "org_id": iam["org_1"],
                        "email": f"l10-admin-{l10_admin[:6]}@example.or.kr",
                        "display_name": "L10Admin",
                        "auth_level": 10,
                        "can_unmask_pii": True,
                    },
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "dept_memberships",
                    {
                        "org_id": iam["org_1"],
                        "principal_id": l10_admin,
                        "dept_id": iam["dept_a"],
                        "dept_role": "dept_admin",
                    },
                )

    admin_key = _create_api_key(
        api_client,
        org_id=iam["org_1"],
        principal_id=l10_admin,
        auth_level=10,
        can_unmask_pii=True,
        dept_ids=[iam["dept_a"]],
        dept_admin_ids=[iam["dept_a"]],
    )
    l5_key = _create_api_key(
        api_client,
        org_id=iam["org_1"],
        principal_id=iam["dave"],
        auth_level=5,
        dept_ids=[iam["dept_b"]],
    )

    admin_client = api_client.with_api_key(admin_key)
    bkt = admin_client.create_bucket(
        {
            "slug": f"f09-lvl-{uuid.uuid4().hex[:6]}",
            "name": "Level Filter Bucket",
            "org_id": iam["org_1"],
            "access_scope": "org",
        }
    ).assert_status((200, 201)).json()
    bucket_id = str(bkt.get("id") or bkt.get("bucket_id"))

    shared_token = f"F09LEVELTOKEN{uuid.uuid4().hex[:6]}"
    admin_client.create_document(
        bucket_id,
        {
            "title": "Low L2 Doc",
            "text": f"Public internal {shared_token} LOW_L2",
            "access_scope": "org",
            "min_auth_level": 2,
        },
    ).assert_status((200, 201))
    admin_client.create_document(
        bucket_id,
        {
            "title": "Secret L8 Doc",
            "text": f"Restricted secret {shared_token} HIGH_L8",
            "access_scope": "org",
            "min_auth_level": 8,
        },
    ).assert_status((200, 201))

    res = (
        api_client.with_api_key(l5_key)
        .search_bucket(bucket_id, {"query": shared_token, "top_k": 10})
        .assert_status(200)
        .json()
    )
    hits_text = str(res)
    assert "LOW_L2" in hits_text
    assert "HIGH_L8" not in hits_text


# ---------------------------------------------------------------------------
# F10 — Explicit Deny > Allow & 3-Axis Policy Simulator
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f10
def test_f10_arn_policy_explicit_allow_grants_access(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    pol_id = str(uuid.uuid4())
    target_arn = f"arn:openrag:{iam['org_1']}:dept-a:dept:document:*"

    api_client.studio_iam_request(
        "policies",
        method="POST",
        payload={
            "id": pol_id,
            "org_id": iam["org_1"],
            "name": f"allow-ext-{pol_id[:6]}",
            "document": {
                "statements": [
                    {"effect": "Allow", "actions": ["read"], "resources": [target_arn]}
                ]
            },
            "attachments": [{"target_type": "principal", "target_id": iam["dave"]}],
        },
    ).assert_status((200, 201))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 5,
            "dept_ids": [iam["dept_b"]],
            "access_scope": "dept",
            "min_auth_level": 2,
            "resource_arn": f"arn:openrag:{iam['org_1']}:dept-a:dept:document:L2/doc-allow",
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is True
    assert len(sim.get("matched_allows") or []) >= 1
    assert (sim.get("matched_denies") or []) == []


@pytest.mark.tier1
@pytest.mark.f10
def test_f10_explicit_arn_deny_overrides_arn_allow_and_org_scope(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    doc_id = str(uuid.uuid4())
    doc_arn = f"arn:openrag:{iam['org_1']}:dept-a:org:document:L2/{doc_id}"

    for effect, name in (("Allow", "allow-all-org"), ("Deny", "deny-specific-doc")):
        api_client.studio_iam_request(
            "policies",
            method="POST",
            payload={
                "org_id": iam["org_1"],
                "name": f"{name}-{uuid.uuid4().hex[:6]}",
                "document": {
                    "statements": [
                        {
                            "effect": effect,
                            "actions": ["read"],
                            "resources": [doc_arn if effect == "Deny" else f"arn:openrag:{iam['org_1']}:*"],
                        }
                    ]
                },
                "attachments": [{"target_type": "principal", "target_id": iam["dave"]}],
            },
        ).assert_status((200, 201))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 5,
            "access_scope": "org",
            "min_auth_level": 1,
            "resource_arn": doc_arn,
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is False
    assert len(sim.get("matched_denies") or []) >= 1


@pytest.mark.tier1
@pytest.mark.f10
def test_f10_explicit_group_deny_overrides_group_allow_and_dept_scope(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    g_allow = str(uuid.uuid4())
    g_deny = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for gid, label in ((g_allow, "AllowGrp"), (g_deny, "DenyGrp")):
                    _dynamic_insert(
                        cur,
                        "iam",
                        "groups",
                        {
                            "id": gid,
                            "org_id": iam["org_1"],
                            "name": f"{label}-{gid[:6]}",
                            "min_auth_level": 1,
                            "max_auth_level": 10,
                            "allowed_actions": ["read", "search"],
                        },
                    )
                    _dynamic_insert(
                        cur,
                        "iam",
                        "group_members",
                        {"org_id": iam["org_1"], "group_id": gid, "principal_id": iam["bob"]},
                    )
    res = _seed_scoped_resources(
        migrated_pg,
        org_id=iam["org_1"],
        dept_id=iam["dept_a"],
        bucket_id=iam["bkt_1"],
        owner_id=iam["alice"],
        access_scope="dept",
        min_auth_level=1,
        allowed_group_ids=[g_allow],
        denied_group_ids=[g_deny],
    )

    bob_ctx = PgPrincipalContext(
        org_id=iam["org_1"],
        principal_id=iam["bob"],
        auth_level=3,
        dept_ids=(iam["dept_a"],),
        group_ids=(g_allow, g_deny),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.documents WHERE id = %s::uuid",
            principal=bob_ctx,
            params=(res["doc_id"],),
        )
        == []
    )

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["bob"],
            "auth_level": 3,
            "dept_ids": [iam["dept_a"]],
            "group_ids": [g_allow, g_deny],
            "allowed_group_ids": [g_allow],
            "denied_group_ids": [g_deny],
            "access_scope": "dept",
            "min_auth_level": 1,
            "resource_arn": f"arn:openrag:{iam['org_1']}:dept-a:dept:document:L1/{res['doc_id']}",
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim["allowed"] is False


@pytest.mark.tier1
@pytest.mark.f10
def test_f10_studio_iam_simulator_returns_complete_3_axis_trace(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    grp_id = str(uuid.uuid4())
    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={
            "id": grp_id,
            "org_id": iam["org_1"],
            "slug": f"trace-{grp_id[:6]}",
            "name": "Trace Group",
            "min_auth_level": 4,
            "allowed_actions": ["read", "search"],
        },
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{grp_id}/members",
        method="POST",
        payload={"principal_id": iam["dave"]},
    ).assert_status((200, 201, 204))

    sim = api_client.studio_iam_simulate(
        {
            "org_id": iam["org_1"],
            "principal_id": iam["dave"],
            "auth_level": 5,
            "group_ids": [grp_id],
            "allowed_group_ids": [grp_id],
            "access_scope": "group",
            "min_auth_level": 3,
            "resource_arn": f"arn:openrag:{iam['org_1']}:dept-a:group:document:L3/doc-trace",
            "action": "read",
        }
    ).assert_status(200).json()

    assert sim["allowed"] is True
    assert isinstance(sim.get("reason"), str) and sim["reason"]
    assert isinstance(sim.get("matched_denies"), list)
    assert isinstance(sim.get("matched_allows"), list)
    assert ("effective_level" in sim) or ("effective_min_level" in sim)
    assert isinstance(sim.get("effective_actions"), list)


@pytest.mark.tier1
@pytest.mark.f10
def test_f10_combined_3_axis_allow_requires_arn_group_and_level_together(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    iam = _seed_iam_fixture(migrated_pg)
    l8_member = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _dynamic_insert(
                    cur,
                    "iam",
                    "principals",
                    {
                        "id": l8_member,
                        "org_id": iam["org_1"],
                        "email": f"l8-combo-{l8_member[:6]}@example.or.kr",
                        "display_name": "L8ComboMember",
                        "auth_level": 8,
                    },
                )
                _dynamic_insert(
                    cur,
                    "iam",
                    "dept_memberships",
                    {
                        "org_id": iam["org_1"],
                        "principal_id": l8_member,
                        "dept_id": iam["dept_b"],
                        "dept_role": "member",
                    },
                )

    g1 = str(uuid.uuid4())
    doc_arn = f"arn:openrag:{iam['org_1']}:dept-a:group:document:L6/{uuid.uuid4()}"

    api_client.studio_iam_request(
        "groups",
        method="POST",
        payload={
            "id": g1,
            "org_id": iam["org_1"],
            "slug": f"combo-{g1[:6]}",
            "name": "Combo Group",
            "min_auth_level": 6,
            "allowed_actions": ["read"],
        },
    ).assert_status((200, 201))
    api_client.studio_iam_request(
        f"groups/{g1}/members",
        method="POST",
        payload={"principal_id": l8_member},
    ).assert_status((200, 201, 204))

    base = {
        "org_id": iam["org_1"],
        "principal_id": l8_member,
        "auth_level": 6,
        "group_ids": [g1],
        "allowed_group_ids": [g1],
        "access_scope": "group",
        "min_auth_level": 6,
        "resource_arn": doc_arn,
        "action": "read",
    }
    assert api_client.studio_iam_simulate(base).assert_status(200).json()["allowed"] is True

    no_group = dict(base, group_ids=[])
    assert api_client.studio_iam_simulate(no_group).assert_status(200).json()["allowed"] is False

    low_level = dict(base, auth_level=5)
    assert api_client.studio_iam_simulate(low_level).assert_status(200).json()["allowed"] is False

    wrong_action = dict(base, action="write")
    assert api_client.studio_iam_simulate(wrong_action).assert_status(200).json()["allowed"] is False
