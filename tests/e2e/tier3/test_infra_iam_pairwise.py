"""Tier 3 pairwise E2E tests: Infra, DB Fallback, 128d Vectors, IAM/ARN/Groups/Levels (6 tests)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import psycopg
import pytest
import yaml

from conftest import fail_missing_target
from lib import (
    CliBinaryNotFoundError,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PgPrincipalContext,
)
from tier3._helpers import (
    ensure_migrated_all,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_org_with_users,
    sha256_hex,
    unit_vec_str,
)

pytestmark = [pytest.mark.tier3]

_TEN_SCHEMAS = (
    "iam",
    "vectors",
    "corpus",
    "pii",
    "memory",
    "wiki",
    "indexing",
    "gateway",
    "plugins",
    "eval",
)


@pytest.mark.f01
@pytest.mark.f03
@pytest.mark.f28
def test_t3_f01_f03_repo_clean_configs_validate_through_cli_and_studio(
    openrag_root: Path,
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    lic = openrag_root / "LICENSE"
    cargo_toml = openrag_root / "Cargo.toml"
    default_yaml = openrag_root / "configs" / "pipeline.default.yaml"
    airgap_yaml = openrag_root / "configs" / "pipeline.airgap.yaml"
    space_yaml = openrag_root / "configs" / "search_space.example.yaml"
    for req in (lic, cargo_toml, default_yaml, airgap_yaml, space_yaml):
        if not req.is_file():
            fail_missing_target(f"Missing required repo config/file: {req}")

    assert "Apache License" in lic.read_text(encoding="utf-8")
    forbidden_a = "-".join(("schift", "db"))
    forbidden_b = "-".join(("schift", "rag", "ce"))
    cargo_text = cargo_toml.read_text(encoding="utf-8").lower()
    assert forbidden_a not in cargo_text and forbidden_b not in cargo_text

    default_cfg = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
    airgap_cfg = yaml.safe_load(airgap_yaml.read_text(encoding="utf-8"))
    space_cfg = yaml.safe_load(space_yaml.read_text(encoding="utf-8"))
    assert default_cfg["thumbnail"]["dim"] == 128
    assert airgap_cfg["llm"]["local_only_min_auth_level"] == 1
    assert space_cfg["security_gates"]["require_zero_pii_leak"] is True

    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    help_res = openrag_cli.run(["--help"], check=True)
    assert "pipeline" in help_res.stdout

    ctx = seed_org_with_users(migrated_pg, label="T3F01")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-f01")
    client = api_client.with_api_key(key)

    pipe_get = client.studio_pipeline_get().assert_status(200).json()
    assert pipe_get is not None
    pipe_test = client.studio_pipeline_test(
        {"pipeline_yaml": default_yaml.read_text(encoding="utf-8"), "query": "출장 규정"}
    ).assert_status(200).json()
    assert isinstance(pipe_test, dict)


@pytest.mark.f02
@pytest.mark.f04
def test_t3_f02_f04_docker_compose_and_ddl_10_schemas_alignment(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
) -> None:
    compose_path = openrag_root / "docker" / "docker-compose.yml"
    dockerfile_path = openrag_root / "docker" / "postgres" / "Dockerfile"
    for req in (compose_path, dockerfile_path):
        if not req.is_file():
            fail_missing_target(f"Missing Docker artifact: {req}")

    compose_data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    pg_svc = compose_data["services"]["postgres"]
    assert str(pg_svc["build"]["args"]["PG_MAJOR"]) in ("17", "18")
    env_map = pg_svc["environment"]
    assert "POSTGRES_DB" in env_map and "POSTGRES_USER" in env_map
    assert any("migrations:/docker-entrypoint-initdb.d" in str(v) for v in pg_svc["volumes"])

    df_text = dockerfile_path.read_text(encoding="utf-8")
    assert "pg_pinecone" in df_text and "pg_search" in df_text

    ensure_migrated_all(pg_ephemeral, openrag_root)
    with psycopg.connect(pg_ephemeral.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name = ANY(%s)
                """,
                (list(_TEN_SCHEMAS),),
            )
            found_schemas = {row[0] for row in cur.fetchall()}
            assert found_schemas == set(_TEN_SCHEMAS), found_schemas

            cur.execute(
                """
                SELECT table_schema || '.' || table_name
                FROM information_schema.tables
                WHERE table_schema = ANY(%s) AND table_type = 'BASE TABLE'
                """,
                (list(_TEN_SCHEMAS),),
            )
            tables = {row[0] for row in cur.fetchall()}
            for expected_table in (
                "iam.orgs",
                "iam.groups",
                "iam.resource_group_bindings",
                "vectors.index_bindings",
                "corpus.buckets",
                "corpus.documents",
                "corpus.pages",
                "corpus.page_figures",
                "corpus.chunks",
                "pii.guard_policies",
                "pii.guard_rules",
                "pii.redaction_jobs",
                "memory.episodes",
                "memory.entries",
                "memory.patches",
                "memory.edges",
                "wiki.pages",
                "wiki.concepts",
                "wiki.concept_aliases",
                "indexing.outbox",
                "indexing.watermarks",
                "indexing.quarantine",
                "gateway.providers",
                "plugins.connectors",
                "eval.golden_sets",
                "eval.bench_runs",
            ):
                assert expected_table in tables, f"Missing table {expected_table}"


@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f06
def test_t3_f04_f05_f06_capability_fallback_preserves_128d_vectors_and_rls(
    migrated_pg: PgEphemeralCluster,
) -> None:
    caps = migrated_pg.probe_capabilities()
    assert caps["vector"] is True
    assert caps["pg_trgm"] is True
    assert caps["pgcrypto"] is True

    ctx = seed_org_with_users(migrated_pg, label="T3F06")
    other_org = seed_org_with_users(migrated_pg, label="T3F06OUT")

    bucket_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    fig_id = str(uuid.uuid4())
    chunk_near_id = str(uuid.uuid4())
    chunk_far_id = str(uuid.uuid4())
    wiki_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    acl_key = f"{ctx['org_id']}:{ctx['dept1_id']}:dept:L3"

    vec_target = unit_vec_str(0)
    vec_far = unit_vec_str(17)

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
                    VALUES (%s, %s, %s, %s, %s, 'VecBucket', 'dept', 1, %s)
                    """,
                    (
                        bucket_id,
                        ctx["org_id"],
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"vb-{bucket_id[:6]}",
                        f"arn:openrag:{ctx['slug']}:policy:dept:bucket:L1/{bucket_id}",
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
                    VALUES (%s, %s, %s, %s, %s, 'dept', 3, %s, %s, 'VecDoc', %s)
                    """,
                    (
                        doc_id,
                        ctx["org_id"],
                        bucket_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"arn:openrag:{ctx['slug']}:policy:dept:document:L3/{doc_id}",
                        acl_key,
                        sha256_hex(doc_id),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO corpus.pages (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, thumb_embedding)
                    VALUES (%s, %s, %s, %s, %s, 'dept', 3, %s, %s, 1, %s::vector(128))
                    """,
                    (
                        page_id,
                        ctx["org_id"],
                        doc_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"arn:openrag:{ctx['slug']}:policy:dept:page:L3/{page_id}",
                        acl_key,
                        vec_target,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO corpus.page_figures (id, org_id, page_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, figure_idx, figure_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, 'dept', 3, %s, %s, 0, %s::vector(128))
                    """,
                    (
                        fig_id,
                        ctx["org_id"],
                        page_id,
                        doc_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"arn:openrag:{ctx['slug']}:policy:dept:figure:L3/{fig_id}",
                        acl_key,
                        vec_target,
                    ),
                )
                for cid, idx, text_val, vec in (
                    (chunk_near_id, 0, "travelpolicy 고유출장여비정책 벡터근접", vec_target),
                    (chunk_far_id, 1, "unrelated 무관한참고문단 원거리", vec_far),
                ):
                    cur.execute(
                        """
                        INSERT INTO corpus.chunks (id, org_id, document_id, page_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, chunk_idx, redacted_text, bm25_text, text_hash, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, 'dept', 3, %s, %s, %s, %s, %s, %s, %s::vector(128))
                        """,
                        (
                            cid,
                            ctx["org_id"],
                            doc_id,
                            page_id,
                            ctx["dept1_id"],
                            ctx["owner_id"],
                            f"arn:openrag:{ctx['slug']}:policy:dept:chunk:L3/{cid}",
                            acl_key,
                            idx,
                            text_val,
                            text_val,
                            sha256_hex(cid),
                            vec,
                        ),
                    )
                cur.execute(
                    """
                    INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding)
                    VALUES (%s, %s, %s, %s, %s, 'dept', 3, %s, %s, %s, 'VecWiki', '고유출장여비정책 위키', '고유출장여비정책 위키', %s, %s::vector(128))
                    """,
                    (
                        wiki_id,
                        ctx["org_id"],
                        bucket_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"arn:openrag:{ctx['slug']}:policy:dept:wiki:L3/{wiki_id}",
                        acl_key,
                        f"vw-{wiki_id[:6]}",
                        sha256_hex(wiki_id),
                        vec_target,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, embedding)
                    VALUES (%s, %s, %s, %s, %s, 'dept', 3, %s, %s, 'fact', '고유출장여비정책 메모리', '고유출장여비정책 메모리', %s, %s::vector(128))
                    """,
                    (
                        mem_id,
                        ctx["org_id"],
                        bucket_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"arn:openrag:{ctx['slug']}:policy:dept:memory:L3/{mem_id}",
                        acl_key,
                        sha256_hex(mem_id),
                        vec_target,
                    ),
                )

    owner_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["owner_id"],
        auth_level=6,
        dept_ids=(ctx["dept1_id"],),
    )
    chunk_rows = migrated_pg.query_as_principal(
        """
        SELECT id::text AS id, vector_dims(embedding) AS dims, acl_scope_key,
               (embedding <=> %s::vector(128)) AS dist,
               similarity(bm25_text, %s) AS trgm_sim,
               (to_tsvector('simple', bm25_text) @@ plainto_tsquery('simple', %s)) AS tsv_hit
        FROM corpus.chunks
        WHERE document_id = %s
        ORDER BY dist ASC
        """,
        principal=owner_ctx,
        params=(vec_target, "travelpolicy", "고유출장여비정책", doc_id),
    )
    assert [r["id"] for r in chunk_rows] == [chunk_near_id, chunk_far_id]
    assert all(r["dims"] == 128 and r["acl_scope_key"] == acl_key for r in chunk_rows)
    assert chunk_rows[0]["trgm_sim"] > chunk_rows[1]["trgm_sim"]
    assert chunk_rows[0]["tsv_hit"] is True and chunk_rows[1]["tsv_hit"] is False

    for tbl_sql in (
        "SELECT id::text FROM corpus.pages WHERE id = %s",
        "SELECT id::text FROM corpus.page_figures WHERE id = %s",
        "SELECT id::text FROM wiki.pages WHERE id = %s",
        "SELECT id::text FROM memory.entries WHERE id = %s",
    ):
        target = {
            "corpus.pages": page_id,
            "corpus.page_figures": fig_id,
            "wiki.pages": wiki_id,
            "memory.entries": mem_id,
        }[tbl_sql.split("FROM ")[1].split(" WHERE")[0]]
        assert len(migrated_pg.query_as_principal(tbl_sql, principal=owner_ctx, params=(target,))) == 1

    outsider_ctx = PgPrincipalContext(
        org_id=other_org["org_id"],
        principal_id=other_org["admin_id"],
        auth_level=9,
        dept_ids=(other_org["dept2_id"],),
        dept_admin_ids=(other_org["dept2_id"],),
    )
    assert (
        migrated_pg.query_as_principal(
            "SELECT id FROM corpus.chunks WHERE document_id = %s",
            principal=outsider_ctx,
            params=(doc_id,),
        )
        == []
    )


@pytest.mark.f07
@pytest.mark.f08
def test_t3_f07_f08_private_isolation_vs_cross_dept_group_allow(
    migrated_pg: PgEphemeralCluster,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F0708")
    tf_group_id = str(uuid.uuid4())
    bucket_id = str(uuid.uuid4())
    priv_doc_id = str(uuid.uuid4())
    grp_doc_id = str(uuid.uuid4())
    priv_mem_id = str(uuid.uuid4())
    grp_mem_id = str(uuid.uuid4())

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO iam.groups (id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level, allowed_actions)
                    VALUES (%s, %s, NULL, %s, 'project_tf', 1, 10, ARRAY['read', 'search'])
                    """,
                    (tf_group_id, ctx["org_id"], f"project-tf-{tf_group_id[:6]}"),
                )
                for pid in (ctx["owner_id"], ctx["peer_id"], ctx["other_dept_id"]):
                    cur.execute(
                        """
                        INSERT INTO iam.group_members (org_id, group_id, principal_id, member_role)
                        VALUES (%s, %s, %s, 'member')
                        """,
                        (ctx["org_id"], tf_group_id, pid),
                    )
                cur.execute(
                    """
                    INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
                    VALUES (%s, %s, %s, %s, %s, 'T3ScopeBucket', 'org', 1, %s)
                    """,
                    (
                        bucket_id,
                        ctx["org_id"],
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"sb-{bucket_id[:6]}",
                        f"arn:openrag:{ctx['slug']}:policy:org:bucket:L1/{bucket_id}",
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, min_auth_level, arn, acl_scope_key, title, content_sha256)
                    VALUES (%s, %s, %s, %s, %s, 'private', ARRAY[%s::UUID], 3, %s, %s, 'PrivateDoc', %s)
                    """,
                    (
                        priv_doc_id,
                        ctx["org_id"],
                        bucket_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        tf_group_id,
                        f"arn:openrag:{ctx['slug']}:policy:private:document:L3/{priv_doc_id}",
                        f"{ctx['org_id']}:private:L3",
                        sha256_hex(priv_doc_id),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, min_auth_level, arn, acl_scope_key, title, content_sha256)
                    VALUES (%s, %s, %s, %s, %s, 'group', ARRAY[]::UUID[], 3, %s, %s, 'GroupDoc', %s)
                    """,
                    (
                        grp_doc_id,
                        ctx["org_id"],
                        bucket_id,
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        f"arn:openrag:{ctx['slug']}:policy:group:document:L3/{grp_doc_id}",
                        f"{ctx['org_id']}:group:L3",
                        sha256_hex(grp_doc_id),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO iam.resource_group_bindings (org_id, resource_type, resource_id, group_id, effect, actions, min_auth_level)
                    VALUES (%s, 'document', %s, %s, 'allow', ARRAY['read', 'search'], 3)
                    """,
                    (ctx["org_id"], grp_doc_id, tf_group_id),
                )
                cur.execute(
                    """
                    INSERT INTO memory.entries (id, org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
                    VALUES
                        (%s, %s, %s, %s, 'private', ARRAY[%s::UUID], 3, %s, %s, 'fact', 'PrivateMem', 'PrivateMem', %s),
                        (%s, %s, %s, %s, 'group', ARRAY[%s::UUID], 3, %s, %s, 'fact', 'GroupMem', 'GroupMem', %s)
                    """,
                    (
                        priv_mem_id,
                        ctx["org_id"],
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        tf_group_id,
                        f"arn:openrag:{ctx['slug']}:policy:private:memory:L3/{priv_mem_id}",
                        f"{ctx['org_id']}:private:L3",
                        sha256_hex(priv_mem_id),
                        grp_mem_id,
                        ctx["org_id"],
                        ctx["dept1_id"],
                        ctx["owner_id"],
                        tf_group_id,
                        f"arn:openrag:{ctx['slug']}:policy:group:memory:L3/{grp_mem_id}",
                        f"{ctx['org_id']}:group:L3",
                        sha256_hex(grp_mem_id),
                    ),
                )

    peer_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["peer_id"],
        auth_level=6,
        dept_ids=(ctx["dept1_id"],),
        group_ids=(tf_group_id,),
    )
    other_dept_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["other_dept_id"],
        auth_level=6,
        dept_ids=(ctx["dept2_id"],),
        group_ids=(tf_group_id,),
    )
    dept_admin_ctx = PgPrincipalContext(
        org_id=ctx["org_id"],
        principal_id=ctx["dept_admin_id"],
        auth_level=8,
        dept_ids=(ctx["dept1_id"],),
        dept_admin_ids=(ctx["dept1_id"],),
    )

    # Same-dept peer AND cross-dept TF member see group doc & group memory, NEVER private doc/memory
    for pctx in (peer_ctx, other_dept_ctx):
        docs = {
            r["id"]
            for r in migrated_pg.query_as_principal(
                "SELECT id::text AS id FROM corpus.documents WHERE id = ANY(%s::uuid[])",
                principal=pctx,
                params=([priv_doc_id, grp_doc_id],),
            )
        }
        assert docs == {grp_doc_id}, f"Expected only group doc for {pctx.principal_id}, got {docs}"
        mems = {
            r["id"]
            for r in migrated_pg.query_as_principal(
                "SELECT id::text AS id FROM memory.entries WHERE id = ANY(%s::uuid[])",
                principal=pctx,
                params=([priv_mem_id, grp_mem_id],),
            )
        }
        assert mems == {grp_mem_id}

    # dept_admin of dept1 reads private doc & private memory
    admin_docs = {
        r["id"]
        for r in migrated_pg.query_as_principal(
            "SELECT id::text AS id FROM corpus.documents WHERE id = ANY(%s::uuid[])",
            principal=dept_admin_ctx,
            params=([priv_doc_id, grp_doc_id],),
        )
    }
    assert priv_doc_id in admin_docs
    admin_mems = {
        r["id"]
        for r in migrated_pg.query_as_principal(
            "SELECT id::text AS id FROM memory.entries WHERE id = %s",
            principal=dept_admin_ctx,
            params=(priv_mem_id,),
        )
    }
    assert admin_mems == {priv_mem_id}


@pytest.mark.f08
@pytest.mark.f09
def test_t3_f08_f09_group_level_threshold_and_member_level_override(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F0809")
    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-0809-adm")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="t3-0809-peer")
    other_key = mint_api_key(api_client, ctx["org_id"], ctx["other_dept_id"], tag="t3-0809-oth")

    # peer is L6 -> promote other_dept_id to L7 in DB so they satisfy binding min_auth_level=7
    migrated_pg.execute_sql(
        f"UPDATE iam.principals SET auth_level = 7 WHERE id = '{ctx['other_dept_id']}';"
    )

    grp_id = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level, allowed_actions)
        VALUES ('{grp_id}', '{ctx['org_id']}', NULL, 'lvl-tf-{grp_id[:6]}', 'project_tf', 6, 9, ARRAY['read', 'search']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id, member_role, level_override) VALUES
            ('{ctx['org_id']}', '{grp_id}', '{ctx['peer_id']}', 'member', 4),
            ('{ctx['org_id']}', '{grp_id}', '{ctx['other_dept_id']}', 'member', NULL);
        """
    )

    b = (
        api_client.with_api_key(admin_key)
        .create_bucket(
            {
                "slug": f"glvl-{uuid.uuid4().hex[:6]}",
                "name": "GroupLevel",
                "access_scope": "org",
                "min_auth_level": 1,
            }
        )
        .assert_status((200, 201))
        .json()
    )
    bid = str(b.get("id") or b.get("bucket_id"))
    doc = (
        api_client.with_api_key(admin_key)
        .create_document(
            bid,
            {
                "title": "LevelBoundDoc",
                "text": "GROUP_LEVEL_OVERRIDE_SECRET_T3",
                "access_scope": "group",
                "allowed_group_ids": [grp_id],
                "min_auth_level": 5,
            },
        )
        .assert_status((200, 201))
        .json()
    )
    doc_id = str(doc.get("id") or doc.get("document_id"))
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.resource_group_bindings (org_id, resource_type, resource_id, group_id, effect, actions, min_auth_level)
        VALUES ('{ctx['org_id']}', 'document', '{doc_id}', '{grp_id}', 'allow', ARRAY['read', 'search'], 7);
        """
    )

    # PG RLS check: peer (level_override=4 < 5/6/7) denied; other_dept (L7, no lower override) allowed
    peer_rls = migrated_pg.query_as_principal(
        "SELECT id::text FROM corpus.documents WHERE id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"],
            principal_id=ctx["peer_id"],
            auth_level=6,
            dept_ids=(ctx["dept1_id"],),
            group_ids=(grp_id,),
        ),
        params=(doc_id,),
    )
    assert peer_rls == []

    other_rls = migrated_pg.query_as_principal(
        "SELECT id::text FROM corpus.documents WHERE id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"],
            principal_id=ctx["other_dept_id"],
            auth_level=7,
            dept_ids=(ctx["dept2_id"],),
            group_ids=(grp_id,),
        ),
        params=(doc_id,),
    )
    assert len(other_rls) == 1

    # HTTP search check
    s_peer = api_client.with_api_key(peer_key).search_bucket(
        bid, {"query": "GROUP_LEVEL_OVERRIDE_SECRET_T3", "top_k": 5}
    )
    assert s_peer.status in (200, 403) and "GROUP_LEVEL_OVERRIDE_SECRET_T3" not in s_peer.text

    s_other = (
        api_client.with_api_key(other_key)
        .search_bucket(bid, {"query": "GROUP_LEVEL_OVERRIDE_SECRET_T3", "top_k": 5})
        .assert_status(200)
    )
    assert "GROUP_LEVEL_OVERRIDE_SECRET_T3" in s_other.text


@pytest.mark.f09
@pytest.mark.f10
def test_t3_f09_f10_explicit_deny_overrides_level_and_group_allow_in_simulator_and_rls(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F0910")
    # Promote peer to L9 and create allow_grp + deny_grp + Allow * ARN policy + Explicit Deny policy
    migrated_pg.execute_sql(
        f"UPDATE iam.principals SET auth_level = 9 WHERE id = '{ctx['peer_id']}';"
    )
    allow_grp = str(uuid.uuid4())
    deny_grp = str(uuid.uuid4())
    allow_pol = str(uuid.uuid4())
    deny_pol = str(uuid.uuid4())
    allow_doc_json = json.dumps(
        {"statements": [{"effect": "Allow", "actions": ["*"], "resources": ["*"]}]}
    )
    deny_doc_json = json.dumps(
        {
            "statements": [
                {
                    "effect": "Deny",
                    "actions": ["read", "search"],
                    "resources": [f"arn:openrag:{ctx['slug']}:*"],
                }
            ]
        }
    )
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, name, group_type, min_auth_level, max_auth_level, allowed_actions) VALUES
            ('{allow_grp}', '{ctx['org_id']}', 'allow-g-{allow_grp[:6]}', 'permission_group', 1, 10, ARRAY['read', 'search']),
            ('{deny_grp}', '{ctx['org_id']}', 'deny-g-{deny_grp[:6]}', 'permission_group', 1, 10, ARRAY['read', 'search']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
            ('{ctx['org_id']}', '{allow_grp}', '{ctx['peer_id']}'),
            ('{ctx['org_id']}', '{deny_grp}', '{ctx['peer_id']}');
        INSERT INTO iam.policies (id, org_id, name, document) VALUES
            ('{allow_pol}', '{ctx['org_id']}', 'allow-all-{allow_pol[:6]}', '{allow_doc_json}'::jsonb),
            ('{deny_pol}', '{ctx['org_id']}', 'deny-org-{deny_pol[:6]}', '{deny_doc_json}'::jsonb);
        INSERT INTO iam.policy_attachments (org_id, policy_id, target_type, target_id) VALUES
            ('{ctx['org_id']}', '{allow_pol}', 'principal', '{ctx['peer_id']}'),
            ('{ctx['org_id']}', '{deny_pol}', 'group', '{deny_grp}');
        """
    )

    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-0910-adm")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="t3-0910-peer")
    admin_client = api_client.with_api_key(admin_key)

    b = admin_client.create_bucket(
        {"slug": f"deny-{uuid.uuid4().hex[:6]}", "name": "DenyBucket", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    doc = admin_client.create_document(
        bid,
        {
            "title": "ExplicitDenyDoc",
            "text": "EXPLICIT_DENY_OVERRIDES_ALL_T3",
            "access_scope": "group",
            "allowed_group_ids": [allow_grp],
            "denied_group_ids": [deny_grp],
            "min_auth_level": 2,
        },
    ).assert_status((200, 201)).json()
    doc_id = str(doc.get("id") or doc.get("document_id"))

    sim = admin_client.studio_iam_simulate(
        {
            "principal_id": ctx["peer_id"],
            "resource_type": "document",
            "resource_id": doc_id,
            "action": "read",
        }
    ).assert_status(200).json()
    assert sim.get("allowed") is False
    assert sim.get("matched_denies"), f"Expected non-empty matched_denies in {sim!r}"

    rls_rows = migrated_pg.query_as_principal(
        "SELECT id::text FROM corpus.documents WHERE id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"],
            principal_id=ctx["peer_id"],
            auth_level=9,
            dept_ids=(ctx["dept1_id"],),
            group_ids=(allow_grp, deny_grp),
        ),
        params=(doc_id,),
    )
    assert rls_rows == []

    srch = api_client.with_api_key(peer_key).search_bucket(
        bid, {"query": "EXPLICIT_DENY_OVERRIDES_ALL_T3", "top_k": 5}
    )
    assert srch.status in (200, 403) and "EXPLICIT_DENY_OVERRIDES_ALL_T3" not in srch.text
