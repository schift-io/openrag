"""Tier 4 Scenarios 6-11: Stdio MCP 13 Tools, Kumiho, Promote+KoParaRel, Obsidian Sync, 3-Engine+S3, Outbox/Quarantine."""

from __future__ import annotations

import hashlib
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
    PgPrincipalContext,
)
from lib.mcp_client import OPENRAG_MCP_TOOLS
from tier4._helpers import (
    extract_id,
    insert_returning_one,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_institution_org,
)

pytestmark = [pytest.mark.tier4]


@pytest.mark.f08
@pytest.mark.f09
@pytest.mark.f16
@pytest.mark.f29
@pytest.mark.f36
@pytest.mark.f37
def test_s06_local_stdio_mcp_all_13_tools_under_arn_group_level_and_guard(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    ctx = seed_institution_org(migrated_pg, label="S06")
    tf_grp = str(uuid.uuid4())
    deny_grp = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, name, group_type, allowed_actions) VALUES
            ('{tf_grp}', '{ctx['org_id']}', 's06-tf-{tf_grp[:6]}', 'project_tf', ARRAY['read', 'search', 'write']),
            ('{deny_grp}', '{ctx['org_id']}', 's06-deny-{deny_grp[:6]}', 'permission_group', ARRAY['read', 'search']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
            ('{ctx['org_id']}', '{tf_grp}', '{ctx['org_admin_id']}'),
            ('{ctx['org_id']}', '{tf_grp}', '{ctx['other_dept_id']}'),
            ('{ctx['org_id']}', '{deny_grp}', '{ctx['low_id']}');
        """
    )
    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s06-adm")
    low_key = mint_api_key(api_client, ctx["org_id"], ctx["low_id"], tag="s06-low")
    admin_client = api_client.with_api_key(admin_key)

    bslug = f"s06-mcp-{uuid.uuid4().hex[:6]}"
    b = admin_client.create_bucket({"slug": bslug, "name": "S06Mcp", "access_scope": "org"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    admin_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "ORION-MCP-S06",
            "action": "block",
            "surfaces": ["inbound_query", "mcp_egress"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    wiki_id = str(uuid.uuid4())
    wiki_slug = f"s06-w-{wiki_id[:6]}"
    migrated_pg.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{wiki_id}', '{ctx['org_id']}', '{bid}', '{ctx['dept_a_id']}', '{ctx['org_admin_id']}', 'dept', 1, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L1/{wiki_id}', '{ctx['org_id']}:dept', '{wiki_slug}', 'S06 Wiki', 'S06 위키 본문', 'S06 위키 본문', '{hashlib.sha256(wiki_id.encode()).hexdigest()}');
        """
    )

    # Verify both ndjson and content-length framing list all 13 tools
    with mcp_stdio(
        api_key=admin_key,
        principal_id=ctx["org_admin_id"],
        framing="content-length",
        prefer_standalone=True,
        env={"DATABASE_URL": migrated_pg.dsn()},
    ) as cl_mcp:
        cl_mcp.handshake()
        assert {t["name"] for t in cl_mcp.tools_list()} == set(OPENRAG_MCP_TOOLS)

    doc_id = ""
    with mcp_stdio(
        api_key=admin_key,
        principal_id=ctx["org_admin_id"],
        framing="ndjson",
        env={"DATABASE_URL": migrated_pg.dsn()},
    ) as mcp:
        mcp.handshake()
        assert {t["name"] for t in mcp.tools_list()} == set(OPENRAG_MCP_TOOLS)

        assert mcp.concept_link("llmwiki", "세컨드브레인", canonical_name="llmwiki", lang="ko").is_error is False
        assert mcp.concept_resolve("세컨드브레인").is_error is False

        ing = mcp.rag_ingest(
            bucket=bslug,
            text="S06 제한 정책 본문 SECRET_S06_TOKEN",
            access_scope="group",
            allowed_group_ids=[tf_grp],
            denied_group_ids=[deny_grp],
            min_auth_level=5,
            title="S06 Group Doc",
            tags=["s06-policy"],
        )
        assert ing.is_error is False
        doc_id = extract_id(ing.json(), "document_id", "id", "doc_id")

        assert "SECRET_S06_TOKEN" in mcp.rag_search("SECRET_S06_TOKEN", bucket=bslug, tags_any=["s06-policy"]).text
        assert mcp.rag_ask("S06 정책 요약", bucket=bslug).is_error is False
        assert "SECRET_S06_TOKEN" in mcp.rag_get_page(document_id=doc_id, page_no=1).text

        ref = mcp.memory_reflect("S06 초기 메모리", scope="dept")
        assert ref.is_error is False
        mem_id = extract_id(ref.json(), "memory_id", "id")
        assert mcp.memory_patch(mem_id, "amend", text="S06 개정 메모리").is_error is False
        assert "S06 개정 메모리" in mcp.memory_engage("S06 개정 메모리", scope="dept").text
        assert "S06 개정 메모리" in mcp.memory_search("S06 개정 메모리", scope="dept").text

        assert mcp.wiki_search("S06 위키 본문").is_error is False
        assert "S06 위키 본문" in mcp.wiki_get(slug=wiki_slug).text
        assert mcp.guard_check("정상 업무 메시지", surface="mcp_egress").json().get("decision") == "allow"

    with mcp_stdio(
        api_key=low_key,
        principal_id=ctx["low_id"],
        framing="ndjson",
        env={"DATABASE_URL": migrated_pg.dsn()},
    ) as low_mcp:
        low_mcp.handshake()
        pg_den = low_mcp.rag_get_page(document_id=doc_id, page_no=1)
        assert pg_den.is_error is True and "SECRET_S06_TOKEN" not in pg_den.text
        srch_den = low_mcp.rag_search("SECRET_S06_TOKEN", bucket=bslug)
        assert "SECRET_S06_TOKEN" not in srch_den.text

        ask_blk = low_mcp.rag_ask("ORION-MCP-S06 질문")
        assert ask_blk.is_error is True
        chk_blk = low_mcp.guard_check("ORION-MCP-S06", surface="mcp_egress")
        assert chk_blk.is_error is True or chk_blk.json().get("decision") == "block"


@pytest.mark.f03
@pytest.mark.f18
@pytest.mark.f29
@pytest.mark.f34
def test_s07_kumiho_pluggable_memory_backend_and_belief_revision_edges(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_institution_org(migrated_pg, label="S07")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s07")
    client = api_client.with_api_key(key)

    kumiho_yaml = sample_factory.pipeline_yaml(
        "s07_kumiho.yaml", memory_backend="kumiho", memory_mirror_to="kumiho"
    ).read_text(encoding="utf-8")
    client.studio_pipeline_update({"pipeline_yaml": kumiho_yaml}).assert_status((200, 204))
    assert "kumiho" in client.studio_pipeline_get().assert_status(200).text

    try:
        with mcp_stdio(api_key=key, principal_id=ctx["org_admin_id"], env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
            mcp.handshake()
            m1_id = extract_id(
                mcp.memory_reflect("초기 출장 일비 5만원 S07", scope="dept", backend="kumiho").json(),
                "memory_id",
                "id",
            )
            mcp.memory_reflect(
                "영수증 증빙 기준 확인 S07",
                scope="dept",
                edge_type="supports",
                target_memory_id=m1_id,
                backend="kumiho",
            )
            mcp.memory_reflect(
                "5만원 기준과 모순 발견 S07",
                scope="dept",
                edge_type="contradicts",
                target_memory_id=m1_id,
                backend="kumiho",
            )
            m4_id = extract_id(
                mcp.memory_reflect(
                    "개정 출장 일비 7만원 S07",
                    scope="dept",
                    edge_type="supersedes",
                    target_memory_id=m1_id,
                    backend="kumiho",
                ).json(),
                "memory_id",
                "id",
            )

            eng = mcp.memory_engage("출장 일비 S07", scope="dept", backend="kumiho")
            assert eng.is_error is False and "7만원" in eng.text

            pg_eng = mcp.memory_engage("출장 일비 S07", scope="dept", backend="openrag_pg")
            assert pg_eng.is_error is False

        edges = migrated_pg.execute_sql(
            f"SELECT string_agg(relation_type, ',') FROM memory.edges WHERE org_id = '{ctx['org_id']}' AND dst_memory_id = '{m1_id}';"
        )
        for rel in ("supports", "contradicts", "supersedes"):
            assert rel in edges
        assert m4_id
    finally:
        default_yaml = sample_factory.pipeline_yaml("s07_default.yaml", memory_backend="openrag_pg").read_text(encoding="utf-8")
        client.studio_pipeline_update({"pipeline_yaml": default_yaml})


@pytest.mark.f18
@pytest.mark.f19
@pytest.mark.f20
@pytest.mark.f33
@pytest.mark.f34
def test_s08_cclg_memory_promote_ladder_and_kopararel_conflict_resolution(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S08")
    tf_grp = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, name, group_type, allowed_actions)
        VALUES ('{tf_grp}', '{ctx['org_id']}', 's08-grp-{tf_grp[:6]}', 'project_tf', ARRAY['read', 'search']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
            ('{ctx['org_id']}', '{tf_grp}', '{ctx['owner_id']}'),
            ('{ctx['org_id']}', '{tf_grp}', '{ctx['other_dept_id']}');
        """
    )

    owner_key = mint_api_key(api_client, ctx["org_id"], ctx["owner_id"], tag="s08-own")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="s08-peer")
    other_key = mint_api_key(api_client, ctx["org_id"], ctx["other_dept_id"], tag="s08-oth")
    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s08-adm")

    owner_client = api_client.with_api_key(owner_key)
    bslug = f"s08-lad-{uuid.uuid4().hex[:6]}"
    b = owner_client.create_bucket({"slug": bslug, "name": "LadderS08", "access_scope": "org"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    mcp = McpHttpClient(owner_client)
    mcp.handshake()
    mem_id = extract_id(
        mcp.memory_reflect("여비 정산 S08_LADDER 5만원", scope="private").json(), "memory_id", "id"
    )

    # Private: peer cannot see
    assert (
        migrated_pg.query_as_principal(
            "SELECT id::text FROM memory.entries WHERE id = %s",
            principal=PgPrincipalContext(
                org_id=ctx["org_id"], principal_id=ctx["peer_id"], auth_level=8, dept_ids=(ctx["dept_a_id"],)
            ),
            params=(mem_id,),
        )
        == []
    )

    owner_client.app_memory_patch({"memory_id": mem_id, "op": "amend", "text": "여비 정산 S08_LADDER 6만원"}).assert_status((200, 201))
    owner_client.app_memory_promote({"memory_id": mem_id, "to": "dept"}).assert_status((200, 201))
    peer_mcp = McpHttpClient(api_client.with_api_key(peer_key))
    peer_mcp.handshake()
    assert "S08_LADDER" in peer_mcp.memory_search("S08_LADDER", scope="dept").text

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": owner_key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.memory_promote(mem_id, to="group", env=env, check=True)
    openrag_cli.memory_promote(mem_id, to="org", env=env, check=True)
    other_mcp = McpHttpClient(api_client.with_api_key(other_key))
    other_mcp.handshake()
    assert "S08_LADDER" in other_mcp.memory_search("S08_LADDER", scope="org").text

    openrag_cli.memory_promote(mem_id, to="wiki", bucket=bslug, env=env, check=True)
    wiki_id = migrated_pg.execute_sql(
        f"SELECT promoted_wiki_id::text FROM memory.entries WHERE id = '{mem_id}';"
    ).strip()
    assert wiki_id

    conflict_wiki_id = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{conflict_wiki_id}', '{ctx['org_id']}', '{bid}', '{ctx['dept_a_id']}', '{ctx['org_admin_id']}', 'dept', 1, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L1/{conflict_wiki_id}', '{ctx['org_id']}:dept', 's08-conf-{conflict_wiki_id[:6]}', '충돌 규정', '여비 정산 4만원 구기준', '여비 정산 4만원 구기준', '{hashlib.sha256(conflict_wiki_id.encode()).hexdigest()}');
        INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type)
        VALUES ('{ctx['org_id']}', '{wiki_id}', '{conflict_wiki_id}', 'contradicts');
        """
    )
    conflicts = api_client.with_api_key(admin_key).studio_wiki_conflicts().assert_status(200).text
    assert wiki_id in conflicts or conflict_wiki_id in conflicts
    assert "S08_LADDER" in owner_client.app_wiki().assert_status(200).text


@pytest.mark.f20
@pytest.mark.f24
@pytest.mark.f28
def test_s09_obsidian_md_bidirectional_wiki_sync_and_latest_revision_retrieval(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S09")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s09")
    client = api_client.with_api_key(key)
    bslug = f"s09-obs-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "ObsidianS09", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    vault_dir = sample_factory.obsidian_vault()
    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.wiki_import(vault_dir, bucket=bslug, scope="dept", auth_level=3, env=env, check=True)

    page_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM wiki.pages WHERE org_id = '{ctx['org_id']}' AND embedding IS NOT NULL;"
        ).strip()
    )
    assert page_cnt >= 4

    mcp = McpHttpClient(client)
    mcp.handshake()
    res = mcp.rag_search("출장 일비", bucket=bslug, spaces=["chunks", "wiki", "memory"], fusion="rrf")
    assert res.is_error is False and "70,000원" in res.text
    if "50,000원" in res.text:
        assert res.text.index("70,000원") < res.text.index("50,000원")

    out_dir = tmp_path / "vault_out"
    openrag_cli.wiki_export(bslug, out_dir=out_dir, env=env, check=True)
    exported = list(out_dir.rglob("*.md"))
    assert len(exported) >= 4 and any("70,000원" in p.read_text(encoding="utf-8") for p in exported)

    export_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM wiki.sync_changes WHERE org_id = '{ctx['org_id']}' AND direction = 'export';"
        ).strip()
    )
    assert export_cnt >= 1 and bid


@pytest.mark.f04
@pytest.mark.f05
@pytest.mark.f06
@pytest.mark.f21
@pytest.mark.f22
@pytest.mark.f28
@pytest.mark.f34
def test_s10_pg_pinecone_3_engine_live_switching_and_s3_vectors_mirror(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    caps = migrated_pg.probe_capabilities()
    assert caps["vector"] is True and caps["pg_trgm"] is True

    ctx_a = seed_institution_org(migrated_pg, label="S10A")
    ctx_b = seed_institution_org(migrated_pg, label="S10B")
    key_a = mint_api_key(api_client, ctx_a["org_id"], ctx_a["org_admin_id"], tag="s10-a")
    key_b = mint_api_key(api_client, ctx_b["org_id"], ctx_b["org_admin_id"], tag="s10-b")
    client_a = api_client.with_api_key(key_a)
    client_b = api_client.with_api_key(key_b)

    ba = client_a.create_bucket({"slug": f"s10a-{uuid.uuid4().hex[:6]}", "name": "S10A", "access_scope": "dept"}).assert_status((200, 201)).json()
    bb = client_b.create_bucket({"slug": f"s10b-{uuid.uuid4().hex[:6]}", "name": "S10B", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid_a = extract_id(ba, "id", "bucket_id")
    bid_b = extract_id(bb, "id", "bucket_id")

    hwpx = sample_factory.hwpx_bytes(title="삼중엔진", paragraphs=["삼중엔진무중단전환S10 고유 검색어"])
    client_a.upload_document(
        bid_a,
        filename="s10.hwpx",
        content=hwpx,
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202))

    query_body = {"query": "삼중엔진무중단전환S10", "top_k": 5}
    env_a = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key_a, "OPENRAG_ORG_ID": ctx_a["org_id"]}
    for engine in ("native", "pgvector", "remote", "native"):
        openrag_cli.index_engine("corpus.chunks", engine, env=env_a, check=True)
        srch = client_a.search_bucket(bid_a, query_body).assert_status(200)
        assert "삼중엔진무중단전환S10" in srch.text

    client_a.studio_index_set_engine({"table": "corpus.chunks", "engine": "pgvector"}).assert_status((200, 204))
    assert "삼중엔진무중단전환S10" in client_a.search_bucket(bid_a, query_body).assert_status(200).text

    openrag_cli.index_mirror("corpus.chunks", to="s3-vectors", env=env_a, check=True)
    env_b = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key_b, "OPENRAG_ORG_ID": ctx_b["org_id"]}
    openrag_cli.index_mirror("corpus.chunks", to="s3-vectors", env=env_b, check=True)

    # Tenant B search must not leak Tenant A token
    b_res = client_b.search_bucket(bid_b, query_body)
    assert "삼중엔진무중단전환S10" not in b_res.text


@pytest.mark.f23
@pytest.mark.f28
@pytest.mark.f34
def test_s11_incremental_outbox_noop_3fail_quarantine_and_retry_recovery(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S11")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s11")
    client = api_client.with_api_key(key)
    bslug = f"s11-out-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "OutboxS11", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    text_v1 = "S11 초기 본문 3페이지 증분 색인"
    sha_v1 = hashlib.sha256(text_v1.encode("utf-8")).hexdigest()
    payload_v1 = {
        "documents": [
            {
                "external_key": "s11-doc",
                "title": "S11 Doc",
                "text": text_v1,
                "content_sha256": sha_v1,
                "access_scope": "dept",
            }
        ]
    }
    client.sync_documents(bslug, payload_v1).assert_status((200, 201))
    wm_1 = int(
        migrated_pg.execute_sql(
            f"SELECT COALESCE(SUM(indexed_count), 0) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    client.sync_documents(bslug, payload_v1).assert_status((200, 201))
    wm_2 = int(
        migrated_pg.execute_sql(
            f"SELECT COALESCE(SUM(indexed_count), 0) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    assert wm_2 == wm_1

    doc_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.documents WHERE bucket_id = '{bid}' LIMIT 1;"
    ).strip()
    chunk_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.chunks WHERE document_id = '{doc_id}' LIMIT 1;"
    ).strip()

    # Priority 10 vs Priority 1 drain order
    p10_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{chunk_id}', 'upsert_embed', 10, 'pending')
        RETURNING id;
        """,
    )
    p1_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{chunk_id}', 'redact_reindex', 1, 'pending')
        RETURNING id;
        """,
    )

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.run(["index", "drain", "--limit", "1"], env=env, check=True)
    p1_status = migrated_pg.execute_sql(f"SELECT status FROM indexing.outbox WHERE id = {p1_id};").strip()
    p10_status = migrated_pg.execute_sql(f"SELECT status FROM indexing.outbox WHERE id = {p10_id};").strip()
    assert p1_status == "done" and p10_status == "pending"

    # Simulate 3-failure quarantine & recovery
    q_ob_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status, attempts, max_attempts, last_error)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{chunk_id}', 'sync_external_index', 5, 'quarantined', 3, 3, 's11 simulated failure')
        RETURNING id;
        """,
    )
    q_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO indexing.quarantine (outbox_id, org_id, target_table, target_id, op, error_class, error_detail)
        VALUES ({q_ob_id}, '{ctx['org_id']}', 'corpus.chunks', '{chunk_id}', 'sync_external_index', 'transient', 's11 simulated failure')
        RETURNING id::text;
        """,
    )

    assert q_id in client.studio_index_outbox().assert_status(200).text or "quarantined" in openrag_cli.index_status(env=env, check=True).stdout.lower()
    client.studio_quarantine_retry(q_id).assert_status((200, 202, 204))
    openrag_cli.index_retry(quarantine_id=q_id, env=env, check=True)
    assert (
        migrated_pg.execute_sql(f"SELECT resolved_at IS NOT NULL FROM indexing.quarantine WHERE id = '{q_id}';").strip()
        == "t"
    )
