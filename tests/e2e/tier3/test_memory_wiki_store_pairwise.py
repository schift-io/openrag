"""Tier 3 pairwise E2E tests: Memory, Kumiho, Promote Ladder, Wiki/KoParaRel, 3-Engine, S3/OSS/BM25, Outbox (7 tests)."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

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
from tier3._helpers import (
    extract_id,
    insert_returning_one,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_org_with_users,
)

pytestmark = [pytest.mark.tier3]


@pytest.mark.f18
@pytest.mark.f07
def test_t3_f18_f07_cclg_memory_patch_preserves_history_and_private_rls(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F1807")
    owner_key = mint_api_key(api_client, ctx["org_id"], ctx["owner_id"], tag="t3-1807-own")
    mcp = McpHttpClient(api_client.with_api_key(owner_key))
    mcp.handshake()

    ref = mcp.memory_reflect("초기 개인 메모리 V1_T3", scope="private")
    assert ref.is_error is False
    mem_id = extract_id(ref.json(), "memory_id", "id")

    openrag_cli.memory_patch(
        mem_id,
        op="amend",
        text="개정 개인 메모리 V2_T3",
        reason="t3 amend",
        env={"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": owner_key, "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    eff = migrated_pg.execute_sql(
        f"SELECT effective_text || '|' || version::text FROM memory.entries WHERE id = '{mem_id}';"
    ).strip()
    assert "개정 개인 메모리 V2_T3" in eff and eff.endswith("|2")

    patch_row = migrated_pg.execute_sql(
        f"SELECT patch_op || '|' || prev_version::text || '|' || new_version::text || '|' || COALESCE(prev_text, '') FROM memory.patches WHERE memory_entry_id = '{mem_id}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert patch_row.startswith("amend|1|2|") and "V1_T3" in patch_row

    peer_rows = migrated_pg.query_as_principal(
        "SELECT id::text FROM memory.entries WHERE id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"],
            principal_id=ctx["peer_id"],
            auth_level=6,
            dept_ids=(ctx["dept1_id"],),
        ),
        params=(mem_id,),
    )
    assert peer_rows == []

    admin_rows = migrated_pg.query_as_principal(
        "SELECT id::text FROM memory.entries WHERE id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"],
            principal_id=ctx["dept_admin_id"],
            auth_level=8,
            dept_ids=(ctx["dept1_id"],),
            dept_admin_ids=(ctx["dept1_id"],),
        ),
        params=(mem_id,),
    )
    assert len(admin_rows) == 1


@pytest.mark.f18
@pytest.mark.f19
def test_t3_f18_f19_kumiho_reflect_contradiction_edge_and_promote_ladder_to_wiki(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F1819")
    key = mint_api_key(api_client, ctx["org_id"], ctx["owner_id"], tag="t3-1819")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"prom-{uuid.uuid4().hex[:6]}", "name": "PromoteBucket", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bucket_slug = str(b.get("slug"))

    mcp = McpHttpClient(client)
    mcp.handshake()
    m1 = mcp.memory_reflect("기존 여비 기준 5만원 T3_KUMIHO", scope="private", backend="kumiho")
    assert m1.is_error is False
    m1_id = extract_id(m1.json(), "memory_id", "id")

    m2 = mcp.memory_reflect(
        "신규 여비 기준 7만원 T3_KUMIHO",
        scope="private",
        edge_type="contradicts",
        target_memory_id=m1_id,
        backend="kumiho",
    )
    assert m2.is_error is False
    m2_id = extract_id(m2.json(), "memory_id", "id")

    edge_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM memory.edges WHERE org_id = '{ctx['org_id']}' AND src_memory_id = '{m2_id}' AND dst_memory_id = '{m1_id}' AND relation_type = 'contradicts';"
        ).strip()
    )
    assert edge_cnt == 1

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.memory_promote(m2_id, to="dept", env=env, check=True)
    openrag_cli.memory_promote(m2_id, to="wiki", bucket=bucket_slug, env=env, check=True)

    promoted_wiki_id = migrated_pg.execute_sql(
        f"SELECT COALESCE(promoted_wiki_id::text, '') FROM memory.entries WHERE id = '{m2_id}';"
    ).strip()
    assert promoted_wiki_id, "Expected promoted_wiki_id after promoting memory to wiki"
    wiki_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM wiki.pages WHERE id = '{promoted_wiki_id}';"
        ).strip()
    )
    assert wiki_cnt == 1


@pytest.mark.f19
@pytest.mark.f20
def test_t3_f19_f20_promoted_memory_links_into_wiki_kopararel_and_exports_markdown(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    tmp_path: Path,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F1920")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1920")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"wk-exp-{uuid.uuid4().hex[:6]}", "name": "WikiExport", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")
    bslug = str(b.get("slug"))

    old_wiki_id = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{old_wiki_id}', '{ctx['org_id']}', '{bid}', '{ctx['dept1_id']}', '{ctx['admin_id']}', 'dept', 1, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L1/{old_wiki_id}', '{ctx['org_id']}:dept', 'old-travel-t3', '구 규정', '구 여비 5만원', '구 여비 5만원', '{hashlib.sha256(old_wiki_id.encode()).hexdigest()}');
        """
    )

    mcp = McpHttpClient(client)
    mcp.handshake()
    ref = mcp.memory_reflect("신규 여비 규정 승격 T3_EXP_70000", scope="dept")
    assert ref.is_error is False
    mem_id = extract_id(ref.json(), "memory_id", "id")

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.memory_promote(mem_id, to="wiki", bucket=bslug, env=env, check=True)
    new_wiki_id = migrated_pg.execute_sql(
        f"SELECT promoted_wiki_id::text FROM memory.entries WHERE id = '{mem_id}';"
    ).strip()
    assert new_wiki_id

    migrated_pg.execute_sql(
        f"INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type) VALUES ('{ctx['org_id']}', '{new_wiki_id}', '{old_wiki_id}', 'supersedes');"
    )

    out_dir = tmp_path / "exported_wiki"
    openrag_cli.wiki_export(bslug, out_dir=out_dir, env=env, check=True)
    md_files = list(out_dir.rglob("*.md"))
    assert md_files and any("T3_EXP_70000" in p.read_text(encoding="utf-8") for p in md_files)

    sync_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM wiki.sync_changes WHERE org_id = '{ctx['org_id']}' AND direction = 'export';"
        ).strip()
    )
    assert sync_cnt >= 1


@pytest.mark.f20
@pytest.mark.f17
def test_t3_f20_f17_cascade_redaction_cleans_wiki_revisions_and_resyncs_markdown(
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

    ctx = seed_org_with_users(migrated_pg, label="T3F2017")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2017")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"wk-red-{uuid.uuid4().hex[:6]}", "name": "WikiRedact", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bslug = str(b.get("slug"))

    secret_token = "WIKI-CANARY-900101-1234567"
    vault_dir = sample_factory.obsidian_vault(
        extra_pages={
            "canary-page.md": f"---\ntitle: 감사 위키\n---\n\n# 감사 위키\n\n비공개 식별자: {secret_token}\n"
        }
    )
    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.wiki_import(vault_dir, bucket=bslug, scope="dept", auth_level=3, env=env, check=True)

    wiki_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM wiki.pages WHERE org_id = '{ctx['org_id']}' AND body_md LIKE '%{secret_token}%' LIMIT 1;"
    ).strip()
    assert wiki_id

    openrag_cli.redact_run(
        wiki_id=wiki_id,
        scope="wiki",
        hard_purge=True,
        reason="t3 wiki cascade redact",
        env=env,
        check=True,
    )

    body_after = migrated_pg.execute_sql(
        f"SELECT body_md || ' ' || bm25_text FROM wiki.pages WHERE id = '{wiki_id}';"
    )
    assert secret_token not in body_after

    redactor_revs = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM wiki.revisions WHERE page_id = '{wiki_id}' AND editor_type = 'redactor';"
        ).strip()
    )
    assert redactor_revs >= 1

    reexport_dir = tmp_path / "reexported"
    openrag_cli.wiki_export(bslug, out_dir=reexport_dir, env=env, check=True)
    for md in reexport_dir.rglob("*.md"):
        assert secret_token not in md.read_text(encoding="utf-8")


@pytest.mark.f21
@pytest.mark.f22
def test_t3_f21_f22_pg_pinecone_3_engine_switch_with_external_s3_oss_mirror_and_bm25(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F2122")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2122")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"eng3-{uuid.uuid4().hex[:6]}", "name": "ThreeEngine", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    token = "삼중엔진외부미러링고유토큰T3"
    hwpx = sample_factory.hwpx_bytes(title="엔진 전환 검증", paragraphs=[f"본문 키워드 {token}"])
    client.upload_document(
        bid,
        filename="engine.hwpx",
        content=hwpx,
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202))

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.index_mirror("corpus.chunks", to="s3-vectors", env=env, check=True)
    openrag_cli.index_mirror("corpus.chunks", to="qdrant", env=env, check=True)

    for engine in ("native", "pgvector", "remote", "native"):
        client.studio_index_set_engine({"table": "corpus.chunks", "engine": engine}).assert_status((200, 204))
        for fusion_payload in (
            {"query": token, "top_k": 5, "fusion": "convex", "alpha": 0.5},
            {"query": token, "top_k": 5, "fusion": "convex", "alpha": 1.0},
        ):
            srch = client.search_bucket(bid, fusion_payload).assert_status(200)
            assert token in srch.text


@pytest.mark.f21
@pytest.mark.f23
def test_t3_f21_f23_engine_switch_triggers_outbox_watermark_sync_without_noop_reembed(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F2123")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2123")
    client = api_client.with_api_key(key)
    bslug = f"noop-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "NoopSwitch", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    body_text = "해시동일무중단스위치검증본문T3"
    sha = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
    sync_payload = {
        "documents": [
            {
                "external_key": "noop-doc-1",
                "title": "NoOp Doc",
                "text": body_text,
                "content_sha256": sha,
                "access_scope": "dept",
            }
        ]
    }
    client.sync_documents(bslug, sync_payload).assert_status((200, 201))
    count_first = migrated_pg.execute_sql(
        f"SELECT COALESCE(SUM(indexed_count), 0) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}';"
    ).strip()

    res_noop = client.sync_documents(bslug, sync_payload).assert_status((200, 201)).json()
    count_second = migrated_pg.execute_sql(
        f"SELECT COALESCE(SUM(indexed_count), 0) FROM indexing.watermarks WHERE org_id = '{ctx['org_id']}';"
    ).strip()
    assert int(count_second) == int(count_first), res_noop

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.index_engine("corpus.chunks", "pgvector", env=env, check=True)
    eng = migrated_pg.execute_sql(
        f"SELECT pg_pinecone_engine FROM vectors.index_bindings WHERE org_id = '{ctx['org_id']}' AND table_name = 'corpus.chunks' LIMIT 1;"
    ).strip()
    assert eng == "pgvector"
    assert body_text in client.search_bucket(bid, {"query": body_text, "top_k": 5}).assert_status(200).text


@pytest.mark.f22
@pytest.mark.f23
def test_t3_f22_f23_external_adapter_failure_quarantines_outbox_and_recovers_via_retry(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F2223")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2223")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"quar-{uuid.uuid4().hex[:6]}", "name": "Quarantine", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    doc = client.create_document(
        bid, {"title": "QuarDoc", "text": "격리복구테스트본문T3", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")
    chunk_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.chunks WHERE document_id = '{doc_id}' LIMIT 1;"
    ).strip()
    assert chunk_id

    outbox_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO indexing.outbox (org_id, target_table, target_id, op, priority, status, attempts, max_attempts, last_error)
        VALUES ('{ctx['org_id']}', 'corpus.chunks', '{chunk_id}', 'sync_external_index', 5, 'quarantined', 3, 3, 'simulated s3_vectors timeout')
        RETURNING id;
        """,
    )
    q_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO indexing.quarantine (outbox_id, org_id, target_table, target_id, op, error_class, error_detail)
        VALUES ({outbox_id}, '{ctx['org_id']}', 'corpus.chunks', '{chunk_id}', 'sync_external_index', 'external_timeout', 'simulated s3_vectors timeout')
        RETURNING id::text;
        """,
    )

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    status_out = openrag_cli.index_status(env=env, check=True)
    assert "quarantined" in status_out.stdout.lower() or q_id in status_out.stdout

    client.studio_quarantine_retry(q_id).assert_status((200, 202, 204))
    openrag_cli.index_retry(retry_all=True, env=env, check=True)

    resolved = migrated_pg.execute_sql(
        f"SELECT resolved_at IS NOT NULL FROM indexing.quarantine WHERE id = '{q_id}';"
    ).strip()
    assert resolved == "t"
