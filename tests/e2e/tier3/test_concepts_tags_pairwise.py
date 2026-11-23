"""Tier 3 pairwise E2E tests: Follow-up 4 Concept/Alias Graph, NFKC Guard & Tag Search (6 tests)."""

from __future__ import annotations

import uuid

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
from lib.http_client import extract_tag_facets
from tier3._helpers import (
    extract_hits,
    extract_id,
    hit_doc_ids,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_org_with_users,
)

pytestmark = [pytest.mark.tier3]


@pytest.mark.f36
@pytest.mark.f11
@pytest.mark.f24
def test_t3_f36_f11_hwpx_parenthetical_extraction_auto_links_and_expands_search(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F3611")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3611")
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"paren-{uuid.uuid4().hex[:6]}", "name": "ParenHWPX", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    pdoc = sample_factory.parenthetical_alias_doc(
        title="공공 조달 용어집",
        pairs=(("제안요청서", "RFP"), ("세컨드 브레인", "LLM Wiki")),
        extra_paragraphs=("고유조달문구T3_PAREN_99",),
    )
    client.upload_document(
        bid,
        filename="paren_glossary.hwpx",
        content=pdoc["hwpx_bytes"],
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202))

    alias_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM wiki.concept_aliases WHERE org_id = '{ctx['org_id']}' AND source = 'ingest_parenthetical';"
        ).strip()
    )
    assert alias_cnt >= 2

    srch_rfp = client.search_bucket(bid, query="RFP", top_k=5).assert_status(200)
    assert "제안요청서" in srch_rfp.text or "T3_PAREN_99" in srch_rfp.text
    srch_wiki = client.search_bucket(bid, query="LLM Wiki", top_k=5).assert_status(200)
    assert "세컨드 브레인" in srch_wiki.text or "T3_PAREN_99" in srch_wiki.text


@pytest.mark.f36
@pytest.mark.f16
@pytest.mark.f29
def test_t3_f36_f16_f29_concept_alias_safeword_blocks_nfkc_obfuscated_mcp_and_chat(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: object,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F3616")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3616")
    client = api_client.with_api_key(key)

    rule = client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "사과",
            "action": "block",
            "surfaces": ["inbound_query", "mcp_egress", "llm_egress", "ingestion_gate"],
            "enabled": True,
        },
    ).assert_status((200, 201)).json()
    rule_id = extract_id(rule, "id", "rule_id")

    obf = sample_factory.obfuscated_variants("apple")
    try:
        with mcp_stdio(  # type: ignore[operator]
            api_key=key,
            principal_id=ctx["admin_id"],
            env={"DATABASE_URL": migrated_pg.dsn()},
        ) as mcp:
            mcp.handshake()
            link_res = mcp.concept_link("사과", "apple", canonical_name="apple", lang="ko")
            assert link_res.is_error is False

            for probe in ("apple", obf["fullwidth_nfkc"], obf["zero_width"], obf["mixed"]):
                chk = mcp.guard_check(f"요청 {probe} 확인", surface="mcp_egress")
                assert chk.is_error is True or chk.json().get("decision") == "block"

                ask = mcp.rag_ask(f"{probe} 정보 알려줘")
                assert ask.is_error is True

                chat = client.chat_completions(
                    {"model": "openrag-auto", "messages": [{"role": "user", "content": f"{probe} 보고"}]}
                )
                assert chat.status == 403 and "guard_blocked" in chat.text.lower()

            ok_chk = mcp.guard_check("application 프로세스 확인", surface="mcp_egress")
            assert ok_chk.is_error is False and ok_chk.json().get("decision") == "allow"
    finally:
        client.studio_pii_request(f"guard-rules/{rule_id}", method="DELETE")

    ev_meta = migrated_pg.execute_sql(
        f"SELECT COALESCE(string_agg(metadata::text, ' '), '') FROM pii.guard_events WHERE org_id = '{ctx['org_id']}';"
    )
    assert "apple" not in ev_meta.lower() and "사과" not in ev_meta


@pytest.mark.f36
@pytest.mark.f18
@pytest.mark.f20
def test_t3_f36_f18_f20_concept_deduplicates_memory_reflect_and_wiki_entities(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F3618")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3618")
    client = api_client.with_api_key(key)
    bslug = f"dedup-{uuid.uuid4().hex[:6]}"
    client.create_bucket({"slug": bslug, "name": "ConceptDedup", "access_scope": "dept"}).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    assert mcp.concept_link("llmwiki", "세컨드 브레인", canonical_name="llmwiki", lang="ko").is_error is False
    assert mcp.concept_link("llmwiki", "Second Brain", canonical_name="llmwiki", lang="en").is_error is False

    r1 = mcp.memory_reflect("llmwiki 그래프 구조 메모 T3_DEDUP_A", scope="dept")
    r2 = mcp.memory_reflect("세컨드 브레인 백링크 메모 T3_DEDUP_B", scope="dept")
    assert r1.is_error is False and r2.is_error is False

    client.sync_documents(
        bslug,
        {
            "documents": [
                {"external_key": "w-1", "title": "llmwiki 안내", "text": "llmwiki 페이지 본문", "access_scope": "dept"},
                {"external_key": "w-2", "title": "세컨드 브레인 안내", "text": "세컨드 브레인 페이지 본문", "access_scope": "dept"},
            ]
        },
    ).assert_status((200, 201))

    eng = mcp.memory_engage("세컨드 브레인", scope="dept")
    assert eng.is_error is False
    assert "T3_DEDUP_A" in eng.text and "T3_DEDUP_B" in eng.text

    concepts_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(DISTINCT c.id) FROM wiki.concepts c JOIN wiki.concept_aliases a ON a.concept_id = c.id WHERE c.org_id = '{ctx['org_id']}' AND a.normalized_alias IN ('llmwiki', '세컨드 브레인', 'second brain');"
        ).strip()
    )
    assert concepts_cnt == 1


@pytest.mark.f37
@pytest.mark.f07
@pytest.mark.f08
@pytest.mark.f24
def test_t3_f37_f07_f24_tag_filtering_and_facets_respect_private_rls_and_groups(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F3707")
    tf_grp = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, name, group_type, allowed_actions)
        VALUES ('{tf_grp}', '{ctx['org_id']}', 'tag-tf-{tf_grp[:6]}', 'project_tf', ARRAY['read', 'search']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
            ('{ctx['org_id']}', '{tf_grp}', '{ctx['owner_id']}'),
            ('{ctx['org_id']}', '{tf_grp}', '{ctx['other_dept_id']}');
        """
    )

    owner_key = mint_api_key(api_client, ctx["org_id"], ctx["owner_id"], tag="t3-3707-own")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="t3-3707-peer")
    other_key = mint_api_key(api_client, ctx["org_id"], ctx["other_dept_id"], tag="t3-3707-oth")

    b = (
        api_client.with_api_key(owner_key)
        .create_bucket({"slug": f"tag-rls-{uuid.uuid4().hex[:6]}", "name": "TagRls", "access_scope": "org"})
        .assert_status((200, 201))
        .json()
    )
    bid = extract_id(b, "id", "bucket_id")

    d_priv = extract_id(
        api_client.with_api_key(owner_key)
        .create_document(
            bid,
            {
                "title": "PrivTagged",
                "text": "TAG_RLS_KEY 비밀 개인 문서",
                "access_scope": "private",
                "tags": ["shared-tag", "secret-tag"],
            },
        )
        .assert_status((200, 201))
        .json(),
        "document_id",
        "id",
        "doc_id",
    )
    d_grp = extract_id(
        api_client.with_api_key(owner_key)
        .create_document(
            bid,
            {
                "title": "GrpTagged",
                "text": "TAG_RLS_KEY TF 공유 문서",
                "access_scope": "group",
                "allowed_group_ids": [tf_grp],
                "tags": ["shared-tag", "tf-tag"],
            },
        )
        .assert_status((200, 201))
        .json(),
        "document_id",
        "id",
        "doc_id",
    )
    d_dept = extract_id(
        api_client.with_api_key(owner_key)
        .create_document(
            bid,
            {
                "title": "DeptTagged",
                "text": "TAG_RLS_KEY 부서 문서",
                "access_scope": "dept",
                "tags": ["shared-tag"],
            },
        )
        .assert_status((200, 201))
        .json(),
        "document_id",
        "id",
        "doc_id",
    )

    peer_srch = api_client.with_api_key(peer_key).search_bucket(
        bid, query="TAG_RLS_KEY", tags_any=["shared-tag", "secret-tag"], top_k=10
    ).assert_status(200)
    peer_ids = hit_doc_ids(extract_hits(peer_srch.json()))
    assert d_dept in peer_ids
    assert d_priv not in peer_ids
    assert d_grp not in peer_ids
    assert "secret-tag" not in extract_tag_facets(peer_srch)

    other_srch = api_client.with_api_key(other_key).search_bucket(
        bid, query="TAG_RLS_KEY", tags_all=["shared-tag", "tf-tag"], tags_exclude=["secret-tag"], top_k=10
    ).assert_status(200)
    other_ids = hit_doc_ids(extract_hits(other_srch.json()))
    assert d_grp in other_ids and d_priv not in other_ids

    peer_pg = migrated_pg.query_as_principal(
        "SELECT id::text FROM corpus.documents WHERE 'secret-tag' = ANY(tags) AND org_id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"], principal_id=ctx["peer_id"], auth_level=6, dept_ids=(ctx["dept1_id"],)
        ),
        params=(ctx["org_id"],),
    )
    assert peer_pg == []


@pytest.mark.f37
@pytest.mark.f36
@pytest.mark.f29
def test_t3_f37_f36_expand_tag_aliases_across_http_cli_and_mcp_rag_search(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    mcp_stdio: object,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F3736")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3736")
    client = api_client.with_api_key(key)
    bslug = f"tag-al-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "TagAlias", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.concept_link("llmwiki", "세컨드브레인", canonical_name="llmwiki", lang="ko", env=env, check=True)

    md_path = sample_factory.write_text(
        "llmwiki_doc.md",
        sample_factory.tagged_markdown(
            title="LLMWiki 태그 테스트",
            tags=["llmwiki"],
            body="별칭 태그 확장 검색 검증 본문 T3_TAG_ALIAS",
        ),
    )
    openrag_cli.ingest(md_path, bucket=bslug, scope="dept", auth_level=3, tags=["llmwiki"], env=env, check=True)

    no_exp = client.search_bucket(
        bid, query="T3_TAG_ALIAS", tags_any=["세컨드브레인"], expand_tag_aliases=False, top_k=5
    ).assert_status(200)
    assert len(extract_hits(no_exp.json())) == 0

    with_exp = client.search_bucket(
        bid, query="T3_TAG_ALIAS", tags_any=["세컨드브레인"], expand_tag_aliases=True, top_k=5
    ).assert_status(200)
    assert len(extract_hits(with_exp.json())) >= 1

    with mcp_stdio(api_key=key, principal_id=ctx["admin_id"], env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:  # type: ignore[operator]
        mcp.handshake()
        mcp_res = mcp.rag_search(
            "#세컨드브레인 T3_TAG_ALIAS",
            bucket=bslug,
            expand_tag_aliases=True,
            top_k=5,
        )
        assert mcp_res.is_error is False and "T3_TAG_ALIAS" in mcp_res.text


@pytest.mark.f37
@pytest.mark.f33
@pytest.mark.f34
def test_t3_f37_f33_f34_app_chat_inline_tags_and_studio_knowledge_tags_console(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
) -> None:
    try:
        assert web_client.scan_app_tsx_copy() == {}
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F3733")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3733")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"tag-ui-{uuid.uuid4().hex[:6]}", "name": "TagUI", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    doc = client.create_document(
        bid, {"title": "TagUiDoc", "text": "인라인 태그 챗 검증 T3_INLINE_TAG", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")
    client.add_document_tags(bid, doc_id, ["security", "policy-2026"]).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    mem_id = extract_id(
        mcp.memory_reflect("태그 UI 메모리 T3", scope="dept").json(), "memory_id", "id"
    )
    client.add_memory_tags(mem_id, ["security"]).assert_status((200, 201))

    wiki_id = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES ('{wiki_id}', '{ctx['org_id']}', '{bid}', '{ctx['dept1_id']}', '{ctx['admin_id']}', 'dept', 1, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L1/{wiki_id}', '{ctx['org_id']}:dept', 'tag-ui-w-{wiki_id[:6]}', 'TagWiki', 'WikiBody', 'WikiBody', '{uuid.uuid4().hex}');
        """
    )
    client.add_wiki_tags(wiki_id, ["policy-2026"]).assert_status((200, 201))

    chat_res = client.app_chat(
        {"bucket_id": bid, "message": "#security T3_INLINE_TAG 안내해줘", "query": "#security T3_INLINE_TAG"}
    ).assert_status(200)
    assert len(chat_res.text) > 0

    client.app_concepts_request(
        "link", method="POST", payload={"source": "보안", "target": "security", "lang": "ko"}
    ).assert_status((200, 201))
    assert "보안" in client.app_concepts_request("").assert_status(200).text
    assert "보안" in client.studio_concepts_request("").assert_status(200).text

    studio_tags = client.studio_knowledge_request("tags").assert_status(200).text
    assert "security" in studio_tags and "policy-2026" in studio_tags
