"""Tier 4 Scenarios 17-18: Tier 1 /app vs Tier 2 /studio 6 Consoles & Follow-up 4 RFP Concepts/Guard/Tags."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from conftest import fail_missing_target
from lib import (
    APP_ROUTES,
    STUDIO_ROUTES,
    CliBinaryNotFoundError,
    FixtureFactory,
    McpHttpClient,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    WebClient,
)
from lib.http_client import extract_tag_facets
from tier4._helpers import (
    extract_id,
    extract_search_hits,
    hit_doc_ids,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    running_web_url,
    seed_institution_org,
)

pytestmark = [pytest.mark.tier4]


@pytest.mark.f28
@pytest.mark.f29
@pytest.mark.f33
@pytest.mark.f34
def test_s17_tier1_app_zero_jargon_journey_vs_tier2_studio_6_consoles(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        tsx_violations = web_client.scan_app_tsx_copy()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert tsx_violations == {}, f"Forbidden technical jargon in /app TSX: {tsx_violations}"

    web_client.run_typecheck()
    web_client.run_lint()
    web_client.run_build()

    ctx = seed_institution_org(migrated_pg, label="S17")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s17")
    client = api_client.with_api_key(key)

    # Tier 1 /app user journey
    hwpx = sample_factory.hwpx_bytes(title="사용자 콘솔 업로드", paragraphs=["전사 휴가 및 경조사 안내 S17"])
    up = client.app_upload(
        filename="user_console.hwpx",
        content=hwpx,
        fields={"access_scope": "dept", "sensitivity": "normal"},
    ).assert_status((200, 201, 202)).json()
    doc_id = extract_id(up, "document_id", "id", "doc_id")

    chat = client.app_chat({"message": "경조사 안내 S17", "query": "경조사 안내 S17"}).assert_status(200).json()
    assert isinstance(chat, dict)

    page_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.pages WHERE document_id = '{doc_id}' LIMIT 1;"
    ).strip()
    if page_id:
        thumb = client.app_page_thumbnail(page_id).assert_status(200)
        assert len(thumb.body) > 0

    client.app_feedback({"query_text": "경조사 안내 S17", "verdict": "up"}).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    mem_id = extract_id(mcp.memory_reflect("S17 사용자 메모리", scope="private").json(), "memory_id", "id")
    client.app_memory_patch({"memory_id": mem_id, "op": "amend", "text": "S17 사용자 메모리 개정"}).assert_status((200, 201))
    client.app_memory_promote({"memory_id": mem_id, "to": "dept"}).assert_status((200, 201))
    client.app_wiki().assert_status(200)

    connect_api = client.app_connect().assert_status(200).text
    assert any(tok in connect_api for tok in ("mcp", "Claude", "Cursor", "Codex"))

    # Tier 2 /studio 6 operational consoles
    client.studio_pipeline_get().assert_status(200)
    client.studio_pipeline_test({"query": "경조사 안내 S17"}).assert_status(200)
    client.studio_index_engines().assert_status(200)
    client.studio_index_outbox().assert_status(200)
    client.studio_iam_request("groups").assert_status(200)
    client.studio_iam_simulate(
        {"principal_id": ctx["org_admin_id"], "action": "read", "resource_arn": f"arn:openrag:{ctx['slug']}:*"}
    ).assert_status(200)
    client.studio_guard_test({"text": "010-1234-5678 확인", "surface": "llm_egress"}).assert_status(200)
    client.studio_gateway_request("providers").assert_status(200)
    client.studio_eval_request("golden-sets").assert_status(200)
    client.studio_eval_request("leaderboard").assert_status(200)
    client.studio_wiki_conflicts().assert_status(200)
    client.studio_plugins_connectors().assert_status(200)
    client.studio_concepts_request("").assert_status(200)

    # Live Next.js HTML route verification + /app/connect 4-client MCP snippets
    with running_web_url(web_client, api_client.base_url) as web_url:
        for route in APP_ROUTES:
            resp = web_client.check_route_http(web_url, route)
            if route == "/app/connect":
                checks = web_client.verify_connect_mcp_snippets(resp.text)
                assert all(checks.values()), f"Missing MCP snippets on /app/connect: {checks}"
        for route in STUDIO_ROUTES:
            web_client.check_route_http(web_url, route)


@pytest.mark.f11
@pytest.mark.f16
@pytest.mark.f24
@pytest.mark.f28
@pytest.mark.f29
@pytest.mark.f33
@pytest.mark.f34
@pytest.mark.f36
@pytest.mark.f37
def test_s18_followup4_bilingual_rfp_concept_aliases_obfuscated_guard_and_tag_search(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S18")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s18")
    client = api_client.with_api_key(key)
    bslug = f"s18-rfp-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "RfpFollowup4", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    # 1. Parenthetical HWPX ingest auto-extracts 제안요청서(RFP), 사과(apple)
    pdoc = sample_factory.parenthetical_alias_doc(
        title="2026 조달 제안요청서 안내",
        pairs=(("제안요청서", "RFP"), ("사과", "apple")),
        extra_paragraphs=("공공 조달 RFP_S18_CANARY 표준 절차",),
    )
    hwpx_path = sample_factory.write_bytes("rfp_bilingual.hwpx", pdoc["hwpx_bytes"])
    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.ingest(hwpx_path, bucket=bslug, scope="dept", auth_level=3, tags=["procurement", "2026"], env=env, check=True)

    http_mcp = McpHttpClient(client)
    http_mcp.handshake()
    rfp_res = http_mcp.concept_resolve("RFP")
    assert rfp_res.is_error is False and "제안요청서" in rfp_res.text

    # Manual concept linking across CLI, HTTP, and Stdio MCP
    openrag_cli.concept_link("llmwiki", "세컨드 브레인", canonical_name="llmwiki", lang="ko", env=env, check=True)
    client.link_concepts({"source": "llmwiki", "target": "LLM Wiki", "canonical_name": "llmwiki", "lang": "en"}).assert_status((200, 201))

    with mcp_stdio(api_key=key, principal_id=ctx["org_admin_id"], env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        assert mcp.concept_link("llmwiki", "Second Brain", canonical_name="llmwiki", lang="en").is_error is False
        assert mcp.concept_link("llmwiki", "세컨드브레인", canonical_name="llmwiki", lang="ko").is_error is False

        # 2. Cross-lingual / synonym search + wiki/memory entity deduplication
        client.create_document(
            bid,
            {
                "title": "LLMWiki Guide",
                "text": "llmwiki 기반 지식 연결 안내 S18_WIKI_ALIAS",
                "access_scope": "dept",
                "tags": ["llmwiki", "approved"],
            },
        ).assert_status((200, 201))
        mcp.memory_reflect("llmwiki 그래프 메모리 S18_MEM_1", scope="dept")
        mcp.memory_reflect("Second Brain 그래프 메모리 S18_MEM_2", scope="dept")

        assert "S18_WIKI_ALIAS" in client.search_bucket(bid, query="세컨드 브레인", top_k=5).assert_status(200).text
        assert "S18_WIKI_ALIAS" in mcp.rag_search("세컨드 브레인", bucket=bslug, top_k=5).text
        assert "RFP_S18_CANARY" in client.search_bucket(bid, query="제안요청서", top_k=5).assert_status(200).text

        concepts_list = client.list_concepts(query="llmwiki").assert_status(200).json()
        matching_ids = {
            str(c.get("id") or c.get("concept_id"))
            for c in (concepts_list if isinstance(concepts_list, list) else concepts_list.get("concepts", []))
            if isinstance(c, dict)
        }
        assert len(matching_ids) == 1
        assert "S18_MEM_1" in mcp.memory_engage("Second Brain", scope="dept").text

        # 3. NFKC & obfuscated Safe-Word alias blocking
        rule = client.studio_pii_request(
            "guard-rules",
            method="POST",
            payload={
                "rule_kind": "safe_word",
                "pattern_or_category": "사과",
                "action": "block",
                "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"],
                "enabled": True,
            },
        ).assert_status((200, 201)).json()
        rule_id = extract_id(rule, "id", "rule_id")
        try:
            obf = sample_factory.obfuscated_variants("apple")
            for variant in ("사과", "apple", obf["fullwidth_nfkc"], obf["zero_width"], obf["punct_spaced"], obf["mixed"]):
                assert client.chat_completions(
                    {"model": "openrag-auto", "messages": [{"role": "user", "content": f"{variant} 조달"}]}
                ).status == 403
                g_std = client.studio_guard_test({"text": f"{variant} 확인", "surface": "inbound_query"}).assert_status(200).json()
                assert g_std.get("decision") == "block"
                g_mcp = mcp.guard_check(f"{variant} 확인", surface="mcp_egress")
                assert g_mcp.is_error is True or g_mcp.json().get("decision") == "block"
                assert mcp.rag_ask(f"{variant} 요약").is_error is True

            assert client.chat_completions(
                {"model": "openrag-auto", "messages": [{"role": "user", "content": "바나나 조달 안내"}]}
            ).status == 200
        finally:
            client.studio_pii_request(f"guard-rules/{rule_id}", method="DELETE")

        # 4. Tag-based search (`tags_any`, `tags_all`, `tags_exclude`, inline `#tag` / `tag:...`, `expand_tag_aliases`, facets)
        bundle = sample_factory.tagged_corpus_bundle()
        ids_by_key: dict[str, str] = {}
        for item in bundle:
            created = client.create_document(
                bid,
                {
                    "title": item["title"],
                    "text": item["body"],
                    "access_scope": "dept",
                    "tags": item["tags"],
                },
            ).assert_status((200, 201)).json()
            ids_by_key[item["key"]] = extract_id(created, "document_id", "id", "doc_id")

        client.add_document_tags(bid, ids_by_key["d4_finance_2026_approved"], ["security"]).assert_status((200, 201))

        r_any = client.search_bucket(
            bid, query="공통규정키워드T9", tags_any=["finance", "2025"], top_k=10
        ).assert_status(200)
        assert hit_doc_ids(extract_search_hits(r_any.json())) >= {
            ids_by_key["d3_hr_2025_approved"],
            ids_by_key["d4_finance_2026_approved"],
        }

        r_all_exc = client.search_bucket(
            bid, query="공통규정키워드T9", tags_all=["hr", "2026"], tags_exclude=["draft"], top_k=10
        ).assert_status(200)
        got_all_exc = hit_doc_ids(extract_search_hits(r_all_exc.json()))
        assert ids_by_key["d2_hr_2026_approved"] in got_all_exc
        assert ids_by_key["d1_hr_2026_draft"] not in got_all_exc

        r_inline = mcp.rag_search("#security 공통규정키워드T9", bucket=bslug, top_k=5)
        assert r_inline.is_error is False and "재무 지출 기준" in r_inline.text
        assert client.app_chat({"bucket_id": bid, "message": "tag:security 기준", "query": "tag:security 기준"}).status == 200

        r_no_exp = client.search_bucket(
            bid, query="지식관리", tags_any=["세컨드브레인"], expand_tag_aliases=False, top_k=5
        ).assert_status(200)
        assert ids_by_key["d5_llmwiki"] not in hit_doc_ids(extract_search_hits(r_no_exp.json()))

        r_exp = client.search_bucket(
            bid, query="#세컨드브레인 지식관리", expand_tag_aliases=True, top_k=5
        ).assert_status(200)
        assert ids_by_key["d5_llmwiki"] in hit_doc_ids(extract_search_hits(r_exp.json()))
        facets = extract_tag_facets(r_exp)
        assert facets, f"Expected non-empty tag facets in {r_exp.json()!r}"

    assert "llmwiki" in client.app_concepts_request("").assert_status(200).text
    assert "llmwiki" in client.studio_concepts_request("").assert_status(200).text
    assert "security" in client.studio_knowledge_request("tags").assert_status(200).text
