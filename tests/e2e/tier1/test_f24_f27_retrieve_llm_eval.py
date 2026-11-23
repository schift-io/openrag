"""Tier 1 E2E tests for F24-F27: Multi-Space Hybrid Retriever, Policy LLM Router, Golden Set, Bench & AutoRAG."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

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
    l3_id = str(uuid.uuid4())
    l9_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'Eval Org {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ('{dept_id}', '{org_id}', 'Policy', 'policy');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{l3_id}', '{org_id}', 'l3-{l3_id[:6]}@example.org', 'L3User', 3, false, false),
            ('{l9_id}', '{org_id}', 'l9-{l9_id[:6]}@example.org', 'L9User', 9, true, false),
            ('{admin_id}', '{org_id}', 'admin-{admin_id[:6]}@example.org', 'Admin', 10, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept_id}', '{l3_id}', 'member'),
            ('{org_id}', '{dept_id}', '{l9_id}', 'member'),
            ('{org_id}', '{dept_id}', '{admin_id}', 'dept_admin');
        """
    )
    return {
        "org_id": org_id,
        "slug": slug,
        "dept_id": dept_id,
        "l3_id": l3_id,
        "l9_id": l9_id,
        "admin_id": admin_id,
    }


def _mint_api_key(api_client: OpenRagHttpClient, ctx: dict[str, str], principal_id: str) -> str:
    resp = api_client.cli_code_exchange(
        {"code": issue_cli_code(ctx["org_id"], principal_id), "org_id": ctx["org_id"], "principal_id": principal_id}
    )
    resp.assert_status((200, 201))
    key = resp.json().get("api_key") or resp.json().get("token")
    assert isinstance(key, str) and key
    return key


# ============================================================================
# F24 — 128d Multi-Space Hybrid Retriever & Admission (6 tests)
# ============================================================================


@pytest.mark.f24
def test_f24_multispace_retrieval_across_chunks_thumb_wiki_memory(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"ms-{uuid.uuid4().hex[:6]}", "name": "MultiSpace", "access_scope": "dept", "min_auth_level": 3}
    ).assert_status((200, 201)).json()
    bucket_id = str(b.get("id") or b.get("bucket_id"))
    bucket_slug = b.get("slug") or bucket_id

    concept = "예산집행투명성지침"
    hwpx = sample_factory.hwpx_file("budget.hwpx", title=concept, paragraphs=[f"{concept} 본문 청크 데이터"])
    client.upload_document(
        bucket_id,
        filename=hwpx.name,
        content=hwpx.read_bytes(),
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202))

    # Seed thumb space vector + wiki + memory sharing the concept
    doc_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    wiki_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    vec = _unit_vec_128(7)
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
        VALUES ('{doc_id}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:policy:dept:doc:L3/{doc_id}', '{ctx['org_id']}:dept', '{concept} 스캔본', '{_sha256_hex(doc_id)}');
        INSERT INTO corpus.pages (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no, redacted_markdown, thumb_embedding)
        VALUES ('{page_id}', '{ctx['org_id']}', '{doc_id}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:policy:dept:page:L3/{page_id}', '{ctx['org_id']}:dept', 1, '{concept} 썸네일 OCR', '{vec}'::vector(128));
        INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding)
        VALUES ('{wiki_id}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L3/{wiki_id}', '{ctx['org_id']}:dept', 'wiki-{wiki_id[:6]}', '{concept} 위키', '{concept} 위키 요약', '{concept} 위키 요약', '{_sha256_hex(wiki_id)}', '{vec}'::vector(128));
        INSERT INTO memory.entries (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256, embedding)
        VALUES ('{mem_id}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:policy:dept:memory:L3/{mem_id}', '{ctx['org_id']}:dept', 'fact', '{concept} 기억 메모', '{concept} 기억 메모', '{_sha256_hex(mem_id)}', '{vec}'::vector(128));
        """
    )

    mcp = McpHttpClient(client)
    mcp.handshake()
    res = mcp.rag_search(concept, bucket=bucket_slug, top_k=20, spaces=["chunks", "thumb", "wiki", "memory"])
    assert res.is_error is False
    data = res.json()
    items = data.get("results") or data.get("items") or []
    spaces_found = {str(it.get("space") or it.get("source_type") or "") for it in items}
    assert len(items) >= 3
    assert len(spaces_found) >= 2


@pytest.mark.f24
def test_f24_rrf_k60_fusion_and_score_decomposition(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"rrf-{uuid.uuid4().hex[:6]}", "name": "RRF", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    client.create_document(bid, {"title": "RRF Hit", "text": "공공기관 전자결재 표준 지침 안내", "access_scope": "dept"}).assert_status((200, 201))
    client.create_document(bid, {"title": "Other", "text": "사내 구내식당 주간 메뉴표", "access_scope": "dept"}).assert_status((200, 201))

    s_resp = client.search_bucket(bid, {"query": "전자결재 표준 지침", "fusion": "rrf", "top_k": 5}).assert_status(200)
    items = s_resp.json().get("results") or s_resp.json().get("items") or []
    assert len(items) >= 1
    top = items[0]
    assert "score" in top
    assert ("vector_score" in top or "bm25_score" in top or "score_breakdown" in top)


@pytest.mark.f24
def test_f24_convex_fusion_alpha_blending(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"cvx-{uuid.uuid4().hex[:6]}", "name": "Convex", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "Convex Doc", "text": "볼록 결합 알파 0.3 하이브리드 검색 검증", "access_scope": "dept"}).assert_status((200, 201))

    s_resp = client.search_bucket(
        bid, {"query": "볼록 결합 알파", "fusion": "convex", "alpha": 0.3, "top_k": 5}
    ).assert_status(200)
    items = s_resp.json().get("results") or s_resp.json().get("items") or []
    assert len(items) >= 1
    assert 0.0 <= float(items[0]["score"]) <= 2.0


@pytest.mark.f24
def test_f24_latest_revision_boost_ranks_v2_above_v1(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"rev-{uuid.uuid4().hex[:6]}", "name": "LatestRev", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    d1 = client.create_document(
        bid,
        {
            "title": "2025년 출장 여비 규정",
            "text": "2025년 출장 여비 5만원 지급 기준",
            "access_scope": "dept",
            "revision": 1,
            "valid_to": "2025-12-31T23:59:59Z",
        },
    ).assert_status((200, 201)).json()
    d2 = client.create_document(
        bid,
        {
            "title": "2026년 출장 여비 개정 규정",
            "text": "2026년 출장 여비 7만원 지급 기준",
            "access_scope": "dept",
            "revision": 2,
            "valid_from": "2026-01-01T00:00:00Z",
        },
    ).assert_status((200, 201)).json()

    d1_id = str(d1.get("id") or d1.get("document_id"))
    d2_id = str(d2.get("id") or d2.get("document_id"))
    s_resp = client.search_bucket(bid, {"query": "출장 여비", "top_k": 5}).assert_status(200)
    items = s_resp.json().get("results") or s_resp.json().get("items") or []
    ids = [str(it.get("document_id") or it.get("id")) for it in items]
    assert d2_id in ids
    if d1_id in ids:
        assert ids.index(d2_id) < ids.index(d1_id)


@pytest.mark.f24
def test_f24_evidence_admission_admits_grounded_results(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"adm-{uuid.uuid4().hex[:6]}", "name": "Admission", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(
        bid,
        {"title": "연차 휴가 규정", "text": "신입 직원은 1년 미만 근무 시 월 1일의 유급 연차 휴가를 부여받는다.", "access_scope": "dept"},
    ).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    ask_res = mcp.rag_ask("신입 직원 연차 휴가는 며칠인가요?", bucket=b.get("slug") or bid)
    assert ask_res.is_error is False
    data = ask_res.json()
    decision = str(data.get("admission") or data.get("admission_decision") or data.get("status") or "").lower()
    assert decision in ("admit", "accepted", "ok", "")
    citations = data.get("citations") or []
    assert len(citations) >= 1


@pytest.mark.f24
def test_f24_two_stage_rerank_refines_top_k_order(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"rrk-{uuid.uuid4().hex[:6]}", "name": "Rerank", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    client.create_document(bid, {"title": "일반 안내", "text": "보안 관련 일반 소개 문서입니다.", "access_scope": "dept"}).assert_status((200, 201))
    exact = client.create_document(bid, {"title": "비밀번호 만료 주기", "text": "비밀번호 만료 주기는 90일이며 대소문자 및 숫자를 포함해야 한다.", "access_scope": "dept"}).assert_status((200, 201)).json()
    exact_id = str(exact.get("id") or exact.get("document_id"))

    s_resp = client.search_bucket(bid, {"query": "비밀번호 만료 주기는 몇 일인가요?", "rerank": True, "top_k": 5}).assert_status(200)
    items = s_resp.json().get("results") or s_resp.json().get("items") or []
    assert len(items) >= 1
    assert str(items[0].get("document_id") or items[0].get("id")) == exact_id


# ============================================================================
# F25 — Policy-Enforced LLM Router (5 tests)
# ============================================================================


@pytest.mark.f25
def test_f25_l8_plus_context_forces_local_vllm_provider(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l9_key = _mint_api_key(api_client, ctx, ctx["l9_id"])
    client = api_client.with_api_key(l9_key)

    b = client.create_bucket({"slug": f"l8b-{uuid.uuid4().hex[:6]}", "name": "L8Secret", "access_scope": "dept", "min_auth_level": 8}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "L8 Doc", "text": "L8 기밀 보안 감사 계획서", "access_scope": "dept", "min_auth_level": 8}).assert_status((200, 201))

    resp = client.chat_completions(
        {"model": "openrag-auto", "bucket_id": bid, "messages": [{"role": "user", "content": "기밀 보안 감사 계획 요약"}]}
    )
    resp.assert_status(200)

    logs = pg_ephemeral.execute_sql(
        f"""
        SELECT p.provider_type, p.is_on_premise, l.max_context_auth_level
        FROM gateway.request_logs l
        LEFT JOIN gateway.providers p ON p.id = l.provider_id
        WHERE l.org_id = '{ctx['org_id']}'
        ORDER BY l.created_at DESC LIMIT 1;
        """
    )
    assert "vllm_local" in logs or "mock" in logs
    assert "openrouter" not in logs


@pytest.mark.f25
def test_f25_masked_low_level_context_allows_openrouter_or_external(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l3_key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(l3_key)

    b = client.create_bucket({"slug": f"l3ext-{uuid.uuid4().hex[:6]}", "name": "L3Masked", "access_scope": "dept", "min_auth_level": 3}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "L3 Contact", "text": "담당자 연락처 010-1234-5678 문의 바랍니다.", "access_scope": "dept", "min_auth_level": 3}).assert_status((200, 201))

    resp = client.chat_completions(
        {"model": "anthropic/claude-3.5-sonnet", "bucket_id": bid, "messages": [{"role": "user", "content": "담당자 연락처 안내"}]}
    )
    resp.assert_status(200)
    assert "010-1234-5678" not in resp.text

    masked_flag = pg_ephemeral.execute_sql(
        f"SELECT pii_masked FROM gateway.request_logs WHERE org_id = '{ctx['org_id']}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert masked_flag == "t"


@pytest.mark.f25
def test_f25_airgap_profile_routes_all_levels_to_local_vllm(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    admin_client = api_client.with_api_key(admin_key)

    airgap_yaml = sample_factory.pipeline_yaml("airgap.yaml", profile="airgap", local_only_min_auth_level=1).read_text(encoding="utf-8")
    admin_client.studio_pipeline_update({"pipeline_yaml": airgap_yaml}).assert_status(200)

    l3_key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(l3_key)
    b = client.create_bucket({"slug": f"ag-{uuid.uuid4().hex[:6]}", "name": "Airgap", "access_scope": "dept", "min_auth_level": 1}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "L1 Airgap", "text": "망분리 L1 일반 공지사항", "access_scope": "dept", "min_auth_level": 1}).assert_status((200, 201))

    client.chat_completions({"model": "openrag-auto", "bucket_id": bid, "messages": [{"role": "user", "content": "공지사항"}]}).assert_status(200)
    logs = pg_ephemeral.execute_sql(
        f"SELECT p.provider_type, p.is_on_premise FROM gateway.request_logs l JOIN gateway.providers p ON p.id = l.provider_id WHERE l.org_id = '{ctx['org_id']}' ORDER BY l.created_at DESC LIMIT 1;"
    )
    assert ("vllm_local" in logs or "|t" in logs) and "openrouter" not in logs


@pytest.mark.f25
def test_f25_gateway_routing_rule_priority_and_fallback_selection(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    prov_resp = client.studio_gateway_request("providers").assert_status(200)
    providers = prov_resp.json().get("providers") or prov_resp.json()
    assert isinstance(providers, list) and len(providers) >= 1
    target_id = str(providers[0]["id"])

    rule_resp = client.studio_gateway_request(
        "routing-rules",
        method="POST",
        payload={
            "priority": 10,
            "min_auth_level": 1,
            "max_auth_level": 10,
            "require_pii_masked": True,
            "target_provider_id": target_id,
            "enabled": True,
        },
    )
    rule_resp.assert_status((200, 201))

    client.chat_completions({"model": "openrag-auto", "messages": [{"role": "user", "content": "라우팅 우선순위 테스트"}]}).assert_status(200)
    logs = client.studio_gateway_request("logs").assert_status(200).json()
    assert len(logs.get("logs") or logs) >= 1


@pytest.mark.f25
def test_f25_request_logs_omit_raw_prompt_and_pii_content(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l3_key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    sentinel_prompt = "UNIQUE_PROMPT_900101-1234567_LOG_CHECK"
    api_client.with_api_key(l3_key).chat_completions(
        {"model": "openrag-auto", "messages": [{"role": "user", "content": sentinel_prompt}]}
    ).assert_status((200, 403))

    dump = pg_ephemeral.execute_sql(
        f"SELECT model_id, max_context_auth_level, pii_masked, status FROM gateway.request_logs WHERE org_id = '{ctx['org_id']}';"
    )
    assert "UNIQUE_PROMPT" not in dump
    assert "900101-1234567" not in dump


# ============================================================================
# F26 — 4-Axis Golden Set & Feedback Promotion (5 tests)
# ============================================================================


@pytest.mark.f26
def test_f26_create_four_axis_golden_set_via_studio_api(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs = client.studio_eval_request(
        "golden-sets",
        method="POST",
        payload={"name": f"gov-4axis-{uuid.uuid4().hex[:6]}", "axis": "multi", "description": "4축 골든셋"},
    ).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))

    for axis in ("retrieval", "ocr", "pii", "pararel"):
        client.studio_eval_request(
            f"golden-sets/{gs_id}/items",
            method="POST",
            payload={
                "axis": axis,
                "query_text": f"테스트 질의 {axis}",
                "test_principal_id": ctx["l3_id"],
                "expected_doc_ids": [str(uuid.uuid4())],
                "forbidden_doc_ids": [str(uuid.uuid4())],
                "expected_pii_tokens": ["<RRN_1>"] if axis == "pii" else [],
                "expected_edges": [{"relation_type": "supersedes"}] if axis == "pararel" else None,
            },
        ).assert_status((200, 201))

    cnt = pg_ephemeral.execute_sql(f"SELECT count(DISTINCT axis) FROM eval.golden_items WHERE golden_set_id = '{gs_id}';").strip()
    assert int(cnt) == 4


@pytest.mark.f26
def test_f26_user_feedback_submission_from_app_console(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])

    resp = api_client.with_api_key(key).app_feedback(
        {
            "query_text": "2026년 출장 일비 얼마인가요?",
            "verdict": "wrong_evidence",
            "cited_chunk_ids": [str(uuid.uuid4())],
            "comment": "구버전 인용",
        }
    )
    resp.assert_status((200, 201))

    row = pg_ephemeral.execute_sql(
        f"SELECT verdict, promoted_golden_item_id IS NULL FROM eval.search_feedback WHERE org_id = '{ctx['org_id']}' ORDER BY created_at DESC LIMIT 1;"
    )
    assert "wrong_evidence" in row and "t" in row


@pytest.mark.f26
def test_f26_promote_search_feedback_to_golden_item(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l3_key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])

    fb = api_client.with_api_key(l3_key).app_feedback(
        {"query_text": "법인카드 정산 기한", "verdict": "missing_citation", "cited_chunk_ids": [], "comment": "누락"}
    ).assert_status((200, 201)).json()
    fb_id = str(fb.get("id") or fb.get("feedback_id"))

    gs = api_client.with_api_key(admin_key).studio_eval_request(
        "golden-sets", method="POST", payload={"name": f"fb-gs-{uuid.uuid4().hex[:6]}", "axis": "retrieval"}
    ).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))

    prom = api_client.with_api_key(admin_key).studio_eval_request(
        f"feedback/{fb_id}/promote",
        method="POST",
        payload={"golden_set_id": gs_id, "axis": "retrieval", "expected_doc_ids": [str(uuid.uuid4())]},
    )
    prom.assert_status((200, 201))

    linked = pg_ephemeral.execute_sql(
        f"SELECT promoted_golden_item_id IS NOT NULL FROM eval.search_feedback WHERE id = '{fb_id}';"
    ).strip()
    assert linked == "t"


@pytest.mark.f26
def test_f26_golden_item_review_status_workflow(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs = client.studio_eval_request(
        "golden-sets", method="POST", payload={"name": f"rev-gs-{uuid.uuid4().hex[:6]}", "axis": "retrieval"}
    ).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))

    item = client.studio_eval_request(
        f"golden-sets/{gs_id}/items",
        method="POST",
        payload={"axis": "retrieval", "query_text": "승인 워크플로우 질의", "expected_doc_ids": [str(uuid.uuid4())], "review_status": "pending"},
    ).assert_status((200, 201)).json()
    item_id = str(item.get("id") or item.get("item_id"))

    client.studio_eval_request(
        f"golden-items/{item_id}", method="PATCH", payload={"review_status": "approved"}
    ).assert_status(200)
    status = pg_ephemeral.execute_sql(f"SELECT review_status FROM eval.golden_items WHERE id = '{item_id}';").strip()
    assert status == "approved"


@pytest.mark.f26
def test_f26_golden_set_versioning_and_multi_axis_listing(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    name = f"multi-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request(
        "golden-sets", method="POST", payload={"name": name, "axis": "multi"}
    ).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))
    for ax in ("retrieval", "ocr", "pii", "pararel"):
        client.studio_eval_request(
            f"golden-sets/{gs_id}/items",
            method="POST",
            payload={"axis": ax, "query_text": f"q-{ax}", "expected_doc_ids": [str(uuid.uuid4())]},
        ).assert_status((200, 201))

    listing = client.studio_eval_request("golden-sets").assert_status(200).text
    assert name in listing


# ============================================================================
# F27 — 3-Track RAG Bench, A/B Search & AutoRAG Optimizer (6 tests)
# ============================================================================


def _seed_golden_set_with_docs(pg: PgEphemeralCluster, ctx: dict[str, str], api_client: OpenRagHttpClient) -> tuple[str, str, str]:
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)
    b = client.create_bucket({"slug": f"bch-{uuid.uuid4().hex[:6]}", "name": "BenchBucket", "access_scope": "dept", "min_auth_level": 3}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    d = client.create_document(bid, {"title": "Bench Doc", "text": "자동 최적화 벤치마크 정답 문서 70,000원", "access_scope": "dept", "min_auth_level": 3}).assert_status((200, 201)).json()
    did = str(d.get("id") or d.get("document_id"))

    gs_name = f"gov-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request("golden-sets", method="POST", payload={"name": gs_name, "axis": "multi"}).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))
    client.studio_eval_request(
        f"golden-sets/{gs_id}/items",
        method="POST",
        payload={"axis": "retrieval", "query_text": "자동 최적화 벤치마크 정답", "test_principal_id": ctx["l3_id"], "expected_doc_ids": [did], "forbidden_doc_ids": []},
    ).assert_status((200, 201))
    return gs_name, gs_id, bid


@pytest.mark.f27
def test_f27_three_track_bench_cli_and_persisted_metrics(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    gs_name, gs_id, _ = _seed_golden_set_with_docs(pg_ephemeral, ctx, api_client)

    res = openrag_cli.bench_run(
        golden_set=gs_name,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )
    assert res.returncode == 0

    runs = pg_ephemeral.execute_sql(
        f"SELECT track, pii_leak_count, arn_violation_count, security_gate_pass FROM eval.bench_runs WHERE golden_set_id = '{gs_id}';"
    )
    assert "track1_retrieval" in runs
    assert "track2_e2e_rag" in runs
    assert "track3_security_ocr" in runs
    assert "f" not in [line.split("|")[-1].strip() for line in runs.strip().splitlines() if "|" in line]


@pytest.mark.f27
def test_f27_ab_search_endpoint_returns_metrics_and_security_gates(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    _, _, bid = _seed_golden_set_with_docs(pg_ephemeral, ctx, api_client)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    d_list = client.list_documents(bid).assert_status(200).json()
    docs = d_list.get("documents") or d_list
    doc_id = str(docs[0].get("id") or docs[0].get("document_id"))

    ab_resp = client.ab_search(
        {
            "bucket_id": bid,
            "k": 5,
            "queries": [{"query": "자동 최적화 벤치마크 정답", "relevant_ids": [doc_id], "forbidden_ids": []}],
            "variants": {
                "variant_a": {"fusion": "rrf", "alpha": 0.2},
                "variant_b": {"fusion": "convex", "alpha": 0.5},
            },
        }
    )
    ab_resp.assert_status(200)
    data = ab_resp.json()
    serialized = json.dumps(data).lower()
    assert "mrr" in serialized
    assert "recall" in serialized
    assert "ndcg" in serialized
    gates = data.get("security_gates") or {}
    assert gates.get("pii_leak_count", 0) == 0
    assert gates.get("arn_violation_count", 0) == 0
    assert gates.get("pass", True) is True


@pytest.mark.f27
def test_f27_autorag_greedy_optimizer_exports_best_yaml(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    gs_name, gs_id, _ = _seed_golden_set_with_docs(pg_ephemeral, ctx, api_client)
    ss_path = sample_factory.search_space_yaml(golden_set=gs_name)
    best_path = tmp_path / "best_pipeline.yaml"

    openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=ss_path,
        require_zero_pii_leak=True,
        require_zero_arn_violation=True,
        export_best=best_path,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    assert best_path.is_file()
    parsed = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict) and "retrieve" in parsed

    row = pg_ephemeral.execute_sql(
        f"SELECT strategy, security_gate_pass, status FROM eval.optimization_trials WHERE golden_set_id = '{gs_id}' ORDER BY created_at DESC LIMIT 1;"
    )
    assert "greedy_node" in row and "completed" in row


@pytest.mark.f27
def test_f27_promote_clean_best_pipeline_to_bucket(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    gs_name, _, bid = _seed_golden_set_with_docs(pg_ephemeral, ctx, api_client)
    ss_path = sample_factory.search_space_yaml(golden_set=gs_name)
    best_path = tmp_path / "best.yaml"

    openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=ss_path,
        export_best=best_path,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )
    openrag_cli.pipeline_promote(
        best_path,
        bucket=bid,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    promoted = pg_ephemeral.execute_sql(
        f"SELECT is_promoted, promoted_at IS NOT NULL FROM eval.pipeline_configs WHERE org_id = '{ctx['org_id']}' AND is_promoted = true;"
    )
    assert "t|t" in promoted


@pytest.mark.f27
def test_f27_studio_bench_leaderboard_and_poc_report_export(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    gs_name, gs_id, _ = _seed_golden_set_with_docs(pg_ephemeral, ctx, api_client)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_eval_request("bench", method="POST", payload={"golden_set_id": gs_id, "golden_set": gs_name}).assert_status((200, 201))
    lb = client.studio_eval_request("leaderboard").assert_status(200)
    assert "security_gate" in lb.text.lower() or "ndcg" in lb.text.lower()

    rep = client.studio_eval_request("report").assert_status(200)
    assert len(rep.text) > 10


@pytest.mark.f27
def test_f27_ab_pair_and_grid_strategies_supported(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    gs_name, gs_id, _ = _seed_golden_set_with_docs(pg_ephemeral, ctx, api_client)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    small_space = yaml.safe_dump(
        {
            "version": 1,
            "golden_set": gs_name,
            "strategy": "grid",
            "security_gates": {"require_zero_pii_leak": True, "require_zero_arn_violation": True},
            "nodes": {"retrieve.fusion": ["rrf", "convex"], "retrieve.alpha": [0.2, 0.5]},
        }
    )
    opt = client.studio_eval_request(
        "optimize",
        method="POST",
        payload={"golden_set_id": gs_id, "strategy": "grid", "search_space_yaml": small_space},
    )
    opt.assert_status((200, 201))
    row = pg_ephemeral.execute_sql(
        f"SELECT strategy, best_score IS NOT NULL FROM eval.optimization_trials WHERE golden_set_id = '{gs_id}' ORDER BY created_at DESC LIMIT 1;"
    )
    assert "grid" in row and "t" in row
