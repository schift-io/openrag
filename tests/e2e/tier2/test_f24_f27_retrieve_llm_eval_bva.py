"""Tier 2 BVA & negative tests for F24-F27: Retrieval, LLM Router, Golden Set, Bench & AutoRAG."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import hashlib
import uuid
from pathlib import Path

import pytest
import yaml

from lib import (
    FixtureFactory,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
)

pytestmark = [pytest.mark.tier2]


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> None:
    pg.ensure_started()
    out = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;")
    if out.strip() != "t":
        pg.apply_migrations(root / "migrations")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_org_and_users(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    l3_id = str(uuid.uuid4())
    l7_id = str(uuid.uuid4())
    l8_id = str(uuid.uuid4())
    l9_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'BVA Eval {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ('{dept_id}', '{org_id}', 'Ops', 'ops');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{l3_id}', '{org_id}', 'l3-{l3_id[:6]}@example.org', 'L3', 3, false, false),
            ('{l7_id}', '{org_id}', 'l7-{l7_id[:6]}@example.org', 'L7', 7, false, false),
            ('{l8_id}', '{org_id}', 'l8-{l8_id[:6]}@example.org', 'L8', 8, true, false),
            ('{l9_id}', '{org_id}', 'l9-{l9_id[:6]}@example.org', 'L9', 9, true, false),
            ('{admin_id}', '{org_id}', 'admin-{admin_id[:6]}@example.org', 'Admin', 10, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept_id}', '{l3_id}', 'member'),
            ('{org_id}', '{dept_id}', '{l7_id}', 'member'),
            ('{org_id}', '{dept_id}', '{l8_id}', 'member'),
            ('{org_id}', '{dept_id}', '{l9_id}', 'member'),
            ('{org_id}', '{dept_id}', '{admin_id}', 'dept_admin');
        """
    )
    return {
        "org_id": org_id,
        "slug": slug,
        "dept_id": dept_id,
        "l3_id": l3_id,
        "l7_id": l7_id,
        "l8_id": l8_id,
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
# F24 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f24
def test_f24_bva_convex_alpha_boundaries_0_0_and_1_0_and_invalid(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"alp-{uuid.uuid4().hex[:6]}", "name": "Alpha", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "Alpha Doc", "text": "알파 경계 테스트 문서", "access_scope": "dept"}).assert_status((200, 201))

    client.search_bucket(bid, {"query": "알파 경계", "fusion": "convex", "alpha": 0.0}).assert_status(200)
    client.search_bucket(bid, {"query": "알파 경계", "fusion": "convex", "alpha": 1.0}).assert_status(200)

    assert client.search_bucket(bid, {"query": "알파 경계", "fusion": "convex", "alpha": -0.01}).status in (400, 422)
    assert client.search_bucket(bid, {"query": "알파 경계", "fusion": "convex", "alpha": 1.01}).status in (400, 422)


@pytest.mark.f24
def test_f24_bva_evidence_admission_abstains_below_threshold(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"abs-{uuid.uuid4().hex[:6]}", "name": "Abstain", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "Office Supplies", "text": "사무용품 구매 신청 절차 안내", "access_scope": "dept"}).assert_status((200, 201))

    resp = client.chat_completions(
        {
            "model": "openrag-auto",
            "bucket_id": bid,
            "admission_threshold": 0.95,
            "messages": [{"role": "user", "content": "화성 탐사선 로켓 연료 배합 비율은?"}],
        }
    ).assert_status(200)
    data = resp.json()
    adm = str(data.get("admission") or data.get("admission_decision") or "").lower()
    assert adm in ("abstain", "denied", "insufficient_evidence") or data.get("citations", []) == []


@pytest.mark.f24
def test_f24_bva_retrieval_strictly_excludes_unauthorized_private_and_high_level_docs(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    l3_key = _mint_api_key(api_client, ctx, ctx["l3_id"])

    b = api_client.with_api_key(admin_key).create_bucket(
        {"slug": f"sec-{uuid.uuid4().hex[:6]}", "name": "SecRet", "access_scope": "dept", "min_auth_level": 1}
    ).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    secret_token = f"HIGH_CONFIDENTIAL_{uuid.uuid4().hex[:6]}"
    api_client.with_api_key(admin_key).create_document(
        bid, {"title": "L9 Secret", "text": f"검색어매칭 {secret_token}", "access_scope": "private", "min_auth_level": 9}
    ).assert_status((200, 201))
    api_client.with_api_key(admin_key).create_document(
        bid, {"title": "L3 Public Dept", "text": "검색어매칭 일반 공개 부서 안내", "access_scope": "dept", "min_auth_level": 3}
    ).assert_status((200, 201))

    s_resp = api_client.with_api_key(l3_key).search_bucket(bid, {"query": f"검색어매칭 {secret_token}", "top_k": 10}).assert_status(200)
    assert secret_token not in s_resp.text


@pytest.mark.f24
def test_f24_bva_top_k_boundaries_0_1_and_exceeding_max(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"topk-{uuid.uuid4().hex[:6]}", "name": "TopK", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "Doc1", "text": "토큰 경계 테스트 1", "access_scope": "dept"}).assert_status((200, 201))

    r1 = client.search_bucket(bid, {"query": "토큰 경계", "top_k": 1}).assert_status(200)
    assert len(r1.json().get("results") or r1.json().get("items") or []) <= 1

    assert client.search_bucket(bid, {"query": "토큰 경계", "top_k": 0}).status in (400, 422)
    r201 = client.search_bucket(bid, {"query": "토큰 경계", "top_k": 201})
    assert r201.status in (200, 400, 422)
    if r201.status == 200:
        assert len(r201.json().get("results") or r201.json().get("items") or []) <= 200


@pytest.mark.f24
def test_f24_bva_expired_valid_to_document_deprioritized_or_filtered(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"exp-{uuid.uuid4().hex[:6]}", "name": "Expired", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    old = client.create_document(
        bid, {"title": "Expired", "text": "주차권 지원 만료 규정 2024", "access_scope": "dept", "revision": 1, "valid_to": "2024-12-31T00:00:00Z"}
    ).assert_status((200, 201)).json()
    cur = client.create_document(
        bid, {"title": "Current", "text": "주차권 지원 현행 규정 2026", "access_scope": "dept", "revision": 2, "valid_from": "2026-01-01T00:00:00Z"}
    ).assert_status((200, 201)).json()

    old_id = str(old.get("id") or old.get("document_id"))
    cur_id = str(cur.get("id") or cur.get("document_id"))
    res = client.search_bucket(bid, {"query": "주차권 지원 규정", "latest_revision_only": True, "top_k": 5}).assert_status(200)
    ids = [str(it.get("document_id") or it.get("id")) for it in (res.json().get("results") or res.json().get("items") or [])]
    assert cur_id in ids
    if old_id in ids:
        assert ids.index(cur_id) < ids.index(old_id)


@pytest.mark.f24
def test_f24_bva_unknown_retrieval_space_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["l3_id"])
    client = api_client.with_api_key(key)
    b = client.create_bucket({"slug": f"unksp-{uuid.uuid4().hex[:6]}", "name": "UnkSpace", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    resp = client.search_bucket(bid, {"query": "테스트", "spaces": ["unknown_space"]})
    assert resp.status in (400, 422)


# ============================================================================
# F25 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f25
def test_f25_bva_auth_level_boundary_L7_vs_L8_routing(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l7_key = _mint_api_key(api_client, ctx, ctx["l7_id"])
    l8_key = _mint_api_key(api_client, ctx, ctx["l8_id"])

    b7 = api_client.with_api_key(l7_key).create_bucket({"slug": f"b7-{uuid.uuid4().hex[:6]}", "name": "B7", "access_scope": "dept", "min_auth_level": 7}).assert_status((200, 201)).json()
    b8 = api_client.with_api_key(l8_key).create_bucket({"slug": f"b8-{uuid.uuid4().hex[:6]}", "name": "B8", "access_scope": "dept", "min_auth_level": 8}).assert_status((200, 201)).json()
    bid7 = str(b7.get("id") or b7.get("bucket_id"))
    bid8 = str(b8.get("id") or b8.get("bucket_id"))

    api_client.with_api_key(l7_key).create_document(bid7, {"title": "L7", "text": "L7 경계 문서", "access_scope": "dept", "min_auth_level": 7}).assert_status((200, 201))
    api_client.with_api_key(l8_key).create_document(bid8, {"title": "L8", "text": "L8 경계 문서", "access_scope": "dept", "min_auth_level": 8}).assert_status((200, 201))

    api_client.with_api_key(l7_key).chat_completions({"model": "anthropic/claude-3.5-sonnet", "bucket_id": bid7, "messages": [{"role": "user", "content": "L7 경계"}]}).assert_status(200)
    api_client.with_api_key(l8_key).chat_completions({"model": "anthropic/claude-3.5-sonnet", "bucket_id": bid8, "messages": [{"role": "user", "content": "L8 경계"}]}).assert_status(200)

    logs = pg_ephemeral.execute_sql(
        f"SELECT l.max_context_auth_level, p.provider_type, p.is_on_premise FROM gateway.request_logs l JOIN gateway.providers p ON p.id = l.provider_id WHERE l.org_id = '{ctx['org_id']}' AND l.max_context_auth_level = 8 ORDER BY l.created_at DESC LIMIT 1;"
    )
    assert "vllm_local" in logs or "|t" in logs


@pytest.mark.f25
def test_f25_bva_block_unmasked_pii_to_external_provider(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    # Disable on-prem providers and enforce unmasked PII block to external provider
    pg_ephemeral.execute_sql(
        f"UPDATE gateway.providers SET enabled = false WHERE org_id = '{ctx['org_id']}' AND is_on_premise = true;"
    )
    cfg = sample_factory.pipeline_yaml("nomask.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace("capsule_enabled: true", "capsule_enabled: false")
    client.studio_pipeline_update({"pipeline_yaml": cfg})

    resp = client.chat_completions(
        {
            "model": "anthropic/claude-3.5-sonnet",
            "allow_unmasked_pii": True,
            "messages": [{"role": "user", "content": "주민번호 900101-1234567 외부 전송 시도"}],
        }
    )
    assert resp.status in (403, 503) or "900101-1234567" not in resp.text


@pytest.mark.f25
def test_f25_bva_block_l9_context_when_external_provider_max_level_5(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    l9_key = _mint_api_key(api_client, ctx, ctx["l9_id"])
    client = api_client.with_api_key(l9_key)

    b = client.create_bucket({"slug": f"l9blk-{uuid.uuid4().hex[:6]}", "name": "L9Blk", "access_scope": "dept", "min_auth_level": 9}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    client.create_document(bid, {"title": "L9", "text": "9등급 최고 기밀", "access_scope": "dept", "min_auth_level": 9}).assert_status((200, 201))

    pg_ephemeral.execute_sql(
        f"""
        UPDATE gateway.providers SET enabled = false WHERE org_id = '{ctx['org_id']}' AND is_on_premise = true;
        UPDATE gateway.providers SET max_auth_level = 5 WHERE org_id = '{ctx['org_id']}' AND is_on_premise = false;
        """
    )

    resp = client.chat_completions(
        {"model": "anthropic/claude-3.5-sonnet", "bucket_id": bid, "messages": [{"role": "user", "content": "최고 기밀 요약"}]}
    )
    assert resp.status in (403, 503)


@pytest.mark.f25
def test_f25_bva_llm_egress_guard_blocks_safeword_before_provider_call(
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
            "surfaces": ["inbound_query", "llm_egress"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    resp = client.chat_completions(
        {"model": "openrag-auto", "messages": [{"role": "user", "content": "PROJECT-ORION 예산 보고"}]}
    )
    assert resp.status == 403
    assert "guard_blocked" in resp.text.lower()


@pytest.mark.f25
def test_f25_bva_invalid_gateway_routing_rule_min_gt_max_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    providers = client.studio_gateway_request("providers").assert_status(200).json()
    plist = providers.get("providers") or providers
    target_id = str(plist[0]["id"]) if plist else str(uuid.uuid4())

    bad_rule = client.studio_gateway_request(
        "routing-rules",
        method="POST",
        payload={"priority": 10, "min_auth_level": 8, "max_auth_level": 5, "target_provider_id": target_id},
    )
    assert bad_rule.status in (400, 422)


# ============================================================================
# F26 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f26
def test_f26_bva_reject_invalid_golden_axis_and_verdict(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs = client.studio_eval_request(
        "golden-sets", method="POST", payload={"name": f"badax-{uuid.uuid4().hex[:6]}", "axis": "multi"}
    ).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))

    assert (
        client.studio_eval_request(
            f"golden-sets/{gs_id}/items",
            method="POST",
            payload={"axis": "unknown", "query_text": "test", "expected_doc_ids": [str(uuid.uuid4())]},
        ).status
        in (400, 422)
    )
    assert client.app_feedback({"query_text": "test", "verdict": "maybe"}).status in (400, 422)


@pytest.mark.f26
def test_f26_bva_reject_golden_item_with_empty_query_or_missing_expected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs = client.studio_eval_request(
        "golden-sets", method="POST", payload={"name": f"emptyit-{uuid.uuid4().hex[:6]}", "axis": "retrieval"}
    ).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))

    assert (
        client.studio_eval_request(
            f"golden-sets/{gs_id}/items",
            method="POST",
            payload={"axis": "retrieval", "query_text": "", "expected_doc_ids": [], "forbidden_doc_ids": []},
        ).status
        in (400, 422)
    )


@pytest.mark.f26
def test_f26_bva_duplicate_golden_set_name_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    name = f"dup-gs-{uuid.uuid4().hex[:6]}"
    client.studio_eval_request("golden-sets", method="POST", payload={"name": name, "axis": "retrieval"}).assert_status((200, 201))
    dup = client.studio_eval_request("golden-sets", method="POST", payload={"name": name, "axis": "retrieval"})
    assert dup.status in (400, 409)


@pytest.mark.f26
def test_f26_bva_double_promote_same_search_feedback_prevented_or_idempotent(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    fb = client.app_feedback({"query_text": "이중 승격 방지", "verdict": "down"}).assert_status((200, 201)).json()
    fb_id = str(fb.get("id") or fb.get("feedback_id"))
    gs = client.studio_eval_request("golden-sets", method="POST", payload={"name": f"dp-{uuid.uuid4().hex[:6]}", "axis": "retrieval"}).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))

    p1 = client.studio_eval_request(
        f"feedback/{fb_id}/promote", method="POST", payload={"golden_set_id": gs_id, "axis": "retrieval", "expected_doc_ids": [str(uuid.uuid4())]}
    ).assert_status((200, 201))
    p2 = client.studio_eval_request(
        f"feedback/{fb_id}/promote", method="POST", payload={"golden_set_id": gs_id, "axis": "retrieval", "expected_doc_ids": [str(uuid.uuid4())]}
    )
    assert p2.status in (200, 409)
    cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM eval.golden_items WHERE golden_set_id = '{gs_id}';").strip()
    assert int(cnt) == 1
    assert p1.status in (200, 201)


@pytest.mark.f26
def test_f26_bva_rejected_golden_item_excluded_from_bench_run(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs_name = f"rej-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request("golden-sets", method="POST", payload={"name": gs_name, "axis": "retrieval"}).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))
    ok_item = client.studio_eval_request(
        f"golden-sets/{gs_id}/items", method="POST", payload={"axis": "retrieval", "query_text": "승인 항목", "expected_doc_ids": [str(uuid.uuid4())], "review_status": "approved"}
    ).assert_status((200, 201)).json()
    rej_item = client.studio_eval_request(
        f"golden-sets/{gs_id}/items", method="POST", payload={"axis": "retrieval", "query_text": "거절 항목", "expected_doc_ids": [str(uuid.uuid4())], "review_status": "rejected"}
    ).assert_status((200, 201)).json()
    rej_id = str(rej_item.get("id") or rej_item.get("item_id"))
    ok_id = str(ok_item.get("id") or ok_item.get("item_id"))

    openrag_cli.bench_run(golden_set=gs_name, env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]}, check=True)
    rej_cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM eval.bench_item_results WHERE golden_item_id = '{rej_id}';").strip()
    ok_cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM eval.bench_item_results WHERE golden_item_id = '{ok_id}';").strip()
    assert int(rej_cnt) == 0
    assert int(ok_cnt) >= 1


# ============================================================================
# F27 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f27
def test_f27_bva_autorag_hard_blocked_on_pii_leak_count_gt_0(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs_name = f"piileak-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request("golden-sets", method="POST", payload={"name": gs_name, "axis": "pii"}).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))
    client.studio_eval_request(
        f"golden-sets/{gs_id}/items",
        method="POST",
        payload={"axis": "pii", "query_text": "900101-1234567", "expected_pii_tokens": ["900101-1234567"], "expected_doc_ids": [str(uuid.uuid4())]},
    ).assert_status((200, 201))

    # Candidate space intentionally disables capsule to trigger pii_leak_count > 0
    bad_space = tmp_path / "bad_pii_space.yaml"
    bad_space.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "golden_set": gs_name,
                "strategy": "greedy_node",
                "security_gates": {"require_zero_pii_leak": True, "require_zero_arn_violation": True},
                "nodes": {"pii.capsule_enabled": [False], "retrieve.fusion": ["rrf"]},
            }
        ),
        encoding="utf-8",
    )
    best_out = tmp_path / "should_not_export.yaml"
    res = openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=bad_space,
        require_zero_pii_leak=True,
        export_best=best_out,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
    )
    assert res.returncode != 0
    assert not best_out.exists()


@pytest.mark.f27
def test_f27_bva_autorag_hard_blocked_on_arn_violation_count_gt_0(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    b = client.create_bucket({"slug": f"arnv-{uuid.uuid4().hex[:6]}", "name": "ArnV", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    forb = client.create_document(bid, {"title": "Forbidden", "text": "금지 문서 누출 검증 테스트", "access_scope": "private", "min_auth_level": 9}).assert_status((200, 201)).json()
    forb_id = str(forb.get("id") or forb.get("document_id"))

    gs_name = f"arnleak-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request("golden-sets", method="POST", payload={"name": gs_name, "axis": "retrieval"}).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))
    client.studio_eval_request(
        f"golden-sets/{gs_id}/items",
        method="POST",
        payload={"axis": "retrieval", "query_text": "금지 문서 누출 검증", "test_principal_id": ctx["l3_id"], "expected_doc_ids": [], "forbidden_doc_ids": [forb_id]},
    ).assert_status((200, 201))

    bad_space = tmp_path / "bad_arn_space.yaml"
    bad_space.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "golden_set": gs_name,
                "strategy": "greedy_node",
                "security_gates": {"require_zero_pii_leak": True, "require_zero_arn_violation": True},
                "nodes": {"iam.bypass_rls_test_only": [True], "retrieve.fusion": ["rrf"]},
            }
        ),
        encoding="utf-8",
    )
    best_out = tmp_path / "blocked_arn.yaml"
    res = openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=bad_space,
        require_zero_arn_violation=True,
        export_best=best_out,
        env={"DATABASE_URL": pg_ephemeral.dsn(), "OPENRAG_ORG_ID": ctx["org_id"]},
    )
    assert res.returncode != 0
    assert not best_out.exists()


@pytest.mark.f27
def test_f27_bva_pipeline_promote_blocked_for_unsafe_config(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])

    cfg_id = str(uuid.uuid4())
    gs_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO eval.pipeline_configs (id, org_id, name, profile, pipeline_yaml, config_json, is_promoted)
        VALUES ('{cfg_id}', '{ctx['org_id']}', 'unsafe-{cfg_id[:6]}', 'custom', 'version: 1', '{{}}'::jsonb, false);
        INSERT INTO eval.golden_sets (id, org_id, name, axis) VALUES ('{gs_id}', '{ctx['org_id']}', 'gs-{gs_id[:6]}', 'retrieval');
        INSERT INTO eval.bench_runs (id, org_id, golden_set_id, pipeline_config_id, track, pii_leak_count, arn_violation_count)
        VALUES ('{run_id}', '{ctx['org_id']}', '{gs_id}', '{cfg_id}', 'track3_security_ocr', 1, 0);
        """
    )

    resp = api_client.with_api_key(admin_key).studio_eval_request(
        "promote", method="POST", payload={"pipeline_config_id": cfg_id}
    )
    assert resp.status in (403, 409, 422)
    promoted = pg_ephemeral.execute_sql(f"SELECT is_promoted FROM eval.pipeline_configs WHERE id = '{cfg_id}';").strip()
    assert promoted == "f"


@pytest.mark.f27
def test_f27_bva_ab_search_detects_forbidden_doc_violation_in_security_gates(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    b = client.create_bucket({"slug": f"abv-{uuid.uuid4().hex[:6]}", "name": "ABViol", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))
    doc = client.create_document(bid, {"title": "ForbiddenDoc", "text": "보안 게이트 금지 문서 탐지", "access_scope": "dept"}).assert_status((200, 201)).json()
    doc_id = str(doc.get("id") or doc.get("document_id"))

    ab = client.ab_search(
        {
            "bucket_id": bid,
            "k": 5,
            "queries": [{"query": "보안 게이트 금지 문서 탐지", "relevant_ids": [], "forbidden_ids": [doc_id]}],
            "variants": {"a": {"fusion": "rrf"}, "b": {"fusion": "convex", "alpha": 0.5}},
        }
    ).assert_status(200).json()
    gates = ab.get("security_gates") or {}
    assert gates.get("arn_violation_count", 0) >= 1
    assert gates.get("pass") is False


@pytest.mark.f27
def test_f27_bva_search_space_empty_nodes_or_unknown_strategy_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    empty_nodes = tmp_path / "empty_nodes.yaml"
    empty_nodes.write_text(
        yaml.safe_dump({"version": 1, "golden_set": "gov_v1", "strategy": "greedy_node", "nodes": {}}),
        encoding="utf-8",
    )
    bad_strat = tmp_path / "bad_strat.yaml"
    bad_strat.write_text(
        yaml.safe_dump({"version": 1, "golden_set": "gov_v1", "strategy": "random_walk", "nodes": {"retrieve.fusion": ["rrf"]}}),
        encoding="utf-8",
    )

    assert openrag_cli.bench_optimize(golden_set="gov_v1", search_space=empty_nodes, env={"DATABASE_URL": pg_ephemeral.dsn()}).returncode != 0
    assert openrag_cli.bench_optimize(golden_set="gov_v1", search_space=bad_strat, env={"DATABASE_URL": pg_ephemeral.dsn()}).returncode != 0


@pytest.mark.f27
def test_f27_bva_bench_run_nonexistent_golden_set_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    res = openrag_cli.bench_run(
        golden_set=f"nonexistent_set_{uuid.uuid4().hex[:8]}",
        env={"DATABASE_URL": pg_ephemeral.dsn()},
    )
    assert res.returncode != 0
