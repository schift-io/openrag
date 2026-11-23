"""Tier 3 pairwise E2E tests: Hybrid Retriever, Policy LLM Router, Golden Set, Bench & AutoRAG (6 tests)."""

from __future__ import annotations

import hashlib
import uuid

import pytest

from conftest import fail_missing_target
from lib import (
    CliBinaryNotFoundError,
    FixtureFactory,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
)
from tier3._helpers import (
    extract_id,
    insert_returning_one,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_org_with_users,
)

pytestmark = [pytest.mark.tier3]


@pytest.mark.f24
@pytest.mark.f20
def test_t3_f24_f20_hybrid_retriever_fuses_chunks_thumb_wiki_memory_with_latest_revision(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F2420")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2420")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"hyb-{uuid.uuid4().hex[:6]}", "name": "HybridRev", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    w1 = str(uuid.uuid4())
    w2 = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, version)
        VALUES
            ('{w1}', '{ctx['org_id']}', '{bid}', '{ctx['dept1_id']}', '{ctx['admin_id']}', 'dept', 1, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L1/{w1}', '{ctx['org_id']}:dept', 'trv-2025-t3', '2025 출장 여비', '출장 일비 50,000원 기준', '출장 일비 50,000원 기준', '{hashlib.sha256(w1.encode()).hexdigest()}', 1),
            ('{w2}', '{ctx['org_id']}', '{bid}', '{ctx['dept1_id']}', '{ctx['admin_id']}', 'dept', 1, 'arn:openrag:{ctx['slug']}:policy:dept:wiki:L1/{w2}', '{ctx['org_id']}:dept', 'trv-2026-t3', '2026 출장 여비', '출장 일비 70,000원 개정 기준', '출장 일비 70,000원 개정 기준', '{hashlib.sha256(w2.encode()).hexdigest()}', 2);
        INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type)
        VALUES ('{ctx['org_id']}', '{w2}', '{w1}', 'supersedes');
        """
    )
    client.create_document(
        bid, {"title": "여비 공지", "text": "개정 출장 일비 70,000원 적용 안내", "access_scope": "dept"}
    ).assert_status((200, 201))

    for fusion, extra in (("rrf", {}), ("convex", {"alpha": 0.3})):
        res = client.search_bucket(
            bid,
            {"query": "출장 일비", "top_k": 5, "spaces": ["chunks", "thumb", "wiki", "memory"], "fusion": fusion, **extra},
        ).assert_status(200).text
        assert "70,000원" in res
        if "50,000원" in res:
            assert res.index("70,000원") < res.index("50,000원")


@pytest.mark.f24
@pytest.mark.f25
@pytest.mark.f09
def test_t3_f24_f25_l8_confidential_retrieval_forces_local_vllm_and_blocks_openrouter(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F2425")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2425")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"l8llm-{uuid.uuid4().hex[:6]}", "name": "L8Llm", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    client.create_document(
        bid,
        {
            "title": "L8 Secret Brief",
            "text": "L8_LOCAL_VLLM_ONLY_TOKEN 비밀 예산 보고",
            "access_scope": "dept",
            "min_auth_level": 8,
        },
    ).assert_status((200, 201))

    resp = client.chat_completions(
        {
            "model": "openrag-auto",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "L8_LOCAL_VLLM_ONLY_TOKEN 요약"}],
        }
    ).assert_status(200)
    assert resp.json()["choices"][0]["message"]["content"]

    prov_row = migrated_pg.execute_sql(
        f"""
        SELECT p.provider_type || '|' || p.is_on_premise::text || '|' || COALESCE(l.max_context_auth_level::text, '')
        FROM gateway.request_logs l
        JOIN gateway.providers p ON p.id = l.provider_id
        WHERE l.org_id = '{ctx['org_id']}'
        ORDER BY l.created_at DESC
        LIMIT 1;
        """
    ).strip()
    assert prov_row.startswith("vllm_local|true|"), prov_row


@pytest.mark.f25
@pytest.mark.f15
@pytest.mark.f16
def test_t3_f25_f15_f16_llm_egress_masks_capsule_pii_and_blocks_safeword_before_gateway(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F2516")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2516")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"egress-{uuid.uuid4().hex[:6]}", "name": "LlmEgress", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION-T3",
            "action": "block",
            "surfaces": ["inbound_query", "llm_egress"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    pii_doc = sample_factory.korean_pii_doc(safe_words=())
    client.create_document(
        bid, {"title": "EgressPii", "text": pii_doc["text"], "access_scope": "org", "min_auth_level": 3}
    ).assert_status((200, 201))

    ok_resp = client.chat_completions(
        {
            "model": "openrouter/default",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "담당자 연락처 안내 요약"}],
        }
    ).assert_status(200)
    assert "900101-1234567" not in ok_resp.text and "010-1234-5678" not in ok_resp.text

    pii_masked_flag = migrated_pg.execute_sql(
        f"SELECT pii_masked::text FROM gateway.request_logs WHERE org_id = '{ctx['org_id']}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert pii_masked_flag == "true", pii_masked_flag

    logs_before = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM gateway.request_logs WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    blocked = client.chat_completions(
        {
            "model": "openrouter/default",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "PROJECT-ORION-T3 설명해줘"}],
        }
    )
    assert blocked.status == 403 and "guard_blocked" in blocked.text.lower()
    logs_after = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM gateway.request_logs WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    assert logs_after == logs_before


@pytest.mark.f26
@pytest.mark.f24
def test_t3_f26_f24_search_feedback_promotes_to_4axis_golden_set_and_improves_ab_search(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F2624")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2624")
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"fb-{uuid.uuid4().hex[:6]}", "name": "FeedbackAB", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")
    doc = client.create_document(
        bid, {"title": "복리후생 기준", "text": "연차 휴가 및 경조사 지원금 기준 안내", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")

    fb = client.app_feedback(
        {
            "query_text": "경조사 지원금 기준",
            "response_text": "잘못된 문서가 인용됨",
            "verdict": "wrong_evidence",
            "comment": "경조사 기준 문서가 1순위여야 함",
            "expected_doc_ids": [doc_id],
        }
    ).assert_status((200, 201)).json()
    fb_id = extract_id(fb, "id", "feedback_id")

    gs = client.studio_eval_request(
        "golden-sets",
        method="POST",
        payload={"name": f"gs-t3-{uuid.uuid4().hex[:6]}", "axis": "multi"},
    ).assert_status((200, 201)).json()
    gs_id = extract_id(gs, "id", "golden_set_id")

    client.studio_eval_request(
        f"feedback/{fb_id}/promote",
        method="POST",
        payload={"golden_set_id": gs_id, "axis": "retrieval", "expected_doc_ids": [doc_id]},
    ).assert_status((200, 201))

    ab = client.ab_search(
        {
            "bucket_id": bid,
            "golden_set_id": gs_id,
            "queries": ["경조사 지원금 기준"],
            "variants": {
                "a": {"fusion": "rrf", "rerank": False},
                "b": {"fusion": "convex", "alpha": 0.3, "rerank": True},
            },
        }
    ).assert_status(200).json()
    assert "a" in ab and "b" in ab


@pytest.mark.f26
@pytest.mark.f27
def test_t3_f26_f27_4axis_golden_set_drives_3track_bench_and_autorag_optimizer(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F2627")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2627")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"gs4-{uuid.uuid4().hex[:6]}", "name": "Golden4Axis", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")
    doc = client.create_document(
        bid, {"title": "표준 지침", "text": "표준 지침 본문 T3_BENCH", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")

    gs_name = f"gs4-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request(
        "golden-sets",
        method="POST",
        payload={"name": gs_name, "axis": "multi"},
    ).assert_status((200, 201)).json()
    gs_id = extract_id(gs, "id", "golden_set_id")

    for axis in ("retrieval", "ocr", "pii", "pararel"):
        client.studio_eval_request(
            f"golden-sets/{gs_id}/items",
            method="POST",
            payload={"axis": axis, "query_text": f"질의 {axis}", "expected_doc_ids": [doc_id]},
        ).assert_status((200, 201))

    ss_path = sample_factory.search_space_yaml("t3_space.yaml", golden_set=gs_name)
    out_yaml = sample_factory.base_dir / "best_t3.yaml"
    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.bench_run(golden_set=gs_name, env=env, check=True)
    openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=ss_path,
        require_zero_pii_leak=True,
        require_zero_arn_violation=True,
        export_best=out_yaml,
        env=env,
        check=True,
    )
    assert out_yaml.is_file() and len(out_yaml.read_text(encoding="utf-8").strip()) > 0
    gate_pass = migrated_pg.execute_sql(
        f"SELECT security_gate_pass::text FROM eval.bench_runs WHERE org_id = '{ctx['org_id']}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert gate_pass == "true", gate_pass


@pytest.mark.f27
@pytest.mark.f10
@pytest.mark.f15
def test_t3_f27_f10_autorag_security_gate_blocks_promotion_on_arn_or_pii_violation(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F2710")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2710")
    client = api_client.with_api_key(key)

    gs_name = f"unsafe-gs-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request(
        "golden-sets",
        method="POST",
        payload={"name": gs_name, "axis": "pii"},
    ).assert_status((200, 201)).json()
    gs_id = extract_id(gs, "id", "golden_set_id")

    pipe_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO eval.pipeline_configs (org_id, name, profile, pipeline_yaml, config_json)
        VALUES ('{ctx['org_id']}', 'unsafe-cfg-{uuid.uuid4().hex[:6]}', 'custom', 'version: 1', '{{"pii":{{"capsule_enabled":false}}}}'::jsonb)
        RETURNING id::text;
        """,
    )
    bench_id = insert_returning_one(
        migrated_pg,
        f"""
        INSERT INTO eval.bench_runs (org_id, golden_set_id, pipeline_config_id, track, metrics, pii_leak_count, arn_violation_count)
        VALUES ('{ctx['org_id']}', '{gs_id}', '{pipe_id}', 'track3_security_ocr', '{{"ndcg_at_10": 0.9}}'::jsonb, 1, 1)
        RETURNING id::text;
        """,
    )

    gate_pass = migrated_pg.execute_sql(
        f"SELECT security_gate_pass::text FROM eval.bench_runs WHERE id = '{bench_id}';"
    ).strip()
    assert gate_pass == "false", gate_pass

    prom_resp = client.studio_eval_request(
        "promote",
        method="POST",
        payload={"bench_run_id": bench_id, "pipeline_config_id": pipe_id},
    )
    assert prom_resp.status in (400, 403, 409, 422)

    unsafe_yaml = sample_factory.pipeline_yaml("unsafe_promote.yaml")
    unsafe_yaml.write_text(
        unsafe_yaml.read_text(encoding="utf-8").replace("capsule_enabled: true", "capsule_enabled: false"),
        encoding="utf-8",
    )
    cli_prom = openrag_cli.pipeline_promote(
        unsafe_yaml,
        bucket="default",
        env={"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]},
    )
    assert cli_prom.returncode != 0
