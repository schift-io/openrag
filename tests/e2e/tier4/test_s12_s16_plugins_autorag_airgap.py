"""Tier 4 Scenarios 12-16: Plugins (kordoc, docufinder, ai-memory-hooks), AutoRAG Gates, Airgap Stack."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import yaml

from conftest import fail_missing_target
from lib import (
    CliBinaryNotFoundError,
    FixtureFactory,
    McpHttpClient,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PgPrincipalContext,
    PluginClient,
    extract_sse_text,
)
from tier4._helpers import (
    extract_id,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_institution_org,
)

pytestmark = [pytest.mark.tier4]


@pytest.mark.f11
@pytest.mark.f23
@pytest.mark.f29
@pytest.mark.f30
def test_s12_kordoc_mcp_bridge_immediate_write_and_index_priority1_form_fill(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        info = plugin_client.inspect_kordoc_bridge()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert info["files"]

    ctx = seed_institution_org(migrated_pg, label="S12")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s12")
    client = api_client.with_api_key(key)
    bslug = f"s12-kor-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "KordocS12", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    try:
        with plugin_client.invoke_kordoc_mcp(api_url=api_client.base_url, api_key=key) as kmcp:
            init = kmcp.handshake()
            assert "serverInfo" in init or "capabilities" in init
            assert len(kmcp.tools_list()) >= 1
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    hwpx_v1 = sample_factory.hwpx_file(
        "form_v1.hwpx",
        title="공문 서식 S12",
        paragraphs=["1차 작성본"],
        tables=[[["항목", "상태"], ["공문S12", "초안V1_S12"]]],
    )
    plugin_client.trigger_kordoc_hook(hwpx_v1, bucket=bslug, api_url=api_client.base_url, api_key=key)
    assert "초안V1_S12" in client.search_bucket(bid, {"query": "초안V1_S12", "top_k": 5}).assert_status(200).text

    prio_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM indexing.outbox WHERE org_id = '{ctx['org_id']}' AND priority = 1;"
        ).strip()
    )
    assert prio_cnt >= 1
    assert "kordoc" in client.studio_plugins_connectors().assert_status(200).text.lower()

    hwpx_v2 = sample_factory.hwpx_file(
        "form_v1.hwpx",
        title="공문 서식 S12",
        paragraphs=["최종 결재본"],
        tables=[[["항목", "상태"], ["공문S12", "확정V2_S12"]]],
    )
    plugin_client.trigger_kordoc_hook(hwpx_v2, bucket=bslug, api_url=api_client.base_url, api_key=key)
    assert "확정V2_S12" in client.search_bucket(bid, {"query": "확정V2_S12", "top_k": 5}).assert_status(200).text


@pytest.mark.f23
@pytest.mark.f28
@pytest.mark.f31
def test_s13_docufinder_delta_folder_sync_and_anything_chat_sse_citations(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    try:
        plugin_client.inspect_docufinder_connector()
        cfg = plugin_client.get_docufinder_anything_config()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert "/v1/chat/completions" in str(cfg.get("endpoint", cfg))

    ctx = seed_institution_org(migrated_pg, label="S13")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s13")
    client = api_client.with_api_key(key)
    bslug = f"s13-df-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "DocufinderS13", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    watch_dir = tmp_path / "df_watch"
    watch_dir.mkdir()
    state_file = tmp_path / "df_state.json"
    (watch_dir / "a.md").write_text("# 정책 A\n\nDOCUFINDER_V1_S13 본문\n", encoding="utf-8")
    (watch_dir / "b.hwpx").write_bytes(
        sample_factory.hwpx_bytes(title="문서 B", paragraphs=["DOCUFINDER_DEL_S13"])
    )

    plugin_client.run_docufinder_sync(
        watch_dir, bucket=bslug, api_url=api_client.base_url, api_key=key, state_file=state_file
    )
    assert "DOCUFINDER_V1_S13" in client.search_bucket(bid, {"query": "DOCUFINDER_V1_S13", "top_k": 5}).assert_status(200).text
    assert "DOCUFINDER_DEL_S13" in client.search_bucket(bid, {"query": "DOCUFINDER_DEL_S13", "top_k": 5}).assert_status(200).text

    # Re-run unchanged -> SHA-256 No-Op skip
    r_noop = plugin_client.run_docufinder_sync(
        watch_dir, bucket=bslug, api_url=api_client.base_url, api_key=key, state_file=state_file
    )
    assert r_noop.returncode == 0

    # Modify a.md and delete b.hwpx
    (watch_dir / "a.md").write_text("# 정책 A\n\nDOCUFINDER_V2_S13 개정 본문\n", encoding="utf-8")
    (watch_dir / "b.hwpx").unlink()
    plugin_client.run_docufinder_sync(
        watch_dir, bucket=bslug, api_url=api_client.base_url, api_key=key, state_file=state_file
    )

    assert "DOCUFINDER_V2_S13" in client.search_bucket(bid, {"query": "DOCUFINDER_V2_S13", "top_k": 5}).assert_status(200).text
    assert "DOCUFINDER_DEL_S13" not in client.search_bucket(bid, {"query": "DOCUFINDER_DEL_S13", "top_k": 5}).assert_status(200).text

    model_name = str(cfg.get("model") or "openrag-anything")
    status, _, events = client.chat_completions_stream(
        {
            "model": model_name,
            "bucket": bslug,
            "include_citations": True,
            "messages": [{"role": "user", "content": "DOCUFINDER_V2_S13 내용 요약"}],
        }
    )
    assert status == 200
    assert any(ev.data.strip() == "[DONE]" for ev in events)
    assert len(extract_sse_text(events)) > 0
    conn_txt = client.studio_plugins_connectors().assert_status(200).text.lower()
    assert "docufinder" in conn_txt or "schift_index" in conn_txt


@pytest.mark.f18
@pytest.mark.f32
@pytest.mark.f33
def test_s14_ai_memory_hooks_secret_preredaction_and_private_episode_recall(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        info = plugin_client.inspect_ai_memory_hooks()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert "hooks/hooks.json" in info

    ctx = seed_institution_org(migrated_pg, label="S14")
    owner_key = mint_api_key(api_client, ctx["org_id"], ctx["owner_id"], tag="s14-own")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="s14-peer")

    secrets = sample_factory.secret_payload()
    payload = {
        "session_id": "s14-work-session",
        "prompt": f"Inspect {secrets['local_path']}",
        "transcript": secrets["transcript"] + " WORK_S14_MEM_TOKEN",
    }

    for event_name in ("UserPromptSubmit", "PostToolUse", "Stop"):
        dry = plugin_client.invoke_memory_hook(
            event_name, payload, api_url=api_client.base_url, api_key=owner_key, dry_run=True
        )
        assert dry.returncode == 0
        for secret_val in secrets["secrets"]:
            assert secret_val not in dry.stdout and secret_val not in dry.stderr

    live = plugin_client.invoke_memory_hook(
        "Stop", payload, api_url=api_client.base_url, api_key=owner_key, dry_run=False
    )
    assert live.returncode == 0

    ep_blob = migrated_pg.execute_sql(
        f"SELECT string_agg(summary_md || ' ' || COALESCE(raw_redacted_json::text, ''), ' ') FROM memory.episodes WHERE org_id = '{ctx['org_id']}' AND redaction_applied = true;"
    )
    assert "WORK_S14_MEM_TOKEN" in ep_blob
    for secret_val in secrets["secrets"]:
        assert secret_val not in ep_blob

    owner_client = api_client.with_api_key(owner_key)
    owner_mcp = McpHttpClient(owner_client)
    owner_mcp.handshake()
    assert "WORK_S14_MEM_TOKEN" in owner_mcp.memory_search("WORK_S14_MEM_TOKEN", scope="private").text
    assert "WORK_S14_MEM_TOKEN" in owner_client.app_memory_list().assert_status(200).text

    peer_mcp = McpHttpClient(api_client.with_api_key(peer_key))
    peer_mcp.handshake()
    assert "WORK_S14_MEM_TOKEN" not in peer_mcp.memory_search("WORK_S14_MEM_TOKEN", scope="private").text
    assert (
        migrated_pg.query_as_principal(
            "SELECT id::text FROM memory.episodes WHERE org_id = %s",
            principal=PgPrincipalContext(
                org_id=ctx["org_id"], principal_id=ctx["peer_id"], auth_level=8, dept_ids=(ctx["dept_a_id"],)
            ),
            params=(ctx["org_id"],),
        )
        == []
    )


@pytest.mark.f03
@pytest.mark.f26
@pytest.mark.f27
@pytest.mark.f34
def test_s15_autorag_greedy_optimizer_security_gate_block_vs_clean_promotion(
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

    ctx = seed_institution_org(migrated_pg, label="S15")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s15")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"s15-eval-{uuid.uuid4().hex[:6]}", "name": "EvalS15", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")
    doc = client.create_document(
        bid, {"title": "평가 문서 S15", "text": "AutoRAG 보안 게이트 검증용 본문 S15", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")

    fb = client.app_feedback(
        {
            "query_text": "AutoRAG 보안 게이트 검증",
            "response_text": "부적절 인용",
            "verdict": "wrong_evidence",
            "expected_doc_ids": [doc_id],
        }
    ).assert_status((200, 201)).json()
    fb_id = extract_id(fb, "id", "feedback_id")

    gs_name = f"s15-gs-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request(
        "golden-sets", method="POST", payload={"name": gs_name, "axis": "multi"}
    ).assert_status((200, 201)).json()
    gs_id = extract_id(gs, "id", "golden_set_id")

    client.studio_eval_request(
        f"feedback/{fb_id}/promote",
        method="POST",
        payload={"golden_set_id": gs_id, "axis": "retrieval", "expected_doc_ids": [doc_id]},
    ).assert_status((200, 201))
    for axis in ("ocr", "pii", "pararel"):
        client.studio_eval_request(
            f"golden-sets/{gs_id}/items",
            method="POST",
            payload={"axis": axis, "query_text": f"S15 {axis} 점검", "expected_doc_ids": [doc_id]},
        ).assert_status((200, 201))

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.bench_run(golden_set=gs_name, env=env, check=True)

    # Unsafe search space must fail security gates and NOT export best_unsafe.yaml
    unsafe_space = tmp_path / "unsafe_space.yaml"
    unsafe_space.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "golden_set": gs_name,
                "strategy": "greedy_node",
                "security_gates": {"require_zero_pii_leak": True, "require_zero_arn_violation": True},
                "nodes": {"pii.capsule_enabled": [False], "retrieve.fusion": ["rrf"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    best_unsafe = tmp_path / "best_unsafe.yaml"
    unsafe_run = openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=unsafe_space,
        require_zero_pii_leak=True,
        require_zero_arn_violation=True,
        export_best=best_unsafe,
        env=env,
    )
    assert unsafe_run.returncode != 0
    assert not best_unsafe.exists()

    # Clean optimization passes and promotes
    clean_space = sample_factory.search_space_yaml("clean_space.yaml", golden_set=gs_name)
    best_clean = tmp_path / "best_clean.yaml"
    openrag_cli.bench_optimize(
        golden_set=gs_name,
        search_space=clean_space,
        require_zero_pii_leak=True,
        require_zero_arn_violation=True,
        export_best=best_clean,
        env=env,
        check=True,
    )
    assert best_clean.is_file()
    openrag_cli.pipeline_promote(best_clean, bucket=str(b.get("slug")), env=env, check=True)

    promoted_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM eval.pipeline_configs WHERE org_id = '{ctx['org_id']}' AND is_promoted = true;"
        ).strip()
    )
    assert promoted_cnt >= 1
    client.ab_search(
        {
            "bucket_id": bid,
            "golden_set_id": gs_id,
            "queries": ["AutoRAG 보안 게이트 검증"],
            "variants": {"a": {"fusion": "rrf"}, "b": {"fusion": "convex", "alpha": 0.3}},
        }
    ).assert_status(200)
    client.studio_eval_request("leaderboard").assert_status(200)


@pytest.mark.f01
@pytest.mark.f02
@pytest.mark.f03
@pytest.mark.f12
@pytest.mark.f25
def test_s16_airgap_stack_config_compose_and_local_ocr_vllm_enforcement(
    openrag_root: Path,
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    lic_path = openrag_root / "LICENSE"
    df_path = openrag_root / "docker" / "postgres" / "Dockerfile"
    airgap_compose = openrag_root / "docker" / "docker-compose.airgap.yml"
    airgap_yaml_path = openrag_root / "configs" / "pipeline.airgap.yaml"
    for p in (lic_path, df_path, airgap_compose, airgap_yaml_path):
        if not p.is_file():
            fail_missing_target(f"Missing required airgap/repo artifact: {p}")

    assert "Apache License" in lic_path.read_text(encoding="utf-8")
    forbidden_a = "-".join(("schift", "db"))
    forbidden_b = "-".join(("schift", "rag", "ce"))
    for p in (df_path, airgap_compose, airgap_yaml_path):
        txt = p.read_text(encoding="utf-8").lower()
        assert forbidden_a not in txt and forbidden_b not in txt

    df_txt = df_path.read_text(encoding="utf-8")
    assert "pg_pinecone" in df_txt and "pg_search" in df_txt
    compose_cfg = yaml.safe_load(airgap_compose.read_text(encoding="utf-8"))
    assert compose_cfg["networks"]["openrag_airgap"]["internal"] is True
    for svc in ("vllm", "local-ocr", "local-embed"):
        assert svc in compose_cfg["services"]

    airgap_raw = airgap_yaml_path.read_text(encoding="utf-8")
    airgap_cfg = yaml.safe_load(airgap_raw)
    assert airgap_cfg["parse"]["ocr_engine"] == "local_vllm"
    assert airgap_cfg["llm"]["local_only_min_auth_level"] == 1

    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S16")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s16")
    client = api_client.with_api_key(key)
    bslug = f"s16-air-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "AirgapS16", "access_scope": "org"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    client.studio_pipeline_update({"pipeline_yaml": airgap_raw}).assert_status((200, 204))
    pdf_path = sample_factory.pdf_file("airgap_l1.pdf", ["Airgap L1 Local Only Verification S16"])
    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.ingest(pdf_path, bucket=bslug, scope="org", auth_level=1, pipeline=airgap_yaml_path, env=env, check=True)

    sens = migrated_pg.execute_sql(
        f"SELECT sensitivity FROM corpus.documents WHERE bucket_id = '{bid}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert sens != "share"

    client.chat_completions(
        {
            "model": "openrag-auto",
            "bucket_id": bid,
            "messages": [{"role": "user", "content": "Airgap L1 Local Only 요약"}],
        }
    ).assert_status(200)

    prov_types = migrated_pg.execute_sql(
        f"""
        SELECT string_agg(p.provider_type, ',')
        FROM gateway.request_logs l
        JOIN gateway.providers p ON p.id = l.provider_id
        WHERE l.org_id = '{ctx['org_id']}';
        """
    )
    assert "vllm_local" in prov_types and "openrouter" not in prov_types
