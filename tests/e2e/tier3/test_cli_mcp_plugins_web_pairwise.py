"""Tier 3 pairwise E2E tests: CLI, 13-Tool MCP, P1-P3 Plugins, Tier 1 /app & Tier 2 /studio (6 tests)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from conftest import fail_missing_target
from lib import (
    APP_ROUTES,
    STUDIO_ROUTES,
    FixtureFactory,
    McpHttpClient,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PluginClient,
    WebClient,
    extract_sse_text,
)
from lib.mcp_client import OPENRAG_MCP_TOOLS
from tier3._helpers import (
    extract_id,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    running_web_url,
    seed_org_with_users,
)

pytestmark = [pytest.mark.tier3]


@pytest.mark.f28
@pytest.mark.f29
@pytest.mark.f09
def test_t3_f28_f29_cli_code_exchange_scoped_token_governs_stdio_and_http_mcp(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F2829")
    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-2829-adm")
    admin_client = api_client.with_api_key(admin_key)

    b1 = admin_client.create_bucket(
        {"slug": f"d1-{uuid.uuid4().hex[:6]}", "name": "Dept1", "access_scope": "dept", "min_auth_level": 1}
    ).assert_status((200, 201)).json()
    b1_id = extract_id(b1, "id", "bucket_id")
    b1_slug = str(b1.get("slug"))

    admin_client.create_document(
        b1_id,
        {
            "title": "L3 Public Dept1",
            "text": "SCOPED_TOKEN_L3_OK_T3",
            "access_scope": "org",
            "min_auth_level": 3,
        },
    ).assert_status((200, 201))
    d_high = admin_client.create_document(
        b1_id,
        {
            "title": "L6 High Dept1",
            "text": "SCOPED_TOKEN_L6_BLOCKED_T3",
            "access_scope": "org",
            "min_auth_level": 6,
        },
    ).assert_status((200, 201)).json()
    d_high_id = extract_id(d_high, "document_id", "id", "doc_id")

    scoped_key = mint_api_key(
        api_client,
        ctx["org_id"],
        ctx["admin_id"],
        tag="t3-2829-scoped",
        max_auth_level=4,
        scoped_arns=[f"arn:openrag:{ctx['slug']}:*"],
    )

    http_mcp = McpHttpClient(api_client.with_api_key(scoped_key))
    http_mcp.handshake()
    assert {t["name"] for t in http_mcp.tools_list()} == set(OPENRAG_MCP_TOOLS)

    with mcp_stdio(
        api_key=scoped_key,
        principal_id=ctx["admin_id"],
        env={"DATABASE_URL": migrated_pg.dsn()},
    ) as stdio_mcp:
        stdio_mcp.handshake()
        assert {t["name"] for t in stdio_mcp.tools_list()} == set(OPENRAG_MCP_TOOLS)

        ok = stdio_mcp.rag_search("SCOPED_TOKEN_L3_OK_T3", bucket=b1_slug, top_k=5)
        assert ok.is_error is False and "SCOPED_TOKEN_L3_OK_T3" in ok.text

        denied = stdio_mcp.rag_get_page(document_id=d_high_id, page_no=1)
        assert denied.is_error is True
        assert "SCOPED_TOKEN_L6_BLOCKED_T3" not in denied.text


@pytest.mark.f30
@pytest.mark.f11
@pytest.mark.f28
def test_t3_f30_f11_f28_kordoc_mcp_bridge_writes_hwpx_table_and_immediately_indexes(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        plugin_client.inspect_kordoc_bridge()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F3011")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3011")
    client = api_client.with_api_key(key)
    bslug = f"kordoc-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "KordocBridge", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    cell_token = f"KORDOC_CELL_{uuid.uuid4().hex[:8]}"
    hwpx_path = sample_factory.hwpx_file(
        "kordoc_t3.hwpx",
        title="공문서 표 자동적재",
        paragraphs=["즉시 색인 훅 테스트"],
        tables=[[["항목", "값"], ["코드", cell_token]]],
    )
    try:
        plugin_client.trigger_kordoc_hook(
            hwpx_path,
            bucket=bslug,
            api_url=api_client.base_url,
            api_key=key,
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    srch = client.search_bucket(bid, {"query": cell_token, "top_k": 5}).assert_status(200)
    assert cell_token in srch.text

    prio_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM indexing.outbox WHERE org_id = '{ctx['org_id']}' AND priority = 1;"
        ).strip()
    )
    src_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM plugins.source_entries WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    assert prio_cnt >= 1 and src_cnt >= 1


@pytest.mark.f31
@pytest.mark.f23
@pytest.mark.f28
def test_t3_f31_f23_f28_docufinder_delta_sync_and_anything_chat_completions(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    try:
        cfg = plugin_client.get_docufinder_anything_config()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F3123")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3123")
    client = api_client.with_api_key(key)
    bslug = f"docufinder-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": bslug, "name": "Docufinder", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    token_v1 = f"DF_V1_{uuid.uuid4().hex[:6]}"
    token_v2 = f"DF_V2_{uuid.uuid4().hex[:6]}"
    (watch_dir / "policy.md").write_text(f"# 사내 안내\n\n버전1 {token_v1}\n", encoding="utf-8")
    (watch_dir / "form.hwpx").write_bytes(sample_factory.hwpx_bytes(title="서식", paragraphs=[token_v1]))
    state_file = tmp_path / "sync_state.json"

    try:
        plugin_client.run_docufinder_sync(
            watch_dir,
            bucket=bslug,
            api_url=api_client.base_url,
            api_key=key,
            state_file=state_file,
        )
        # Modify one file; second file unchanged (SHA-256 skip)
        (watch_dir / "policy.md").write_text(f"# 사내 안내\n\n버전2 {token_v2}\n", encoding="utf-8")
        res2 = plugin_client.run_docufinder_sync(
            watch_dir,
            bucket=bslug,
            api_url=api_client.base_url,
            api_key=key,
            state_file=state_file,
        )
        assert res2.returncode == 0
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    assert token_v2 in client.search_bucket(bid, {"query": token_v2, "top_k": 5}).assert_status(200).text

    model_name = str(cfg.get("model") or "openrag-anything")
    status, _, events = client.chat_completions_stream(
        {
            "model": model_name,
            "bucket_id": bid,
            "include_citations": True,
            "messages": [{"role": "user", "content": f"{token_v2} 내용 요약해줘"}],
        }
    )
    assert status == 200
    assert any(ev.data.strip() == "[DONE]" for ev in events)
    assert len(extract_sse_text(events)) > 0


@pytest.mark.f32
@pytest.mark.f16
@pytest.mark.f18
def test_t3_f32_f16_f18_ai_memory_hooks_preredacts_secrets_and_persists_episode(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        plugin_client.inspect_ai_memory_hooks()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F3216")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3216")
    client = api_client.with_api_key(key)

    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION-HOOK",
            "action": "block",
            "surfaces": ["ingestion_gate"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    secrets = sample_factory.secret_payload()
    try:
        res_clean = plugin_client.invoke_memory_hook(
            "Stop",
            {"transcript": secrets["transcript"] + " WORK_T3_CLEAN", "session_id": "s-t3-1"},
            api_url=api_client.base_url,
            api_key=key,
            dry_run=False,
        )
        assert res_clean.returncode == 0
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    ep_blob = migrated_pg.execute_sql(
        f"SELECT string_agg(summary_md || ' ' || COALESCE(raw_redacted_json::text, ''), ' ') FROM memory.episodes WHERE org_id = '{ctx['org_id']}' AND redaction_applied = true;"
    )
    assert "WORK_T3_CLEAN" in ep_blob
    assert secrets["openai_key"] not in ep_blob
    assert secrets["aws_key"] not in ep_blob
    assert secrets["local_path"] not in ep_blob

    res_blocked = plugin_client.invoke_memory_hook(
        "Stop",
        {"transcript": "PROJECT-ORION-HOOK confidential session", "session_id": "s-t3-2"},
        api_url=api_client.base_url,
        api_key=key,
        dry_run=False,
    )
    assert res_blocked.returncode != 0 or "guard_blocked" in (res_blocked.stdout + res_blocked.stderr).lower()


@pytest.mark.f33
@pytest.mark.f17
@pytest.mark.f28
def test_t3_f33_f17_f28_app_console_upload_chat_bbox_thumbnail_after_redaction(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        tsx_violations = web_client.scan_app_tsx_copy()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert tsx_violations == {}, f"Forbidden jargon in /app TSX: {tsx_violations}"

    ctx = seed_org_with_users(migrated_pg, label="T3F3317")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3317")
    client = api_client.with_api_key(key)

    bbox = (100, 100, 700, 400)
    png_bytes = sample_factory.png_bytes(300, 300, bg_rgb=(255, 255, 255), boxes=[(bbox, (220, 40, 40))])
    up = client.app_upload(
        filename="app_redact.png",
        content=png_bytes,
        content_type="image/png",
        fields={"access_scope": "dept", "sensitivity": "confidential"},
    ).assert_status((200, 201, 202)).json()
    doc_id = extract_id(up, "document_id", "id", "doc_id")

    chat = client.app_chat({"query": "방금 올린 문서 확인", "message": "방금 올린 문서 확인"}).assert_status(200).json()
    assert isinstance(chat, dict)

    page_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.pages WHERE document_id = '{doc_id}' LIMIT 1;"
    ).strip()
    assert page_id

    openrag_cli.redact_run(
        doc_id=doc_id,
        scope="document",
        redact_thumbnails=True,
        hard_purge=True,
        reason="t3 app console redact",
        env={"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )
    thumb = client.app_page_thumbnail(page_id).assert_status(200).body
    assert sample_factory.inspect_bbox_blackout(thumb, bbox, min_black_ratio=0.90).is_blacked_out is True

    with running_web_url(web_client, api_client.base_url) as web_url:
        for route in APP_ROUTES:
            web_client.check_route_http(web_url, route)


@pytest.mark.f34
@pytest.mark.f21
@pytest.mark.f27
def test_t3_f34_f21_f27_studio_ops_switches_engine_runs_guard_and_displays_eval(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
) -> None:
    try:
        web_client.require_web_dir()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F3421")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-3421")
    client = api_client.with_api_key(key)

    client.studio_index_set_engine({"table": "corpus.chunks", "engine": "pgvector"}).assert_status((200, 204))
    eng_data = client.studio_index_engines().assert_status(200).text
    assert "pgvector" in eng_data

    guard_res = client.studio_guard_test({"text": "연락처 010-1234-5678", "surface": "llm_egress"}).assert_status(200).json()
    assert guard_res.get("decision") in ("mask", "block", "warn", "allow")

    sim_res = client.studio_iam_simulate(
        {"principal_id": ctx["admin_id"], "action": "read", "resource_arn": f"arn:openrag:{ctx['slug']}:*"}
    ).assert_status(200).json()
    assert "allowed" in sim_res

    client.studio_eval_request("golden-sets").assert_status(200)
    client.studio_plugins_connectors().assert_status(200)

    with running_web_url(web_client, api_client.base_url) as web_url:
        for route in STUDIO_ROUTES:
            web_client.check_route_http(web_url, route)
