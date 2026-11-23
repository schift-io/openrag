"""Tier 1 E2E tests for F30-F34: P1-P3 Plugins, Tier 1 /app Console, Tier 2 /studio Ops."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
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
    PgPrincipalContext,
    PluginClient,
    WebClient,
    extract_sse_text,
)

pytestmark = [pytest.mark.tier1]


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> None:
    pg.ensure_started()
    out = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;")
    if out.strip() != "t":
        pg.apply_migrations(root / "migrations")


def _seed_org_and_users(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'Web Org {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ('{dept_id}', '{org_id}', 'General', 'general');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{user_id}', '{org_id}', 'u-{user_id[:6]}@example.org', 'User', 5, false, false),
            ('{peer_id}', '{org_id}', 'p-{peer_id[:6]}@example.org', 'Peer', 5, false, false),
            ('{admin_id}', '{org_id}', 'a-{admin_id[:6]}@example.org', 'Admin', 10, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept_id}', '{user_id}', 'member'),
            ('{org_id}', '{dept_id}', '{peer_id}', 'member'),
            ('{org_id}', '{dept_id}', '{admin_id}', 'dept_admin');
        """
    )
    return {"org_id": org_id, "slug": slug, "dept_id": dept_id, "user_id": user_id, "peer_id": peer_id, "admin_id": admin_id}


def _mint_api_key(api_client: OpenRagHttpClient, ctx: dict[str, str], principal_id: str) -> str:
    resp = api_client.cli_code_exchange(
        {"code": issue_cli_code(ctx["org_id"], principal_id), "org_id": ctx["org_id"], "principal_id": principal_id}
    )
    resp.assert_status((200, 201))
    key = resp.json().get("api_key") or resp.json().get("token")
    assert isinstance(key, str) and key
    return key


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _running_web_url(web_client: WebClient, api_url: str) -> Iterator[str]:
    env_url = os.environ.get("OPENRAG_WEB_URL", "").strip()
    if env_url:
        yield env_url.rstrip("/")
        return

    try:
        web_dir = web_client.require_web_dir()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    if not (web_dir / ".next").is_dir():
        web_client.run_build()

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        fail_missing_target("`pnpm` binary not found on PATH for running Next.js web server.")

    port = _free_port()
    env = os.environ.copy()
    env["NEXT_PUBLIC_OPENRAG_API_URL"] = api_url
    env["PORT"] = str(port)
    proc = subprocess.Popen(
        [pnpm, "--dir", str(web_dir), "start", "-p", str(port), "-H", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=web_dir,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 25.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2)
                pytest.fail(f"Next.js server exited early (rc={proc.returncode}): {out} {err}")
            try:
                with urllib.request.urlopen(f"{base_url}/app", timeout=1.5) as resp:
                    if 200 <= resp.status < 500:
                        break
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.25)
        else:
            pytest.fail(f"Next.js server did not become ready at {base_url}/app within 25s")
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


# ============================================================================
# F30 — P1 kordoc-mcp-bridge Plugin (5 tests)
# ============================================================================


@pytest.mark.f30
def test_f30_kordoc_bridge_manifest_and_entrypoint_present(plugin_client: PluginClient) -> None:
    try:
        info = plugin_client.inspect_kordoc_bridge()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert Path(info["path"]).is_dir()
    assert info["files"], "kordoc-mcp-bridge must contain manifest/entrypoint files"
    assert any("readme" in f.lower() for f in info["files"])


@pytest.mark.f30
def test_f30_kordoc_mcp_stdio_handshake_and_tools(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])

    try:
        with plugin_client.invoke_kordoc_mcp(api_url=api_client.base_url, api_key=key) as mcp:
            mcp.handshake()
            tools = mcp.tools_list()
            assert len(tools) >= 1
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))


@pytest.mark.f30
def test_f30_kordoc_hook_immediate_write_and_index_hwpx(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)
    slug = f"kordoc-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": slug, "name": "Kordoc", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    hwpx = sample_factory.hwpx_file(
        "gongmun.hwpx",
        title="공문 여비 안내",
        tables=[[["항목", "금액"], ["출장여비고유토큰K30", "70,000원"]]],
    )
    try:
        res = plugin_client.trigger_kordoc_hook(hwpx, bucket=slug, api_url=api_client.base_url, api_key=key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0, f"kordoc hook failed: {res.stderr}"

    srch = client.search_bucket(bid, {"query": "출장여비고유토큰K30", "top_k": 5}).assert_status(200)
    assert "70,000" in srch.text or "출장여비고유토큰K30" in srch.text


@pytest.mark.f30
def test_f30_kordoc_priority_1_outbox_and_connector_heartbeat(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)
    slug = f"kprio-{uuid.uuid4().hex[:6]}"
    client.create_bucket({"slug": slug, "name": "KPrio", "access_scope": "dept"}).assert_status((200, 201))

    hwpx = sample_factory.hwpx_file("prio.hwpx", title="우선순위1 공문", paragraphs=["우선순위 1 즉시 색인"])
    try:
        res = plugin_client.trigger_kordoc_hook(hwpx, bucket=slug, api_url=api_client.base_url, api_key=admin_key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0

    prio = pg_ephemeral.execute_sql(
        f"SELECT min(priority) FROM indexing.outbox WHERE org_id = '{ctx['org_id']}';"
    ).strip()
    assert prio == "1"

    conn = client.studio_plugins_connectors().assert_status(200).text
    assert "kordoc" in conn.lower()


@pytest.mark.f30
def test_f30_kordoc_form_fill_updates_reindexes_via_sha256_diff(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)
    slug = f"kfill-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": slug, "name": "KFill", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    f1 = sample_factory.hwpx_file("form.hwpx", title="서식", tables=[[["상태", "초기값OLD99"]]])
    try:
        r1 = plugin_client.trigger_kordoc_hook(f1, bucket=slug, api_url=api_client.base_url, api_key=key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert r1.returncode == 0

    f2 = sample_factory.hwpx_file("form.hwpx", title="서식", tables=[[["상태", "갱신값NEW88"]]])
    r2 = plugin_client.trigger_kordoc_hook(f2, bucket=slug, api_url=api_client.base_url, api_key=key)
    assert r2.returncode == 0

    srch = client.search_bucket(bid, {"query": "갱신값NEW88", "top_k": 5}).assert_status(200)
    assert "갱신값NEW88" in srch.text


# ============================================================================
# F31 — P2 docufinder-connector Plugin (5 tests)
# ============================================================================


@pytest.mark.f31
def test_f31_docufinder_manifest_and_anything_config_valid(plugin_client: PluginClient) -> None:
    try:
        info = plugin_client.inspect_docufinder_connector()
        cfg = plugin_client.get_docufinder_anything_config()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert Path(info["path"]).is_dir()
    serialized = json.dumps(cfg)
    assert "/v1/chat/completions" in serialized


@pytest.mark.f31
def test_f31_docufinder_delta_sync_ingests_new_and_skips_unchanged(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)
    slug = f"df-{uuid.uuid4().hex[:6]}"
    client.create_bucket({"slug": slug, "name": "Docufinder", "access_scope": "dept"}).assert_status((200, 201))

    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    (watch_dir / "a.md").write_text("# 문서 A\n도큐파인더 델타 동기화 A", encoding="utf-8")
    (watch_dir / "b.hwpx").write_bytes(sample_factory.hwpx_bytes(title="문서 B", paragraphs=["도큐파인더 HWPX B"]))
    state_file = tmp_path / "df_state.json"

    try:
        r1 = plugin_client.run_docufinder_sync(
            watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key, state_file=state_file
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert r1.returncode == 0
    assert state_file.is_file()

    r2 = plugin_client.run_docufinder_sync(
        watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key, state_file=state_file
    )
    assert r2.returncode == 0
    assert "skip" in (r2.stdout + r2.stderr).lower() or "0" in r2.stdout


@pytest.mark.f31
def test_f31_docufinder_delta_sync_modifies_and_deletes_removed_files(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)
    slug = f"dfmod-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": slug, "name": "DFMod", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    watch_dir = tmp_path / "watch_mod"
    watch_dir.mkdir()
    a_md = watch_dir / "a.md"
    b_hwpx = watch_dir / "b.hwpx"
    a_md.write_text("초기 텍스트 V1", encoding="utf-8")
    b_hwpx.write_bytes(sample_factory.hwpx_bytes(title="삭제예정", paragraphs=["삭제대상고유토큰DEL77"]))
    state_file = tmp_path / "state_mod.json"

    try:
        r1 = plugin_client.run_docufinder_sync(
            watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key, state_file=state_file
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert r1.returncode == 0

    a_md.write_text("수정된 텍스트 UPDATED_DF_88", encoding="utf-8")
    b_hwpx.unlink()
    r2 = plugin_client.run_docufinder_sync(
        watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key, state_file=state_file
    )
    assert r2.returncode == 0

    s_new = client.search_bucket(bid, {"query": "UPDATED_DF_88", "top_k": 5}).assert_status(200)
    assert "UPDATED_DF_88" in s_new.text
    s_del = client.search_bucket(bid, {"query": "삭제대상고유토큰DEL77", "top_k": 5}).assert_status(200)
    assert "삭제대상고유토큰DEL77" not in s_del.text


@pytest.mark.f31
def test_f31_docufinder_anything_chat_completions_sse_roundtrip(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)
    slug = f"dfany-{uuid.uuid4().hex[:6]}"
    client.create_bucket({"slug": slug, "name": "DFAny", "access_scope": "dept"}).assert_status((200, 201))

    watch_dir = tmp_path / "watch_any"
    watch_dir.mkdir()
    (watch_dir / "a.md").write_text("# 재택근무 규정\n주 2회 원격 근무가 허용된다.", encoding="utf-8")

    try:
        cfg = plugin_client.get_docufinder_anything_config()
        r = plugin_client.run_docufinder_sync(watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert r.returncode == 0

    model = cfg.get("model") or "openrag-anything"
    status, _, events = client.chat_completions_stream(
        {"model": model, "bucket": slug, "messages": [{"role": "user", "content": "원격 근무는 주 몇 회 허용되나요?"}]}
    )
    assert status == 200
    assert len(extract_sse_text(events)) > 0


@pytest.mark.f31
def test_f31_docufinder_connector_registers_health_in_plugins_connectors(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)
    slug = f"dfh-{uuid.uuid4().hex[:6]}"
    client.create_bucket({"slug": slug, "name": "DFHealth", "access_scope": "dept"}).assert_status((200, 201))
    watch_dir = tmp_path / "watch_h"
    watch_dir.mkdir()
    (watch_dir / "h.md").write_text("health check", encoding="utf-8")

    try:
        r = plugin_client.run_docufinder_sync(watch_dir, bucket=slug, api_url=api_client.base_url, api_key=admin_key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert r.returncode == 0

    conns = client.studio_plugins_connectors().assert_status(200).text.lower()
    assert "docufinder" in conns or "schift_index" in conns
    assert "healthy" in conns


# ============================================================================
# F32 — P3 ai-memory-hooks Plugin (5 tests)
# ============================================================================


@pytest.mark.f32
def test_f32_ai_memory_hooks_manifest_and_hooks_json_present(plugin_client: PluginClient) -> None:
    try:
        info = plugin_client.inspect_ai_memory_hooks()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert Path(info["path"]).is_dir()
    assert info.get("hooks/hooks.json") is not None
    assert "_parse_error" not in info["hooks/hooks.json"]


@pytest.mark.f32
def test_f32_ai_memory_hook_client_side_pre_redacts_secrets_and_paths(
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    payload = sample_factory.secret_payload()
    try:
        res = plugin_client.invoke_memory_hook(
            "Stop", payload, api_url="http://127.0.0.1:9999", api_key="dry-run-key", dry_run=True
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0, f"ai-memory-hooks dry_run failed: {res.stderr}"
    for secret in (payload["openai_key"], payload["aws_key"], payload["local_path"]):
        assert secret not in res.stdout
        assert secret not in res.stderr


@pytest.mark.f32
def test_f32_ai_memory_hook_live_ingests_episode_and_entry(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    payload = sample_factory.secret_payload()
    payload["summary"] = f"작업 완료 요약: pgvector 인덱스 튜닝 ({payload['openai_key']})"

    try:
        res = plugin_client.invoke_memory_hook(
            "Stop", payload, api_url=api_client.base_url, api_key=key, dry_run=False
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0

    ep = pg_ephemeral.execute_sql(
        f"SELECT source_harness, redaction_applied, summary_md FROM memory.episodes WHERE org_id = '{ctx['org_id']}' ORDER BY created_at DESC LIMIT 1;"
    )
    assert "claude_code" in ep or "codex" in ep
    assert payload["openai_key"] not in ep
    assert payload["aws_key"] not in ep


@pytest.mark.f32
def test_f32_ai_memory_hook_supports_user_prompt_submit_and_post_tool_use(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])

    for event_name, body in (
        ("UserPromptSubmit", {"session_id": "s1", "prompt": "출장비 규정 확인"}),
        ("PostToolUse", {"session_id": "s1", "tool_name": "Edit", "tool_output": "updated config"}),
    ):
        try:
            res = plugin_client.invoke_memory_hook(
                event_name, body, api_url=api_client.base_url, api_key=key, dry_run=True
            )
        except FileNotFoundError as exc:
            fail_missing_target(str(exc))
        assert res.returncode == 0


@pytest.mark.f32
def test_f32_ai_memory_hook_episode_private_isolation_and_recall(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    owner_key = _mint_api_key(api_client, ctx, ctx["user_id"])
    peer_key = _mint_api_key(api_client, ctx, ctx["peer_id"])
    token = f"HOOK_PRIV_{uuid.uuid4().hex[:8]}"

    try:
        res = plugin_client.invoke_memory_hook(
            "Stop",
            {"session_id": f"sess-{token}", "transcript": f"개인 작업 기록 {token}", "harness": "claude_code"},
            api_url=api_client.base_url,
            api_key=owner_key,
            dry_run=False,
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0

    owner_mcp = McpHttpClient(api_client.with_api_key(owner_key))
    owner_mcp.handshake()
    assert token in owner_mcp.memory_search(token).text

    peer_mcp = McpHttpClient(api_client.with_api_key(peer_key))
    peer_mcp.handshake()
    assert token not in peer_mcp.memory_search(token).text


# ============================================================================
# F33 — Tier 1 User Console (/app) (6 tests)
# ============================================================================


@pytest.mark.f33
def test_f33_web_nextjs_typecheck_lint_and_build(web_client: WebClient) -> None:
    try:
        web_client.require_web_dir()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert web_client.run_typecheck().returncode == 0
    assert web_client.run_lint().returncode == 0
    assert web_client.run_build().returncode == 0


@pytest.mark.f33
def test_f33_app_tsx_static_zero_jargon_scan(web_client: WebClient) -> None:
    try:
        violations = web_client.scan_app_tsx_copy()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert violations == {}, f"Tier 1 /app TSX leaked forbidden jargon: {violations}"


@pytest.mark.f33
def test_f33_app_all_6_routes_http_200_and_rendered_html_zero_jargon(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    with _running_web_url(web_client, api_client.base_url) as web_url:
        for route in APP_ROUTES:
            web_client.check_route_http(web_url, route)


@pytest.mark.f33
def test_f33_app_chat_thumbnail_bbox_and_feedback_api(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)

    hwpx = sample_factory.hwpx_file("app_up.hwpx", title="사용자 콘솔 업로드", paragraphs=["사용자 콘솔 채팅 출처 썸네일 테스트"])
    up = client.app_upload(
        filename=hwpx.name,
        content=hwpx.read_bytes(),
        fields={"scope": "dept", "sensitivity": "normal"},
    )
    up.assert_status((200, 201, 202))

    chat = client.app_chat({"message": "사용자 콘솔 채팅 출처 썸네일"}).assert_status(200).json()
    sources = chat.get("sources") or chat.get("citations") or []
    assert len(sources) >= 1
    page_id = str(sources[0].get("page_id") or "")
    if page_id:
        thumb = client.app_page_thumbnail(page_id).assert_status(200)
        assert len(thumb.body) > 0

    fb = client.app_feedback({"query_text": "사용자 콘솔 채팅", "verdict": "up"})
    fb.assert_status((200, 201))


@pytest.mark.f33
def test_f33_app_wiki_and_memory_patch_retract_promote_apis(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    client = api_client.with_api_key(key)

    mcp = McpHttpClient(client)
    mcp.handshake()
    m1 = mcp.memory_reflect("App 콘솔 메모리 테스트 1", scope="private").json()
    m2 = mcp.memory_reflect("App 콘솔 메모리 철회용", scope="private").json()
    id1 = str(m1.get("memory_id") or m1.get("id"))
    id2 = str(m2.get("memory_id") or m2.get("id"))

    assert id1 in client.app_memory_list().assert_status(200).text
    client.app_memory_patch({"memory_id": id1, "op": "amend", "text": "App 콘솔 수정됨"}).assert_status(200)
    client.app_memory_promote({"memory_id": id1, "to": "dept"}).assert_status((200, 201))
    client.app_memory_retract({"memory_id": id2, "reason": "사용자 철회"}).assert_status(200)
    client.app_wiki().assert_status(200)


@pytest.mark.f33
def test_f33_app_connect_exposes_all_4_mcp_client_snippets_and_api_keys(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["user_id"])
    api_resp = api_client.with_api_key(key).app_connect().assert_status(200)
    assert all(web_client.verify_connect_mcp_snippets(api_resp.text).values())

    with _running_web_url(web_client, api_client.base_url) as web_url:
        page_resp = web_client.check_route_http(web_url, "/app/connect")
        assert all(web_client.verify_connect_mcp_snippets(page_resp.text).values())


# ============================================================================
# F34 — Tier 2 Admin/Ops Studio (/studio — 6 Ops Consoles) (8 tests)
# ============================================================================


@pytest.mark.f34
def test_f34_studio_all_6_routes_render_http_200(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    web_client: WebClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    with _running_web_url(web_client, api_client.base_url) as web_url:
        for route in STUDIO_ROUTES:
            web_client.check_route_http(web_url, route)


@pytest.mark.f34
def test_f34_studio_pipeline_yaml_and_score_decomposition_console(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_pipeline_get().assert_status(200)
    yaml_body = sample_factory.pipeline_yaml("studio.yaml").read_text(encoding="utf-8")
    client.studio_pipeline_update({"pipeline_yaml": yaml_body}).assert_status(200)
    test_res = client.studio_pipeline_test({"query": "점수 분해 테스트"}).assert_status(200)
    lower = test_res.text.lower()
    assert "vector" in lower or "bm25" in lower or "score" in lower


@pytest.mark.f34
def test_f34_studio_index_engine_switcher_outbox_and_quarantine_console(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_index_engines().assert_status(200)
    client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": "pgvector"}).assert_status(200)
    client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": "native"}).assert_status(200)
    client.studio_index_outbox().assert_status(200)

    qid = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"INSERT INTO indexing.quarantine (id, org_id, target_table, target_id, op, error_class, error_detail) VALUES ('{qid}', '{ctx['org_id']}', 'corpus.chunks', '{uuid.uuid4()}', 'upsert_embed', 'transient', 'err');"
    )
    client.studio_quarantine_retry(qid).assert_status((200, 202))


@pytest.mark.f34
def test_f34_studio_iam_groups_levels_and_3axis_simulator(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    grp = client.studio_iam_request(
        "groups",
        method="POST",
        payload={"name": f"tf-sim-{uuid.uuid4().hex[:6]}", "group_type": "project_tf", "min_auth_level": 4, "allowed_actions": ["read", "search"]},
    ).assert_status((200, 201)).json()
    grp_id = str(grp.get("id") or grp.get("group_id"))
    client.studio_iam_request(
        f"groups/{grp_id}/members", method="POST", payload={"principal_id": ctx["user_id"]}
    ).assert_status((200, 201))

    sim = client.studio_iam_simulate(
        {
            "principal_id": ctx["user_id"],
            "action": "read",
            "access_scope": "group",
            "allowed_group_ids": [grp_id],
            "min_auth_level": 4,
            "arn": f"arn:openrag:{ctx['slug']}:general:group:doc:L4/{uuid.uuid4()}",
        }
    ).assert_status(200).json()
    assert sim.get("allowed") is True
    assert "reason" in sim


@pytest.mark.f34
def test_f34_studio_pii_guard_sandbox_cascade_redaction_and_llm_router(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_pii_request("profiles").assert_status(200)
    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["llm_egress", "mcp_egress"], "enabled": True},
    ).assert_status((200, 201))

    g_block = client.studio_guard_test({"text": "PROJECT-ORION 기밀", "surface": "llm_egress"}).assert_status(200).json()
    assert g_block.get("decision") == "block"

    g_mask = client.studio_guard_test({"text": "전화 010-1234-5678", "surface": "llm_egress"}).assert_status(200).json()
    assert g_mask.get("decision") == "mask"
    assert "010-1234-5678" not in g_mask.get("sanitized_text", "")

    client.studio_gateway_request("providers").assert_status(200)
    client.studio_gateway_request("logs").assert_status(200)


@pytest.mark.f34
def test_f34_studio_eval_golden_autorag_bench_and_promote_console(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    gs_name = f"st-eval-{uuid.uuid4().hex[:6]}"
    gs = client.studio_eval_request("golden-sets", method="POST", payload={"name": gs_name, "axis": "multi"}).assert_status((200, 201)).json()
    gs_id = str(gs.get("id") or gs.get("golden_set_id"))
    client.studio_eval_request(
        f"golden-sets/{gs_id}/items",
        method="POST",
        payload={"axis": "retrieval", "query_text": "평가 콘솔 검증", "expected_doc_ids": [str(uuid.uuid4())]},
    ).assert_status((200, 201))

    opt = client.studio_eval_request(
        "optimize",
        method="POST",
        payload={"golden_set_id": gs_id, "strategy": "greedy_node", "search_space_yaml": sample_factory.search_space_yaml(golden_set=gs_name).read_text(encoding="utf-8")},
    ).assert_status((200, 201))
    assert "best" in opt.text.lower()


@pytest.mark.f34
def test_f34_studio_knowledge_wiki_conflicts_memory_backend_and_plugins_ops(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    client.studio_wiki_conflicts().assert_status(200)
    client.studio_plugins_connectors().assert_status(200)

    kumiho_yaml = sample_factory.pipeline_yaml("k.yaml", memory_backend="kumiho", memory_mirror_to="kumiho").read_text(encoding="utf-8")
    client.studio_pipeline_update({"pipeline_yaml": kumiho_yaml}).assert_status(200)
    got = client.studio_pipeline_get().assert_status(200)
    assert "kumiho" in got.text.lower()


@pytest.mark.f34
def test_f34_studio_knowledge_resolve_wiki_conflict_and_restore_default_memory(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    p1, p2, edge_id = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO wiki.pages (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256)
        VALUES
            ('{p1}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:general:dept:wiki:L3/{p1}', '{ctx['org_id']}:dept', 'c1-{p1[:6]}', 'C1', 'Old', 'Old', '{hashlib.sha256(p1.encode()).hexdigest()}'),
            ('{p2}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['admin_id']}', 'dept', 3, 'arn:openrag:{ctx['slug']}:general:dept:wiki:L3/{p2}', '{ctx['org_id']}:dept', 'c2-{p2[:6]}', 'C2', 'New', 'New', '{hashlib.sha256(p2.encode()).hexdigest()}');
        INSERT INTO wiki.edges (id, org_id, src_page_id, dst_page_id, relation_type)
        VALUES ('{edge_id}', '{ctx['org_id']}', '{p2}', '{p1}', 'contradicts');
        """
    )

    conflicts = client.studio_wiki_conflicts().assert_status(200)
    assert edge_id in conflicts.text or "contradicts" in conflicts.text

    pg_yaml = sample_factory.pipeline_yaml("default_mem.yaml", memory_backend="openrag_pg").read_text(encoding="utf-8")
    client.studio_pipeline_update({"pipeline_yaml": pg_yaml}).assert_status(200)
    assert "openrag_pg" in client.studio_pipeline_get().assert_status(200).text
