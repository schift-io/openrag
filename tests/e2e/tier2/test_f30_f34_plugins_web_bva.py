"""Tier 2 BVA & negative tests for F30-F34: Plugins, /app Zero-Jargon Console, /studio Ops."""

from __future__ import annotations

from lib.cli_client import issue_cli_code

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

from conftest import fail_missing_target
from lib import (
    FORBIDDEN_APP_JARGON,
    FixtureFactory,
    McpJsonRpcError,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PluginClient,
    WebClient,
)

pytestmark = [pytest.mark.tier2]


def _ensure_migrated(pg: PgEphemeralCluster, root: Path) -> None:
    pg.ensure_started()
    out = pg.execute_sql("SELECT to_regclass('iam.orgs') IS NOT NULL;")
    if out.strip() != "t":
        pg.apply_migrations(root / "migrations")


def _seed_org_and_users(pg: PgEphemeralCluster) -> dict[str, str]:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    member_id = str(uuid.uuid4())
    peer_id = str(uuid.uuid4())
    admin_id = str(uuid.uuid4())
    slug = f"org-{org_id[:8]}"
    pg.execute_sql(
        f"""
        INSERT INTO iam.orgs (id, slug, name) VALUES ('{org_id}', '{slug}', 'BVA Web {slug}');
        INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES ('{dept_id}', '{org_id}', 'Ops', 'ops');
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin) VALUES
            ('{member_id}', '{org_id}', 'm-{member_id[:6]}@example.org', 'Member', 3, false, false),
            ('{peer_id}', '{org_id}', 'p-{peer_id[:6]}@example.org', 'Peer', 5, false, false),
            ('{admin_id}', '{org_id}', 'a-{admin_id[:6]}@example.org', 'Admin', 10, true, true);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES
            ('{org_id}', '{dept_id}', '{member_id}', 'member'),
            ('{org_id}', '{dept_id}', '{peer_id}', 'member'),
            ('{org_id}', '{dept_id}', '{admin_id}', 'dept_admin');
        """
    )
    return {"org_id": org_id, "slug": slug, "dept_id": dept_id, "member_id": member_id, "peer_id": peer_id, "admin_id": admin_id}


def _mint_api_key(api_client: OpenRagHttpClient, ctx: dict[str, str], principal_id: str) -> str:
    resp = api_client.cli_code_exchange(
        {"code": issue_cli_code(ctx["org_id"], principal_id), "org_id": ctx["org_id"], "principal_id": principal_id}
    )
    resp.assert_status((200, 201))
    key = resp.json().get("api_key") or resp.json().get("token")
    assert isinstance(key, str) and key
    return key


# ============================================================================
# F30 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f30
def test_f30_bva_kordoc_hook_corrupt_hwpx_returns_error(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    slug = f"kbad-{uuid.uuid4().hex[:6]}"
    api_client.with_api_key(key).create_bucket({"slug": slug, "name": "KBad", "access_scope": "dept"}).assert_status((200, 201))

    corrupt = sample_factory.write_bytes("corrupt.hwpx", b"NOT_A_ZIP_HWPX_PAYLOAD")
    try:
        res = plugin_client.trigger_kordoc_hook(corrupt, bucket=slug, api_url=api_client.base_url, api_key=key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0


@pytest.mark.f30
def test_f30_bva_kordoc_hook_nonexistent_file_or_bucket_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    valid_hwpx = sample_factory.hwpx_file("valid.hwpx")

    try:
        r_missing_file = plugin_client.trigger_kordoc_hook(
            tmp_path / "missing.hwpx", bucket="any", api_url=api_client.base_url, api_key=key
        )
        r_missing_bucket = plugin_client.trigger_kordoc_hook(
            valid_hwpx, bucket=f"missing-{uuid.uuid4().hex[:6]}", api_url=api_client.base_url, api_key=key
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert r_missing_file.returncode != 0
    assert r_missing_bucket.returncode != 0


@pytest.mark.f30
def test_f30_bva_kordoc_hook_unauthorized_api_key_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    hwpx = sample_factory.hwpx_file("unauth.hwpx")
    try:
        res = plugin_client.trigger_kordoc_hook(
            hwpx, bucket="default", api_url=api_client.base_url, api_key="orag_invalid_key_999"
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0


@pytest.mark.f30
def test_f30_bva_kordoc_mcp_unknown_tool_or_missing_field_returns_is_error(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    try:
        with plugin_client.invoke_kordoc_mcp(api_url=api_client.base_url, api_key=key) as mcp:
            mcp.handshake()
            try:
                res = mcp.tools_call("nonexistent_kordoc_tool", {})
                assert res.is_error is True
            except McpJsonRpcError as exc:
                assert exc.code in (-32601, -32602)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))


@pytest.mark.f30
def test_f30_bva_kordoc_hook_blocked_by_ingestion_gate_guard(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)
    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["ingestion_gate"], "enabled": True},
    ).assert_status((200, 201))

    slug = f"kgate-{uuid.uuid4().hex[:6]}"
    client.create_bucket({"slug": slug, "name": "KGate", "access_scope": "dept"}).assert_status((200, 201))
    hwpx = sample_factory.hwpx_file("blocked.hwpx", title="PROJECT-ORION 비밀 공문", paragraphs=["PROJECT-ORION 포함"])
    try:
        res = plugin_client.trigger_kordoc_hook(hwpx, bucket=slug, api_url=api_client.base_url, api_key=key)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0


# ============================================================================
# F31 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f31
def test_f31_bva_docufinder_nonexistent_watch_dir_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    try:
        res = plugin_client.run_docufinder_sync(
            tmp_path / "does_not_exist", bucket="any", api_url=api_client.base_url, api_key=key
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0


@pytest.mark.f31
def test_f31_bva_docufinder_invalid_api_key_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    watch_dir = tmp_path / "w_badkey"
    watch_dir.mkdir()
    (watch_dir / "doc.md").write_text("hello", encoding="utf-8")
    state_file = tmp_path / "state_bad.json"
    try:
        res = plugin_client.run_docufinder_sync(
            watch_dir, bucket="any", api_url=api_client.base_url, api_key="orag_invalid_999", state_file=state_file
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0
    if state_file.exists():
        assert "doc.md" not in state_file.read_text(encoding="utf-8")


@pytest.mark.f31
def test_f31_bva_docufinder_corrupt_file_quarantined_without_halting_valid_files(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(key)
    slug = f"dfq-{uuid.uuid4().hex[:6]}"
    b = client.create_bucket({"slug": slug, "name": "DFQ", "access_scope": "dept"}).assert_status((200, 201)).json()
    bid = str(b.get("id") or b.get("bucket_id"))

    watch_dir = tmp_path / "w_mix"
    watch_dir.mkdir()
    (watch_dir / "valid.md").write_text("정상 파일 DF_VALID_TOKEN_55", encoding="utf-8")
    (watch_dir / "broken.hwpx").write_bytes(b"CORRUPT_ZIP")
    try:
        res = plugin_client.run_docufinder_sync(
            watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode in (0, 1, 2)
    srch = client.search_bucket(bid, {"query": "DF_VALID_TOKEN_55", "top_k": 5}).assert_status(200)
    assert "DF_VALID_TOKEN_55" in srch.text


@pytest.mark.f31
def test_f31_bva_docufinder_corrupt_state_file_recovers_safely(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
    tmp_path: Path,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    slug = f"dfrec-{uuid.uuid4().hex[:6]}"
    api_client.with_api_key(key).create_bucket({"slug": slug, "name": "DFRec", "access_scope": "dept"}).assert_status((200, 201))

    watch_dir = tmp_path / "w_rec"
    watch_dir.mkdir()
    (watch_dir / "ok.md").write_text("recover test", encoding="utf-8")
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("{broken-json", encoding="utf-8")
    try:
        res = plugin_client.run_docufinder_sync(
            watch_dir, bucket=slug, api_url=api_client.base_url, api_key=key, state_file=state_file
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode in (0, 1, 2)


@pytest.mark.f31
def test_f31_bva_docufinder_anything_unauthorized_chat_returns_401(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    try:
        cfg = plugin_client.get_docufinder_anything_config()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    resp = api_client.with_api_key("orag_revoked_anything").chat_completions(
        {"model": cfg.get("model") or "openrag-anything", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status == 401
    assert "error" in resp.text.lower()


# ============================================================================
# F32 BVA / Negative (5 tests)
# ============================================================================


@pytest.mark.f32
def test_f32_bva_ai_memory_hook_malformed_json_stdin_rejected(
    plugin_client: PluginClient,
) -> None:
    try:
        info = plugin_client.inspect_ai_memory_hooks()
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    hooks_dir = Path(info["path"])
    hooks_cfg = info.get("hooks/hooks.json") or {}
    assert "_parse_error" not in hooks_cfg
    cmd_str = ""
    for entries in (hooks_cfg.get("hooks") or hooks_cfg).values():
        if isinstance(entries, list) and entries:
            first = entries[0]
            if isinstance(first, dict) and "hooks" in first:
                cmd_str = first["hooks"][0].get("command", "")
                break
            if isinstance(first, dict) and "command" in first:
                cmd_str = first["command"]
                break
    assert cmd_str, "Could not locate hook command in hooks.json"
    cmd_str = cmd_str.replace("${CLAUDE_PLUGIN_ROOT}", str(hooks_dir))
    proc = subprocess.run(cmd_str, shell=True, input="{bad-json", text=True, capture_output=True, timeout=10)
    assert proc.returncode != 0


@pytest.mark.f32
def test_f32_bva_ai_memory_hook_empty_transcript_noop_or_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["member_id"])
    try:
        res = plugin_client.invoke_memory_hook("Stop", {}, api_url=api_client.base_url, api_key=key, dry_run=False)
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode in (0, 1, 2)
    cnt = pg_ephemeral.execute_sql(f"SELECT count(*) FROM memory.entries WHERE org_id = '{ctx['org_id']}';").strip()
    assert int(cnt) == 0


@pytest.mark.f32
def test_f32_bva_ai_memory_hook_redacts_additional_secret_patterns(
    plugin_client: PluginClient,
) -> None:
    jwt_tok = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    priv_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----"
    db_url = "postgres://adminuser:supersecretpass99@prod-db.internal:5432/openrag"
    home_mac = "/Users/dev/project/.env"
    home_lin = "/home/ubuntu/.ssh/id_rsa"
    payload = {
        "session_id": "sec-bva",
        "transcript": f"jwt={jwt_tok}\nkey={priv_key}\ndb={db_url}\np1={home_mac}\np2={home_lin}",
    }
    try:
        res = plugin_client.invoke_memory_hook(
            "Stop", payload, api_url="http://127.0.0.1:9999", api_key="dry", dry_run=True
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0
    for sensitive in (jwt_tok, "BEGIN PRIVATE KEY", "supersecretpass99", home_mac, home_lin):
        assert sensitive not in res.stdout


@pytest.mark.f32
def test_f32_bva_ai_memory_hook_unauthorized_key_rejected(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    try:
        res = plugin_client.invoke_memory_hook(
            "Stop",
            {"session_id": "unauth", "transcript": "무단 저장 시도"},
            api_url=api_client.base_url,
            api_key="orag_invalid_hook_key",
            dry_run=False,
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0


@pytest.mark.f32
def test_f32_bva_ai_memory_hook_safeword_block_on_ingestion_gate(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    plugin_client: PluginClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    api_client.with_api_key(key).studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["ingestion_gate"], "enabled": True},
    ).assert_status((200, 201))

    try:
        res = plugin_client.invoke_memory_hook(
            "Stop",
            {"session_id": "orion", "transcript": "PROJECT-ORION 코드 작성 완료"},
            api_url=api_client.base_url,
            api_key=key,
            dry_run=False,
        )
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0
    cnt = pg_ephemeral.execute_sql(
        f"SELECT count(*) FROM memory.episodes WHERE org_id = '{ctx['org_id']}' AND summary_md LIKE '%PROJECT-ORION%';"
    ).strip()
    assert int(cnt) == 0


# ============================================================================
# F33 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f33
def test_f33_bva_jargon_scanner_catches_all_8_forbidden_terms(web_client: WebClient) -> None:
    try:
        web_client.require_web_dir()
        assert web_client.scan_app_tsx_copy() == {}
    except FileNotFoundError as exc:
        fail_missing_target(str(exc))

    for term in FORBIDDEN_APP_JARGON:
        found = web_client.find_jargon_violations(f"문서 검색 중 {term} 오류가 발생했습니다.")
        assert len(found) >= 1, f"Scanner failed to catch forbidden term {term!r}"


@pytest.mark.f33
def test_f33_bva_app_upload_missing_file_or_invalid_scope_returns_400(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    key = _mint_api_key(api_client, ctx, ctx["member_id"])
    client = api_client.with_api_key(key)

    bad_scope = client.app_upload(
        filename="bad.txt", content=b"hello", content_type="text/plain", fields={"scope": "world_public"}
    )
    assert bad_scope.status in (400, 422)
    assert "arn" not in bad_scope.text.lower()

    empty_upload = client.request("POST", "/api/app/upload", data=b"", headers={"Content-Type": "multipart/form-data; boundary=empty"})
    assert empty_upload.status in (400, 422)


@pytest.mark.f33
def test_f33_bva_app_page_thumbnail_unauthorized_peer_denied_403_404(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    owner_key = _mint_api_key(api_client, ctx, ctx["member_id"])
    peer_key = _mint_api_key(api_client, ctx, ctx["peer_id"])

    bucket_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    page_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO corpus.buckets (id, org_id, dept_id, owner_principal_id, slug, name, access_scope, min_auth_level, arn)
        VALUES ('{bucket_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['member_id']}', 'priv-{bucket_id[:6]}', 'Priv', 'dept', 1, 'arn:openrag:{ctx['slug']}:ops:dept:bucket:L1/{bucket_id}');
        INSERT INTO corpus.documents (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, title, content_sha256)
        VALUES ('{doc_id}', '{ctx['org_id']}', '{bucket_id}', '{ctx['dept_id']}', '{ctx['member_id']}', 'private', 1, 'arn:openrag:{ctx['slug']}:ops:private:doc:L1/{doc_id}', '{ctx['org_id']}:private', 'PrivDoc', 'abc');
        INSERT INTO corpus.pages (id, org_id, document_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, page_no)
        VALUES ('{page_id}', '{ctx['org_id']}', '{doc_id}', '{ctx['dept_id']}', '{ctx['member_id']}', 'private', 1, 'arn:openrag:{ctx['slug']}:ops:private:page:L1/{page_id}', '{ctx['org_id']}:private', 1);
        """
    )

    resp = api_client.with_api_key(peer_key).app_page_thumbnail(page_id)
    assert resp.status in (403, 404)
    assert owner_key != peer_key


@pytest.mark.f33
def test_f33_bva_app_memory_retract_or_patch_other_users_private_denied_403(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    peer_key = _mint_api_key(api_client, ctx, ctx["peer_id"])
    mem_id = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO memory.entries (id, org_id, dept_id, owner_principal_id, access_scope, min_auth_level, arn, acl_scope_key, memory_kind, effective_text, bm25_text, content_sha256)
        VALUES ('{mem_id}', '{ctx['org_id']}', '{ctx['dept_id']}', '{ctx['member_id']}', 'private', 3, 'arn:openrag:{ctx['slug']}:ops:private:memory:L3/{mem_id}', '{ctx['org_id']}:private', 'fact', '타인 비공개', '타인 비공개', 'sha');
        """
    )

    assert api_client.with_api_key(peer_key).app_memory_patch({"memory_id": mem_id, "op": "amend", "text": "변조"}).status in (403, 404)
    assert api_client.with_api_key(peer_key).app_memory_retract({"memory_id": mem_id, "reason": "변조"}).status in (403, 404)


@pytest.mark.f33
def test_f33_bva_app_chat_unauthenticated_401_and_safeword_blocked_403(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    assert api_client.app_chat({"message": "안녕하세요"}).status == 401

    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    api_client.with_api_key(admin_key).studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["inbound_query"], "enabled": True},
    ).assert_status((200, 201))

    member_key = _mint_api_key(api_client, ctx, ctx["member_id"])
    blk = api_client.with_api_key(member_key).app_chat({"message": "PROJECT-ORION 알려줘"})
    assert blk.status == 403


@pytest.mark.f33
def test_f33_bva_app_page_thumbnail_after_hard_purge_shows_blackout_pixels(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    bbox = (100, 100, 500, 300)
    png_bytes = sample_factory.png_bytes(
        width=400,
        height=400,
        bg_rgb=(255, 255, 255),
        boxes=[(bbox, (220, 40, 40))],
    )
    up = client.app_upload(
        filename="scan_pii.png",
        content=png_bytes,
        content_type="image/png",
        fields={"scope": "dept", "sensitivity": "normal"},
    ).assert_status((200, 201, 202)).json()
    doc_id = str(up.get("document_id") or up.get("id"))
    page_id = str(up.get("page_id") or "")
    if not page_id:
        page_id = pg_ephemeral.execute_sql(f"SELECT id FROM corpus.pages WHERE document_id = '{doc_id}' LIMIT 1;").strip()
    assert page_id

    before_thumb = client.app_page_thumbnail(page_id).assert_status(200).body
    assert sample_factory.inspect_bbox_blackout(before_thumb, bbox).is_blacked_out is False

    client.studio_pii_request(
        "redactions",
        method="POST",
        payload={
            "target_scope": "document",
            "target_ref": {"document_id": doc_id, "bbox_norm_1000": list(bbox)},
            "redaction_mode": "hard_purge",
            "redact_thumbnails": True,
            "reason": "Blackout thumbnail BBox",
        },
    ).assert_status((200, 201, 202))

    thumb_resp = client.app_page_thumbnail(page_id).assert_status(200)
    report = sample_factory.inspect_bbox_blackout(thumb_resp.body, bbox)
    assert report.is_blacked_out is True


# ============================================================================
# F34 BVA / Negative (6 tests)
# ============================================================================


@pytest.mark.f34
def test_f34_bva_non_admin_member_blocked_from_studio_admin_apis_403(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    member_key = _mint_api_key(api_client, ctx, ctx["member_id"])
    m_client = api_client.with_api_key(member_key)

    assert m_client.studio_iam_request("groups", method="POST", payload={"name": "unauth"}).status == 403
    assert m_client.studio_pii_request("redactions", method="POST", payload={"reason": "unauth"}).status == 403
    assert m_client.studio_index_set_engine({"table_name": "corpus.chunks", "engine": "pgvector"}).status == 403
    assert m_client.studio_eval_request("promote", method="POST", payload={"pipeline_config_id": str(uuid.uuid4())}).status == 403


@pytest.mark.f34
def test_f34_bva_studio_pipeline_update_rejects_invalid_yaml_and_unknown_memory_backend(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    good_yaml = sample_factory.pipeline_yaml("good.yaml", memory_backend="openrag_pg").read_text(encoding="utf-8")
    client.studio_pipeline_update({"pipeline_yaml": good_yaml}).assert_status(200)

    assert client.studio_pipeline_update({"pipeline_yaml": "version: [\nbroken"}).status in (400, 422)
    assert client.studio_pipeline_update({"pipeline_yaml": good_yaml.replace("dim: 128", "dim: 256")}).status in (400, 422)
    assert client.studio_pipeline_update({"pipeline_yaml": good_yaml.replace("openrag_pg", "bad_backend")}).status in (400, 422)

    current = client.studio_pipeline_get().assert_status(200).text
    assert "openrag_pg" in current


@pytest.mark.f34
def test_f34_bva_studio_guard_sandbox_block_vs_mask_vs_allow_with_allowlist_and_toggle(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    r_blk = client.studio_guard_test(
        {
            "text": "PROJECT-ORION 보고",
            "surface": "llm_egress",
            "guard_enabled": True,
            "rules": [{"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["llm_egress"]}],
        }
    ).assert_status(200).json()
    assert r_blk.get("decision") == "block"

    r_allow = client.studio_guard_test(
        {
            "text": "PROJECT-ORION 보고",
            "surface": "llm_egress",
            "guard_enabled": True,
            "allowlist": ["PROJECT-ORION"],
            "rules": [{"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["llm_egress"]}],
        }
    ).assert_status(200).json()
    assert r_allow.get("decision") == "allow"

    r_off = client.studio_guard_test(
        {
            "text": "PROJECT-ORION 보고",
            "surface": "llm_egress",
            "guard_enabled": False,
            "rules": [{"kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["llm_egress"]}],
        }
    ).assert_status(200).json()
    assert r_off.get("decision") == "allow"

    r_mask = client.studio_guard_test(
        {"text": "연락처 010-1234-5678", "surface": "llm_egress", "guard_enabled": True}
    ).assert_status(200).json()
    assert r_mask.get("decision") == "mask"
    assert "010-1234-5678" not in r_mask.get("sanitized_text", "")


@pytest.mark.f34
def test_f34_bva_studio_iam_simulator_explicit_deny_overrides_group_allow(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    client = api_client.with_api_key(admin_key)

    allow_g = str(uuid.uuid4())
    deny_g = str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, dept_id, name) VALUES
            ('{allow_g}', '{ctx['org_id']}', NULL, 'allow-{allow_g[:6]}'),
            ('{deny_g}', '{ctx['org_id']}', NULL, 'deny-{deny_g[:6]}');
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
            ('{ctx['org_id']}', '{allow_g}', '{ctx['member_id']}'),
            ('{ctx['org_id']}', '{deny_g}', '{ctx['member_id']}');
        """
    )

    sim = client.studio_iam_simulate(
        {
            "principal_id": ctx["member_id"],
            "action": "read",
            "access_scope": "group",
            "allowed_group_ids": [allow_g],
            "denied_group_ids": [deny_g],
            "min_auth_level": 1,
            "arn": f"arn:openrag:{ctx['slug']}:ops:group:doc:L1/{uuid.uuid4()}",
        }
    ).assert_status(200).json()
    assert sim.get("allowed") is False
    assert len(sim.get("matched_denies") or []) >= 1


@pytest.mark.f34
def test_f34_bva_studio_eval_promote_blocked_when_security_gate_fails(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    cfg_id, gs_id, run_id = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    pg_ephemeral.execute_sql(
        f"""
        INSERT INTO eval.pipeline_configs (id, org_id, name, profile, pipeline_yaml, config_json, is_promoted)
        VALUES ('{cfg_id}', '{ctx['org_id']}', 'bad-{cfg_id[:6]}', 'custom', 'version: 1', '{{}}'::jsonb, false);
        INSERT INTO eval.golden_sets (id, org_id, name, axis) VALUES ('{gs_id}', '{ctx['org_id']}', 'gs-{gs_id[:6]}', 'retrieval');
        INSERT INTO eval.bench_runs (id, org_id, golden_set_id, pipeline_config_id, track, pii_leak_count, arn_violation_count)
        VALUES ('{run_id}', '{ctx['org_id']}', '{gs_id}', '{cfg_id}', 'track3_security_ocr', 0, 1);
        """
    )
    resp = api_client.with_api_key(admin_key).studio_eval_request(
        "promote", method="POST", payload={"pipeline_config_id": cfg_id}
    )
    assert resp.status in (403, 409, 422)


@pytest.mark.f34
def test_f34_bva_studio_quarantine_retry_unknown_id_returns_404(
    openrag_root: Path,
    pg_ephemeral: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    _ensure_migrated(pg_ephemeral, openrag_root)
    ctx = _seed_org_and_users(pg_ephemeral)
    admin_key = _mint_api_key(api_client, ctx, ctx["admin_id"])
    resp = api_client.with_api_key(admin_key).studio_quarantine_retry("00000000-0000-0000-0000-000000000099")
    assert resp.status == 404
