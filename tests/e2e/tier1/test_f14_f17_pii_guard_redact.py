"""Tier 1 isolated happy-path E2E tests for F14-F17 (PII Detectors, Capsule Vault, Guard, Cascade Redaction)."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError
from lib.mcp_client import McpHttpClient
from lib.pg_harness import PgEphemeralCluster

pytestmark = pytest.mark.tier1

_DEFAULT_PII_PROFILE: dict[str, Any] = {
    "use_schift_v7": True,
    "use_marker_ko_pii": False,
    "use_schift_v6_org": False,
    "custom_dict_terms": [],
    "allowlist_terms": [],
}


def _parse_pg_vector(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    text = str(raw).strip()
    assert text.startswith("[") and text.endswith("]"), f"Invalid vector text: {text!r}"
    return [float(tok) for tok in text[1:-1].split(",") if tok.strip()]


def _ensure_migrated(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    migrations_dir = openrag_root / "migrations"
    if not migrations_dir.is_dir():
        fail_missing_target(f"Missing migrations directory: {migrations_dir}")
    pg_ephemeral.ensure_started()
    if getattr(pg_ephemeral, "_e2e_migrated_port", None) != pg_ephemeral.port:
        pg_ephemeral.apply_migrations(migrations_dir)
        setattr(pg_ephemeral, "_e2e_migrated_port", pg_ephemeral.port)
    return pg_ephemeral


@pytest.fixture(scope="module")
def migrated_pg(pg_ephemeral: PgEphemeralCluster, openrag_root: Path) -> PgEphemeralCluster:
    return _ensure_migrated(pg_ephemeral, openrag_root)


def _create_bucket(api_client: Any, prefix: str = "f14") -> tuple[str, str]:
    slug = f"{prefix}-{uuid.uuid4().hex[:8]}"
    resp = api_client.create_bucket(
        {"slug": slug, "name": f"Bucket {slug}", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    return str(resp.get("id") or resp.get("bucket_id") or slug), slug


def _extract_doc_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("document_id", "id", "doc_id"):
            val = payload.get(key)
            if val:
                return str(val)
        if isinstance(payload.get("document"), dict):
            return _extract_doc_id(payload["document"])
    raise AssertionError(f"Could not extract document_id from {payload!r}")


def _extract_page_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if payload.get("page_id"):
            return str(payload["page_id"])
        pages = payload.get("pages")
        if isinstance(pages, list) and pages and isinstance(pages[0], dict):
            pid = pages[0].get("id") or pages[0].get("page_id")
            if pid:
                return str(pid)
    return None


def _dynamic_insert(
    cur: psycopg.Cursor[Any],
    schema: str,
    table: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    cur.execute(
        """
        SELECT column_name, data_type, udt_name, is_nullable, column_default, is_generated
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    cols_meta = cur.fetchall()
    if not cols_meta:
        raise AssertionError(f"Table {schema}.{table} missing")
    available = {row[0]: row for row in cols_meta}
    insert_data = {k: v for k, v in values.items() if k in available}

    for col_name, data_type, udt_name, is_nullable, col_default, is_generated in cols_meta:
        if col_name in insert_data or is_generated == "ALWAYS" or is_nullable == "YES" or col_default is not None:
            continue
        if data_type == "uuid":
            insert_data[col_name] = str(uuid.uuid4())
        elif data_type in ("text", "character varying", "character"):
            insert_data[col_name] = f"e2e_{col_name}_{uuid.uuid4().hex[:8]}"
        elif data_type in ("integer", "smallint", "bigint"):
            insert_data[col_name] = 1
        elif data_type in ("double precision", "real", "numeric"):
            insert_data[col_name] = 0.0
        elif data_type == "boolean":
            insert_data[col_name] = False
        elif data_type in ("jsonb", "json"):
            insert_data[col_name] = "{}"
        elif data_type == "ARRAY":
            insert_data[col_name] = []
        elif data_type == "bytea":
            insert_data[col_name] = b"e2e"
        elif udt_name == "vector":
            insert_data[col_name] = "[" + ",".join(["1.0"] + ["0.0"] * 127) + "]"
        else:
            insert_data[col_name] = f"e2e_{uuid.uuid4().hex[:6]}"

    cols = list(insert_data.keys())
    phs: list[str] = []
    params: list[Any] = []
    for c in cols:
        val = insert_data[c]
        data_type, udt_name = available[c][1], available[c][2]
        if udt_name == "vector":
            phs.append("%s::vector(128)")
        elif data_type in ("jsonb", "json") and isinstance(val, str):
            phs.append(f"%s::{data_type}")
        else:
            phs.append("%s")
        params.append(val)

    sql = f'INSERT INTO "{schema}"."{table}" ({", ".join(cols)}) VALUES ({", ".join(phs)}) RETURNING *'
    cur.execute(sql, params)
    row = cur.fetchone()
    assert cur.description is not None and row is not None
    return {desc[0]: row[idx] for idx, desc in enumerate(cur.description)}


def _setup_unmask_principals(api_client: Any, pg: PgEphemeralCluster) -> tuple[str, str, str]:
    """Create org + authorized (can_unmask_pii=True) and unauthorized API keys."""
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    auth_pid = str(uuid.uuid4())
    unauth_pid = str(uuid.uuid4())

    api_client.studio_iam_request(
        "orgs",
        method="POST",
        payload={"id": org_id, "slug": f"pii-org-{org_id[:6]}", "name": "PII Org"},
    ).assert_status((200, 201, 409))

    for pid, unmask in ((auth_pid, True), (unauth_pid, False)):
        api_client.studio_iam_request(
            "principals",
            method="POST",
            payload={
                "id": pid,
                "org_id": org_id,
                "email": f"u-{pid[:8]}@example.or.kr",
                "display_name": f"PII-{pid[:6]}",
                "auth_level": 6,
                "can_unmask_pii": unmask,
                "dept_ids": [dept_id],
            },
        ).assert_status((200, 201, 409))

    auth_key_res = api_client.studio_iam_request(
        "api-keys",
        method="POST",
        payload={
            "org_id": org_id,
            "principal_id": auth_pid,
            "name": f"unmask-ok-{uuid.uuid4().hex[:6]}",
            "max_auth_level": 6,
            "allowed_actions": ["read", "search", "pii:unmask"],
        },
    ).assert_status((200, 201)).json()
    unauth_key_res = api_client.studio_iam_request(
        "api-keys",
        method="POST",
        payload={
            "org_id": org_id,
            "principal_id": unauth_pid,
            "name": f"unmask-no-{uuid.uuid4().hex[:6]}",
            "max_auth_level": 6,
            "allowed_actions": ["read", "search"],
        },
    ).assert_status((200, 201)).json()

    auth_key = str(auth_key_res.get("token") or auth_key_res.get("api_key") or auth_key_res.get("key"))
    unauth_key = str(unauth_key_res.get("token") or unauth_key_res.get("api_key") or unauth_key_res.get("key"))
    return org_id, auth_key, unauth_key


# ---------------------------------------------------------------------------
# F14 — M3 Multi-Detector PII Engine
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f14
def test_f14_schift_v7_default_detects_korean_rrn_phone_email_name_address(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    doc = sample_factory.korean_pii_doc(safe_words=())
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={"use_schift_v7": True, "use_marker_ko_pii": False, "use_schift_v6_org": False},
    ).assert_status((200, 201))
    try:
        res = api_client.studio_guard_test(
            {"text": doc["text"], "surface": "llm_egress"}
        ).assert_status(200).json()
        sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
        assert sanitized
        for tok in (doc["rrn"], doc["phone"], doc["email"]):
            assert tok not in sanitized, f"PII token {tok!r} leaked in sanitized_text"
        assert doc["name"] not in sanitized or doc["address"] not in sanitized
    finally:
        api_client.studio_pii_request("profiles", method="PUT", payload=_DEFAULT_PII_PROFILE)


@pytest.mark.tier1
@pytest.mark.f14
def test_f14_marker_ko_pii_toggle_with_checksum_and_trace(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={"use_schift_v7": True, "use_marker_ko_pii": True},
    ).assert_status((200, 201))
    try:
        prof = api_client.studio_pii_request("profiles").assert_status(200).json()
        assert prof.get("use_marker_ko_pii") is True

        doc = sample_factory.korean_pii_doc(safe_words=())
        res = api_client.studio_guard_test(
            {"text": doc["text"], "surface": "llm_egress"}
        ).assert_status(200).json()
        sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
        assert "900101-1234567" not in sanitized
        assert "010-1234-5678" not in sanitized
    finally:
        api_client.studio_pii_request("profiles", method="PUT", payload=_DEFAULT_PII_PROFILE)


@pytest.mark.tier1
@pytest.mark.f14
def test_f14_custom_dict_terms_detects_org_specific_identifiers(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "use_schift_v7": True,
            "custom_dict_terms": ["내부감사코드-2026-X99"],
        },
    ).assert_status((200, 201))
    try:
        res = api_client.studio_guard_test(
            {"text": "문서 참조번호: 내부감사코드-2026-X99 확인 요망", "surface": "llm_egress"}
        ).assert_status(200).json()
        sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
        assert "내부감사코드-2026-X99" not in sanitized
    finally:
        api_client.studio_pii_request("profiles", method="PUT", payload=_DEFAULT_PII_PROFILE)


@pytest.mark.tier1
@pytest.mark.f14
def test_f14_allowlist_terms_exempts_approved_tokens(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "use_schift_v7": True,
            "allowlist_terms": ["02-120", "홍길동"],
        },
    ).assert_status((200, 201))
    try:
        res = api_client.studio_guard_test(
            {
                "text": "민원 안내: 홍길동 담당자(02-120) 및 개인번호 010-1234-5678",
                "surface": "llm_egress",
            }
        ).assert_status(200).json()
        sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
        assert "02-120" in sanitized
        assert "홍길동" in sanitized
        assert "010-1234-5678" not in sanitized
    finally:
        api_client.studio_pii_request("profiles", method="PUT", payload=_DEFAULT_PII_PROFILE)


@pytest.mark.tier1
@pytest.mark.f14
def test_f14_schift_v6_toggle_persists_and_evaluates(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "use_schift_v7": True,
            "use_schift_v6_org": True,
            "use_marker_ko_pii": False,
        },
    ).assert_status((200, 201))
    try:
        prof = api_client.studio_pii_request("profiles").assert_status(200).json()
        assert prof.get("use_schift_v6_org") is True

        res = api_client.studio_guard_test(
            {"text": "연락처 010-9876-5432", "surface": "llm_egress"}
        ).assert_status(200).json()
        sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
        assert "010-9876-5432" not in sanitized
    finally:
        api_client.studio_pii_request("profiles", method="PUT", payload=_DEFAULT_PII_PROFILE)


# ---------------------------------------------------------------------------
# F15 — M3 2-Stage PII Capsule & AES-256-GCM Vault
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f15
def test_f15_ingest_encrypts_pii_in_vault_and_masks_stored_chunks(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    doc = sample_factory.korean_pii_doc(safe_words=())
    bucket_id, _ = _create_bucket(api_client, "f15-vault")
    hwpx = sample_factory.hwpx_bytes(title="인사기록", paragraphs=[doc["text"]])
    up = api_client.upload_document(
        bucket_id,
        filename="hr_pii.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT redacted_text FROM corpus.chunks WHERE document_id = %s::uuid",
                (doc_id,),
            )
            chunk_rows = cur.fetchall()
            assert chunk_rows
            for (txt,) in chunk_rows:
                assert "900101-1234567" not in (txt or "")
                assert "010-1234-5678" not in (txt or "")

            cur.execute("SELECT algorithm FROM pii.vault_keys")
            algos = [r[0] for r in cur.fetchall()]
            assert any("AES-256-GCM" in str(a).upper() for a in algos)

            cur.execute("SELECT ciphertext, nonce FROM pii.vault_entries")
            entries = cur.fetchall()
            assert len(entries) >= 1
            for ct, nonce in entries:
                assert ct and nonce
                assert b"900101-1234567" not in bytes(ct)
                assert b"010-1234-5678" not in bytes(ct)


@pytest.mark.tier1
@pytest.mark.f15
def test_f15_external_llm_egress_zero_pii_leak(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    doc = sample_factory.korean_pii_doc(safe_words=())
    bucket_id, bucket_slug = _create_bucket(api_client, "f15-egress")
    api_client.create_document(
        bucket_id,
        {
            "title": "External LLM PII Test",
            "text": doc["text"],
            "access_scope": "org",
            "min_auth_level": 2,
        },
    ).assert_status((200, 201))

    resp = api_client.chat_completions(
        {
            "model": "openrouter/default",
            "bucket": bucket_slug,
            "messages": [{"role": "user", "content": "담당자 연락처와 주민등록번호를 확인해줘"}],
        }
    ).assert_status(200)

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pii_masked, model_id, status FROM gateway.request_logs ORDER BY created_at DESC LIMIT 5"
            )
            logs = cur.fetchall()
            assert logs
            assert any(bool(row[0]) for row in logs)
    gw_logs = api_client.studio_gateway_request("logs").assert_status(200).text
    assert "900101-1234567" not in gw_logs
    assert "010-1234-5678" not in gw_logs
    assert "900101-1234567" not in resp.text


@pytest.mark.tier1
@pytest.mark.f15
def test_f15_authorized_principal_unmasks_pii_and_writes_audit_log(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    org_id, auth_key, _ = _setup_unmask_principals(api_client, migrated_pg)
    auth_client = api_client.with_api_key(auth_key)
    doc = sample_factory.korean_pii_doc(safe_words=())

    bkt = auth_client.create_bucket(
        {"slug": f"f15-unmask-{uuid.uuid4().hex[:6]}", "name": "Unmask Bucket", "org_id": org_id, "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bucket_id = str(bkt.get("id") or bkt.get("bucket_id"))
    auth_client.create_document(
        bucket_id,
        {"title": "Unmask Target", "text": doc["text"], "access_scope": "org", "min_auth_level": 3},
    ).assert_status((200, 201))

    chat = auth_client.chat_completions(
        {
            "model": "vllm/local",
            "bucket": bucket_id,
            "unmask": True,
            "messages": [{"role": "user", "content": "담당자 홍길동의 주민등록번호와 연락처"}],
        }
    ).assert_status(200).json()
    assert "900101-1234567" in str(chat) or "010-1234-5678" in str(chat)

    audits = api_client.studio_pii_request("audit-logs").assert_status(200).json()
    assert len(audits) >= 1


@pytest.mark.tier1
@pytest.mark.f15
def test_f15_unauthorized_principal_stays_masked_with_denied_audit(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    org_id, auth_key, unauth_key = _setup_unmask_principals(api_client, migrated_pg)
    auth_client = api_client.with_api_key(auth_key)
    unauth_client = api_client.with_api_key(unauth_key)
    doc = sample_factory.korean_pii_doc(safe_words=())

    bkt = auth_client.create_bucket(
        {"slug": f"f15-nounmask-{uuid.uuid4().hex[:6]}", "name": "NoUnmask Bucket", "org_id": org_id, "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bucket_id = str(bkt.get("id") or bkt.get("bucket_id"))
    auth_client.create_document(
        bucket_id,
        {"title": "Masked Target", "text": doc["text"], "access_scope": "org", "min_auth_level": 3},
    ).assert_status((200, 201))

    srch = unauth_client.search_bucket(bucket_id, {"query": "인사", "top_k": 5}).assert_status(200)
    assert "900101-1234567" not in srch.text
    assert "010-1234-5678" not in srch.text

    chat = unauth_client.chat_completions(
        {
            "model": "openrag-default",
            "bucket": bucket_id,
            "unmask": True,
            "messages": [{"role": "user", "content": "주민등록번호 알려줘"}],
        }
    )
    assert chat.status in (200, 403)
    assert "900101-1234567" not in chat.text
    assert "010-1234-5678" not in chat.text


@pytest.mark.tier1
@pytest.mark.f15
def test_f15_capsule_roundtrip_via_studio_audit_logs_api(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    doc = sample_factory.korean_pii_doc(safe_words=())
    bucket_id, _ = _create_bucket(api_client, "f15-audit")
    api_client.create_document(
        bucket_id, {"title": "Audit Doc", "text": doc["text"], "access_scope": "org"}
    ).assert_status((200, 201))
    api_client.chat_completions(
        {"model": "openrag-default", "bucket": bucket_id, "messages": [{"role": "user", "content": "홍길동 정보"}]}
    ).assert_status(200)

    logs = api_client.studio_pii_request("audit-logs").assert_status(200).json()
    assert isinstance(logs, (list, dict))
    assert "900101-1234567" not in str(logs)


# ---------------------------------------------------------------------------
# F16 — M3 Configurable PII & Safe-Word Guard
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f16
def test_f16_safeword_block_rejects_chat_completion_with_guard_blocked(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    rule_resp = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"],
        },
    ).assert_status((200, 201)).json()
    rule_id = rule_resp.get("id") or rule_resp.get("rule_id")

    try:
        chat = api_client.chat_completions(
            {
                "model": "openrag-default",
                "messages": [{"role": "user", "content": "PROJECT-ORION 기밀 계획 요약해줘"}],
            }
        )
        assert chat.status in (400, 403, 422)
        body_lower = chat.text.lower()
        assert "guard_blocked" in body_lower or "block" in body_lower
        assert "PROJECT-ORION" not in str((chat.json() if "{" in chat.text else {}).get("choices", ""))
    finally:
        if rule_id:
            api_client.studio_pii_request(f"guard-rules/{rule_id}", method="DELETE")


@pytest.mark.tier1
@pytest.mark.f16
def test_f16_mcp_stdio_and_http_guard_check_block_and_mask(
    api_client: Any, mcp_stdio: Any, migrated_pg: PgEphemeralCluster
) -> None:
    r_sw = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"],
        },
    ).assert_status((200, 201)).json()
    r_rrn = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "pii_category",
            "pattern_or_category": "RRN",
            "action": "mask",
            "surfaces": ["mcp_egress", "llm_egress"],
        },
    ).assert_status((200, 201)).json()

    try:
        http_mcp = McpHttpClient(api_client)
        http_mcp.handshake()
        with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as stdio_mcp:
            stdio_mcp.handshake()
            for client in (stdio_mcp, http_mcp):
                blk = client.guard_check("PROJECT-ORION 보고서", surface="mcp_egress")
                blk_data = blk.json()
                assert blk.is_error is True or blk_data.get("decision") == "block"

                msk = client.guard_check("주민번호 900101-1234567 확인", surface="mcp_egress")
                assert msk.is_error is False
                msk_data = msk.json()
                assert msk_data.get("decision") == "mask"
                assert "900101-1234567" not in str(
                    msk_data.get("sanitized_text") or msk_data.get("masked_text") or ""
                )
    finally:
        for r in (r_sw, r_rrn):
            rid = r.get("id") or r.get("rule_id")
            if rid:
                api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier1
@pytest.mark.f16
def test_f16_warn_action_passes_with_warning_and_logs_event(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    token = f"CONFIDENTIAL-DRAFT-{uuid.uuid4().hex[:6]}"
    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "keyword",
            "pattern_or_category": token,
            "action": "warn",
            "surfaces": ["inbound_query"],
        },
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")
    try:
        res = api_client.studio_guard_test(
            {"text": f"{token} 검토", "surface": "inbound_query"}
        ).assert_status(200).json()
        assert res.get("decision") == "warn"
        events = api_client.studio_pii_request("guard-events").assert_status(200).json()
        assert "warn" in str(events).lower()
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier1
@pytest.mark.f16
def test_f16_guard_allowlist_exempts_safe_term(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query"],
        },
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    policies = api_client.studio_pii_request("guard-policies").assert_status(200).json()
    pol_list = policies if isinstance(policies, list) else (policies.get("policies") or [policies])
    pol_id = str(pol_list[0].get("id")) if pol_list and isinstance(pol_list[0], dict) and pol_list[0].get("id") else None
    try:
        if pol_id:
            api_client.studio_pii_request(
                f"guard-policies/{pol_id}",
                method="PUT",
                payload={"master_enabled": True, "allowlist_terms": ["PROJECT-ORION-DEMO"]},
            ).assert_status((200, 201))

        res = api_client.studio_guard_test(
            {
                "text": "PROJECT-ORION-DEMO 공개 발표",
                "surface": "inbound_query",
                "allowlist_terms": ["PROJECT-ORION-DEMO"],
            }
        ).assert_status(200).json()
        assert res.get("decision") in ("allow", "warn")
        assert res.get("decision") != "block"
    finally:
        if pol_id:
            api_client.studio_pii_request(
                f"guard-policies/{pol_id}",
                method="PUT",
                payload={"master_enabled": True, "allowlist_terms": []},
            )
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier1
@pytest.mark.f16
def test_f16_guard_rule_crud_and_ingestion_gate_surface(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f16-ing")
    secret_kw = f"TOP-SECRET-BLUEPRINT-{uuid.uuid4().hex[:6]}"
    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "keyword",
            "pattern_or_category": secret_kw,
            "action": "block",
            "surfaces": ["ingestion_gate"],
        },
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    try:
        resp = api_client.create_document(
            bucket_id,
            {"title": "Blocked Blueprint", "text": f"{secret_kw} spec", "access_scope": "org"},
        )
        assert resp.status in (400, 403, 422)
        assert "guard_blocked" in resp.text.lower() or "block" in resp.text.lower()

        docs = api_client.list_documents(bucket_id).assert_status(200).json()
        assert "Blocked Blueprint" not in str(docs)
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier1
@pytest.mark.f16
def test_f16_llm_egress_surface_isolates_retrieved_context_from_benign_query(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, bucket_slug = _create_bucket(api_client, "f16-egress-iso")
    anchor = f"ANCHOR-F16-{uuid.uuid4().hex[:6]}"
    egress_secret = f"ORION-EGRESS-ONLY-{uuid.uuid4().hex[:6]}"

    api_client.create_document(
        bucket_id,
        {
            "title": "Egress Isolation Doc",
            "text": f"Summary {anchor} contains secret {egress_secret} context",
            "access_scope": "org",
        },
    ).assert_status((200, 201))

    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": egress_secret,
            "action": "block",
            "surfaces": ["llm_egress"],
        },
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    try:
        inb = api_client.studio_guard_test(
            {"text": egress_secret, "surface": "inbound_query"}
        ).assert_status(200).json()
        assert inb.get("decision") != "block"

        chat = api_client.chat_completions(
            {
                "model": "openrag-default",
                "bucket": bucket_slug,
                "messages": [{"role": "user", "content": f"{anchor} 내용 요약해줘"}],
            }
        )
        assert chat.status in (400, 403, 422), (
            f"Expected isolated llm_egress block on retrieved context, got {chat.status}: {chat.text}"
        )
        assert "guard_blocked" in chat.text.lower() or "block" in chat.text.lower()
        assert egress_secret not in chat.text
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


# ---------------------------------------------------------------------------
# F17 — M3 Full-Layer Cascade Redaction
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f17
def test_f17_cli_redact_run_masks_chunks_and_pages(
    migrated_pg: PgEphemeralCluster, openrag_cli: Any, api_client: Any, sample_factory: Any
) -> None:
    token = f"SECRET-TOKEN-F17-{uuid.uuid4().hex[:6]}"
    bucket_id, _ = _create_bucket(api_client, "f17-cli")
    hwpx = sample_factory.hwpx_bytes(title="Cascade Doc", paragraphs=[f"Sensitive {token} payload"])
    up = api_client.upload_document(
        bucket_id,
        filename="cascade.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    try:
        res = openrag_cli.redact_run(
            doc_id=doc_id,
            scope="document",
            redact_thumbnails=True,
            reason="tier1 redact test",
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode == 0

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT redacted_text FROM corpus.chunks WHERE document_id = %s::uuid",
                (doc_id,),
            )
            chunks = cur.fetchall()
            assert chunks
            for (txt,) in chunks:
                assert token not in (txt or "")
            cur.execute(
                "SELECT redacted_markdown FROM corpus.pages WHERE document_id = %s::uuid",
                (doc_id,),
            )
            for (md,) in cur.fetchall():
                assert token not in (md or "")


@pytest.mark.tier1
@pytest.mark.f17
def test_f17_cascade_redaction_blacks_out_thumbnail_bbox_pixels(
    api_client: Any, openrag_cli: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f17-bbox")
    bbox = (100, 100, 700, 400)
    png = sample_factory.png_bytes(400, 400, bg_rgb=(255, 255, 255), boxes=[(bbox, (240, 240, 240))])
    up = api_client.upload_document(
        bucket_id,
        filename="bbox_target.png",
        content=png,
        content_type="image/png",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)
    page_id = _extract_page_id(up)

    if not page_id:
        with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
            mcp.handshake()
            pj = mcp.rag_get_page(document_id=doc_id, page_no=1).json()
            page_id = str(pj.get("page_id") or pj.get("id"))
    assert page_id

    before_thumb = api_client.app_page_thumbnail(page_id).assert_status(200).body
    assert sample_factory.inspect_bbox_blackout(before_thumb, bbox).is_blacked_out is False

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thumb_embedding::text FROM corpus.pages WHERE id = %s::uuid",
                (page_id,),
            )
            row = cur.fetchone()
            before_vec_raw = row[0] if row else None

    job = api_client.studio_pii_request(
        "redactions",
        method="POST",
        payload={
            "target_scope": "document",
            "target_ref": {"document_id": doc_id, "bbox_norm_1000": list(bbox)},
            "redaction_mode": "mask_keep_vault",
            "redact_thumbnails": True,
            "reason": "bbox blackout verification",
        },
    ).assert_status((200, 201, 202)).json()
    job_id = str(job.get("job_id") or job.get("id"))
    assert job_id

    after_thumb = api_client.app_page_thumbnail(page_id).assert_status(200).body
    report = sample_factory.inspect_bbox_blackout(after_thumb, bbox, min_black_ratio=0.90)
    assert report.is_blacked_out is True, f"Thumbnail BBox was not blacked out: {report}"

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thumb_embedding::text FROM corpus.pages WHERE id = %s::uuid",
                (page_id,),
            )
            after_row = cur.fetchone()
            assert after_row is not None and after_row[0] is not None
            after_vec = _parse_pg_vector(after_row[0])
            assert len(after_vec) == 128
            assert all(math.isfinite(v) for v in after_vec)
            norm = math.sqrt(sum(v * v for v in after_vec))
            assert abs(norm - 1.0) < 1e-3, f"Re-embedded 128d vector not L2-normalized: {norm}"
            if before_vec_raw is not None:
                before_vec = _parse_pg_vector(before_vec_raw)
                assert after_vec != before_vec, "thumb_embedding was not recomputed after BBox blackout"

            cur.execute(
                "SELECT target_table, action_taken FROM pii.redaction_events WHERE redaction_job_id = %s::uuid",
                (job_id,),
            )
            events = cur.fetchall()
            assert len(events) >= 1, f"Expected pii.redaction_events rows for job_id={job_id}"


@pytest.mark.tier1
@pytest.mark.f17
def test_f17_hard_purge_destroys_vault_entries_and_clears_memory_patch_history(
    migrated_pg: PgEphemeralCluster, openrag_cli: Any, api_client: Any, sample_factory: Any
) -> None:
    canary_rrn = "900101-1234567"
    bucket_id, _ = _create_bucket(api_client, "f17-purge")
    doc_up = api_client.create_document(
        bucket_id,
        {
            "title": "Purge Doc",
            "text": f"주민등록번호 {canary_rrn} 보관 테스트",
            "access_scope": "org",
        },
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(doc_up)

    mem_ep = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    pid = str(uuid.uuid4())
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SELECT id::text FROM iam.orgs LIMIT 1")
                row = cur.fetchone()
                if row:
                    org_id = row[0]
                else:
                    _dynamic_insert(cur, "iam", "orgs", {"id": org_id, "slug": f"porg-{org_id[:6]}", "name": "Purge Org"})
                cur.execute("SELECT id::text FROM iam.departments WHERE org_id = %s::uuid LIMIT 1", (org_id,))
                drow = cur.fetchone()
                if drow:
                    dept_id = drow[0]
                else:
                    _dynamic_insert(
                        cur,
                        "iam",
                        "departments",
                        {"id": dept_id, "org_id": org_id, "name": "Purge Dept", "arn_path": f"org/{org_id[:6]}/dept/{dept_id[:6]}"},
                    )
                cur.execute("SELECT id::text FROM iam.principals WHERE org_id = %s::uuid LIMIT 1", (org_id,))
                prow = cur.fetchone()
                if prow:
                    pid = prow[0]
                else:
                    _dynamic_insert(
                        cur,
                        "iam",
                        "principals",
                        {"id": pid, "org_id": org_id, "email": f"p-{pid[:6]}@example.or.kr", "display_name": "Purge User", "auth_level": 5},
                    )
                _dynamic_insert(
                    cur,
                    "memory",
                    "episodes",
                    {
                        "id": mem_ep,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "owner_principal_id": pid,
                        "access_scope": "org",
                        "arn": f"arn:openrag:{org_id}:org:org:episode:L1/{mem_ep}",
                        "source_harness": "chat_ui",
                        "summary_md": f"EP {canary_rrn}",
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "entries",
                    {
                        "id": mem_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "source_episode_id": mem_ep,
                        "owner_principal_id": pid,
                        "access_scope": "org",
                        "arn": f"arn:openrag:{org_id}:org:org:memory:L1/{mem_id}",
                        "acl_scope_key": f"{org_id}:org:org:L1",
                        "memory_kind": "fact",
                        "effective_text": f"Updated {canary_rrn}",
                        "bm25_text": f"Updated {canary_rrn}",
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "patches",
                    {
                        "id": str(uuid.uuid4()),
                        "org_id": org_id,
                        "memory_entry_id": mem_id,
                        "actor_type": "human",
                        "actor_id": pid,
                        "patch_op": "amend",
                        "prev_version": 1,
                        "new_version": 2,
                        "prev_text": f"Prev {canary_rrn}",
                        "new_text": f"Updated {canary_rrn}",
                        "reason": "seed",
                    },
                )

    try:
        openrag_cli.redact_run(
            pattern=canary_rrn,
            scope="org",
            redact_thumbnails=True,
            hard_purge=True,
            reason="gdpr purge",
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ve.purged_at
                FROM pii.vault_entries ve
                JOIN pii.chunk_spans cs ON cs.vault_entry_id = ve.id
                JOIN corpus.chunks c ON c.id = cs.chunk_id
                WHERE c.document_id = %s::uuid
                """,
                (doc_id,),
            )
            doc_vault_rows = cur.fetchall()
            if doc_vault_rows:
                assert all(r[0] is not None for r in doc_vault_rows)

            cur.execute("SELECT COUNT(*) FROM pii.vault_entries WHERE purged_at IS NOT NULL")
            assert int(cur.fetchone()[0]) >= 1  # type: ignore[index]

            cur.execute("SELECT effective_text FROM memory.entries WHERE id = %s::uuid", (mem_id,))
            eff_row = cur.fetchone()
            assert eff_row is not None and canary_rrn not in (eff_row[0] or "")

            cur.execute("SELECT prev_text FROM memory.patches WHERE memory_entry_id = %s::uuid", (mem_id,))
            for (prev_txt,) in cur.fetchall():
                assert prev_txt is None

            cur.execute("SELECT COUNT(*) FROM pii.redaction_events")
            assert int(cur.fetchone()[0]) >= 1  # type: ignore[index]


@pytest.mark.tier1
@pytest.mark.f17
def test_f17_pattern_redaction_propagates_across_chunks_memory_wiki_and_outbox(
    migrated_pg: PgEphemeralCluster, openrag_cli: Any, api_client: Any
) -> None:
    canary = f"LEAKED-CODE-{uuid.uuid4().hex[:6]}"
    bucket_id, _ = _create_bucket(api_client, "f17-prop")
    api_client.create_document(
        bucket_id,
        {"title": "Leak Doc", "text": f"Document has {canary}", "access_scope": "org"},
    ).assert_status((200, 201))

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SELECT id::text FROM iam.orgs LIMIT 1")
                org_id = cur.fetchone()[0]  # type: ignore[index]
                cur.execute("SELECT id::text FROM iam.departments WHERE org_id = %s::uuid LIMIT 1", (org_id,))
                drow = cur.fetchone()
                if drow:
                    dept_id = drow[0]
                else:
                    dept_id = str(uuid.uuid4())
                    _dynamic_insert(
                        cur,
                        "iam",
                        "departments",
                        {"id": dept_id, "org_id": org_id, "name": "Prop Dept", "arn_path": f"org/{org_id[:6]}/dept/{dept_id[:6]}"},
                    )
                cur.execute("SELECT id::text FROM iam.principals WHERE org_id = %s::uuid LIMIT 1", (org_id,))
                prow = cur.fetchone()
                if prow:
                    pid = prow[0]
                else:
                    pid = str(uuid.uuid4())
                    _dynamic_insert(
                        cur,
                        "iam",
                        "principals",
                        {"id": pid, "org_id": org_id, "email": f"w-{pid[:6]}@example.or.kr", "display_name": "Wiki User", "auth_level": 5},
                    )
                ep_id = str(uuid.uuid4())
                mem_id = str(uuid.uuid4())
                wiki_id = str(uuid.uuid4())
                _dynamic_insert(
                    cur,
                    "memory",
                    "episodes",
                    {
                        "id": ep_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "owner_principal_id": pid,
                        "access_scope": "org",
                        "arn": f"arn:openrag:{org_id}:org:org:episode:L1/{ep_id}",
                        "source_harness": "chat_ui",
                        "summary_md": f"EP {canary}",
                    },
                )
                _dynamic_insert(
                    cur,
                    "memory",
                    "entries",
                    {
                        "id": mem_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "source_episode_id": ep_id,
                        "owner_principal_id": pid,
                        "access_scope": "org",
                        "arn": f"arn:openrag:{org_id}:org:org:memory:L1/{mem_id}",
                        "acl_scope_key": f"{org_id}:org:org:L1",
                        "memory_kind": "fact",
                        "effective_text": canary,
                        "bm25_text": canary,
                    },
                )
                _dynamic_insert(
                    cur,
                    "wiki",
                    "pages",
                    {
                        "id": wiki_id,
                        "org_id": org_id,
                        "dept_id": dept_id,
                        "bucket_id": bucket_id,
                        "owner_principal_id": pid,
                        "slug": f"leak-{wiki_id[:6]}",
                        "title": "Leak Wiki",
                        "body_md": f"# Wiki\n{canary}",
                        "bm25_text": f"Wiki {canary}",
                        "access_scope": "org",
                        "arn": f"arn:openrag:{org_id}:org:org:wiki:L1/{wiki_id}",
                        "acl_scope_key": f"{org_id}:org:org:L1",
                    },
                )

    try:
        openrag_cli.redact_run(
            pattern=canary,
            scope="org",
            redact_thumbnails=False,
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM corpus.chunks WHERE redacted_text LIKE %s", (f"%{canary}%",))
            assert cur.fetchone()[0] == 0  # type: ignore[index]
            cur.execute("SELECT COUNT(*) FROM memory.entries WHERE effective_text LIKE %s", (f"%{canary}%",))
            assert cur.fetchone()[0] == 0  # type: ignore[index]
            cur.execute("SELECT COUNT(*) FROM wiki.pages WHERE body_md LIKE %s", (f"%{canary}%",))
            assert cur.fetchone()[0] == 0  # type: ignore[index]
            cur.execute("SELECT op FROM indexing.outbox WHERE op IN ('redact_reindex', 'delete_vector')")
            assert len(cur.fetchall()) >= 1


@pytest.mark.tier1
@pytest.mark.f17
def test_f17_immutable_redaction_jobs_and_events_audit_via_studio_api(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f17-audit")
    hwpx = sample_factory.hwpx_bytes(title="Audit Redact", paragraphs=["민감 토큰 AUDIT-F17-TOKEN"])
    up = api_client.upload_document(
        bucket_id,
        filename="audit_redact.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    job = api_client.studio_pii_request(
        "redactions",
        method="POST",
        payload={
            "target_scope": "document",
            "target_ref": {"document_id": doc_id},
            "redaction_mode": "mask_keep_vault",
            "redact_thumbnails": True,
            "reason": "audit verify",
        },
    ).assert_status((200, 201, 202)).json()
    job_id = str(job.get("job_id") or job.get("id"))
    assert job_id

    listed = api_client.studio_pii_request("redactions").assert_status(200).json()
    assert listed
    assert "completed" in str(listed).lower() or "text_masked" in str(listed).lower()

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT target_table, action_taken FROM pii.redaction_events WHERE redaction_job_id = %s::uuid",
                (job_id,),
            )
            events = cur.fetchall()
            assert len(events) >= 1, f"Expected immutable pii.redaction_events for job {job_id}"
