"""Tier 2 boundary, negative, and corner-case E2E tests for F14-F17 (PII, Vault, Guard, Redaction)."""

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

pytestmark = pytest.mark.tier2

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


def _create_bucket(api_client: Any, prefix: str = "f14bva") -> tuple[str, str]:
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


# ---------------------------------------------------------------------------
# F14 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f14
def test_f14_bva_invalid_rrn_checksum_vs_valid_rrn_under_ko_pii(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={"use_schift_v7": True, "use_marker_ko_pii": True},
    ).assert_status((200, 201))
    try:
        valid_res = api_client.studio_guard_test(
            {"text": "정상 주민번호 900101-1234567 확인", "surface": "llm_egress"}
        ).assert_status(200).json()
        valid_sanitized = str(
            valid_res.get("sanitized_text") or valid_res.get("masked_text") or ""
        )
        assert "900101-1234567" not in valid_sanitized

        invalid_res = api_client.studio_guard_test(
            {"text": "비정상 번호 991332-9999999 확인", "surface": "llm_egress"}
        ).assert_status(200).json()
        invalid_sanitized = str(
            invalid_res.get("sanitized_text") or invalid_res.get("masked_text") or ""
        )
        assert "991332-9999999" in invalid_sanitized or (
            (invalid_res.get("matched_categories") or []) != (valid_res.get("matched_categories") or [])
        ), (
            f"Invalid RRN 991332-9999999 should not match valid RRN checksum/categories: "
            f"valid={valid_res!r}, invalid={invalid_res!r}"
        )
    finally:
        api_client.studio_pii_request(
            "profiles",
            method="PUT",
            payload=_DEFAULT_PII_PROFILE,
        )


@pytest.mark.tier2
@pytest.mark.f14
def test_f14_bva_adjacent_korean_multibyte_utf8_no_space_around_pii(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    text = "성명:홍길동(900101-1234567)연락처:010-1234-5678이메일:hong.gildong@example.or.kr끝"
    res = api_client.studio_guard_test({"text": text, "surface": "llm_egress"}).assert_status(200).json()
    sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
    assert "성명:" in sanitized
    assert sanitized.endswith("끝")
    assert "900101-1234567" not in sanitized
    assert "010-1234-5678" not in sanitized
    assert "hong.gildong@example.or.kr" not in sanitized


@pytest.mark.tier2
@pytest.mark.f14
def test_f14_bva_overlapping_custom_dict_and_regex_spans(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "use_schift_v7": True,
            "custom_dict_terms": ["010-1234-5678-내선99", "홍길동팀장"],
        },
    ).assert_status((200, 201))
    try:
        res = api_client.studio_guard_test(
            {"text": "담당자 홍길동팀장 연락처 010-1234-5678-내선99", "surface": "llm_egress"}
        ).assert_status(200).json()
        sanitized = str(res.get("sanitized_text") or res.get("masked_text") or "")
        assert "1234-5678" not in sanitized
        assert "홍길동팀장" not in sanitized
    finally:
        api_client.studio_pii_request(
            "profiles", method="PUT", payload=_DEFAULT_PII_PROFILE
        )


@pytest.mark.tier2
@pytest.mark.f14
def test_f14_bva_empty_and_whitespace_only_input(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    for sample in ("", "   \n\t  "):
        res = api_client.studio_guard_test(
            {"text": sample, "surface": "llm_egress"}
        ).assert_status(200).json()
        assert res.get("decision") == "allow"
        assert (res.get("matched_categories") or []) == []


@pytest.mark.tier2
@pytest.mark.f14
def test_f14_bva_all_detectors_disabled_returns_unmodified_text(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "use_schift_v7": False,
            "use_marker_ko_pii": False,
            "use_schift_v6_org": False,
            "custom_dict_terms": [],
        },
    ).assert_status((200, 201))
    try:
        res = api_client.studio_guard_test(
            {"text": "주민등록번호 900101-1234567", "surface": "llm_egress"}
        ).assert_status(200).json()
        assert "900101-1234567" in str(
            res.get("sanitized_text") or res.get("masked_text") or ""
        )
        if "detector_matches" in res:
            assert res["detector_matches"] == []
    finally:
        api_client.studio_pii_request(
            "profiles",
            method="PUT",
            payload=_DEFAULT_PII_PROFILE,
        )


# ---------------------------------------------------------------------------
# F15 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f15
def test_f15_bva_tampered_vault_ciphertext_fails_closed_on_unmask(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    org_id = str(uuid.uuid4())
    pid = str(uuid.uuid4())
    api_client.studio_iam_request(
        "orgs", method="POST", payload={"id": org_id, "slug": f"tamp-{org_id[:6]}", "name": "Tamper Org"}
    ).assert_status((200, 201, 409))
    api_client.studio_iam_request(
        "principals",
        method="POST",
        payload={"id": pid, "org_id": org_id, "email": f"t-{pid[:6]}@example.or.kr", "display_name": "TamperAdmin", "auth_level": 8, "can_unmask_pii": True},
    ).assert_status((200, 201, 409))
    token = str(
        api_client.studio_iam_request(
            "api-keys",
            method="POST",
            payload={"org_id": org_id, "principal_id": pid, "name": "tamper-k", "max_auth_level": 8, "allowed_actions": ["read", "search", "pii:unmask"]},
        ).assert_status((200, 201)).json().get("token")
    )

    client = api_client.with_api_key(token)
    bkt = client.create_bucket(
        {"slug": f"f15-tamp-{uuid.uuid4().hex[:6]}", "name": "Tamper Bkt", "org_id": org_id, "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bkt_id = str(bkt.get("id") or bkt.get("bucket_id"))
    doc = sample_factory.korean_pii_doc(safe_words=())
    client.create_document(
        bkt_id, {"title": "Tamper Doc", "text": doc["text"], "access_scope": "org"}
    ).assert_status((200, 201))

    # Corrupt ciphertext bytes in pii.vault_entries
    migrated_pg.execute_sql(
        "UPDATE pii.vault_entries SET ciphertext = decode('deadbeefcafebabe', 'hex')"
    )
    resp = client.chat_completions(
        {
            "model": "vllm/local",
            "bucket": bkt_id,
            "unmask": True,
            "messages": [{"role": "user", "content": "주민등록번호 확인"}],
        }
    )
    assert resp.status in (200, 400, 422, 500)
    assert "900101-1234567" not in resp.text


@pytest.mark.tier2
@pytest.mark.f15
def test_f15_bva_purged_vault_entry_cannot_be_unmasked(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    org_id = str(uuid.uuid4())
    pid = str(uuid.uuid4())
    api_client.studio_iam_request(
        "orgs", method="POST", payload={"id": org_id, "slug": f"prg-{org_id[:6]}", "name": "Purge Org"}
    ).assert_status((200, 201, 409))
    api_client.studio_iam_request(
        "principals",
        method="POST",
        payload={"id": pid, "org_id": org_id, "email": f"prg-{pid[:6]}@example.or.kr", "display_name": "PurgeAdmin", "auth_level": 8, "can_unmask_pii": True},
    ).assert_status((200, 201, 409))
    token = str(
        api_client.studio_iam_request(
            "api-keys",
            method="POST",
            payload={"org_id": org_id, "principal_id": pid, "name": "prg-k", "max_auth_level": 8, "allowed_actions": ["read", "search", "pii:unmask"]},
        ).assert_status((200, 201)).json().get("token")
    )

    client = api_client.with_api_key(token)
    bkt = client.create_bucket(
        {"slug": f"f15-prg-{uuid.uuid4().hex[:6]}", "name": "Purge Bkt", "org_id": org_id, "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bkt_id = str(bkt.get("id") or bkt.get("bucket_id"))
    doc = sample_factory.korean_pii_doc(safe_words=())
    client.create_document(
        bkt_id, {"title": "Purged Vault Doc", "text": doc["text"], "access_scope": "org"}
    ).assert_status((200, 201))

    migrated_pg.execute_sql("UPDATE pii.vault_entries SET purged_at = now()")
    resp = client.chat_completions(
        {
            "model": "vllm/local",
            "bucket": bkt_id,
            "unmask": True,
            "messages": [{"role": "user", "content": "주민번호 확인"}],
        }
    )
    assert resp.status in (200, 403, 410)
    assert "900101-1234567" not in resp.text


@pytest.mark.tier2
@pytest.mark.f15
def test_f15_bva_cross_org_vault_token_isolation(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    org_a, org_b = str(uuid.uuid4()), str(uuid.uuid4())
    pid_a, pid_b = str(uuid.uuid4()), str(uuid.uuid4())
    for oid, pid, tag in ((org_a, pid_a, "oa"), (org_b, pid_b, "ob")):
        api_client.studio_iam_request(
            "orgs", method="POST", payload={"id": oid, "slug": f"{tag}-{oid[:6]}", "name": tag}
        ).assert_status((200, 201, 409))
        api_client.studio_iam_request(
            "principals",
            method="POST",
            payload={"id": pid, "org_id": oid, "email": f"{tag}-{pid[:6]}@example.or.kr", "display_name": tag, "auth_level": 9, "can_unmask_pii": True},
        ).assert_status((200, 201, 409))

    key_a = str(
        api_client.studio_iam_request(
            "api-keys", method="POST", payload={"org_id": org_a, "principal_id": pid_a, "name": "ka", "max_auth_level": 9, "allowed_actions": ["read", "search", "pii:unmask"]}
        ).assert_status((200, 201)).json().get("token")
    )
    key_b = str(
        api_client.studio_iam_request(
            "api-keys", method="POST", payload={"org_id": org_b, "principal_id": pid_b, "name": "kb", "max_auth_level": 9, "allowed_actions": ["read", "search", "pii:unmask"]}
        ).assert_status((200, 201)).json().get("token")
    )

    bkt = api_client.with_api_key(key_a).create_bucket(
        {"slug": f"f15-oa-{uuid.uuid4().hex[:6]}", "name": "OrgA Bkt", "org_id": org_a, "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bkt_id = str(bkt.get("id") or bkt.get("bucket_id"))
    doc = sample_factory.korean_pii_doc(safe_words=())
    api_client.with_api_key(key_a).create_document(
        bkt_id, {"title": "OrgA PII", "text": doc["text"], "access_scope": "org"}
    ).assert_status((200, 201))

    resp_b = api_client.with_api_key(key_b).chat_completions(
        {
            "model": "vllm/local",
            "bucket": bkt_id,
            "unmask": True,
            "messages": [{"role": "user", "content": "주민번호 확인"}],
        }
    )
    assert resp_b.status in (200, 403, 404)
    assert "900101-1234567" not in resp_b.text
    assert "010-1234-5678" not in resp_b.text


@pytest.mark.tier2
@pytest.mark.f15
def test_f15_bva_gateway_and_audit_logs_contain_zero_plaintext_pii(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    doc = sample_factory.korean_pii_doc(safe_words=())
    bucket_id, _ = _create_bucket(api_client, "f15-logs")
    api_client.create_document(
        bucket_id, {"title": "Log Check", "text": doc["text"], "access_scope": "org"}
    ).assert_status((200, 201))
    api_client.chat_completions(
        {"model": "openrouter/default", "bucket": bucket_id, "messages": [{"role": "user", "content": "연락처 확인"}]}
    ).assert_status(200)

    gw_logs = api_client.studio_gateway_request("logs").assert_status(200).text
    pii_logs = api_client.studio_pii_request("audit-logs").assert_status(200).text
    for blob in (gw_logs, pii_logs):
        assert "900101-1234567" not in blob
        assert "010-1234-5678" not in blob


@pytest.mark.tier2
@pytest.mark.f15
def test_f15_bva_no_pii_document_creates_zero_spurious_vault_entries(
    migrated_pg: PgEphemeralCluster, api_client: Any, sample_factory: Any
) -> None:
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pii.vault_entries")
            before_cnt = int(cur.fetchone()[0])  # type: ignore[index]

    bucket_id, _ = _create_bucket(api_client, "f15-clean")
    pdf = sample_factory.pdf_bytes(["Quarterly infrastructure capacity summary 2026"])
    api_client.upload_document(
        bucket_id,
        filename="clean_summary.pdf",
        content=pdf,
        content_type="application/pdf",
        fields={"access_scope": "org"},
    ).assert_status((200, 201))

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM pii.vault_entries")
            after_cnt = int(cur.fetchone()[0])  # type: ignore[index]
    assert after_cnt == before_cnt


# ---------------------------------------------------------------------------
# F16 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f16
def test_f16_bva_action_precedence_block_overrides_mask_warn_allow(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    created_ids: list[str] = []
    for kind, pat, act in (
        ("keyword", "ORION", "allow"),
        ("keyword", "PROJECT-ORION", "warn"),
        ("pii_category", "RRN", "mask"),
        ("safe_word", "PROJECT-ORION", "block"),
    ):
        r = api_client.studio_pii_request(
            "guard-rules",
            method="POST",
            payload={"rule_kind": kind, "pattern_or_category": pat, "action": act, "surfaces": ["inbound_query"]},
        ).assert_status((200, 201)).json()
        rid = r.get("id") or r.get("rule_id")
        if rid:
            created_ids.append(str(rid))

    try:
        res = api_client.studio_guard_test(
            {"text": "PROJECT-ORION 900101-1234567", "surface": "inbound_query"}
        ).assert_status(200).json()
        assert res.get("decision") == "block"

        chat = api_client.chat_completions(
            {"model": "openrag-default", "messages": [{"role": "user", "content": "PROJECT-ORION 900101-1234567"}]}
        )
        assert chat.status in (400, 403, 422)
        assert "900101-1234567" not in chat.text
    finally:
        for rid in created_ids:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier2
@pytest.mark.f16
def test_f16_bva_master_toggle_off_bypasses_guard_while_preserving_iam(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"rule_kind": "safe_word", "pattern_or_category": "PROJECT-ORION", "action": "block", "surfaces": ["inbound_query"]},
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    policies = api_client.studio_pii_request("guard-policies").assert_status(200).json()
    pol_list = policies if isinstance(policies, list) else (policies.get("policies") or [policies])
    pol_id = str(pol_list[0].get("id")) if pol_list and isinstance(pol_list[0], dict) and pol_list[0].get("id") else None

    try:
        if pol_id:
            api_client.studio_pii_request(
                f"guard-policies/{pol_id}", method="PUT", payload={"master_enabled": False}
            ).assert_status((200, 201))

        g_res = api_client.studio_guard_test(
            {"text": "PROJECT-ORION 테스트", "surface": "inbound_query", "master_enabled": False}
        ).assert_status(200).json()
        assert g_res.get("decision") == "allow"

        # IAM must still reject unauthorized simulation even when Guard master toggle is OFF
        sim = api_client.studio_iam_simulate(
            {
                "org_id": str(uuid.uuid4()),
                "principal_id": str(uuid.uuid4()),
                "auth_level": 1,
                "access_scope": "private",
                "min_auth_level": 9,
                "resource_arn": "arn:openrag:org1:dept1:private:document:L9/d1",
                "action": "read",
            }
        ).assert_status(200).json()
        assert sim["allowed"] is False
    finally:
        if pol_id:
            api_client.studio_pii_request(
                f"guard-policies/{pol_id}", method="PUT", payload={"master_enabled": True}
            )
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier2
@pytest.mark.f16
def test_f16_bva_surface_isolation_mcp_egress_rule_does_not_block_inbound_query(
    api_client: Any, mcp_stdio: Any, migrated_pg: PgEphemeralCluster
) -> None:
    token = f"INTERNAL-ONLY-MCP-{uuid.uuid4().hex[:6]}"
    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"rule_kind": "safe_word", "pattern_or_category": token, "action": "block", "surfaces": ["mcp_egress"]},
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    try:
        inb = api_client.studio_guard_test({"text": token, "surface": "inbound_query"}).assert_status(200).json()
        assert inb.get("decision") != "block"

        with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
            mcp.handshake()
            egress = mcp.guard_check(token, surface="mcp_egress")
            assert egress.is_error is True or egress.json().get("decision") == "block"
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier2
@pytest.mark.f16
def test_f16_bva_invalid_regex_rule_rejected_at_creation_and_valid_regex_matches(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bad = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"rule_kind": "regex", "pattern_or_category": "[unclosed(group", "action": "block", "surfaces": ["inbound_query"]},
    )
    assert bad.status in (400, 422)

    good = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"rule_kind": "regex", "pattern_or_category": r"KOREA-SECRET-\d{4}", "action": "block", "surfaces": ["inbound_query"]},
    ).assert_status((200, 201)).json()
    rid = good.get("id") or good.get("rule_id")
    try:
        res = api_client.studio_guard_test(
            {"text": "문서 KOREA-SECRET-4821 포함", "surface": "inbound_query"}
        ).assert_status(200).json()
        assert res.get("decision") == "block"
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier2
@pytest.mark.f16
def test_f16_bva_guard_events_audit_never_stores_raw_blocked_content(
    migrated_pg: PgEphemeralCluster, api_client: Any
) -> None:
    secret = f"PROJECT-ORION-9988-{uuid.uuid4().hex[:6]}"
    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={"rule_kind": "safe_word", "pattern_or_category": secret, "action": "block", "surfaces": ["inbound_query", "mcp_egress"]},
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    try:
        api_client.chat_completions(
            {"model": "openrag-default", "messages": [{"role": "user", "content": f"Leak {secret}"}]}
        )
        http_mcp = McpHttpClient(api_client)
        http_mcp.handshake()
        http_mcp.guard_check(f"MCP {secret}", surface="mcp_egress")

        events_text = api_client.studio_pii_request("guard-events").assert_status(200).text
        assert "block" in events_text.lower()
        assert secret not in events_text

        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT decision, metadata::text FROM pii.guard_events")
                rows = cur.fetchall()
                assert any(r[0] == "block" for r in rows)
                for _, meta in rows:
                    assert secret not in (meta or "")
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


@pytest.mark.tier2
@pytest.mark.f16
def test_f16_bva_llm_egress_mask_surface_isolation_with_benign_query(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, bucket_slug = _create_bucket(api_client, "f16-bva-egress")
    anchor = f"ANCHOR-BVA-{uuid.uuid4().hex[:6]}"
    egress_tok = f"EGRESS-MASK-ONLY-{uuid.uuid4().hex[:6]}"

    api_client.create_document(
        bucket_id,
        {
            "title": "Egress Mask BVA Doc",
            "text": f"Public {anchor} references {egress_tok} token",
            "access_scope": "org",
        },
    ).assert_status((200, 201))

    rule = api_client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": egress_tok,
            "action": "mask",
            "surfaces": ["llm_egress"],
        },
    ).assert_status((200, 201)).json()
    rid = rule.get("id") or rule.get("rule_id")

    try:
        inb = api_client.studio_guard_test(
            {"text": egress_tok, "surface": "inbound_query"}
        ).assert_status(200).json()
        assert inb.get("decision") == "allow"

        chat = api_client.chat_completions(
            {
                "model": "openrag-default",
                "bucket": bucket_slug,
                "messages": [{"role": "user", "content": f"{anchor} 요약해줘"}],
            }
        ).assert_status(200)
        assert egress_tok not in chat.text
    finally:
        if rid:
            api_client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")


# ---------------------------------------------------------------------------
# F17 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f17
def test_f17_bva_non_admin_member_forbidden_from_cascade_redaction(
    api_client: Any, openrag_cli: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    org_id = str(uuid.uuid4())
    member_id = str(uuid.uuid4())
    api_client.studio_iam_request(
        "orgs", method="POST", payload={"id": org_id, "slug": f"morg-{org_id[:6]}", "name": "MOrg"}
    ).assert_status((200, 201, 409))
    api_client.studio_iam_request(
        "principals",
        method="POST",
        payload={"id": member_id, "org_id": org_id, "email": f"m-{member_id[:6]}@example.or.kr", "display_name": "Member", "auth_level": 2},
    ).assert_status((200, 201, 409))
    member_key = str(
        api_client.studio_iam_request(
            "api-keys", method="POST", payload={"org_id": org_id, "principal_id": member_id, "name": "mk", "max_auth_level": 2}
        ).assert_status((200, 201)).json().get("token")
    )

    bucket_id, _ = _create_bucket(api_client, "f17-forbid")
    up = api_client.create_document(
        bucket_id, {"title": "Org Target", "text": "PROTECTED-DOC-F17", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    resp = api_client.with_api_key(member_key).studio_pii_request(
        "redactions",
        method="POST",
        payload={"target_scope": "document", "target_ref": {"document_id": doc_id}, "redaction_mode": "mask_keep_vault", "reason": "unauth"},
    )
    assert resp.status == 403

    try:
        cli_res = openrag_cli.redact_run(
            doc_id=doc_id,
            scope="org",
            reason="unauth cli",
            env={"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": member_key},
        )
        assert cli_res.returncode != 0
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))


@pytest.mark.tier2
@pytest.mark.f17
def test_f17_bva_nonexistent_target_id_and_invalid_mode_rejected(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bad_mode = api_client.studio_pii_request(
        "redactions",
        method="POST",
        payload={
            "target_scope": "document",
            "target_ref": {"document_id": str(uuid.uuid4())},
            "redaction_mode": "burn_everything",
            "reason": "bad mode",
        },
    )
    assert bad_mode.status in (400, 422)

    missing_target = api_client.studio_pii_request(
        "redactions",
        method="POST",
        payload={
            "target_scope": "document",
            "target_ref": {"document_id": "00000000-0000-0000-0000-00000000dead"},
            "redaction_mode": "mask_keep_vault",
            "reason": "missing doc",
        },
    )
    assert missing_target.status in (200, 400, 404, 422)
    if missing_target.status == 200:
        counts = missing_target.json().get("affected_counts") or {}
        assert sum(int(v) for v in counts.values()) == 0


@pytest.mark.tier2
@pytest.mark.f17
def test_f17_bva_redact_thumbnails_false_preserves_original_pixels(
    api_client: Any, openrag_cli: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f17-keeppx")
    bbox = (100, 100, 700, 400)
    png = sample_factory.png_bytes(400, 400, bg_rgb=(255, 255, 255), boxes=[(bbox, (240, 240, 240))])
    up = api_client.upload_document(
        bucket_id,
        filename="keep_pixels.png",
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

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thumb_embedding::text FROM corpus.pages WHERE id = %s::uuid",
                (page_id,),
            )
            row = cur.fetchone()
            before_vec_raw = row[0] if row else None

    try:
        openrag_cli.redact_run(
            doc_id=doc_id,
            scope="document",
            redact_thumbnails=False,
            reason="keep pixels",
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    thumb = api_client.app_page_thumbnail(page_id).assert_status(200).body
    assert sample_factory.inspect_bbox_blackout(thumb, bbox).is_blacked_out is False

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thumb_embedding::text FROM corpus.pages WHERE id = %s::uuid",
                (page_id,),
            )
            after_row = cur.fetchone()
            assert after_row is not None and after_row[0] is not None
            after_vec = _parse_pg_vector(after_row[0])
            assert len(after_vec) == 128 and all(math.isfinite(v) for v in after_vec)
            norm = math.sqrt(sum(v * v for v in after_vec))
            assert abs(norm - 1.0) < 1e-3, f"Expected L2-normalized 128d vector, got norm={norm}"
            if before_vec_raw is not None:
                assert after_vec == _parse_pg_vector(before_vec_raw)

            cur.execute(
                "SELECT COUNT(*) FROM pii.redaction_events WHERE target_id = %s::uuid OR target_table = 'corpus.chunks'",
                (doc_id,),
            )
            assert int(cur.fetchone()[0]) >= 1  # type: ignore[index]


@pytest.mark.tier2
@pytest.mark.f17
def test_f17_bva_idempotent_rerun_cascade_redaction_completes_cleanly(
    api_client: Any, openrag_cli: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f17-idem")
    up = api_client.create_document(
        bucket_id,
        {"title": "Idem Redact", "text": "Token REPEAT-REDACT-777 here", "access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    try:
        for _ in range(2):
            res = openrag_cli.redact_run(
                doc_id=doc_id,
                scope="document",
                hard_purge=True,
                redact_thumbnails=True,
                reason="idempotent rerun",
                env={"DATABASE_URL": migrated_pg.dsn()},
                expect_exit=0,
            )
            assert res.returncode == 0
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))


@pytest.mark.tier2
@pytest.mark.f17
def test_f17_bva_post_redaction_search_and_mcp_get_page_leak_zero_purged_data(
    api_client: Any, mcp_stdio: Any, openrag_cli: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    canary = f"PURGE-ME-CANARY-{uuid.uuid4().hex[:6]}"
    bucket_id, _ = _create_bucket(api_client, "f17-zeroleak")
    hwpx = sample_factory.hwpx_bytes(title="Zero Leak HWPX", paragraphs=[f"Sensitive {canary} phrase"])
    up = api_client.upload_document(
        bucket_id,
        filename="zero_leak.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    pre = api_client.search_bucket(bucket_id, {"query": canary, "top_k": 5}).assert_status(200).json()
    assert canary in str(pre)

    try:
        openrag_cli.redact_run(
            pattern=canary,
            scope="org",
            hard_purge=True,
            redact_thumbnails=True,
            reason="zero leak purge",
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    post = api_client.search_bucket(bucket_id, {"query": canary, "top_k": 5}).assert_status(200).json()
    assert canary not in str(post)

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        page = mcp.rag_get_page(document_id=doc_id, page_no=1)
        assert canary not in page.text
