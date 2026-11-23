"""Tier 3 pairwise E2E tests: Parse, OCR, Thumbnails, PII Capsule, Guard & Cascade Redaction (7 tests)."""

from __future__ import annotations

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
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_org_with_users,
)

pytestmark = [pytest.mark.tier3]


@pytest.mark.f11
@pytest.mark.f13
def test_t3_f11_f13_hwpx_table_ingest_generates_webp_thumbs_and_128d_visual_vectors(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F1113")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1113")
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"hwpx-th-{uuid.uuid4().hex[:6]}", "name": "HWPXThumb", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    hwpx = sample_factory.hwpx_bytes(
        title="표 포함 공문",
        paragraphs=["본문 요약"],
        tables=[[["구분", "금액"], ["여비", "85,000원_T3_CELL"]]],
    )
    up = client.upload_document(
        bid,
        filename="table_thumb.hwpx",
        content=hwpx,
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202)).json()
    doc_id = extract_id(up, "document_id", "id", "doc_id")

    tbl_html = migrated_pg.execute_sql(
        f"SELECT COALESCE(string_agg(table_html, ' '), '') FROM corpus.chunks WHERE document_id = '{doc_id}';"
    ).strip()
    assert "<table" in tbl_html.lower() and "<tr" in tbl_html.lower() and "<td" in tbl_html.lower()
    assert "85,000원_T3_CELL" in tbl_html

    page_rows = migrated_pg.execute_sql(
        f"SELECT id::text, vector_dims(thumb_embedding) FROM corpus.pages WHERE document_id = '{doc_id}' ORDER BY page_no LIMIT 1;"
    ).strip()
    assert page_rows, "Expected at least 1 page row in corpus.pages"
    page_id, dims_str = page_rows.split("|")
    assert int(dims_str) == 128

    thumb_resp = client.app_page_thumbnail(page_id).assert_status(200)
    assert thumb_resp.body[:4] == b"RIFF" and thumb_resp.body[8:12] == b"WEBP"


@pytest.mark.f12
@pytest.mark.f09
def test_t3_f12_f09_schift_ocr_mode_enforces_auth_level_and_zdr_batch_rejection(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F1209")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1209")
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"ocr-{uuid.uuid4().hex[:6]}", "name": "OcrLevel", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    pdf_bytes = sample_factory.pdf_bytes(["Confidential L8 OCR Page"])
    # L8+ with share mode must be rejected or forced to zdr/private (never stay share)
    resp_l8_share = client.upload_document(
        bid,
        filename="l8_share.pdf",
        content=pdf_bytes,
        content_type="application/pdf",
        fields={"access_scope": "dept", "min_auth_level": "8", "ocr_mode": "share"},
    )
    assert resp_l8_share.status in (200, 201, 202, 400, 403, 422)
    if resp_l8_share.status in (200, 201, 202):
        doc_id = extract_id(resp_l8_share.json(), "document_id", "id", "doc_id")
        sens = migrated_pg.execute_sql(
            f"SELECT sensitivity FROM corpus.documents WHERE id = '{doc_id}';"
        ).strip()
        assert sens in ("zdr", "private", "confidential", "secret"), sens

    # ZDR + batch=true must return 400
    resp_zdr_batch = client.upload_document(
        bid,
        filename="zdr_batch.pdf",
        content=pdf_bytes,
        content_type="application/pdf",
        fields={"access_scope": "dept", "min_auth_level": "8", "ocr_mode": "zdr", "batch": "true"},
    )
    assert resp_zdr_batch.status == 400


@pytest.mark.f13
@pytest.mark.f17
def test_t3_f13_f17_cascade_redaction_blackbox_masks_thumbnail_and_reembeds_128d(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F1317")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1317")
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"red-{uuid.uuid4().hex[:6]}", "name": "RedactThumb", "access_scope": "dept"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    bbox = (150, 150, 650, 420)
    bright_png = sample_factory.png_bytes(320, 320, bg_rgb=(250, 250, 250), boxes=[(bbox, (240, 60, 60))])
    up = client.upload_document(
        bid,
        filename="pii_thumb.png",
        content=bright_png,
        content_type="image/png",
        fields={"access_scope": "dept", "min_auth_level": "3"},
    ).assert_status((200, 201, 202)).json()
    doc_id = extract_id(up, "document_id", "id", "doc_id")

    page_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.pages WHERE document_id = '{doc_id}' LIMIT 1;"
    ).strip()
    assert page_id
    before_vec = migrated_pg.execute_sql(
        f"SELECT thumb_embedding::text FROM corpus.pages WHERE id = '{page_id}';"
    ).strip()

    openrag_cli.redact_run(
        doc_id=doc_id,
        scope="document",
        redact_thumbnails=True,
        hard_purge=True,
        reason="t3-bbox-audit",
        env={"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    after_thumb = client.app_page_thumbnail(page_id).assert_status(200).body
    report = sample_factory.inspect_bbox_blackout(after_thumb, bbox, min_black_ratio=0.90)
    assert report.is_blacked_out is True, report

    after_vec = migrated_pg.execute_sql(
        f"SELECT thumb_embedding::text FROM corpus.pages WHERE id = '{page_id}';"
    ).strip()
    assert after_vec and after_vec != before_vec

    job_status = migrated_pg.execute_sql(
        f"SELECT status FROM pii.redaction_jobs WHERE org_id = '{ctx['org_id']}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert job_status == "completed"
    ev_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.redaction_events WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    assert ev_cnt >= 1
    unpurged_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.vault_entries ve JOIN pii.chunk_spans cs ON cs.vault_entry_id = ve.id JOIN corpus.chunks c ON c.id = cs.chunk_id WHERE c.document_id = '{doc_id}' AND ve.purged_at IS NULL;"
        ).strip()
    )
    assert unpurged_cnt == 0


@pytest.mark.f14
@pytest.mark.f15
def test_t3_f14_f15_multi_detector_custom_dict_feeds_aes_vault_and_authorized_unmask(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F1415")
    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1415-adm")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="t3-1415-peer")
    admin_client = api_client.with_api_key(admin_key)

    admin_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "name": "default",
            "use_schift_v7": True,
            "use_marker_ko_pii": True,
            "custom_dict_terms": ["ORAG-CUSTOM-99"],
            "allowlist_terms": ["PUBLIC-ALLOW-01"],
        },
    ).assert_status((200, 201))

    b = admin_client.create_bucket(
        {"slug": f"pii-cap-{uuid.uuid4().hex[:6]}", "name": "PiiCap", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    pii_doc = sample_factory.korean_pii_doc(safe_words=(), allowlist_terms=("PUBLIC-ALLOW-01",))
    full_text = pii_doc["text"] + "\n내부식별코드: ORAG-CUSTOM-99"
    created = admin_client.create_document(
        bid, {"title": "CapsuleDoc", "text": full_text, "access_scope": "org", "min_auth_level": 3}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(created, "document_id", "id", "doc_id")

    stored_text = migrated_pg.execute_sql(
        f"SELECT string_agg(redacted_text, ' ') FROM corpus.chunks WHERE document_id = '{doc_id}';"
    )
    assert "900101-1234567" not in stored_text
    assert "ORAG-CUSTOM-99" not in stored_text
    assert "PUBLIC-ALLOW-01" in stored_text

    vault_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.vault_entries WHERE org_id = '{ctx['org_id']}' AND octet_length(ciphertext) > 0 AND octet_length(nonce) >= 12;"
        ).strip()
    )
    assert vault_cnt >= 1

    peer_chat = api_client.with_api_key(peer_key).chat_completions(
        {
            "model": "vllm/local",
            "bucket_id": bid,
            "unmask_pii": True,
            "messages": [{"role": "user", "content": "담당자 주민등록번호와 내부식별코드를 알려줘"}],
        }
    ).assert_status(200)
    assert "900101-1234567" not in peer_chat.text

    admin_chat = admin_client.chat_completions(
        {
            "model": "vllm/local",
            "bucket_id": bid,
            "unmask_pii": True,
            "messages": [{"role": "user", "content": "담당자 주민등록번호와 내부식별코드를 알려줘"}],
        }
    ).assert_status(200)
    assert "900101-1234567" in admin_chat.text or "010-1234-5678" in admin_chat.text


@pytest.mark.f15
@pytest.mark.f08
def test_t3_f15_f08_pii_unmask_requires_both_principal_flag_and_group_permission(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F1508")
    # Principal has can_unmask_pii=true, placed in grp_no_unmask vs grp_with_unmask
    migrated_pg.execute_sql(
        f"UPDATE iam.principals SET can_unmask_pii = true WHERE id = '{ctx['other_dept_id']}';"
    )
    grp_a = str(uuid.uuid4())
    grp_b = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.groups (id, org_id, name, group_type, allowed_actions) VALUES
            ('{grp_a}', '{ctx['org_id']}', 'read-only-{grp_a[:6]}', 'permission_group', ARRAY['read', 'search']),
            ('{grp_b}', '{ctx['org_id']}', 'unmask-grp-{grp_b[:6]}', 'permission_group', ARRAY['read', 'search', 'pii:unmask']);
        INSERT INTO iam.group_members (org_id, group_id, principal_id) VALUES
            ('{ctx['org_id']}', '{grp_a}', '{ctx['other_dept_id']}'),
            ('{ctx['org_id']}', '{grp_b}', '{ctx['other_dept_id']}');
        """
    )

    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1508-adm")
    key_a = mint_api_key(
        api_client, ctx["org_id"], ctx["other_dept_id"], tag="t3-1508-a", scoped_group_ids=[grp_a]
    )
    key_b = mint_api_key(
        api_client, ctx["org_id"], ctx["other_dept_id"], tag="t3-1508-b", scoped_group_ids=[grp_b]
    )

    b = (
        api_client.with_api_key(admin_key)
        .create_bucket({"slug": f"unm-g-{uuid.uuid4().hex[:6]}", "name": "UnmGrp", "access_scope": "org"})
        .assert_status((200, 201))
        .json()
    )
    bid = extract_id(b, "id", "bucket_id")
    pii_doc = sample_factory.korean_pii_doc(safe_words=())
    api_client.with_api_key(admin_key).create_document(
        bid,
        {
            "title": "GroupUnmaskDoc",
            "text": pii_doc["text"],
            "access_scope": "group",
            "allowed_group_ids": [grp_a, grp_b],
            "min_auth_level": 3,
        },
    ).assert_status((200, 201))

    resp_a = api_client.with_api_key(key_a).chat_completions(
        {
            "model": "vllm/local",
            "bucket_id": bid,
            "unmask_pii": True,
            "messages": [{"role": "user", "content": "주민등록번호 알려줘"}],
        }
    ).assert_status(200)
    assert "900101-1234567" not in resp_a.text

    resp_b = api_client.with_api_key(key_b).chat_completions(
        {
            "model": "vllm/local",
            "bucket_id": bid,
            "unmask_pii": True,
            "messages": [{"role": "user", "content": "주민등록번호 알려줘"}],
        }
    ).assert_status(200)
    assert "900101-1234567" in resp_b.text

    audit_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.audit_logs WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    assert audit_cnt >= 2


@pytest.mark.f14
@pytest.mark.f16
def test_t3_f14_f16_custom_dict_pii_and_safeword_guard_mask_vs_block_surfaces(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_org_with_users(migrated_pg, label="T3F1416")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1416")
    client = api_client.with_api_key(key)

    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "pii_category",
            "pattern_or_category": "rrn_kr",
            "action": "mask",
            "surfaces": ["llm_egress", "inbound_query"],
            "enabled": True,
        },
    ).assert_status((200, 201))
    client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query", "ingestion_gate"],
            "enabled": True,
        },
    ).assert_status((200, 201))

    masked = client.chat_completions(
        {"model": "openrag-auto", "messages": [{"role": "user", "content": "직원 900101-1234567 문의"}]}
    ).assert_status(200)
    assert "900101-1234567" not in masked.text

    blocked = client.chat_completions(
        {"model": "openrag-auto", "messages": [{"role": "user", "content": "PROJECT-ORION 예산"}]}
    )
    assert blocked.status == 403
    assert "guard_blocked" in blocked.text.lower()

    ev_blob = migrated_pg.execute_sql(
        f"SELECT COALESCE(string_agg(metadata::text, ' '), '') FROM pii.guard_events WHERE org_id = '{ctx['org_id']}';"
    )
    assert "PROJECT-ORION" not in ev_blob and "900101-1234567" not in ev_blob


@pytest.mark.f16
@pytest.mark.f17
def test_t3_f16_f17_guard_ingestion_gate_plus_retroactive_cascade_redaction(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_org_with_users(migrated_pg, label="T3F1617")
    key = mint_api_key(api_client, ctx["org_id"], ctx["admin_id"], tag="t3-1617")
    client = api_client.with_api_key(key)

    b = client.create_bucket(
        {"slug": f"retro-{uuid.uuid4().hex[:6]}", "name": "RetroRedact", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    canary = f"RETRO-SECRET-{uuid.uuid4().hex[:8]}"
    rule = client.studio_pii_request(
        "guard-rules",
        method="POST",
        payload={
            "rule_kind": "safe_word",
            "pattern_or_category": canary,
            "action": "warn",
            "surfaces": ["ingestion_gate"],
            "enabled": True,
        },
    ).assert_status((200, 201)).json()
    rule_id = extract_id(rule, "id", "rule_id")

    doc = client.create_document(
        bid, {"title": "RetroDoc", "text": f"본문 내 {canary} 포함", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")

    # Upgrade rule to block on ingestion_gate and retroactively redact existing data
    client.studio_pii_request(
        f"guard-rules/{rule_id}",
        method="PUT",
        payload={"action": "block", "surfaces": ["ingestion_gate", "inbound_query"], "enabled": True},
    ).assert_status((200, 204))

    openrag_cli.redact_run(
        pattern=canary,
        scope="pattern_org",
        redact_thumbnails=True,
        hard_purge=True,
        reason="retroactive safeword promotion",
        env={"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]},
        check=True,
    )

    chunk_txt = migrated_pg.execute_sql(
        f"SELECT string_agg(redacted_text, ' ') FROM corpus.chunks WHERE document_id = '{doc_id}';"
    )
    assert canary not in chunk_txt

    blocked_new = client.create_document(
        bid, {"title": "BlockedNew", "text": f"신규 {canary} 시도", "access_scope": "org"}
    )
    assert blocked_new.status in (400, 403, 422)
