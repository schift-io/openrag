"""Tier 4 Scenarios 1-5: Ministry Classified HWPX, TF Groups/Deny, PII Capsule, Guard, Cascade Redaction."""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest

from conftest import fail_missing_target
from lib import (
    CliBinaryNotFoundError,
    FixtureFactory,
    McpHttpClient,
    OpenRagCli,
    OpenRagHttpClient,
    PgEphemeralCluster,
    PgPrincipalContext,
)
from tier4._helpers import (
    extract_id,
    migrated_pg,  # noqa: F401 - pytest fixture
    mint_api_key,
    seed_institution_org,
)

pytestmark = [pytest.mark.tier4]


@pytest.mark.f07
@pytest.mark.f09
@pytest.mark.f11
@pytest.mark.f12
@pytest.mark.f25
@pytest.mark.f28
def test_s01_gov_ministry_classified_hwpx_dept_admin_vs_peer_and_l8_vllm(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S01")
    owner_key = mint_api_key(api_client, ctx["org_id"], ctx["owner_id"], tag="s01-own")
    peer_key = mint_api_key(api_client, ctx["org_id"], ctx["peer_id"], tag="s01-peer")
    dadmin_key = mint_api_key(api_client, ctx["org_id"], ctx["dept_admin_id"], tag="s01-dadm")
    low_key = mint_api_key(api_client, ctx["org_id"], ctx["low_id"], tag="s01-low")

    owner_client = api_client.with_api_key(owner_key)
    bslug = f"ministry-{uuid.uuid4().hex[:6]}"
    b = owner_client.create_bucket(
        {"slug": bslug, "name": "ClassifiedPolicy", "access_scope": "dept", "min_auth_level": 1}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    hwpx_path = sample_factory.hwpx_file(
        "classified_budget.hwpx",
        title="기밀 예산 배정표",
        paragraphs=["외부 반출 금지 대외비 문서"],
        tables=[[["등급", "예산"], ["대외비", "42억원_S01_SECRET"]]],
    )
    # ZDR + batch=true must return HTTP 400
    zdr_batch = owner_client.upload_document(
        bid,
        filename="zdr_batch.pdf",
        content=sample_factory.pdf_bytes(["L8 ZDR"]),
        content_type="application/pdf",
        fields={"access_scope": "private", "min_auth_level": "8", "ocr_mode": "zdr", "batch": "true"},
    )
    assert zdr_batch.status == 400

    # Ingest classified HWPX via CLI requesting share mode on L8 -> must enforce private/zdr (never share)
    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": owner_key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.ingest(
        hwpx_path,
        bucket=bslug,
        scope="private",
        auth_level=8,
        mode="share",
        env=env,
        check=True,
    )

    doc_row = migrated_pg.execute_sql(
        f"SELECT id::text || '|' || sensitivity FROM corpus.documents WHERE bucket_id = '{bid}' ORDER BY created_at DESC LIMIT 1;"
    ).strip()
    assert doc_row and "|" in doc_row
    doc_id, sensitivity = doc_row.split("|", 1)
    assert sensitivity != "share", f"L8 document must never stay in share OCR mode: {sensitivity}"

    tbl_html = migrated_pg.execute_sql(
        f"SELECT COALESCE(string_agg(table_html, ' '), '') FROM corpus.chunks WHERE document_id = '{doc_id}';"
    )
    assert "<table" in tbl_html.lower() and "<td" in tbl_html.lower()
    assert "42억원_S01_SECRET" in tbl_html

    # PG17 RLS checks: owner & dept_admin allowed, same-level peer(L8) & low(L3) strictly blocked
    for allowed_pid, is_dadmin in ((ctx["owner_id"], False), (ctx["dept_admin_id"], True)):
        rows = migrated_pg.query_as_principal(
            "SELECT id::text FROM corpus.documents WHERE id = %s",
            principal=PgPrincipalContext(
                org_id=ctx["org_id"],
                principal_id=allowed_pid,
                auth_level=8,
                dept_ids=(ctx["dept_a_id"],),
                dept_admin_ids=(ctx["dept_a_id"],) if is_dadmin else (),
            ),
            params=(doc_id,),
        )
        assert len(rows) == 1

    for denied_pid, lvl in ((ctx["peer_id"], 8), (ctx["low_id"], 3)):
        rows = migrated_pg.query_as_principal(
            "SELECT id::text FROM corpus.documents WHERE id = %s",
            principal=PgPrincipalContext(
                org_id=ctx["org_id"],
                principal_id=denied_pid,
                auth_level=lvl,
                dept_ids=(ctx["dept_a_id"],),
            ),
            params=(doc_id,),
        )
        assert rows == []

    # HTTP search: owner & dept_admin succeed; same-level peer & low_member get zero leak
    assert "42억원_S01_SECRET" in owner_client.search_bucket(bid, {"query": "42억원_S01_SECRET", "top_k": 5}).assert_status(200).text
    assert "42억원_S01_SECRET" in api_client.with_api_key(dadmin_key).search_bucket(bid, {"query": "42억원_S01_SECRET", "top_k": 5}).assert_status(200).text
    for dkey in (peer_key, low_key):
        s_den = api_client.with_api_key(dkey).search_bucket(bid, {"query": "42억원_S01_SECRET", "top_k": 5})
        assert s_den.status in (200, 403) and "42억원_S01_SECRET" not in s_den.text

    # L8+ local vLLM routing enforcement
    chat = owner_client.chat_completions(
        {
            "model": "openrag-auto",
            "bucket_id": bid,
            "include_citations": True,
            "messages": [{"role": "user", "content": "대외비 예산 42억원_S01_SECRET 요약해줘"}],
        }
    ).assert_status(200).json()
    assert chat["choices"][0]["message"]["content"]

    route_log = migrated_pg.execute_sql(
        f"""
        SELECT p.provider_type || '|' || p.is_on_premise::text
        FROM gateway.request_logs l
        JOIN gateway.providers p ON p.id = l.provider_id
        WHERE l.org_id = '{ctx['org_id']}'
        ORDER BY l.created_at DESC
        LIMIT 1;
        """
    ).strip()
    assert route_log == "vllm_local|true", route_log


@pytest.mark.f08
@pytest.mark.f09
@pytest.mark.f10
@pytest.mark.f24
@pytest.mark.f28
def test_s02_cross_dept_tf_permission_group_share_and_explicit_deny_override(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
) -> None:
    ctx = seed_institution_org(migrated_pg, label="S02")
    denied_member_id = str(uuid.uuid4())
    migrated_pg.execute_sql(
        f"""
        INSERT INTO iam.principals (id, org_id, email, display_name, auth_level)
        VALUES ('{denied_member_id}', '{ctx['org_id']}', 'den-{denied_member_id[:6]}@gov.or.kr', 'DeniedL7', 7);
        INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
        VALUES ('{ctx['org_id']}', '{ctx['dept_b_id']}', '{denied_member_id}', 'member');
        """
    )

    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s02-adm")
    legal_key = mint_api_key(api_client, ctx["org_id"], ctx["other_dept_id"], tag="s02-leg")
    low_key = mint_api_key(api_client, ctx["org_id"], ctx["low_id"], tag="s02-low")
    denied_key = mint_api_key(api_client, ctx["org_id"], denied_member_id, tag="s02-den")
    admin_client = api_client.with_api_key(admin_key)

    tf_allow = extract_id(
        admin_client.studio_iam_request(
            "groups",
            method="POST",
            payload={
                "name": f"tf-allow-{uuid.uuid4().hex[:6]}",
                "group_type": "project_tf",
                "min_auth_level": 5,
                "allowed_actions": ["read", "search"],
                "member_ids": [ctx["owner_id"], ctx["other_dept_id"], ctx["low_id"], denied_member_id],
            },
        ).assert_status((200, 201)).json(),
        "id",
        "group_id",
    )
    tf_deny = extract_id(
        admin_client.studio_iam_request(
            "groups",
            method="POST",
            payload={
                "name": f"tf-deny-{uuid.uuid4().hex[:6]}",
                "group_type": "permission_group",
                "min_auth_level": 1,
                "allowed_actions": ["read", "search"],
                "member_ids": [denied_member_id],
            },
        ).assert_status((200, 201)).json(),
        "id",
        "group_id",
    )

    b = admin_client.create_bucket(
        {"slug": f"tf-s02-{uuid.uuid4().hex[:6]}", "name": "TFShare", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")
    doc = admin_client.create_document(
        bid,
        {
            "title": "TF Orchid Budget",
            "text": "TF_ORCHID_2026_BUDGET 교차부서 프로젝트 문서",
            "access_scope": "group",
            "allowed_group_ids": [tf_allow],
            "denied_group_ids": [tf_deny],
            "min_auth_level": 5,
        },
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")

    sim_ok = admin_client.studio_iam_simulate(
        {"principal_id": ctx["other_dept_id"], "resource_type": "document", "resource_id": doc_id, "action": "read"}
    ).assert_status(200).json()
    assert sim_ok.get("allowed") is True
    assert sim_ok.get("matched_allows")

    for bad_pid in (denied_member_id, ctx["low_id"], ctx["peer_id"]):
        sim_no = admin_client.studio_iam_simulate(
            {"principal_id": bad_pid, "resource_type": "document", "resource_id": doc_id, "action": "read"}
        ).assert_status(200).json()
        assert sim_no.get("allowed") is False
        if bad_pid == denied_member_id:
            assert sim_no.get("matched_denies")

    assert "TF_ORCHID_2026_BUDGET" in api_client.with_api_key(legal_key).search_bucket(
        bid, {"query": "TF_ORCHID_2026_BUDGET", "top_k": 5}
    ).assert_status(200).text

    for bad_key in (denied_key, low_key):
        s_bad = api_client.with_api_key(bad_key).search_bucket(bid, {"query": "TF_ORCHID_2026_BUDGET", "top_k": 5})
        assert s_bad.status in (200, 403) and "TF_ORCHID_2026_BUDGET" not in s_bad.text

    rls_denied = migrated_pg.query_as_principal(
        "SELECT id::text FROM corpus.documents WHERE id = %s",
        principal=PgPrincipalContext(
            org_id=ctx["org_id"],
            principal_id=denied_member_id,
            auth_level=7,
            dept_ids=(ctx["dept_b_id"],),
            group_ids=(tf_allow, tf_deny),
        ),
        params=(doc_id,),
    )
    assert rls_denied == []


@pytest.mark.f14
@pytest.mark.f15
@pytest.mark.f25
@pytest.mark.f28
def test_s03_external_llm_pii_capsule_custom_dict_and_unmask_authorization(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    ctx = seed_institution_org(migrated_pg, label="S03")
    auth_key = mint_api_key(api_client, ctx["org_id"], ctx["dept_admin_id"], tag="s03-auth")
    unauth_key = mint_api_key(api_client, ctx["org_id"], ctx["other_dept_id"], tag="s03-unauth")
    admin_key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s03-adm")
    admin_client = api_client.with_api_key(admin_key)

    admin_client.studio_pii_request(
        "profiles",
        method="PUT",
        payload={
            "name": "default",
            "use_schift_v7": True,
            "use_marker_ko_pii": True,
            "custom_dict_terms": ["내부감사코드-S03-99"],
        },
    ).assert_status((200, 201))

    b = admin_client.create_bucket(
        {"slug": f"s03-pii-{uuid.uuid4().hex[:6]}", "name": "CapsuleS03", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    pii_doc = sample_factory.korean_pii_doc(safe_words=())
    doc = admin_client.create_document(
        bid,
        {
            "title": "감사 대상자 명단",
            "text": pii_doc["text"] + "\n특수코드: 내부감사코드-S03-99",
            "access_scope": "org",
            "min_auth_level": 3,
        },
    ).assert_status((200, 201)).json()
    doc_id = extract_id(doc, "document_id", "id", "doc_id")

    chunks_txt = migrated_pg.execute_sql(
        f"SELECT string_agg(redacted_text, ' ') FROM corpus.chunks WHERE document_id = '{doc_id}';"
    )
    assert "900101-1234567" not in chunks_txt
    assert "010-1234-5678" not in chunks_txt
    assert "내부감사코드-S03-99" not in chunks_txt

    vault_keys = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.vault_keys WHERE org_id = '{ctx['org_id']}' AND algorithm = 'AES-256-GCM';"
        ).strip()
    )
    vault_entries = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.vault_entries WHERE org_id = '{ctx['org_id']}' AND octet_length(ciphertext) > 0;"
        ).strip()
    )
    assert vault_keys >= 1 and vault_entries >= 1

    ext_resp = api_client.with_api_key(unauth_key).chat_completions(
        {
            "model": "openrouter/default",
            "bucket_id": bid,
            "unmask_pii": True,
            "messages": [{"role": "user", "content": "감사 대상자 연락처와 주민등록번호 알려줘"}],
        }
    ).assert_status(200)
    assert "900101-1234567" not in ext_resp.text and "010-1234-5678" not in ext_resp.text

    auth_resp = api_client.with_api_key(auth_key).chat_completions(
        {
            "model": "vllm/local",
            "bucket_id": bid,
            "unmask_pii": True,
            "messages": [{"role": "user", "content": "감사 대상자 주민등록번호와 연락처 복원해줘"}],
        }
    ).assert_status(200)
    assert "900101-1234567" in auth_resp.text or "010-1234-5678" in auth_resp.text

    audit_resp = admin_client.studio_pii_request("audit-logs").assert_status(200).text
    assert "900101-1234567" not in audit_resp


@pytest.mark.f16
@pytest.mark.f28
@pytest.mark.f29
@pytest.mark.f34
def test_s04_configurable_safeword_and_pii_guard_4_actions_allowlist_and_master_toggle(
    migrated_pg: PgEphemeralCluster,
    api_client: OpenRagHttpClient,
    mcp_stdio: Any,
) -> None:
    ctx = seed_institution_org(migrated_pg, label="S04")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s04")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"s04-g-{uuid.uuid4().hex[:6]}", "name": "GuardS04", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    created_rules: list[str] = []
    for kind, pat, action in (
        ("safe_word", "PROJECT-ORION", "block"),
        ("pii_category", "phone_kr", "mask"),
        ("keyword", "INTERNAL-DRAFT-S04", "warn"),
        ("keyword", "PUBLIC-BULLETIN-S04", "allow"),
    ):
        r = client.studio_pii_request(
            "guard-rules",
            method="POST",
            payload={
                "rule_kind": kind,
                "pattern_or_category": pat,
                "action": action,
                "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"],
                "enabled": True,
            },
        ).assert_status((200, 201)).json()
        created_rules.append(extract_id(r, "id", "rule_id"))

    policy_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM pii.guard_policies WHERE org_id = '{ctx['org_id']}' AND is_active = true LIMIT 1;"
    ).strip()
    assert policy_id

    try:
        # ingestion_gate block
        ing_blk = client.create_document(
            bid, {"title": "OrionBlocked", "text": "비밀 PROJECT-ORION 문건", "access_scope": "org"}
        )
        assert ing_blk.status in (400, 403, 422)
        cnt_blk = int(
            migrated_pg.execute_sql(
                f"SELECT count(*) FROM corpus.documents WHERE org_id = '{ctx['org_id']}' AND title = 'OrionBlocked';"
            ).strip()
        )
        assert cnt_blk == 0

        # HTTP + MCP 4 actions
        chat_blk = client.chat_completions(
            {"model": "openrag-auto", "messages": [{"role": "user", "content": "PROJECT-ORION 보고"}]}
        )
        assert chat_blk.status == 403 and "guard_blocked" in chat_blk.text.lower()

        with mcp_stdio(api_key=key, principal_id=ctx["org_admin_id"], env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
            mcp.handshake()
            g_blk = mcp.guard_check("PROJECT-ORION 자료", surface="mcp_egress")
            assert g_blk.is_error is True or g_blk.json().get("decision") == "block"

            g_mask = mcp.guard_check("담당 010-1234-5678", surface="mcp_egress")
            assert g_mask.is_error is False and g_mask.json().get("decision") == "mask"
            assert "010-1234-5678" not in g_mask.json().get("sanitized_text", "")

            g_warn = mcp.guard_check("INTERNAL-DRAFT-S04 초안", surface="mcp_egress")
            assert g_warn.is_error is False and g_warn.json().get("decision") == "warn"

            g_allow = mcp.guard_check("PUBLIC-BULLETIN-S04 공지", surface="mcp_egress")
            assert g_allow.is_error is False and g_allow.json().get("decision") == "allow"

        # Allowlist term bypasses block while base safe-word stays blocked
        client.studio_pii_request(
            f"guard-policies/{policy_id}",
            method="PUT",
            payload={"allowlist_terms": ["PROJECT-ORION-DEMO"], "master_enabled": True},
        ).assert_status((200, 204))
        demo_res = client.studio_guard_test({"text": "PROJECT-ORION-DEMO 공개 시연", "surface": "inbound_query"}).assert_status(200).json()
        assert demo_res.get("decision") != "block"

        # Master toggle OFF
        client.studio_pii_request(
            f"guard-policies/{policy_id}",
            method="PUT",
            payload={"master_enabled": False},
        ).assert_status((200, 204))
        off_res = client.studio_guard_test({"text": "PROJECT-ORION", "surface": "inbound_query"}).assert_status(200).json()
        assert off_res.get("decision") == "allow"
    finally:
        client.studio_pii_request(
            f"guard-policies/{policy_id}",
            method="PUT",
            payload={"master_enabled": True, "allowlist_terms": []},
        )
        for rid in created_rules:
            client.studio_pii_request(f"guard-rules/{rid}", method="DELETE")

    ev_meta = migrated_pg.execute_sql(
        f"SELECT COALESCE(string_agg(metadata::text, ' '), '') FROM pii.guard_events WHERE org_id = '{ctx['org_id']}';"
    )
    assert "PROJECT-ORION" not in ev_meta


@pytest.mark.f13
@pytest.mark.f17
@pytest.mark.f23
@pytest.mark.f28
def test_s05_full_layer_cascade_redaction_hard_purge_bbox_blackout_and_reembed(
    migrated_pg: PgEphemeralCluster,
    openrag_cli: OpenRagCli,
    api_client: OpenRagHttpClient,
    sample_factory: FixtureFactory,
) -> None:
    try:
        openrag_cli.require_binary("openrag")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    ctx = seed_institution_org(migrated_pg, label="S05")
    key = mint_api_key(api_client, ctx["org_id"], ctx["org_admin_id"], tag="s05")
    client = api_client.with_api_key(key)
    b = client.create_bucket(
        {"slug": f"s05-red-{uuid.uuid4().hex[:6]}", "name": "CascadeS05", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bid = extract_id(b, "id", "bucket_id")

    canary = f"CANARY-S05-{uuid.uuid4().hex[:6]}"
    bbox = (120, 120, 680, 380)
    png_bytes = sample_factory.png_bytes(400, 400, bg_rgb=(250, 250, 250), boxes=[(bbox, (230, 80, 40))])
    up_img = client.upload_document(
        bid,
        filename="s05_scan.png",
        content=png_bytes,
        content_type="image/png",
        fields={"access_scope": "org", "min_auth_level": "3"},
    ).assert_status((200, 201, 202)).json()
    img_doc_id = extract_id(up_img, "document_id", "id", "doc_id")

    client.create_document(
        bid, {"title": "CanaryText", "text": f"본문 내 {canary} 및 900101-1234567", "access_scope": "org"}
    ).assert_status((200, 201))

    mcp = McpHttpClient(client)
    mcp.handshake()
    mem_id = extract_id(mcp.memory_reflect(f"기억 내 {canary}", scope="org").json(), "memory_id", "id")

    page_id = migrated_pg.execute_sql(
        f"SELECT id::text FROM corpus.pages WHERE document_id = '{img_doc_id}' LIMIT 1;"
    ).strip()
    assert page_id
    before_thumb = client.app_page_thumbnail(page_id).assert_status(200).body
    assert sample_factory.inspect_bbox_blackout(before_thumb, bbox).is_blacked_out is False
    before_vec = migrated_pg.execute_sql(
        f"SELECT thumb_embedding::text FROM corpus.pages WHERE id = '{page_id}';"
    ).strip()

    env = {"DATABASE_URL": migrated_pg.dsn(), "OPENRAG_API_KEY": key, "OPENRAG_ORG_ID": ctx["org_id"]}
    openrag_cli.redact_run(
        pattern=canary,
        scope="pattern_org",
        redact_thumbnails=True,
        hard_purge=True,
        reason="s05 pattern purge",
        env=env,
        check=True,
    )
    client.studio_pii_request(
        "redactions",
        method="POST",
        payload={
            "target_scope": "document",
            "target_ref": {"document_id": img_doc_id, "bbox_norm_1000": list(bbox)},
            "redaction_mode": "hard_purge",
            "redact_thumbnails": True,
            "reason": "s05 bbox purge",
        },
    ).assert_status((200, 201, 202))

    assert canary not in migrated_pg.execute_sql(
        f"SELECT COALESCE(string_agg(redacted_text, ' '), '') FROM corpus.chunks WHERE org_id = '{ctx['org_id']}';"
    )
    assert canary not in migrated_pg.execute_sql(
        f"SELECT effective_text FROM memory.entries WHERE id = '{mem_id}';"
    )
    after_thumb = client.app_page_thumbnail(page_id).assert_status(200).body
    assert sample_factory.inspect_bbox_blackout(after_thumb, bbox, min_black_ratio=0.90).is_blacked_out is True

    after_vec_row = migrated_pg.execute_sql(
        f"SELECT thumb_embedding::text FROM corpus.pages WHERE id = '{page_id}';"
    ).strip()
    assert after_vec_row and after_vec_row != before_vec
    vec_vals = [float(x) for x in after_vec_row.strip("[]").split(",")]
    assert len(vec_vals) == 128
    norm = math.sqrt(sum(v * v for v in vec_vals))
    assert abs(norm - 1.0) < 1e-3

    outbox_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM indexing.outbox WHERE org_id = '{ctx['org_id']}' AND op IN ('redact_reindex', 'delete_vector');"
        ).strip()
    )
    red_ev_cnt = int(
        migrated_pg.execute_sql(
            f"SELECT count(*) FROM pii.redaction_events WHERE org_id = '{ctx['org_id']}';"
        ).strip()
    )
    assert outbox_cnt >= 1 and red_ev_cnt >= 1
