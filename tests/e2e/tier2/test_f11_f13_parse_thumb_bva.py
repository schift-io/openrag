"""Tier 2 boundary, negative, and corner-case E2E tests for F11-F13."""

from __future__ import annotations

import math
import uuid
import xml.sax.saxutils as xml_escape
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError
from lib.fixtures import make_pipeline_yaml
from lib.pg_harness import PgEphemeralCluster

pytestmark = pytest.mark.tier2


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


def _create_bucket(api_client: Any, prefix: str = "f11bva") -> tuple[str, str]:
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


def _parse_pg_vector(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    text = str(raw).strip()
    return [float(tok) for tok in text[1:-1].split(",") if tok.strip()]


# ---------------------------------------------------------------------------
# F11 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f11
def test_f11_bva_corrupt_truncated_hwpx_returns_explicit_client_error(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-corrupt")
    resp = api_client.upload_document(
        bucket_id,
        filename="broken.hwpx",
        content=b"PK\x03\x04corrupt_hwpx_payload",
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    )
    assert resp.status in (400, 422), f"Expected 400/422, got {resp.status}: {resp.text}"


@pytest.mark.tier2
@pytest.mark.f11
def test_f11_bva_zip_missing_section0_xml_rejected(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-nosec")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", b"application/hwp+zip", compress_type=zipfile.ZIP_STORED)

    resp = api_client.upload_document(
        bucket_id,
        filename="incomplete.hwpx",
        content=buf.getvalue(),
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    )
    assert resp.status in (400, 422), f"Expected 400/422, got {resp.status}: {resp.text}"
    docs = api_client.list_documents(bucket_id).assert_status(200).json()
    assert "incomplete.hwpx" not in str(docs) or "indexed" not in str(docs).lower()


@pytest.mark.tier2
@pytest.mark.f11
def test_f11_bva_empty_hwpx_paragraphs_and_zero_cell_table(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-empty")
    hwpx = sample_factory.hwpx_bytes(title="", paragraphs=(), tables=[[]])
    resp = api_client.upload_document(
        bucket_id,
        filename="empty_table.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    )
    assert resp.status in (200, 201, 400, 422), f"Unexpected server error {resp.status}: {resp.text}"


@pytest.mark.tier2
@pytest.mark.f11
def test_f11_bva_hwpx_table_xss_and_colspan_rowspan_boundary(
    api_client: Any, mcp_stdio: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-xss")
    buf = BytesIO()
    xss_cell = xml_escape.escape('<script>alert(1)</script> & "비고"')
    sec_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
        'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">'
        "<hp:p><hp:run><hp:t>XSS 및 병합 셀 표 테스트</hp:t></hp:run></hp:p>"
        '<hp:tbl rowCnt="2" colCnt="2">'
        "<hp:tr>"
        f'<hp:tc colAddr="0" rowAddr="0" colSpan="2" rowSpan="1"><hp:subList><hp:p><hp:run><hp:t>{xss_cell}</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
        "</hp:tr>"
        "<hp:tr>"
        '<hp:tc colAddr="0" rowAddr="1" colSpan="1" rowSpan="2"><hp:subList><hp:p><hp:run><hp:t>셀A</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
        '<hp:tc colAddr="1" rowAddr="1" colSpan="1" rowSpan="1"><hp:subList><hp:p><hp:run><hp:t>셀B</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
        "</hp:tr>"
        "</hp:tbl></hs:sec>"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", b"application/hwp+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("Contents/section0.xml", sec_xml.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

    up = api_client.upload_document(
        bucket_id,
        filename="xss_span.hwpx",
        content=buf.getvalue(),
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        page_text = mcp.rag_get_page(document_id=doc_id, page_no=1).text
        assert "<script>" not in page_text.lower()
        assert "<table" in page_text.lower()
        assert "<tr" in page_text.lower()
        assert "<td" in page_text.lower()
        for disallowed in ("<th", "<thead", "<br", "<b>"):
            assert disallowed not in page_text.lower()


@pytest.mark.tier2
@pytest.mark.f11
def test_f11_bva_large_25_row_korean_table_hwpx_preserves_last_row(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-25row")
    rows = [[f"행{r}", f"부서{r}", f"항목{r}", f"값{r}"] for r in range(1, 25)]
    rows.append(["행25", "최종부서", "최종항목", "최종행-고유키-R25C4"])
    hwpx = sample_factory.hwpx_bytes(title="대용량 표", paragraphs=["25행 표 검증"], tables=[rows])

    api_client.upload_document(
        bucket_id,
        filename="large_table.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201))

    res = api_client.search_bucket(
        bucket_id, {"query": "최종행-고유키-R25C4", "top_k": 5}
    ).assert_status(200).json()
    assert "최종행-고유키-R25C4" in str(res)


@pytest.mark.tier2
@pytest.mark.f11
def test_f11_bva_cli_ingest_corrupt_hwp_exits_nonzero_with_stderr(
    openrag_cli: Any, sample_factory: Any
) -> None:
    bad_path = sample_factory.write_bytes(
        "bad.hwp", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1INVALID_OLE2_CORRUPT"
    )
    try:
        res = openrag_cli.ingest(bad_path, bucket="f11-bva")
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))
    assert res.returncode != 0
    assert (res.stderr or res.stdout).strip() != ""
    assert "panicked at" not in res.stderr.lower()


# ---------------------------------------------------------------------------
# F12 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f12
def test_f12_bva_zdr_plus_batch_conflict_returns_400_zdr_batch_unsupported(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f12-zdr")
    pdf = sample_factory.pdf_bytes(["ZDR Batch Conflict Test"])
    resp = api_client.upload_document(
        bucket_id,
        filename="zdr_batch.pdf",
        content=pdf,
        content_type="application/pdf",
        fields={"access_scope": "org", "mode": "zdr", "batch": "true"},
    )
    assert resp.status == 400, f"Expected 400 for zdr+batch, got {resp.status}: {resp.text}"
    assert "zdr_batch_unsupported" in resp.text.lower() or "batch" in resp.text.lower()


@pytest.mark.tier2
@pytest.mark.f12
def test_f12_bva_high_auth_level_l8_rejects_or_upgrades_share_mode(
    api_client: Any, openrag_cli: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f12-l8")
    pdf = sample_factory.pdf_bytes(["L8 Classified Share Reject"])
    resp = api_client.upload_document(
        bucket_id,
        filename="l8_share.pdf",
        content=pdf,
        content_type="application/pdf",
        fields={"access_scope": "org", "min_auth_level": "8", "sensitivity": "confidential", "mode": "share"},
    )
    assert resp.status in (200, 201, 400, 403, 422)
    if resp.status in (200, 201):
        data = resp.json()
        eff_mode = str(data.get("ocr_mode") or data.get("mode") or "").lower()
        assert eff_mode != "share", f"L8 document must never execute OCR in 'share' mode: {data}"

    pdf_path = sample_factory.pdf_file("l8_cli.pdf", ["L8 CLI Check"])
    try:
        cli_res = openrag_cli.ingest(
            pdf_path,
            bucket="f12-l8-cli",
            auth_level=8,
            mode="share",
            env={"DATABASE_URL": migrated_pg.dsn()},
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))
    if cli_res.returncode == 0:
        assert '"mode": "share"' not in cli_res.stdout.lower()


@pytest.mark.tier2
@pytest.mark.f12
def test_f12_bva_auth_level_boundary_3_vs_4_ocr_mode_transition(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pipeline_update({"pipeline_yaml": make_pipeline_yaml(profile="standard")}).assert_status(200)

    r3 = api_client.studio_pipeline_test({"query": "ocr", "auth_level": 3}).assert_status(200).json()
    r4 = api_client.studio_pipeline_test({"query": "ocr", "auth_level": 4}).assert_status(200).json()
    r7 = api_client.studio_pipeline_test({"query": "ocr", "auth_level": 7}).assert_status(200).json()

    assert "share" in str(r3).lower()
    if "valid" in r3:
        assert r3["valid"] is True
    assert "private" in str(r4).lower() or "local" in str(r4).lower()
    assert "zdr" in str(r7).lower() or "local" in str(r7).lower()


@pytest.mark.tier2
@pytest.mark.f12
def test_f12_bva_corrupt_pdf_bytes_rejected_without_500(
    api_client: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f12-badpdf")
    resp = api_client.upload_document(
        bucket_id,
        filename="corrupt.pdf",
        content=b"%PDF-1.4\n1 0 obj\n<< /Broken >>\n",
        content_type="application/pdf",
        fields={"access_scope": "org"},
    )
    assert resp.status in (400, 422), f"Expected 400/422 for corrupt PDF, got {resp.status}: {resp.text}"


@pytest.mark.tier2
@pytest.mark.f12
def test_f12_bva_zero_byte_and_1x1_png_boundary(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f12-zero")
    r_empty = api_client.upload_document(
        bucket_id,
        filename="empty.png",
        content=b"",
        content_type="image/png",
        fields={"access_scope": "org"},
    )
    assert r_empty.status in (400, 422)

    r_1x1 = api_client.upload_document(
        bucket_id,
        filename="tiny.png",
        content=sample_factory.png_bytes(1, 1),
        content_type="image/png",
        fields={"access_scope": "org"},
    )
    assert r_1x1.status in (200, 201, 400, 422)


# ---------------------------------------------------------------------------
# F13 BVA / Negative
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f13
def test_f13_bva_unauthorized_peer_blocked_from_private_page_thumbnail(
    api_client: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    org_id = str(uuid.uuid4())
    dept_id = str(uuid.uuid4())
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())

    api_client.studio_iam_request(
        "orgs", method="POST", payload={"id": org_id, "slug": f"f13o-{org_id[:6]}", "name": "F13 Org"}
    ).assert_status((200, 201, 409))
    for pid in (user_a, user_b):
        api_client.studio_iam_request(
            "principals",
            method="POST",
            payload={
                "id": pid,
                "org_id": org_id,
                "email": f"{pid[:8]}@example.or.kr",
                "display_name": pid[:8],
                "auth_level": 5,
                "dept_ids": [dept_id],
            },
        ).assert_status((200, 201, 409))

    key_a = str(
        api_client.studio_iam_request(
            "api-keys", method="POST", payload={"org_id": org_id, "principal_id": user_a, "name": "ka", "max_auth_level": 5}
        ).assert_status((200, 201)).json().get("token")
    )
    key_b = str(
        api_client.studio_iam_request(
            "api-keys", method="POST", payload={"org_id": org_id, "principal_id": user_b, "name": "kb", "max_auth_level": 5}
        ).assert_status((200, 201)).json().get("token")
    )

    client_a = api_client.with_api_key(key_a)
    bkt = client_a.create_bucket(
        {"slug": f"f13-priv-{uuid.uuid4().hex[:6]}", "name": "F13 Priv", "org_id": org_id, "dept_id": dept_id, "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bkt_id = str(bkt.get("id") or bkt.get("bucket_id"))

    up = client_a.upload_document(
        bkt_id,
        filename="priv.pdf",
        content=sample_factory.pdf_bytes(["Private Thumbnail Test"]),
        content_type="application/pdf",
        fields={"access_scope": "private"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)
    page_id = _extract_page_id(up)
    if not page_id:
        with psycopg.connect(migrated_pg.dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id::text FROM corpus.pages WHERE document_id = %s::uuid LIMIT 1", (doc_id,))
                row = cur.fetchone()
                page_id = row[0] if row else None
    assert page_id

    resp_b = api_client.with_api_key(key_b).app_page_thumbnail(page_id)
    assert resp_b.status in (403, 404)
    assert not resp_b.body.startswith(b"RIFF")
    assert not resp_b.body.startswith(b"\x89PNG")


@pytest.mark.tier2
@pytest.mark.f13
def test_f13_bva_nonexistent_page_id_thumbnail_and_mcp_get_page(
    api_client: Any, mcp_stdio: Any, migrated_pg: PgEphemeralCluster
) -> None:
    fake_id = "00000000-0000-0000-0000-000000000099"
    resp = api_client.app_page_thumbnail(fake_id)
    assert resp.status in (400, 404)

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        res = mcp.rag_get_page(page_id=fake_id)
        assert res.is_error is True or "not found" in res.text.lower()


@pytest.mark.tier2
@pytest.mark.f13
def test_f13_bva_bbox_norm_1000_bounds_stay_within_0_to_999(
    api_client: Any, mcp_stdio: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f13-bounds")
    png = sample_factory.png_bytes(500, 500, boxes=[((0, 0, 999, 999), (0, 0, 255))])
    up = api_client.upload_document(
        bucket_id,
        filename="edge_box.png",
        content=png,
        content_type="image/png",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bbox_norm_1000 FROM corpus.page_figures WHERE document_id = %s::uuid",
                (doc_id,),
            )
            for (bbox,) in cur.fetchall():
                if bbox is not None:
                    vals = [int(v) for v in bbox]
                    assert 0 <= vals[0] < vals[2] <= 999
                    assert 0 <= vals[1] < vals[3] <= 999

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        pj = mcp.rag_get_page(document_id=doc_id, page_no=1).json()
        for item in (pj.get("boxes") or pj.get("figures") or []):
            if isinstance(item, dict) and item.get("bbox_norm_1000"):
                x0, y0, x1, y1 = [int(v) for v in item["bbox_norm_1000"]]
                assert 0 <= x0 < x1 <= 999
                assert 0 <= y0 < y1 <= 999


@pytest.mark.tier2
@pytest.mark.f13
def test_f13_bva_128d_visual_embedding_norm_and_no_nan_inf(
    openrag_cli: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    blank_path = sample_factory.write_bytes(
        "blank.png", sample_factory.png_bytes(100, 100, bg_rgb=(255, 255, 255))
    )
    try:
        openrag_cli.ingest(
            blank_path,
            bucket=f"f13-blank-{uuid.uuid4().hex[:6]}",
            scope="org",
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.thumb_embedding::text
                FROM corpus.pages p
                JOIN corpus.documents d ON d.id = p.document_id
                WHERE d.title LIKE '%blank%' OR COALESCE(d.source_uri, '') LIKE '%blank%'
                """
            )
            rows = cur.fetchall()
            assert rows
            for (raw_vec,) in rows:
                vec = _parse_pg_vector(raw_vec)
                assert len(vec) == 128
                assert all(math.isfinite(v) for v in vec)
                norm = math.sqrt(sum(v * v for v in vec))
                assert norm > 0.0
                assert abs(norm - 1.0) < 1e-3, f"Expected L2-normalized 128d thumb_embedding, got norm={norm}"


@pytest.mark.tier2
@pytest.mark.f13
def test_f13_bva_hwpx_thumbnail_generation_without_external_kordoc_cli(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f13-hwpx-thumb")
    hwpx = sample_factory.hwpx_bytes(title="한글 썸네일 테스트", paragraphs=["외부 CLI 없이 생성"])
    up = api_client.upload_document(
        bucket_id,
        filename="no_kordoc.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
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

    thumb = api_client.app_page_thumbnail(page_id).assert_status(200)
    assert len(thumb.body) > 0
    assert (thumb.body[:4] == b"RIFF" and thumb.body[8:12] == b"WEBP") or thumb.body.startswith(
        b"\x89PNG"
    )
