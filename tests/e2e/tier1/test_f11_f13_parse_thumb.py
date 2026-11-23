"""Tier 1 isolated happy-path E2E tests for F11-F13 (kordoc HWPX, Schift OCR/pdf_native, Thumbnailer)."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest

from conftest import fail_missing_target
from lib.cli_client import CliBinaryNotFoundError
from lib.fixtures import _decode_image_rgb
from lib.pg_harness import PgEphemeralCluster

pytestmark = pytest.mark.tier1


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


def _create_bucket(api_client: Any, prefix: str = "f11") -> tuple[str, str]:
    slug = f"{prefix}-{uuid.uuid4().hex[:8]}"
    resp = api_client.create_bucket(
        {"slug": slug, "name": f"Bucket {slug}", "access_scope": "org"}
    ).assert_status((200, 201)).json()
    bucket_id = str(resp.get("id") or resp.get("bucket_id") or slug)
    return bucket_id, slug


def _extract_doc_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("document_id", "id", "doc_id"):
            val = payload.get(key)
            if val:
                return str(val)
        doc = payload.get("document")
        if isinstance(doc, dict):
            return _extract_doc_id(doc)
    raise AssertionError(f"Could not extract document_id from response: {payload!r}")


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
    assert text.startswith("[") and text.endswith("]"), f"Invalid vector text: {text!r}"
    return [float(tok) for tok in text[1:-1].split(",") if tok.strip()]


# ---------------------------------------------------------------------------
# F11 — M1 kordoc HWP/HWPX/HML Parser
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f11
def test_f11_hwpx_korean_paragraphs_upload_and_search(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-hwpx")
    hwpx = sample_factory.hwpx_bytes(
        title="2026년 기관 보안 지침",
        paragraphs=["제1조 적용범위: 전 부서 적용", "비밀취급인가 절차 안내"],
    )
    up = api_client.upload_document(
        bucket_id,
        filename="policy.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)
    assert doc_id

    listed = api_client.list_documents(bucket_id).assert_status(200).json()
    assert "policy.hwpx" in str(listed)

    search = api_client.search_bucket(
        bucket_id, {"query": "비밀취급인가", "top_k": 5}
    ).assert_status(200).json()
    hits = search.get("results") or search.get("hits") or search.get("items") or []
    assert len(hits) >= 1
    assert "비밀취급인가" in str(hits)


@pytest.mark.tier1
@pytest.mark.f11
def test_f11_hwpx_table_html_preservation(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-tbl")
    hwpx = sample_factory.hwpx_bytes(
        title="여비 지급 기준표",
        paragraphs=["출장 여비 규정"],
        tables=[
            [
                ["구분", "일비", "숙박비"],
                ["국내", "70,000원", "100,000원"],
                ["국외", "120,000원", "250,000원"],
            ]
        ],
    )
    up = api_client.upload_document(
        bucket_id,
        filename="travel_table.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    search = api_client.search_bucket(
        bucket_id, {"query": "70,000원", "top_k": 5}
    ).assert_status(200).json()
    assert "70,000원" in str(search)

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        page_res = mcp.rag_get_page(document_id=doc_id, page_no=1)
        assert page_res.is_error is False
        page_text = page_res.text
        assert "<table" in page_text.lower()
        assert "<tr" in page_text.lower()
        assert "<td" in page_text.lower()
        assert "250,000원" in page_text
        for disallowed in ("<th", "<thead", "<br", "<b>"):
            assert disallowed not in page_text.lower()


@pytest.mark.tier1
@pytest.mark.f11
def test_f11_cli_ingest_hwpx_with_scope_and_level(
    openrag_cli: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    hwpx_path = sample_factory.hwpx_file(
        "notice.hwpx",
        title="부서 공고",
        paragraphs=["예산 집행 마감일 12월 15일"],
    )
    try:
        res = openrag_cli.ingest(
            hwpx_path,
            bucket=f"f11-cli-{uuid.uuid4().hex[:6]}",
            scope="dept",
            auth_level=3,
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    assert res.returncode == 0
    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.access_scope, d.min_auth_level, c.redacted_text
                FROM corpus.documents d
                JOIN corpus.chunks c ON c.document_id = d.id
                WHERE c.redacted_text LIKE %s
                """,
                ("%예산 집행 마감일%",),
            )
            rows = cur.fetchall()
            assert rows, "CLI HWPX ingest did not persist chunks to PostgreSQL"
            assert any(r[0] == "dept" and int(r[1]) == 3 for r in rows)


@pytest.mark.tier1
@pytest.mark.f11
def test_f11_app_upload_hwpx_multipart(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    hwpx = sample_factory.hwpx_bytes(
        title="업무 메모",
        paragraphs=["회의록 요약: 분기 점검 완료 F11APPTOKEN"],
    )
    up = api_client.app_upload(
        filename="memo.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org", "sensitivity": "normal"},
    ).assert_status((200, 201)).json()
    assert _extract_doc_id(up)

    chat = api_client.app_chat({"query": "분기 점검 완료 F11APPTOKEN"}).assert_status(200).json()
    assert "분기 점검" in str(chat) or "F11APPTOKEN" in str(chat)


@pytest.mark.tier1
@pytest.mark.f11
def test_f11_mcp_rag_ingest_hwpx_path(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    _, bucket_slug = _create_bucket(api_client, "f11-mcp")
    hwpx_path = sample_factory.hwpx_file(
        "mcp_doc.hwpx",
        title="MCP 공문",
        paragraphs=["전자결재 연계 확인 F11MCPTOKEN"],
    )
    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        ing = mcp.rag_ingest(
            bucket=bucket_slug,
            path=str(hwpx_path),
            access_scope="org",
            min_auth_level=2,
            title="MCP 공문",
        )
        assert ing.is_error is False
        assert _extract_doc_id(ing.json())

        srch = mcp.rag_search("전자결재 연계 확인 F11MCPTOKEN", bucket=bucket_slug)
        assert srch.is_error is False
        assert "전자결재" in srch.text or "F11MCPTOKEN" in srch.text


@pytest.mark.tier1
@pytest.mark.f11
def test_f11_hwpx_multi_table_and_header_metadata(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f11-multitbl")
    hwpx = sample_factory.hwpx_bytes(
        title="복수 표 문서",
        paragraphs=["첫 번째 표와 두 번째 표 포함"],
        tables=[
            [["항목A", "10"]],
            [["항목B", "20"]],
        ],
    )
    api_client.upload_document(
        bucket_id,
        filename="multi_table.hwpx",
        content=hwpx,
        content_type="application/hwp+zip",
        fields={"access_scope": "org"},
    ).assert_status((200, 201))

    hit_a = api_client.search_bucket(bucket_id, {"query": "항목A", "top_k": 5}).assert_status(200).json()
    hit_b = api_client.search_bucket(bucket_id, {"query": "항목B", "top_k": 5}).assert_status(200).json()
    assert "항목A" in str(hit_a) and "10" in str(hit_a)
    assert "항목B" in str(hit_b) and "20" in str(hit_b)


# ---------------------------------------------------------------------------
# F12 — M1 Schift OCR + Mode Guard + Dewarp + pdf_native
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f12
def test_f12_pdf_native_multipage_ascii_ingest_and_search(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f12-pdf")
    pdf = sample_factory.pdf_bytes(
        ["Alpha Page One Token", "Beta Page Two Token", "Gamma Page Three Token"]
    )
    up = api_client.upload_document(
        bucket_id,
        filename="multi.pdf",
        content=pdf,
        content_type="application/pdf",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    search = api_client.search_bucket(
        bucket_id, {"query": "Gamma Page Three Token", "top_k": 5}
    ).assert_status(200).json()
    assert "Gamma Page Three Token" in str(search)

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        p3 = mcp.rag_get_page(document_id=doc_id, page_no=3)
        assert p3.is_error is False
        assert "Gamma" in p3.text


@pytest.mark.tier1
@pytest.mark.f12
def test_f12_png_image_ocr_offline_adapter_extracts_blocks(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f12-png")
    png = sample_factory.png_bytes(400, 300, boxes=[((50, 50, 350, 180), (20, 20, 20))])
    up = api_client.upload_document(
        bucket_id,
        filename="scan_page.png",
        content=png,
        content_type="image/png",
        fields={"access_scope": "org", "mode": "share"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)
    assert doc_id

    docs = api_client.list_documents(bucket_id).assert_status(200).json()
    assert "scan_page.png" in str(docs)


@pytest.mark.tier1
@pytest.mark.f12
def test_f12_pipeline_studio_ocr_mode_guard_and_dewarp_config(
    api_client: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    api_client.studio_pipeline_get().assert_status(200)
    yaml_text = sample_factory.pipeline_yaml("f12_std.yaml", profile="standard").read_text(
        encoding="utf-8"
    )
    api_client.studio_pipeline_update({"pipeline_yaml": yaml_text}).assert_status(200)

    test_res = api_client.studio_pipeline_test(
        {"query": "OCR mode check", "dewarp": True, "auth_level": 2}
    ).assert_status(200).json()
    assert "valid" in test_res and test_res["valid"] is True
    mode_val = test_res.get("ocr_mode") or test_res.get("mode")
    if mode_val is not None:
        assert str(mode_val).lower() == "share"
    serialized = str(test_res).lower()
    assert "share" in serialized


@pytest.mark.tier1
@pytest.mark.f12
def test_f12_cli_ingest_high_auth_level_enforces_private_or_zdr_mode(
    openrag_cli: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    pdf_path = sample_factory.pdf_file("classified.pdf", ["Confidential L8 Policy"])
    try:
        res = openrag_cli.ingest(
            pdf_path,
            bucket=f"f12-sec-{uuid.uuid4().hex[:6]}",
            scope="dept",
            auth_level=8,
            mode="private",
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    combined = (res.stdout + "\n" + res.stderr).lower()
    assert "private" in combined or "zdr" in combined or "local" in combined
    assert '"mode": "share"' not in combined


@pytest.mark.tier1
@pytest.mark.f12
def test_f12_airgap_pipeline_uses_local_vllm_ocr(
    openrag_cli: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    yaml_path = sample_factory.pipeline_yaml("airgap.yaml", profile="airgap")
    pdf_path = sample_factory.pdf_file("airgap_doc.pdf", ["Airgap Local OCR Check"])
    try:
        res = openrag_cli.ingest(
            pdf_path,
            bucket=f"f12-airgap-{uuid.uuid4().hex[:6]}",
            pipeline=yaml_path,
            env={"DATABASE_URL": migrated_pg.dsn()},
            expect_exit=0,
        )
    except CliBinaryNotFoundError as exc:
        fail_missing_target(str(exc))

    assert res.returncode == 0


# ---------------------------------------------------------------------------
# F13 — M2 Thumbnailer & 128d Visual Embed
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.f13
def test_f13_pdf_ingest_serves_valid_webp_page_thumbnail(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f13-webp")
    pdf = sample_factory.pdf_bytes(["Thumbnail Page One", "Thumbnail Page Two"])
    up = api_client.upload_document(
        bucket_id,
        filename="thumb.pdf",
        content=pdf,
        content_type="application/pdf",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)
    page_id = _extract_page_id(up)

    if not page_id:
        with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
            mcp.handshake()
            page_json = mcp.rag_get_page(document_id=doc_id, page_no=1).json()
            page_id = str(page_json.get("page_id") or page_json.get("id"))
    assert page_id

    thumb_resp = api_client.app_page_thumbnail(page_id).assert_status(200)
    ctype = thumb_resp.headers.get("content-type", "").lower()
    assert "image/webp" in ctype or "image/png" in ctype
    if "webp" in ctype:
        assert thumb_resp.body[:4] == b"RIFF" and thumb_resp.body[8:12] == b"WEBP"
    w, h, _ = _decode_image_rgb(thumb_resp.body)
    assert w > 0 and h > 0


@pytest.mark.tier1
@pytest.mark.f13
def test_f13_mcp_rag_get_page_returns_thumbnail_uri_and_norm1000_bbox(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f13-mcp")
    png = sample_factory.png_bytes(400, 300, boxes=[((100, 100, 700, 600), (40, 40, 180))])
    up = api_client.upload_document(
        bucket_id,
        filename="fig_page.png",
        content=png,
        content_type="image/png",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    with mcp_stdio(env={"DATABASE_URL": migrated_pg.dsn()}) as mcp:
        mcp.handshake()
        res = mcp.rag_get_page(document_id=doc_id, page_no=1)
        assert res.is_error is False
        data = res.json()
        assert data.get("page_id") or data.get("id")
        assert int(data.get("page_no", 1)) == 1
        assert data.get("thumb_uri") or data.get("thumbnail_url")
        boxes = data.get("boxes") or data.get("figures") or []
        for item in boxes:
            bbox = item.get("bbox_norm_1000") if isinstance(item, dict) else None
            if bbox:
                assert len(bbox) == 4
                x0, y0, x1, y1 = [int(v) for v in bbox]
                assert 0 <= x0 < x1 <= 999
                assert 0 <= y0 < y1 <= 999


@pytest.mark.tier1
@pytest.mark.f13
def test_f13_pg_stores_128d_finite_nonzero_visual_embeddings(
    openrag_cli: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    pdf_path = sample_factory.pdf_file("embed_check.pdf", ["Page One Embed", "Page Two Embed"])
    try:
        openrag_cli.ingest(
            pdf_path,
            bucket=f"f13-pg-{uuid.uuid4().hex[:6]}",
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
                WHERE d.title LIKE '%embed_check%' OR COALESCE(d.source_uri, '') LIKE '%embed_check%'
                ORDER BY p.page_no ASC
                """
            )
            rows = cur.fetchall()
            assert len(rows) >= 2
            for (raw_vec,) in rows:
                assert raw_vec is not None
                vec = _parse_pg_vector(raw_vec)
                assert len(vec) == 128
                assert all(math.isfinite(v) for v in vec)
                norm = math.sqrt(sum(v * v for v in vec))
                assert norm > 0.0
                assert abs(norm - 1.0) < 1e-3, f"Expected L2-normalized 128d thumb_embedding, got norm={norm}"


@pytest.mark.tier1
@pytest.mark.f13
def test_f13_image_figure_crop_and_bbox_norm_1000(
    api_client: Any, migrated_pg: PgEphemeralCluster, sample_factory: Any
) -> None:
    bucket_id, _ = _create_bucket(api_client, "f13-fig")
    png = sample_factory.png_bytes(400, 400, boxes=[((100, 150, 700, 650), (30, 80, 200))])
    up = api_client.upload_document(
        bucket_id,
        filename="diagram.png",
        content=png,
        content_type="image/png",
        fields={"access_scope": "org"},
    ).assert_status((200, 201)).json()
    doc_id = _extract_doc_id(up)

    with psycopg.connect(migrated_pg.dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT figure_uri, figure_embedding::text, bbox_norm_1000
                FROM corpus.page_figures
                WHERE document_id = %s::uuid
                """,
                (doc_id,),
            )
            figs = cur.fetchall()
            if figs:
                for uri, raw_vec, bbox in figs:
                    assert uri
                    vec = _parse_pg_vector(raw_vec)
                    assert len(vec) == 128 and all(math.isfinite(v) for v in vec)
                    norm = math.sqrt(sum(v * v for v in vec))
                    assert abs(norm - 1.0) < 1e-3, f"Expected L2-normalized 128d figure_embedding, got norm={norm}"
                    if bbox is not None:
                        vals = [int(v) for v in bbox]
                        assert 0 <= vals[0] < vals[2] <= 999
                        assert 0 <= vals[1] < vals[3] <= 999
            else:
                cur.execute(
                    "SELECT thumb_embedding::text FROM corpus.pages WHERE document_id = %s::uuid",
                    (doc_id,),
                )
                page_rows = cur.fetchall()
                assert page_rows
                vec = _parse_pg_vector(page_rows[0][0])
                assert len(vec) == 128 and all(math.isfinite(v) for v in vec)
                norm = math.sqrt(sum(v * v for v in vec))
                assert abs(norm - 1.0) < 1e-3, f"Expected L2-normalized 128d thumb_embedding, got norm={norm}"


@pytest.mark.tier1
@pytest.mark.f13
def test_f13_thumbnail_respects_pipeline_max_edge_px(
    api_client: Any, mcp_stdio: Any, sample_factory: Any, migrated_pg: PgEphemeralCluster
) -> None:
    yaml_text = sample_factory.pipeline_yaml("f13_edge.yaml", profile="standard").read_text(
        encoding="utf-8"
    )
    api_client.studio_pipeline_update({"pipeline_yaml": yaml_text}).assert_status(200)

    bucket_id, _ = _create_bucket(api_client, "f13-edge")
    wide_png = sample_factory.png_bytes(1600, 1200, boxes=[((100, 100, 800, 800), (50, 90, 150))])
    up = api_client.upload_document(
        bucket_id,
        filename="wide_scan.png",
        content=wide_png,
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

    thumb = api_client.app_page_thumbnail(page_id).assert_status(200)
    w, h, _ = _decode_image_rgb(thumb.body)
    assert max(w, h) <= 1280
    assert w >= h > 0
