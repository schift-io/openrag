"""Deterministic pure-Python fixture builders and BBox pixel redaction inspector."""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import unicodedata
import xml.sax.saxutils as xml_escape
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

try:
    from PIL import Image, features as _pil_features
except ImportError:  # pragma: no cover - verified installed on host, safe fallback kept
    Image = None  # type: ignore[assignment]
    _pil_features = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BBoxPixelReport:
    """Pixel-level verification result for a normalized 0..999 BBox region."""

    width: int
    height: int
    pixel_box: tuple[int, int, int, int]
    total_pixels: int
    black_pixels: int
    black_ratio: float
    mean_rgb: tuple[float, float, float]
    is_blacked_out: bool


def _norm_box_to_pixels(
    width: int, height: int, bbox_norm_1000: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_norm_1000
    px0 = max(0, min(width - 1, int(x0 * width / 1000)))
    py0 = max(0, min(height - 1, int(y0 * height / 1000)))
    px1 = max(px0 + 1, min(width, int((x1 + 1) * width / 1000)))
    py1 = max(py0 + 1, min(height, int((y1 + 1) * height / 1000)))
    return px0, py0, px1, py1


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def make_png_bytes(
    width: int = 100,
    height: int = 100,
    *,
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
    boxes: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = (),
) -> bytes:
    """Build a valid deterministic 8-bit RGB PNG using pure Python stdlib."""
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG dimensions must be positive, got {width}x{height}")

    row_stride = width * 3
    pixels = bytearray(bg_rgb) * (width * height)

    for bbox, color in boxes:
        px0, py0, px1, py1 = _norm_box_to_pixels(width, height, bbox)
        rgb_bytes = bytes(color)
        for y in range(py0, py1):
            row_start = y * row_stride
            for x in range(px0, px1):
                idx = row_start + x * 3
                pixels[idx : idx + 3] = rgb_bytes

    raw_scanlines = bytearray()
    for y in range(height):
        raw_scanlines.append(0)  # Filter type 0 (None)
        row_start = y * row_stride
        raw_scanlines.extend(pixels[row_start : row_start + row_stride])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_scanlines), level=6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_rgb(data: bytes) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Pure-Python 8-bit RGB/RGBA PNG decoder supporting all 5 PNG scanline filters."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG magic header")

    pos = 8
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    idat_chunks: list[bytes] = []

    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        pos += 8
        cdata = data[pos : pos + length]
        pos += length + 4  # skip chunk + CRC
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", cdata)
        elif ctype == b"IDAT":
            idat_chunks.append(cdata)
        elif ctype == b"IEND":
            break

    if width <= 0 or height <= 0:
        raise ValueError("Missing or invalid PNG IHDR")
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(
            f"Pure-Python PNG fallback supports 8-bit RGB/RGBA, got depth={bit_depth} color={color_type}"
        )

    bpp = 3 if color_type == 2 else 4
    stride = width * bpp
    raw = zlib.decompress(b"".join(idat_chunks))

    prev_row = bytearray(stride)
    pixels: list[tuple[int, int, int]] = []
    offset = 0

    for _ in range(height):
        flt = raw[offset]
        offset += 1
        row = bytearray(raw[offset : offset + stride])
        offset += stride

        if flt == 0:
            pass
        elif flt == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif flt == 2:
            for i in range(stride):
                row[i] = (row[i] + prev_row[i]) & 0xFF
        elif flt == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev_row[i]
                row[i] = (row[i] + ((left + up) >> 1)) & 0xFF
        elif flt == 4:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev_row[i]
                up_left = prev_row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"Unsupported PNG filter type: {flt}")

        for x in range(width):
            base = x * bpp
            pixels.append((row[base], row[base + 1], row[base + 2]))
        prev_row = row

    return width, height, pixels


def make_webp_bytes(
    width: int = 100,
    height: int = 100,
    *,
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
    boxes: Sequence[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = (),
) -> bytes:
    """Build a lossless WebP image via Pillow or `cwebp` fallback."""
    png_bytes = make_png_bytes(width, height, bg_rgb=bg_rgb, boxes=boxes)

    if Image is not None and _pil_features is not None and _pil_features.check("webp"):
        with Image.open(BytesIO(png_bytes)) as img:
            out = BytesIO()
            img.save(out, format="WEBP", lossless=True)
            return out.getvalue()

    cwebp = shutil.which("cwebp")
    if cwebp is not None:
        with tempfile.TemporaryDirectory(prefix="orag_webp_") as td:
            in_png = Path(td) / "in.png"
            out_webp = Path(td) / "out.webp"
            in_png.write_bytes(png_bytes)
            subprocess.run(
                [cwebp, "-quiet", "-lossless", str(in_png), "-o", str(out_webp)],
                check=True,
                capture_output=True,
            )
            return out_webp.read_bytes()

    raise RuntimeError("Neither Pillow WebP support nor `cwebp` CLI is available")


def _decode_image_rgb(image_bytes: bytes) -> tuple[int, int, list[tuple[int, int, int]]]:
    if Image is not None:
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                rgb = img.convert("RGB")
                w, h = rgb.size
                raw = list(rgb.getdata())
                return w, h, [(int(r), int(g), int(b)) for (r, g, b) in raw]
        except Exception:
            pass

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return decode_png_rgb(image_bytes)

    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        dwebp = shutil.which("dwebp")
        if dwebp is not None:
            with tempfile.TemporaryDirectory(prefix="orag_dwebp_") as td:
                in_webp = Path(td) / "in.webp"
                out_png = Path(td) / "out.png"
                in_webp.write_bytes(image_bytes)
                subprocess.run(
                    [dwebp, str(in_webp), "-o", str(out_png)],
                    check=True,
                    capture_output=True,
                )
                return decode_png_rgb(out_png.read_bytes())

    raise ValueError("Unsupported or unreadable image format (expected PNG or WebP)")


def inspect_bbox_blackout(
    image_bytes: bytes,
    bbox_norm_1000: tuple[int, int, int, int],
    *,
    black_threshold: int = 16,
    min_black_ratio: float = 0.90,
) -> BBoxPixelReport:
    """Inspect pixels inside `bbox_norm_1000` (0..999) and check if black-box masked."""
    w, h, pixels = _decode_image_rgb(image_bytes)
    px0, py0, px1, py1 = _norm_box_to_pixels(w, h, bbox_norm_1000)

    total = 0
    black = 0
    sum_r = 0
    sum_g = 0
    sum_b = 0

    for y in range(py0, py1):
        row_offset = y * w
        for x in range(px0, px1):
            r, g, b = pixels[row_offset + x]
            total += 1
            sum_r += r
            sum_g += g
            sum_b += b
            if r <= black_threshold and g <= black_threshold and b <= black_threshold:
                black += 1

    ratio = (black / total) if total > 0 else 0.0
    mean_rgb = (
        (sum_r / total, sum_g / total, sum_b / total) if total > 0 else (0.0, 0.0, 0.0)
    )
    return BBoxPixelReport(
        width=w,
        height=h,
        pixel_box=(px0, py0, px1, py1),
        total_pixels=total,
        black_pixels=black,
        black_ratio=ratio,
        mean_rgb=mean_rgb,
        is_blacked_out=(ratio >= min_black_ratio),
    )


def _pdf_escape(text: str) -> str:
    ascii_safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return ascii_safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf_bytes(
    pages: Sequence[str] = ("OpenRAG E2E Test Document",),
) -> bytes:
    """Build a valid multi-page PDF 1.4 with accurate byte-offset xref table."""
    page_texts = list(pages) if pages else ["OpenRAG E2E Test Document"]
    n = len(page_texts)

    # Object numbers:
    # 1: Catalog, 2: Pages, 3: Helvetica Font
    # For page i in 0..n-1: page_obj = 4 + 2*i, content_obj = 5 + 2*i
    objects: dict[int, bytes] = {}
    kids_refs = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [ {kids_refs} ] /Count {n} >>".encode("ascii")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, text in enumerate(page_texts):
        page_id = 4 + 2 * i
        content_id = 5 + 2 * i
        lines = text.splitlines() or [text]
        stream_ops = ["BT", "/F1 12 Tf", "72 720 Td"]
        for line_idx, line in enumerate(lines):
            if line_idx > 0:
                stream_ops.append("0 -16 Td")
            stream_ops.append(f"({_pdf_escape(line)}) Tj")
        stream_ops.append("ET")
        stream_bytes = "\n".join(stream_ops).encode("latin-1")

        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )

    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    total_objs = 3 + 2 * n
    offsets: dict[int, int] = {}
    for obj_id in range(1, total_objs + 1):
        offsets[obj_id] = len(buf)
        buf.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        buf.extend(objects[obj_id])
        buf.extend(b"\nendobj\n")

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {total_objs + 1}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, total_objs + 1):
        buf.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))

    buf.extend(
        (
            f"trailer\n<< /Size {total_objs + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(buf)


def make_hwpx_bytes(
    *,
    title: str = "OpenRAG 테스트 문서",
    paragraphs: Sequence[str] = ("본문 테스트",),
    tables: Sequence[Sequence[Sequence[str]]] | None = None,
) -> bytes:
    """Build a valid OWPML/HWPX ZIP archive with uncompressed `mimetype` and XML sections."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", b"application/hwp+zip", compress_type=zipfile.ZIP_STORED)

        container_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ocf:container xmlns:ocf="urn:oasis:names:tc:opendocument:xmlns:container">\n'
            "  <ocf:rootfiles>\n"
            '    <ocf:rootfile full-path="Contents/content.hpf" '
            'media-type="application/hwpml-package+xml"/>\n'
            "  </ocf:rootfiles>\n"
            "</ocf:container>\n"
        )
        zf.writestr("META-INF/container.xml", container_xml.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

        content_hpf = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<opf:package xmlns:opf="http://www.idpf.org/2007/opf" version="1.0">\n'
            f"  <opf:metadata><opf:title>{xml_escape.escape(title)}</opf:title></opf:metadata>\n"
            "  <opf:manifest>\n"
            '    <opf:item id="header" href="Contents/header.xml" media-type="application/xml"/>\n'
            '    <opf:item id="section0" href="Contents/section0.xml" media-type="application/xml"/>\n'
            "  </opf:manifest>\n"
            "  <opf:spine>\n"
            '    <opf:itemref idref="section0"/>\n'
            "  </opf:spine>\n"
            "</opf:package>\n"
        )
        zf.writestr("Contents/content.hpf", content_hpf.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

        header_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<hh:head xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head" version="1.4" secCnt="1">\n'
            "  <hh:beginNum page=\"1\" footnote=\"1\" endnote=\"1\" pic=\"1\" tbl=\"1\" equation=\"1\"/>\n"
            "</hh:head>\n"
        )
        zf.writestr("Contents/header.xml", header_xml.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

        sec_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
            'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">',
            f"  <hp:p><hp:run><hp:t>{xml_escape.escape(title)}</hp:t></hp:run></hp:p>",
        ]
        for para in paragraphs:
            sec_parts.append(
                f"  <hp:p><hp:run><hp:t>{xml_escape.escape(para)}</hp:t></hp:run></hp:p>"
            )

        if tables:
            for tbl in tables:
                row_cnt = len(tbl)
                col_cnt = max((len(r) for r in tbl), default=0)
                sec_parts.append(f'  <hp:tbl rowCnt="{row_cnt}" colCnt="{col_cnt}">')
                for r_idx, row in enumerate(tbl):
                    sec_parts.append("    <hp:tr>")
                    for c_idx, cell_text in enumerate(row):
                        sec_parts.append(
                            f'      <hp:tc colAddr="{c_idx}" rowAddr="{r_idx}" colSpan="1" rowSpan="1">'
                            f"<hp:subList><hp:p><hp:run><hp:t>{xml_escape.escape(cell_text)}</hp:t>"
                            "</hp:run></hp:p></hp:subList></hp:tc>"
                        )
                    sec_parts.append("    </hp:tr>")
                sec_parts.append("  </hp:tbl>")

        sec_parts.append("</hs:sec>\n")
        zf.writestr(
            "Contents/section0.xml",
            "\n".join(sec_parts).encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    return buf.getvalue()


def make_korean_pii_doc(
    *,
    include_rrn: bool = True,
    include_phone: bool = True,
    include_email: bool = True,
    include_address: bool = True,
    include_name: bool = True,
    safe_words: Sequence[str] = ("PROJECT-ORION",),
    allowlist_terms: Sequence[str] = (),
) -> dict[str, Any]:
    """Build deterministic Korean PII + Safe-Word institutional fixture payload."""
    rrn = "900101-1234567"
    phone = "010-1234-5678"
    email = "hong.gildong@example.or.kr"
    address = "서울특별시 종로구 세종대로 209 정부서울청사 1204호"
    name = "홍길동"

    lines = ["[공공기관 내부 검토 보고서]"]
    expected_tokens: list[str] = []

    if include_name:
        lines.append(f"담당자 성명: {name}")
        expected_tokens.append(name)
    if include_rrn:
        lines.append(f"주민등록번호: {rrn}")
        expected_tokens.append(rrn)
    if include_phone:
        lines.append(f"연락처: {phone}")
        expected_tokens.append(phone)
    if include_email:
        lines.append(f"이메일: {email}")
        expected_tokens.append(email)
    if include_address:
        lines.append(f"소속 주소: {address}")
        expected_tokens.append(address)

    sw_list = list(safe_words)
    if sw_list:
        lines.append("보안 코드명: " + ", ".join(sw_list))
    al_list = list(allowlist_terms)
    if al_list:
        lines.append("예외 허용 용어: " + ", ".join(al_list))

    return {
        "rrn": rrn if include_rrn else None,
        "phone": phone if include_phone else None,
        "email": email if include_email else None,
        "address": address if include_address else None,
        "name": name if include_name else None,
        "safe_words": sw_list,
        "allowlist_terms": al_list,
        "expected_tokens": expected_tokens,
        "text": "\n".join(lines),
    }


def make_secret_payload() -> dict[str, Any]:
    """Deterministic fake credential / local-path payload for ai-memory-hooks pre-redaction tests."""
    openai_key = "sk-test-openrag-1234567890abcdef"
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    local_path = "/Users/dev/secret/.env"
    return {
        "openai_key": openai_key,
        "aws_key": aws_key,
        "local_path": local_path,
        "secrets": [openai_key, aws_key, local_path],
        "transcript": (
            f"Checked {local_path} with OPENAI_API_KEY={openai_key} "
            f"and AWS_ACCESS_KEY_ID={aws_key} before deploying."
        ),
    }


def make_obsidian_vault(
    vault_dir: Path | None = None,
    *,
    extra_pages: Mapping[str, str] | None = None,
) -> Path:
    """Populate a deterministic Obsidian Markdown directory with YAML frontmatter and [[wikilinks]]."""
    root = Path(vault_dir) if vault_dir is not None else Path(tempfile.mkdtemp(prefix="orag_vault_"))
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        "index.md": (
            "---\n"
            "title: 기관 규정 위키 인덱스\n"
            "scope: dept\n"
            "auth_level: 3\n"
            "---\n\n"
            "# 기관 규정 위키\n\n"
            "- [[travel-policy-2025]]\n"
            "- [[travel-policy-2026]]\n"
            "- [[security-guidelines]]\n"
        ),
        "travel-policy-2025.md": (
            "---\n"
            "title: 2025년 출장 여비 규정\n"
            "revision: 1\n"
            "valid_from: 2025-01-01\n"
            "valid_to: 2025-12-31\n"
            "---\n\n"
            "# 2025년 출장 여비 규정\n\n"
            "국내 출장 일비는 1일 50,000원으로 지급한다.\n"
            "참고: [[security-guidelines]]\n"
        ),
        "travel-policy-2026.md": (
            "---\n"
            "title: 2026년 출장 여비 개정 규정\n"
            "revision: 2\n"
            "supersedes: travel-policy-2025\n"
            "valid_from: 2026-01-01\n"
            "---\n\n"
            "# 2026년 출장 여비 개정 규정\n\n"
            "2026년 1월부터 국내 출장 일비는 1일 70,000원으로 상향한다.\n"
            "본 규정은 [[travel-policy-2025]] 를 대체한다.\n"
        ),
        "security-guidelines.md": (
            "---\n"
            "title: 정보보안 및 개인정보 보호 지침\n"
            "min_auth_level: 5\n"
            "---\n\n"
            "# 정보보안 및 개인정보 보호 지침\n\n"
            "주민등록번호 등 민감 정보는 외부 LLM 전송 전 반드시 마스킹한다.\n"
        ),
    }
    if extra_pages:
        for rel, content in extra_pages.items():
            files[rel] = content

    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return root


def make_pipeline_yaml(
    *,
    profile: str = "standard",
    engine: str = "native",
    fusion: str = "rrf",
    alpha: float = 0.3,
    local_only_min_auth_level: int = 8,
    guard_enabled: bool = True,
    guard_rules: Sequence[dict[str, Any]] | None = None,
    memory_backend: str = "openrag_pg",
    memory_mirror_to: str | None = None,
) -> str:
    """Emit a deterministic OpenRAG pipeline YAML config."""
    rules = list(guard_rules) if guard_rules is not None else [
        {
            "kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query", "llm_egress", "mcp_egress", "ingestion_gate"],
        },
        {
            "kind": "pii_category",
            "pattern_or_category": "RRN",
            "action": "mask",
            "surfaces": ["llm_egress", "mcp_egress"],
        },
    ]

    memory_cfg: dict[str, Any] = {"backend": memory_backend}
    if memory_mirror_to is not None:
        memory_cfg["mirror_to"] = memory_mirror_to

    doc = {
        "version": 1,
        "profile": profile,
        "parse": {
            "hwp": "kordoc",
            "ocr": "local_vllm" if profile == "airgap" else "schift_ocr",
            "dewarp": True,
            "ocr_mode_by_auth_level": {"share_max_level": 3, "private_min_level": 4, "zdr_min_level": 7},
        },
        "thumbnail": {
            "format": "webp",
            "dim": 128,
            "max_edge_px": 1280,
        },
        "pii": {
            "schift_v7": True,
            "marker_ko_pii": False,
            "schift_v6": False,
            "custom_dict": [],
            "capsule_enabled": True,
            "guard": {
                "enabled": guard_enabled,
                "default_action": "mask",
                "allowlist": [],
                "rules": rules,
            },
        },
        "memory": memory_cfg,
        "store": {
            "pg_pinecone_engine": engine,
            "dim": 128,
            "bm25": "pg_search",
            "external_mirror": "none",
        },
        "retrieve": {
            "spaces": ["chunks", "thumb", "wiki", "memory"],
            "fusion": fusion,
            "alpha": alpha,
            "rerank": True,
            "latest_revision_boost": True,
            "admission_threshold": 0.25,
        },
        "llm": {
            "external_provider": "vllm_local" if profile == "airgap" else "openrouter",
            "local_provider": "vllm_local",
            "local_only_min_auth_level": local_only_min_auth_level,
        },
    }
    header = "# memory.backend: openrag_pg | kumiho (optional mirror_to: kumiho)\n"
    return header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def make_search_space_yaml(
    *,
    golden_set: str = "gov_v1",
    require_zero_pii_leak: bool = True,
    require_zero_arn_violation: bool = True,
) -> str:
    """Emit a deterministic AutoRAG search-space YAML config."""
    doc = {
        "version": 1,
        "golden_set": golden_set,
        "strategy": "greedy_node",
        "security_gates": {
            "require_zero_pii_leak": require_zero_pii_leak,
            "require_zero_arn_violation": require_zero_arn_violation,
        },
        "nodes": {
            "parse": ["pdf_native", "schift_ocr"],
            "chunking": ["semantic_512", "fixed_384"],
            "retrieve.fusion": ["rrf", "convex"],
            "retrieve.alpha": [0.2, 0.3, 0.5],
            "retrieve.rerank": [True, False],
            "llm.model": ["local-vllm", "openrouter/masked"],
        },
    }
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def make_concept_alias_pairs() -> dict[str, Any]:
    """Return canonical cross-lingual and domain synonym equivalence groups for F36/F37."""
    return {
        "cross_lingual": [
            {"canonical": "apple", "aliases": ["apple", "사과"], "langs": ["en", "ko"]},
            {"canonical": "rfp", "aliases": ["RFP", "제안요청서"], "langs": ["en", "ko"]},
        ],
        "domain_synonyms": [
            {
                "canonical": "llmwiki",
                "aliases": ["llmwiki", "LLM Wiki", "세컨드 브레인", "Second Brain"],
                "langs": ["en", "ko"],
            },
        ],
        "tag_aliases": [
            {"canonical": "llmwiki", "aliases": ["llmwiki", "세컨드브레인"]},
        ],
    }


def make_parenthetical_alias_doc(
    *,
    title: str = "개념 자동 추출 테스트 문서",
    pairs: Sequence[tuple[str, str]] = (
        ("사과", "apple"),
        ("세컨드 브레인", "LLM Wiki"),
        ("제안요청서", "RFP"),
    ),
    extra_paragraphs: Sequence[str] = (),
) -> dict[str, Any]:
    """Emit UTF-8 text, Markdown, and HWPX bytes containing `한글(English)` parenthetical pairs."""
    pair_list = [(str(k), str(v)) for k, v in pairs]
    lines = [f"{k}({v}) 관련 기관 표준 문서 정리" for k, v in pair_list]
    lines.extend(str(p) for p in extra_paragraphs)
    text = "\n".join(lines)
    markdown = f"# {title}\n\n" + "\n\n".join(lines) + "\n"
    hwpx_bytes = make_hwpx_bytes(title=title, paragraphs=lines)
    return {
        "title": title,
        "pairs": pair_list,
        "text": text,
        "markdown": markdown,
        "hwpx_bytes": hwpx_bytes,
    }


def make_obfuscated_variants(term: str) -> dict[str, str]:
    """Emit deterministic NFKC fullwidth, zero-width (\\u200b), punct-spaced, and mixed variants."""
    if not term:
        raise ValueError("term must be non-empty")

    def _to_fullwidth(ch: str) -> str:
        if ch == " ":
            return "\u3000"
        code = ord(ch)
        if 0x21 <= code <= 0x7E:
            return chr(code + 0xFEE0)
        return ch

    fullwidth_nfkc = "".join(_to_fullwidth(c) for c in term)
    # Also verify NFKC round-trip property on ASCII parts; keep deterministic output
    _ = unicodedata.normalize("NFKC", fullwidth_nfkc)

    chars = list(term)
    zero_width = "\u200b".join(chars)

    punct_cycle = (".", "_", "-", " ")
    punct_parts: list[str] = []
    for idx, ch in enumerate(chars):
        punct_parts.append(ch)
        if idx < len(chars) - 1:
            punct_parts.append(punct_cycle[idx % len(punct_cycle)])
    punct_spaced = "".join(punct_parts)

    mixed_parts: list[str] = []
    for idx, ch in enumerate(chars):
        mixed_parts.append(_to_fullwidth(ch))
        if idx < len(chars) - 1:
            mixed_parts.append("\u200b" if idx % 2 == 0 else ".")
    mixed = "".join(mixed_parts)

    return {
        "original": term,
        "fullwidth_nfkc": fullwidth_nfkc,
        "zero_width": zero_width,
        "punct_spaced": punct_spaced,
        "mixed": mixed,
    }


def make_tagged_markdown(
    *,
    title: str = "태그 검색 테스트 문서",
    tags: Sequence[str] = ("llmwiki", "policy-2026"),
    body: str = "태그 기반 검색 및 패싯 집계 본문입니다.",
    scope: str = "dept",
    min_auth_level: int = 3,
    inline_hashtags: bool = True,
) -> str:
    """Emit UTF-8 Markdown with YAML frontmatter tags and optional inline #tag tokens."""
    tag_list = [str(t).lstrip("#").strip() for t in tags if str(t).strip()]
    fm = yaml.safe_dump(
        {
            "title": title,
            "scope": scope,
            "min_auth_level": min_auth_level,
            "tags": tag_list,
        },
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    parts = [f"---\n{fm}\n---\n", f"# {title}\n", body.strip()]
    if inline_hashtags and tag_list:
        parts.append(" ".join(f"#{t}" for t in tag_list))
    return "\n\n".join(parts) + "\n"


def make_tagged_corpus_bundle() -> list[dict[str, Any]]:
    """Return a deterministic multi-document corpus for tags_any/tags_all/tags_exclude and alias tags."""
    docs = [
        {
            "key": "d1_hr_2026_draft",
            "title": "인사 규정 초안 2026",
            "tags": ["hr", "2026", "draft"],
            "body": "공통규정키워드T9 인사 규정 초안 문서입니다.",
        },
        {
            "key": "d2_hr_2026_approved",
            "title": "인사 규정 확정본 2026",
            "tags": ["hr", "2026", "approved"],
            "body": "공통규정키워드T9 인사 규정 확정 문서입니다.",
        },
        {
            "key": "d3_hr_2025_approved",
            "title": "인사 규정 확정본 2025",
            "tags": ["hr", "2025", "approved"],
            "body": "공통규정키워드T9 2025년도 인사 규정입니다.",
        },
        {
            "key": "d4_finance_2026_approved",
            "title": "재무 지출 기준 2026",
            "tags": ["finance", "2026", "approved"],
            "body": "공통규정키워드T9 재무 지출 기준 문서입니다.",
        },
        {
            "key": "d5_llmwiki",
            "title": "지식관리 아키텍처",
            "tags": ["llmwiki"],
            "body": "지식관리 시스템 및 개인 지식 연결 가이드입니다.",
        },
    ]
    for item in docs:
        item["markdown"] = make_tagged_markdown(
            title=item["title"],
            tags=item["tags"],
            body=item["body"],
        )
    return docs


class FixtureFactory:
    """Convenience wrapper binding deterministic generators to a per-test temp directory."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, rel_name: str, data: bytes) -> Path:
        path = self.base_dir / rel_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def write_text(self, rel_name: str, text: str) -> Path:
        path = self.base_dir / rel_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def pdf_bytes(self, pages: Sequence[str] = ("OpenRAG E2E Test Document",)) -> bytes:
        return make_pdf_bytes(pages)

    def pdf_file(
        self, filename: str = "sample.pdf", pages: Sequence[str] = ("OpenRAG E2E Test Document",)
    ) -> Path:
        return self.write_bytes(filename, make_pdf_bytes(pages))

    def hwpx_bytes(
        self,
        *,
        title: str = "OpenRAG 테스트 문서",
        paragraphs: Sequence[str] = ("본문 테스트",),
        tables: Sequence[Sequence[Sequence[str]]] | None = None,
    ) -> bytes:
        return make_hwpx_bytes(title=title, paragraphs=paragraphs, tables=tables)

    def hwpx_file(
        self,
        filename: str = "sample.hwpx",
        *,
        title: str = "OpenRAG 테스트 문서",
        paragraphs: Sequence[str] = ("본문 테스트",),
        tables: Sequence[Sequence[Sequence[str]]] | None = None,
    ) -> Path:
        return self.write_bytes(
            filename, make_hwpx_bytes(title=title, paragraphs=paragraphs, tables=tables)
        )

    def png_bytes(self, *args: Any, **kwargs: Any) -> bytes:
        return make_png_bytes(*args, **kwargs)

    def webp_bytes(self, *args: Any, **kwargs: Any) -> bytes:
        return make_webp_bytes(*args, **kwargs)

    def inspect_bbox_blackout(self, *args: Any, **kwargs: Any) -> BBoxPixelReport:
        return inspect_bbox_blackout(*args, **kwargs)

    def korean_pii_doc(self, **kwargs: Any) -> dict[str, Any]:
        return make_korean_pii_doc(**kwargs)

    def secret_payload(self) -> dict[str, Any]:
        return make_secret_payload()

    def obsidian_vault(
        self, subdir: str = "obsidian_vault", *, extra_pages: Mapping[str, str] | None = None
    ) -> Path:
        return make_obsidian_vault(self.base_dir / subdir, extra_pages=extra_pages)

    def pipeline_yaml(self, filename: str = "pipeline.yaml", **kwargs: Any) -> Path:
        return self.write_text(filename, make_pipeline_yaml(**kwargs))

    def search_space_yaml(self, filename: str = "search_space.yaml", **kwargs: Any) -> Path:
        return self.write_text(filename, make_search_space_yaml(**kwargs))

    def concept_alias_pairs(self) -> dict[str, Any]:
        return make_concept_alias_pairs()

    def parenthetical_alias_doc(self, **kwargs: Any) -> dict[str, Any]:
        return make_parenthetical_alias_doc(**kwargs)

    def obfuscated_variants(self, term: str) -> dict[str, str]:
        return make_obfuscated_variants(term)

    def tagged_markdown(self, **kwargs: Any) -> str:
        return make_tagged_markdown(**kwargs)

    def tagged_corpus_bundle(self) -> list[dict[str, Any]]:
        return make_tagged_corpus_bundle()
