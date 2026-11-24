use std::io::{Cursor, Write};

use chrono::{TimeZone, Utc};
use flate2::write::DeflateEncoder;
use flate2::Compression;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use openrag_core::{AuthLevel, OcrModeRule, OpenRagError, ParseConfig, Sensitivity, ThumbnailConfig};
use openrag_parse::{
    extract_parenthetical_aliases, encode_png_bounded, parse_hml, parse_hwp5, parse_hwpx,
    parse_image_with_ocr, parse_pdf_native, resolve_ocr_policy, sanitize_table_html,
    DocumentFormat, DocumentParser, InMemoryBlobStore, OcrExecutionTarget, OcrTask,
    ParsePolicyCtx, ShareQuotaTracker, ThumbnailBlobStore, Thumbnailer,
};
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

fn default_parse_cfg() -> ParseConfig {
    ParseConfig {
        hwp_engine: "kordoc".into(),
        ocr_engine: "schift_ocr".into(),
        dewarp_enabled: true,
        pdf_native_enabled: true,
        default_ocr_mode: Sensitivity::Share,
        ocr_mode_by_auth_level: vec![
            OcrModeRule {
                min_auth_level: AuthLevel::new(8).unwrap_or(AuthLevel::MAX),
                required_mode: Sensitivity::Zdr,
            },
            OcrModeRule {
                min_auth_level: AuthLevel::new(5).unwrap_or(AuthLevel::MAX),
                required_mode: Sensitivity::Private,
            },
        ],
        reject_zdr_with_batch: true,
    }
}

fn default_thumb_cfg() -> ThumbnailConfig {
    ThumbnailConfig {
        enabled: true,
        format: "webp".into(),
        max_edge_px: 1280,
        figure_max_edge_px: 1280,
        figure_max_bytes: 307_200,
    }
}

fn build_test_hwpx(section_xml: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(buf);
    let opts = SimpleFileOptions::default();

    zip.start_file("Contents/content.hpf", opts)?;
    zip.write_all(
        br#"<?xml version="1.0" encoding="UTF-8"?>
        <opf:package xmlns:opf="http://www.idpf.org/2007/opf">
          <opf:metadata><opf:title>2026 RFP Draft</opf:title></opf:metadata>
        </opf:package>"#,
    )?;

    zip.start_file("Contents/section0.xml", opts)?;
    zip.write_all(section_xml.as_bytes())?;

    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

#[test]
fn f11_hwpx_table_html_escaping_and_span_preservation() -> Result<(), Box<dyn std::error::Error>> {
    let section = r#"<?xml version="1.0" encoding="UTF-8"?>
    <hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"
            xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
      <hp:p><hp:run><hp:t>제안요청서(RFP) 요약 문단</hp:t></hp:run></hp:p>
      <hp:equation><hp:script>E = mc^2</hp:script></hp:equation>
      <hp:tbl>
        <hp:tr>
          <hp:tc colSpan="2"><hp:subList><hp:p><hp:run><hp:t>항목 &lt;script&gt;alert(1)&lt;/script&gt;</hp:t></hp:run></hp:p></hp:subList></hp:tc>
        </hp:tr>
        <hp:tr>
          <hp:tc rowSpan="2"><hp:subList><hp:p><hp:run><hp:t>예산</hp:t></hp:run></hp:p></hp:subList></hp:tc>
          <hp:tc><hp:subList><hp:p><hp:run><hp:t>100억 &amp; 부가세 별도</hp:t></hp:run></hp:p></hp:subList></hp:tc>
        </hp:tr>
      </hp:tbl>
    </hs:sec>"#;

    let bytes = build_test_hwpx(section)?;
    let doc = parse_hwpx(&bytes, Some("rfp.hwpx"))?;
    assert_eq!(doc.title, "2026 RFP Draft");
    assert_eq!(doc.format, DocumentFormat::Hwpx);
    assert_eq!(doc.pages.len(), 1);
    assert_eq!(doc.extracted_aliases, vec![("제안요청서".into(), "RFP".into())]);
    assert_eq!(doc.pages[0].formulas, vec!["E = mc^2".to_string()]);

    let table = doc.pages[0]
        .tables_html
        .first()
        .ok_or("missing parsed table")?;
    assert!(table.contains("<table>"));
    assert!(table.contains("<tr>"));
    assert!(table.contains("<td colspan=\"2\">"));
    assert!(table.contains("<td rowspan=\"2\">"));
    assert!(table.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
    assert!(table.contains("100억 &amp; 부가세 별도"));
    assert!(!table.contains("<script>"));
    assert!(!table.contains("<th"));
    assert!(!table.contains("<thead"));
    assert!(!table.contains("<br"));
    assert!(!table.contains("<b>"));
    Ok(())
}

#[test]
fn f11_corrupt_hwpx_missing_section0_and_corrupt_ole2_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // Corrupt ZIP
    let err_zip = parse_hwpx(b"PK\x03\x04corrupt-zip-bytes", Some("bad.hwpx"));
    assert!(matches!(err_zip, Err(OpenRagError::Validation(_))));

    // Valid ZIP missing Contents/section0.xml
    let cursor = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(cursor);
    zip.start_file("Contents/content.hpf", SimpleFileOptions::default())?;
    zip.write_all(b"<package/>")?;
    let no_sec = zip.finish()?.into_inner();
    let err_no_sec = parse_hwpx(&no_sec, Some("empty.hwpx"));
    assert!(matches!(err_no_sec, Err(OpenRagError::Validation(ref msg)) if msg.contains("section0.xml")));

    // Corrupt OLE2 HWP
    let err_ole = parse_hwp5(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1INVALID_OLE2_CORRUPT",
        Some("bad.hwp"),
    );
    assert!(matches!(err_ole, Err(OpenRagError::Validation(_))));
    Ok(())
}

#[test]
fn f11_hml_and_hwp5_pure_rust_parsing() -> Result<(), Box<dyn std::error::Error>> {
    let hml = r#"<?xml version="1.0" encoding="UTF-8"?>
    <HWPML Version="2.9">
      <HEAD><TITLE>HML Report</TITLE></HEAD>
      <BODY><SECTION>
        <P><TEXT><CHAR>사과(apple) 공급 계약</CHAR></TEXT></P>
        <TABLE><ROW><CELL ColSpan="2"><P><TEXT><CHAR>합계</CHAR></TEXT></P></CELL></ROW></TABLE>
      </SECTION></BODY>
    </HWPML>"#
        .as_bytes();
    let hml_doc = parse_hml(hml, Some("contract.hml"))?;
    assert_eq!(hml_doc.title, "HML Report");
    assert!(hml_doc.full_markdown.contains("<td colspan=\"2\">합계</td>"));
    assert!(!hml_doc.full_markdown.contains("<th"));
    assert_eq!(hml_doc.extracted_aliases, vec![("사과".into(), "apple".into())]);

    // Build minimal valid HWP 5.0 OLE2 compound file with compressed BodyText/Section0
    let cursor = Cursor::new(Vec::new());
    let mut cfb = cfb::CompoundFile::create(cursor)?;
    {
        let mut hdr = [0u8; 256];
        let sig = b"HWP Document File";
        hdr[..sig.len()].copy_from_slice(sig);
        // Set compression bit 0 at offset 36
        hdr[36..40].copy_from_slice(&1u32.to_le_bytes());
        let mut stream = cfb.create_stream("FileHeader")?;
        stream.write_all(&hdr)?;
    }
    {
        cfb.create_storage("BodyText")?;
        let text = "한글바이너리 테스트";
        let mut utf16 = Vec::new();
        for u in text.encode_utf16() {
            utf16.extend_from_slice(&u.to_le_bytes());
        }
        let tag_id: u32 = 67; // HWPTAG_PARA_TEXT
        let level: u32 = 0;
        let size: u32 = utf16.len() as u32;
        let rec_hdr: u32 = tag_id | (level << 10) | (size << 20);
        let mut raw_rec = Vec::new();
        raw_rec.extend_from_slice(&rec_hdr.to_le_bytes());
        raw_rec.extend_from_slice(&utf16);

        let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&raw_rec)?;
        let compressed = enc.finish()?;

        let mut stream = cfb.create_stream("BodyText/Section0")?;
        stream.write_all(&compressed)?;
    }
    let hwp5_bytes = cfb.into_inner().into_inner();
    let hwp5_doc = parse_hwp5(&hwp5_bytes, Some("legacy.hwp"))?;
    assert_eq!(hwp5_doc.format, DocumentFormat::Hwp);
    assert!(hwp5_doc.full_markdown.contains("한글바이너리 테스트"));
    // Public offices publish HWP 5 files named .hwpx: the bytes decide, not the name.
    assert_eq!(DocumentFormat::detect(&hwp5_bytes, None, Some("공고문.hwpx")), DocumentFormat::Hwp);
    Ok(())
}

#[test]
fn f11_mislabeled_files_are_read_by_their_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let hwpx = build_test_hwpx("<hs:sec/>")?;
    assert_eq!(DocumentFormat::detect(&hwpx, None, Some("공고문.hwp")), DocumentFormat::Hwpx);
    let mut docx = ZipWriter::new(Cursor::new(Vec::new()));
    docx.start_file("word/document.xml", SimpleFileOptions::default())?;
    docx.write_all(b"<w:document/>")?;
    let docx = docx.finish()?.into_inner();
    assert_eq!(DocumentFormat::detect(&docx, Some("application/hwp+zip"), Some("공고문.hwpx")), DocumentFormat::Docx);
    assert_eq!(DocumentFormat::detect(b"%PDF-1.7\n", None, Some("공고문.hwp")), DocumentFormat::Pdf);
    // Text has no signature: the extension still decides.
    assert_eq!(DocumentFormat::detect(b"# PK", None, Some("notes.md")), DocumentFormat::Markdown);
    assert_eq!(DocumentFormat::detect(b"a,b\n1,2\n", None, Some("t.csv")), DocumentFormat::Csv);
    Ok(())
}

#[test]
fn f12_ocr_mode_guard_and_share_quota() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = default_parse_cfg();

    // L2 -> share
    let d2 = resolve_ocr_policy(
        &cfg,
        &ParsePolicyCtx {
            auth_level: AuthLevel::new(2)?,
            sensitivity: Sensitivity::Normal,
            requested_ocr_mode: Some(Sensitivity::Share),
            batch: false,
            ocr_task: OcrTask::Markdown,
            tenant_key: None,
        },
    )?;
    assert_eq!(d2.effective_mode, Sensitivity::Share);
    assert!(!d2.upgraded_from_share);

    // L5 -> private
    let d5 = resolve_ocr_policy(
        &cfg,
        &ParsePolicyCtx {
            auth_level: AuthLevel::new(5)?,
            sensitivity: Sensitivity::Normal,
            requested_ocr_mode: Some(Sensitivity::Share),
            batch: false,
            ocr_task: OcrTask::Markdown,
            tenant_key: None,
        },
    )?;
    assert_eq!(d5.effective_mode, Sensitivity::Private);
    assert!(d5.upgraded_from_share);

    // L8 -> never allows share (upgrades to zdr)
    let d8 = resolve_ocr_policy(
        &cfg,
        &ParsePolicyCtx {
            auth_level: AuthLevel::new(8)?,
            sensitivity: Sensitivity::Normal,
            requested_ocr_mode: Some(Sensitivity::Share),
            batch: false,
            ocr_task: OcrTask::Markdown,
            tenant_key: None,
        },
    )?;
    assert_eq!(d8.effective_mode, Sensitivity::Zdr);
    assert!(d8.upgraded_from_share);

    // Airgap local_vllm never allows share even at L1
    let mut airgap = cfg.clone();
    airgap.ocr_engine = "local_vllm".into();
    airgap.default_ocr_mode = Sensitivity::Share;
    airgap.ocr_mode_by_auth_level.clear();
    let d_air = resolve_ocr_policy(&airgap, &ParsePolicyCtx::default())?;
    assert_eq!(d_air.effective_mode, Sensitivity::Private);
    assert_eq!(d_air.execution_target, OcrExecutionTarget::LocalVllm);

    // ZDR + batch=true rejected with zdr_batch_unsupported
    let zdr_batch = resolve_ocr_policy(
        &cfg,
        &ParsePolicyCtx {
            auth_level: AuthLevel::new(8)?,
            sensitivity: Sensitivity::Zdr,
            requested_ocr_mode: Some(Sensitivity::Zdr),
            batch: true,
            ocr_task: OcrTask::Markdown,
            tenant_key: None,
        },
    );
    assert!(matches!(
        zdr_batch,
        Err(OpenRagError::Validation(ref msg)) if msg.contains("zdr_batch_unsupported")
    ));

    // ShareQuotaTracker: 50 pages/day KST allowed, 51st rejected
    let tracker = ShareQuotaTracker::default();
    let kst_time = Utc.with_ymd_and_hms(2026, 11, 22, 1, 0, 0).single().ok_or("invalid timestamp")?;
    assert_eq!(tracker.try_consume_at("org-1", 50, kst_time)?, 0);
    let over = tracker.try_consume_at("org-1", 1, kst_time);
    assert!(matches!(over, Err(OpenRagError::Validation(ref msg)) if msg.contains("share_quota_exceeded")));
    Ok(())
}

fn build_multipage_test_pdf() -> Vec<u8> {
    let mut pdf = Vec::new();
    let mut offsets = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let objs: Vec<String> = vec![
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n".into(),
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>\nendobj\n".into(),
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 6 0 R >>\nendobj\n".into(),
        "4 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 7 0 R >>\nendobj\n".into(),
        "5 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 8 0 R >>\nendobj\n".into(),
        "6 0 obj\n<< /Length 31 >>\nstream\nBT (Alpha page one) Tj ET\nendstream\nendobj\n".into(),
        "7 0 obj\n<< /Length 30 >>\nstream\nBT (Beta page two) Tj ET\nendstream\nendobj\n".into(),
        "8 0 obj\n<< /Length 33 >>\nstream\nBT (Gamma page three) Tj ET\nendstream\nendobj\n".into(),
    ];

    for obj in &objs {
        offsets.push(pdf.len());
        pdf.extend_from_slice(obj.as_bytes());
    }
    let xref_pos = pdf.len();
    pdf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for off in offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n").as_bytes(),
    );
    pdf
}

#[tokio::test]
async fn f12_pdf_native_multipage_and_corrupt_rejection() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = default_parse_cfg();
    let pdf_bytes = build_multipage_test_pdf();
    let doc = parse_pdf_native(&pdf_bytes, Some("multi.pdf"), &cfg).await?;
    assert_eq!(doc.pages.len(), 3);
    assert!(doc.pages[0].raw_markdown.contains("Alpha"));
    assert!(doc.pages[1].raw_markdown.contains("Beta"));
    assert!(doc.pages[2].raw_markdown.contains("Gamma"));

    // Truncated broken PDF rejected
    let broken = parse_pdf_native(b"%PDF-1.4\n1 0 obj\n<< /Broken >>\n", Some("bad.pdf"), &cfg).await;
    assert!(matches!(broken, Err(OpenRagError::Validation(_))));
    Ok(())
}

#[test]
fn f36_parenthetical_alias_extraction() {
    let input = "제안요청서(RFP)와 사과(apple) 및 인공지능(Artificial Intelligence) (2026) 그리고 (p.3)";
    let aliases = extract_parenthetical_aliases(input);
    assert_eq!(
        aliases,
        vec![
            ("제안요청서".to_string(), "RFP".to_string()),
            ("사과".to_string(), "apple".to_string()),
            (
                "인공지능".to_string(),
                "Artificial Intelligence".to_string()
            ),
        ]
    );
}

#[tokio::test]
async fn f12_f13_image_ocr_webp_thumbnail_figure_crop_and_256d_embed() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = default_parse_cfg();
    let thumb_cfg = default_thumb_cfg();

    // 0-byte image rejected
    let empty_err = parse_image_with_ocr(
        b"",
        DocumentFormat::Png,
        Some("empty.png"),
        &cfg,
        &ParsePolicyCtx::default(),
        None,
        None,
    )
    .await;
    assert!(matches!(empty_err, Err(OpenRagError::Validation(_))));

    // Synthetic 1600x1200 PNG with colored figure block -> downscaled <= 1280 WebP + PNG crop
    let mut img = RgbImage::from_pixel(1600, 1200, Rgb([255, 255, 255]));
    for y in 200..500 {
        for x in 300..900 {
            img.put_pixel(x, y, Rgb([20, 80, 200]));
        }
    }
    let mut png_cur = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(img.clone()).write_to(&mut png_cur, ImageFormat::Png)?;
    let png_bytes = png_cur.into_inner();

    let parser = DocumentParser::new();
    let parsed = parser
        .parse(
            &png_bytes,
            Some("image/png"),
            Some("scan.png"),
            &cfg,
            &ParsePolicyCtx::default(),
        )
        .await?;
    assert_eq!(parsed.pages.len(), 1);
    assert!(!parsed.pages[0].blocks.is_empty());
    let first_bbox = parsed.pages[0].blocks[0]
        .bbox_norm_1000
        .ok_or("missing detected bbox")?;
    assert!(first_bbox.x_min >= 0 && first_bbox.x_max <= 999);

    let thumbs = Thumbnailer::new().render(&parsed, Some(&png_bytes), &thumb_cfg)?;
    assert_eq!(thumbs.len(), 1);
    let t = &thumbs[0];
    assert!(t.webp_bytes.starts_with(b"RIFF"));
    assert_eq!(&t.webp_bytes[8..12], b"WEBP");
    assert!(t.width.max(t.height) <= 1280);
    assert!(!t.figures.is_empty());
    assert_eq!(t.figures[0].filename, "page-001-fig-0.png");
    assert!((t.figures[0].png_bytes.len() as u64) <= 307_200);

    // Store in InMemoryBlobStore roundtrip
    let blob_store = InMemoryBlobStore::new();
    let uri = blob_store.put_bytes("org/doc/page-001.webp", &t.webp_bytes).await?;
    assert_eq!(blob_store.get_bytes(&uri).await?, t.webp_bytes);
    Ok(())
}

#[test]
fn f11_nested_hwpx_and_hml_tables_preserve_outer_cells_and_spans() -> Result<(), Box<dyn std::error::Error>> {
    let hwpx_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
    <hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"
            xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
      <hp:tbl>
        <hp:tr>
          <hp:tc colSpan="3" rowSpan="2">
            <hp:subList>
              <hp:p><hp:run><hp:t>OuterHeader</hp:t></hp:run></hp:p>
              <hp:tbl>
                <hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>InnerCell</hp:t></hp:run></hp:p></hp:subList></hp:tc></hp:tr>
              </hp:tbl>
            </hp:subList>
          </hp:tc>
          <hp:tc><hp:subList><hp:p><hp:run><hp:t>OuterRight</hp:t></hp:run></hp:p></hp:subList></hp:tc>
        </hp:tr>
        <hp:tr>
          <hp:tc><hp:subList><hp:p><hp:run><hp:t>SecondRowCell</hp:t></hp:run></hp:p></hp:subList></hp:tc>
        </hp:tr>
      </hp:tbl>
    </hs:sec>"#;
    let hwpx_bytes = build_test_hwpx(hwpx_xml)?;
    let hwpx_doc = parse_hwpx(&hwpx_bytes, Some("nested.hwpx"))?;
    let tbl = hwpx_doc.pages[0].tables_html.first().ok_or("missing HWPX table")?;
    assert!(tbl.contains("OuterHeader"), "missing OuterHeader in {tbl}");
    assert!(tbl.contains("InnerCell"), "missing InnerCell in {tbl}");
    assert!(tbl.contains("OuterRight"), "missing OuterRight in {tbl}");
    assert!(tbl.contains("SecondRowCell"), "missing SecondRowCell in {tbl}");
    assert!(tbl.contains("colspan=\"3\""), "missing colspan=3 in {tbl}");
    assert!(tbl.contains("rowspan=\"2\""), "missing rowspan=2 in {tbl}");

    let hml_xml = br#"<?xml version="1.0" encoding="UTF-8"?>
    <HWPML Version="2.9">
      <HEAD><TITLE>Nested HML</TITLE></HEAD>
      <BODY><SECTION>
        <TABLE>
          <ROW>
            <CELL ColSpan="3" RowSpan="2">
              <P><TEXT><CHAR>OuterHML</CHAR></TEXT></P>
              <TABLE><ROW><CELL><P><TEXT><CHAR>InnerHML</CHAR></TEXT></P></CELL></ROW></TABLE>
            </CELL>
            <CELL><P><TEXT><CHAR>OuterRightHML</CHAR></TEXT></P></CELL>
          </ROW>
        </TABLE>
      </SECTION></BODY>
    </HWPML>"#;
    let hml_doc = parse_hml(hml_xml, Some("nested.hml"))?;
    let hml_tbl = hml_doc.pages[0].tables_html.first().ok_or("missing HML table")?;
    assert!(hml_tbl.contains("OuterHML"), "missing OuterHML in {hml_tbl}");
    assert!(hml_tbl.contains("InnerHML"), "missing InnerHML in {hml_tbl}");
    assert!(hml_tbl.contains("OuterRightHML"), "missing OuterRightHML in {hml_tbl}");
    assert!(hml_tbl.contains("colspan=\"3\""), "missing colspan=3 in {hml_tbl}");
    assert!(hml_tbl.contains("rowspan=\"2\""), "missing rowspan=2 in {hml_tbl}");
    Ok(())
}

#[test]
fn f11_sanitize_table_html_is_case_insensitive_and_attribute_aware() {
    let input = "<TABLE><THEAD><TR><TH class=\"h\">Head<BR class=\"x\">Line</TH></TR></THEAD><TBODY id=\"b\"><TR><TD><B class=\"bold\">Val</B></TD></TR></TBODY></TABLE>";
    let out = sanitize_table_html(input);
    let lower = out.to_ascii_lowercase();
    assert!(!lower.contains("<th"), "found <th in {out}");
    assert!(!lower.contains("<thead"), "found <thead in {out}");
    assert!(!lower.contains("<br"), "found <br in {out}");
    assert!(!lower.contains("<b"), "found <b in {out}");
    assert!(!lower.contains("<tbody"), "found <tbody in {out}");
    assert!(out.contains("<table>"));
    assert!(out.contains("<tr>"));
    assert!(out.contains("<td class=\"h\">Head Line</td>"));
    assert!(out.contains("<td>Val</td>"));

    let self_closing = "<TABLE><THEAD/><TBODY/><TFOOT/><TR><TH data-ref=\"a/b\"/><TD><B/>X<B class=\"k\"/>Y</TD></TR></TABLE>";
    let sc_out = sanitize_table_html(self_closing);
    assert_eq!(
        sc_out,
        "<table><tr><td data-ref=\"a/b\"></td><td>XY</td></tr></table>"
    );
}

#[test]
fn f11_unclosed_xml_at_eof_returns_validation_error() -> Result<(), Box<dyn std::error::Error>> {
    let hwpx_bytes = build_test_hwpx("<hs:sec><hp:p><hp:t>unclosed")?;
    let hwpx_res = parse_hwpx(&hwpx_bytes, Some("truncated.hwpx"));
    assert!(matches!(hwpx_res, Err(OpenRagError::Validation(_))));

    let hml_res = parse_hml(b"<HWPML><BODY><SECTION><P><TEXT><CHAR>unclosed", Some("truncated.hml"));
    assert!(matches!(hml_res, Err(OpenRagError::Validation(_))));
    Ok(())
}

#[test]
fn f11_hwp5_inline_controls_consume_16_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let cursor = Cursor::new(Vec::new());
    let mut cfb = cfb::CompoundFile::create(cursor)?;
    {
        let mut hdr = [0u8; 256];
        let sig = b"HWP Document File";
        hdr[..sig.len()].copy_from_slice(sig);
        let mut stream = cfb.create_stream("FileHeader")?;
        stream.write_all(&hdr)?;
    }
    {
        cfb.create_storage("BodyText")?;
        let mut units: Vec<u16> = Vec::new();
        units.extend("Before".encode_utf16());
        // Inline control code 4 + 7 dummy WCHARs ('X') = 16 bytes
        units.push(4);
        units.extend(std::iter::repeat_n(b'X' as u16, 7));
        // Inline control code 19 + 7 dummy WCHARs ('Y') = 16 bytes
        units.push(19);
        units.extend(std::iter::repeat_n(b'Y' as u16, 7));
        units.extend("After".encode_utf16());

        let mut payload = Vec::with_capacity(units.len() * 2);
        for u in units {
            payload.extend_from_slice(&u.to_le_bytes());
        }
        let rec_hdr: u32 = 67 | ((payload.len() as u32) << 20);
        let mut raw_rec = Vec::new();
        raw_rec.extend_from_slice(&rec_hdr.to_le_bytes());
        raw_rec.extend_from_slice(&payload);

        let mut stream = cfb.create_stream("BodyText/Section0")?;
        stream.write_all(&raw_rec)?;
    }
    let bytes = cfb.into_inner().into_inner();
    let doc = parse_hwp5(&bytes, Some("inline.hwp"))?;
    assert_eq!(doc.full_markdown.trim(), "BeforeAfter");
    Ok(())
}

#[test]
fn f13_encode_png_bounded_strictly_respects_max_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let mut noisy = RgbImage::new(1280, 1280);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    for p in noisy.pixels_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (state >> 56) as u8;
        let g = (state >> 48) as u8;
        let b = (state >> 40) as u8;
        *p = Rgb([r, g, b]);
    }

    let bounded_64k = encode_png_bounded(&noisy, 1280, 65_536)?;
    assert!(bounded_64k.len() <= 65_536, "got {} bytes", bounded_64k.len());

    let bounded_2k = encode_png_bounded(&noisy, 1200, 2048)?;
    assert!(bounded_2k.len() <= 2048, "got {} bytes", bounded_2k.len());

    let impossible = encode_png_bounded(&noisy, 128, 10);
    assert!(matches!(impossible, Err(OpenRagError::Validation(_))));
    Ok(())
}

#[tokio::test]
async fn f12_pdf_non_utf8_dictionary_bytes_do_not_drop_page() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = default_parse_cfg();
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /CustomNote (\xFF\xFE\x80) /Contents 4 0 R >>\nendobj\n");
    pdf.extend_from_slice(b"4 0 obj\n<< /Length 30 >>\nstream\nBT (Latin1 dict page) Tj ET\nendstream\nendobj\n");
    pdf.extend_from_slice(b"trailer\n<< /Root 1 0 R >>\n%%EOF\n");

    let doc = parse_pdf_native(&pdf, Some("latin1-dict.pdf"), &cfg).await?;
    assert_eq!(doc.pages.len(), 1);
    assert!(doc.pages[0].raw_markdown.contains("Latin1 dict page"));
    Ok(())
}
