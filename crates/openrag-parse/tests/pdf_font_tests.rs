//! PDF text through fonts: ToUnicode CMaps (bfchar, bfrange, 2-byte CID codes), object streams,
//! ActualText spans and form XObjects. (Fonts that cannot be decoded are tested inside the crate,
//! where the `pdftotext` fallback does not run.)

use openrag_core::PipelineConfig;
use openrag_parse::parse_pdf_with_scan_pages;

type R = Result<(), Box<dyn std::error::Error>>;

fn cfg() -> Result<openrag_core::ParseConfig, Box<dyn std::error::Error>> {
    let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/pipeline.default.yaml"));
    Ok(PipelineConfig::from_yaml_file(path)?.parse)
}

/// UTF-16BE hex of a string, as a ToUnicode destination.
fn u16hex(s: &str) -> String {
    s.encode_utf16().map(|u| format!("{u:04X}")).collect()
}

fn stream(body: &str) -> String {
    format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len())
}

/// Writes numbered objects (1-based) as a PDF; object 1 must be the catalog.
fn pdf(objs: &[String]) -> Vec<u8> {
    let mut out = String::from("%PDF-1.7\n");
    for (i, o) in objs.iter().enumerate() {
        out.push_str(&format!("{} 0 obj\n{o}\nendobj\n", i + 1));
    }
    out.push_str("trailer\n<< /Root 1 0 R >>\n%%EOF\n");
    out.into_bytes()
}

/// A one-page PDF: `resources` and `content` for the page, plus extra objects from number 5.
fn one_page(resources: &str, content: &str, extra: &[String]) -> Vec<u8> {
    let mut objs = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 /Resources ".to_string() + resources + " >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>".to_string(),
        stream(content),
    ];
    objs.extend_from_slice(extra);
    pdf(&objs)
}

async fn text_of(bytes: &[u8]) -> Result<(String, Vec<usize>), Box<dyn std::error::Error>> {
    let (doc, scan) = parse_pdf_with_scan_pages(bytes, None, &cfg()?).await?;
    Ok((doc.full_markdown, scan))
}

#[tokio::test]
async fn subset_truetype_with_tounicode_bfchar() -> R {
    // LibreOffice style: one-byte glyph codes <01><02>..., a ToUnicode map, a space glyph.
    let cmap = format!(
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n1 begincodespacerange <00> <FF> endcodespacerange\n\
         5 beginbfchar <01> <{}> <02> <{}> <03> <{}> <04> <{}> <05> <0020> endbfchar\nendcmap end end",
        u16hex("출"), u16hex("장"), u16hex("비"), u16hex("\0")
    );
    let content = "BT /F1 12 Tf 56 780 Td <010203> Tj ET\nBT /F1 12 Tf 56 760 Td [<01>2<02>-1<05><03>] TJ <04> Tj ET";
    let bytes = one_page(
        "<< /Font << /F1 5 0 R >> >>",
        content,
        &["<< /Type /Font /Subtype /TrueType /BaseFont /AAAAAA+NanumGothic /ToUnicode 6 0 R >>".into(), stream(&cmap)],
    );
    let (text, scan) = text_of(&bytes).await?;
    // The second line is a new line (y moved); the NUL from <04> never reaches storage.
    assert_eq!(text, "출장비\n출장 비");
    assert!(scan.is_empty());
    Ok(())
}

#[tokio::test]
async fn cid_font_two_byte_codes_with_bfrange() -> R {
    // Word / Hangul style: Type0 Identity-H, two-byte codes, ranges and an array range.
    let cmap = format!(
        "begincmap 1 begincodespacerange <0000> <FFFF> endcodespacerange\n\
         2 beginbfrange <0010> <0012> <{}> <0020> <0021> [<{}> <{}>] endbfrange\n\
         1 beginbfchar <0003> <0020> endbfchar endcmap",
        u16hex("가"),
        u16hex("휴"),
        u16hex("가"),
    );
    // 가, 각, 갂 are U+AC00.. consecutive; then a space, then 휴가.
    let content = "BT /C0 10 Tf 1 0 0 1 72 700 Tm <001000110012> Tj <0003> Tj <00200021> Tj ET";
    let bytes = one_page(
        "<< /Font << /C0 5 0 R >> >>",
        content,
        &[
            "<< /Type /Font /Subtype /Type0 /BaseFont /MalgunGothic /Encoding /Identity-H /DescendantFonts [7 0 R] /ToUnicode 6 0 R >>".into(),
            stream(&cmap),
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /MalgunGothic >>".into(),
        ],
    );
    assert_eq!(text_of(&bytes).await?.0, "가각갂 휴가");
    Ok(())
}

#[tokio::test]
async fn pages_and_fonts_inside_object_streams() -> R {
    // PDF 1.5+: the page, its fonts and the ToUnicode reference live in an /ObjStm.
    let cmap = format!("1 begincodespacerange <00> <FF> endcodespacerange 2 beginbfchar <41> <{}> <42> <{}> endbfchar", u16hex("연"), u16hex("차"));
    let members = [
        (3, "<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /Font << /F1 6 0 R >> >> >>".to_string()),
        (6, "<< /Type /Font /Subtype /TrueType /ToUnicode 7 0 R >>".to_string()),
    ];
    let mut header = String::new();
    let mut body = String::new();
    for (num, text) in &members {
        header.push_str(&format!("{num} {} ", body.len()));
        body.push_str(text);
        body.push(' ');
    }
    let objstm = format!("<< /Type /ObjStm /N 2 /First {} /Length {} >>\nstream\n{header}{body}\nendstream", header.len(), header.len() + body.len());
    // Objects 3 and 6 exist only inside the object stream (5).
    let top = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>".to_string()),
        (2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string()),
        (4, stream("BT /F1 12 Tf 72 700 Td (AB) Tj ET")),
        (5, objstm),
        (7, stream(&cmap)),
    ];
    let mut out = String::from("%PDF-1.7\n");
    for (num, o) in &top {
        out.push_str(&format!("{num} 0 obj\n{o}\nendobj\n"));
    }
    out.push_str("trailer\n<< /Root 1 0 R >>\n%%EOF\n");
    let bytes = out.into_bytes();
    assert_eq!(text_of(&bytes).await?.0, "연차");
    Ok(())
}

#[tokio::test]
async fn actual_text_spans_and_form_xobjects() -> R {
    // Chrome writes a space as a glyph inside `/Span <</ActualText ( )>> BDC … EMC`.
    let cmap = format!("1 begincodespacerange <00> <FF> endcodespacerange 3 beginbfchar <01> <{}> <02> <{}> <03> <0078> endbfchar", u16hex("재"), u16hex("택"));
    let content = "BT /F1 16 Tf 1 0 0 -1 8 82 Tm <01> Tj /Span<</ActualText ( ) >> BDC 4 0 Td <03> Tj EMC 4 0 Td <02> Tj ET\n/Fm0 Do";
    let form = "<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /Font << /F1 5 0 R >> >> /Length 34 >>\nstream\nBT /F1 9 Tf 1 0 0 1 8 300 Tm <0102> Tj ET\nendstream";
    let bytes = one_page(
        "<< /Font << /F1 5 0 R >> /XObject << /Fm0 7 0 R >> >>",
        content,
        &["<< /Type /Font /Subtype /Type3 /ToUnicode 6 0 R >>".into(), stream(&cmap), form.into()],
    );
    assert_eq!(text_of(&bytes).await?.0, "재 택\n재택");
    Ok(())
}

#[tokio::test]
async fn unicode_cmap_encoding_and_winansi() -> R {
    // A CID font whose encoding is UCS-2 itself, and a simple font in WinAnsi.
    let content = format!("BT /K0 12 Tf 72 700 Td <{}> Tj ET BT /F1 12 Tf 72 680 Td (Caf\\351 \\223ok\\224) Tj ET", u16hex("휴일"));
    let bytes = one_page(
        "<< /Font << /K0 5 0 R /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >> >> >>",
        &content,
        &["<< /Type /Font /Subtype /Type0 /Encoding /UniKS-UCS2-H /DescendantFonts [] >>".into()],
    );
    assert_eq!(text_of(&bytes).await?.0, "휴일\nCafé “ok”");
    Ok(())
}
