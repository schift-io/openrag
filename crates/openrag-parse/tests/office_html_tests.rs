use std::io::{Cursor, Write};

use openrag_parse::{html_to_markdown, parse_docx, parse_html, parse_pptx, DocumentFormat};
use zip::write::SimpleFileOptions;

type R = Result<(), Box<dyn std::error::Error>>;

fn zip_of(parts: &[(&str, &str)]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, body) in parts {
        w.start_file(*name, SimpleFileOptions::default())?;
        w.write_all(body.as_bytes())?;
    }
    Ok(w.finish()?.into_inner())
}

const W: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main""#;

#[test]
fn docx_headings_lists_tables_and_page_breaks() -> R {
    let styles = format!(
        r#"<w:styles {W}>
          <w:style w:type="paragraph" w:styleId="1"><w:name w:val="heading 1"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
          <w:style w:type="paragraph" w:styleId="a3"><w:name w:val="제목 2"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
        </w:styles>"#
    );
    let body = format!(
        r#"<w:document {W}><w:body>
          <w:p><w:pPr><w:pStyle w:val="1"/><w:tabs><w:tab w:val="left" w:pos="720"/></w:tabs></w:pPr><w:r><w:t>출장비 규정</w:t></w:r></w:p>
          <w:p><w:r><w:t xml:space="preserve">숙박비 한도는 </w:t></w:r><w:r><w:t>1박 120,000원 &amp; 일비 30,000원</w:t></w:r></w:p>
          <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>영수증 첨부</w:t></w:r></w:p>
          <w:p><w:r><w:delText>지운 글</w:delText><w:instrText>PAGE</w:instrText></w:r></w:p>
          <w:tbl><w:tr><w:tc><w:p><w:r><w:t>지역</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>한도</w:t></w:r></w:p></w:tc></w:tr>
                 <w:tr><w:tc><w:p><w:r><w:t>부산</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>120,000</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          <w:p><w:r><w:br w:type="page"/></w:r></w:p>
          <w:p><w:pPr><w:pStyle w:val="a3"/></w:pPr><w:r><w:lastRenderedPageBreak/><w:t>부칙</w:t></w:r></w:p>
          <w:p><w:r><w:t>2026년 1월 1일부터 시행</w:t></w:r></w:p>
        </w:body></w:document>"#
    );
    let bytes = zip_of(&[("word/document.xml", &body), ("word/styles.xml", &styles)])?;
    assert_eq!(DocumentFormat::detect(&bytes, None, None), DocumentFormat::Docx);
    assert_eq!(DocumentFormat::detect(&bytes, None, Some("규정.docx")), DocumentFormat::Docx);

    let doc = parse_docx(&bytes, Some("출장비_규정.docx"))?;
    assert_eq!(doc.format, DocumentFormat::Docx);
    assert_eq!(doc.title, "출장비_규정");
    assert_eq!(doc.pages.len(), 2, "{:#?}", doc.pages);
    let p1 = &doc.pages[0].raw_markdown;
    assert!(p1.starts_with("# 출장비 규정\n\n숙박비 한도는 1박 120,000원 & 일비 30,000원"), "{p1}");
    assert!(p1.contains("- 영수증 첨부"), "{p1}");
    assert!(p1.contains("| 지역 | 한도 |\n| 부산 | 120,000 |"), "{p1}");
    assert!(!p1.contains("지운 글") && !p1.contains("PAGE"), "{p1}");
    assert_eq!(doc.pages[1].page_no, 2);
    assert_eq!(doc.pages[1].raw_markdown, "## 부칙\n\n2026년 1월 1일부터 시행");
    Ok(())
}

#[test]
fn docx_without_text_is_refused() -> R {
    let body = format!(r#"<w:document {W}><w:body><w:p/></w:body></w:document>"#);
    let bytes = zip_of(&[("word/document.xml", &body)])?;
    let err = parse_docx(&bytes, Some("빈.docx")).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(err.contains("no text"), "{err}");
    Ok(())
}

const P: &str = r#"xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main""#;

fn slide(title: &str, body: &str) -> String {
    format!(
        r#"<p:sld {P}><p:cSld><p:spTree>
          <p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
            <p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
          <p:sp><p:nvSpPr><p:cNvPr id="3" name="Body"/><p:cNvSpPr/><p:nvPr><p:ph idx="1"/></p:nvPr></p:nvSpPr>
            <p:txBody><a:p><a:r><a:t>{body}</a:t></a:r><a:br><a:rPr/></a:br><a:r><a:t>둘째 줄</a:t></a:r></a:p></p:txBody></p:sp>
          <p:sp><p:nvSpPr><p:cNvPr id="4" name="Num"/><p:cNvSpPr/><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr>
            <p:txBody><a:p><a:fld type="slidenum"><a:t>7</a:t></a:fld></a:p></p:txBody></p:sp>
        </p:spTree></p:cSld></p:sld>"#
    )
}

#[test]
fn pptx_slides_in_presentation_order_with_notes() -> R {
    // slide2.xml is shown first: the order comes from presentation.xml, not the file names.
    let pres = format!(
        r#"<p:presentation {P}><p:sldIdLst><p:sldId id="256" r:id="rId3"/><p:sldId id="257" r:id="rId2"/></p:sldIdLst></p:presentation>"#
    );
    let pres_rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
        <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
      </Relationships>"#;
    let slide2_rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
        <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>
      </Relationships>"#;
    let notes = format!(
        r#"<p:notes {P}><p:cSld><p:spTree>
          <p:sp><p:nvSpPr><p:cNvPr id="2" name="Img"/><p:cNvSpPr/><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr></p:sp>
          <p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes"/><p:cNvSpPr/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
            <p:txBody><a:p><a:r><a:t>발표자 메모: 예산은 3월 확정</a:t></a:r></a:p></p:txBody></p:sp>
        </p:spTree></p:cSld></p:notes>"#
    );
    let table_slide = format!(
        r#"<p:sld {P}><p:cSld><p:spTree><p:graphicFrame><a:graphic><a:graphicData><a:tbl>
          <a:tr><a:tc><a:txBody><a:p><a:r><a:t>항목</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>금액</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
          <a:tr><a:tc><a:txBody><a:p><a:r><a:t>숙박</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>120,000</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
        </a:tbl></a:graphicData></a:graphic></p:graphicFrame></p:spTree></p:cSld></p:sld>"#
    );
    let first = slide("2026 예산 보고", "총액 1억 2천만 원");
    let bytes = zip_of(&[
        ("ppt/presentation.xml", &pres),
        ("ppt/_rels/presentation.xml.rels", pres_rels),
        ("ppt/slides/slide1.xml", &table_slide),
        ("ppt/slides/slide2.xml", &first),
        ("ppt/slides/_rels/slide2.xml.rels", slide2_rels),
        ("ppt/notesSlides/notesSlide1.xml", &notes),
    ])?;
    assert_eq!(DocumentFormat::detect(&bytes, None, None), DocumentFormat::Pptx);

    let doc = parse_pptx(&bytes, Some("예산.pptx"))?;
    assert_eq!(doc.format, DocumentFormat::Pptx);
    assert_eq!(doc.pages.len(), 2);
    assert_eq!(doc.pages[0].page_no, 1);
    assert_eq!(
        doc.pages[0].raw_markdown,
        "# 2026 예산 보고\n\n총액 1억 2천만 원\n둘째 줄\n\n발표자 메모: 예산은 3월 확정"
    );
    assert_eq!(doc.pages[1].raw_markdown, "| 항목 | 금액 |\n| 숙박 | 120,000 |");
    Ok(())
}

#[test]
fn html_text_without_scripts_styles_or_markup() -> R {
    let html = r#"<!DOCTYPE html><html><head><meta charset="utf-8"><title>사내 공지</title>
      <style>body { color: red }</style><script>var x = "<p>not text</p>";</script></head>
      <body><nav>홈 &gt; 공지</nav><!-- 주석 -->
      <h1>보안 서약</h1><p>출입증은&nbsp;퇴사일에   반납합니다.<br>문의: 총무팀</p>
      <ul><li>노트북 반납</li><li>계정 &amp; 메일 정지</li></ul>
      <table><tr><th>항목</th><th>기한</th></tr><tr><td>출입증</td><td>당일</td></tr></table>
      <p>&#49;&#x32;3 &copy; ACME</p></body></html>"#;
    let (title, md) = html_to_markdown(html);
    assert_eq!(title.as_deref(), Some("사내 공지"));
    assert_eq!(
        md,
        "홈 > 공지\n\n# 보안 서약\n\n출입증은 퇴사일에 반납합니다.\n문의: 총무팀\n\n- 노트북 반납\n- 계정 & 메일 정지\n\n| 항목 | 기한 |\n| 출입증 | 당일 |\n\n123 © ACME"
    );
    assert_eq!(DocumentFormat::detect(html.as_bytes(), None, None), DocumentFormat::Html);
    assert_eq!(DocumentFormat::detect(html.as_bytes(), Some("text/html; charset=utf-8"), None), DocumentFormat::Html);
    Ok(())
}

#[test]
fn html_in_euc_kr_is_decoded() -> R {
    let (encoded, _, _) = encoding_rs::EUC_KR.encode("<html><head><meta charset=\"euc-kr\"></head><body><p>퇴직금 지급 기준</p></body></html>");
    let doc = parse_html(&encoded, Some("퇴직금.html"))?;
    assert_eq!(doc.full_markdown, "퇴직금 지급 기준");
    assert_eq!(doc.title, "퇴직금");
    // No charset declared and not UTF-8: read as EUC-KR.
    let (bare, _, _) = encoding_rs::EUC_KR.encode("<p>연차 휴가</p>");
    assert_eq!(parse_html(&bare, None)?.full_markdown, "연차 휴가");
    Ok(())
}
