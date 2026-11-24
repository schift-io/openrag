use std::io::{Cursor, Write};

use openrag_parse::{parse_spreadsheet, read_sheet, DocumentFormat};
use zip::write::SimpleFileOptions;

type R = Result<(), Box<dyn std::error::Error>>;

fn zip_of(parts: &[(&str, String)]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, body) in parts {
        w.start_file(*name, SimpleFileOptions::default())?;
        w.write_all(body.as_bytes())?;
    }
    Ok(w.finish()?.into_inner())
}

const NS: &str = r#"xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships""#;

/// 매출 sheet: a title row, a header, 30 data rows (date, 지점, 금액); a hidden 코드 sheet.
fn workbook() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let strings = ["2024년 지점별 매출", "날짜", "지점", "금액", "강남", "판교", "부산", "코드", "이름", "A01", "본사"];
    let sst = format!(
        "<sst {NS}>{}</sst>",
        strings.iter().map(|s| format!("<si><t>{s}</t></si>")).collect::<String>()
    );
    let mut rows = String::from(r#"<row r="1"><c r="A1" t="s"><v>0</v></c></row>"#);
    rows.push_str(r#"<row r="3"><c r="A3" t="s"><v>1</v></c><c r="B3" t="s"><v>2</v></c><c r="C3" t="s"><v>3</v></c></row>"#);
    for i in 0..30 {
        let branch = [4, 5, 6][i % 3];
        let n = 4 + i;
        // 45292 = 2024-01-01; style 1 shows a date.
        rows.push_str(&format!(
            r#"<row r="{n}"><c r="A{n}" s="1"><v>{}</v></c><c r="B{n}" t="s"><v>{branch}</v></c><c r="C{n}"><f>B{n}*2</f><v>{}</v></c></row>"#,
            45292 + i,
            (i + 1) * 1000
        ));
    }
    let sheet1 = format!("<worksheet {NS}><sheetData>{rows}</sheetData></worksheet>");
    let sheet2 = format!(
        r#"<worksheet {NS}><sheetData><row r="1"><c r="A1" t="s"><v>7</v></c><c r="B1" t="s"><v>8</v></c></row><row r="2"><c r="A2" t="s"><v>9</v></c><c r="B2" t="inlineStr"><is><t>본사 &amp; 지원</t></is></c></row></sheetData></worksheet>"#
    );
    let wb = format!(
        r#"<workbook {NS}><workbookPr/><sheets><sheet name="매출" sheetId="1" r:id="rId1"/><sheet name="코드" sheetId="2" state="hidden" r:id="rId2"/></sheets></workbook>"#
    );
    let rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="x" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="x" Target="/xl/worksheets/sheet2.xml"/></Relationships>"#;
    let styles = format!(r#"<styleSheet {NS}><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>"#);
    zip_of(&[
        ("xl/workbook.xml", wb),
        ("xl/_rels/workbook.xml.rels", rels.to_string()),
        ("xl/sharedStrings.xml", sst),
        ("xl/styles.xml", styles),
        ("xl/worksheets/sheet1.xml", sheet1),
        ("xl/worksheets/sheet2.xml", sheet2),
    ])
}

#[test]
fn excel_becomes_catalog_cards() -> R {
    let bytes = workbook()?;
    assert_eq!(DocumentFormat::detect(&bytes, None, None), DocumentFormat::Xlsx);
    let doc = parse_spreadsheet(&bytes, DocumentFormat::Xlsx, Some("매출실적.xlsx"))?;
    assert_eq!(doc.pages.len(), 3, "{:#?}", doc.pages);
    let overview = &doc.pages[0].raw_markdown;
    assert!(overview.contains("엑셀 파일 \"매출실적.xlsx\""), "{overview}");
    assert!(overview.contains("매출 (30행)") && overview.contains("코드·숨김 (1행)"), "{overview}");
    assert!(overview.contains("rag_read_sheet"), "{overview}");

    let card = &doc.pages[1].raw_markdown;
    assert!(card.starts_with("## 시트 \"매출\" (30행 × 3열)"), "{card}");
    assert!(card.contains("제목: 2024년 지점별 매출"), "{card}");
    assert!(card.contains("열: 날짜 | 지점 | 금액"), "{card}");
    assert!(card.contains("날짜 2024-01-01 ~ 2024-01-30"), "{card}");
    assert!(card.contains("금액 1000 ~ 30000"), "{card}");
    assert!(card.contains("지점 — 강남(10), 부산(10), 판교(10)"), "{card}");
    assert!(card.contains("| 2024-01-01 | 강남 | 1000 |"), "{card}");
    assert!(!card.contains("| 2024-01-11 |"), "only the first 10 rows: {card}");
    // 금액 is numeric (a range); a column of all-different text gets two examples, not a list.
    let many: Vec<(String, String)> = (0..40).map(|i| (format!("{}", i), format!("010-{:04}-5678", 1000 + i))).collect();
    let csv = format!("번호,연락처\n{}\n", many.iter().map(|(a, b)| format!("{a},{b}")).collect::<Vec<_>>().join("\n"));
    let phones = parse_spreadsheet(csv.as_bytes(), DocumentFormat::Csv, Some("연락처.csv"))?;
    let pc = &phones.pages[1].raw_markdown;
    assert!(pc.contains("연락처 — 거의 모두 다른 값 40개 (예: 010-1000-5678, 010-1001-5678)"), "{pc}");

    let hidden = &doc.pages[2].raw_markdown;
    assert!(hidden.starts_with("## 시트 \"코드\" (숨긴 시트)"), "{hidden}");
    assert!(hidden.contains("본사 & 지원"), "{hidden}");
    Ok(())
}

#[test]
fn rows_are_read_on_demand() -> R {
    let bytes = workbook()?;
    let s = read_sheet(&bytes, DocumentFormat::Xlsx, None, Some("매출"), 28, 10)?;
    assert_eq!(s.sheets, ["매출", "코드"]);
    assert_eq!(s.header, ["날짜", "지점", "금액"]);
    assert_eq!(s.total_rows, 30);
    assert_eq!(s.rows, [vec!["2024-01-29", "판교", "29000"], vec!["2024-01-30", "부산", "30000"]]);
    let err = read_sheet(&bytes, DocumentFormat::Xlsx, None, Some("없는시트"), 0, 5).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(err.contains("sheets: 매출, 코드"), "{err}");
    Ok(())
}

#[test]
fn csv_in_euc_kr_with_quotes_and_tabs() -> R {
    let text = "사번\t이름\t부서\n1001\t\"김, 철수\"\t영업\n1002\t이영희\t영업\n1003\t\"박 \"\"민\"\"\"\t개발\n";
    let (bytes, _, _) = encoding_rs::EUC_KR.encode(text);
    assert_eq!(DocumentFormat::detect(&bytes, None, Some("직원.tsv")), DocumentFormat::Csv);
    let doc = parse_spreadsheet(&bytes, DocumentFormat::Csv, Some("직원.tsv"))?;
    let card = &doc.pages[1].raw_markdown;
    assert!(card.starts_with("## 시트 \"직원\" (3행 × 3열)"), "{card}");
    assert!(card.contains("열: 사번 | 이름 | 부서"), "{card}");
    assert!(card.contains("부서 — 영업(2), 개발"), "{card}");
    assert!(card.contains("| 1001 | 김, 철수 | 영업 |"), "{card}");
    let s = read_sheet(&bytes, DocumentFormat::Csv, Some("직원.tsv"), None, 2, 5)?;
    assert_eq!(s.rows, [vec!["1003", "박 \"민\"", "개발"]]);
    Ok(())
}
