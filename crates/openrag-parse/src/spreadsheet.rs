//! Spreadsheets (.xlsx, .csv/.tsv) are not split into passages cell by cell. Ingestion stores a
//! catalog card per workbook and per sheet: title rows, column names, size, value ranges, frequent
//! values and the first rows, so search can tell what the file is about. The cells themselves are
//! read on demand ([`read_sheet`]) from the original file.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use chrono::{Duration, NaiveDate};
use openrag_core::OpenRagError;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesRef, BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use zip::ZipArchive;

use crate::aliases::extract_parenthetical_aliases;
use crate::model::{sha256_hex_bytes, sha256_hex_str, DocumentFormat, ParsedDocument, ParsedPage};

const MAX_PART_BYTES: u64 = 64 * 1024 * 1024;
/// Rows looked at for a sheet's statistics.
const STAT_ROWS: usize = 200_000;
/// First rows shown on a card.
const SAMPLE_ROWS: usize = 10;
/// Frequent values listed per text column.
const TOP_VALUES: usize = 12;
/// A card stops growing here (characters).
const CARD_LIMIT: usize = 8_000;

#[derive(Debug, Clone, PartialEq)]
pub struct Sheet {
    pub name: String,
    pub hidden: bool,
    pub rows: Vec<Vec<String>>,
}

// ---------------------------------------------------------------------------
// XLSX
// ---------------------------------------------------------------------------

fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn attr(e: &BytesStart<'_>, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| local(a.key.as_ref()) == key)
        .and_then(|a| a.normalized_value(XmlVersion::Implicit1_0).ok().map(|v| v.into_owned()))
}

fn prefixed_attr(e: &BytesStart<'_>, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| {
            let k: &str = a.key.as_ref();
            k.contains(':') && local(k) == key
        })
        .and_then(|a| a.normalized_value(XmlVersion::Implicit1_0).ok().map(|v| v.into_owned()))
}

fn general_ref(r: &BytesRef<'_>) -> String {
    if let Ok(Some(ch)) = r.resolve_char_ref() {
        return ch.to_string();
    }
    resolve_predefined_entity(r.as_ref()).map(ToString::to_string).unwrap_or_default()
}

type Archive<'a> = ZipArchive<Cursor<&'a [u8]>>;

fn read_part(archive: &mut Archive<'_>, name: &str) -> Result<Option<String>, OpenRagError> {
    let file = match archive.by_name(name) {
        Ok(f) => f,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(e) => return Err(OpenRagError::Validation(format!("cannot read {name}: {e}"))),
    };
    if file.size() > MAX_PART_BYTES {
        return Err(OpenRagError::Validation(format!("{name} is larger than 64 MiB")));
    }
    let mut out = String::new();
    file.take(MAX_PART_BYTES)
        .read_to_string(&mut out)
        .map_err(|e| OpenRagError::Validation(format!("cannot read {name}: {e}")))?;
    Ok(Some(out))
}

fn shared_strings(xml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let (mut cur, mut in_t, mut in_phonetic) = (String::new(), false, false);
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                "si" => cur.clear(),
                "t" => in_t = !in_phonetic,
                "rPh" => in_phonetic = true,
                _ => {}
            },
            Ok(Event::Empty(e)) if local(e.name().as_ref()) == "si" => out.push(String::new()),
            Ok(Event::Text(t)) if in_t => cur.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(r)) if in_t => cur.push_str(&general_ref(&r)),
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                "t" => in_t = false,
                "rPh" => in_phonetic = false,
                "si" => out.push(std::mem::take(&mut cur)),
                _ => {}
            },
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

fn is_date_format(id: u32, code: Option<&str>) -> bool {
    if matches!(id, 14..=22 | 27..=36 | 45..=47 | 50..=58) {
        return true;
    }
    let Some(code) = code else { return false };
    // Ignore quoted text and [colour]/[locale] sections, then look for date/time letters.
    let mut plain = String::new();
    let (mut quoted, mut bracket) = (false, false);
    for c in code.chars() {
        match c {
            '"' => quoted = !quoted,
            '[' if !quoted => bracket = true,
            ']' if !quoted => bracket = false,
            _ if !quoted && !bracket => plain.push(c.to_ascii_lowercase()),
            _ => {}
        }
    }
    plain.contains('y') || plain.contains('d') || (plain.contains('m') && !plain.contains('0') && !plain.contains('#'))
        || plain.contains("hh")
}

/// Style index -> whether the cell shows a date.
fn date_styles(xml: &str) -> Vec<bool> {
    let mut formats: HashMap<u32, String> = HashMap::new();
    let mut xfs: Vec<u32> = Vec::new();
    let mut in_cell_xfs = false;
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e) | Event::Empty(e)) => match local(e.name().as_ref()) {
                "numFmt" => {
                    if let (Some(id), Some(code)) = (attr(&e, "numFmtId").and_then(|v| v.parse().ok()), attr(&e, "formatCode")) {
                        formats.insert(id, code);
                    }
                }
                "cellXfs" => in_cell_xfs = true,
                "xf" if in_cell_xfs => xfs.push(attr(&e, "numFmtId").and_then(|v| v.parse().ok()).unwrap_or(0)),
                _ => {}
            },
            Ok(Event::End(e)) if local(e.name().as_ref()) == "cellXfs" => in_cell_xfs = false,
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    xfs.iter().map(|id| is_date_format(*id, formats.get(id).map(String::as_str))).collect()
}

fn serial_to_date(v: f64, date1904: bool) -> Option<String> {
    if !(0.0..2_958_466.0).contains(&v) {
        return None;
    }
    let base = if date1904 { NaiveDate::from_ymd_opt(1904, 1, 1)? } else { NaiveDate::from_ymd_opt(1899, 12, 30)? };
    let days = v.floor() as i64;
    let date = base.checked_add_signed(Duration::days(days))?;
    let secs = ((v - v.floor()) * 86_400.0).round() as i64;
    if secs == 0 {
        Some(date.format("%Y-%m-%d").to_string())
    } else {
        Some(format!("{} {:02}:{:02}", date.format("%Y-%m-%d"), secs / 3600, (secs % 3600) / 60))
    }
}

fn number_text(v: &str) -> String {
    match v.parse::<f64>() {
        Ok(n) if n.fract() == 0.0 && n.abs() < 1e15 => format!("{}", n as i64),
        Ok(n) => {
            let s = format!("{n}");
            if s.len() > v.len() { v.to_string() } else { s }
        }
        Err(_) => v.to_string(),
    }
}

/// Column index (0-based) from a cell reference such as `AB12`.
fn column_of(r: &str) -> Option<usize> {
    let letters: String = r.chars().take_while(char::is_ascii_alphabetic).collect();
    if letters.is_empty() {
        return None;
    }
    let mut n = 0usize;
    for c in letters.chars() {
        n = n * 26 + (c.to_ascii_uppercase() as usize - 'A' as usize + 1);
    }
    Some(n - 1)
}

fn sheet_rows(xml: &str, strings: &[String], dates: &[bool], date1904: bool, max_rows: usize) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut row_no: Option<usize> = None;
    let (mut cell_col, mut cell_type, mut cell_style) = (0usize, String::new(), 0usize);
    let (mut value, mut in_v, mut in_t) = (String::new(), false, false);
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match local(e.name().as_ref()) {
                "row" => {
                    row.clear();
                    row_no = attr(&e, "r").and_then(|r| r.parse::<usize>().ok());
                }
                "c" => {
                    cell_col = attr(&e, "r").and_then(|r| column_of(&r)).unwrap_or(row.len());
                    cell_type = attr(&e, "t").unwrap_or_default();
                    cell_style = attr(&e, "s").and_then(|s| s.parse().ok()).unwrap_or(0);
                    value.clear();
                }
                "v" => in_v = true,
                "t" => in_t = true,
                _ => {}
            },
            Ok(Event::Text(t)) if in_v || in_t => value.push_str(&t.xml10_content()),
            Ok(Event::GeneralRef(r)) if in_v || in_t => value.push_str(&general_ref(&r)),
            Ok(Event::End(e)) => match local(e.name().as_ref()) {
                "v" => in_v = false,
                "t" => in_t = false,
                "c" => {
                    let text = match cell_type.as_str() {
                        "s" => value.trim().parse::<usize>().ok().and_then(|i| strings.get(i).cloned()).unwrap_or_default(),
                        "b" => if value.trim() == "1" { "TRUE".into() } else { "FALSE".into() },
                        "inlineStr" | "str" | "e" | "d" => value.clone(),
                        _ if value.is_empty() => String::new(),
                        _ => match (dates.get(cell_style).copied().unwrap_or(false), value.trim().parse::<f64>()) {
                            (true, Ok(v)) => serial_to_date(v, date1904).unwrap_or_else(|| number_text(&value)),
                            _ => number_text(value.trim()),
                        },
                    };
                    if cell_col < 16_384 {
                        if row.len() <= cell_col {
                            row.resize(cell_col + 1, String::new());
                        }
                        row[cell_col] = text.trim().to_string();
                    }
                }
                "row" => {
                    // Keep row numbers: blank rows between data rows stay blank.
                    if let Some(n) = row_no {
                        while rows.len() + 1 < n && rows.len() < max_rows {
                            rows.push(Vec::new());
                        }
                    }
                    if rows.len() >= max_rows {
                        break;
                    }
                    while row.last().is_some_and(String::is_empty) {
                        row.pop();
                    }
                    rows.push(std::mem::take(&mut row));
                }
                _ => {}
            },
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    while rows.last().is_some_and(Vec::is_empty) {
        rows.pop();
    }
    rows
}

fn resolve(base: &str, target: &str) -> String {
    if let Some(abs) = target.strip_prefix('/') {
        return abs.to_string();
    }
    let mut parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
    for seg in target.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

/// Every sheet of a workbook, each with at most `max_rows` rows.
pub fn xlsx_sheets(bytes: &[u8], max_rows: usize) -> Result<Vec<Sheet>, OpenRagError> {
    let mut archive = ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| OpenRagError::Validation(format!("invalid Excel file (not a zip archive): {e}")))?;
    let workbook = read_part(&mut archive, "xl/workbook.xml")?
        .ok_or_else(|| OpenRagError::Validation("not an Excel workbook: xl/workbook.xml is missing".into()))?;
    let rels = read_part(&mut archive, "xl/_rels/workbook.xml.rels")?.unwrap_or_default();
    let strings = read_part(&mut archive, "xl/sharedStrings.xml")?.map(|s| shared_strings(&s)).unwrap_or_default();
    let dates = read_part(&mut archive, "xl/styles.xml")?.map(|s| date_styles(&s)).unwrap_or_default();

    let mut targets: HashMap<String, String> = HashMap::new();
    let mut reader = Reader::from_str(&rels);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e) | Event::Empty(e)) if local(e.name().as_ref()) == "Relationship" => {
                if let (Some(id), Some(t)) = (attr(&e, "Id"), attr(&e, "Target")) {
                    targets.insert(id, resolve("xl", &t));
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    let mut entries: Vec<(String, bool, String)> = Vec::new();
    let mut date1904 = false;
    let mut reader = Reader::from_str(&workbook);
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e) | Event::Empty(e)) => match local(e.name().as_ref()) {
                "workbookPr" => date1904 = matches!(attr(&e, "date1904").as_deref(), Some("1" | "true")),
                "sheet" => {
                    let name = attr(&e, "name").unwrap_or_default();
                    let hidden = matches!(attr(&e, "state").as_deref(), Some("hidden" | "veryHidden"));
                    if let Some(part) = prefixed_attr(&e, "id").and_then(|rid| targets.get(&rid).cloned()) {
                        entries.push((name, hidden, part));
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    let mut sheets = Vec::with_capacity(entries.len());
    for (name, hidden, part) in entries {
        // Chart sheets and dialog sheets have no cell data.
        let Some(xml) = read_part(&mut archive, &part)? else { continue };
        if !xml.contains("sheetData") {
            continue;
        }
        sheets.push(Sheet { name, hidden, rows: sheet_rows(&xml, &strings, &dates, date1904, max_rows) });
    }
    Ok(sheets)
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

fn decode_text(bytes: &[u8]) -> String {
    if let Some((enc, bom)) = encoding_rs::Encoding::for_bom(bytes) {
        return enc.decode_without_bom_handling(&bytes[bom..]).0.into_owned();
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => encoding_rs::EUC_KR.decode_without_bom_handling(bytes).0.into_owned(),
    }
}

fn csv_rows(text: &str, max_rows: usize) -> Vec<Vec<String>> {
    let first = text.lines().next().unwrap_or("");
    let delim = [',', '\t', ';', '|'].into_iter().max_by_key(|d| first.matches(*d).count()).unwrap_or(',');
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => quoted = true,
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field).trim().to_string());
                rows.push(std::mem::take(&mut row));
                if rows.len() >= max_rows {
                    return rows;
                }
            }
            c if c == delim => row.push(std::mem::take(&mut field).trim().to_string()),
            c => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field.trim().to_string());
        rows.push(row);
    }
    for r in &mut rows {
        while r.last().is_some_and(String::is_empty) {
            r.pop();
        }
    }
    while rows.last().is_some_and(Vec::is_empty) {
        rows.pop();
    }
    rows
}

fn stem(filename: Option<&str>) -> String {
    filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("Untitled")
        .to_string()
}

/// The sheets of a spreadsheet file (one for CSV, named after the file).
pub fn sheets_of(bytes: &[u8], format: DocumentFormat, filename: Option<&str>, max_rows: usize) -> Result<Vec<Sheet>, OpenRagError> {
    match format {
        DocumentFormat::Xlsx => xlsx_sheets(bytes, max_rows),
        DocumentFormat::Csv => Ok(vec![Sheet { name: stem(filename), hidden: false, rows: csv_rows(&decode_text(bytes), max_rows) }]),
        other => Err(OpenRagError::Validation(format!("{other:?} is not a spreadsheet"))),
    }
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

fn is_number(s: &str) -> bool {
    let t = s.replace(',', "");
    !t.is_empty() && t.parse::<f64>().is_ok()
}

fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 8 && b[..4].iter().all(u8::is_ascii_digit) && matches!(b[4], b'-' | b'.' | b'/')
}

/// (title lines, header row index) — the header is the first row with two or more cells that
/// are mostly text; single-cell rows above it are titles.
fn header_of(rows: &[Vec<String>]) -> (Vec<String>, Option<usize>) {
    let mut titles = Vec::new();
    for (i, r) in rows.iter().take(20).enumerate() {
        let filled: Vec<&String> = r.iter().filter(|c| !c.is_empty()).collect();
        if filled.len() >= 2 {
            let texty = filled.iter().filter(|c| !is_number(c) && !is_date(c)).count();
            if texty * 2 >= filled.len() {
                return (titles, Some(i));
            }
            return (titles, None);
        }
        if let Some(t) = filled.first() {
            titles.push((*t).clone());
        }
    }
    (titles, None)
}

fn short(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n { format!("{t}…") } else { t }
}

fn md_row(cells: &[String], width: usize) -> String {
    let mut c: Vec<String> = cells.iter().map(|x| short(&x.replace('|', "\\|").replace('\n', " "), 40)).collect();
    c.resize(width, String::new());
    format!("| {} |", c.join(" | "))
}

fn column_name(header: Option<&Vec<String>>, i: usize) -> String {
    header
        .and_then(|h| h.get(i))
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| format!("열{}", i + 1))
}

fn group_digits(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn sheet_card(sheet: &Sheet, total_rows: usize) -> String {
    let (titles, header_idx) = header_of(&sheet.rows);
    let header = header_idx.and_then(|i| sheet.rows.get(i));
    let data_start = header_idx.map_or(titles.len(), |i| i + 1);
    let data: Vec<&Vec<String>> = sheet.rows.iter().skip(data_start).filter(|r| r.iter().any(|c| !c.is_empty())).collect();
    let width = sheet.rows.iter().map(Vec::len).max().unwrap_or(0);
    let data_rows = total_rows.saturating_sub(data_start);

    let mut out = format!(
        "## 시트 \"{}\"{} ({}행 × {}열)\n",
        sheet.name,
        if sheet.hidden { " (숨긴 시트)" } else { "" },
        group_digits(data_rows),
        width
    );
    if !titles.is_empty() {
        out.push_str(&format!("제목: {}\n", titles.iter().map(|t| short(t, 80)).collect::<Vec<_>>().join(" / ")));
    }
    if let Some(h) = header {
        out.push_str(&format!("열: {}\n", h.iter().map(|c| short(c, 30)).collect::<Vec<_>>().join(" | ")));
    }

    let mut ranges = Vec::new();
    let mut frequent = Vec::new();
    for col in 0..width {
        let vals: Vec<&str> = data.iter().filter_map(|r| r.get(col)).map(String::as_str).filter(|v| !v.is_empty()).collect();
        if vals.is_empty() {
            continue;
        }
        let name = column_name(header, col);
        let dates: Vec<&str> = vals.iter().copied().filter(|v| is_date(v)).collect();
        let nums: Vec<f64> = vals.iter().filter_map(|v| v.replace(',', "").parse::<f64>().ok()).collect();
        if dates.len() * 10 >= vals.len() * 8 {
            let (lo, hi) = (dates.iter().min(), dates.iter().max());
            if let (Some(lo), Some(hi)) = (lo, hi) {
                ranges.push(format!("{name} {} ~ {}", short(lo, 10), short(hi, 10)));
            }
        } else if nums.len() * 10 >= vals.len() * 8 {
            let lo = nums.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ranges.push(format!("{name} {} ~ {}", number_text(&lo.to_string()), number_text(&hi.to_string())));
        } else {
            let avg = vals.iter().map(|v| v.chars().count()).sum::<usize>() / vals.len();
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for v in &vals {
                *counts.entry(v).or_default() += 1;
            }
            let mut top: Vec<(&str, usize)> = counts.into_iter().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
            if avg > 60 {
                frequent.push(format!("{name} (긴 글): {}", top.iter().take(2).map(|(v, _)| short(v, 80)).collect::<Vec<_>>().join(" / ")));
            } else if top.len() > TOP_VALUES && top.len() * 10 >= vals.len() * 9 {
                // Nearly every value differs (ids, phone numbers): two examples say more than a list.
                frequent.push(format!(
                    "{name} — 거의 모두 다른 값 {}개 (예: {})",
                    group_digits(top.len()),
                    top.iter().take(2).map(|(v, _)| short(v, 40)).collect::<Vec<_>>().join(", ")
                ));
            } else {
                let distinct = top.len();
                let shown: Vec<String> = top.iter().take(TOP_VALUES).map(|(v, n)| if *n > 1 { format!("{}({n})", short(v, 40)) } else { short(v, 40) }).collect();
                frequent.push(format!("{name} — {}{}", shown.join(", "), if distinct > TOP_VALUES { format!(" 외 {}종", distinct - TOP_VALUES) } else { String::new() }));
            }
        }
    }
    if !ranges.is_empty() {
        out.push_str(&format!("값 범위: {}\n", ranges.join("; ")));
    }
    if !frequent.is_empty() {
        out.push_str("자주 나오는 값:\n");
        for f in &frequent {
            out.push_str(&format!("- {f}\n"));
        }
    }
    if !data.is_empty() {
        out.push_str(&format!("앞쪽 {}행:\n", data.len().min(SAMPLE_ROWS)));
        if let Some(h) = header {
            out.push_str(&md_row(h, width));
            out.push('\n');
        }
        for r in data.iter().take(SAMPLE_ROWS) {
            out.push_str(&md_row(r, width));
            out.push('\n');
        }
    }
    if out.chars().count() > CARD_LIMIT {
        out = short(&out, CARD_LIMIT);
    }
    out.trim_end().to_string()
}

/// Catalog cards: page 1 describes the workbook, then one page per sheet.
pub fn parse_spreadsheet(bytes: &[u8], format: DocumentFormat, filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    let sheets = sheets_of(bytes, format, filename, STAT_ROWS)?;
    let title = stem(filename);
    let kind = if format == DocumentFormat::Csv { "CSV 파일" } else { "엑셀 파일" };
    let data_rows = |s: &Sheet| {
        let (titles, header_idx) = header_of(&s.rows);
        s.rows.len().saturating_sub(header_idx.map_or(titles.len(), |i| i + 1))
    };
    let visible: Vec<String> = sheets
        .iter()
        .map(|s| format!("{}{} ({}행)", s.name, if s.hidden { "·숨김" } else { "" }, group_digits(data_rows(s))))
        .collect();
    let mut overview = format!(
        "# {kind} \"{}\"\n이 문서는 {kind}의 안내 카드예요. 시트 {}개: {}.\n셀 내용은 rag_read_sheet 도구로 읽어요: 시작 행과 행 수로 넘겨 읽거나, find(찾을 값)와 column(열 이름)으로 그 값이 든 행만 골라 읽어요.",
        filename.unwrap_or(&title),
        sheets.len(),
        visible.join(", ")
    );
    if sheets.iter().all(|s| s.rows.is_empty()) {
        overview.push_str("\n모든 시트가 비어 있어요.");
    }
    let mut pages_md = vec![overview];
    for s in &sheets {
        if !s.rows.is_empty() {
            pages_md.push(sheet_card(s, s.rows.len()));
        }
    }
    let pages: Vec<ParsedPage> = pages_md
        .into_iter()
        .enumerate()
        .map(|(i, md)| ParsedPage {
            page_no: i32::try_from(i + 1).unwrap_or(i32::MAX),
            extracted_aliases: extract_parenthetical_aliases(&md),
            content_sha256: sha256_hex_str(&md),
            raw_markdown: md,
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks: Vec::new(),
            page_image_rgb: None,
        })
        .collect();
    let full_markdown = pages.iter().map(|p| p.raw_markdown.as_str()).collect::<Vec<_>>().join("\n\n");
    Ok(ParsedDocument {
        title,
        format,
        mime_type: format.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        extracted_aliases: extract_parenthetical_aliases(&full_markdown),
        full_markdown,
        pages,
        ocr_policy: None,
        parser_engine: if format == DocumentFormat::Csv { "csv_card".into() } else { "xlsx_card".into() },
    })
}

/// A slice of one sheet, read on demand.
#[derive(Debug, Clone, PartialEq)]
pub struct SheetSlice {
    pub sheets: Vec<String>,
    pub sheet: String,
    pub header: Vec<String>,
    /// Data rows (below the header), 0-based.
    pub offset: usize,
    pub rows: Vec<Vec<String>>,
    /// Position of each returned row among the data rows: the `offset` that reads it unfiltered.
    pub row_index: Vec<usize>,
    pub total_rows: usize,
    /// Rows of `sheet` that pass the filter (`None` when the read is not filtered).
    pub total_matches: Option<usize>,
    /// Matching rows per searched sheet, in workbook order (empty when not filtered).
    pub matches_by_sheet: Vec<(String, usize)>,
}

/// Which rows a filtered read keeps: rows with a cell that contains `text` (equals it when
/// `exact`), looking in `column` only when one is named. Case, runs of spaces and thousands
/// separators (`1,350,000`) are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowFilter {
    pub text: String,
    pub column: Option<String>,
    pub exact: bool,
}

fn pick<'a>(sheets: &'a [Sheet], names: &[String], want: &str) -> Result<&'a Sheet, OpenRagError> {
    sheets
        .iter()
        .find(|s| s.name == want)
        .or_else(|| sheets.iter().find(|s| s.name.eq_ignore_ascii_case(want.trim())))
        .ok_or_else(|| OpenRagError::NotFound(format!("no sheet named '{want}' (sheets: {})", names.join(", "))))
}

/// (header, data rows below it).
fn split_sheet(sheet: &Sheet) -> (Vec<String>, &[Vec<String>]) {
    let (titles, header_idx) = header_of(&sheet.rows);
    let start = header_idx.map_or(titles.len(), |i| i + 1);
    let header = header_idx.and_then(|i| sheet.rows.get(i)).cloned().unwrap_or_default();
    (header, &sheet.rows[start.min(sheet.rows.len())..])
}

/// Lower case, single spaces, no commas between digits.
fn normalize(s: &str) -> String {
    let lower = s.trim().to_lowercase();
    let chars: Vec<char> = lower.split_whitespace().collect::<Vec<_>>().join(" ").chars().collect();
    let mut out = String::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        let between_digits = i > 0 && chars[i - 1].is_ascii_digit() && chars.get(i + 1).is_some_and(char::is_ascii_digit);
        if !(c == ',' && between_digits) {
            out.push(c);
        }
    }
    out
}

/// Column by its header name, by the `열N` name the cards use for unnamed columns, or by its
/// letter (`C`).
fn column_index(header: &[String], data: &[Vec<String>], want: &str) -> Option<usize> {
    let w = normalize(want);
    let width = data.iter().map(Vec::len).chain([header.len()]).max().unwrap_or(0);
    let named = Some(header.to_vec());
    (0..width)
        .find(|&i| header.get(i).is_some_and(|h| normalize(h) == w))
        .or_else(|| (0..width).find(|&i| normalize(&column_name(named.as_ref(), i)) == w))
        .or_else(|| {
            let t = want.trim();
            (!t.is_empty() && t.len() <= 3 && t.chars().all(|c| c.is_ascii_alphabetic()))
                .then(|| column_of(t))
                .flatten()
                .filter(|&i| i < width)
        })
}

/// Rows `offset..offset+limit` of a sheet (by name; the first sheet when `None`), below its header.
pub fn read_sheet(
    bytes: &[u8],
    format: DocumentFormat,
    filename: Option<&str>,
    sheet: Option<&str>,
    offset: usize,
    limit: usize,
) -> Result<SheetSlice, OpenRagError> {
    let sheets = sheets_of(bytes, format, filename, usize::MAX)?;
    let names: Vec<String> = sheets.iter().map(|s| s.name.clone()).collect();
    let chosen = match sheet {
        Some(want) => pick(&sheets, &names, want)?,
        None => sheets.first().ok_or_else(|| OpenRagError::NotFound("the file has no sheets".into()))?,
    };
    let (header, data) = split_sheet(chosen);
    let rows: Vec<Vec<String>> = data.iter().skip(offset).take(limit).cloned().collect();
    let row_index = (offset..offset + rows.len()).collect();
    Ok(SheetSlice {
        sheets: names,
        sheet: chosen.name.clone(),
        header,
        offset,
        rows,
        row_index,
        total_rows: data.len(),
        total_matches: None,
        matches_by_sheet: Vec::new(),
    })
}

/// One searched sheet: its header, data rows and the rows that passed the filter.
struct Searched<'a> {
    sheet: &'a Sheet,
    header: Vec<String>,
    data: &'a [Vec<String>],
    hits: Vec<usize>,
}

/// The rows that pass `filter`, `offset..offset+limit` of them. With `sheet` the search stays in
/// that sheet; without it every sheet is searched, `matches_by_sheet` counts them all and the rows
/// come from the first sheet that has a match.
pub fn find_rows(
    bytes: &[u8],
    format: DocumentFormat,
    filename: Option<&str>,
    sheet: Option<&str>,
    filter: &RowFilter,
    offset: usize,
    limit: usize,
) -> Result<SheetSlice, OpenRagError> {
    let needle = normalize(&filter.text);
    if needle.is_empty() {
        return Err(OpenRagError::Validation("the text to find is empty".into()));
    }
    let sheets = sheets_of(bytes, format, filename, usize::MAX)?;
    let names: Vec<String> = sheets.iter().map(|s| s.name.clone()).collect();
    let searched: Vec<&Sheet> = match sheet {
        Some(want) => vec![pick(&sheets, &names, want)?],
        None => sheets.iter().collect(),
    };
    if searched.is_empty() {
        return Err(OpenRagError::NotFound("the file has no sheets".into()));
    }
    let mut found: Vec<Searched<'_>> = Vec::with_capacity(searched.len());
    let mut has_column = filter.column.is_none();
    for s in searched {
        let (header, data) = split_sheet(s);
        let col = match filter.column.as_deref() {
            Some(c) => match column_index(&header, data, c) {
                Some(i) => {
                    has_column = true;
                    Some(i)
                }
                None => {
                    found.push(Searched { sheet: s, header, data, hits: Vec::new() });
                    continue;
                }
            },
            None => None,
        };
        let hits = data
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                let cells: Box<dyn Iterator<Item = &String>> = match col {
                    Some(i) => Box::new(row.get(i).into_iter()),
                    None => Box::new(row.iter()),
                };
                cells.into_iter().any(|cell| {
                    let c = normalize(cell);
                    if filter.exact { c == needle } else { c.contains(&needle) }
                })
            })
            .map(|(i, _)| i)
            .collect();
        found.push(Searched { sheet: s, header, data, hits });
    }
    if !has_column {
        let Searched { header, data, .. } = &found[0];
        let width = data.iter().map(Vec::len).chain([header.len()]).max().unwrap_or(0);
        let named = Some(header.clone());
        let columns: Vec<String> = (0..width).map(|i| column_name(named.as_ref(), i)).collect();
        return Err(OpenRagError::NotFound(format!(
            "no column named '{}' (columns: {})",
            filter.column.as_deref().unwrap_or_default(),
            columns.join(", ")
        )));
    }
    let matches_by_sheet = found.iter().map(|f| (f.sheet.name.clone(), f.hits.len())).collect();
    let at = found.iter().position(|f| !f.hits.is_empty()).unwrap_or(0);
    let Searched { sheet: chosen, header, data, hits } = found.swap_remove(at);
    let row_index: Vec<usize> = hits.iter().copied().skip(offset).take(limit).collect();
    Ok(SheetSlice {
        sheets: names,
        sheet: chosen.name.clone(),
        header,
        offset,
        rows: row_index.iter().map(|&i| data[i].clone()).collect(),
        row_index,
        total_rows: data.len(),
        total_matches: Some(hits.len()),
        matches_by_sheet,
    })
}
