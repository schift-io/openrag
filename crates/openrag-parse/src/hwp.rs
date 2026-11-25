use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use cfb::CompoundFile;
use flate2::read::DeflateDecoder;
use openrag_core::{OpenRagError, ParseConfig};
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesRef, Event};
use quick_xml::{Reader, XmlVersion};
use regex::Regex;
use tokio::process::Command;
use zip::ZipArchive;

use crate::aliases::extract_parenthetical_aliases;
use crate::model::{sha256_hex_bytes, sha256_hex_str, DocumentFormat, ParsedDocument, ParsedPage};

#[derive(Debug, Clone)]
pub struct KordocCliConfig {
    pub bin_path: Option<PathBuf>,
    pub timeout: Duration,
    /// Longer timeout for PDFs because local OCR on large scans can take minutes.
    pub pdf_timeout: Duration,
}

impl Default for KordocCliConfig {
    fn default() -> Self {
        Self {
            bin_path: None,
            timeout: Duration::from_secs(30),
            pdf_timeout: Duration::from_secs(600),
        }
    }
}

pub fn escape_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Normalizes table HTML so that only `<table>`, `<tr>`, and `<td ...>` tags remain
/// (`<th>`, `<thead>`, `<tbody>`, `<tfoot>`, `<br>`, `<b>` are stripped or normalized to `<td>`).
pub fn sanitize_table_html(raw: &str) -> String {
    let mut s = raw.to_string();
    if let Ok(re) = Regex::new(
        r#"(?is)</?(?:thead|tbody|tfoot|b)(?:\s+(?:[^>"']|"[^"]*"|'[^']*')*)?\s*/?>"#,
    ) {
        s = re.replace_all(&s, "").into_owned();
    }
    if let Ok(re) = Regex::new(r"(?is)<br(?:\s[^>]*)?/?>") {
        s = re.replace_all(&s, " ").into_owned();
    }
    if let Ok(re) = Regex::new(r"(?is)</br\s*>") {
        s = re.replace_all(&s, "").into_owned();
    }
    if let Ok(re) = Regex::new(
        r#"(?is)<th(?:\s+((?:[^>"'/]|"[^"]*"|'[^']*'|/(?:[^>"']|"[^"]*"|'[^']*'))*))?\s*/>"#,
    ) {
        s = re.replace_all(&s, |caps: &regex::Captures<'_>| {
            if let Some(attrs) = caps.get(1) {
                let a = attrs.as_str().trim_end();
                if a.is_empty() {
                    "<td></td>".to_string()
                } else {
                    format!("<td {a}></td>")
                }
            } else {
                "<td></td>".to_string()
            }
        })
        .into_owned();
    }
    if let Ok(re) = Regex::new(r"(?is)<th(\s[^>]*)?>") {
        s = re.replace_all(&s, "<td$1>").into_owned();
    }
    if let Ok(re) = Regex::new(r"(?is)</th\s*>") {
        s = re.replace_all(&s, "</td>").into_owned();
    }
    if let Ok(re) = Regex::new(r"(?is)<(/?)(table|tr|td)(\s[^>]*)?>") {
        s = re.replace_all(&s, |caps: &regex::Captures<'_>| {
            let slash = caps.get(1).map_or("", |m| m.as_str());
            let tag = caps.get(2).map_or("", |m| m.as_str()).to_ascii_lowercase();
            let attrs = caps.get(3).map_or("", |m| m.as_str());
            format!("<{slash}{tag}{attrs}>")
        })
        .into_owned();
    }
    s
}

#[derive(Debug, Clone, Default)]
struct CellData {
    text: String,
    nested_tables: Vec<String>,
    colspan: u32,
    rowspan: u32,
}

#[derive(Debug, Clone, Default)]
struct TableData {
    rows: Vec<Vec<CellData>>,
}

#[derive(Debug, Clone, Default)]
struct TableFrame {
    table: TableData,
    current_row: Option<Vec<CellData>>,
    current_cell: Option<CellData>,
    cell_para_buf: String,
}

impl TableData {
    fn to_strict_html(&self) -> String {
        let mut out = String::from("<table>");
        for row in &self.rows {
            out.push_str("<tr>");
            for cell in row {
                out.push_str("<td");
                if cell.colspan > 1 {
                    out.push_str(&format!(" colspan=\"{}\"", cell.colspan));
                }
                if cell.rowspan > 1 {
                    out.push_str(&format!(" rowspan=\"{}\"", cell.rowspan));
                }
                out.push('>');
                let escaped_text = escape_html(cell.text.trim());
                let mut has_content = !escaped_text.is_empty();
                out.push_str(&escaped_text);
                for nested in &cell.nested_tables {
                    if has_content {
                        out.push(' ');
                    }
                    out.push_str(nested);
                    has_content = true;
                }
                out.push_str("</td>");
            }
            out.push_str("</tr>");
        }
        out.push_str("</table>");
        out
    }
}

pub async fn parse_hwp_family(
    bytes: &[u8],
    filename: Option<&str>,
    cfg: &ParseConfig,
    kordoc: Option<&KordocCliConfig>,
) -> Result<ParsedDocument, OpenRagError> {
    if bytes.is_empty() {
        return Err(OpenRagError::Validation("empty HWP/HWPX/HML document".into()));
    }

    let format = DocumentFormat::detect(bytes, None, filename);

    if cfg.hwp_engine == "kordoc" {
        if let Some(doc) = try_kordoc_cli(bytes, filename, format, kordoc).await? {
            return Ok(doc);
        }
    }

    match format {
        DocumentFormat::Hwpx => parse_hwpx(bytes, filename),
        DocumentFormat::Hml => parse_hml(bytes, filename),
        DocumentFormat::Hwp => parse_hwp5(bytes, filename),
        _ => {
            if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
                parse_hwpx(bytes, filename)
            } else if bytes.starts_with(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") {
                parse_hwp5(bytes, filename)
            } else {
                parse_hml(bytes, filename)
            }
        }
    }
}

/// Resolves the kordoc CLI binary from explicit config, `OPENRAG_KORDOC_BIN`, or `PATH`.
pub(crate) fn resolve_kordoc_bin(kordoc: Option<&KordocCliConfig>) -> Option<PathBuf> {
    let bin = kordoc
        .and_then(|c| c.bin_path.clone())
        .or_else(|| std::env::var_os("OPENRAG_KORDOC_BIN").map(PathBuf::from))
        .or_else(|| find_on_path("kordoc"))?;
    bin.exists().then_some(bin)
}

async fn try_kordoc_cli(
    bytes: &[u8],
    filename: Option<&str>,
    format: DocumentFormat,
    kordoc: Option<&KordocCliConfig>,
) -> Result<Option<ParsedDocument>, OpenRagError> {
    let Some(bin_path) = resolve_kordoc_bin(kordoc) else {
        return Ok(None);
    };

    let timeout = kordoc.map_or_else(|| Duration::from_secs(30), |c| c.timeout);
    let ext = match format {
        DocumentFormat::Hwpx => "hwpx",
        DocumentFormat::Hml => "hml",
        _ => "hwp",
    };

    let tmp_dir = std::env::temp_dir();
    let unique = format!("openrag-kordoc-{}.{ext}", uuid::Uuid::now_v7());
    let tmp_path = tmp_dir.join(unique);
    tokio::fs::write(&tmp_path, bytes)
        .await
        .map_err(|e| OpenRagError::Internal(format!("failed to write kordoc temp input: {e}")))?;

    let mut cmd = Command::new(&bin_path);
    cmd.arg(&tmp_path).arg("--format").arg("markdown");

    let output_res = tokio::time::timeout(timeout, cmd.output()).await;
    let _ = tokio::fs::remove_file(&tmp_path).await;

    let Ok(Ok(output)) = output_res else {
        return Ok(None);
    };
    if !output.status.success() {
        return Ok(None);
    }

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let sanitized = sanitize_table_html(&stdout);
    if sanitized.trim().is_empty() {
        return Ok(None);
    }

    let aliases = extract_parenthetical_aliases(&sanitized);
    let tables_html = extract_tables_from_text(&sanitized);
    let page = ParsedPage {
        page_no: 1,
        raw_markdown: sanitized.clone(),
        tables_html,
        formulas: Vec::new(),
        fields: Vec::new(),
        blocks: Vec::new(),
        extracted_aliases: aliases.clone(),
        page_image_rgb: None,
        content_sha256: sha256_hex_str(&sanitized),
    };

    let title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("Untitled HWP")
        .to_string();

    Ok(Some(ParsedDocument {
        title,
        format,
        mime_type: format.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages: vec![page],
        full_markdown: sanitized,
        extracted_aliases: aliases,
        ocr_policy: None,
        parser_engine: "kordoc".to_string(),
    }))
}

fn find_on_path(bin_name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(bin_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub(crate) fn extract_tables_from_text(text: &str) -> Vec<String> {
    let mut tables = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<table") {
        let after = &rest[start..];
        if let Some(end) = after.find("</table>") {
            let end_idx = end + "</table>".len();
            tables.push(after[..end_idx].to_string());
            rest = &after[end_idx..];
        } else {
            break;
        }
    }
    tables
}

pub fn parse_hwpx(bytes: &[u8], filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    if bytes.is_empty() {
        return Err(OpenRagError::Validation("empty HWPX payload".into()));
    }

    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor)
        .map_err(|e| OpenRagError::Validation(format!("invalid HWPX zip archive: {e}")))?;

    let mut title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("Untitled HWPX")
        .to_string();

    let mut section_names: Vec<(usize, String)> = Vec::new();

    for idx in 0..archive.len() {
        let mut file = archive.by_index(idx).map_err(|e| {
            OpenRagError::Validation(format!("corrupt HWPX zip entry #{idx}: {e}"))
        })?;
        let name = file.name().to_string();

        if name.eq_ignore_ascii_case("Contents/content.hpf") {
            let mut hpf_xml = String::new();
            file.read_to_string(&mut hpf_xml).map_err(|e| {
                OpenRagError::Validation(format!("failed to read HWPX content.hpf: {e}"))
            })?;
            if let Some(parsed_title) = extract_hpf_title(&hpf_xml) {
                title = parsed_title;
            }
            continue;
        }

        if let Some(sec_idx) = parse_hwpx_section_index(&name) {
            section_names.push((sec_idx, name));
        }
    }

    if section_names.is_empty() {
        return Err(OpenRagError::Validation(
            "HWPX missing Contents/section0.xml".into(),
        ));
    }

    section_names.sort_by_key(|(idx, _)| *idx);

    let mut pages = Vec::new();
    let mut page_no = 1i32;

    for (_, sec_name) in section_names {
        let mut file = archive.by_name(&sec_name).map_err(|e| {
            OpenRagError::Validation(format!("failed to open HWPX section '{sec_name}': {e}"))
        })?;
        let mut xml = String::new();
        file.read_to_string(&mut xml).map_err(|e| {
            OpenRagError::Validation(format!("invalid UTF-8 in HWPX '{sec_name}': {e}"))
        })?;

        let sec_pages = parse_hwpx_section_xml(&xml, &mut page_no)?;
        pages.extend(sec_pages);
    }

    if pages.is_empty() {
        pages.push(ParsedPage {
            page_no: 1,
            raw_markdown: String::new(),
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks: Vec::new(),
            extracted_aliases: Vec::new(),
            page_image_rgb: None,
            content_sha256: sha256_hex_str(""),
        });
    }

    let full_markdown = pages
        .iter()
        .map(|p| p.raw_markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let extracted_aliases = extract_parenthetical_aliases(&full_markdown);

    Ok(ParsedDocument {
        title,
        format: DocumentFormat::Hwpx,
        mime_type: DocumentFormat::Hwpx.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages,
        full_markdown,
        extracted_aliases,
        ocr_policy: None,
        parser_engine: "hwpx_rust".to_string(),
    })
}

fn parse_hwpx_section_index(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    let stripped = lower.strip_prefix("contents/section")?;
    let num_str = stripped.strip_suffix(".xml")?;
    num_str.parse::<usize>().ok()
}

fn extract_hpf_title(xml: &str) -> Option<String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut in_title = false;
    while let Ok(ev) = reader.read_event_into(&mut buf) {
        match ev {
            Event::Start(e) => {
                if local_name(e.name().as_ref()).eq_ignore_ascii_case("title") {
                    in_title = true;
                }
            }
            Event::Text(t) if in_title => {
                let txt = t.xml10_content();
                let trimmed = txt.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
            Event::End(e) if local_name(e.name().as_ref()).eq_ignore_ascii_case("title") => {
                in_title = false;
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    None
}

fn parse_hwpx_section_xml(xml: &str, next_page_no: &mut i32) -> Result<Vec<ParsedPage>, OpenRagError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    struct PageAcc {
        lines: Vec<String>,
        tables_html: Vec<String>,
        formulas: Vec<String>,
        fields: Vec<(String, String)>,
    }

    let mut pages_acc = vec![PageAcc {
        lines: Vec::new(),
        tables_html: Vec::new(),
        formulas: Vec::new(),
        fields: Vec::new(),
    }];

    let mut table_stack: Vec<TableFrame> = Vec::new();
    let mut open_tags: Vec<String> = Vec::new();
    let mut para_buf = String::new();
    let mut in_text = false;
    let mut in_formula = false;
    let mut formula_buf = String::new();
    let mut active_field_name: Option<String> = None;
    let mut field_buf = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                open_tags.push(e.name().as_ref().to_string());
                let lname = local_name(e.name().as_ref()).to_string();
                match lname.as_str() {
                    "p" => {
                        let has_page_break = attr_bool(&e, "pageBreak");
                        if has_page_break {
                            let cur_empty = pages_acc.last().is_none_or(|p| p.lines.is_empty());
                            if !cur_empty {
                                pages_acc.push(PageAcc {
                                    lines: Vec::new(),
                                    tables_html: Vec::new(),
                                    formulas: Vec::new(),
                                    fields: Vec::new(),
                                });
                            }
                        }
                        if let Some(frame) = table_stack.last_mut().filter(|f| f.current_cell.is_some()) {
                            frame.cell_para_buf.clear();
                        } else if table_stack.is_empty() {
                            para_buf.clear();
                        }
                    }
                    "t" | "char" => {
                        in_text = true;
                    }
                    "tbl" => {
                        table_stack.push(TableFrame::default());
                    }
                    "tr" => {
                        if let Some(frame) = table_stack.last_mut() {
                            frame.current_row = Some(Vec::new());
                        }
                    }
                    "tc" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if frame.current_row.is_some() {
                                let colspan = attr_u32(&e, "colSpan").max(1);
                                let rowspan = attr_u32(&e, "rowSpan").max(1);
                                frame.current_cell = Some(CellData {
                                    text: String::new(),
                                    nested_tables: Vec::new(),
                                    colspan,
                                    rowspan,
                                });
                                frame.cell_para_buf.clear();
                            }
                        }
                    }
                    "cellSpan" => {
                        if let Some(cell) = table_stack
                            .last_mut()
                            .and_then(|f| f.current_cell.as_mut())
                        {
                            let cs = attr_u32(&e, "colSpan");
                            let rs = attr_u32(&e, "rowSpan");
                            if cs > 1 {
                                cell.colspan = cs;
                            }
                            if rs > 1 {
                                cell.rowspan = rs;
                            }
                        }
                    }
                    "equation" | "script" => {
                        in_formula = true;
                        formula_buf.clear();
                    }
                    "fieldBegin" => {
                        let name = attr_str(&e, "name").unwrap_or_else(|| "field".to_string());
                        active_field_name = Some(name);
                        field_buf.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let qname = e.name();
                let lname = local_name(qname.as_ref());
                if lname == "cellSpan" {
                    if let Some(cell) = table_stack
                        .last_mut()
                        .and_then(|f| f.current_cell.as_mut())
                    {
                        let cs = attr_u32(&e, "colSpan");
                        let rs = attr_u32(&e, "rowSpan");
                        if cs > 1 {
                            cell.colspan = cs;
                        }
                        if rs > 1 {
                            cell.rowspan = rs;
                        }
                    }
                } else if lname == "pageBreak" {
                    let cur_empty = pages_acc.last().is_none_or(|p| p.lines.is_empty());
                    if !cur_empty {
                        pages_acc.push(PageAcc {
                            lines: Vec::new(),
                            tables_html: Vec::new(),
                            formulas: Vec::new(),
                            fields: Vec::new(),
                        });
                    }
                }
            }
            Ok(Event::Text(t)) => {
                let decoded = t.xml10_content();
                if in_formula {
                    formula_buf.push_str(&decoded);
                } else if in_text {
                    if let Some(frame) = table_stack.last_mut().filter(|f| f.current_cell.is_some()) {
                        frame.cell_para_buf.push_str(&decoded);
                    } else if table_stack.is_empty() {
                        para_buf.push_str(&decoded);
                    }
                    if active_field_name.is_some() {
                        field_buf.push_str(&decoded);
                    }
                }
            }
            Ok(Event::GeneralRef(r)) => {
                let decoded = resolve_general_ref(&r);
                if in_formula {
                    formula_buf.push_str(&decoded);
                } else if in_text {
                    if let Some(frame) = table_stack.last_mut().filter(|f| f.current_cell.is_some()) {
                        frame.cell_para_buf.push_str(&decoded);
                    } else if table_stack.is_empty() {
                        para_buf.push_str(&decoded);
                    }
                    if active_field_name.is_some() {
                        field_buf.push_str(&decoded);
                    }
                }
            }
            Ok(Event::End(e)) => {
                let _ = open_tags.pop();
                let lname = local_name(e.name().as_ref()).to_string();
                match lname.as_str() {
                    "t" | "char" => {
                        in_text = false;
                    }
                    "p" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if let Some(cell) = frame.current_cell.as_mut() {
                                let trimmed = frame.cell_para_buf.trim();
                                if !trimmed.is_empty() {
                                    if !cell.text.is_empty() {
                                        cell.text.push(' ');
                                    }
                                    cell.text.push_str(trimmed);
                                }
                                frame.cell_para_buf.clear();
                            }
                        } else {
                            let trimmed = para_buf.trim();
                            if !trimmed.is_empty() {
                                if let Some(cur) = pages_acc.last_mut() {
                                    cur.lines.push(trimmed.to_string());
                                }
                            }
                            para_buf.clear();
                        }
                    }
                    "tc" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if let (Some(row), Some(cell)) =
                                (frame.current_row.as_mut(), frame.current_cell.take())
                            {
                                row.push(cell);
                            }
                        }
                    }
                    "tr" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if let Some(row) = frame.current_row.take() {
                                frame.table.rows.push(row);
                            }
                        }
                    }
                    "tbl" => {
                        if let Some(frame) = table_stack.pop() {
                            let html = frame.table.to_strict_html();
                            if let Some(outer_cell) = table_stack
                                .last_mut()
                                .and_then(|f| f.current_cell.as_mut())
                            {
                                outer_cell.nested_tables.push(html);
                            } else if let Some(cur) = pages_acc.last_mut() {
                                cur.tables_html.push(html.clone());
                                cur.lines.push(html);
                            }
                        }
                    }
                    "equation" | "script" => {
                        in_formula = false;
                        let eq = formula_buf.trim();
                        if !eq.is_empty() {
                            if let Some(cur) = pages_acc.last_mut() {
                                cur.formulas.push(eq.to_string());
                                cur.lines.push(format!("$${eq}$$"));
                            }
                        }
                        formula_buf.clear();
                    }
                    "fieldEnd" => {
                        if let Some(name) = active_field_name.take() {
                            let val = field_buf.trim().to_string();
                            if let Some(cur) = pages_acc.last_mut() {
                                cur.fields.push((name, val));
                            }
                            field_buf.clear();
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => {
                if !open_tags.is_empty() {
                    return Err(OpenRagError::Validation(format!(
                        "malformed HWPX section XML: unclosed element(s) at EOF: {}",
                        open_tags.join(", ")
                    )));
                }
                break;
            }
            Err(e) => {
                return Err(OpenRagError::Validation(format!(
                    "malformed HWPX section XML: {e}"
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    let mut out = Vec::with_capacity(pages_acc.len());
    for acc in pages_acc {
        let raw_markdown = acc.lines.join("\n\n");
        let aliases = extract_parenthetical_aliases(&raw_markdown);
        let sha = sha256_hex_str(&raw_markdown);
        out.push(ParsedPage {
            page_no: *next_page_no,
            raw_markdown,
            tables_html: acc.tables_html,
            formulas: acc.formulas,
            fields: acc.fields,
            blocks: Vec::new(),
            extracted_aliases: aliases,
            page_image_rgb: None,
            content_sha256: sha,
        });
        *next_page_no += 1;
    }

    Ok(out)
}

pub fn parse_hml(bytes: &[u8], filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    let xml = std::str::from_utf8(bytes)
        .map_err(|e| OpenRagError::Validation(format!("HML document is not valid UTF-8: {e}")))?;
    if !xml.contains("<HWPML") && !xml.contains("<hwpml") {
        return Err(OpenRagError::Validation(
            "invalid HML document: missing <HWPML> root element".into(),
        ));
    }

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    let mut title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("Untitled HML")
        .to_string();

    let mut lines: Vec<String> = Vec::new();
    let mut tables_html: Vec<String> = Vec::new();
    let mut formulas: Vec<String> = Vec::new();

    let mut in_title = false;
    let mut in_char = false;
    let mut in_script = false;
    let mut para_buf = String::new();
    let mut script_buf = String::new();
    let mut table_stack: Vec<TableFrame> = Vec::new();
    let mut open_tags: Vec<String> = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                open_tags.push(e.name().as_ref().to_string());
                let lname = local_name(e.name().as_ref()).to_ascii_uppercase();
                match lname.as_str() {
                    "TITLE" => in_title = true,
                    "P" => {
                        if table_stack.is_empty() {
                            para_buf.clear();
                        }
                    }
                    "CHAR" => in_char = true,
                    "TABLE" => table_stack.push(TableFrame::default()),
                    "ROW" => {
                        if let Some(frame) = table_stack.last_mut() {
                            frame.current_row = Some(Vec::new());
                        }
                    }
                    "CELL" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if frame.current_row.is_some() {
                                frame.current_cell = Some(CellData {
                                    text: String::new(),
                                    nested_tables: Vec::new(),
                                    colspan: attr_u32_ci(&e, "colspan").max(1),
                                    rowspan: attr_u32_ci(&e, "rowspan").max(1),
                                });
                            }
                        }
                    }
                    "SCRIPT" => {
                        in_script = true;
                        script_buf.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                let decoded = t.xml10_content();
                if in_title {
                    let tr = decoded.trim();
                    if !tr.is_empty() {
                        title = tr.to_string();
                    }
                } else if in_script {
                    script_buf.push_str(&decoded);
                } else if in_char {
                    if let Some(cell) = table_stack
                        .last_mut()
                        .and_then(|f| f.current_cell.as_mut())
                    {
                        cell.text.push_str(&decoded);
                    } else if table_stack.is_empty() {
                        para_buf.push_str(&decoded);
                    }
                }
            }
            Ok(Event::GeneralRef(r)) => {
                let decoded = resolve_general_ref(&r);
                if in_title {
                    title.push_str(&decoded);
                } else if in_script {
                    script_buf.push_str(&decoded);
                } else if in_char {
                    if let Some(cell) = table_stack
                        .last_mut()
                        .and_then(|f| f.current_cell.as_mut())
                    {
                        cell.text.push_str(&decoded);
                    } else if table_stack.is_empty() {
                        para_buf.push_str(&decoded);
                    }
                }
            }
            Ok(Event::End(e)) => {
                let _ = open_tags.pop();
                let lname = local_name(e.name().as_ref()).to_ascii_uppercase();
                match lname.as_str() {
                    "TITLE" => in_title = false,
                    "CHAR" => in_char = false,
                    "P" => {
                        if let Some(cell) = table_stack
                            .last_mut()
                            .and_then(|f| f.current_cell.as_mut())
                        {
                            if !cell.text.ends_with(' ') && !cell.text.is_empty() {
                                cell.text.push(' ');
                            }
                        } else if table_stack.is_empty() {
                            let tr = para_buf.trim();
                            if !tr.is_empty() {
                                lines.push(tr.to_string());
                            }
                            para_buf.clear();
                        }
                    }
                    "CELL" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if let (Some(row), Some(cell)) =
                                (frame.current_row.as_mut(), frame.current_cell.take())
                            {
                                row.push(cell);
                            }
                        }
                    }
                    "ROW" => {
                        if let Some(frame) = table_stack.last_mut() {
                            if let Some(row) = frame.current_row.take() {
                                frame.table.rows.push(row);
                            }
                        }
                    }
                    "TABLE" => {
                        if let Some(frame) = table_stack.pop() {
                            let html = frame.table.to_strict_html();
                            if let Some(outer_cell) = table_stack
                                .last_mut()
                                .and_then(|f| f.current_cell.as_mut())
                            {
                                outer_cell.nested_tables.push(html);
                            } else {
                                tables_html.push(html.clone());
                                lines.push(html);
                            }
                        }
                    }
                    "SCRIPT" => {
                        in_script = false;
                        let eq = script_buf.trim();
                        if !eq.is_empty() {
                            formulas.push(eq.to_string());
                            lines.push(format!("$${eq}$$"));
                        }
                        script_buf.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => {
                if !open_tags.is_empty() {
                    return Err(OpenRagError::Validation(format!(
                        "malformed HML XML: unclosed element(s) at EOF: {}",
                        open_tags.join(", ")
                    )));
                }
                break;
            }
            Err(e) => {
                return Err(OpenRagError::Validation(format!(
                    "malformed HML XML: {e}"
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    let raw_markdown = lines.join("\n\n");
    let aliases = extract_parenthetical_aliases(&raw_markdown);
    let page = ParsedPage {
        page_no: 1,
        raw_markdown: raw_markdown.clone(),
        tables_html,
        formulas,
        fields: Vec::new(),
        blocks: Vec::new(),
        extracted_aliases: aliases.clone(),
        page_image_rgb: None,
        content_sha256: sha256_hex_str(&raw_markdown),
    };

    Ok(ParsedDocument {
        title,
        format: DocumentFormat::Hml,
        mime_type: DocumentFormat::Hml.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages: vec![page],
        full_markdown: raw_markdown,
        extracted_aliases: aliases,
        ocr_policy: None,
        parser_engine: "hml_rust".to_string(),
    })
}

pub fn parse_hwp5(bytes: &[u8], filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    if bytes.is_empty() {
        return Err(OpenRagError::Validation("empty HWP 5.0 payload".into()));
    }

    let mut comp = CompoundFile::open(Cursor::new(bytes))
        .map_err(|e| OpenRagError::Validation(format!("corrupt HWP OLE2 container: {e}")))?;

    let mut header_buf = Vec::new();
    {
        let mut hdr_stream = comp.open_stream("FileHeader").map_err(|e| {
            OpenRagError::Validation(format!("HWP OLE2 missing FileHeader stream: {e}"))
        })?;
        hdr_stream.read_to_end(&mut header_buf).map_err(|e| {
            OpenRagError::Validation(format!("failed reading HWP FileHeader: {e}"))
        })?;
    }

    if header_buf.len() < 40 || !header_buf.starts_with(b"HWP Document File") {
        return Err(OpenRagError::Validation(
            "invalid HWP 5.0 FileHeader signature".into(),
        ));
    }

    let flags = u32::from_le_bytes([header_buf[36], header_buf[37], header_buf[38], header_buf[39]]);
    let compressed = (flags & 0x01) != 0;

    let mut pages = Vec::new();
    let mut section_idx = 0usize;
    loop {
        let stream_path = format!("BodyText/Section{section_idx}");
        let Ok(mut stream) = comp.open_stream(&stream_path) else {
            break;
        };
        let mut raw_stream = Vec::new();
        stream.read_to_end(&mut raw_stream).map_err(|e| {
            OpenRagError::Validation(format!("corrupt HWP stream '{stream_path}': {e}"))
        })?;

        let decoded_stream = if compressed {
            let mut decoder = DeflateDecoder::new(&raw_stream[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).map_err(|e| {
                OpenRagError::Validation(format!(
                    "failed to decompress HWP '{stream_path}': {e}"
                ))
            })?;
            decompressed
        } else {
            raw_stream
        };

        let text = extract_hwp5_section_text(&decoded_stream);
        let aliases = extract_parenthetical_aliases(&text);
        pages.push(ParsedPage {
            page_no: (section_idx + 1) as i32,
            raw_markdown: text.clone(),
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks: Vec::new(),
            extracted_aliases: aliases,
            page_image_rgb: None,
            content_sha256: sha256_hex_str(&text),
        });
        section_idx += 1;
    }

    if pages.is_empty() || pages.iter().all(|p| p.raw_markdown.trim().is_empty()) {
        if let Ok(mut prv) = comp.open_stream("PrvText") {
            let mut prv_bytes = Vec::new();
            if prv.read_to_end(&mut prv_bytes).is_ok() {
                let text = decode_utf16_le_lossy(&prv_bytes);
                if !text.trim().is_empty() {
                    let aliases = extract_parenthetical_aliases(&text);
                    pages = vec![ParsedPage {
                        page_no: 1,
                        raw_markdown: text.clone(),
                        tables_html: Vec::new(),
                        formulas: Vec::new(),
                        fields: Vec::new(),
                        blocks: Vec::new(),
                        extracted_aliases: aliases,
                        page_image_rgb: None,
                        content_sha256: sha256_hex_str(&text),
                    }];
                }
            }
        }
    }

    if pages.is_empty() {
        return Err(OpenRagError::Validation(
            "HWP 5.0 file contains no BodyText/Section or PrvText streams".into(),
        ));
    }

    let title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("Untitled HWP")
        .to_string();
    let full_markdown = pages
        .iter()
        .map(|p| p.raw_markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let extracted_aliases = extract_parenthetical_aliases(&full_markdown);

    Ok(ParsedDocument {
        title,
        format: DocumentFormat::Hwp,
        mime_type: DocumentFormat::Hwp.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages,
        full_markdown,
        extracted_aliases,
        ocr_policy: None,
        parser_engine: "hwp5_rust".to_string(),
    })
}

fn extract_hwp5_section_text(data: &[u8]) -> String {
    const HWPTAG_PARA_TEXT: u32 = 67;
    let mut pos = 0usize;
    let mut paragraphs: Vec<String> = Vec::new();

    while pos + 4 <= data.len() {
        let header = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;
        let tag_id = header & 0x3FF;
        let mut size = ((header >> 20) & 0xFFF) as usize;
        if size == 0xFFF {
            if pos + 4 > data.len() {
                break;
            }
            size = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
        }
        if pos + size > data.len() {
            break;
        }
        let payload = &data[pos..pos + size];
        pos += size;

        if tag_id == HWPTAG_PARA_TEXT {
            let text = decode_hwp5_para_text(payload);
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                paragraphs.push(trimmed.to_string());
            }
        }
    }

    paragraphs.join("\n\n")
}

fn decode_hwp5_para_text(payload: &[u8]) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(payload.len() / 2);
    let mut i = 0usize;
    while i + 1 < payload.len() {
        let code = u16::from_le_bytes([payload[i], payload[i + 1]]);
        i += 2;
        match code {
            // Extended / inline HWP control characters consume 8 WCHAR units total (1 already read + 7 more)
            1..=8 | 11 | 12 | 14..=23 => {
                i = i.saturating_add(14);
            }
            10 | 13 => units.push(b'\n' as u16),
            0 | 9 | 24..=31 => {}
            _ => units.push(code),
        }
    }
    String::from_utf16_lossy(&units)
}

fn decode_utf16_le_lossy(bytes: &[u8]) -> String {
    let mut units = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        units.push(u16::from_le_bytes([bytes[i], bytes[i + 1]]));
        i += 2;
    }
    String::from_utf16_lossy(&units)
}

fn resolve_general_ref(r: &BytesRef<'_>) -> String {
    if let Ok(Some(ch)) = r.resolve_char_ref() {
        return ch.to_string();
    }
    if let Some(ent) = resolve_predefined_entity(r.as_ref()) {
        return ent.to_string();
    }
    format!("&{};", r.as_ref())
}

fn local_name(s: &str) -> &str {
    s.rsplit(':').next().unwrap_or(s)
}

fn attr_str(e: &quick_xml::events::BytesStart<'_>, key: &str) -> Option<String> {
    for attr in e.attributes().flatten() {
        let k = local_name(attr.key.as_ref());
        if k == key {
            if let Ok(val) = attr.normalized_value(XmlVersion::Implicit1_0) {
                return Some(val.into_owned());
            }
        }
    }
    None
}

fn attr_u32(e: &quick_xml::events::BytesStart<'_>, key: &str) -> u32 {
    attr_str(e, key)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

fn attr_u32_ci(e: &quick_xml::events::BytesStart<'_>, key: &str) -> u32 {
    for attr in e.attributes().flatten() {
        let k = local_name(attr.key.as_ref());
        if k.eq_ignore_ascii_case(key) {
            if let Ok(val) = attr.normalized_value(XmlVersion::Implicit1_0) {
                return val.parse::<u32>().unwrap_or(0);
            }
        }
    }
    0
}

fn attr_bool(e: &quick_xml::events::BytesStart<'_>, key: &str) -> bool {
    attr_str(e, key)
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}
