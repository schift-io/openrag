//! Word (.docx) and PowerPoint (.pptx) text extraction.
//!
//! Both are zip archives of XML. Word paragraphs become Markdown paragraphs (headings from the
//! style's outline level, list items as `- `, tables as `| a | b |` rows) and Word's saved page
//! breaks split pages. Each PowerPoint slide becomes one page, in presentation order, followed by
//! its speaker notes.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use openrag_core::OpenRagError;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesRef, BytesStart, Event};
use quick_xml::{Reader, XmlVersion};
use zip::ZipArchive;

use crate::aliases::extract_parenthetical_aliases;
use crate::model::{sha256_hex_bytes, sha256_hex_str, DocumentFormat, ParsedDocument, ParsedPage};

/// Upper bound for one XML part; larger parts are refused (zip bombs).
const MAX_PART_BYTES: u64 = 64 * 1024 * 1024;

type Archive<'a> = ZipArchive<Cursor<&'a [u8]>>;

/// The Office format of a zip archive, by the parts it contains.
pub fn office_zip_kind(bytes: &[u8]) -> Option<DocumentFormat> {
    let archive = ZipArchive::new(Cursor::new(bytes)).ok()?;
    let mut kind = None;
    for name in archive.file_names() {
        if name == "word/document.xml" {
            kind = Some(DocumentFormat::Docx);
            break;
        }
        if name == "ppt/presentation.xml" {
            kind = Some(DocumentFormat::Pptx);
            break;
        }
    }
    kind
}

fn open<'a>(bytes: &'a [u8], what: &str) -> Result<Archive<'a>, OpenRagError> {
    ZipArchive::new(Cursor::new(bytes))
        .map_err(|e| OpenRagError::Validation(format!("invalid {what} file (not a zip archive): {e}")))
}

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

fn local(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn attr(e: &BytesStart<'_>, key: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| local(a.key.as_ref()) == key)
        .and_then(|a| a.normalized_value(XmlVersion::Implicit1_0).ok().map(|v| v.into_owned()))
}

/// A namespaced attribute such as `r:id`, told apart from a plain `id` on the same element.
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

fn title_of(filename: Option<&str>, fallback: &str) -> String {
    filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or(fallback)
        .to_string()
}

fn markdown_table(rows: &[Vec<String>]) -> String {
    rows.iter()
        .filter(|r| r.iter().any(|c| !c.is_empty()))
        .map(|r| format!("| {} |", r.join(" | ")))
        .collect::<Vec<_>>()
        .join("\n")
}

fn finish(
    bytes: &[u8],
    filename: Option<&str>,
    format: DocumentFormat,
    engine: &str,
    pages: Vec<(i32, String)>,
    empty_msg: &str,
) -> Result<ParsedDocument, OpenRagError> {
    let pages: Vec<ParsedPage> = pages
        .into_iter()
        .filter(|(_, md)| !md.trim().is_empty())
        .map(|(page_no, md)| ParsedPage {
            page_no,
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
    if pages.is_empty() {
        return Err(OpenRagError::Validation(empty_msg.into()));
    }
    let full_markdown = pages.iter().map(|p| p.raw_markdown.as_str()).collect::<Vec<_>>().join("\n\n");
    Ok(ParsedDocument {
        title: title_of(filename, "Untitled Document"),
        format,
        mime_type: format.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        extracted_aliases: extract_parenthetical_aliases(&full_markdown),
        full_markdown,
        pages,
        ocr_policy: None,
        parser_engine: engine.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Word
// ---------------------------------------------------------------------------

/// styleId -> heading level, from each style's outline level (`제목 1` and `Heading 1` both
/// carry outlineLvl 0) or its name.
fn docx_heading_styles(xml: &str) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut current: Option<String> = None;
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if local(e.name().as_ref()) == "style" => {
                current = attr(&e, "styleId");
            }
            Ok(Event::End(e)) if local(e.name().as_ref()) == "style" => current = None,
            Ok(Event::Start(e) | Event::Empty(e)) => {
                let Some(id) = current.as_ref() else { continue };
                match local(e.name().as_ref()) {
                    "outlineLvl" => {
                        if let Some(lvl) = attr(&e, "val").and_then(|v| v.parse::<usize>().ok()) {
                            if lvl < 9 {
                                out.insert(id.clone(), (lvl + 1).min(6));
                            }
                        }
                    }
                    "name" => {
                        let name = attr(&e, "val").unwrap_or_default().to_ascii_lowercase();
                        let level = match name.as_str() {
                            "title" => Some(1),
                            n => n.strip_prefix("heading ").and_then(|d| d.trim().parse::<usize>().ok()),
                        };
                        if let Some(l) = level.filter(|l| (1..=9).contains(l)) {
                            out.entry(id.clone()).or_insert(l.min(6));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

#[derive(Default)]
struct Para {
    text: String,
    heading: Option<usize>,
    list: bool,
}

#[derive(Default)]
struct Table {
    rows: Vec<Vec<String>>,
    row: Vec<String>,
    cell: String,
}

#[derive(Default)]
struct WordWriter {
    pages: Vec<(i32, String)>,
    blocks: Vec<String>,
    break_pending: bool,
}

impl WordWriter {
    fn push_block(&mut self, block: String) {
        if !block.trim().is_empty() {
            self.blocks.push(block);
        }
    }

    fn new_page(&mut self) {
        self.break_pending = false;
        if self.blocks.is_empty() {
            return;
        }
        let page_no = i32::try_from(self.pages.len() + 1).unwrap_or(i32::MAX);
        self.pages.push((page_no, std::mem::take(&mut self.blocks).join("\n\n")));
    }
}

fn docx_body(xml: &str, styles: &HashMap<String, usize>) -> Result<Vec<(i32, String)>, OpenRagError> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut w = WordWriter::default();
    let mut paras: Vec<Para> = Vec::new();
    let mut tables: Vec<Table> = Vec::new();
    let mut run_depth = 0usize;
    let mut in_text = false;

    let page_marker = |w: &mut WordWriter, paras: &[Para], tables: &[Table]| {
        let at_start = paras.last().is_none_or(|p| p.text.trim().is_empty());
        if tables.is_empty() && paras.len() <= 1 && at_start {
            w.new_page();
        } else {
            w.break_pending = true;
        }
    };

    loop {
        let ev = reader
            .read_event_into(&mut buf)
            .map_err(|e| OpenRagError::Validation(format!("corrupt word/document.xml: {e}")))?;
        match ev {
            Event::Start(e) => match local(e.name().as_ref()) {
                "p" => paras.push(Para::default()),
                "r" => run_depth += 1,
                "t" => in_text = run_depth > 0,
                "tbl" => tables.push(Table::default()),
                "tr" => {
                    if let Some(t) = tables.last_mut() {
                        t.row.clear();
                    }
                }
                "tc" => {
                    if let Some(t) = tables.last_mut() {
                        t.cell.clear();
                    }
                }
                "pStyle" => {
                    if let (Some(p), Some(level)) = (paras.last_mut(), attr(&e, "val").and_then(|v| styles.get(&v))) {
                        p.heading = Some(*level);
                    }
                }
                "numPr" => {
                    if let Some(p) = paras.last_mut() {
                        p.list = true;
                    }
                }
                _ => {}
            },
            Event::Empty(e) => match local(e.name().as_ref()) {
                "pStyle" => {
                    if let (Some(p), Some(level)) = (paras.last_mut(), attr(&e, "val").and_then(|v| styles.get(&v))) {
                        p.heading = Some(*level);
                    }
                }
                "outlineLvl" => {
                    if let (Some(p), Some(lvl)) = (paras.last_mut(), attr(&e, "val").and_then(|v| v.parse::<usize>().ok())) {
                        if lvl < 9 {
                            p.heading = Some((lvl + 1).min(6));
                        }
                    }
                }
                "numPr" => {
                    if let Some(p) = paras.last_mut() {
                        p.list = true;
                    }
                }
                "tab" if run_depth > 0 => {
                    if let Some(p) = paras.last_mut() {
                        p.text.push('\t');
                    }
                }
                "br" | "cr" if run_depth > 0 => {
                    if attr(&e, "type").as_deref() == Some("page") {
                        page_marker(&mut w, &paras, &tables);
                    } else if let Some(p) = paras.last_mut() {
                        p.text.push('\n');
                    }
                }
                "lastRenderedPageBreak" => page_marker(&mut w, &paras, &tables),
                _ => {}
            },
            Event::Text(t) if in_text => {
                if let Some(p) = paras.last_mut() {
                    p.text.push_str(&t.xml10_content());
                }
            }
            Event::GeneralRef(r) if in_text => {
                if let Some(p) = paras.last_mut() {
                    p.text.push_str(&general_ref(&r));
                }
            }
            Event::End(e) => match local(e.name().as_ref()) {
                "t" => in_text = false,
                "r" => run_depth = run_depth.saturating_sub(1),
                "p" => {
                    let Some(p) = paras.pop() else { continue };
                    let text = p.text.trim().to_string();
                    if let Some(t) = tables.last_mut() {
                        if !text.is_empty() {
                            if !t.cell.is_empty() {
                                t.cell.push(' ');
                            }
                            t.cell.push_str(&text.replace('\n', " "));
                        }
                        continue;
                    }
                    if !text.is_empty() {
                        let block = match (p.heading, p.list) {
                            (Some(level), _) => format!("{} {}", "#".repeat(level), text.replace('\n', " ")),
                            (None, true) => format!("- {text}"),
                            (None, false) => text,
                        };
                        w.push_block(block);
                    }
                    if paras.is_empty() && w.break_pending {
                        w.new_page();
                    }
                }
                "tc" => {
                    if let Some(t) = tables.last_mut() {
                        let cell = std::mem::take(&mut t.cell).replace('|', "\\|");
                        t.row.push(cell.trim().to_string());
                    }
                }
                "tr" => {
                    if let Some(t) = tables.last_mut() {
                        let row = std::mem::take(&mut t.row);
                        t.rows.push(row);
                    }
                }
                "tbl" => {
                    let Some(t) = tables.pop() else { continue };
                    let rendered = markdown_table(&t.rows);
                    if let Some(parent) = tables.last_mut() {
                        // A table inside a cell is kept as that cell's text.
                        let flat = t.rows.iter().map(|r| r.join(" ")).collect::<Vec<_>>().join(" ");
                        if !parent.cell.is_empty() {
                            parent.cell.push(' ');
                        }
                        parent.cell.push_str(flat.trim());
                    } else {
                        w.push_block(rendered);
                        if w.break_pending {
                            w.new_page();
                        }
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    w.new_page();
    Ok(w.pages)
}

/// Text of a Word document as Markdown, split at Word's saved page breaks.
pub fn parse_docx(bytes: &[u8], filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    let mut archive = open(bytes, "Word")?;
    let body = read_part(&mut archive, "word/document.xml")?
        .ok_or_else(|| OpenRagError::Validation("not a Word document: word/document.xml is missing".into()))?;
    let styles = read_part(&mut archive, "word/styles.xml")?
        .map(|s| docx_heading_styles(&s))
        .unwrap_or_default();
    let pages = docx_body(&body, &styles)?;
    finish(bytes, filename, DocumentFormat::Docx, "docx_native", pages, "this Word document has no text")
}

// ---------------------------------------------------------------------------
// PowerPoint
// ---------------------------------------------------------------------------

/// Relationship id -> target part, resolved against the folder of the part that owns the rels.
fn rels(xml: &str, base_dir: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e) | Event::Empty(e)) if local(e.name().as_ref()) == "Relationship" => {
                let (Some(id), Some(target)) = (attr(&e, "Id"), attr(&e, "Target")) else { continue };
                if attr(&e, "TargetMode").as_deref() == Some("External") {
                    continue;
                }
                let kind = attr(&e, "Type").unwrap_or_default();
                out.push((id, resolve_part(base_dir, &target), kind));
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

fn resolve_part(base_dir: &str, target: &str) -> String {
    if let Some(abs) = target.strip_prefix('/') {
        return abs.to_string();
    }
    let mut parts: Vec<&str> = base_dir.split('/').filter(|p| !p.is_empty()).collect();
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

fn slide_order(archive: &mut Archive<'_>) -> Result<Vec<String>, OpenRagError> {
    let pres = read_part(archive, "ppt/presentation.xml")?
        .ok_or_else(|| OpenRagError::Validation("not a PowerPoint file: ppt/presentation.xml is missing".into()))?;
    let pres_rels = read_part(archive, "ppt/_rels/presentation.xml.rels")?
        .map(|r| rels(&r, "ppt"))
        .unwrap_or_default();
    let mut ids = Vec::new();
    let mut reader = Reader::from_str(&pres);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e) | Event::Empty(e)) if local(e.name().as_ref()) == "sldId" => {
                if let Some(rid) = prefixed_attr(&e, "id") {
                    ids.push(rid);
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    let mut order: Vec<String> = ids
        .iter()
        .filter_map(|rid| pres_rels.iter().find(|(id, _, _)| id == rid).map(|(_, t, _)| t.clone()))
        .collect();
    if order.is_empty() {
        // No usable slide list: fall back to slideN.xml in number order.
        let mut numbered: Vec<(u32, String)> = archive
            .file_names()
            .filter_map(|n| {
                let num = n.strip_prefix("ppt/slides/slide")?.strip_suffix(".xml")?.parse::<u32>().ok()?;
                Some((num, n.to_string()))
            })
            .collect();
        numbered.sort();
        order = numbered.into_iter().map(|(_, n)| n).collect();
    }
    Ok(order)
}

/// Text of one slide (or notes) part. Title placeholders become `# ` headings; with
/// `body_only`, only body placeholders are read (the notes page also holds a slide image and
/// the slide number).
fn slide_text(xml: &str, body_only: bool) -> Result<String, OpenRagError> {
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    let mut blocks: Vec<String> = Vec::new();
    let mut shape: Vec<String> = Vec::new();
    let mut shape_depth = 0usize;
    let mut placeholder: Option<String> = None;
    let mut para = String::new();
    let mut in_text = false;
    let mut field_depth = 0usize;
    let mut tables: Vec<Table> = Vec::new();

    let keep = |ph: &Option<String>| -> bool {
        if body_only {
            ph.as_deref() == Some("body")
        } else {
            !matches!(ph.as_deref(), Some("sldNum" | "dt" | "ftr" | "hdr"))
        }
    };

    loop {
        let ev = reader
            .read_event_into(&mut buf)
            .map_err(|e| OpenRagError::Validation(format!("corrupt slide XML: {e}")))?;
        match ev {
            Event::Start(e) => match local(e.name().as_ref()) {
                "sp" | "graphicFrame" => {
                    if shape_depth == 0 {
                        placeholder = None;
                        shape.clear();
                    }
                    shape_depth += 1;
                }
                "p" => para.clear(),
                "ph" => placeholder = Some(attr(&e, "type").unwrap_or_else(|| "body".into())),
                "br" => para.push('\n'),
                "t" => in_text = field_depth == 0,
                "fld" => field_depth += 1,
                "tbl" => tables.push(Table::default()),
                "tr" => {
                    if let Some(t) = tables.last_mut() {
                        t.row.clear();
                    }
                }
                "tc" => {
                    if let Some(t) = tables.last_mut() {
                        t.cell.clear();
                    }
                }
                _ => {}
            },
            Event::Empty(e) => match local(e.name().as_ref()) {
                "ph" => placeholder = Some(attr(&e, "type").unwrap_or_else(|| "body".into())),
                "br" => para.push('\n'),
                _ => {}
            },
            Event::Text(t) if in_text => para.push_str(&t.xml10_content()),
            Event::GeneralRef(r) if in_text => para.push_str(&general_ref(&r)),
            Event::End(e) => match local(e.name().as_ref()) {
                "t" => in_text = false,
                "fld" => field_depth = field_depth.saturating_sub(1),
                "p" => {
                    let text = para.trim().to_string();
                    para.clear();
                    if text.is_empty() {
                        continue;
                    }
                    if let Some(t) = tables.last_mut() {
                        if !t.cell.is_empty() {
                            t.cell.push(' ');
                        }
                        t.cell.push_str(&text.replace('\n', " "));
                    } else {
                        shape.push(text);
                    }
                }
                "tc" => {
                    if let Some(t) = tables.last_mut() {
                        let cell = std::mem::take(&mut t.cell).replace('|', "\\|");
                        t.row.push(cell.trim().to_string());
                    }
                }
                "tr" => {
                    if let Some(t) = tables.last_mut() {
                        let row = std::mem::take(&mut t.row);
                        t.rows.push(row);
                    }
                }
                "tbl" => {
                    if let Some(t) = tables.pop() {
                        shape.push(markdown_table(&t.rows));
                    }
                }
                "sp" | "graphicFrame" => {
                    shape_depth = shape_depth.saturating_sub(1);
                    if shape_depth == 0 && !shape.is_empty() && keep(&placeholder) {
                        let is_title = matches!(placeholder.as_deref(), Some("title" | "ctrTitle"));
                        let mut lines = std::mem::take(&mut shape);
                        if is_title {
                            let joined = lines.join(" ").replace('\n', " ");
                            lines = vec![format!("# {joined}")];
                        }
                        blocks.push(lines.join("\n"));
                    }
                    if shape_depth == 0 {
                        shape.clear();
                    }
                }
                _ => {}
            },
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(blocks.join("\n\n"))
}

/// One page per slide, in presentation order, each followed by its speaker notes.
pub fn parse_pptx(bytes: &[u8], filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    let mut archive = open(bytes, "PowerPoint")?;
    let order = slide_order(&mut archive)?;
    let mut pages = Vec::with_capacity(order.len());
    for (idx, part) in order.iter().enumerate() {
        let Some(xml) = read_part(&mut archive, part)? else { continue };
        let mut text = slide_text(&xml, false)?;
        let (dir, file) = part.rsplit_once('/').unwrap_or(("", part.as_str()));
        let rels_part = format!("{dir}/_rels/{file}.rels");
        if let Some(r) = read_part(&mut archive, &rels_part)? {
            for (_, target, kind) in rels(&r, dir) {
                if !kind.ends_with("/notesSlide") {
                    continue;
                }
                if let Some(notes_xml) = read_part(&mut archive, &target)? {
                    let notes = slide_text(&notes_xml, true)?;
                    if !notes.trim().is_empty() {
                        if !text.is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str(&notes);
                    }
                }
            }
        }
        pages.push((i32::try_from(idx + 1).unwrap_or(i32::MAX), text));
    }
    finish(bytes, filename, DocumentFormat::Pptx, "pptx_native", pages, "this PowerPoint file has no text")
}
