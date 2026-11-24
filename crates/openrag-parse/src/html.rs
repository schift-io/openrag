//! HTML pages as Markdown text: headings as `#`, list items as `- `, table rows as `| a | b |`.
//! Scripts, styles and other non-text elements are dropped. The character set comes from the
//! byte-order mark or `<meta charset>`; pages without one are read as UTF-8, else as EUC-KR.

use std::path::Path;

use openrag_core::OpenRagError;

use crate::aliases::extract_parenthetical_aliases;
use crate::model::{sha256_hex_bytes, sha256_hex_str, DocumentFormat, ParsedDocument, ParsedPage};

/// Elements whose content is never text.
const SKIP_CONTENT: &[&str] = &["script", "style", "noscript", "template", "svg", "math", "iframe", "object", "canvas"];

/// Elements that start and end a paragraph.
const BLOCK: &[&str] = &[
    "p", "div", "section", "article", "header", "footer", "main", "aside", "nav", "blockquote",
    "figure", "figcaption", "form", "fieldset", "address", "dl", "dt", "dd", "ul", "ol", "table",
    "thead", "tbody", "tfoot", "caption", "hr", "body", "details", "summary",
];

/// Whether the bytes look like an HTML page (for files without an extension).
pub fn looks_like_html(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    let lower = String::from_utf8_lossy(head).to_ascii_lowercase();
    let t = lower.trim_start_matches('\u{feff}').trim_start();
    t.starts_with("<!doctype html") || t.starts_with("<html")
}

fn decode(bytes: &[u8]) -> String {
    if let Some((enc, bom_len)) = encoding_rs::Encoding::for_bom(bytes) {
        return enc.decode_without_bom_handling(&bytes[bom_len..]).0.into_owned();
    }
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]).to_ascii_lowercase();
    if let Some(pos) = head.find("charset=") {
        let label: String = head[pos + 8..]
            .trim_start_matches(['"', '\''])
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if let Some(enc) = encoding_rs::Encoding::for_label(label.as_bytes()) {
            return enc.decode_without_bom_handling(bytes).0.into_owned();
        }
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => encoding_rs::EUC_KR.decode_without_bom_handling(bytes).0.into_owned(),
    }
}

fn entity(name: &str) -> Option<String> {
    if let Some(num) = name.strip_prefix('#') {
        let code = match num.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => num.parse::<u32>().ok()?,
        };
        return char::from_u32(code).map(|c| c.to_string());
    }
    let s = match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" | "ensp" | "emsp" | "thinsp" => " ",
        "middot" => "·",
        "bull" => "•",
        "hellip" => "…",
        "ndash" => "–",
        "mdash" => "—",
        "lsquo" => "‘",
        "rsquo" => "’",
        "ldquo" => "“",
        "rdquo" => "”",
        "laquo" => "«",
        "raquo" => "»",
        "copy" => "©",
        "reg" => "®",
        "trade" => "™",
        "times" => "×",
        "divide" => "÷",
        "deg" => "°",
        "plusmn" => "±",
        "larr" => "←",
        "rarr" => "→",
        "uarr" => "↑",
        "darr" => "↓",
        "won" => "₩",
        "euro" => "€",
        "pound" => "£",
        "yen" => "¥",
        "cent" => "¢",
        "sect" => "§",
        "para" => "¶",
        _ => return None,
    };
    Some(s.to_string())
}

fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let end = after.find(';').filter(|&e| e > 0 && e <= 10);
        match end.and_then(|e| entity(&after[..e]).map(|s| (e, s))) {
            Some((e, s)) => {
                out.push_str(&s);
                rest = &after[e + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

#[derive(Default)]
struct Out {
    text: String,
    pre: usize,
}

impl Out {
    fn ends_with_space(&self) -> bool {
        self.text.is_empty() || self.text.ends_with([' ', '\n'])
    }

    fn push_text(&mut self, raw: &str) {
        let decoded = decode_entities(raw);
        if self.pre > 0 {
            self.text.push_str(&decoded);
            return;
        }
        let mut first = true;
        let starts_ws = decoded.starts_with(char::is_whitespace);
        for word in decoded.split_whitespace() {
            if (!first || starts_ws) && !self.ends_with_space() {
                self.text.push(' ');
            }
            self.text.push_str(word);
            first = false;
        }
        if !first && decoded.ends_with(char::is_whitespace) {
            self.text.push(' ');
        }
    }

    fn newline(&mut self) {
        let trimmed = self.text.trim_end_matches(' ').len();
        self.text.truncate(trimmed);
        if !self.text.is_empty() && !self.text.ends_with('\n') {
            self.text.push('\n');
        }
    }

    fn paragraph(&mut self) {
        self.newline();
        if !self.text.is_empty() && !self.text.ends_with("\n\n") {
            self.text.push('\n');
        }
    }
}

/// Reads a tag starting at `<`; returns (lowercase name, is_closing, index after `>`).
fn read_tag(s: &str, start: usize) -> Option<(String, bool, usize)> {
    let bytes = s.as_bytes();
    let mut i = start + 1;
    let closing = bytes.get(i) == Some(&b'/');
    if closing {
        i += 1;
    }
    let name_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-' || bytes[i] == b':') {
        i += 1;
    }
    if i == name_start {
        return None;
    }
    let name = s[name_start..i].to_ascii_lowercase();
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => {}
            None if b == b'"' || b == b'\'' => quote = Some(b),
            None if b == b'>' => return Some((name, closing, i + 1)),
            None => {}
        }
        i += 1;
    }
    Some((name, closing, bytes.len()))
}

fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    hay.get(from..)?.to_ascii_lowercase().find(needle).map(|p| p + from)
}

/// (`<title>`, body as Markdown).
pub fn html_to_markdown(html: &str) -> (Option<String>, String) {
    let mut out = Out::default();
    let mut title: Option<String> = None;
    let mut i = 0usize;
    let s = html;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        let Some(lt) = s[i..].find('<').map(|p| p + i) else {
            out.push_text(&s[i..]);
            break;
        };
        if lt > i {
            out.push_text(&s[i..lt]);
        }
        let rest = &s[lt..];
        if rest.starts_with("<!--") {
            i = rest.find("-->").map_or(bytes.len(), |p| lt + p + 3);
            continue;
        }
        if rest.starts_with("<![CDATA[") {
            let end = rest.find("]]>").map_or(bytes.len(), |p| lt + p);
            out.push_text(&s[lt + 9..end.max(lt + 9)]);
            i = (end + 3).min(bytes.len());
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            i = rest.find('>').map_or(bytes.len(), |p| lt + p + 1);
            continue;
        }
        let Some((name, closing, after)) = read_tag(s, lt) else {
            out.push_text("<");
            i = lt + 1;
            continue;
        };
        i = after;
        if !closing && SKIP_CONTENT.contains(&name.as_str()) {
            let close = format!("</{name}");
            i = find_ci(s, &close, i)
                .and_then(|p| s[p..].find('>').map(|q| p + q + 1))
                .unwrap_or(bytes.len());
            continue;
        }
        if !closing && name == "title" {
            let end = find_ci(s, "</title", i).unwrap_or(bytes.len());
            let t = decode_entities(&s[i..end]).split_whitespace().collect::<Vec<_>>().join(" ");
            if !t.is_empty() && title.is_none() {
                title = Some(t);
            }
            i = s[end..].find('>').map_or(bytes.len(), |q| end + q + 1);
            continue;
        }
        match name.as_str() {
            "br" => out.newline(),
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                out.paragraph();
                if !closing {
                    let level = usize::from(name.as_bytes()[1] - b'0');
                    out.text.push_str(&"#".repeat(level));
                    out.text.push(' ');
                }
            }
            "li" => {
                out.newline();
                if !closing {
                    out.text.push_str("- ");
                }
            }
            "tr" => {
                out.newline();
                if !closing {
                    out.text.push('|');
                }
            }
            "td" | "th" => {
                if closing {
                    let trimmed = out.text.trim_end_matches(' ').len();
                    out.text.truncate(trimmed);
                    out.text.push_str(" |");
                } else {
                    out.text.push(' ');
                }
            }
            "pre" => {
                out.paragraph();
                if closing {
                    out.pre = out.pre.saturating_sub(1);
                } else {
                    out.pre += 1;
                }
            }
            n if BLOCK.contains(&n) => out.paragraph(),
            _ => {}
        }
    }

    let mut lines: Vec<String> = Vec::new();
    for line in out.text.lines() {
        let l = line.trim_end();
        let l = if out.pre > 0 { l } else { l.trim_start() };
        // Lone markers left by empty elements.
        if l == "-" || l == "|" || l.chars().all(|c| c == '#') && !l.is_empty() {
            continue;
        }
        if l.is_empty() && lines.last().is_none_or(String::is_empty) {
            continue;
        }
        lines.push(l.to_string());
    }
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    (title, lines.join("\n"))
}

/// The text of an HTML page.
pub fn parse_html(bytes: &[u8], filename: Option<&str>) -> Result<ParsedDocument, OpenRagError> {
    let html = decode(bytes);
    let (page_title, md) = html_to_markdown(&html);
    if md.trim().is_empty() {
        return Err(OpenRagError::Validation("this HTML page has no text".into()));
    }
    let title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .map(ToString::to_string)
        .or(page_title)
        .unwrap_or_else(|| "Untitled Document".into());
    let aliases = extract_parenthetical_aliases(&md);
    let page = ParsedPage {
        page_no: 1,
        content_sha256: sha256_hex_str(&md),
        raw_markdown: md.clone(),
        tables_html: Vec::new(),
        formulas: Vec::new(),
        fields: Vec::new(),
        blocks: Vec::new(),
        extracted_aliases: aliases.clone(),
        page_image_rgb: None,
    };
    Ok(ParsedDocument {
        title,
        format: DocumentFormat::Html,
        mime_type: DocumentFormat::Html.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages: vec![page],
        full_markdown: md,
        extracted_aliases: aliases,
        ocr_policy: None,
        parser_engine: "html_native".to_string(),
    })
}
