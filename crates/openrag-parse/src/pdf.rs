use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use flate2::read::ZlibDecoder;
use openrag_core::{OpenRagError, ParseConfig};
use tokio::process::Command;

use crate::aliases::extract_parenthetical_aliases;
use crate::model::{sha256_hex_bytes, sha256_hex_str, DocumentFormat, ParsedDocument, ParsedPage};

#[derive(Debug, Clone)]
struct PdfObj<'a> {
    dict: Cow<'a, str>,
    stream: Option<&'a [u8]>,
}

pub async fn parse_pdf_native(
    bytes: &[u8],
    filename: Option<&str>,
    cfg: &ParseConfig,
) -> Result<ParsedDocument, OpenRagError> {
    parse_pdf_with_scan_pages(bytes, filename, cfg).await.map(|(doc, _)| doc)
}

/// Like [`parse_pdf_native`], plus the pages (0-based) that need OCR: pages without a text
/// layer that draw an image (a scan or a photo), or every page when none has any text.
pub async fn parse_pdf_with_scan_pages(
    bytes: &[u8],
    filename: Option<&str>,
    cfg: &ParseConfig,
) -> Result<(ParsedDocument, Vec<usize>), OpenRagError> {
    if !cfg.pdf_native_enabled {
        return Err(OpenRagError::Config(
            "pdf_native parser is disabled in ParseConfig".into(),
        ));
    }
    if bytes.len() < 16 {
        return Err(OpenRagError::Validation(
            "PDF payload is too short or empty".into(),
        ));
    }

    let header_slice = &bytes[..bytes.len().min(1024)];
    let has_pdf_header = header_slice.windows(5).any(|w| w == b"%PDF-");
    if !has_pdf_header {
        return Err(OpenRagError::Validation(
            "invalid PDF: missing %PDF- header".into(),
        ));
    }

    let has_eof = bytes.windows(5).any(|w| w == b"%%EOF")
        || (bytes.windows(9).any(|w| w == b"startxref") && bytes.windows(6).any(|w| w == b"endobj"));
    if !has_eof {
        return Err(OpenRagError::Validation(
            "corrupt or truncated PDF: missing %%EOF / endobj structure".into(),
        ));
    }

    let objects = parse_pdf_objects(bytes);
    if objects.is_empty() {
        return Err(OpenRagError::Validation(
            "corrupt PDF: no valid indirect objects found".into(),
        ));
    }

    let ordered_page_ids = collect_page_object_ids(&objects);
    let mut pages = Vec::new();
    let mut empty_pages = Vec::new();
    let mut scan_pages = Vec::new();

    for (idx, page_id) in ordered_page_ids.iter().enumerate() {
        let Some(page_obj) = objects.get(page_id) else {
            continue;
        };
        let mut page_text = extract_page_text(page_obj, &objects)?;
        if page_text.trim().is_empty() {
            if let Some(fallback_text) = try_pdftotext_page(bytes, idx + 1).await {
                page_text = fallback_text;
            }
        }
        if page_text.trim().is_empty() {
            empty_pages.push(pages.len());
            if page_draws_image(page_obj, &objects) {
                scan_pages.push(pages.len());
            }
        }
        let aliases = extract_parenthetical_aliases(&page_text);
        let sha = sha256_hex_str(&page_text);
        pages.push(ParsedPage {
            page_no: (idx + 1) as i32,
            raw_markdown: page_text,
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks: Vec::new(),
            extracted_aliases: aliases,
            page_image_rgb: None,
            content_sha256: sha,
        });
    }

    if pages.is_empty() {
        return Err(OpenRagError::Validation(
            "corrupt PDF: no /Page objects found".into(),
        ));
    }

    let full_markdown = pages
        .iter()
        .map(|p| p.raw_markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let extracted_aliases = extract_parenthetical_aliases(&full_markdown);
    let title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("Untitled PDF")
        .to_string();

    if empty_pages.len() == pages.len() {
        scan_pages = empty_pages;
    }
    let doc = ParsedDocument {
        title,
        format: DocumentFormat::Pdf,
        mime_type: DocumentFormat::Pdf.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages,
        full_markdown,
        extracted_aliases,
        ocr_policy: None,
        parser_engine: "pdf_native".to_string(),
    };
    Ok((doc, scan_pages))
}

/// Whether a page's resources hold an image XObject (`/Subtype /Image`), the mark of a scanned
/// or photographed page. Resources may sit on the page, behind a reference, or on a parent.
fn page_draws_image(page_obj: &PdfObj<'_>, objects: &HashMap<u32, PdfObj<'_>>) -> bool {
    let mut dict: Cow<'_, str> = page_obj.dict.clone();
    for _ in 0..4 {
        let resources = match parse_ref_list(&dict, "/Resources").first() {
            Some(id) if !inline_dict_follows(&dict, "/Resources") => {
                objects.get(id).map(|o| o.dict.clone())
            }
            _ => dict.find("/Resources").map(|p| Cow::Owned(dict[p..].to_string())),
        };
        if let Some(res) = resources {
            let Some(xo_pos) = res.find("/XObject") else {
                return false;
            };
            let xobjects: Cow<'_, str> = match parse_ref_list(&res, "/XObject").first() {
                Some(id) if !inline_dict_follows(&res, "/XObject") => {
                    objects.get(id).map_or(Cow::Borrowed(""), |o| o.dict.clone())
                }
                _ => {
                    let rest = &res[xo_pos + "/XObject".len()..];
                    Cow::Owned(rest.split(">>").next().unwrap_or("").to_string())
                }
            };
            return refs_in(&xobjects).into_iter().any(|id| {
                objects.get(&id).is_some_and(|o| {
                    o.dict.contains("/Subtype /Image") || o.dict.contains("/Subtype/Image")
                })
            });
        }
        match parse_ref_list(&dict, "/Parent").first().and_then(|id| objects.get(id)) {
            Some(parent) => dict = parent.dict.clone(),
            None => return false,
        }
    }
    false
}

fn inline_dict_follows(dict: &str, key: &str) -> bool {
    dict.find(key)
        .is_some_and(|p| dict[p + key.len()..].trim_start().starts_with("<<"))
}

/// Every `N G R` reference in a dictionary fragment.
fn refs_in(fragment: &str) -> Vec<u32> {
    let tokens: Vec<&str> = fragment
        .split(|c: char| c.is_whitespace() || c == '/' || c == '<' || c == '>' || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .collect();
    tokens
        .windows(3)
        .filter(|w| w[2] == "R" && w[1].parse::<u32>().is_ok())
        .filter_map(|w| w[0].parse::<u32>().ok())
        .collect()
}

fn parse_pdf_objects(bytes: &[u8]) -> HashMap<u32, PdfObj<'_>> {
    let mut map = HashMap::new();
    // Walk byte offsets in `bytes` using ASCII marker search on latin1-compatible view.
    let mut pos = 0usize;
    while pos < bytes.len() {
        let Some((obj_id, after_hdr)) = find_next_obj_header(bytes, pos) else {
            break;
        };
        let Some(endobj_rel) = find_subslice(&bytes[after_hdr..], b"endobj") else {
            break;
        };
        let body = &bytes[after_hdr..after_hdr + endobj_rel];
        pos = after_hdr + endobj_rel + 6;

        let stream_pos = find_subslice(body, b"stream");
        let (dict_bytes, stream_bytes) = if let Some(sp) = stream_pos {
            let mut s_start = sp + 6;
            if body.get(s_start..s_start + 2) == Some(b"\r\n") {
                s_start += 2;
            } else if body.get(s_start) == Some(&b'\n') || body.get(s_start) == Some(&b'\r') {
                s_start += 1;
            }
            let endstream_rel = find_subslice(&body[s_start..], b"endstream");
            let stream_slice = endstream_rel.map(|er| {
                let mut slice = &body[s_start..s_start + er];
                if slice.ends_with(b"\r\n") {
                    slice = &slice[..slice.len() - 2];
                } else if slice.ends_with(b"\n") || slice.ends_with(b"\r") {
                    slice = &slice[..slice.len() - 1];
                }
                slice
            });
            (&body[..sp], stream_slice)
        } else {
            (body, None)
        };

        let dict_str = String::from_utf8_lossy(dict_bytes);
        map.insert(
            obj_id,
            PdfObj {
                dict: dict_str,
                stream: stream_bytes,
            },
        );
    }
    map
}

fn find_next_obj_header(bytes: &[u8], mut start: usize) -> Option<(u32, usize)> {
    while let Some(rel) = find_subslice(&bytes[start..], b" obj") {
        let obj_kw = start + rel;
        let after = obj_kw + 4;
        if let Some(&next_b) = bytes.get(after) {
            if !next_b.is_ascii_whitespace() && next_b != b'<' && next_b != b'[' {
                start = after;
                continue;
            }
        }
        // Scan backwards for `<id> <gen> obj`
        let line_start = bytes[..obj_kw]
            .iter()
            .rposition(|&b| b == b'\n' || b == b'\r')
            .map_or(0, |p| p + 1);
        let prefix = std::str::from_utf8(&bytes[line_start..obj_kw]).ok()?;
        let parts: Vec<&str> = prefix.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(id), Ok(_gen)) = (
                parts[parts.len() - 2].parse::<u32>(),
                parts[parts.len() - 1].parse::<u32>(),
            ) {
                return Some((id, after));
            }
        }
        start = after;
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn collect_page_object_ids(objects: &HashMap<u32, PdfObj<'_>>) -> Vec<u32> {
    // Prefer /Type /Pages /Kids tree traversal if root /Pages exists
    let mut roots: Vec<u32> = objects
        .iter()
        .filter(|(_, obj)| is_pages_node(&obj.dict) && !obj.dict.contains("/Parent"))
        .map(|(&id, _)| id)
        .collect();
    roots.sort_unstable();

    let mut ordered = Vec::new();
    let mut visited = std::collections::HashSet::new();
    for root in roots {
        walk_pages_tree(root, objects, &mut ordered, &mut visited);
    }

    if !ordered.is_empty() {
        return ordered;
    }

    let mut fallback: Vec<u32> = objects
        .iter()
        .filter(|(_, obj)| is_leaf_page(&obj.dict))
        .map(|(&id, _)| id)
        .collect();
    fallback.sort_unstable();
    fallback
}

fn walk_pages_tree(
    node_id: u32,
    objects: &HashMap<u32, PdfObj<'_>>,
    out: &mut Vec<u32>,
    visited: &mut std::collections::HashSet<u32>,
) {
    if !visited.insert(node_id) {
        return;
    }
    let Some(obj) = objects.get(&node_id) else {
        return;
    };
    if is_leaf_page(&obj.dict) {
        out.push(node_id);
        return;
    }
    for kid in parse_ref_list(&obj.dict, "/Kids") {
        walk_pages_tree(kid, objects, out, visited);
    }
}

fn is_pages_node(dict: &str) -> bool {
    dict.contains("/Type /Pages") || dict.contains("/Type/Pages")
}

fn is_leaf_page(dict: &str) -> bool {
    (dict.contains("/Type /Page") || dict.contains("/Type/Page")) && !is_pages_node(dict)
}

fn parse_ref_list(dict: &str, key: &str) -> Vec<u32> {
    let Some(pos) = dict.find(key) else {
        return Vec::new();
    };
    let rest = &dict[pos + key.len()..];
    let trimmed = rest.trim_start();
    let segment = if let Some(open) = trimmed.strip_prefix('[') {
        open.split(']').next().unwrap_or("")
    } else {
        // Single ref like `4 0 R`
        let tokens: Vec<&str> = trimmed.split_whitespace().take(3).collect();
        if tokens.len() == 3 && tokens[2].starts_with('R') {
            if let Ok(id) = tokens[0].parse::<u32>() {
                return vec![id];
            }
        }
        return Vec::new();
    };

    let tokens: Vec<&str> = segment.split_whitespace().collect();
    let mut ids = Vec::new();
    let mut i = 0usize;
    while i + 2 < tokens.len() {
        if tokens[i + 2] == "R" {
            if let Ok(id) = tokens[i].parse::<u32>() {
                ids.push(id);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    ids
}

fn extract_page_text(
    page_obj: &PdfObj<'_>,
    objects: &HashMap<u32, PdfObj<'_>>,
) -> Result<String, OpenRagError> {
    let content_ids = parse_ref_list(&page_obj.dict, "/Contents");
    let mut combined = Vec::new();

    for cid in content_ids {
        let Some(cobj) = objects.get(&cid) else {
            continue;
        };
        let Some(stream) = cobj.stream else {
            continue;
        };
        let decompressed = if cobj.dict.contains("/FlateDecode") {
            let mut dec = ZlibDecoder::new(stream);
            let mut buf = Vec::new();
            if dec.read_to_end(&mut buf).is_ok() {
                buf
            } else {
                stream.to_vec()
            }
        } else {
            stream.to_vec()
        };
        let chunk = extract_text_operators(&decompressed);
        if !chunk.trim().is_empty() {
            combined.push(chunk);
        }
    }

    Ok(combined.join("\n"))
}

fn extract_text_operators(stream: &[u8]) -> String {
    let text = String::from_utf8_lossy(stream);
    let bytes = text.as_bytes();
    let mut lines: Vec<String> = Vec::new();
    let mut current_line = String::new();
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'(' => {
                let (decoded, next_i) = parse_pdf_literal_string(bytes, i);
                i = next_i;
                let _look = skip_ws_slice(&bytes[i..]);
                current_line.push_str(&decoded);
            }
            b'<' if bytes.get(i + 1) != Some(&b'<') => {
                if let Some((decoded, next_i)) = parse_pdf_hex_string(bytes, i) {
                    i = next_i;
                    current_line.push_str(&decoded);
                } else {
                    i += 1;
                }
            }
            b'[' => {
                let (arr_text, next_i) = parse_pdf_tj_array(bytes, i);
                if !arr_text.is_empty() {
                    current_line.push_str(&arr_text);
                }
                i = next_i;
            }
            b'T' if matches!(bytes.get(i + 1), Some(b'*' | b'd' | b'D')) => {
                if !current_line.trim().is_empty() {
                    lines.push(current_line.trim().to_string());
                    current_line.clear();
                }
                i += 2;
            }
            b'E' if bytes.get(i + 1) == Some(&b'T') => {
                if !current_line.trim().is_empty() {
                    lines.push(current_line.trim().to_string());
                    current_line.clear();
                }
                i += 2;
            }
            _ => i += 1,
        }
    }
    if !current_line.trim().is_empty() {
        lines.push(current_line.trim().to_string());
    }
    lines.join("\n")
}

fn skip_ws_slice(s: &[u8]) -> &[u8] {
    let mut idx = 0;
    while idx < s.len() && s[idx].is_ascii_whitespace() {
        idx += 1;
    }
    &s[idx..]
}

fn parse_pdf_literal_string(bytes: &[u8], start: usize) -> (String, usize) {
    let mut i = start + 1;
    let mut depth = 1usize;
    let mut raw: Vec<u8> = Vec::new();
    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    break;
                }
                match bytes[i] {
                    b'n' => raw.push(b'\n'),
                    b'r' => raw.push(b'\r'),
                    b't' => raw.push(b'\t'),
                    b'b' => raw.push(0x08),
                    b'f' => raw.push(0x0C),
                    b'(' | b')' | b'\\' => raw.push(bytes[i]),
                    b'0'..=b'7' => {
                        let mut val = bytes[i] - b'0';
                        for _ in 0..2 {
                            if let Some(&d @ b'0'..=b'7') = bytes.get(i + 1) {
                                val = val.wrapping_mul(8).wrapping_add(d - b'0');
                                i += 1;
                            } else {
                                break;
                            }
                        }
                        raw.push(val);
                    }
                    b'\r' | b'\n' => {}
                    other => raw.push(other),
                }
            }
            b'(' => {
                depth += 1;
                raw.push(b'(');
            }
            b')' => {
                depth -= 1;
                if depth > 0 {
                    raw.push(b')');
                }
            }
            b => raw.push(b),
        }
        i += 1;
    }

    let decoded = if raw.starts_with(&[0xFE, 0xFF]) {
        decode_utf16_be(&raw[2..])
    } else {
        String::from_utf8_lossy(&raw).into_owned()
    };
    (decoded, i)
}

fn parse_pdf_hex_string(bytes: &[u8], start: usize) -> Option<(String, usize)> {
    let close = bytes[start + 1..].iter().position(|&b| b == b'>')?;
    let hex_slice = &bytes[start + 1..start + 1 + close];
    let clean: Vec<u8> = hex_slice
        .iter()
        .copied()
        .filter(|b| b.is_ascii_hexdigit())
        .collect();
    let mut raw = Vec::with_capacity(clean.len().div_ceil(2));
    let mut idx = 0usize;
    while idx < clean.len() {
        let hi = hex_val(clean[idx])?;
        let lo = if idx + 1 < clean.len() {
            hex_val(clean[idx + 1])?
        } else {
            0
        };
        raw.push((hi << 4) | lo);
        idx += 2;
    }
    let decoded = if raw.starts_with(&[0xFE, 0xFF]) {
        decode_utf16_be(&raw[2..])
    } else {
        String::from_utf8_lossy(&raw).into_owned()
    };
    Some((decoded, start + 1 + close + 1))
}

fn parse_pdf_tj_array(bytes: &[u8], start: usize) -> (String, usize) {
    let mut i = start + 1;
    let mut out = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b']' => {
                i += 1;
                break;
            }
            b'(' => {
                let (s, next_i) = parse_pdf_literal_string(bytes, i);
                out.push_str(&s);
                i = next_i;
            }
            b'<' if bytes.get(i + 1) != Some(&b'<') => {
                if let Some((s, next_i)) = parse_pdf_hex_string(bytes, i) {
                    out.push_str(&s);
                    i = next_i;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    (out, i)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn decode_utf16_be(bytes: &[u8]) -> String {
    let mut units = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        units.push(u16::from_be_bytes([bytes[i], bytes[i + 1]]));
        i += 2;
    }
    String::from_utf16_lossy(&units)
}

async fn try_pdftotext_page(bytes: &[u8], page_no: usize) -> Option<String> {
    let path_var = std::env::var_os("PATH")?;
    let bin = std::env::split_paths(&path_var)
        .map(|d| d.join("pdftotext"))
        .find(|p| p.is_file())?;

    let tmp_path = std::env::temp_dir().join(format!("openrag-pdf-{}.pdf", uuid::Uuid::now_v7()));
    if tokio::fs::write(&tmp_path, bytes).await.is_err() {
        return None;
    }

    let mut cmd = Command::new(bin);
    cmd.arg("-f")
        .arg(page_no.to_string())
        .arg("-l")
        .arg(page_no.to_string())
        .arg(&tmp_path)
        .arg("-");

    let res = tokio::time::timeout(Duration::from_secs(10), cmd.output()).await;
    let _ = tokio::fs::remove_file(&tmp_path).await;
    let Ok(Ok(out)) = res else {
        return None;
    };
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use openrag_core::Sensitivity;

    pub(crate) fn test_parse_cfg() -> ParseConfig {
        ParseConfig {
            hwp_engine: "kordoc".into(),
            ocr_engine: "schift_ocr".into(),
            dewarp_enabled: false,
            pdf_native_enabled: true,
            default_ocr_mode: Sensitivity::Share,
            ocr_mode_by_auth_level: Vec::new(),
            reject_zdr_with_batch: true,
        }
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// A PDF whose pages either show text (`Some`) or only draw an image (`None`), the way a
    /// scanner writes them.
    pub(crate) fn pdf_with_pages(pages: &[Option<&str>]) -> Vec<u8> {
        let mut objs: Vec<String> = vec![String::new()]; // 1: Pages, filled in below
        objs.push("<< /Type /Catalog /Pages 1 0 R >>".into()); // 2
        let mut kids = Vec::new();
        for page in pages {
            let (resources, content) = match page {
                Some(text) => (
                    "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>".to_string(),
                    format!("BT /F1 12 Tf 72 700 Td ({text}) Tj ET"),
                ),
                None => {
                    objs.push("<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\nstream\n\u{0}\nendstream".into());
                    (format!("<< /XObject << /Im0 {} 0 R >> >>", objs.len()), "q 595 0 0 842 0 0 cm /Im0 Do Q".to_string())
                }
            };
            objs.push(format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()));
            let contents = objs.len();
            objs.push(format!(
                "<< /Type /Page /Parent 1 0 R /MediaBox [0 0 595 842] /Resources {resources} /Contents {contents} 0 R >>"
            ));
            kids.push(format!("{} 0 R", objs.len()));
        }
        objs[0] = format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), kids.len());
        let mut out = String::from("%PDF-1.4\n");
        for (i, o) in objs.iter().enumerate() {
            out.push_str(&format!("{} 0 obj\n{o}\nendobj\n", i + 1));
        }
        out.push_str("trailer\n<< /Root 2 0 R >>\n%%EOF\n");
        out.into_bytes()
    }

    #[tokio::test]
    async fn scanned_pages_are_the_image_pages_without_text() -> TestResult {
        let cfg = test_parse_cfg();
        let (doc, scan) =
            parse_pdf_with_scan_pages(&pdf_with_pages(&[Some("Hello"), None]), Some("mixed.pdf"), &cfg).await?;
        assert_eq!(doc.pages.len(), 2);
        assert!(doc.pages[0].raw_markdown.contains("Hello"));
        assert_eq!(scan, [1]);

        let (_, scan) = parse_pdf_with_scan_pages(&pdf_with_pages(&[None, None]), None, &cfg).await?;
        assert_eq!(scan, [0, 1]);

        let (_, scan) = parse_pdf_with_scan_pages(&pdf_with_pages(&[Some("A"), Some("B")]), None, &cfg).await?;
        assert!(scan.is_empty());
        Ok(())
    }
}
