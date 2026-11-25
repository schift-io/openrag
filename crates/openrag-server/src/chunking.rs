//! Page chunking strategies: `fixed`, `structure`, and `semantic`.
//!
//! Chunking runs per page so every chunk keeps a single `page_id` for citations.

use openrag_core::{ChunkerKind, OpenRagError};
use openrag_memory::TextEmbedder;

const MAX_OVERLAP_CHARS: usize = 200;
const MAX_HEADING_TRAIL_CHARS: usize = 200;
const TINY_TAIL_CHARS: usize = 150;

/// Dispatches page text to the configured chunker.
pub async fn chunk_page(
    kind: ChunkerKind,
    text: &str,
    max_chars: usize,
    fallback_title: &str,
    embedder: &dyn TextEmbedder,
) -> Result<Vec<String>, OpenRagError> {
    match kind {
        ChunkerKind::Fixed => Ok(split_text_chunks(text, max_chars, fallback_title)),
        ChunkerKind::Structure => Ok(split_structured(text, max_chars, fallback_title)),
        ChunkerKind::Semantic => split_semantic(text, max_chars, fallback_title, embedder).await,
    }
}

/// Fixed-size character slicer with 10% (capped at 100 chars) overlap.
pub fn split_text_chunks(text: &str, max_chars: usize, fallback_title: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![fallback_title.to_string()];
    }
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() <= max_chars {
        return vec![trimmed.to_string()];
    }
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let overlap = (max_chars / 10).min(100);
    while start < chars.len() {
        let end = (start + max_chars).min(chars.len());
        let piece: String = chars[start..end].iter().collect();
        let piece_trimmed = piece.trim();
        if !piece_trimmed.is_empty() {
            chunks.push(piece_trimmed.to_string());
        }
        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    if chunks.is_empty() {
        chunks.push(fallback_title.to_string());
    }
    chunks
}

/// Structure-aware chunker for PDF text, Korean government documents, and Markdown.
pub fn split_structured(text: &str, max_chars: usize, fallback_title: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![fallback_title.to_string()];
    }
    if max_chars == 0 {
        return vec![trimmed.to_string()];
    }

    let blocks = parse_blocks(text);
    if blocks.is_empty() {
        return vec![fallback_title.to_string()];
    }

    let items: Vec<PackItem> = blocks
        .into_iter()
        .map(|b| PackItem {
            kind: b.kind,
            text: b.text,
            sep_before: b.sep_before.to_string(),
            force_break_before: false,
        })
        .collect();

    pack_items(&items, max_chars, fallback_title)
}

/// Embedding-guided chunker: parses structure first, splits text blocks into sentences,
/// and breaks where consecutive sentence cosine distance exceeds the 90th percentile.
pub async fn split_semantic(
    text: &str,
    max_chars: usize,
    fallback_title: &str,
    embedder: &dyn TextEmbedder,
) -> Result<Vec<String>, OpenRagError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(vec![fallback_title.to_string()]);
    }
    if max_chars == 0 {
        return Ok(vec![trimmed.to_string()]);
    }

    let blocks = parse_blocks(text);
    if blocks.is_empty() {
        return Ok(vec![fallback_title.to_string()]);
    }

    // Collect sentence spans per text block to count sentences and embed them in order.
    let block_spans: Vec<Vec<(usize, usize)>> = blocks
        .iter()
        .map(|b| {
            if b.kind == BlockKind::Text {
                find_sentence_spans(&b.text)
            } else {
                Vec::new()
            }
        })
        .collect();

    let total_sentences: usize = block_spans.iter().map(Vec::len).sum();
    if total_sentences < 4 {
        return Ok(split_structured(text, max_chars, fallback_title));
    }

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(total_sentences);
    for (block, spans) in blocks.iter().zip(&block_spans) {
        for &(start, end) in spans {
            let sent = &block.text[start..end];
            vectors.push(embedder.embed_text(sent).await?);
        }
    }

    let mut distances: Vec<f32> = Vec::with_capacity(vectors.len().saturating_sub(1));
    for pair in vectors.windows(2) {
        distances.push(cosine_distance(&pair[0], &pair[1]));
    }
    let threshold = percentile_f32(&distances, 0.90);

    // `break_before[k]` means a semantic topic shift occurred between sentence k-1 and sentence k.
    let mut break_before = vec![false; total_sentences];
    for (idx, &dist) in distances.iter().enumerate() {
        if dist > threshold {
            break_before[idx + 1] = true;
        }
    }

    // Map each block to its first sentence's global index so headings can look ahead.
    let mut block_first_sent: Vec<Option<usize>> = Vec::with_capacity(blocks.len());
    let mut cursor = 0usize;
    for spans in &block_spans {
        if spans.is_empty() {
            block_first_sent.push(None);
        } else {
            block_first_sent.push(Some(cursor));
            cursor += spans.len();
        }
    }

    let mut items: Vec<PackItem> = Vec::new();
    for (b_idx, (block, spans)) in blocks.iter().zip(&block_spans).enumerate() {
        match block.kind {
            BlockKind::Heading { .. } => {
                // Keep a heading with the new topic when the next text block starts a semantic break.
                let next_sent_breaks = block_first_sent[b_idx + 1..]
                    .iter()
                    .find_map(|opt| *opt)
                    .is_some_and(|s_idx| break_before[s_idx]);
                items.push(PackItem {
                    kind: block.kind,
                    text: block.text.clone(),
                    sep_before: block.sep_before.to_string(),
                    force_break_before: next_sent_breaks,
                });
            }
            BlockKind::PipeTable | BlockKind::HtmlTable => {
                items.push(PackItem {
                    kind: block.kind,
                    text: block.text.clone(),
                    sep_before: block.sep_before.to_string(),
                    force_break_before: false,
                });
            }
            BlockKind::Text => {
                let Some(first_global) = block_first_sent[b_idx] else {
                    continue;
                };
                // Split the text block into contiguous runs separated by internal semantic breaks.
                let mut run_start_idx = 0usize;
                for local_idx in 0..spans.len() {
                    let is_internal_break =
                        local_idx > run_start_idx && break_before[first_global + local_idx];
                    if is_internal_break {
                        let slice_start = spans[run_start_idx].0;
                        let slice_end = spans[local_idx - 1].1;
                        let sep = if run_start_idx == 0 {
                            block.sep_before.to_string()
                        } else {
                            block.text[spans[run_start_idx - 1].1..slice_start].to_string()
                        };
                        items.push(PackItem {
                            kind: BlockKind::Text,
                            text: block.text[slice_start..slice_end].to_string(),
                            sep_before: sep,
                            force_break_before: break_before[first_global + run_start_idx],
                        });
                        run_start_idx = local_idx;
                    }
                }
                let slice_start = spans[run_start_idx].0;
                let slice_end = spans[spans.len() - 1].1;
                let sep = if run_start_idx == 0 {
                    block.sep_before.to_string()
                } else {
                    block.text[spans[run_start_idx - 1].1..slice_start].to_string()
                };
                items.push(PackItem {
                    kind: BlockKind::Text,
                    text: block.text[slice_start..slice_end].to_string(),
                    sep_before: sep,
                    force_break_before: break_before[first_global + run_start_idx],
                });
            }
        }
    }

    Ok(pack_items(&items, max_chars, fallback_title))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Heading { level: u8 },
    PipeTable,
    HtmlTable,
    Text,
}

#[derive(Debug, Clone)]
struct Block {
    kind: BlockKind,
    text: String,
    sep_before: &'static str,
}

#[derive(Debug, Clone)]
struct PackItem {
    kind: BlockKind,
    text: String,
    sep_before: String,
    force_break_before: bool,
}

#[derive(Debug, Clone)]
struct DraftChunk {
    heading_trail: Option<String>,
    body: String,
    is_heading_only: bool,
    /// Active heading trail at the end of this chunk, used to drop redundant trails on tail merge.
    end_heading_trail: Option<String>,
    /// True when this chunk continues the same split block as the previous chunk.
    same_block_as_prev: bool,
    /// Byte length of the overlap or repeated table header at the start of `body`.
    overlap_bytes: usize,
    /// Separator to insert if merged into the previous chunk when `overlap_bytes == 0`.
    sep_from_prev: String,
}

impl DraftChunk {
    fn rendered_char_count(&self) -> usize {
        match &self.heading_trail {
            Some(trail) => char_count(trail) + 1 + char_count(&self.body),
            None => char_count(&self.body),
        }
    }

    fn render(&self) -> String {
        let raw = match &self.heading_trail {
            Some(trail) => format!("{trail}\n{}", self.body),
            None => self.body.clone(),
        };
        raw.trim().to_string()
    }
}

struct TextPiece {
    text: String,
    overlap_bytes: usize,
    sep_from_prev: String,
}

fn char_count(s: &str) -> usize {
    s.chars().count()
}

/// Finds the byte length of the prefix of `next` that overlaps with the suffix of `prev`
/// (bounded by `max_chars` characters), used to deduplicate char-sliced overlaps on tail merge.
fn suffix_prefix_overlap_bytes(prev: &str, next: &str, max_chars: usize) -> usize {
    let prev_chars: Vec<char> = prev.chars().collect();
    let next_chars: Vec<(usize, char)> = next.char_indices().collect();
    let limit = prev_chars.len().min(next_chars.len()).min(max_chars);
    for len in (1..=limit).rev() {
        let matched = prev_chars[prev_chars.len() - len..]
            .iter()
            .zip(next_chars[..len].iter())
            .all(|(&a, &(_, b))| a == b);
        if matched {
            return next_chars.get(len).map_or(next.len(), |&(byte_idx, _)| byte_idx);
        }
    }
    0
}

fn is_hangul_syllable(ch: char) -> bool {
    ('\u{AC00}'..='\u{D7A3}').contains(&ch)
}

fn is_korean_list_char(ch: char) -> bool {
    matches!(
        ch,
        '가' | '나' | '다' | '라' | '마' | '바' | '사' | '아' | '자' | '차' | '카' | '타' | '파' | '하'
    )
}

fn parse_md_heading_level(line: &str) -> Option<u8> {
    let trimmed = line.trim_start();
    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &trimmed[hashes..];
    if !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None;
    }
    if rest.trim().is_empty() {
        return None;
    }
    u8::try_from(hashes).ok()
}

fn starts_html_table(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.len() < 6 {
        return false;
    }
    let Some(prefix) = trimmed.get(..6) else {
        return false;
    };
    if !prefix.eq_ignore_ascii_case("<table") {
        return false;
    }
    match trimmed.as_bytes().get(6) {
        None => true,
        Some(&b) => b == b'>' || b == b'/' || b.is_ascii_whitespace(),
    }
}

fn contains_ascii_ci(haystack: &str, needle: &str) -> bool {
    haystack.to_ascii_lowercase().contains(needle)
}

fn is_pipe_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') {
        return false;
    }
    let mut has_dash = false;
    for ch in trimmed.chars() {
        if ch == '-' {
            has_dash = true;
        } else if !matches!(ch, '|' | ':' | ' ' | '\t') {
            return false;
        }
    }
    has_dash
}

/// Detects Korean/Markdown bullet markers, numbered items, and legal article headings (`제N조/장/절`).
fn starts_with_item_marker(line: &str) -> bool {
    let s = line.trim_start();
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let rest = chars.as_str();

    // Standalone symbol bullets and circled numbers (①..⑳) never start normal prose words.
    if matches!(
        first,
        '□' | '○' | '◦' | '※' | '•' | '●' | '■' | '▶' | '◆' | '·' | '①'..='⑳'
    ) {
        return true;
    }

    // Require whitespace after ㅇ, -, and * so we don't split negative numbers or *emphasis*.
    if matches!(first, 'ㅇ' | '-' | '*') {
        return rest.chars().next().is_some_and(char::is_whitespace);
    }

    // Legal articles: 제3조, 제3조(목적), 제3조의2, 제1장 총칙, 제2절.
    if first == '제' {
        let after_je = rest.trim_start();
        let digit_bytes = after_je.bytes().take_while(u8::is_ascii_digit).count();
        if digit_bytes > 0 {
            let after_digits = after_je[digit_bytes..].trim_start();
            let mut unit_chars = after_digits.chars();
            if matches!(unit_chars.next(), Some('조' | '장' | '절')) {
                let tail = unit_chars.as_str();
                let valid_after = match tail.chars().next() {
                    None => true,
                    Some(c) if c.is_whitespace() || matches!(c, '(' | ':' | '.' | '[' | '<' | '①'..='⑳') => {
                        true
                    }
                    Some('의') => tail['의'.len_utf8()..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_digit()),
                    _ => false,
                };
                if valid_after {
                    return true;
                }
            }
        }
    }

    // Numbered items: `1. ...`, `1) ...` (require body after marker so bare footnotes like `2)` stay in paragraph).
    if first.is_ascii_digit() {
        let digit_bytes = s.bytes().take_while(u8::is_ascii_digit).count();
        if (1..=4).contains(&digit_bytes) {
            let after_digits = &s[digit_bytes..];
            if let Some(after_dot) = after_digits.strip_prefix('.') {
                return after_dot.chars().next().is_some_and(char::is_whitespace);
            }
            if let Some(after_paren) = after_digits.strip_prefix(')') {
                return after_paren
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_whitespace() || c.is_alphabetic());
            }
        }
        return false;
    }

    // Parenthesized numbering: `(1) ...`, `(가) ...`.
    if first == '(' {
        let digit_bytes = rest.bytes().take_while(u8::is_ascii_digit).count();
        if (1..=4).contains(&digit_bytes) {
            if let Some(after_paren) = rest[digit_bytes..].strip_prefix(')') {
                return after_paren
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_whitespace() || c.is_alphabetic());
            }
        }
        let mut inner = rest.chars();
        if inner.next().is_some_and(is_korean_list_char) {
            if let Some(after_paren) = inner.as_str().strip_prefix(')') {
                return after_paren
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_whitespace() || c.is_alphabetic());
            }
        }
        return false;
    }

    // Korean alphabetic items: `가. ...`, `가) ...`.
    if is_korean_list_char(first) {
        if let Some(after_dot) = rest.strip_prefix('.') {
            return after_dot.chars().next().is_some_and(char::is_whitespace);
        }
        if let Some(after_paren) = rest.strip_prefix(')') {
            return after_paren
                .chars()
                .next()
                .is_some_and(|c| c.is_whitespace() || c.is_alphabetic());
        }
        return false;
    }

    // Unicode Roman numerals: `Ⅰ. ...`, `Ⅰ) ...`.
    if matches!(first, 'Ⅰ'..='Ⅻ' | 'ⅰ'..='ⅻ') {
        if let Some(after_dot) = rest.strip_prefix('.') {
            return after_dot.chars().next().is_some_and(char::is_whitespace);
        }
        if let Some(after_paren) = rest.strip_prefix(')') {
            return after_paren
                .chars()
                .next()
                .is_some_and(|c| c.is_whitespace() || c.is_alphabetic());
        }
    }

    false
}

fn parse_blocks(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.lines().collect();
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0usize;
    let mut saw_blank = false;

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            saw_blank = true;
            i += 1;
            continue;
        }
        let sep_before = if blocks.is_empty() {
            ""
        } else if saw_blank {
            "\n\n"
        } else {
            "\n"
        };
        saw_blank = false;

        if let Some(level) = parse_md_heading_level(line) {
            blocks.push(Block {
                kind: BlockKind::Heading { level },
                text: line.to_string(),
                sep_before,
            });
            i += 1;
            continue;
        }

        if line.trim_start().starts_with('|') {
            let start = i;
            i += 1;
            while i < lines.len() && lines[i].trim_start().starts_with('|') {
                i += 1;
            }
            blocks.push(Block {
                kind: BlockKind::PipeTable,
                text: lines[start..i].join("\n"),
                sep_before,
            });
            continue;
        }

        if starts_html_table(line) {
            let start = i;
            let mut end = i + 1;
            if !contains_ascii_ci(line, "</table>") {
                while end < lines.len() {
                    let closed = contains_ascii_ci(lines[end], "</table>");
                    end += 1;
                    if closed {
                        break;
                    }
                }
            }
            blocks.push(Block {
                kind: BlockKind::HtmlTable,
                text: lines[start..end].join("\n"),
                sep_before,
            });
            i = end;
            continue;
        }

        let start = i;
        i += 1;
        while i < lines.len() {
            let next = lines[i];
            if next.trim().is_empty()
                || parse_md_heading_level(next).is_some()
                || next.trim_start().starts_with('|')
                || starts_html_table(next)
                || starts_with_item_marker(next)
            {
                break;
            }
            i += 1;
        }
        blocks.push(Block {
            kind: BlockKind::Text,
            text: lines[start..i].join("\n"),
            sep_before,
        });
    }

    blocks
}

/// Returns true if `piece` is solely a numbering prefix like `1.`, `가.`, or `Ⅰ.`,
/// so we don't split a sentence right after the list number.
fn is_list_prefix_only(piece: &str) -> bool {
    let Some(stem) = piece.trim().strip_suffix('.') else {
        return false;
    };
    if stem.is_empty() {
        return false;
    }
    if stem.len() <= 4 && stem.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    let mut chars = stem.chars();
    let Some(c) = chars.next() else {
        return false;
    };
    if chars.next().is_some() {
        return false;
    }
    is_korean_list_char(c) || matches!(c, 'Ⅰ'..='Ⅻ' | 'ⅰ'..='ⅻ')
}

/// Finds non-empty sentence byte spans `(start, end)` inside a text block.
///
/// Matches Korean sentence endings (`다.`, `요.`, `음.`, `함.`, `임.`, `니다.`) and
/// `.` `?` `!` `。` followed by whitespace or line end.
fn find_sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut sent_start: Option<usize> = None;

    for idx in 0..chars.len() {
        let (byte_idx, ch) = chars[idx];
        if sent_start.is_none() {
            if ch.is_whitespace() {
                continue;
            }
            sent_start = Some(byte_idx);
        }

        let Some(start) = sent_start else {
            continue;
        };

        let next_ch = chars.get(idx + 1).map(|&(_, c)| c);
        let followed_by_ws_or_end = next_ch.is_none_or(char::is_whitespace);
        let prev_ch = idx.checked_sub(1).map(|p| chars[p].1);

        let is_ko_end = ch == '.'
            && prev_ch.is_some_and(|p| matches!(p, '다' | '요' | '음' | '함' | '임'))
            && (followed_by_ws_or_end || next_ch.is_some_and(is_hangul_syllable));
        let is_punct_end =
            matches!(ch, '.' | '?' | '!' | '。') && (followed_by_ws_or_end || ch == '。');

        if is_ko_end || is_punct_end {
            let end_byte = byte_idx + ch.len_utf8();
            if ch == '.' && next_ch.is_some() && is_list_prefix_only(&text[start..end_byte]) {
                continue;
            }
            spans.push((start, end_byte));
            sent_start = None;
        }
    }

    if let Some(start) = sent_start {
        let tail = text[start..].trim_end();
        if !tail.is_empty() {
            spans.push((start, start + tail.len()));
        }
    }

    spans
}

fn find_line_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut offset = 0usize;
    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let lead = line.len() - line.trim_start().len();
            let start = offset + lead;
            let end = start + trimmed.len();
            spans.push((start, end));
        }
        offset += raw_line.len() + 1;
    }
    spans
}

fn split_oversized_by_lines(text: &str, first_cap: usize, next_cap: usize) -> Vec<TextPiece> {
    let line_spans = find_line_spans(text);
    if line_spans.is_empty() {
        return Vec::new();
    }

    let mut pieces: Vec<TextPiece> = Vec::new();
    let mut prev_overlap: Option<(usize, usize)> = None;
    let mut prev_end: Option<usize> = None;
    let mut idx = 0usize;

    while idx < line_spans.len() {
        let cap = if pieces.is_empty() { first_cap } else { next_cap };
        let (l_start, l_end) = line_spans[idx];

        let mut piece_start = l_start;
        let mut overlap_bytes = 0usize;
        let mut sep_from_prev = prev_end
            .map(|pe| text[pe..l_start].to_string())
            .unwrap_or_default();

        if let Some((ov_start, ov_end)) = prev_overlap {
            if ov_end <= l_start && char_count(&text[ov_start..l_end]) <= cap {
                piece_start = ov_start;
                overlap_bytes = ov_end - ov_start;
                sep_from_prev.clear();
            }
        }

        if char_count(&text[piece_start..l_end]) > cap {
            // Single line exceeds budget: fall back to fixed character slicing with its overlap.
            let sliced = split_text_chunks(&text[l_start..l_end], cap, "");
            let max_char_ov = (cap / 10).min(100);
            let mut prev_slice: Option<String> = None;
            for (s_idx, s) in sliced.into_iter().enumerate() {
                let ov_bytes = prev_slice
                    .as_deref()
                    .map_or(0, |p| suffix_prefix_overlap_bytes(p, &s, max_char_ov));
                prev_slice = Some(s.clone());
                pieces.push(TextPiece {
                    text: s,
                    overlap_bytes: ov_bytes,
                    sep_from_prev: if s_idx == 0 {
                        sep_from_prev.clone()
                    } else {
                        String::new()
                    },
                });
            }
            prev_overlap = None;
            prev_end = Some(l_end);
            idx += 1;
            continue;
        }

        let mut last_idx = idx;
        while last_idx + 1 < line_spans.len() {
            let next_end = line_spans[last_idx + 1].1;
            if char_count(&text[piece_start..next_end]) <= cap {
                last_idx += 1;
            } else {
                break;
            }
        }

        let piece_end = line_spans[last_idx].1;
        pieces.push(TextPiece {
            text: text[piece_start..piece_end].to_string(),
            overlap_bytes,
            sep_from_prev,
        });

        let (last_l_start, last_l_end) = line_spans[last_idx];
        prev_overlap = if char_count(&text[last_l_start..last_l_end]) <= MAX_OVERLAP_CHARS {
            Some((last_l_start, last_l_end))
        } else {
            None
        };
        prev_end = Some(piece_end);
        idx = last_idx + 1;
    }

    pieces
}

fn split_oversized_text(text: &str, first_cap: usize, next_cap: usize) -> Vec<TextPiece> {
    let spans = find_sentence_spans(text);
    if spans.is_empty() {
        return Vec::new();
    }

    let mut pieces: Vec<TextPiece> = Vec::new();
    let mut prev_overlap: Option<(usize, usize)> = None;
    let mut prev_end: Option<usize> = None;
    let mut idx = 0usize;

    while idx < spans.len() {
        let cap = if pieces.is_empty() { first_cap } else { next_cap };
        let (s_start, s_end) = spans[idx];

        let mut piece_start = s_start;
        let mut overlap_bytes = 0usize;
        let mut sep_from_prev = prev_end
            .map(|pe| text[pe..s_start].to_string())
            .unwrap_or_default();

        if let Some((ov_start, ov_end)) = prev_overlap {
            if ov_end <= s_start && char_count(&text[ov_start..s_end]) <= cap {
                piece_start = ov_start;
                overlap_bytes = ov_end - ov_start;
                sep_from_prev.clear();
            }
        }

        if char_count(&text[piece_start..s_end]) > cap {
            // Single sentence exceeds budget: fall back to line breaks, then char slicing.
            let sub_pieces = split_oversized_by_lines(&text[s_start..s_end], cap, next_cap);
            for (p_idx, mut p) in sub_pieces.into_iter().enumerate() {
                if p_idx == 0 {
                    p.sep_from_prev = sep_from_prev.clone();
                }
                pieces.push(p);
            }
            prev_overlap = None;
            prev_end = Some(s_end);
            idx += 1;
            continue;
        }

        let mut last_idx = idx;
        while last_idx + 1 < spans.len() {
            let next_end = spans[last_idx + 1].1;
            if char_count(&text[piece_start..next_end]) <= cap {
                last_idx += 1;
            } else {
                break;
            }
        }

        let piece_end = spans[last_idx].1;
        pieces.push(TextPiece {
            text: text[piece_start..piece_end].to_string(),
            overlap_bytes,
            sep_from_prev,
        });

        let (last_s_start, last_s_end) = spans[last_idx];
        prev_overlap = if char_count(&text[last_s_start..last_s_end]) <= MAX_OVERLAP_CHARS {
            Some((last_s_start, last_s_end))
        } else {
            None
        };
        prev_end = Some(piece_end);
        idx = last_idx + 1;
    }

    pieces
}

fn split_pipe_table(table_text: &str, first_cap: usize, next_cap: usize) -> Vec<TextPiece> {
    let lines: Vec<&str> = table_text.lines().collect();
    if lines.len() < 2 || !is_pipe_table_separator(lines[1]) {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    }

    let header = format!("{}\n{}", lines[0], lines[1]);
    let header_chars = char_count(&header);
    let data_rows = &lines[2..];

    if data_rows.is_empty() || header_chars + 2 > next_cap || header_chars + 2 > first_cap {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    }

    let mut pieces: Vec<TextPiece> = Vec::new();
    let mut cur = header.clone();
    let mut cur_chars = header_chars;
    let mut cur_has_row = false;

    for &row in data_rows {
        let cap = if pieces.is_empty() { first_cap } else { next_cap };
        let row_chars = char_count(row);

        if cur_chars + 1 + row_chars <= cap {
            cur.push('\n');
            cur.push_str(row);
            cur_chars += 1 + row_chars;
            cur_has_row = true;
            continue;
        }

        if cur_has_row {
            let overlap_bytes = if pieces.is_empty() { 0 } else { header.len() };
            pieces.push(TextPiece {
                text: cur,
                overlap_bytes,
                sep_from_prev: "\n".to_string(),
            });
            cur = header.clone();
            cur_chars = header_chars;
            cur_has_row = false;
        }

        let next_piece_cap = if pieces.is_empty() { first_cap } else { next_cap };
        if header_chars + 1 + row_chars <= next_piece_cap {
            cur.push('\n');
            cur.push_str(row);
            cur_chars += 1 + row_chars;
            cur_has_row = true;
        } else {
            // Single table row exceeds remaining budget: slice the row and repeat header on each part.
            let row_budget = next_piece_cap.saturating_sub(header_chars + 1).max(1);
            for part in split_text_chunks(row, row_budget, "") {
                let overlap_bytes = if pieces.is_empty() { 0 } else { header.len() };
                pieces.push(TextPiece {
                    text: format!("{header}\n{part}"),
                    overlap_bytes,
                    sep_from_prev: "\n".to_string(),
                });
            }
        }
    }

    if cur_has_row {
        let overlap_bytes = if pieces.is_empty() { 0 } else { header.len() };
        pieces.push(TextPiece {
            text: cur,
            overlap_bytes,
            sep_from_prev: "\n".to_string(),
        });
    }

    pieces
}

fn split_html_table(table_text: &str, first_cap: usize, next_cap: usize) -> Vec<TextPiece> {
    let lower = table_text.to_ascii_lowercase();
    let Some(open_end) = lower.find('>') else {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    };
    let Some(close_start) = lower.rfind("</table>") else {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    };
    if close_start <= open_end + 1 {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    }

    let open_tag = &table_text[..=open_end];
    let close_tag = &table_text[close_start..];
    let inner = &table_text[open_end + 1..close_start];
    let inner_lower = &lower[open_end + 1..close_start];

    // Collect `<tr>...</tr>` row slices.
    let mut rows: Vec<&str> = Vec::new();
    let mut pos = 0usize;
    while let Some(rel_end) = inner_lower[pos..].find("</tr>") {
        let row_end = pos + rel_end + 5;
        let slice = inner[pos..row_end].trim();
        if !slice.is_empty() {
            rows.push(slice);
        }
        pos = row_end;
    }
    let remainder = inner[pos..].trim();
    if !remainder.is_empty() {
        rows.push(remainder);
    }

    if rows.len() < 2 {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    }

    let (header_row, body_rows): (Option<&str>, &[&str]) =
        if contains_ascii_ci(rows[0], "<th") || contains_ascii_ci(rows[0], "<thead") {
            (Some(rows[0]), &rows[1..])
        } else {
            (None, &rows[..])
        };

    let prefix = match header_row {
        Some(h) => format!("{open_tag}\n{h}"),
        None => open_tag.to_string(),
    };
    let wrapper_chars = char_count(&prefix) + 1 + char_count(close_tag);
    if wrapper_chars + 2 > first_cap || wrapper_chars + 2 > next_cap || body_rows.is_empty() {
        return split_oversized_by_lines(table_text, first_cap, next_cap);
    }

    let mut pieces: Vec<TextPiece> = Vec::new();
    let mut cur = prefix.clone();
    let mut cur_chars = char_count(&cur);
    let mut cur_has_row = false;
    let close_chars = 1 + char_count(close_tag);

    for &row in body_rows {
        let cap = if pieces.is_empty() { first_cap } else { next_cap };
        let row_chars = char_count(row);

        if cur_chars + 1 + row_chars + close_chars <= cap {
            cur.push('\n');
            cur.push_str(row);
            cur_chars += 1 + row_chars;
            cur_has_row = true;
            continue;
        }

        if cur_has_row {
            cur.push('\n');
            cur.push_str(close_tag);
            pieces.push(TextPiece {
                text: cur,
                overlap_bytes: 0,
                sep_from_prev: "\n".to_string(),
            });
            cur = prefix.clone();
            cur_chars = char_count(&cur);
        }

        let next_piece_cap = if pieces.is_empty() { first_cap } else { next_cap };
        if cur_chars + 1 + row_chars + close_chars <= next_piece_cap {
            cur.push('\n');
            cur.push_str(row);
            cur_chars += 1 + row_chars;
            cur_has_row = true;
        } else {
            return split_oversized_by_lines(table_text, first_cap, next_cap);
        }
    }

    if cur_has_row {
        cur.push('\n');
        cur.push_str(close_tag);
        pieces.push(TextPiece {
            text: cur,
            overlap_bytes: 0,
            sep_from_prev: "\n".to_string(),
        });
    }

    pieces
}

fn update_heading_stack(stack: &mut Vec<(u8, String)>, level: u8, line: &str) {
    while let Some(&(top_level, _)) = stack.last() {
        if top_level >= level {
            stack.pop();
        } else {
            break;
        }
    }
    stack.push((level, line.trim().to_string()));
}

fn format_heading_trail(stack: &[(u8, String)], max_trail_chars: usize) -> Option<String> {
    if stack.is_empty() || max_trail_chars == 0 {
        return None;
    }
    // Prefer keeping the innermost headings when the full trail exceeds max_trail_chars.
    let mut start = stack.len() - 1;
    let mut total = char_count(&stack[start].1);
    while start > 0 {
        let candidate = char_count(&stack[start - 1].1) + 3 + total;
        if candidate <= max_trail_chars {
            start -= 1;
            total = candidate;
        } else {
            break;
        }
    }
    let joined = stack[start..]
        .iter()
        .map(|(_, h)| h.as_str())
        .collect::<Vec<_>>()
        .join(" > ");
    if char_count(&joined) <= max_trail_chars {
        Some(joined)
    } else {
        Some(joined.chars().take(max_trail_chars).collect())
    }
}

fn pack_items_draft(items: &[PackItem], max_chars: usize) -> Vec<DraftChunk> {
    let max_trail_chars = MAX_HEADING_TRAIL_CHARS.min(max_chars / 2);
    let mut heading_stack: Vec<(u8, String)> = Vec::new();
    let mut chunks: Vec<DraftChunk> = Vec::new();
    let mut current: Option<DraftChunk> = None;

    for item in items {
        let sep = if item.sep_before.is_empty() {
            "\n"
        } else {
            item.sep_before.as_str()
        };

        if item.force_break_before {
            if let Some(cur) = current.as_ref() {
                if !cur.is_heading_only {
                    if let Some(flushed) = current.take() {
                        chunks.push(flushed);
                    }
                }
            }
        }

        if let BlockKind::Heading { level } = item.kind {
            update_heading_stack(&mut heading_stack, level, &item.text);
            let active_trail = format_heading_trail(&heading_stack, max_trail_chars);
            let item_chars = char_count(&item.text);

            if let Some(cur) = current.as_mut() {
                let candidate_chars = cur.rendered_char_count() + char_count(sep) + item_chars;
                if candidate_chars <= max_chars {
                    cur.body.push_str(sep);
                    cur.body.push_str(&item.text);
                    cur.end_heading_trail = active_trail;
                    continue;
                }
            }
            if let Some(flushed) = current.take() {
                chunks.push(flushed);
            }

            if item_chars <= max_chars {
                current = Some(DraftChunk {
                    heading_trail: None,
                    body: item.text.clone(),
                    is_heading_only: true,
                    end_heading_trail: active_trail,
                    same_block_as_prev: false,
                    overlap_bytes: 0,
                    sep_from_prev: sep.to_string(),
                });
            } else {
                for (p_idx, part) in split_text_chunks(&item.text, max_chars, "").into_iter().enumerate() {
                    chunks.push(DraftChunk {
                        heading_trail: None,
                        body: part,
                        is_heading_only: true,
                        end_heading_trail: active_trail.clone(),
                        same_block_as_prev: p_idx > 0,
                        overlap_bytes: 0,
                        sep_from_prev: if p_idx == 0 {
                            sep.to_string()
                        } else {
                            String::new()
                        },
                    });
                }
            }
            continue;
        }

        let active_trail = format_heading_trail(&heading_stack, max_trail_chars);
        let trail_overhead = active_trail.as_ref().map_or(0, |t| char_count(t) + 1);
        let fresh_budget = max_chars.saturating_sub(trail_overhead).max(1);
        let item_chars = char_count(&item.text);

        // Try packing the whole block into the current chunk.
        if let Some(cur) = current.as_mut() {
            let candidate_chars = cur.rendered_char_count() + char_count(sep) + item_chars;
            if candidate_chars <= max_chars {
                cur.body.push_str(sep);
                cur.body.push_str(&item.text);
                cur.is_heading_only = false;
                cur.end_heading_trail = active_trail;
                continue;
            }
        }

        // Block fits in a fresh chunk (with carried heading trail): never split it.
        if item_chars <= fresh_budget {
            if let Some(flushed) = current.take() {
                chunks.push(flushed);
            }
            current = Some(DraftChunk {
                heading_trail: active_trail.clone(),
                body: item.text.clone(),
                is_heading_only: false,
                end_heading_trail: active_trail,
                same_block_as_prev: false,
                overlap_bytes: 0,
                sep_from_prev: sep.to_string(),
            });
            continue;
        }

        // Oversized block: if `current` only holds headings and leaves enough room for a piece,
        // attach the first piece directly to those headings instead of emitting an orphan heading chunk.
        let heading_prefix_room = current
            .as_ref()
            .filter(|c| c.is_heading_only)
            .and_then(|c| max_chars.checked_sub(c.rendered_char_count() + char_count(sep)))
            .filter(|&rem| rem >= fresh_budget / 2 && rem >= 40);

        let attach_first_to_heading = heading_prefix_room.is_some();
        let first_cap = if let Some(rem) = heading_prefix_room {
            rem
        } else {
            if let Some(flushed) = current.take() {
                chunks.push(flushed);
            }
            fresh_budget
        };

        let pieces = match item.kind {
            BlockKind::PipeTable => split_pipe_table(&item.text, first_cap, fresh_budget),
            BlockKind::HtmlTable => split_html_table(&item.text, first_cap, fresh_budget),
            BlockKind::Text => split_oversized_text(&item.text, first_cap, fresh_budget),
            BlockKind::Heading { .. } => Vec::new(),
        };

        let mut iter = pieces.into_iter().enumerate();
        if attach_first_to_heading {
            if let (Some(mut cur), Some((_, first_piece))) = (current.take(), iter.next()) {
                if cur.rendered_char_count() + char_count(sep) + char_count(&first_piece.text)
                    <= max_chars
                {
                    cur.body.push_str(sep);
                    cur.body.push_str(&first_piece.text);
                    cur.is_heading_only = false;
                    cur.end_heading_trail = active_trail.clone();
                    current = Some(cur);
                } else {
                    chunks.push(cur);
                    current = Some(DraftChunk {
                        heading_trail: active_trail.clone(),
                        body: first_piece.text,
                        is_heading_only: false,
                        end_heading_trail: active_trail.clone(),
                        same_block_as_prev: false,
                        overlap_bytes: first_piece.overlap_bytes,
                        sep_from_prev: sep.to_string(),
                    });
                }
            }
        }

        for (p_idx, piece) in iter {
            if let Some(flushed) = current.take() {
                chunks.push(flushed);
            }
            current = Some(DraftChunk {
                heading_trail: active_trail.clone(),
                body: piece.text,
                is_heading_only: false,
                end_heading_trail: active_trail.clone(),
                same_block_as_prev: p_idx > 0,
                overlap_bytes: piece.overlap_bytes,
                sep_from_prev: if p_idx == 0 {
                    sep.to_string()
                } else {
                    piece.sep_from_prev
                },
            });
        }
    }

    if let Some(flushed) = current {
        chunks.push(flushed);
    }

    merge_tiny_tail(&mut chunks, max_chars);
    chunks
}

fn pack_items(items: &[PackItem], max_chars: usize, fallback_title: &str) -> Vec<String> {
    let chunks = pack_items_draft(items, max_chars);
    let out: Vec<String> = chunks
        .into_iter()
        .map(|c| c.render())
        .filter(|s| !s.is_empty())
        .collect();

    if out.is_empty() {
        vec![fallback_title.to_string()]
    } else {
        out
    }
}

fn merge_tiny_tail(chunks: &mut Vec<DraftChunk>, max_chars: usize) {
    if chunks.len() < 2 || max_chars <= TINY_TAIL_CHARS {
        return;
    }
    let [.., prev, last] = chunks.as_mut_slice() else {
        return;
    };
    if last.rendered_char_count() >= TINY_TAIL_CHARS {
        return;
    }

    let addition = if last.same_block_as_prev {
        if last.overlap_bytes > 0 && last.overlap_bytes <= last.body.len() {
            last.body[last.overlap_bytes..].to_string()
        } else {
            format!("{}{}", last.sep_from_prev, last.body)
        }
    } else {
        let sep = if last.sep_from_prev.is_empty() {
            "\n\n"
        } else {
            last.sep_from_prev.as_str()
        };
        if last.heading_trail.is_some() && last.heading_trail == prev.end_heading_trail {
            format!("{sep}{}", last.body)
        } else if let Some(trail) = &last.heading_trail {
            format!("{sep}{trail}\n{}", last.body)
        } else {
            format!("{sep}{}", last.body)
        }
    };

    if prev.rendered_char_count() + char_count(&addition) <= max_chars + TINY_TAIL_CHARS {
        prev.body.push_str(&addition);
        prev.is_heading_only = false;
        prev.end_heading_trail = last.end_heading_trail.clone();
        chunks.pop();
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    let sim = (dot / denom).clamp(-1.0, 1.0);
    (1.0 - sim) as f32
}

/// Linear-interpolated percentile (matches numpy.percentile with default linear method).
fn percentile_f32(values: &[f32], q: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = (pos - lo as f64) as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use openrag_core::PipelineConfig;
    use openrag_memory::DeterministicTextEmbedder;

    use super::*;

    const PDF_SHAPE_1: &str = "\
가정의학
발생 순위)이 21.8명(4위), 33.4명(4위)인 것과 비교하여 볼 때
그 발생이 급격히 증가하고 있다. 또한 2010년 대장암의 암사
망분율은 남자에서 9.6% (4위), 여자에서 12.4% (3위)를 차지
하고 있다.
2)
국제암연구기관(International Agency for Research on
Cancer)에서 발표한 2008년 선진국의 연령표준화 암발생률은";

    const PDF_SHAPE_2: &str = "\
□ 전문교육기관ㆍ항공훈련기관 지정 개요
ㅇ 전문교육기관 조종사ㆍ항공정비사ㆍ항공교통관제사를 양성하려는
대학 등 교육기관이 전문교육기관으로 지정 을 받으려는 경우

* 조종사 양성 교육기관은 의무적으로 전문교육기관 지정을 받아야 되나,
항공정비사ㆍ항공교통관제사 양성 교육기관은 선택사항임";

    fn repeat_to_chars(sample: &str, sep: &str, target_chars: usize) -> String {
        let mut out = String::new();
        while char_count(&out) < target_chars {
            if !out.is_empty() {
                out.push_str(sep);
            }
            out.push_str(sample);
        }
        out
    }

    fn strip_ws(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    /// Verifies non-whitespace coverage in exact input order by stripping carried headings and
    /// recorded overlaps/repeated headers from the structured chunk drafts, and also verifies
    /// that rendered chunks match the public `split_structured` output.
    fn assert_structured_coverage(input: &str, max_chars: usize) {
        let blocks = parse_blocks(input);
        let items: Vec<PackItem> = blocks
            .into_iter()
            .map(|b| PackItem {
                kind: b.kind,
                text: b.text,
                sep_before: b.sep_before.to_string(),
                force_break_before: false,
            })
            .collect();
        let drafts = pack_items_draft(&items, max_chars);
        let public_chunks = split_structured(input, max_chars, "fallback");
        let rendered: Vec<String> = drafts
            .iter()
            .map(DraftChunk::render)
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(rendered, public_chunks);

        let mut rebuilt = String::new();
        for draft in &drafts {
            // Strip carried heading (omitted because we read `draft.body`) and strip recorded overlap.
            let body = if draft.overlap_bytes > 0 && draft.overlap_bytes <= draft.body.len() {
                &draft.body[draft.overlap_bytes..]
            } else {
                draft.body.as_str()
            };
            rebuilt.push_str(&strip_ws(body));
        }
        assert_eq!(rebuilt, strip_ws(input), "non-whitespace coverage mismatch");
    }

    /// Public-output coverage check for non-periodic inputs where heading trails and overlaps
    /// can be stripped directly from the returned strings.
    fn assert_public_coverage_no_overlap(input: &str, chunks: &[String], repeated_prefix: Option<&str>) {
        let mut rebuilt = String::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            let mut body = chunk.as_str();
            if idx > 0 {
                if let Some((first_line, rest)) = body.split_once('\n') {
                    if first_line.contains(" > #")
                        || (parse_md_heading_level(first_line).is_some()
                            && chunks[..idx]
                                .iter()
                                .any(|p| p.lines().any(|l| l.trim() == first_line.trim())))
                    {
                        body = rest;
                    }
                }
                if let Some(prefix) = repeated_prefix {
                    if let Some(stripped) = body.strip_prefix(prefix) {
                        body = stripped;
                    }
                }
            }
            rebuilt.push_str(&strip_ws(body));
        }
        assert_eq!(rebuilt, strip_ws(input), "public non-whitespace coverage mismatch");
    }

    struct FakeEmbedder {
        by_substr: Vec<(String, Vec<f32>)>,
        default_vec: Vec<f32>,
        calls: AtomicUsize,
    }

    impl FakeEmbedder {
        fn new(rules: Vec<(&str, Vec<f32>)>, default_vec: Vec<f32>) -> Self {
            Self {
                by_substr: rules
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                default_vec,
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl TextEmbedder for FakeEmbedder {
        fn model_name(&self) -> &str {
            "fake-semantic-embedder"
        }

        async fn embed_text(&self, text: &str) -> Result<Vec<f32>, OpenRagError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            for (needle, vec) in &self.by_substr {
                if text.contains(needle.as_str()) {
                    return Ok(vec.clone());
                }
            }
            Ok(self.default_vec.clone())
        }
    }

    #[test]
    fn fixed_chunker_preserves_original_behavior() {
        assert_eq!(split_text_chunks("   \n\t ", 1000, "Fallback"), vec!["Fallback"]);
        assert_eq!(split_text_chunks("short text", 1000, "Fallback"), vec!["short text"]);

        let long = "가".repeat(2500);
        let chunks = split_text_chunks(&long, 1000, "Fallback");
        assert_eq!(chunks.len(), 3);
        assert_eq!(char_count(&chunks[0]), 1000);
        assert_eq!(char_count(&chunks[1]), 1000);
        assert_eq!(char_count(&chunks[2]), 700);
    }

    #[test]
    fn structure_empty_and_multibyte_korean() {
        assert_eq!(split_structured("", 1000, "빈 문서"), vec!["빈 문서"]);
        assert_eq!(split_structured(" \n\t  ", 1000, "빈 문서"), vec!["빈 문서"]);

        let korean = "한글멀티바이트문자열테스트입니다".repeat(200);
        let chunks = split_structured(&korean, 1000, "제목");
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(char_count(c) <= 1000, "chunk len {} > 1000", char_count(c));
        }
    }

    #[test]
    fn structure_respects_max_chars_on_pdf_shapes_and_markdown() {
        let pdf1 = repeat_to_chars(PDF_SHAPE_1, "\n", 3000);
        let pdf2 = repeat_to_chars(PDF_SHAPE_2, "\n\n", 3000);
        let md = format!(
            "# 여비 규정\n\n## 3. 출장비\n\n{}\n\n### 세부 기준\n\n{}\n",
            "출장비는 국내 여비와 국외 여비로 구분하여 지급한다. ".repeat(35),
            "숙박비는 실비 정산을 원칙으로 하며 영수증을 첨부하여야 한다. ".repeat(35),
        );

        for (label, input, allow_tail_merge) in [
            ("pdf1", pdf1.as_str(), false),
            ("pdf2", pdf2.as_str(), true),
            ("md", md.as_str(), false),
        ] {
            let chunks = split_structured(input, 1000, "제목");
            assert!(chunks.len() >= 2, "{label} should split into multiple chunks");
            for (i, c) in chunks.iter().enumerate() {
                let limit = if allow_tail_merge && i + 1 == chunks.len() {
                    1000 + TINY_TAIL_CHARS
                } else {
                    1000
                };
                assert!(
                    char_count(c) <= limit,
                    "{label} chunk {i} exceeded limit {limit}: {}",
                    char_count(c)
                );
            }
            assert_structured_coverage(input, 1000);
        }

        // Heading trail is carried on Markdown continuation chunks, not on plain PDF chunks.
        let md_chunks = split_structured(&md, 1000, "제목");
        assert!(md_chunks[0].starts_with("# 여비 규정"));
        assert!(
            md_chunks[1].starts_with("# 여비 규정 > ## 3. 출장비\n"),
            "expected heading trail on chunk 1, got: {}",
            md_chunks[1].lines().next().unwrap_or("")
        );
        let pdf1_chunks = split_structured(&pdf1, 1000, "제목");
        assert!(!pdf1_chunks.iter().any(|c| c.starts_with('#')));
    }

    #[test]
    fn structure_never_splits_fitting_bullet_or_numbered_items() {
        let items: Vec<String> = vec![
            format!("□ 개요 {}", "설명 문장입니다. ".repeat(18).trim()),
            format!("ㅇ 첫 번째 항목 {}", "세부 내용입니다. ".repeat(18).trim()),
            format!("* 별표 항목 {}", "주의 사항입니다. ".repeat(18).trim()),
            format!("1. 번호 항목 {}", "절차 안내입니다. ".repeat(18).trim()),
            format!("(2) 괄호 번호 항목 {}", "추가 조건입니다. ".repeat(18).trim()),
            format!("가. 한글 번호 항목 {}", "예외 규정입니다. ".repeat(18).trim()),
            format!("Ⅰ. 로마 숫자 항목 {}", "부칙 내용입니다. ".repeat(18).trim()),
            format!("제3조(목적) {}", "이 규정은 기준을 정한다. ".repeat(18).trim()),
            format!("• 불릿 항목 {}", "참고 사항입니다. ".repeat(18).trim()),
        ];
        let page = items.join("\n");
        let chunks = split_structured(&page, 1000, "규정");
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(char_count(c) <= 1000, "chunk len {} > 1000", char_count(c));
        }
        // Every individual item must appear intact inside some single chunk.
        for item in &items {
            assert!(
                chunks.iter().any(|c| c.contains(item.as_str())),
                "item was split across chunks: {item}"
            );
        }
        assert_public_coverage_no_overlap(&page, &chunks, None);
    }

    #[test]
    fn structure_splits_pipe_table_repeating_header_and_separator() {
        let header = "| 부서 | 항목 | 금액 | 비고 |";
        let sep = "| --- | --- | ---: | --- |";
        let mut lines = vec![header.to_string(), sep.to_string()];
        for i in 1..=35 {
            lines.push(format!(
                "| 항공안전본부 제{i}과 | 전문교육기관 지정 심사 수수료 항목 {i} | {}0,000원 | 정기 점검 대상 |",
                i + 10
            ));
        }
        let table = lines.join("\n");
        assert!(char_count(&table) > 1500);

        let chunks = split_structured(&table, 600, "표");
        assert!(chunks.len() >= 3);
        let expected_prefix = format!("{header}\n{sep}\n");
        for (idx, chunk) in chunks.iter().enumerate() {
            assert!(
                char_count(chunk) <= 600,
                "table chunk {idx} exceeded max_chars: {}",
                char_count(chunk)
            );
            assert!(
                chunk.starts_with(&expected_prefix),
                "table chunk {idx} missing header+separator:\n{chunk}"
            );
        }
        assert_public_coverage_no_overlap(&table, &chunks, Some(&expected_prefix));
        assert_structured_coverage(&table, 600);
    }

    #[test]
    fn structure_splits_3000_char_paragraph_without_blank_lines_at_sentence_ends() {
        let sentence_a = "발생 순위가 21.8명(4위)인 것과 비교하여 볼 때 그 발생이 급격히 증가하고 있다.";
        let sentence_b = "또한 2010년 대장암의 암사망분율은 남자에서 9.6%, 여자에서 12.4%를 차지하고 있다.";
        let sentence_c = "국제암연구기관에서 발표한 선진국의 연령표준화 암발생률은 지속적으로 상승함.";
        let mut para = String::new();
        let mut i = 0usize;
        while char_count(&para) < 3000 {
            if !para.is_empty() {
                para.push('\n');
            }
            let s = match i % 3 {
                0 => sentence_a,
                1 => sentence_b,
                _ => sentence_c,
            };
            para.push_str(s);
            i += 1;
        }
        assert!(!para.contains("\n\n"));

        let chunks = split_structured(&para, 1000, "암통계");
        assert!(chunks.len() >= 3);
        for (idx, c) in chunks.iter().enumerate() {
            if idx + 1 < chunks.len() {
                assert!(char_count(c) <= 1000, "chunk {idx} len {} > 1000", char_count(c));
                assert!(
                    c.ends_with("다.")
                        || c.ends_with("요.")
                        || c.ends_with("음.")
                        || c.ends_with("함.")
                        || c.ends_with("임.")
                        || c.ends_with('.')
                        || c.ends_with('?')
                        || c.ends_with('!')
                        || c.ends_with('。'),
                    "chunk {idx} did not end at sentence end: ...{}",
                    c.chars().rev().take(20).collect::<String>().chars().rev().collect::<String>()
                );
            } else {
                assert!(char_count(c) <= 1150);
            }
        }
        // Consecutive chunks inside the same split paragraph share the last sentence as overlap (<= 200 chars).
        assert!(
            chunks[1].starts_with(sentence_a)
                || chunks[1].starts_with(sentence_b)
                || chunks[1].starts_with(sentence_c),
            "expected previous piece's last sentence as overlap at start of chunk 1"
        );
        assert_structured_coverage(&para, 1000);
    }

    #[test]
    fn structure_falls_back_to_char_slicing_on_unbroken_2500_char_run() {
        let run = "가나다라마바사아자차".repeat(250);
        assert_eq!(char_count(&run), 2500);

        let chunks = split_structured(&run, 1000, "제목");
        assert_eq!(chunks.len(), 3);
        for c in &chunks {
            assert!(char_count(c) <= 1000, "chunk len {} > 1000", char_count(c));
        }
        assert_structured_coverage(&run, 1000);
    }

    #[test]
    fn structure_tiny_tail_merge_and_heading_trail_cap() {
        // Tail under 150 chars merges when prev + tail <= max_chars + 150.
        let p1 = format!("첫째 문단입니다. {}", "가".repeat(940));
        let p2 = "짧은 꼬리 문단입니다.";
        let page = format!("{p1}\n\n{p2}");
        let chunks = split_structured(&page, 1000, "제목");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains(p2));
        assert!(char_count(&chunks[0]) <= 1150);

        // Long nested headings clamp trail to <= 200 chars and still keep chunks within max_chars.
        let h1 = format!("# {}", "가".repeat(120));
        let h2 = format!("## {}", "나".repeat(120));
        let body = "여비 지급 기준에 관한 상세한 설명 문장입니다. ".repeat(45);
        let md = format!("{h1}\n\n{h2}\n\n{body}");
        let md_chunks = split_structured(&md, 1000, "제목");
        assert!(md_chunks.len() >= 2);
        for (idx, c) in md_chunks.iter().enumerate() {
            assert!(char_count(c) <= 1000, "chunk {idx} len {} > 1000", char_count(c));
        }
        let trail_line = md_chunks[1].lines().next().unwrap_or("");
        assert!(char_count(trail_line) <= 200, "heading trail exceeded 200 chars: {}", char_count(trail_line));
    }

    #[tokio::test]
    async fn semantic_breaks_at_sharp_vector_shifts_and_respects_max_cap()
    -> Result<(), Box<dyn std::error::Error>> {
        // Two clearly separated topics (> 150 chars each so tiny-tail merge doesn't collapse them),
        // with orthogonal vectors between topic A and topic B.
        let sentences = [
            "TOPIC_A 첫 번째 항공 안전 규정과 운항 절차에 대한 설명 문장입니다.",
            "TOPIC_A 두 번째 항공기 정비 점검 주기와 안전 관리 체계 안내 문장입니다.",
            "TOPIC_A 세 번째 조종사 자격 유지와 정기 훈련 기준에 관한 설명 문장입니다.",
            "TOPIC_A 네 번째 항공교통관제 협조 및 비상 대응 매뉴얼 안내 문장입니다.",
            "TOPIC_A 다섯 번째 활주로 안전 점검과 기상 관측 보고 절차 설명 문장입니다.",
            "TOPIC_B 여섯 번째 사내 출장비 정산 기준과 숙박비 한도에 관한 규정입니다.",
            "TOPIC_B 일곱 번째 국내 출장 일비와 교통비 영수증 제출 방법 안내입니다.",
            "TOPIC_B 여덟 번째 국외 출장 환율 적용 기준과 법인카드 사용 수칙입니다.",
            "TOPIC_B 아홉 번째 출장 결과 보고서 작성 기한과 결재 절차 설명입니다.",
            "TOPIC_B 열 번째 여비 지급 예외 사항 및 사후 정산 처리 기준 안내입니다.",
        ];
        let page = sentences.join(" ");

        let embedder = FakeEmbedder::new(
            vec![
                ("TOPIC_A", vec![1.0, 0.0, 0.0, 0.0]),
                ("TOPIC_B", vec![0.0, 1.0, 0.0, 0.0]),
            ],
            vec![0.0, 0.0, 1.0, 0.0],
        );

        let chunks = split_semantic(&page, 1000, "혼합문서", &embedder).await?;
        assert_eq!(embedder.call_count(), sentences.len());
        assert_eq!(chunks.len(), 2, "expected exact break at TOPIC_A -> TOPIC_B shift");
        assert!(chunks[0].contains("TOPIC_A 다섯 번째"));
        assert!(!chunks[0].contains("TOPIC_B"));
        assert!(chunks[1].starts_with("TOPIC_B 여섯 번째"));
        assert!(!chunks[1].contains("TOPIC_A"));
        for c in &chunks {
            assert!(char_count(c) <= 1000);
        }

        // Verify max cap still holds when a single topic exceeds max_chars (each topic is ~220 chars).
        let capped = split_semantic(&page, 120, "혼합문서", &embedder).await?;
        assert!(capped.len() >= 4, "expected >= 4 chunks at max_chars=120, got {}", capped.len());
        for (idx, c) in capped.iter().enumerate() {
            assert!(char_count(c) <= 120, "chunk {idx} len {} > 120", char_count(c));
        }

        Ok(())
    }

    #[tokio::test]
    async fn semantic_falls_back_to_structure_when_fewer_than_four_sentences()
    -> Result<(), Box<dyn std::error::Error>> {
        let short_page = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.";
        let embedder = FakeEmbedder::new(vec![], vec![1.0, 0.0]);
        let sem = split_semantic(short_page, 1000, "짧은문서", &embedder).await?;
        let st = split_structured(short_page, 1000, "짧은문서");
        assert_eq!(sem, st);
        assert_eq!(embedder.call_count(), 0, "embedder should not be called for < 4 sentences");
        Ok(())
    }

    #[tokio::test]
    async fn chunk_page_dispatches_all_variants() -> Result<(), Box<dyn std::error::Error>> {
        let embedder = DeterministicTextEmbedder::default();
        for (sample, sep) in [(PDF_SHAPE_1, "\n"), (PDF_SHAPE_2, "\n\n")] {
            let text = repeat_to_chars(sample, sep, 3000);
            let fixed = chunk_page(ChunkerKind::Fixed, &text, 1000, "t", &embedder).await?;
            let structure = chunk_page(ChunkerKind::Structure, &text, 1000, "t", &embedder).await?;
            let semantic = chunk_page(ChunkerKind::Semantic, &text, 1000, "t", &embedder).await?;
            assert!(!fixed.is_empty());
            assert!(!structure.is_empty());
            assert!(!semantic.is_empty());
        }
        Ok(())
    }

    #[test]
    fn config_chunker_defaults_and_yaml_parsing() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(ChunkerKind::default(), ChunkerKind::Structure);

        let base = include_str!("../../../configs/pipeline.default.yaml");
        let cfg = PipelineConfig::from_yaml_str(base)?;
        assert_eq!(cfg.store.chunker, ChunkerKind::Structure);

        let as_fixed = base.replace("chunker: structure", "chunker: fixed");
        assert_eq!(PipelineConfig::from_yaml_str(&as_fixed)?.store.chunker, ChunkerKind::Fixed);

        let as_struct = as_fixed.replace("chunker: fixed", "chunker: structure");
        assert_eq!(PipelineConfig::from_yaml_str(&as_struct)?.store.chunker, ChunkerKind::Structure);

        let as_sem = base.replace("chunker: structure", "chunker: semantic");
        assert_eq!(PipelineConfig::from_yaml_str(&as_sem)?.store.chunker, ChunkerKind::Semantic);

        let omitted = base.replace("  chunker: structure\n", "");
        assert_eq!(PipelineConfig::from_yaml_str(&omitted)?.store.chunker, ChunkerKind::Structure);

        // Verify serde lowercase roundtrip directly.
        let mut map = HashMap::new();
        map.insert("fixed", ChunkerKind::Fixed);
        map.insert("structure", ChunkerKind::Structure);
        map.insert("semantic", ChunkerKind::Semantic);
        for (raw, expected) in map {
            let parsed: ChunkerKind = serde_yaml_ng::from_str(raw)?;
            assert_eq!(parsed, expected);
        }
        Ok(())
    }
}
