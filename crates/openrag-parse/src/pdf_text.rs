//! Text of a PDF page. The content stream is read as bytes and each string is decoded with the
//! font it is shown in: the font's `/ToUnicode` CMap when it has one (subset TrueType fonts from
//! LibreOffice, `Identity-H` CID fonts from Word, Hangul, Chrome and macOS), a Unicode CMap
//! encoding (`UniKS-UCS2-H`), or WinAnsi for simple fonts without one. Text inside form XObjects
//! is read too. Lines break where the text moves to another line.

use std::collections::HashMap;
use std::io::Read;

use flate2::read::ZlibDecoder;

use crate::pdf::PdfObj;

/// Upper bound for a decompressed stream (content, CMap, object stream).
const MAX_STREAM_BYTES: u64 = 64 * 1024 * 1024;
/// Form XObjects inside form XObjects.
const MAX_FORM_DEPTH: usize = 4;

// ---------------------------------------------------------------------------
// Dictionaries
// ---------------------------------------------------------------------------

fn is_delim(c: u8) -> bool {
    matches!(c, b'/' | b'<' | b'>' | b'[' | b']' | b'(' | b')' | b'%' | b'{' | b'}')
}

fn is_ws(c: u8) -> bool {
    c.is_ascii_whitespace() || c == 0
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() {
        if is_ws(b[i]) {
            i += 1;
        } else if b[i] == b'%' {
            while i < b.len() && b[i] != b'\n' && b[i] != b'\r' {
                i += 1;
            }
        } else {
            break;
        }
    }
    i
}

fn token_end(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && !is_ws(b[i]) && !is_delim(b[i]) {
        i += 1;
    }
    i
}

fn literal_end(b: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    let mut depth = 1usize;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 1,
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    b.len()
}

/// End (exclusive) of the object starting at `i`.
fn object_end(b: &[u8], i: usize) -> usize {
    match b.get(i) {
        Some(b'<') if b.get(i + 1) == Some(&b'<') => {
            let mut j = i + 2;
            loop {
                j = skip_ws(b, j);
                if j >= b.len() {
                    return b.len();
                }
                if b[j] == b'>' && b.get(j + 1) == Some(&b'>') {
                    return j + 2;
                }
                let next = object_end(b, j);
                j = if next > j { next } else { j + 1 };
            }
        }
        Some(b'<') => b[i..].iter().position(|&c| c == b'>').map_or(b.len(), |p| i + p + 1),
        Some(b'[') => {
            let mut j = i + 1;
            loop {
                j = skip_ws(b, j);
                if j >= b.len() {
                    return b.len();
                }
                if b[j] == b']' {
                    return j + 1;
                }
                let next = object_end(b, j);
                j = if next > j { next } else { j + 1 };
            }
        }
        Some(b'(') => literal_end(b, i),
        Some(b'/') => token_end(b, i + 1),
        Some(_) => {
            let end = token_end(b, i);
            // `12 0 R` is one value.
            if b[i..end].iter().all(u8::is_ascii_digit) && end > i {
                let g = skip_ws(b, end);
                let g_end = token_end(b, g);
                if g_end > g && b[g..g_end].iter().all(u8::is_ascii_digit) {
                    let r = skip_ws(b, g_end);
                    if b.get(r) == Some(&b'R') && b.get(r + 1).is_none_or(|&c| is_ws(c) || is_delim(c)) {
                        return r + 1;
                    }
                }
            }
            end
        }
        None => i,
    }
}

/// Top-level entries of the first dictionary in `src`: (key without `/`, raw value).
pub(crate) fn dict_entries(src: &str) -> Vec<(String, String)> {
    let b = src.as_bytes();
    let Some(open) = src.find("<<") else { return Vec::new() };
    let mut out = Vec::new();
    let mut i = open + 2;
    loop {
        i = skip_ws(b, i);
        if i >= b.len() || (b[i] == b'>' && b.get(i + 1) == Some(&b'>')) {
            break;
        }
        if b[i] != b'/' {
            let next = object_end(b, i);
            i = if next > i { next } else { i + 1 };
            continue;
        }
        let key_end = token_end(b, i + 1);
        let key = src.get(i + 1..key_end).unwrap_or("").to_string();
        let v = skip_ws(b, key_end);
        let v_end = object_end(b, v);
        out.push((key, src.get(v..v_end).unwrap_or("").to_string()));
        i = if v_end > v { v_end } else { v + 1 };
    }
    out
}

fn entry<'e>(entries: &'e [(String, String)], key: &str) -> Option<&'e str> {
    entries.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn ref_id(v: &str) -> Option<u32> {
    let mut it = v.split_whitespace();
    let (id, _gen, r) = (it.next()?, it.next()?, it.next()?);
    (r == "R").then(|| id.parse().ok()).flatten()
}

/// A value as a dictionary: inline, or the dictionary of the object it refers to.
fn resolve_dict(v: &str, objects: &HashMap<u32, PdfObj<'_>>) -> String {
    match ref_id(v) {
        Some(id) => objects.get(&id).map(|o| o.dict.to_string()).unwrap_or_default(),
        None => v.to_string(),
    }
}

fn ref_list(v: &str) -> Vec<u32> {
    let toks: Vec<&str> = v
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split_whitespace()
        .collect();
    toks.windows(3)
        .filter(|w| w[2] == "R")
        .filter_map(|w| w[0].parse::<u32>().ok())
        .collect()
}

/// The stream of an object, decompressed when it is Flate-encoded.
pub(crate) fn stream_data(obj: &PdfObj<'_>) -> Option<Vec<u8>> {
    let raw = obj.stream?;
    if obj.dict.contains("/FlateDecode") || obj.dict.contains("/Fl ") || obj.dict.contains("/Fl]") {
        let mut out = Vec::new();
        match ZlibDecoder::new(raw).take(MAX_STREAM_BYTES).read_to_end(&mut out) {
            Ok(_) => Some(out),
            // A truncated stream still yields what was decoded before the error.
            Err(_) if !out.is_empty() => Some(out),
            Err(_) => None,
        }
    } else {
        Some(raw.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Object streams
// ---------------------------------------------------------------------------

/// Objects packed in `/Type /ObjStm` streams (PDF 1.5+, e.g. Word): (number, object text).
pub(crate) fn object_stream_members(obj: &PdfObj<'_>) -> Vec<(u32, String)> {
    if !(obj.dict.contains("/Type /ObjStm") || obj.dict.contains("/Type/ObjStm")) {
        return Vec::new();
    }
    let entries = dict_entries(&obj.dict);
    let (Some(n), Some(first)) = (
        entry(&entries, "N").and_then(|v| v.trim().parse::<usize>().ok()),
        entry(&entries, "First").and_then(|v| v.trim().parse::<usize>().ok()),
    ) else {
        return Vec::new();
    };
    let Some(data) = stream_data(obj) else { return Vec::new() };
    let header = String::from_utf8_lossy(&data[..first.min(data.len())]).into_owned();
    let nums: Vec<usize> = header.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    let pairs: Vec<(usize, usize)> = nums.chunks(2).take(n).filter_map(|c| Some((*c.first()?, *c.get(1)?))).collect();
    let mut out = Vec::with_capacity(pairs.len());
    for (k, &(num, off)) in pairs.iter().enumerate() {
        let start = first.saturating_add(off);
        let end = pairs.get(k + 1).map_or(data.len(), |&(_, o)| first.saturating_add(o)).min(data.len());
        if start >= end {
            continue;
        }
        if let Ok(num) = u32::try_from(num) {
            out.push((num, String::from_utf8_lossy(&data[start..end]).into_owned()));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct CMap {
    /// (byte length, low, high) of each codespace range.
    ranges: Vec<(usize, u32, u32)>,
    map: HashMap<(usize, u32), String>,
}

fn be(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0u32, |acc, &b| (acc << 8) | u32::from(b))
}

fn utf16(bytes: &[u8]) -> String {
    let units: Vec<u16> = bytes.chunks(2).map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)])).collect();
    String::from_utf16_lossy(&units)
}

#[derive(Debug)]
enum CTok {
    Hex(Vec<u8>),
    Open,
    Close,
    Word(String),
}

fn hex_bytes(s: &[u8]) -> Vec<u8> {
    let digits: Vec<u8> = s
        .iter()
        .filter_map(|c| match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        })
        .collect();
    digits.chunks(2).map(|p| (p[0] << 4) | p.get(1).copied().unwrap_or(0)).collect()
}

fn cmap_tokens(b: &[u8]) -> Vec<CTok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        i = skip_ws(b, i);
        let Some(&c) = b.get(i) else { break };
        match c {
            b'<' if b.get(i + 1) == Some(&b'<') => i += 2,
            b'>' if b.get(i + 1) == Some(&b'>') => i += 2,
            b'<' => {
                let end = b[i..].iter().position(|&x| x == b'>').map_or(b.len(), |p| i + p);
                out.push(CTok::Hex(hex_bytes(&b[i + 1..end])));
                i = end + 1;
            }
            b'[' => {
                out.push(CTok::Open);
                i += 1;
            }
            b']' => {
                out.push(CTok::Close);
                i += 1;
            }
            b'(' => i = literal_end(b, i),
            b'/' => i = token_end(b, i + 1),
            _ => {
                let end = token_end(b, i).max(i + 1);
                out.push(CTok::Word(String::from_utf8_lossy(&b[i..end]).into_owned()));
                i = end;
            }
        }
    }
    out
}

impl CMap {
    fn parse(b: &[u8]) -> Self {
        let toks = cmap_tokens(b);
        let mut cm = CMap::default();
        let mut i = 0;
        let word = |t: Option<&CTok>, w: &str| matches!(t, Some(CTok::Word(x)) if x == w);
        while i < toks.len() {
            match &toks[i] {
                CTok::Word(w) if w == "begincodespacerange" => {
                    i += 1;
                    while i + 1 < toks.len() && !word(toks.get(i), "endcodespacerange") {
                        if let (CTok::Hex(lo), CTok::Hex(hi)) = (&toks[i], &toks[i + 1]) {
                            if !lo.is_empty() && lo.len() <= 4 {
                                cm.ranges.push((lo.len(), be(lo), be(hi)));
                            }
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                }
                CTok::Word(w) if w == "beginbfchar" => {
                    i += 1;
                    while i + 1 < toks.len() && !word(toks.get(i), "endbfchar") {
                        if let (CTok::Hex(src), CTok::Hex(dst)) = (&toks[i], &toks[i + 1]) {
                            if !src.is_empty() && src.len() <= 4 {
                                cm.map.insert((src.len(), be(src)), utf16(dst));
                            }
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                }
                CTok::Word(w) if w == "beginbfrange" => {
                    i += 1;
                    while i + 2 < toks.len() && !word(toks.get(i), "endbfrange") {
                        let (CTok::Hex(lo), CTok::Hex(hi)) = (&toks[i], &toks[i + 1]) else {
                            i += 1;
                            continue;
                        };
                        let (len, lo_v, hi_v) = (lo.len(), be(lo), be(hi));
                        if len == 0 || len > 4 || hi_v < lo_v || hi_v - lo_v > 0xFFFF {
                            i += 2;
                            continue;
                        }
                        match &toks[i + 2] {
                            CTok::Hex(dst) if !dst.is_empty() => {
                                let mut units: Vec<u16> =
                                    dst.chunks(2).map(|c| u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)])).collect();
                                let last = units.len() - 1;
                                let base = units[last];
                                for (k, code) in (lo_v..=hi_v).enumerate() {
                                    units[last] = base.wrapping_add(u16::try_from(k).unwrap_or(u16::MAX));
                                    cm.map.insert((len, code), String::from_utf16_lossy(&units));
                                }
                                i += 3;
                            }
                            CTok::Open => {
                                let mut j = i + 3;
                                let mut code = lo_v;
                                while j < toks.len() {
                                    match &toks[j] {
                                        CTok::Hex(dst) => {
                                            if code <= hi_v {
                                                cm.map.insert((len, code), utf16(dst));
                                            }
                                            code = code.saturating_add(1);
                                            j += 1;
                                        }
                                        CTok::Close => {
                                            j += 1;
                                            break;
                                        }
                                        _ => j += 1,
                                    }
                                }
                                i = j;
                            }
                            _ => i += 3,
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
        cm
    }

    fn lengths(&self) -> Vec<usize> {
        let mut lens: Vec<usize> = if self.ranges.is_empty() {
            self.map.keys().map(|(l, _)| *l).collect()
        } else {
            self.ranges.iter().map(|(l, _, _)| *l).collect()
        };
        lens.sort_unstable();
        lens.dedup();
        if lens.is_empty() {
            lens.push(1);
        }
        lens
    }

    fn in_space(&self, len: usize, code: u32) -> bool {
        self.ranges.is_empty() || self.ranges.iter().any(|&(l, lo, hi)| l == len && (lo..=hi).contains(&code))
    }

    fn decode(&self, bytes: &[u8]) -> String {
        let lens = self.lengths();
        let mut out = String::new();
        let mut i = 0;
        while i < bytes.len() {
            let mut step = 1;
            for &l in &lens {
                if i + l > bytes.len() {
                    continue;
                }
                let code = be(&bytes[i..i + l]);
                if self.in_space(l, code) {
                    if let Some(s) = self.map.get(&(l, code)) {
                        out.push_str(s);
                    }
                    step = l;
                    break;
                }
            }
            i += step;
        }
        out
    }
}

/// WinAnsi (cp1252) for simple fonts without a CMap.
fn win_ansi(bytes: &[u8]) -> String {
    const HIGH: [char; 32] = [
        '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}', '\u{90}',
        '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ',
    ];
    bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => HIGH[usize::from(b - 0x80)],
            _ => char::from(b),
        })
        .collect()
}

#[derive(Debug)]
enum FontText {
    CMap(CMap),
    /// A CID font whose encoding is Unicode itself (`UniKS-UCS2-H`, `…-UTF16-H`).
    Utf16,
    /// A CID font without a ToUnicode map: its codes are glyph numbers, not characters.
    Unreadable,
    Simple,
}

fn font_text(dict: &str, objects: &HashMap<u32, PdfObj<'_>>) -> FontText {
    let e = dict_entries(dict);
    if let Some(cm) = entry(&e, "ToUnicode")
        .and_then(ref_id)
        .and_then(|id| objects.get(&id))
        .and_then(stream_data)
        .map(|data| CMap::parse(&data))
        .filter(|cm| !cm.map.is_empty())
    {
        return FontText::CMap(cm);
    }
    let subtype = entry(&e, "Subtype").unwrap_or("");
    if subtype.contains("Type0") {
        let enc = entry(&e, "Encoding").unwrap_or("");
        if enc.contains("UCS2") || enc.contains("UTF16") {
            return FontText::Utf16;
        }
        return FontText::Unreadable;
    }
    FontText::Simple
}

fn fallback_text(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xFE, 0xFF]) {
        return utf16(&bytes[2..]);
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => win_ansi(bytes),
    }
}

// ---------------------------------------------------------------------------
// Content streams
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Operand {
    Num(f64),
    Name(String),
    Str(Vec<u8>),
    Arr(Vec<Operand>),
    /// An inline dictionary, as written (marked-content properties).
    Dict(String),
    Other,
}

enum Item {
    Operand(Operand),
    Op(String),
}

struct Lexer<'b> {
    b: &'b [u8],
    i: usize,
}

fn literal_bytes(b: &[u8], start: usize) -> (Vec<u8>, usize) {
    let mut i = start + 1;
    let mut depth = 1usize;
    let mut out = Vec::new();
    while i < b.len() {
        match b[i] {
            b'\\' => {
                i += 1;
                let Some(&c) = b.get(i) else { break };
                match c {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0C),
                    b'0'..=b'7' => {
                        let mut v = u32::from(c - b'0');
                        for _ in 0..2 {
                            match b.get(i + 1) {
                                Some(&d @ b'0'..=b'7') => {
                                    v = v * 8 + u32::from(d - b'0');
                                    i += 1;
                                }
                                _ => break,
                            }
                        }
                        out.push((v & 0xFF) as u8);
                    }
                    b'\r' => {
                        if b.get(i + 1) == Some(&b'\n') {
                            i += 1;
                        }
                    }
                    b'\n' => {}
                    other => out.push(other),
                }
            }
            b'(' => {
                depth += 1;
                out.push(b'(');
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return (out, i + 1);
                }
                out.push(b')');
            }
            c => out.push(c),
        }
        i += 1;
    }
    (out, b.len())
}

impl<'b> Lexer<'b> {
    fn operand(&mut self) -> Option<Operand> {
        let b = self.b;
        self.i = skip_ws(b, self.i);
        let c = *b.get(self.i)?;
        Some(match c {
            b'(' => {
                let (s, end) = literal_bytes(b, self.i);
                self.i = end;
                Operand::Str(s)
            }
            b'<' if b.get(self.i + 1) == Some(&b'<') => {
                let start = self.i;
                self.i = object_end(b, self.i);
                Operand::Dict(String::from_utf8_lossy(&b[start..self.i]).into_owned())
            }
            b'<' => {
                let end = b[self.i..].iter().position(|&x| x == b'>').map_or(b.len(), |p| self.i + p);
                let s = hex_bytes(&b[self.i + 1..end]);
                self.i = (end + 1).min(b.len());
                Operand::Str(s)
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.i = skip_ws(b, self.i);
                    match b.get(self.i) {
                        None => break,
                        Some(b']') => {
                            self.i += 1;
                            break;
                        }
                        Some(_) => {
                            let before = self.i;
                            match self.operand() {
                                Some(op) => items.push(op),
                                None => break,
                            }
                            if self.i == before {
                                self.i += 1;
                            }
                        }
                    }
                }
                Operand::Arr(items)
            }
            b'/' => {
                let end = token_end(b, self.i + 1);
                let name = String::from_utf8_lossy(&b[self.i + 1..end]).into_owned();
                self.i = end;
                Operand::Name(name)
            }
            b')' | b'>' | b']' | b'{' | b'}' => {
                self.i += 1;
                Operand::Other
            }
            _ => {
                let end = token_end(b, self.i).max(self.i + 1);
                let tok = &b[self.i..end];
                self.i = end;
                match std::str::from_utf8(tok).ok().and_then(|t| t.parse::<f64>().ok()) {
                    Some(n) if tok.first().is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.')) => Operand::Num(n),
                    _ => Operand::Other,
                }
            }
        })
    }

    fn next(&mut self) -> Option<Item> {
        let b = self.b;
        self.i = skip_ws(b, self.i);
        let c = *b.get(self.i)?;
        let starts_operand = matches!(c, b'(' | b'<' | b'[' | b'/' | b'-' | b'+' | b'.' | b'0'..=b'9' | b')' | b'>' | b']' | b'{' | b'}');
        if starts_operand {
            return self.operand().map(Item::Operand);
        }
        let end = token_end(b, self.i).max(self.i + 1);
        let op = String::from_utf8_lossy(&b[self.i..end]).into_owned();
        self.i = end;
        if op == "ID" {
            // Inline image data runs to `EI` between whitespace.
            let mut j = self.i + 1;
            while j + 1 < b.len() {
                if b[j] == b'E' && b[j + 1] == b'I' && is_ws(b[j - 1]) && b.get(j + 2).is_none_or(|&x| is_ws(x)) {
                    break;
                }
                j += 1;
            }
            self.i = (j + 2).min(b.len());
        }
        Some(Item::Op(op))
    }
}

#[derive(Default)]
struct TextOut {
    lines: Vec<String>,
    cur: String,
    /// Vertical position of the text matrix and of the current line.
    y: f64,
    line_y: Option<f64>,
}

impl TextOut {
    fn newline(&mut self) {
        let t = self.cur.trim();
        if !t.is_empty() {
            self.lines.push(t.to_string());
        }
        self.cur.clear();
    }

    fn moved_to(&mut self, y: f64) {
        self.y = y;
        if self.line_y.is_some_and(|ly| (ly - y).abs() > 2.0) {
            self.newline();
        }
        self.line_y = Some(y);
    }

    fn space(&mut self) {
        if !self.cur.is_empty() && !self.cur.ends_with(' ') {
            self.cur.push(' ');
        }
    }
}

struct Interp<'o, 'a> {
    objects: &'o HashMap<u32, PdfObj<'a>>,
    out: TextOut,
    /// Open marked-content sections; `Some` carries an `/ActualText` replacement (Chrome writes
    /// spaces this way) and hides the glyphs shown inside it.
    marked: Vec<Option<String>>,
    /// Characters decoded, and bytes shown in fonts that cannot be decoded.
    decoded: usize,
    unreadable: usize,
}

/// The `/ActualText` of marked-content properties.
fn actual_text(props: &str) -> Option<String> {
    let e = dict_entries(props);
    let v = entry(&e, "ActualText")?.trim();
    let bytes = if v.starts_with('(') {
        literal_bytes(v.as_bytes(), 0).0
    } else if v.starts_with('<') {
        hex_bytes(v.trim_start_matches('<').trim_end_matches('>').as_bytes())
    } else {
        return None;
    };
    Some(fallback_text(&bytes))
}

fn num(o: Option<&Operand>) -> f64 {
    match o {
        Some(Operand::Num(n)) => *n,
        _ => 0.0,
    }
}

impl Interp<'_, '_> {
    fn fonts(&self, resources: &str) -> HashMap<String, FontText> {
        let res = dict_entries(resources);
        let Some(fonts_v) = entry(&res, "Font") else { return HashMap::new() };
        dict_entries(&resolve_dict(fonts_v, self.objects))
            .into_iter()
            .map(|(name, v)| (name, font_text(&resolve_dict(&v, self.objects), self.objects)))
            .collect()
    }

    fn show(&mut self, fonts: &HashMap<String, FontText>, font: Option<&str>, bytes: &[u8]) {
        if self.marked.iter().any(Option::is_some) {
            return;
        }
        let text = match font.and_then(|f| fonts.get(f)) {
            Some(FontText::CMap(cm)) => cm.decode(bytes),
            Some(FontText::Utf16) => utf16(bytes),
            Some(FontText::Unreadable) => {
                self.unreadable += bytes.len() / 2;
                String::new()
            }
            Some(FontText::Simple) => win_ansi(bytes),
            None => fallback_text(bytes),
        };
        self.decoded += text.chars().filter(|c| !c.is_whitespace()).count();
        self.out.cur.push_str(&text);
    }

    fn run(&mut self, data: &[u8], resources: &str, depth: usize) {
        let fonts = self.fonts(resources);
        let mut lx = Lexer { b: data, i: 0 };
        let mut ops: Vec<Operand> = Vec::new();
        let mut font: Option<String> = None;
        // Text matrix: x scale (c, d) for Td and the line origin y (f).
        let (mut tm_d, mut line_f) = (1.0f64, 0.0f64);
        while let Some(item) = lx.next() {
            let op = match item {
                Item::Operand(o) => {
                    ops.push(o);
                    continue;
                }
                Item::Op(op) => op,
            };
            match op.as_str() {
                "BT" => {
                    tm_d = 1.0;
                    line_f = 0.0;
                }
                "Tf" => {
                    if let Some(Operand::Name(n)) = ops.first() {
                        font = Some(n.clone());
                    }
                }
                "Tm" => {
                    tm_d = num(ops.get(3));
                    line_f = num(ops.get(5));
                    self.out.moved_to(line_f);
                }
                "Td" | "TD" => {
                    line_f += num(ops.get(1)) * if tm_d == 0.0 { 1.0 } else { tm_d };
                    self.out.moved_to(line_f);
                }
                "T*" => {
                    self.out.newline();
                    self.out.line_y = None;
                }
                "Tj" | "'" | "\"" => {
                    if op != "Tj" {
                        self.out.newline();
                        self.out.line_y = None;
                    }
                    if let Some(Operand::Str(s)) = ops.last() {
                        let s = s.clone();
                        self.show(&fonts, font.as_deref(), &s);
                    }
                }
                "TJ" => {
                    if let Some(Operand::Arr(items)) = ops.last() {
                        let items = items.clone();
                        for it in items {
                            match it {
                                Operand::Str(s) => self.show(&fonts, font.as_deref(), &s),
                                // A large negative adjustment is a word gap.
                                Operand::Num(n) if n < -200.0 => self.out.space(),
                                _ => {}
                            }
                        }
                    }
                }
                "BMC" => self.marked.push(None),
                "BDC" => {
                    let props = match ops.last() {
                        Some(Operand::Dict(d)) => actual_text(d),
                        _ => None,
                    };
                    self.marked.push(props);
                }
                "EMC" => {
                    if let Some(Some(text)) = self.marked.pop() {
                        if !self.marked.iter().any(Option::is_some) {
                            self.decoded += text.chars().filter(|c| !c.is_whitespace()).count();
                            self.out.cur.push_str(&text);
                        }
                    }
                }
                "Do" if depth < MAX_FORM_DEPTH => {
                    if let Some(Operand::Name(n)) = ops.first() {
                        self.form(resources, n, depth);
                    }
                }
                _ => {}
            }
            ops.clear();
        }
    }

    fn form(&mut self, resources: &str, name: &str, depth: usize) {
        let res = dict_entries(resources);
        let Some(xo) = entry(&res, "XObject") else { return };
        let xobjects = dict_entries(&resolve_dict(xo, self.objects));
        let Some(obj) = entry(&xobjects, name).and_then(ref_id).and_then(|id| self.objects.get(&id)) else {
            return;
        };
        if !(obj.dict.contains("/Subtype /Form") || obj.dict.contains("/Subtype/Form")) {
            return;
        }
        let Some(data) = stream_data(obj) else { return };
        let fe = dict_entries(&obj.dict);
        let form_res = entry(&fe, "Resources").map(|v| resolve_dict(v, self.objects)).unwrap_or_else(|| resources.to_string());
        let saved = (self.out.y, self.out.line_y);
        self.run(&data, &form_res, depth + 1);
        (self.out.y, self.out.line_y) = saved;
    }
}

/// `/Resources` of a page, inherited from the page tree when the page has none.
fn page_resources(page: &PdfObj<'_>, objects: &HashMap<u32, PdfObj<'_>>) -> String {
    let mut dict = page.dict.to_string();
    for _ in 0..16 {
        let e = dict_entries(&dict);
        if let Some(v) = entry(&e, "Resources") {
            return resolve_dict(v, objects);
        }
        match entry(&e, "Parent").and_then(ref_id).and_then(|id| objects.get(&id)) {
            Some(parent) => dict = parent.dict.to_string(),
            None => break,
        }
    }
    String::new()
}

/// Control characters other than newline and tab never reach storage (PostgreSQL refuses NUL).
pub(crate) fn clean(text: &str) -> String {
    text.chars().filter(|&c| !c.is_control() || c == '\n' || c == '\t').collect()
}

/// The text of one page.
pub(crate) fn page_text(page: &PdfObj<'_>, objects: &HashMap<u32, PdfObj<'_>>) -> String {
    let e = dict_entries(&page.dict);
    let Some(contents) = entry(&e, "Contents") else { return String::new() };
    let ids = match ref_id(contents) {
        Some(id) => match objects.get(&id) {
            // `/Contents 5 0 R` may point at an array of streams.
            Some(o) if o.stream.is_none() && o.dict.trim_start().starts_with('[') => ref_list(&o.dict),
            _ => vec![id],
        },
        None => ref_list(contents),
    };
    let mut data = Vec::new();
    for id in ids {
        if let Some(chunk) = objects.get(&id).and_then(stream_data) {
            data.extend_from_slice(&chunk);
            data.push(b'\n');
        }
    }
    let resources = page_resources(page, objects);
    let mut it = Interp { objects, out: TextOut::default(), marked: Vec::new(), decoded: 0, unreadable: 0 };
    it.run(&data, &resources, 0);
    it.out.newline();
    // Mostly glyph numbers without a way to characters (a CID font with no ToUnicode): the
    // page counts as having no text, so OCR can read it, instead of storing the scraps.
    if it.unreadable > it.decoded {
        return String::new();
    }
    clean(&it.out.lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cid_font_without_tounicode_gives_no_text() {
        // Glyph numbers with no way back to characters: the page counts as text-less (so OCR
        // can read it) instead of keeping the few ASCII scraps around them.
        let content = "BT /C0 12 Tf 72 700 Td <0A1B0C2D0E3F0A1B> Tj /F1 12 Tf (:) Tj ET";
        let pdf = format!(
            "%PDF-1.7\n1 0 obj\n<< /Type /Page /Resources << /Font << /C0 3 0 R /F1 4 0 R >> >> /Contents 2 0 R >>\nendobj\n\
             2 0 obj\n<< /Length {} >>\nstream\n{content}\nendstream\nendobj\n\
             3 0 obj\n<< /Type /Font /Subtype /Type0 /Encoding /Identity-H /DescendantFonts [] >>\nendobj\n\
             4 0 obj\n<< /Type /Font /Subtype /TrueType /Encoding /MacRomanEncoding >>\nendobj\n%%EOF\n",
            content.len()
        );
        let objects = crate::pdf::parse_pdf_objects(pdf.as_bytes());
        let page = objects.get(&1).map(|p| page_text(p, &objects));
        assert_eq!(page.as_deref(), Some(""));
    }

    #[test]
    fn control_characters_never_reach_storage() {
        assert_eq!(clean("a\u{0}b\u{7}\tc\nd"), "ab\tc\nd");
    }
}
