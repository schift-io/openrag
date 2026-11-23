use openrag_core::{AccessScope, AuthLevel, BBoxNorm1000, Sensitivity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentFormat {
    Hwpx,
    Hml,
    Hwp,
    Pdf,
    Png,
    Jpeg,
    Webp,
    Markdown,
    PlainText,
}

impl DocumentFormat {
    pub const fn default_mime(self) -> &'static str {
        match self {
            Self::Hwpx => "application/vnd.hancom.hwpx",
            Self::Hml => "application/vnd.hancom.hml",
            Self::Hwp => "application/x-hwp",
            Self::Pdf => "application/pdf",
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
            Self::Markdown => "text/markdown",
            Self::PlainText => "text/plain",
        }
    }

    pub fn detect(bytes: &[u8], mime: Option<&str>, filename: Option<&str>) -> Self {
        if let Some(m) = mime.map(str::to_ascii_lowercase) {
            if m.contains("hwpx") {
                return Self::Hwpx;
            }
            if m.contains("hml") {
                return Self::Hml;
            }
            if m.contains("x-hwp") || m.contains("haansoft.hwp") {
                return Self::Hwp;
            }
            if m == "application/pdf" {
                return Self::Pdf;
            }
            if m == "image/png" {
                return Self::Png;
            }
            if m == "image/jpeg" || m == "image/jpg" {
                return Self::Jpeg;
            }
            if m == "image/webp" {
                return Self::Webp;
            }
            if m == "text/markdown" {
                return Self::Markdown;
            }
        }

        if let Some(ext) = filename
            .and_then(|f| f.rsplit('.').next())
            .map(str::to_ascii_lowercase)
        {
            match ext.as_str() {
                "hwpx" => return Self::Hwpx,
                "hml" => return Self::Hml,
                "hwp" => return Self::Hwp,
                "pdf" => return Self::Pdf,
                "png" => return Self::Png,
                "jpg" | "jpeg" => return Self::Jpeg,
                "webp" => return Self::Webp,
                "md" | "markdown" => return Self::Markdown,
                "txt" => return Self::PlainText,
                _ => {}
            }
        }

        if bytes.starts_with(b"%PDF-") {
            return Self::Pdf;
        }
        if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Self::Png;
        }
        if bytes.starts_with(b"\xFF\xD8\xFF") {
            return Self::Jpeg;
        }
        if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            return Self::Webp;
        }
        if bytes.starts_with(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") {
            return Self::Hwp;
        }
        if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
            return Self::Hwpx;
        }
        let head = std::str::from_utf8(&bytes[..bytes.len().min(512)]).unwrap_or("");
        if head.contains("<HWPML") || head.contains("<hwpml") {
            return Self::Hml;
        }
        Self::PlainText
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrTask {
    Markdown,
    Structured,
    Layout,
}

impl OcrTask {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Structured => "structured",
            Self::Layout => "layout",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrExecutionTarget {
    SchiftCloud,
    LocalVllm,
    OfflineFallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsePolicyCtx {
    pub auth_level: AuthLevel,
    pub sensitivity: Sensitivity,
    pub requested_ocr_mode: Option<Sensitivity>,
    pub batch: bool,
    pub ocr_task: OcrTask,
    pub tenant_key: Option<String>,
}

impl Default for ParsePolicyCtx {
    fn default() -> Self {
        Self {
            auth_level: AuthLevel::MIN,
            sensitivity: Sensitivity::Normal,
            requested_ocr_mode: None,
            batch: false,
            ocr_task: OcrTask::Markdown,
            tenant_key: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrPolicyDecision {
    pub effective_mode: Sensitivity,
    pub execution_target: OcrExecutionTarget,
    pub upgraded_from_share: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrBlock {
    pub kind: String,
    pub text: String,
    pub bbox_norm_1000: Option<BBoxNorm1000>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedPage {
    pub page_no: i32,
    pub raw_markdown: String,
    pub tables_html: Vec<String>,
    pub formulas: Vec<String>,
    pub fields: Vec<(String, String)>,
    pub blocks: Vec<OcrBlock>,
    pub extracted_aliases: Vec<(String, String)>,
    #[serde(skip)]
    pub page_image_rgb: Option<(u32, u32, Vec<u8>)>,
    pub content_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedDocument {
    pub title: String,
    pub format: DocumentFormat,
    pub mime_type: String,
    pub content_sha256: String,
    pub pages: Vec<ParsedPage>,
    pub full_markdown: String,
    pub extracted_aliases: Vec<(String, String)>,
    pub ocr_policy: Option<OcrPolicyDecision>,
    pub parser_engine: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedFigure {
    pub figure_idx: i32,
    pub filename: String,
    pub kind: String,
    pub caption: Option<String>,
    pub bbox_norm_1000: Option<BBoxNorm1000>,
    pub png_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderedPageThumbnail {
    pub page_no: i32,
    pub width: u32,
    pub height: u32,
    pub webp_bytes: Vec<u8>,
    pub figures: Vec<ParsedFigure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AclScopeInfo {
    pub scope: AccessScope,
    pub auth_level: AuthLevel,
}

pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

pub fn sha256_hex_str(s: &str) -> String {
    sha256_hex_bytes(s.as_bytes())
}
