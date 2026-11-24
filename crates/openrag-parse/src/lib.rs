#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

pub mod aliases;
pub mod html;
pub mod hwp;
pub mod model;
pub mod office;
pub mod ocr;
pub mod pdf;
mod pdf_text;
pub mod spreadsheet;
pub mod s3;
pub mod store;
pub mod thumbnail;

pub use aliases::extract_parenthetical_aliases;
pub use html::{html_to_markdown, parse_html};
pub use office::{parse_docx, parse_pptx};
pub use spreadsheet::{find_rows, parse_spreadsheet, read_sheet, RowFilter, SheetSlice};
pub use hwp::{escape_html, parse_hml, parse_hwp5, parse_hwp_family, parse_hwpx, sanitize_table_html, KordocCliConfig};
pub use model::{
    sha256_hex_bytes, sha256_hex_str, DocumentFormat, OcrBlock, OcrExecutionTarget,
    OcrPolicyDecision, OcrTask, ParsePolicyCtx, ParsedDocument, ParsedFigure, ParsedPage,
    RenderedPageThumbnail,
};
pub use ocr::{
    detect_foreground_blocks, dewarp_image, ocr_pdf_pages, ocr_whole_pdf, parse_image_with_ocr,
    resolve_ocr_policy, resolve_ocr_policy_with_target, SchiftOcrClient, SchiftOcrResponse,
    ShareQuotaTracker, DEFAULT_SHARE_DAILY_LIMIT,
};
pub use pdf::{parse_pdf_native, parse_pdf_with_scan_pages};
pub use s3::{S3BlobConfig, S3BlobStore};
pub use store::{ParsePersister, PersistParsedDocumentInput, PersistedDocumentIds};
pub use thumbnail::{
    encode_png_bounded, encode_webp, InMemoryBlobStore, LocalFsBlobStore, ThumbnailBlobStore,
    Thumbnailer,
};

use std::path::Path;

use openrag_core::{OpenRagError, ParseConfig};

#[derive(Debug, Default)]
pub struct DocumentParser {
    pub kordoc: Option<KordocCliConfig>,
    pub share_quota: ShareQuotaTracker,
    pub ocr_client: Option<SchiftOcrClient>,
}

impl DocumentParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// The OCR endpoint, when the policy would send this document to one.
    fn remote_ocr(&self, cfg: &ParseConfig, ctx: &ParsePolicyCtx) -> Option<&SchiftOcrClient> {
        let client = self.ocr_client.as_ref()?;
        let policy = resolve_ocr_policy_with_target(cfg, ctx, true).ok()?;
        (policy.execution_target == OcrExecutionTarget::SchiftCloud).then_some(client)
    }

    /// PDFs keep their text layer; scanned pages (no text, an image) go to OCR. Without OCR a
    /// PDF with no text at all is refused instead of being stored empty.
    async fn parse_pdf(
        &self,
        bytes: &[u8],
        filename: Option<&str>,
        cfg: &ParseConfig,
        ctx: &ParsePolicyCtx,
    ) -> Result<ParsedDocument, OpenRagError> {
        let ocr = self.remote_ocr(cfg, ctx);
        let (mut doc, scan_pages) = match parse_pdf_with_scan_pages(bytes, filename, cfg).await {
            Ok(v) => v,
            // The native parser cannot walk every PDF (e.g. compressed object streams).
            Err(OpenRagError::Validation(msg)) if msg.starts_with("corrupt PDF") => match ocr {
                Some(client) => {
                    return ocr_whole_pdf(bytes, filename, cfg, ctx, Some(&self.share_quota), client).await
                }
                None => return Err(OpenRagError::Validation(msg)),
            },
            Err(e) => return Err(e),
        };
        if scan_pages.is_empty() {
            return Ok(doc);
        }
        if let Some(client) = ocr {
            ocr_pdf_pages(&mut doc, bytes, filename, &scan_pages, cfg, ctx, Some(&self.share_quota), client)
                .await?;
            return Ok(doc);
        }
        if scan_pages.len() == doc.pages.len() {
            return Err(OpenRagError::Validation(
                "this PDF has no text layer (a scan?) and OCR is not configured: set OPENRAG_SCHIFT_OCR_URL and SCHIFT_OCR_API_KEY".into(),
            ));
        }
        Ok(doc)
    }

    pub async fn parse(
        &self,
        bytes: &[u8],
        mime: Option<&str>,
        filename: Option<&str>,
        cfg: &ParseConfig,
        ctx: &ParsePolicyCtx,
    ) -> Result<ParsedDocument, OpenRagError> {
        if bytes.is_empty() {
            return Err(OpenRagError::Validation("cannot parse empty document".into()));
        }

        let format = DocumentFormat::detect(bytes, mime, filename);
        match format {
            DocumentFormat::Hwpx | DocumentFormat::Hml | DocumentFormat::Hwp => {
                parse_hwp_family(bytes, filename, cfg, self.kordoc.as_ref()).await
            }
            DocumentFormat::Pdf => self.parse_pdf(bytes, filename, cfg, ctx).await,
            DocumentFormat::Png | DocumentFormat::Jpeg | DocumentFormat::Webp => {
                parse_image_with_ocr(
                    bytes,
                    format,
                    filename,
                    cfg,
                    ctx,
                    Some(&self.share_quota),
                    self.ocr_client.as_ref(),
                )
                .await
            }
            DocumentFormat::Docx => crate::office::parse_docx(bytes, filename),
            DocumentFormat::Pptx => crate::office::parse_pptx(bytes, filename),
            DocumentFormat::Html => crate::html::parse_html(bytes, filename),
            DocumentFormat::Xlsx | DocumentFormat::Csv => crate::spreadsheet::parse_spreadsheet(bytes, format, filename),
            DocumentFormat::Markdown | DocumentFormat::PlainText => {
                let text = std::str::from_utf8(bytes).map_err(|e| {
                    OpenRagError::Validation(format!("text document is not valid UTF-8: {e}"))
                })?;
                let aliases = extract_parenthetical_aliases(text);
                let page = ParsedPage {
                    page_no: 1,
                    raw_markdown: text.to_string(),
                    tables_html: Vec::new(),
                    formulas: Vec::new(),
                    fields: Vec::new(),
                    blocks: Vec::new(),
                    extracted_aliases: aliases.clone(),
                    page_image_rgb: None,
                    content_sha256: sha256_hex_str(text),
                };
                let title = filename
                    .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
                    .unwrap_or("Untitled Document")
                    .to_string();
                Ok(ParsedDocument {
                    title,
                    format,
                    mime_type: format.default_mime().to_string(),
                    content_sha256: sha256_hex_bytes(bytes),
                    pages: vec![page],
                    full_markdown: text.to_string(),
                    extracted_aliases: aliases,
                    ocr_policy: None,
                    parser_engine: "text_native".to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scanned_pdf_without_ocr_is_refused_not_stored_empty() {
        let parser = DocumentParser::new();
        let scan = crate::pdf::tests::pdf_with_pages(&[None]);
        let r = parser
            .parse(&scan, Some("application/pdf"), Some("scan.pdf"), &crate::pdf::tests::test_parse_cfg(), &ParsePolicyCtx::default())
            .await;
        assert!(
            matches!(&r, Err(OpenRagError::Validation(m)) if m.contains("no text layer")),
            "{r:?}"
        );
        let mixed = crate::pdf::tests::pdf_with_pages(&[Some("Hello"), None]);
        let doc = parser
            .parse(&mixed, Some("application/pdf"), Some("mixed.pdf"), &crate::pdf::tests::test_parse_cfg(), &ParsePolicyCtx::default())
            .await;
        assert!(doc.is_ok_and(|d| d.pages.len() == 2), "a PDF with some text is kept");
    }
}
