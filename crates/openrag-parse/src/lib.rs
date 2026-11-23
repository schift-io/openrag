#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

pub mod aliases;
pub mod hwp;
pub mod model;
pub mod ocr;
pub mod pdf;
pub mod s3;
pub mod store;
pub mod thumbnail;

pub use aliases::extract_parenthetical_aliases;
pub use hwp::{escape_html, parse_hml, parse_hwp5, parse_hwp_family, parse_hwpx, sanitize_table_html, KordocCliConfig};
pub use model::{
    sha256_hex_bytes, sha256_hex_str, DocumentFormat, OcrBlock, OcrExecutionTarget,
    OcrPolicyDecision, OcrTask, ParsePolicyCtx, ParsedDocument, ParsedFigure, ParsedPage,
    RenderedPageThumbnail,
};
pub use ocr::{
    detect_foreground_blocks, dewarp_image, parse_image_with_ocr, resolve_ocr_policy,
    resolve_ocr_policy_with_target, SchiftOcrClient, SchiftOcrResponse, ShareQuotaTracker,
    DEFAULT_SHARE_DAILY_LIMIT,
};
pub use pdf::parse_pdf_native;
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
            DocumentFormat::Pdf => parse_pdf_native(bytes, filename, cfg).await,
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
