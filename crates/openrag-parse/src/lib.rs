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
    RemoteOcrConsent, RenderedPageThumbnail,
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

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use openrag_core::{OpenRagError, ParseConfig};
use serde::Deserialize;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
struct KordocPdfOutput {
    success: bool,
    #[serde(rename = "pageCount", default)]
    page_count: usize,
    #[serde(default)]
    pages: Vec<KordocPdfPage>,
}

#[derive(Debug, Deserialize)]
struct KordocPdfPage {
    #[serde(rename = "pageNumber")]
    page_number: usize,
    #[serde(default)]
    markdown: String,
}

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

    /// Parses a PDF through the kordoc CLI (with optional local Korean OCR), returning the
    /// document and 0-based indices of pages that remain empty. Returns `None` on any failure so
    /// the caller can fall back to the native PDF path.
    async fn try_kordoc_pdf(
        &self,
        bytes: &[u8],
        filename: Option<&str>,
        cfg: &ParseConfig,
    ) -> Option<(ParsedDocument, Vec<usize>)> {
        let Some(bin_path) = hwp::resolve_kordoc_bin(self.kordoc.as_ref()) else {
            tracing::warn!("kordoc binary not found; falling back to native PDF parser");
            return None;
        };

        let tmp_path =
            std::env::temp_dir().join(format!("openrag-kordoc-{}.pdf", uuid::Uuid::now_v7()));
        if let Err(e) = tokio::fs::write(&tmp_path, bytes).await {
            tracing::warn!(
                error = %e,
                "failed to write temp PDF for kordoc; falling back to native PDF parser"
            );
            return None;
        }

        let timeout = self
            .kordoc
            .as_ref()
            .map_or_else(|| Duration::from_secs(600), |c| c.pdf_timeout);
        let mut cmd = Command::new(&bin_path);
        cmd.kill_on_drop(true)
            .arg(&tmp_path)
            .arg("--format")
            .arg("json")
            .arg("--plain")
            .arg("--no-images")
            .arg("--silent");
        if cfg.kordoc_ocr {
            cmd.arg("--ocr");
        }

        let output_res = tokio::time::timeout(timeout, cmd.output()).await;
        let _ = tokio::fs::remove_file(&tmp_path).await;

        let output = match output_res {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                tracing::warn!(
                    error = %e,
                    "failed to run kordoc for PDF; falling back to native PDF parser"
                );
                return None;
            }
            Err(_) => {
                tracing::warn!(
                    timeout_secs = timeout.as_secs(),
                    "kordoc timed out parsing PDF; falling back to native PDF parser"
                );
                return None;
            }
        };
        if !output.status.success() {
            tracing::warn!(
                exit_code = output.status.code(),
                "kordoc exited non-zero on PDF; falling back to native PDF parser"
            );
            return None;
        }

        let parsed: KordocPdfOutput = match serde_json::from_slice(&output.stdout) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "kordoc produced invalid JSON for PDF; falling back to native PDF parser"
                );
                return None;
            }
        };
        if !parsed.success {
            tracing::warn!("kordoc returned success=false for PDF; falling back to native PDF parser");
            return None;
        }

        let max_page_no = parsed.pages.iter().map(|p| p.page_number).max().unwrap_or(0);
        let page_count = if parsed.page_count > 0 {
            parsed.page_count
        } else {
            max_page_no
        };
        if page_count == 0 {
            tracing::warn!("kordoc returned zero pages for PDF; falling back to native PDF parser");
            return None;
        }

        let mut by_page: HashMap<usize, String> = HashMap::with_capacity(parsed.pages.len());
        for page in parsed.pages {
            by_page.insert(page.page_number, page.markdown);
        }

        let mut pages = Vec::with_capacity(page_count);
        let mut scan_pages = Vec::new();
        for page_num in 1..=page_count {
            let raw = by_page.get(&page_num).map_or("", String::as_str);
            let raw_markdown = sanitize_table_html(raw);
            if raw_markdown.trim().is_empty() {
                scan_pages.push(page_num - 1);
            }
            let tables_html = hwp::extract_tables_from_text(&raw_markdown);
            let extracted_aliases = extract_parenthetical_aliases(&raw_markdown);
            let content_sha256 = sha256_hex_str(&raw_markdown);
            pages.push(ParsedPage {
                page_no: i32::try_from(page_num).unwrap_or(i32::MAX),
                raw_markdown,
                tables_html,
                formulas: Vec::new(),
                fields: Vec::new(),
                blocks: Vec::new(),
                extracted_aliases,
                page_image_rgb: None,
                content_sha256,
            });
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

        let doc = ParsedDocument {
            title,
            format: DocumentFormat::Pdf,
            mime_type: DocumentFormat::Pdf.default_mime().to_string(),
            content_sha256: sha256_hex_bytes(bytes),
            pages,
            full_markdown,
            extracted_aliases,
            ocr_policy: None,
            parser_engine: "kordoc".to_string(),
        };
        Some((doc, scan_pages))
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
        if cfg.pdf_engine == "kordoc" {
            if let Some((mut doc, scan_pages)) = self.try_kordoc_pdf(bytes, filename, cfg).await {
                if scan_pages.is_empty() {
                    return Ok(doc);
                }
                if let Some(client) = ocr {
                    match ctx.remote_ocr {
                        RemoteOcrConsent::Ask => {
                            return Err(OpenRagError::RemoteOcrConsentRequired {
                                pages_needing_ocr: scan_pages.len(),
                                total_pages: doc.pages.len(),
                            });
                        }
                        RemoteOcrConsent::Deny => {}
                        RemoteOcrConsent::Allow | RemoteOcrConsent::Default => {
                            ocr_pdf_pages(
                                &mut doc,
                                bytes,
                                filename,
                                &scan_pages,
                                cfg,
                                ctx,
                                Some(&self.share_quota),
                                client,
                            )
                            .await?;
                            // ocr_pdf_pages names the native engine; this text came from kordoc.
                            doc.parser_engine = "kordoc+schift_ocr".to_string();
                            return Ok(doc);
                        }
                    }
                }
                if scan_pages.len() == doc.pages.len() {
                    return Err(OpenRagError::Validation(
                        "kordoc OCR found no text in this PDF and Schift OCR is not configured: set OPENRAG_SCHIFT_OCR_URL and SCHIFT_OCR_API_KEY".into(),
                    ));
                }
                return Ok(doc);
            }
        }
        let (mut doc, scan_pages) = match parse_pdf_with_scan_pages(bytes, filename, cfg).await {
            Ok(v) => v,
            // The native parser cannot walk every PDF (e.g. compressed object streams).
            Err(OpenRagError::Validation(msg)) if msg.starts_with("corrupt PDF") => match ocr {
                Some(client) => match ctx.remote_ocr {
                    RemoteOcrConsent::Ask => {
                        return Err(OpenRagError::RemoteOcrConsentRequired {
                            pages_needing_ocr: 1,
                            total_pages: 1,
                        });
                    }
                    RemoteOcrConsent::Deny => return Err(OpenRagError::Validation(msg)),
                    RemoteOcrConsent::Allow | RemoteOcrConsent::Default => {
                        return ocr_whole_pdf(
                            bytes,
                            filename,
                            cfg,
                            ctx,
                            Some(&self.share_quota),
                            client,
                        )
                        .await;
                    }
                },
                None => return Err(OpenRagError::Validation(msg)),
            },
            Err(e) => return Err(e),
        };
        if scan_pages.is_empty() {
            return Ok(doc);
        }
        if let Some(client) = ocr {
            match ctx.remote_ocr {
                RemoteOcrConsent::Ask => {
                    return Err(OpenRagError::RemoteOcrConsentRequired {
                        pages_needing_ocr: scan_pages.len(),
                        total_pages: doc.pages.len(),
                    });
                }
                RemoteOcrConsent::Deny => {}
                RemoteOcrConsent::Allow | RemoteOcrConsent::Default => {
                    ocr_pdf_pages(
                        &mut doc,
                        bytes,
                        filename,
                        &scan_pages,
                        cfg,
                        ctx,
                        Some(&self.share_quota),
                        client,
                    )
                    .await?;
                    return Ok(doc);
                }
            }
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
    use std::path::PathBuf;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn write_fake_kordoc(dir: &Path, name: &str, body: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
        }
        Ok(path)
    }

    fn parser_with_kordoc(bin_path: PathBuf) -> DocumentParser {
        let mut parser = DocumentParser::new();
        parser.kordoc = Some(KordocCliConfig {
            bin_path: Some(bin_path),
            ..Default::default()
        });
        parser
    }

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

    #[tokio::test]
    async fn kordoc_pdf_parses_pages_keeps_empty_and_refuses_all_empty() -> TestResult {
        let tmp = tempfile::tempdir()?;
        let script = write_fake_kordoc(
            tmp.path(),
            "kordoc-two-pages.sh",
            r#"cat <<'EOF'
{"success":true,"fileType":"pdf","pageCount":2,"pages":[{"pageNumber":1,"markdown":"한국어 본문 (KO)\n<table><tr><th>항목</th></tr><tr><td>값</td></tr></table>"},{"pageNumber":2,"markdown":""}]}
EOF"#,
        )?;
        let parser = parser_with_kordoc(script);
        let pdf = crate::pdf::tests::pdf_with_pages(&[Some("Native"), None]);
        let cfg = crate::pdf::tests::test_parse_cfg();
        let doc = parser
            .parse(&pdf, Some("application/pdf"), Some("report.pdf"), &cfg, &ParsePolicyCtx::default())
            .await?;

        assert_eq!(doc.title, "report");
        assert_eq!(doc.parser_engine, "kordoc");
        assert_eq!(doc.pages.len(), 2);
        assert_eq!(doc.pages[0].page_no, 1);
        assert!(doc.pages[0].raw_markdown.contains("한국어 본문 (KO)"));
        assert!(doc.pages[0].raw_markdown.contains("<td>항목</td>"));
        assert_eq!(doc.pages[0].tables_html.len(), 1);
        assert_eq!(doc.pages[1].page_no, 2);
        assert!(doc.pages[1].raw_markdown.is_empty());

        let all_empty = write_fake_kordoc(
            tmp.path(),
            "kordoc-empty.sh",
            r#"printf '{"success":true,"fileType":"pdf","pageCount":2,"pages":[{"pageNumber":1,"markdown":""}]}'"#,
        )?;
        let empty_parser = parser_with_kordoc(all_empty);
        let err = empty_parser
            .parse(&pdf, Some("application/pdf"), Some("blank.pdf"), &cfg, &ParsePolicyCtx::default())
            .await;
        assert!(
            matches!(&err, Err(OpenRagError::Validation(m)) if m.contains("kordoc OCR found no text") && m.contains("Schift OCR is not configured")),
            "unexpected result: {err:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn kordoc_pdf_errors_fall_back_to_native() -> TestResult {
        let tmp = tempfile::tempdir()?;
        let pdf = crate::pdf::tests::pdf_with_pages(&[Some("Native fallback text")]);
        let cfg = crate::pdf::tests::test_parse_cfg();

        for (name, body) in [
            ("exit1.sh", "exit 1"),
            ("garbage.sh", "printf 'not-json-at-all'"),
            ("success_false.sh", r#"printf '{"success":false,"error":"parse failed"}'"#),
        ] {
            let script = write_fake_kordoc(tmp.path(), name, body)?;
            let parser = parser_with_kordoc(script);
            let doc = parser
                .parse(&pdf, Some("application/pdf"), Some("fallback.pdf"), &cfg, &ParsePolicyCtx::default())
                .await?;
            assert_eq!(doc.parser_engine, "pdf_native", "case {name}");
            assert!(doc.full_markdown.contains("Native fallback text"), "case {name}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn pdf_engine_native_never_calls_kordoc() -> TestResult {
        let tmp = tempfile::tempdir()?;
        let marker = tmp.path().join("kordoc-was-called");
        let script = write_fake_kordoc(
            tmp.path(),
            "must-not-run.sh",
            &format!("touch '{}'\nexit 1", marker.display()),
        )?;
        let parser = parser_with_kordoc(script);
        let mut cfg = crate::pdf::tests::test_parse_cfg();
        cfg.pdf_engine = "native".into();
        let pdf = crate::pdf::tests::pdf_with_pages(&[Some("Native only")]);
        let doc = parser
            .parse(&pdf, Some("application/pdf"), Some("native.pdf"), &cfg, &ParsePolicyCtx::default())
            .await?;
        assert!(!marker.exists(), "kordoc script must not run when pdf_engine=native");
        assert_eq!(doc.parser_engine, "pdf_native");
        assert!(doc.full_markdown.contains("Native only"));
        Ok(())
    }

    #[tokio::test]
    async fn kordoc_ocr_flag_passed_only_when_enabled() -> TestResult {
        let tmp = tempfile::tempdir()?;
        let script = write_fake_kordoc(
            tmp.path(),
            "echo-args.sh",
            r#"printf '{"success":true,"fileType":"pdf","pageCount":1,"pages":[{"pageNumber":1,"markdown":"ARGS:%s"}]}' "$*""#,
        )?;
        let parser = parser_with_kordoc(script);
        let pdf = crate::pdf::tests::pdf_with_pages(&[Some("Hello")]);

        let mut cfg = crate::pdf::tests::test_parse_cfg();
        cfg.kordoc_ocr = true;
        let with_ocr = parser
            .parse(&pdf, Some("application/pdf"), Some("doc.pdf"), &cfg, &ParsePolicyCtx::default())
            .await?;
        assert_eq!(with_ocr.parser_engine, "kordoc");
        assert!(with_ocr.full_markdown.contains("--ocr"), "{}", with_ocr.full_markdown);
        assert!(with_ocr.full_markdown.contains("--format json"), "{}", with_ocr.full_markdown);
        assert!(with_ocr.full_markdown.contains("--plain"), "{}", with_ocr.full_markdown);
        assert!(with_ocr.full_markdown.contains("--no-images"), "{}", with_ocr.full_markdown);
        assert!(with_ocr.full_markdown.contains("--silent"), "{}", with_ocr.full_markdown);

        cfg.kordoc_ocr = false;
        let without_ocr = parser
            .parse(&pdf, Some("application/pdf"), Some("doc.pdf"), &cfg, &ParsePolicyCtx::default())
            .await?;
        assert_eq!(without_ocr.parser_engine, "kordoc");
        assert!(!without_ocr.full_markdown.contains("--ocr"), "{}", without_ocr.full_markdown);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires a kordoc binary (OPENRAG_KORDOC_BIN) and a PDF (OPENRAG_KORDOC_SAMPLE_PDF)"]
    async fn real_kordoc_parses_sample_hwp_export_pdf() -> TestResult {
        let sample = std::env::var_os("OPENRAG_KORDOC_SAMPLE_PDF")
            .ok_or("set OPENRAG_KORDOC_SAMPLE_PDF to a PDF file")?;
        let bytes = std::fs::read(sample)?;
        let parser = DocumentParser::new();
        let cfg = crate::pdf::tests::test_parse_cfg();
        let doc = parser
            .parse(&bytes, Some("application/pdf"), Some("hwp_export.pdf"), &cfg, &ParsePolicyCtx::default())
            .await?;
        assert_eq!(doc.parser_engine, "kordoc");
        assert!(!doc.pages.is_empty());
        println!("page_count={}", doc.pages.len());
        for page in &doc.pages {
            println!("page {} chars={}", page.page_no, page.raw_markdown.chars().count());
        }
        Ok(())
    }

    fn make_fake_ocr_hook(
        response: SchiftOcrResponse,
    ) -> (std::sync::Arc<std::sync::atomic::AtomicUsize>, crate::ocr::FakeOcrHook) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let hook = Arc::new(move |_: &[u8], _: &str, _: openrag_core::Sensitivity, _: bool, _: OcrTask| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            Ok(response.clone())
        });
        (calls, hook)
    }

    fn tiny_png_bytes() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([240, 240, 240]));
        let mut cursor = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img).write_to(&mut cursor, image::ImageFormat::Png)?;
        Ok(cursor.into_inner())
    }

    #[tokio::test]
    async fn remote_ocr_consent_native_pdf_branch() -> TestResult {
        use std::sync::atomic::Ordering;

        let ocr_resp: SchiftOcrResponse = serde_json::from_value(serde_json::json!({
            "pages": [
                {"page_no": 1, "result": {"markdown": "OCR page 1"}},
                {"page_no": 2, "result": {"markdown": "OCR page 2"}}
            ]
        }))?;
        let (calls, hook) = make_fake_ocr_hook(ocr_resp);

        crate::ocr::with_fake_ocr_hook(hook, async move {
            let mut parser = DocumentParser::new();
            parser.ocr_client = Some(SchiftOcrClient::new("http://ocr.test.invalid", None)?);

            let mut cfg = crate::pdf::tests::test_parse_cfg();
            cfg.pdf_engine = "native".into();

            // 2-page mixed PDF: page 1 has text, page 2 is scanned (empty).
            let mixed = crate::pdf::tests::pdf_with_pages(&[Some("Native page one"), None]);
            let all_empty = crate::pdf::tests::pdf_with_pages(&[None]);
            let text_only = crate::pdf::tests::pdf_with_pages(&[Some("Already readable")]);

            // Text-only PDF with Ask never needs remote OCR -> succeeds without asking or calling OCR.
            let ok_text = parser
                .parse(
                    &text_only,
                    Some("application/pdf"),
                    Some("readable.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Ask,
                        ..Default::default()
                    },
                )
                .await?;
            assert_eq!(ok_text.pages.len(), 1);
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Mixed PDF with Ask -> consent error (1 of 2), calls nothing.
            let ask_err = parser
                .parse(
                    &mixed,
                    Some("application/pdf"),
                    Some("mixed.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Ask,
                        ..Default::default()
                    },
                )
                .await;
            assert_eq!(
                ask_err,
                Err(OpenRagError::RemoteOcrConsentRequired {
                    pages_needing_ocr: 1,
                    total_pages: 2,
                })
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Mixed PDF with Deny -> keeps page 2 empty, calls nothing.
            let deny_mixed = parser
                .parse(
                    &mixed,
                    Some("application/pdf"),
                    Some("mixed.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Deny,
                        ..Default::default()
                    },
                )
                .await?;
            assert_eq!(deny_mixed.pages.len(), 2);
            assert!(deny_mixed.pages[0].raw_markdown.contains("Native page one"));
            assert!(deny_mixed.pages[1].raw_markdown.is_empty());
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // All-empty PDF with Deny -> today's refusal error, calls nothing.
            let deny_empty = parser
                .parse(
                    &all_empty,
                    Some("application/pdf"),
                    Some("scan.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Deny,
                        ..Default::default()
                    },
                )
                .await;
            assert!(
                matches!(&deny_empty, Err(OpenRagError::Validation(m)) if m.contains("no text layer")),
                "{deny_empty:?}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Default and Allow -> call OCR as before.
            for (expected_calls, consent) in [(1, RemoteOcrConsent::Default), (2, RemoteOcrConsent::Allow)]
            {
                let doc = parser
                    .parse(
                        &mixed,
                        Some("application/pdf"),
                        Some("mixed.pdf"),
                        &cfg,
                        &ParsePolicyCtx {
                            remote_ocr: consent,
                            ..Default::default()
                        },
                    )
                    .await?;
                assert_eq!(doc.parser_engine, "pdf_native+schift_ocr");
                assert!(doc.pages[0].raw_markdown.contains("Native page one"));
                assert_eq!(doc.pages[1].raw_markdown, "OCR page 2");
                assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            }
            Ok(())
        })
        .await
    }

    #[tokio::test]
    async fn remote_ocr_consent_kordoc_branch() -> TestResult {
        use std::sync::atomic::Ordering;

        let tmp = tempfile::tempdir()?;
        let ocr_resp: SchiftOcrResponse = serde_json::from_value(serde_json::json!({
            "pages": [
                {"page_no": 1, "result": {"markdown": "OCR k1"}},
                {"page_no": 2, "result": {"markdown": "OCR k2"}}
            ]
        }))?;
        let (calls, hook) = make_fake_ocr_hook(ocr_resp);

        let mixed_script = write_fake_kordoc(
            tmp.path(),
            "kordoc-mixed.sh",
            r#"printf '{"success":true,"fileType":"pdf","pageCount":2,"pages":[{"pageNumber":1,"markdown":"Kordoc page 1"},{"pageNumber":2,"markdown":""}]}'"#,
        )?;
        let empty_script = write_fake_kordoc(
            tmp.path(),
            "kordoc-all-empty.sh",
            r#"printf '{"success":true,"fileType":"pdf","pageCount":2,"pages":[{"pageNumber":1,"markdown":""},{"pageNumber":2,"markdown":""}]}'"#,
        )?;

        crate::ocr::with_fake_ocr_hook(hook, async move {
            let pdf = crate::pdf::tests::pdf_with_pages(&[Some("Native"), None]);
            let cfg = crate::pdf::tests::test_parse_cfg();

            let mut parser = parser_with_kordoc(mixed_script);
            parser.ocr_client = Some(SchiftOcrClient::new("http://ocr.test.invalid", None)?);

            // Ask -> 1 of 2 pages, no OCR call.
            let ask_err = parser
                .parse(
                    &pdf,
                    Some("application/pdf"),
                    Some("kdoc.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Ask,
                        ..Default::default()
                    },
                )
                .await;
            assert_eq!(
                ask_err,
                Err(OpenRagError::RemoteOcrConsentRequired {
                    pages_needing_ocr: 1,
                    total_pages: 2,
                })
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Deny on mixed -> keeps page 2 empty, no OCR call.
            let deny_doc = parser
                .parse(
                    &pdf,
                    Some("application/pdf"),
                    Some("kdoc.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Deny,
                        ..Default::default()
                    },
                )
                .await?;
            assert_eq!(deny_doc.parser_engine, "kordoc");
            assert_eq!(deny_doc.pages[0].raw_markdown, "Kordoc page 1");
            assert!(deny_doc.pages[1].raw_markdown.is_empty());
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Deny on all-empty -> refuses with today's error, no OCR call.
            let mut empty_parser = parser_with_kordoc(empty_script);
            empty_parser.ocr_client = Some(SchiftOcrClient::new("http://ocr.test.invalid", None)?);
            let deny_empty = empty_parser
                .parse(
                    &pdf,
                    Some("application/pdf"),
                    Some("blank.pdf"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Deny,
                        ..Default::default()
                    },
                )
                .await;
            assert!(
                matches!(&deny_empty, Err(OpenRagError::Validation(m)) if m.contains("kordoc OCR found no text")),
                "{deny_empty:?}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Default and Allow -> call OCR and produce kordoc+schift_ocr.
            for (expected_calls, consent) in [(1, RemoteOcrConsent::Default), (2, RemoteOcrConsent::Allow)]
            {
                let doc = parser
                    .parse(
                        &pdf,
                        Some("application/pdf"),
                        Some("kdoc.pdf"),
                        &cfg,
                        &ParsePolicyCtx {
                            remote_ocr: consent,
                            ..Default::default()
                        },
                    )
                    .await?;
                assert_eq!(doc.parser_engine, "kordoc+schift_ocr");
                assert_eq!(doc.pages[0].raw_markdown, "Kordoc page 1");
                assert_eq!(doc.pages[1].raw_markdown, "OCR k2");
                assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            }
            Ok(())
        })
        .await
    }

    #[tokio::test]
    async fn remote_ocr_consent_image_branch() -> TestResult {
        use std::sync::atomic::Ordering;

        let ocr_resp: SchiftOcrResponse = serde_json::from_value(serde_json::json!({
            "result": {"markdown": "OCR image text"}
        }))?;
        let (calls, hook) = make_fake_ocr_hook(ocr_resp);

        crate::ocr::with_fake_ocr_hook(hook, async move {
            let mut parser = DocumentParser::new();
            parser.ocr_client = Some(SchiftOcrClient::new("http://ocr.test.invalid", None)?);
            let cfg = crate::pdf::tests::test_parse_cfg();
            let png = tiny_png_bytes()?;

            // Ask -> 1 of 1 pages, calls nothing, consumes no share quota.
            let ask_err = parser
                .parse(
                    &png,
                    Some("image/png"),
                    Some("scan.png"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Ask,
                        tenant_key: Some("t-ask".into()),
                        ..Default::default()
                    },
                )
                .await;
            assert_eq!(
                ask_err,
                Err(OpenRagError::RemoteOcrConsentRequired {
                    pages_needing_ocr: 1,
                    total_pages: 1,
                })
            );
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(parser.share_quota.used_today("t-ask"), 0);

            // Deny -> offline fallback, calls nothing.
            let deny_doc = parser
                .parse(
                    &png,
                    Some("image/png"),
                    Some("scan.png"),
                    &cfg,
                    &ParsePolicyCtx {
                        remote_ocr: RemoteOcrConsent::Deny,
                        ..Default::default()
                    },
                )
                .await?;
            assert_eq!(deny_doc.parser_engine, "offline_ocr");
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            // Default and Allow -> call OCR.
            for (expected_calls, consent) in [(1, RemoteOcrConsent::Default), (2, RemoteOcrConsent::Allow)]
            {
                let doc = parser
                    .parse(
                        &png,
                        Some("image/png"),
                        Some("scan.png"),
                        &cfg,
                        &ParsePolicyCtx {
                            remote_ocr: consent,
                            ..Default::default()
                        },
                    )
                    .await?;
                assert_eq!(doc.parser_engine, "schift_ocr");
                assert_eq!(doc.full_markdown, "OCR image text");
                assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
            }
            Ok(())
        })
        .await
    }
}
