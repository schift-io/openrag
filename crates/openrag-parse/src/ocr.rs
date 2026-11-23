use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use chrono::{DateTime, FixedOffset, NaiveDate, Utc};
use image::{GenericImageView, Rgb, RgbImage};
use openrag_core::{BBoxNorm1000, OpenRagError, ParseConfig, Sensitivity};
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::aliases::extract_parenthetical_aliases;
use crate::model::{
    sha256_hex_bytes, sha256_hex_str, DocumentFormat, OcrBlock, OcrExecutionTarget,
    OcrPolicyDecision, OcrTask, ParsePolicyCtx, ParsedDocument, ParsedPage,
};

pub const DEFAULT_SHARE_DAILY_LIMIT: u32 = 50;

fn ocr_mode_rank(mode: Sensitivity) -> u8 {
    match mode {
        Sensitivity::Share | Sensitivity::Normal => 1,
        Sensitivity::Private | Sensitivity::Confidential | Sensitivity::Secret => 2,
        Sensitivity::Zdr => 3,
    }
}

fn normalize_ocr_mode(s: Sensitivity) -> Sensitivity {
    match s {
        Sensitivity::Share | Sensitivity::Normal => Sensitivity::Share,
        Sensitivity::Private | Sensitivity::Confidential | Sensitivity::Secret => {
            Sensitivity::Private
        }
        Sensitivity::Zdr => Sensitivity::Zdr,
    }
}

fn max_ocr_mode(a: Sensitivity, b: Sensitivity) -> Sensitivity {
    let na = normalize_ocr_mode(a);
    let nb = normalize_ocr_mode(b);
    if ocr_mode_rank(na) >= ocr_mode_rank(nb) {
        na
    } else {
        nb
    }
}

/// Evaluates the OCR privacy mode (`share` | `private` | `zdr`) and execution target
/// (`SchiftCloud` | `LocalVllm` | `OfflineFallback`).
///
/// Guarantees:
/// - Never allows `share` when `auth_level` requires `private` / `zdr`, when `sensitivity` is
///   `Private`/`Confidential`/`Secret`/`Zdr`, or when `cfg.ocr_engine == "local_vllm"`.
/// - Rejects `zdr` + `batch=true` with `OpenRagError::Validation("zdr_batch_unsupported: ...")`.
pub fn resolve_ocr_policy(
    cfg: &ParseConfig,
    ctx: &ParsePolicyCtx,
) -> Result<OcrPolicyDecision, OpenRagError> {
    let has_remote_url = std::env::var("OPENRAG_SCHIFT_OCR_URL")
        .ok()
        .is_some_and(|v| !v.trim().is_empty());
    resolve_ocr_policy_with_target(cfg, ctx, has_remote_url)
}

pub fn resolve_ocr_policy_with_target(
    cfg: &ParseConfig,
    ctx: &ParsePolicyCtx,
    has_remote_ocr_endpoint: bool,
) -> Result<OcrPolicyDecision, OpenRagError> {
    let mut required_by_level = normalize_ocr_mode(cfg.default_ocr_mode);
    let mut best_rule_level = 0u8;
    for rule in &cfg.ocr_mode_by_auth_level {
        if ctx.auth_level >= rule.min_auth_level && rule.min_auth_level.as_u8() >= best_rule_level
        {
            best_rule_level = rule.min_auth_level.as_u8();
            required_by_level = normalize_ocr_mode(rule.required_mode);
        }
    }

    let sensitivity_floor = match ctx.sensitivity {
        Sensitivity::Share | Sensitivity::Normal => Sensitivity::Share,
        Sensitivity::Private | Sensitivity::Confidential | Sensitivity::Secret => {
            Sensitivity::Private
        }
        Sensitivity::Zdr => Sensitivity::Zdr,
    };

    let requested = ctx
        .requested_ocr_mode
        .map(normalize_ocr_mode)
        .unwrap_or_else(|| normalize_ocr_mode(cfg.default_ocr_mode));

    let mut effective_mode = max_ocr_mode(requested, max_ocr_mode(required_by_level, sensitivity_floor));

    // Airgap / local_vllm never transmits to free external `share` tier.
    if cfg.ocr_engine == "local_vllm" && effective_mode == Sensitivity::Share {
        effective_mode = Sensitivity::Private;
    }

    if cfg.reject_zdr_with_batch
        && ctx.batch
        && (effective_mode == Sensitivity::Zdr
            || ctx.requested_ocr_mode == Some(Sensitivity::Zdr))
    {
        return Err(OpenRagError::Validation(
            "zdr_batch_unsupported: ZDR mode cannot be used with batch=true; use_mode=private"
                .into(),
        ));
    }

    let upgraded_from_share =
        ctx.requested_ocr_mode == Some(Sensitivity::Share) && effective_mode != Sensitivity::Share;

    // A configured endpoint speaks Schift's /v1/run API: Schift's service, or an on-prem Schift
    // (or compatible) server in the air-gapped profile, where local_vllm only rules out `share`.
    let execution_target = if has_remote_ocr_endpoint
        && (cfg.ocr_engine == "schift_ocr" || cfg.ocr_engine == "local_vllm")
    {
        OcrExecutionTarget::SchiftCloud
    } else if cfg.ocr_engine == "local_vllm" {
        OcrExecutionTarget::LocalVllm
    } else {
        OcrExecutionTarget::OfflineFallback
    };

    Ok(OcrPolicyDecision {
        effective_mode,
        execution_target,
        upgraded_from_share,
    })
}

/// Thread-safe KST (UTC+09:00) daily quota tracker enforcing the 50 pages/day free `share` limit.
#[derive(Debug)]
pub struct ShareQuotaTracker {
    daily_limit: u32,
    counts: Mutex<HashMap<(String, NaiveDate), u32>>,
}

impl Default for ShareQuotaTracker {
    fn default() -> Self {
        Self::new(DEFAULT_SHARE_DAILY_LIMIT)
    }
}

impl ShareQuotaTracker {
    pub fn new(daily_limit: u32) -> Self {
        Self {
            daily_limit,
            counts: Mutex::new(HashMap::new()),
        }
    }

    pub fn kst_date(now: DateTime<Utc>) -> NaiveDate {
        match FixedOffset::east_opt(9 * 3600) {
            Some(kst) => now.with_timezone(&kst).date_naive(),
            None => now.date_naive(),
        }
    }

    pub fn try_consume(&self, tenant_key: &str, pages: u32) -> Result<u32, OpenRagError> {
        self.try_consume_at(tenant_key, pages, Utc::now())
    }

    pub fn try_consume_at(
        &self,
        tenant_key: &str,
        pages: u32,
        now: DateTime<Utc>,
    ) -> Result<u32, OpenRagError> {
        let date = Self::kst_date(now);
        let mut guard = self.counts.lock().map_err(|_| {
            OpenRagError::Internal("share quota tracker mutex poisoned".into())
        })?;
        let key = (tenant_key.to_string(), date);
        let used = guard.get(&key).copied().unwrap_or(0);
        let next = used.saturating_add(pages);
        if next > self.daily_limit {
            return Err(OpenRagError::Validation(format!(
                "share_quota_exceeded: daily free share limit of {} pages/KST day reached (used={used}, requested={pages})",
                self.daily_limit
            )));
        }
        guard.insert(key, next);
        Ok(self.daily_limit.saturating_sub(next))
    }

    pub fn used_today(&self, tenant_key: &str) -> u32 {
        let date = Self::kst_date(Utc::now());
        self.counts
            .lock()
            .ok()
            .and_then(|g| g.get(&(tenant_key.to_string(), date)).copied())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchiftOcrImageItem {
    pub image: String,
    #[serde(default)]
    pub image_base64: Option<String>,
    #[serde(default)]
    pub image_url: Option<String>,
    #[serde(default)]
    pub bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub caption: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchiftOcrBlockItem {
    #[serde(default = "default_block_kind")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub bbox: Option<[i32; 4]>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

fn default_block_kind() -> String {
    "text".to_string()
}

/// One page's model output. Schift's `/v1/run` sends it under `result`
/// (`{"result": {"markdown": "…", "images": […]}, "job_id": …, "usage": …}`, seen on schift.io
/// 2026-11-23, model schift-ocr-2-20261112). The same fields at the top level are still read.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchiftOcrResult {
    #[serde(default)]
    pub markdown: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub blocks: Vec<SchiftOcrBlockItem>,
    #[serde(default)]
    pub images: Vec<SchiftOcrImageItem>,
}

impl SchiftOcrResult {
    fn page_text(&self) -> Option<&str> {
        self.markdown.as_deref().or(self.text.as_deref())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchiftOcrPageItem {
    pub page_no: Option<i32>,
    #[serde(default)]
    pub markdown: String,
    #[serde(default)]
    pub blocks: Vec<SchiftOcrBlockItem>,
    #[serde(default)]
    pub images: Vec<SchiftOcrImageItem>,
    #[serde(default)]
    pub result: Option<SchiftOcrResult>,
}

impl SchiftOcrPageItem {
    /// `result.markdown` (or `result.text`), else the top-level `markdown`.
    pub fn page_markdown(&self) -> &str {
        self.result
            .as_ref()
            .and_then(SchiftOcrResult::page_text)
            .unwrap_or(&self.markdown)
    }

    pub fn page_blocks(&self) -> &[SchiftOcrBlockItem] {
        match &self.result {
            Some(r) if !r.blocks.is_empty() => &r.blocks,
            _ => &self.blocks,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchiftOcrResponse {
    #[serde(default)]
    pub markdown: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub blocks: Vec<SchiftOcrBlockItem>,
    #[serde(default)]
    pub pages: Vec<SchiftOcrPageItem>,
    #[serde(default)]
    pub images: Vec<SchiftOcrImageItem>,
    #[serde(default)]
    pub result: Option<SchiftOcrResult>,
}

impl SchiftOcrResponse {
    /// Text of a single-page response: `result.markdown` / `result.text`, else the same fields
    /// at the top level.
    pub fn page_markdown(&self) -> Option<&str> {
        self.result
            .as_ref()
            .and_then(SchiftOcrResult::page_text)
            .or(self.markdown.as_deref())
            .or(self.text.as_deref())
    }

    pub fn page_blocks(&self) -> &[SchiftOcrBlockItem] {
        match &self.result {
            Some(r) if !r.blocks.is_empty() => &r.blocks,
            _ => &self.blocks,
        }
    }

    /// Whether the response carries page content in a shape this client reads.
    fn has_content(&self) -> bool {
        self.page_markdown().is_some()
            || !self.page_blocks().is_empty()
            || self
                .pages
                .iter()
                .any(|p| p.result.is_some() || !p.markdown.is_empty() || !p.blocks.is_empty())
    }
}

/// Decode a `/v1/run` body. A 200 whose page text sits under no field this client reads is an
/// error: the document would otherwise be indexed with an empty body.
fn decode_ocr_body(body: serde_json::Value) -> Result<SchiftOcrResponse, OpenRagError> {
    let keys = body
        .as_object()
        .map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "))
        .unwrap_or_default();
    let resp: SchiftOcrResponse =
        serde_json::from_value(body).map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "schift_ocr".into(),
            message: format!("failed to parse OCR JSON response: {e}"),
        })?;
    if !resp.has_content() {
        return Err(OpenRagError::ExternalAdapter {
            adapter: "schift_ocr".into(),
            message: format!(
                "OCR response has no page text (expected result.markdown); keys: [{keys}]"
            ),
        });
    }
    Ok(resp)
}

#[derive(Debug, Clone)]
pub struct SchiftOcrClient {
    endpoint_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl SchiftOcrClient {
    pub fn new(
        endpoint_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self, OpenRagError> {
        let endpoint = endpoint_url.into();
        if endpoint.trim().is_empty() {
            return Err(OpenRagError::Config(
                "Schift OCR endpoint_url must not be empty".into(),
            ));
        }
        // Whole PDFs go in one request, so allow for a long document.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| {
                OpenRagError::Internal(format!("failed to construct Schift OCR HTTP client: {e}"))
            })?;
        Ok(Self {
            endpoint_url: endpoint.trim_end_matches('/').to_string(),
            api_key,
            client,
        })
    }

    pub async fn run_ocr(
        &self,
        image_bytes: &[u8],
        filename: &str,
        mode: Sensitivity,
        batch: bool,
        task: OcrTask,
    ) -> Result<SchiftOcrResponse, OpenRagError> {
        if batch && normalize_ocr_mode(mode) == Sensitivity::Zdr {
            return Err(OpenRagError::Validation(
                "zdr_batch_unsupported: ZDR mode cannot be used with batch=true; use_mode=private"
                    .into(),
            ));
        }
        if image_bytes.is_empty() {
            return Err(OpenRagError::Validation(
                "cannot submit empty image bytes to OCR".into(),
            ));
        }

        let url = format!("{}/v1/run", self.endpoint_url);
        let mime = ocr_upload_mime(filename);
        // 429 (too many concurrent jobs) is waited out: up to 8 retries, each wait retry_after_sec or
        // the backoff (1, 2, 4, 8, 16 s), capped at 20 s; about 80 s in all without hints, 160 s at most.
        let mut attempt = 0u32;
        let resp = loop {
            let file_part = multipart::Part::bytes(image_bytes.to_vec())
                .file_name(filename.to_string())
                .mime_str(mime)
                .map_err(|e| OpenRagError::Internal(format!("OCR upload mime: {e}")))?;
            let form = multipart::Form::new()
                .part("image", file_part)
                .text("task", task.as_str().to_string())
                .text("mode", normalize_ocr_mode(mode).as_str().to_string())
                .text("batch", if batch { "true" } else { "false" }.to_string());

            let mut req = self.client.post(&url).multipart(form);
            if let Some(key) = self.api_key.as_deref() {
                req = req.bearer_auth(key);
            }

            let resp = req.send().await.map_err(|e| OpenRagError::ExternalAdapter {
                adapter: "schift_ocr".into(),
                message: format!("HTTP request failed: {e}"),
            })?;
            if resp.status().as_u16() != 429 || attempt >= OCR_BUSY_RETRIES {
                break resp;
            }
            let body = resp.text().await.unwrap_or_default();
            let hinted = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("retry_after_sec").and_then(serde_json::Value::as_u64))
                .unwrap_or(1);
            let wait = hinted.max(1u64 << attempt.min(4)).min(20);
            tokio::time::sleep(Duration::from_secs(wait)).await;
            attempt += 1;
        };

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 400 && body.contains("zdr_batch_unsupported") {
                return Err(OpenRagError::Validation(format!(
                    "zdr_batch_unsupported: {body}"
                )));
            }
            return Err(OpenRagError::ExternalAdapter {
                adapter: "schift_ocr".into(),
                message: format!("HTTP {status}: {body}"),
            });
        }

        let body = resp.json::<serde_json::Value>().await;
        let body = body.map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "schift_ocr".into(),
            message: format!("failed to parse OCR JSON response: {e}"),
        })?;
        decode_ocr_body(body)
    }
}

/// How many times a 429 from the OCR endpoint is retried (waits of 1, 2, 4, 8, 16, 20, 20, 20 s).
const OCR_BUSY_RETRIES: u32 = 8;

fn ocr_upload_mime(filename: &str) -> &'static str {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("pdf") => "application/pdf",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

/// OCR text per page index (0-based) of a `/v1/run` reply: `pages[]` for a multi-page PDF,
/// otherwise the single `result`.
fn ocr_pages_by_index(
    resp: &SchiftOcrResponse,
) -> Result<HashMap<usize, (String, Vec<OcrBlock>)>, OpenRagError> {
    let mut out = HashMap::new();
    if resp.pages.is_empty() {
        let md = resp.page_markdown().unwrap_or_default().to_string();
        out.insert(0, (md, convert_ocr_blocks(resp.page_blocks())?));
    } else {
        for (idx, p) in resp.pages.iter().enumerate() {
            let at = p
                .page_no
                .and_then(|n| usize::try_from(n).ok())
                .and_then(|n| n.checked_sub(1))
                .unwrap_or(idx);
            out.insert(at, (p.page_markdown().to_string(), convert_ocr_blocks(p.page_blocks())?));
        }
    }
    Ok(out)
}

/// Sends a PDF to the OCR endpoint once and fills the listed pages (0-based) with its text.
/// Used for scanned pages, which have no text layer; pages that have one keep it.
#[allow(clippy::too_many_arguments)]
pub async fn ocr_pdf_pages(
    doc: &mut ParsedDocument,
    bytes: &[u8],
    filename: Option<&str>,
    pages: &[usize],
    cfg: &ParseConfig,
    ctx: &ParsePolicyCtx,
    quota: Option<&ShareQuotaTracker>,
    client: &SchiftOcrClient,
) -> Result<(), OpenRagError> {
    let policy = resolve_ocr_policy_with_target(cfg, ctx, true)?;
    if policy.effective_mode == Sensitivity::Share {
        if let Some(q) = quota {
            let tenant = ctx.tenant_key.as_deref().unwrap_or("default");
            q.try_consume(tenant, u32::try_from(doc.pages.len()).unwrap_or(u32::MAX))?;
        }
    }
    let fname = filename.unwrap_or("document.pdf");
    let resp = client
        .run_ocr(bytes, fname, policy.effective_mode, ctx.batch, ctx.ocr_task)
        .await?;
    let by_index = ocr_pages_by_index(&resp)?;
    for &i in pages {
        let (Some(page), Some((md, blocks))) = (doc.pages.get_mut(i), by_index.get(&i)) else {
            continue;
        };
        page.raw_markdown = md.clone();
        page.blocks = blocks.clone();
        page.extracted_aliases = extract_parenthetical_aliases(md);
        page.content_sha256 = sha256_hex_str(md);
    }
    doc.full_markdown = doc
        .pages
        .iter()
        .map(|p| p.raw_markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    doc.extracted_aliases = extract_parenthetical_aliases(&doc.full_markdown);
    doc.ocr_policy = Some(policy);
    doc.parser_engine = "pdf_native+schift_ocr".to_string();
    Ok(())
}

/// Reads a whole PDF with OCR, for PDFs whose structure the native parser cannot walk.
pub async fn ocr_whole_pdf(
    bytes: &[u8],
    filename: Option<&str>,
    cfg: &ParseConfig,
    ctx: &ParsePolicyCtx,
    quota: Option<&ShareQuotaTracker>,
    client: &SchiftOcrClient,
) -> Result<ParsedDocument, OpenRagError> {
    let policy = resolve_ocr_policy_with_target(cfg, ctx, true)?;
    let fname = filename.unwrap_or("document.pdf");
    let resp = client
        .run_ocr(bytes, fname, policy.effective_mode, ctx.batch, ctx.ocr_task)
        .await?;
    let by_index = ocr_pages_by_index(&resp)?;
    if policy.effective_mode == Sensitivity::Share {
        if let Some(q) = quota {
            let tenant = ctx.tenant_key.as_deref().unwrap_or("default");
            q.try_consume(tenant, u32::try_from(by_index.len()).unwrap_or(u32::MAX))?;
        }
    }
    let mut idx: Vec<usize> = by_index.keys().copied().collect();
    idx.sort_unstable();
    let pages: Vec<ParsedPage> = idx
        .iter()
        .filter_map(|i| by_index.get(i).map(|v| (*i, v)))
        .map(|(i, (md, blocks))| ParsedPage {
            page_no: i32::try_from(i + 1).unwrap_or(i32::MAX),
            raw_markdown: md.clone(),
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks: blocks.clone(),
            extracted_aliases: extract_parenthetical_aliases(md),
            page_image_rgb: None,
            content_sha256: sha256_hex_str(md),
        })
        .collect();
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
    Ok(ParsedDocument {
        title,
        format: DocumentFormat::Pdf,
        mime_type: DocumentFormat::Pdf.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages,
        full_markdown,
        extracted_aliases,
        ocr_policy: Some(policy),
        parser_engine: "schift_ocr".to_string(),
    })
}

/// Deterministic skew estimator (`-5..=+5` degrees) + bilinear rotation when `dewarp_enabled` is true.
pub fn dewarp_image(img: &RgbImage, cfg: &ParseConfig) -> RgbImage {
    if !cfg.dewarp_enabled || img.width() < 16 || img.height() < 16 {
        return img.clone();
    }

    let (w, h) = img.dimensions();
    // Estimate skew angle on a small proxy grid for fast deterministic execution.
    let step_x = (w / 128).max(1);
    let step_y = (h / 128).max(1);
    let mut best_angle = 0.0f32;
    let mut best_var = f64::NEG_INFINITY;

    let mut deg = -5.0f32;
    while deg <= 5.001 {
        let rad = deg.to_radians();
        let sin_a = rad.sin();
        let mut row_sums: HashMap<i32, u64> = HashMap::new();
        for y in (0..h).step_by(step_y as usize) {
            for x in (0..w).step_by(step_x as usize) {
                let px = img.get_pixel(x, y);
                let lum = (0.299 * (px[0] as f32) + 0.587 * (px[1] as f32) + 0.114 * (px[2] as f32))
                    as u64;
                let darkness = 255u64.saturating_sub(lum);
                let proj_y = ((y as f32) - (x as f32) * sin_a).round() as i32;
                *row_sums.entry(proj_y).or_insert(0) += darkness;
            }
        }
        let n = (row_sums.len().max(1)) as f64;
        let mean = row_sums.values().sum::<u64>() as f64 / n;
        let var = row_sums
            .values()
            .map(|&v| {
                let d = (v as f64) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        if var > best_var {
            best_var = var;
            best_angle = deg;
        }
        deg += 1.0;
    }

    if best_angle.abs() < 0.5 {
        return img.clone();
    }

    rotate_rgb_bilinear(img, -best_angle.to_radians())
}

fn rotate_rgb_bilinear(img: &RgbImage, rad: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    let cx = (w as f32) / 2.0;
    let cy = (h as f32) / 2.0;
    let cos_a = rad.cos();
    let sin_a = rad.sin();
    let mut out = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));

    for y in 0..h {
        let dy = (y as f32) - cy;
        for x in 0..w {
            let dx = (x as f32) - cx;
            let sx = cos_a * dx + sin_a * dy + cx;
            let sy = -sin_a * dx + cos_a * dy + cy;
            if sx >= 0.0 && sx < (w - 1) as f32 && sy >= 0.0 && sy < (h - 1) as f32 {
                let x0 = sx.floor() as u32;
                let y0 = sy.floor() as u32;
                let fx = sx - (x0 as f32);
                let fy = sy - (y0 as f32);

                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x0 + 1, y0);
                let p01 = img.get_pixel(x0, y0 + 1);
                let p11 = img.get_pixel(x0 + 1, y0 + 1);

                let mut rgb = [0u8; 3];
                for c in 0..3 {
                    let v0 = (p00[c] as f32) * (1.0 - fx) + (p10[c] as f32) * fx;
                    let v1 = (p01[c] as f32) * (1.0 - fx) + (p11[c] as f32) * fx;
                    rgb[c] = (v0 * (1.0 - fy) + v1 * fy).round().clamp(0.0, 255.0) as u8;
                }
                out.put_pixel(x, y, Rgb(rgb));
            }
        }
    }
    out
}

pub async fn parse_image_with_ocr(
    bytes: &[u8],
    format: DocumentFormat,
    filename: Option<&str>,
    cfg: &ParseConfig,
    ctx: &ParsePolicyCtx,
    quota: Option<&ShareQuotaTracker>,
    remote_client: Option<&SchiftOcrClient>,
) -> Result<ParsedDocument, OpenRagError> {
    if bytes.is_empty() {
        return Err(OpenRagError::Validation(
            "empty image payload cannot be parsed".into(),
        ));
    }

    let dyn_img = image::load_from_memory(bytes)
        .map_err(|e| OpenRagError::Validation(format!("invalid or corrupt image payload: {e}")))?;
    let (orig_w, orig_h) = dyn_img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err(OpenRagError::Validation("image has zero dimensions".into()));
    }

    let policy = resolve_ocr_policy_with_target(cfg, ctx, remote_client.is_some())?;
    if policy.effective_mode == Sensitivity::Share {
        if let Some(q) = quota {
            let tenant = ctx.tenant_key.as_deref().unwrap_or("default");
            q.try_consume(tenant, 1)?;
        }
    }

    let rgb_raw = dyn_img.to_rgb8();
    let dewarped = dewarp_image(&rgb_raw, cfg);

    if policy.execution_target == OcrExecutionTarget::SchiftCloud {
        if let Some(client) = remote_client {
            let fname = filename.unwrap_or("page.png");
            let resp = client
                .run_ocr(bytes, fname, policy.effective_mode, ctx.batch, ctx.ocr_task)
                .await?;
            return build_doc_from_schift_response(bytes, format, filename, &dewarped, resp, policy);
        }
    }

    // Deterministic offline OCR fallback (optionally augmented by tesseract CLI)
    let mut blocks = detect_foreground_blocks(&dewarped);
    let tesseract_text = try_tesseract_cli(bytes).await;

    let raw_markdown = if let Some(txt) = tesseract_text.filter(|t| !t.trim().is_empty()) {
        if blocks.is_empty() {
            let full_box = BBoxNorm1000::new(0, 0, 999, 999)?;
            blocks.push(OcrBlock {
                kind: "text".into(),
                text: txt.clone(),
                bbox_norm_1000: Some(full_box),
                confidence: Some(0.90),
            });
        } else if let Some(first) = blocks.first_mut() {
            first.text = txt.clone();
        }
        txt
    } else if !blocks.is_empty() {
        blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let full_box = BBoxNorm1000::new(0, 0, 999, 999)?;
        let fallback_line = format!("[OCR image {}x{}]", dewarped.width(), dewarped.height());
        blocks.push(OcrBlock {
            kind: "text".into(),
            text: fallback_line.clone(),
            bbox_norm_1000: Some(full_box),
            confidence: Some(0.50),
        });
        fallback_line
    };

    let aliases = extract_parenthetical_aliases(&raw_markdown);
    let (w, h) = dewarped.dimensions();
    let page = ParsedPage {
        page_no: 1,
        raw_markdown: raw_markdown.clone(),
        tables_html: Vec::new(),
        formulas: Vec::new(),
        fields: Vec::new(),
        blocks,
        extracted_aliases: aliases.clone(),
        page_image_rgb: Some((w, h, dewarped.into_raw())),
        content_sha256: sha256_hex_str(&raw_markdown),
    };

    let title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("OCR Image")
        .to_string();

    let engine = match policy.execution_target {
        OcrExecutionTarget::LocalVllm => "local_vllm",
        OcrExecutionTarget::SchiftCloud => "schift_ocr",
        OcrExecutionTarget::OfflineFallback => "offline_ocr",
    };

    Ok(ParsedDocument {
        title,
        format,
        mime_type: format.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages: vec![page],
        full_markdown: raw_markdown,
        extracted_aliases: aliases,
        ocr_policy: Some(policy),
        parser_engine: engine.to_string(),
    })
}

fn build_doc_from_schift_response(
    bytes: &[u8],
    format: DocumentFormat,
    filename: Option<&str>,
    img: &RgbImage,
    resp: SchiftOcrResponse,
    policy: OcrPolicyDecision,
) -> Result<ParsedDocument, OpenRagError> {
    let mut pages = Vec::new();
    let (w, h) = img.dimensions();

    if resp.pages.is_empty() {
        let md = resp.page_markdown().unwrap_or_default().to_string();
        let blocks = convert_ocr_blocks(resp.page_blocks())?;
        let aliases = extract_parenthetical_aliases(&md);
        pages.push(ParsedPage {
            page_no: 1,
            raw_markdown: md.clone(),
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks,
            extracted_aliases: aliases,
            page_image_rgb: Some((w, h, img.clone().into_raw())),
            content_sha256: sha256_hex_str(&md),
        });
    } else {
        for (idx, p) in resp.pages.iter().enumerate() {
            let md = p.page_markdown();
            let blocks = convert_ocr_blocks(p.page_blocks())?;
            let aliases = extract_parenthetical_aliases(md);
            pages.push(ParsedPage {
                page_no: p.page_no.unwrap_or((idx + 1) as i32),
                raw_markdown: md.to_string(),
                tables_html: Vec::new(),
                formulas: Vec::new(),
                fields: Vec::new(),
                blocks,
                extracted_aliases: aliases,
                page_image_rgb: if idx == 0 {
                    Some((w, h, img.clone().into_raw()))
                } else {
                    None
                },
                content_sha256: sha256_hex_str(md),
            });
        }
    }

    let full_markdown = pages
        .iter()
        .map(|p| p.raw_markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let extracted_aliases = extract_parenthetical_aliases(&full_markdown);
    let title = filename
        .and_then(|f| Path::new(f).file_stem().and_then(|s| s.to_str()))
        .unwrap_or("OCR Document")
        .to_string();

    Ok(ParsedDocument {
        title,
        format,
        mime_type: format.default_mime().to_string(),
        content_sha256: sha256_hex_bytes(bytes),
        pages,
        full_markdown,
        extracted_aliases,
        ocr_policy: Some(policy),
        parser_engine: "schift_ocr".to_string(),
    })
}

fn convert_ocr_blocks(items: &[SchiftOcrBlockItem]) -> Result<Vec<OcrBlock>, OpenRagError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let bbox = match item.bbox {
            Some([x0, y0, x1, y1]) => Some(BBoxNorm1000::new(
                x0.clamp(0, 999),
                y0.clamp(0, 999),
                x1.clamp(0, 999).max(x0.clamp(0, 999)),
                y1.clamp(0, 999).max(y0.clamp(0, 999)),
            )?),
            None => None,
        };
        out.push(OcrBlock {
            kind: item.kind.clone(),
            text: item.text.clone(),
            bbox_norm_1000: bbox,
            confidence: item.confidence,
        });
    }
    Ok(out)
}

/// Detects non-background connected rectangular regions in normalized `0..=999` coordinates.
pub fn detect_foreground_blocks(img: &RgbImage) -> Vec<OcrBlock> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }

    // Estimate background color from the four corners
    let corners = [
        img.get_pixel(0, 0),
        img.get_pixel(w - 1, 0),
        img.get_pixel(0, h - 1),
        img.get_pixel(w - 1, h - 1),
    ];
    let bg_r = corners.iter().map(|p| p[0] as i32).sum::<i32>() / 4;
    let bg_g = corners.iter().map(|p| p[1] as i32).sum::<i32>() / 4;
    let bg_b = corners.iter().map(|p| p[2] as i32).sum::<i32>() / 4;

    // 25x25 coarse grid occupancy mask
    const GX: usize = 25;
    const GY: usize = 25;
    let mut active = [[false; GX]; GY];

    for (gy, active_row) in active.iter_mut().enumerate().take(GY) {
        let y0 = (gy * (h as usize)) / GY;
        let y1 = (((gy + 1) * (h as usize)) / GY).max(y0 + 1).min(h as usize);
        for (gx, active_cell) in active_row.iter_mut().enumerate().take(GX) {
            let x0 = (gx * (w as usize)) / GX;
            let x1 = (((gx + 1) * (w as usize)) / GX).max(x0 + 1).min(w as usize);
            let mut fg = 0usize;
            let mut total = 0usize;
            for py in y0..y1 {
                for px in x0..x1 {
                    let p = img.get_pixel(px as u32, py as u32);
                    let diff = (p[0] as i32 - bg_r).abs()
                        + (p[1] as i32 - bg_g).abs()
                        + (p[2] as i32 - bg_b).abs();
                    if diff > 36 {
                        fg += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 && fg * 10 >= total {
                *active_cell = true;
            }
        }
    }

    // BFS connected components on the 25x25 grid
    let mut visited = [[false; GX]; GY];
    let mut blocks = Vec::new();

    for sy in 0..GY {
        for sx in 0..GX {
            if !active[sy][sx] || visited[sy][sx] {
                continue;
            }
            let mut queue = vec![(sx, sy)];
            visited[sy][sx] = true;
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (sx, sy, sx, sy);
            let mut cell_count = 0usize;

            while let Some((cx, cy)) = queue.pop() {
                cell_count += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);

                let neighbors = [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx < GX && ny < GY && active[ny][nx] && !visited[ny][nx] {
                        visited[ny][nx] = true;
                        queue.push((nx, ny));
                    }
                }
            }

            if cell_count == 0 {
                continue;
            }

            let bx0 = ((min_x * 1000) / GX).min(998) as i32;
            let by0 = ((min_y * 1000) / GY).min(998) as i32;
            let bx1 = ((((max_x + 1) * 1000) / GX).saturating_sub(1))
                .clamp(bx0 as usize + 1, 999) as i32;
            let by1 = ((((max_y + 1) * 1000) / GY).saturating_sub(1))
                .clamp(by0 as usize + 1, 999) as i32;

            if let Ok(bbox) = BBoxNorm1000::new(bx0, by0, bx1, by1) {
                let idx = blocks.len();
                blocks.push(OcrBlock {
                    kind: "figure".into(),
                    text: format!(
                        "[OCR region #{idx} bbox=({bx0},{by0},{bx1},{by1})]"
                    ),
                    bbox_norm_1000: Some(bbox),
                    confidence: Some(0.85),
                });
            }
        }
    }

    blocks
}

async fn try_tesseract_cli(bytes: &[u8]) -> Option<String> {
    let path_var = std::env::var_os("PATH")?;
    let bin = std::env::split_paths(&path_var)
        .map(|d| d.join("tesseract"))
        .find(|p| p.is_file())?;

    let tmp_path = std::env::temp_dir().join(format!("openrag-ocr-{}.png", uuid::Uuid::now_v7()));
    if tokio::fs::write(&tmp_path, bytes).await.is_err() {
        return None;
    }

    let mut cmd = Command::new(bin);
    cmd.arg(&tmp_path).arg("stdout").arg("--psm").arg("6");
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
mod tests {
    use super::*;
    use serde_json::json;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn build(body: serde_json::Value) -> Result<ParsedDocument, OpenRagError> {
        let policy = OcrPolicyDecision {
            effective_mode: Sensitivity::Share,
            execution_target: OcrExecutionTarget::SchiftCloud,
            upgraded_from_share: false,
        };
        build_doc_from_schift_response(
            b"png",
            DocumentFormat::Png,
            Some("card.png"),
            &RgbImage::new(2, 2),
            decode_ocr_body(body)?,
            policy,
        )
    }

    /// Shape of a real `POST https://schift.io/v1/run` reply (2026-11-23, task=markdown, share);
    /// the page text is shortened.
    #[test]
    fn run_response_text_comes_from_result_markdown() -> TestResult {
        let md = "# 인사 기록 카드 (시험용 가상 데이터)\n\n연차: 입사 1년 후 15일, 이월 불가";
        let doc = build(json!({
            "id": "5f66c3726f73", "task": "markdown", "model_version": "schift-ocr-2-20261112",
            "image_size": [1100, 440], "result": {"markdown": md}, "truncated": false,
            "model_usage": {"input_tokens": 1314, "output_tokens": 85, "ms": 615},
            "job_id": "d7a1bee9c86f4cf4a14e8afff8816284", "mode": "share", "stored": true,
            "usage": {"pages": 1, "credits": 0, "credits_paid": 0, "free": 0, "paid": 0, "mode": "share"}
        }))?;
        assert_eq!(doc.pages.len(), 1);
        assert_eq!(doc.pages[0].raw_markdown, md);
        assert_eq!(doc.full_markdown, md);
        Ok(())
    }

    /// A blank page comes back as one picture (real reply, 2026-11-23): kept, not an error.
    #[test]
    fn picture_only_page_is_kept() -> TestResult {
        let doc = build(json!({
            "task": "markdown", "job_id": "5dcfab062ca94079bcf620d1aac54ebf", "mode": "share",
            "result": {"markdown": "![](page-001-fig-1.png)", "images": [{
                "image": "page-001-fig-1.png", "bbox": [0, 0, 400, 300], "caption": null,
                "image_url": "/v1/jobs/5dcfab062ca94079bcf620d1aac54ebf/images/page-001-fig-1.png"
            }]}
        }))?;
        assert_eq!(doc.full_markdown, "![](page-001-fig-1.png)");
        Ok(())
    }

    #[test]
    fn top_level_fields_and_per_page_results_are_read() -> TestResult {
        assert_eq!(build(json!({"markdown": "top"}))?.full_markdown, "top");
        assert_eq!(build(json!({"text": "plain"}))?.full_markdown, "plain");
        let doc = build(json!({"pages": [
            {"page_no": 1, "result": {"markdown": "one"}},
            {"markdown": "two"}
        ]}))?;
        let texts: Vec<&str> = doc.pages.iter().map(|p| p.raw_markdown.as_str()).collect();
        assert_eq!(texts, ["one", "two"]);
        Ok(())
    }

    /// Shape of a real reply to a 2-page scanned PDF (2026-11-23, private): the top-level
    /// `result` is page 1 and `pages[]` holds every page in order, without page numbers.
    #[test]
    fn pdf_reply_maps_pages_in_order() -> TestResult {
        let resp = decode_ocr_body(json!({
            "id": "1d9ac0658287", "task": "markdown", "result": {"markdown": "# 인사 기록 카드"},
            "truncated": false, "job_id": "c8dc3f0385a541d182006a8c0443a5da", "mode": "private",
            "stored": true,
            "pages": [
                {"id": "1d9ac0658287", "task": "markdown", "image_size": [1653, 662],
                 "result": {"markdown": "# 인사 기록 카드"}, "truncated": false},
                {"id": "745bfcc6ffab", "task": "markdown", "image_size": [1653, 900],
                 "result": {"markdown": "# 보안 서약 안내"}, "truncated": false}
            ],
            "usage": {"pages": 2, "credits": 0, "mode": "private"}
        }))?;
        let by_index = ocr_pages_by_index(&resp)?;
        assert_eq!(by_index.len(), 2);
        assert_eq!(by_index.get(&0).map(|p| p.0.as_str()), Some("# 인사 기록 카드"));
        assert_eq!(by_index.get(&1).map(|p| p.0.as_str()), Some("# 보안 서약 안내"));
        Ok(())
    }

    /// The air-gapped profile (`local_vllm`) sends OCR to its configured endpoint, never as share.
    #[test]
    fn local_vllm_with_an_endpoint_uses_it() -> TestResult {
        let cfg = ParseConfig {
            ocr_engine: "local_vllm".into(),
            ..crate::pdf::tests::test_parse_cfg()
        };
        let ctx = ParsePolicyCtx::default();
        let with = resolve_ocr_policy_with_target(&cfg, &ctx, true)?;
        assert_eq!(with.execution_target, OcrExecutionTarget::SchiftCloud);
        assert_eq!(with.effective_mode, Sensitivity::Private);
        let without = resolve_ocr_policy_with_target(&cfg, &ctx, false)?;
        assert_eq!(without.execution_target, OcrExecutionTarget::LocalVllm);
        assert_eq!(ocr_upload_mime("scan.PDF"), "application/pdf");
        assert_eq!(ocr_upload_mime("p.jpg"), "image/jpeg");
        Ok(())
    }

    #[test]
    fn reply_without_page_text_is_an_error() -> TestResult {
        let reply = json!({"job_id": "x", "output": {"md": "text in a field we do not read"}});
        let Err(OpenRagError::ExternalAdapter { adapter, message }) = build(reply) else {
            return Err("expected a schift_ocr adapter error".into());
        };
        assert_eq!(adapter, "schift_ocr");
        assert!(message.contains("no page text"), "{message}");
        assert!(message.contains("output"), "{message}");
        Ok(())
    }
}
