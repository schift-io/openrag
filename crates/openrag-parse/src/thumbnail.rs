use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use openrag_core::{BBoxNorm1000, OpenRagError, ThumbnailConfig};

use crate::model::{ParsedDocument, ParsedFigure, ParsedPage, RenderedPageThumbnail};
use crate::ocr::detect_foreground_blocks;

/// Where page thumbnails and figure crops are kept: local files, memory (tests) or an S3 bucket.
#[async_trait::async_trait]
pub trait ThumbnailBlobStore: Send + Sync {
    /// Store `bytes` under `key` and return the URI to keep in the database.
    async fn put_bytes(&self, key: &str, bytes: &[u8]) -> Result<String, OpenRagError>;
    /// Write `bytes` at a URI this store handed out, or one derived from it (`*.redacted.webp`).
    async fn put_uri(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError>;
    async fn get_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError>;
}

#[derive(Debug, Clone)]
pub struct LocalFsBlobStore {
    root_dir: PathBuf,
}

impl LocalFsBlobStore {
    pub fn new(root_dir: impl Into<PathBuf>) -> Result<Self, OpenRagError> {
        let root_dir = root_dir.into();
        std::fs::create_dir_all(&root_dir).map_err(|e| {
            OpenRagError::Internal(format!(
                "failed to create thumbnail blob root '{}': {e}",
                root_dir.display()
            ))
        })?;
        Ok(Self { root_dir })
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }
}

async fn write_file(path: &Path, bytes: &[u8]) -> Result<(), OpenRagError> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            OpenRagError::Internal(format!(
                "failed to create thumbnail dir '{}': {e}",
                parent.display()
            ))
        })?;
    }
    tokio::fs::write(path, bytes).await.map_err(|e| {
        OpenRagError::Internal(format!(
            "failed to write thumbnail '{}': {e}",
            path.display()
        ))
    })
}

#[async_trait::async_trait]
impl ThumbnailBlobStore for LocalFsBlobStore {
    async fn put_bytes(&self, key: &str, bytes: &[u8]) -> Result<String, OpenRagError> {
        let clean = key.trim_start_matches('/').replace("..", "_");
        let path = self.root_dir.join(clean);
        write_file(&path, bytes).await?;
        Ok(path.to_string_lossy().into_owned())
    }

    async fn put_uri(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError> {
        write_file(Path::new(uri.strip_prefix("file://").unwrap_or(uri)), bytes).await
    }

    async fn get_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError> {
        let path_str = uri.strip_prefix("file://").unwrap_or(uri);
        tokio::fs::read(path_str).await.map_err(|e| {
            OpenRagError::NotFound(format!("failed to read thumbnail '{uri}': {e}"))
        })
    }
}

#[derive(Debug, Default)]
pub struct InMemoryBlobStore {
    inner: Mutex<HashMap<String, Vec<u8>>>,
}

impl InMemoryBlobStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl ThumbnailBlobStore for InMemoryBlobStore {
    async fn put_bytes(&self, key: &str, bytes: &[u8]) -> Result<String, OpenRagError> {
        let uri = format!("mem://{}", key.trim_start_matches('/'));
        self.put_uri(&uri, bytes).await?;
        Ok(uri)
    }

    async fn put_uri(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError> {
        if !uri.starts_with("mem://") {
            return Err(OpenRagError::Validation(format!(
                "InMemoryBlobStore cannot write '{uri}'"
            )));
        }
        let mut guard = self.inner.lock().map_err(|_| {
            OpenRagError::Internal("InMemoryBlobStore mutex poisoned".into())
        })?;
        guard.insert(uri.to_string(), bytes.to_vec());
        Ok(())
    }

    async fn get_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError> {
        let guard = self.inner.lock().map_err(|_| {
            OpenRagError::Internal("InMemoryBlobStore mutex poisoned".into())
        })?;
        guard
            .get(uri)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("blob '{uri}' not found")))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Thumbnailer;

impl Thumbnailer {
    pub const fn new() -> Self {
        Self
    }

    /// Renders a WebP thumbnail per page and crops its figures to PNG. Neither is embedded:
    /// pages and figures are found through their text and tags.
    pub fn render(
        &self,
        doc: &ParsedDocument,
        source_bytes: Option<&[u8]>,
        cfg: &ThumbnailConfig,
    ) -> Result<Vec<RenderedPageThumbnail>, OpenRagError> {
        if cfg.max_edge_px == 0 || cfg.figure_max_edge_px == 0 || cfg.figure_max_bytes == 0 {
            return Err(OpenRagError::Config(
                "thumbnail bounds must be >= 1".into(),
            ));
        }

        let mut results = Vec::with_capacity(doc.pages.len());
        let decoded_source_rgb = source_bytes
            .and_then(|b| image::load_from_memory(b).ok())
            .map(|img| img.to_rgb8());

        for (idx, page) in doc.pages.iter().enumerate() {
            let base_rgb = if let Some((w, h, ref buf)) = page.page_image_rgb {
                RgbImage::from_raw(w, h, buf.clone()).ok_or_else(|| {
                    OpenRagError::Internal("invalid page_image_rgb dimensions".into())
                })?
            } else if idx == 0 {
                if let Some(ref src) = decoded_source_rgb {
                    src.clone()
                } else {
                    render_synthetic_page_image(page, cfg.max_edge_px)
                }
            } else {
                render_synthetic_page_image(page, cfg.max_edge_px)
            };

            let resized = resize_to_max_edge(&base_rgb, cfg.max_edge_px);
            let webp_bytes = encode_webp(&resized)?;
            let figures = extract_and_crop_figures(page, &base_rgb, cfg)?;

            results.push(RenderedPageThumbnail {
                page_no: page.page_no,
                width: resized.width(),
                height: resized.height(),
                webp_bytes,
                figures,
            });
        }

        Ok(results)
    }
}

fn resize_to_max_edge(img: &RgbImage, max_edge: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 || (w <= max_edge && h <= max_edge) {
        return img.clone();
    }
    let scale = (max_edge as f64) / (w.max(h) as f64);
    let nw = ((w as f64) * scale).round().max(1.0) as u32;
    let nh = ((h as f64) * scale).round().max(1.0) as u32;
    DynamicImage::ImageRgb8(img.clone())
        .resize_exact(nw, nh, FilterType::Triangle)
        .to_rgb8()
}

pub fn encode_webp(img: &RgbImage) -> Result<Vec<u8>, OpenRagError> {
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(img.clone())
        .write_to(&mut out, ImageFormat::WebP)
        .map_err(|e| OpenRagError::Internal(format!("failed to encode WebP thumbnail: {e}")))?;
    Ok(out.into_inner())
}

pub fn encode_png_bounded(
    img: &RgbImage,
    max_edge_px: u32,
    max_bytes: u64,
) -> Result<Vec<u8>, OpenRagError> {
    if max_edge_px == 0 || max_bytes == 0 {
        return Err(OpenRagError::Validation(
            "PNG bounds max_edge_px and max_bytes must be >= 1".into(),
        ));
    }
    let mut current = resize_to_max_edge(img, max_edge_px);
    loop {
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(current.clone())
            .write_to(&mut out, ImageFormat::Png)
            .map_err(|e| OpenRagError::Internal(format!("failed to encode PNG figure: {e}")))?;
        let bytes = out.into_inner();
        if (bytes.len() as u64) <= max_bytes {
            return Ok(bytes);
        }
        let cur_edge = current.width().max(current.height());
        if cur_edge <= 1 {
            return Err(OpenRagError::Validation(format!(
                "PNG figure cannot fit within max_bytes={max_bytes} even at 1x1 (got {} bytes)",
                bytes.len()
            )));
        }
        let next_edge = ((cur_edge * 3) / 4).min(cur_edge - 1).max(1);
        current = resize_to_max_edge(&current, next_edge);
    }
}

fn extract_and_crop_figures(
    page: &ParsedPage,
    base_rgb: &RgbImage,
    cfg: &ThumbnailConfig,
) -> Result<Vec<ParsedFigure>, OpenRagError> {
    let mut figure_specs: Vec<(String, Option<String>, BBoxNorm1000)> = Vec::new();
    for block in &page.blocks {
        let k = block.kind.to_ascii_lowercase();
        if matches!(k.as_str(), "figure" | "image" | "chart" | "table" | "region") {
            if let Some(bbox) = block.bbox_norm_1000 {
                figure_specs.push((
                    block.kind.clone(),
                    if block.text.trim().is_empty() {
                        None
                    } else {
                        Some(block.text.clone())
                    },
                    bbox,
                ));
            }
        }
    }

    if figure_specs.is_empty() && page.page_image_rgb.is_some() {
        for blk in detect_foreground_blocks(base_rgb) {
            if let Some(bbox) = blk.bbox_norm_1000 {
                figure_specs.push((blk.kind, Some(blk.text), bbox));
            }
        }
    }

    let (w, h) = base_rgb.dimensions();
    let mut out = Vec::with_capacity(figure_specs.len());
    for (fig_idx, (kind, caption, bbox)) in figure_specs.into_iter().enumerate() {
        let (rx, ry, rw, rh) = bbox.to_pixel_rect(w, h);
        if rw == 0 || rh == 0 {
            continue;
        }
        let cropped = DynamicImage::ImageRgb8(base_rgb.clone())
            .crop_imm(rx, ry, rw, rh)
            .to_rgb8();
        let png_bytes =
            encode_png_bounded(&cropped, cfg.figure_max_edge_px, cfg.figure_max_bytes)?;
        let filename = format!("page-{:03}-fig-{fig_idx}.png", page.page_no);
        out.push(ParsedFigure {
            figure_idx: fig_idx as i32,
            filename,
            kind,
            caption,
            bbox_norm_1000: Some(bbox),
            png_bytes,
        });
    }

    Ok(out)
}

/// Renders a crisp A4-proportioned synthetic page image reflecting page text lines & tables.
fn render_synthetic_page_image(page: &ParsedPage, max_edge_px: u32) -> RgbImage {
    let target_h = max_edge_px.clamp(240, 1200);
    let target_w = ((target_h as f64) * (210.0 / 297.0)).round() as u32;
    let mut img = RgbImage::from_pixel(target_w, target_h, Rgb([252, 252, 253]));

    // Header band
    let header_h = (target_h / 16).max(8);
    for y in 8..header_h.min(target_h) {
        for x in 12..target_w.saturating_sub(12) {
            img.put_pixel(x, y, Rgb([30, 58, 138]));
        }
    }

    // Deterministic horizontal text bars proportional to markdown lines
    let mut cursor_y = header_h + 14;
    let margin_x = (target_w / 14).max(8);
    let usable_w = target_w.saturating_sub(margin_x * 2).max(16);

    for (line_idx, line) in page.raw_markdown.lines().take(28).enumerate() {
        if cursor_y + 10 >= target_h {
            break;
        }
        let char_count = line.chars().count().min(80) as u32;
        if char_count == 0 {
            cursor_y += 8;
            continue;
        }
        let bar_w = ((usable_w * char_count) / 80).max(12).min(usable_w);
        let color = if line.contains("<table") {
            Rgb([15, 118, 110])
        } else if line_idx == 0 {
            Rgb([17, 24, 39])
        } else {
            Rgb([75, 85, 99])
        };
        let bar_h = if line.contains("<table") { 10 } else { 5 };
        for y in cursor_y..(cursor_y + bar_h).min(target_h) {
            for x in margin_x..(margin_x + bar_w).min(target_w) {
                img.put_pixel(x, y, color);
            }
        }
        cursor_y += bar_h + 7;
    }

    img
}
