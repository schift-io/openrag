use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgba};
use openrag_core::{BBoxNorm1000, OpenRagError};

#[async_trait::async_trait]
pub trait ThumbnailStore: Send + Sync {
    async fn read_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError>;
    async fn write_bytes(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError>;
}

#[derive(Debug, Clone, Default)]
pub struct LocalThumbnailStore;

impl LocalThumbnailStore {
    pub const fn new() -> Self {
        Self
    }

    fn resolve_path(uri: &str) -> &str {
        uri.strip_prefix("file://").unwrap_or(uri)
    }

    pub fn derive_redacted_uri(orig_uri: &str) -> String {
        if let Some(dot) = orig_uri.rfind('.') {
            let (stem, ext) = orig_uri.split_at(dot);
            if stem.ends_with(".redacted") {
                return orig_uri.to_string();
            }
            format!("{stem}.redacted{ext}")
        } else {
            format!("{orig_uri}.redacted")
        }
    }
}

#[async_trait::async_trait]
impl ThumbnailStore for LocalThumbnailStore {
    async fn read_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError> {
        let path = Self::resolve_path(uri);
        tokio::fs::read(path).await.map_err(|e| {
            OpenRagError::NotFound(format!("failed to read thumbnail '{uri}': {e}"))
        })
    }

    async fn write_bytes(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError> {
        let path_str = Self::resolve_path(uri);
        let path = std::path::Path::new(path_str);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    OpenRagError::Internal(format!(
                        "failed creating thumbnail directory '{}': {e}",
                        parent.display()
                    ))
                })?;
            }
        }
        tokio::fs::write(path, bytes).await.map_err(|e| {
            OpenRagError::Internal(format!("failed writing thumbnail '{uri}': {e}"))
        })
    }
}

/// Paints pure opaque black `(0, 0, 0, 255)` over every `BBoxNorm1000` rectangle on a PNG,
/// WebP, or JPEG thumbnail image, preserving the original image format.
pub fn mask_image_bboxes(
    image_bytes: &[u8],
    bboxes: &[BBoxNorm1000],
) -> Result<Vec<u8>, OpenRagError> {
    if image_bytes.is_empty() {
        return Err(OpenRagError::Validation(
            "cannot mask empty thumbnail image bytes".into(),
        ));
    }

    let format = image::guess_format(image_bytes)
        .map_err(|e| OpenRagError::Validation(format!("unsupported thumbnail format: {e}")))?;

    let dyn_img = image::load_from_memory(image_bytes)
        .map_err(|e| OpenRagError::Validation(format!("corrupt thumbnail image: {e}")))?;
    let mut rgba = dyn_img.to_rgba8();
    let (w, h) = rgba.dimensions();

    for bbox in bboxes {
        let (rx, ry, rw, rh) = bbox.to_pixel_rect(w, h);
        let x_end = rx.saturating_add(rw).min(w);
        let y_end = ry.saturating_add(rh).min(h);
        for py in ry..y_end {
            for px in rx..x_end {
                rgba.put_pixel(px, py, Rgba([0, 0, 0, 255]));
            }
        }
    }

    let mut out = Cursor::new(Vec::new());
    let out_format = match format {
        ImageFormat::Png => ImageFormat::Png,
        ImageFormat::WebP => ImageFormat::WebP,
        ImageFormat::Jpeg => ImageFormat::Jpeg,
        _ => ImageFormat::Png,
    };

    if out_format == ImageFormat::Jpeg {
        DynamicImage::ImageRgba8(rgba)
            .to_rgb8()
            .write_to(&mut out, ImageFormat::Jpeg)
            .map_err(|e| OpenRagError::Internal(format!("failed encoding masked JPEG: {e}")))?;
    } else {
        DynamicImage::ImageRgba8(rgba)
            .write_to(&mut out, out_format)
            .map_err(|e| OpenRagError::Internal(format!("failed encoding masked image: {e}")))?;
    }

    Ok(out.into_inner())
}
