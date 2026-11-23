use serde::{Deserialize, Serialize};

use crate::error::OpenRagError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BBoxNorm1000 {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

impl BBoxNorm1000 {
    pub fn new(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> Result<Self, OpenRagError> {
        let coords = [x_min, y_min, x_max, y_max];
        if coords.iter().any(|&v| !(0..=999).contains(&v)) {
            return Err(OpenRagError::Validation(format!(
                "bbox_norm_1000 coordinates must be within 0..=999, got {coords:?}"
            )));
        }
        if x_min > x_max || y_min > y_max {
            return Err(OpenRagError::Validation(format!(
                "bbox_norm_1000 requires min <= max, got {coords:?}"
            )));
        }
        Ok(Self {
            x_min,
            y_min,
            x_max,
            y_max,
        })
    }

    pub fn from_array(coords: [i32; 4]) -> Result<Self, OpenRagError> {
        Self::new(coords[0], coords[1], coords[2], coords[3])
    }

    pub const fn to_array(self) -> [i32; 4] {
        [self.x_min, self.y_min, self.x_max, self.y_max]
    }

    /// Maps normalized 0..999 coordinates to `(x, y, width, height)` in pixels.
    pub fn to_pixel_rect(self, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
        if img_w == 0 || img_h == 0 {
            return (0, 0, 0, 0);
        }
        let x0 = ((self.x_min as u64) * (img_w as u64) / 1000) as u32;
        let y0 = ((self.y_min as u64) * (img_h as u64) / 1000) as u32;
        let x1 = (((self.x_max as u64 + 1) * (img_w as u64)).div_ceil(1000)).min(img_w as u64) as u32;
        let y1 = (((self.y_max as u64 + 1) * (img_h as u64)).div_ceil(1000)).min(img_h as u64) as u32;
        (x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
    }
}
