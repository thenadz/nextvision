//! Pre-processing: letterbox resize with aspect-ratio preservation.
//!
//! The input RGB8 or RGBA8 pixel buffer is resized to `target_size × target_size`,
//! centered with gray (114/255) padding, and converted to a float32
//! NCHW tensor with values normalised to the 0–1 range.

/// Metadata about the letterbox transform so that bounding boxes can be
/// mapped back to the original frame's normalised `[0, 1]` space.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    /// Scale factor applied to the image (same for x and y).
    pub scale: f32,
    /// Horizontal offset (pixels in the letterboxed image).
    pub pad_x: f32,
    /// Vertical offset (pixels in the letterboxed image).
    pub pad_y: f32,
    /// Target (letterboxed) image size.
    pub target: u32,
    /// Original frame width.
    pub orig_w: u32,
    /// Original frame height.
    pub orig_h: u32,
}

impl LetterboxInfo {
    /// Remap a bounding box from letterbox pixel coordinates back to
    /// normalised `[0, 1]` coordinates relative to the original frame.
    pub fn remap_to_normalised(&self, x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> (f32, f32, f32, f32) {
        let x0 = ((x_min - self.pad_x) / self.scale).clamp(0.0, self.orig_w as f32);
        let y0 = ((y_min - self.pad_y) / self.scale).clamp(0.0, self.orig_h as f32);
        let x1 = ((x_max - self.pad_x) / self.scale).clamp(0.0, self.orig_w as f32);
        let y1 = ((y_max - self.pad_y) / self.scale).clamp(0.0, self.orig_h as f32);
        (
            x0 / self.orig_w as f32,
            y0 / self.orig_h as f32,
            x1 / self.orig_w as f32,
            y1 / self.orig_h as f32,
        )
    }
}

/// Perform letterbox pre-processing: resize + pad + convert to NCHW float32.
///
/// Returns `(tensor_data, info)` where `tensor_data` is a flat `Vec<f32>`
/// in NCHW layout `[1, 3, H, W]` with values in 0–255 range, and `info`
/// carries the transform metadata for coordinate remapping.
///
/// For batch processing where multiple frames are concatenated into one
/// tensor, prefer [`letterbox_preprocess_into`] to avoid per-frame
/// allocations.
///
/// # Arguments
/// - `rgb_data` — pixel buffer in row-major RGB8 or RGBA8 order.
/// - `src_w`, `src_h` — original frame dimensions in pixels.
/// - `src_stride` — byte stride per row (may include padding).
/// - `bpp` — bytes per pixel (3 for RGB8, 4 for RGBA8).
/// - `target` — target square size for the model input.
pub fn letterbox_preprocess(
    rgb_data: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    bpp: u32,
    target: u32,
) -> (Vec<f32>, LetterboxInfo) {
    let t = target as usize;
    let pixels = t * t;
    let mut tensor = vec![114.0_f32 / 255.0; pixels * 3];
    let info = letterbox_preprocess_into(rgb_data, src_w, src_h, src_stride, bpp, target, &mut tensor);
    (tensor, info)
}

/// Letterbox pre-processing into a caller-provided buffer.
///
/// Writes NCHW float32 data (0–255 range) into `output`, which must be
/// at least `3 * target * target` elements. Padding pixels are set to
/// 114.0. Returns the [`LetterboxInfo`] for coordinate remapping.
///
/// This variant avoids per-frame allocation when building a batched
/// tensor: the caller pre-allocates the full `[N, 3, H, W]` buffer
/// and passes successive slices.
///
/// # Panics
///
/// Panics if `output.len() < 3 * target * target`.
pub fn letterbox_preprocess_into(
    rgb_data: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    bpp: u32,
    target: u32,
    output: &mut [f32],
) -> LetterboxInfo {
    let t = target as usize;
    let pixels = t * t;
    assert!(
        output.len() >= pixels * 3,
        "output buffer too small: need {}, got {}",
        pixels * 3,
        output.len()
    );

    // Fill with gray padding value (114/255 ≈ 0.447).
    output[..pixels * 3].fill(114.0 / 255.0);

    let scale = (target as f32 / src_w as f32).min(target as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as u32;
    let new_h = (src_h as f32 * scale).round() as u32;
    let pad_x = (target - new_w) as f32 / 2.0;
    let pad_y = (target - new_h) as f32 / 2.0;
    let pad_x_int = pad_x as u32;
    let pad_y_int = pad_y as u32;

    // Nearest-neighbour resize + place into padded canvas.
    for dst_y in 0..new_h {
        let src_y = ((dst_y as f32 / scale) as u32).min(src_h.saturating_sub(1));
        let out_y = (dst_y + pad_y_int) as usize;
        for dst_x in 0..new_w {
            let src_x = ((dst_x as f32 / scale) as u32).min(src_w.saturating_sub(1));
            let src_idx = (src_y * src_stride + src_x * bpp) as usize;
            let out_x = (dst_x + pad_x_int) as usize;
            if src_idx + 2 < rgb_data.len() {
                let r = rgb_data[src_idx] as f32 / 255.0;
                let g = rgb_data[src_idx + 1] as f32 / 255.0;
                let b = rgb_data[src_idx + 2] as f32 / 255.0;
                // NCHW layout, RGB order (Ultralytics convention).
                output[out_y * t + out_x] = r;
                output[pixels + out_y * t + out_x] = g;
                output[2 * pixels + out_y * t + out_x] = b;
            }
        }
    }

    LetterboxInfo {
        scale,
        pad_x,
        pad_y,
        target,
        orig_w: src_w,
        orig_h: src_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_identity_when_no_padding() {
        // If the image exactly fits the target, remap should be identity-ish.
        let info = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            target: 640,
            orig_w: 640,
            orig_h: 640,
        };
        let (x0, y0, x1, y1) = info.remap_to_normalised(100.0, 200.0, 300.0, 400.0);
        assert!((x0 - 100.0 / 640.0).abs() < 1e-5);
        assert!((y0 - 200.0 / 640.0).abs() < 1e-5);
        assert!((x1 - 300.0 / 640.0).abs() < 1e-5);
        assert!((y1 - 400.0 / 640.0).abs() < 1e-5);
    }

    #[test]
    fn remap_with_padding() {
        // 1920×1080 → 640. scale = 640/1920 = 0.333..
        // new_w = 640, new_h = 360; pad_y = (640-360)/2 = 140
        let info = LetterboxInfo {
            scale: 640.0 / 1920.0,
            pad_x: 0.0,
            pad_y: 140.0,
            target: 640,
            orig_w: 1920,
            orig_h: 1080,
        };
        // A point at (0, 140) in letterbox = top-left of actual image
        let (x0, y0, _, _) = info.remap_to_normalised(0.0, 140.0, 1.0, 141.0);
        assert!(x0.abs() < 1e-3);
        assert!(y0.abs() < 1e-3);
    }

    #[test]
    fn letterbox_output_shape() {
        // 4×2 RGB image, solid red.
        let w = 4u32;
        let h = 2u32;
        let stride = w * 3;
        let data = [255u8, 0, 0].repeat((w * h) as usize);
        let target = 8;
        let (tensor, info) = letterbox_preprocess(&data, w, h, stride, 3, target);
        assert_eq!(tensor.len(), 3 * 8 * 8);
        assert!(info.scale > 0.0);
    }

    #[test]
    fn letterbox_output_shape_rgba() {
        // 4×2 RGBA image, solid red.
        let w = 4u32;
        let h = 2u32;
        let stride = w * 4;
        let data = [255u8, 0, 0, 255].repeat((w * h) as usize);
        let target = 8;
        let (tensor, info) = letterbox_preprocess(&data, w, h, stride, 4, target);
        assert_eq!(tensor.len(), 3 * 8 * 8);
        assert!(info.scale > 0.0);
    }
}
