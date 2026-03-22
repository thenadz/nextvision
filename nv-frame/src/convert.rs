//! Pixel format conversion utilities.
//!
//! These are **opt-in** and allocate a new [`FrameEnvelope`]. They are not
//! used on the hot path — stages that need a specific format should convert
//! explicitly.

use crate::frame::{FrameEnvelope, FrameAccessError, PixelFormat};

/// Errors from [`convert()`].
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    /// The source and target formats are identical — clone instead.
    #[error("source format already matches target")]
    SameFormat,
    /// No conversion path exists for the given format pair.
    #[error("unsupported conversion: {from:?} -> {to:?}")]
    Unsupported {
        /// Source pixel format.
        from: PixelFormat,
        /// Target pixel format.
        to: PixelFormat,
    },
    /// Host bytes could not be obtained from the frame.
    #[error("frame data not accessible: {0}")]
    Access(#[from] FrameAccessError),
}

/// Convert a frame to a different pixel format.
///
/// Works with host-resident **and** [`MappableToHost`](crate::DataAccess::MappableToHost)
/// device frames. Conversion always allocates a new output buffer, so the
/// additional cost of materializing device data is negligible.
///
/// # Errors
///
/// - [`ConvertError::SameFormat`] — source already matches `target`; clone instead.
/// - [`ConvertError::Unsupported`] — no conversion path for this format pair.
/// - [`ConvertError::Access`] — host bytes could not be obtained (opaque device frame).
///
/// Currently supported paths:
/// - `Bgr8` → `Rgb8`
/// - `Rgb8` → `Bgr8`
/// - `Rgba8` → `Rgb8`
/// - `Rgb8` → `Gray8`
///
/// Other conversions can be contributed by adding a match arm below.
pub fn convert(frame: &FrameEnvelope, target: PixelFormat) -> Result<FrameEnvelope, ConvertError> {
    if frame.format() == target {
        return Err(ConvertError::SameFormat);
    }

    let host_bytes = frame.require_host_data()?;

    let w = frame.width();
    let h = frame.height();
    let src_stride = frame.stride();

    // Validate that the frame data is large enough for the declared dimensions.
    // A truncated or corrupt frame would otherwise panic in pixel_rows().
    // All arithmetic is checked to prevent overflow from adversarial dimensions.
    let src_bpp = match frame.format() {
        PixelFormat::Gray8 => 1u32,
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
        PixelFormat::Rgba8 => 4,
        _ => {
            return Err(ConvertError::Unsupported {
                from: frame.format(),
                to: target,
            })
        }
    };
    let required = checked_frame_size(w, h, src_stride, src_bpp).ok_or_else(|| {
        ConvertError::Access(FrameAccessError::MaterializationFailed {
            detail: format!(
                "dimension overflow: {}x{} stride={} bpp={}",
                w, h, src_stride, src_bpp,
            ),
        })
    })?;
    if host_bytes.len() < required {
        return Err(ConvertError::Access(FrameAccessError::MaterializationFailed {
            detail: format!(
                "frame data too short: {} bytes for {}x{} stride={} bpp={}",
                host_bytes.len(), w, h, src_stride, src_bpp,
            ),
        }));
    }

    let converted = match (frame.format(), target) {
        (PixelFormat::Bgr8, PixelFormat::Rgb8) | (PixelFormat::Rgb8, PixelFormat::Bgr8) => {
            swap_rb(&host_bytes, w, h, src_stride)
        }
        (PixelFormat::Rgba8, PixelFormat::Rgb8) => rgba_to_rgb(&host_bytes, w, h, src_stride),
        (PixelFormat::Rgb8, PixelFormat::Gray8) => rgb_to_gray(&host_bytes, w, h, src_stride),
        _ => {
            return Err(ConvertError::Unsupported {
                from: frame.format(),
                to: target,
            })
        }
    };

    let out_stride = match target {
        PixelFormat::Gray8 => w,
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => w.checked_mul(3).ok_or_else(|| {
            ConvertError::Access(FrameAccessError::MaterializationFailed {
                detail: format!("output stride overflow: width={} bpp=3", w),
            })
        })?,
        PixelFormat::Rgba8 => w.checked_mul(4).ok_or_else(|| {
            ConvertError::Access(FrameAccessError::MaterializationFailed {
                detail: format!("output stride overflow: width={} bpp=4", w),
            })
        })?,
        _ => {
            return Err(ConvertError::Unsupported {
                from: frame.format(),
                to: target,
            })
        }
    };

    Ok(FrameEnvelope::new_owned(
        frame.feed_id(),
        frame.seq(),
        frame.ts(),
        frame.wall_ts(),
        w,
        h,
        target,
        out_stride,
        converted,
        frame.metadata().clone(),
    ))
}

/// Compute the minimum buffer size for the given frame dimensions using
/// checked arithmetic. Returns `None` on overflow.
fn checked_frame_size(width: u32, height: u32, stride: u32, bpp: u32) -> Option<usize> {
    if height == 0 {
        return Some(0);
    }
    let last_row_bytes = (width as usize).checked_mul(bpp as usize)?;
    let prefix_rows = (height as usize).checked_sub(1)?;
    let prefix_bytes = prefix_rows.checked_mul(stride as usize)?;
    prefix_bytes.checked_add(last_row_bytes)
}

/// Extract tightly-packed pixel rows from potentially padded data.
///
/// GStreamer (and other backends) may deliver rows with padding bytes
/// at the end (stride > width × bpp). This iterator yields only the
/// meaningful bytes of each row, skipping any trailing padding.
///
/// # Safety contract
///
/// The caller must ensure that `data.len()` is at least
/// `checked_frame_size(width, height, stride, bpp)`. The `convert()`
/// function validates this before calling `pixel_rows`.
fn pixel_rows(data: &[u8], width: u32, height: u32, stride: u32, bpp: u32) -> Vec<&[u8]> {
    let row_bytes = (width as usize) * (bpp as usize);
    let stride = stride as usize;
    (0..height as usize)
        .map(|y| {
            let start = y * stride;
            &data[start..start + row_bytes]
        })
        .collect()
}

/// Swap R and B channels in a 3-byte-per-pixel buffer, respecting stride.
fn swap_rb(data: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let rows = pixel_rows(data, width, height, stride, 3);
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for row in rows {
        for pixel in row.chunks_exact(3) {
            out.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
    }
    out
}

/// Drop the alpha channel from RGBA data, respecting stride.
fn rgba_to_rgb(data: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let rows = pixel_rows(data, width, height, stride, 4);
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for row in rows {
        for px in row.chunks_exact(4) {
            out.extend_from_slice(&[px[0], px[1], px[2]]);
        }
    }
    out
}

/// Convert RGB to grayscale using luminance weights, respecting stride.
fn rgb_to_gray(data: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let rows = pixel_rows(data, width, height, stride, 3);
    let mut out = Vec::with_capacity((width * height) as usize);
    for row in rows {
        for px in row.chunks_exact(3) {
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;
            out.push((0.299 * r + 0.587 * g + 0.114 * b).round() as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::PixelFormat;
    use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};

    fn make_frame(format: PixelFormat, data: Vec<u8>, w: u32, h: u32) -> FrameEnvelope {
        let stride = match format {
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => w * 3,
            PixelFormat::Rgba8 => w * 4,
            PixelFormat::Gray8 => w,
            _ => w,
        };
        FrameEnvelope::new_owned(
            FeedId::new(1),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            w,
            h,
            format,
            stride,
            data,
            TypedMetadata::new(),
        )
    }

    #[test]
    fn same_format_returns_error() {
        let f = make_frame(PixelFormat::Rgb8, vec![0; 12], 2, 2);
        assert!(matches!(
            convert(&f, PixelFormat::Rgb8),
            Err(ConvertError::SameFormat)
        ));
    }

    #[test]
    fn bgr_to_rgb() {
        let data = vec![10, 20, 30, 40, 50, 60];
        let f = make_frame(PixelFormat::Bgr8, data, 2, 1);
        let converted = convert(&f, PixelFormat::Rgb8).unwrap();
        assert_eq!(converted.format(), PixelFormat::Rgb8);
        assert_eq!(converted.host_data().unwrap(), &[30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn bgr_to_rgb_with_stride_padding() {
        // 2 pixels wide × 2 rows, BGR8 = 6 bytes/row, but stride = 8 (2 padding bytes)
        let data = vec![
            10, 20, 30, 40, 50, 60, 0xAA, 0xBB, // row 0 + 2 pad bytes
            70, 80, 90, 11, 22, 33, 0xCC, 0xDD, // row 1 + 2 pad bytes
        ];
        let f = FrameEnvelope::new_owned(
            FeedId::new(1),
            0,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            2,
            2,
            PixelFormat::Bgr8,
            8, // stride > width*3
            data,
            TypedMetadata::new(),
        );
        let converted = convert(&f, PixelFormat::Rgb8).unwrap();
        // Padding bytes must NOT appear in output
        assert_eq!(
            converted.host_data().unwrap(),
            &[30, 20, 10, 60, 50, 40, 90, 80, 70, 33, 22, 11]
        );
        assert_eq!(converted.stride(), 6); // tightly packed output
    }

    #[test]
    fn conversion_preserves_metadata() {
        #[derive(Clone, Debug, PartialEq)]
        struct Tag(u32);

        let mut meta = TypedMetadata::new();
        meta.insert(Tag(42));

        let f = FrameEnvelope::new_owned(
            FeedId::new(1),
            7,
            MonotonicTs::ZERO,
            WallTs::from_micros(0),
            2,
            1,
            PixelFormat::Bgr8,
            6,
            vec![10, 20, 30, 40, 50, 60],
            meta,
        );
        let converted = convert(&f, PixelFormat::Rgb8).unwrap();
        assert_eq!(converted.metadata().get::<Tag>(), Some(&Tag(42)));
        assert_eq!(converted.seq(), 7);
    }
}
