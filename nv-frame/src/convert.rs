//! Pixel format conversion utilities.
//!
//! These are **opt-in** and allocate a new [`FrameEnvelope`]. They are not
//! used on the hot path — stages that need a specific format should convert
//! explicitly.

use crate::frame::{FrameEnvelope, PixelFormat};

/// Convert a frame to a different pixel format.
///
/// Returns `None` if the conversion is not supported or the source format
/// matches the target (in which case, just clone the frame).
///
/// Currently supports:
/// - `Bgr8` → `Rgb8`
/// - `Rgb8` → `Bgr8`
/// - `Rgba8` → `Rgb8`
/// - `Rgb8` → `Gray8`
///
/// Additional conversions will be added as needed.
#[must_use]
pub fn convert(frame: &FrameEnvelope, target: PixelFormat) -> Option<FrameEnvelope> {
    if frame.format() == target {
        return None; // caller should clone instead
    }

    let converted = match (frame.format(), target) {
        (PixelFormat::Bgr8, PixelFormat::Rgb8) | (PixelFormat::Rgb8, PixelFormat::Bgr8) => {
            swap_rb(frame.data())
        }
        (PixelFormat::Rgba8, PixelFormat::Rgb8) => rgba_to_rgb(frame.data()),
        (PixelFormat::Rgb8, PixelFormat::Gray8) => rgb_to_gray(frame.data()),
        _ => return None,
    };

    let stride = match target {
        PixelFormat::Gray8 => frame.width(),
        PixelFormat::Rgb8 | PixelFormat::Bgr8 => frame.width() * 3,
        PixelFormat::Rgba8 => frame.width() * 4,
        _ => return None,
    };

    Some(FrameEnvelope::new_owned(
        frame.feed_id(),
        frame.seq(),
        frame.ts(),
        frame.wall_ts(),
        frame.width(),
        frame.height(),
        target,
        stride,
        converted,
        frame.metadata().clone(),
    ))
}

/// Swap R and B channels in a 3-byte-per-pixel buffer.
fn swap_rb(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for pixel in out.chunks_exact_mut(3) {
        pixel.swap(0, 2);
    }
    out
}

/// Drop the alpha channel from RGBA data.
fn rgba_to_rgb(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(4)
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

/// Convert RGB to grayscale using luminance weights.
fn rgb_to_gray(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(3)
        .map(|px| {
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b).round() as u8
        })
        .collect()
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
    fn same_format_returns_none() {
        let f = make_frame(PixelFormat::Rgb8, vec![0; 12], 2, 2);
        assert!(convert(&f, PixelFormat::Rgb8).is_none());
    }

    #[test]
    fn bgr_to_rgb() {
        let data = vec![10, 20, 30, 40, 50, 60];
        let f = make_frame(PixelFormat::Bgr8, data, 2, 1);
        let converted = convert(&f, PixelFormat::Rgb8).unwrap();
        assert_eq!(converted.format(), PixelFormat::Rgb8);
        assert_eq!(converted.data(), &[30, 20, 10, 60, 50, 40]);
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
