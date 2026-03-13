//! Synthetic frame construction for tests.

use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};
use nv_frame::{FrameEnvelope, PixelFormat};

/// Create a synthetic RGB frame filled with a solid color.
///
/// # Arguments
///
/// - `feed_id` — the feed ID to stamp on the frame.
/// - `seq` — monotonic sequence number.
/// - `ts` — monotonic timestamp.
/// - `width`, `height` — frame dimensions.
/// - `r`, `g`, `b` — fill color.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn solid_rgb(
    feed_id: FeedId,
    seq: u64,
    ts: MonotonicTs,
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
) -> FrameEnvelope {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * 3);
    for _ in 0..pixel_count {
        data.extend_from_slice(&[r, g, b]);
    }
    FrameEnvelope::new_owned(
        feed_id,
        seq,
        ts,
        WallTs::from_micros(0),
        width,
        height,
        PixelFormat::Rgb8,
        width * 3,
        data,
        TypedMetadata::new(),
    )
}

/// Create a synthetic grayscale frame filled with a uniform value.
#[must_use]
pub fn solid_gray(
    feed_id: FeedId,
    seq: u64,
    ts: MonotonicTs,
    width: u32,
    height: u32,
    value: u8,
) -> FrameEnvelope {
    let pixel_count = (width * height) as usize;
    let data = vec![value; pixel_count];
    FrameEnvelope::new_owned(
        feed_id,
        seq,
        ts,
        WallTs::from_micros(0),
        width,
        height,
        PixelFormat::Gray8,
        width,
        data,
        TypedMetadata::new(),
    )
}

/// Create a series of synthetic frames with incrementing sequence numbers
/// and monotonic timestamps spaced by `interval_ns` nanoseconds.
#[must_use]
pub fn frame_sequence(
    feed_id: FeedId,
    count: u64,
    width: u32,
    height: u32,
    interval_ns: u64,
) -> Vec<FrameEnvelope> {
    (0..count)
        .map(|i| {
            solid_rgb(
                feed_id,
                i,
                MonotonicTs::from_nanos(i * interval_ns),
                width,
                height,
                128,
                128,
                128,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_rgb_dimensions() {
        let f = solid_rgb(FeedId::new(1), 0, MonotonicTs::ZERO, 64, 48, 255, 0, 0);
        assert_eq!(f.width(), 64);
        assert_eq!(f.height(), 48);
        assert_eq!(f.data().len(), 64 * 48 * 3);
        // First pixel should be red.
        assert_eq!(&f.data()[..3], &[255, 0, 0]);
    }

    #[test]
    fn frame_sequence_ordering() {
        let frames = frame_sequence(FeedId::new(1), 5, 32, 32, 33_333_333);
        assert_eq!(frames.len(), 5);
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f.seq(), i as u64);
        }
        assert!(frames[1].ts() > frames[0].ts());
    }
}
