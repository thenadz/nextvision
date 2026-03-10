//! GstSample → FrameEnvelope bridge.
//!
//! This module is the **single crossing point** between GStreamer buffer
//! types and the library's frame abstraction.
//!
//! # Zero-copy path (when GStreamer is linked)
//!
//! [`bridge_gst_sample()`] extracts the buffer from a `gst::Sample`, clones
//! the ref-counted buffer handle, maps it read-only, and wraps the mapping
//! in a `FrameEnvelope` with `PixelData::Mapped`. When the last
//! `Arc<FrameInner>` drops, the mapping drops, releasing the GStreamer
//! buffer back to its pool. No pixel data is copied.
//!
//! # Owned-copy fallback
//!
//! [`bridge_sample()`] is the non-GStreamer path that takes pre-extracted
//! metadata and pixel bytes (used by tests and synthetic feeds).
//!
//! # PTZ metadata hook
//!
//! PTZ telemetry uses [`nv_view::PtzTelemetry`] — the canonical type from
//! the view crate — ensuring end-to-end type alignment. If telemetry is
//! present, it is stored in the frame's [`TypedMetadata`](nv_core::TypedMetadata).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nv_core::TypedMetadata;
use nv_core::error::MediaError;
use nv_core::id::FeedId;
use nv_core::timestamp::{MonotonicTs, WallTs};
use nv_frame::{FrameEnvelope, PixelFormat};

// Re-export PtzTelemetry from nv-view — this is the canonical PTZ type
// used everywhere in the library. No separate definition in nv-media.
pub use nv_view::PtzTelemetry;

use crate::pipeline::OutputFormat;

/// Metadata extracted from a GStreamer sample's caps and buffer.
///
/// The bridge parses `GstVideoInfo` into this struct before constructing the
/// `FrameEnvelope`. All GStreamer types are erased at this boundary.
#[derive(Debug, Clone)]
pub(crate) struct SampleMetadata {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride: u32,
    /// Pixel format (mapped from `GstVideoFormat`).
    pub format: PixelFormat,
    /// Presentation timestamp from the GStreamer buffer (nanoseconds).
    pub pts_ns: u64,
}

/// Bridge a real GStreamer sample into a [`FrameEnvelope`] (zero-copy).
///
/// Extracts caps metadata, maps the buffer read-only, and wraps the
/// mapped buffer in a `FrameEnvelope`. The GStreamer buffer is kept alive
/// inside the frame's `PinGuard` and released when the last clone drops.
///
/// # Arguments
///
/// * `feed_id` — ID of the feed this frame belongs to.
/// * `seq` — shared atomic counter for monotonic frame sequencing.
/// * `output_format` — expected output format (for stride calculation fallback).
/// * `sample` — the GStreamer sample from the appsink callback.
/// * `ptz` — optional PTZ telemetry to attach to the frame's metadata.
///   PTZ data typically comes from an external provider (ONVIF, serial)
///   rather than from the GStreamer sample itself.
#[cfg(feature = "gst-backend")]
pub(crate) fn bridge_gst_sample(
    feed_id: FeedId,
    seq: &Arc<AtomicU64>,
    output_format: OutputFormat,
    sample: &gstreamer::Sample,
    ptz: Option<PtzTelemetry>,
) -> Result<FrameEnvelope, MediaError> {
    let caps = sample.caps().ok_or_else(|| MediaError::DecodeFailed {
        detail: "sample has no caps".into(),
    })?;

    let video_info =
        gstreamer_video::VideoInfo::from_caps(caps).map_err(|e| MediaError::DecodeFailed {
            detail: format!("failed to parse VideoInfo from caps: {e}"),
        })?;

    let width = video_info.width();
    let height = video_info.height();
    let stride = video_info.stride()[0] as u32;
    let format = output_format.to_pixel_format();

    let buffer = sample.buffer().ok_or_else(|| MediaError::DecodeFailed {
        detail: "sample has no buffer".into(),
    })?;

    let pts_ns = buffer.pts().map(|pts| pts.nseconds()).unwrap_or(0);

    let ts = MonotonicTs::from_nanos(pts_ns);
    let wall_ts = WallTs::now();
    let frame_seq = seq.fetch_add(1, Ordering::Relaxed);

    // Clone the ref-counted GStreamer buffer (cheap — increments the refcount)
    // and map it read-only. The `MappedBuffer` keeps the mapping alive; when
    // the frame's PinGuard drops, it releases the buffer back to GStreamer's
    // pool. No pixel data is copied.
    let owned_buffer = buffer.copy();
    let map = owned_buffer
        .into_mapped_buffer_readable()
        .map_err(|_| MediaError::DecodeFailed {
            detail: "failed to map buffer read-only".into(),
        })?;

    let ptr = map.as_slice().as_ptr();
    let len = map.size();

    let mut metadata = TypedMetadata::new();
    if let Some(telemetry) = ptz {
        metadata.insert(telemetry);
    }

    // SAFETY: `ptr` points to `len` readable bytes inside the mapped
    // GStreamer buffer. The `map` (moved into the PinGuard) keeps the
    // mapping and buffer alive for the lifetime of the frame and all its
    // clones. The data is immutable while mapped.
    Ok(unsafe {
        FrameEnvelope::new_mapped(
            feed_id,
            frame_seq,
            ts,
            wall_ts,
            width,
            height,
            format,
            stride,
            ptr,
            len,
            Box::new(map),
            metadata,
        )
    })
}

/// Bridge a decoded sample into a [`FrameEnvelope`] (owned-copy path).
///
/// Used when GStreamer's zero-copy mapping is not available (tests,
/// non-GStreamer builds, synthetic frames).
pub(crate) fn bridge_sample(
    feed_id: FeedId,
    seq: u64,
    meta: &SampleMetadata,
    wall_ts: WallTs,
    pixel_data: &[u8],
    ptz: Option<PtzTelemetry>,
) -> Result<FrameEnvelope, MediaError> {
    let ts = MonotonicTs::from_nanos(meta.pts_ns);

    let expected_len = (meta.stride as usize) * (meta.height as usize);
    if pixel_data.len() < expected_len {
        return Err(MediaError::DecodeFailed {
            detail: format!(
                "pixel data too short: got {} bytes, expected at least {} ({}×{})",
                pixel_data.len(),
                expected_len,
                meta.stride,
                meta.height,
            ),
        });
    }

    let mut metadata = TypedMetadata::new();
    if let Some(telemetry) = ptz {
        metadata.insert(telemetry);
    }

    Ok(FrameEnvelope::new_owned(
        feed_id,
        seq,
        ts,
        wall_ts,
        meta.width,
        meta.height,
        meta.format,
        meta.stride,
        pixel_data.to_vec(),
        metadata,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_meta(w: u32, h: u32) -> SampleMetadata {
        let stride = w * 3;
        SampleMetadata {
            width: w,
            height: h,
            stride,
            format: PixelFormat::Rgb8,
            pts_ns: 1_000_000,
        }
    }

    #[test]
    fn bridge_sample_round_trip() {
        let meta = rgb_meta(2, 2);
        let pixels = vec![128u8; 12]; // 2×2 RGB
        let frame = bridge_sample(
            FeedId::new(1),
            0,
            &meta,
            WallTs::from_micros(0),
            &pixels,
            None,
        )
        .unwrap();
        assert_eq!(frame.width(), 2);
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.format(), PixelFormat::Rgb8);
        assert_eq!(frame.data().len(), 12);
    }

    #[test]
    fn bridge_rejects_short_data() {
        let meta = rgb_meta(2, 2);
        let pixels = vec![0u8; 6]; // too short
        let result = bridge_sample(
            FeedId::new(1),
            0,
            &meta,
            WallTs::from_micros(0),
            &pixels,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn bridge_with_ptz_telemetry() {
        let meta = rgb_meta(2, 2);
        let pixels = vec![255u8; 12];
        let ptz = PtzTelemetry {
            pan: 45.0,
            tilt: -10.0,
            zoom: 0.5,
            ts: MonotonicTs::from_nanos(1_000_000),
        };
        let frame = bridge_sample(
            FeedId::new(1),
            0,
            &meta,
            WallTs::from_micros(0),
            &pixels,
            Some(ptz),
        )
        .unwrap();
        let stored = frame.metadata().get::<PtzTelemetry>().unwrap();
        assert!((stored.pan - 45.0).abs() < f32::EPSILON);
        assert!((stored.zoom - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn monotonic_ts_from_pts() {
        let meta = SampleMetadata {
            width: 1,
            height: 1,
            stride: 3,
            format: PixelFormat::Rgb8,
            pts_ns: 5_000_000_000, // 5 seconds
        };
        let pixels = vec![0u8; 3];
        let frame = bridge_sample(
            FeedId::new(1),
            0,
            &meta,
            WallTs::from_micros(0),
            &pixels,
            None,
        )
        .unwrap();
        assert_eq!(frame.ts().as_nanos(), 5_000_000_000);
    }
}
