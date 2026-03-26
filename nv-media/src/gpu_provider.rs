//! GPU pipeline provider — extension point for platform-specific GPU residency.
//!
//! [`GpuPipelineProvider`] is the public trait that platform-specific crates
//! implement to provide GPU-resident frame delivery through GStreamer.
//! The built-in CUDA path (via `DeviceResidency::Cuda`) uses upstream
//! GStreamer CUDA elements (`cudaupload`, `cudaconvert`), available on
//! GStreamer >= 1.20 (including JetPack 6 / L4T R36).  For older
//! platforms (e.g., JetPack 5.x), external crates like `nv-jetson`
//! implement this trait to provide an alternative GPU memory path.
//!
//! External crates (e.g., `nv-jetson`) implement this trait to support
//! hardware where the upstream CUDA elements are not available — for
//! example, NVMM-based GPU residency on JetPack 5.x (GStreamer 1.16).
//!
//! # GStreamer dependency
//!
//! This trait takes GStreamer types (`gstreamer::Sample`) in its method
//! signatures because it operates at the media backend boundary. External
//! crates implementing this trait explicitly opt into the `gstreamer`
//! dependency. This is the deliberate extension surface for the GStreamer
//! backend — upstream of this point, no GStreamer types are visible.
//!
//! # Pipeline topology
//!
//! The provider controls two parts of the per-feed pipeline:
//!
//! 1. **Pipeline tail** — the GStreamer elements between the decoder and
//!    the appsink (`build_pipeline_tail`). For upstream CUDA this is
//!    `cudaupload → cudaconvert → appsink(CUDAMemory)`. For Jetson NVMM
//!    it might be `nvvidconv → appsink(NVMM)` or just `appsink(NVMM)`.
//!
//! 2. **Frame bridge** — the function that converts a `GstSample` into a
//!    `FrameEnvelope` with device-resident
//!    pixel data (`bridge_sample`).
//!
//! The provider controls the full pipeline tail; any decoder-to-tail
//! bridging elements should be included as the first element(s) in
//! the tail returned by `build_pipeline_tail`.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use nv_core::error::MediaError;
use nv_core::id::FeedId;
use nv_frame::FrameEnvelope;
use nv_frame::frame::PixelFormat;

use crate::bridge::PtzTelemetry;

/// Result of [`GpuPipelineProvider::build_pipeline_tail`].
///
/// Contains the GStreamer elements that form the pipeline segment between
/// the decoder (or post-decode hook) and the appsink, plus the configured
/// appsink itself.
#[cfg(feature = "gst-backend")]
pub struct GpuPipelineTail {
    /// Ordered converter elements to insert before the appsink.
    ///
    /// May be empty if the decoder output negotiates directly with the
    /// appsink (e.g., NVMM passthrough).  Elements are linked in order:
    /// `elements[0] → elements[1] → ... → appsink`.
    pub elements: Vec<gstreamer::Element>,
    /// The configured appsink with appropriate caps set.
    pub appsink: gstreamer_app::AppSink,
}

/// Extension point for GPU-resident pipeline construction.
///
/// Platform-specific crates implement this trait to provide tailored
/// pipeline topology and frame bridging for their GPU memory model.
///
/// The built-in CUDA path is available via `DeviceResidency::Cuda`
/// without implementing this trait.  Applications only need a custom
/// provider when the built-in elements are unavailable (e.g., NVMM
/// on JetPack 5.x) or when a different GPU memory model is required.
///
/// # Thread safety
///
/// Implementations must be `Send + Sync` because the provider is shared
/// between the pipeline-building code (source management thread) and the
/// appsink callback (GStreamer streaming thread) via `Arc`.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use nv_media::gpu_provider::GpuPipelineProvider;
/// use nv_media::DeviceResidency;
///
/// let provider: Arc<dyn GpuPipelineProvider> = Arc::new(MyJetsonProvider::new());
///
/// let config = FeedConfig::builder()
///     .device_residency(DeviceResidency::Provider(provider))
///     // ...
///     .build()?;
/// ```
pub trait GpuPipelineProvider: Send + Sync {
    /// Human-readable name for logging and diagnostics.
    fn name(&self) -> &str;

    /// Build the GPU pipeline tail (converter elements + appsink).
    ///
    /// Called once per session during pipeline construction. The returned
    /// elements are added to the pipeline and linked in order, followed
    /// by the appsink.
    ///
    /// # Arguments
    ///
    /// * `pixel_format` — the target pixel format for the appsink caps.
    ///
    /// # Errors
    ///
    /// Return `MediaError::Unsupported` if the required GStreamer elements
    /// or capabilities are not available at runtime.
    #[cfg(feature = "gst-backend")]
    fn build_pipeline_tail(&self, pixel_format: PixelFormat)
    -> Result<GpuPipelineTail, MediaError>;

    /// Bridge a GStreamer sample into a device-resident [`FrameEnvelope`].
    ///
    /// Called on every frame from the appsink streaming thread. Must be
    /// efficient — allocations and blocking should be minimized.
    ///
    /// The returned `FrameEnvelope` should carry `PixelData::Device` with
    /// a platform-specific handle (e.g., `CudaBufferHandle` or
    /// `NvmmBufferHandle`) and optionally a `HostMaterializeFn` for
    /// transparent CPU fallback.
    #[cfg(feature = "gst-backend")]
    fn bridge_sample(
        &self,
        feed_id: FeedId,
        seq: &Arc<AtomicU64>,
        pixel_format: PixelFormat,
        sample: &gstreamer::Sample,
        ptz: Option<PtzTelemetry>,
    ) -> Result<FrameEnvelope, MediaError>;
}

/// Shared handle to a [`GpuPipelineProvider`].
///
/// Used by `IngressOptions`,
/// `SessionConfig`, and the pipeline
/// builder.
pub type SharedGpuProvider = Arc<dyn GpuPipelineProvider>;

// ---------------------------------------------------------------------------
// Provider authoring helpers
// ---------------------------------------------------------------------------

/// Pre-extracted metadata from a GStreamer sample.
///
/// Providers call [`SampleInfo::extract()`] at the top of their
/// [`bridge_sample`](GpuPipelineProvider::bridge_sample) implementation
/// to avoid re-deriving width/height/stride/timestamps from raw
/// GStreamer types.
///
/// # Example
///
/// ```rust,ignore
/// fn bridge_sample(
///     &self,
///     feed_id: FeedId,
///     seq: &Arc<AtomicU64>,
///     pixel_format: PixelFormat,
///     sample: &gstreamer::Sample,
///     ptz: Option<PtzTelemetry>,
/// ) -> Result<FrameEnvelope, MediaError> {
///     let info = SampleInfo::extract(sample, seq)?;
///     // … platform-specific handle extraction …
///     Ok(info.into_device_envelope(feed_id, pixel_format, handle, Some(materialize), ptz))
/// }
/// ```
#[cfg(feature = "gst-backend")]
pub struct SampleInfo {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Row stride in bytes (from the first video plane).
    pub stride: u32,
    /// Monotonic presentation timestamp.
    pub ts: nv_core::timestamp::MonotonicTs,
    /// Wall-clock timestamp captured at extraction time.
    pub wall_ts: nv_core::timestamp::WallTs,
    /// Monotonically increasing frame sequence number.
    pub frame_seq: u64,
    /// Owned copy of the GStreamer buffer (ref-count bump, not a data copy).
    pub buffer: gstreamer::Buffer,
}

#[cfg(feature = "gst-backend")]
impl SampleInfo {
    /// Extract common metadata from a GStreamer sample.
    ///
    /// Parses `VideoInfo` from the sample caps, reads the PTS, bumps
    /// the sequence counter, and captures a wall-clock timestamp.
    pub fn extract(sample: &gstreamer::Sample, seq: &Arc<AtomicU64>) -> Result<Self, MediaError> {
        use std::sync::atomic::Ordering;

        let caps = sample.caps().ok_or_else(|| MediaError::DecodeFailed {
            detail: "sample has no caps".into(),
        })?;

        let video_info =
            gstreamer_video::VideoInfo::from_caps(caps).map_err(|e| MediaError::DecodeFailed {
                detail: format!("failed to parse VideoInfo from caps: {e}"),
            })?;

        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| MediaError::DecodeFailed {
                detail: "sample has no buffer".into(),
            })?;

        let pts_ns = buffer.pts().map(|pts| pts.nseconds()).unwrap_or(0);

        Ok(Self {
            width: video_info.width(),
            height: video_info.height(),
            stride: video_info.stride()[0] as u32,
            ts: nv_core::timestamp::MonotonicTs::from_nanos(pts_ns),
            wall_ts: nv_core::timestamp::WallTs::now(),
            frame_seq: seq.fetch_add(1, Ordering::Relaxed),
            buffer,
        })
    }

    /// Build a device-resident [`FrameEnvelope`] from this sample info.
    ///
    /// Assembles the `TypedMetadata` (including optional PTZ telemetry)
    /// and delegates to [`FrameEnvelope::new_device`].
    pub fn into_device_envelope(
        self,
        feed_id: nv_core::id::FeedId,
        pixel_format: PixelFormat,
        handle: Arc<dyn std::any::Any + Send + Sync>,
        materialize: Option<nv_frame::HostMaterializeFn>,
        ptz: Option<crate::bridge::PtzTelemetry>,
    ) -> FrameEnvelope {
        let mut metadata = nv_core::TypedMetadata::new();
        if let Some(telemetry) = ptz {
            metadata.insert(telemetry);
        }

        FrameEnvelope::new_device(
            feed_id,
            self.frame_seq,
            self.ts,
            self.wall_ts,
            self.width,
            self.height,
            pixel_format,
            self.stride,
            handle,
            materialize,
            metadata,
        )
    }
}
