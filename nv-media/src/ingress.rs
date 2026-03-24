//! Public trait contracts for media ingress.
//!
//! These traits define the boundary between the media backend and the rest of
//! the library. They ensure that no backend-specific types leak into the public
//! API, and that alternative media backends can be substituted by implementing
//! [`MediaIngressFactory`].
//!
//! The default implementation is GStreamer-backed (see [`DefaultMediaFactory`](super::DefaultMediaFactory)).
//! All other crates interact through these traits, never through GStreamer types.

use std::sync::Arc;
use std::time::Duration;

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::MediaError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_frame::FrameEnvelope;

use crate::bridge::PtzTelemetry;
use crate::decode::DecodePreference;
use crate::gpu_provider::SharedGpuProvider;
use crate::hook::PostDecodeHook;

/// Where decoded frames reside after the media bridge.
///
/// Controls whether the pipeline tail maps frames to host memory
/// (default), uses the built-in CUDA path, or delegates to a custom
/// [`GpuPipelineProvider`](crate::GpuPipelineProvider).
///
/// # Resolution order
///
/// The pipeline builder resolves the device residency strategy in this
/// order:
///
/// 1. **`Provider`** — the provider builds the pipeline tail and bridges
///    frames.  No built-in CUDA elements are needed.  **If the provider
///    fails, pipeline construction returns an error** — there is no
///    silent fallback to Host, because the user explicitly selected a
///    specific hardware integration.
/// 2. **`Cuda`** — the built-in upstream CUDA path (`cudaupload`,
///    `cudaconvert`).  Requires the `cuda` cargo feature and GStreamer
///    ≥ 1.20.  Falls back to `Host` with a warning if the elements are
///    not available.
/// 3. **`Host`** — standard `videoconvert → appsink` path.  Frames
///    arrive in CPU-accessible mapped memory.
///
/// # Feature gating
///
/// - `Cuda` requires the `cuda` cargo feature on `nv-media`.  Without
///   it, requesting `Cuda` returns `MediaError::Unsupported`.
/// - `Provider` does **not** require the `cuda` feature — the provider
///   decides which GStreamer elements to use.
///
/// # Hardware extension
///
/// To add support for a new hardware backend (e.g., AMD ROCm), implement
/// [`GpuPipelineProvider`](crate::GpuPipelineProvider) in a new crate
/// and pass `DeviceResidency::Provider(Arc::new(MyProvider::new()))`.
/// No changes to the core library are needed.
#[derive(Clone, Default)]
pub enum DeviceResidency {
    /// Frames are mapped to CPU-accessible memory (zero-copy GStreamer
    /// buffer mapping). This is the standard path.
    #[default]
    Host,
    /// Frames remain on the GPU as CUDA device memory via the built-in
    /// upstream GStreamer CUDA elements (`cudaupload`, `cudaconvert`).
    ///
    /// Requires the `cuda` cargo feature. On GStreamer < 1.20 (where the
    /// CUDA elements are unavailable), falls back to `Host` with a
    /// warning.
    ///
    /// Stages access the device pointer via
    /// [`FrameEnvelope::accelerated_handle::<CudaBufferHandle>()`](nv_frame::FrameEnvelope::accelerated_handle).
    /// CPU consumers can still call
    /// [`FrameEnvelope::require_host_data()`](nv_frame::FrameEnvelope::require_host_data)
    /// — the materializer downloads device data on first access and
    /// caches the result.
    Cuda,
    /// Frames remain on a device managed by the supplied provider.
    ///
    /// The provider controls pipeline tail construction and frame
    /// bridging.  This variant does **not** require the `cuda` cargo
    /// feature — the provider decides which GStreamer elements and
    /// memory model to use.
    ///
    /// See [`GpuPipelineProvider`](crate::GpuPipelineProvider) for the
    /// trait contract.
    Provider(SharedGpuProvider),
}

impl std::fmt::Debug for DeviceResidency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => write!(f, "Host"),
            Self::Cuda => write!(f, "Cuda"),
            Self::Provider(p) => write!(f, "Provider({})", p.name()),
        }
    }
}

impl DeviceResidency {
    /// Whether this residency requests device-resident frames (non-host).
    #[inline]
    pub fn is_device(&self) -> bool {
        !matches!(self, Self::Host)
    }

    /// Extract the provider reference, if this is the `Provider` variant.
    #[inline]
    pub fn provider(&self) -> Option<&SharedGpuProvider> {
        match self {
            Self::Provider(p) => Some(p),
            _ => None,
        }
    }
}

/// Reported lifecycle state of a media source after a [`tick()`](MediaIngress::tick).
///
/// The runtime uses this to decide whether the feed is still alive,
/// currently recovering, or permanently stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStatus {
    /// The source is running normally — frames are (or will be) flowing.
    Running,
    /// The source is attempting to reconnect after a failure.
    ///
    /// Frames are not flowing. The runtime should keep ticking to drive
    /// the reconnection state machine forward.
    Reconnecting,
    /// The source is permanently stopped (reconnection budget exhausted,
    /// terminal EOS, or explicit stop).
    Stopped,
}

/// Result of a [`MediaIngress::tick()`] call.
///
/// Combines the source's lifecycle status with an optional scheduling hint
/// that tells the runtime how soon the next tick is needed.
#[derive(Debug, Clone)]
pub struct TickOutcome {
    /// Current lifecycle state of the source.
    pub status: SourceStatus,
    /// Suggested delay before the next `tick()` call.
    ///
    /// - `Some(d)` — the source has a pending deadline (e.g., a reconnect
    ///   backoff that expires in `d`). The runtime should arrange to tick
    ///   again after at most `d`.
    /// - `None` — no specific urgency. The runtime will wait indefinitely
    ///   for the next frame, EOS, or error — no periodic tick occurs.
    ///   Sources that rely on polling to discover state changes (e.g., a
    ///   stopped flag) **must** provide a `Some` hint.
    pub next_tick: Option<Duration>,
}

impl TickOutcome {
    /// Source is running, no specific tick urgency.
    #[inline]
    pub fn running() -> Self {
        Self {
            status: SourceStatus::Running,
            next_tick: None,
        }
    }

    /// Source is reconnecting; tick again after `delay`.
    #[inline]
    pub fn reconnecting(delay: Duration) -> Self {
        Self {
            status: SourceStatus::Reconnecting,
            next_tick: Some(delay),
        }
    }

    /// Source is permanently stopped.
    #[inline]
    pub fn stopped() -> Self {
        Self {
            status: SourceStatus::Stopped,
            next_tick: None,
        }
    }
}

/// Trait contract for a media ingress source.
///
/// Abstracts source lifecycle so the runtime and pipeline code do not depend
/// on GStreamer internals. The GStreamer-backed `MediaSource` implements this
/// trait within `nv-media`.
///
/// Each `MediaIngress` instance handles exactly one feed. Implementations must
/// be `Send` (moved onto the feed's I/O thread at startup) but need not be `Sync`.
///
/// # Lifecycle
///
/// 1. `start(sink)` — begin producing frames, delivering them via `sink`.
/// 2. Frames flow: decoded frames are pushed to `sink.on_frame()`.
/// 3. The runtime calls `tick()` on every processed frame **and** whenever
///    the frame queue times out according to [`TickOutcome::next_tick`].
///    This is event-driven: when `next_tick` is `None`, the runtime waits
///    indefinitely for the next frame, EOS, or error — there is **no**
///    fixed polling interval.
/// 4. `pause()` / `resume()` — temporarily halt/restart frame production.
/// 5. `stop()` — tear down the source and release all resources.
///
/// # Tick scheduling
///
/// The runtime does **not** poll at a fixed interval. Tick frequency is
/// determined entirely by frame arrivals and the `next_tick` hint:
///
/// - When frames are flowing, `tick()` is called after each frame.
/// - When the queue is idle and `next_tick` is `Some(d)`, the runtime
///   wakes after `d` to call `tick()`.
/// - When the queue is idle and `next_tick` is `None`, the runtime
///   sleeps indefinitely — only a new frame, `on_error()`, `on_eos()`,
///   or shutdown will wake it.
///
/// Sources that need periodic management (e.g., reconnection with
/// backoff) **must** return `Some(remaining)` in `next_tick` so the
/// runtime wakes at the right time. Sources that are purely
/// frame-driven (e.g., test doubles) can return `None`.
pub trait MediaIngress: Send + 'static {
    /// Begin producing frames.
    ///
    /// Decoded frames are delivered to `sink`. The implementation may spawn
    /// internal threads or use the calling thread, depending on the backend.
    ///
    /// On success the source enters the `Running` state immediately.
    /// On failure the source remains in the `Idle` state and the caller
    /// (typically the runtime) may retry. The returned [`MediaError`] carries
    /// the typed reason for the failure (e.g., `Unsupported` when the
    /// backend is not linked, `ConnectionFailed` when the stream cannot
    /// be reached).
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if the pipeline cannot be constructed or the
    /// source cannot be connected on initial attempt.
    fn start(&mut self, sink: Box<dyn FrameSink>) -> Result<(), MediaError>;

    /// Stop the source and release all backend resources.
    ///
    /// After `stop()`, the source may not be restarted.
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if teardown encounters an issue.
    fn stop(&mut self) -> Result<(), MediaError>;

    /// Pause frame production without releasing the connection.
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if the backend does not support pausing or
    /// the source is not in a running state.
    fn pause(&mut self) -> Result<(), MediaError>;

    /// Resume frame production after a pause.
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if the source is not paused.
    fn resume(&mut self) -> Result<(), MediaError>;

    /// Drive internal lifecycle management: poll the backend bus for
    /// errors/events and advance the reconnection state machine.
    ///
    /// Called by the runtime after each processed frame and whenever
    /// the frame queue times out according to the previous
    /// [`TickOutcome::next_tick`] hint. There is no fixed polling
    /// interval — tick frequency is entirely event-driven.
    ///
    /// Implementations should:
    ///
    /// 1. Drain pending bus messages and process lifecycle events.
    /// 2. If in a reconnecting state and the backoff deadline has elapsed,
    ///    attempt reconnection.
    /// 3. Return a [`TickOutcome`] carrying the current status and an
    ///    optional hint for when the next tick is needed.
    ///
    /// The `next_tick` hint allows the runtime to sleep efficiently.
    /// For example, when reconnecting with a 2-second backoff,
    /// `next_tick` should be `Some(remaining)` so the runtime wakes
    /// exactly when the backoff expires. Returning `None` means the
    /// runtime will wait indefinitely for the next frame/error/EOS.
    ///
    /// Sources that need to poll for state changes **must** provide a
    /// `Some` hint — without one, the runtime will not call `tick()`
    /// again until a frame or error arrives.
    ///
    /// For sources that do not need periodic management (e.g., test
    /// doubles that produce all frames upfront), the default
    /// implementation returns [`TickOutcome::running()`].
    fn tick(&mut self) -> TickOutcome {
        TickOutcome::running()
    }

    /// The source specification this ingress was created from.
    fn source_spec(&self) -> &SourceSpec;

    /// The feed ID this ingress is associated with.
    fn feed_id(&self) -> FeedId;

    /// The effective decode status after backend negotiation.
    ///
    /// Returns `None` if no decode decision has been made yet (the stream
    /// has not started, or the backend does not report decoder identity).
    ///
    /// The tuple contains `(outcome, backend_detail)` where
    /// `backend_detail` is an opaque diagnostic string (e.g., the
    /// GStreamer element name). Do not match on its contents.
    fn decode_status(&self) -> Option<(nv_core::health::DecodeOutcome, String)> {
        None
    }
}

/// Receives decoded frames from a [`MediaIngress`] source.
///
/// Implemented by the runtime's pipeline executor. The media backend calls
/// these methods on its internal thread — implementations must be
/// `Send + Sync` because the callback may be invoked from a GStreamer
/// streaming thread.
pub trait FrameSink: Send + Sync + 'static {
    /// A new decoded frame is available.
    fn on_frame(&self, frame: FrameEnvelope);

    /// The source encountered an error (e.g., transient decode failure,
    /// connection loss, or timeout).
    ///
    /// The error preserves the original typed [`MediaError`] variant so
    /// the receiver can distinguish connection failures from decode errors,
    /// timeouts, etc. without parsing strings.
    fn on_error(&self, error: MediaError);

    /// End of stream — the source has no more frames to produce from
    /// the current session.
    ///
    /// Called by the source FSM only when the session is definitively
    /// over: a non-looping file reaching its end, or a live source whose
    /// reconnection budget is exhausted. This is a terminal signal.
    ///
    /// Implementations should close the frame queue so the worker thread
    /// observes the `Closed` state and exits the processing loop.
    fn on_eos(&self);

    /// Wake the consumer thread for control-plane processing.
    ///
    /// Called by the backend when a lifecycle-relevant bus event occurs
    /// (error, EOS) to ensure the consumer thread ticks the source
    /// promptly — even when no frames are flowing and the queue pop
    /// has no deadline.
    ///
    /// The default implementation is a no-op. Implementations backed by
    /// a frame queue should notify the queue's consumer condvar.
    fn wake(&self) {}
}

/// Configuration bundle passed to [`MediaIngressFactory::create()`].
///
/// Replaces positional arguments with a named struct so new options can
/// be added without breaking the trait signature.
///
/// Construct via [`IngressOptions::new()`] and the `with_*` builder
/// methods. The struct is `#[non_exhaustive]`, so adding fields is
/// not a semver break.
///
/// # Examples
///
/// ```
/// use nv_core::config::{ReconnectPolicy, SourceSpec};
/// use nv_core::id::FeedId;
/// use nv_media::{DecodePreference, DeviceResidency, IngressOptions};
///
/// let options = IngressOptions::new(
///         FeedId::new(1),
///         SourceSpec::rtsp("rtsp://cam/stream"),
///         ReconnectPolicy::default(),
///     )
///     .with_decode_preference(DecodePreference::CpuOnly)
///     .with_device_residency(DeviceResidency::Host);
/// ```
#[non_exhaustive]
pub struct IngressOptions {
    /// Feed identifier.
    pub feed_id: FeedId,
    /// Source specification (RTSP URL, file path, etc.).
    pub spec: SourceSpec,
    /// Reconnection policy for the source.
    pub reconnect: ReconnectPolicy,
    /// Optional PTZ telemetry provider.
    pub ptz_provider: Option<Arc<dyn PtzProvider>>,
    /// Decode preference — controls hardware vs. software decode selection.
    pub decode_preference: DecodePreference,
    /// Optional post-decode hook — can inject a pipeline element between
    /// the decoder and the color-space converter.
    ///
    /// **Ignored when `device_residency` is `Provider`** — the provider
    /// controls the full decoder-to-tail path. Hooks are only evaluated
    /// for `Host` and `Cuda` residency modes.
    pub post_decode_hook: Option<PostDecodeHook>,
    /// Maximum number of media events buffered before new events are
    /// dropped. Default: 64.
    pub event_queue_capacity: usize,
    /// Where decoded frames reside after the bridge.
    ///
    /// [`DeviceResidency::Host`] (default) maps frames to CPU memory.
    /// [`DeviceResidency::Cuda`] uses the built-in CUDA path.
    /// [`DeviceResidency::Provider`] delegates to a custom provider.
    ///
    /// See [`DeviceResidency`] for resolution semantics.
    pub device_residency: DeviceResidency,
}

impl IngressOptions {
    /// Create a new options bundle with required fields.
    ///
    /// Optional fields default to:
    /// - `ptz_provider`: `None`
    /// - `decode_preference`: [`DecodePreference::Auto`]
    #[must_use]
    pub fn new(feed_id: FeedId, spec: SourceSpec, reconnect: ReconnectPolicy) -> Self {
        Self {
            feed_id,
            spec,
            reconnect,
            ptz_provider: None,
            decode_preference: DecodePreference::default(),
            post_decode_hook: None,
            event_queue_capacity: 64,
            device_residency: DeviceResidency::default(),
        }
    }

    /// Attach a PTZ telemetry provider.
    #[must_use]
    pub fn with_ptz_provider(mut self, provider: Arc<dyn PtzProvider>) -> Self {
        self.ptz_provider = Some(provider);
        self
    }

    /// Set the decode preference.
    #[must_use]
    pub fn with_decode_preference(mut self, pref: DecodePreference) -> Self {
        self.decode_preference = pref;
        self
    }

    /// Attach a post-decode hook.
    #[must_use]
    pub fn with_post_decode_hook(mut self, hook: PostDecodeHook) -> Self {
        self.post_decode_hook = Some(hook);
        self
    }

    /// Set the event queue capacity. Default: 64.
    #[must_use]
    pub fn with_event_queue_capacity(mut self, capacity: usize) -> Self {
        self.event_queue_capacity = capacity;
        self
    }

    /// Set the device residency mode for decoded frames.
    ///
    /// [`DeviceResidency::Host`] (default) maps frames to CPU memory.
    /// [`DeviceResidency::Cuda`] uses the built-in CUDA path.
    /// [`DeviceResidency::Provider`] delegates to a custom
    /// [`GpuPipelineProvider`](crate::GpuPipelineProvider).
    ///
    /// See [`DeviceResidency`] for resolution semantics and feature
    /// gating details.
    #[must_use]
    pub fn with_device_residency(mut self, residency: DeviceResidency) -> Self {
        self.device_residency = residency;
        self
    }
}

/// Factory for creating [`MediaIngress`] instances from a source spec.
///
/// The runtime holds one factory and calls `create()` for each new feed.
/// The default implementation ([`DefaultMediaFactory`](super::DefaultMediaFactory))
/// produces backend-specific media sources. Custom implementations can
/// substitute alternative backends or test doubles.
pub trait MediaIngressFactory: Send + Sync + 'static {
    /// Create a new media ingress for the given feed configuration.
    ///
    /// All feed-level options are bundled in [`IngressOptions`], which
    /// includes an optional [`PtzProvider`] for feeds that have external
    /// PTZ telemetry (e.g., ONVIF).
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if the source specification is invalid or
    /// the backend cannot handle the requested format.
    fn create(&self, options: IngressOptions) -> Result<Box<dyn MediaIngress>, MediaError>;
}

/// Receives [`HealthEvent`]s emitted by a [`MediaIngress`] source.
///
/// The runtime typically provides an implementation that forwards events
/// to subscribers via a channel. The media layer calls these methods from
/// its management thread; implementations must be `Send + Sync`.
pub trait HealthSink: Send + Sync + 'static {
    /// Emit a health event.
    fn emit(&self, event: HealthEvent);
}

/// Provides the latest PTZ telemetry for a feed.
///
/// Implemented by the runtime or an external ONVIF/serial adapter. The
/// appsink callback queries this on every decoded frame to attach telemetry
/// to the frame's [`TypedMetadata`](nv_core::TypedMetadata).
///
/// Must be `Send + Sync` since it is called from the GStreamer streaming thread.
pub trait PtzProvider: Send + Sync + 'static {
    /// Return the latest PTZ telemetry, or `None` if unavailable.
    fn latest(&self) -> Option<PtzTelemetry>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_residency_default_is_host() {
        assert!(matches!(DeviceResidency::default(), DeviceResidency::Host));
    }

    #[test]
    fn device_residency_host_is_not_device() {
        assert!(!DeviceResidency::Host.is_device());
    }

    #[test]
    fn device_residency_cuda_is_device() {
        assert!(DeviceResidency::Cuda.is_device());
    }

    #[test]
    fn device_residency_host_has_no_provider() {
        assert!(DeviceResidency::Host.provider().is_none());
    }

    #[test]
    fn device_residency_cuda_has_no_provider() {
        assert!(DeviceResidency::Cuda.provider().is_none());
    }

    #[test]
    fn device_residency_debug_variants() {
        assert_eq!(format!("{:?}", DeviceResidency::Host), "Host");
        assert_eq!(format!("{:?}", DeviceResidency::Cuda), "Cuda");
    }

    #[test]
    fn device_residency_provider_is_device() {
        use std::sync::Arc;
        use nv_core::error::MediaError;
        use nv_core::id::FeedId;
        use nv_frame::PixelFormat;
        use crate::gpu_provider::GpuPipelineProvider;

        struct StubProvider;
        impl GpuPipelineProvider for StubProvider {
            fn name(&self) -> &str { "stub" }
            #[cfg(feature = "gst-backend")]
            fn build_pipeline_tail(
                &self, _: PixelFormat,
            ) -> Result<crate::gpu_provider::GpuPipelineTail, MediaError> {
                unimplemented!()
            }
            #[cfg(feature = "gst-backend")]
            fn bridge_sample(
                &self, _: FeedId, _: &Arc<std::sync::atomic::AtomicU64>,
                _: PixelFormat, _: &gstreamer::Sample,
                _: Option<crate::PtzTelemetry>,
            ) -> Result<nv_frame::FrameEnvelope, MediaError> {
                unimplemented!()
            }
        }

        let p = DeviceResidency::Provider(Arc::new(StubProvider));
        assert!(p.is_device());
        assert!(p.provider().is_some());
        assert_eq!(p.provider().unwrap().name(), "stub");
        assert_eq!(format!("{p:?}"), "Provider(stub)");
    }

    fn test_feed_id() -> nv_core::id::FeedId {
        nv_core::id::FeedId::new(1)
    }

    fn test_reconnect() -> nv_core::config::ReconnectPolicy {
        nv_core::config::ReconnectPolicy::default()
    }

    #[test]
    fn ingress_options_default_residency_is_host() {
        let opts = IngressOptions::new(
            test_feed_id(),
            SourceSpec::file("/tmp/test.mp4"),
            test_reconnect(),
        );
        assert!(matches!(opts.device_residency, DeviceResidency::Host));
    }

    #[test]
    fn ingress_options_with_device_residency() {
        let opts = IngressOptions::new(
            test_feed_id(),
            SourceSpec::file("/tmp/test.mp4"),
            test_reconnect(),
        )
            .with_device_residency(DeviceResidency::Cuda);
        assert!(matches!(opts.device_residency, DeviceResidency::Cuda));
    }
}
