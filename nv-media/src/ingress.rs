//! Public trait contracts for media ingress.
//!
//! These traits define the boundary between the media backend (GStreamer) and
//! the rest of the library. They ensure that no GStreamer types leak into the
//! public API, and that alternative media backends could be substituted.

use std::sync::Arc;

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::MediaError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_frame::FrameEnvelope;

use crate::bridge::PtzTelemetry;

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
/// 3. `pause()` / `resume()` — temporarily halt/restart frame production.
/// 4. `stop()` — tear down the source and release all resources.
///
/// Reconnection with backoff is handled internally by the implementation
/// according to the configured [`ReconnectPolicy`].
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

    /// The source specification this ingress was created from.
    fn source_spec(&self) -> &SourceSpec;

    /// The feed ID this ingress is associated with.
    fn feed_id(&self) -> FeedId;
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

    /// End of stream — the source has no more frames to produce.
    ///
    /// This signals clean termination (e.g., a file source reaching its end).
    /// After `on_eos()`, no further `on_frame()` calls will occur unless the
    /// source is restarted.
    fn on_eos(&self);
}

/// Factory for creating [`MediaIngress`] instances from a source spec.
///
/// The runtime holds one factory and calls `create()` for each new feed.
/// The default factory produces GStreamer-backed media sources.
pub trait MediaIngressFactory: Send + Sync + 'static {
    /// Create a new media ingress for the given feed and source specification.
    ///
    /// An optional [`PtzProvider`] can be supplied for feeds that have
    /// external PTZ telemetry (e.g., ONVIF). The provider is queried on
    /// every decoded frame so that PTZ metadata can be attached.
    ///
    /// # Errors
    ///
    /// Returns `MediaError` if the source specification is invalid or
    /// the backend cannot handle the requested format.
    fn create(
        &self,
        feed_id: FeedId,
        spec: SourceSpec,
        reconnect: ReconnectPolicy,
        ptz_provider: Option<Arc<dyn PtzProvider>>,
    ) -> Result<Box<dyn MediaIngress>, MediaError>;
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
