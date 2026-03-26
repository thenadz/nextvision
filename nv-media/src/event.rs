//! Media events produced by the backend and consumed by the source layer.
//!
//! These events drive the source lifecycle state machine (reconnection,
//! EOS handling, health reporting). They are **not** part of the public API —
//! the public surface uses [`HealthEvent`](nv_core::HealthEvent).

use nv_core::error::MediaError;

/// Internal event produced by the GStreamer backend.
///
/// These events are the library-internal vocabulary for what happens inside
/// a media pipeline. The source layer maps them to lifecycle transitions
/// (reconnection, stop) and the runtime maps them to [`HealthEvent`]s.
#[derive(Debug)]
#[allow(dead_code)] // Variants constructed under gst-backend; tested via bus unit tests
pub(crate) enum MediaEvent {
    /// Pipeline reached the Playing state and is producing frames.
    StreamStarted,

    /// End of stream — no more frames will be produced.
    ///
    /// For file sources this is normal termination. For live sources
    /// (RTSP, V4L2) it typically indicates an unexpected server-side closure.
    Eos,

    /// A non-fatal warning from the pipeline (e.g., clock drift, minor
    /// decode glitch). The pipeline continues producing frames.
    Warning {
        message: String,
        debug: Option<String>,
    },

    /// A fatal pipeline error. The session should be torn down and
    /// reconnection attempted if the policy allows.
    Error {
        error: MediaError,
        debug: Option<String>,
    },

    /// A PTS discontinuity was detected in the stream.
    ///
    /// This can indicate a stream restart, server-side seek, network
    /// interruption causing frame loss, or a clock reset. The gap
    /// magnitude is included so downstream consumers (e.g., the view
    /// system) can decide whether to trigger an epoch change.
    Discontinuity {
        /// Absolute gap size in nanoseconds.
        gap_ns: u64,
        /// Previous PTS in nanoseconds.
        prev_pts_ns: u64,
        /// Current PTS in nanoseconds.
        current_pts_ns: u64,
    },
}
