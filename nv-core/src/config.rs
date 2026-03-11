//! Configuration types shared across crates.
//!
//! Source specifications, camera modes, reconnection policies, and
//! backoff strategies live here so downstream crates can reference
//! them without depending on the media or runtime crates.

use std::path::PathBuf;

/// Specification of a video source.
///
/// Defined in `nv-core` (not `nv-media`) to prevent downstream crates
/// from transitively depending on GStreamer.
#[derive(Debug, Clone)]
pub enum SourceSpec {
    /// An RTSP stream.
    Rtsp {
        url: String,
        transport: RtspTransport,
    },

    /// A local video file.
    File {
        path: PathBuf,
        /// Whether to loop the file when it reaches the end.
        loop_: bool,
    },

    /// A Video4Linux2 device (Linux only).
    V4l2 { device: String },

    /// Escape hatch: a raw GStreamer launch-line fragment.
    ///
    /// The library constructs the pipeline internally for all other variants.
    /// Use this only for exotic sources not covered above.
    Custom { gst_launch_fragment: String },
}

impl SourceSpec {
    /// Convenience constructor for an RTSP source with TCP transport.
    #[must_use]
    pub fn rtsp(url: impl Into<String>) -> Self {
        Self::Rtsp {
            url: url.into(),
            transport: RtspTransport::Tcp,
        }
    }

    /// Convenience constructor for a local video file (non-looping).
    #[must_use]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File {
            path: path.into(),
            loop_: false,
        }
    }

    /// Convenience constructor for a looping local video file.
    ///
    /// The source seeks back to the start on EOS instead of stopping.
    #[must_use]
    pub fn file_looping(path: impl Into<PathBuf>) -> Self {
        Self::File {
            path: path.into(),
            loop_: true,
        }
    }

    /// Returns `true` if this is a non-looping file source.
    ///
    /// Non-looping file sources treat EOS as terminal (not an error):
    /// the feed stops with [`StopReason::EndOfStream`](crate::health::StopReason::EndOfStream)
    /// rather than attempting a restart.
    #[must_use]
    pub fn is_file_nonloop(&self) -> bool {
        matches!(self, Self::File { loop_: false, .. })
    }
}

/// RTSP transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtspTransport {
    /// TCP interleaved — more reliable over lossy networks.
    Tcp,
    /// UDP unicast — lower latency, may lose packets.
    UdpUnicast,
}

/// Declared camera installation mode.
///
/// This is a **required** field on feed configuration — there is no default.
/// The mode determines whether the view system is engaged.
///
/// See the architecture docs §9 for full details on why this is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    /// Camera is physically fixed (bolted mount, no PTZ, no gimbal).
    ///
    /// The view system is bypassed entirely. No `ViewStateProvider` is needed.
    /// `CameraMotionState` is always `Stable`, `MotionSource` is always `None`.
    Fixed,

    /// Camera may move (PTZ, gimbal, handheld, vehicle-mounted, drone, etc.).
    ///
    /// A `ViewStateProvider` is **required**. If the provider returns no data,
    /// the view system defaults to `CameraMotionState::Unknown` and
    /// `ContextValidity::Degraded` — never to `Stable`.
    Observed,
}

/// Reconnection policy for video sources.
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    /// Maximum reconnection attempts. `0` = infinite retries.
    pub max_attempts: u32,
    /// Base delay between reconnection attempts.
    pub base_delay: std::time::Duration,
    /// Maximum delay (caps exponential backoff).
    pub max_delay: std::time::Duration,
    /// Backoff strategy.
    pub backoff: BackoffKind,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 0,
            base_delay: std::time::Duration::from_secs(1),
            max_delay: std::time::Duration::from_secs(30),
            backoff: BackoffKind::Exponential,
        }
    }
}

/// Backoff strategy for reconnection attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffKind {
    /// Delay doubles each attempt (capped at `max_delay`).
    Exponential,
    /// Delay increases linearly by `base_delay` each attempt.
    Linear,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_looping_creates_looping_file_spec() {
        let spec = SourceSpec::file_looping("/tmp/test.mp4");
        match &spec {
            SourceSpec::File { path, loop_ } => {
                assert_eq!(path.to_str().unwrap(), "/tmp/test.mp4");
                assert!(*loop_, "file_looping should create a looping spec");
            }
            _ => panic!("expected File variant"),
        }
        assert!(!spec.is_file_nonloop());
    }

    #[test]
    fn file_creates_nonlooping_file_spec() {
        let spec = SourceSpec::file("/tmp/test.mp4");
        assert!(spec.is_file_nonloop());
    }

    #[test]
    fn rtsp_creates_tcp_spec() {
        let spec = SourceSpec::rtsp("rtsp://example.com/stream");
        match &spec {
            SourceSpec::Rtsp { url, transport } => {
                assert_eq!(url, "rtsp://example.com/stream");
                assert_eq!(*transport, RtspTransport::Tcp);
            }
            _ => panic!("expected Rtsp variant"),
        }
    }
}
