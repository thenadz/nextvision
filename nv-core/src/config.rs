//! Configuration types shared across crates.
//!
//! Source specifications, camera modes, reconnection policies, and
//! backoff strategies live here so downstream crates can reference
//! them without depending on the media or runtime crates.

use std::fmt;
use std::path::PathBuf;

use crate::security::{RtspSecurityPolicy, redact_url};

/// Specification of a video source.
///
/// Defined in `nv-core` (not `nv-media`) to prevent downstream crates
/// from transitively depending on GStreamer.
#[derive(Clone)]
pub enum SourceSpec {
    /// An RTSP stream.
    ///
    /// The `security` field controls TLS enforcement. The default
    /// ([`PreferTls`](RtspSecurityPolicy::PreferTls)) promotes bare
    /// `rtsp://` URLs to `rtsps://` at pipeline construction time.
    Rtsp {
        url: String,
        transport: RtspTransport,
        /// TLS security policy for this source. Default: `PreferTls`.
        security: RtspSecurityPolicy,
    },

    /// A local video file.
    File {
        path: PathBuf,
        /// Whether to loop the file when it reaches the end.
        loop_: bool,
    },

    /// A Video4Linux2 device (Linux only).
    V4l2 { device: String },

    /// Escape hatch: a raw pipeline launch-line fragment.
    ///
    /// The library constructs the pipeline internally for all other variants.
    /// Use this only for exotic sources not covered above.
    ///
    /// **Security note:** Custom pipelines are gated by
    /// [`CustomPipelinePolicy`](crate::security::CustomPipelinePolicy) at
    /// the runtime layer. The default policy rejects custom pipelines;
    /// set `CustomPipelinePolicy::AllowTrusted` on the runtime builder to
    /// enable them.
    Custom { pipeline_fragment: String },
}

impl SourceSpec {
    /// Convenience constructor for an RTSP source with TCP transport.
    ///
    /// The security policy is inferred from the URL scheme:
    ///
    /// - `rtsps://` or no recognized scheme → [`PreferTls`](RtspSecurityPolicy::PreferTls)
    /// - `rtsp://` → [`AllowInsecure`](RtspSecurityPolicy::AllowInsecure)
    ///
    /// An explicit `rtsp://` scheme is treated as a deliberate choice by
    /// the caller and is **not** promoted to `rtsps://`. Use
    /// [`rtsp_tls`](Self::rtsp_tls) to force TLS promotion on a bare
    /// `rtsp://` URL.
    #[must_use]
    pub fn rtsp(url: impl Into<String>) -> Self {
        let url = url.into();
        // Explicit rtsp:// means the caller chose plaintext.
        let security = if url.starts_with("rtsp://") {
            RtspSecurityPolicy::AllowInsecure
        } else {
            // rtsps://, scheme-less, or unknown → try TLS.
            RtspSecurityPolicy::PreferTls
        };
        Self::Rtsp {
            url,
            transport: RtspTransport::Tcp,
            security,
        }
    }

    /// Convenience constructor that forces [`PreferTls`](RtspSecurityPolicy::PreferTls).
    ///
    /// Unlike [`rtsp()`](Self::rtsp), this promotes bare `rtsp://` URLs to
    /// `rtsps://` at pipeline construction time. Use this when you know
    /// the camera supports TLS but the URL was provided without the
    /// `rtsps://` scheme.
    #[must_use]
    pub fn rtsp_tls(url: impl Into<String>) -> Self {
        Self::Rtsp {
            url: url.into(),
            transport: RtspTransport::Tcp,
            security: RtspSecurityPolicy::PreferTls,
        }
    }

    /// Convenience constructor for an RTSP source with explicit insecure
    /// transport ([`AllowInsecure`](RtspSecurityPolicy::AllowInsecure)).
    ///
    /// Equivalent to [`rtsp()`](Self::rtsp) for `rtsp://` URLs. Useful
    /// when constructing a spec from a variable where you want to
    /// guarantee `AllowInsecure` regardless of the scheme.
    #[must_use]
    pub fn rtsp_insecure(url: impl Into<String>) -> Self {
        Self::Rtsp {
            url: url.into(),
            transport: RtspTransport::Tcp,
            security: RtspSecurityPolicy::AllowInsecure,
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

impl fmt::Debug for SourceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rtsp {
                url,
                transport,
                security,
            } => f
                .debug_struct("Rtsp")
                .field("url", &redact_url(url))
                .field("transport", transport)
                .field("security", security)
                .finish(),
            Self::File { path, loop_ } => f
                .debug_struct("File")
                .field("path", path)
                .field("loop_", loop_)
                .finish(),
            Self::V4l2 { device } => f.debug_struct("V4l2").field("device", device).finish(),
            Self::Custom { .. } => f
                .debug_struct("Custom")
                .field("pipeline_fragment", &"<redacted>")
                .finish(),
        }
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
    use crate::security::RtspSecurityPolicy;

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
    fn rtsp_plain_scheme_infers_allow_insecure() {
        let spec = SourceSpec::rtsp("rtsp://example.com/stream");
        match &spec {
            SourceSpec::Rtsp {
                url,
                transport,
                security,
            } => {
                assert_eq!(url, "rtsp://example.com/stream");
                assert_eq!(*transport, RtspTransport::Tcp);
                assert_eq!(*security, RtspSecurityPolicy::AllowInsecure);
            }
            _ => panic!("expected Rtsp variant"),
        }
    }

    #[test]
    fn rtsp_tls_scheme_infers_prefer_tls() {
        let spec = SourceSpec::rtsp("rtsps://example.com/stream");
        match &spec {
            SourceSpec::Rtsp {
                url,
                transport,
                security,
            } => {
                assert_eq!(url, "rtsps://example.com/stream");
                assert_eq!(*transport, RtspTransport::Tcp);
                assert_eq!(*security, RtspSecurityPolicy::PreferTls);
            }
            _ => panic!("expected Rtsp variant"),
        }
    }

    #[test]
    fn rtsp_no_scheme_infers_prefer_tls() {
        let spec = SourceSpec::rtsp("example.com/stream");
        match &spec {
            SourceSpec::Rtsp { security, .. } => {
                assert_eq!(*security, RtspSecurityPolicy::PreferTls);
            }
            _ => panic!("expected Rtsp variant"),
        }
    }

    #[test]
    fn rtsp_tls_forces_prefer_tls() {
        let spec = SourceSpec::rtsp_tls("rtsp://example.com/stream");
        match &spec {
            SourceSpec::Rtsp { url, security, .. } => {
                assert_eq!(url, "rtsp://example.com/stream");
                assert_eq!(*security, RtspSecurityPolicy::PreferTls);
            }
            _ => panic!("expected Rtsp variant"),
        }
    }

    #[test]
    fn rtsp_insecure_creates_allow_insecure_spec() {
        let spec = SourceSpec::rtsp_insecure("rtsp://example.com/stream");
        match &spec {
            SourceSpec::Rtsp { url, security, .. } => {
                assert_eq!(url, "rtsp://example.com/stream");
                assert_eq!(*security, RtspSecurityPolicy::AllowInsecure);
            }
            _ => panic!("expected Rtsp variant"),
        }
    }
}
