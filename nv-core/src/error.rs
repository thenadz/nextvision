//! Typed error hierarchy for the NextVision runtime.
//!
//! Every error is typed and informative — no opaque string errors in core flows.
//! Each variant carries enough context for operational debugging.

use crate::id::{FeedId, StageId};

/// Top-level error enum encompassing all NextVision error categories.
#[derive(Debug, thiserror::Error)]
pub enum NvError {
    /// Error originating from the media/source layer.
    #[error("media error: {0}")]
    Media(#[from] MediaError),

    /// Error originating from a perception stage.
    #[error("stage error: {0}")]
    Stage(#[from] StageError),

    /// Error from the temporal state system.
    #[error("temporal error: {0}")]
    Temporal(#[from] TemporalError),

    /// Error from the view/PTZ system.
    #[error("view error: {0}")]
    View(#[from] ViewError),

    /// Error from the runtime/orchestration layer.
    #[error("runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    /// Configuration error — returned at feed/runtime creation time.
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
}

/// Errors from the media ingress layer (source connection, decoding).
///
/// `Clone` is derived so that the same typed error can be delivered to
/// both the health-event path and the frame-sink callback without
/// downgrading one copy to a lossy display string.
///
/// **Security:** The `Display` implementation redacts credentials from
/// URLs and sanitizes untrusted backend strings. This means log output
/// and health events never contain raw secrets.
#[derive(Debug, Clone, thiserror::Error)]
pub enum MediaError {
    /// Failed to connect to the video source.
    #[error("connection failed to `{redacted_url}`: {detail}", redacted_url = crate::security::redact_url(url))]
    ConnectionFailed { url: String, detail: String },

    /// Decoding a video frame failed.
    #[error("decode failed: {detail}")]
    DecodeFailed { detail: String },

    /// End of stream reached (file sources).
    #[error("end of stream")]
    Eos,

    /// Source timed out (no data received within deadline).
    #[error("source timeout")]
    Timeout,

    /// The source format or codec is not supported.
    #[error("unsupported: {detail}")]
    Unsupported { detail: String },

    /// An RTSP source with `RequireTls` policy was given a non-TLS URL.
    #[error("insecure RTSP rejected by RequireTls policy (use rtsps:// or set AllowInsecure)")]
    InsecureRtspRejected,

    /// A `SourceSpec::Custom` pipeline was rejected by the security policy.
    #[error("custom pipeline rejected: set CustomPipelinePolicy::AllowTrusted on the runtime builder to enable custom pipelines")]
    CustomPipelineRejected,
}

/// Errors from perception stages.
///
/// `Clone` is derived so that stage errors can be broadcast through
/// health-event channels without wrapping in `Arc`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StageError {
    /// The stage's processing logic failed.
    #[error("stage `{stage_id}` processing failed: {detail}")]
    ProcessingFailed { stage_id: StageId, detail: String },

    /// The stage ran out of a resource (GPU OOM, buffer limit, etc.).
    #[error("stage `{stage_id}` resource exhausted")]
    ResourceExhausted { stage_id: StageId },

    /// The stage could not load its model or contact an external dependency.
    #[error("stage `{stage_id}` model/dependency load failed: {detail}")]
    ModelLoadFailed { stage_id: StageId, detail: String },
}

/// Errors from the temporal state system.
#[derive(Debug, thiserror::Error)]
pub enum TemporalError {
    /// A referenced track was not found in the temporal store.
    #[error("track not found: {0}")]
    TrackNotFound(crate::id::TrackId),

    /// The retention policy rejected an operation.
    #[error("retention limit exceeded: {detail}")]
    RetentionLimitExceeded { detail: String },
}

/// Errors from the view/PTZ system.
#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    /// The view state provider returned invalid data.
    #[error("invalid motion report: {detail}")]
    InvalidMotionReport { detail: String },

    /// A transform computation failed (e.g., degenerate homography).
    #[error("transform computation failed: {detail}")]
    TransformFailed { detail: String },
}

/// Errors from the runtime/orchestration layer.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The specified feed was not found.
    #[error("feed not found: {feed_id}")]
    FeedNotFound { feed_id: FeedId },

    /// The runtime is already running.
    #[error("runtime is already running")]
    AlreadyRunning,

    /// The feed is already paused.
    #[error("feed is already paused")]
    AlreadyPaused,

    /// The feed is not paused.
    #[error("feed is not paused")]
    NotPaused,

    /// Shutdown is in progress; new operations are rejected.
    #[error("shutdown in progress")]
    ShutdownInProgress,

    /// The maximum number of concurrent feeds has been reached.
    #[error("feed limit exceeded (max: {max})")]
    FeedLimitExceeded { max: usize },

    /// An internal lock is poisoned (a thread panicked while holding it).
    #[error("internal registry lock poisoned")]
    RegistryPoisoned,

    /// Failed to spawn a feed worker thread.
    #[error("thread spawn failed: {detail}")]
    ThreadSpawnFailed { detail: String },
}

/// Configuration errors — returned at feed or runtime construction time.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The source specification is invalid.
    #[error("invalid source: {detail}")]
    InvalidSource { detail: String },

    /// A policy configuration is invalid.
    #[error("invalid policy: {detail}")]
    InvalidPolicy { detail: String },

    /// A required configuration field is missing.
    #[error("missing required field: `{field}`")]
    MissingRequired { field: &'static str },

    /// `CameraMode` and `ViewStateProvider` are inconsistent.
    ///
    /// For example: `Observed` without a provider, or `Fixed` with a provider.
    #[error("camera mode conflict: {detail}")]
    CameraModeConflict { detail: String },

    /// A capacity or depth value is zero (which would deadlock or panic).
    #[error("invalid capacity: {field} must be > 0")]
    InvalidCapacity { field: &'static str },

    /// Stage capability validation failed.
    #[error("stage validation failed: {detail}")]
    StageValidation { detail: String },

    /// A batch coordinator with this processor ID already exists.
    #[error("duplicate batch processor id: {id}")]
    DuplicateBatchProcessorId { id: StageId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv_error_from_media_error() {
        let err: NvError = MediaError::Timeout.into();
        assert!(matches!(err, NvError::Media(MediaError::Timeout)));
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn nv_error_from_config_error() {
        let err: NvError = ConfigError::MissingRequired { field: "source" }.into();
        assert!(matches!(err, NvError::Config(ConfigError::MissingRequired { .. })));
        assert!(err.to_string().contains("source"));
    }

    #[test]
    fn stage_error_includes_stage_id() {
        let err = StageError::ProcessingFailed {
            stage_id: StageId("detector"),
            detail: "NaN output".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("detector"));
        assert!(msg.contains("NaN output"));
    }

    #[test]
    fn runtime_error_display() {
        let err = RuntimeError::FeedLimitExceeded { max: 64 };
        assert!(err.to_string().contains("64"));
    }

    #[test]
    fn media_error_is_clone() {
        let err = MediaError::ConnectionFailed {
            url: "rtsp://cam".into(),
            detail: "timeout".into(),
        };
        let err2 = err.clone();
        assert_eq!(err.to_string(), err2.to_string());
    }
}
