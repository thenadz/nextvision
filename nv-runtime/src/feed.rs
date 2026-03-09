//! Feed configuration and handle.

use nv_core::config::{CameraMode, ReconnectPolicy, SourceSpec};
use nv_core::error::{ConfigError, NvError};
use nv_core::id::FeedId;
use nv_core::metrics::FeedMetrics;
use nv_perception::Stage;
use nv_temporal::RetentionPolicy;
use nv_view::{EpochPolicy, ViewStateProvider};

use crate::backpressure::BackpressurePolicy;
use crate::output::OutputSink;
use crate::shutdown::RestartPolicy;

/// Configuration for a single video feed.
///
/// Constructed via [`FeedConfig::builder()`].
pub struct FeedConfig {
    pub(crate) source: SourceSpec,
    pub(crate) camera_mode: CameraMode,
    pub(crate) stages: Vec<Box<dyn Stage>>,
    pub(crate) view_state_provider: Option<Box<dyn ViewStateProvider>>,
    pub(crate) epoch_policy: Box<dyn EpochPolicy>,
    pub(crate) output_sink: Box<dyn OutputSink>,
    pub(crate) backpressure: BackpressurePolicy,
    pub(crate) temporal: RetentionPolicy,
    pub(crate) reconnect: ReconnectPolicy,
    pub(crate) restart: RestartPolicy,
}

/// Builder for [`FeedConfig`].
///
/// # Required fields
///
/// - `source` — the video source specification.
/// - `camera_mode` — `Fixed` or `Observed` (no default).
/// - `stages` — at least one perception stage.
/// - `output_sink` — where to send processed outputs.
///
/// # Validation
///
/// `build()` validates:
/// - `Observed` mode requires a `ViewStateProvider`.
/// - `Fixed` mode must not have a `ViewStateProvider`.
pub struct FeedConfigBuilder {
    source: Option<SourceSpec>,
    camera_mode: Option<CameraMode>,
    stages: Option<Vec<Box<dyn Stage>>>,
    view_state_provider: Option<Box<dyn ViewStateProvider>>,
    epoch_policy: Option<Box<dyn EpochPolicy>>,
    output_sink: Option<Box<dyn OutputSink>>,
    backpressure: BackpressurePolicy,
    temporal: RetentionPolicy,
    reconnect: ReconnectPolicy,
    restart: RestartPolicy,
}

impl FeedConfig {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> FeedConfigBuilder {
        FeedConfigBuilder {
            source: None,
            camera_mode: None,
            stages: None,
            view_state_provider: None,
            epoch_policy: None,
            output_sink: None,
            backpressure: BackpressurePolicy::default(),
            temporal: RetentionPolicy::default(),
            reconnect: ReconnectPolicy::default(),
            restart: RestartPolicy::default(),
        }
    }
}

impl FeedConfigBuilder {
    /// Set the video source.
    #[must_use]
    pub fn source(mut self, source: SourceSpec) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the camera mode (`Fixed` or `Observed`). **Required.**
    #[must_use]
    pub fn camera_mode(mut self, mode: CameraMode) -> Self {
        self.camera_mode = Some(mode);
        self
    }

    /// Set the ordered list of perception stages.
    #[must_use]
    pub fn stages(mut self, stages: Vec<Box<dyn Stage>>) -> Self {
        self.stages = Some(stages);
        self
    }

    /// Set the view state provider (required for `CameraMode::Observed`).
    #[must_use]
    pub fn view_state_provider(mut self, provider: Box<dyn ViewStateProvider>) -> Self {
        self.view_state_provider = Some(provider);
        self
    }

    /// Set the epoch policy (optional; defaults to `DefaultEpochPolicy`).
    #[must_use]
    pub fn epoch_policy(mut self, policy: Box<dyn EpochPolicy>) -> Self {
        self.epoch_policy = Some(policy);
        self
    }

    /// Set the output sink. **Required.**
    #[must_use]
    pub fn output_sink(mut self, sink: Box<dyn OutputSink>) -> Self {
        self.output_sink = Some(sink);
        self
    }

    /// Set the backpressure policy. Default: `DropOldest { queue_depth: 4 }`.
    #[must_use]
    pub fn backpressure(mut self, policy: BackpressurePolicy) -> Self {
        self.backpressure = policy;
        self
    }

    /// Set the temporal retention policy.
    #[must_use]
    pub fn temporal(mut self, policy: RetentionPolicy) -> Self {
        self.temporal = policy;
        self
    }

    /// Set the reconnection policy.
    #[must_use]
    pub fn reconnect(mut self, policy: ReconnectPolicy) -> Self {
        self.reconnect = policy;
        self
    }

    /// Set the restart policy.
    #[must_use]
    pub fn restart(mut self, policy: RestartPolicy) -> Self {
        self.restart = policy;
        self
    }

    /// Build the feed configuration.
    ///
    /// # Errors
    ///
    /// - `MissingRequired` if `source`, `camera_mode`, `stages`, or `output_sink` are not set.
    /// - `CameraModeConflict` if `Observed` is set without a provider, or `Fixed` with a provider.
    pub fn build(self) -> Result<FeedConfig, NvError> {
        let source = self.source.ok_or(ConfigError::MissingRequired {
            field: "source",
        })?;
        let camera_mode = self.camera_mode.ok_or(ConfigError::MissingRequired {
            field: "camera_mode",
        })?;
        let stages = self.stages.ok_or(ConfigError::MissingRequired {
            field: "stages",
        })?;
        if stages.is_empty() {
            return Err(ConfigError::InvalidPolicy {
                detail: "at least one perception stage is required".into(),
            }
            .into());
        }
        let output_sink = self.output_sink.ok_or(ConfigError::MissingRequired {
            field: "output_sink",
        })?;

        // Validate camera mode vs. provider.
        match camera_mode {
            CameraMode::Observed if self.view_state_provider.is_none() => {
                return Err(ConfigError::CameraModeConflict {
                    detail: "CameraMode::Observed requires a ViewStateProvider".into(),
                }
                .into());
            }
            CameraMode::Fixed if self.view_state_provider.is_some() => {
                return Err(ConfigError::CameraModeConflict {
                    detail: "CameraMode::Fixed must not have a ViewStateProvider".into(),
                }
                .into());
            }
            _ => {}
        }

        // Default epoch policy when none is explicitly provided.
        let epoch_policy = self
            .epoch_policy
            .unwrap_or_else(|| Box::new(nv_view::DefaultEpochPolicy::default()));

        Ok(FeedConfig {
            source,
            camera_mode,
            stages,
            view_state_provider: self.view_state_provider,
            epoch_policy,
            output_sink,
            backpressure: self.backpressure,
            temporal: self.temporal,
            reconnect: self.reconnect,
            restart: self.restart,
        })
    }
}

/// Handle to a running feed.
///
/// Provides per-feed monitoring and control: health events, metrics,
/// pause/resume, and stop.
///
/// The handle is cheaply cloneable (`Arc`-backed shared state).
pub struct FeedHandle {
    inner: std::sync::Arc<FeedInner>,
}

/// Interior state behind `FeedHandle`.
struct FeedInner {
    id: FeedId,
    paused: std::sync::atomic::AtomicBool,
}

impl FeedHandle {
    /// Create a feed handle (internal — constructed by the runtime).
    pub(crate) fn new(id: FeedId) -> Self {
        Self {
            inner: std::sync::Arc::new(FeedInner {
                id,
                paused: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    /// The feed's unique identifier.
    #[must_use]
    pub fn id(&self) -> FeedId {
        self.inner.id
    }

    /// Whether and feed is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.inner
            .paused
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get a snapshot of the feed's current metrics.
    ///
    /// When the full pipeline executor is implemented, this will query
    /// live counters. Currently returns zeroed counters for the correct
    /// feed ID.
    #[must_use]
    pub fn metrics(&self) -> FeedMetrics {
        // Will query pipeline executor counters when implemented.
        FeedMetrics {
            feed_id: self.inner.id,
            frames_received: 0,
            frames_dropped: 0,
            frames_processed: 0,
            tracks_active: 0,
            view_epoch: 0,
            restarts: 0,
        }
    }

    /// Pause the feed (stop pulling frames from source; stages idle).
    ///
    /// # Errors
    ///
    /// Returns an error if the feed is already paused.
    pub fn pause(&self) -> Result<(), nv_core::NvError> {
        let was_paused = self
            .inner
            .paused
            .swap(true, std::sync::atomic::Ordering::Relaxed);
        if was_paused {
            return Err(nv_core::NvError::Runtime(
                nv_core::error::RuntimeError::AlreadyPaused,
            ));
        }
        // Full runtime will signal the feed's I/O thread to pause.
        Ok(())
    }

    /// Resume a paused feed.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed is not paused.
    pub fn resume(&self) -> Result<(), nv_core::NvError> {
        let was_paused = self
            .inner
            .paused
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        if !was_paused {
            return Err(nv_core::NvError::Runtime(
                nv_core::error::RuntimeError::NotPaused,
            ));
        }
        // Full runtime will signal the feed's I/O thread to resume.
        Ok(())
    }
}
