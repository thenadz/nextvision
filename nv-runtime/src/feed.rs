//! Feed configuration and handle.

use std::sync::Arc;

use nv_core::config::{CameraMode, ReconnectPolicy, SourceSpec};
use nv_core::error::{ConfigError, NvError};
use nv_core::id::FeedId;
use nv_core::metrics::FeedMetrics;
use nv_media::PtzProvider;
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
    pub(crate) ptz_provider: Option<Arc<dyn PtzProvider>>,
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
    ptz_provider: Option<Arc<dyn PtzProvider>>,
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
            ptz_provider: None,
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

    /// Set an optional PTZ telemetry provider.
    ///
    /// When provided, the media backend queries this on every decoded
    /// frame to attach PTZ telemetry to the frame metadata.
    #[must_use]
    pub fn ptz_provider(mut self, provider: Arc<dyn PtzProvider>) -> Self {
        self.ptz_provider = Some(provider);
        self
    }

    /// Build the feed configuration.
    ///
    /// # Errors
    ///
    /// - `MissingRequired` if `source`, `camera_mode`, `stages`, or `output_sink` are not set.
    /// - `CameraModeConflict` if `Observed` is set without a provider, or `Fixed` with a provider.
    pub fn build(self) -> Result<FeedConfig, NvError> {
        let source = self
            .source
            .ok_or(ConfigError::MissingRequired { field: "source" })?;
        let camera_mode = self.camera_mode.ok_or(ConfigError::MissingRequired {
            field: "camera_mode",
        })?;
        let stages = self
            .stages
            .ok_or(ConfigError::MissingRequired { field: "stages" })?;
        if stages.is_empty() {
            return Err(ConfigError::InvalidPolicy {
                detail: "at least one perception stage is required".into(),
            }
            .into());
        }
        let output_sink = self.output_sink.ok_or(ConfigError::MissingRequired {
            field: "output_sink",
        })?;

        // Validate queue depth.
        if self.backpressure.queue_depth() == 0 {
            return Err(ConfigError::InvalidCapacity {
                field: "queue_depth",
            }
            .into());
        }

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
            ptz_provider: self.ptz_provider,
        })
    }
}

/// Handle to a running feed.
///
/// Provides per-feed monitoring and control: health events, metrics,
/// pause/resume, and stop.
///
/// Backed by `Arc<FeedSharedState>` — metrics are read from the same
/// atomic counters that the feed worker thread writes.
pub struct FeedHandle {
    shared: std::sync::Arc<crate::worker::FeedSharedState>,
}

impl FeedHandle {
    /// Create a feed handle (internal — constructed by the runtime).
    pub(crate) fn new(shared: std::sync::Arc<crate::worker::FeedSharedState>) -> Self {
        Self { shared }
    }

    /// The feed's unique identifier.
    #[must_use]
    pub fn id(&self) -> FeedId {
        self.shared.id
    }

    /// Whether the feed is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.shared
            .paused
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the worker thread is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.shared.alive.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get a snapshot of the feed's current metrics.
    ///
    /// Reads live atomic counters maintained by the feed worker thread.
    #[must_use]
    pub fn metrics(&self) -> FeedMetrics {
        self.shared.metrics()
    }

    /// Pause the feed (stop pulling frames from source; stages idle).
    ///
    /// Uses a condvar to wake the worker without spin-sleeping.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed is already paused.
    pub fn pause(&self) -> Result<(), nv_core::NvError> {
        let was_paused = self
            .shared
            .paused
            .swap(true, std::sync::atomic::Ordering::Relaxed);
        if was_paused {
            return Err(nv_core::NvError::Runtime(
                nv_core::error::RuntimeError::AlreadyPaused,
            ));
        }
        // Mirror into the condvar-guarded bool.
        let (lock, _cvar) = &self.shared.pause_condvar;
        *lock.lock().unwrap() = true;
        Ok(())
    }

    /// Resume a paused feed.
    ///
    /// Notifies the worker thread via condvar so it wakes immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed is not paused.
    pub fn resume(&self) -> Result<(), nv_core::NvError> {
        let was_paused = self
            .shared
            .paused
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        if !was_paused {
            return Err(nv_core::NvError::Runtime(
                nv_core::error::RuntimeError::NotPaused,
            ));
        }
        // Mirror into the condvar-guarded bool and wake the worker.
        let (lock, cvar) = &self.shared.pause_condvar;
        *lock.lock().unwrap() = false;
        cvar.notify_one();
        Ok(())
    }
}
