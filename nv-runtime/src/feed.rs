//! Feed configuration and handle.

use std::sync::Arc;
use std::time::Duration;

use nv_core::config::{CameraMode, ReconnectPolicy, SourceSpec};
use nv_core::error::{ConfigError, NvError};
use nv_core::id::FeedId;
use nv_core::metrics::FeedMetrics;
use nv_media::PtzProvider;
use nv_media::DecodePreference;
use nv_perception::{Stage, StagePipeline, ValidationMode, validate_pipeline_phased};
use nv_temporal::RetentionPolicy;
use nv_view::{EpochPolicy, ViewStateProvider};

use crate::backpressure::BackpressurePolicy;
use crate::batch::BatchHandle;
use crate::output::{FrameInclusion, OutputSink};
use crate::pipeline::FeedPipeline;
use crate::shutdown::RestartPolicy;

/// Configuration for a single video feed.
///
/// Constructed via [`FeedConfig::builder()`].
pub struct FeedConfig {
    pub(crate) source: SourceSpec,
    pub(crate) camera_mode: CameraMode,
    pub(crate) stages: Vec<Box<dyn Stage>>,
    pub(crate) batch: Option<BatchHandle>,
    pub(crate) post_batch_stages: Vec<Box<dyn Stage>>,
    pub(crate) view_state_provider: Option<Box<dyn ViewStateProvider>>,
    pub(crate) epoch_policy: Box<dyn EpochPolicy>,
    pub(crate) output_sink: Box<dyn OutputSink>,
    pub(crate) backpressure: BackpressurePolicy,
    pub(crate) temporal: RetentionPolicy,
    pub(crate) reconnect: ReconnectPolicy,
    pub(crate) restart: RestartPolicy,
    pub(crate) ptz_provider: Option<Arc<dyn PtzProvider>>,
    pub(crate) frame_inclusion: FrameInclusion,
    pub(crate) sink_queue_capacity: usize,
    pub(crate) decode_preference: DecodePreference,
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
    feed_pipeline: Option<FeedPipeline>,
    view_state_provider: Option<Box<dyn ViewStateProvider>>,
    epoch_policy: Option<Box<dyn EpochPolicy>>,
    output_sink: Option<Box<dyn OutputSink>>,
    backpressure: BackpressurePolicy,
    temporal: RetentionPolicy,
    reconnect: ReconnectPolicy,
    restart: RestartPolicy,
    ptz_provider: Option<Arc<dyn PtzProvider>>,
    frame_inclusion: FrameInclusion,
    validation_mode: ValidationMode,
    sink_queue_capacity: usize,
    decode_preference: DecodePreference,
}

impl FeedConfig {
    /// Create a new builder.
    #[must_use]
    pub fn builder() -> FeedConfigBuilder {
        FeedConfigBuilder {
            source: None,
            camera_mode: None,
            stages: None,
            feed_pipeline: None,
            view_state_provider: None,
            epoch_policy: None,
            output_sink: None,
            backpressure: BackpressurePolicy::default(),
            temporal: RetentionPolicy::default(),
            reconnect: ReconnectPolicy::default(),
            restart: RestartPolicy::default(),
            ptz_provider: None,
            frame_inclusion: FrameInclusion::default(),
            validation_mode: ValidationMode::default(),
            sink_queue_capacity: 16,
            decode_preference: DecodePreference::default(),
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
    ///
    /// Mutually exclusive with [`feed_pipeline()`](Self::feed_pipeline).
    /// For pipelines with a shared batch point, use `feed_pipeline()`.
    #[must_use]
    pub fn stages(mut self, stages: Vec<Box<dyn Stage>>) -> Self {
        self.stages = Some(stages);
        self
    }

    /// Set the perception pipeline from a [`StagePipeline`].
    ///
    /// This is a convenience alternative to [`stages()`](Self::stages)
    /// that accepts a pre-composed pipeline.
    /// Mutually exclusive with [`feed_pipeline()`](Self::feed_pipeline).
    #[must_use]
    pub fn pipeline(mut self, pipeline: StagePipeline) -> Self {
        self.stages = Some(pipeline.into_stages());
        self
    }

    /// Set the feed pipeline, optionally including a shared batch point.
    ///
    /// This is the recommended way to configure feeds that participate in
    /// shared batch inference. Use [`FeedPipeline::builder()`] to compose
    /// pre-batch stages, a batch point, and post-batch stages.
    ///
    /// Mutually exclusive with [`stages()`](Self::stages) and
    /// [`pipeline()`](Self::pipeline).
    #[must_use]
    pub fn feed_pipeline(mut self, pipeline: FeedPipeline) -> Self {
        self.feed_pipeline = Some(pipeline);
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

    /// Set the frame inclusion policy.
    ///
    /// When set to [`FrameInclusion::Always`], each [`OutputEnvelope`]
    /// includes a zero-copy reference to the source [`FrameEnvelope`].
    /// Default is [`FrameInclusion::Never`].
    #[must_use]
    pub fn frame_inclusion(mut self, policy: FrameInclusion) -> Self {
        self.frame_inclusion = policy;
        self
    }

    /// Set the stage capability validation mode.
    ///
    /// - [`ValidationMode::Off`] (default) — no validation.
    /// - [`ValidationMode::Warn`] — log warnings via `tracing::warn!`.
    /// - [`ValidationMode::Error`] — any warning becomes a build error.
    #[must_use]
    pub fn validation_mode(mut self, mode: ValidationMode) -> Self {
        self.validation_mode = mode;
        self
    }

    /// Set the per-feed output sink queue capacity.
    ///
    /// This is the bounded channel between the feed worker thread and
    /// the sink thread. Default: `16`. Must be at least 1.
    #[must_use]
    pub fn sink_queue_capacity(mut self, capacity: usize) -> Self {
        self.sink_queue_capacity = capacity;
        self
    }

    /// Set the decode preference for this feed.
    ///
    /// Controls hardware vs. software decoder selection. Default is
    /// [`DecodePreference::Auto`], which preserves existing behavior.
    ///
    /// See [`DecodePreference`] for variant semantics.
    #[must_use]
    pub fn decode_preference(mut self, pref: DecodePreference) -> Self {
        self.decode_preference = pref;
        self
    }

    /// Append a single stage to the pipeline.
    ///
    /// Convenience alternative to [`stages()`](Self::stages) when
    /// building one stage at a time.
    #[must_use]
    pub fn add_stage(mut self, stage: impl Stage) -> Self {
        self.stages.get_or_insert_with(Vec::new).push(Box::new(stage));
        self
    }

    /// Append a boxed stage to the pipeline.
    #[must_use]
    pub fn add_boxed_stage(mut self, stage: Box<dyn Stage>) -> Self {
        self.stages.get_or_insert_with(Vec::new).push(stage);
        self
    }

    /// Build the feed configuration.
    ///
    /// # Errors
    ///
    /// - `MissingRequired` if `source`, `camera_mode`, stages (or feed_pipeline), or `output_sink` are not set.
    /// - `CameraModeConflict` if `Observed` is set without a provider, or `Fixed` with a provider.
    pub fn build(self) -> Result<FeedConfig, NvError> {
        let source = self.source.ok_or(ConfigError::MissingRequired {
            field: "source",
        })?;
        let camera_mode = self.camera_mode.ok_or(ConfigError::MissingRequired {
            field: "camera_mode",
        })?;

        // Resolve stages: either from feed_pipeline or from stages/pipeline.
        let (stages, batch, post_batch_stages) = if let Some(fp) = self.feed_pipeline {
            if self.stages.is_some() {
                return Err(ConfigError::InvalidPolicy {
                    detail: "cannot set both stages() and feed_pipeline() — use one or the other".into(),
                }.into());
            }
            fp.into_parts()
        } else {
            let stages = self.stages.ok_or(ConfigError::MissingRequired {
                field: "stages (or feed_pipeline)",
            })?;
            (stages, None, Vec::new())
        };

        // At least one stage must exist somewhere in the pipeline.
        if stages.is_empty() && post_batch_stages.is_empty() && batch.is_none() {
            return Err(ConfigError::InvalidPolicy {
                detail: "at least one perception stage or a batch point is required".into(),
            }
            .into());
        }

        let output_sink = self.output_sink.ok_or(ConfigError::MissingRequired {
            field: "output_sink",
        })?;

        // Stage capability validation — batch-aware: pre-batch stages
        // are validated first, then batch processor capabilities update
        // the availability state, then post-batch stages are validated.
        let batch_caps = batch.as_ref().and_then(|b| b.capabilities().cloned());
        let batch_id = batch.as_ref().map(|b| b.processor_id());
        match self.validation_mode {
            ValidationMode::Off => {}
            ValidationMode::Warn => {
                for w in validate_pipeline(&stages, batch_caps.as_ref(), batch_id, &post_batch_stages) {
                    tracing::warn!("stage validation: {w:?}");
                }
            }
            ValidationMode::Error => {
                let warnings = validate_pipeline(&stages, batch_caps.as_ref(), batch_id, &post_batch_stages);
                if !warnings.is_empty() {
                    let detail = warnings
                        .iter()
                        .map(|w| format!("{w:?}"))
                        .collect::<Vec<_>>()
                        .join("; ");
                    return Err(ConfigError::StageValidation { detail }.into());
                }
            }
        }

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
            batch,
            post_batch_stages,
            view_state_provider: self.view_state_provider,
            epoch_policy,
            output_sink,
            backpressure: self.backpressure,
            temporal: self.temporal,
            reconnect: self.reconnect,
            restart: self.restart,
            ptz_provider: self.ptz_provider,
            frame_inclusion: self.frame_inclusion,
            sink_queue_capacity: self.sink_queue_capacity.max(1),
            decode_preference: self.decode_preference,
        })
    }
}

/// Validate pipeline stages accounting for a batch processor between
/// pre-batch and post-batch stages.
fn validate_pipeline(
    pre_batch: &[Box<dyn Stage>],
    batch_caps: Option<&nv_perception::stage::StageCapabilities>,
    batch_id: Option<nv_core::id::StageId>,
    post_batch: &[Box<dyn Stage>],
) -> Vec<nv_perception::ValidationWarning> {
    validate_pipeline_phased(pre_batch, batch_caps, batch_id, post_batch)
}

/// Handle to a running feed.
///
/// Provides per-feed monitoring and control: health events, metrics,
/// queue telemetry, uptime, pause/resume, and stop.
///
/// Backed by `Arc<FeedSharedState>` — cloning is cheap (Arc bump).
/// Metrics are read from the same atomic counters that the feed worker
/// thread writes.
#[derive(Clone)]
pub struct FeedHandle {
    shared: std::sync::Arc<crate::worker::FeedSharedState>,
}

/// Snapshot of queue depths and capacities for a feed.
///
/// **Source queue**: bounded frame queue between media ingress and pipeline.
/// **Sink queue**: bounded channel between pipeline and output sink.
///
/// Values are approximate under concurrent access — sufficient for
/// monitoring and dashboards, not for synchronization.
#[derive(Debug, Clone, Copy)]
pub struct QueueTelemetry {
    /// Current number of frames in the source queue.
    pub source_depth: usize,
    /// Maximum capacity of the source queue.
    pub source_capacity: usize,
    /// Current number of outputs in the sink queue.
    pub sink_depth: usize,
    /// Maximum capacity of the sink queue.
    pub sink_capacity: usize,
}

/// Snapshot of the decode method selected by the media backend.
///
/// Available after the stream starts and the backend confirms decoder
/// negotiation. Use [`FeedHandle::decode_status()`] to poll.
#[derive(Debug, Clone)]
pub struct DecodeStatus {
    /// Whether hardware or software decoding was selected.
    pub outcome: nv_core::health::DecodeOutcome,
    /// Backend-specific detail string (e.g., GStreamer element name).
    ///
    /// Intended for diagnostics and dashboards — do not match on its
    /// contents programmatically.
    pub detail: String,
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
        self.shared
            .alive
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get a snapshot of the feed's current metrics.
    ///
    /// Reads live atomic counters maintained by the feed worker thread.
    #[must_use]
    pub fn metrics(&self) -> FeedMetrics {
        self.shared.metrics()
    }

    /// Get a snapshot of the feed's source and sink queue depths/capacities.
    ///
    /// Source queue depth reads the frame queue's internal lock briefly.
    /// Sink queue depth reads an atomic counter with no locking.
    ///
    /// If no processing session is active (between restarts or after
    /// shutdown), both depths return 0.
    #[must_use]
    pub fn queue_telemetry(&self) -> QueueTelemetry {
        let (source_depth, source_capacity) = self.shared.source_queue_telemetry();
        QueueTelemetry {
            source_depth,
            source_capacity,
            sink_depth: self.shared.sink_occupancy.load(std::sync::atomic::Ordering::Relaxed),
            sink_capacity: self.shared.sink_capacity.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Elapsed time since the feed's current processing session started.
    ///
    /// **Semantics: session-scoped uptime.** The clock resets on each
    /// successful start or restart. If the feed has not started yet or
    /// is between restart attempts, the value reflects the time since the
    /// last session began.
    ///
    /// Useful for monitoring feed stability: a feed that restarts
    /// frequently will show low uptime values.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.shared.session_uptime()
    }

    /// The decode method confirmed by the media backend for this feed.
    ///
    /// Returns `None` if no decode decision has been made yet (the
    /// stream has not started, the backend has not negotiated a decoder,
    /// or the feed is between restarts).
    #[must_use]
    pub fn decode_status(&self) -> Option<DecodeStatus> {
        let (outcome, detail) = self.shared.decode_status()?;
        Some(DecodeStatus { outcome, detail })
    }

    /// Get a consolidated diagnostics snapshot of this feed.
    ///
    /// Composes lifecycle state, metrics, queue depths, decode status,
    /// and view-system health into a single read. All data comes from
    /// the same atomic counters the individual accessors use — this is
    /// a convenience composite, not a new data source.
    ///
    /// Suitable for periodic polling (1–5 s) by dashboards and health
    /// probes.
    #[must_use]
    pub fn diagnostics(&self) -> crate::diagnostics::FeedDiagnostics {
        use crate::diagnostics::{ViewDiagnostics, ViewStatus};

        let metrics = self.metrics();
        let validity_ordinal = self
            .shared
            .view_context_validity
            .load(std::sync::atomic::Ordering::Relaxed);
        let status = match validity_ordinal {
            0 => ViewStatus::Stable,
            1 => ViewStatus::Degraded,
            _ => ViewStatus::Invalid,
        };
        let stability_bits = self
            .shared
            .view_stability_score
            .load(std::sync::atomic::Ordering::Relaxed);

        crate::diagnostics::FeedDiagnostics {
            feed_id: self.id(),
            alive: self.is_alive(),
            paused: self.is_paused(),
            uptime: self.uptime(),
            metrics,
            queues: self.queue_telemetry(),
            decode: self.decode_status(),
            view: ViewDiagnostics {
                epoch: metrics.view_epoch,
                stability_score: f32::from_bits(stability_bits),
                status,
            },
            batch_processor_id: self.shared.batch_processor_id,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::id::StageId;
    use nv_perception::stage::StageCapabilities;
    use nv_perception::{Stage, StageContext, StageOutput, ValidationWarning};

    struct CapStage {
        id: &'static str,
        caps: StageCapabilities,
    }
    impl Stage for CapStage {
        fn id(&self) -> StageId { StageId(self.id) }
        fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, nv_core::error::StageError> {
            Ok(StageOutput::empty())
        }
        fn capabilities(&self) -> Option<StageCapabilities> {
            Some(self.caps)
        }
    }

    #[test]
    fn batch_consumes_validated_against_pre_batch() {
        let pre: Vec<Box<dyn Stage>> = vec![];
        let caps = StageCapabilities::new()
            .consumes_detections()
            .produces_tracks();
        let batch_id = StageId("detector");
        let post: Vec<Box<dyn Stage>> = vec![];

        let warnings = validate_pipeline(&pre, Some(&caps), Some(batch_id), &post);
        assert!(
            warnings.iter().any(|w| matches!(
                w,
                ValidationWarning::UnsatisfiedDependency { stage_id, missing: "detections" }
                if *stage_id == StageId("detector")
            )),
            "expected UnsatisfiedDependency for detections, got: {warnings:?}"
        );
    }

    #[test]
    fn batch_consumes_satisfied_by_pre_batch() {
        let pre: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            id: "det_stage",
            caps: StageCapabilities::new().produces_detections(),
        })];
        let caps = StageCapabilities::new()
            .consumes_detections()
            .produces_tracks();
        let batch_id = StageId("tracker");
        let post: Vec<Box<dyn Stage>> = vec![];

        let warnings = validate_pipeline(&pre, Some(&caps), Some(batch_id), &post);
        assert!(
            !warnings.iter().any(|w| matches!(w, ValidationWarning::UnsatisfiedDependency { .. })),
            "no unsatisfied dependencies expected, got: {warnings:?}"
        );
    }

    #[test]
    fn batch_id_collision_detected() {
        let pre: Vec<Box<dyn Stage>> = vec![Box::new(CapStage {
            id: "detector",
            caps: StageCapabilities::new().produces_detections(),
        })];
        let batch_id = StageId("detector"); // same as pre-batch stage
        let post: Vec<Box<dyn Stage>> = vec![];

        let warnings = validate_pipeline(&pre, None, Some(batch_id), &post);
        assert!(
            warnings.iter().any(|w| matches!(
                w,
                ValidationWarning::DuplicateStageId { stage_id } if *stage_id == StageId("detector")
            )),
            "expected DuplicateStageId, got: {warnings:?}"
        );
    }
}
