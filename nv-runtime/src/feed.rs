//! Feed configuration and validation.

use std::sync::Arc;
use std::time::Duration;

use nv_core::config::{CameraMode, ReconnectPolicy, SourceSpec};
use nv_core::error::{ConfigError, NvError};
use nv_media::DecodePreference;
use nv_media::DeviceResidency;
use nv_media::PostDecodeHook;
use nv_media::PtzProvider;
use nv_perception::{Stage, StagePipeline, ValidationMode, validate_pipeline_phased};
use nv_temporal::RetentionPolicy;
use nv_view::{EpochPolicy, ViewStateProvider};

use crate::backpressure::BackpressurePolicy;
use crate::batch::BatchHandle;
use crate::output::{FrameInclusion, OutputSink, SinkFactory};
use crate::pipeline::FeedPipeline;
use crate::shutdown::RestartPolicy;
use crate::worker::sink::DEFAULT_SINK_SHUTDOWN_TIMEOUT;

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
    pub(crate) sink_factory: Option<SinkFactory>,
    pub(crate) backpressure: BackpressurePolicy,
    pub(crate) temporal: RetentionPolicy,
    pub(crate) reconnect: ReconnectPolicy,
    pub(crate) restart: RestartPolicy,
    pub(crate) ptz_provider: Option<Arc<dyn PtzProvider>>,
    pub(crate) frame_inclusion: FrameInclusion,
    pub(crate) sink_queue_capacity: usize,
    pub(crate) sink_shutdown_timeout: Duration,
    pub(crate) decode_preference: DecodePreference,
    pub(crate) post_decode_hook: Option<PostDecodeHook>,
    pub(crate) device_residency: DeviceResidency,
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
    sink_factory: Option<SinkFactory>,
    backpressure: BackpressurePolicy,
    temporal: RetentionPolicy,
    reconnect: ReconnectPolicy,
    restart: RestartPolicy,
    ptz_provider: Option<Arc<dyn PtzProvider>>,
    frame_inclusion: FrameInclusion,
    validation_mode: ValidationMode,
    sink_queue_capacity: usize,
    sink_shutdown_timeout: Duration,
    decode_preference: DecodePreference,
    post_decode_hook: Option<PostDecodeHook>,
    device_residency: DeviceResidency,
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
            sink_factory: None,
            backpressure: BackpressurePolicy::default(),
            temporal: RetentionPolicy::default(),
            reconnect: ReconnectPolicy::default(),
            restart: RestartPolicy::default(),
            ptz_provider: None,
            frame_inclusion: FrameInclusion::default(),
            validation_mode: ValidationMode::default(),
            sink_queue_capacity: 16,
            sink_shutdown_timeout: DEFAULT_SINK_SHUTDOWN_TIMEOUT,
            decode_preference: DecodePreference::default(),
            post_decode_hook: None,
            device_residency: DeviceResidency::default(),
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

    /// Set an optional sink factory for reconstructing the sink after
    /// timeout or panic.
    ///
    /// Without a factory, a sink that times out during shutdown is
    /// permanently replaced with a silent no-op. With a factory, the
    /// next feed restart constructs a fresh sink.
    #[must_use]
    pub fn sink_factory(mut self, factory: SinkFactory) -> Self {
        self.sink_factory = Some(factory);
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
    /// - [`FrameInclusion::Never`] (default) — no pixel data in outputs.
    /// - [`FrameInclusion::Always`] — every output carries a frame.
    /// - [`FrameInclusion::Sampled { interval }`] — a frame is included
    ///   every `interval` outputs, reducing host materialization and sink
    ///   thread cost while keeping perception artifacts at full rate.
    /// - [`FrameInclusion::TargetFps { target, fallback_interval }`] —
    ///   resolve the sample interval dynamically from the observed source
    ///   rate. During a warmup window (first ~30 frames),
    ///   `fallback_interval` is used. Once the source cadence is
    ///   estimated from frame timestamps, the interval is computed as
    ///   `round(source_fps / target)` and locked for the feed's lifetime.
    ///
    /// Use [`TargetFps`](FrameInclusion::TargetFps) when the source FPS
    /// is unknown at config time or varies across feeds. Use
    /// [`Sampled`](FrameInclusion::Sampled) when the interval is known
    /// statically.
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

    /// Set the timeout for joining the sink worker thread during
    /// shutdown or restart. Default: 5 seconds.
    #[must_use]
    pub fn sink_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.sink_shutdown_timeout = timeout;
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

    /// Set a post-decode hook that can inject a pipeline element between
    /// the decoder and the color-space converter.
    ///
    /// **Ignored when `device_residency` is `Provider`** — the provider
    /// controls the full decoder-to-tail path.  Hooks are only evaluated
    /// for `Host` and `Cuda` residency modes, so this can be set
    /// unconditionally without conflicting with provider-managed feeds.
    ///
    /// See [`PostDecodeHook`] for details and usage examples.
    #[must_use]
    pub fn post_decode_hook(mut self, hook: PostDecodeHook) -> Self {
        self.post_decode_hook = Some(hook);
        self
    }

    /// Set the device residency mode for decoded frames.
    ///
    /// Controls the pipeline tail strategy:
    /// - [`DeviceResidency::Host`] (default) — `videoconvert → appsink`
    /// - [`DeviceResidency::Cuda`] — `cudaupload → cudaconvert → appsink(memory:CUDAMemory)`
    /// - `DeviceResidency::Provider(p)` — delegates to the provider
    ///
    /// See [`DeviceResidency`] for details.
    #[must_use]
    pub fn device_residency(mut self, residency: DeviceResidency) -> Self {
        self.device_residency = residency;
        self
    }

    /// Append a single stage to the pipeline.
    ///
    /// Convenience alternative to [`stages()`](Self::stages) when
    /// building one stage at a time.
    #[must_use]
    pub fn add_stage(mut self, stage: impl Stage) -> Self {
        self.stages
            .get_or_insert_with(Vec::new)
            .push(Box::new(stage));
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
        let source = self
            .source
            .ok_or(ConfigError::MissingRequired { field: "source" })?;
        let camera_mode = self.camera_mode.ok_or(ConfigError::MissingRequired {
            field: "camera_mode",
        })?;

        // Resolve stages: either from feed_pipeline or from stages/pipeline.
        let (stages, batch, post_batch_stages) = if let Some(fp) = self.feed_pipeline {
            if self.stages.is_some() {
                return Err(ConfigError::InvalidPolicy {
                    detail: "cannot set both stages() and feed_pipeline() — use one or the other"
                        .into(),
                }
                .into());
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
                for w in
                    validate_pipeline(&stages, batch_caps.as_ref(), batch_id, &post_batch_stages)
                {
                    tracing::warn!("stage validation: {w:?}");
                }
            }
            ValidationMode::Error => {
                let warnings =
                    validate_pipeline(&stages, batch_caps.as_ref(), batch_id, &post_batch_stages);
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
            sink_factory: self.sink_factory,
            backpressure: self.backpressure,
            temporal: self.temporal,
            reconnect: self.reconnect,
            restart: self.restart,
            ptz_provider: self.ptz_provider,
            frame_inclusion: self.frame_inclusion,
            sink_queue_capacity: self.sink_queue_capacity.max(1),
            sink_shutdown_timeout: self.sink_shutdown_timeout,
            decode_preference: self.decode_preference,
            post_decode_hook: self.post_decode_hook,
            device_residency: self.device_residency,
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
        fn id(&self) -> StageId {
            StageId(self.id)
        }
        fn process(
            &mut self,
            _: &StageContext<'_>,
        ) -> Result<StageOutput, nv_core::error::StageError> {
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
            !warnings
                .iter()
                .any(|w| matches!(w, ValidationWarning::UnsatisfiedDependency { .. })),
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
