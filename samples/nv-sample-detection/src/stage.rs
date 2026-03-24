use ort::session::Session;

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::stage::{Stage, StageCapabilities, StageCategory, StageContext, StageOutput};
use nv_view::ViewEpoch;

use crate::config::DetectorConfig;
use crate::inference;
use crate::postprocess::decode_end2end_output;
use crate::preprocess::{FramePreprocessor, HostPreprocessor};
use crate::session;

/// Sample ONNX detector stage.
///
/// Wraps an ONNX Runtime session for inference. The session is
/// created in [`on_start`](Stage::on_start) so that model-load failures
/// surface as typed errors at feed startup.
///
/// Frame preprocessing is delegated to a [`FramePreprocessor`]. The
/// default [`HostPreprocessor`] handles host-readable and host-mappable
/// frames via CPU letterbox resize. For device-native preprocessing
/// (e.g., CUDA/TensorRT on Jetson), pass a custom preprocessor via
/// [`with_preprocessor`](Self::with_preprocessor).
///
/// When no custom preprocessor is provided, the host fallback policy is
/// resolved via [`DetectorConfig::effective_host_fallback`](crate::DetectorConfig::effective_host_fallback):
/// the default [`Auto`](crate::HostFallbackPolicy::Auto) resolves to
/// [`Warn`](crate::HostFallbackPolicy::Warn) in GPU mode and
/// [`Allow`](crate::HostFallbackPolicy::Allow) otherwise. Explicit
/// `Allow`, `Warn`, or `Forbid` are always respected.
pub struct DetectorStage {
    config: DetectorConfig,
    session: Option<Session>,
    det_counter: u64,
    preprocessor: Box<dyn FramePreprocessor>,
}

impl DetectorStage {
    const STAGE_ID: StageId = StageId("sample-detector");

    /// Create a new detector stage with the default host preprocessor.
    ///
    /// The model is **not** loaded until [`on_start`](Stage::on_start) is
    /// called by the runtime. The host fallback policy is resolved from
    /// [`DetectorConfig::effective_host_fallback`](crate::DetectorConfig::effective_host_fallback):
    /// [`Auto`](crate::HostFallbackPolicy::Auto) yields `Warn` for GPU,
    /// `Allow` for CPU; explicit variants are passed through unchanged.
    pub fn new(config: DetectorConfig) -> Self {
        let policy = config.effective_host_fallback();
        Self {
            preprocessor: Box::new(HostPreprocessor::with_policy(Self::STAGE_ID, policy)),
            config,
            session: None,
            det_counter: 0,
        }
    }

    /// Create a detector stage with a custom frame preprocessor.
    ///
    /// Use this when the default host-path letterbox is insufficient,
    /// e.g., for device-native preprocessing on Jetson (NVMM → CUDA)
    /// or AMD (ROCm).
    pub fn with_preprocessor(
        config: DetectorConfig,
        preprocessor: Box<dyn FramePreprocessor>,
    ) -> Self {
        Self {
            config,
            session: None,
            det_counter: 0,
            preprocessor,
        }
    }
}

impl Stage for DetectorStage {
    fn id(&self) -> StageId {
        Self::STAGE_ID
    }

    fn category(&self) -> StageCategory {
        StageCategory::FrameAnalysis
    }

    fn capabilities(&self) -> Option<StageCapabilities> {
        Some(StageCapabilities::new().produces_detections())
    }

    fn on_start(&mut self) -> Result<(), StageError> {
        let session = session::load_session(&self.config, Self::STAGE_ID)?;
        self.session = Some(session);
        Ok(())
    }

    fn on_stop(&mut self) -> Result<(), StageError> {
        self.session = None;
        Ok(())
    }

    fn on_view_epoch_change(&mut self, _new_epoch: ViewEpoch) -> Result<(), StageError> {
        Ok(())
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let session = inference::require_session(&mut self.session, Self::STAGE_ID)?;
        let frame = ctx.frame;
        let input_size = self.config.input_size;

        // Preprocess: routing between host and device paths is handled
        // by the configured FramePreprocessor.
        let preprocessed = self.preprocessor.preprocess(frame, input_size)?;

        // Run inference + decode in a single borrow (zero-copy output).
        let shape = vec![1i64, 3, input_size as i64, input_size as i64];
        let conf_threshold = self.config.confidence_threshold;
        let det_offset = self.det_counter;

        let (output, new_counter) = inference::with_inference(
            session,
            shape,
            preprocessed.tensor,
            Self::STAGE_ID,
            |_shape, output_flat| {
                let detection_set = decode_end2end_output(
                    output_flat,
                    conf_threshold,
                    &preprocessed.letterbox,
                    det_offset,
                );
                let counter = det_offset + detection_set.len() as u64;
                Ok((StageOutput::with_detections(detection_set), counter))
            },
        )?;
        self.det_counter = new_counter;

        Ok(output)
    }
}
