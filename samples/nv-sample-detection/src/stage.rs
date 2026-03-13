use ort::session::Session;

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::stage::{Stage, StageCapabilities, StageCategory, StageContext, StageOutput};
use nv_view::ViewEpoch;

use crate::config::DetectorConfig;
use crate::inference;
use crate::letterbox::letterbox_preprocess;
use crate::postprocess::decode_end2end_output;
use crate::session;

/// Sample ONNX detector stage.
///
/// Wraps an ONNX Runtime session for inference. The session is
/// created in [`on_start`](Stage::on_start) so that model-load failures
/// surface as typed errors at feed startup.
pub struct DetectorStage {
    config: DetectorConfig,
    session: Option<Session>,
    det_counter: u64,
}

impl DetectorStage {
    const STAGE_ID: StageId = StageId("sample-detector");

    /// Create a new detector stage.
    ///
    /// The model is **not** loaded until [`on_start`](Stage::on_start) is
    /// called by the runtime.
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            session: None,
            det_counter: 0,
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

        inference::require_rgb8(frame.format(), Self::STAGE_ID)?;

        let input_size = self.config.input_size;

        // Letterbox preprocess.
        let (tensor_data, lb_info) = letterbox_preprocess(
            frame.data(),
            frame.width(),
            frame.height(),
            frame.stride(),
            input_size,
        );

        // Run inference + decode in a single borrow (zero-copy output).
        let shape = vec![1i64, 3, input_size as i64, input_size as i64];
        let conf_threshold = self.config.confidence_threshold;
        let det_offset = self.det_counter;

        let (output, new_counter) = inference::with_inference(
            session,
            shape,
            tensor_data,
            Self::STAGE_ID,
            |_shape, output_flat| {
                let detection_set = decode_end2end_output(
                    output_flat,
                    conf_threshold,
                    &lb_info,
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
