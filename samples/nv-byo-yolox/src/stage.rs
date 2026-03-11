use ort::session::Session;
use ort::value::Tensor;
use tracing::debug;

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_frame::PixelFormat;
use nv_perception::stage::{Stage, StageCategory, StageContext, StageOutput};
use nv_view::ViewEpoch;

use crate::config::YoloxConfig;
use crate::letterbox::letterbox_preprocess;
use crate::postprocess::decode_yolox_output;

/// YOLOX-s detector stage.
///
/// Wraps an ONNX Runtime session for YOLOX-s inference. The session is
/// created in [`on_start`](Stage::on_start) so that model-load failures
/// surface as typed errors at feed startup.
///
/// Holds the session as a direct owned value (`&mut self` on the single
/// feed thread).
pub struct YoloxStage {
    config: YoloxConfig,
    session: Option<Session>,
    det_counter: u64,
    num_classes: usize,
}

impl YoloxStage {
    const STAGE_ID: StageId = StageId("byo-yolox");

    /// Create a new YOLOX stage.
    ///
    /// The model is **not** loaded until [`on_start`](Stage::on_start) is
    /// called by the runtime.
    pub fn new(config: YoloxConfig) -> Self {
        Self {
            config,
            session: None,
            det_counter: 0,
            num_classes: 80, // COCO default; overridden on model load
        }
    }
}

impl Stage for YoloxStage {
    fn id(&self) -> StageId {
        Self::STAGE_ID
    }

    fn category(&self) -> StageCategory {
        StageCategory::FrameAnalysis
    }

    fn on_start(&mut self) -> Result<(), StageError> {
        let mut builder = Session::builder().map_err(|e| StageError::ModelLoadFailed {
            stage_id: Self::STAGE_ID,
            detail: format!("session builder failed: {e}"),
        })?;

        let session =
            builder
                .commit_from_file(&self.config.model_path)
                .map_err(|e| StageError::ModelLoadFailed {
                    stage_id: Self::STAGE_ID,
                    detail: format!("ONNX load failed: {e}"),
                })?;

        // Infer num_classes from output shape if available.
        if let Some(output_info) = session.outputs().first() {
            let dtype: &ort::value::ValueType = output_info.dtype();
            if let ort::value::ValueType::Tensor { ref shape, .. } = *dtype {
                if let Some(&last) = shape.last() {
                    if last > 5 {
                        self.num_classes = (last - 5) as usize;
                    }
                }
            }
        }

        debug!(
            model = %self.config.model_path.display(),
            num_classes = self.num_classes,
            input_size = self.config.input_size,
            "YOLOX model loaded"
        );

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
        let session = self.session.as_mut().ok_or_else(|| StageError::ProcessingFailed {
            stage_id: Self::STAGE_ID,
            detail: "session not initialised (on_start not called)".into(),
        })?;

        let frame = ctx.frame;

        if frame.format() != PixelFormat::Rgb8 {
            return Err(StageError::ProcessingFailed {
                stage_id: Self::STAGE_ID,
                detail: format!("unsupported pixel format {:?}, expected Rgb8", frame.format()),
            });
        }

        let input_size = self.config.input_size;

        // Letterbox preprocess.
        let (tensor_data, lb_info) = letterbox_preprocess(
            frame.data(),
            frame.width(),
            frame.height(),
            frame.stride(),
            input_size,
        );

        // Build tensor and run inference.
        let shape = vec![1i64, 3, input_size as i64, input_size as i64];
        let input_tensor =
            Tensor::from_array((shape, tensor_data)).map_err(|e| StageError::ProcessingFailed {
                stage_id: Self::STAGE_ID,
                detail: format!("tensor creation error: {e}"),
            })?;

        let outputs =
            session
                .run(ort::inputs![input_tensor])
                .map_err(|e| StageError::ProcessingFailed {
                    stage_id: Self::STAGE_ID,
                    detail: format!("inference failed: {e}"),
                })?;

        // Extract output tensor (first output).
        let output_value = &outputs[0];
        let (_shape, output_slice) = output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| StageError::ProcessingFailed {
                stage_id: Self::STAGE_ID,
                detail: format!("output tensor extraction failed: {e}"),
            })?;

        // Decode + NMS.
        let offset = self.det_counter;
        let detection_set = decode_yolox_output(
            output_slice,
            self.num_classes,
            input_size,
            self.config.confidence_threshold,
            self.config.nms_threshold,
            &lb_info,
            offset,
        );
        self.det_counter += detection_set.len() as u64;

        Ok(StageOutput::with_detections(detection_set))
    }
}
