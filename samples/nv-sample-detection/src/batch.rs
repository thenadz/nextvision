//! Sample batch detector — shared cross-feed inference via [`BatchProcessor`].
//!
//! Preprocesses each frame individually (letterbox), concatenates them
//! into a single `[N, 3, H, W]` tensor, runs one ONNX session call,
//! then splits the output back into per-item detection sets.
//!
//! Frame preprocessing is delegated to a [`BatchFramePreprocessor`].
//! The default [`HostBatchPreprocessor`] uses CPU letterbox; hardware
//! backends provide their own implementations to bypass host
//! materialization. Inference, batching, and postprocessing logic are
//! shared regardless of the preprocessing backend.

use ort::session::Session;
use tracing::debug;

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::batch::{BatchEntry, BatchProcessor};
use nv_perception::stage::{StageCapabilities, StageCategory, StageOutput};

use crate::config::DetectorConfig;
use crate::inference;
use crate::letterbox::LetterboxInfo;
use crate::postprocess::decode_end2end_output;
use crate::preprocess::{BatchFramePreprocessor, HostBatchPreprocessor};
use crate::session;

/// Sample batch detector.
///
/// Implements [`BatchProcessor`] so multiple feeds can share a single
/// model session. Frames from different feeds are batched into one
/// inference call.
///
/// Frame preprocessing is delegated to a [`BatchFramePreprocessor`]:
/// - Default [`HostBatchPreprocessor`] handles host/mappable frames
///   via CPU letterbox.
/// - Hardware backends (CUDA, TensorRT) provide GPU-native
///   implementations to avoid host materialization.
///
/// Inference, output validation, and postprocessing are shared.
pub struct DetectorBatchProcessor {
    config: DetectorConfig,
    session: Option<Session>,
    det_counter: u64,
    preprocessor: Box<dyn BatchFramePreprocessor>,
}

impl DetectorBatchProcessor {
    const PROCESSOR_ID: StageId = StageId("sample-detector-batch");

    /// Create a new batch processor with the default host preprocessor.
    ///
    /// The host fallback policy is resolved from
    /// [`DetectorConfig::effective_host_fallback`](crate::DetectorConfig::effective_host_fallback):
    /// [`Auto`](crate::HostFallbackPolicy::Auto) yields `Warn` for GPU,
    /// `Allow` for CPU; explicit variants are passed through unchanged.
    pub fn new(config: DetectorConfig) -> Self {
        let policy = config.effective_host_fallback();
        Self {
            preprocessor: Box::new(HostBatchPreprocessor::with_policy(
                Self::PROCESSOR_ID,
                policy,
            )),
            config,
            session: None,
            det_counter: 0,
        }
    }

    /// Create a batch processor with a custom frame preprocessor.
    ///
    /// Use this when the default host-path letterbox is insufficient,
    /// e.g., for device-native batch preprocessing on Jetson.
    pub fn with_preprocessor(
        config: DetectorConfig,
        preprocessor: Box<dyn BatchFramePreprocessor>,
    ) -> Self {
        Self {
            config,
            session: None,
            det_counter: 0,
            preprocessor,
        }
    }
}

impl BatchProcessor for DetectorBatchProcessor {
    fn id(&self) -> StageId {
        Self::PROCESSOR_ID
    }

    fn category(&self) -> StageCategory {
        StageCategory::FrameAnalysis
    }

    fn capabilities(&self) -> Option<StageCapabilities> {
        Some(StageCapabilities::new().produces_detections())
    }

    fn on_start(&mut self) -> Result<(), StageError> {
        let session = session::load_session(&self.config, Self::PROCESSOR_ID)?;
        self.session = Some(session);
        Ok(())
    }

    fn on_stop(&mut self) -> Result<(), StageError> {
        self.session = None;
        Ok(())
    }

    fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
        let session = inference::require_session(&mut self.session, Self::PROCESSOR_ID)?;

        let batch_size = items.len();
        let input_size = self.config.input_size;
        let pixels_per_frame = 3 * (input_size as usize) * (input_size as usize);

        // Pre-allocate the full batch tensor and write each frame's
        // letterbox output directly into its slice (no per-item allocation).
        let mut batch_tensor = vec![0.0_f32; batch_size * pixels_per_frame];
        let mut lb_infos: Vec<LetterboxInfo> = Vec::with_capacity(batch_size);

        for (i, item) in items.iter().enumerate() {
            let offset = i * pixels_per_frame;

            // Delegate to the configured BatchFramePreprocessor.
            // The default HostBatchPreprocessor uses resolve_host_pixels
            // + letterbox_preprocess_into. Device-native backends bypass
            // host materialization entirely.
            let lb_info = self.preprocessor.preprocess_into(
                &item.frame,
                input_size,
                &mut batch_tensor[offset..offset + pixels_per_frame],
            )?;
            lb_infos.push(lb_info);
        }

        // Run batched inference: [N, 3, H, W] → decode per-item output
        // using a zero-copy closure over the ORT output.
        let shape = vec![batch_size as i64, 3, input_size as i64, input_size as i64];
        let conf_threshold = self.config.confidence_threshold;
        let det_counter = self.det_counter;

        let new_counter = inference::with_inference(
            session,
            shape,
            batch_tensor,
            Self::PROCESSOR_ID,
            |out_shape, output_flat| {
                // Expected output shape: [N, max_dets, 6].
                if out_shape.len() != 3 {
                    return Err(StageError::ProcessingFailed {
                        stage_id: Self::PROCESSOR_ID,
                        detail: format!(
                            "unexpected output rank: expected 3 dimensions \
                             [batch, max_dets, 6], got {:?}",
                            out_shape,
                        ),
                    });
                }
                if out_shape[0] as usize != batch_size {
                    return Err(StageError::ProcessingFailed {
                        stage_id: Self::PROCESSOR_ID,
                        detail: format!(
                            "output batch dimension mismatch: expected {}, got {}",
                            batch_size, out_shape[0],
                        ),
                    });
                }
                if out_shape[2] != 6 {
                    return Err(StageError::ProcessingFailed {
                        stage_id: Self::PROCESSOR_ID,
                        detail: format!(
                            "output columns mismatch: expected 6 \
                             [x1, y1, x2, y2, conf, class], got {}",
                            out_shape[2],
                        ),
                    });
                }

                let max_dets = out_shape[1] as usize;
                let cols = out_shape[2] as usize;
                let item_len = max_dets * cols;

                // Fail fast: if the output doesn't have enough data for
                // all batch items, surface an error immediately instead
                // of silently producing empty detections.
                let expected_len = batch_size * item_len;
                if output_flat.len() < expected_len {
                    return Err(StageError::ProcessingFailed {
                        stage_id: Self::PROCESSOR_ID,
                        detail: format!(
                            "batch output too short: expected at least {} floats \
                             ({} items × {} per item), got {}",
                            expected_len,
                            batch_size,
                            item_len,
                            output_flat.len(),
                        ),
                    });
                }

                debug!(batch_size, max_dets, "batch inference complete");

                // Decode per-item output.
                let mut counter = det_counter;
                for (i, item) in items.iter_mut().enumerate() {
                    let start = i * item_len;
                    let item_output = &output_flat[start..start + item_len];

                    let detection_set =
                        decode_end2end_output(item_output, conf_threshold, &lb_infos[i], counter);
                    counter += detection_set.len() as u64;
                    item.output = Some(StageOutput::with_detections(detection_set));
                }
                Ok(counter)
            },
        )?;
        self.det_counter = new_counter;

        Ok(())
    }
}
