//! Shared ONNX session loading for detection models.
//!
//! Used by both [`DetectorStage`](crate::DetectorStage) (per-feed) and
//! [`DetectorBatchProcessor`](crate::DetectorBatchProcessor) (cross-feed batch).

use ort::session::Session;
use tracing::debug;

use nv_core::error::StageError;
use nv_core::id::StageId;

use crate::config::DetectorConfig;

/// Expected number of columns in the end-to-end output tensor.
/// Each detection row is `[x1, y1, x2, y2, confidence, class_id]`.
const EXPECTED_OUTPUT_COLS: i64 = 6;

/// Load a detector ONNX model session, validating the output schema.
///
/// Checks that the model's first output has 3 dimensions with the last
/// dimension equal to 6 (`[batch, max_dets, 6]`). This catches mismatched
/// exports (e.g., raw 85-column output) at startup instead of at
/// first inference.
pub(crate) fn load_session(config: &DetectorConfig, stage_id: StageId) -> Result<Session, StageError> {
    let mut builder = Session::builder().map_err(|e| StageError::ModelLoadFailed {
        stage_id,
        detail: format!("session builder failed: {e}"),
    })?;

    let session = builder
        .commit_from_file(&config.model_path)
        .map_err(|e| StageError::ModelLoadFailed {
            stage_id,
            detail: format!("ONNX load failed: {e}"),
        })?;

    // Validate output schema: expect [batch, max_dets, 6].
    if let Some(output) = session.outputs().first() {
        if let ort::value::ValueType::Tensor { shape, .. } = output.dtype() {
            if shape.len() != 3 {
                return Err(StageError::ModelLoadFailed {
                    stage_id,
                    detail: format!(
                        "unexpected output rank: expected 3 dimensions [batch, max_dets, 6], \
                         got {} dimensions {:?}. Ensure the model is an end-to-end export.",
                        shape.len(),
                        &shape[..],
                    ),
                });
            }
            // The last dimension may be dynamic (-1). Only reject
            // when it is a fixed positive value that isn't 6.
            let cols = shape[2];
            if cols > 0 && cols != EXPECTED_OUTPUT_COLS {
                return Err(StageError::ModelLoadFailed {
                    stage_id,
                    detail: format!(
                        "unexpected output columns: expected {EXPECTED_OUTPUT_COLS} \
                         [x1, y1, x2, y2, conf, class], got {cols}. \
                         Ensure the model is an end-to-end export.",
                    ),
                });
            }
        }
    }

    debug!(
        model = %config.model_path.display(),
        input_size = config.input_size,
        stage = %stage_id,
        "detector model loaded"
    );

    Ok(session)
}
