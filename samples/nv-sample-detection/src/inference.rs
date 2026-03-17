//! Shared inference helpers for the sample detector stage and batch processor.
//!
//! Encapsulates the common ONNX Runtime call pattern — tensor construction,
//! session execution, and output extraction — so that [`DetectorStage`](crate::DetectorStage)
//! and [`DetectorBatchProcessor`](crate::DetectorBatchProcessor) share a single
//! error-handling path.
//!
//! The [`with_inference`] function uses a closure to give callers a borrowed
//! view of the output tensor, avoiding a full copy of the model output.

use ort::session::Session;
use ort::value::Tensor;

use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_frame::PixelFormat;

/// Assert that a frame is RGB8, returning a typed [`StageError`] on mismatch.
pub(crate) fn require_rgb8(format: PixelFormat, stage_id: StageId) -> Result<(), StageError> {
    if format != PixelFormat::Rgb8 {
        return Err(StageError::ProcessingFailed {
            stage_id,
            detail: format!("unsupported pixel format {format:?}, expected Rgb8"),
        });
    }
    Ok(())
}

/// Run ONNX inference and process the output via a callback.
///
/// Constructs the input [`Tensor`] from `(shape, data)`, calls
/// `session.run()`, extracts the first float32 output tensor, and
/// passes a borrowed view to `f`. This avoids copying the (potentially
/// large) output tensor — the callback decodes directly from ORT's
/// internal memory.
///
/// Returns whatever `f` returns, wrapped in the outer `Result`.
pub(crate) fn with_inference<R>(
    session: &mut Session,
    shape: Vec<i64>,
    data: Vec<f32>,
    stage_id: StageId,
    f: impl FnOnce(&[i64], &[f32]) -> Result<R, StageError>,
) -> Result<R, StageError> {
    let input_tensor =
        Tensor::from_array((shape, data)).map_err(|e| StageError::ProcessingFailed {
            stage_id,
            detail: format!("tensor creation error: {e}"),
        })?;

    let outputs =
        session
            .run(ort::inputs![input_tensor])
            .map_err(|e| StageError::ProcessingFailed {
                stage_id,
                detail: format!("inference failed: {e}"),
            })?;

    let output_value = &outputs[0];
    let (out_shape, output_data) =
        output_value
            .try_extract_tensor::<f32>()
            .map_err(|e| StageError::ProcessingFailed {
                stage_id,
                detail: format!("output tensor extraction failed: {e}"),
            })?;

    f(out_shape, output_data)
}

/// Obtain a mutable reference to the session, returning a typed error if
/// `on_start` was not called.
pub(crate) fn require_session(
    session: &mut Option<Session>,
    stage_id: StageId,
) -> Result<&mut Session, StageError> {
    session
        .as_mut()
        .ok_or_else(|| StageError::ProcessingFailed {
            stage_id,
            detail: "session not initialised (on_start not called)".into(),
        })
}
