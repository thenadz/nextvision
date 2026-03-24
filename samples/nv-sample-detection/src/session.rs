//! Shared ONNX session loading for detection models.
//!
//! Used by both [`DetectorStage`](crate::DetectorStage) (per-feed) and
//! [`DetectorBatchProcessor`](crate::DetectorBatchProcessor) (cross-feed batch).

use ort::session::Session;
use tracing::{debug, warn};
#[cfg(feature = "gpu")]
use tracing::info;

use nv_core::error::StageError;
use nv_core::id::StageId;

use crate::config::DetectorConfig;

/// Expected number of columns in the end-to-end output tensor.
/// Each detection row is `[x1, y1, x2, y2, confidence, class_id]`.
const EXPECTED_OUTPUT_COLS: i64 = 6;

/// Register the CUDA execution provider on the session builder.
///
/// Uses [`Heuristic`](ort::ep::cuda::ConvAlgorithmSearch::Heuristic) cuDNN
/// convolution algorithm search instead of the default `Exhaustive`.
/// `Exhaustive` benchmarks every cuDNN implementation on first inference
/// (causing multi-second latency spikes) and on some GPU architectures
/// (notably Jetson Xavier SM 7.2) selects algorithms that produce
/// numerically divergent results, systematically suppressing confidence
/// values by 10–20×.
#[cfg(feature = "gpu")]
fn configure_gpu(
    mut builder: ort::session::builder::SessionBuilder,
    stage_id: StageId,
) -> Result<ort::session::builder::SessionBuilder, StageError> {
    use ort::ep::{self, ExecutionProvider};

    let cuda_available = ep::CUDA::default().is_available().unwrap_or(false);
    info!(
        stage = %stage_id,
        cuda_available,
        "ORT execution-provider probe"
    );

    if cuda_available {
        let ep = ep::CUDA::default()
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic);
        match ep.register(&mut builder) {
            Ok(()) => info!(stage = %stage_id, "CUDA EP registered (conv search: heuristic)"),
            Err(e) => warn!(
                stage = %stage_id,
                error = %e,
                "CUDA EP registration failed — CPU fallback only"
            ),
        }
    } else {
        warn!(
            stage = %stage_id,
            "CUDA EP not available in loaded ORT library — inference will run on CPU"
        );
    }

    Ok(builder)
}

/// Load a detector ONNX model session, validating the output schema.
///
/// Checks that the model's first output has 3 dimensions with the last
/// dimension equal to 6 (`[batch, max_dets, 6]`). This catches mismatched
/// exports (e.g., raw 85-column output) at startup instead of at
/// first inference.
///
/// When `config.gpu` is true and the `gpu` crate feature is enabled,
/// the session is configured with the CUDA execution provider before
/// falling back to CPU.
pub fn load_session(config: &DetectorConfig, stage_id: StageId) -> Result<Session, StageError> {
    let mut builder = Session::builder().map_err(|e| StageError::ModelLoadFailed {
        stage_id,
        detail: format!("session builder failed: {e}"),
    })?;

    #[cfg(feature = "gpu")]
    if config.gpu {
        builder = configure_gpu(builder, stage_id)?;
    }

    #[cfg(not(feature = "gpu"))]
    if config.gpu {
        warn!(
            stage = %stage_id,
            "GPU requested but `gpu` feature not enabled — falling back to CPU"
        );
    }

    #[cfg(feature = "gpu")]
    if config.gpu {
        warn!(
            model = %config.model_path.display(),
            stage = %stage_id,
            "loading ONNX model with CUDA EP — first inference will trigger \
             JIT compilation and may take 30+ seconds; subsequent calls are fast"
        );
    }

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
