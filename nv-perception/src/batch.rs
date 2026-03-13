//! Shared batch processing trait and entry types.
//!
//! # Design intent
//!
//! Most perception stages are per-feed: each feed has its own detector,
//! tracker, classifier, etc. However, inference-heavy stages (object
//! detection, embedding extraction, scene classification) benefit from
//! batching frames across multiple feeds into a single accelerator call.
//!
//! The [`BatchProcessor`] trait captures this pattern. The runtime
//! collects frames from multiple feeds, dispatches them as a batch, and
//! routes per-item results back to each feed's pipeline continuation.
//!
//! # Backend independence
//!
//! `BatchProcessor` does not assume ONNX, TensorRT, OpenVINO, or any
//! specific inference framework. Implementors choose their own backend.

use nv_core::error::StageError;
use nv_core::id::{FeedId, StageId};
use nv_frame::FrameEnvelope;
use nv_view::ViewSnapshot;

use crate::stage::{StageCapabilities, StageCategory, StageOutput};

/// An entry in a batch, passed to [`BatchProcessor::process`].
///
/// Each entry represents one frame from one feed. The processor reads
/// `frame` (and optionally `view` and `feed_id`) then writes the
/// per-item result into `output`.
///
/// # Contract
///
/// After [`BatchProcessor::process`] returns `Ok(())`, every entry's
/// `output` should be `Some(StageOutput)`. Entries left as `None` are
/// treated as if the processor returned [`StageOutput::empty()`] for
/// that item.
pub struct BatchEntry {
    /// The feed that submitted this frame.
    pub feed_id: FeedId,
    /// The video frame to process.
    ///
    /// `FrameEnvelope` is `Arc`-backed — zero-copy reference, cheap to
    /// clone. Pixel data is accessed via `frame.data()`.
    pub frame: FrameEnvelope,
    /// View-state snapshot at the time of this frame.
    ///
    /// Processors may use this to skip inference during rapid camera
    /// movement or adapt behavior based on camera stability.
    pub view: ViewSnapshot,
    /// Slot for the processor to write its per-item output.
    ///
    /// Must be set to `Some(...)` for each successfully processed item.
    pub output: Option<StageOutput>,
}

/// User-implementable trait for shared batch inference.
///
/// A `BatchProcessor` receives frames from potentially multiple feeds,
/// processes them together (typically via GPU-accelerated inference),
/// and writes per-frame results back into each [`BatchEntry::output`].
///
/// # Ownership model
///
/// The processor is **owned** by its coordinator thread (moved in via
/// `Box<dyn BatchProcessor>`). It is never shared across threads —
/// `Sync` is not required. The coordinator is the sole caller of
/// every method on the trait.
///
/// `process()` takes `&mut self`, so the processor can hold mutable
/// state (GPU session handles, scratch buffers, etc.) without interior
/// mutability.
///
/// # Lifecycle
///
/// 1. `on_start()` — called once when the batch coordinator starts.
/// 2. `process()` — called once per formed batch.
/// 3. `on_stop()` — called once when the coordinator shuts down.
///
/// # Error handling
///
/// If `process()` returns `Err(StageError)`, the entire batch fails.
/// All feed threads waiting on that batch receive the error and drop
/// their frames (same semantics as a per-feed stage error).
///
/// # Example
///
/// ```rust,no_run
/// use nv_perception::batch::{BatchProcessor, BatchEntry};
/// use nv_perception::{StageId, StageOutput, DetectionSet};
/// use nv_core::error::StageError;
///
/// struct MyDetector { /* model handle */ }
///
/// impl BatchProcessor for MyDetector {
///     fn id(&self) -> StageId { StageId("my_detector") }
///
///     fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
///         for item in items.iter_mut() {
///             let _pixels = item.frame.data();
///             // ... run model ...
///             item.output = Some(StageOutput::with_detections(DetectionSet::empty()));
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait BatchProcessor: Send + 'static {
    /// Unique name for this processor (used in provenance, metrics, logging).
    fn id(&self) -> StageId;

    /// Process a batch of frames.
    ///
    /// For each entry in `items`, read `frame` (and optionally `view`),
    /// perform inference, and set `output` to `Some(StageOutput)`.
    ///
    /// The batch may contain frames from multiple feeds.
    /// `items.len()` is bounded by the configured `max_batch_size`.
    fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError>;

    /// Called once when the batch coordinator starts.
    ///
    /// Allocate GPU resources, load models, warm up the runtime here.
    fn on_start(&mut self) -> Result<(), StageError> {
        Ok(())
    }

    /// Called once when the batch coordinator shuts down.
    ///
    /// Release resources here. Best-effort — errors are logged but not
    /// fatal.
    fn on_stop(&mut self) -> Result<(), StageError> {
        Ok(())
    }

    /// Optional category hint — defaults to [`StageCategory::FrameAnalysis`].
    ///
    /// Aligns with [`Stage::category()`](crate::Stage::category) so that
    /// validation and metrics treat the batch processor as a pipeline
    /// participant, not a foreign construct.
    fn category(&self) -> StageCategory {
        StageCategory::FrameAnalysis
    }

    /// Declare input/output capabilities for pipeline validation.
    ///
    /// When provided, the feed pipeline builder can validate that
    /// pre-batch stages satisfy the processor's input requirements
    /// and post-batch stages' inputs are met by the processor's outputs.
    ///
    /// Returns `None` by default (opts out of validation).
    fn capabilities(&self) -> Option<StageCapabilities> {
        None
    }
}
