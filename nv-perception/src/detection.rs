//! Detection types — individual detections and per-frame detection sets.

use nv_core::{BBox, DetectionId, TypedMetadata};

/// A single object detection within one frame.
///
/// Detections are produced by detection stages and stored in [`DetectionSet`].
/// They carry spatial, classification, and optional re-identification data.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Unique ID within this frame's detection set.
    pub id: DetectionId,
    /// Numeric class identifier (model-defined).
    pub class_id: u32,
    /// Detection confidence score, typically in `[0, 1]`.
    pub confidence: f32,
    /// Axis-aligned bounding box in normalized `[0, 1]` coordinates.
    pub bbox: BBox,
    /// Optional re-identification or feature embedding vector.
    pub embedding: Option<Vec<f32>>,
    /// Extensible metadata (domain-specific fields, additional scores, etc.).
    pub metadata: TypedMetadata,
}

/// All detections for one frame.
///
/// This is the authoritative set for a frame at any point in the pipeline.
/// When a stage returns `Some(DetectionSet)` in its [`StageOutput`](super::StageOutput),
/// it **replaces** the accumulator's detection set entirely.
#[derive(Clone, Debug, Default)]
pub struct DetectionSet {
    /// The detections in this set.
    pub detections: Vec<Detection>,
}

impl DetectionSet {
    /// Create an empty detection set.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            detections: Vec::new(),
        }
    }

    /// Number of detections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.detections.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }
}
