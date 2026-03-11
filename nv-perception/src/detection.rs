//! Detection types — individual detections and per-frame detection sets.

use nv_core::{BBox, DetectionId, TypedMetadata};

/// A single object detection within one frame.
///
/// Detections are produced by detection stages and stored in [`DetectionSet`].
/// They carry spatial, classification, and optional re-identification data.
///
/// Use [`Detection::builder()`] for ergonomic construction with optional fields.
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

impl Detection {
    /// Create a builder for a [`Detection`].
    ///
    /// # Example
    ///
    /// ```
    /// use nv_core::{BBox, DetectionId};
    /// use nv_perception::Detection;
    ///
    /// let det = Detection::builder(
    ///     DetectionId::new(1),
    ///     0,
    ///     0.95,
    ///     BBox::new(0.1, 0.2, 0.3, 0.4),
    /// )
    /// .embedding(vec![0.1, 0.2, 0.3])
    /// .build();
    /// ```
    #[must_use]
    pub fn builder(id: DetectionId, class_id: u32, confidence: f32, bbox: BBox) -> DetectionBuilder {
        DetectionBuilder {
            id,
            class_id,
            confidence,
            bbox,
            embedding: None,
            metadata: TypedMetadata::new(),
        }
    }
}

/// Builder for [`Detection`].
///
/// All required fields are set in [`Detection::builder()`]. Optional fields
/// can be chained before calling [`build()`](DetectionBuilder::build).
pub struct DetectionBuilder {
    id: DetectionId,
    class_id: u32,
    confidence: f32,
    bbox: BBox,
    embedding: Option<Vec<f32>>,
    metadata: TypedMetadata,
}

impl DetectionBuilder {
    /// Set the re-identification / feature embedding vector.
    #[must_use]
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Insert a typed metadata value.
    #[must_use]
    pub fn meta<T: Clone + Send + Sync + 'static>(mut self, val: T) -> Self {
        self.metadata.insert(val);
        self
    }

    /// Build the detection.
    #[must_use]
    pub fn build(self) -> Detection {
        Detection {
            id: self.id,
            class_id: self.class_id,
            confidence: self.confidence,
            bbox: self.bbox,
            embedding: self.embedding,
            metadata: self.metadata,
        }
    }
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

impl From<Vec<Detection>> for DetectionSet {
    fn from(detections: Vec<Detection>) -> Self {
        Self { detections }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::BBox;

    #[test]
    fn builder_required_fields_only() {
        let det = Detection::builder(
            DetectionId::new(1),
            0,
            0.95,
            BBox::new(0.1, 0.2, 0.3, 0.4),
        )
        .build();

        assert_eq!(det.id, DetectionId::new(1));
        assert_eq!(det.class_id, 0);
        assert!((det.confidence - 0.95).abs() < f32::EPSILON);
        assert!(det.embedding.is_none());
        assert!(det.metadata.is_empty());
    }

    #[test]
    fn builder_with_embedding_and_meta() {
        #[derive(Clone, Debug, PartialEq)]
        struct Extra(u32);

        let det = Detection::builder(
            DetectionId::new(2),
            5,
            0.8,
            BBox::new(0.0, 0.0, 1.0, 1.0),
        )
        .embedding(vec![0.1, 0.2, 0.3])
        .meta(Extra(42))
        .build();

        assert_eq!(det.embedding.as_ref().unwrap().len(), 3);
        assert_eq!(det.metadata.get::<Extra>(), Some(&Extra(42)));
    }

    #[test]
    fn detection_set_from_vec() {
        let dets = vec![
            Detection::builder(DetectionId::new(1), 0, 0.9, BBox::new(0.0, 0.0, 0.5, 0.5))
                .build(),
            Detection::builder(DetectionId::new(2), 1, 0.7, BBox::new(0.5, 0.5, 1.0, 1.0))
                .build(),
        ];
        let set: DetectionSet = dets.into();
        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
    }
}
