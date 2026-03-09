//! Scene-level features — observations about the entire visible scene.
//!
//! Unlike per-detection or per-track features, scene features describe whole-scene
//! properties such as complexity estimates, ambient conditions, or scene embeddings.
//!
//! Scene features are domain-agnostic: names and semantics are chosen by stage
//! authors. The core provides the structural type only.

use nv_core::MonotonicTs;

/// A single scene-level feature observation produced by a perception stage.
///
/// # Example
///
/// ```
/// use nv_perception::{SceneFeature, SceneFeatureValue};
/// use nv_core::MonotonicTs;
///
/// let complexity = SceneFeature {
///     name: "scene_complexity",
///     value: SceneFeatureValue::Scalar(0.82),
///     confidence: Some(0.95),
///     ts: MonotonicTs::from_nanos(1_000_000),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct SceneFeature {
    /// Feature name — compile-time constant chosen by the stage author.
    pub name: &'static str,
    /// The observed value.
    pub value: SceneFeatureValue,
    /// Optional confidence in `[0.0, 1.0]` (`None` = not applicable).
    pub confidence: Option<f32>,
    /// Timestamp at which this feature was computed.
    pub ts: MonotonicTs,
}

/// Possible values for a scene feature.
#[derive(Clone, Debug)]
pub enum SceneFeatureValue {
    /// A single scalar (e.g., density = 0.73).
    Scalar(f64),
    /// A numeric vector (e.g., scene embedding).
    Vector(Vec<f64>),
    /// A boolean flag (e.g., daytime = true).
    Boolean(bool),
    /// A categorical label (e.g., "indoor").
    Categorical(&'static str),
    /// A distribution over named categories.
    Distribution(Vec<(&'static str, f64)>),
}
