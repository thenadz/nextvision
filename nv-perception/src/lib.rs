//! # nv-perception
//!
//! Perception stage model and artifact types for the NextVision runtime.
//!
//! This crate defines:
//!
//! - **[`Stage`]** — the core user-implementable trait for per-frame processing.
//! - **[`Detection`]**, **[`DetectionSet`]** — per-frame object detections.
//! - **[`Track`]**, **[`TrackObservation`]**, **[`TrackState`]** — tracked objects.
//! - **[`PerceptionArtifacts`]** — accumulated outputs of all stages for a frame.
//! - **[`DerivedSignal`]**, **[`SignalValue`]** — generic named signals.
//! - **[`SceneFeature`]**, **[`SceneFeatureValue`]** — scene-level feature observations.
//!
//! ## Stage execution model
//!
//! Stages execute linearly and synchronously within a feed. For each frame:
//! 1. A `PerceptionArtifacts` accumulator starts empty.
//! 2. Each stage receives a [`StageContext`] with the frame, prior artifacts,
//!    temporal snapshot, and view snapshot.
//! 3. Each stage returns a [`StageOutput`] that is merged into the accumulator.
//! 4. After all stages run, the accumulator feeds the temporal store and output.
//!
//! Stages take `&mut self` — they run on a dedicated thread per feed and are
//! never shared across threads.

pub mod artifact;
pub mod detection;
pub mod scene;
pub mod signal;
pub mod stage;
pub mod temporal_access;
pub mod track;

pub use artifact::PerceptionArtifacts;
pub use detection::{Detection, DetectionSet};
pub use scene::{SceneFeature, SceneFeatureValue};
pub use signal::{DerivedSignal, SignalValue};
pub use stage::{Stage, StageContext, StageOutput};
pub use temporal_access::TemporalAccess;
pub use track::{Track, TrackObservation, TrackState};
