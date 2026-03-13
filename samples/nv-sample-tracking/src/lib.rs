//! # nv-sample-tracking
//!
//! Sample multi-object tracker adapter for the NextVision perception pipeline.
//!
//! This crate is a **reference implementation** — it demonstrates how to
//! integrate a multi-object tracker with the library's [`Stage`](nv_perception::Stage)
//! trait. Library users will typically replace this with their preferred
//! tracking backend.
//!
//! The provided [`TrackerStage`] consumes
//! [`DetectionSet`](nv_perception::DetectionSet) from an upstream detector
//! and produces [`Track`](nv_perception::Track) outputs.
//!
//! ## Algorithm overview
//!
//! The bundled tracker is an observation-centric SORT variant featuring:
//!
//! 1. **Observation-centric re-update** — When a lost track is
//!    re-associated, the Kalman filter is re-run forward through cached
//!    observations to correct accumulated drift.
//!
//! 2. **Observation-centric momentum** — The cost matrix penalises
//!    direction changes, preferring associations that maintain smooth motion.
//!
//! ## Track lifecycle
//!
//! - **Tentative** — Newly created tracks remain tentative until they
//!   accumulate `min_hits` consecutive observations.
//! - **Confirmed** — Tracks with enough observations. These appear in output.
//! - **Coasted** — Confirmed tracks that miss a detection frame are coast-
//!   predicted by the Kalman filter.
//! - **Lost** — Tracks that have coasted for more than `max_age` frames
//!   are pruned.

mod config;
mod kalman;
mod matching;
mod ocsort;
mod stage;

pub use config::TrackerConfig;
pub use stage::TrackerStage;
