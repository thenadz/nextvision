//! # nv-byo-ocsort
//!
//! OC-SORT (Observation-Centric SORT) multi-object tracker adapter for the
//! NextVision perception pipeline.
//!
//! This crate provides [`OcSortStage`], an implementation of
//! [`nv_perception::Stage`] that consumes [`DetectionSet`](nv_perception::DetectionSet)
//! from an upstream detector and produces [`Track`](nv_perception::Track) outputs.
//!
//! ## Algorithm overview
//!
//! OC-SORT extends classic SORT with:
//!
//! 1. **Observation-centric re-update (ORU)** — When a lost track is
//!    re-associated, the Kalman filter is re-run forward through cached
//!    observations to correct accumulated drift from coast-only predictions.
//!
//! 2. **Observation-centric momentum (OCM)** — The cost matrix penalises
//!    direction changes, preferring associations that maintain smooth motion
//!    trajectories.
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

pub use config::OcSortConfig;
pub use stage::OcSortStage;
