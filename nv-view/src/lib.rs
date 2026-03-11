//! # nv-view
//!
//! Camera view-state, PTZ modeling, and motion-aware epoch management
//! for the NextVision runtime.
//!
//! This crate handles the fundamental challenge of video perception on
//! moving cameras: when the camera's view changes, spatial relationships
//! between frames change and temporal state may become invalid.
//!
//! ## Core types
//!
//! - **[`ViewState`]** — the current best estimate of the camera's view.
//! - **[`ViewSnapshot`]** — a read-only, `Arc`-wrapped snapshot given to stages.
//! - **[`ViewEpoch`]** / **[`ViewVersion`]** — monotonic counters for discontinuity and change tracking.
//! - **[`CameraMotionState`]** — whether the camera is stable, moving, or unknown.
//! - **[`TransitionPhase`]** — where in a motion transition the current frame sits.
//!
//! ## User extension points
//!
//! - **[`ViewStateProvider`]** — supplies motion information each frame (telemetry or inferred).
//! - **[`EpochPolicy`]** — controls the response to detected camera motion.
//! - **[`ViewBoundContext`]** — binds user data to a specific view for staleness checks.

pub mod bound;
pub mod camera_motion;
pub mod epoch;
pub mod provider;
pub mod ptz;
pub mod transform;
pub mod transition;
pub mod validity;
pub mod view_state;

pub use bound::{BoundContextValidity, ViewBoundContext};
pub use camera_motion::{CameraMotionState, MotionSource};
pub use epoch::{DefaultEpochPolicy, EpochDecision, EpochPolicy, EpochPolicyContext};
pub use provider::{MotionPollContext, MotionReport, ViewStateProvider};
pub use ptz::{PtzEvent, PtzTelemetry};
pub use transform::{GlobalTransformEstimate, TransformEstimationMethod};
pub use transition::TransitionPhase;
pub use validity::{ContextValidity, DegradationReason};
pub use view_state::{ViewEpoch, ViewSnapshot, ViewState, ViewVersion};
