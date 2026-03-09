//! # nv-temporal
//!
//! Temporal state management for the NextVision video perception runtime.
//!
//! Each feed owns a [`TemporalStore`] that manages:
//!
//! - **Track histories** — past observations and lifecycle for each tracked object.
//! - **Trajectories** — spatial paths segmented by camera view epochs.
//! - **Motion features** — displacement, speed, direction for each trajectory segment.
//! - **Continuity state** — segment boundaries with causal explanations.
//! - **Retention** — bounded eviction to control memory growth.
//!
//! Stages receive a [`TemporalStoreSnapshot`] — a read-only, `Arc`-wrapped
//! view of the temporal state, captured once per frame before stage execution.

pub mod continuity;
pub mod retention;
pub mod store;
pub mod trajectory;

pub use continuity::SegmentBoundary;
pub use retention::RetentionPolicy;
pub use store::{SnapshotTrackHistory, TemporalStore, TemporalStoreSnapshot, TrackHistory};
pub use trajectory::{MotionFeatures, Trajectory, TrajectoryPoint, TrajectorySegment};
