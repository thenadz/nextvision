//! # nv-core
//!
//! Foundational types for the NextVision video perception runtime.
//!
//! This crate provides the shared vocabulary used by all other NextVision crates:
//!
//! - **IDs** — [`FeedId`], [`TrackId`], [`DetectionId`], [`StageId`]
//! - **Timestamps** — [`MonotonicTs`], [`WallTs`], [`Duration`]
//! - **Geometry** — [`BBox`], [`Point2`], [`Polygon`], [`AffineTransform2D`]
//! - **Errors** — [`NvError`] and its sub-enums
//! - **Metadata** — [`TypedMetadata`] type-map pattern
//! - **Health** — [`HealthEvent`], [`StopReason`]
//! - **Metrics** — [`FeedMetrics`], [`StageMetrics`]
//! - **Source** — [`SourceSpec`], [`RtspTransport`], [`ReconnectPolicy`]
//! - **Config** — [`CameraMode`], [`BackoffKind`]
//!
//! All types are domain-agnostic. No GStreamer types appear here.

pub mod config;
pub mod error;
pub mod geom;
pub mod health;
pub mod id;
pub mod metadata;
pub mod metrics;
pub mod timestamp;

// Re-export key types at crate root for ergonomic imports.
pub use config::*;
pub use error::*;
pub use geom::*;
pub use health::*;
pub use id::*;
pub use metadata::TypedMetadata;
pub use metrics::*;
pub use timestamp::*;
