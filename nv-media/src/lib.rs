//! # nv-media
//!
//! GStreamer backend for the NextVision video perception runtime.
//!
//! This is the **only** crate in the workspace that depends on `gstreamer-rs`.
//! It provides:
//!
//! - **[`MediaIngress`]** — public trait contract for media source lifecycle.
//! - **[`FrameSink`]** — callback trait for delivering decoded frames.
//! - **[`MediaIngressFactory`]** — factory for creating ingress instances.
//! - **[`MediaSource`]** — GStreamer-backed implementation of `MediaIngress`.
//! - **[`PtzTelemetry`]** — optional PTZ metadata extracted from stream.
//! - **Source management** — constructing GStreamer pipelines from [`SourceSpec`](nv_core::SourceSpec).
//! - **Decode** — codec handling, hardware acceleration negotiation.
//! - **Zero-copy bridge** — converting `GstSample` into [`FrameEnvelope`](nv_frame::FrameEnvelope)
//!   via a ref-counted GStreamer buffer mapping that avoids copying pixel data.
//! - **Reconnection** — automatic reconnection with configurable backoff.
//! - **Discontinuity detection** — PTS gap monitoring for stream health.
//!
//! ## Architecture boundary
//!
//! GStreamer types **never** cross this crate's public API. The bridge erases
//! all GStreamer types before handing frames to downstream crates. This ensures
//! that users of the library never need to understand GStreamer internals.
//! All external consumers interact only through the [`MediaIngress`] trait
//! and its associated types.
//!
//! ## Feature flags
//!
//! - **`gst-backend`** — Enables the real GStreamer backend. Without this feature,
//!   [`MediaSource::start()`](source::MediaSource) returns
//!   `MediaError::Unsupported`. All types, traits, and state-machine logic
//!   compile without it, which allows development, testing, and downstream
//!   integration without GStreamer development libraries.
//!
//! ## Module overview
//!
//! | Module | Visibility | Purpose |
//! |---|---|---|
//! | [`ingress`] | **public** | Trait contracts (`MediaIngress`, `FrameSink`, `MediaIngressFactory`) |
//! | [`source`] | **public** | `MediaSource` — concrete implementation with reconnection |
//! | `backend` | `pub(crate)` | `GstSession` — safe adapter around GStreamer pipeline |
//! | `bridge` | `pub(crate)` | GstSample → `FrameEnvelope` conversion |
//! | `bus` | `pub(crate)` | Bus message types and mapping to `MediaEvent` |
//! | `clock` | `pub(crate)` | PTS tracking and discontinuity detection |
//! | `decode` | `pub(crate)` | Codec handling and decoder selection |
//! | `event` | `pub(crate)` | Internal `MediaEvent` enum |
//! | `pipeline` | `pub(crate)` | Pipeline builder and configuration |

// -- Public modules --
pub mod ingress;
pub mod source;
pub mod factory;

// -- Internal modules --
pub(crate) mod backend;
pub(crate) mod bridge;
pub(crate) mod bus;
pub(crate) mod clock;
pub(crate) mod decode;
pub(crate) mod event;
pub(crate) mod pipeline;
pub(crate) mod reconnect;

// -- Public re-exports --
pub use ingress::PtzProvider;
pub use ingress::SourceStatus;
pub use ingress::TickOutcome;
pub use bridge::PtzTelemetry;
pub use ingress::{FrameSink, HealthSink, MediaIngress, MediaIngressFactory};
pub use factory::{DefaultMediaFactory, GstMediaIngressFactory};
pub use source::MediaSource;
