//! # nv-frame
//!
//! Frame abstraction for the NextVision video perception runtime.
//!
//! The central type is [`FrameEnvelope`] — an immutable, ref-counted frame
//! that carries pixel data, timing information, and extensible metadata.
//!
//! ## Design goals
//!
//! - **Immutable after construction** — no mutable access to pixel data.
//! - **Cheap to clone** — `Clone` is an `Arc` bump, not a pixel copy.
//! - **Zero-copy from GStreamer** — via the `Mapped` pixel data variant.
//! - **Thread-safe** — `Send + Sync` for cross-thread handoff.
//! - **Self-describing** — carries format, dimensions, stride, and metadata.
//!
//! ## Pixel formats
//!
//! The library normalizes frames to a known set of [`PixelFormat`]s.
//! Conversion utilities are in the [`convert`] module (opt-in, allocates).

pub mod convert;
pub mod frame;

pub use frame::{FrameEnvelope, PixelFormat};
