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
//!
//! ## Data residency
//!
//! Frames are either **host-resident** (CPU-accessible bytes, the default)
//! or **device-resident** (opaque accelerated buffer on a GPU/NPU).
//!
//! [`Residency`] describes where the data lives. [`DataAccess`] describes
//! what host-access is available:
//!
//! | `Residency` | `DataAccess` | Meaning |
//! |---|---|---|
//! | `Host` | `HostReadable` | Zero-copy `&[u8]` via `host_data()`. |
//! | `Device` | `MappableToHost` | Materializable via `require_host_data()`. |
//! | `Device` | `Opaque` | No host path; use `accelerated_handle::<T>()`. |
//!
//! ### CPU-only stages
//!
//! Use [`FrameEnvelope::require_host_data()`] — all paths return
//! `Cow::Borrowed`: host frames borrow directly from the frame,
//! device frames borrow from a per-frame cache populated on first access.
//! Opaque frames return `Err(`[`FrameAccessError`]`)`.
//!
//! ```ignore
//! let pixels = frame.require_host_data()
//!     .map_err(|e| StageError::ProcessingFailed {
//!         stage_id: MY_STAGE,
//!         detail: e.to_string(),
//!     })?;
//! process_cpu(&pixels);
//! ```
//!
//! ### Memoization
//!
//! Materialized host bytes are cached in the frame's `Arc`-shared inner
//! state. Repeated calls (including from clones) reuse the cache at zero
//! cost. Failures are also cached — frame data is immutable, so a
//! transfer that fails will not succeed on retry.
//!
//! ### Mixed CPU/GPU stages
//!
//! Branch on [`DataAccess`]:
//!
//! ```ignore
//! match frame.data_access() {
//!     DataAccess::HostReadable => { /* host_data() */ }
//!     DataAccess::MappableToHost => { /* require_host_data() or accelerated_handle */ }
//!     DataAccess::Opaque => { /* accelerated_handle::<T>() only */ }
//!     _ => {}
//! }
//! ```
//!
//! ## Adapter crates and the opaque handle
//!
//! The accelerated handle is intended only for:
//! - Backend adapter crates bridging accelerated decode buffers
//! - GPU tensors destined for inference
//! - Accelerator-native frame storage
//!
//! It must **not** be used for general stage metadata or cross-stage
//! messaging. Use [`nv_core::TypedMetadata`] for those purposes.
//!
//! When constructing device frames, adapter crates may optionally provide
//! a [`HostMaterializeFn`] that enables CPU fallback. The materializer
//! returns [`HostBytes`] — either an owned `Vec<u8>` or a zero-copy
//! mapped view. See [`FrameEnvelope::new_device()`] for details.

pub mod convert;
pub mod frame;

pub use convert::ConvertError;
pub use frame::{
    DataAccess, FrameAccessError, FrameEnvelope, HostBytes, HostMaterializeFn, PixelFormat,
    Residency,
};
