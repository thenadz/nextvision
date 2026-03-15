//! Per-feed worker thread — owns the source, executor, and processing loop.
//!
//! Each feed runs on a dedicated OS thread. This gives perfect isolation:
//! a stage that blocks or panics affects only its own feed.
//!
//! # Thread model
//!
//! ```text
//! ┌──────────────┐    FrameQueue     ┌─────────────────┐   SinkQueue   ┌───────────┐
//! │ GStreamer     │──── push() ──────▶│ Feed Worker      │── push() ───▶│ Sink      │
//! │ streaming     │                   │ (OS thread)      │              │ Thread    │
//! │ thread        │                   │                  │              │           │
//! │               │                   │  pop() → stages  │              │ emit()    │
//! │  on_error()   │                   │  → broadcast     │              │ (user     │
//! │  on_eos()  ───┼── close() ──────▶│  → health events │              │  sink)    │
//! └──────────────┘                   └─────────────────┘              └───────────┘
//! ```
//!
//! The worker thread owns:
//! - `PipelineExecutor` (stages, temporal store, view state)
//! - Source handle (via `MediaIngressFactory`)
//! - `FrameQueue` (shared with `FeedFrameSink`)
//!
//! Output is decoupled from the feed thread via a bounded
//! per-feed sink queue. The sink thread calls `OutputSink::emit()`
//! asynchronously, preventing slow sinks from blocking perception.
//!
//! Shutdown is coordinated via `FeedSharedState.shutdown` (`AtomicBool`)
//! and the queue's `close()` / `wake_consumer()` methods.

mod feed_loop;
mod ingress_adapter;
mod shared_state;
mod sink;

pub(crate) use feed_loop::spawn_feed_worker;
pub(crate) use shared_state::{BroadcastHealthSink, FeedSharedState};
