//! Consolidated diagnostics snapshots for feeds and the runtime.
//!
//! The library exposes several independent observability surfaces:
//! [`FeedMetrics`], [`QueueTelemetry`], [`DecodeStatus`], health events,
//! [`BatchMetrics`], and per-frame [`Provenance`]. Each serves a specific
//! purpose, but downstream systems that want a comprehensive view of
//! runtime state must stitch them together manually.
//!
//! This module provides **composite snapshot types** that unify the most
//! commonly needed diagnostics into single, coherent reads:
//!
//! - [`FeedDiagnostics`] — everything about one feed in one call.
//! - [`RuntimeDiagnostics`] — every feed, batch coordinator, and output
//!   channel status in one call.
//!
//! # Recommended consumption pattern
//!
//! ```text
//! ┌──────────────────┐
//! │ Dashboard / OPS  │
//! └──────┬───────────┘
//!        │ poll every 1–5 s
//!        ▼
//!   runtime.diagnostics()
//!        │
//!        ├─▶ RuntimeDiagnostics
//!        │     .uptime
//!        │     .feed_count / .max_feeds
//!        │     .output_lag  (in_lag, pending_lost)
//!        │     .batches: Vec<BatchDiagnostics>
//!        │         .processor_id, .metrics
//!        │     .feeds: Vec<FeedDiagnostics>  (sorted by FeedId)
//!        │         .alive / .paused / .uptime
//!        │         .metrics  (frames counters, restarts)
//!        │         .queues   (source/sink depth + capacity)
//!        │         .decode   (hw/sw codec status)
//!        │         .view     (stability score, context status)
//!        │         .batch_processor_id  (links to .batches)
//!        │
//!        │  (complement with event-driven streams:)
//!        │
//!   runtime.health_subscribe()   ← state transitions, errors, degradation
//!   runtime.output_subscribe()   ← per-frame provenance, admission, detections
//! ```
//!
//! The snapshot approach is intentionally poll-oriented. Snapshots are
//! cheap (atomic loads plus small allocations for the `Vec` and decode
//! detail `String`), idempotent, and do not perturb the pipeline.
//! Event-driven details (individual health events, per-frame provenance)
//! remain on their respective broadcast channels.

use std::time::Duration;

use nv_core::id::{FeedId, StageId};
use nv_core::metrics::FeedMetrics;

use crate::batch::BatchMetrics;
use crate::feed_handle::{DecodeStatus, QueueTelemetry};

// ---------------------------------------------------------------------------
// Per-feed diagnostics
// ---------------------------------------------------------------------------

/// Consolidated per-feed diagnostics snapshot.
///
/// Combines lifecycle state, throughput metrics, queue depths, decode
/// status, and view-system health into a single best-effort coherent
/// snapshot. Each field is read from independent atomics/mutexes, so
/// the snapshot is approximately — not transactionally — consistent.
///
/// Obtained via [`FeedHandle::diagnostics()`](crate::FeedHandle::diagnostics).
#[derive(Debug, Clone)]
pub struct FeedDiagnostics {
    /// The feed's unique identifier.
    pub feed_id: FeedId,
    /// Whether the worker thread is still alive.
    pub alive: bool,
    /// Whether the feed is currently paused.
    pub paused: bool,
    /// Time since the current processing session started.
    ///
    /// Resets on each restart. A feed that restarts frequently will
    /// show low uptime values.
    pub uptime: Duration,
    /// Frame counters, track count, view epoch, and restart count.
    pub metrics: FeedMetrics,
    /// Source and sink queue depths and capacities.
    pub queues: QueueTelemetry,
    /// Decode method selected by the media backend, if known.
    ///
    /// `None` until the stream starts and the backend confirms decoder
    /// negotiation.
    pub decode: Option<DecodeStatus>,
    /// Current camera view-system status.
    pub view: ViewDiagnostics,
    /// The batch coordinator this feed submits to, if any.
    ///
    /// Use this to correlate with [`RuntimeDiagnostics::batches`] for
    /// the coordinator's metrics.
    pub batch_processor_id: Option<StageId>,
}

/// Summary of the camera view-system's current health.
///
/// Fixed cameras report `status: ViewStatus::Stable` and `stability_score: 1.0`.
/// Observed (PTZ/moving) cameras reflect the live epoch policy output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewDiagnostics {
    /// Current view epoch — incremented on significant view discontinuities.
    pub epoch: u64,
    /// Stability score in `[0.0, 1.0]`. `1.0` = fully stable.
    pub stability_score: f32,
    /// High-level view health status.
    pub status: ViewStatus,
}

/// High-level camera view health.
///
/// This is a diagnostic summary of the underlying [`ContextValidity`]
/// (from `nv-view`) — intentionally simpler to avoid forcing downstream
/// consumers to depend on the view crate.
///
/// [`ContextValidity`]: nv_view::ContextValidity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewStatus {
    /// View is stable; temporal context is valid.
    Stable,
    /// View is changing; temporal context is degraded.
    Degraded,
    /// View has changed significantly; prior context is invalid.
    Invalid,
}

// ---------------------------------------------------------------------------
// Runtime-wide diagnostics
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batch diagnostics
// ---------------------------------------------------------------------------

/// Diagnostics snapshot for a batch coordinator.
///
/// One entry per coordinator created via [`Runtime::create_batch()`].
/// Included in [`RuntimeDiagnostics::batches`].
#[derive(Debug, Clone)]
pub struct BatchDiagnostics {
    /// The processor's unique stage ID.
    pub processor_id: StageId,
    /// Live metrics snapshot (counters, timing, error state).
    pub metrics: BatchMetrics,
}

// ---------------------------------------------------------------------------
// Output lag diagnostics
// ---------------------------------------------------------------------------

/// Snapshot of the output broadcast channel's saturation state.
///
/// Obtained from the runtime's internal sentinel-based lag detector.
/// A non-zero `pending_lost` during `in_lag == true` indicates the
/// channel is saturated and subscribers may be losing messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLagStatus {
    /// Whether the output channel is currently saturated.
    pub in_lag: bool,
    /// Messages lost (sentinel-observed) since the last emitted
    /// [`HealthEvent::OutputLagged`] event. Non-zero only during
    /// active saturation.
    pub pending_lost: u64,
}

// ---------------------------------------------------------------------------
// Runtime-wide diagnostics
// ---------------------------------------------------------------------------

/// Consolidated runtime-wide diagnostics snapshot.
///
/// Provides a one-call overview of every feed, batch coordinator, and
/// output channel health. Fields are read from independent sources, so
/// the snapshot is best-effort coherent — not transactionally consistent.
///
/// Obtained via [`Runtime::diagnostics()`](crate::Runtime::diagnostics)
/// or [`RuntimeHandle::diagnostics()`](crate::RuntimeHandle::diagnostics).
#[derive(Debug, Clone)]
pub struct RuntimeDiagnostics {
    /// Elapsed time since the runtime was created.
    pub uptime: Duration,
    /// Number of currently active feeds.
    pub feed_count: usize,
    /// Maximum allowed concurrent feeds.
    pub max_feeds: usize,
    /// Per-feed diagnostics, sorted by [`FeedId`] for stable iteration.
    pub feeds: Vec<FeedDiagnostics>,
    /// Diagnostics for each batch coordinator owned by this runtime.
    pub batches: Vec<BatchDiagnostics>,
    /// Current output broadcast channel saturation status.
    pub output_lag: OutputLagStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_status_equality() {
        assert_eq!(ViewStatus::Stable, ViewStatus::Stable);
        assert_ne!(ViewStatus::Stable, ViewStatus::Degraded);
        assert_ne!(ViewStatus::Degraded, ViewStatus::Invalid);
    }

    #[test]
    fn view_diagnostics_debug() {
        let v = ViewDiagnostics {
            epoch: 3,
            stability_score: 0.75,
            status: ViewStatus::Degraded,
        };
        let dbg = format!("{v:?}");
        assert!(dbg.contains("Degraded"));
        assert!(dbg.contains("0.75"));
    }
}
