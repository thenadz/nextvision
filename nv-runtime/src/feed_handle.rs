//! Feed handle and runtime-observable types.

use std::sync::Arc;
use std::time::Duration;

use nv_core::id::FeedId;
use nv_core::metrics::FeedMetrics;

/// Handle to a running feed.
///
/// Provides per-feed monitoring and control: health events, metrics,
/// queue telemetry, uptime, pause/resume, and stop.
///
/// Backed by `Arc<FeedSharedState>` — cloning is cheap (Arc bump).
/// Metrics are read from the same atomic counters that the feed worker
/// thread writes.
#[derive(Clone)]
pub struct FeedHandle {
    shared: Arc<crate::worker::FeedSharedState>,
}

/// Snapshot of queue depths and capacities for a feed.
///
/// **Source queue**: bounded frame queue between media ingress and pipeline.
/// **Sink queue**: bounded channel between pipeline and output sink.
///
/// Values are approximate under concurrent access — sufficient for
/// monitoring and dashboards, not for synchronization.
#[derive(Debug, Clone, Copy)]
pub struct QueueTelemetry {
    /// Current number of frames in the source queue.
    pub source_depth: usize,
    /// Maximum capacity of the source queue.
    pub source_capacity: usize,
    /// Current number of outputs in the sink queue.
    pub sink_depth: usize,
    /// Maximum capacity of the sink queue.
    pub sink_capacity: usize,
}

/// Snapshot of the decode method selected by the media backend.
///
/// Available after the stream starts and the backend confirms decoder
/// negotiation. Use [`FeedHandle::decode_status()`] to poll.
#[derive(Debug, Clone)]
pub struct DecodeStatus {
    /// Whether hardware or software decoding was selected.
    pub outcome: nv_core::health::DecodeOutcome,
    /// Backend-specific detail string (e.g., GStreamer element name).
    ///
    /// Intended for diagnostics and dashboards — do not match on its
    /// contents programmatically.
    pub detail: String,
}

impl FeedHandle {
    /// Create a feed handle (internal — constructed by the runtime).
    pub(crate) fn new(shared: Arc<crate::worker::FeedSharedState>) -> Self {
        Self { shared }
    }

    /// The feed's unique identifier.
    #[must_use]
    pub fn id(&self) -> FeedId {
        self.shared.id
    }

    /// Whether the feed is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.shared
            .paused
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether the worker thread is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.shared
            .alive
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get a snapshot of the feed's current metrics.
    ///
    /// Reads live atomic counters maintained by the feed worker thread.
    #[must_use]
    pub fn metrics(&self) -> FeedMetrics {
        self.shared.metrics()
    }

    /// Get a snapshot of the feed's source and sink queue depths/capacities.
    ///
    /// Source queue depth reads the frame queue's internal lock briefly.
    /// Sink queue depth reads an atomic counter with no locking.
    ///
    /// If no processing session is active (between restarts or after
    /// shutdown), both depths return 0.
    #[must_use]
    pub fn queue_telemetry(&self) -> QueueTelemetry {
        let (source_depth, source_capacity) = self.shared.source_queue_telemetry();
        QueueTelemetry {
            source_depth,
            source_capacity,
            sink_depth: self.shared.sink_occupancy.load(std::sync::atomic::Ordering::Relaxed),
            sink_capacity: self.shared.sink_capacity.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Elapsed time since the feed's current processing session started.
    ///
    /// **Semantics: session-scoped uptime.** The clock resets on each
    /// successful start or restart. If the feed has not started yet or
    /// is between restart attempts, the value reflects the time since the
    /// last session began.
    ///
    /// Useful for monitoring feed stability: a feed that restarts
    /// frequently will show low uptime values.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.shared.session_uptime()
    }

    /// The decode method confirmed by the media backend for this feed.
    ///
    /// Returns `None` if no decode decision has been made yet (the
    /// stream has not started, the backend has not negotiated a decoder,
    /// or the feed is between restarts).
    #[must_use]
    pub fn decode_status(&self) -> Option<DecodeStatus> {
        let (outcome, detail) = self.shared.decode_status()?;
        Some(DecodeStatus { outcome, detail })
    }

    /// Get a consolidated diagnostics snapshot of this feed.
    ///
    /// Composes lifecycle state, metrics, queue depths, decode status,
    /// and view-system health into a single read. All data comes from
    /// the same atomic counters the individual accessors use — this is
    /// a convenience composite, not a new data source.
    ///
    /// Suitable for periodic polling (1–5 s) by dashboards and health
    /// probes.
    #[must_use]
    pub fn diagnostics(&self) -> crate::diagnostics::FeedDiagnostics {
        use crate::diagnostics::{ViewDiagnostics, ViewStatus};

        let metrics = self.metrics();
        let validity_ordinal = self
            .shared
            .view_context_validity
            .load(std::sync::atomic::Ordering::Relaxed);
        let status = match validity_ordinal {
            0 => ViewStatus::Stable,
            1 => ViewStatus::Degraded,
            _ => ViewStatus::Invalid,
        };
        let stability_bits = self
            .shared
            .view_stability_score
            .load(std::sync::atomic::Ordering::Relaxed);

        crate::diagnostics::FeedDiagnostics {
            feed_id: self.id(),
            alive: self.is_alive(),
            paused: self.is_paused(),
            uptime: self.uptime(),
            metrics,
            queues: self.queue_telemetry(),
            decode: self.decode_status(),
            view: ViewDiagnostics {
                epoch: metrics.view_epoch,
                stability_score: f32::from_bits(stability_bits),
                status,
            },
            batch_processor_id: self.shared.batch_processor_id,
        }
    }

    /// Pause the feed (stop pulling frames from source; stages idle).
    ///
    /// Uses a condvar to wake the worker without spin-sleeping.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed is already paused.
    pub fn pause(&self) -> Result<(), nv_core::NvError> {
        let was_paused = self
            .shared
            .paused
            .swap(true, std::sync::atomic::Ordering::Relaxed);
        if was_paused {
            return Err(nv_core::NvError::Runtime(
                nv_core::error::RuntimeError::AlreadyPaused,
            ));
        }
        // Mirror into the condvar-guarded bool.
        let (lock, _cvar) = &self.shared.pause_condvar;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
        Ok(())
    }

    /// Resume a paused feed.
    ///
    /// Notifies the worker thread via condvar so it wakes immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if the feed is not paused.
    pub fn resume(&self) -> Result<(), nv_core::NvError> {
        let was_paused = self
            .shared
            .paused
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        if !was_paused {
            return Err(nv_core::NvError::Runtime(
                nv_core::error::RuntimeError::NotPaused,
            ));
        }
        // Mirror into the condvar-guarded bool and wake the worker.
        let (lock, cvar) = &self.shared.pause_condvar;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = false;
        cvar.notify_one();
        Ok(())
    }
}
