//! Output types: [`OutputEnvelope`], [`OutputSink`] trait, [`SinkFactory`], and lag detection.

use std::sync::Arc;
use std::time::Instant;

use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::timestamp::{MonotonicTs, WallTs};
use nv_core::TypedMetadata;
use nv_frame::FrameEnvelope;
use nv_perception::{DerivedSignal, DetectionSet, SceneFeature, Track};
use nv_view::ViewState;
use tokio::sync::broadcast;

use crate::provenance::Provenance;

/// Factory for constructing fresh [`OutputSink`] instances.
///
/// When a sink thread times out or panics during shutdown, the feed
/// worker loses the original sink. If a `SinkFactory` was provided,
/// the next restart can construct a fresh sink rather than falling
/// back to a silent `NullSink`.
pub type SinkFactory = Box<dyn Fn() -> Box<dyn OutputSink> + Send + Sync>;

/// Controls whether the source [`FrameEnvelope`] is included in the
/// [`OutputEnvelope`].
///
/// Default is [`Never`](FrameInclusion::Never) — output contains only
/// perception artifacts. Use [`Always`](FrameInclusion::Always) when
/// downstream consumers need access to the pixel data (e.g., annotation
/// overlays, frame archival, or visual debugging).
///
/// [`Sampled`](FrameInclusion::Sampled) provides a middle ground: frames
/// are included every `interval` outputs, keeping metadata (detections,
/// tracks, signals) at full rate while reducing the cost of host
/// materialization and downstream pixel processing in the sink thread.
/// For example, `Sampled { interval: 6 }` on a 30 fps source yields
/// ~5 fps of frame delivery while perception runs at full rate.
///
/// Because `FrameEnvelope` is `Arc`-backed, inclusion is zero-copy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FrameInclusion {
    /// Never include frames in output (default).
    #[default]
    Never,
    /// Always include the source frame in output.
    Always,
    /// Include the source frame every `interval` outputs.
    ///
    /// Perception artifacts (detections, tracks, signals, provenance)
    /// flow at full rate regardless. Only the pixel payload is gated.
    ///
    /// An `interval` of `1` behaves like [`Always`](Self::Always).
    /// An `interval` of `0` behaves like [`Never`](Self::Never).
    Sampled {
        /// Include a frame every N-th output envelope.
        interval: u32,
    },
    /// Include frames at a target preview FPS, resolved dynamically from
    /// the observed source rate.
    ///
    /// During a warmup window (first ~30 frames), `fallback_interval` is
    /// used. Once the source FPS is estimated, the interval is computed
    /// as `round(source_fps / target_fps)` and the variant is resolved
    /// in-place to [`Sampled`](Self::Sampled).
    ///
    /// This avoids hardcoding an assumed source FPS at config time.
    TargetFps {
        /// Desired preview frames per second.
        target: f32,
        /// Interval to use before the source rate is known.
        fallback_interval: u32,
    },
}

impl FrameInclusion {
    /// Create a sampled frame inclusion policy with edge-case normalization.
    ///
    /// - `interval == 0` → [`Never`](Self::Never)
    /// - `interval == 1` → [`Always`](Self::Always)
    /// - `interval > 1` → [`Sampled`](Self::Sampled)
    ///
    /// Prefer this over constructing [`Sampled`](Self::Sampled) directly
    /// to avoid footgun values that silently alias other variants.
    #[must_use]
    pub fn sampled(interval: u32) -> Self {
        match interval {
            0 => Self::Never,
            1 => Self::Always,
            n => Self::Sampled { interval: n },
        }
    }

    /// Create a target-FPS frame inclusion that resolves dynamically
    /// from the observed source rate.
    ///
    /// Until the source rate is known (warmup window), falls back to
    /// `fallback_interval`. Once observed, resolves to [`Sampled`].
    ///
    /// `fallback_interval` is normalized: 0 → Never, 1 → Always.
    #[must_use]
    pub fn target_fps(target: f32, fallback_interval: u32) -> Self {
        if target <= 0.0 {
            return Self::Never;
        }
        Self::TargetFps {
            target,
            fallback_interval,
        }
    }

    /// Compute a sampled frame inclusion from a target preview FPS and
    /// an assumed source FPS.
    ///
    /// The interval is `round(source / target)`, clamped to valid range.
    ///
    /// For runtime-adaptive behavior, prefer [`target_fps`](Self::target_fps)
    /// which resolves from observed source rate instead of a static assumption.
    ///
    /// # Examples
    ///
    /// ```
    /// # use nv_runtime::FrameInclusion;
    /// assert_eq!(
    ///     FrameInclusion::from_target_fps(5.0, 30.0),
    ///     FrameInclusion::Sampled { interval: 6 },
    /// );
    /// assert_eq!(
    ///     FrameInclusion::from_target_fps(60.0, 30.0),
    ///     FrameInclusion::Always,
    /// );
    /// ```
    #[must_use]
    pub fn from_target_fps(target_fps: f32, assumed_source_fps: f32) -> Self {
        if target_fps <= 0.0 {
            return Self::Never;
        }
        if assumed_source_fps <= 0.0 || target_fps >= assumed_source_fps {
            return Self::Always;
        }
        let interval = (assumed_source_fps / target_fps).round() as u32;
        Self::sampled(interval)
    }

    /// The effective sample interval.
    ///
    /// - [`Never`](Self::Never) → `0`
    /// - [`Always`](Self::Always) → `1`
    /// - [`Sampled`](Self::Sampled) → the configured interval
    /// - [`TargetFps`](Self::TargetFps) → the fallback interval
    ///   (actual interval is determined at runtime)
    #[must_use]
    pub fn effective_interval(&self) -> u32 {
        match self {
            Self::Never => 0,
            Self::Always => 1,
            Self::Sampled { interval } => *interval,
            Self::TargetFps { fallback_interval, .. } => *fallback_interval,
        }
    }

    /// Resolve a [`TargetFps`](Self::TargetFps) variant to a concrete
    /// [`Sampled`](Self::Sampled) interval given the observed source FPS.
    ///
    /// Returns the resolved variant. Non-`TargetFps` variants are
    /// returned unchanged.
    #[must_use]
    pub fn resolve_with_source_fps(self, source_fps: f32) -> Self {
        match self {
            Self::TargetFps { target, fallback_interval } => {
                if source_fps <= 0.0 {
                    Self::sampled(fallback_interval)
                } else {
                    Self::from_target_fps(target, source_fps)
                }
            }
            other => other,
        }
    }
}

/// Summary of temporal-store admission for this frame.
///
/// Populated during the temporal commit phase. Tells downstream consumers
/// how many tracks were admitted vs. rejected due to the concurrent-track
/// cap, without requiring them to subscribe to health events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdmissionSummary {
    /// Number of tracks successfully committed to the temporal store.
    pub admitted: u32,
    /// Number of new tracks rejected because the store was at capacity
    /// and no eviction victim was available.
    pub rejected: u32,
}

/// Structured output for one processed frame.
///
/// Contains the complete perception result, view state, and full provenance.
/// Delivered to the user via [`OutputSink::emit`].
///
/// Broadcast subscribers receive `Arc<OutputEnvelope>` to avoid cloning the
/// full payload on every send. The per-feed [`OutputSink`] receives owned
/// values.
#[derive(Debug, Clone)]
pub struct OutputEnvelope {
    /// Which feed produced this output.
    pub feed_id: FeedId,
    /// Monotonic frame sequence number.
    pub frame_seq: u64,
    /// Monotonic timestamp of the source frame.
    pub ts: MonotonicTs,
    /// Wall-clock timestamp of the source frame.
    pub wall_ts: WallTs,
    /// Final detection set after all stages.
    pub detections: DetectionSet,
    /// Final track set after all stages.
    pub tracks: Vec<Track>,
    /// All derived signals from all stages.
    pub signals: Vec<DerivedSignal>,
    /// Scene-level features from all stages.
    pub scene_features: Vec<SceneFeature>,
    /// View state at the time of this frame.
    pub view: ViewState,
    /// Full provenance: stage timings, view decisions, pipeline latency.
    pub provenance: Provenance,
    /// Extensible output metadata.
    pub metadata: TypedMetadata,
    /// The source frame, present when [`FrameInclusion::Always`] is
    /// configured, or on sampled frames when [`FrameInclusion::Sampled`]
    /// is configured.
    ///
    /// This is a zero-copy `Arc` clone of the frame the pipeline processed.
    pub frame: Option<FrameEnvelope>,
    /// Temporal-store admission outcome for this frame's tracks.
    pub admission: AdmissionSummary,
}

/// User-implementable trait: receives structured outputs from the pipeline.
///
/// `emit()` is called on a dedicated per-feed sink thread, decoupled from
/// the feed's processing loop by a bounded queue. This isolation ensures
/// that a slow sink does not block perception.
///
/// The output arrives as `Arc<OutputEnvelope>` for zero-copy handoff.
/// Sinks that need an owned copy can call `Arc::unwrap_or_clone()` or
/// clone specific fields as needed.
///
/// `emit()` is wrapped in `catch_unwind` — a panicking sink emits a
/// [`HealthEvent::SinkPanic`] and the output is dropped, but the feed
/// continues. If a sink blocks during shutdown, a
/// [`HealthEvent::SinkTimeout`] is emitted and the sink thread is detached.
///
/// `emit()` is deliberately **not** async and **not** fallible:
///
/// - If the sink needs async I/O, it should buffer and channel internally.
/// - If the sink fails, it should log and drop — the perception pipeline
///   must never block on downstream consumption.
pub trait OutputSink: Send + 'static {
    /// Receive a processed output envelope.
    fn emit(&self, output: Arc<OutputEnvelope>);
}

/// Arc-wrapped output envelope for zero-copy broadcast fan-out.
///
/// Returned by [`Runtime::output_subscribe`](crate::Runtime::output_subscribe).
/// Subscribers share the same allocation; no full clone is needed per receiver.
pub type SharedOutput = Arc<OutputEnvelope>;

// ---------------------------------------------------------------------------
// LagDetector — deterministic output-channel overflow detection
// ---------------------------------------------------------------------------

/// Minimum interval between consecutive `OutputLagged` health events.
const LAG_THROTTLE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Detects output broadcast channel saturation using an internal sentinel
/// receiver.
///
/// A single instance is shared (via `Arc`) across all feed workers.
/// After every broadcast `send()`, each worker calls
/// [`check_after_send`](LagDetector::check_after_send) which:
///
/// 1. Increments a send counter (atomic, lock-free).
/// 2. When enough sends have accumulated for the ring buffer to potentially
///    wrap (≥ `capacity`), acquires the sentinel mutex via `try_lock`.
/// 3. Drains the sentinel receiver. A `TryRecvError::Lagged(n)` indicates
///    the ring buffer wrapped past the sentinel's read position by `n`
///    messages.
/// 4. Accumulates sentinel-observed wrap counts and emits a throttled
///    [`HealthEvent::OutputLagged`] event:
///    - **Transition into saturation:** always emitted immediately.
///    - **Sustained saturation:** at most once per second, carrying the
///      accumulated delta for the interval.
///    - **Recovery:** a final event flushes any remaining accumulated
///      delta.
///
/// The sentinel receiver intentionally does **not** consume messages between
/// checks, so it naturally falls behind when the channel overflows. This
/// makes saturation detection deterministic — it observes
/// `TryRecvError::Lagged(n)` rather than predicting overflow from queue
/// length.
///
/// **Semantics:** the sentinel reports ring-buffer wrap, **not** guaranteed
/// per-subscriber loss. The sentinel is the *slowest possible* consumer;
/// any external subscriber keeping up with production will experience less
/// (or no) loss. The emitted event is a channel-saturation /
/// backpressure-risk signal — not a per-subscriber loss report.
///
/// # Thread safety
///
/// The send counter is `AtomicU64`. All remaining mutable state is behind
/// a single `std::sync::Mutex` guarded by `try_lock`, so contention never
/// blocks the per-frame hot path.
///
/// # Hot-path cost
///
/// - **Most frames:** one `fetch_add` + one comparison (sends < capacity).
/// - **Every ~capacity frames:** one `try_lock` + sentinel drain + optional
///   throttled event emission.
pub(crate) struct LagDetector {
    /// Total sends since the last sentinel drain.
    sends_since_check: std::sync::atomic::AtomicU64,
    inner: std::sync::Mutex<LagDetectorInner>,
    /// Broadcast channel capacity — determines drain interval.
    capacity: usize,
    /// Minimum interval between consecutive throttled events.
    throttle_interval: std::time::Duration,
}

struct LagDetectorInner {
    sentinel: broadcast::Receiver<SharedOutput>,
    in_lag: bool,
    /// Messages evicted since the last emitted `OutputLagged` event.
    accumulated_lost: u64,
    /// When the last `OutputLagged` event was emitted.
    last_event_time: Instant,
}

impl LagDetector {
    /// Create a new lag detector from a sentinel receiver.
    ///
    /// `capacity` must match the broadcast channel's capacity so the
    /// detector knows when the ring buffer can wrap.
    ///
    /// The sentinel is a dedicated `broadcast::Receiver` created internally
    /// by the runtime — it is never exposed to users. It counts as one
    /// receiver in `Sender::receiver_count()`.
    pub fn new(sentinel: broadcast::Receiver<SharedOutput>, capacity: usize) -> Self {
        Self::with_config(sentinel, capacity, LAG_THROTTLE_INTERVAL)
    }

    /// Internal constructor with configurable throttle interval.
    ///
    /// `check_after_send` uses `throttle_interval` to cap emission rate
    /// during sustained saturation.
    fn with_config(
        sentinel: broadcast::Receiver<SharedOutput>,
        capacity: usize,
        throttle_interval: std::time::Duration,
    ) -> Self {
        Self {
            sends_since_check: std::sync::atomic::AtomicU64::new(0),
            inner: std::sync::Mutex::new(LagDetectorInner {
                sentinel,
                in_lag: false,
                accumulated_lost: 0,
                last_event_time: Instant::now(),
            }),
            capacity,
            throttle_interval,
        }
    }

    /// Record a send and, when enough sends have accumulated, drain the
    /// sentinel to detect channel saturation.
    ///
    /// Call this after every successful `broadcast::Sender::send()`.
    pub fn check_after_send(&self, health_tx: &broadcast::Sender<HealthEvent>) {
        use std::sync::atomic::Ordering;

        let sends = self
            .sends_since_check
            .fetch_add(1, Ordering::Relaxed)
            + 1;

        // The sentinel hasn't consumed any messages since the last drain.
        // It can only observe Lagged(n) when the ring buffer wraps past
        // its position, which requires more than `capacity` sends
        // (since the ring buffer holds exactly `capacity` messages).
        if (sends as usize) <= self.capacity {
            return;
        }

        let Ok(mut inner) = self.inner.try_lock() else {
            return;
        };

        // Reset send counter now that we're draining.
        self.sends_since_check.store(0, Ordering::Relaxed);

        // Drain the sentinel: it has been idle since the last drain, so
        // any ring-buffer wrap shows up as Lagged(n).
        let mut lost: u64 = 0;
        loop {
            match inner.sentinel.try_recv() {
                Ok(_) => {} // consume to catch up
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    lost += n;
                    // Receiver position was advanced past the gap.
                    // Continue draining to fully catch up.
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }

        if lost == 0 {
            // Sentinel caught up with no missed messages — buffer did not wrap.
            if inner.in_lag && inner.accumulated_lost > 0 {
                // Recovery: flush remaining accumulated delta.
                let delta = inner.accumulated_lost;
                inner.accumulated_lost = 0;
                inner.in_lag = false;
                inner.last_event_time = Instant::now();
                drop(inner);
                let _ = health_tx.send(HealthEvent::OutputLagged {
                    messages_lost: delta,
                });
            } else {
                inner.in_lag = false;
            }
            return;
        }

        // Saturation detected: sentinel missed `lost` messages due to
        // ring-buffer wrap.
        inner.accumulated_lost += lost;

        let should_emit = if !inner.in_lag {
            // Transition into saturation — always emit immediately.
            inner.in_lag = true;
            true
        } else {
            // Already saturated — emit at most once per throttle interval.
            inner.last_event_time.elapsed() >= self.throttle_interval
        };

        if should_emit {
            let delta = inner.accumulated_lost;
            inner.accumulated_lost = 0;
            inner.last_event_time = Instant::now();
            drop(inner);
            let _ = health_tx.send(HealthEvent::OutputLagged {
                messages_lost: delta,
            });
        }
    }

    /// Snapshot the current lag status without emitting any health events.
    ///
    /// Uses a blocking lock with poison recovery — safe for the
    /// diagnostics polling path (off hot path, 1–5 s interval).
    pub fn status(&self) -> crate::diagnostics::OutputLagStatus {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        crate::diagnostics::OutputLagStatus {
            in_lag: inner.in_lag,
            pending_lost: inner.accumulated_lost,
        }
    }

    /// Reset the detector after transitioning to a no-external-subscriber
    /// state.
    ///
    /// If accumulated sentinel-observed wrap from a prior saturation period
    /// is pending, it is flushed as a single final `OutputLagged` event
    /// before the reset. The sentinel is then drained without reporting
    /// (no subscriber cares about sentinel-only wrap), and all state is
    /// cleared.
    ///
    /// Uses a blocking `lock()` — this is called off the hot path, only
    /// when `receiver_count` drops below the external threshold.
    pub fn realign(&self, health_tx: &broadcast::Sender<HealthEvent>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());

        // Flush any pending sentinel-observed delta first.
        if inner.accumulated_lost > 0 {
            let delta = inner.accumulated_lost;
            inner.accumulated_lost = 0;
            let _ = health_tx.send(HealthEvent::OutputLagged {
                messages_lost: delta,
            });
        }

        // Drain the sentinel completely — discard everything, no external
        // subscriber is present to care.
        loop {
            match inner.sentinel.try_recv() {
                Ok(_) => {}
                Err(broadcast::error::TryRecvError::Lagged(_)) => {}
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }

        // Reset saturation state.
        inner.in_lag = false;
        inner.last_event_time = Instant::now();
        self.sends_since_check
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Flush any pending accumulated sentinel-observed delta as a single
    /// final event.
    ///
    /// Called on shutdown or any other point where no further sends will
    /// occur. Does **not** drain the sentinel — only emits the delta that
    /// was already accumulated by prior `check_after_send` calls.
    ///
    /// Uses a blocking `lock()`.
    pub fn flush(&self, health_tx: &broadcast::Sender<HealthEvent>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.accumulated_lost > 0 {
            let delta = inner.accumulated_lost;
            inner.accumulated_lost = 0;
            inner.in_lag = false;
            let _ = health_tx.send(HealthEvent::OutputLagged {
                messages_lost: delta,
            });
        }
    }
}

// ===========================================================================
// Unit tests for LagDetector
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use nv_core::health::HealthEvent;
    use nv_core::id::FeedId;
    use nv_core::timestamp::{MonotonicTs, WallTs};
    use nv_core::TypedMetadata;
    use nv_perception::DetectionSet;
    use nv_view::ViewState;
    use tokio::sync::broadcast;

    use crate::provenance::{Provenance, ViewProvenance};

    /// Create a minimal dummy `SharedOutput` for broadcast sending.
    fn make_dummy_output() -> SharedOutput {
        Arc::new(OutputEnvelope {
            feed_id: FeedId::new(0),
            frame_seq: 0,
            ts: MonotonicTs::from_nanos(0),
            wall_ts: WallTs::from_micros(0),
            detections: DetectionSet::empty(),
            tracks: Vec::new(),
            signals: Vec::new(),
            scene_features: Vec::new(),
            view: ViewState::fixed_initial(),
            provenance: Provenance {
                stages: Vec::new(),
                view_provenance: ViewProvenance {
                    motion_source: nv_view::MotionSource::None,
                    epoch_decision: None,
                    transition: nv_view::TransitionPhase::Settled,
                    stability_score: 1.0,
                    epoch: nv_view::view_state::ViewEpoch::INITIAL,
                    version: nv_view::view_state::ViewVersion::INITIAL,
                },
                frame_receive_ts: MonotonicTs::from_nanos(0),
                pipeline_complete_ts: MonotonicTs::from_nanos(0),
                total_latency: nv_core::Duration::from_nanos(0),
                frame_age: None,
                queue_hold_time: std::time::Duration::ZERO,
                frame_included: false,
            },
            metadata: TypedMetadata::new(),
            frame: None,
            admission: AdmissionSummary::default(),
        })
    }

    fn make_detector(
        capacity: usize,
        throttle: Duration,
    ) -> (broadcast::Sender<SharedOutput>, LagDetector) {
        let (tx, sentinel_rx) = broadcast::channel(capacity);
        let detector = LagDetector::with_config(sentinel_rx, capacity, throttle);
        (tx, detector)
    }

    fn make_health() -> (broadcast::Sender<HealthEvent>, broadcast::Receiver<HealthEvent>) {
        broadcast::channel(128)
    }

    /// Collect all `OutputLagged` deltas from the health channel.
    fn collect_lag_deltas(rx: &mut broadcast::Receiver<HealthEvent>) -> Vec<u64> {
        let mut deltas = Vec::new();
        while let Ok(evt) = rx.try_recv() {
            if let HealthEvent::OutputLagged { messages_lost } = evt {
                deltas.push(messages_lost);
            }
        }
        deltas
    }

    // D.1: delta_not_cumulative_exact
    //
    // Two equal loss intervals must produce equal deltas, not
    // cumulative totals.
    #[test]
    fn delta_not_cumulative_exact() {
        let capacity = 4;
        // Zero throttle — emit every check so we get crisp per-interval deltas.
        let (tx, detector) = make_detector(capacity, Duration::ZERO);
        let (health_tx, mut health_rx) = make_health();

        // First interval: capacity+1 sends ⇒ sentinel lags by 1.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d1 = collect_lag_deltas(&mut health_rx);

        // Second interval: same pattern.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d2 = collect_lag_deltas(&mut health_rx);

        // Both intervals should report exactly the same delta, not cumulative.
        assert!(!d1.is_empty(), "first interval should emit a lag event");
        assert!(!d2.is_empty(), "second interval should emit a lag event");

        let sum1: u64 = d1.iter().sum();
        let sum2: u64 = d2.iter().sum();
        assert_eq!(sum1, sum2, "equal loss intervals must produce equal deltas");
    }

    // D.2: throttle_blocks_event_storm
    //
    // Many overflow checks within one throttle interval must produce at most
    // one event (the transition event).
    #[test]
    fn throttle_blocks_event_storm() {
        let capacity = 4;
        // 1-second throttle — the loop below runs in well under 1 second.
        let (tx, detector) = make_detector(capacity, Duration::from_secs(1));
        let (health_tx, mut health_rx) = make_health();

        // 10 full drain cycles, each producing Lagged(1).
        for _ in 0..10 {
            for _ in 0..capacity + 1 {
                let _ = tx.send(make_dummy_output());
                detector.check_after_send(&health_tx);
            }
        }

        let deltas = collect_lag_deltas(&mut health_rx);

        // Only the first transition event should have been emitted;
        // all subsequent overflow within the same second is accumulated.
        assert_eq!(
            deltas.len(),
            1,
            "throttle should block storm: got {:?}",
            deltas
        );
        // The single event carries the first interval's delta.
        assert!(deltas[0] > 0, "emitted delta must be positive");
    }

    // D.3: throttle_allows_periodic_emission
    //
    // Sustained lag across >1 throttle interval should produce bounded
    // periodic emissions.
    #[test]
    fn throttle_allows_periodic_emission() {
        let capacity = 4;
        // Very short throttle so the test doesn't sleep long.
        let throttle = Duration::from_millis(10);
        let (tx, detector) = make_detector(capacity, throttle);
        let (health_tx, mut health_rx) = make_health();

        // First interval — triggers transition event.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d1 = collect_lag_deltas(&mut health_rx);
        assert_eq!(d1.len(), 1, "transition event");

        // Wait past throttle interval.
        std::thread::sleep(throttle + Duration::from_millis(5));

        // Another drain cycle — should now be eligible for periodic emission.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d2 = collect_lag_deltas(&mut health_rx);
        assert_eq!(d2.len(), 1, "periodic emission after interval elapsed");
        assert!(d2[0] > 0, "periodic delta must be positive");
    }

    // D.4: no_subscriber_reset_prevents_false_positive
    //
    // Induce a sentinel-only window via realign, then verify no lag is
    // reported from that window. A subsequent subscriber window must
    // report only its own losses.
    #[test]
    fn no_subscriber_reset_prevents_false_positive() {
        let capacity = 4;
        let (tx, detector) = make_detector(capacity, Duration::ZERO);
        let (health_tx, mut health_rx) = make_health();

        // Phase 1: normal sends — accumulate sentinel backlog.
        for _ in 0..capacity {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }

        // No lag events yet (sentinel is exactly capacity behind, not lagged).
        let d = collect_lag_deltas(&mut health_rx);
        assert!(d.is_empty(), "no lag before buffer wraps");

        // Phase 2: realign — simulates no-subscriber transition.
        // Any stale sentinel state is discarded.
        detector.realign(&health_tx);
        let flushed = collect_lag_deltas(&mut health_rx);
        // No accumulated loss to flush (we never exceeded capacity).
        assert!(flushed.is_empty(), "no pending loss to flush on realign");

        // Phase 3: more sends in a "new subscriber" window.
        // After realign, sends_since_check is 0 and sentinel is caught up.
        // Send capacity+1 to trigger one drain cycle with Lagged(1).
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d3 = collect_lag_deltas(&mut health_rx);

        // Should report only the loss from this interval, not from the
        // pre-realign window.
        assert!(!d3.is_empty(), "new window should produce its own lag events");
        let total: u64 = d3.iter().sum();
        assert!(total > 0 && total <= 2, "delta should reflect only new-window loss");
    }

    // D.5: flush_pending_emits_final_delta
    //
    // Create pending accumulated loss via check_after_send (throttled,
    // so not yet emitted), then flush and verify the final event.
    #[test]
    fn flush_pending_emits_final_delta() {
        let capacity = 4;
        // Long throttle — the second drain interval's loss remains pending.
        let (tx, detector) = make_detector(capacity, Duration::from_secs(60));
        let (health_tx, mut health_rx) = make_health();

        // First drain cycle — transition event emitted.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d1 = collect_lag_deltas(&mut health_rx);
        assert_eq!(d1.len(), 1, "transition event emitted");

        // Second drain cycle — throttled, loss is accumulated but not emitted.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let d2 = collect_lag_deltas(&mut health_rx);
        assert!(d2.is_empty(), "throttled — no event emitted yet");

        // Flush — should emit the stranded accumulated loss.
        detector.flush(&health_tx);
        let d3 = collect_lag_deltas(&mut health_rx);
        assert_eq!(d3.len(), 1, "flush must emit exactly one event");
        assert!(d3[0] > 0, "flushed delta must be positive");

        // Flush again — nothing pending, no event.
        detector.flush(&health_tx);
        let d4 = collect_lag_deltas(&mut health_rx);
        assert!(d4.is_empty(), "double flush must not emit");
    }

    // Verify that realign flushes real pending loss before resetting.
    #[test]
    fn realign_flushes_pending_before_reset() {
        let capacity = 4;
        // Long throttle so second-interval loss stays pending.
        let (tx, detector) = make_detector(capacity, Duration::from_secs(60));
        let (health_tx, mut health_rx) = make_health();

        // Transition event.
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let _ = collect_lag_deltas(&mut health_rx);

        // Accumulate more loss (throttled, stays pending).
        for _ in 0..capacity + 1 {
            let _ = tx.send(make_dummy_output());
            detector.check_after_send(&health_tx);
        }
        let pending = collect_lag_deltas(&mut health_rx);
        assert!(pending.is_empty(), "loss stays pending under throttle");

        // Realign should flush the pending loss, then reset.
        detector.realign(&health_tx);
        let flushed = collect_lag_deltas(&mut health_rx);
        assert_eq!(flushed.len(), 1, "realign must flush pending loss");
        assert!(flushed[0] > 0, "flushed delta must be positive");

        // After realign, no more events until new loss accumulates.
        detector.flush(&health_tx);
        let after = collect_lag_deltas(&mut health_rx);
        assert!(after.is_empty(), "no pending loss after realign + flush");
    }

    // ---------------------------------------------------------------
    // FrameInclusion normalization + constructor tests
    // ---------------------------------------------------------------

    #[test]
    fn sampled_zero_normalizes_to_never() {
        assert_eq!(FrameInclusion::sampled(0), FrameInclusion::Never);
    }

    #[test]
    fn sampled_one_normalizes_to_always() {
        assert_eq!(FrameInclusion::sampled(1), FrameInclusion::Always);
    }

    #[test]
    fn sampled_above_one_creates_sampled() {
        assert_eq!(
            FrameInclusion::sampled(6),
            FrameInclusion::Sampled { interval: 6 },
        );
    }

    #[test]
    fn from_target_fps_5_at_30_yields_interval_6() {
        assert_eq!(
            FrameInclusion::from_target_fps(5.0, 30.0),
            FrameInclusion::Sampled { interval: 6 },
        );
    }

    #[test]
    fn from_target_fps_10_at_30_yields_interval_3() {
        assert_eq!(
            FrameInclusion::from_target_fps(10.0, 30.0),
            FrameInclusion::Sampled { interval: 3 },
        );
    }

    #[test]
    fn from_target_fps_zero_is_never() {
        assert_eq!(
            FrameInclusion::from_target_fps(0.0, 30.0),
            FrameInclusion::Never,
        );
    }

    #[test]
    fn from_target_fps_negative_is_never() {
        assert_eq!(
            FrameInclusion::from_target_fps(-5.0, 30.0),
            FrameInclusion::Never,
        );
    }

    #[test]
    fn from_target_fps_above_source_is_always() {
        assert_eq!(
            FrameInclusion::from_target_fps(60.0, 30.0),
            FrameInclusion::Always,
        );
    }

    #[test]
    fn from_target_fps_equal_to_source_is_always() {
        assert_eq!(
            FrameInclusion::from_target_fps(30.0, 30.0),
            FrameInclusion::Always,
        );
    }

    #[test]
    fn from_target_fps_with_zero_source_is_always() {
        assert_eq!(
            FrameInclusion::from_target_fps(5.0, 0.0),
            FrameInclusion::Always,
        );
    }

    #[test]
    fn effective_interval_values() {
        assert_eq!(FrameInclusion::Never.effective_interval(), 0);
        assert_eq!(FrameInclusion::Always.effective_interval(), 1);
        assert_eq!(
            FrameInclusion::Sampled { interval: 6 }.effective_interval(),
            6,
        );
        assert_eq!(
            FrameInclusion::TargetFps { target: 5.0, fallback_interval: 6 }
                .effective_interval(),
            6,
        );
    }

    // ---------------------------------------------------------------
    // TargetFps constructor + resolution tests
    // ---------------------------------------------------------------

    #[test]
    fn target_fps_zero_is_never() {
        assert_eq!(FrameInclusion::target_fps(0.0, 6), FrameInclusion::Never);
    }

    #[test]
    fn target_fps_negative_is_never() {
        assert_eq!(FrameInclusion::target_fps(-5.0, 6), FrameInclusion::Never);
    }

    #[test]
    fn target_fps_positive_creates_variant() {
        assert_eq!(
            FrameInclusion::target_fps(5.0, 6),
            FrameInclusion::TargetFps { target: 5.0, fallback_interval: 6 },
        );
    }

    #[test]
    fn resolve_target_fps_with_30_source() {
        let fi = FrameInclusion::target_fps(5.0, 6);
        let resolved = fi.resolve_with_source_fps(30.0);
        assert_eq!(resolved, FrameInclusion::Sampled { interval: 6 });
    }

    #[test]
    fn resolve_target_fps_with_25_source() {
        let fi = FrameInclusion::target_fps(5.0, 6);
        let resolved = fi.resolve_with_source_fps(25.0);
        assert_eq!(resolved, FrameInclusion::Sampled { interval: 5 });
    }

    #[test]
    fn resolve_target_fps_with_15_source() {
        let fi = FrameInclusion::target_fps(5.0, 6);
        let resolved = fi.resolve_with_source_fps(15.0);
        assert_eq!(resolved, FrameInclusion::Sampled { interval: 3 });
    }

    #[test]
    fn resolve_target_fps_above_source_is_always() {
        let fi = FrameInclusion::target_fps(60.0, 6);
        let resolved = fi.resolve_with_source_fps(30.0);
        assert_eq!(resolved, FrameInclusion::Always);
    }

    #[test]
    fn resolve_target_fps_zero_source_uses_fallback() {
        let fi = FrameInclusion::target_fps(5.0, 6);
        let resolved = fi.resolve_with_source_fps(0.0);
        assert_eq!(resolved, FrameInclusion::Sampled { interval: 6 });
    }

    #[test]
    fn resolve_noop_for_sampled() {
        let fi = FrameInclusion::Sampled { interval: 3 };
        let resolved = fi.resolve_with_source_fps(30.0);
        assert_eq!(resolved, FrameInclusion::Sampled { interval: 3 });
    }

    #[test]
    fn resolve_noop_for_never() {
        let fi = FrameInclusion::Never;
        let resolved = fi.resolve_with_source_fps(30.0);
        assert_eq!(resolved, FrameInclusion::Never);
    }
}
