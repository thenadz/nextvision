//! Per-feed worker thread — owns the source, executor, and processing loop.
//!
//! Each feed runs on a dedicated OS thread. This gives perfect isolation:
//! a stage that blocks or panics affects only its own feed.
//!
//! # Thread model
//!
//! ```text
//! ┌──────────────┐    FrameQueue     ┌─────────────────┐
//! │ GStreamer     │──── push() ──────▶│ Feed Worker      │
//! │ streaming     │                   │ (OS thread)      │
//! │ thread        │                   │                  │
//! │               │                   │  pop() → stages  │
//! │  on_error()   │                   │  → output_sink   │
//! │  on_eos()  ───┼── close() ──────▶│  → health events │
//! └──────────────┘                   └─────────────────┘
//! ```
//!
//! The worker thread owns:
//! - `PipelineExecutor` (stages, temporal store, view state)
//! - Source handle (via `MediaIngressFactory`)
//! - `FrameQueue` (shared with `FeedFrameSink`)
//!
//! Shutdown is coordinated via `FeedSharedState.shutdown` (`AtomicBool`)
//! and the queue's `close()` method.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::{MediaError, NvError, RuntimeError};
use nv_core::health::{HealthEvent, StopReason};
use nv_core::id::FeedId;
use nv_core::metrics::FeedMetrics;
use nv_frame::FrameEnvelope;
use nv_media::ingress::{FrameSink, HealthSink, MediaIngressFactory};
use tokio::sync::broadcast;

use crate::backpressure::BackpressurePolicy;
use crate::executor::PipelineExecutor;
use crate::feed::FeedConfig;
use crate::output::{LagDetector, OutputSink, SharedOutput};
use crate::queue::{FrameQueue, PushOutcome};
use crate::shutdown::{RestartPolicy, RestartTrigger};

// ---------------------------------------------------------------------------
// Shared state between FeedHandle and the worker thread
// ---------------------------------------------------------------------------

/// Shared state accessed atomically by both the `FeedHandle` (user thread)
/// and the feed worker thread.
pub(crate) struct FeedSharedState {
    pub id: FeedId,
    pub paused: AtomicBool,
    pub shutdown: AtomicBool,
    pub frames_received: AtomicU64,
    pub frames_dropped: AtomicU64,
    pub frames_processed: AtomicU64,
    pub tracks_active: AtomicU64,
    pub view_epoch: AtomicU64,
    pub restarts: AtomicU32,
    /// Set to `false` when the worker thread exits.
    pub alive: AtomicBool,
    /// Condvar for pause/resume: the worker waits on this instead of spin-sleeping.
    /// The Mutex guards a `bool` that mirrors `paused` — the Condvar wakes
    /// the worker when the pause state changes or shutdown is requested.
    pub pause_condvar: (Mutex<bool>, Condvar),
}

impl FeedSharedState {
    pub fn new(id: FeedId) -> Self {
        Self {
            id,
            paused: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            frames_received: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            frames_processed: AtomicU64::new(0),
            tracks_active: AtomicU64::new(0),
            view_epoch: AtomicU64::new(0),
            restarts: AtomicU32::new(0),
            alive: AtomicBool::new(true),
            pause_condvar: (Mutex::new(false), Condvar::new()),
        }
    }

    /// Snapshot the current metrics.
    pub fn metrics(&self) -> FeedMetrics {
        FeedMetrics {
            feed_id: self.id,
            frames_received: self.frames_received.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            tracks_active: self.tracks_active.load(Ordering::Relaxed),
            view_epoch: self.view_epoch.load(Ordering::Relaxed),
            restarts: self.restarts.load(Ordering::Relaxed),
        }
    }

    /// Request shutdown and wake the worker if it is paused.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        let (_lock, cvar) = &self.pause_condvar;
        cvar.notify_one();
    }
}

// ---------------------------------------------------------------------------
// FrameSink adapter — bridges media ingress → FrameQueue
// ---------------------------------------------------------------------------

/// Adapter that implements [`FrameSink`] by pushing into a [`FrameQueue`].
///
/// Created per feed session. The `Arc<FrameQueue>` is shared with the
/// feed worker thread's pop loop.
struct FeedFrameSink {
    queue: Arc<FrameQueue>,
    shared: Arc<FeedSharedState>,
    health_tx: broadcast::Sender<HealthEvent>,
    feed_id: FeedId,
}

impl FrameSink for FeedFrameSink {
    fn on_frame(&self, frame: FrameEnvelope) {
        self.shared.frames_received.fetch_add(1, Ordering::Relaxed);
        let outcome = self.queue.push(frame);
        match outcome {
            PushOutcome::Accepted => {}
            PushOutcome::DroppedOldest | PushOutcome::Rejected => {
                let dropped = self.shared.frames_dropped.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = self.health_tx.send(HealthEvent::BackpressureDrop {
                    feed_id: self.feed_id,
                    frames_dropped: dropped,
                });
            }
        }
    }

    fn on_error(&self, _error: MediaError) {
        // Source-level errors are reported by the source's HealthSink.
        // The FrameSink on_error is informational — the source handles
        // reconnection internally.
    }

    fn on_eos(&self) {
        self.queue.close();
    }
}

// ---------------------------------------------------------------------------
// HealthSink adapter — forwards to broadcast channel
// ---------------------------------------------------------------------------

/// Forwards [`HealthEvent`]s from a [`MediaSource`] to the runtime's
/// broadcast channel.
pub(crate) struct BroadcastHealthSink {
    tx: broadcast::Sender<HealthEvent>,
}

impl BroadcastHealthSink {
    pub fn new(tx: broadcast::Sender<HealthEvent>) -> Self {
        Self { tx }
    }
}

impl HealthSink for BroadcastHealthSink {
    fn emit(&self, event: HealthEvent) {
        let _ = self.tx.send(event);
    }
}

// ---------------------------------------------------------------------------
// FeedWorker — the per-feed thread entry point
// ---------------------------------------------------------------------------

/// Why the processing loop exited.
enum ExitReason {
    /// Graceful shutdown was requested.
    Shutdown,
    /// Source ended (EOS/queue closed) — may be eligible for restart.
    SourceEnded,
    /// Non-looping file source reached end-of-stream — terminal, no restart.
    FileEos,
    /// A stage panicked — may trigger restart if policy allows.
    StagePanic,
}

/// Spawns and runs the per-feed worker thread.
///
/// Returns the `Arc<FeedSharedState>` and the thread `JoinHandle`.
///
/// # Errors
///
/// Returns `RuntimeError::ThreadSpawnFailed` if the OS thread cannot be created.
pub(crate) fn spawn_feed_worker(
    feed_id: FeedId,
    config: FeedConfig,
    factory: Arc<dyn MediaIngressFactory>,
    health_tx: broadcast::Sender<HealthEvent>,
    output_tx: broadcast::Sender<SharedOutput>,
    lag_detector: Arc<LagDetector>,
) -> Result<(Arc<FeedSharedState>, std::thread::JoinHandle<()>), NvError> {
    let shared = Arc::new(FeedSharedState::new(feed_id));
    let shared_clone = Arc::clone(&shared);
    let is_file_nonloop = config.source.is_file_nonloop();

    let handle = std::thread::Builder::new()
        .name(format!("nv-feed-{}", feed_id))
        .spawn(move || {
            let mut worker = FeedWorker {
                feed_id,
                factory,
                source_spec: config.source,
                reconnect_policy: config.reconnect,
                backpressure: config.backpressure,
                restart_policy: config.restart,
                executor: PipelineExecutor::new(
                    feed_id,
                    config.stages,
                    config.temporal,
                    config.camera_mode,
                    config.view_state_provider,
                    config.epoch_policy,
                ),
                output_sink: config.output_sink,
                ptz_provider: config.ptz_provider,
                health_tx,
                output_tx,
                shared: shared_clone,
                is_file_nonloop,
                lag_detector,
                had_external_subscribers: false,
            };
            worker.run();
        })
        .map_err(|e| {
            NvError::Runtime(RuntimeError::ThreadSpawnFailed {
                detail: e.to_string(),
            })
        })?;

    Ok((shared, handle))
}

/// Per-feed worker: owns all feed-local state and runs the processing loop.
struct FeedWorker {
    feed_id: FeedId,
    factory: Arc<dyn MediaIngressFactory>,
    source_spec: SourceSpec,
    reconnect_policy: ReconnectPolicy,
    backpressure: BackpressurePolicy,
    restart_policy: RestartPolicy,
    executor: PipelineExecutor,
    output_sink: Box<dyn OutputSink>,
    ptz_provider: Option<Arc<dyn nv_media::PtzProvider>>,
    health_tx: broadcast::Sender<HealthEvent>,
    output_tx: broadcast::Sender<SharedOutput>,
    shared: Arc<FeedSharedState>,
    /// Whether this is a non-looping file source (EOS is terminal).
    is_file_nonloop: bool,
    /// Shared lag detector — sentinel-based, runtime-global.
    lag_detector: Arc<LagDetector>,
    /// Whether external subscribers existed when we last emitted output.
    /// Used to detect the transition to no-subscribers so we can realign
    /// the lag detector even when the send is skipped.
    had_external_subscribers: bool,
}

impl FeedWorker {
    /// Main entry point — runs until shutdown or restart budget exhausted.
    fn run(&mut self) {
        let mut restart_count: u32 = 0;
        let mut session_start = Instant::now();

        loop {
            if self.shared.shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Reset temporal state on restart.
            if restart_count > 0 {
                self.executor.clear_temporal();
            }

            // Start stages.
            if let Err(e) = self.executor.start_stages() {
                tracing::error!(
                    feed_id = %self.feed_id,
                    error = %e,
                    "stage on_start failed"
                );
                if !self.can_restart(restart_count, &session_start, &ExitReason::SourceEnded) {
                    self.emit_feed_stopped(StopReason::Fatal {
                        detail: format!("stage startup failed: {e}"),
                    });
                    break;
                }
                restart_count = self.bump_restart(&mut session_start, restart_count);
                self.sleep_restart_delay();
                continue;
            }

            // Mark session start (for restart_window tracking).
            session_start = Instant::now();

            // Create queue + source.
            let queue = Arc::new(FrameQueue::new(self.backpressure.clone()));
            let sink_adapter = FeedFrameSink {
                queue: Arc::clone(&queue),
                shared: Arc::clone(&self.shared),
                health_tx: self.health_tx.clone(),
                feed_id: self.feed_id,
            };

            let source = self.factory.create(
                self.feed_id,
                self.source_spec.clone(),
                self.reconnect_policy.clone(),
                self.ptz_provider.clone(),
            );

            let mut source = match source {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "failed to create media source"
                    );
                    self.executor.stop_stages();
                    if !self.can_restart(restart_count, &session_start, &ExitReason::SourceEnded) {
                        self.emit_feed_stopped(StopReason::Fatal {
                            detail: format!("source creation failed: {e}"),
                        });
                        break;
                    }
                    restart_count = self.bump_restart(&mut session_start, restart_count);
                    self.sleep_restart_delay();
                    continue;
                }
            };

            // Start the source — frames begin flowing to the queue.
            match source.start(Box::new(sink_adapter)) {
                Ok(()) => {
                    tracing::info!(feed_id = %self.feed_id, "feed started");
                }
                Err(e) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "source start failed"
                    );
                    self.executor.stop_stages();
                    if !self.can_restart(restart_count, &session_start, &ExitReason::SourceEnded) {
                        self.emit_feed_stopped(StopReason::Fatal {
                            detail: format!("source start failed: {e}"),
                        });
                        break;
                    }
                    restart_count = self.bump_restart(&mut session_start, restart_count);
                    self.sleep_restart_delay();
                    continue;
                }
            }

            // Processing loop.
            let exit_reason = self.processing_loop(&queue);

            // Cleanup.
            let _ = source.stop();
            self.executor.stop_stages();

            match exit_reason {
                ExitReason::Shutdown => {
                    self.emit_feed_stopped(StopReason::UserRequested);
                    break;
                }
                ExitReason::FileEos => {
                    // Non-looping file source — terminal, no restart.
                    let _ = self.health_tx.send(HealthEvent::SourceEos {
                        feed_id: self.feed_id,
                    });
                    self.emit_feed_stopped(StopReason::EndOfStream);
                    break;
                }
                ExitReason::SourceEnded | ExitReason::StagePanic => {
                    if !self.can_restart(restart_count, &session_start, &exit_reason) {
                        let detail = match exit_reason {
                            ExitReason::StagePanic => format!(
                                "stage panic (trigger {:?} does not allow restart, or budget exhausted after {} restarts)",
                                self.restart_policy.restart_on, restart_count,
                            ),
                            _ => {
                                format!("restart budget exhausted after {} restarts", restart_count,)
                            }
                        };
                        self.emit_feed_stopped(StopReason::Fatal { detail });
                        break;
                    }
                    restart_count = self.bump_restart(&mut session_start, restart_count);
                    self.sleep_restart_delay();
                }
            }
        }

        self.shared.alive.store(false, Ordering::Relaxed);
    }

    /// Frame processing loop: pop → execute stages → emit output.
    ///
    /// Returns the reason the loop exited.
    fn processing_loop(&mut self, queue: &Arc<FrameQueue>) -> ExitReason {
        loop {
            // Check shutdown first.
            if self.shared.shutdown.load(Ordering::Relaxed) {
                return ExitReason::Shutdown;
            }

            // Handle pause: wait on condvar until resumed or shutdown.
            if self.shared.paused.load(Ordering::Relaxed) {
                let (lock, cvar) = &self.shared.pause_condvar;
                let mut paused = lock.lock().unwrap();
                while *paused && !self.shared.shutdown.load(Ordering::Relaxed) {
                    paused = cvar.wait(paused).unwrap();
                }
                continue;
            }

            // Pop the next frame.
            let frame = match queue.pop(&self.shared.shutdown) {
                Some(f) => f,
                None => {
                    // Queue closed (EOS) or shutdown.
                    if self.shared.shutdown.load(Ordering::Relaxed) {
                        return ExitReason::Shutdown;
                    }
                    // File EOS is terminal for non-looping file sources.
                    if self.is_file_nonloop {
                        return ExitReason::FileEos;
                    }
                    return ExitReason::SourceEnded;
                }
            };

            // Run the pipeline.
            let (maybe_output, health_events) = self.executor.process_frame(&frame);

            // Check if any stage panicked.
            let had_panic = health_events
                .iter()
                .any(|e| matches!(e, HealthEvent::StagePanic { .. }));

            // Broadcast health events.
            for event in health_events {
                let _ = self.health_tx.send(event);
            }

            // Update shared metrics.
            self.shared.frames_processed.fetch_add(1, Ordering::Relaxed);
            self.shared
                .tracks_active
                .store(self.executor.track_count() as u64, Ordering::Relaxed);
            self.shared
                .view_epoch
                .store(self.executor.view_epoch(), Ordering::Relaxed);

            // Only emit output if the frame was not dropped.
            if let Some(output) = maybe_output {
                // Broadcast to external subscribers. receiver_count()
                // includes the internal sentinel receiver; only send when
                // there is at least one external subscriber (count > 1).
                let has_external = self.output_tx.receiver_count() > 1;

                // Arc-wrap first so both broadcast and sink share the
                // same allocation instead of cloning the full output.
                let shared_out: SharedOutput = Arc::new(output);

                if has_external {
                    self.had_external_subscribers = true;
                    let _ = self.output_tx.send(Arc::clone(&shared_out));
                    self.lag_detector.check_after_send(&self.health_tx);
                }

                // Emit output to the per-feed sink with panic containment.
                // A panicking OutputSink must not tear down the feed thread.
                let sink_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let output = Arc::try_unwrap(shared_out).unwrap_or_else(|arc| (*arc).clone());
                    self.output_sink.emit(output);
                }));
                if sink_result.is_err() {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        "OutputSink::emit() panicked — output dropped",
                    );
                    let _ = self.health_tx.send(HealthEvent::SinkPanic {
                        feed_id: self.feed_id,
                    });
                }
            }

            // Subscriber transition: external subscribers were present on
            // a prior frame but are gone now. Realign the lag detector to
            // flush any pending accumulated loss and drain stale sentinel
            // backlog. Checked unconditionally — outside the output gate —
            // so the transition is detected even on frames that produced no
            // output or where the broadcast send was skipped.
            if self.had_external_subscribers && self.output_tx.receiver_count() <= 1 {
                self.had_external_subscribers = false;
                self.lag_detector.realign(&self.health_tx);
            }

            // If a stage panicked, exit the loop so the worker can
            // decide whether to restart based on the restart policy.
            if had_panic {
                return ExitReason::StagePanic;
            }
        }
    }

    /// Check whether another restart is allowed given the exit reason and policy.
    ///
    /// Returns `false` (no restart) when any of these hold:
    /// - `restart_on == Never`
    /// - `max_restarts == 0` (semantics: never restart regardless of window)
    /// - trigger mode does not match the exit reason (e.g. `SourceFailure`
    ///   policy with a `StagePanic` exit)
    /// - restart budget exhausted within the current restart window
    ///
    /// The `restart_window` resets the running counter only when
    /// `max_restarts > 0`. A window cannot override `max_restarts == 0`.
    fn can_restart(
        &self,
        current_count: u32,
        session_start: &Instant,
        reason: &ExitReason,
    ) -> bool {
        // Never-restart policy: unconditional.
        if self.restart_policy.restart_on == RestartTrigger::Never {
            return false;
        }

        // max_restarts == 0 means "never restart", regardless of window.
        if self.restart_policy.max_restarts == 0 {
            return false;
        }

        // Check that the trigger mode matches the exit reason.
        match (&self.restart_policy.restart_on, reason) {
            // SourceFailure only restarts on source-level issues, not panics.
            (RestartTrigger::SourceFailure, ExitReason::StagePanic) => return false,
            // SourceOrStagePanic covers both.
            (RestartTrigger::SourceOrStagePanic, _) => {}
            (RestartTrigger::SourceFailure, _) => {}
            (RestartTrigger::Never, _) => return false,
        }

        // Apply restart_window: if the session ran longer than the window,
        // the counter effectively resets (we allow the restart).
        // This only applies when max_restarts > 0 (checked above).
        let window: std::time::Duration = self.restart_policy.restart_window.into();
        if session_start.elapsed() >= window {
            return true; // counter is considered reset
        }

        current_count < self.restart_policy.max_restarts
    }

    /// Increment restart counter (resetting if the window elapsed) and
    /// emit the FeedRestarting health event.
    ///
    /// Returns the new *window-scoped* counter (used for budget checking).
    /// The shared `restarts` metric tracks the cumulative total.
    fn bump_restart(&self, session_start: &mut Instant, current_count: u32) -> u32 {
        let window: std::time::Duration = self.restart_policy.restart_window.into();
        let new_count = if session_start.elapsed() >= window {
            // Window elapsed — reset counter.
            1
        } else {
            current_count + 1
        };
        *session_start = Instant::now();

        let total = self.shared.restarts.fetch_add(1, Ordering::Relaxed) + 1;
        self.emit_restarting(total);
        new_count
    }

    fn emit_restarting(&self, count: u32) {
        let _ = self.health_tx.send(HealthEvent::FeedRestarting {
            feed_id: self.feed_id,
            restart_count: count,
        });
    }

    fn emit_feed_stopped(&self, reason: StopReason) {
        let _ = self.health_tx.send(HealthEvent::FeedStopped {
            feed_id: self.feed_id,
            reason,
        });
    }

    fn sleep_restart_delay(&self) {
        let delay: std::time::Duration = self.restart_policy.restart_delay.into();
        // Check shutdown during sleep so we don't block shutdown by the full delay.
        let step = std::time::Duration::from_millis(50);
        let mut remaining = delay;
        while remaining > std::time::Duration::ZERO {
            if self.shared.shutdown.load(Ordering::Relaxed) {
                return;
            }
            let sleep_for = remaining.min(step);
            std::thread::sleep(sleep_for);
            remaining = remaining.saturating_sub(sleep_for);
        }
    }
}
