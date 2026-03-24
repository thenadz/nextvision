use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use nv_core::config::{ReconnectPolicy, SourceSpec};
use nv_core::error::{NvError, RuntimeError};
use nv_core::health::{HealthEvent, StopReason};
use nv_core::id::FeedId;
use nv_media::ingress::{IngressOptions, MediaIngress, MediaIngressFactory, SourceStatus};
use tokio::sync::broadcast;

use crate::backpressure::BackpressurePolicy;
use crate::executor::PipelineExecutor;
use crate::feed::FeedConfig;
use crate::output::{LagDetector, OutputSink, SharedOutput, SinkFactory};
use crate::queue::{FrameQueue, PopResult};
use crate::shutdown::{RestartPolicy, RestartTrigger};

use super::ingress_adapter::{BackpressureThrottle, FeedFrameSink};
use super::shared_state::FeedSharedState;
use super::sink::{NullSink, SinkBpThrottle, SinkRecovery, SinkWorker};

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
    /// A stage failed during startup (`on_start`) — restart may be allowed
    /// under `SourceOrStagePanic` policy but not under `SourceFailure`.
    StageStartFailed,
    /// Source reported permanently stopped (reconnection budget exhausted).
    SourceStopped,
    /// The sink worker thread could not be spawned. A terminal health
    /// event has already been emitted by `processing_loop`.
    SinkSpawnFailed,
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
    let batch_processor_id = config.batch.as_ref().map(|b| b.processor_id());
    let shared = Arc::new(FeedSharedState::new(feed_id, batch_processor_id));
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
                    config.batch,
                    config.post_batch_stages,
                    config.temporal,
                    config.camera_mode,
                    config.view_state_provider,
                    config.epoch_policy,
                    config.frame_inclusion,
                    Arc::clone(&shared_clone.shutdown),
                ),
                output_sink: config.output_sink,
                sink_factory: config.sink_factory,
                ptz_provider: config.ptz_provider,
                health_tx,
                output_tx,
                shared: shared_clone,
                is_file_nonloop,
                lag_detector,
                had_external_subscribers: false,
                sink_queue_capacity: config.sink_queue_capacity,
                sink_shutdown_timeout: config.sink_shutdown_timeout,
                decode_preference: config.decode_preference,
                post_decode_hook: config.post_decode_hook,
                device_residency: config.device_residency,
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
    /// Optional factory for constructing a fresh sink after timeout/panic.
    sink_factory: Option<SinkFactory>,
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
    /// Per-feed output sink queue capacity (bounded channel to sink thread).
    sink_queue_capacity: usize,
    /// Timeout for joining the sink worker thread during shutdown/restart.
    sink_shutdown_timeout: Duration,
    /// Decode preference — plumbed through to the media ingress factory.
    decode_preference: nv_media::DecodePreference,
    /// Optional post-decode hook — plumbed through to the media ingress.
    post_decode_hook: Option<nv_media::PostDecodeHook>,
    /// Device residency mode — plumbed through to the media ingress.
    device_residency: nv_media::DeviceResidency,
}

/// Drop guard that ensures `FeedSharedState::alive` is set to `false`
/// when the worker exits — even on panic.
struct AliveGuard(Arc<FeedSharedState>);

impl Drop for AliveGuard {
    fn drop(&mut self) {
        self.0.alive.store(false, Ordering::Relaxed);
    }
}

impl FeedWorker {
    /// Main entry point — runs until shutdown or restart budget exhausted.
    fn run(&mut self) {
        // Guard ensures alive is set false even on unexpected panic.
        let _guard = AliveGuard(Arc::clone(&self.shared));
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
                if !self.try_restart(
                    &mut restart_count,
                    &mut session_start,
                    &ExitReason::StageStartFailed,
                    format!("stage startup failed: {e}"),
                ) {
                    break;
                }
                continue;
            }

            // Mark session start (for restart_window tracking).
            session_start = Instant::now();

            // Create queue + source.
            let queue = Arc::new(FrameQueue::new(self.backpressure.clone()));
            self.shared.set_queue(Some(Arc::clone(&queue)));

            let bp_throttle = BackpressureThrottle::new();
            let sink_adapter = FeedFrameSink {
                queue: Arc::clone(&queue),
                shared: Arc::clone(&self.shared),
                health_tx: self.health_tx.clone(),
                feed_id: self.feed_id,
                bp_throttle,
            };

            let mut options = IngressOptions::new(
                self.feed_id,
                self.source_spec.clone(),
                self.reconnect_policy.clone(),
            )
            .with_decode_preference(self.decode_preference)
            .with_device_residency(self.device_residency.clone());
            if let Some(ref ptz) = self.ptz_provider {
                options = options.with_ptz_provider(Arc::clone(ptz));
            }
            if let Some(ref hook) = self.post_decode_hook {
                options = options.with_post_decode_hook(Arc::clone(hook));
            }
            let source = self.factory.create(options);

            let mut source = match source {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "failed to create media source"
                    );
                    if !self.cleanup_and_try_restart(
                        &mut restart_count,
                        &mut session_start,
                        format!("source creation failed: {e}"),
                    ) {
                        break;
                    }
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
                    if !self.cleanup_and_try_restart(
                        &mut restart_count,
                        &mut session_start,
                        format!("source start failed: {e}"),
                    ) {
                        break;
                    }
                    continue;
                }
            }

            // Processing loop.
            let exit_reason = self.processing_loop(&queue, &mut *source);

            // Cleanup.
            queue.close();
            self.shared.set_queue(None);
            let _ = source.stop();
            self.stop_and_flush_stages();

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
                ExitReason::SourceStopped => {
                    // Source FSM reported permanently stopped.
                    // For non-looping files this is terminal EndOfStream.
                    // For everything else it means reconnect budget was
                    // exhausted — also terminal.
                    if self.is_file_nonloop {
                        let _ = self.health_tx.send(HealthEvent::SourceEos {
                            feed_id: self.feed_id,
                        });
                        self.emit_feed_stopped(StopReason::EndOfStream);
                    } else {
                        self.emit_feed_stopped(StopReason::Fatal {
                            detail: "source stopped (reconnection budget exhausted)".into(),
                        });
                    }
                    break;
                }
                ExitReason::SinkSpawnFailed => {
                    // Terminal health event already emitted by processing_loop.
                    break;
                }
                ExitReason::SourceEnded | ExitReason::StagePanic | ExitReason::StageStartFailed => {
                    let detail = match exit_reason {
                        ExitReason::StagePanic | ExitReason::StageStartFailed => format!(
                            "stage failure (trigger {:?} does not allow restart, or budget exhausted after {} restarts)",
                            self.restart_policy.restart_on, restart_count,
                        ),
                        _ => format!("restart budget exhausted after {} restarts", restart_count,),
                    };
                    if !self.try_restart(
                        &mut restart_count,
                        &mut session_start,
                        &exit_reason,
                        detail,
                    ) {
                        break;
                    }
                }
            }
        }

        // `alive` is set to `false` by the AliveGuard drop.
    }

    /// Frame processing loop: pop → execute stages → emit output.
    ///
    /// The source is ticked on every frame and on queue pop timeouts. The
    /// timeout deadline is driven entirely by the source's
    /// [`TickOutcome::next_tick`] hint (e.g., reconnect backoff). When the
    /// source has no specific deadline (`next_tick: None`), the queue pop
    /// waits indefinitely — woken by incoming frames, source errors
    /// (via `wake_consumer()`), shutdown, or EOS.
    ///
    /// Returns the reason the loop exited.
    fn processing_loop(
        &mut self,
        queue: &Arc<FrameQueue>,
        source: &mut dyn MediaIngress,
    ) -> ExitReason {
        // Record session start time and sink capacity for telemetry.
        if let Ok(mut t) = self.shared.session_started_at.lock() {
            *t = Instant::now();
        }
        self.shared
            .sink_capacity
            .store(self.sink_queue_capacity, Ordering::Relaxed);
        self.shared.sink_occupancy.store(0, Ordering::Relaxed);

        // Spawn sink worker — output is decoupled from this thread.
        let sink = std::mem::replace(&mut self.output_sink, Box::new(NullSink));
        let sink_worker = match SinkWorker::spawn(
            self.feed_id,
            sink,
            self.health_tx.clone(),
            self.sink_queue_capacity,
            Arc::clone(&self.shared.sink_occupancy),
        ) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!(
                    feed_id = %self.feed_id,
                    error = %e,
                    "failed to spawn sink worker thread",
                );
                self.emit_feed_stopped(StopReason::Fatal {
                    detail: format!("sink worker spawn failed: {e}"),
                });
                // The real sink was already moved out; self.output_sink
                // holds NullSink. We cannot recover the original sink,
                // so exit immediately.
                return ExitReason::SinkSpawnFailed;
            }
        };

        let result = self.run_processing_loop(queue, source, &sink_worker);

        // Recover the sink from the worker thread so it can be reused
        // if the feed restarts.
        match sink_worker.shutdown(&self.health_tx, self.feed_id, self.sink_shutdown_timeout) {
            SinkRecovery::Recovered(sink) => {
                self.output_sink = sink;
            }
            SinkRecovery::Lost => {
                // The sink thread timed out or panicked. When a factory is
                // available, construct a fresh sink so the next restart
                // produces output instead of silently discarding it.
                if let Some(ref factory) = self.sink_factory {
                    tracing::info!(
                        feed_id = %self.feed_id,
                        "sink was lost (timeout/panic) — reconstructing from factory",
                    );
                    self.output_sink = factory();
                } else {
                    tracing::warn!(
                        feed_id = %self.feed_id,
                        "sink was lost (timeout/panic) and no sink_factory configured — \
                         output will be discarded on restart",
                    );
                    // Alert operators: the feed will continue processing but
                    // all output is silently discarded until the feed is
                    // reconfigured with a new sink or removed.
                    self.emit_feed_stopped(StopReason::Fatal {
                        detail: "sink lost with no sink_factory — \
                                 output will be discarded on restart"
                            .into(),
                    });
                    self.output_sink = Box::new(NullSink);
                }
            }
        }

        result
    }

    fn run_processing_loop(
        &mut self,
        queue: &Arc<FrameQueue>,
        source: &mut dyn MediaIngress,
        sink_worker: &SinkWorker,
    ) -> ExitReason {
        // Seed the initial tick so that any deadline armed during source
        // start (e.g., liveness timeout, reconnect backoff) is honoured
        // from the very first queue pop. Without this, a source that
        // arms a deadline but emits no frames would cause the worker to
        // wait indefinitely.
        //
        // We intentionally do NOT short-circuit on SourceStatus::Stopped
        // here: the producer thread may have raced ahead and signalled
        // EOS before we enter the loop. The correct exit for that case
        // is through queue.pop() → Closed → SourceEnded (restartable),
        // not SourceStopped (terminal).
        let initial = source.tick();
        let mut next_tick_hint: Option<Duration> = initial.next_tick;

        // Sync initial decode status.
        Self::sync_decode_status(source, &self.shared);

        let mut sink_bp = SinkBpThrottle::new();

        let reason = self.run_loop_inner(
            queue,
            source,
            sink_worker,
            &mut sink_bp,
            &mut next_tick_hint,
        );

        // Flush any accumulated tail from the sink backpressure
        // coalescer so the final delta is not silently lost.
        sink_bp.flush(&self.health_tx, self.feed_id);

        reason
    }

    fn run_loop_inner(
        &mut self,
        queue: &Arc<FrameQueue>,
        source: &mut dyn MediaIngress,
        sink_worker: &SinkWorker,
        sink_bp: &mut SinkBpThrottle,
        next_tick_hint: &mut Option<Duration>,
    ) -> ExitReason {
        loop {
            // Check shutdown first.
            if self.shared.shutdown.load(Ordering::Relaxed) {
                return ExitReason::Shutdown;
            }

            // Handle pause: pause the source, wait on condvar, then resume.
            if self.shared.paused.load(Ordering::Relaxed) {
                // Pause the media source to stop decoding/network I/O.
                if let Err(e) = source.pause() {
                    tracing::warn!(
                        feed_id = %self.feed_id,
                        error = %e,
                        "source pause failed (continuing paused state)"
                    );
                }
                let (lock, cvar) = &self.shared.pause_condvar;
                let mut paused = lock.lock().unwrap_or_else(|e| e.into_inner());
                while *paused && !self.shared.shutdown.load(Ordering::Relaxed) {
                    paused = cvar.wait(paused).unwrap_or_else(|e| e.into_inner());
                }
                // Resume the source when leaving paused state.
                if !self.shared.shutdown.load(Ordering::Relaxed) {
                    if let Err(e) = source.resume() {
                        tracing::warn!(
                            feed_id = %self.feed_id,
                            error = %e,
                            "source resume failed"
                        );
                    }
                }
                continue;
            }

            // Pop the next frame. The deadline is driven entirely by
            // the source's tick hint. None → wait indefinitely.
            let deadline = next_tick_hint.map(|d| Instant::now() + d);
            let pop_result = queue.pop(&self.shared.shutdown, deadline);
            let frame = match pop_result {
                PopResult::Frame(f) => f,
                PopResult::Closed => {
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
                PopResult::Timeout | PopResult::Wake => {
                    // No frame available — tick the source to drive
                    // bus polling and reconnection.
                    let outcome = source.tick();
                    *next_tick_hint = outcome.next_tick;
                    Self::sync_decode_status(source, &self.shared);
                    if outcome.status == SourceStatus::Stopped {
                        return ExitReason::SourceStopped;
                    }
                    continue;
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
            self.shared
                .view_stability_score
                .store(self.executor.stability_score().to_bits(), Ordering::Relaxed);
            self.shared
                .view_context_validity
                .store(self.executor.context_validity_ordinal(), Ordering::Relaxed);

            // Only emit output if the frame was not dropped.
            if let Some(output) = maybe_output {
                // Broadcast to external subscribers. receiver_count()
                // includes the internal sentinel receiver; only send when
                // there is at least one external subscriber (count > 1).
                let has_external = self.output_tx.receiver_count() > 1;

                // Arc-wrap first so broadcast and sink share the same
                // allocation — no deep clone needed.
                let shared_out: SharedOutput = Arc::new(output);

                if has_external {
                    self.had_external_subscribers = true;
                    let _ = self.output_tx.send(Arc::clone(&shared_out));
                    self.lag_detector.check_after_send(&self.health_tx);
                }

                // Enqueue for the sink worker (non-blocking).
                sink_worker.send(shared_out, sink_bp, &self.health_tx, self.feed_id);
            }

            // Tick the source after processing to drain any pending bus
            // events and update the next_tick_hint for the next pop
            // deadline.  Do NOT exit immediately when the source reports
            // Stopped — buffered frames in the queue must be drained
            // first.  The next pop() will return either another Frame
            // (continue processing) or Closed/Timeout/Wake, at which
            // point the stopped status is picked up and the loop exits
            // cleanly.
            let outcome = source.tick();
            *next_tick_hint = outcome.next_tick;
            Self::sync_decode_status(source, &self.shared);

            // Subscriber transition: external subscribers were present on
            // a prior frame but are gone now. Realign the lag detector to
            // flush any pending accumulated loss and drain stale sentinel
            // backlog.
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

    /// Copy the latest decode status from the media source into shared state.
    ///
    /// This is a cheap check: the source returns a cached `Option` and
    /// the write only happens when the value actually changes.
    fn sync_decode_status(source: &dyn MediaIngress, shared: &FeedSharedState) {
        if let Some(status) = source.decode_status() {
            if let Ok(mut guard) = shared.decode_status.lock() {
                *guard = Some(status);
            }
        }
    }

    /// Stop all stages and flush any pending batch rejections/timeouts,
    /// emitting the corresponding health events.
    fn stop_and_flush_stages(&mut self) {
        self.executor.stop_stages();
        for evt in [
            self.executor.flush_batch_rejections(),
            self.executor.flush_batch_timeouts(),
            self.executor.flush_batch_in_flight_rejections(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = self.health_tx.send(evt);
        }
    }

    /// Common cleanup for source-create or source-start failures:
    /// clear queue, stop stages, flush batch rejections, then try restart.
    ///
    /// Returns `true` if restart was accepted (caller should `continue`),
    /// `false` if denied (caller should `break`).
    fn cleanup_and_try_restart(
        &mut self,
        restart_count: &mut u32,
        session_start: &mut Instant,
        detail: String,
    ) -> bool {
        self.shared.set_queue(None);
        self.stop_and_flush_stages();
        self.try_restart(
            restart_count,
            session_start,
            &ExitReason::SourceEnded,
            detail,
        )
    }

    /// Attempt a restart. Returns `true` if the restart was accepted
    /// (caller should `continue`), `false` if restart was denied
    /// (caller should `break`).
    fn try_restart(
        &self,
        restart_count: &mut u32,
        session_start: &mut Instant,
        reason: &ExitReason,
        fatal_detail: String,
    ) -> bool {
        if !self.can_restart(*restart_count, session_start, reason) {
            self.emit_feed_stopped(StopReason::Fatal {
                detail: fatal_detail,
            });
            return false;
        }
        *restart_count = self.bump_restart(session_start, *restart_count);
        self.sleep_restart_delay();
        true
    }

    /// Check whether another restart is allowed given the exit reason and policy.
    fn can_restart(
        &self,
        current_count: u32,
        session_start: &Instant,
        reason: &ExitReason,
    ) -> bool {
        if self.restart_policy.restart_on == RestartTrigger::Never {
            return false;
        }
        if self.restart_policy.max_restarts == 0 {
            return false;
        }
        match (&self.restart_policy.restart_on, reason) {
            (RestartTrigger::SourceFailure, ExitReason::StagePanic | ExitReason::StageStartFailed) => return false,
            (RestartTrigger::SourceOrStagePanic, _) => {}
            (RestartTrigger::SourceFailure, _) => {}
            (RestartTrigger::Never, _) => return false,
        }
        let window: std::time::Duration = self.restart_policy.restart_window.into();
        if session_start.elapsed() >= window {
            return true;
        }
        current_count < self.restart_policy.max_restarts
    }

    /// Increment restart counter and emit FeedRestarting health event.
    fn bump_restart(&self, session_start: &mut Instant, current_count: u32) -> u32 {
        let window: std::time::Duration = self.restart_policy.restart_window.into();
        let new_count = if session_start.elapsed() >= window {
            1
        } else {
            current_count + 1
        };
        *session_start = Instant::now();

        let total = self.shared.restarts.fetch_add(1, Ordering::Relaxed) + 1;
        let _ = self.health_tx.send(HealthEvent::FeedRestarting {
            feed_id: self.feed_id,
            restart_count: total,
        });
        new_count
    }

    fn emit_feed_stopped(&self, reason: StopReason) {
        let _ = self.health_tx.send(HealthEvent::FeedStopped {
            feed_id: self.feed_id,
            reason,
        });
    }

    fn sleep_restart_delay(&self) {
        let delay: std::time::Duration = self.restart_policy.restart_delay.into();
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
