use std::sync::atomic::Ordering;
use std::time::Instant;

use nv_core::error::StageError;
use nv_core::health::HealthEvent;
use nv_core::id::FeedId;
use nv_core::metrics::StageMetrics;
use nv_core::timestamp::{Duration, MonotonicTs};
use nv_frame::FrameEnvelope;
use nv_perception::batch::BatchEntry;
use nv_perception::{PerceptionArtifacts, Stage, StageContext};
use nv_temporal::TemporalStoreSnapshot;
use nv_view::ViewSnapshot;

use crate::batch::BatchSubmitError;
use crate::output::{FrameInclusion, OutputEnvelope};
use crate::provenance::{
    Provenance, StageOutcomeCategory, StageProvenance, StageResult, ViewProvenance,
};

use super::{
    BATCH_IN_FLIGHT_THROTTLE, BATCH_REJECTION_THROTTLE, BATCH_TIMEOUT_THROTTLE,
    FRAME_LAG_THRESHOLD_MS, FRAME_LAG_THROTTLE, PipelineExecutor, instant_to_ts_impl,
};

/// Number of frames to observe before resolving [`FrameInclusion::TargetFps`]
/// to a concrete interval. Must be ≥ 2 for a meaningful FPS estimate.
const TARGET_FPS_WARMUP_FRAMES: u64 = 30;

impl PipelineExecutor {
    /// Flush any accumulated batch rejection count as a final
    /// [`HealthEvent::BatchSubmissionRejected`].
    ///
    /// Called at lifecycle boundaries (stop/restart) so that short
    /// rejection bursts that didn't reach the throttle window are
    /// still surfaced.
    pub fn flush_batch_rejections(&mut self) -> Option<HealthEvent> {
        if self.batch_rejection_count == 0 {
            return None;
        }
        let processor_id = self.batch.as_ref()?.processor_id();
        let count = self.batch_rejection_count;
        self.batch_rejection_count = 0;
        self.last_batch_rejection_event = None;
        Some(HealthEvent::BatchSubmissionRejected {
            feed_id: self.feed_id,
            processor_id,
            dropped_count: count,
        })
    }

    /// Flush any accumulated batch timeout count as a final
    /// [`HealthEvent::BatchTimeout`].
    ///
    /// Called at lifecycle boundaries (stop/restart) so that short
    /// timeout bursts that didn't reach the throttle window are
    /// still surfaced.
    pub fn flush_batch_timeouts(&mut self) -> Option<HealthEvent> {
        if self.batch_timeout_count == 0 {
            return None;
        }
        let processor_id = self.batch.as_ref()?.processor_id();
        let count = self.batch_timeout_count;
        self.batch_timeout_count = 0;
        self.last_batch_timeout_event = None;
        Some(HealthEvent::BatchTimeout {
            feed_id: self.feed_id,
            processor_id,
            timed_out_count: count,
        })
    }

    /// Flush any accumulated batch in-flight cap rejections as a final
    /// [`HealthEvent::BatchInFlightExceeded`].
    ///
    /// Called at lifecycle boundaries (stop/restart) so that short
    /// bursts that didn't reach the throttle window are still surfaced.
    pub fn flush_batch_in_flight_rejections(&mut self) -> Option<HealthEvent> {
        if self.batch_in_flight_rejection_count == 0 {
            return None;
        }
        let processor_id = self.batch.as_ref()?.processor_id();
        let count = self.batch_in_flight_rejection_count;
        self.batch_in_flight_rejection_count = 0;
        self.last_batch_in_flight_rejection_event = None;
        Some(HealthEvent::BatchInFlightExceeded {
            feed_id: self.feed_id,
            processor_id,
            rejected_count: count,
        })
    }

    // ------------------------------------------------------------------
    // Frame processing
    // ------------------------------------------------------------------

    /// Attempt to resolve [`FrameInclusion::TargetFps`] to a concrete
    /// [`FrameInclusion::Sampled`] interval using observed source cadence.
    ///
    /// Called once per frame when the inclusion policy is `TargetFps`.
    /// Records the source-domain timestamp ([`FrameEnvelope::ts()`]) of
    /// the first frame, then after [`TARGET_FPS_WARMUP_FRAMES`] frames
    /// computes:
    ///   `source_fps = (frames − 1) / (ts_last − ts_first)`
    /// and resolves in-place with
    /// [`FrameInclusion::resolve_with_source_fps`].
    ///
    /// Using source timestamps rather than wall-clock [`Instant`] ensures
    /// that processing delays (CUDA/TRT JIT compilation, transient compute
    /// stalls) do not distort the FPS estimate.
    fn try_resolve_target_fps(&mut self, frame_ts: MonotonicTs) {
        if self.fps_warmup_start_ts.is_none() {
            self.fps_warmup_start_ts = Some(frame_ts);
            return;
        }

        if self.frames_processed < TARGET_FPS_WARMUP_FRAMES {
            return;
        }

        // Compute observed FPS from source-domain timestamps.
        let start_nanos = self.fps_warmup_start_ts.unwrap().as_nanos();
        let end_nanos = frame_ts.as_nanos();
        if end_nanos <= start_nanos {
            return; // Clock glitch or non-monotonic — try again next frame.
        }

        let elapsed_secs = (end_nanos - start_nanos) as f64 / 1_000_000_000.0;
        if elapsed_secs <= 0.0 {
            return;
        }

        let source_fps = (self.frames_processed - 1) as f64 / elapsed_secs;
        let resolved = self.frame_inclusion.resolve_with_source_fps(source_fps as f32);

        tracing::info!(
            feed = %self.feed_id,
            observed_source_fps = format!("{source_fps:.1}"),
            resolved = %format!("{resolved:?}"),
            "resolved TargetFps frame inclusion from observed source rate",
        );

        // Reset sample counter to avoid stale state from the fallback window.
        self.frame_sample_counter = 0;
        self.frame_inclusion = resolved;
    }

    /// Process a single frame through the pipeline.
    ///
    /// Execution order:
    /// 1. View orchestration (for observed cameras).
    /// 2. Pre-batch stages (sequential, on feed thread).
    /// 3. Batch point (if present) — submit to shared coordinator, block
    ///    until result, merge into artifacts.
    /// 4. Post-batch stages (sequential, on feed thread).
    /// 5. Temporal commit + retention.
    ///
    /// Returns `Some((output, health_events))` on success (even if individual
    /// stages produced errors that were recorded).
    /// Returns `None` if a stage error drops the frame.
    ///
    /// A stage *panic* is a sentinel: health events are returned and the
    /// caller decides whether to restart.
    pub fn process_frame(
        &mut self,
        frame: &FrameEnvelope,
        queue_hold_time: std::time::Duration,
    ) -> (Option<OutputEnvelope>, Vec<HealthEvent>) {
        let t_pipeline_start = Instant::now();
        let frame_receive_ts = self.instant_to_ts(t_pipeline_start);

        // Compute wall-clock age: how stale is this frame?
        // Skip if wall_ts is zero (sentinel for "no wall timestamp").
        let frame_wall_micros = frame.wall_ts().as_micros();
        let frame_age = if frame_wall_micros > 0 {
            let wall_now = nv_core::timestamp::WallTs::now();
            let age_micros = wall_now.as_micros().saturating_sub(frame_wall_micros);
            if age_micros > 0 {
                Some(Duration::from_nanos(age_micros as u64 * 1_000))
            } else {
                None
            }
        } else {
            None
        };

        let mut health_events = Vec::new();

        // --- Frame lag detection ---
        if let Some(age) = frame_age {
            let age_ms = age.as_nanos() / 1_000_000;
            if age_ms >= FRAME_LAG_THRESHOLD_MS as u64 {
                self.frame_lag_count += 1;
                self.frame_lag_peak_age_ms = self.frame_lag_peak_age_ms.max(age_ms);
                let now = Instant::now();
                let should_emit = self
                    .last_frame_lag_event
                    .is_none_or(|t| now.duration_since(t) >= FRAME_LAG_THROTTLE);
                if should_emit {
                    let count = self.frame_lag_count;
                    let peak = self.frame_lag_peak_age_ms;
                    self.frame_lag_count = 0;
                    self.frame_lag_peak_age_ms = 0;
                    self.last_frame_lag_event = Some(now);
                    health_events.push(HealthEvent::FrameLag {
                        feed_id: self.feed_id,
                        frame_age_ms: peak,
                        frames_lagged: count,
                    });
                }
            }
        }

        // --- View orchestration (Issue 5) ---
        let (motion_source, epoch_decision) = self.update_view(frame, &mut health_events);

        // Snapshot temporal store once for the whole frame.
        let temporal_snapshot = self.temporal.snapshot();
        let mut artifacts = PerceptionArtifacts::empty();
        let pre_batch_count = self.stages.len();
        let post_batch_count = self.post_batch_stages.len();
        let total_prov_capacity =
            pre_batch_count + post_batch_count + usize::from(self.batch.is_some());
        let mut stage_provs = Vec::with_capacity(total_prov_capacity);

        // Capture clock anchors so we can call the free function inside
        // the mutable-borrow loop over self.stages.
        let anchor = self.clock_anchor;
        let anchor_ts = self.clock_anchor_ts;

        // --- Pre-batch stages ---
        let pre_outcome = run_stage_sequence(
            &mut self.stages,
            &mut self.stage_metrics,
            0,
            self.feed_id,
            frame,
            &mut artifacts,
            &self.view_snapshot,
            &temporal_snapshot,
            &mut stage_provs,
            &mut health_events,
            anchor,
            anchor_ts,
        );

        // Early exit on pre-batch failure.
        match pre_outcome {
            StageSeqOutcome::Panic => {
                self.frames_processed += 1;
                return (None, health_events);
            }
            StageSeqOutcome::FrameDropped => {
                self.frames_processed += 1;
                return (None, health_events);
            }
            StageSeqOutcome::Ok => {}
        }

        // --- Batch point ---
        if let Some(ref batch_handle) = self.batch {
            let batch_id = batch_handle.processor_id();
            let t_batch_start = Instant::now();

            let entry = BatchEntry {
                feed_id: self.feed_id,
                frame: frame.clone(),
                view: self.view_snapshot.clone(),
                output: None,
            };

            let batch_result = batch_handle.submit_and_wait(entry, self.batch_in_flight.as_ref());
            let t_batch_end = Instant::now();
            let batch_latency = Duration::from_nanos(t_batch_start.elapsed().as_nanos() as u64);

            match batch_result {
                Ok(output) => {
                    // Flush any accumulated rejection count from a prior
                    // overload period now that submissions are succeeding
                    // again (recovery boundary).
                    if self.batch_rejection_count > 0 {
                        health_events.push(HealthEvent::BatchSubmissionRejected {
                            feed_id: self.feed_id,
                            processor_id: batch_id,
                            dropped_count: self.batch_rejection_count,
                        });
                        self.batch_rejection_count = 0;
                        self.last_batch_rejection_event = None;
                    }

                    // Flush any accumulated timeout count on recovery.
                    if self.batch_timeout_count > 0 {
                        health_events.push(HealthEvent::BatchTimeout {
                            feed_id: self.feed_id,
                            processor_id: batch_id,
                            timed_out_count: self.batch_timeout_count,
                        });
                        self.batch_timeout_count = 0;
                        self.last_batch_timeout_event = None;
                    }

                    // Flush any accumulated in-flight cap rejections on recovery.
                    if self.batch_in_flight_rejection_count > 0 {
                        health_events.push(HealthEvent::BatchInFlightExceeded {
                            feed_id: self.feed_id,
                            processor_id: batch_id,
                            rejected_count: self.batch_in_flight_rejection_count,
                        });
                        self.batch_in_flight_rejection_count = 0;
                        self.last_batch_in_flight_rejection_event = None;
                    }

                    artifacts.merge(output);
                    stage_provs.push(StageProvenance {
                        stage_id: batch_id,
                        start_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_start),
                        end_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_end),
                        latency: batch_latency,
                        result: StageResult::Ok,
                    });
                }
                Err(submit_err) => {
                    let (result, health) = match submit_err {
                        BatchSubmitError::QueueFull => {
                            // Throttle: accumulate rejections, emit at
                            // most once per second.
                            self.batch_rejection_count += 1;
                            let now = Instant::now();
                            let should_emit = self
                                .last_batch_rejection_event
                                .is_none_or(|t| now.duration_since(t) >= BATCH_REJECTION_THROTTLE);
                            let health = if should_emit {
                                let count = self.batch_rejection_count;
                                self.batch_rejection_count = 0;
                                self.last_batch_rejection_event = Some(now);
                                Some(HealthEvent::BatchSubmissionRejected {
                                    feed_id: self.feed_id,
                                    processor_id: batch_id,
                                    dropped_count: count,
                                })
                            } else {
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::ResourceExhausted),
                                health,
                            )
                        }
                        BatchSubmitError::ProcessingFailed(ref e) => {
                            // The coordinator already emitted the
                            // authoritative BatchError with the real
                            // batch_size — do NOT emit a duplicate here.
                            (categorize_stage_error(e), None)
                        }
                        BatchSubmitError::CoordinatorShutdown => {
                            // Distinguish expected feed/runtime shutdown
                            // from unexpected coordinator death.
                            let is_expected = self.feed_shutdown.load(Ordering::Relaxed);
                            let health = if is_expected {
                                // Expected lifecycle event — no health noise.
                                None
                            } else if !self.coordinator_loss_emitted {
                                // Unexpected coordinator loss — emit once.
                                self.coordinator_loss_emitted = true;
                                Some(HealthEvent::StageError {
                                    feed_id: self.feed_id,
                                    stage_id: batch_id,
                                    error: StageError::ProcessingFailed {
                                        stage_id: batch_id,
                                        detail: "batch coordinator shut down unexpectedly".into(),
                                    },
                                })
                            } else {
                                // Already emitted — suppress duplicates.
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::DependencyUnavailable),
                                health,
                            )
                        }
                        BatchSubmitError::Timeout => {
                            batch_handle.record_timeout();
                            // Throttle: accumulate timeouts, emit at
                            // most once per second.
                            self.batch_timeout_count += 1;
                            let now = Instant::now();
                            let should_emit = self
                                .last_batch_timeout_event
                                .is_none_or(|t| now.duration_since(t) >= BATCH_TIMEOUT_THROTTLE);
                            let health = if should_emit {
                                let count = self.batch_timeout_count;
                                self.batch_timeout_count = 0;
                                self.last_batch_timeout_event = Some(now);
                                Some(HealthEvent::BatchTimeout {
                                    feed_id: self.feed_id,
                                    processor_id: batch_id,
                                    timed_out_count: count,
                                })
                            } else {
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::ProcessingFailed),
                                health,
                            )
                        }
                        BatchSubmitError::InFlightCapReached => {
                            // Prior timed-out item still in coordinator.
                            // Throttle: same pattern as QueueFull.
                            self.batch_in_flight_rejection_count += 1;
                            let now = Instant::now();
                            let should_emit = self
                                .last_batch_in_flight_rejection_event
                                .is_none_or(|t| now.duration_since(t) >= BATCH_IN_FLIGHT_THROTTLE);
                            let health = if should_emit {
                                let count = self.batch_in_flight_rejection_count;
                                self.batch_in_flight_rejection_count = 0;
                                self.last_batch_in_flight_rejection_event = Some(now);
                                Some(HealthEvent::BatchInFlightExceeded {
                                    feed_id: self.feed_id,
                                    processor_id: batch_id,
                                    rejected_count: count,
                                })
                            } else {
                                None
                            };
                            (
                                StageResult::Error(StageOutcomeCategory::ResourceExhausted),
                                health,
                            )
                        }
                    };

                    if let Some(evt) = health {
                        health_events.push(evt);
                    }
                    stage_provs.push(StageProvenance {
                        stage_id: batch_id,
                        start_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_start),
                        end_ts: instant_to_ts_impl(anchor, anchor_ts, t_batch_end),
                        latency: batch_latency,
                        result,
                    });

                    // Batch failure drops the frame — skip post-batch stages.
                    self.frames_processed += 1;
                    return (None, health_events);
                }
            }
        }

        // --- Post-batch stages ---
        let post_outcome = run_stage_sequence(
            &mut self.post_batch_stages,
            &mut self.stage_metrics,
            pre_batch_count,
            self.feed_id,
            frame,
            &mut artifacts,
            &self.view_snapshot,
            &temporal_snapshot,
            &mut stage_provs,
            &mut health_events,
            anchor,
            anchor_ts,
        );

        self.frames_processed += 1;

        // If a stage panicked, return health events but no output.
        // The caller (worker) will decide whether to restart.
        match post_outcome {
            StageSeqOutcome::Panic => {
                return (None, health_events);
            }
            StageSeqOutcome::FrameDropped => {
                // Issue 9: frame was dropped due to stage error — emit no output.
                return (None, health_events);
            }
            StageSeqOutcome::Ok => {}
        }

        // --- Track ending (authoritative set semantics) ---
        //
        // End absent tracks *before* committing incoming tracks so that
        // cap space freed by ended tracks is available for admission.
        // Without this ordering, ID churn can cause spurious
        // TrackAdmissionRejected health events when the store is near
        // capacity.
        let now_ts = frame.ts();
        let current_epoch = self.view_state.epoch;

        if artifacts.tracks_authoritative {
            self.track_id_buf.clear();
            self.track_id_buf
                .extend(artifacts.tracks.iter().map(|t| t.id));
            self.ended_buf.clear();
            self.ended_buf.extend(
                self.temporal
                    .track_ids()
                    .filter(|id| !self.track_id_buf.contains(id))
                    .copied(),
            );
            for id in &self.ended_buf {
                self.temporal.end_track(id);
            }
        }

        // --- Temporal commit ---
        //
        // Tracks in the output envelope are **stage-authoritative**: they
        // reflect exactly what the perception stages produced, regardless
        // of temporal-store admission. If the store rejects a track
        // (e.g., capacity limit), a health event is emitted but the track
        // still appears in the output. Consumers who need to know which
        // tracks have temporal history should consult the store snapshot.
        let mut track_rejections = 0u32;
        let track_total = artifacts.tracks.len() as u32;
        for track in &artifacts.tracks {
            if !self.temporal.commit_track(track, now_ts, current_epoch) {
                track_rejections += 1;
            }
        }
        let admission = crate::output::AdmissionSummary {
            admitted: track_total - track_rejections,
            rejected: track_rejections,
        };
        if track_rejections > 0 {
            health_events.push(HealthEvent::TrackAdmissionRejected {
                feed_id: self.feed_id,
                rejected_count: track_rejections,
            });
        }

        // --- Retention enforcement ---
        self.temporal.enforce_retention(now_ts);

        let t_pipeline_end = Instant::now();
        let pipeline_complete_ts = self.instant_to_ts(t_pipeline_end);
        let total_latency = Duration::from_nanos(t_pipeline_start.elapsed().as_nanos() as u64);

        // --- TargetFps resolution ---
        //
        // If the frame inclusion policy is TargetFps, estimate the source
        // FPS from source-domain timestamps and resolve to a concrete
        // Sampled interval. Uses frame timestamps (not wall-clock) so
        // that CUDA/TRT startup stalls don't distort the estimate.
        if let FrameInclusion::TargetFps { .. } = self.frame_inclusion {
            self.try_resolve_target_fps(frame.ts());
        }

        // Determine frame inclusion for this output.
        let include_frame = match self.frame_inclusion {
            FrameInclusion::Always => true,
            FrameInclusion::Never => false,
            FrameInclusion::Sampled { interval } => {
                if interval == 0 {
                    false
                } else {
                    self.frame_sample_counter += 1;
                    if self.frame_sample_counter >= interval as u64 {
                        self.frame_sample_counter = 0;
                        true
                    } else {
                        false
                    }
                }
            }
            // Before resolution, use fallback_interval.
            FrameInclusion::TargetFps { fallback_interval, .. } => {
                if fallback_interval == 0 {
                    false
                } else {
                    self.frame_sample_counter += 1;
                    if self.frame_sample_counter >= fallback_interval as u64 {
                        self.frame_sample_counter = 0;
                        true
                    } else {
                        false
                    }
                }
            }
        };

        let output = OutputEnvelope {
            feed_id: self.feed_id,
            frame_seq: frame.seq(),
            ts: frame.ts(),
            wall_ts: frame.wall_ts(),
            detections: artifacts.detections,
            tracks: artifacts.tracks,
            signals: artifacts.signals,
            scene_features: artifacts.scene_features,
            view: self.view_state.clone(),
            provenance: Provenance {
                stages: stage_provs,
                view_provenance: ViewProvenance {
                    motion_source,
                    epoch_decision,
                    transition: self.view_snapshot.transition(),
                    stability_score: self.view_snapshot.stability_score(),
                    epoch: self.view_snapshot.epoch(),
                    version: self.view_snapshot.version(),
                },
                frame_receive_ts,
                pipeline_complete_ts,
                total_latency,
                frame_age,
                queue_hold_time,
                frame_included: include_frame,
            },
            metadata: artifacts.stage_artifacts,
            frame: if include_frame { Some(frame.clone()) } else { None },
            admission,
        };

        (Some(output), health_events)
    }
}

/// Outcome of [`run_stage_sequence`]: indicates whether execution
/// should continue to the next pipeline phase.
enum StageSeqOutcome {
    /// All stages ran successfully.
    Ok,
    /// A stage error caused the frame to be dropped.
    FrameDropped,
    /// A stage panicked.
    Panic,
}

/// Run a sequence of stages, collecting artifacts, provenance, and
/// health events. Shared between pre-batch and post-batch execution.
#[allow(clippy::too_many_arguments)]
fn run_stage_sequence(
    stages: &mut [Box<dyn Stage>],
    metrics: &mut [StageMetrics],
    metrics_offset: usize,
    feed_id: FeedId,
    frame: &FrameEnvelope,
    artifacts: &mut PerceptionArtifacts,
    view_snapshot: &ViewSnapshot,
    temporal_snapshot: &TemporalStoreSnapshot,
    stage_provs: &mut Vec<StageProvenance>,
    health_events: &mut Vec<HealthEvent>,
    anchor: Instant,
    anchor_ts: MonotonicTs,
) -> StageSeqOutcome {
    for (i, stage) in stages.iter_mut().enumerate() {
        let stage_id = stage.id();
        let midx = metrics_offset + i;
        let t_stage_start = Instant::now();

        let ctx = StageContext {
            feed_id,
            frame,
            artifacts,
            view: view_snapshot,
            temporal: temporal_snapshot,
            metrics: &metrics[midx],
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stage.process(&ctx)));

        let t_stage_end = Instant::now();
        let stage_latency = Duration::from_nanos(t_stage_start.elapsed().as_nanos() as u64);

        let stage_result = match result {
            Ok(Ok(output)) => {
                artifacts.merge(output);
                metrics[midx].frames_processed += 1;
                StageResult::Ok
            }
            Ok(Err(e)) => {
                metrics[midx].errors += 1;
                health_events.push(HealthEvent::StageError {
                    feed_id,
                    stage_id,
                    error: e.clone(),
                });
                stage_provs.push(StageProvenance {
                    stage_id,
                    start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
                    end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
                    latency: stage_latency,
                    result: categorize_stage_error(&e),
                });
                return StageSeqOutcome::FrameDropped;
            }
            Err(_panic) => {
                metrics[midx].errors += 1;
                health_events.push(HealthEvent::StagePanic { feed_id, stage_id });
                stage_provs.push(StageProvenance {
                    stage_id,
                    start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
                    end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
                    latency: stage_latency,
                    result: StageResult::Error(StageOutcomeCategory::Panic),
                });
                return StageSeqOutcome::Panic;
            }
        };

        stage_provs.push(StageProvenance {
            stage_id,
            start_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_start),
            end_ts: instant_to_ts_impl(anchor, anchor_ts, t_stage_end),
            latency: stage_latency,
            result: stage_result,
        });
    }

    StageSeqOutcome::Ok
}

/// Map a [`StageError`] variant to the provenance category.
fn categorize_stage_error(e: &StageError) -> StageResult {
    match e {
        StageError::ProcessingFailed { .. } => {
            StageResult::Error(StageOutcomeCategory::ProcessingFailed)
        }
        StageError::ResourceExhausted { .. } => {
            StageResult::Error(StageOutcomeCategory::ResourceExhausted)
        }
        StageError::ModelLoadFailed { .. } => {
            StageResult::Error(StageOutcomeCategory::DependencyUnavailable)
        }
    }
}
