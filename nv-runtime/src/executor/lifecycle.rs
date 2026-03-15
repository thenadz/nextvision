use nv_core::config::CameraMode;
use nv_core::error::StageError;
use nv_core::id::FeedId;
use nv_perception::Stage;
use nv_view::{TransitionPhase, ViewEpoch, ViewSnapshot};

use super::PipelineExecutor;

impl PipelineExecutor {
    /// Call `on_start()` on each stage in order (pre-batch then post-batch).
    ///
    /// If any stage fails or panics, previously-started stages are stopped
    /// (best-effort) and the error is returned.
    pub fn start_stages(&mut self) -> Result<(), StageError> {
        // Start pre-batch stages.
        if let Err((started, e)) = start_stage_slice(&self.feed_id, &mut self.stages) {
            self.rollback_started_stages(started, 0);
            return Err(e);
        }

        // Start post-batch stages.
        if let Err((started, e)) = start_stage_slice(&self.feed_id, &mut self.post_batch_stages) {
            self.rollback_started_stages(self.stages.len(), started);
            return Err(e);
        }

        Ok(())
    }

    /// Best-effort stop of pre-batch stages `[0..pre_count)` and
    /// post-batch stages `[0..post_count)`. Used on startup failure.
    fn rollback_started_stages(&mut self, pre_count: usize, post_count: usize) {
        for s in &mut self.stages[..pre_count] {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = s.on_stop();
            }));
        }
        for s in &mut self.post_batch_stages[..post_count] {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = s.on_stop();
            }));
        }
    }

    /// Call `on_stop()` on each stage (pre-batch then post-batch) — best-effort,
    /// errors and panics are logged.
    pub fn stop_stages(&mut self) {
        for stage in self.stages.iter_mut().chain(self.post_batch_stages.iter_mut()) {
            let stage_id = stage.id();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stage.on_stop()
            }));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!(
                        feed_id = %self.feed_id,
                        stage_id = %stage_id,
                        error = %e,
                        "stage on_stop error (ignored)"
                    );
                }
                Err(_) => {
                    tracing::error!(
                        feed_id = %self.feed_id,
                        stage_id = %stage_id,
                        "stage on_stop() panicked (ignored)"
                    );
                }
            }
        }
    }

    /// Clear temporal state and increment epoch (called on feed restart).
    ///
    /// All active trajectory segments are closed with
    /// [`SegmentBoundary::FeedRestart`](nv_temporal::SegmentBoundary::FeedRestart)
    /// before clearing, ensuring proper segment boundaries in the data
    /// model even though the data itself is discarded.
    ///
    /// Incrementing the epoch ensures that tracks carried over from a
    /// prior session are segmented, and stages that cache view-dependent
    /// state get an `on_view_epoch_change` notification on the first
    /// frame of the new session.
    pub fn clear_temporal(&mut self) {
        self.temporal
            .close_all_segments(nv_temporal::SegmentBoundary::FeedRestart);
        self.temporal.clear();
        let new_epoch = self.view_state.epoch.next();
        self.view_state.epoch = new_epoch;
        self.view_state.version = self.view_state.version.next();
        self.view_state.transition = TransitionPhase::Settled;
        self.view_state.stability_score = match self.camera_mode {
            CameraMode::Fixed => 1.0,
            CameraMode::Observed => 0.0,
        };
        self.temporal.set_view_epoch(new_epoch);
        self.view_snapshot = ViewSnapshot::new(self.view_state.clone());
    }
}

/// Start each stage in `stages` in order, with panic safety.
///
/// Returns `Ok(())` if all stages started. On error, returns
/// `(started_count, error)` — the caller rolls back `[0..started_count)`.
fn start_stage_slice(
    feed_id: &FeedId,
    stages: &mut [Box<dyn Stage>],
) -> Result<(), (usize, StageError)> {
    for (i, stage) in stages.iter_mut().enumerate() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.on_start()
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err((i, e)),
            Err(_) => {
                let stage_id = stage.id();
                tracing::error!(
                    feed_id = %feed_id,
                    stage_id = %stage_id,
                    "stage on_start() panicked"
                );
                return Err((i, StageError::ProcessingFailed {
                    stage_id,
                    detail: "on_start() panicked".into(),
                }));
            }
        }
    }
    Ok(())
}

/// Notify all stages (pre-batch + post-batch) of a view epoch change,
/// with panic safety. Errors and panics are logged but do not propagate.
pub(super) fn notify_epoch_change(
    feed_id: FeedId,
    pre: &mut [Box<dyn Stage>],
    post: &mut [Box<dyn Stage>],
    epoch: ViewEpoch,
) {
    for stage in pre.iter_mut().chain(post.iter_mut()) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stage.on_view_epoch_change(epoch)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!(
                    feed_id = %feed_id,
                    stage_id = %stage.id(),
                    error = %e,
                    "stage on_view_epoch_change error (ignored)"
                );
            }
            Err(_) => {
                tracing::error!(
                    feed_id = %feed_id,
                    stage_id = %stage.id(),
                    "stage on_view_epoch_change panicked (ignored)"
                );
            }
        }
    }
}
