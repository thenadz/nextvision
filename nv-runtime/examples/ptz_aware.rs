//! PTZ-aware feed example: using `ViewStateProvider` for moving cameras.
//!
//! This example demonstrates:
//! - Configuring `CameraMode::Observed` for a PTZ camera.
//! - Implementing `ViewStateProvider` with mock telemetry.
//! - Handling view epoch changes in a stage.
//! - Inspecting view provenance in the output.

use nv_core::config::{CameraMode, SourceSpec};
use nv_core::error::StageError;
use nv_core::id::StageId;
use nv_perception::{Stage, StageContext, StageOutput};
use nv_runtime::{FeedConfig, OutputEnvelope, OutputSink, Runtime};
use nv_view::provider::{MotionPollContext, MotionReport, ViewStateProvider};
use nv_view::{CameraMotionState, PtzTelemetry, ViewEpoch};

use std::sync::Arc;

/// A mock PTZ telemetry provider.
///
/// In production, this would poll an ONVIF endpoint or serial port
/// and return actual pan/tilt/zoom readings.
struct MockPtzProvider;

impl ViewStateProvider for MockPtzProvider {
    fn poll(&self, ctx: &MotionPollContext<'_>) -> MotionReport {
        // Simulate a slow pan: motion hint alternates based on frame index.
        let is_moving = ctx.frame.seq() % 20 < 5;
        MotionReport {
            ptz: Some(PtzTelemetry {
                pan: 0.5 + (ctx.frame.seq() as f32) * 0.01,
                tilt: 0.0,
                zoom: 1.0,
                ts: ctx.ts,
            }),
            frame_transform: None,
            motion_hint: Some(if is_moving {
                CameraMotionState::Moving {
                    angular_velocity: Some(5.0),
                    displacement: Some(0.02),
                }
            } else {
                CameraMotionState::Stable
            }),
            ..Default::default()
        }
    }
}

/// A stage that resets internal state on view epoch changes.
struct EpochAwareStage {
    current_epoch: ViewEpoch,
    frames_in_epoch: u64,
}

impl EpochAwareStage {
    fn new() -> Self {
        Self {
            current_epoch: ViewEpoch::INITIAL,
            frames_in_epoch: 0,
        }
    }
}

impl Stage for EpochAwareStage {
    fn id(&self) -> StageId {
        StageId("epoch_aware")
    }

    fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        self.frames_in_epoch += 1;
        Ok(StageOutput::empty())
    }

    fn on_view_epoch_change(&mut self, new_epoch: ViewEpoch) -> Result<(), StageError> {
        println!(
            "  Epoch change: {:?} → {:?} (processed {} frames in old epoch)",
            self.current_epoch, new_epoch, self.frames_in_epoch,
        );
        self.current_epoch = new_epoch;
        self.frames_in_epoch = 0;
        Ok(())
    }
}

/// Output sink that logs view provenance.
struct ViewProvenanceLogger;

impl OutputSink for ViewProvenanceLogger {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let vp = &output.provenance.view_provenance;
        println!(
            "seq={} epoch={:?} motion={:?} transition={:?} stability={:.2}",
            output.frame_seq, vp.epoch, vp.motion_source, vp.transition, vp.stability_score,
        );
    }
}

fn main() -> Result<(), nv_core::error::NvError> {
    let runtime = Runtime::builder().build()?;

    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://192.168.1.20/ptz_stream"))
        .camera_mode(CameraMode::Observed)
        .view_state_provider(Box::new(MockPtzProvider))
        .stages(vec![Box::new(EpochAwareStage::new())])
        .output_sink(Box::new(ViewProvenanceLogger))
        .build()?;

    let handle = runtime.add_feed(config)?;
    println!("PTZ feed {:?} running...", handle.id());

    std::thread::sleep(std::time::Duration::from_secs(5));

    let m = handle.metrics();
    println!(
        "Feed {:?}: processed={} restarts={}",
        handle.id(),
        m.frames_processed,
        m.restarts,
    );

    runtime.shutdown()?;
    println!("Done.");
    Ok(())
}
