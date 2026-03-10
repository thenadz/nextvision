//! Example: multi-feed pipeline using the stage system.
//!
//! This example demonstrates how to compose a multi-stage perception
//! pipeline using `StagePipeline` and run it through the `PipelineExecutor`.
//!
//! Since this example does not require a live video source or GStreamer,
//! it drives the executor directly with synthetic frames — the same pattern
//! used by the runtime's feed worker threads.
//!
//! ## Pipeline shape
//!
//! ```text
//! Feed A (2 detections/frame):
//!   [MockDetector] → [MockTracker] → [TemporalAnalysis] → [LoggingSink]
//!
//! Feed B (5 detections/frame):
//!   [MockDetector] → [MockTracker] → [TemporalAnalysis] → [LoggingSink]
//! ```
//!
//! Each feed runs its own isolated executor with independent stage instances
//! and temporal stores — exactly as the runtime would do on separate OS threads.
//!
//! Run with: `cargo run --example multi_feed_pipeline`

use nv_core::error::StageError;
use nv_core::id::{FeedId, StageId};
use nv_perception::{
    DerivedSignal, SignalValue, Stage, StageCategory, StageContext, StageOutput, StagePipeline,
};

// Re-use shared mock stages from nv-test-util.
use nv_test_util::mock_stage::{MockDetectorStage, MockTemporalStage, MockTrackerStage};
use nv_test_util::synthetic;

// PipelineExecutor is pub(crate) in nv-runtime, so this example
// demonstrates the same execution model using the public types.
// In production, feeds are driven by Runtime::add_feed().

// ---------------------------------------------------------------------------
// Example: LoggingSink stage
// ---------------------------------------------------------------------------

/// A sink stage that prints a summary of each frame's perception results.
///
/// Demonstrates the `Sink` stage category: it reads accumulated artifacts
/// from prior stages and performs side-effect output (printing), without
/// modifying the artifact accumulator.
struct LoggingSink {
    feed_label: &'static str,
}

impl Stage for LoggingSink {
    fn id(&self) -> StageId {
        StageId("logging_sink")
    }

    fn category(&self) -> StageCategory {
        StageCategory::Sink
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let det_count = ctx.artifacts.detections.len();
        let trk_count = ctx.artifacts.tracks.len();
        let sig_count = ctx.artifacts.signals.len();

        println!(
            "[{}] seq={} ts={:?} | {} dets, {} tracks, {} signals",
            self.feed_label,
            ctx.frame.seq(),
            ctx.frame.ts(),
            det_count,
            trk_count,
            sig_count,
        );

        // Sinks return empty output.
        Ok(StageOutput::empty())
    }
}

// ---------------------------------------------------------------------------
// Example: Dwell time estimator (temporal analysis)
// ---------------------------------------------------------------------------

/// A simple temporal analysis stage that reports how many tracks have been
/// observed for more than N frames.
///
/// Demonstrates reading temporal state (`ctx.temporal`) for higher-level
/// analysis without depending on any domain-specific concepts.
struct DwellTimeEstimator {
    min_observations: usize,
}

impl Stage for DwellTimeEstimator {
    fn id(&self) -> StageId {
        StageId("dwell_estimator")
    }

    fn category(&self) -> StageCategory {
        StageCategory::TemporalAnalysis
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        let long_dwell_count = ctx
            .temporal
            .track_ids()
            .iter()
            .filter(|id| ctx.temporal.trajectory_point_count(id) >= self.min_observations)
            .count();

        Ok(StageOutput::with_signal(DerivedSignal {
            name: "long_dwell_tracks",
            value: SignalValue::Scalar(long_dwell_count as f64),
            ts: ctx.frame.ts(),
        }))
    }
}

fn main() {
    println!("=== NextVision Multi-Feed Pipeline Example ===\n");

    // ---------------------------------------------------------------
    // Build pipelines using StagePipeline
    // ---------------------------------------------------------------

    let pipeline_a = StagePipeline::builder()
        .add(MockDetectorStage::new("detector", 2))
        .add(MockTrackerStage::new("tracker"))
        .add(MockTemporalStage::new("temporal_analysis"))
        .add(DwellTimeEstimator {
            min_observations: 3,
        })
        .add(LoggingSink {
            feed_label: "Feed-A",
        })
        .build();

    let pipeline_b = StagePipeline::builder()
        .add(MockDetectorStage::new("detector", 5))
        .add(MockTrackerStage::new("tracker"))
        .add(MockTemporalStage::new("temporal_analysis"))
        .add(LoggingSink {
            feed_label: "Feed-B",
        })
        .build();

    // Describe the pipelines.
    println!("Feed A pipeline ({} stages):", pipeline_a.len());
    for (id, cat) in pipeline_a.categories() {
        println!("  {}: {:?}", id, cat);
    }

    println!("\nFeed B pipeline ({} stages):", pipeline_b.len());
    for (id, cat) in pipeline_b.categories() {
        println!("  {}: {:?}", id, cat);
    }
    println!();

    // ---------------------------------------------------------------
    // Simulate feed execution with synthetic frames
    // ---------------------------------------------------------------
    // In production, the Runtime spawns a dedicated OS thread per feed
    // and drives the executor automatically. Here we drive two executors
    // sequentially to demonstrate the execution model.

    let feed_a = FeedId::new(1);
    let feed_b = FeedId::new(2);

    let frames_a = synthetic::frame_sequence(feed_a, 10, 64, 48, 33_333_333);
    let frames_b = synthetic::frame_sequence(feed_b, 10, 64, 48, 33_333_333);

    // PipelineExecutor is pub(crate) in nv-runtime, so external code
    // uses Runtime::add_feed() instead. This example drives stages
    // directly to show the execution flow.
    let stages_a = pipeline_a.into_stages();
    let stages_b = pipeline_b.into_stages();

    // Since PipelineExecutor is not public, demonstrate the flow
    // conceptually using the StageHarness for each stage:
    use nv_perception::PerceptionArtifacts;
    use nv_test_util::mock_stage::NullTemporalAccess;
    use nv_view::{ViewSnapshot, ViewState};

    fn run_pipeline(
        feed_label: &str,
        stages: &mut [Box<dyn Stage>],
        frames: &[nv_frame::FrameEnvelope],
    ) {
        let feed_id = frames[0].feed_id();
        let view = ViewSnapshot::new(ViewState::fixed_initial());
        let temporal = NullTemporalAccess;
        let metrics = nv_core::metrics::StageMetrics {
            frames_processed: 0,
            errors: 0,
        };

        for stage in stages.iter_mut() {
            stage.on_start().expect("stage start failed");
        }

        for frame in frames {
            let mut artifacts = PerceptionArtifacts::empty();

            for stage in stages.iter_mut() {
                let ctx = StageContext {
                    feed_id,
                    frame,
                    artifacts: &artifacts,
                    view: &view,
                    temporal: &temporal,
                    metrics: &metrics,
                };
                match stage.process(&ctx) {
                    Ok(output) => artifacts.merge(output),
                    Err(e) => {
                        eprintln!("[{feed_label}] stage {} failed: {e}", stage.id());
                        break;
                    }
                }
            }
        }

        for stage in stages.iter_mut() {
            let _ = stage.on_stop();
        }
    }

    println!("--- Processing Feed A (2 detections/frame) ---");
    run_pipeline("Feed-A", &mut stages_a.into_iter().collect::<Vec<_>>(), &frames_a);

    println!("\n--- Processing Feed B (5 detections/frame) ---");
    run_pipeline("Feed-B", &mut stages_b.into_iter().collect::<Vec<_>>(), &frames_b);

    println!("\n=== Example complete ===");
}
