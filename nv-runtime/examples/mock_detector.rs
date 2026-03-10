//! Mock detector example: a stage that produces synthetic detections.
//!
//! This example demonstrates:
//! - Implementing the `Stage` trait with detection output.
//! - Using `StageContext` to access frame metadata.
//! - Composing multiple stages in a pipeline.
//! - Receiving structured `OutputEnvelope` with detections and provenance.

use nv_core::config::{CameraMode, SourceSpec};
use nv_core::error::StageError;
use nv_core::geom::BBox;
use nv_core::id::StageId;
use nv_perception::detection::{Detection, DetectionSet};
use nv_perception::{Stage, StageContext, StageOutput};
use nv_runtime::{FeedConfig, OutputEnvelope, OutputSink, Runtime};

use std::sync::Arc;

/// A mock object detector that produces one synthetic detection per frame.
///
/// In a real implementation, this would call an inference engine
/// (e.g., TensorRT, ONNX Runtime) on the frame pixels.
struct MockDetectorStage {
    next_det_id: u64,
}

impl MockDetectorStage {
    fn new() -> Self {
        Self { next_det_id: 0 }
    }
}

impl Stage for MockDetectorStage {
    fn id(&self) -> StageId {
        StageId("mock_detector")
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        // Simulate detecting a person in the center of the frame.
        let det = Detection {
            id: nv_core::DetectionId::new(self.next_det_id),
            class_id: 0, // "person"
            confidence: 0.87,
            bbox: BBox::new(0.3, 0.2, 0.4, 0.6),
            embedding: None,
            metadata: nv_core::TypedMetadata::new(),
        };
        self.next_det_id += 1;

        let _ = ctx.frame.seq(); // Access frame metadata.
        Ok(StageOutput::with_detections(DetectionSet {
            detections: vec![det],
        }))
    }
}

/// A simple classifier stage that reads detections from prior stages.
struct MockClassifierStage;

impl Stage for MockClassifierStage {
    fn id(&self) -> StageId {
        StageId("mock_classifier")
    }

    fn process(&mut self, ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
        // Access detections produced by the upstream detector.
        let det_count = ctx.artifacts.detections.len();
        if det_count > 0 {
            // In a real stage, you would refine class labels or confidence scores.
        }
        Ok(StageOutput::empty())
    }
}

/// Output sink that logs detection summaries.
struct DetectionLogger;

impl OutputSink for DetectionLogger {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let det_count = output.detections.len();
        let stage_names: Vec<_> = output
            .provenance
            .stages
            .iter()
            .map(|s| s.stage_id.0)
            .collect();

        println!(
            "seq={} detections={} stages={:?} total_latency={:?}",
            output.frame_seq,
            det_count,
            stage_names,
            output.provenance.total_latency,
        );
    }
}

fn main() -> Result<(), nv_core::error::NvError> {
    let runtime = Runtime::builder().build()?;

    // Build a two-stage pipeline: detector → classifier.
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://192.168.1.10/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![
            Box::new(MockDetectorStage::new()),
            Box::new(MockClassifierStage),
        ])
        .output_sink(Box::new(DetectionLogger))
        .build()?;

    let handle = runtime.add_feed(config)?;
    println!("Feed {:?} running with mock detector + classifier", handle.id());

    // Let the feed run for a while.
    std::thread::sleep(std::time::Duration::from_secs(5));

    let m = handle.metrics();
    println!(
        "Processed {} frames, {} dropped",
        m.frames_processed, m.frames_dropped,
    );

    runtime.shutdown()?;
    Ok(())
}
