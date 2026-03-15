//! Configuration validation tests: zero capacity/depth rejection,
//! FeedConfigBuilder validation modes, sink_queue_capacity, and
//! DecodePreference propagation.

use super::super::*;

use nv_core::config::{CameraMode, SourceSpec};
use nv_core::id::StageId;
use nv_perception::{Stage, StageContext, StageOutput};
use nv_test_util::mock_stage::NoOpStage;

use super::harness::*;

// ---------------------------------------------------------------------------
// Config validation — zero capacity / depth rejection
// ---------------------------------------------------------------------------

#[test]
fn zero_health_capacity_rejected() {
    let result = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(1)))
        .health_capacity(0)
        .build();
    assert!(result.is_err(), "health_capacity=0 must be rejected");
}

#[test]
fn zero_output_capacity_rejected() {
    let result = Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(1)))
        .output_capacity(0)
        .build();
    assert!(result.is_err(), "output_capacity=0 must be rejected");
}

#[test]
fn zero_queue_depth_rejected() {
    use crate::backpressure::BackpressurePolicy;
    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(NoOpStage::new("noop"))])
        .output_sink(Box::new(CountingSink::new().0))
        .backpressure(BackpressurePolicy::DropOldest { queue_depth: 0 })
        .build();
    assert!(result.is_err(), "queue_depth=0 must be rejected");
}

// ---------------------------------------------------------------------------
// FeedConfigBuilder — validation_mode, add_stage helpers
// ---------------------------------------------------------------------------

/// `ValidationMode::Error` rejects a misordered pipeline.
#[test]
fn validation_mode_error_rejects_bad_ordering() {
    use nv_perception::{StageCapabilities, ValidationMode};

    struct DetStage;
    impl Stage for DetStage {
        fn id(&self) -> StageId { StageId("det") }
        fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, nv_core::error::StageError> {
            Ok(StageOutput::empty())
        }
        fn capabilities(&self) -> Option<StageCapabilities> {
            Some(StageCapabilities::new().produces_detections())
        }
    }

    struct TrkStage;
    impl Stage for TrkStage {
        fn id(&self) -> StageId { StageId("trk") }
        fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, nv_core::error::StageError> {
            Ok(StageOutput::empty())
        }
        fn capabilities(&self) -> Option<StageCapabilities> {
            Some(StageCapabilities::new().consumes_detections().produces_tracks())
        }
    }

    // Wrong order: tracker before detector.
    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(TrkStage), Box::new(DetStage)])
        .output_sink(Box::new(CountingSink::new().0))
        .validation_mode(ValidationMode::Error)
        .build();
    assert!(result.is_err(), "misordered pipeline must be rejected in Error mode");

    // Correct order: detector then tracker.
    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(DetStage), Box::new(TrkStage)])
        .output_sink(Box::new(CountingSink::new().0))
        .validation_mode(ValidationMode::Error)
        .build();
    assert!(result.is_ok(), "correctly ordered pipeline must pass in Error mode");
}

/// `ValidationMode::Off` (default) does not reject a misordered pipeline.
#[test]
fn validation_mode_off_allows_bad_ordering() {
    use nv_perception::{StageCapabilities, ValidationMode};

    struct BadStage;
    impl Stage for BadStage {
        fn id(&self) -> StageId { StageId("bad") }
        fn process(&mut self, _: &StageContext<'_>) -> Result<StageOutput, nv_core::error::StageError> {
            Ok(StageOutput::empty())
        }
        fn capabilities(&self) -> Option<StageCapabilities> {
            Some(StageCapabilities::new().consumes_detections())
        }
    }

    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(BadStage)])
        .output_sink(Box::new(CountingSink::new().0))
        .validation_mode(ValidationMode::Off)
        .build();
    assert!(result.is_ok(), "Off mode must not reject");
}

/// `add_stage` / `add_boxed_stage` build a valid feed config.
#[test]
fn add_stage_helpers_build_valid_config() {
    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .add_stage(NoOpStage::new("a"))
        .add_boxed_stage(Box::new(NoOpStage::new("b")))
        .output_sink(Box::new(CountingSink::new().0))
        .build();
    assert!(result.is_ok(), "add_stage helpers must produce valid config");
}

/// `add_stage` without any stage call fails (empty stages).
#[test]
fn add_stage_empty_is_rejected() {
    let result = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/test"))
        .camera_mode(CameraMode::Fixed)
        .output_sink(Box::new(CountingSink::new().0))
        .build();
    assert!(result.is_err(), "missing stages must be rejected");
}

// ===========================================================================
// sink_queue_capacity
// ===========================================================================

#[test]
fn sink_queue_capacity_configurable() {
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(nv_test_util::mock_stage::NoOpStage::new("noop")) as Box<dyn Stage>])
        .output_sink(Box::new(sink))
        .sink_queue_capacity(32)
        .build()
        .expect("valid config");
    assert_eq!(config.sink_queue_capacity, 32);
}

#[test]
fn sink_queue_capacity_defaults_to_16() {
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(nv_test_util::mock_stage::NoOpStage::new("noop")) as Box<dyn Stage>])
        .output_sink(Box::new(sink))
        .build()
        .expect("valid config");
    assert_eq!(config.sink_queue_capacity, 16);
}

#[test]
fn sink_queue_capacity_clamped_to_min_1() {
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(nv_test_util::mock_stage::NoOpStage::new("noop")) as Box<dyn Stage>])
        .output_sink(Box::new(sink))
        .sink_queue_capacity(0)
        .build()
        .expect("valid config");
    assert_eq!(config.sink_queue_capacity, 1);
}

// ===========================================================================
// DecodePreference builder propagation
// ===========================================================================

#[test]
fn feed_config_builder_default_decode_preference_is_auto() {
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(NoOpStage::new("noop"))])
        .output_sink(Box::new(sink))
        .build()
        .expect("valid config");
    assert_eq!(config.decode_preference, nv_media::DecodePreference::Auto);
}

#[test]
fn feed_config_builder_sets_cpu_only() {
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(NoOpStage::new("noop"))])
        .output_sink(Box::new(sink))
        .decode_preference(nv_media::DecodePreference::CpuOnly)
        .build()
        .expect("valid config");
    assert_eq!(config.decode_preference, nv_media::DecodePreference::CpuOnly);
}

#[test]
fn feed_config_builder_sets_require_hardware() {
    let (sink, _) = CountingSink::new();
    let config = FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(vec![Box::new(NoOpStage::new("noop"))])
        .output_sink(Box::new(sink))
        .decode_preference(nv_media::DecodePreference::RequireHardware)
        .build()
        .expect("valid config");
    assert_eq!(config.decode_preference, nv_media::DecodePreference::RequireHardware);
}
