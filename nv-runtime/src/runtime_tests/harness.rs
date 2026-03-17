//! Shared mock infrastructure for runtime integration tests.

use super::super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nv_core::config::{CameraMode, SourceSpec};
use nv_frame::PixelFormat;
use nv_media::ingress::{FrameSink, IngressOptions, MediaIngress, MediaIngressFactory};
use nv_perception::Stage;

use crate::output::{OutputSink, SharedOutput};
use crate::shutdown::RestartPolicy;

// ---------------------------------------------------------------------------
// Mock infrastructure
// ---------------------------------------------------------------------------

/// Mock source: sends `frame_count` frames then EOS.
pub(super) struct MockIngress {
    feed_id: FeedId,
    spec: SourceSpec,
    frame_count: u64,
    fail_on_start: bool,
    frame_delay: std::time::Duration,
    eos_signaled: Arc<std::sync::atomic::AtomicBool>,
}

impl MediaIngress for MockIngress {
    fn start(&mut self, sink: Box<dyn FrameSink>) -> Result<(), nv_core::error::MediaError> {
        if self.fail_on_start {
            return Err(nv_core::error::MediaError::ConnectionFailed {
                url: "mock://fail".into(),
                detail: "mock start failure".into(),
            });
        }
        let count = self.frame_count;
        let feed_id = self.feed_id;
        let delay = self.frame_delay;
        let eos_flag = Arc::clone(&self.eos_signaled);
        std::thread::spawn(move || {
            for i in 0..count {
                let frame = make_test_frame(feed_id, i);
                sink.on_frame(frame);
                if delay > std::time::Duration::ZERO {
                    std::thread::sleep(delay);
                }
            }
            eos_flag.store(true, Ordering::Release);
            sink.on_eos();
        });
        Ok(())
    }

    fn stop(&mut self) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }
    fn pause(&mut self) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }
    fn resume(&mut self) -> Result<(), nv_core::error::MediaError> {
        Ok(())
    }

    fn tick(&mut self) -> nv_media::ingress::TickOutcome {
        if self.eos_signaled.load(Ordering::Acquire) {
            nv_media::ingress::TickOutcome::stopped()
        } else {
            nv_media::ingress::TickOutcome::running()
        }
    }

    fn source_spec(&self) -> &SourceSpec {
        &self.spec
    }
    fn feed_id(&self) -> FeedId {
        self.feed_id
    }
}

/// Mock factory: creates `MockIngress` sources.
pub(super) struct MockFactory {
    pub frame_count: u64,
    pub fail_on_start: bool,
    pub frame_delay: std::time::Duration,
}

impl MockFactory {
    pub fn new(frame_count: u64) -> Self {
        Self {
            frame_count,
            fail_on_start: false,
            frame_delay: std::time::Duration::from_millis(1),
        }
    }

    pub fn failing() -> Self {
        Self {
            frame_count: 0,
            fail_on_start: true,
            frame_delay: std::time::Duration::ZERO,
        }
    }
}

impl MediaIngressFactory for MockFactory {
    fn create(
        &self,
        options: IngressOptions,
    ) -> Result<Box<dyn MediaIngress>, nv_core::error::MediaError> {
        Ok(Box::new(MockIngress {
            feed_id: options.feed_id,
            spec: options.spec,
            frame_count: self.frame_count,
            fail_on_start: self.fail_on_start,
            frame_delay: self.frame_delay,
            eos_signaled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }))
    }
}

/// Output sink that counts received outputs.
pub(super) struct CountingSink {
    count: Arc<AtomicU64>,
}

impl CountingSink {
    pub fn new() -> (Self, Arc<AtomicU64>) {
        let count = Arc::new(AtomicU64::new(0));
        (
            Self {
                count: Arc::clone(&count),
            },
            count,
        )
    }
}

impl OutputSink for CountingSink {
    fn emit(&self, _output: SharedOutput) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

pub(super) fn make_test_frame(feed_id: FeedId, seq: u64) -> nv_frame::FrameEnvelope {
    nv_frame::FrameEnvelope::new_owned(
        feed_id,
        seq,
        nv_core::MonotonicTs::from_nanos(seq * 33_333_333),
        nv_core::WallTs::from_micros(0),
        2,
        2,
        PixelFormat::Rgb8,
        6,
        vec![0u8; 12],
        nv_core::TypedMetadata::new(),
    )
}

pub(super) fn build_config(stages: Vec<Box<dyn Stage>>, sink: Box<dyn OutputSink>) -> FeedConfig {
    FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(stages)
        .output_sink(sink)
        .build()
        .expect("valid config")
}

pub(super) fn build_config_with_restart(
    stages: Vec<Box<dyn Stage>>,
    sink: Box<dyn OutputSink>,
    restart: RestartPolicy,
) -> FeedConfig {
    FeedConfig::builder()
        .source(SourceSpec::rtsp("rtsp://mock/stream"))
        .camera_mode(CameraMode::Fixed)
        .stages(stages)
        .output_sink(sink)
        .restart(restart)
        .build()
        .expect("valid config")
}

/// Wait for a feed to stop (with timeout).
pub(super) fn wait_for_stop(handle: &FeedHandle, timeout: std::time::Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while handle.is_alive() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Build a runtime with `MockFactory::new(frame_count)` and defaults.
pub(super) fn build_runtime(frame_count: u64) -> Runtime {
    Runtime::builder()
        .ingress_factory(Box::new(MockFactory::new(frame_count)))
        .build()
        .unwrap()
}
