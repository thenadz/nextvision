use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use minifb::{Window, WindowOptions};
use nv_runtime::{OutputEnvelope, OutputSink};
use tracing::{info, warn};

use crate::overlay;

/// A frame ready for display on the UI thread.
struct UiFrame {
    /// Packed ARGB pixels for minifb (`0x00RRGGBB`).
    buf: Vec<u32>,
    width: usize,
    height: usize,
}

/// Output sink that renders detection/track overlays on each frame and
/// forwards the result to a dedicated display thread.
///
/// The UI thread owns the `minifb` window and runs independently so that
/// rendering never stalls the perception pipeline. If the UI cannot keep
/// up, excess frames are silently dropped (bounded channel, try-send).
pub struct OverlaySink {
    tx: SyncSender<UiFrame>,
    frame_count: AtomicU64,
    start: Instant,
}

impl OverlaySink {
    /// Spawn the UI thread and return a sink that feeds it.
    ///
    /// `init_width` / `init_height` set the initial window dimensions.
    pub fn spawn(init_width: usize, init_height: usize) -> Self {
        // Bounded to 2 — if the UI falls behind we drop frames.
        let (tx, rx) = mpsc::sync_channel::<UiFrame>(2);

        thread::Builder::new()
            .name("overlay-ui".into())
            .spawn(move || ui_thread(rx, init_width, init_height))
            .expect("failed to spawn overlay UI thread");

        Self {
            tx,
            frame_count: AtomicU64::new(0),
            start: Instant::now(),
        }
    }
}

impl OutputSink for OverlaySink {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let count = self.frame_count.fetch_add(1, Ordering::Relaxed) + 1;
        let elapsed = self.start.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            count as f64 / elapsed
        } else {
            0.0
        };

        // Extract the frame from FrameInclusion::Always output.
        let Some(frame) = output.frame.as_ref() else {
            warn!("no frame in output — is frame_inclusion set to Always?");
            return;
        };

        let w = frame.width();
        let h = frame.height();
        let stride = frame.stride();

        // Work on a mutable copy so we can draw on it.
        let mut rgb = frame.data().to_vec();

        overlay::draw_tracks(&mut rgb, w, h, stride, &output.tracks);
        overlay::draw_fps(&mut rgb, w, h, stride, fps);

        let buf = rgb_to_packed(&rgb, w, h, stride);

        // Non-blocking: drop the frame if the UI thread is behind.
        let _ = self.tx.try_send(UiFrame {
            buf,
            width: w as usize,
            height: h as usize,
        });

        // Log a summary periodically.
        if count % 60 == 0 {
            info!(
                seq = output.frame_seq,
                detections = output.detections.len(),
                tracks = output.tracks.len(),
                fps = format!("{fps:.1}"),
                "pipeline running"
            );
        }
    }
}

/// Convert strided RGB8 pixels to packed `0x00RRGGBB` u32 for minifb.
fn rgb_to_packed(rgb: &[u8], w: u32, h: u32, stride: u32) -> Vec<u32> {
    let mut buf = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let row = (y * stride) as usize;
        for x in 0..w {
            let i = row + (x as usize) * 3;
            let r = rgb[i] as u32;
            let g = rgb[i + 1] as u32;
            let b = rgb[i + 2] as u32;
            buf.push((r << 16) | (g << 8) | b);
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// UI thread
// ---------------------------------------------------------------------------

fn ui_thread(rx: Receiver<UiFrame>, init_w: usize, init_h: usize) {
    let mut window = match Window::new(
        "YOLOX + OC-SORT — NextVision",
        init_w,
        init_h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    ) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!("failed to create window: {e}");
            return;
        }
    };

    window.set_target_fps(60);

    let mut last_w = init_w;
    let mut last_h = init_h;

    while window.is_open() {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => {
                // Resize the window if the frame dimensions changed.
                if frame.width != last_w || frame.height != last_h {
                    last_w = frame.width;
                    last_h = frame.height;
                }
                if let Err(e) = window.update_with_buffer(&frame.buf, frame.width, frame.height) {
                    tracing::warn!("update_with_buffer failed: {e}");
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Keep the window responsive even when no frames arrive.
                window.update();
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}
