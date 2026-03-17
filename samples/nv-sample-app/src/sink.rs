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
    /// EMA of FPS, stored as `f64` bits via `to_bits()`/`from_bits()`.
    fps_ema_bits: AtomicU64,
    /// Last emit timestamp in nanoseconds from `start`.
    last_ns: AtomicU64,
    start: Instant,
}

/// EMA smoothing factor (α). α = 0.05 gives a ~20-frame half-life.
const FPS_ALPHA: f64 = 0.05;

impl OverlaySink {
    /// Spawn the UI thread and return a sink that feeds it.
    ///
    /// `init_width` / `init_height` set the initial window dimensions.
    ///
    /// Returns an error if the OS thread cannot be spawned.
    pub fn spawn(init_width: usize, init_height: usize) -> Result<Self, std::io::Error> {
        // Bounded to 2 — if the UI falls behind we drop frames.
        let (tx, rx) = mpsc::sync_channel::<UiFrame>(2);

        thread::Builder::new()
            .name("overlay-ui".into())
            .spawn(move || ui_thread(rx, init_width, init_height))?;

        Ok(Self {
            tx,
            frame_count: AtomicU64::new(0),
            fps_ema_bits: AtomicU64::new(0_f64.to_bits()),
            last_ns: AtomicU64::new(0),
            start: Instant::now(),
        })
    }

    /// Compute a smoothed FPS using exponential moving average.
    fn update_fps(&self) -> f64 {
        let now_ns = self.start.elapsed().as_nanos() as u64;
        let prev_ns = self.last_ns.swap(now_ns, Ordering::Relaxed);

        if prev_ns == 0 {
            // First frame — no delta to compute yet.
            return 0.0;
        }

        let delta_s = (now_ns.saturating_sub(prev_ns)) as f64 / 1_000_000_000.0;
        let instant_fps = if delta_s > 0.0 { 1.0 / delta_s } else { 0.0 };

        loop {
            let old_bits = self.fps_ema_bits.load(Ordering::Relaxed);
            let old_ema = f64::from_bits(old_bits);
            let new_ema = if old_ema == 0.0 {
                instant_fps
            } else {
                FPS_ALPHA * instant_fps + (1.0 - FPS_ALPHA) * old_ema
            };
            if self
                .fps_ema_bits
                .compare_exchange_weak(old_bits, new_ema.to_bits(), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return new_ema;
            }
        }
    }
}

impl OutputSink for OverlaySink {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let count = self.frame_count.fetch_add(1, Ordering::Relaxed) + 1;
        let fps = self.update_fps();

        // Extract the frame from FrameInclusion::Always output.
        let Some(frame) = output.frame.as_ref() else {
            warn!("no frame in output — is frame_inclusion set to Always?");
            return;
        };

        let w = frame.width();
        let h = frame.height();
        let stride = frame.stride();

        // Work on a mutable copy so we can draw on it.
        let mut rgb = match frame.require_host_data() {
            Ok(cow) => cow.into_owned(),
            Err(_) => {
                warn!("frame is not host-accessible, skipping");
                return;
            }
        };

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
                feed = %output.feed_id,
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
        "Detection + Tracking — NextVision",
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
