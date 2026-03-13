//! Multi-panel egui UI with telemetry dashboard.
//!
//! Displays all feed video panels in a grid layout with detection/track
//! overlays, plus a top telemetry bar showing real-time system metrics.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use eframe::egui;
use nv_core::id::FeedId;
use nv_runtime::{
    BatchHandle, BatchMetrics, FeedHandle, OutputEnvelope, OutputSink,
    QueueTelemetry, RuntimeHandle,
};
use tracing::info;

// ---------------------------------------------------------------------------
// Helpers

/// Format a `Duration` as a compact human-readable string.
///
/// Examples: `"3s"`, `"2m 15s"`, `"1h 03m"`, `"2d 05h"`.
fn format_uptime(d: std::time::Duration) -> String {
    let total_secs = d.as_secs();
    let days = total_secs / 86_400;
    let hours = (total_secs % 86_400) / 3_600;
    let mins = (total_secs % 3_600) / 60;
    let secs = total_secs % 60;

    if days > 0 {
        format!("{days}d {:02}h", hours)
    } else if hours > 0 {
        format!("{hours}h {:02}m", mins)
    } else if mins > 0 {
        format!("{mins}m {secs:02}s")
    } else {
        format!("{secs}s")
    }
}

// ---------------------------------------------------------------------------
// Shared state between OutputSinks and the UI thread
// ---------------------------------------------------------------------------

/// Per-feed snapshot delivered from a sink to the UI.
///
/// Holds the `Arc<OutputEnvelope>` directly, avoiding per-frame deep
/// copies of pixel data and track vectors on the hot path.
struct FeedSnapshot {
    output: Arc<OutputEnvelope>,
}

/// Per-feed telemetry for the dashboard.
struct FeedTelemetry {
    frame_count: u64,
    fps_ema: f64,
    last_detection_count: usize,
    last_track_count: usize,
    last_pipeline_latency_us: u64,
    last_stage_latencies: Vec<(String, u64)>,
}

/// Shared state between all sinks and the UI thread.
pub struct UiState {
    /// Latest snapshot per feed, consumed by the UI thread.
    snapshots: HashMap<FeedId, FeedSnapshot>,
    /// Telemetry per feed.
    telemetry: HashMap<FeedId, FeedTelemetry>,
    /// Feed ordering (insertion order).
    feed_order: Vec<FeedId>,
}

impl UiState {
    fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            telemetry: HashMap::new(),
            feed_order: Vec::new(),
        }
    }
}

/// Thread-safe handle to the shared UI state.
pub type SharedUiState = Arc<Mutex<UiState>>;

/// Create a new shared UI state.
pub fn new_shared_state() -> SharedUiState {
    Arc::new(Mutex::new(UiState::new()))
}

// ---------------------------------------------------------------------------
// OutputSink implementation — one per feed, all push to shared state
// ---------------------------------------------------------------------------

/// EMA smoothing factor (α ≈ 0.05 → ~20-frame half-life).
const FPS_ALPHA: f64 = 0.05;

/// Output sink that pushes frames and telemetry to the shared UI state.
///
/// Each feed gets its own `UiSink`. The egui app reads from the shared
/// state on every repaint. The feed ID is learned from the first output
/// envelope, so the sink can be constructed before the feed is assigned
/// an ID by the runtime.
pub struct UiSink {
    state: SharedUiState,
    frame_count: AtomicU64,
    fps_ema_bits: AtomicU64,
    last_ns: AtomicU64,
    start: Instant,
}

impl UiSink {
    /// Create a new sink bound to the given shared state.
    pub fn new(state: SharedUiState) -> Self {
        Self {
            state,
            frame_count: AtomicU64::new(0),
            fps_ema_bits: AtomicU64::new(0_f64.to_bits()),
            last_ns: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    fn update_fps(&self) -> f64 {
        let now_ns = self.start.elapsed().as_nanos() as u64;
        let prev_ns = self.last_ns.swap(now_ns, Ordering::Relaxed);

        if prev_ns == 0 {
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
                .compare_exchange_weak(
                    old_bits,
                    new_ema.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return new_ema;
            }
        }
    }
}

impl OutputSink for UiSink {
    fn emit(&self, output: Arc<OutputEnvelope>) {
        let count = self.frame_count.fetch_add(1, Ordering::Relaxed) + 1;
        let fps = self.update_fps();
        let feed_id = output.feed_id;

        let detection_count = output.detections.len();
        let track_count = output.tracks.len();
        let pipeline_latency_us = output.provenance.total_latency.as_nanos() / 1000;
        let stage_latencies: Vec<(String, u64)> = output
            .provenance
            .stages
            .iter()
            .map(|sp| (format!("{}", sp.stage_id), sp.latency.as_nanos() / 1000))
            .collect();

        // Build snapshot if frame is available (zero-copy: just Arc-clone).
        if output.frame.is_some() {
            let snapshot = FeedSnapshot {
                output: Arc::clone(&output),
            };

            if let Ok(mut s) = self.state.lock() {
                // Register feed on first emit.
                if !s.feed_order.contains(&feed_id) {
                    s.feed_order.push(feed_id);
                }
                s.snapshots.insert(feed_id, snapshot);
                let t = s.telemetry.entry(feed_id).or_insert(FeedTelemetry {
                    frame_count: 0,
                    fps_ema: 0.0,
                    last_detection_count: 0,
                    last_track_count: 0,
                    last_pipeline_latency_us: 0,
                    last_stage_latencies: Vec::new(),
                });
                t.frame_count = count;
                t.fps_ema = fps;
                t.last_detection_count = detection_count;
                t.last_track_count = track_count;
                t.last_pipeline_latency_us = pipeline_latency_us;
                t.last_stage_latencies = stage_latencies;
            }
        }

        // Periodic log.
        if count % 60 == 0 {
            info!(
                feed = %feed_id,
                seq = output.frame_seq,
                detections = detection_count,
                tracks = track_count,
                fps = format!("{fps:.1}"),
                "pipeline running"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Track-colour palette (reused from overlay.rs)
// ---------------------------------------------------------------------------

const TRACK_COLOURS: &[[u8; 3]] = &[
    [255, 0, 0],
    [0, 255, 0],
    [0, 128, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0],
    [128, 255, 0],
    [0, 255, 128],
    [128, 0, 255],
];

fn colour_for_track(track_id: u64) -> egui::Color32 {
    let c = TRACK_COLOURS[track_id as usize % TRACK_COLOURS.len()];
    egui::Color32::from_rgb(c[0], c[1], c[2])
}

// ---------------------------------------------------------------------------
// rgb_to_rgba — convert strided RGB8 to RGBA for egui::ColorImage
// ---------------------------------------------------------------------------

fn rgb_to_rgba(rgb: &[u8], w: u32, h: u32, stride: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let row_start = (y * stride) as usize;
        for x in 0..w {
            let i = row_start + (x as usize) * 3;
            rgba.push(rgb[i]);
            rgba.push(rgb[i + 1]);
            rgba.push(rgb[i + 2]);
            rgba.push(255);
        }
    }
    rgba
}

// ---------------------------------------------------------------------------
// egui Application
// ---------------------------------------------------------------------------

/// The egui application state.
pub struct NvApp {
    state: SharedUiState,
    /// Cached textures per feed.
    textures: HashMap<FeedId, egui::TextureHandle>,
    /// Persisted last-good frame per feed (survives across repaints).
    last_frames: HashMap<FeedId, FeedSnapshot>,
    /// When each feed last delivered a frame.
    last_frame_times: HashMap<FeedId, Instant>,
    /// Feed handles for live telemetry queries.
    feed_handles: Vec<FeedHandle>,
    /// Optional batch handle for batch utilization telemetry.
    batch_handle: Option<BatchHandle>,
    /// Runtime handle for runtime-level telemetry.
    runtime_handle: RuntimeHandle,
}

impl NvApp {
    pub fn new(
        state: SharedUiState,
        feed_handles: Vec<FeedHandle>,
        batch_handle: Option<BatchHandle>,
        runtime_handle: RuntimeHandle,
    ) -> Self {
        Self {
            state,
            textures: HashMap::new(),
            last_frames: HashMap::new(),
            last_frame_times: HashMap::new(),
            feed_handles,
            batch_handle,
            runtime_handle,
        }
    }
}

impl eframe::App for NvApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repaint for live video.
        ctx.request_repaint();

        let state = self.state.lock().unwrap();
        let feed_order = state.feed_order.clone();
        let feed_count = feed_order.len();

        // Collect live telemetry from feed handles (outside of UiState lock).
        let queue_snapshots: Vec<(FeedId, QueueTelemetry)> = self
            .feed_handles
            .iter()
            .map(|h| (h.id(), h.queue_telemetry()))
            .collect();
        let feed_uptimes: Vec<(FeedId, std::time::Duration)> = self
            .feed_handles
            .iter()
            .map(|h| (h.id(), h.uptime()))
            .collect();
        let batch_metrics: Option<BatchMetrics> =
            self.batch_handle.as_ref().map(|b| b.metrics());
        let runtime_uptime = self.runtime_handle.uptime();

        // ------- Telemetry top panel -------
        egui::TopBottomPanel::top("telemetry").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("NextVision");
                ui.separator();
                ui.label(format!(
                    "{} feed(s)  up {}",
                    feed_count,
                    format_uptime(runtime_uptime)
                ));
                ui.separator();

                // Per-feed telemetry compact display.
                for (_idx, fid) in feed_order.iter().enumerate() {
                    if let Some(t) = state.telemetry.get(fid) {
                        let qt = queue_snapshots
                            .iter()
                            .find(|(id, _)| id == fid)
                            .map(|(_, q)| q);
                        let uptime = feed_uptimes
                            .iter()
                            .find(|(id, _)| id == fid)
                            .map(|(_, d)| *d);
                        ui.group(|ui| {
                            ui.vertical(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("{}", fid))
                                        .strong()
                                        .size(11.0),
                                );
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!("{:.1} fps", t.fps_ema))
                                            .size(11.0)
                                            .color(fps_colour(t.fps_ema)),
                                    ).on_hover_text("Frames per second (exponential moving average)");
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{}d {}t",
                                            t.last_detection_count, t.last_track_count
                                        ))
                                        .size(11.0),
                                    ).on_hover_text("Detections and active tracks in the last frame");
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{:.1}ms",
                                            t.last_pipeline_latency_us as f64 / 1000.0
                                        ))
                                        .size(11.0)
                                        .color(latency_colour(t.last_pipeline_latency_us)),
                                    ).on_hover_text("End-to-end pipeline latency for the last frame");
                                    if let Some(qt) = qt {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "src_q:{}/{}",
                                                qt.source_depth, qt.source_capacity
                                            ))
                                            .size(11.0)
                                            .color(queue_colour(qt.source_depth, qt.source_capacity)),
                                        ).on_hover_text("Source queue: frames buffered from media ingest (depth/capacity)");
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "sink_q:{}/{}",
                                                qt.sink_depth, qt.sink_capacity
                                            ))
                                            .size(11.0)
                                            .color(queue_colour(qt.sink_depth, qt.sink_capacity)),
                                        ).on_hover_text("Sink queue: processed frames waiting for output delivery (depth/capacity)");
                                    }
                                    if let Some(d) = uptime {
                                        ui.label(
                                            egui::RichText::new(format!("up:{}", format_uptime(d)))
                                                .size(9.0)
                                                .color(egui::Color32::GRAY),
                                        ).on_hover_text("Feed uptime since start");
                                    }
                                });
                                // Stage breakdown.
                                if !t.last_stage_latencies.is_empty() {
                                    ui.horizontal(|ui| {
                                        for (name, us) in &t.last_stage_latencies {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "{}: {:.1}ms",
                                                    name,
                                                    *us as f64 / 1000.0
                                                ))
                                                .size(9.0)
                                                .color(egui::Color32::GRAY),
                                            ).on_hover_text(format!("Latency of stage '{}' for the last frame", name));
                                        }
                                    });
                                }
                            });
                        });
                    }
                }

                // Batch utilization panel (only when batched mode is active).
                if let Some(ref bm) = batch_metrics {
                    ui.separator();
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new("Batch").strong().size(11.0));
                            let fill_text = match bm.avg_fill_ratio() {
                                Some(r) => format!("{:.0}% fill", r * 100.0),
                                None => "—".to_string(),
                            };
                            let avg_text = match bm.avg_batch_size() {
                                Some(a) => format!("{:.1}/{}", a, bm.configured_max_batch_size),
                                None => "—".to_string(),
                            };
                            ui.label(egui::RichText::new(avg_text).size(11.0))
                                .on_hover_text("Average batch size / configured maximum");
                            ui.label(
                                egui::RichText::new(fill_text)
                                    .size(11.0)
                                    .color(batch_fill_colour(bm.avg_fill_ratio())),
                            ).on_hover_text("Average batch fill ratio (actual / max)");
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}b {}p {}r pend:{}",
                                    bm.batches_dispatched,
                                    bm.items_processed,
                                    bm.items_rejected,
                                    bm.pending_items()
                                ))
                                .size(9.0)
                                .color(egui::Color32::GRAY),
                            ).on_hover_text("b=batches dispatched, p=items processed, r=items rejected, pend=items pending");
                        });
                    });
                }

                // Drops summary.
                ui.separator();
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label(egui::RichText::new("Drops").strong().size(11.0));
                        let total_drops: u64 = self
                            .feed_handles
                            .iter()
                            .map(|h| h.metrics().frames_dropped)
                            .sum();
                        let colour = if total_drops == 0 {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::YELLOW
                        };
                        ui.label(
                            egui::RichText::new(format!("{}", total_drops))
                                .size(11.0)
                                .color(colour),
                        ).on_hover_text("Total frames dropped across all feeds due to backpressure");
                    });
                });
            });
        });

        // ------- Video panels in central area -------

        // Compute grid dimensions: ceil(sqrt(n)) columns.
        let cols = if feed_count == 0 {
            1
        } else {
            (feed_count as f64).sqrt().ceil() as usize
        };

        // Move any new snapshots from shared state into our persisted last_frames.
        drop(state);
        {
            let mut state = self.state.lock().unwrap();
            for fid in &feed_order {
                if let Some(snap) = state.snapshots.remove(fid) {
                    self.last_frames.insert(*fid, snap);
                    self.last_frame_times.insert(*fid, Instant::now());
                }
            }
        }

        let now = Instant::now();

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let rows = (feed_count + cols - 1) / cols;
            let panel_w = available.x / cols as f32;
            let panel_h = available.y / rows.max(1) as f32;

            egui::Grid::new("video_grid")
                .num_columns(cols)
                .spacing([2.0, 2.0])
                .show(ui, |ui| {
                    // Pre-compute stale durations to avoid borrow issues.
                    let stale: Vec<(FeedId, f32)> = feed_order
                        .iter()
                        .map(|fid| {
                            let s = self
                                .last_frame_times
                                .get(fid)
                                .map(|t| now.duration_since(*t).as_secs_f32())
                                .unwrap_or(f32::MAX);
                            (*fid, s)
                        })
                        .collect();

                    for (i, (fid, stale_secs)) in stale.iter().enumerate() {
                        // Each cell is a fixed-size group.
                        ui.allocate_ui(egui::vec2(panel_w - 4.0, panel_h - 4.0), |ui| {
                            self.draw_feed_panel(ui, *fid, *stale_secs);
                        });

                        if (i + 1) % cols == 0 {
                            ui.end_row();
                        }
                    }
                });
        });
    }
}

impl NvApp {
    fn draw_feed_panel(
        &mut self,
        ui: &mut egui::Ui,
        feed_id: FeedId,
        stale_secs: f32,
    ) {
        let available = ui.available_size();
        let snapshot = self.last_frames.get(&feed_id);

        if let Some(snap) = snapshot {
            let frame = snap.output.frame.as_ref().expect("snapshot has frame");
            let rgb = frame.data();
            let width = frame.width();
            let height = frame.height();
            let stride = frame.stride();

            // Upload / update texture.
            let rgba = rgb_to_rgba(rgb, width, height, stride);
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [width as usize, height as usize],
                &rgba,
            );

            let tex = self
                .textures
                .entry(feed_id)
                .or_insert_with(|| {
                    ui.ctx().load_texture(
                        format!("feed-{}", feed_id),
                        image.clone(),
                        egui::TextureOptions::LINEAR,
                    )
                });
            tex.set(image, egui::TextureOptions::LINEAR);

            // Compute scaled size maintaining aspect ratio.
            let img_w = width as f32;
            let img_h = height as f32;
            let scale = (available.x / img_w).min(available.y / img_h);
            let disp_w = img_w * scale;
            let disp_h = img_h * scale;

            // Draw the video frame.
            let (rect, _response) =
                ui.allocate_exact_size(egui::vec2(disp_w, disp_h), egui::Sense::hover());

            ui.painter().image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Draw track overlays.
            for t in &snap.output.tracks {
                let bb = &t.current.bbox;
                let x0 = rect.min.x + bb.x_min * disp_w;
                let y0 = rect.min.y + bb.y_min * disp_h;
                let x1 = rect.min.x + bb.x_max * disp_w;
                let y1 = rect.min.y + bb.y_max * disp_h;

                let colour = colour_for_track(t.id.as_u64());
                let box_rect = egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1));
                ui.painter().rect_stroke(
                    box_rect,
                    0.0,
                    egui::Stroke::new(2.0, colour),
                    egui::StrokeKind::Outside,
                );

                // Detection type + track ID label.
                let label = format!("{} (T{})", crate::overlay::coco_class_name(t.class_id), t.id.as_u64());
                ui.painter().text(
                    egui::pos2(x0, y0 - 2.0),
                    egui::Align2::LEFT_BOTTOM,
                    &label,
                    egui::FontId::proportional(11.0),
                    colour,
                );
            }

            // Detection count + seq overlay in bottom-left.
            let info_text = format!(
                "seq:{} det:{} trk:{} lat:{:.1}ms",
                snap.output.frame_seq,
                snap.output.detections.len(),
                snap.output.tracks.len(),
                snap.output.provenance.total_latency.as_nanos() as f64 / 1_000_000.0,
            );
            ui.painter().text(
                egui::pos2(rect.min.x + 4.0, rect.max.y - 4.0),
                egui::Align2::LEFT_BOTTOM,
                &info_text,
                egui::FontId::proportional(10.0),
                egui::Color32::WHITE,
            );

            // Stale frame warning overlay (>10s without a new frame).
            if stale_secs > 10.0 {
                ui.painter().rect_filled(
                    rect,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(180, 0, 0, 120),
                );
                let stale_text = format!("STALE — no frames for {:.0}s", stale_secs);
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    &stale_text,
                    egui::FontId::proportional(16.0),
                    egui::Color32::WHITE,
                );
            }

            // Feed ID overlay — top-left of the video.
            let label_text = format!("{}", feed_id);
            let label_pos = egui::pos2(rect.min.x + 4.0, rect.min.y + 4.0);
            // Dark background pill for readability.
            let galley = ui.painter().layout_no_wrap(
                label_text.clone(),
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
            let label_rect = egui::Rect::from_min_size(
                label_pos,
                galley.size() + egui::vec2(6.0, 2.0),
            );
            ui.painter().rect_filled(
                label_rect,
                3.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160),
            );
            ui.painter().text(
                label_pos + egui::vec2(3.0, 1.0),
                egui::Align2::LEFT_TOP,
                &label_text,
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
        } else {
            // No frame ever received — show placeholder.
            let (rect, _) =
                ui.allocate_exact_size(available, egui::Sense::hover());
            ui.painter().rect_filled(rect, 0.0, egui::Color32::from_gray(30));
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Waiting for frames...",
                egui::FontId::proportional(14.0),
                egui::Color32::GRAY,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Colour helpers for telemetry
// ---------------------------------------------------------------------------

fn fps_colour(fps: f64) -> egui::Color32 {
    if fps >= 5.0 {
        egui::Color32::GREEN
    } else if fps >= 1.0 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::RED
    }
}

fn latency_colour(us: u64) -> egui::Color32 {
    if us < 100_000 {
        egui::Color32::GREEN
    } else if us < 500_000 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::RED
    }
}

fn queue_colour(depth: usize, capacity: usize) -> egui::Color32 {
    if capacity == 0 {
        return egui::Color32::GRAY;
    }
    let ratio = depth as f64 / capacity as f64;
    if ratio < 0.5 {
        egui::Color32::GREEN
    } else if ratio < 0.9 {
        egui::Color32::YELLOW
    } else {
        egui::Color32::RED
    }
}

fn batch_fill_colour(ratio: Option<f64>) -> egui::Color32 {
    match ratio {
        Some(r) if r >= 0.8 => egui::Color32::GREEN,
        Some(r) if r >= 0.4 => egui::Color32::YELLOW,
        Some(_) => egui::Color32::from_rgb(255, 165, 0), // orange — underutilized
        None => egui::Color32::GRAY,
    }
}

// ---------------------------------------------------------------------------
// Launch function
// ---------------------------------------------------------------------------

/// Launch the egui window. This blocks the calling thread.
///
/// Call this from the main thread after setting up the runtime and feeds.
/// The window renders all feeds registered via `UiSink` instances that
/// share the same `SharedUiState`.
pub fn run_ui(
    state: SharedUiState,
    feed_handles: Vec<FeedHandle>,
    batch_handle: Option<BatchHandle>,
    runtime_handle: RuntimeHandle,
) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NextVision — Detection + Tracking")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "NextVision",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(NvApp::new(state, feed_handles, batch_handle, runtime_handle)))
        }),
    )
}
