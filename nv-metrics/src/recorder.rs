//! Maps [`RuntimeDiagnostics`] snapshots and [`HealthEvent`]s to
//! OpenTelemetry instrument recordings.
//!
//! **Gauges** (phase 1) — created once at startup. Each poll cycle calls
//! [`Instruments::record()`] with a fresh snapshot.
//!
//! **Counters** (phase 2) — event-driven. Each [`HealthEvent`] received
//! from the broadcast channel increments the corresponding counter via
//! [`HealthCounters::record()`].

use nv_runtime::diagnostics::{RuntimeDiagnostics, ViewStatus};
use nv_runtime::HealthEvent;
use opentelemetry::metrics::{Counter, Gauge, Meter};
use opentelemetry::KeyValue;

// ---------------------------------------------------------------------------
// Instrument set
// ---------------------------------------------------------------------------

/// Holds all OTel instruments. Created once, reused every poll cycle.
pub(crate) struct Instruments {
    // -- Runtime-level --
    runtime_uptime_seconds: Gauge<f64>,
    runtime_feed_count: Gauge<u64>,
    runtime_max_feeds: Gauge<u64>,
    runtime_output_lag: Gauge<u64>,
    runtime_output_lag_pending_lost: Gauge<u64>,

    // -- Per-feed --
    feed_alive: Gauge<u64>,
    feed_paused: Gauge<u64>,
    feed_uptime_seconds: Gauge<f64>,
    feed_frames_received: Gauge<u64>,
    feed_frames_dropped: Gauge<u64>,
    feed_frames_processed: Gauge<u64>,
    feed_tracks_active: Gauge<u64>,
    feed_view_epoch: Gauge<u64>,
    feed_restarts: Gauge<u64>,
    feed_queue_source_depth: Gauge<u64>,
    feed_queue_source_capacity: Gauge<u64>,
    feed_queue_sink_depth: Gauge<u64>,
    feed_queue_sink_capacity: Gauge<u64>,
    feed_view_stability_score: Gauge<f64>,
    feed_view_status: Gauge<u64>,

    // -- Per-batch --
    batch_dispatched: Gauge<u64>,
    batch_items_processed: Gauge<u64>,
    batch_items_submitted: Gauge<u64>,
    batch_items_rejected: Gauge<u64>,
    batch_items_timed_out: Gauge<u64>,
    batch_avg_processing_ms: Gauge<f64>,
    batch_avg_formation_ms: Gauge<f64>,
    batch_avg_fill_ratio: Gauge<f64>,
    batch_consecutive_errors: Gauge<u64>,
    batch_pending_items: Gauge<u64>,
}

impl Instruments {
    pub(crate) fn new(meter: &Meter) -> Self {
        Self {
            // Runtime
            runtime_uptime_seconds: meter
                .f64_gauge("nv.runtime.uptime_seconds")
                .with_description("Elapsed seconds since the runtime started")
                .build(),
            runtime_feed_count: meter
                .u64_gauge("nv.runtime.feed_count")
                .with_description("Number of currently active feeds")
                .build(),
            runtime_max_feeds: meter
                .u64_gauge("nv.runtime.max_feeds")
                .with_description("Maximum allowed concurrent feeds")
                .build(),
            runtime_output_lag: meter
                .u64_gauge("nv.runtime.output_lag")
                .with_description("1 if the output broadcast channel is saturated, 0 otherwise")
                .build(),
            runtime_output_lag_pending_lost: meter
                .u64_gauge("nv.runtime.output_lag_pending_lost")
                .with_description("Pending lost messages during output channel saturation")
                .build(),

            // Feed
            feed_alive: meter
                .u64_gauge("nv.feed.alive")
                .with_description("1 if the feed worker thread is alive, 0 otherwise")
                .build(),
            feed_paused: meter
                .u64_gauge("nv.feed.paused")
                .with_description("1 if the feed is paused, 0 otherwise")
                .build(),
            feed_uptime_seconds: meter
                .f64_gauge("nv.feed.uptime_seconds")
                .with_description("Seconds since the feed's current session started (resets on restart)")
                .build(),
            feed_frames_received: meter
                .u64_gauge("nv.feed.frames_received")
                .with_description("Total frames received from the source")
                .build(),
            feed_frames_dropped: meter
                .u64_gauge("nv.feed.frames_dropped")
                .with_description("Total frames dropped due to backpressure")
                .build(),
            feed_frames_processed: meter
                .u64_gauge("nv.feed.frames_processed")
                .with_description("Total frames processed through all stages")
                .build(),
            feed_tracks_active: meter
                .u64_gauge("nv.feed.tracks_active")
                .with_description("Number of active tracks in the temporal store")
                .build(),
            feed_view_epoch: meter
                .u64_gauge("nv.feed.view_epoch")
                .with_description("Current view epoch value")
                .build(),
            feed_restarts: meter
                .u64_gauge("nv.feed.restarts")
                .with_description("Number of feed restarts")
                .build(),
            feed_queue_source_depth: meter
                .u64_gauge("nv.feed.queue_source_depth")
                .with_description("Current source queue depth")
                .build(),
            feed_queue_source_capacity: meter
                .u64_gauge("nv.feed.queue_source_capacity")
                .with_description("Source queue capacity")
                .build(),
            feed_queue_sink_depth: meter
                .u64_gauge("nv.feed.queue_sink_depth")
                .with_description("Current sink queue depth")
                .build(),
            feed_queue_sink_capacity: meter
                .u64_gauge("nv.feed.queue_sink_capacity")
                .with_description("Sink queue capacity")
                .build(),
            feed_view_stability_score: meter
                .f64_gauge("nv.feed.view_stability_score")
                .with_description("View stability score [0.0, 1.0] — 1.0 is fully stable")
                .build(),
            feed_view_status: meter
                .u64_gauge("nv.feed.view_status")
                .with_description("View health: 0=Stable, 1=Degraded, 2=Invalid")
                .build(),

            // Batch
            batch_dispatched: meter
                .u64_gauge("nv.batch.dispatched")
                .with_description("Total batches dispatched to the processor")
                .build(),
            batch_items_processed: meter
                .u64_gauge("nv.batch.items_processed")
                .with_description("Total items processed by the batch processor")
                .build(),
            batch_items_submitted: meter
                .u64_gauge("nv.batch.items_submitted")
                .with_description("Total items submitted by feed threads")
                .build(),
            batch_items_rejected: meter
                .u64_gauge("nv.batch.items_rejected")
                .with_description("Total items rejected (queue full or disconnected)")
                .build(),
            batch_items_timed_out: meter
                .u64_gauge("nv.batch.items_timed_out")
                .with_description("Total items whose response timed out")
                .build(),
            batch_avg_processing_ms: meter
                .f64_gauge("nv.batch.avg_processing_ms")
                .with_description("Average batch processing time in milliseconds")
                .build(),
            batch_avg_formation_ms: meter
                .f64_gauge("nv.batch.avg_formation_ms")
                .with_description("Average batch formation wait time in milliseconds")
                .build(),
            batch_avg_fill_ratio: meter
                .f64_gauge("nv.batch.avg_fill_ratio")
                .with_description("Average batch fill ratio [0.0, 1.0]")
                .build(),
            batch_consecutive_errors: meter
                .u64_gauge("nv.batch.consecutive_errors")
                .with_description("Consecutive batch processor errors since last success")
                .build(),
            batch_pending_items: meter
                .u64_gauge("nv.batch.pending_items")
                .with_description("Approximate items in-flight (submitted but not processed/rejected)")
                .build(),
        }
    }

    /// Record all metrics from a single diagnostics snapshot.
    pub(crate) fn record(&self, diag: &RuntimeDiagnostics) {
        // -- Runtime --
        self.runtime_uptime_seconds
            .record(diag.uptime.as_secs_f64(), &[]);
        self.runtime_feed_count
            .record(diag.feed_count as u64, &[]);
        self.runtime_max_feeds
            .record(diag.max_feeds as u64, &[]);
        self.runtime_output_lag
            .record(u64::from(diag.output_lag.in_lag), &[]);
        self.runtime_output_lag_pending_lost
            .record(diag.output_lag.pending_lost, &[]);

        // -- Per-feed --
        for feed in &diag.feeds {
            let attrs = [KeyValue::new(
                "feed_id",
                feed.feed_id.to_string(),
            )];

            self.feed_alive.record(u64::from(feed.alive), &attrs);
            self.feed_paused.record(u64::from(feed.paused), &attrs);
            self.feed_uptime_seconds
                .record(feed.uptime.as_secs_f64(), &attrs);
            self.feed_frames_received
                .record(feed.metrics.frames_received, &attrs);
            self.feed_frames_dropped
                .record(feed.metrics.frames_dropped, &attrs);
            self.feed_frames_processed
                .record(feed.metrics.frames_processed, &attrs);
            self.feed_tracks_active
                .record(feed.metrics.tracks_active, &attrs);
            self.feed_view_epoch
                .record(feed.metrics.view_epoch, &attrs);
            self.feed_restarts
                .record(u64::from(feed.metrics.restarts), &attrs);
            self.feed_queue_source_depth
                .record(feed.queues.source_depth as u64, &attrs);
            self.feed_queue_source_capacity
                .record(feed.queues.source_capacity as u64, &attrs);
            self.feed_queue_sink_depth
                .record(feed.queues.sink_depth as u64, &attrs);
            self.feed_queue_sink_capacity
                .record(feed.queues.sink_capacity as u64, &attrs);
            self.feed_view_stability_score
                .record(f64::from(feed.view.stability_score), &attrs);
            self.feed_view_status
                .record(view_status_to_u64(feed.view.status), &attrs);
        }

        // -- Per-batch --
        for batch in &diag.batches {
            let attrs = [KeyValue::new(
                "processor_id",
                batch.processor_id.as_str().to_owned(),
            )];

            self.batch_dispatched
                .record(batch.metrics.batches_dispatched, &attrs);
            self.batch_items_processed
                .record(batch.metrics.items_processed, &attrs);
            self.batch_items_submitted
                .record(batch.metrics.items_submitted, &attrs);
            self.batch_items_rejected
                .record(batch.metrics.items_rejected, &attrs);
            self.batch_items_timed_out
                .record(batch.metrics.items_timed_out, &attrs);
            self.batch_consecutive_errors
                .record(batch.metrics.consecutive_errors, &attrs);
            self.batch_pending_items
                .record(batch.metrics.pending_items(), &attrs);

            if let Some(ms) = batch.metrics.avg_processing_ns() {
                self.batch_avg_processing_ms
                    .record(ms / 1_000_000.0, &attrs);
            }
            if let Some(ms) = batch.metrics.avg_formation_ns() {
                self.batch_avg_formation_ms
                    .record(ms / 1_000_000.0, &attrs);
            }
            if let Some(ratio) = batch.metrics.avg_fill_ratio() {
                self.batch_avg_fill_ratio.record(ratio, &attrs);
            }
        }
    }
}

fn view_status_to_u64(status: ViewStatus) -> u64 {
    match status {
        ViewStatus::Stable => 0,
        ViewStatus::Degraded => 1,
        ViewStatus::Invalid => 2,
    }
}

// ---------------------------------------------------------------------------
// Health-event counters (phase 2)
// ---------------------------------------------------------------------------

/// OTel counters driven by [`HealthEvent`]s from the runtime broadcast.
///
/// Each counter is monotonic — the OTel SDK accumulates totals and
/// downstream systems derive rates as needed.
pub(crate) struct HealthCounters {
    source_disconnections: Counter<u64>,
    source_reconnections: Counter<u64>,
    stage_errors: Counter<u64>,
    stage_panics: Counter<u64>,
    feed_restarts: Counter<u64>,
    feeds_stopped: Counter<u64>,
    backpressure_drops: Counter<u64>,
    view_epoch_changes: Counter<u64>,
    view_degradations: Counter<u64>,
    output_lag_events: Counter<u64>,
    sink_panics: Counter<u64>,
    sink_timeouts: Counter<u64>,
    sink_backpressure: Counter<u64>,
    track_admission_rejected: Counter<u64>,
    batch_errors: Counter<u64>,
    batch_submission_rejected: Counter<u64>,
    batch_timeouts: Counter<u64>,
    batch_in_flight_exceeded: Counter<u64>,
}

impl HealthCounters {
    pub(crate) fn new(meter: &Meter) -> Self {
        Self {
            source_disconnections: meter
                .u64_counter("nv.health.source_disconnections")
                .with_description("Source disconnection events")
                .build(),
            source_reconnections: meter
                .u64_counter("nv.health.source_reconnections")
                .with_description("Source reconnection attempts")
                .build(),
            stage_errors: meter
                .u64_counter("nv.health.stage_errors")
                .with_description("Stage processing errors")
                .build(),
            stage_panics: meter
                .u64_counter("nv.health.stage_panics")
                .with_description("Stage panics")
                .build(),
            feed_restarts: meter
                .u64_counter("nv.health.feed_restarts")
                .with_description("Feed restart events")
                .build(),
            feeds_stopped: meter
                .u64_counter("nv.health.feeds_stopped")
                .with_description("Feeds permanently stopped")
                .build(),
            backpressure_drops: meter
                .u64_counter("nv.health.backpressure_drops")
                .with_description("Frames dropped due to backpressure")
                .build(),
            view_epoch_changes: meter
                .u64_counter("nv.health.view_epoch_changes")
                .with_description("View epoch change events")
                .build(),
            view_degradations: meter
                .u64_counter("nv.health.view_degradations")
                .with_description("View degradation events")
                .build(),
            output_lag_events: meter
                .u64_counter("nv.health.output_lag_events")
                .with_description("Output channel saturation events")
                .build(),
            sink_panics: meter
                .u64_counter("nv.health.sink_panics")
                .with_description("Output sink panic events")
                .build(),
            sink_timeouts: meter
                .u64_counter("nv.health.sink_timeouts")
                .with_description("Output sink timeout events")
                .build(),
            sink_backpressure: meter
                .u64_counter("nv.health.sink_backpressure")
                .with_description("Output sink backpressure drop events")
                .build(),
            track_admission_rejected: meter
                .u64_counter("nv.health.track_admission_rejected")
                .with_description("Tracks rejected by temporal store admission control")
                .build(),
            batch_errors: meter
                .u64_counter("nv.health.batch_errors")
                .with_description("Batch processor errors")
                .build(),
            batch_submission_rejected: meter
                .u64_counter("nv.health.batch_submission_rejected")
                .with_description("Batch submissions rejected (queue full)")
                .build(),
            batch_timeouts: meter
                .u64_counter("nv.health.batch_timeouts")
                .with_description("Batch response timeouts")
                .build(),
            batch_in_flight_exceeded: meter
                .u64_counter("nv.health.batch_in_flight_exceeded")
                .with_description("Batch submissions rejected (in-flight cap)")
                .build(),
        }
    }

    /// Increment the appropriate counter for a single health event.
    pub(crate) fn record(&self, event: &HealthEvent) {
        match event {
            HealthEvent::SourceConnected { .. } => {}
            HealthEvent::SourceEos { .. } => {}
            HealthEvent::DecodeDecision { .. } => {}
            HealthEvent::ViewCompensationApplied { .. } => {}

            HealthEvent::SourceDisconnected { feed_id, .. } => {
                self.source_disconnections.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::SourceReconnecting { feed_id, .. } => {
                self.source_reconnections.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::StageError { feed_id, stage_id, .. } => {
                self.stage_errors.add(1, &[
                    KeyValue::new("feed_id", feed_id.to_string()),
                    KeyValue::new("stage_id", stage_id.as_str().to_owned()),
                ]);
            }
            HealthEvent::StagePanic { feed_id, stage_id } => {
                self.stage_panics.add(1, &[
                    KeyValue::new("feed_id", feed_id.to_string()),
                    KeyValue::new("stage_id", stage_id.as_str().to_owned()),
                ]);
            }
            HealthEvent::FeedRestarting { feed_id, .. } => {
                self.feed_restarts.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::FeedStopped { feed_id, .. } => {
                self.feeds_stopped.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::BackpressureDrop { feed_id, frames_dropped } => {
                self.backpressure_drops.add(*frames_dropped, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::ViewEpochChanged { feed_id, .. } => {
                self.view_epoch_changes.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::ViewDegraded { feed_id, .. } => {
                self.view_degradations.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::OutputLagged { messages_lost } => {
                self.output_lag_events.add(*messages_lost, &[]);
            }
            HealthEvent::SinkPanic { feed_id } => {
                self.sink_panics.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::SinkTimeout { feed_id } => {
                self.sink_timeouts.add(1, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::SinkBackpressure { feed_id, outputs_dropped } => {
                self.sink_backpressure.add(*outputs_dropped, &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::TrackAdmissionRejected { feed_id, rejected_count } => {
                self.track_admission_rejected.add(u64::from(*rejected_count), &[KeyValue::new("feed_id", feed_id.to_string())]);
            }
            HealthEvent::BatchError { processor_id, .. } => {
                self.batch_errors.add(1, &[KeyValue::new("processor_id", processor_id.as_str().to_owned())]);
            }
            HealthEvent::BatchSubmissionRejected { feed_id, processor_id, dropped_count } => {
                self.batch_submission_rejected.add(*dropped_count, &[
                    KeyValue::new("feed_id", feed_id.to_string()),
                    KeyValue::new("processor_id", processor_id.as_str().to_owned()),
                ]);
            }
            HealthEvent::BatchTimeout { feed_id, processor_id, timed_out_count } => {
                self.batch_timeouts.add(*timed_out_count, &[
                    KeyValue::new("feed_id", feed_id.to_string()),
                    KeyValue::new("processor_id", processor_id.as_str().to_owned()),
                ]);
            }
            HealthEvent::BatchInFlightExceeded { feed_id, processor_id, rejected_count } => {
                self.batch_in_flight_exceeded.add(*rejected_count, &[
                    KeyValue::new("feed_id", feed_id.to_string()),
                    KeyValue::new("processor_id", processor_id.as_str().to_owned()),
                ]);
            }
            HealthEvent::ResidencyDowngrade { .. } | HealthEvent::InsecureRtspSource { .. } => {
                // Informational — no counter needed. Operators observe this
                // via health event subscribers or logs.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nv_runtime::diagnostics::{
        BatchDiagnostics, FeedDiagnostics, OutputLagStatus, ViewDiagnostics,
    };
    use nv_runtime::BatchMetrics;
    use nv_runtime::QueueTelemetry;
    use nv_core::id::{FeedId, StageId};
    use nv_core::metrics::FeedMetrics;
    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
    use std::time::Duration;

    fn sample_diagnostics() -> RuntimeDiagnostics {
        RuntimeDiagnostics {
            uptime: Duration::from_secs(120),
            feed_count: 2,
            max_feeds: 8,
            feeds: vec![
                FeedDiagnostics {
                    feed_id: FeedId::new(1),
                    alive: true,
                    paused: false,
                    uptime: Duration::from_secs(100),
                    metrics: FeedMetrics {
                        feed_id: FeedId::new(1),
                        frames_received: 5000,
                        frames_dropped: 10,
                        frames_processed: 4990,
                        tracks_active: 7,
                        view_epoch: 3,
                        restarts: 1,
                    },
                    queues: QueueTelemetry {
                        source_depth: 2,
                        source_capacity: 16,
                        sink_depth: 0,
                        sink_capacity: 8,
                    },
                    decode: None,
                    view: ViewDiagnostics {
                        epoch: 3,
                        stability_score: 0.95,
                        status: ViewStatus::Stable,
                    },
                    batch_processor_id: Some(StageId("yolov8")),
                },
            ],
            batches: vec![
                BatchDiagnostics {
                    processor_id: StageId("yolov8"),
                    metrics: BatchMetrics {
                        batches_dispatched: 250,
                        items_processed: 980,
                        items_submitted: 1000,
                        items_rejected: 5,
                        items_timed_out: 2,
                        total_processing_ns: 25_000_000_000,
                        total_formation_ns: 5_000_000_000,
                        min_batch_size: 2,
                        max_batch_size_seen: 4,
                        configured_max_batch_size: 4,
                        consecutive_errors: 0,
                    },
                },
            ],
            output_lag: OutputLagStatus {
                in_lag: false,
                pending_lost: 0,
            },
            detached_thread_count: 0,
        }
    }

    /// Verify that `Instruments::record()` does not panic with valid diagnostics.
    #[test]
    fn record_does_not_panic() {
        let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);
        let diag = sample_diagnostics();
        instruments.record(&diag);
    }

    /// Verify recording with an empty diagnostics snapshot.
    #[test]
    fn record_empty_diagnostics() {
        let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);
        let diag = RuntimeDiagnostics {
            uptime: Duration::ZERO,
            feed_count: 0,
            max_feeds: 4,
            feeds: vec![],
            batches: vec![],
            output_lag: OutputLagStatus {
                in_lag: false,
                pending_lost: 0,
            },
            detached_thread_count: 0,
        };
        instruments.record(&diag);
    }

    /// Verify recording with a batch that has no dispatches yet (avg returns None).
    #[test]
    fn record_batch_no_dispatches() {
        let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);
        let diag = RuntimeDiagnostics {
            uptime: Duration::from_secs(1),
            feed_count: 0,
            max_feeds: 4,
            feeds: vec![],
            batches: vec![BatchDiagnostics {
                processor_id: StageId("empty"),
                metrics: BatchMetrics::default(),
            }],
            output_lag: OutputLagStatus {
                in_lag: false,
                pending_lost: 0,
            },
            detached_thread_count: 0,
        };
        instruments.record(&diag);
    }

    #[test]
    fn view_status_encoding() {
        assert_eq!(view_status_to_u64(ViewStatus::Stable), 0);
        assert_eq!(view_status_to_u64(ViewStatus::Degraded), 1);
        assert_eq!(view_status_to_u64(ViewStatus::Invalid), 2);
    }

    // -- In-memory exporter tests: verify actual metric values --

    /// Build a provider backed by the in-memory exporter for assertion.
    fn in_memory_provider() -> (SdkMeterProvider, InMemoryMetricExporter) {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        (provider, exporter)
    }

    /// Find a gauge data point value by metric name in exported data.
    fn find_u64_gauge(exporter: &InMemoryMetricExporter, name: &str) -> Option<u64> {
        let metrics = exporter.get_finished_metrics().ok()?;
        for rm in &metrics {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == name {
                        if let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = m.data() {
                            return gauge.data_points().next().map(|dp| dp.value());
                        }
                    }
                }
            }
        }
        None
    }

    fn find_f64_gauge(exporter: &InMemoryMetricExporter, name: &str) -> Option<f64> {
        let metrics = exporter.get_finished_metrics().ok()?;
        for rm in &metrics {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == name {
                        if let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = m.data() {
                            return gauge.data_points().next().map(|dp| dp.value());
                        }
                    }
                }
            }
        }
        None
    }

    /// Count how many distinct metric names were exported.
    fn metric_names(exporter: &InMemoryMetricExporter) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(metrics) = exporter.get_finished_metrics() {
            for rm in &metrics {
                for sm in rm.scope_metrics() {
                    for m in sm.metrics() {
                        names.push(m.name().to_owned());
                    }
                }
            }
        }
        names.sort();
        names.dedup();
        names
    }

    #[test]
    fn in_memory_runtime_metrics() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        instruments.record(&diag);

        // Force a collection cycle so in-memory exporter captures data.
        drop(instruments);
        let _ = provider.force_flush();

        assert_eq!(find_u64_gauge(&exporter, "nv.runtime.feed_count"), Some(2));
        assert_eq!(find_u64_gauge(&exporter, "nv.runtime.max_feeds"), Some(8));
        assert_eq!(find_u64_gauge(&exporter, "nv.runtime.output_lag"), Some(0));

        let uptime = find_f64_gauge(&exporter, "nv.runtime.uptime_seconds");
        assert!(uptime.is_some());
        assert!((uptime.unwrap() - 120.0).abs() < 0.001);
    }

    #[test]
    fn in_memory_feed_metrics() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        assert_eq!(find_u64_gauge(&exporter, "nv.feed.frames_received"), Some(5000));
        assert_eq!(find_u64_gauge(&exporter, "nv.feed.frames_dropped"), Some(10));
        assert_eq!(find_u64_gauge(&exporter, "nv.feed.tracks_active"), Some(7));
        assert_eq!(find_u64_gauge(&exporter, "nv.feed.alive"), Some(1));
        assert_eq!(find_u64_gauge(&exporter, "nv.feed.view_status"), Some(0)); // Stable

        let stability = find_f64_gauge(&exporter, "nv.feed.view_stability_score");
        assert!(stability.is_some());
        assert!((stability.unwrap() - 0.95).abs() < 0.001);
    }

    #[test]
    fn in_memory_batch_metrics() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        assert_eq!(find_u64_gauge(&exporter, "nv.batch.dispatched"), Some(250));
        assert_eq!(find_u64_gauge(&exporter, "nv.batch.items_processed"), Some(980));
        assert_eq!(find_u64_gauge(&exporter, "nv.batch.items_rejected"), Some(5));

        let avg_proc = find_f64_gauge(&exporter, "nv.batch.avg_processing_ms");
        assert!(avg_proc.is_some());
        // 25_000_000_000 ns / 250 batches = 100_000_000 ns = 100 ms
        assert!((avg_proc.unwrap() - 100.0).abs() < 0.1);
    }

    #[test]
    fn all_expected_metrics_present() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        let names = metric_names(&exporter);

        // Runtime metrics
        assert!(names.contains(&"nv.runtime.uptime_seconds".to_owned()));
        assert!(names.contains(&"nv.runtime.feed_count".to_owned()));
        assert!(names.contains(&"nv.runtime.max_feeds".to_owned()));
        assert!(names.contains(&"nv.runtime.output_lag".to_owned()));
        assert!(names.contains(&"nv.runtime.output_lag_pending_lost".to_owned()));

        // Feed metrics
        assert!(names.contains(&"nv.feed.alive".to_owned()));
        assert!(names.contains(&"nv.feed.paused".to_owned()));
        assert!(names.contains(&"nv.feed.uptime_seconds".to_owned()));
        assert!(names.contains(&"nv.feed.frames_received".to_owned()));
        assert!(names.contains(&"nv.feed.frames_dropped".to_owned()));
        assert!(names.contains(&"nv.feed.frames_processed".to_owned()));
        assert!(names.contains(&"nv.feed.tracks_active".to_owned()));
        assert!(names.contains(&"nv.feed.view_epoch".to_owned()));
        assert!(names.contains(&"nv.feed.restarts".to_owned()));
        assert!(names.contains(&"nv.feed.queue_source_depth".to_owned()));
        assert!(names.contains(&"nv.feed.queue_source_capacity".to_owned()));
        assert!(names.contains(&"nv.feed.queue_sink_depth".to_owned()));
        assert!(names.contains(&"nv.feed.queue_sink_capacity".to_owned()));
        assert!(names.contains(&"nv.feed.view_stability_score".to_owned()));
        assert!(names.contains(&"nv.feed.view_status".to_owned()));

        // Batch metrics
        assert!(names.contains(&"nv.batch.dispatched".to_owned()));
        assert!(names.contains(&"nv.batch.items_processed".to_owned()));
        assert!(names.contains(&"nv.batch.items_submitted".to_owned()));
        assert!(names.contains(&"nv.batch.items_rejected".to_owned()));
        assert!(names.contains(&"nv.batch.items_timed_out".to_owned()));
        assert!(names.contains(&"nv.batch.avg_processing_ms".to_owned()));
        assert!(names.contains(&"nv.batch.avg_formation_ms".to_owned()));
        assert!(names.contains(&"nv.batch.avg_fill_ratio".to_owned()));
        assert!(names.contains(&"nv.batch.consecutive_errors".to_owned()));
        assert!(names.contains(&"nv.batch.pending_items".to_owned()));
    }

    #[test]
    fn feed_metrics_carry_feed_id_attribute() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        let metrics = exporter.get_finished_metrics().unwrap();
        let mut found_attr = false;
        for rm in &metrics {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == "nv.feed.frames_received" {
                        if let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = m.data() {
                            for dp in gauge.data_points() {
                                for attr in dp.attributes() {
                                    if attr.key.as_str() == "feed_id" {
                                        found_attr = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found_attr, "feed metrics must carry a feed_id attribute");
    }

    #[test]
    fn batch_metrics_carry_processor_id_attribute() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        let metrics = exporter.get_finished_metrics().unwrap();
        let mut found_attr = false;
        for rm in &metrics {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == "nv.batch.dispatched" {
                        if let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = m.data() {
                            for dp in gauge.data_points() {
                                for attr in dp.attributes() {
                                    if attr.key.as_str() == "processor_id" {
                                        found_attr = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found_attr, "batch metrics must carry a processor_id attribute");
    }

    #[test]
    fn output_lag_records_pending_lost() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let mut diag = sample_diagnostics();
        diag.output_lag.in_lag = true;
        diag.output_lag.pending_lost = 42;
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        assert_eq!(find_u64_gauge(&exporter, "nv.runtime.output_lag"), Some(1));
        assert_eq!(find_u64_gauge(&exporter, "nv.runtime.output_lag_pending_lost"), Some(42));
    }

    #[test]
    fn feed_paused_recorded() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let mut diag = sample_diagnostics();
        diag.feeds[0].paused = true;
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        assert_eq!(find_u64_gauge(&exporter, "nv.feed.paused"), Some(1));
    }

    #[test]
    fn batch_pending_items_recorded() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let instruments = Instruments::new(&meter);

        let diag = sample_diagnostics();
        // pending = submitted - processed - rejected = 1000 - 980 - 5 = 15
        instruments.record(&diag);

        drop(instruments);
        let _ = provider.force_flush();

        assert_eq!(find_u64_gauge(&exporter, "nv.batch.pending_items"), Some(15));
    }

    // -- Health counter helpers & tests --

    fn find_u64_sum(exporter: &InMemoryMetricExporter, name: &str) -> Option<u64> {
        let metrics = exporter.get_finished_metrics().ok()?;
        for rm in &metrics {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == name {
                        if let AggregatedMetrics::U64(MetricData::Sum(sum)) = m.data() {
                            return sum.data_points().next().map(|dp| dp.value());
                        }
                    }
                }
            }
        }
        None
    }

    #[test]
    fn health_counter_does_not_panic_on_all_variants() {
        use nv_core::error::{MediaError, StageError};

        let provider = SdkMeterProvider::builder().build();
        let meter = provider.meter("test");
        let counters = HealthCounters::new(&meter);

        let feed = FeedId::new(1);
        let stage = StageId("det");
        let proc = StageId("yolo");

        let make_stage_err = || StageError::ProcessingFailed { stage_id: StageId("det"), detail: "err".into() };

        // Fire every variant — none should panic.
        let events: Vec<HealthEvent> = vec![
            HealthEvent::SourceConnected { feed_id: feed },
            HealthEvent::SourceDisconnected { feed_id: feed, reason: MediaError::Timeout },
            HealthEvent::SourceReconnecting { feed_id: feed, attempt: 1 },
            HealthEvent::StageError { feed_id: feed, stage_id: stage, error: make_stage_err() },
            HealthEvent::StagePanic { feed_id: feed, stage_id: stage },
            HealthEvent::FeedRestarting { feed_id: feed, restart_count: 1 },
            HealthEvent::FeedStopped { feed_id: feed, reason: nv_core::health::StopReason::UserRequested },
            HealthEvent::BackpressureDrop { feed_id: feed, frames_dropped: 5 },
            HealthEvent::ViewEpochChanged { feed_id: feed, epoch: 2 },
            HealthEvent::ViewDegraded { feed_id: feed, stability_score: 0.5 },
            HealthEvent::ViewCompensationApplied { feed_id: feed, epoch: 2 },
            HealthEvent::OutputLagged { messages_lost: 10 },
            HealthEvent::SourceEos { feed_id: feed },
            HealthEvent::DecodeDecision {
                feed_id: feed,
                outcome: nv_core::health::DecodeOutcome::Software,
                preference: nv_core::health::DecodePreference::Auto,
                fallback_active: false,
                fallback_reason: None,
                detail: "test".into(),
            },
            HealthEvent::SinkPanic { feed_id: feed },
            HealthEvent::SinkTimeout { feed_id: feed },
            HealthEvent::SinkBackpressure { feed_id: feed, outputs_dropped: 3 },
            HealthEvent::TrackAdmissionRejected { feed_id: feed, rejected_count: 2 },
            HealthEvent::BatchError { processor_id: proc, batch_size: 4, error: make_stage_err() },
            HealthEvent::BatchSubmissionRejected { feed_id: feed, processor_id: proc, dropped_count: 1 },
            HealthEvent::BatchTimeout { feed_id: feed, processor_id: proc, timed_out_count: 1 },
            HealthEvent::BatchInFlightExceeded { feed_id: feed, processor_id: proc, rejected_count: 1 },
        ];

        for event in &events {
            counters.record(event);
        }
    }

    #[test]
    fn health_counter_increments() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let counters = HealthCounters::new(&meter);

        let feed = FeedId::new(1);

        counters.record(&HealthEvent::BackpressureDrop { feed_id: feed, frames_dropped: 3 });
        counters.record(&HealthEvent::BackpressureDrop { feed_id: feed, frames_dropped: 7 });

        drop(counters);
        let _ = provider.force_flush();

        // 3 + 7 = 10
        assert_eq!(find_u64_sum(&exporter, "nv.health.backpressure_drops"), Some(10));
    }

    #[test]
    fn health_counter_output_lag_accumulates() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let counters = HealthCounters::new(&meter);

        counters.record(&HealthEvent::OutputLagged { messages_lost: 5 });
        counters.record(&HealthEvent::OutputLagged { messages_lost: 15 });

        drop(counters);
        let _ = provider.force_flush();

        assert_eq!(find_u64_sum(&exporter, "nv.health.output_lag_events"), Some(20));
    }

    #[test]
    fn health_counter_stage_error_carries_attributes() {
        use nv_core::error::StageError;

        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let counters = HealthCounters::new(&meter);

        counters.record(&HealthEvent::StageError {
            feed_id: FeedId::new(42),
            stage_id: StageId("detector"),
            error: StageError::ProcessingFailed { stage_id: StageId("detector"), detail: "test".into() },
        });

        drop(counters);
        let _ = provider.force_flush();

        let metrics = exporter.get_finished_metrics().unwrap();
        let mut found_feed = false;
        let mut found_stage = false;
        for rm in &metrics {
            for sm in rm.scope_metrics() {
                for m in sm.metrics() {
                    if m.name() == "nv.health.stage_errors" {
                        if let AggregatedMetrics::U64(MetricData::Sum(sum)) = m.data() {
                            for dp in sum.data_points() {
                                for attr in dp.attributes() {
                                    if attr.key.as_str() == "feed_id" {
                                        found_feed = true;
                                    }
                                    if attr.key.as_str() == "stage_id" {
                                        found_stage = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found_feed, "stage_errors counter must carry feed_id");
        assert!(found_stage, "stage_errors counter must carry stage_id");
    }

    #[test]
    fn health_counter_ignored_events_do_not_emit() {
        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let counters = HealthCounters::new(&meter);

        // These variants are intentionally not counted.
        counters.record(&HealthEvent::SourceConnected { feed_id: FeedId::new(1) });
        counters.record(&HealthEvent::SourceEos { feed_id: FeedId::new(1) });

        drop(counters);
        let _ = provider.force_flush();

        let names = metric_names(&exporter);
        // No counter instruments should have been emitted for
        // informational-only events.
        assert!(names.is_empty(), "informational events should not produce metrics, got: {names:?}");
    }

    #[test]
    fn health_counter_all_names_present() {
        use nv_core::error::{MediaError, StageError};

        let (provider, exporter) = in_memory_provider();
        let meter = provider.meter("test");
        let counters = HealthCounters::new(&meter);

        let feed = FeedId::new(1);
        let stage = StageId("s");
        let proc = StageId("p");
        let make_err = || StageError::ProcessingFailed { stage_id: StageId("s"), detail: "e".into() };

        // Fire one of each counted variant.
        counters.record(&HealthEvent::SourceDisconnected { feed_id: feed, reason: MediaError::Timeout });
        counters.record(&HealthEvent::SourceReconnecting { feed_id: feed, attempt: 1 });
        counters.record(&HealthEvent::StageError { feed_id: feed, stage_id: stage, error: make_err() });
        counters.record(&HealthEvent::StagePanic { feed_id: feed, stage_id: stage });
        counters.record(&HealthEvent::FeedRestarting { feed_id: feed, restart_count: 1 });
        counters.record(&HealthEvent::FeedStopped { feed_id: feed, reason: nv_core::health::StopReason::UserRequested });
        counters.record(&HealthEvent::BackpressureDrop { feed_id: feed, frames_dropped: 1 });
        counters.record(&HealthEvent::ViewEpochChanged { feed_id: feed, epoch: 1 });
        counters.record(&HealthEvent::ViewDegraded { feed_id: feed, stability_score: 0.5 });
        counters.record(&HealthEvent::OutputLagged { messages_lost: 1 });
        counters.record(&HealthEvent::SinkPanic { feed_id: feed });
        counters.record(&HealthEvent::SinkTimeout { feed_id: feed });
        counters.record(&HealthEvent::SinkBackpressure { feed_id: feed, outputs_dropped: 1 });
        counters.record(&HealthEvent::TrackAdmissionRejected { feed_id: feed, rejected_count: 1 });
        counters.record(&HealthEvent::BatchError { processor_id: proc, batch_size: 4, error: make_err() });
        counters.record(&HealthEvent::BatchSubmissionRejected { feed_id: feed, processor_id: proc, dropped_count: 1 });
        counters.record(&HealthEvent::BatchTimeout { feed_id: feed, processor_id: proc, timed_out_count: 1 });
        counters.record(&HealthEvent::BatchInFlightExceeded { feed_id: feed, processor_id: proc, rejected_count: 1 });

        drop(counters);
        let _ = provider.force_flush();

        let names = metric_names(&exporter);

        let expected = [
            "nv.health.source_disconnections",
            "nv.health.source_reconnections",
            "nv.health.stage_errors",
            "nv.health.stage_panics",
            "nv.health.feed_restarts",
            "nv.health.feeds_stopped",
            "nv.health.backpressure_drops",
            "nv.health.view_epoch_changes",
            "nv.health.view_degradations",
            "nv.health.output_lag_events",
            "nv.health.sink_panics",
            "nv.health.sink_timeouts",
            "nv.health.sink_backpressure",
            "nv.health.track_admission_rejected",
            "nv.health.batch_errors",
            "nv.health.batch_submission_rejected",
            "nv.health.batch_timeouts",
            "nv.health.batch_in_flight_exceeded",
        ];

        for name in expected {
            assert!(names.contains(&name.to_owned()), "missing counter: {name}");
        }
    }
}
