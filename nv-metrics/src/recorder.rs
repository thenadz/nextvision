//! Maps [`RuntimeDiagnostics`] snapshots to OpenTelemetry instrument recordings.
//!
//! All instruments are created once at startup. Each poll cycle calls
//! [`Instruments::record()`] with a fresh snapshot, recording current
//! values as gauges. Downstream systems (Prometheus, Grafana) derive
//! rates from gauge deltas where appropriate.

use nv_runtime::diagnostics::{RuntimeDiagnostics, ViewStatus};
use opentelemetry::metrics::{Gauge, Meter};
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
}
