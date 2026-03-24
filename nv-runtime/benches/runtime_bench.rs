//! Benchmarks for key runtime hot paths: queue operations and output fanout.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;

use nv_core::{FeedId, MonotonicTs, TypedMetadata};

fn output_envelope_construction(c: &mut Criterion) {
    use nv_core::timestamp::WallTs;
    use nv_perception::DetectionSet;
    use nv_runtime::{OutputEnvelope, Provenance, ViewProvenance};
    use nv_view::view_state::{ViewEpoch, ViewState, ViewVersion};
    use nv_view::{MotionSource, TransitionPhase};

    c.bench_function("output_envelope_construct", |b| {
        b.iter(|| {
            black_box(OutputEnvelope {
                feed_id: FeedId::new(1),
                frame_seq: 0,
                ts: MonotonicTs::from_nanos(0),
                wall_ts: WallTs::from_micros(0),
                detections: DetectionSet::empty(),
                tracks: Vec::new(),
                signals: Vec::new(),
                scene_features: Vec::new(),
                view: ViewState::fixed_initial(),
                provenance: Provenance {
                    stages: Vec::new(),
                    view_provenance: ViewProvenance {
                        motion_source: MotionSource::None,
                        epoch_decision: None,
                        transition: TransitionPhase::Settled,
                        stability_score: 1.0,
                        epoch: ViewEpoch::INITIAL,
                        version: ViewVersion::INITIAL,
                    },
                    frame_receive_ts: MonotonicTs::from_nanos(0),
                    pipeline_complete_ts: MonotonicTs::from_nanos(0),
                    total_latency: nv_core::Duration::from_nanos(0),
                    frame_included: false,
                },
                metadata: TypedMetadata::new(),
                frame: None,
                admission: nv_runtime::AdmissionSummary::default(),
            });
        });
    });
}

fn output_arc_clone(c: &mut Criterion) {
    use nv_core::timestamp::WallTs;
    use nv_perception::DetectionSet;
    use nv_runtime::{OutputEnvelope, Provenance, SharedOutput, ViewProvenance};
    use nv_view::view_state::{ViewEpoch, ViewState, ViewVersion};
    use nv_view::{MotionSource, TransitionPhase};

    let shared: SharedOutput = Arc::new(OutputEnvelope {
        feed_id: FeedId::new(1),
        frame_seq: 0,
        ts: MonotonicTs::from_nanos(0),
        wall_ts: WallTs::from_micros(0),
        detections: DetectionSet::empty(),
        tracks: Vec::new(),
        signals: Vec::new(),
        scene_features: Vec::new(),
        view: ViewState::fixed_initial(),
        provenance: Provenance {
            stages: Vec::new(),
            view_provenance: ViewProvenance {
                motion_source: MotionSource::None,
                epoch_decision: None,
                transition: TransitionPhase::Settled,
                stability_score: 1.0,
                epoch: ViewEpoch::INITIAL,
                version: ViewVersion::INITIAL,
            },
            frame_receive_ts: MonotonicTs::from_nanos(0),
            pipeline_complete_ts: MonotonicTs::from_nanos(0),
            total_latency: nv_core::Duration::from_nanos(0),
            frame_included: false,
        },
        metadata: TypedMetadata::new(),
        frame: None,
        admission: nv_runtime::AdmissionSummary::default(),
    });

    c.bench_function("shared_output_arc_clone", |b| {
        b.iter(|| {
            black_box(shared.clone());
        });
    });
}

fn broadcast_fanout(c: &mut Criterion) {
    use nv_core::timestamp::WallTs;
    use nv_perception::DetectionSet;
    use nv_runtime::{OutputEnvelope, Provenance, SharedOutput, ViewProvenance};
    use nv_view::view_state::{ViewEpoch, ViewState, ViewVersion};
    use nv_view::{MotionSource, TransitionPhase};
    use tokio::sync::broadcast;

    let (tx, _rx1) = broadcast::channel::<SharedOutput>(64);
    let _rx2 = tx.subscribe();
    let _rx3 = tx.subscribe();

    let shared: SharedOutput = Arc::new(OutputEnvelope {
        feed_id: FeedId::new(1),
        frame_seq: 0,
        ts: MonotonicTs::from_nanos(0),
        wall_ts: WallTs::from_micros(0),
        detections: DetectionSet::empty(),
        tracks: Vec::new(),
        signals: Vec::new(),
        scene_features: Vec::new(),
        view: ViewState::fixed_initial(),
        provenance: Provenance {
            stages: Vec::new(),
            view_provenance: ViewProvenance {
                motion_source: MotionSource::None,
                epoch_decision: None,
                transition: TransitionPhase::Settled,
                stability_score: 1.0,
                epoch: ViewEpoch::INITIAL,
                version: ViewVersion::INITIAL,
            },
            frame_receive_ts: MonotonicTs::from_nanos(0),
            pipeline_complete_ts: MonotonicTs::from_nanos(0),
            total_latency: nv_core::Duration::from_nanos(0),
            frame_included: false,
        },
        metadata: TypedMetadata::new(),
        frame: None,
        admission: nv_runtime::AdmissionSummary::default(),
    });

    c.bench_function("broadcast_send_3_subscribers", |b| {
        b.iter(|| {
            let _ = black_box(tx.send(shared.clone()));
        });
    });
}

fn batch_channel_alloc(c: &mut Criterion) {
    use nv_core::error::StageError;
    use nv_perception::StageOutput;

    // Measure the per-submit sync_channel(1) allocation + roundtrip
    // that the batch hot path pays for each item. This is the minimum
    // overhead added by the batch coordination layer.
    c.bench_function("batch_response_channel_alloc", |b| {
        b.iter(|| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Result<StageOutput, StageError>>(1);
            let _ = tx.send(Ok(StageOutput::empty()));
            let _ = black_box(rx.recv());
        });
    });
}

fn batch_metrics_snapshot(c: &mut Criterion) {
    use nv_core::error::StageError;
    use nv_core::id::StageId;
    use nv_perception::StageOutput;
    use nv_perception::batch::{BatchEntry, BatchProcessor};
    use nv_runtime::BatchConfig;
    use std::time::Duration;

    struct NoopProcessor;
    impl BatchProcessor for NoopProcessor {
        fn id(&self) -> StageId {
            StageId("bench_metrics")
        }
        fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), StageError> {
            for item in items.iter_mut() {
                item.output = Some(StageOutput::empty());
            }
            Ok(())
        }
    }

    let runtime = nv_runtime::Runtime::builder().build().unwrap();
    let handle = runtime
        .create_batch(
            Box::new(NoopProcessor),
            BatchConfig {
                max_batch_size: 4,
                max_latency: Duration::from_millis(50),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
                startup_timeout: None,
            },
        )
        .unwrap();

    // Measure cost of snapshotting atomic metrics — this is called
    // from dashboards and health reporters.
    c.bench_function("batch_metrics_snapshot", |b| {
        b.iter(|| {
            black_box(handle.metrics());
        });
    });

    runtime.shutdown().ok();
}

fn batch_metrics_display(c: &mut Criterion) {
    use nv_runtime::BatchMetrics;

    let m = BatchMetrics {
        batches_dispatched: 1000,
        items_processed: 4000,
        items_submitted: 4200,
        items_rejected: 150,
        items_timed_out: 5,
        total_processing_ns: 500_000_000,
        total_formation_ns: 200_000_000,
        min_batch_size: 2,
        max_batch_size_seen: 4,
        configured_max_batch_size: 4,
        consecutive_errors: 0,
    };

    // Measure formatting cost — Display is used for diagnostic logging.
    c.bench_function("batch_metrics_display", |b| {
        b.iter(|| {
            black_box(format!("{m}"));
        });
    });
}

fn batch_entry_construction(c: &mut Criterion) {
    use nv_core::id::FeedId;
    use nv_core::timestamp::MonotonicTs;
    use nv_perception::batch::BatchEntry;
    use nv_view::ViewSnapshot;

    let frame = nv_test_util::synthetic::solid_gray(
        FeedId::new(1),
        0,
        MonotonicTs::from_nanos(0),
        640,
        480,
        128,
    );
    let view = ViewSnapshot::new(nv_view::ViewState::fixed_initial());

    // Measure per-item BatchEntry construction cost (Arc clone of
    // frame + ViewSnapshot clone). This runs once per frame per feed.
    c.bench_function("batch_entry_construct", |b| {
        b.iter(|| {
            black_box(BatchEntry {
                feed_id: FeedId::new(1),
                frame: frame.clone(),
                view: view.clone(),
                output: None,
            });
        });
    });
}

fn batch_channel_contention(c: &mut Criterion) {
    use nv_core::error::StageError;
    use nv_perception::StageOutput;

    // Measure try_send contention with N producer threads hammering a
    // bounded sync_channel. Threads are pre-spawned and synchronized
    // with barriers so only send contention is measured, not thread
    // lifecycle overhead.
    let num_producers = 4;
    let sends_per_iter = 10;
    let queue_depth = 16;
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<StageOutput, StageError>>(queue_depth);

    // Drain thread: pull items as fast as possible.
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop_flag);
    let drain_thread = std::thread::spawn(move || {
        while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = rx.recv_timeout(std::time::Duration::from_millis(1));
        }
        while rx.try_recv().is_ok() {}
    });

    // Pre-spawn producer threads. Each waits on a start barrier, sends
    // `sends_per_iter` items, then signals an end barrier.
    let start_barrier = Arc::new(std::sync::Barrier::new(num_producers + 1));
    let end_barrier = Arc::new(std::sync::Barrier::new(num_producers + 1));
    let producer_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut _producer_handles = Vec::with_capacity(num_producers);
    for _ in 0..num_producers {
        let t = tx.clone();
        let sb = Arc::clone(&start_barrier);
        let eb = Arc::clone(&end_barrier);
        let ps = Arc::clone(&producer_stop);
        _producer_handles.push(std::thread::spawn(move || {
            loop {
                sb.wait();
                if ps.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                for _ in 0..sends_per_iter {
                    let _ = t.try_send(Ok(StageOutput::empty()));
                }
                eb.wait();
            }
        }));
    }

    c.bench_function("batch_try_send_4_thread_contention", |b| {
        b.iter(|| {
            // Release producers.
            start_barrier.wait();
            // Wait for all producers to finish their sends.
            end_barrier.wait();
        });
    });

    // Shut down producers.
    producer_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    start_barrier.wait(); // release them one last time so they see the stop flag
    for h in _producer_handles {
        let _ = h.join();
    }
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    drop(tx);
    let _ = drain_thread.join();
}

fn batch_rejection_cost(c: &mut Criterion) {
    use nv_core::error::StageError;
    use nv_perception::StageOutput;

    // Measure the cost of a rejected try_send (queue full). This is
    // the fast-path overhead each feed pays when backpressure fires.
    let (tx, _rx) = std::sync::mpsc::sync_channel::<Result<StageOutput, StageError>>(1);

    // Fill the single-slot queue so every try_send fails.
    let _ = tx.try_send(Ok(StageOutput::empty()));

    c.bench_function("batch_rejected_try_send", |b| {
        b.iter(|| {
            let result = tx.try_send(Ok(StageOutput::empty()));
            black_box(result.is_err());
        });
    });
}

criterion_group!(
    benches,
    output_envelope_construction,
    output_arc_clone,
    broadcast_fanout,
    batch_channel_alloc,
    batch_metrics_snapshot,
    batch_metrics_display,
    batch_entry_construction,
    batch_channel_contention,
    batch_rejection_cost,
);
criterion_main!(benches);
