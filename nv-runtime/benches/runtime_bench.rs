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

criterion_group!(
    benches,
    output_envelope_construction,
    output_arc_clone,
    broadcast_fanout,
    batch_channel_alloc,
);
criterion_main!(benches);
