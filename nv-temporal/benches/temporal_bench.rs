//! Benchmarks for temporal store snapshot operations.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;

use nv_core::{MonotonicTs, TrackId};
use nv_perception::track::{Track, TrackObservation, TrackState};
use nv_temporal::retention::RetentionPolicy;
use nv_temporal::store::{TemporalStore, TrackHistory};
use nv_temporal::trajectory::Trajectory;
use nv_view::ViewEpoch;

fn make_observation(ts_nanos: u64) -> TrackObservation {
    TrackObservation {
        ts: MonotonicTs::from_nanos(ts_nanos),
        bbox: nv_core::geom::BBox::new(0.1, 0.2, 0.3, 0.4),
        confidence: 0.9,
        state: TrackState::Confirmed,
        detection_id: None,
    }
}

fn make_track(id: u64) -> (TrackId, TrackHistory) {
    let track_id = TrackId::new(id);
    let obs = make_observation(id * 33_333_333);
    let track = Arc::new(Track {
        id: track_id,
        class_id: 0,
        state: TrackState::Confirmed,
        current: obs.clone(),
        metadata: nv_core::TypedMetadata::new(),
    });
    let trajectory = Arc::new(Trajectory::new());
    let ts = MonotonicTs::from_nanos(id * 33_333_333);
    let mut history = TrackHistory::new(track, trajectory, ts, ts, ViewEpoch::INITIAL);

    // Add some observations.
    for i in 0..10 {
        history.push_observation(make_observation((id * 10 + i) * 33_333_333));
    }

    (track_id, history)
}

fn snapshot_10_tracks(c: &mut Criterion) {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    for i in 0..10 {
        let (id, history) = make_track(i);
        store.insert_track(id, history);
    }

    c.bench_function("temporal_snapshot_10_tracks", |b| {
        b.iter(|| {
            black_box(store.snapshot());
        });
    });
}

fn snapshot_100_tracks(c: &mut Criterion) {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    for i in 0..100 {
        let (id, history) = make_track(i);
        store.insert_track(id, history);
    }

    c.bench_function("temporal_snapshot_100_tracks", |b| {
        b.iter(|| {
            black_box(store.snapshot());
        });
    });
}

fn snapshot_cached_no_mutations(c: &mut Criterion) {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    for i in 0..50 {
        let (id, history) = make_track(i);
        store.insert_track(id, history);
    }
    // Prime the cache.
    let _ = store.snapshot();

    c.bench_function("temporal_snapshot_50_tracks_cached", |b| {
        b.iter(|| {
            black_box(store.snapshot());
        });
    });
}

criterion_group!(
    benches,
    snapshot_10_tracks,
    snapshot_100_tracks,
    snapshot_cached_no_mutations
);
criterion_main!(benches);
