use super::*;
use crate::continuity::SegmentBoundary;
use nv_core::geom::BBox;
use nv_perception::TrackState;

fn make_observation(ts_ns: u64) -> TrackObservation {
    TrackObservation {
        ts: MonotonicTs::from_nanos(ts_ns),
        bbox: BBox::new(0.0, 0.0, 0.1, 0.1),
        confidence: 0.9,
        state: TrackState::Confirmed,
        detection_id: None,
        metadata: nv_core::TypedMetadata::new(),
    }
}

fn make_track(id: u64) -> Track {
    Track {
        id: TrackId::new(id),
        class_id: 0,
        state: TrackState::Confirmed,
        current: make_observation(0),
        metadata: nv_core::TypedMetadata::new(),
    }
}

/// Insert a track into the store for testing.
fn insert_track(store: &mut TemporalStore, track_id: u64, obs_count: usize) {
    let id = TrackId::new(track_id);
    let mut history = TrackHistory::new(
        Arc::new(make_track(track_id)),
        Arc::new(Trajectory::new()),
        MonotonicTs::from_nanos(0),
        MonotonicTs::from_nanos((obs_count as u64).saturating_sub(1) * 1_000_000),
        ViewEpoch::INITIAL,
    );
    for i in 0..obs_count {
        history.push_observation(make_observation((i as u64) * 1_000_000));
    }
    store.tracks.insert(id, history);
}

/// Insert a track with a specific state directly (bypasses admission control).
fn insert_track_with_state(
    store: &mut TemporalStore,
    track_id: u64,
    state: TrackState,
    last_seen_ns: u64,
) {
    let id = TrackId::new(track_id);
    let track = Arc::new(make_track_with_state(track_id, state));
    let history = TrackHistory::new(
        track,
        Arc::new(Trajectory::new()),
        MonotonicTs::from_nanos(0),
        MonotonicTs::from_nanos(last_seen_ns),
        ViewEpoch::INITIAL,
    );
    store.tracks.insert(id, history);
}

#[test]
fn snapshot_shares_track_arc() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 3);

    let snap = store.snapshot();
    let id = TrackId::new(1);
    let live = store.get_track(&id).unwrap();
    let snapped = snap.get_track(&id).unwrap();

    // Track Arc should be the same allocation.
    assert!(Arc::ptr_eq(&live.track, &snapped.track));
    // Trajectory Arc should be the same allocation.
    assert!(Arc::ptr_eq(&live.trajectory, &snapped.trajectory));
}

#[test]
fn snapshot_reuses_observations_when_unchanged() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 5);

    let snap1 = store.snapshot();
    let snap2 = store.snapshot();

    let id = TrackId::new(1);
    let obs1 = &snap1.get_track(&id).unwrap().observations;
    let obs2 = &snap2.get_track(&id).unwrap().observations;

    // The two snapshots should share the exact same observations Arc
    // since no mutations happened between them.
    assert!(
        Arc::ptr_eq(obs1, obs2),
        "unchanged observations should share Arc allocation"
    );
}

#[test]
fn snapshot_rebuilds_observations_after_mutation() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 3);

    let snap1 = store.snapshot();

    // Mutate observations through the encapsulated API — generation is
    // bumped automatically, no manual `mark_observations_dirty()` needed.
    let id = TrackId::new(1);
    let history = store.tracks.get_mut(&id).unwrap();
    history.push_observation(make_observation(99_000_000));

    let snap2 = store.snapshot();

    let obs1 = &snap1.get_track(&id).unwrap().observations;
    let obs2 = &snap2.get_track(&id).unwrap().observations;

    assert_ne!(
        obs1.len(),
        obs2.len(),
        "snap2 should have the new observation"
    );
    assert!(
        !Arc::ptr_eq(obs1, obs2),
        "changed observations should have new Arc"
    );
    assert_eq!(obs2.len(), 4);
}

#[test]
fn snapshot_clone_is_arc_bump() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 2);

    let snap = store.snapshot();
    let snap_clone = snap.clone();

    // Both should point to the same inner allocation.
    assert!(Arc::ptr_eq(&snap.inner, &snap_clone.inner));
}

#[test]
fn empty_store_snapshot() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let snap = store.snapshot();
    assert_eq!(snap.track_count(), 0);
}

#[test]
fn pop_front_invalidates_cache() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 3);

    let snap1 = store.snapshot();
    let id = TrackId::new(1);

    let history = store.tracks.get_mut(&id).unwrap();
    let popped = history.pop_observation_front();
    assert!(popped.is_some());

    let snap2 = store.snapshot();
    let obs1 = &snap1.get_track(&id).unwrap().observations;
    let obs2 = &snap2.get_track(&id).unwrap().observations;
    assert!(!Arc::ptr_eq(obs1, obs2), "pop_front must invalidate cache");
    assert_eq!(obs2.len(), 2);
}

#[test]
fn clear_observations_invalidates_cache() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 3);

    let snap1 = store.snapshot();
    let id = TrackId::new(1);

    let history = store.tracks.get_mut(&id).unwrap();
    history.clear_observations();

    let snap2 = store.snapshot();
    let obs1 = &snap1.get_track(&id).unwrap().observations;
    let obs2 = &snap2.get_track(&id).unwrap().observations;
    assert!(!Arc::ptr_eq(obs1, obs2));
    assert_eq!(obs2.len(), 0);
}

#[test]
fn clear_empty_observations_does_not_bump_gen() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 1, 0);

    let snap1 = store.snapshot();
    let id = TrackId::new(1);

    let history = store.tracks.get_mut(&id).unwrap();
    history.clear_observations(); // no-op on empty

    let snap2 = store.snapshot();
    let obs1 = &snap1.get_track(&id).unwrap().observations;
    let obs2 = &snap2.get_track(&id).unwrap().observations;
    // Already empty, so the gen should not have bumped → same Arc.
    assert!(Arc::ptr_eq(obs1, obs2));
}

#[test]
fn observation_count_reflects_mutations() {
    let mut history = TrackHistory::new(
        Arc::new(make_track(1)),
        Arc::new(Trajectory::new()),
        MonotonicTs::from_nanos(0),
        MonotonicTs::from_nanos(0),
        ViewEpoch::INITIAL,
    );
    assert_eq!(history.observation_count(), 0);
    history.push_observation(make_observation(1_000_000));
    assert_eq!(history.observation_count(), 1);
    history.push_observation(make_observation(2_000_000));
    assert_eq!(history.observation_count(), 2);
    history.pop_observation_front();
    assert_eq!(history.observation_count(), 1);
    history.clear_observations();
    assert_eq!(history.observation_count(), 0);
}

#[test]
fn remove_track_returns_history() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    insert_track(&mut store, 42, 3);
    assert_eq!(store.track_count(), 1);

    let removed = store.remove_track(&TrackId::new(42));
    assert!(removed.is_some());
    assert_eq!(store.track_count(), 0);

    // Removing again yields None.
    assert!(store.remove_track(&TrackId::new(42)).is_none());
}

// ------------------------------------------------------------------
// Track lifecycle tests
// ------------------------------------------------------------------

fn make_track_with_state(id: u64, state: TrackState) -> Track {
    Track {
        id: TrackId::new(id),
        class_id: 0,
        state,
        current: make_observation(0),
        metadata: nv_core::TypedMetadata::new(),
    }
}

#[test]
fn commit_track_creates_new_history() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let ts = MonotonicTs::from_nanos(1_000_000);
    let epoch = ViewEpoch::INITIAL;

    store.commit_track(&track, ts, epoch);

    assert_eq!(store.track_count(), 1);
    let history = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(history.first_seen, ts);
    assert_eq!(history.last_seen, ts);
    assert_eq!(history.observation_count(), 1);
    assert_eq!(history.trajectory.segment_count(), 1);
    assert_eq!(history.trajectory.total_points(), 1);
}

#[test]
fn commit_track_updates_existing_history() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);
    store.commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch);
    store.commit_track(&track, MonotonicTs::from_nanos(3_000_000), epoch);

    let history = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(history.observation_count(), 3);
    assert_eq!(history.trajectory.total_points(), 3);
    assert_eq!(history.last_seen, MonotonicTs::from_nanos(3_000_000));
    // first_seen should remain unchanged.
    assert_eq!(history.first_seen, MonotonicTs::from_nanos(1_000_000));
}

#[test]
fn commit_track_enforces_observation_cap() {
    let retention = RetentionPolicy {
        max_observations_per_track: 3,
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    for i in 0..10u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    let history = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(
        history.observation_count(),
        3,
        "observations should be capped at max_observations_per_track"
    );
}

#[test]
fn commit_track_respects_concurrent_cap_with_eviction() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 3,
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    // Fill to cap with Lost tracks.
    for i in 1..=3u64 {
        let track = make_track_with_state(i, TrackState::Lost);
        assert!(store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch));
    }
    assert_eq!(store.track_count(), 3);

    // A 4th track should evict the oldest Lost track (id=1).
    let new_track = make_track(4);
    assert!(store.commit_track(&new_track, MonotonicTs::from_nanos(4_000_000), epoch));
    assert_eq!(store.track_count(), 3, "cap must hold after commit");
    assert!(
        store.get_track(&TrackId::new(1)).is_none(),
        "oldest Lost evicted"
    );
    assert!(
        store.get_track(&TrackId::new(4)).is_some(),
        "new track admitted"
    );
}

#[test]
fn commit_track_rejects_when_only_confirmed_at_cap() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 2,
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    // Fill to cap with Confirmed tracks (non-evictable).
    for i in 1..=2u64 {
        let track = make_track_with_state(i, TrackState::Confirmed);
        assert!(store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch));
    }
    assert_eq!(store.track_count(), 2);

    // A 3rd track should be rejected — no eviction victim.
    let new_track = make_track(3);
    assert!(!store.commit_track(&new_track, MonotonicTs::from_nanos(3_000_000), epoch));
    assert_eq!(store.track_count(), 2, "cap must hold");
    assert!(store.get_track(&TrackId::new(3)).is_none());
}

#[test]
fn track_lifecycle_tentative_to_confirmed_to_lost() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    // Tentative track.
    let tentative = make_track_with_state(1, TrackState::Tentative);
    store.commit_track(&tentative, MonotonicTs::from_nanos(1_000_000), epoch);
    assert_eq!(
        store.get_track(&TrackId::new(1)).unwrap().track.state,
        TrackState::Tentative
    );

    // Confirmed.
    let confirmed = make_track_with_state(1, TrackState::Confirmed);
    store.commit_track(&confirmed, MonotonicTs::from_nanos(2_000_000), epoch);
    assert_eq!(
        store.get_track(&TrackId::new(1)).unwrap().track.state,
        TrackState::Confirmed
    );

    // Coasted.
    let coasted = make_track_with_state(1, TrackState::Coasted);
    store.commit_track(&coasted, MonotonicTs::from_nanos(3_000_000), epoch);
    assert_eq!(
        store.get_track(&TrackId::new(1)).unwrap().track.state,
        TrackState::Coasted
    );

    // Lost.
    let lost = make_track_with_state(1, TrackState::Lost);
    store.commit_track(&lost, MonotonicTs::from_nanos(4_000_000), epoch);
    assert_eq!(
        store.get_track(&TrackId::new(1)).unwrap().track.state,
        TrackState::Lost
    );
}

// ------------------------------------------------------------------
// Retention / state-pruning tests
// ------------------------------------------------------------------

#[test]
fn enforce_retention_evicts_old_lost_tracks() {
    let retention = RetentionPolicy {
        max_track_age: nv_core::Duration::from_secs(10),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    let lost = make_track_with_state(1, TrackState::Lost);
    store.commit_track(&lost, MonotonicTs::from_nanos(0), epoch);
    assert_eq!(store.track_count(), 1);

    // Time = 11 seconds → beyond max_track_age.
    store.enforce_retention(MonotonicTs::from_nanos(11_000_000_000));
    assert_eq!(store.track_count(), 0, "old Lost track should be evicted");
}

#[test]
fn enforce_retention_keeps_recently_lost_tracks() {
    let retention = RetentionPolicy {
        max_track_age: nv_core::Duration::from_secs(10),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    let lost = make_track_with_state(1, TrackState::Lost);
    store.commit_track(&lost, MonotonicTs::from_nanos(5_000_000_000), epoch);

    // Time = 8 seconds → within max_track_age of last_seen (5s).
    store.enforce_retention(MonotonicTs::from_nanos(8_000_000_000));
    assert_eq!(store.track_count(), 1, "recent Lost track should be kept");
}

#[test]
fn enforce_retention_does_not_evict_confirmed_tracks() {
    let retention = RetentionPolicy {
        max_track_age: nv_core::Duration::from_secs(1),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    let confirmed = make_track_with_state(1, TrackState::Confirmed);
    store.commit_track(&confirmed, MonotonicTs::from_nanos(0), epoch);

    // Even at max age, Confirmed tracks are not evicted.
    store.enforce_retention(MonotonicTs::from_nanos(10_000_000_000));
    assert_eq!(
        store.track_count(),
        1,
        "Confirmed tracks should never be age-evicted"
    );
}

#[test]
fn enforce_retention_hard_cap_evicts_oldest_lost_first() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 3,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);

    // Insert directly to bypass pre-eviction — testing enforce_retention.
    for i in 1..=5u64 {
        insert_track_with_state(&mut store, i, TrackState::Lost, i * 1_000_000);
    }
    assert_eq!(store.track_count(), 5);

    store.enforce_retention(MonotonicTs::from_nanos(10_000_000));
    assert_eq!(store.track_count(), 3);

    // Oldest (track 1, 2) should be gone; newest (3, 4, 5) remain.
    assert!(store.get_track(&TrackId::new(1)).is_none());
    assert!(store.get_track(&TrackId::new(2)).is_none());
    assert!(store.get_track(&TrackId::new(3)).is_some());
}

#[test]
fn enforce_retention_hard_cap_spares_confirmed_and_tentative() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 2,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);

    // Insert directly to bypass admission control — we're testing
    // enforce_retention eviction, not commit_track admission.
    // 3 Confirmed tracks + 1 Lost.
    for i in 1..=3u64 {
        insert_track_with_state(&mut store, i, TrackState::Confirmed, i * 1_000_000);
    }
    insert_track_with_state(&mut store, 10, TrackState::Lost, 0);
    assert_eq!(store.track_count(), 4);

    store.enforce_retention(MonotonicTs::from_nanos(100_000_000));

    // Lost track should be evicted, but Confirmed tracks stay even though > cap.
    assert!(store.get_track(&TrackId::new(10)).is_none());
    assert_eq!(
        store.track_count(),
        3,
        "Confirmed tracks cannot be evicted by hard cap"
    );
}

#[test]
fn enforce_retention_hard_cap_evicts_tentative_after_lost_and_coasted() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 2,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);

    // Insert directly to bypass admission control.
    // 2 Confirmed + 2 Tentative — no Lost or Coasted.
    for i in 1..=2u64 {
        insert_track_with_state(&mut store, i, TrackState::Confirmed, i * 1_000_000);
    }
    for i in 3..=4u64 {
        insert_track_with_state(&mut store, i, TrackState::Tentative, i * 1_000_000);
    }
    assert_eq!(store.track_count(), 4);

    store.enforce_retention(MonotonicTs::from_nanos(100_000_000));

    // Tentative tracks should be evicted (oldest first) to reach cap.
    assert_eq!(store.track_count(), 2);
    // Oldest tentative (3) evicted first, then (4).
    assert!(store.get_track(&TrackId::new(3)).is_none());
    assert!(store.get_track(&TrackId::new(4)).is_none());
    // Confirmed remain.
    assert!(store.get_track(&TrackId::new(1)).is_some());
    assert!(store.get_track(&TrackId::new(2)).is_some());
}

#[test]
fn hard_cap_strictly_bounded_under_high_id_churn_no_evictable() {
    // Simulates rapid ID churn with only Confirmed tracks:
    // commit_track should reject new admission when at cap.
    let retention = RetentionPolicy {
        max_concurrent_tracks: 3,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    // Fill to cap with Confirmed tracks.
    for i in 1..=3u64 {
        let confirmed = make_track_with_state(i, TrackState::Confirmed);
        let accepted =
            store.commit_track(&confirmed, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        assert!(accepted, "track {i} should be accepted (under cap)");
    }

    // Try to admit new Confirmed tracks beyond cap — should be rejected.
    for i in 100..=105u64 {
        let confirmed = make_track_with_state(i, TrackState::Confirmed);
        let accepted =
            store.commit_track(&confirmed, MonotonicTs::from_nanos(i * 1_000_000), epoch);
        assert!(
            !accepted,
            "track {i} should be rejected (at cap, no evictable)"
        );
    }

    assert_eq!(store.track_count(), 3, "count must never exceed cap");

    // Updates to existing tracks are always accepted.
    let updated = make_track_with_state(1, TrackState::Confirmed);
    let accepted = store.commit_track(&updated, MonotonicTs::from_nanos(200_000_000), epoch);
    assert!(
        accepted,
        "updates to existing tracks must always be accepted"
    );
    assert_eq!(store.track_count(), 3);
}

#[test]
fn admission_allowed_when_evictable_candidate_exists_at_cap() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 2,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    // 1 Confirmed + 1 Lost = at cap.
    let confirmed = make_track_with_state(1, TrackState::Confirmed);
    store.commit_track(&confirmed, MonotonicTs::from_nanos(1_000_000), epoch);
    let lost = make_track_with_state(2, TrackState::Lost);
    store.commit_track(&lost, MonotonicTs::from_nanos(2_000_000), epoch);
    assert_eq!(store.track_count(), 2);

    // New track should be admitted (Lost is pre-evicted) at cap.
    let new_track = make_track_with_state(3, TrackState::Confirmed);
    let accepted = store.commit_track(&new_track, MonotonicTs::from_nanos(3_000_000), epoch);
    assert!(
        accepted,
        "new track should be admitted when evictable candidate exists"
    );
    // Pre-eviction removes the Lost victim during commit — count stays at cap.
    assert_eq!(store.track_count(), 2, "strict cap: count stays at max");
    assert!(
        store.get_track(&TrackId::new(2)).is_none(),
        "Lost victim should be pre-evicted"
    );
    assert!(
        store.get_track(&TrackId::new(3)).is_some(),
        "new track should be present"
    );
}

#[test]
fn strict_cap_mixed_states_burst_never_exceeds() {
    // Burst-admission of mixed-state tracks should never push
    // the store above max_concurrent_tracks.
    let cap = 4;
    let retention = RetentionPolicy {
        max_concurrent_tracks: cap,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    // Phase 1: fill to cap with 2 Confirmed + 1 Lost + 1 Tentative.
    let c1 = make_track_with_state(1, TrackState::Confirmed);
    store.commit_track(&c1, MonotonicTs::from_nanos(1_000_000), epoch);
    let c2 = make_track_with_state(2, TrackState::Confirmed);
    store.commit_track(&c2, MonotonicTs::from_nanos(2_000_000), epoch);
    let l3 = make_track_with_state(3, TrackState::Lost);
    store.commit_track(&l3, MonotonicTs::from_nanos(3_000_000), epoch);
    let t4 = make_track_with_state(4, TrackState::Tentative);
    store.commit_track(&t4, MonotonicTs::from_nanos(4_000_000), epoch);
    assert_eq!(store.track_count(), cap);

    // Phase 2: burst 10 new confirmed tracks.
    let mut admitted = 0u32;
    let mut rejected = 0u32;
    for i in 100..110u64 {
        let t = make_track_with_state(i, TrackState::Confirmed);
        if store.commit_track(&t, MonotonicTs::from_nanos(i * 1_000_000), epoch) {
            admitted += 1;
        } else {
            rejected += 1;
        }
        // Invariant: never exceeds cap.
        assert!(
            store.track_count() <= cap,
            "count {} exceeds cap {} after admitting track {i}",
            store.track_count(),
            cap,
        );
    }

    // Lost (3) + Tentative (4) = 2 evictable victims, so 2 admissions
    // succeed; remaining 8 are rejected (only Confirmed remain).
    assert_eq!(
        admitted, 2,
        "should admit exactly as many as victims available"
    );
    assert_eq!(rejected, 8);
    assert_eq!(store.track_count(), cap);
}

// ------------------------------------------------------------------
// Trajectory segmentation tests
// ------------------------------------------------------------------

#[test]
fn commit_creates_initial_trajectory_segment() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(hist.trajectory.segment_count(), 1);

    let seg = hist.trajectory.active_segment().unwrap();
    assert_eq!(seg.view_epoch, epoch);
    assert_eq!(seg.opened_by, SegmentBoundary::TrackCreated);
    assert!(seg.is_active());
    assert_eq!(seg.points.len(), 1);
}

#[test]
fn trajectory_accumulates_points_within_segment() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    for i in 0..5u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 33_333_333), epoch);
    }

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(hist.trajectory.segment_count(), 1);
    assert_eq!(hist.trajectory.total_points(), 5);

    // Motion features should be computed.
    let seg = hist.trajectory.active_segment().unwrap();
    // With all observations at the same bbox, displacement ≈ 0.
    assert!(seg.motion.is_stationary);
}

// ------------------------------------------------------------------
// Snapshot / TemporalAccess propagation tests
// ------------------------------------------------------------------

#[test]
fn snapshot_exposes_trajectory_info_via_temporal_access() {
    use nv_perception::TemporalAccess;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    for i in 0..3u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    let snap = store.snapshot();
    let id = TrackId::new(1);

    assert_eq!(snap.trajectory_point_count(&id), 3);
    assert_eq!(snap.trajectory_segment_count(&id), 1);

    // Missing track returns 0.
    let missing = TrackId::new(999);
    assert_eq!(snap.trajectory_point_count(&missing), 0);
    assert_eq!(snap.trajectory_segment_count(&missing), 0);
}

#[test]
fn snapshot_track_ids_matches_committed_tracks() {
    use nv_perception::TemporalAccess;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    for i in 1..=4u64 {
        let track = make_track(i);
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    let snap = store.snapshot();
    let mut ids: Vec<u64> = snap.track_ids().iter().map(|id| id.as_u64()).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 3, 4]);
}

// ------------------------------------------------------------------
// Provenance propagation via TemporalAccess
// ------------------------------------------------------------------

#[test]
fn snapshot_preserves_first_and_last_seen() {
    use nv_perception::TemporalAccess;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    store.commit_track(&track, MonotonicTs::from_nanos(100), epoch);
    store.commit_track(&track, MonotonicTs::from_nanos(200), epoch);
    store.commit_track(&track, MonotonicTs::from_nanos(300), epoch);

    let snap = store.snapshot();
    let id = TrackId::new(1);

    assert_eq!(snap.first_seen(&id), Some(MonotonicTs::from_nanos(100)));
    assert_eq!(snap.last_seen(&id), Some(MonotonicTs::from_nanos(300)));
}

#[test]
fn snapshot_observations_reflect_commit_order() {
    use nv_perception::TemporalAccess;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    // Create tracks with distinct observation timestamps.
    let mut track1 = make_track(1);
    track1.current.ts = MonotonicTs::from_nanos(100);
    store.commit_track(&track1, MonotonicTs::from_nanos(100), epoch);

    let mut track2 = make_track(1);
    track2.current.ts = MonotonicTs::from_nanos(200);
    store.commit_track(&track2, MonotonicTs::from_nanos(200), epoch);

    let snap = store.snapshot();
    let obs = snap.recent_observations(&TrackId::new(1));
    assert_eq!(obs.len(), 2);
    // Oldest first.
    assert!(obs[0].ts < obs[1].ts);
}

#[test]
fn clear_resets_all_state() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    for i in 1..=5u64 {
        let track = make_track(i);
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    assert_eq!(store.track_count(), 5);
    store.clear();
    assert_eq!(store.track_count(), 0);
}

// ------------------------------------------------------------------
// Epoch-transition integration tests (P1 / P2)
// ------------------------------------------------------------------

#[test]
fn commit_track_segments_trajectory_on_epoch_change() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch0 = ViewEpoch::INITIAL;
    let epoch1 = epoch0.next();

    // 3 commits in epoch 0.
    for i in 0..3u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch0);
    }
    assert_eq!(
        store
            .get_track(&TrackId::new(1))
            .unwrap()
            .trajectory
            .segment_count(),
        1
    );

    // Epoch changes — next commit should segment.
    store.set_view_epoch(epoch1);
    store.commit_track(&track, MonotonicTs::from_nanos(10_000_000), epoch1);

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(hist.trajectory.segment_count(), 2);

    // First segment should be closed with EpochChange.
    let seg0 = &hist.trajectory.segments[0];
    assert!(!seg0.is_active());
    assert_eq!(seg0.points.len(), 3);
    assert_eq!(
        seg0.closed_by,
        Some(SegmentBoundary::EpochChange {
            from_epoch: epoch0,
            to_epoch: epoch1,
        })
    );

    // Second segment should be active with EpochChange as opened_by.
    let seg1 = &hist.trajectory.segments[1];
    assert!(seg1.is_active());
    assert_eq!(
        seg1.opened_by,
        SegmentBoundary::EpochChange {
            from_epoch: epoch0,
            to_epoch: epoch1,
        }
    );
    assert_eq!(seg1.points.len(), 1);
    assert_eq!(seg1.view_epoch, epoch1);
}

#[test]
fn commit_track_segments_across_multiple_epochs() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let track = make_track(1);
    let epoch0 = ViewEpoch::INITIAL;
    let epoch1 = epoch0.next();
    let epoch2 = epoch1.next();

    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch0);
    store.commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch1);
    store.commit_track(&track, MonotonicTs::from_nanos(3_000_000), epoch2);

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(hist.trajectory.segment_count(), 3);
    assert!(!hist.trajectory.segments[0].is_active());
    assert!(!hist.trajectory.segments[1].is_active());
    assert!(hist.trajectory.segments[2].is_active());
    assert_eq!(hist.trajectory.total_points(), 3);
}

#[test]
fn commit_track_closes_segment_on_lost_state() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    // Transition to Lost.
    let lost = make_track_with_state(1, TrackState::Lost);
    store.commit_track(&lost, MonotonicTs::from_nanos(2_000_000), epoch);

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert_eq!(hist.trajectory.segment_count(), 1);
    let seg = &hist.trajectory.segments[0];
    assert!(!seg.is_active());
    assert_eq!(seg.closed_by, Some(SegmentBoundary::TrackLost));
    // Both points were appended before close.
    assert_eq!(seg.points.len(), 2);
}

#[test]
fn close_all_segments_marks_feed_restart() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    for i in 1..=3u64 {
        let track = make_track(i);
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    // All 3 tracks should have active segments.
    for (_, hist) in store.tracks() {
        assert!(hist.trajectory.active_segment().is_some());
    }

    store.close_all_segments(SegmentBoundary::FeedRestart);

    // All segments should now be closed with FeedRestart.
    for (_, hist) in store.tracks() {
        assert!(hist.trajectory.active_segment().is_none());
        let seg = hist.trajectory.segments.last().unwrap();
        assert_eq!(seg.closed_by, Some(SegmentBoundary::FeedRestart));
    }
}

// ------------------------------------------------------------------
// Trajectory retention (point pruning) tests (P1)
// ------------------------------------------------------------------

#[test]
fn commit_track_prunes_trajectory_points_per_track() {
    let retention = RetentionPolicy {
        max_trajectory_points_per_track: 5,
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    for i in 0..10u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert!(
        hist.trajectory.total_points() <= 5,
        "trajectory should be pruned to max_trajectory_points_per_track"
    );
}

#[test]
fn enforce_retention_prunes_trajectory_points() {
    let retention = RetentionPolicy {
        max_trajectory_points_per_track: 3,
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let track = make_track(1);
    let epoch = ViewEpoch::INITIAL;

    // commit_track also prunes, but enforce_retention does a second pass.
    // Build up points across epoch segments.
    let epoch1 = epoch.next();
    for i in 0..5u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }
    for i in 5..10u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch1);
    }

    store.enforce_retention(MonotonicTs::from_nanos(20_000_000));

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert!(
        hist.trajectory.total_points() <= 3,
        "trajectory points should be pruned by enforce_retention: got {}",
        hist.trajectory.total_points()
    );
}

#[test]
fn trajectory_pruning_removes_oldest_closed_segments_first() {
    let retention = RetentionPolicy {
        max_trajectory_points_per_track: 4,
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let track = make_track(1);
    let epoch0 = ViewEpoch::INITIAL;
    let epoch1 = epoch0.next();

    // 3 points in epoch 0, then 3 points in epoch 1.
    for i in 0..3u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch0);
    }
    for i in 3..6u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch1);
    }

    let hist = store.get_track(&TrackId::new(1)).unwrap();
    assert!(hist.trajectory.total_points() <= 4);
    // The remaining points should come from the most recent segment(s).
    if let Some(seg) = hist.trajectory.segments.last() {
        assert_eq!(seg.view_epoch, epoch1);
    }
}

// ------------------------------------------------------------------
// TrackStore / TrajectoryStoreAccess trait tests (P2)
// ------------------------------------------------------------------

#[test]
fn track_store_trait_on_temporal_store() {
    use super::TrackStore;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;
    let id = TrackId::new(1);

    assert!(!store.has_track(&id));

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    assert!(store.has_track(&id));
    assert_eq!(store.track_state(&id), Some(TrackState::Confirmed));
    assert_eq!(store.track_creation_epoch(&id), Some(epoch));
}

#[test]
fn track_store_trait_on_snapshot() {
    use super::TrackStore;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    let snap = store.snapshot();
    let id = TrackId::new(1);

    assert!(snap.has_track(&id));
    assert_eq!(snap.track_state(&id), Some(TrackState::Confirmed));
    assert_eq!(snap.track_creation_epoch(&id), Some(epoch));
    assert!(!snap.has_track(&TrackId::new(999)));
}

#[test]
fn trajectory_store_trait_on_temporal_store() {
    use super::TrajectoryStoreAccess;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;
    let id = TrackId::new(1);

    assert!(store.trajectory(&id).is_none());

    let track = make_track(1);
    for i in 0..3u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    let traj = store.trajectory(&id).unwrap();
    assert_eq!(traj.total_points(), 3);
    assert_eq!(traj.segment_count(), 1);
    assert!(traj.active_segment().is_some());
}

#[test]
fn trajectory_store_trait_on_snapshot() {
    use super::TrajectoryStoreAccess;

    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    for i in 0..3u64 {
        store.commit_track(&track, MonotonicTs::from_nanos(i * 1_000_000), epoch);
    }

    let snap = store.snapshot();
    let id = TrackId::new(1);

    let traj = TrajectoryStoreAccess::trajectory(&snap, &id).unwrap();
    assert_eq!(traj.total_points(), 3);
    assert_eq!(traj.segment_count(), 1);
    assert!(TrajectoryStoreAccess::trajectory(&snap, &TrackId::new(999)).is_none());
}

// ------------------------------------------------------------------
// P2: Hard-cap eviction priority (Lost before Coasted)
// ------------------------------------------------------------------

#[test]
fn hard_cap_evicts_lost_before_coasted() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 2,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);

    // Insert directly to bypass pre-eviction — testing enforce_retention
    // priority: Lost should be evicted before Coasted.
    insert_track_with_state(&mut store, 1, TrackState::Coasted, 1_000_000);
    insert_track_with_state(&mut store, 2, TrackState::Confirmed, 2_000_000);
    insert_track_with_state(&mut store, 3, TrackState::Lost, 3_000_000);
    assert_eq!(store.track_count(), 3);

    // Enforce — should evict Lost (track 3) first, even though Coasted
    // (track 1) is older.
    store.enforce_retention(MonotonicTs::from_nanos(5_000_000));
    assert_eq!(store.track_count(), 2);
    assert!(
        store.get_track(&TrackId::new(3)).is_none(),
        "Lost track should be evicted before Coasted"
    );
    assert!(
        store.get_track(&TrackId::new(1)).is_some(),
        "Coasted track should survive when Lost tracks are available"
    );
}

#[test]
fn hard_cap_falls_back_to_coasted_when_no_lost() {
    let retention = RetentionPolicy {
        max_concurrent_tracks: 2,
        max_track_age: nv_core::Duration::from_secs(600),
        ..RetentionPolicy::default()
    };
    let mut store = TemporalStore::new(retention);
    let epoch = ViewEpoch::INITIAL;

    // 2 Coasted + 1 Confirmed = 3 total.
    let coasted1 = make_track_with_state(1, TrackState::Coasted);
    store.commit_track(&coasted1, MonotonicTs::from_nanos(1_000_000), epoch);

    let coasted2 = make_track_with_state(2, TrackState::Coasted);
    store.commit_track(&coasted2, MonotonicTs::from_nanos(2_000_000), epoch);

    let confirmed = make_track_with_state(3, TrackState::Confirmed);
    store.commit_track(&confirmed, MonotonicTs::from_nanos(3_000_000), epoch);

    store.enforce_retention(MonotonicTs::from_nanos(5_000_000));
    assert_eq!(store.track_count(), 2);
    assert!(
        store.get_track(&TrackId::new(1)).is_none(),
        "oldest Coasted should be evicted when no Lost tracks exist"
    );
    assert!(store.get_track(&TrackId::new(2)).is_some());
    assert!(store.get_track(&TrackId::new(3)).is_some());
}

// ------------------------------------------------------------------
// P3: end_track emits TrackEnded boundary
// ------------------------------------------------------------------

#[test]
fn end_track_closes_segment_with_track_ended() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);
    store.commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch);

    let removed = store.end_track(&TrackId::new(1));
    assert!(removed.is_some());

    let history = removed.unwrap();
    let seg = history.trajectory.segments.last().unwrap();
    assert!(!seg.is_active());
    assert_eq!(seg.closed_by, Some(SegmentBoundary::TrackEnded));
    assert_eq!(seg.points.len(), 2);
}

#[test]
fn end_track_removes_track_from_store() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);
    assert_eq!(store.track_count(), 1);

    store.end_track(&TrackId::new(1));
    assert_eq!(store.track_count(), 0);
}

#[test]
fn end_track_returns_none_for_missing_track() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    assert!(store.end_track(&TrackId::new(999)).is_none());
}

// ------------------------------------------------------------------
// P1: apply_compensation transforms trajectory points
// ------------------------------------------------------------------

#[test]
fn apply_compensation_transforms_active_segment_points() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    // All points start at bbox center (0.05, 0.05).
    let id = TrackId::new(1);
    let before = store.get_track(&id).unwrap().trajectory.segments[0].points[0].position;

    // Apply a translation of (+0.1, +0.2).
    let t = nv_core::AffineTransform2D::new(1.0, 0.0, 0.1, 0.0, 1.0, 0.2);
    store.apply_compensation(&t, epoch);

    let after = store.get_track(&id).unwrap().trajectory.segments[0].points[0].position;
    assert!(
        (after.x - (before.x + 0.1)).abs() < 1e-5,
        "x should be translated"
    );
    assert!(
        (after.y - (before.y + 0.2)).abs() < 1e-5,
        "y should be translated"
    );

    // Compensation metadata should be recorded.
    let seg = &store.get_track(&id).unwrap().trajectory.segments[0];
    assert!(seg.compensation.is_some());
    assert_eq!(seg.compensation_count, 1);
}

#[test]
fn apply_compensation_ignores_wrong_epoch() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;
    let other_epoch = epoch.next();

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    let id = TrackId::new(1);
    let before = store.get_track(&id).unwrap().trajectory.segments[0].points[0].position;

    // Apply compensation for a different epoch — should be skipped.
    let t = nv_core::AffineTransform2D::new(1.0, 0.0, 0.5, 0.0, 1.0, 0.5);
    store.apply_compensation(&t, other_epoch);

    let after = store.get_track(&id).unwrap().trajectory.segments[0].points[0].position;
    assert!(
        (after.x - before.x).abs() < 1e-6,
        "position should be unchanged for wrong-epoch compensation"
    );
}

#[test]
fn apply_compensation_composes_multiple_transforms() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    let track = make_track(1);
    store.commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch);

    let t1 = nv_core::AffineTransform2D::new(1.0, 0.0, 0.1, 0.0, 1.0, 0.0);
    let t2 = nv_core::AffineTransform2D::new(1.0, 0.0, 0.2, 0.0, 1.0, 0.0);
    store.apply_compensation(&t1, epoch);
    store.apply_compensation(&t2, epoch);

    let seg = &store
        .get_track(&TrackId::new(1))
        .unwrap()
        .trajectory
        .segments[0];
    assert_eq!(seg.compensation_count, 2);
    // Cumulative transform should be t1 then t2 = translate by 0.3.
    let comp = seg.compensation.unwrap();
    assert!(
        (comp.m[2] - 0.3).abs() < 1e-9,
        "cumulative tx should be ~0.3"
    );
}

// ------------------------------------------------------------------
// end_track on already-closed segment (Lost → end_track)
// ------------------------------------------------------------------

#[test]
fn end_track_on_already_closed_segment_preserves_track_lost() {
    let mut store = TemporalStore::new(RetentionPolicy::default());
    let epoch = ViewEpoch::INITIAL;

    // Commit a Lost track — segment is closed with TrackLost.
    let lost = make_track_with_state(1, TrackState::Lost);
    store.commit_track(&lost, MonotonicTs::from_nanos(1_000_000), epoch);
    assert!(
        store
            .get_track(&TrackId::new(1))
            .unwrap()
            .trajectory
            .active_segment()
            .is_none(),
        "Lost track's segment should already be closed"
    );

    // end_track should still remove it even though there's no active segment.
    let removed = store.end_track(&TrackId::new(1));
    assert!(removed.is_some());
    assert_eq!(store.track_count(), 0);

    // The last segment should still have TrackLost (not overwritten).
    let hist = removed.unwrap();
    let seg = hist.trajectory.segments.last().unwrap();
    assert_eq!(
        seg.closed_by,
        Some(SegmentBoundary::TrackLost),
        "pre-existing TrackLost boundary should be preserved"
    );
}
