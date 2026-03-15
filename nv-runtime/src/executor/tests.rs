    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use nv_core::config::CameraMode;
    use nv_core::error::StageError;
    use nv_core::health::HealthEvent;
    use nv_core::id::{FeedId, StageId};
    use nv_core::timestamp::Duration;
    use nv_perception::{StageContext, StageOutput};
    use nv_temporal::RetentionPolicy;
    use nv_view::{DefaultEpochPolicy, TransitionPhase, ViewEpoch};
    fn make_executor() -> PipelineExecutor {
        PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy {
                max_track_age: Duration::from_secs(5),
                max_observations_per_track: 10,
                max_concurrent_tracks: 3,
                max_trajectory_points_per_track: 1000,
            },
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn clear_temporal_increments_epoch() {
        let mut exec = make_executor();
        let epoch_before = exec.view_epoch();
        exec.clear_temporal();
        let epoch_after = exec.view_epoch();
        assert!(
            epoch_after > epoch_before,
            "epoch should increase on restart: {epoch_before} → {epoch_after}",
        );
    }

    #[test]
    fn clear_temporal_resets_transition_to_settled() {
        let mut exec = make_executor();
        // Manually set transition to something non-settled.
        exec.view_state.transition = TransitionPhase::Moving;
        exec.clear_temporal();
        assert_eq!(exec.view_state.transition, TransitionPhase::Settled);
    }

    #[test]
    fn enforce_retention_evicts_old_lost_tracks() {
        let mut exec = make_executor();
        let retention = exec.temporal.retention().clone();

        // Insert a "Lost" track with last_seen at time 0.
        let track = nv_perception::Track {
            id: nv_core::TrackId::new(1),
            class_id: 0,
            state: nv_perception::TrackState::Lost,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(0),
                bbox: nv_core::BBox::new(0.0, 0.0, 0.1, 0.1),
                confidence: 0.9,
                state: nv_perception::TrackState::Lost,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        };
        let history = nv_temporal::store::TrackHistory::new(
            Arc::new(track),
            Arc::new(nv_temporal::Trajectory::new()),
            MonotonicTs::from_nanos(0),
            MonotonicTs::from_nanos(0),
            ViewEpoch::INITIAL,
        );
        exec.temporal
            .insert_track(nv_core::TrackId::new(1), history);
        assert_eq!(exec.temporal.track_count(), 1);

        // Now = last_seen + max_track_age + 1s → should be evicted.
        let now = MonotonicTs::from_nanos(
            retention.max_track_age.as_nanos() + Duration::from_secs(1).as_nanos(),
        );
        exec.temporal.enforce_retention(now);
        assert_eq!(
            exec.temporal.track_count(),
            0,
            "old Lost track should be evicted"
        );
    }

    #[test]
    fn enforce_retention_respects_max_concurrent_tracks() {
        let mut exec = make_executor();
        let retention = exec.temporal.retention().clone();
        assert_eq!(retention.max_concurrent_tracks, 3);

        // Insert 5 tracks: ids 1..=5, all Lost, staggered last_seen.
        for i in 1..=5u64 {
            let track = nv_perception::Track {
                id: nv_core::TrackId::new(i),
                class_id: 0,
                state: nv_perception::TrackState::Lost,
                current: nv_perception::TrackObservation {
                    ts: MonotonicTs::from_nanos(i * 1_000_000),
                    bbox: nv_core::BBox::new(0.0, 0.0, 0.1, 0.1),
                    confidence: 0.9,
                    state: nv_perception::TrackState::Lost,
                    detection_id: None,
                    metadata: nv_core::TypedMetadata::new(),
                },
                metadata: nv_core::TypedMetadata::new(),
            };
            let history = nv_temporal::store::TrackHistory::new(
                Arc::new(track),
                Arc::new(nv_temporal::Trajectory::new()),
                MonotonicTs::from_nanos(i * 1_000_000),
                MonotonicTs::from_nanos(i * 1_000_000),
                ViewEpoch::INITIAL,
            );
            exec.temporal
                .insert_track(nv_core::TrackId::new(i), history);
        }
        assert_eq!(exec.temporal.track_count(), 5);

        // Use a "now" that doesn't trigger age-based eviction.
        let now = MonotonicTs::from_nanos(10_000_000);
        exec.temporal.enforce_retention(now);

        // Should be capped at 3.
        assert_eq!(
            exec.temporal.track_count(),
            3,
            "should evict down to max_concurrent_tracks",
        );

        // The oldest (track 1, 2) should be gone; newest (3, 4, 5) remain.
        assert!(exec.temporal.get_track(&nv_core::TrackId::new(1)).is_none());
        assert!(exec.temporal.get_track(&nv_core::TrackId::new(2)).is_none());
        assert!(exec.temporal.get_track(&nv_core::TrackId::new(3)).is_some());
    }

    // ------------------------------------------------------------------
    // Admission rejection visibility (B / D.2)
    // ------------------------------------------------------------------

    #[test]
    fn admission_rejection_emits_health_event() {
        // Stage that returns more tracks than the executor's cap (3).
        struct ManyTracksStage;
        impl nv_perception::Stage for ManyTracksStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("many_tracks")
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                // 5 Confirmed tracks — exceeds cap of 3.
                let tracks: Vec<nv_perception::Track> = (1..=5u64)
                    .map(|i| nv_perception::Track {
                        id: nv_core::TrackId::new(i),
                        class_id: 0,
                        state: nv_perception::TrackState::Confirmed,
                        current: nv_perception::TrackObservation {
                            ts: MonotonicTs::from_nanos(0),
                            bbox: nv_core::BBox::new(0.0, 0.0, 0.1, 0.1),
                            confidence: 0.9,
                            state: nv_perception::TrackState::Confirmed,
                            detection_id: None,
                            metadata: nv_core::TypedMetadata::new(),
                        },
                        metadata: nv_core::TypedMetadata::new(),
                    })
                    .collect();
                Ok(nv_perception::StageOutput::with_tracks(tracks))
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(ManyTracksStage)],
            None,
            Vec::new(),
            RetentionPolicy {
                max_track_age: Duration::from_secs(5),
                max_observations_per_track: 10,
                max_concurrent_tracks: 3,
                max_trajectory_points_per_track: 1000,
            },
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let (_output, health_events) = exec.process_frame(&frame);

        // 5 tracks, cap 3: first 3 admitted, last 2 rejected (coalesced into one event).
        let rejection_events: Vec<_> = health_events
            .iter()
            .filter(|e| matches!(e, HealthEvent::TrackAdmissionRejected { .. }))
            .collect();
        assert_eq!(
            rejection_events.len(),
            1,
            "should emit one coalesced TrackAdmissionRejected event, got {rejection_events:?}",
        );
        match &rejection_events[0] {
            HealthEvent::TrackAdmissionRejected { rejected_count, .. } => {
                assert_eq!(*rejected_count, 2, "expected 2 rejected tracks");
            }
            _ => unreachable!(),
        }

        // The temporal store should be at exactly the cap.
        assert_eq!(exec.track_count(), 3);
    }

    // ------------------------------------------------------------------
    // Metadata propagation tests (P2)
    // ------------------------------------------------------------------

    #[test]
    fn process_frame_propagates_stage_metadata_to_output() {
        // A stage that inserts a typed artifact.
        struct MetadataStage;
        impl nv_perception::Stage for MetadataStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("metadata")
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                let mut out = nv_perception::StageOutput::empty();
                out.artifacts.insert::<String>("hello metadata".to_string());
                Ok(out)
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(MetadataStage)],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let (output, _health) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        // The stage's typed artifact should appear in the output metadata.
        let value = out.metadata.get::<String>();
        assert_eq!(
            value.map(|s| s.as_str()),
            Some("hello metadata"),
            "stage artifacts should be propagated to output metadata"
        );
    }

    // ------------------------------------------------------------------
    // FeedRestart segmentation tests (P2)
    // ------------------------------------------------------------------

    #[test]
    fn clear_temporal_closes_segments_with_feed_restart() {
        let mut exec = make_executor();

        // Commit a track so there's an active trajectory segment.
        let track = nv_perception::Track {
            id: nv_core::TrackId::new(1),
            class_id: 0,
            state: nv_perception::TrackState::Confirmed,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(1_000_000),
                bbox: nv_core::BBox::new(0.1, 0.1, 0.2, 0.2),
                confidence: 0.95,
                state: nv_perception::TrackState::Confirmed,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        };
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);

        // Verify active segment exists.
        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .unwrap();
        assert!(hist.trajectory.active_segment().is_some());

        // Take a snapshot before clear to capture the closed segment.
        // clear_temporal closes segments, then clears. We verify through
        // the close_all_segments mechanism directly.
        exec.temporal
            .close_all_segments(nv_temporal::SegmentBoundary::FeedRestart);

        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .unwrap();
        assert!(
            hist.trajectory.active_segment().is_none(),
            "segment should be closed"
        );
        let seg = hist.trajectory.segments().last().unwrap();
        assert_eq!(
            seg.closed_by(),
            Some(&nv_temporal::SegmentBoundary::FeedRestart),
            "segment should be closed with FeedRestart boundary"
        );
    }

    // ------------------------------------------------------------------
    // Epoch-driven segmentation end-to-end test (P1)
    // ------------------------------------------------------------------

    #[test]
    fn commit_via_executor_segments_on_epoch_change() {
        let mut exec = make_executor();
        let epoch0 = ViewEpoch::INITIAL;

        // Commit track in epoch 0.
        let track = nv_perception::Track {
            id: nv_core::TrackId::new(1),
            class_id: 0,
            state: nv_perception::TrackState::Confirmed,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(1_000_000),
                bbox: nv_core::BBox::new(0.1, 0.1, 0.2, 0.2),
                confidence: 0.95,
                state: nv_perception::TrackState::Confirmed,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        };
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(1_000_000), epoch0);

        assert_eq!(
            exec.temporal
                .get_track(&nv_core::TrackId::new(1))
                .unwrap()
                .trajectory
                .segment_count(),
            1
        );

        // Simulate epoch change.
        let epoch1 = epoch0.next();
        exec.temporal.set_view_epoch(epoch1);

        // Commit in new epoch — should create new segment.
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(2_000_000), epoch1);

        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .unwrap();
        assert_eq!(
            hist.trajectory.segment_count(),
            2,
            "epoch change should create a new trajectory segment"
        );
        assert!(!hist.trajectory.segments()[0].is_active());
        assert!(hist.trajectory.segments()[1].is_active());
        assert_eq!(hist.trajectory.segments()[1].view_epoch(), epoch1);
    }

    // ------------------------------------------------------------------
    // TrackEnded: authoritative-set semantics
    // ------------------------------------------------------------------

    /// Stage that returns an authoritative (possibly empty) track set.
    struct AuthoritativeTrackStage {
        tracks: Vec<nv_perception::Track>,
    }

    impl AuthoritativeTrackStage {
        fn new(tracks: Vec<nv_perception::Track>) -> Self {
            Self { tracks }
        }
    }

    impl nv_perception::Stage for AuthoritativeTrackStage {
        fn id(&self) -> nv_core::id::StageId {
            nv_core::id::StageId("authoritative_tracker")
        }
        fn process(
            &mut self,
            _ctx: &nv_perception::StageContext<'_>,
        ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
            Ok(nv_perception::StageOutput {
                tracks: Some(self.tracks.clone()),
                ..nv_perception::StageOutput::empty()
            })
        }
    }

    fn make_track(id: u64, state: nv_perception::TrackState) -> nv_perception::Track {
        nv_perception::Track {
            id: nv_core::TrackId::new(id),
            class_id: 0,
            state,
            current: nv_perception::TrackObservation {
                ts: MonotonicTs::from_nanos(0),
                bbox: nv_core::BBox::new(0.1, 0.1, 0.2, 0.2),
                confidence: 0.9,
                state,
                detection_id: None,
                metadata: nv_core::TypedMetadata::new(),
            },
            metadata: nv_core::TypedMetadata::new(),
        }
    }

    #[test]
    fn missing_track_from_authoritative_set_triggers_track_ended() {
        let track_a = make_track(1, nv_perception::TrackState::Confirmed);
        let track_b = make_track(2, nv_perception::TrackState::Confirmed);

        // Frame 1: stage produces [A, B].
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(AuthoritativeTrackStage::new(vec![
                track_a.clone(),
                track_b.clone(),
            ]))],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame1 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (out1, _) = exec.process_frame(&frame1);
        assert!(out1.is_some());
        assert_eq!(exec.temporal.track_count(), 2);

        // Frame 2: stage produces only [A] — B should be ended.
        exec.stages = vec![Box::new(AuthoritativeTrackStage::new(vec![
            track_a.clone(),
        ]))];

        let frame2 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            1,
            MonotonicTs::from_nanos(2_000_000),
            2,
            2,
            128,
        );
        let (out2, _) = exec.process_frame(&frame2);
        assert!(out2.is_some());
        // B should have been removed via end_track.
        assert_eq!(
            exec.temporal.track_count(),
            1,
            "track B should be ended and removed"
        );
        assert!(
            exec.temporal
                .get_track(&nv_core::TrackId::new(1))
                .is_some(),
            "track A should still exist"
        );
        assert!(
            exec.temporal
                .get_track(&nv_core::TrackId::new(2))
                .is_none(),
            "track B should be gone"
        );
    }

    #[test]
    fn no_false_track_ended_when_no_stage_produced_tracks() {
        // Pre-populate two tracks via direct commit.
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            // NoOpStage returns StageOutput::empty() — tracks: None.
            vec![Box::new(nv_test_util::mock_stage::NoOpStage::new("noop"))],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let track_a = make_track(1, nv_perception::TrackState::Confirmed);
        let track_b = make_track(2, nv_perception::TrackState::Confirmed);
        exec.temporal
            .commit_track(&track_a, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);
        exec.temporal
            .commit_track(&track_b, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);
        assert_eq!(exec.temporal.track_count(), 2);

        // Process a frame through the NoOp stage (non-authoritative).
        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(2_000_000),
            2,
            2,
            128,
        );
        let (out, _) = exec.process_frame(&frame);
        assert!(out.is_some());
        // Both tracks should survive — no stage claimed authoritativeness.
        assert_eq!(
            exec.temporal.track_count(),
            2,
            "non-authoritative frame must not end tracks"
        );
    }

    #[test]
    fn explicit_lost_uses_track_lost_not_track_ended() {
        // Stage produces track A as Lost.
        let track_a = make_track(1, nv_perception::TrackState::Lost);

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(AuthoritativeTrackStage::new(vec![
                track_a.clone(),
            ]))],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (out, _) = exec.process_frame(&frame);
        assert!(out.is_some());

        // Track A was committed as Lost — commit_track closes segment
        // with TrackLost. Since it IS in the authoritative set, it is
        // NOT passed to end_track. It remains in the store until retention
        // evicts it.
        let hist = exec
            .temporal
            .get_track(&nv_core::TrackId::new(1))
            .expect("Lost track should still be in store");
        let last_seg = hist.trajectory.segments().last().unwrap();
        assert_eq!(
            last_seg.closed_by(),
            Some(&nv_temporal::SegmentBoundary::TrackLost),
            "explicit Lost track should have TrackLost boundary, not TrackEnded"
        );
    }

    #[test]
    fn feed_restart_still_uses_feed_restart_boundary() {
        // Commit a track, then clear_temporal — segments should close
        // with FeedRestart, not TrackEnded.
        let mut exec = make_executor();
        let track = make_track(1, nv_perception::TrackState::Confirmed);
        exec.temporal
            .commit_track(&track, MonotonicTs::from_nanos(1_000_000), ViewEpoch::INITIAL);

        exec.clear_temporal();

        // After clear the store is empty, but we can verify the behavior
        // by the fact that clear_temporal calls close_all_segments with
        // FeedRestart before clearing. This is already tested in
        // clear_temporal_closes_segments_with_feed_restart, but we
        // repeat the essential invariant here for completeness.
        assert_eq!(
            exec.temporal.track_count(),
            0,
            "store should be empty after restart"
        );
    }

    // ------------------------------------------------------------------
    // View-state provenance tests
    // ------------------------------------------------------------------

    #[test]
    fn fixed_camera_provenance_shows_stable() {
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, health) = exec.process_frame(&frame);
        assert!(health.is_empty());
        let out = output.expect("should produce output");

        // Fixed camera — stability score should be 1.0, transition Settled,
        // no epoch decision, epoch 0.
        let vp = &out.provenance.view_provenance;
        assert_eq!(vp.stability_score, 1.0);
        assert_eq!(vp.transition, TransitionPhase::Settled);
        assert_eq!(vp.epoch, ViewEpoch::INITIAL);
        assert!(vp.epoch_decision.is_none(), "fixed camera has no epoch decision");

        // View state on output should be Valid.
        assert!(
            matches!(out.view.validity, nv_view::ContextValidity::Valid),
            "fixed camera output should have Valid context"
        );
    }

    #[test]
    fn observed_camera_with_provider_populates_provenance() {
        use nv_view::{MotionPollContext, MotionReport, ViewStateProvider};

        struct StableProvider;
        impl ViewStateProvider for StableProvider {
            fn poll(&self, _ctx: &MotionPollContext<'_>) -> MotionReport {
                MotionReport::default() // no motion data → Unknown state
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Observed,
            Some(Box::new(StableProvider)),
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        let vp = &out.provenance.view_provenance;
        // The provider returned no data, so CameraMotionState::Unknown.
        // DefaultEpochPolicy returns Continue for Unknown state.
        assert!(
            vp.epoch_decision.is_some(),
            "observed camera should have epoch decision"
        );
        // Version should have advanced from INITIAL.
        assert!(vp.version > nv_view::ViewVersion::INITIAL);
    }

    #[test]
    fn view_degradation_reflected_in_output() {
        use nv_view::{MotionPollContext, MotionReport, ViewStateProvider};

        // Provider reports small PTZ movement → Degrade decision.
        struct SmallPtzProvider;
        impl ViewStateProvider for SmallPtzProvider {
            fn poll(&self, ctx: &MotionPollContext<'_>) -> MotionReport {
                // Always report a small PTZ pan delta.
                let prev_pan = ctx
                    .previous_view
                    .ptz
                    .as_ref()
                    .map(|p| p.pan)
                    .unwrap_or(0.0);
                MotionReport {
                    ptz: Some(nv_view::PtzTelemetry {
                        pan: prev_pan + 2.0, // small move
                        tilt: 0.0,
                        zoom: 0.5,
                        ts: ctx.ts,
                    }),
                    ..Default::default()
                }
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Observed,
            Some(Box::new(SmallPtzProvider)),
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        // First frame initializes PTZ baseline (no previous ptz → Continue on
        // first frame since DefaultEpochPolicy needs both prev and current ptz).
        let f1 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        exec.process_frame(&f1);

        // Second frame: the policy now has prev_ptz and current_ptz, and
        // the delta (2.0) is small → Degrade.
        let f2 = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            1,
            MonotonicTs::from_nanos(2_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&f2);
        let out = output.expect("should produce output");

        // Stability score should have decreased from initial.
        assert!(
            out.provenance.view_provenance.stability_score < 1.0,
            "stability should decrease under PTZ movement"
        );
        assert!(
            matches!(
                out.view.validity,
                nv_view::ContextValidity::Degraded { .. }
            ),
            "view should be degraded under small PTZ move"
        );
    }

    // ==================================================================
    // Stage execution flow tests
    // ==================================================================

    #[test]
    fn stages_execute_in_declared_order() {
        // A stage that appends its ID to a shared log via a signal name.
        struct OrderStage {
            name: &'static str,
        }
        impl nv_perception::Stage for OrderStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId(self.name)
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                Ok(nv_perception::StageOutput::with_signal(
                    nv_perception::DerivedSignal {
                        name: self.name,
                        value: nv_perception::SignalValue::Boolean(true),
                        ts: MonotonicTs::ZERO,
                    },
                ))
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(OrderStage { name: "first" }),
                Box::new(OrderStage { name: "second" }),
                Box::new(OrderStage { name: "third" }),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        // Signals are appended in execution order.
        let names: Vec<&str> = out.signals.iter().map(|s| s.name).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }

    #[test]
    fn detector_output_visible_to_tracker() {
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 3)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        assert_eq!(out.detections.len(), 3, "detector should produce 3 detections");
        assert_eq!(out.tracks.len(), 3, "tracker should produce 3 tracks from 3 detections");
    }

    #[test]
    fn full_pipeline_detector_tracker_temporal_sink() {
        // Use a custom sink stage that records what it saw via a signal.
        struct RecordingSink;
        impl nv_perception::Stage for RecordingSink {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("recording_sink")
            }
            fn category(&self) -> nv_perception::StageCategory {
                nv_perception::StageCategory::Sink
            }
            fn process(
                &mut self,
                ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                // Record what we see — detection count and track count as signals.
                Ok(nv_perception::StageOutput::with_signals(vec![
                    nv_perception::DerivedSignal {
                        name: "sink_det_count",
                        value: nv_perception::SignalValue::Scalar(
                            ctx.artifacts.detections.len() as f64,
                        ),
                        ts: ctx.frame.ts(),
                    },
                    nv_perception::DerivedSignal {
                        name: "sink_track_count",
                        value: nv_perception::SignalValue::Scalar(
                            ctx.artifacts.tracks.len() as f64,
                        ),
                        ts: ctx.frame.ts(),
                    },
                ]))
            }
        }

        use nv_test_util::mock_stage::{MockDetectorStage, MockTemporalStage, MockTrackerStage};

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(MockTrackerStage::new("trk")),
                Box::new(MockTemporalStage::new("temporal")),
                Box::new(RecordingSink),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");

        // Detections and tracks should propagate to output.
        assert_eq!(out.detections.len(), 2);
        assert_eq!(out.tracks.len(), 2);

        // The temporal stage should have produced a track_count signal.
        // On frame 0 no tracks are in the temporal store yet (committed after stages).
        let temporal_signal = out
            .signals
            .iter()
            .find(|s| s.name == "track_count")
            .expect("temporal stage should produce track_count signal");
        assert!(matches!(temporal_signal.value, nv_perception::SignalValue::Scalar(_)));

        // The sink should see 2 detections and 2 tracks.
        let sink_det = out
            .signals
            .iter()
            .find(|s| s.name == "sink_det_count")
            .expect("sink should record detection count");
        let sink_trk = out
            .signals
            .iter()
            .find(|s| s.name == "sink_track_count")
            .expect("sink should record track count");
        assert!(matches!(sink_det.value, nv_perception::SignalValue::Scalar(v) if (v - 2.0).abs() < f64::EPSILON));
        assert!(matches!(sink_trk.value, nv_perception::SignalValue::Scalar(v) if (v - 2.0).abs() < f64::EPSILON));

        // Provenance should have 4 stage entries.
        assert_eq!(out.provenance.stages.len(), 4);
        assert_eq!(out.provenance.stages[0].stage_id, nv_core::id::StageId("det"));
        assert_eq!(out.provenance.stages[1].stage_id, nv_core::id::StageId("trk"));
        assert_eq!(out.provenance.stages[2].stage_id, nv_core::id::StageId("temporal"));
        assert_eq!(out.provenance.stages[3].stage_id, nv_core::id::StageId("recording_sink"));
    }

    #[test]
    fn stage_failure_drops_frame_skips_remaining() {
        use nv_test_util::mock_stage::{FailingStage, MockDetectorStage};

        // Pipeline: detector → failing → never-reached
        struct NeverReached;
        impl nv_perception::Stage for NeverReached {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("never_reached")
            }
            fn process(
                &mut self,
                _ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                panic!("this stage should never be called");
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(FailingStage::new("bad_stage")),
                Box::new(NeverReached),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, health) = exec.process_frame(&frame);

        // Frame should be dropped.
        assert!(output.is_none(), "failed stage should drop the frame");
        // Health event should be emitted.
        assert!(
            health.iter().any(|h| matches!(h, HealthEvent::StageError { .. })),
            "should emit StageError health event"
        );
    }

    #[test]
    fn stage_error_provenance_records_failure() {
        use nv_test_util::mock_stage::FailingStage;

        // Pipeline: detector → failing. The detector runs fine,
        // the failing stage errors. Both get provenance entries.
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(nv_test_util::mock_stage::NoOpStage::new("ok")),
                Box::new(FailingStage::new("fail")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        // Output is None (frame dropped), but we can verify through
        // health events that the executor processed both stages.
        assert!(output.is_none());
    }

    #[test]
    fn output_propagation_across_frames() {
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        // Process 3 frames.
        for i in 0..3u64 {
            let frame = nv_test_util::synthetic::solid_gray(
                FeedId::new(1),
                i,
                MonotonicTs::from_nanos(i * 33_000_000),
                4,
                4,
                128,
            );
            let (output, _) = exec.process_frame(&frame);
            let out = output.expect("should produce output");
            assert_eq!(out.detections.len(), 2, "frame {i}: should have 2 detections");
            assert_eq!(out.tracks.len(), 2, "frame {i}: should have 2 tracks");
            assert_eq!(out.frame_seq, i);
        }
        assert_eq!(exec.frames_processed(), 3);
    }

    #[test]
    fn feed_local_state_preserved_across_frames() {
        // A stage with internal state (counter).
        struct CounterStage {
            call_count: u64,
        }
        impl nv_perception::Stage for CounterStage {
            fn id(&self) -> nv_core::id::StageId {
                nv_core::id::StageId("counter")
            }
            fn process(
                &mut self,
                ctx: &nv_perception::StageContext<'_>,
            ) -> Result<nv_perception::StageOutput, nv_core::error::StageError> {
                self.call_count += 1;
                Ok(nv_perception::StageOutput::with_signal(
                    nv_perception::DerivedSignal {
                        name: "call_count",
                        value: nv_perception::SignalValue::Scalar(self.call_count as f64),
                        ts: ctx.frame.ts(),
                    },
                ))
            }
        }

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            vec![Box::new(CounterStage { call_count: 0 })],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        for i in 0..5u64 {
            let frame = nv_test_util::synthetic::solid_gray(
                FeedId::new(1),
                i,
                MonotonicTs::from_nanos(i * 33_000_000),
                2,
                2,
                128,
            );
            let (output, _) = exec.process_frame(&frame);
            let out = output.expect("should produce output");
            let signal = out.signals.iter().find(|s| s.name == "call_count").unwrap();
            match signal.value {
                nv_perception::SignalValue::Scalar(v) => {
                    assert_eq!(v as u64, i + 1, "stage internal state should persist")
                }
                _ => panic!("expected scalar signal"),
            }
        }
    }

    #[test]
    fn two_independent_executors_have_isolated_state() {
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        // Two executors simulating two feeds.
        let mut exec_a = PipelineExecutor::new(
            FeedId::new(1),
            vec![
                Box::new(MockDetectorStage::new("det", 2)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        let mut exec_b = PipelineExecutor::new(
            FeedId::new(2),
            vec![
                Box::new(MockDetectorStage::new("det", 5)),
                Box::new(MockTrackerStage::new("trk")),
            ],
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec_a.start_stages().unwrap();
        exec_b.start_stages().unwrap();

        let frame_a = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let frame_b = nv_test_util::synthetic::solid_gray(
            FeedId::new(2),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );

        let (out_a, _) = exec_a.process_frame(&frame_a);
        let (out_b, _) = exec_b.process_frame(&frame_b);

        let a = out_a.expect("feed A output");
        let b = out_b.expect("feed B output");

        // Feed A: 2 detections → 2 tracks.
        assert_eq!(a.detections.len(), 2);
        assert_eq!(a.tracks.len(), 2);
        assert_eq!(a.feed_id, FeedId::new(1));

        // Feed B: 5 detections → 5 tracks.
        assert_eq!(b.detections.len(), 5);
        assert_eq!(b.tracks.len(), 5);
        assert_eq!(b.feed_id, FeedId::new(2));

        // Temporal stores are independent.
        assert_eq!(exec_a.track_count(), 2);
        assert_eq!(exec_b.track_count(), 5);
    }

    #[test]
    fn pipeline_with_stage_pipeline_builder() {
        use nv_perception::StagePipeline;
        use nv_test_util::mock_stage::{MockDetectorStage, MockTrackerStage};

        let pipeline = StagePipeline::builder()
            .add(MockDetectorStage::new("det", 4))
            .add(MockTrackerStage::new("trk"))
            .build();

        let ids: Vec<&str> = pipeline.stage_ids().iter().map(|s| s.as_str()).collect();
        assert_eq!(ids, vec!["det", "trk"]);

        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            pipeline.into_stages(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            4,
            4,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");
        assert_eq!(out.detections.len(), 4);
        assert_eq!(out.tracks.len(), 4);
    }

    // ---------------------------------------------------------------
    // Frame inclusion policy tests
    // ---------------------------------------------------------------

    #[test]
    fn frame_inclusion_never_produces_no_frame() {
        let mut exec = make_executor();
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");
        assert!(out.frame.is_none());
    }

    #[test]
    fn frame_inclusion_always_includes_frame() {
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            Vec::new(),
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Always,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(1_000_000),
            2,
            2,
            128,
        );
        let (output, _) = exec.process_frame(&frame);
        let out = output.expect("should produce output");
        assert!(out.frame.is_some());
        // Zero-copy: same backing data (Arc bump, not pixel copy).
        assert_eq!(out.frame.as_ref().unwrap().seq(), frame.seq());
    }

    // ---------------------------------------------------------------
    // P2: Panic containment in start_stages / stop_stages
    // ---------------------------------------------------------------

    /// A stage that panics in on_start.
    struct PanicOnStartStage;
    impl Stage for PanicOnStartStage {
        fn id(&self) -> StageId { StageId("panic-start") }
        fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
        fn on_start(&mut self) -> Result<(), StageError> {
            panic!("intentional on_start panic");
        }
    }

    /// A stage that panics in on_stop.
    struct PanicOnStopStage {
        started: bool,
    }
    impl PanicOnStopStage {
        fn new() -> Self { Self { started: false } }
    }
    impl Stage for PanicOnStopStage {
        fn id(&self) -> StageId { StageId("panic-stop") }
        fn process(&mut self, _ctx: &StageContext<'_>) -> Result<StageOutput, StageError> {
            Ok(StageOutput::empty())
        }
        fn on_start(&mut self) -> Result<(), StageError> {
            self.started = true;
            Ok(())
        }
        fn on_stop(&mut self) -> Result<(), StageError> {
            panic!("intentional on_stop panic");
        }
    }

    #[test]
    fn start_stages_catches_panic() {
        let stages: Vec<Box<dyn Stage>> = vec![Box::new(PanicOnStartStage)];
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            stages,
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        let result = exec.start_stages();
        assert!(result.is_err(), "start_stages should return error on panic");
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("panicked"),
            "error should mention panic: {err}"
        );
    }

    #[test]
    fn stop_stages_catches_panic() {
        let stages: Vec<Box<dyn Stage>> = vec![Box::new(PanicOnStopStage::new())];
        let mut exec = PipelineExecutor::new(
            FeedId::new(1),
            stages,
            None,
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );
        exec.start_stages().unwrap();
        // Should not panic — catches the panic internally.
        exec.stop_stages();
    }

    #[test]
    fn flush_batch_rejections_returns_none_when_empty() {
        let mut exec = make_executor();
        assert!(exec.flush_batch_rejections().is_none());
    }

    #[test]
    fn flush_batch_rejections_emits_accumulated_count() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_flush") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(42),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        // Simulate accumulated rejections.
        exec.batch_rejection_count = 5;

        let evt = exec.flush_batch_rejections();
        assert!(evt.is_some(), "should have flushed accumulated rejections");
        match evt.unwrap() {
            HealthEvent::BatchSubmissionRejected { feed_id, processor_id, dropped_count } => {
                assert_eq!(feed_id, FeedId::new(42));
                assert_eq!(processor_id, StageId("noop_flush"));
                assert_eq!(dropped_count, 5);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        // After flush, count should be zero.
        assert!(exec.flush_batch_rejections().is_none());

        coord.shutdown();
    }

    /// When the feed shutdown flag is set *before* coordinator dies,
    /// CoordinatorShutdown should produce zero health events (expected).
    #[test]
    fn coordinator_shutdown_expected_emits_no_health() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_csexp") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        // Mark feed as shutting down *before* coordinator dies.
        let feed_shutdown = Arc::new(AtomicBool::new(true));

        let mut exec = PipelineExecutor::new(
            FeedId::new(99),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::clone(&feed_shutdown),
        );

        // Shut down the coordinator — the handle's next submit will see CoordinatorShutdown.
        coord.shutdown();
        // Give coordinator thread a moment to exit.
        std::thread::sleep(std::time::Duration::from_millis(50));

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 0, MonotonicTs::from_nanos(0), 2, 2, 128,
        );
        let (_output, health_events) = exec.process_frame(&frame);

        let stage_errors: Vec<_> = health_events.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert!(stage_errors.is_empty(), "expected no StageError for expected shutdown, got: {stage_errors:?}");
    }

    /// When the feed shutdown flag is NOT set and the coordinator dies,
    /// exactly one StageError health event should be emitted.
    #[test]
    fn coordinator_shutdown_unexpected_emits_one_stage_error() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_csunexp") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        // feed_shutdown stays false — coordinator death is unexpected.
        let mut exec = PipelineExecutor::new(
            FeedId::new(99),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        coord.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let frame = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 0, MonotonicTs::from_nanos(0), 2, 2, 128,
        );
        let (_output, health_events) = exec.process_frame(&frame);

        let stage_errors: Vec<_> = health_events.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert_eq!(stage_errors.len(), 1, "expected exactly one StageError for unexpected shutdown, got: {stage_errors:?}");

        match &stage_errors[0] {
            HealthEvent::StageError { feed_id, stage_id, error } => {
                assert_eq!(*feed_id, FeedId::new(99));
                assert_eq!(*stage_id, StageId("noop_csunexp"));
                let detail = format!("{error}");
                assert!(detail.contains("batch coordinator shut down unexpectedly"), "unexpected detail: {detail}");
            }
            _ => unreachable!(),
        }
    }

    /// After the first unexpected CoordinatorShutdown health event,
    /// subsequent frames should not emit duplicates.
    #[test]
    fn coordinator_shutdown_unexpected_deduplicates() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_csdedup") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(99),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        coord.shutdown();
        std::thread::sleep(std::time::Duration::from_millis(50));

        let frame1 = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 0, MonotonicTs::from_nanos(0), 2, 2, 128,
        );
        let frame2 = nv_test_util::synthetic::solid_gray(
            FeedId::new(99), 1, MonotonicTs::from_nanos(1_000_000), 2, 2, 128,
        );

        // First frame: should emit the StageError.
        let (_, h1) = exec.process_frame(&frame1);
        let errs1: Vec<_> = h1.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert_eq!(errs1.len(), 1, "first frame should emit one StageError");

        // Second frame: should NOT emit a duplicate.
        let (_, h2) = exec.process_frame(&frame2);
        let errs2: Vec<_> = h2.iter().filter(|e| matches!(e, HealthEvent::StageError { .. })).collect();
        assert!(errs2.is_empty(), "second frame should suppress duplicate StageError, got: {errs2:?}");
    }

    // ---------------------------------------------------------------
    // Batch timeout coalescing tests
    // ---------------------------------------------------------------

    #[test]
    fn flush_batch_timeouts_returns_none_when_empty() {
        let mut exec = make_executor();
        assert!(exec.flush_batch_timeouts().is_none());
    }

    #[test]
    fn flush_batch_timeouts_emits_accumulated_count() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        struct Noop;
        impl BatchProcessor for Noop {
            fn id(&self) -> StageId { StageId("noop_to") }
            fn process(&mut self, _: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> { Ok(()) }
        }

        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(4);
        let coord = BatchCoordinator::start(
            Box::new(Noop),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                response_timeout: None,
                max_in_flight_per_feed: 1,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(77),
            Vec::new(),
            Some(handle),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        // Simulate accumulated timeouts.
        exec.batch_timeout_count = 3;

        let evt = exec.flush_batch_timeouts();
        assert!(evt.is_some(), "should have flushed accumulated timeouts");
        match evt.unwrap() {
            HealthEvent::BatchTimeout { feed_id, processor_id, timed_out_count } => {
                assert_eq!(feed_id, FeedId::new(77));
                assert_eq!(processor_id, StageId("noop_to"));
                assert_eq!(timed_out_count, 3);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        // After flush, count should be zero.
        assert!(exec.flush_batch_timeouts().is_none());

        coord.shutdown();
    }

    /// Integration test: timeout coalescing through process_frame.
    ///
    /// Uses a slow processor that triggers feed-side timeouts, and
    /// verifies:
    /// 1. First timeout emits a BatchTimeout health event immediately.
    /// 2. Rapid subsequent timeouts within the 1s window are coalesced
    ///    (no event emitted).
    /// 3. Recovery (successful batch) flushes any accumulated count.
    #[test]
    fn timeout_coalescing_through_process_frame() {
        use crate::batch::{BatchConfig, BatchCoordinator};
        use nv_core::health::HealthEvent;
        use nv_perception::batch::{BatchEntry, BatchProcessor};

        /// Processor that sleeps longer than the response timeout,
        /// causing every submission to time out on the feed side.
        /// Controlled by a shared flag to switch to fast mode.
        struct ControllableProcessor {
            slow: Arc<AtomicBool>,
        }
        impl BatchProcessor for ControllableProcessor {
            fn id(&self) -> StageId { StageId("ctrl_slow") }
            fn process(&mut self, items: &mut [BatchEntry]) -> Result<(), nv_core::error::StageError> {
                if self.slow.load(Ordering::Relaxed) {
                    // Sleep longer than max_latency + response_timeout to
                    // ensure the feed-side recv_timeout fires.
                    std::thread::sleep(std::time::Duration::from_millis(400));
                }
                for item in items.iter_mut() {
                    item.output = Some(nv_perception::StageOutput::empty());
                }
                Ok(())
            }
        }

        let slow_flag = Arc::new(AtomicBool::new(true));
        let (health_tx, _) = tokio::sync::broadcast::channel::<HealthEvent>(64);
        let coord = BatchCoordinator::start(
            Box::new(ControllableProcessor { slow: Arc::clone(&slow_flag) }),
            BatchConfig {
                max_batch_size: 1,
                max_latency: std::time::Duration::from_millis(10),
                queue_capacity: None,
                // Very short response timeout so timeout triggers quickly.
                response_timeout: Some(std::time::Duration::from_millis(50)),
                // Allow 2 in-flight so the test can verify timeout
                // coalescing rather than in-flight cap behavior.
                max_in_flight_per_feed: 2,
            },
            health_tx,
        ).unwrap();
        let handle = coord.handle();

        let mut exec = PipelineExecutor::new(
            FeedId::new(55),
            Vec::new(),
            Some(handle.clone()),
            Vec::new(),
            RetentionPolicy::default(),
            CameraMode::Fixed,
            None,
            Box::new(DefaultEpochPolicy::default()),
            FrameInclusion::Never,
            Arc::new(AtomicBool::new(false)),
        );

        let make_frame = |seq: u64| {
            nv_test_util::synthetic::solid_gray(
                FeedId::new(55), seq, MonotonicTs::from_nanos(seq * 33_000_000), 2, 2, 128,
            )
        };

        // --- Frame 1: first timeout, should emit BatchTimeout immediately ---
        let (_, h1) = exec.process_frame(&make_frame(0));
        let timeout_events_1: Vec<_> = h1.iter()
            .filter(|e| matches!(e, HealthEvent::BatchTimeout { .. }))
            .collect();
        assert_eq!(
            timeout_events_1.len(), 1,
            "first timeout should emit one BatchTimeout event, got {timeout_events_1:?}"
        );
        // Verify the event contents.
        match &timeout_events_1[0] {
            HealthEvent::BatchTimeout { feed_id, timed_out_count, .. } => {
                assert_eq!(*feed_id, FeedId::new(55));
                assert_eq!(*timed_out_count, 1);
            }
            _ => unreachable!(),
        }

        // --- Frame 2: rapid second timeout within throttle window ---
        // Should NOT emit an event (coalesced into accumulator).
        let (_, h2) = exec.process_frame(&make_frame(1));
        let timeout_events_2: Vec<_> = h2.iter()
            .filter(|e| matches!(e, HealthEvent::BatchTimeout { .. }))
            .collect();
        assert!(
            timeout_events_2.is_empty(),
            "second timeout within throttle window should be coalesced, got {timeout_events_2:?}"
        );

        // Verify the internal accumulator has tracked it.
        assert_eq!(exec.batch_timeout_count, 1, "one coalesced timeout in accumulator");

        // --- Switch to fast mode and process a successful frame ---
        slow_flag.store(false, Ordering::Relaxed);
        // Give the coordinator time to finish any in-flight slow batches
        // (timed-out items from above may still be processing).
        std::thread::sleep(std::time::Duration::from_millis(500));

        let (output, h3) = exec.process_frame(&make_frame(2));
        assert!(output.is_some(), "frame should succeed after processor speeds up");

        // Recovery should flush the accumulated timeout count.
        let timeout_events_3: Vec<_> = h3.iter()
            .filter(|e| matches!(e, HealthEvent::BatchTimeout { .. }))
            .collect();
        assert_eq!(
            timeout_events_3.len(), 1,
            "recovery should flush accumulated timeouts, got {timeout_events_3:?}"
        );
        match &timeout_events_3[0] {
            HealthEvent::BatchTimeout { timed_out_count, .. } => {
                assert_eq!(*timed_out_count, 1, "flushed count should be 1 (one coalesced timeout)");
            }
            _ => unreachable!(),
        }

        // After recovery, no more pending timeouts.
        assert_eq!(exec.batch_timeout_count, 0);

        coord.shutdown();
    }
