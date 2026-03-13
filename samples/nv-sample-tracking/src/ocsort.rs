//! OC-SORT tracker core — manages track state, lifecycle, and frame-by-frame update.

use nv_core::id::TrackId;
use nv_core::MonotonicTs;
use nv_perception::{Detection, Track, TrackObservation, TrackState};
use nv_core::TypedMetadata;

use crate::config::TrackerConfig;
use crate::kalman::KalmanBoxTracker;
use crate::matching;

/// Internal tracked object.
pub(crate) struct TrackedObject {
    pub id: TrackId,
    pub class_id: u32,
    pub kf: KalmanBoxTracker,
    /// Number of consecutive frames with an association.
    pub hit_streak: u32,
    /// The last detection confidence (for output).
    pub confidence: f32,
    /// Last detection ID.
    pub last_detection_id: Option<nv_core::DetectionId>,
}

/// The core OC-SORT tracker.
pub(crate) struct OcSortTracker {
    config: TrackerConfig,
    tracks: Vec<TrackedObject>,
    next_id: u64,
}

impl OcSortTracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Clear all tracks (e.g. on view epoch change).
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Process one frame of detections and return the current track set.
    pub fn update(&mut self, detections: &[Detection], ts: MonotonicTs) -> Vec<Track> {
        // 1. Predict all existing tracks.
        for t in &mut self.tracks {
            t.kf.predict();
        }

        // 2. Build detection bboxes in normalised space.
        let det_bboxes: Vec<[f64; 4]> = detections
            .iter()
            .map(|d| {
                [
                    d.bbox.x_min as f64,
                    d.bbox.y_min as f64,
                    d.bbox.x_max as f64,
                    d.bbox.y_max as f64,
                ]
            })
            .collect();

        // 3. Associate.
        let result = matching::associate(
            &self.tracks.iter().map(|t| t.kf.clone()).collect::<Vec<_>>(),
            &det_bboxes,
            self.config.iou_threshold as f64,
            self.config.ocm_weight as f64,
        );

        // 4. Update matched tracks.
        for &(ti, di) in &result.matched {
            let det = &detections[di];
            let z = KalmanBoxTracker::bbox_to_z(det_bboxes[di]);
            let track = &mut self.tracks[ti];

            // OC-SORT ORU: if the track was coasting, re-update.
            if track.kf.time_since_update > 1 {
                track.kf.re_update();
            }

            track.kf.update(z);
            track.hit_streak += 1;
            track.confidence = det.confidence;
            track.class_id = det.class_id;
            track.last_detection_id = Some(det.id);
        }

        // 5. Create new tracks for unmatched detections.
        for &di in &result.unmatched_dets {
            let det = &detections[di];
            let z = KalmanBoxTracker::bbox_to_z(det_bboxes[di]);
            let id = TrackId::new(self.next_id);
            self.next_id += 1;
            self.tracks.push(TrackedObject {
                id,
                class_id: det.class_id,
                kf: KalmanBoxTracker::new(z),
                hit_streak: 1,
                confidence: det.confidence,
                last_detection_id: Some(det.id),
            });
        }

        // 6. Reset hit streak for unmatched tracks (coasting).
        for &ti in &result.unmatched_tracks {
            self.tracks[ti].hit_streak = 0;
        }

        // 7. Remove dead tracks and build output.
        self.tracks.retain(|t| t.kf.time_since_update <= self.config.max_age);

        // 8. Build output tracks.
        let mut output = Vec::new();
        for t in &self.tracks {
            let state = if t.kf.time_since_update > 0 {
                if t.kf.time_since_update > self.config.max_age / 2 {
                    TrackState::Lost
                } else {
                    TrackState::Coasted
                }
            } else if t.hit_streak >= self.config.min_hits {
                TrackState::Confirmed
            } else {
                TrackState::Tentative
            };

            // Optionally skip tentative tracks in output.
            if state == TrackState::Tentative && !self.config.output_tentative {
                continue;
            }

            let bbox_arr = t.kf.get_bbox();
            let bbox = nv_core::BBox::new(
                bbox_arr[0] as f32,
                bbox_arr[1] as f32,
                bbox_arr[2] as f32,
                bbox_arr[3] as f32,
            );

            output.push(Track {
                id: t.id,
                class_id: t.class_id,
                state,
                current: TrackObservation {
                    ts,
                    bbox,
                    confidence: t.confidence,
                    state,
                    detection_id: t.last_detection_id,
                },
                metadata: TypedMetadata::new(),
            });
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::{DetectionId, MonotonicTs};
    use nv_core::geom::BBox;

    fn det(id: u64, x0: f32, y0: f32, x1: f32, y1: f32, conf: f32) -> Detection {
        Detection {
            id: DetectionId::new(id),
            class_id: 0,
            confidence: conf,
            bbox: BBox::new(x0, y0, x1, y1),
            embedding: None,
            metadata: TypedMetadata::new(),
        }
    }

    #[test]
    fn new_detections_create_tracks() {
        let config = TrackerConfig {
            min_hits: 1,
            output_tentative: true,
            ..Default::default()
        };
        let mut tracker = OcSortTracker::new(config);
        let dets = vec![det(1, 0.1, 0.2, 0.3, 0.4, 0.9)];
        let tracks = tracker.update(&dets, MonotonicTs::from_nanos(0));
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn track_id_stability_across_frames() {
        let config = TrackerConfig {
            min_hits: 1,
            output_tentative: true,
            ..Default::default()
        };
        let mut tracker = OcSortTracker::new(config);
        let d1 = vec![det(1, 0.1, 0.2, 0.3, 0.4, 0.9)];
        let t1 = tracker.update(&d1, MonotonicTs::from_nanos(0));
        let id = t1[0].id;

        // Same position — should match same track.
        let d2 = vec![det(2, 0.1, 0.2, 0.3, 0.4, 0.9)];
        let t2 = tracker.update(&d2, MonotonicTs::from_nanos(33_000_000));
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].id, id);
    }

    #[test]
    fn track_expires_after_max_age() {
        let config = TrackerConfig {
            max_age: 2,
            min_hits: 1,
            output_tentative: true,
            ..Default::default()
        };
        let mut tracker = OcSortTracker::new(config);
        let dets = vec![det(1, 0.1, 0.2, 0.3, 0.4, 0.9)];
        tracker.update(&dets, MonotonicTs::from_nanos(0));

        // No detections for several frames.
        for i in 1..=3 {
            tracker.update(&[], MonotonicTs::from_nanos(i * 33_000_000));
        }
        let tracks = tracker.update(&[], MonotonicTs::from_nanos(4 * 33_000_000));
        assert!(tracks.is_empty(), "track should have expired");
    }

    #[test]
    fn reset_clears_all_tracks() {
        let config = TrackerConfig {
            min_hits: 1,
            output_tentative: true,
            ..Default::default()
        };
        let mut tracker = OcSortTracker::new(config);
        let dets = vec![det(1, 0.1, 0.2, 0.3, 0.4, 0.9)];
        tracker.update(&dets, MonotonicTs::from_nanos(0));
        tracker.reset();
        let tracks = tracker.update(&[], MonotonicTs::from_nanos(33_000_000));
        assert!(tracks.is_empty());
    }

    #[test]
    fn confirmed_after_min_hits() {
        let config = TrackerConfig {
            min_hits: 3,
            output_tentative: true,
            ..Default::default()
        };
        let mut tracker = OcSortTracker::new(config);

        for i in 0..3 {
            let d = vec![det(i + 1, 0.1, 0.2, 0.3, 0.4, 0.9)];
            let t = tracker.update(&d, MonotonicTs::from_nanos(i * 33_000_000));
            if i < 2 {
                assert_eq!(t[0].state, TrackState::Tentative);
            } else {
                assert_eq!(t[0].state, TrackState::Confirmed);
            }
        }
    }
}
