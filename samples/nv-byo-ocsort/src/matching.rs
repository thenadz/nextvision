/// Detection–track association via IoU cost matrix + OCM penalty
/// solved with a greedy assignment.

use crate::kalman::KalmanBoxTracker;

/// Result of the matching phase.
pub(crate) struct MatchResult {
    /// `(track_index, detection_index)` pairs.
    pub matched: Vec<(usize, usize)>,
    /// Indices of unmatched detections.
    pub unmatched_dets: Vec<usize>,
    /// Indices of unmatched tracks.
    pub unmatched_tracks: Vec<usize>,
}

/// Compute IoU between two `[x_min, y_min, x_max, y_max]` boxes.
fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Compute the OCM (observation-centric momentum) penalty between a
/// track's velocity direction and the direction from the track's predicted
/// position to the detection.
fn ocm_penalty(track: &KalmanBoxTracker, det_bbox: [f64; 4]) -> f64 {
    let dir = match track.velocity_direction {
        Some(d) => d,
        None => return 0.0,
    };
    let pred = track.get_bbox();
    let pred_cx = (pred[0] + pred[2]) / 2.0;
    let pred_cy = (pred[1] + pred[3]) / 2.0;
    let det_cx = (det_bbox[0] + det_bbox[2]) / 2.0;
    let det_cy = (det_bbox[1] + det_bbox[3]) / 2.0;
    let dx = det_cx - pred_cx;
    let dy = det_cy - pred_cy;
    let norm = (dx * dx + dy * dy).sqrt();
    if norm < 1e-8 {
        return 0.0;
    }
    // Cosine similarity between velocity direction and detection direction.
    let cos_sim = (dir[0] * dx / norm + dir[1] * dy / norm).clamp(-1.0, 1.0);
    // Penalty: higher when directions disagree (cos_sim < 0).
    // Return value in [0, 1]: 0 = perfect agreement, 1 = opposite.
    (1.0 - cos_sim) / 2.0
}

/// Associate detections to existing tracks using IoU + OCM.
///
/// Uses a greedy (Hungarian-free) approach: iterate candidates in
/// descending IoU order and greedily assign. This is faster than the
/// full Hungarian and gives near-identical results in practice.
pub(crate) fn associate(
    tracks: &[KalmanBoxTracker],
    det_bboxes: &[[f64; 4]],
    iou_threshold: f64,
    ocm_weight: f64,
) -> MatchResult {
    let nt = tracks.len();
    let nd = det_bboxes.len();

    if nt == 0 || nd == 0 {
        return MatchResult {
            matched: Vec::new(),
            unmatched_dets: (0..nd).collect(),
            unmatched_tracks: (0..nt).collect(),
        };
    }

    // Build cost matrix (higher = better = IoU - ocm_weight * penalty).
    let mut candidates: Vec<(usize, usize, f64)> = Vec::with_capacity(nt * nd);
    for (ti, track) in tracks.iter().enumerate() {
        let pred_bbox = track.get_bbox();
        for (di, det_bbox) in det_bboxes.iter().enumerate() {
            let iou_val = iou(pred_bbox, *det_bbox);
            if iou_val < iou_threshold {
                continue;
            }
            let penalty = ocm_penalty(track, *det_bbox);
            let score = iou_val - ocm_weight * penalty;
            candidates.push((ti, di, score));
        }
    }

    // Sort by descending score.
    candidates.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut matched = Vec::new();
    let mut used_tracks = vec![false; nt];
    let mut used_dets = vec![false; nd];

    for (ti, di, _score) in candidates {
        if used_tracks[ti] || used_dets[di] {
            continue;
        }
        matched.push((ti, di));
        used_tracks[ti] = true;
        used_dets[di] = true;
    }

    let unmatched_tracks = (0..nt).filter(|&i| !used_tracks[i]).collect();
    let unmatched_dets = (0..nd).filter(|&i| !used_dets[i]).collect();

    MatchResult {
        matched,
        unmatched_dets,
        unmatched_tracks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_overlap() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 5.0, 15.0, 15.0];
        let v = iou(a, b);
        // Intersection = 5×5 = 25; union = 200-25 = 175; IoU ≈ 0.143
        assert!((v - 25.0 / 175.0).abs() < 1e-4);
    }

    #[test]
    fn iou_no_overlap() {
        let a = [0.0, 0.0, 1.0, 1.0];
        let b = [2.0, 2.0, 3.0, 3.0];
        assert_eq!(iou(a, b), 0.0);
    }

    #[test]
    fn associate_empty() {
        let result = associate(&[], &[], 0.3, 0.5);
        assert!(result.matched.is_empty());
        assert!(result.unmatched_dets.is_empty());
        assert!(result.unmatched_tracks.is_empty());
    }

    #[test]
    fn associate_perfect_match() {
        let z = KalmanBoxTracker::bbox_to_z([0.1, 0.2, 0.3, 0.4]);
        let kf = KalmanBoxTracker::new(z);
        let dets = [[0.1, 0.2, 0.3, 0.4]];
        let result = associate(&[kf], &dets, 0.3, 0.0);
        assert_eq!(result.matched.len(), 1);
        assert!(result.unmatched_dets.is_empty());
        assert!(result.unmatched_tracks.is_empty());
    }
}
