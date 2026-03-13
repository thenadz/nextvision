//! Non-maximum suppression.
//!
//! Given a set of candidate detections (each with a bounding box, confidence,
//! and class ID), this performs per-class greedy NMS using IoU overlap.

/// A raw detection candidate before NMS.
#[derive(Debug, Clone)]
pub struct RawDetection {
    /// Pixel-space bounding box in the letterboxed image.
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
    /// Object confidence × class probability.
    pub score: f32,
    /// Winning class index.
    pub class_id: u32,
}

/// Run per-class greedy NMS and return surviving detection indices.
///
/// Detections are sorted by score (descending). For each candidate, any
/// lower-scored detection of the same class with IoU ≥ `iou_threshold`
/// is suppressed.
pub fn nms(detections: &[RawDetection], iou_threshold: f32) -> Vec<usize> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Sort indices by descending score.
    let mut indices: Vec<usize> = (0..detections.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        detections[b]
            .score
            .partial_cmp(&detections[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::with_capacity(detections.len());
    let mut suppressed = vec![false; detections.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        let a = &detections[i];

        for &j in &indices {
            if suppressed[j] || j == i {
                continue;
            }
            let b = &detections[j];
            if b.class_id != a.class_id {
                continue;
            }
            if iou_of(a, b) >= iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

fn iou_of(a: &RawDetection, b: &RawDetection) -> f32 {
    let x0 = a.x_min.max(b.x_min);
    let y0 = a.y_min.max(b.y_min);
    let x1 = a.x_max.min(b.x_max);
    let y1 = a.y_max.min(b.y_max);

    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min);
    let area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x_min: f32, y_min: f32, x_max: f32, y_max: f32, score: f32, class_id: u32) -> RawDetection {
        RawDetection { x_min, y_min, x_max, y_max, score, class_id }
    }

    #[test]
    fn nms_empty() {
        assert!(nms(&[], 0.5).is_empty());
    }

    #[test]
    fn nms_no_overlap() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            det(20.0, 20.0, 30.0, 30.0, 0.8, 0),
        ];
        let kept = nms(&dets, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_suppresses_overlapping_same_class() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            det(1.0, 1.0, 11.0, 11.0, 0.8, 0), // high overlap
        ];
        let kept = nms(&dets, 0.5);
        assert_eq!(kept.len(), 1);
        assert_eq!(dets[kept[0]].score, 0.9);
    }

    #[test]
    fn nms_keeps_different_classes() {
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            det(1.0, 1.0, 11.0, 11.0, 0.8, 1), // same box, different class
        ];
        let kept = nms(&dets, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_respects_threshold() {
        // Two boxes with ~68% IoU: intersection 9×9=81, union 100+100-81=119.
        let dets = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            det(1.0, 1.0, 11.0, 11.0, 0.8, 0),
        ];
        // With high threshold (0.9 > 0.68), both kept.
        assert_eq!(nms(&dets, 0.9).len(), 2);
        // With low threshold (0.5 < 0.68), one suppressed.
        assert_eq!(nms(&dets, 0.5).len(), 1);
    }
}
