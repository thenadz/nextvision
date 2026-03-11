/// YOLOX output decoding.
///
/// The Megvii YOLOX v0.1.1rc0 ONNX export produces an output tensor of
/// shape `[1, N, 5+C]` where `N` is the total number of grid cells across
/// all FPN levels and `C` is the number of classes. Each row contains:
///
///   `[cx_offset, cy_offset, log_w, log_h, objectness, class_0, ..., class_C-1]`
///
/// Coordinates are raw grid offsets / log-space that need spatial decode
/// (grid offset addition, stride multiplication, exp for w/h).
/// Objectness and class scores are already post-sigmoid.

use crate::letterbox::LetterboxInfo;
use crate::nms::{self, RawDetection};
use nv_core::geom::BBox;
use nv_core::id::DetectionId;
use nv_core::TypedMetadata;
use nv_perception::{Detection, DetectionSet};

/// YOLOX strides per FPN level (P3, P4, P5).
const STRIDES: [u32; 3] = [8, 16, 32];

/// Build the grid offsets for all FPN levels.
///
/// Returns `(grid_offsets, expanded_strides)` where each element corresponds
/// to a grid cell.
fn build_grids(input_size: u32) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut offsets = Vec::new();
    let mut strides = Vec::new();
    for &s in &STRIDES {
        let n = input_size / s;
        for y in 0..n {
            for x in 0..n {
                offsets.push([x as f32, y as f32]);
                strides.push(s as f32);
            }
        }
    }
    (offsets, strides)
}

/// Decode YOLOX output tensor into a [`DetectionSet`].
///
/// The v0.1.1rc0 Megvii ONNX export applies sigmoid to objectness and
/// class scores inside the model graph, but coordinates remain raw
/// (grid offsets + log-space width/height). This function:
///
/// 1. Applies grid-offset + stride decode to cx/cy.
/// 2. Applies exp() + stride decode to w/h.
/// 3. Uses objectness and class scores directly (already post-sigmoid).
///
/// # Arguments
/// - `output` — flat f32 tensor of shape `[N, 5+num_classes]`.
/// - `num_classes` — number of object classes the model was trained on.
/// - `input_size` — model input spatial size (e.g. 640).
/// - `conf_threshold` — minimum objectness×class-probability to keep.
/// - `nms_threshold` — IoU threshold for NMS.
/// - `info` — letterbox metadata for remapping coordinates.
/// - `det_id_offset` — starting `DetectionId` counter (for monotonic IDs).
pub fn decode_yolox_output(
    output: &[f32],
    num_classes: usize,
    input_size: u32,
    conf_threshold: f32,
    nms_threshold: f32,
    info: &LetterboxInfo,
    det_id_offset: u64,
) -> DetectionSet {
    let cols = 5 + num_classes;
    let num_anchors = output.len() / cols;
    if num_anchors == 0 || output.len() % cols != 0 {
        return DetectionSet::empty();
    }

    let (grids, strides) = build_grids(input_size);

    // Phase 1: decode and threshold.
    let mut candidates = Vec::new();
    for i in 0..num_anchors {
        let row = &output[i * cols..(i + 1) * cols];

        // Objectness is already post-sigmoid from the ONNX graph.
        let objectness = row[4];
        if objectness < conf_threshold {
            continue;
        }

        // Find best class (scores are already post-sigmoid).
        let (best_cls, &best_cls_score) = row[5..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let score = objectness * best_cls_score;
        if score < conf_threshold {
            continue;
        }

        // Decode grid offsets + log-space w/h to letterbox pixel space.
        let (gx, gy) = if i < grids.len() {
            (grids[i][0], grids[i][1])
        } else {
            (0.0, 0.0)
        };
        let stride = if i < strides.len() { strides[i] } else { 1.0 };

        let cx = (row[0] + gx) * stride;
        let cy = (row[1] + gy) * stride;
        let w = row[2].exp() * stride;
        let h = row[3].exp() * stride;

        let x_min = cx - w / 2.0;
        let y_min = cy - h / 2.0;
        let x_max = cx + w / 2.0;
        let y_max = cy + h / 2.0;

        candidates.push(RawDetection {
            x_min,
            y_min,
            x_max,
            y_max,
            score,
            class_id: best_cls as u32,
        });
    }

    // Phase 2: NMS.
    let kept = nms::nms(&candidates, nms_threshold);

    // Phase 3: remap to normalised coordinates.
    let detections: Vec<Detection> = kept
        .into_iter()
        .enumerate()
        .map(|(idx, k)| {
            let c = &candidates[k];
            let (nx0, ny0, nx1, ny1) =
                info.remap_to_normalised(c.x_min, c.y_min, c.x_max, c.y_max);
            Detection {
                id: DetectionId::new(det_id_offset + idx as u64),
                class_id: c.class_id,
                confidence: c.score,
                bbox: BBox::new(nx0, ny0, nx1, ny1),
                embedding: None,
                metadata: TypedMetadata::new(),
            }
        })
        .collect();

    DetectionSet { detections }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_empty_tensor() {
        let info = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            target: 640,
            orig_w: 640,
            orig_h: 640,
        };
        let ds = decode_yolox_output(&[], 80, 640, 0.25, 0.45, &info, 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn decode_single_detection() {
        // Simulate 1 anchor with 2 classes, high objectness + class-0.
        // row = [cx_raw, cy_raw, w_raw, h_raw, obj, cls0, cls1]
        let num_classes = 2;
        let cols = 5 + num_classes;
        // Grid(0,0), stride=8 → cx=(0+0)*8=0, cy=0, w=exp(2)*8, h=exp(2)*8
        let mut row = vec![0.0f32; cols];
        row[0] = 5.0; // cx_raw → (5+gx)*stride
        row[1] = 5.0; // cy_raw
        row[2] = 2.0; // ln(w/stride)
        row[3] = 2.0; // ln(h/stride)
        row[4] = 0.9; // objectness
        row[5] = 0.8; // class 0
        row[6] = 0.1; // class 1

        let info = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            target: 640,
            orig_w: 640,
            orig_h: 640,
        };
        let ds = decode_yolox_output(&row, num_classes, 640, 0.1, 0.5, &info, 100);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.detections[0].class_id, 0);
        assert!(ds.detections[0].confidence > 0.5);
        assert_eq!(ds.detections[0].id.as_u64(), 100);
    }
}
