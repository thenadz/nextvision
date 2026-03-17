//! End-to-end detection output decoding.
//!
//! End-to-end detection models produce an output tensor of shape
//! `[batch, max_dets, 6]` where each row is:
//!
//!   `[x1, y1, x2, y2, confidence, class_id]`
//!
//! Coordinates are in the input tensor pixel space (letterboxed).
//! The model uses a learned query head — no NMS is required. We only
//! apply confidence thresholding and coordinate remapping.

use crate::letterbox::LetterboxInfo;
use nv_core::TypedMetadata;
use nv_core::geom::BBox;
use nv_core::id::DetectionId;
use nv_perception::{Detection, DetectionSet};

/// Columns per detection in the end-to-end output format.
const END2END_COLS: usize = 6;

/// Decode an end-to-end detection output slice into a [`DetectionSet`].
///
/// Each row is `[x1, y1, x2, y2, confidence, class_id]` in letterbox
/// pixel coordinates. Rows below `conf_threshold` are discarded.
/// Surviving boxes are remapped to normalised `[0, 1]` coordinates
/// relative to the original frame.
///
/// # Arguments
/// - `output` — flat f32 slice for **one batch item**, length `max_dets * 6`.
/// - `conf_threshold` — minimum confidence to keep.
/// - `info` — letterbox metadata for coordinate remapping.
/// - `det_id_offset` — starting [`DetectionId`] counter.
pub fn decode_end2end_output(
    output: &[f32],
    conf_threshold: f32,
    info: &LetterboxInfo,
    det_id_offset: u64,
) -> DetectionSet {
    let num_slots = output.len() / END2END_COLS;
    if num_slots == 0 || output.len() % END2END_COLS != 0 {
        return DetectionSet::empty();
    }

    let mut detections = Vec::new();
    for i in 0..num_slots {
        let row = &output[i * END2END_COLS..(i + 1) * END2END_COLS];
        let confidence = row[4];
        if confidence < conf_threshold {
            continue;
        }

        let (nx0, ny0, nx1, ny1) = info.remap_to_normalised(row[0], row[1], row[2], row[3]);

        detections.push(Detection {
            id: DetectionId::new(det_id_offset + detections.len() as u64),
            class_id: row[5] as u32,
            confidence,
            bbox: BBox::new(nx0, ny0, nx1, ny1),
            embedding: None,
            metadata: TypedMetadata::new(),
        });
    }

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
        let ds = decode_end2end_output(&[], 0.25, &info, 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn decode_single_detection() {
        // row = [x1, y1, x2, y2, confidence, class_id]
        let row = vec![100.0, 200.0, 300.0, 400.0, 0.9, 0.0];

        let info = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            target: 640,
            orig_w: 640,
            orig_h: 640,
        };
        let ds = decode_end2end_output(&row, 0.25, &info, 100);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.detections[0].class_id, 0);
        assert!((ds.detections[0].confidence - 0.9).abs() < 1e-5);
        assert_eq!(ds.detections[0].id.as_u64(), 100);
    }

    #[test]
    fn decode_filters_low_confidence() {
        // Two slots: one above threshold, one below.
        let data = vec![
            100.0, 100.0, 200.0, 200.0, 0.8, 1.0, // kept
            300.0, 300.0, 400.0, 400.0, 0.1, 2.0, // filtered
        ];
        let info = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            target: 640,
            orig_w: 640,
            orig_h: 640,
        };
        let ds = decode_end2end_output(&data, 0.25, &info, 0);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.detections[0].class_id, 1);
    }
}
