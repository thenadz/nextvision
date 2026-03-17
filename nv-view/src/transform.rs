//! Global transform estimate types.

use nv_core::AffineTransform2D;

use crate::view_state::ViewVersion;

/// A frame-to-reference coordinate transform estimate.
///
/// Provided by `ViewStateProvider` implementations that perform
/// feature matching, optical flow, or PTZ-based modeling.
#[derive(Clone, Debug, PartialEq)]
pub struct GlobalTransformEstimate {
    /// The affine transform from current frame coordinates to reference coordinates.
    pub transform: AffineTransform2D,
    /// Confidence in this estimate, in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Method used to compute this transform.
    pub method: TransformEstimationMethod,
    /// `ViewVersion` at which this transform was computed.
    ///
    /// Consumers can compare against the current version to detect staleness.
    pub computed_at: ViewVersion,
}

impl GlobalTransformEstimate {
    /// Displacement magnitude (in normalized coordinates) implied by
    /// the translation component of the transform.
    ///
    /// `sqrt(tx² + ty²)` of the affine matrix.
    #[must_use]
    pub fn displacement_magnitude(&self) -> f32 {
        let tx = self.transform.m[2];
        let ty = self.transform.m[5];
        (tx * tx + ty * ty).sqrt() as f32
    }
}

/// Method used to estimate a global coordinate transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransformEstimationMethod {
    /// Computed from PTZ telemetry combined with a camera model.
    PtzModel,
    /// Computed from inter-frame feature matching (optical flow, homography).
    FeatureMatching,
    /// Provided by a user-supplied external system.
    External,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nv_core::AffineTransform2D;

    fn identity_transform() -> GlobalTransformEstimate {
        GlobalTransformEstimate {
            transform: AffineTransform2D::IDENTITY,
            confidence: 1.0,
            method: TransformEstimationMethod::FeatureMatching,
            computed_at: ViewVersion::new(1),
        }
    }

    #[test]
    fn identity_has_zero_displacement() {
        let est = identity_transform();
        assert!((est.displacement_magnitude() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn translation_displacement() {
        let est = GlobalTransformEstimate {
            // Translation of (0.3, 0.4) → magnitude = 0.5
            transform: AffineTransform2D {
                m: [1.0, 0.0, 0.3, 0.0, 1.0, 0.4],
            },
            confidence: 0.9,
            method: TransformEstimationMethod::PtzModel,
            computed_at: ViewVersion::new(2),
        };
        assert!((est.displacement_magnitude() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn clone_eq() {
        let est = identity_transform();
        assert_eq!(est, est.clone());
    }
}
