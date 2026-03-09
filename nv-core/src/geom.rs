//! Geometry primitives for the NextVision runtime.
//!
//! All spatial types use normalized `[0, 1]` coordinates relative to frame dimensions.
//! This eliminates resolution-dependency throughout the perception pipeline.
//!
//! To convert to pixel coordinates: `px = normalized * dimension`.

use std::fmt;

/// Axis-aligned bounding box in normalized `[0, 1]` coordinates.
///
/// `x_min <= x_max` and `y_min <= y_max` are expected invariants.
/// The origin is the top-left corner of the frame.
#[derive(Clone, Copy, PartialEq)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl BBox {
    /// Create a new bounding box from normalized coordinates.
    #[must_use]
    pub fn new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// Width of the bounding box in normalized coordinates.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    /// Height of the bounding box in normalized coordinates.
    #[must_use]
    pub fn height(&self) -> f32 {
        self.y_max - self.y_min
    }

    /// Area of the bounding box in normalized coordinates.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Center point of the bounding box.
    #[must_use]
    pub fn center(&self) -> Point2 {
        Point2 {
            x: (self.x_min + self.x_max) * 0.5,
            y: (self.y_min + self.y_max) * 0.5,
        }
    }

    /// Compute intersection-over-union with another bounding box.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let ix_min = self.x_min.max(other.x_min);
        let iy_min = self.y_min.max(other.y_min);
        let ix_max = self.x_max.min(other.x_max);
        let iy_max = self.y_max.min(other.y_max);

        let iw = (ix_max - ix_min).max(0.0);
        let ih = (iy_max - iy_min).max(0.0);
        let intersection = iw * ih;

        let union = self.area() + other.area() - intersection;
        if union <= 0.0 {
            return 0.0;
        }
        intersection / union
    }
}

impl fmt::Debug for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BBox([{:.3}, {:.3}] -> [{:.3}, {:.3}])",
            self.x_min, self.y_min, self.x_max, self.y_max
        )
    }
}

/// 2D point in normalized `[0, 1]` coordinates.
#[derive(Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    /// Create a new point.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point (in normalized coordinates).
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl fmt::Debug for Point2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point2({:.4}, {:.4})", self.x, self.y)
    }
}

/// Closed polygon in normalized `[0, 1]` coordinates.
///
/// Defined by an ordered list of vertices. The polygon is implicitly closed
/// (last vertex connects to first).
#[derive(Clone, Debug, PartialEq)]
pub struct Polygon {
    /// Ordered vertices. Minimum 3 for a valid polygon.
    pub vertices: Vec<Point2>,
}

impl Polygon {
    /// Create a polygon from vertices.
    #[must_use]
    pub fn new(vertices: Vec<Point2>) -> Self {
        Self { vertices }
    }

    /// Number of vertices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Whether the polygon has no vertices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// 2D affine transform represented as a 3×3 matrix in row-major order.
///
/// Stored as 6 elements `[a, b, tx, c, d, ty]` representing:
/// ```text
/// | a  b  tx |
/// | c  d  ty |
/// | 0  0   1 |
/// ```
///
/// Uses `f64` for numerical precision in composed transforms.
#[derive(Clone, Copy, PartialEq)]
pub struct AffineTransform2D {
    /// `[a, b, tx, c, d, ty]` — the upper two rows of a 3×3 affine matrix.
    pub m: [f64; 6],
}

impl AffineTransform2D {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };

    /// Create from the six affine parameters.
    #[must_use]
    pub fn new(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Self {
        Self {
            m: [a, b, tx, c, d, ty],
        }
    }

    /// Apply this transform to a [`Point2`].
    ///
    /// The point is converted to `f64`, transformed, and converted back.
    #[must_use]
    pub fn apply(&self, p: Point2) -> Point2 {
        let x = p.x as f64;
        let y = p.y as f64;
        Point2 {
            x: (self.m[0] * x + self.m[1] * y + self.m[2]) as f32,
            y: (self.m[3] * x + self.m[4] * y + self.m[5]) as f32,
        }
    }

    /// Apply this transform to a [`BBox`] using the transform-and-re-bound method.
    ///
    /// All four corners are transformed, then an axis-aligned bounding box
    /// is computed from the results. This may enlarge the box if the transform
    /// involves rotation.
    #[must_use]
    pub fn apply_bbox(&self, bbox: BBox) -> BBox {
        let corners = [
            self.apply(Point2::new(bbox.x_min, bbox.y_min)),
            self.apply(Point2::new(bbox.x_max, bbox.y_min)),
            self.apply(Point2::new(bbox.x_max, bbox.y_max)),
            self.apply(Point2::new(bbox.x_min, bbox.y_max)),
        ];
        BBox {
            x_min: corners.iter().map(|c| c.x).fold(f32::INFINITY, f32::min),
            y_min: corners.iter().map(|c| c.y).fold(f32::INFINITY, f32::min),
            x_max: corners
                .iter()
                .map(|c| c.x)
                .fold(f32::NEG_INFINITY, f32::max),
            y_max: corners
                .iter()
                .map(|c| c.y)
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }

    /// Compose two transforms: `self` followed by `other`.
    ///
    /// Equivalent to matrix multiplication `other * self`.
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        let a = self.m;
        let b = other.m;
        Self {
            m: [
                b[0] * a[0] + b[1] * a[3],
                b[0] * a[1] + b[1] * a[4],
                b[0] * a[2] + b[1] * a[5] + b[2],
                b[3] * a[0] + b[4] * a[3],
                b[3] * a[1] + b[4] * a[4],
                b[3] * a[2] + b[4] * a[5] + b[5],
            ],
        }
    }
}

impl fmt::Debug for AffineTransform2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AffineTransform2D([{:.4}, {:.4}, {:.4}; {:.4}, {:.4}, {:.4}])",
            self.m[0], self.m[1], self.m[2], self.m[3], self.m[4], self.m[5]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_center() {
        let b = BBox::new(0.1, 0.2, 0.5, 0.8);
        let c = b.center();
        assert!((c.x - 0.3).abs() < 1e-6);
        assert!((c.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bbox_iou_identical() {
        let b = BBox::new(0.0, 0.0, 0.5, 0.5);
        assert!((b.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_iou_disjoint() {
        let a = BBox::new(0.0, 0.0, 0.2, 0.2);
        let b = BBox::new(0.5, 0.5, 1.0, 1.0);
        assert!((a.iou(&b)).abs() < 1e-6);
    }

    #[test]
    fn affine_identity() {
        let t = AffineTransform2D::IDENTITY;
        let p = Point2::new(0.3, 0.7);
        let q = t.apply(p);
        assert!((q.x - p.x).abs() < 1e-6);
        assert!((q.y - p.y).abs() < 1e-6);
    }

    #[test]
    fn affine_translation() {
        let t = AffineTransform2D::new(1.0, 0.0, 0.1, 0.0, 1.0, 0.2);
        let p = Point2::new(0.3, 0.4);
        let q = t.apply(p);
        assert!((q.x - 0.4).abs() < 1e-5);
        assert!((q.y - 0.6).abs() < 1e-5);
    }

    #[test]
    fn affine_compose_identity() {
        let t = AffineTransform2D::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0);
        let composed = t.then(&AffineTransform2D::IDENTITY);
        assert_eq!(t.m, composed.m);
    }
}
