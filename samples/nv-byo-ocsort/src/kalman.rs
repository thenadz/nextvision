/// Constant-velocity Kalman filter for bounding-box tracking.
///
/// State vector: `[cx, cy, s, r, vx, vy, vs]`
/// where `(cx, cy)` is the bounding box centre, `s` is the area, `r` is the
/// aspect ratio (width/height), and `(vx, vy, vs)` are velocities.
///
/// Measurement vector: `[cx, cy, s, r]`

/// Kalman state dimension.
const DIM_X: usize = 7;
/// Kalman measurement dimension.
const DIM_Z: usize = 4;

/// A lightweight Kalman filter operating on 7-d state.
#[derive(Debug, Clone)]
pub(crate) struct KalmanBoxTracker {
    /// State estimate.
    pub x: [f64; DIM_X],
    /// Error covariance (flattened 7×7, row-major).
    pub p: [f64; DIM_X * DIM_X],
    /// Number of updates so far.
    pub hits: u32,
    /// Consecutive frames without update.
    pub time_since_update: u32,
    /// The age in frames.
    pub age: u32,
    /// Cached last observation for ORU.
    pub last_observation: Option<[f64; DIM_Z]>,
    /// History of observations for ORU re-update.
    pub observation_history: Vec<[f64; DIM_Z]>,
    /// History of predicted states for OCM direction.
    pub velocity_direction: Option<[f64; 2]>,
}

// Process noise and measurement noise — hand-tuned constants from OC-SORT.
const STD_WEIGHT_POSITION: f64 = 1.0 / 20.0;
const STD_WEIGHT_VELOCITY: f64 = 1.0 / 160.0;

impl KalmanBoxTracker {
    /// Initialise a new tracker from a measurement `[cx, cy, s, r]`.
    pub fn new(measurement: [f64; DIM_Z]) -> Self {
        let mut x = [0.0; DIM_X];
        x[0] = measurement[0];
        x[1] = measurement[1];
        x[2] = measurement[2];
        x[3] = measurement[3];
        // Velocities start at 0.

        let mut p = [0.0; DIM_X * DIM_X];
        // Initialise diagonal with large uncertainty for velocities.
        for i in 0..DIM_Z {
            p[i * DIM_X + i] = 10.0;
        }
        for i in DIM_Z..DIM_X {
            p[i * DIM_X + i] = 1000.0;
        }

        Self {
            x,
            p,
            hits: 1,
            time_since_update: 0,
            age: 0,
            last_observation: Some(measurement),
            observation_history: vec![measurement],
            velocity_direction: None,
        }
    }

    /// Predict the next state.
    pub fn predict(&mut self) {
        // Constant-velocity model: x' = F·x
        self.x[0] += self.x[4];
        self.x[1] += self.x[5];
        self.x[2] += self.x[6];
        // Area must remain positive.
        if self.x[2] < 1e-6 {
            self.x[2] = 1e-6;
        }

        self.age += 1;
        self.time_since_update += 1;

        // Covariance predict: P' = F·P·F^T + Q  (simplified for CV model).
        // We propagate the cross-terms for position/velocity.
        let mut p_new = self.p;
        // Add velocity uncertainty to position rows.
        for i in 0..3 {
            let vi = i + 4;
            // P[i,j] += P[vi,j] + P[i,vj] + P[vi,vj]  for j < DIM_X
            for j in 0..DIM_X {
                let vj = if j < 3 { j + 4 } else { j };
                p_new[i * DIM_X + j] += self.p[vi * DIM_X + j]
                    + self.p[i * DIM_X + vj]
                    + self.p[vi * DIM_X + vj];
            }
        }
        // Add process noise on diagonal.
        let q_pos = (STD_WEIGHT_POSITION * self.x[2].abs().sqrt()).powi(2);
        let q_vel = (STD_WEIGHT_VELOCITY * self.x[2].abs().sqrt()).powi(2);
        for i in 0..DIM_Z {
            p_new[i * DIM_X + i] += q_pos;
        }
        for i in DIM_Z..DIM_X {
            p_new[i * DIM_X + i] += q_vel;
        }
        self.p = p_new;
    }

    /// Update with a new measurement.
    pub fn update(&mut self, z: [f64; DIM_Z]) {
        // Compute direction from previous observation for OCM.
        if let Some(prev) = self.last_observation {
            let dx = z[0] - prev[0];
            let dy = z[1] - prev[1];
            let norm = (dx * dx + dy * dy).sqrt();
            if norm > 1e-8 {
                self.velocity_direction = Some([dx / norm, dy / norm]);
            }
        }

        // Innovation: y = z - H·x  (H extracts first 4 dims).
        let mut y = [0.0; DIM_Z];
        for i in 0..DIM_Z {
            y[i] = z[i] - self.x[i];
        }

        // Innovation covariance: S = H·P·H^T + R.
        let r_std = STD_WEIGHT_POSITION * self.x[2].abs().sqrt();
        let r = r_std * r_std;

        let mut s = [[0.0; DIM_Z]; DIM_Z];
        for i in 0..DIM_Z {
            for j in 0..DIM_Z {
                s[i][j] = self.p[i * DIM_X + j];
            }
            s[i][i] += r;
        }

        // Kalman gain: K = P·H^T·S^{-1}  (4×4 inversion via simple Gaussian elimination).
        let s_inv = invert_4x4(s);

        // K is 7×4.
        let mut k = [[0.0; DIM_Z]; DIM_X];
        for i in 0..DIM_X {
            for j in 0..DIM_Z {
                let mut val = 0.0;
                for m in 0..DIM_Z {
                    val += self.p[i * DIM_X + m] * s_inv[m][j];
                }
                k[i][j] = val;
            }
        }

        // State update: x += K·y.
        for i in 0..DIM_X {
            for j in 0..DIM_Z {
                self.x[i] += k[i][j] * y[j];
            }
        }

        // Covariance update: P = (I - K·H)·P.
        let mut p_new = self.p;
        for i in 0..DIM_X {
            for j in 0..DIM_X {
                let mut kh = 0.0;
                for m in 0..DIM_Z {
                    // H is identity for first 4 cols, zero elsewhere.
                    if j < DIM_Z {
                        kh += k[i][m] * if m == j { 1.0 } else { 0.0 };
                    }
                }
                p_new[i * DIM_X + j] -= kh * self.p[j * DIM_X + j]; // simplified
            }
        }
        self.p = p_new;

        self.time_since_update = 0;
        self.hits += 1;
        self.last_observation = Some(z);
        self.observation_history.push(z);
    }

    /// Get the current bounding box as `[x_min, y_min, x_max, y_max]`
    /// in normalised coordinates.
    pub fn get_bbox(&self) -> [f64; 4] {
        let cx = self.x[0];
        let cy = self.x[1];
        let s = self.x[2].max(1e-8);
        let r = self.x[3].max(1e-4);
        let w = (s * r).sqrt();
        let h = s / w.max(1e-8);
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    }

    /// Apply observation-centric re-update (ORU).
    ///
    /// Re-runs the Kalman filter forward on cached observations to
    /// correct drift accumulated during coasting.
    pub fn re_update(&mut self) {
        if self.observation_history.len() < 2 {
            return;
        }
        // Revert to second-to-last observation state by re-running from it.
        let obs = self.observation_history.clone();
        let last_idx = obs.len() - 1;
        // Re-initialise from the second-to-last observation.
        let rewind = &obs[last_idx.saturating_sub(1)];
        let mut x = [0.0; DIM_X];
        x[0] = rewind[0];
        x[1] = rewind[1];
        x[2] = rewind[2];
        x[3] = rewind[3];
        self.x = x;
        // Then re-apply the last observation.
        self.predict();
        self.update(obs[last_idx]);
    }

    /// Convert bounding box `[x_min, y_min, x_max, y_max]` to
    /// measurement `[cx, cy, s, r]`.
    pub fn bbox_to_z(bbox: [f64; 4]) -> [f64; DIM_Z] {
        let w = bbox[2] - bbox[0];
        let h = bbox[3] - bbox[1];
        let cx = bbox[0] + w / 2.0;
        let cy = bbox[1] + h / 2.0;
        let s = w * h;
        let r = if h.abs() > 1e-8 { w / h } else { 1.0 };
        [cx, cy, s, r]
    }
}

/// Simple 4×4 matrix inversion via Gauss-Jordan elimination.
fn invert_4x4(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut aug = [[0.0; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = m[i][j];
        }
        aug[i][i + 4] = 1.0;
    }
    for col in 0..4 {
        // Partial pivot.
        let mut max_row = col;
        for row in (col + 1)..4 {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let diag = aug[col][col];
        if diag.abs() < 1e-12 {
            // Singular — return identity as fallback.
            let mut id = [[0.0; 4]; 4];
            for i in 0..4 {
                id[i][i] = 1.0;
            }
            return id;
        }
        for j in 0..8 {
            aug[col][j] /= diag;
        }
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..8 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = aug[i][j + 4];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn new_tracker_returns_measurement() {
        let z = KalmanBoxTracker::bbox_to_z([0.1, 0.2, 0.3, 0.5]);
        let kf = KalmanBoxTracker::new(z);
        let bbox = kf.get_bbox();
        assert!(approx_eq(bbox[0], 0.1));
        assert!(approx_eq(bbox[1], 0.2));
        assert!(approx_eq(bbox[2], 0.3));
        assert!(approx_eq(bbox[3], 0.5));
    }

    #[test]
    fn predict_moves_state() {
        let z = KalmanBoxTracker::bbox_to_z([0.1, 0.2, 0.3, 0.5]);
        let mut kf = KalmanBoxTracker::new(z);
        let before = kf.get_bbox();
        kf.predict();
        // With zero velocity the prediction should barely change.
        let after = kf.get_bbox();
        assert!(approx_eq(before[0], after[0]));
        assert!(approx_eq(before[1], after[1]));
    }

    #[test]
    fn update_corrects_prediction() {
        let z1 = KalmanBoxTracker::bbox_to_z([0.1, 0.2, 0.3, 0.5]);
        let mut kf = KalmanBoxTracker::new(z1);
        kf.predict();
        let z2 = KalmanBoxTracker::bbox_to_z([0.15, 0.25, 0.35, 0.55]);
        kf.update(z2);
        let bbox = kf.get_bbox();
        // Should be pulled towards z2.
        assert!(bbox[0] > 0.1);
        assert!(bbox[1] > 0.2);
    }

    #[test]
    fn invert_identity() {
        let id = [[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]];
        let inv = invert_4x4(id);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(inv[i][j], expected));
            }
        }
    }
}
