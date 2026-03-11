/// Configuration for the OC-SORT tracker stage.
#[derive(Debug, Clone)]
pub struct OcSortConfig {
    /// Maximum frames a track can coast (no association) before being removed.
    /// Default: 30.
    pub max_age: u32,
    /// Minimum consecutive hits before a tentative track is confirmed.
    /// Default: 3.
    pub min_hits: u32,
    /// IoU threshold for the association cost matrix. Pairs below this
    /// threshold are not considered.
    /// Default: 0.3.
    pub iou_threshold: f32,
    /// Weight of the observation-centric momentum (OCM) term.
    /// 0.0 disables OCM (pure IoU association). Default: 0.5.
    pub ocm_weight: f32,
    /// Whether to include tentative (unconfirmed) tracks in the stage output.
    /// Default: false.
    pub output_tentative: bool,
}

impl Default for OcSortConfig {
    fn default() -> Self {
        Self {
            max_age: 30,
            min_hits: 3,
            iou_threshold: 0.3,
            ocm_weight: 0.5,
            output_tentative: false,
        }
    }
}
