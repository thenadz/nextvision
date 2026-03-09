//! Optional buffer pool / slab for frame allocation.
//!
//! The primary zero-copy path uses GStreamer-managed buffers. This module
//! provides a simple pool for owned frames (test, synthetic, or fallback paths).
//!
//! The pool implementation will be added when the runtime is built.
//! For now, this module declares the interface.

/// Configuration for an optional frame buffer pool.
///
/// The pool pre-allocates `Vec<u8>` buffers of a fixed size and recycles them
/// to reduce allocation pressure in high-throughput scenarios.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers to keep in the pool.
    pub capacity: usize,
    /// Size of each buffer in bytes.
    pub buffer_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            capacity: 8,
            buffer_size: 1920 * 1080 * 3, // 1080p RGB
        }
    }
}
