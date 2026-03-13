//! Benchmarks for frame construction and cloning.
//!
//! `frame_construct_720p_mapped` measures the production zero-copy path
//! (pointer + guard, no pixel copy). `frame_clone_arc` measures the
//! per-stage handoff cost (Arc refcount bump).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};
use nv_frame::{FrameEnvelope, PixelFormat};

fn frame_construction_720p_mapped(c: &mut Criterion) {
    let width = 1280u32;
    let height = 720u32;
    let stride = width * 3;
    // Simulate a pre-existing buffer (in production this is a GStreamer mapped buffer).
    let buffer = vec![0u8; (stride * height) as usize];
    let ptr = buffer.as_ptr();
    let len = buffer.len();

    c.bench_function("frame_construct_720p_mapped", |b| {
        b.iter(|| {
            // The guard keeps the backing buffer alive. In production this is
            // a GStreamer MappedBuffer; here we use a cheap Box<()> stand-in
            // since we're measuring Arc + FrameInner allocation, not the
            // guard's destructor.
            let guard: Box<dyn std::any::Any + Send + Sync> = Box::new(());
            // SAFETY: `ptr` and `len` point into `buffer` which outlives
            // every frame created in this benchmark iteration.
            black_box(unsafe {
                FrameEnvelope::new_mapped(
                    FeedId::new(1),
                    0,
                    MonotonicTs::from_nanos(0),
                    WallTs::from_micros(0),
                    width,
                    height,
                    PixelFormat::Rgb8,
                    stride,
                    ptr,
                    len,
                    guard,
                    TypedMetadata::new(),
                )
            });
        });
    });
}

fn frame_clone(c: &mut Criterion) {
    let buffer = vec![0u8; 1280 * 720 * 3];
    let ptr = buffer.as_ptr();
    let len = buffer.len();
    let guard: Box<dyn std::any::Any + Send + Sync> = Box::new(());

    let frame = unsafe {
        FrameEnvelope::new_mapped(
            FeedId::new(1),
            0,
            MonotonicTs::from_nanos(0),
            WallTs::from_micros(0),
            1280,
            720,
            PixelFormat::Rgb8,
            1280 * 3,
            ptr,
            len,
            guard,
            TypedMetadata::new(),
        )
    };

    c.bench_function("frame_clone_arc", |b| {
        b.iter(|| {
            black_box(frame.clone());
        });
    });
}

criterion_group!(benches, frame_construction_720p_mapped, frame_clone);
criterion_main!(benches);
