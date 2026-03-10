//! Benchmarks for frame construction and cloning.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nv_core::{FeedId, MonotonicTs, TypedMetadata, WallTs};
use nv_frame::{FrameEnvelope, PixelFormat};

fn frame_construction_720p(c: &mut Criterion) {
    let width = 1280u32;
    let height = 720u32;
    let stride = width * 3;
    let data = vec![0u8; (stride * height) as usize];

    c.bench_function("frame_construct_720p_owned", |b| {
        b.iter(|| {
            black_box(FrameEnvelope::new_owned(
                FeedId::new(1),
                0,
                MonotonicTs::from_nanos(0),
                WallTs::from_micros(0),
                width,
                height,
                PixelFormat::Rgb8,
                stride,
                data.clone(),
                TypedMetadata::new(),
            ));
        });
    });
}

fn frame_clone(c: &mut Criterion) {
    let frame = FrameEnvelope::new_owned(
        FeedId::new(1),
        0,
        MonotonicTs::from_nanos(0),
        WallTs::from_micros(0),
        1280,
        720,
        PixelFormat::Rgb8,
        1280 * 3,
        vec![0u8; 1280 * 720 * 3],
        TypedMetadata::new(),
    );

    c.bench_function("frame_clone_arc", |b| {
        b.iter(|| {
            black_box(frame.clone());
        });
    });
}

criterion_group!(benches, frame_construction_720p, frame_clone);
criterion_main!(benches);
