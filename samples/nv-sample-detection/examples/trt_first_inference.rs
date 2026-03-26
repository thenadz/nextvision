//! Minimal TRT first-inference timing test.
//!
//! Calls the exact same `load_session()` and `DetectorConfig` the real app
//! uses, then runs a single inference. Nothing is reimplemented — the only
//! difference from a live run is that there is no pipeline/batch/runtime
//! machinery around it.
//!
//! Usage inside Docker:
//!   docker run --rm --runtime nvidia \
//!       --entrypoint /app/trt_first_inference \
//!       nextvision /app/models/yolo26s.onnx

use std::path::PathBuf;
use std::time::Instant;

use nv_sample_detection::DetectorConfig;
use nv_sample_detection::session::load_session;

fn main() {
    // Initialise tracing so we see the same log output as the real app.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "samples/models/yolo26s.onnx".to_string());
    let cuda_only = std::env::args().nth(2).as_deref() == Some("cuda-only");

    println!("=== TRT first-inference timing test ===");
    println!("model: {model_path}");
    if cuda_only {
        println!("mode: CUDA only (TRT disabled)");
    } else {
        println!("mode: TRT + CUDA");
    }

    // --- Build session using the EXACT same code path as the real app ---
    let config = DetectorConfig {
        model_path: PathBuf::from(&model_path),
        gpu: true,
        cuda_only,
        ..Default::default()
    };
    let stage_id = nv_core::id::StageId("trt-test");

    println!("[1/2] Loading session (same as real app: load_session())...");
    let t0 = Instant::now();
    let mut session = load_session(&config, stage_id).expect("load_session failed");
    println!("       done in {:.2}s", t0.elapsed().as_secs_f64());

    // --- First inference ---
    println!("[2/2] Running first inference (1x3x640x640 zeros)...");
    let input_data = vec![0.0_f32; 3 * 640 * 640];
    let input_tensor =
        ort::value::Tensor::from_array((vec![1i64, 3, 640, 640], input_data)).expect("tensor");

    let t1 = Instant::now();
    let outputs = session
        .run(ort::inputs![input_tensor])
        .expect("session.run()");
    let first_inference_secs = t1.elapsed().as_secs_f64();
    println!("       done in {:.2}s", first_inference_secs);
    drop(outputs);

    // --- Second inference ---
    println!("[bonus] Running second inference...");
    let input_data2 = vec![0.0_f32; 3 * 640 * 640];
    let input_tensor2 =
        ort::value::Tensor::from_array((vec![1i64, 3, 640, 640], input_data2)).expect("tensor");

    let t2 = Instant::now();
    let _outputs2 = session
        .run(ort::inputs![input_tensor2])
        .expect("session.run() #2");
    println!("       done in {:.2}s", t2.elapsed().as_secs_f64());

    println!();
    println!(
        "=== RESULT: first session.run() took {:.2}s ===",
        first_inference_secs
    );
}
