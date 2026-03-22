# syntax=docker/dockerfile:1
# =============================================================================
# NextVision sample — Jetson (JetPack 5.1.4 / L4T R35.4.1)
# =============================================================================
#
# Build (natively on the Jetson — recommended, from the repo root):
#
#   docker build -t nextvision -f samples/docker/Dockerfile .
#
# Run:
#
#   docker run --rm --runtime nvidia --network host --ipc host \
#       -e DISPLAY=$DISPLAY -v $HOME/.Xauthority:/root/.Xauthority:ro \
#       nextvision \
#       --video-uri rtsp://cam1/stream --video-uri rtsp://cam2/stream
#
# The entrypoint defaults to batched mode with GPU inference.
# All extra arguments are appended to the entrypoint command.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — build the Rust application
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1 AS builder

ARG ORT_URL=https://github.com/thenadz/onnxruntime4jetpack/releases/download/v1.18.2.1/onnxruntime-jetson-v1.18.2-jp5.1.4-cuda11.4-trt8.5.2-aarch64.tar.gz
ARG ORT_SHA256=c3db9511005720c3f09f36099a29c275ebdd331a2a74f7c16bb807ae9dfc6c22
ARG RUST_VERSION=1.92.0

# Build toolchain.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        curl \
        cmake \
        libssl-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libx11-dev libxcursor-dev libxrandr-dev libxi-dev \
        libxkbcommon-dev libxkbcommon-x11-dev libgl-dev

# Download and install pre-built ORT shared libraries (SHA-256 verified).
RUN set -e \
    && curl -fSL -o /tmp/ort.tar.gz "${ORT_URL}" \
    && echo "${ORT_SHA256}  /tmp/ort.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/ort \
    && tar xzf /tmp/ort.tar.gz -C /opt/ort --no-same-owner \
    && rm /tmp/ort.tar.gz \
    && cd /opt/ort && sha256sum -c SHA256SUMS \
    && cp libonnxruntime.so.1.18.2 /usr/local/lib/ \
    && cp libonnxruntime.so        /usr/local/lib/ \
    && cp libonnxruntime_providers_shared.so   /usr/local/lib/ \
    && cp libonnxruntime_providers_cuda.so     /usr/local/lib/ \
    && cp libonnxruntime_providers_tensorrt.so /usr/local/lib/ \
    && ldconfig

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
        -y --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

# --features jetson enables link-time ORT binding + CUDA EP.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/build/target \
    set -e \
    && ORT_LIB_LOCATION=/usr/local/lib \
       ORT_PREFER_DYNAMIC_LINK=1 \
       cargo build --release -p nv-sample-app \
         --no-default-features --features jetson \
    && cp target/release/nv-sample-app /build/nv-sample-app

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
# l4t-tensorrt:r8.5.2-runtime provides CUDA, cuDNN, TensorRT runtime libs,
# GStreamer, X11/Weston/EGL, and Vulkan on Ubuntu 20.04.
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libxcursor1 \
        libxrandr2 \
        libxi6 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the pre-built ORT shared libraries.
COPY --from=builder /usr/local/lib/libonnxruntime*.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=builder /build/nv-sample-app .
COPY samples/models/yolo26s.onnx ./models/

ENV ORT_LOG_LEVEL="WARNING"

# Default: batched mode with GPU inference. Append --video-uri flags.
ENTRYPOINT ["./nv-sample-app", \
            "--batch", "--batch-size", "4", "--batch-latency-ms", "50", \
            "--gpu", \
            "--model", "/app/models/yolo26s.onnx"]
