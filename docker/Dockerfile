FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages required before running the setup scripts ──────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential cmake git wget curl ca-certificates \
        ninja-build \
        libgl1 libglib2.0-0 libgomp1 \
        kmod \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && pip3 install --upgrade pip setuptools wheel

# ── Copy scripts early for better layer caching ────────────────────────────────
WORKDIR /workspace/FoundationPose-TensorRT
COPY scripts/ scripts/

# ── README step 1: source scripts/deps.sh && activate_deps ────────────────────
# Installs CUDA 12.4, cuDNN 9.8, TensorRT 10.9.0 into ./deps/
RUN bash -c 'source scripts/deps.sh && activate_deps'

# Bake the env-vars that activate_deps exports so every subsequent layer and
# every container shell has them without needing to source anything.
ENV CUDA_HOME=/workspace/FoundationPose-TensorRT/deps/cuda-12-4 \
    CUDA_ROOT=/workspace/FoundationPose-TensorRT/deps/cuda-12-4/bin \
    CUDA_TOOLKIT_ROOT_DIR=/workspace/FoundationPose-TensorRT/deps/cuda-12-4 \
    TENSORRT_HOME=/workspace/FoundationPose-TensorRT/deps/TensorRT-10.9.0.34 \
    PATH="/workspace/FoundationPose-TensorRT/deps/TensorRT-10.9.0.34/bin:/workspace/FoundationPose-TensorRT/deps/cuda-12-4/bin:${PATH}" \
    LD_LIBRARY_PATH="/workspace/FoundationPose-TensorRT/deps/TensorRT-10.9.0.34/lib:/workspace/FoundationPose-TensorRT/deps/cuda-12-4/lib"

# ── Copy the rest of the project ───────────────────────────────────────────────
COPY . .

# ── README step 2: bash scripts/setup.sh ──────────────────────────────────────
# activate_deps is re-sourced here so nvcc is on PATH for the pytorch3d build.
RUN bash -c 'source scripts/deps.sh && activate_deps && bash scripts/setup.sh'

# Force cache-bust if setup.sh changes don't invalidate (e.g. transformations)
RUN pip install transformations


# ── README step 3: bash scripts/convert_onnx.sh ───────────────────────────────
# This downloads ONNX models and compiles TRT engines — bind-mount weights/
# at runtime so the compiled .plan files persist outside the container.
# Run it after `docker run`: bash docker/run.sh bash scripts/convert_onnx.sh

CMD ["bash"]
