#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Weights and demo_data are bind-mounted so compiled engines and datasets
# persist across container restarts without needing a rebuild.
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v "$REPO_ROOT/weights:/workspace/FoundationPose-TensorRT/weights" \
    -v "$REPO_ROOT/demo_data:/workspace/FoundationPose-TensorRT/demo_data" \
    -v "$REPO_ROOT/results:/workspace/FoundationPose-TensorRT/results" \
    foundationpose-tensorrt:latest \
    "$@"
