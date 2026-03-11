#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate dependencies if necessary
if ! command -v trtexec &> /dev/null; then
    source "${SCRIPT_DIR}/deps.sh"
    activate_deps
fi

# Set TensorRT max batch size to match the number of pose hypotheses used during
# registration: min_n_views * (360 / inplane_step). This avoids chunking and
# keeps registration inference in a single engine call.
min_n_views=40
inplane_step=60
chunk_size=$((min_n_views * 360 / inplane_step))

MODEL_FOLDER_PATH="$SCRIPT_DIR/../weights"
ONNX_DIR="${MODEL_FOLDER_PATH}/onnx"
TRT_PLAN_DIR="${MODEL_FOLDER_PATH}/tensorrt"
REFINER_ONNX_MODEL="${ONNX_DIR}/refine_model.onnx"
SCORER_ONNX_MODEL="${ONNX_DIR}/score_model.onnx"

mkdir -p "${ONNX_DIR}"

if [ ! -f "${REFINER_ONNX_MODEL}" ]; then
   cd "${ONNX_DIR}" && \
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/refine_model.onnx' -O refine_model.onnx && \
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/score_model.onnx' -O score_model.onnx \
   && cd -
fi

echo "Using models from: ${MODEL_FOLDER_PATH}"

mkdir -p "${TRT_PLAN_DIR}"

# This is the conversion code for the `tao-toolkit-triton-apps` .onnx
# Originally, the conversion code uses `--preview=+fasterDynamicShapes0805`, but our TensorRT version does not support this.
#echo "Converting the FoundationPose refine model with max batch size: ${chunk_size}"
#trtexec --onnx="${REFINER_ONNX_MODEL}" \
#        --minShapes=inputA:1x6x160x160,inputB:1x6x160x160 \
#        --optShapes=inputA:${chunk_size}x6x160x160,inputB:${chunk_size}x6x160x160 \
#        --maxShapes=inputA:${chunk_size}x6x160x160,inputB:${chunk_size}x6x160x160 \
#        --saveEngine="${TRT_PLAN_DIR}/refiner_cs${chunk_size}.plan"
#
#echo "Converting the FoundationPose score model with max batch size: ${chunk_size}"
#trtexec --onnx="${SCORER_ONNX_MODEL}" \
#        --minShapes=inputA:1x6x160x160,inputB:1x6x160x160 \
#        --optShapes=inputA:${chunk_size}x6x160x160,inputB:${chunk_size}x6x160x160 \
#        --maxShapes=inputA:${chunk_size}x6x160x160,inputB:${chunk_size}x6x160x160 \
#        --saveEngine="${TRT_PLAN_DIR}/scorer_cs${chunk_size}.plan"


# This is the new version of the conversion code for the `isaac_ros_foundationpose` .onnx
# Source: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#quickstart
# The isaac_ros_foundationpose .onnx swapped the channel dimension to the back, so the command changed.
echo "Converting the FoundationPose refine model with max batch size: ${chunk_size}"
trtexec --onnx="${REFINER_ONNX_MODEL}" \
        --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --optShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --maxShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --saveEngine="${TRT_PLAN_DIR}/refiner_cs${chunk_size}.plan"

echo "Converting the FoundationPose score model with max batch size: ${chunk_size}"
trtexec --onnx="${SCORER_ONNX_MODEL}" \
        --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --optShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --maxShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --saveEngine="${TRT_PLAN_DIR}/scorer_cs${chunk_size}.plan"

echo "Conversion complete. Engines saved in ${TRT_PLAN_DIR}"
