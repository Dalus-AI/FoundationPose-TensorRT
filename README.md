# FoundationPose-TensorRT

TensorRT-accelerated 6DoF object pose estimation and tracking based on [FoundationPose](https://nvlabs.github.io/FoundationPose/). Given an RGB-D image, a 3D mesh of the object, and an initial segmentation mask, the model estimates the object pose and tracks it across subsequent frames.

## Credits

The core inference code is derived from [tao-toolkit-triton-apps](https://github.com/NVIDIA-AI-IOT/tao-toolkit-triton-apps), with the heavy Triton Inference Server dependencies removed and replaced by a direct TensorRT backend.

The ONNX models are provided by [isaac_ros_foundationpose](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#quickstart).

## Setup

### 1. CUDA and TensorRT dependencies

Install CUDA 12.4 + cuDNN 9.8 and TensorRT 10.9.0 into a local `deps/` folder:

```bash
source scripts/deps.sh
activate_deps
```

This downloads and installs the dependencies locally - no system-wide installation required. The environment variables (`CUDA_HOME`, `TENSORRT_HOME`, `PATH`, `LD_LIBRARY_PATH`) are only active in the current shell session. Run `deactivate_deps` to restore the original environment.

To use a different CUDA or TensorRT version, edit `scripts/deps.sh`. Make sure the PyTorch CUDA version matches (see step 2).

### 2. Python environment

Create and activate a Python environment, e.g. with conda:

```bash
conda create --name fp_tensorrt python=3.10
conda activate fp_tensorrt
```

Then install all Python dependencies (requires `activate_deps` to be active):

```bash
source scripts/deps.sh && activate_deps
bash scripts/setup.sh
```

This installs PyTorch 2.5.0 (CUDA 12.4), nvdiffrast, pytorch3d, TensorRT Python bindings, and other required packages.

### 3. Model compilation

Download the ONNX models from NVIDIA NGC and compile them into TensorRT engine files:

```bash
bash scripts/convert_onnx.sh
```

This produces `weights/tensorrt/refiner_cs252.plan` and `weights/tensorrt/scorer_cs252.plan`.

**`chunk_size`** variable inside `convert_onnx.sh` controls the maximum batch size of the TensorRT engines (default: `252`). A smaller value reduces VRAM usage, which is useful when tracking multiple objects simultaneously or on memory-constrained GPUs. To change it, edit the `chunk_size` variable before running and use the matching value in `FoundationPoseWrapperConfig`.

## Usage

### Demo

Run the benchmark on the YCB mustard bottle sequence (demo data is downloaded automatically):

```bash
source scripts/deps.sh && activate_deps
python demo.py
```

This runs initial pose estimation on the first frame and tracks the object across the remaining frames, printing per-frame poses and mean inference times.

### Python API

```python
from foundationpose_tensorrt import FoundationPoseWrapper, FoundationPoseWrapperConfig

cfg = FoundationPoseWrapperConfig(
    downsample_width=None,   # Set e.g. to 256 for faster inference at lower accuracy
    est_refine_iter=5,       # Refinement iterations for initial pose estimation
    track_refine_iter=2,     # Refinement iterations for tracking
    chunk_size=84,          # Must match the `chunk_size` of the compiled TensorRT engine
)
wrapper = FoundationPoseWrapper(cfg=cfg)

# Set camera intrinsics (3x3 numpy array)
wrapper.set_camera_intrinsics(K)

# Load object mesh
mesh = FoundationPoseWrapper.load_mesh("path/to/mesh.obj")

# --- First frame ---
wrapper.reset_scene(color, depth)                      # color: (H,W,3) uint8, depth: (H,W) float32 in meters
pose = wrapper.add_object("object_name", mesh, mask)   # mask: (H,W) bool

# --- Subsequent frames ---
poses = wrapper.step_scene(color, depth)               # returns dict[name -> (4,4) numpy array]

# Visualize
vis = wrapper.render_results()   # returns BGR image with projected bounding box and axes
```

Poses are returned as 4x4 homogeneous transformation matrices (object-in-camera frame).

## Project structure

```
scripts/
  deps.sh           # Install/activate CUDA, cuDNN, TensorRT locally
  setup.sh          # Install Python dependencies
  convert_onnx.sh   # Download ONNX models and compile to TensorRT
src/foundationpose_tensorrt/
  wrapper.py        # High-level FoundationPoseWrapper API
  model.py          # TensorRT engine wrapper and FoundationposeModel
  postprocessor.py  # Rendering, cropping, and pose utilities
weights/
  onnx/             # Downloaded ONNX models
  tensorrt/         # Compiled TensorRT .plan files
demo.py             # Benchmark on YCB mustard data
```
