#!/bin/bash

function activate_deps()
{
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    INSTALL_DIR="${SCRIPT_DIR}/../deps"

    CUDA_12_4_CUDNN_9_8_HOME="$INSTALL_DIR/cuda-12-4"
    CUDA_12_4_CUDNN_9_8_ROOT="$CUDA_12_4_CUDNN_9_8_HOME/bin"
    CUDA_12_4_CUDNN_9_8_TOOLKIT_ROOT_DIR="$CUDA_12_4_CUDNN_9_8_HOME"
    CUDA_12_4_CUDNN_9_8_PATH="$CUDA_12_4_CUDNN_9_8_HOME/bin"
    CUDA_12_4_CUDNN_9_8_LD_LIBRARY_PATH="$CUDA_12_4_CUDNN_9_8_HOME/lib"
    TENSORRT_10_9_0_HOME="$INSTALL_DIR/TensorRT-10.9.0.34"
    TENSORRT_10_9_0_BIN_PATH="$TENSORRT_10_9_0_HOME/bin"
    TENSORRT_10_9_0_LIB_PATH="$TENSORRT_10_9_0_HOME/lib"

    cuda_version="12.4"
    cudnn_version="9.8.0"
    cuda_version_full="$cuda_version.0"
    cuda_run_file="cuda_${cuda_version_full}_550.54.14_linux.run"
    cudnn_tar_file="cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz"
    cuda_url="https://developer.download.nvidia.com/compute/cuda/$cuda_version_full/local_installers/$cuda_run_file"
    cudnn_url="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/$cudnn_tar_file"
    # Where install files will be placed
    cuda_installer="$INSTALL_DIR/$cuda_run_file"
    cudnn_pkg="$INSTALL_DIR/$cudnn_tar_file"

    # Flag-files to indicate installation status.
    cuda_done="$CUDA_12_4_CUDNN_9_8_HOME/installed_cuda"
    cudnn_done="$CUDA_12_4_CUDNN_9_8_HOME/installed_cudnn"

    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CUDA_12_4_CUDNN_9_8_HOME"

    if [[ ! -f "$cuda_done" ]]; then
        wget $cuda_url -O $cuda_installer
        echo "Installing CUDA 12.4 ('$cuda_installer') to '$CUDA_12_4_CUDNN_9_8_HOME'.  This might take several minutes."
        sh "$cuda_installer" --silent --defaultroot="$CUDA_12_4_CUDNN_9_8_HOME" --toolkit --toolkitpath="$CUDA_12_4_CUDNN_9_8_HOME" --no-man-page --override
        # Install twice to make sure symlinks work correctly...
        echo "Verifying CUDA 12.4 installation.  This might take several minutes."
        sh "$cuda_installer" --silent --defaultroot="$CUDA_12_4_CUDNN_9_8_HOME" --toolkit --toolkitpath="$CUDA_12_4_CUDNN_9_8_HOME" --no-man-page --override && touch "$cuda_done"
        rm $cuda_installer
        echo "Done."
    fi

    if [[ ! -f "$cudnn_done" ]]; then
        wget $cudnn_url -O $cudnn_pkg
        echo "Installing cuDNN 9.8 ('$cudnn_pkg') to '$CUDA_12_4_CUDNN_9_8_HOME'."
        tar -xf "$cudnn_pkg" --strip-components=1 -C "$CUDA_12_4_CUDNN_9_8_HOME" && touch "$cudnn_done"
        # cudnn overwrites the include folder and creates new lib folder
        cp -Rs "$CUDA_12_4_CUDNN_9_8_HOME/targets/x86_64-linux/lib/"* "$CUDA_12_4_CUDNN_9_8_HOME/lib" 2> /dev/null || true
        cp -Rs "$CUDA_12_4_CUDNN_9_8_HOME/targets/x86_64-linux/include/"* "$CUDA_12_4_CUDNN_9_8_HOME/include" 2> /dev/null || true
        rm $cudnn_pkg
        echo "Done."
    fi

    export PRE_CUDA_12_4_CUDNN_9_8_PATH="$PATH"
    export PRE_CUDA_12_4_CUDNN_9_8_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

    export CUDA_HOME="$CUDA_12_4_CUDNN_9_8_HOME"
    export CUDA_ROOT="$CUDA_12_4_CUDNN_9_8_ROOT"
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_12_4_CUDNN_9_8_TOOLKIT_ROOT_DIR"

    export PATH="$CUDA_12_4_CUDNN_9_8_PATH:$PATH"
    export LD_LIBRARY_PATH="$CUDA_12_4_CUDNN_9_8_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

    tensorrt_tar_file="TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz"
    tensorrt_url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/$tensorrt_tar_file"
    tensorrt_pkg="$INSTALL_DIR/$tensorrt_tar_file"
    tensorrt_done="$TENSORRT_10_9_0_HOME/installed_tensorrt"

    mkdir -p "$TENSORRT_10_9_0_HOME"
    if [[ ! -f "$tensorrt_done" ]]; then
        wget $tensorrt_url -O $tensorrt_pkg
        echo "Installing TensorRT 10.9.0 ('$tensorrt_pkg') to '$TENSORRT_10_9_0_HOME'."
        tar -xf "$tensorrt_pkg" -C "$INSTALL_DIR" && touch "$tensorrt_done"
        rm $tensorrt_pkg
        echo "Done."
    fi

    export TENSORRT_HOME="$TENSORRT_10_9_0_HOME"
    export PATH="$TENSORRT_10_9_0_BIN_PATH:$PATH"
    export LD_LIBRARY_PATH="$TENSORRT_10_9_0_LIB_PATH:$LD_LIBRARY_PATH"
}


function deactivate_deps()
{
    export PATH="$PRE_CUDA_12_4_CUDNN_9_8_PATH"
    export LD_LIBRARY_PATH="$PRE_CUDA_12_4_CUDNN_9_8_LD_LIBRARY_PATH"

    unset PRE_CUDA_12_4_CUDNN_9_8_PATH
    unset PRE_CUDA_12_4_CUDNN_9_8_LD_LIBRARY_PATH

    unset CUDA_HOME
    unset CUDA_ROOT
    unset CUDA_TOOLKIT_ROOT_DIR
}
