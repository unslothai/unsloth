#!/bin/bash

# Function to get PyTorch version if installed
get_pytorch_version () {
    if python -c "import torch" &> /dev/null; then
        PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        echo $PYTORCH_VERSION
    else
        echo "not installed"
    fi
}

# Function to get GPU architecture
get_gpu_type () {
    GPU_MAJOR_VERSION=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])")
    if [[ "$GPU_MAJOR_VERSION" -ge 8 ]]; then
        echo "ampere"
    else
        echo ""
    fi
}

# Function to install packages via Conda
conda_install_packages () {
    conda create --name unsloth_env python=3.10 -y
    CONDA_BASE=$(conda info --base)
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    if [[ -f "$CONDA_SH" ]]; then
        echo "Sourcing Conda from $CONDA_SH"
        source "$CONDA_SH"
    else
        echo "Unable to locate conda.sh at $CONDA_SH. Please ensure Conda is properly installed."
        exit 1
    fi
    conda activate unsloth_env
    conda install pytorch cudatoolkit=${CUDA_TAG} torchvision torchaudio pytorch-cuda=${CUDA_TAG} -c pytorch -c nvidia -y
    conda install xformers -c xformers -y
    pip install bitsandbytes
    pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"
}

# Function to install packages via Pip
pip_install_packages () {
    pip install --upgrade --force-reinstall --no-cache-dir torch==${PYTORCH_CORE_VERSION}+${CUDA_TAG} triton --index-url https://download.pytorch.org/whl/${CUDA_TAG}
    if [[ "$PYTORCH_VERSION_TAG" == "torch210" ]]; then
        pip install "unsloth[${CUDA_TAG}${GPU_TYPE:+-$GPU_TYPE}] @ git+https://github.com/unslothai/unsloth.git"
    else
        pip install "unsloth[${CUDA_TAG}${GPU_TYPE:+-$GPU_TYPE}-$PYTORCH_VERSION_TAG] @ git+https://github.com/unslothai/unsloth.git"
    fi
}

# Check if conda is installed
if type conda &> /dev/null; then
    echo "Anaconda/Miniconda is installed, proceeding with Conda installation."

    # Determine CUDA version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA version detected: $CUDA_VERSION"
    
    # Choose the right tag for pytorch-cuda
    # TODO: This is janky, we should find a better way to do this
    cuda_version_value=$(echo "$CUDA_VERSION" | bc)
    if [ "$(echo "$cuda_version_value < 12" | bc -l)" -eq 1 ]; then
        CUDA_TAG="11.8"
    else
        CUDA_TAG="12.1"
    fi

    conda_install_packages

# If conda is not installed, use pip
else
    echo "Anaconda/Miniconda is not installed, checking for CUDA and proceeding with Pip installation."

    # Check if CUDA is available
    if type nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo "CUDA version detected: $CUDA_VERSION"
        PYTORCH_VERSION=$(get_pytorch_version)
        echo "PyTorch version detected: $PYTORCH_VERSION"
        GPU_TYPE=$(get_gpu_type)
        if [[ $GPU_TYPE == "ampere" ]]; then
            echo "Ampere or newer architecture detected. Proceeding with ampere specific installation."
        else
            echo "Older GPU architecture detected. Proceeding with non-ampere specific installation."
        fi
        # Define CUDA tag based on CUDA version
        if [[ "$CUDA_VERSION" == "11.8" ]]; then
            CUDA_TAG="cu118"
        elif [[ "$CUDA_VERSION" == "12.1" ]]; then
            CUDA_TAG="cu121"
        else
            echo "Unsupported CUDA version for Pip installation. Exiting."
            exit 1
        fi

        # Extract PyTorch version (ignoring any suffix)
        PYTORCH_CORE_VERSION=$(echo $PYTORCH_VERSION | cut -d'+' -f1)
        PYTORCH_VERSION_TAG="torch${PYTORCH_CORE_VERSION//./}"

        pip_install_packages

    else
        echo "CUDA not detected. Pip installation requires CUDA. Exiting."
        exit 1
    fi
fi
