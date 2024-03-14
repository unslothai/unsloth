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

# Function to install packages via Conda
conda_install_packages () {
    conda create --name unsloth_env python=3.10 -y
    conda activate unsloth_env
    conda install pytorch cudatoolkit torchvision torchaudio pytorch-cuda=${CUDA_TAG} -c pytorch -c nvidia -y
    conda install xformers -c xformers -y
    pip install bitsandbytes
    pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"
}

# Function to install packages via Pip
pip_install_packages () {
    pip install --upgrade --force-reinstall --no-cache-dir torch==${PYTORCH_CORE_VERSION}+${CUDA_TAG} triton --index-url https://download.pytorch.org/whl/${CUDA_TAG}
    pip install "unsloth[${CUDA_TAG}] @ git+https://github.com/unslothai/unsloth.git"
}

# Check if conda is installed
if type conda &> /dev/null; then
    echo "Anaconda/Miniconda is installed, proceeding with Conda installation."

    # Determine CUDA version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA version detected: $CUDA_VERSION"
    
    # Choose the right tag for pytorch-cuda
    if [[ "$CUDA_VERSION" == "11.8" ]]; then
        CUDA_TAG="11.8"
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        CUDA_TAG="12.1"
    else
        echo "Unsupported CUDA version for Conda installation. Exiting."
        exit 1
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

        pip_install_packages

    else
        echo "CUDA not detected. Pip installation requires CUDA. Exiting."
        exit 1
    fi
fi
