# Copid and modified based on https://github.com/vllm-project/vllm/blob/main/setup.py
# SPDX-License-Identifier: Apache-2.0

import ctypes
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
import shutil

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

from setuptools.command.install import install

MAIN_CUDA_VERSION = "12.8"

UNSLOTH_TARGET_DEVICE = os.environ.get('UNSLOTH_TARGET_DEVICE', 'rocm')

ROOT_DIR = Path(__file__).parent


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return UNSLOTH_TARGET_DEVICE == "cuda" and has_cuda


def _is_hip() -> bool:
    return (UNSLOTH_TARGET_DEVICE == "cuda"
            or UNSLOTH_TARGET_DEVICE == "rocm") and torch.version.hip is not None


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_rocm_version():
    # Get the Rocm version from the ROCM_HOME/bin/librocm-core.so
    # see https://github.com/ROCm/rocm-core/blob/d11f5c20d500f729c393680a01fa902ebf92094b/rocm_version.cpp#L21
    try:
        librocm_core_file = Path(ROCM_HOME) / "lib" / "librocm-core.so"
        if not librocm_core_file.is_file():
            return None
        librocm_core = ctypes.CDLL(librocm_core_file)
        VerErrors = ctypes.c_uint32
        get_rocm_core_version = librocm_core.getROCmVersion
        get_rocm_core_version.restype = VerErrors
        get_rocm_core_version.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()

        if (get_rocm_core_version(ctypes.byref(major), ctypes.byref(minor),
                                  ctypes.byref(patch)) == 0):
            return f"{major.value}.{minor.value}.{patch.value}"
        return None
    except Exception:
        return None


def get_unsloth_version() -> str:
    # TODO: need to remove magic number 
    # import unsloth.models._utils as unsloth_utils
    # version = unsloth_utils.__version__
    version = "2025.3.19"
    if version is None:
        raise RuntimeError("unsloth version not found")

    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _is_cuda():
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            # skip this for source tarball, required for pypi
            if "sdist" not in sys.argv:
                version += f"{sep}cu{cuda_version_str}"
    elif _is_hip():
        # Get the Rocm Version
        rocm_version = get_rocm_version() or torch.version.hip
        if rocm_version and rocm_version != MAIN_CUDA_VERSION:
            version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version

def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif not line.startswith("--") and not line.startswith(
                    "#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    if _is_cuda():
        requirements = _read_requirements("cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if ("vllm-flash-attn" in req and cuda_major != "12"):
                # vllm-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements
    elif _is_hip():
        requirements = _read_requirements("rocm.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, "
            "or CPU.")

    print(f"bill-dbg: requirements: {requirements}")
    return requirements


class RocmExtraInstallCommand(install):
    def run(self):

        if os.path.exists('thirdparties'):
            shutil.rmtree('thirdparties')

        os.mkdir('thirdparties')
        os.chdir('thirdparties')

        # # xformers
        subprocess.check_call(['git', 'clone', 'https://github.com/ROCm/xformers.git'])
        os.chdir('xformers')
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'
        subprocess.check_call(['python', 'setup.py', 'install'])
        os.chdir('..')

        # bitsandbytes
        subprocess.check_call(['git', 'clone', '--recurse-submodules', 'https://github.com/ROCm/bitsandbytes'])
        os.chdir('bitsandbytes')
        subprocess.check_call(['git', 'checkout', 'rocm_enabled_multi_backend'])
        subprocess.check_call(['pip', 'install', '-r', 'requirements-dev.txt'])
        subprocess.check_call(['cmake', '-DCOMPUTE_BACKEND=hip', '-S', '.'])  # Add -DBNB_ROCM_ARCH if needed
        subprocess.check_call(['make'])
        subprocess.check_call(['pip', 'install', '.'])
        os.chdir('..')

        # flash-attention
        subprocess.check_call(['git', 'clone', '--recursive', 'https://github.com/ROCm/flash-attention.git'])
        os.chdir('flash-attention')
        num_jobs = os.cpu_count() - 1
        subprocess.check_call(['pip', 'install', '-v', '.', f'MAX_JOBS={num_jobs}'], shell=True)
        os.chdir('../..')

        # Continue with regular install
        install.run(self)

package_data = {
    "unsloth": [
        "py.typed",
    ]
}

extras_require = {
    "triton": [
        "triton-windows ; platform_system == 'Windows'",
    ],

    "huggingface": [
        "unsloth_zoo>=2025.5.1",
        "packaging",
        "tyro",
        "transformers==4.51.3,!=4.47.0",
        "datasets>=3.4.1",
        "sentencepiece>=0.2.0",
        "tqdm",
        "psutil",
        "wheel>=0.42.0",
        "numpy",
        "accelerate>=0.34.1",
        "trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3,!=0.15.0,<=0.15.2",
        "peft>=0.7.1,!=0.11.0",
        "protobuf<4.0.0",
        "huggingface_hub",
        "hf_transfer",
        "unsloth[triton]",
    ],

    "windows":[
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5 ; platform_system == 'Windows'",
        "xformers>=0.0.22.post7 ; platform_system == 'Windows'",
    ],
    "cu118only": [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],

    "cu121only" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu118onlytorch211" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu121onlytorch211" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu118onlytorch212" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu121onlytorch212" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23.post1-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23.post1-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23.post1-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu118onlytorch220" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.24%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.24%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.24%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu121onlytorch220" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
    ],
    "cu118onlytorch230" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp312-cp312-manylinux2014_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu121onlytorch230" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp312-cp312-manylinux2014_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu118onlytorch240" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp312-cp312-manylinux2014_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu121onlytorch240" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu124onlytorch240" : [
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu118onlytorch250" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.28.post2-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.28.post2-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.28.post2-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.28.post2-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu121onlytorch250" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu124onlytorch250" : [
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu118onlytorch251" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu121onlytorch251" : [
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu124onlytorch251" : [
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu118onlytorch260" : [
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post3-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post3-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.29.post3-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
    ],
    "cu124onlytorch260" : [
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post3-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu126onlytorch260" : [
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.29.post3-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu126onlytorch270" : [
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu126/xformers-0.0.30-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu128onlytorch270" : [
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
    ],
    "cu118" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118only]",
    ],
    "cu121" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121only]",
    ],
    "cu118-torch211" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu118onlytorch211]",
    ],
    "cu121-torch211" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu121onlytorch211]",
    ],
    "cu118-torch212" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu118onlytorch212]",
    ],
    "cu121-torch212" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu121onlytorch212]",
    ],
    "cu118-torch220" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch220]",
    ],
    "cu121-torch220" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch220]",
    ],
    "cu118-torch230" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch230]",
    ],
    "cu121-torch230" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch230]",
    ],
    "cu118-torch240" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch240]",
    ],
    "cu121-torch240" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch240]",
    ],
    "cu124-torch240" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch240]",
    ],
    "cu118-torch250" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch250]",
    ],
    "cu121-torch250" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch250]",
    ],
    "cu124-torch250" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch250]",
    ],
    "cu118-torch251" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch251]",
    ],
    "cu121-torch251" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch251]",
    ],
    "cu124-torch251" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch251]",
    ],
    "cu118-torch260" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch260]",
    ],
    "cu124-torch260" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch260]",
    ],
    "cu126-torch260" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu126onlytorch260]",
    ],
    "cu126-torch270" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu126onlytorch270]",
    ],
    "cu128-torch270" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu128onlytorch270]",
    ],
    "kaggle" : [
        "unsloth[huggingface]",
    ],
    "kaggle-new" :[
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
    ],
    "conda" : [
        "unsloth[huggingface]",
    ],
    "colab-torch211" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu121onlytorch211]",
    ],
    "colab-ampere-torch211" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu121onlytorch211]",
        "packaging",
        "ninja",
        "flash-attn>=2.6.3",
    ],
    "colab-torch220" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch220]",
    ],
    "colab-ampere-torch220" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch220]",
        "packaging",
        "ninja",
        "flash-attn>=2.6.3",
    ],
    "colab-new" : [
        "unsloth_zoo>=2025.5.1",
        "packaging",
        "tyro",
        "transformers==4.51.3,!=4.47.0",
        "datasets>=3.4.1",
        "sentencepiece>=0.2.0",
        "tqdm",
        "psutil",
        "wheel>=0.42.0",
        "numpy",
        "protobuf<4.0.0",
        "huggingface_hub",
        "hf_transfer",
        "bitsandbytes>=0.45.5",
        "unsloth[triton]",
    ],
    "colab-no-deps" : [
        "accelerate>=0.34.1",
        "trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3,!=0.15.0,<=0.15.2",
        "peft>=0.7.1",
        "xformers",
        "bitsandbytes>=0.45.5",
        "protobuf<4.0.0",
    ],
    "colab" : [
        "unsloth[cu121]",
    ],
    "flashattention" : [
        "packaging ; platform_system == 'Linux'",
        "ninja ; platform_system == 'Linux'",
        "flash-attn>=2.6.3 ; platform_system == 'Linux'",
    ],
    "colab-ampere" : [
        "unsloth[colab-ampere-torch220]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118only]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121only]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch211" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu118onlytorch211]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere-torch211" : [
        "unsloth[huggingface]",
        "bitsandbytes==0.45.5",
        "unsloth[cu121onlytorch211]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch220" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch220]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere-torch220" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch220]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch230" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch230]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere-torch230" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch230]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch240" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch240]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere-torch240" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch240]",
        "unsloth[flashattention]",
    ],
    "cu124-ampere-torch240" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch240]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch250" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch250]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere-torch250" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch250]",
        "unsloth[flashattention]",
    ],
    "cu124-ampere-torch250" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch250]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch251" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch251]",
        "unsloth[flashattention]",
    ],
    "cu121-ampere-torch251" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu121onlytorch251]",
        "unsloth[flashattention]",
    ],
    "cu124-ampere-torch251" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch251]",
        "unsloth[flashattention]",
    ],
    "cu118-ampere-torch260" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu118onlytorch260]",
        "unsloth[flashattention]",
    ],
    "cu124-ampere-torch260" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu124onlytorch260]",
        "unsloth[flashattention]",
    ],
    "cu126-ampere-torch260" :[
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu126onlytorch260]",
        "unsloth[flashattention]",
    ],
    "cu126-ampere-torch270" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu126onlytorch270]",
        "unsloth[flashattention]",
    ],
    "cu128-ampere-torch270" : [
        "unsloth[huggingface]",
        "bitsandbytes>=0.45.5",
        "unsloth[cu128onlytorch270]",
        "unsloth[flashattention]",
    ],

    "intel-gpu-torch260" : [
        "unsloth[huggingface]",

        "pytorch_triton_xpu @ https://download.pytorch.org/whl/pytorch_triton_xpu-3.2.0-cp39-cp39-linux_x86_64.whl#sha256=147607f190a7d7aa24ba454def5977fbbfec792fdae18e4ed278cfec29b69271 ; platform_system == 'Linux' and python_version == '3.9' and platform_machine == 'x86_64'",
        "pytorch_triton_xpu @ https://download.pytorch.org/whl/pytorch_triton_xpu-3.2.0-cp310-cp310-linux_x86_64.whl#sha256=23aa423fa1542afc34f67eb3ba8ef20060f6d1b3a4697eaeab22b11c92b30f2b ; platform_system == 'Linux' and python_version == '3.10' and platform_machine == 'x86_64'",
        "pytorch_triton_xpu @ https://download.pytorch.org/whl/pytorch_triton_xpu-3.2.0-cp311-cp311-linux_x86_64.whl#sha256=bcfa995229bbfd9ffd8d6c8d9f6428d393e876fa6e23ee3c20e3c0d73ca75ca5 ; platform_system == 'Linux' and python_version == '3.11' and platform_machine == 'x86_64'",
        "pytorch_triton_xpu @ https://download.pytorch.org/whl/pytorch_triton_xpu-3.2.0-cp312-cp312-linux_x86_64.whl#sha256=bd340903d03470708df3442438acb8b7e08087ab9e61fbe349b2872bf9257ab0 ; platform_system == 'Linux' and python_version == '3.12' and platform_machine == 'x86_64'",
        "pytorch_triton_xpu @ https://download.pytorch.org/whl/pytorch_triton_xpu-3.2.0-cp313-cp313-linux_x86_64.whl#sha256=814dccc8a07159e6eca74bed70091bc8fea2d9dd87b0d91845f9f38cde62f01c ; platform_system == 'Linux' and python_version == '3.13' and platform_machine == 'x86_64'",

        "torch @ https://download.pytorch.org/whl/xpu/torch-2.6.0%2Bxpu-cp39-cp39-linux_x86_64.whl#sha256=6a8adf6dc4c089406e8b3a7e58ab57a463bddf9b07130d2576e76eced43e92af ; platform_system == 'Linux' and python_version == '3.9' and platform_machine == 'x86_64'",
        "torch @ https://download.pytorch.org/whl/xpu/torch-2.6.0%2Bxpu-cp310-cp310-linux_x86_64.whl#sha256=ff4561cbf07c83bbccaa0f6e9bb0e6dcf721bacd53c9c43c4eb0e7331b4792f9 ; platform_system == 'Linux' and python_version == '3.10' and platform_machine == 'x86_64'",
        "torch @ https://download.pytorch.org/whl/xpu/torch-2.6.0%2Bxpu-cp311-cp311-linux_x86_64.whl#sha256=12005f66b810ddd3ab93f86c4522bcfdd412cbd27fc9d189b661ff7509bc5e8a ; platform_system == 'Linux' and python_version == '3.11' and platform_machine == 'x86_64'",
        "torch @ https://download.pytorch.org/whl/xpu/torch-2.6.0%2Bxpu-cp312-cp312-linux_x86_64.whl#sha256=c4c5c67625cdacf35765c2b94e61fe166e3c3f4a14521b1212a59ad1b3eb0f2e ; platform_system == 'Linux' and python_version == '3.12' and platform_machine == 'x86_64'",
        "torch @ https://download.pytorch.org/whl/xpu/torch-2.6.0%2Bxpu-cp313-cp313-linux_x86_64.whl#sha256=e6864f7a60a5ecc43d5d38f59a16e5dd132384f73dfd3a697f74944026038f7b ; platform_system == 'Linux' and python_version == '3.13' and platform_machine == 'x86_64'",
    ],
}

setup(
    # static metadata should rather go in pyproject.toml
    version=get_unsloth_version(),
    install_requires=get_requirements(),
    extras_require=extras_require,
    cmdclass={
        'install': RocmExtraInstallCommand if UNSLOTH_TARGET_DEVICE == "rocm" else None,
    },
    package_data=package_data,
    
)
