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
    return requirements


class RocmExtraInstallCommand(install):
    def run(self):

        if os.path.exists('thirdparties'):
            shutil.rmtree('thirdparties')

        os.mkdir('thirdparties')
        os.chdir('thirdparties')

        # xformers
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

setup(
    # static metadata should rather go in pyproject.toml
    version=get_unsloth_version(),
    install_requires=get_requirements(),
    cmdclass={
        'install': RocmExtraInstallCommand if UNSLOTH_TARGET_DEVICE == "rocm" else None,
    },
    package_data=package_data,
    
)
