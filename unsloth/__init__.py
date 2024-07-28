# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import warnings
import importlib
import sys
from packaging.version import Version

# # Define a list of modules to check
# MODULES_TO_CHECK = ["bitsandbytes"]

# # Check if any of the modules in the list have been imported
# for module in MODULES_TO_CHECK:
#     if module in sys.modules:
#         raise ImportError(f"Unsloth: Please import Unsloth before {module}.")
#     pass
# pass

# Unsloth currently does not work on multi GPU setups - sadly we are a 2 brother team so
# enabling it will require much more work, so we have to prioritize. Please understand!
# We do have a beta version, which you can contact us about!
# Thank you for your understanding and we appreciate it immensely!
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = os.environ["CUDA_VISIBLE_DEVICES"]
    # Check if there are multiple cuda devices set in env
    if not devices.isdigit():
        first_id = devices.split(",")[0]
        warnings.warn(
            f"Unsloth: 'CUDA_VISIBLE_DEVICES' is currently {devices} \n"\
            "Unsloth currently does not support multi GPU setups - but we are working on it!\n"\
            "Multiple CUDA devices detected but we require a single device.\n"\
            f"We will override CUDA_VISIBLE_DEVICES to first device: {first_id}."
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(first_id)
else:
    # warnings.warn("Unsloth: 'CUDA_VISIBLE_DEVICES' is not set. We shall set it ourselves.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pass

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import torch
except:
    raise ImportError("Pytorch is not installed. Go to https://pytorch.org/.\n"\
                      "We have some installation instructions on our Github page.")
pass

import os, re
import numpy as np
import subprocess

# Hugging Face Hub faster downloads (only enable during Colab and Kaggle sessions)
keynames = "\n" + "\n".join(os.environ.keys())
if "\nCOLAB_"  in keynames or "\nKAGGLE_" in keynames:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
pass

# We support Pytorch 2
# Fixes https://github.com/unslothai/unsloth/issues/38
torch_version = torch.__version__.split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch < 2):
    raise ImportError("Unsloth only supports Pytorch 2 for now. Please update your Pytorch to 2.1.\n"\
                      "We have some installation instructions on our Github page.")
elif (major_torch == 2) and (minor_torch < 2):
    # Disable expandable_segments
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
pass

# Torch 2.5 has including_emulation
major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = (major_version >= 8)

if (major_torch == 2) and (minor_torch >= 5): 
    old_is_bf16_supported = torch.cuda.is_bf16_supported
    def is_bf16_supported(including_emulation = False):
        return old_is_bf16_supported(including_emulation)
    torch.cuda.is_bf16_supported = is_bf16_supported
else:
    def is_bf16_supported(): return SUPPORTS_BFLOAT16
    torch.cuda.is_bf16_supported = is_bf16_supported
pass

# Try loading bitsandbytes and triton
import bitsandbytes as bnb

import triton
libcuda_dirs = lambda: None
if Version(triton.__version__) >= Version("3.0.0"):
    try: from triton.backends.nvidia.driver import libcuda_dirs
    except: pass
else: from triton.common.build import libcuda_dirs

try:
    cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
    libcuda_dirs()
except:
    warnings.warn(
        "Unsloth: Running `ldconfig /usr/lib64-nvidia` to link CUDA."\
    )

    if os.path.exists("/usr/lib64-nvidia"):
        os.system("ldconfig /usr/lib64-nvidia")
    elif os.path.exists("/usr/local"):
        # Sometimes bitsandbytes cannot be linked properly in Runpod for example
        possible_cudas = subprocess.check_output(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
        find_cuda = re.compile(r"[\s](cuda\-[\d\.]{2,})$")
        possible_cudas = [find_cuda.search(x) for x in possible_cudas]
        possible_cudas = [x.group(1) for x in possible_cudas if x is not None]

        # Try linking cuda folder, or everything in local
        if len(possible_cudas) == 0:
            os.system(f"ldconfig /usr/local/")
        else:
            find_number = re.compile(r"([\d\.]{2,})")
            latest_cuda = np.argsort([float(find_number.search(x).group(1)) for x in possible_cudas])[::-1][0]
            latest_cuda = possible_cudas[latest_cuda]
            os.system(f"ldconfig /usr/local/{latest_cuda}")
    pass

    importlib.reload(bnb)
    importlib.reload(triton)
    try:
        libcuda_dirs = lambda: None
        if Version(triton.__version__) >= Version("3.0.0"):
            try: from triton.backends.nvidia.driver import libcuda_dirs
            except: pass
        else: from triton.common.build import libcuda_dirs
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        libcuda_dirs()
    except:
        warnings.warn(
            "Unsloth: CUDA is not linked properly.\n"\
            "Try running `python -m bitsandbytes` then `python -m xformers.info`\n"\
            "We tried running `ldconfig /usr/lib64-nvidia` ourselves, but it didn't work.\n"\
            "You need to run in your terminal `sudo ldconfig /usr/lib64-nvidia` yourself, then import Unsloth.\n"\
            "Also try `sudo ldconfig /usr/local/cuda-xx.x` - find the latest cuda version.\n"\
            "Unsloth will still run for now, but maybe it might crash - let's hope it works!"
        )
pass

from .models import *
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *
