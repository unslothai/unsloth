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

# Currently only supports 1 GPU, or else seg faults will occur.
if "CUDA_VISIBLE_DEVICES" in os.environ:
    device = os.environ["CUDA_VISIBLE_DEVICES"]
    if not device.isdigit():
        warnings.warn(
            f"Unsloth: 'CUDA_VISIBLE_DEVICES' is currently {device} "\
             "but we require 'CUDA_VISIBLE_DEVICES=0'\n"\
             "We shall set it ourselves."
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif "CUDA_DEVICE_ORDER" not in os.environ:
        warnings.warn(
            f"Unsloth: 'CUDA_DEVICE_ORDER' is not set "\
             "but we require 'CUDA_DEVICE_ORDER=PCI_BUS_ID'\n"\
             "We shall set it ourselves."
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
else:
    # warnings.warn("Unsloth: 'CUDA_VISIBLE_DEVICES' is not set. We shall set it ourselves.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pass

try:
    import torch
except:
    raise ImportError("Pytorch is not installed. Go to https://pytorch.org/.\n"\
                      "We have some installation instructions on our Github page.")

# We support torch 2.1 and 2.1.1
# Fixes https://github.com/unslothai/unsloth/issues/38
torch_version = torch.__version__.split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch != 2):# or (major_torch == 2 and minor_torch < 1):
    raise ImportError("Unsloth only supports Pytorch 2.1 for now. Please update your Pytorch to 2.1.\n"\
                      "We have some installation instructions on our Github page.")


# Try loading bitsandbytes and triton
import bitsandbytes as bnb
import triton
from triton.common.build import libcuda_dirs
import os
import re
import numpy as np
import subprocess

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
        import bitsandbytes as bnb
        from triton.common.build import libcuda_dirs
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
