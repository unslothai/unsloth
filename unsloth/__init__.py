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
__version__ = "2023.12"
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

# We only support torch 2.1
# Fixes https://github.com/unslothai/unsloth/issues/38
torch_version = torch.__version__.split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch != 2) or (major_torch == 2 and minor_torch < 1):
    raise ImportError("Unsloth only supports Pytorch 2.1 for now. Please update your Pytorch to 2.1.\n"\
                      "We have some installation instructions on our Github page.")


# Try loading bitsandbytes and triton
import bitsandbytes as bnb
import triton
from triton.common.build import libcuda_dirs
try:
    cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
    libcuda_dirs()
except:
    warnings.warn(
        "CUDA is not linked properly.\n"\
        "We shall run `ldconfig /usr/lib64-nvidia` to try to fix it."
    )
    os.system("ldconfig /usr/lib64-nvidia")
    importlib.reload(bnb)
    importlib.reload(triton)
    try:
        import bitsandbytes as bnb
        from triton.common.build import libcuda_dirs
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        libcuda_dirs()
    except:
        raise ImportError("CUDA is not linked properly.\n"\
                          "We tried running `ldconfig /usr/lib64-nvidia` ourselves, but it didn't work.\n"\
                          "You need to run in your terminal `ldconfig /usr/lib64-nvidia` yourself, then import Unsloth.")
pass

from .models import *
