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
__version__ = "2023.11"
import os
try:
	import torch
except:
	raise ImportError("Pytorch is not installed. Go to https://pytorch.org/.\n"\
					  "We have some installation instructions on our Github page.")

# We only support torch 2.1
major_torch, minor_torch, _ = torch.__version__.split(".")
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch != 2) or (major_torch == 2 and minor_torch < 1):
	raise ImportError("Unsloth only supports Pytorch 2.1 for now. Please update your Pytorch to 2.1.\n"\
					  "We have some installation instructions on our Github page.")

# Currently only supports 1 GPU, or else seg faults will occur.
reload_package = False
n_gpus = torch.cuda.device_count()
if n_gpus == 0:
	raise RuntimeError("Unsloth: Requires at least 1 GPU. Found 0.")
elif n_gpus > 1:
	if "CUDA_VISIBLE_DEVICES" in os.environ:
		device = os.environ["CUDA_VISIBLE_DEVICES"]
		if not device.isdigit():
			print(f"Unsloth: 'CUDA_VISIBLE_DEVICES' is currently {device} "\
				   "but we require 'CUDA_VISIBLE_DEVICES=0'\n"\
				   "We shall set it ourselves.")
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			reload_package = True
	else:
		print("Unsloth: 'CUDA_VISIBLE_DEVICES' is not set. We shall set it ourselves.")
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
		reload_package = True
pass

# Reload Pytorch with CUDA_VISIBLE_DEVICES
if reload_package:
	import importlib
	importlib.reload(torch)
pass

# Try loading bitsandbytes
import bitsandbytes as bnb
try:
	cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
except:
	raise ImportError("CUDA is not linked properly. Try running `ldconfig /usr/lib64-nvidia` first.")

from .models import *
