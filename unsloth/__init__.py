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

import warnings, importlib, sys
from packaging.version import Version
import os, re, subprocess, inspect, functools
import numpy as np

# Log Unsloth is being used
os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Check if modules that need patching are already imported
critical_modules = ["trl", "transformers", "peft"]
already_imported = [mod for mod in critical_modules if mod in sys.modules]

# Fix some issues before importing other packages
from .import_fixes import (
    fix_message_factory_issue,
    check_fbgemm_gpu_version,
    torchvision_compatibility_check,
    fix_diffusers_warnings,
    fix_huggingface_hub,
)

fix_message_factory_issue()
check_fbgemm_gpu_version()
torchvision_compatibility_check()
fix_diffusers_warnings()
fix_huggingface_hub()
del fix_message_factory_issue
del check_fbgemm_gpu_version
del torchvision_compatibility_check
del fix_diffusers_warnings
del fix_huggingface_hub

# This check is critical because Unsloth optimizes these libraries by modifying
# their code at import time. If they're imported first, the original (slower,
# more memory-intensive) implementations will be used instead of Unsloth's
# optimized versions, potentially causing OOM errors or slower training.
if already_imported:
    # stacklevel=2 makes warning point to user's import line rather than this library code,
    # showing them exactly where to fix the import order in their script
    warnings.warn(
        f"WARNING: Unsloth should be imported before [{', '.join(already_imported)}] "
        f"to ensure all optimizations are applied. Your code may run slower or encounter "
        f"memory issues without these optimizations.\n\n"
        f"Please restructure your imports with 'import unsloth' at the top of your file.",
        stacklevel = 2,
    )
del already_imported, critical_modules

# Unsloth currently does not work on multi GPU setups - sadly we are a 2 brother team so
# enabling it will require much more work, so we have to prioritize. Please understand!
# We do have a beta version, which you can contact us about!
# Thank you for your understanding and we appreciate it immensely!

# Fixes https://github.com/unslothai/unsloth/issues/1266
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# [TODO] Check why some GPUs don't work
#    "pinned_use_cuda_host_register:True,"\
#    "pinned_num_register_threads:8"


from importlib.metadata import version as importlib_version
from importlib.metadata import PackageNotFoundError

# Check for unsloth_zoo
try:
    unsloth_zoo_version = importlib_version("unsloth_zoo")
    if Version(unsloth_zoo_version) < Version("2026.1.1"):
        print(
            "Unsloth: Please update Unsloth and Unsloth-Zoo to the latest version!\n"
            "Do this via `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo`"
        )
        # if os.environ.get("UNSLOTH_DISABLE_AUTO_UPDATES", "0") == "0":
        #     try:
        #         os.system("pip install --upgrade --no-cache-dir --no-deps unsloth_zoo")
        #     except:
        #         try:
        #             os.system("pip install --upgrade --no-cache-dir --no-deps --user unsloth_zoo")
        #         except:
        #             raise ImportError("Unsloth: Please update unsloth_zoo via `pip install --upgrade --no-cache-dir --no-deps unsloth_zoo`")
    import unsloth_zoo
except PackageNotFoundError:
    raise ImportError(
        f"Unsloth: Please install unsloth_zoo via `pip install unsloth_zoo` then retry!"
    )
except:
    raise
del PackageNotFoundError, importlib_version

# Try importing PyTorch and check version
try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"
        "We have some installation instructions on our Github page."
    )
except:
    raise

from unsloth_zoo.device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)

# Fix other issues
from .import_fixes import (
    fix_xformers_performance_issue,
    fix_vllm_aimv2_issue,
    fix_vllm_guided_decoding_params,
    ignore_logger_messages,
    patch_ipykernel_hf_xet,
    patch_trackio,
    patch_datasets,
    patch_enable_input_require_grads,
    fix_openenv_no_vllm,
    fix_executorch,
)

fix_xformers_performance_issue()
fix_vllm_aimv2_issue()
fix_vllm_guided_decoding_params()
ignore_logger_messages()
patch_ipykernel_hf_xet()
patch_trackio()
patch_datasets()
patch_enable_input_require_grads()
fix_openenv_no_vllm()
fix_executorch()

del fix_xformers_performance_issue
del fix_vllm_aimv2_issue
del fix_vllm_guided_decoding_params
del ignore_logger_messages
del patch_ipykernel_hf_xet
del patch_trackio
del patch_datasets
del patch_enable_input_require_grads
del fix_openenv_no_vllm
del fix_executorch

# Torch 2.4 has including_emulation
if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = major_version >= 8

    old_is_bf16_supported = torch.cuda.is_bf16_supported
    if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):

        def is_bf16_supported(including_emulation = False):
            return old_is_bf16_supported(including_emulation)

        torch.cuda.is_bf16_supported = is_bf16_supported
    else:

        def is_bf16_supported():
            return SUPPORTS_BFLOAT16

        torch.cuda.is_bf16_supported = is_bf16_supported
    del major_version, minor_version
elif DEVICE_TYPE == "hip":
    SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
elif DEVICE_TYPE == "xpu":
    # torch.xpu.is_bf16_supported() does not have including_emulation
    # set SUPPORTS_BFLOAT16 as torch.xpu.is_bf16_supported()
    SUPPORTS_BFLOAT16 = torch.xpu.is_bf16_supported()

# For Gradio HF Spaces?
# if "SPACE_AUTHOR_NAME" not in os.environ and "SPACE_REPO_NAME" not in os.environ:
import triton

if DEVICE_TYPE == "cuda":
    libcuda_dirs = lambda: None
    if Version(triton.__version__) >= Version("3.0.0"):
        try:
            from triton.backends.nvidia.driver import libcuda_dirs
        except:
            pass
    else:
        from triton.common.build import libcuda_dirs

    # Try loading bitsandbytes and triton
    try:
        import bitsandbytes as bnb
    except:
        print(
            "Unsloth: `bitsandbytes` is not installed - 4bit QLoRA unallowed, but 16bit and full finetuning works!"
        )
        bnb = None
    try:
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        libcuda_dirs()
    except:
        warnings.warn("Unsloth: Running `ldconfig /usr/lib64-nvidia` to link CUDA.")

        if os.path.exists("/usr/lib64-nvidia"):
            os.system("ldconfig /usr/lib64-nvidia")
        elif os.path.exists("/usr/local"):
            # Sometimes bitsandbytes cannot be linked properly in Runpod for example
            possible_cudas = (
                subprocess.check_output(["ls", "-al", "/usr/local"])
                .decode("utf-8")
                .split("\n")
            )
            find_cuda = re.compile(r"[\s](cuda\-[\d\.]{2,})$")
            possible_cudas = [find_cuda.search(x) for x in possible_cudas]
            possible_cudas = [x.group(1) for x in possible_cudas if x is not None]

            # Try linking cuda folder, or everything in local
            if len(possible_cudas) == 0:
                os.system("ldconfig /usr/local/")
            else:
                find_number = re.compile(r"([\d\.]{2,})")
                latest_cuda = np.argsort(
                    [float(find_number.search(x).group(1)) for x in possible_cudas]
                )[::-1][0]
                latest_cuda = possible_cudas[latest_cuda]
                os.system(f"ldconfig /usr/local/{latest_cuda}")
                del find_number, latest_cuda
            del possible_cudas, find_cuda

        if bnb is not None:
            importlib.reload(bnb)
        importlib.reload(triton)
        try:
            libcuda_dirs = lambda: None
            if Version(triton.__version__) >= Version("3.0.0"):
                try:
                    from triton.backends.nvidia.driver import libcuda_dirs
                except:
                    pass
            else:
                from triton.common.build import libcuda_dirs
            cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
            libcuda_dirs()
        except:
            warnings.warn(
                "Unsloth: CUDA is not linked properly.\n"
                "Try running `python -m bitsandbytes` then `python -m xformers.info`\n"
                "We tried running `ldconfig /usr/lib64-nvidia` ourselves, but it didn't work.\n"
                "You need to run in your terminal `sudo ldconfig /usr/lib64-nvidia` yourself, then import Unsloth.\n"
                "Also try `sudo ldconfig /usr/local/cuda-xx.x` - find the latest cuda version.\n"
                "Unsloth will still run for now, but maybe it might crash - let's hope it works!"
            )
    del libcuda_dirs
elif DEVICE_TYPE == "hip":
    # NO-OP for rocm device
    pass
elif DEVICE_TYPE == "xpu":
    import bitsandbytes as bnb

    # TODO: check triton for intel installed properly.
    pass

from .models import *
from .models import __version__
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *
from unsloth_zoo.rl_environments import (
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
    Benchmarker,
    is_port_open,
    launch_openenv,
)

# Patch TRL trainers for backwards compatibility
_patch_trl_trainer()
