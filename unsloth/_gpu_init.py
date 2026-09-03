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

os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Modules that need patching but may already be imported
critical_modules = ["trl", "transformers", "peft"]
already_imported = [mod for mod in critical_modules if mod in sys.modules]

from .import_fixes import (
    fix_message_factory_issue,
    fix_torch_check_is_size,
    fix_torchao_torch_symbol_skew,
    propagate_torchao_fix_to_subprocesses,
    check_fbgemm_gpu_version,
    check_transformers_dependency_versions,
    disable_broken_causal_conv1d,
    disable_broken_vllm,
    configure_amdgpu_asic_id_table_path,
    fix_bitsandbytes_rocm_arch_detection,
    torchvision_compatibility_check,
    disable_torchaudio_if_cuda_mismatched,
    fix_diffusers_warnings,
    fix_huggingface_hub,
)

# Redirect a read-only Hugging Face cache before anything below imports huggingface_hub /
# transformers / vllm (disable_broken_vllm probes `import vllm`, check_fbgemm_gpu_version imports
# transformers, fix_huggingface_hub imports huggingface_hub), any of which would freeze Hub's
# cache constants with the un-redirected paths. hf_cache.py is stdlib-only, so it loads straight
# from its file without triggering the full unsloth_zoo init this early.
try:
    import importlib.util as _importlib_util
    from pathlib import Path as _Path

    _zoo_spec = _importlib_util.find_spec("unsloth_zoo")
    if _zoo_spec is not None and _zoo_spec.origin:
        _hf_cache_file = _Path(_zoo_spec.origin).with_name("hf_cache.py")
        if _hf_cache_file.is_file():
            _hf_cache_spec = _importlib_util.spec_from_file_location(
                "unsloth_zoo._early_hf_cache", _hf_cache_file
            )
            _hf_cache = _importlib_util.module_from_spec(_hf_cache_spec)
            _hf_cache_spec.loader.exec_module(_hf_cache)
            _hf_cache.redirect_hf_cache_if_readonly()
            del _hf_cache, _hf_cache_spec
        del _hf_cache_file
    del _zoo_spec, _importlib_util, _Path
except Exception:
    pass

# Configure libdrm ids table path early so ROCm can resolve AMD GPU names.
configure_amdgpu_asic_id_table_path()
# Must precede `import unsloth_zoo` below, which imports bnb on ROCm.
fix_bitsandbytes_rocm_arch_detection()
disable_broken_causal_conv1d()
disable_broken_vllm()
fix_message_factory_issue()
fix_torch_check_is_size()
fix_torchao_torch_symbol_skew()
# The above fixes THIS process only; vLLM's model-architecture inspector is a subprocess that
# imports torchao itself and hits the same ImportError.
propagate_torchao_fix_to_subprocesses()
# Warn, do not raise: this only adds the correct remedy just before transformers prints its misleading one.
check_transformers_dependency_versions()
check_fbgemm_gpu_version()
torchvision_compatibility_check()
# Must precede `import unsloth_zoo`: its temporary_patches reach transformers.processing_utils ->
# audio_utils -> torchaudio, so a torchaudio raising at extension init takes the whole unsloth
# import down before the late guard would have neutralised it.
disable_torchaudio_if_cuda_mismatched()
fix_diffusers_warnings()
fix_huggingface_hub()
del configure_amdgpu_asic_id_table_path
del fix_bitsandbytes_rocm_arch_detection
del disable_broken_causal_conv1d
del disable_broken_vllm
del fix_message_factory_issue
del fix_torch_check_is_size
del fix_torchao_torch_symbol_skew
del propagate_torchao_fix_to_subprocesses
del check_fbgemm_gpu_version
del check_transformers_dependency_versions
del torchvision_compatibility_check
del fix_diffusers_warnings
del fix_huggingface_hub

# Unsloth patches these libraries at import time; if they are imported first the unoptimized
# versions run, risking OOM or slower training.
if already_imported:
    # stacklevel=2 points the warning at the user's import line
    warnings.warn(
        f"WARNING: Unsloth should be imported before [{', '.join(already_imported)}] "
        f"to ensure all optimizations are applied. Your code may run slower or encounter "
        f"memory issues without these optimizations.\n\n"
        f"Please restructure your imports with 'import unsloth' at the top of your file.",
        stacklevel = 2,
    )
del already_imported, critical_modules

# Pin BNB_ROCM_VERSION before bitsandbytes is first imported (`import unsloth_zoo` below pulls it in on ROCm hosts).
from .import_fixes import maybe_set_windows_rocm_bnb_version

maybe_set_windows_rocm_bnb_version()
del maybe_set_windows_rocm_bnb_version

# Multi-GPU is not yet supported (beta available on request).

# Fixes https://github.com/unslothai/unsloth/issues/1266
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


from importlib.metadata import version as importlib_version
from importlib.metadata import PackageNotFoundError


def _nvidia_smi_gpu_name():
    """The first GPU nvidia-smi reports, or None when it cannot answer.

    Returns instead of raising so the caller re-raises the original error OUTSIDE this
    handler. Raising in here chains the probe's FileNotFoundError onto the device error,
    and every CPU-only host would then read a spurious nvidia-smi traceback plus "During
    handling of the above exception" ahead of the real message.
    """
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            timeout = 5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if smi.returncode != 0 or not smi.stdout.strip():
        return None
    return smi.stdout.strip().splitlines()[0].strip()


def _cuda_visible_devices_hides_nvidia():
    """CUDA_VISIBLE_DEVICES set to "" or "-1" hides every NVIDIA device deliberately
    (install.sh `_cvd_hides_nvidia`). nvidia-smi ignores the variable, so a mask looks
    exactly like a mismatched torch build -- and telling that user to force-reinstall
    torch is wrong advice about a working install."""
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return False
    mask = "".join(mask.split())
    return mask in ("", "-1")


def _reraise_device_type_error_with_gpu_hint(exception):
    """Re-raise a device-detection failure with the actual remedy when nvidia-smi
    can see a GPU that this torch build cannot.

    unsloth_zoo raises NotImplementedError("... only works on NVIDIA/AMD/Intel GPUs")
    whenever torch reports no accelerator. On DGX Spark GB10 and other aarch64 hosts
    that is routinely a torch built for the wrong CUDA -- the GPU is present and
    nvidia-smi lists it, but torch.cuda.is_available() is False -- so the bare "you
    need a GPU" message sends users looking for the wrong problem (#4446, #3553).

    Only the message changes. The exception type is preserved so callers that catch
    NotImplementedError keep working, and zoo's own vendor-specific advice (the AMD
    ROCm repair hint) is left untouched.
    """
    # `raise exception`, never a bare `raise`: a bare raise re-raises whatever is being
    # handled at the raise site, not the error we were handed.
    # Match "ROCm", NOT "AMD": zoo's ROCm repair hint always opens with "an AMD ROCm GPU",
    # while its generic no-accelerator message lists AMD as a supported vendor ("Unsloth
    # currently only works on NVIDIA, AMD and Intel GPUs.") -- and that generic message is
    # what a torch older than 2.6 gets, since get_device_type only reaches the shorter
    # "cannot find any torch accelerator" text behind `hasattr(torch, "accelerator")`.
    if "ROCm" in str(exception):
        raise exception
    if _cuda_visible_devices_hides_nvidia():
        raise exception
    gpu_name = _nvidia_smi_gpu_name()
    if gpu_name is None:
        raise exception
    try:
        import torch as _torch
        torch_cuda_build = getattr(getattr(_torch, "version", None), "cuda", None) or "unknown"
    except Exception:
        torch_cuda_build = "unknown"
    # Recommend all three wheels and the installer, not a lone `pip install torch`: pip
    # leaves an already-installed torchvision/torchaudio behind, and the very next import
    # dies in torchvision_compatibility_check(). "Newest cuXXX" is wrong advice too --
    # recent cu128/cu130 wheels drop pre-Turing GPUs, which is why install.sh caps them.
    raise NotImplementedError(
        f"Unsloth: an NVIDIA GPU ({gpu_name}) is present -- nvidia-smi sees it -- but this "
        f"PyTorch build cannot use it (torch.cuda.is_available() is False).\n"
        f"This usually means the installed PyTorch does not match the system's CUDA driver "
        f"or platform, which is common on DGX Spark GB10 and other aarch64 hosts.\n"
        f"PyTorch was compiled with CUDA: {torch_cuda_build}.\n"
        f"Fix: rerun install.sh, which picks a CUDA wheel family this GPU can run and "
        f"reinstalls torchvision and torchaudio to match. By hand, replace all three "
        f"together:\n"
        f"    pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX\n"
        f"choosing cuXXX at https://pytorch.org/get-started/locally/ -- the newest family is "
        f"not always the right one, since recent cu128/cu130 wheels drop pre-Turing GPUs."
    ) from exception


# Try importing PyTorch and check version
try:
    unsloth_zoo_version = importlib_version("unsloth_zoo")
    if Version(unsloth_zoo_version) < Version("2026.8.15"):
        print(
            "Unsloth: Please update Unsloth and Unsloth-Zoo to the latest version!\n"
            "Do this via `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo`"
        )
    import unsloth_zoo
except PackageNotFoundError:
    raise ImportError(
        f"Unsloth: Please install unsloth_zoo via `pip install unsloth_zoo` then retry!"
    )
except NotImplementedError as device_type_error:
    _reraise_device_type_error_with_gpu_hint(device_type_error)
except:
    raise
del PackageNotFoundError, importlib_version

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"
        "We have some installation instructions on our Github page."
    )
except:
    raise

try:
    from unsloth_zoo.device_type import (
        is_hip,
        get_device_type,
        DEVICE_TYPE,
        DEVICE_TYPE_TORCH,
        DEVICE_COUNT,
        ALLOW_PREQUANTIZED_MODELS,
    )
except NotImplementedError as device_type_error:
    _reraise_device_type_error_with_gpu_hint(device_type_error)
del (
    _reraise_device_type_error_with_gpu_hint,
    _nvidia_smi_gpu_name,
    _cuda_visible_devices_hides_nvidia,
)
from .device_type import arch_lacks_bf16, hip_visible_archs

from .import_fixes import (
    fix_transformers5_bare_annotation_configs,
    fix_transformers_fully_masked_rows,
    fix_xformers_performance_issue,
    fix_flash_attn_4_namespace_shadow,
    fix_vllm_aimv2_issue,
    fix_vllm_lora_tokenizer_module,
    fix_torchao_nf4tensor_move,
    check_vllm_torch_sm100_compatibility,
    fix_vllm_guided_decoding_params,
    fix_vllm_pdl_blackwell,
    fix_triton_compiled_kernel_missing_attrs,
    fix_dynamo_config_thread_visibility,
    patch_trunc_normal_precision_issue,
    ignore_logger_messages,
    patch_ipykernel_hf_xet,
    patch_trackio,
    patch_datasets,
    patch_psutil_cpu_freq,
    patch_enable_input_require_grads,
    patch_unsafe_trainer_rng_load,
    fix_openenv_no_vllm,
    patch_openspiel_env_async,
    fix_executorch,
    patch_vllm_for_notebooks,
    patch_torchcodec_audio_decoder,
    disable_torchcodec_if_broken,
    disable_broken_wandb,
    fix_trl_vllm_ascend,
    fix_peft_transformers_tensor_parallel_import_compat,
    fix_peft_transformers_weight_conversion_import,
    patch_peft_weight_converter_compatibility,
    fix_peft_stale_torchao_import_error,
    patch_accelerate_recursively_apply,
)

# Must run first: guards PretrainedConfig before vLLM defines its config classes.
fix_transformers5_bare_annotation_configs()
# Probe-gated: no-ops unless this transformers really hands SDPA a query row that attends to
# nothing. Ordered here, before anything imports a model, so a plain transformers.generate in the
# same process is covered too (#9708).
fix_transformers_fully_masked_rows()
fix_xformers_performance_issue()
# Must run AFTER fix_xformers_performance_issue (it rewrites xformers' cutlass.py on disk) and
# BEFORE models/_utils.py imports xformers.ops.
fix_flash_attn_4_namespace_shadow()
fix_vllm_aimv2_issue()
fix_vllm_lora_tokenizer_module()
# torchao 0.18.0 moved nf4tensor; torchtune (via xcodec2) still imports the old path. Lazy alias, so
# it costs nothing unless asked for.
fix_torchao_nf4tensor_move()
# Check vLLM + torch < 2.9.0 + SM100 compatibility BEFORE importing vLLM
check_vllm_torch_sm100_compatibility()
fix_vllm_guided_decoding_params()
fix_trl_vllm_ascend()
fix_vllm_pdl_blackwell()
fix_triton_compiled_kernel_missing_attrs()
# Must run before unsloth_zoo's patch_torch_compile and the gpt-oss patches raise the dynamo
# recompile limits, so those settings reach the autograd worker threads on torch >= 2.12.
fix_dynamo_config_thread_visibility()
patch_trunc_normal_precision_issue()
ignore_logger_messages()
patch_ipykernel_hf_xet()
patch_trackio()
patch_datasets()
# Apple Silicon M4+ only: psutil <= 7.2.2 reads the clock 1000x too small.
patch_psutil_cpu_freq()
patch_enable_input_require_grads()
patch_unsafe_trainer_rng_load()
fix_openenv_no_vllm()
patch_openspiel_env_async()
fix_executorch()
patch_vllm_for_notebooks()
patch_torchcodec_audio_decoder()
disable_torchcodec_if_broken()
disable_broken_wandb()
# Must run before patch_peft_weight_converter_compatibility: it stubs the transformers v5
# submodules peft 0.19.x imports, so the next patch can wrap build_peft_weight_mapping instead of
# being swallowed by its ImportError.
fix_peft_transformers_tensor_parallel_import_compat()
fix_peft_transformers_weight_conversion_import()
patch_peft_weight_converter_compatibility()
# After peft is importable, so the already-bound is_torchao_available in peft.tuners.lora.torchao is
# replaced too, not just import_utils'.
fix_peft_stale_torchao_import_error()
patch_accelerate_recursively_apply()

del fix_transformers5_bare_annotation_configs
del fix_xformers_performance_issue
del fix_flash_attn_4_namespace_shadow
del fix_vllm_aimv2_issue
del fix_vllm_lora_tokenizer_module
del fix_torchao_nf4tensor_move
del check_vllm_torch_sm100_compatibility
del fix_vllm_guided_decoding_params
del fix_trl_vllm_ascend
del fix_vllm_pdl_blackwell
del fix_triton_compiled_kernel_missing_attrs
del fix_dynamo_config_thread_visibility
del patch_trunc_normal_precision_issue
del ignore_logger_messages
del patch_ipykernel_hf_xet
del patch_trackio
del patch_datasets
del patch_psutil_cpu_freq
del patch_enable_input_require_grads
del fix_openenv_no_vllm
del patch_openspiel_env_async
del fix_executorch
del patch_vllm_for_notebooks
del patch_torchcodec_audio_decoder
del disable_torchcodec_if_broken
del disable_torchaudio_if_cuda_mismatched
del disable_broken_wandb
del fix_peft_transformers_tensor_parallel_import_compat
del fix_peft_transformers_weight_conversion_import
del patch_peft_weight_converter_compatibility
del fix_peft_stale_torchao_import_error
del patch_accelerate_recursively_apply

# Torch 2.4 has including_emulation
if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():
    # UNSLOTH_ALLOW_CPU=1 keeps DEVICE_TYPE at "cuda" on a host with a CUDA-built torch and no usable
    # device, so ask whether a device is present before asking what it can do: get_device_capability()
    # would raise out of _lazy_init(). is_bf16_supported() is stubbed to match.
    SUPPORTS_BFLOAT16 = False
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
elif DEVICE_TYPE == "cuda":
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
    old_is_bf16_supported = torch.cuda.is_bf16_supported

    # SUPPORTS_BFLOAT16 is process-wide, so one gfx10 in the visible set must disable it for all.
    SUPPORTS_BFLOAT16 = (
        not any(arch_lacks_bf16(arch) for arch in hip_visible_archs()) and old_is_bf16_supported()
    )

    def is_bf16_supported(*args, **kwargs):
        return SUPPORTS_BFLOAT16

    torch.cuda.is_bf16_supported = is_bf16_supported
    del old_is_bf16_supported
elif DEVICE_TYPE == "xpu":
    # torch.xpu.is_bf16_supported() does not have including_emulation set SUPPORTS_BFLOAT16 as
    # torch.xpu.is_bf16_supported()
    SUPPORTS_BFLOAT16 = torch.xpu.is_bf16_supported()

# For Gradio HF Spaces?
# if "SPACE_AUTHOR_NAME" not in os.environ and "SPACE_REPO_NAME" not in os.environ:
import triton

if DEVICE_TYPE == "cuda":
    libcuda_dirs = lambda: None
    if Version(triton.__version__) >= Version("3.0.0"):
        # Try loading bitsandbytes and triton
        try:
            from triton.backends.nvidia.driver import libcuda_dirs
        except:
            pass
    else:
        from triton.common.build import libcuda_dirs

    try:
        import bitsandbytes as bnb

        # Bind the submodule by name: a half-imported bitsandbytes leaves the parent without a `functional`
        # attribute, which would otherwise be misreported below as a CUDA linking failure.
        import bitsandbytes.functional as bnb_functional
    except:
        print(
            "Unsloth: `bitsandbytes` is not installed - 4bit QLoRA unallowed, but 16bit and full finetuning works!"
        )
        bnb = None
        bnb_functional = None
    try:
        cdequantize_blockwise_fp32 = bnb_functional.lib.cdequantize_blockwise_fp32
        libcuda_dirs()
    except:
        if not torch.cuda.is_available():
            # UNSLOTH_ALLOW_CPU=1 on a driverless host: a missing libcuda is the expected state, not broken
            # linkage, and repairing it would ldconfig the host's linker cache as root for a device that is
            # not there.
            pass
        elif hasattr(os, "geteuid") and os.geteuid() == 0:
            warnings.warn("Unsloth: Running `ldconfig /usr/lib64-nvidia` to link CUDA.")

            if os.path.exists("/usr/lib64-nvidia"):
                os.system("ldconfig /usr/lib64-nvidia")
            elif os.path.exists("/usr/local"):
                # Sometimes bitsandbytes cannot be linked properly in Runpod for example
                possible_cudas = (
                    subprocess.check_output(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
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
            # Same degradation as the cuda branch above: no bnb means no 4bit, not a failed `import unsloth`.
            try:
                libcuda_dirs = lambda: None
                if Version(triton.__version__) >= Version("3.0.0"):
                    try:
                        from triton.backends.nvidia.driver import libcuda_dirs
                    except:
                        # TODO: check triton for intel installed properly.
                        pass
                else:
                    from triton.common.build import libcuda_dirs
                cdequantize_blockwise_fp32 = bnb_functional.lib.cdequantize_blockwise_fp32
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
        elif bnb is not None:
            warnings.warn(
                "Unsloth: CUDA is not linked properly.\n"
                "You need to run in your terminal `sudo ldconfig /usr/lib64-nvidia` yourself, then import Unsloth.\n"
                "Also try `sudo ldconfig /usr/local/cuda-xx.x` - find the latest cuda version.\n"
                "Unsloth will still run for now, but maybe it might crash - let's hope it works!"
            )
    del libcuda_dirs
elif DEVICE_TYPE == "hip":
    # NO-OP for rocm device
    pass
elif DEVICE_TYPE == "xpu":
    # Same degradation as the cuda branch above: no bnb means no 4bit, not a failed `import unsloth`.
    try:
        import bitsandbytes as bnb
    except Exception:
        print(
            "Unsloth: `bitsandbytes` is not installed - 4bit QLoRA unallowed, but 16bit and full finetuning works!"
        )
        bnb = None

    pass

from .models import *
from .models import __version__
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *

# Export dataprep utilities for CLI and downstream users
from .dataprep.raw_text import RawTextDataLoader, TextPreprocessor
from unsloth_zoo.rl_environments import (
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
    Benchmarker,
    is_port_open,
    launch_openenv,
)

# Skipped under UNSLOTH_ALLOW_CPU=1 (CPU-only CI): rebinding trl.SFTTrainer.__init__ changes
# inspect.getsource() and corrupts downstream drift detectors.
if os.environ.get("UNSLOTH_ALLOW_CPU", "0") != "1":
    _patch_trl_trainer()
