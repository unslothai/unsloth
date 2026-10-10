# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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

critical_modules = ["trl", "transformers", "peft"]
already_imported = [mod for mod in critical_modules if mod in sys.modules]

from .import_fixes import (
    fix_message_factory_issue,
    patch_torch_missing_attribute_error,
    check_triton_py_ssize_t_clean,
    check_transformers_prequantized_vlm_quant_state,
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
    fix_broken_hf_xet_wheel,
)

# Redirect a read-only HF cache before anything imports huggingface_hub, which freezes cache paths.
# hf_cache.py is stdlib-only, so load it from its file without the full unsloth_zoo init.
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

# Before anything imports huggingface_hub, which freezes HF_HUB_DISABLE_XET at import time.
fix_broken_hf_xet_wheel()
# Must precede the first `import torch`: it sets AMDGPU_ASIC_ID_TABLE_PATH, read when libdrm loads.
configure_amdgpu_asic_id_table_path()
# Before every fix below and `import unsloth_zoo`, which import transformers; this imports torch.
patch_torch_missing_attribute_error()
# Must precede `import unsloth_zoo` below, which imports bnb on ROCm.
fix_bitsandbytes_rocm_arch_detection()
# Torch-only, so it can run first; torchao 0.18 on torch < 2.10 needs it before any torchao import.
fix_torchao_torch_symbol_skew()
# Before unsloth_zoo and transformers (and vllm, below): real torchao on a torch without torch.distributed.
from ._torchao_nodist import fix_torchao_without_torch_distributed

fix_torchao_without_torch_distributed()
del fix_torchao_without_torch_distributed
disable_broken_causal_conv1d()
disable_broken_vllm()
fix_message_factory_issue()
fix_torch_check_is_size()
# vLLM's architecture inspector is a subprocess that imports torchao itself.
propagate_torchao_fix_to_subprocesses()
check_transformers_dependency_versions()
check_triton_py_ssize_t_clean()
check_fbgemm_gpu_version()
torchvision_compatibility_check()
# Before `import unsloth_zoo`: its patches import torchaudio via transformers, and a broken
# torchaudio would fail the whole import before the later fixes run.
disable_torchaudio_if_cuda_mismatched()
fix_diffusers_warnings()
fix_huggingface_hub()
del configure_amdgpu_asic_id_table_path
del fix_bitsandbytes_rocm_arch_detection
del disable_broken_causal_conv1d
del disable_broken_vllm
del fix_message_factory_issue
del patch_torch_missing_attribute_error
del fix_torch_check_is_size
del fix_torchao_torch_symbol_skew
del propagate_torchao_fix_to_subprocesses
del check_fbgemm_gpu_version
del check_transformers_dependency_versions
del check_triton_py_ssize_t_clean
del torchvision_compatibility_check
del fix_diffusers_warnings
del fix_huggingface_hub
del fix_broken_hf_xet_wheel

# If imported before Unsloth, the unpatched versions run, risking OOM or slower training.
if already_imported:
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

# Pure-python protobuf avoids "sentencepiece_model.proto already in the pool" errors.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# docker --gpus sets NVIDIA_VISIBLE_DEVICES but not CUDA_VISIBLE_DEVICES, so Inductor's
# compile workers cannot see the GPU. Opt out with UNSLOTH_FORCE_SINGLE_COMPILE_WORKER=0.
_nvd = os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip().lower()
_cgroup_pinned = _nvd not in ("", "all", "none", "void")
if (
    os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "auto") != "0"
    and _cgroup_pinned
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") in (None, "", "1"):
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"] = "1"
del _nvd, _cgroup_pinned


from importlib.metadata import version as importlib_version
from importlib.metadata import PackageNotFoundError


def _nvidia_smi_gpu_name():
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            # A UnicodeDecodeError here would replace the original error.
            errors = "replace",
            timeout = 5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if smi.returncode != 0 or not smi.stdout.strip():
        return None
    return smi.stdout.strip().splitlines()[0].strip()


def _reraise_device_type_error_with_gpu_hint(exception):
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Zoo's generic error lists AMD, so only "ROCm" identifies its ROCm advice.
    # An empty or "-1" mask hides every GPU on purpose.
    if "ROCm" in str(exception) or (mask is not None and "".join(mask.split()) in ("", "-1")):
        raise exception
    gpu_name = _nvidia_smi_gpu_name()
    if gpu_name is None:
        raise exception
    try:
        import torch as _torch
        torch_build = _torch.__version__
    except Exception:
        torch_build = "unknown"
    mask_note = "" if mask is None else f", CUDA_VISIBLE_DEVICES={mask!r}"
    raise NotImplementedError(
        f"Unsloth: nvidia-smi sees {gpu_name} but torch.cuda.is_available() is False "
        f"(torch {torch_build}{mask_note}). PyTorch likely does not match this "
        f"machine; reinstall it for {sys.executable} per "
        f"https://github.com/unslothai/unsloth#-install"
    ) from exception


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

# Re-assert: unsloth_zoo's patch_torch_compile may pop TORCHINDUCTOR_COMPILE_THREADS.
if os.environ.get("UNSLOTH_FORCE_SINGLE_COMPILE_WORKER", "0") == "1":
    try:
        torch._inductor.config.compile_threads = 1
    except Exception:
        pass
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    def _force_single_compile_worker_in_zoo():
        setattr(
            importlib.import_module("unsloth_zoo.temporary_patches.common"),
            "determine_compile_threads",
            lambda: 1,
        )
        # Zoo's options dicts were snapshotted at import and outrank the env var in Inductor; rewrite them
        # in place, since they are shared by identity across modules and the torch_compile partial.
        for module in list(sys.modules.values()):
            name = getattr(module, "__name__", "")
            if name != "unsloth_zoo" and not name.startswith("unsloth_zoo."):
                continue
            try:
                values = list(vars(module).values())
            except Exception:
                continue
            for value in values:
                if isinstance(value, dict) and "compile_threads" in value:
                    value["compile_threads"] = 1

    try:
        _force_single_compile_worker_in_zoo()
    except Exception:
        pass
    del _force_single_compile_worker_in_zoo

from unsloth_zoo.device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)

# UNSLOTH_ZOO_DISABLE_GPU_INIT makes zoo answer "cpu", so unsloth's own check raises instead.
try:
    from .device_type import (
        arch_lacks_bf16,
        arch_lacks_buffer_ops,
        apply_gfx101x_triton_workaround,
        hip_visible_archs,
    )
except NotImplementedError as device_type_error:
    _reraise_device_type_error_with_gpu_hint(device_type_error)
del _reraise_device_type_error_with_gpu_hint, _nvidia_smi_gpu_name

from .import_fixes import (
    fix_transformers5_bare_annotation_configs,
    fix_transformers5_legacy_config_types,
    fix_transformers5_image_processing_reexports,
    fix_transformers_composite_prefix_renaming,
    fix_transformers_bnb_prequantized_save,
    fix_transformers_fully_masked_rows,
    fix_transformers_flash_attention_mrope_packed_sequence,
    fix_transformers_untrusted_config_fields,
    fix_transformers_chat_template_path_traversal,
    fix_transformers_chunked_mask_block_sequence_ids,
    fix_transformers_flex_mask_graph_breaks,
    fix_transformers_longcat_lsa_config,
    fix_transformers_rope_scaling_drops_theta,
    fix_transformers_fp8_modulelist_experts,
    fix_transformers_fp8_unscaled_checkpoint_linears,
    fix_transformers_validate_rope_ignore_keys,
    fix_transformers5_remote_code_legacy_defaults,
    fix_transformers_config_only_remote_code,
    fix_transformers_remote_rope_scaling_none,
    fix_transformers_is_torch_fx_available,
    fix_transformers5_remote_code_model_api,
    fix_xformers_performance_issue,
    fix_flash_attn_4_namespace_shadow,
    fix_vllm_aimv2_issue,
    fix_vllm_lora_tokenizer_module,
    fix_torchao_nf4tensor_move,
    fix_compressed_tensors_activation_quant_gradient,
    fix_torchao_safe_int_mm_repr_probe,
    check_vllm_torch_sm100_compatibility,
    fix_vllm_guided_decoding_params,
    fix_vllm_pdl_blackwell,
    fix_cudnn_sdpa_d256_masked_backward,
    fix_rocm_windows_fused_sdpa,
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
    patch_torch_export_pt2_unsafe_load,
    fix_openenv_no_vllm,
    patch_openspiel_env_async,
    fix_executorch,
    patch_vllm_for_notebooks,
    patch_torchcodec_audio_decoder,
    disable_torchcodec_if_broken,
    disable_broken_wandb,
    fix_accelerate_dtensor_check_without_torch_distributed,
    fix_trl_vllm_ascend,
    fix_peft_transformers_tensor_parallel_import_compat,
    fix_peft_transformers_weight_conversion_import,
    patch_peft_weight_converter_compatibility,
    patch_peft_float8_adapter_upcast,
    fix_peft_stale_torchao_import_error,
    fix_peft_torchao_missing_tensor_subclass,
    patch_accelerate_recursively_apply,
)

fix_transformers5_legacy_config_types()
# Must run first: guards PretrainedConfig before vLLM defines its config classes.
fix_transformers5_bare_annotation_configs()
# Probe-gated; before any model import so plain transformers.generate is covered too.
fix_transformers_fully_masked_rows()
fix_transformers_flash_attention_mrope_packed_sequence()
fix_transformers_chunked_mask_block_sequence_ids()
fix_transformers_flex_mask_graph_breaks()
# CVE-2026-4372 / 5241 / 9856, no-ops once transformers carries the fix; before any config loads.
fix_transformers_untrusted_config_fields()
fix_transformers_chat_template_path_traversal()
# Probe-gated; before any checkpoint load so plain from_pretrained keeps bnb quant_state too.
fix_transformers_composite_prefix_renaming()
fix_transformers_bnb_prequantized_save()
# After the repair above, never before it, and this is the ONLY call: on exactly the releases
# the repair covers, warning first tells users to downgrade away from a version that now works,
# and a second call cannot retract a warning already logged. The check reads the live attribute,
# so a repair that declined to install still warns. Being this late also keeps it below
# `disable_torchaudio_if_cuda_mismatched`, which matters because this is the only check here
# that IMPORTS transformers rather than reading its metadata.
# A run that loads no pre-quantized multimodal checkpoint never fails either way.
check_transformers_prequantized_vlm_quant_state()
del check_transformers_prequantized_vlm_quant_state
# Probe-gated; before any config is built so llama.py's rope delegation retry keeps the RoPE base.
fix_transformers_rope_scaling_drops_theta()
fix_transformers_fp8_modulelist_experts()
fix_transformers_fp8_unscaled_checkpoint_linears()
fix_transformers_validate_rope_ignore_keys()
fix_transformers5_remote_code_legacy_defaults()
fix_transformers_config_only_remote_code()
fix_transformers_longcat_lsa_config()
# Remote code written for 4.x reads plain RoPE as rope_scaling None and imports is_torch_fx_available.
fix_transformers_remote_rope_scaling_none()
fix_transformers_is_torch_fx_available()
# Probe-gated and lazy: only wraps get_class_in_module.
fix_transformers5_image_processing_reexports()
fix_transformers5_remote_code_model_api()
fix_xformers_performance_issue()
# Must run AFTER fix_xformers_performance_issue (it rewrites xformers' cutlass.py on disk) and
# BEFORE models/_utils.py imports xformers.ops.
fix_flash_attn_4_namespace_shadow()
fix_vllm_aimv2_issue()
fix_vllm_lora_tokenizer_module()
# torchao 0.18.0 moved nf4tensor; torchtune (via xcodec2) imports the old path. Lazy alias.
fix_torchao_nf4tensor_move()
fix_compressed_tensors_activation_quant_gradient()
fix_torchao_safe_int_mm_repr_probe()
# Before importing vLLM.
check_vllm_torch_sm100_compatibility()
fix_vllm_guided_decoding_params()
fix_trl_vllm_ascend()
fix_vllm_pdl_blackwell()
fix_cudnn_sdpa_d256_masked_backward()
# Windows ROCm only, probe-gated: fused attention fails on every call there (gfx1151, torch 2.11).
fix_rocm_windows_fused_sdpa()
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
patch_torch_export_pt2_unsafe_load()
fix_openenv_no_vllm()
patch_openspiel_env_async()
fix_executorch()
patch_vllm_for_notebooks()
patch_torchcodec_audio_decoder()
disable_torchcodec_if_broken()
disable_broken_wandb()
# After unsloth_zoo, whose ROCm torchao loader must be in place before accelerate is imported.
fix_accelerate_dtensor_check_without_torch_distributed()
# Must run before patch_peft_weight_converter_compatibility: it stubs the transformers v5
# submodules peft 0.19.x imports, so the next patch can wrap build_peft_weight_mapping instead of
# being swallowed by its ImportError.
fix_peft_transformers_tensor_parallel_import_compat()
fix_peft_transformers_weight_conversion_import()
patch_peft_weight_converter_compatibility()
patch_peft_float8_adapter_upcast()
# After peft is importable, so peft.tuners.lora.torchao's bound copy is replaced too.
fix_peft_stale_torchao_import_error()
# peft.tuners.lora.model imported dispatch_torchao by value, so both copies must be replaced.
fix_peft_torchao_missing_tensor_subclass()
patch_accelerate_recursively_apply()

del fix_transformers5_bare_annotation_configs
del fix_transformers5_legacy_config_types
del fix_transformers_untrusted_config_fields
del fix_transformers_chat_template_path_traversal
del fix_transformers_rope_scaling_drops_theta
del fix_transformers_bnb_prequantized_save
del fix_transformers_fp8_modulelist_experts
del fix_transformers_fp8_unscaled_checkpoint_linears
del fix_transformers_validate_rope_ignore_keys
del fix_transformers_longcat_lsa_config
del fix_transformers_remote_rope_scaling_none
del fix_transformers_is_torch_fx_available
del fix_transformers5_remote_code_model_api
del fix_xformers_performance_issue
del fix_flash_attn_4_namespace_shadow
del fix_vllm_aimv2_issue
del fix_vllm_lora_tokenizer_module
del fix_torchao_nf4tensor_move
del fix_compressed_tensors_activation_quant_gradient
del fix_torchao_safe_int_mm_repr_probe
del check_vllm_torch_sm100_compatibility
del fix_vllm_guided_decoding_params
del fix_trl_vllm_ascend
del fix_vllm_pdl_blackwell
del fix_cudnn_sdpa_d256_masked_backward
del fix_rocm_windows_fused_sdpa
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
del fix_accelerate_dtensor_check_without_torch_distributed
del fix_peft_transformers_tensor_parallel_import_compat
del fix_peft_transformers_weight_conversion_import
del patch_peft_weight_converter_compatibility
del patch_peft_float8_adapter_upcast
del fix_peft_stale_torchao_import_error
del fix_peft_torchao_missing_tensor_subclass
del patch_accelerate_recursively_apply

if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():
    # UNSLOTH_ALLOW_CPU=1 keeps DEVICE_TYPE "cuda" with no device; get_device_capability() would raise.
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
    # torch.xpu.is_bf16_supported() has no including_emulation argument.
    SUPPORTS_BFLOAT16 = torch.xpu.is_bf16_supported()
elif DEVICE_TYPE == "npu":
    SUPPORTS_BFLOAT16 = torch.npu.is_bf16_supported()

# gfx101x: Triton buffer-op kernels silently write nothing; must be set before the first compile.
if DEVICE_TYPE == "hip" and any(arch_lacks_buffer_ops(arch) for arch in hip_visible_archs()):
    apply_gfx101x_triton_workaround()

# For Gradio HF Spaces?
# `triton` is optional: PyPI ships no Windows wheel, and a CPU-only install must still import.
try:
    import triton
except Exception as _triton_import_exception:
    triton = None
    TRITON_IMPORT_ERROR = f"{type(_triton_import_exception).__name__}: {_triton_import_exception}"
    del _triton_import_exception
else:
    TRITON_IMPORT_ERROR = None

if TRITON_IMPORT_ERROR is not None:
    if DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
        warnings.warn(
            f"Unsloth: `triton` could not be imported ({TRITON_IMPORT_ERROR}), so the fused Triton "
            "kernels are unavailable and training will be slower or may fail.\n"
            "On Windows install `triton-windows`, on Linux reinstall `triton`.",
            stacklevel = 2,
        )
    else:
        print(
            f"Unsloth: `triton` is not available ({TRITON_IMPORT_ERROR}) - continuing without the "
            "fused Triton kernels, which a CPU only install does not use.\n"
            "On Windows, Triton is published as the separate `triton-windows` package."
        )

if DEVICE_TYPE == "cuda":
    libcuda_dirs = lambda: None
    if triton is None:
        pass
    elif Version(triton.__version__) >= Version("3.0.0"):
        try:
            from triton.backends.nvidia.driver import libcuda_dirs
        except:
            pass
    else:
        from triton.common.build import libcuda_dirs

    try:
        import bitsandbytes as bnb

        # Bind the submodule by name: a half-imported bitsandbytes lacks `functional`, which would be misreported.
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
            # Driverless UNSLOTH_ALLOW_CPU host: missing libcuda is expected; do not ldconfig as root.
            pass
        elif hasattr(os, "geteuid") and os.geteuid() == 0:
            warnings.warn("Unsloth: Running `ldconfig /usr/lib64-nvidia` to link CUDA.")

            if os.path.exists("/usr/lib64-nvidia"):
                os.system("ldconfig /usr/lib64-nvidia")
            elif os.path.exists("/usr/local"):
                # bitsandbytes sometimes fails to link CUDA (e.g. Runpod).
                possible_cudas = (
                    subprocess.check_output(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
                )
                find_cuda = re.compile(r"[\s](cuda\-[\d\.]{2,})$")
                possible_cudas = [find_cuda.search(x) for x in possible_cudas]
                possible_cudas = [x.group(1) for x in possible_cudas if x is not None]

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
            if triton is not None:
                importlib.reload(triton)
            # No bnb means no 4bit, not a failed import; missing Triton was already reported above.
            try:
                libcuda_dirs = lambda: None
                if triton is None:
                    pass
                elif Version(triton.__version__) >= Version("3.0.0"):
                    try:
                        from triton.backends.nvidia.driver import libcuda_dirs
                    except:
                        # TODO: check triton for Intel is installed properly.
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
    pass
elif DEVICE_TYPE == "xpu":
    # No bnb means no 4bit, not a failed `import unsloth`.
    try:
        import bitsandbytes as bnb
    except Exception:
        print(
            "Unsloth: `bitsandbytes` is not installed - 4bit QLoRA unallowed, but 16bit and full finetuning works!"
        )
        bnb = None

    pass

# After the bitsandbytes import above, never before: it only patches an already-imported bitsandbytes.
from .import_fixes import patch_bitsandbytes_paged_optimizer_resume

patch_bitsandbytes_paged_optimizer_resume()
del patch_bitsandbytes_paged_optimizer_resume

from .models import *
from .models import __version__
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *

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
