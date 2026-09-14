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

"""Loader-side orchestration: backend routing and EXL3 checkpoint prep."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import Exl3Config, normalize_exl3_config
from .patcher import (
    is_exl3_model_dir,
    patch_transformers_exl3,
    read_exl3_bitrate,
)
from .quantize import quantize_to_exl3
from .utils import is_exllama_available, require_exllama


@dataclass
class Exl3LoadPlan:
    """Result of preparing an EXL3 load."""

    checkpoint_dir: str
    config: Exl3Config
    was_quantized: bool  # True if we ran a conversion, False if reused/prequant
    source_model: str  # original model id/path requested


def _looks_like_exl3_request(load_in_exl3, quantization_config) -> bool:
    if load_in_exl3:
        return True
    if quantization_config is None:
        return False
    if isinstance(quantization_config, Exl3Config):
        return True
    if isinstance(quantization_config, dict):
        return str(quantization_config.get("quant_method", "")).lower() == "exl3"
    return str(getattr(quantization_config, "quant_method", "")).lower() == "exl3"


# Users / CI can opt out of the EXL3-by-default behaviour and fall back to the
# legacy bitsandbytes backend with UNSLOTH_QUANT_BACKEND=bnb (or =bitsandbytes).
def _exl3_default_disabled_by_env() -> bool:
    backend = os.environ.get("UNSLOTH_QUANT_BACKEND", "").strip().lower()
    return backend in ("bnb", "bitsandbytes", "bits_and_bytes")


def _bnb_explicitly_requested(quantization_config) -> bool:
    """True if the caller passed an explicit non-EXL3 quant config (bnb/gptq/
    awq/hqq/...), which opts out of the default EXL3 route."""
    if quantization_config is None:
        return False
    if isinstance(quantization_config, Exl3Config):
        return False
    method = None
    if isinstance(quantization_config, dict):
        method = quantization_config.get("quant_method")
        has_bnb_flags = bool(
            quantization_config.get("load_in_4bit") or quantization_config.get("load_in_8bit")
        )
    else:
        method = getattr(quantization_config, "quant_method", None)
        has_bnb_flags = bool(
            getattr(quantization_config, "load_in_4bit", False)
            or getattr(quantization_config, "load_in_8bit", False)
        )
    # transformers' QuantizationMethod is an enum whose ``.value`` is
    # "bitsandbytes" but whose ``str()`` is "QuantizationMethod.BITS_AND_BYTES".
    # Normalise both, and also accept the presence of bnb load flags.
    method_value = getattr(method, "value", method)
    method_str = str(method_value).lower() if method_value is not None else ""
    # Any explicit non-EXL3 quant method (bnb/gptq/awq/hqq/...) opts out.
    if method_str and method_str != "exl3":
        return True
    return has_bnb_flags


def exl3_is_default_backend(
    *,
    load_in_4bit = False,
    load_in_8bit = False,
    quantization_config = None,
) -> bool:
    """Decide whether a *plain* quantized load should default to EXL3.

    Per the project goal, EXL3 replaces bitsandbytes as the default quantization
    backend. So when the caller requests quantization (the ``load_in_4bit=True``
    default) without explicitly asking for bitsandbytes, and EXL3 can actually
    run here, we route the load through EXL3 rather than bnb.

    EXL3 requires a CUDA GPU + the exllamav3 extension; when either is missing
    (CPU, ROCm/XPU, exllamav3 not installed) we return False so Unsloth falls
    back to the legacy bitsandbytes path. 8-bit-only requests also stay on bnb
    (EXL3's sweet spot is <=6-bit and its 8-bit gains little over bnb int8).
    """
    if _exl3_default_disabled_by_env():
        return False
    if not (load_in_4bit or load_in_8bit):
        return False  # a 16-bit / bf16 load is not a quantized load
    if load_in_8bit and not load_in_4bit:
        return False  # leave pure 8-bit on bnb int8
    if _bnb_explicitly_requested(quantization_config):
        return False
    if not is_exllama_available():
        return False
    if not _cuda_available():
        return False
    return True


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def should_use_exl3(
    model_name: str,
    *,
    load_in_exl3 = False,
    quantization_config = None,
    load_in_4bit = False,
    load_in_8bit = False,
) -> bool:
    """Decide whether this load should go through the EXL3 backend.

    EXL3 is used when:

    * the user explicitly asks for it (``load_in_exl3`` or an EXL3
      ``quantization_config``), or
    * the target is already an EXL3 checkpoint on disk, or
    * it is the default backend for a plain quantized load (see
      :func:`exl3_is_default_backend`) - i.e. EXL3 has replaced bitsandbytes as
      the default quantizer on CUDA when exllamav3 is installed.
    """
    if _looks_like_exl3_request(load_in_exl3, quantization_config):
        return True
    # Auto-detect an EXL3 checkpoint directory even without an explicit flag.
    if isinstance(model_name, str):
        expanded = os.path.expanduser(model_name)
        if is_exl3_model_dir(expanded):
            return True
        # Never hijack an existing non-EXL3 quant checkpoint or a PEFT adapter
        # repo into the default EXL3 route.
        from .patcher import is_nonexl3_quantized_dir, is_peft_adapter_dir

        if is_nonexl3_quantized_dir(expanded) or is_peft_adapter_dir(expanded):
            return False
    # Default-backend routing: a plain quantized load prefers EXL3 over bnb.
    if exl3_is_default_backend(
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
        quantization_config = quantization_config,
    ):
        # Auto-route to EXL3 only for supported archs; else fall back to bnb.
        # Local paths are checked now; Hub ids are validated at prepare time.
        if isinstance(model_name, str) and os.path.isdir(os.path.expanduser(model_name)):
            from .patcher import exllama_supports_arch
            if not exllama_supports_arch(os.path.expanduser(model_name)):
                return False
        return True
    return False


def resolve_exl3_config(
    load_in_exl3,
    quantization_config,
    *,
    compute_dtype = None,
    calibrate = None,
) -> Exl3Config:
    """Build the concrete :class:`Exl3Config` for this request."""
    if isinstance(quantization_config, Exl3Config):
        cfg = quantization_config
    elif (
        isinstance(quantization_config, dict)
        and str(quantization_config.get("quant_method", "")).lower() == "exl3"
    ):
        cfg = Exl3Config.from_dict(quantization_config)
    elif load_in_exl3 not in (True, False, None):
        # e.g. load_in_exl3 = 3 or "2.5bit"
        cfg = normalize_exl3_config(load_in_exl3)
    else:
        cfg = Exl3Config()
    if compute_dtype is not None and cfg.compute_dtype is None:
        cfg.compute_dtype = compute_dtype
    # An explicit calibrate= override (from the loader) wins over the config's
    # default, unless the user passed their own Exl3Config that already set it.
    if calibrate is not None and not isinstance(quantization_config, Exl3Config):
        cfg.calibrate = bool(calibrate)
    return cfg


def _resolve_local_dir(
    model_name: str,
    *,
    token = None,
    revision = None,
    trust_remote_code = False,
    local_files_only = False,
) -> str:
    """Return a local directory for ``model_name`` (downloading if a Hub id).

    ExLlamaV3's converter operates on a local model directory, so a Hub id must
    be materialized to a snapshot first.
    """
    expanded = os.path.expanduser(model_name)
    if os.path.isdir(expanded):
        return expanded

    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name,
        revision = revision,
        token = token,
        local_files_only = local_files_only,
        # We need the full weights to quantize; ignore only obvious non-weights.
        ignore_patterns = ["*.gguf", "*.pth", "original/*"],
    )


def prepare_exl3_checkpoint(
    model_name: str,
    *,
    load_in_exl3 = True,
    quantization_config = None,
    compute_dtype = None,
    token = None,
    revision = None,
    trust_remote_code = False,
    local_files_only = False,
    devices: str = "0",
    calibrate = None,
    is_explicit = None,
) -> Optional[Exl3LoadPlan]:
    """Prepare a local EXL3 checkpoint for ``model_name``.

    Resolves the model to a local directory, quantizes it to EXL3 at the
    requested bitrate if it is not already an EXL3 checkpoint, patches
    transformers, and returns an :class:`Exl3LoadPlan`.
    """
    require_exllama()
    cfg = resolve_exl3_config(
        load_in_exl3, quantization_config, compute_dtype = compute_dtype, calibrate = calibrate
    )

    local_dir = _resolve_local_dir(
        model_name,
        token = token,
        revision = revision,
        trust_remote_code = trust_remote_code,
        local_files_only = local_files_only,
    )

    # Register the EXL3 quantizer with transformers up front.
    if not patch_transformers_exl3():
        raise RuntimeError(
            "Unsloth: failed to register the EXL3 quantizer with transformers. "
            "Is exllamav3 installed correctly?"
        )

    if is_exl3_model_dir(local_dir):
        # Already quantized - reuse. Reflect the stored bitrate if we can read it.
        stored = read_exl3_bitrate(local_dir)
        if stored is not None:
            cfg.bits = stored
        return Exl3LoadPlan(
            checkpoint_dir = local_dir,
            config = cfg,
            was_quantized = False,
            source_model = model_name,
        )

    # An explicit EXL3 request for an architecture ExLlamaV3 does not implement
    # (e.g. OlmoeForCausalLM) would otherwise crash deep in the converter with a
    # cryptic assertion. Surface a clear, actionable error listing the fallback.
    from .patcher import exllama_supports_arch, exllama_supported_architectures

    if not exllama_supports_arch(local_dir):
        # Default-backend route (not an explicit request): fall back to bnb for
        # an unsupported arch instead of erroring.
        explicit = (
            is_explicit
            if is_explicit is not None
            else _looks_like_exl3_request(load_in_exl3, quantization_config)
        )
        if not explicit:
            return None
        import json as _json

        _arch = "?"
        try:
            with open(os.path.join(local_dir, "config.json"), encoding = "utf-8") as _f:
                _cfg = _json.load(_f)
            _archs = list(_cfg.get("architectures") or [])
            tc = _cfg.get("text_config")
            if isinstance(tc, dict):
                _archs += list(tc.get("architectures") or [])
            _arch = ", ".join(_archs) or "?"
        except Exception:
            pass
        raise ValueError(
            f"Unsloth: ExLlamaV3 (EXL3) does not support the architecture "
            f"'{_arch}' yet, so it cannot be quantized with the EXL3 backend.\n"
            f"Use Unsloth's other backends for this model instead:\n"
            f"  - 4-bit bitsandbytes: `load_in_exl3=False, load_in_4bit=True` "
            f"(set env UNSLOTH_QUANT_BACKEND=bnb), or\n"
            f"  - 16-bit LoRA:        `load_in_exl3=False, load_in_16bit=True`, or\n"
            f"  - full finetuning:    `full_finetuning=True`.\n"
            f"EXL3 currently supports {len(exllama_supported_architectures())} "
            f"architectures; upgrade exllamav3 for newer ones."
        )

    checkpoint_dir = quantize_to_exl3(
        local_dir,
        cfg,
        devices = devices,
    )
    return Exl3LoadPlan(
        checkpoint_dir = checkpoint_dir,
        config = cfg,
        was_quantized = True,
        source_model = model_name,
    )


def finalize_exl3_model(model, compute_dtype = None) -> int:
    """Stamp EXL3 quant states onto a freshly loaded model. Returns layer count."""
    if not is_exllama_available():
        return 0
    import torch

    from .quant_linear import attach_exl3_quant_states, densify_exl3_head

    dtype = compute_dtype or torch.float16
    # Attach quant states first (densify relies on them), then materialize the
    # head to a dense nn.Linear (the fused loss reads lm_head.weight directly).
    n = attach_exl3_quant_states(model, compute_dtype = dtype)
    densify_exl3_head(model, compute_dtype = dtype)
    # Align dense layers (defaulted to fp16 by the quantizer) with the model's
    # dtype so bf16-requiring archs don't hit fp16-vs-bf16 mismatches.
    from .quant_linear import harmonize_nonquant_dtype

    harmonize_nonquant_dtype(model, dtype)
    model._unsloth_exl3_backend = True
    return n


def finalize_exl3_experts(
    model,
    checkpoint_dir,
    compute_dtype = None,
) -> int:
    """Reconstruct fused MoE expert weights from the EXL3 checkpoint.

    transformers 5 stores MoE experts as fused 3-D tensors that ExLlamaV3's
    stock integration cannot quantize/load, so the per-expert EXL3 tensors go
    unused. This rebuilds the fused expert parameters from those tensors. Safe
    no-op for dense models. Returns the number of expert matrices rebuilt.

    For large MoEs (many experts) reconstructing every expert to a DENSE tensor
    will not fit in VRAM, so we keep the experts EXL3-quantized and reconstruct
    per-forward (:func:`reload_exl3_experts_quantized`). The threshold is set by
    ``UNSLOTH_EXL3_QUANTIZED_EXPERTS`` (default: auto - quantized when a model
    has > 32 experts per layer, dense otherwise; ``1`` forces quantized, ``0``
    forces dense).
    """
    if not is_exllama_available():
        return 0
    import torch

    dtype = compute_dtype or torch.float16
    from .moe import (
        reload_exl3_experts,
        reload_exl3_experts_quantized,
        _find_experts_modules,
        _infer_num_experts,
    )

    mode = os.environ.get("UNSLOTH_EXL3_QUANTIZED_EXPERTS", "auto").strip().lower()
    use_quantized = False
    if mode in ("1", "true", "yes", "quantized"):
        use_quantized = True
    elif mode in ("0", "false", "no", "dense"):
        use_quantized = False
    else:  # auto: quantized for many-expert MoEs (dense would not fit VRAM)
        max_experts = 0
        for module, _ in _find_experts_modules(model):
            max_experts = max(
                max_experts, int(getattr(module, "num_experts", 0)) or _infer_num_experts(module)
            )
        use_quantized = max_experts > 32

    if use_quantized:
        n = reload_exl3_experts_quantized(model, checkpoint_dir, compute_dtype = dtype)
        if n:
            print(
                f"Unsloth: kept {n} MoE expert matrices EXL3-quantized "
                f"(reconstruct-on-forward) to fit VRAM."
            )
            return n
        # Fall back to dense if the quantized path found nothing.
    return reload_exl3_experts(model, checkpoint_dir, compute_dtype = dtype)
