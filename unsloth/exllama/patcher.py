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

"""Register the EXL3 quantizer with transformers and detect EXL3 checkpoints."""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

from .utils import is_exllama_available

_PATCHED = False
_PATCH_LOCK = threading.Lock()


def _relax_trainable(quantizer_cls) -> None:
    """Make the EXL3 HfQuantizer advertise itself as trainable.

    Unsloth freezes the base EXL3 weights and only trains LoRA adapters, which
    is exactly the QLoRA pattern. transformers/peft gate adapter attachment on
    ``HfQuantizer.is_trainable``; the stock EXL3 quantizer returns False, so we
    override it to True. This does not change the base weights - it only lets
    PEFT wrap the frozen layers with trainable adapters.
    """
    try:
        # ``is_trainable`` is defined as a property on the stock class; replace it
        # with one that returns True so peft accepts the model.
        quantizer_cls.is_trainable = property(lambda self: True)
    except Exception:
        pass


def _patch_moe_expert_replacement(quantizer_cls) -> None:
    """Replace fused MoE experts with quantized experts BEFORE weight loading.

    transformers instantiates a MoE's experts as full-size dense ``gate_up_proj``
    / ``down_proj`` Parameters. For a large MoE (e.g. a 35B with 256 experts)
    those dense params are tens of GB and OOM the GPU during load - long before
    our post-load reconstruction runs. We wrap the quantizer's
    ``_process_model_before_weight_loading`` so that, right after ExLlamaV3
    swaps the nn.Linear layers for meta placeholders, we ALSO swap every fused
    experts module for a meta-light :class:`Exl3QuantizedExperts` and load its
    per-expert trellis directly from the checkpoint. transformers then never
    allocates or loads the dense expert params.
    """
    if getattr(quantizer_cls, "_unsloth_moe_patched", False):
        return
    original = quantizer_cls._process_model_before_weight_loading

    def patched(self, model, *args, **kwargs):
        original(self, model, *args, **kwargs)
        try:
            import os

            path = getattr(model, "name_or_path", None)
            if not path or not os.path.isdir(path):
                return
            from .moe import install_quantized_experts_before_load

            install_quantized_experts_before_load(model, path)
        except Exception as e:
            # Non-fatal: fall back to the post-load reconstruction path.
            print(f"Unsloth: pre-load MoE expert replacement skipped ({e}).")

    quantizer_cls._process_model_before_weight_loading = patched
    quantizer_cls._unsloth_moe_patched = True


def _patch_get_modules_to_replace(quantizer_cls) -> None:
    """Fix EXL3 layer discovery for nested (VLM) module names.

    ExLlamaV3's ``get_modules_to_replace`` unconditionally rewrites every module
    name starting with ``model.language_model.`` to ``language_model.model.``
    before checking it against the checkpoint. That kludge suits some models but
    BREAKS others (e.g. Qwen3.5, whose checkpoint keeps ``model.language_model.``)
    - the rewritten name never matches, so almost no layer is replaced and the
    model loads mostly unquantized. We wrap the method to try BOTH the original
    and the rewritten name against the checkpoint, so a layer is replaced
    whenever either form matches.
    """
    if getattr(quantizer_cls, "_unsloth_modules_patched", False):
        return
    try:
        import torch
        from exllamav3.loader import SafetensorsCollection
        from exllamav3.integration.transformers import Exl3HfLinear
    except Exception:
        return
    import os as _os

    def get_modules_to_replace(self, model):
        path = model.name_or_path
        if not _os.path.isdir(path):
            raise ValueError("EXL3 model must be initialized from local directory")
        stc = SafetensorsCollection(path)
        group = [["sv", "svh"], ["su", "suh"], "trellis"]
        modules_to_replace = {}
        try:
            for name, module in tuple(model.named_modules()):
                if not isinstance(module, torch.nn.Linear):
                    continue
                # Candidate checkpoint names: the module path as-is, and the
                # exllamav3 language_model<->model swaps in both directions.
                candidates = [name]
                if name.startswith("model.language_model."):
                    candidates.append(
                        "language_model.model." + name[len("model.language_model.") :]
                    )
                if name.startswith("language_model.model."):
                    candidates.append(
                        "model.language_model." + name[len("language_model.model.") :]
                    )
                matched = next((c for c in candidates if stc.has_tensor_group(c, group)), None)
                if matched is not None:
                    modules_to_replace[name] = Exl3HfLinear(
                        module.in_features,
                        module.out_features,
                        stc.list_tensors(matched),
                    )
        finally:
            stc.close()
        return modules_to_replace

    quantizer_cls.get_modules_to_replace = get_modules_to_replace
    quantizer_cls._unsloth_modules_patched = True


def _patch_hf_linear_weight(hf_linear_cls) -> None:
    """Make ``Exl3HfLinear.weight`` an ``nn.Parameter`` with the real 2-D shape.

    The stock wrapper stores ``weight`` as a bare ``torch.zeros((1,), device=meta)``
    tensor. Newer transformers (>=5.6) finalize tied weights via
    ``get_parameter("lm_head.weight")``, which asserts the attribute is an
    ``nn.Parameter`` and raises otherwise. We wrap ``__init__`` so the placeholder
    is a frozen meta ``nn.Parameter`` shaped ``[out_features, in_features]``; this
    also lets shape-based logic (PEFT target discovery, Unsloth's EXL3 dequant)
    read the correct dimensions without materializing the dense weight.
    """
    import torch

    if getattr(hf_linear_cls, "_unsloth_weight_patched", False):
        return
    original_init = hf_linear_cls.__init__
    original_finalize = hf_linear_cls.finalize

    def patched_init(self, in_features, out_features, exl3_tensors, *args, **kwargs):
        original_init(self, in_features, out_features, exl3_tensors, *args, **kwargs)
        placeholder = torch.zeros(
            (int(out_features), int(in_features)),
            dtype = torch.float16,
            device = "meta",
        )
        self.weight = torch.nn.Parameter(placeholder, requires_grad = False)
        # Register the codebook side tensors (mcg/mul1) the stock wrapper omits,
        # else the default mcg codebook reconstructs garbage weights.
        key = getattr(self, "key", None)
        if key is not None:
            for subkey in ("mcg", "mul1"):
                meta = exl3_tensors.get(f"{key}.{subkey}")
                # Clear any stock placeholder attribute so register_buffer works.
                if hasattr(self, subkey):
                    try:
                        delattr(self, subkey)
                    except Exception:
                        pass
                if meta is not None:
                    t = torch.empty(meta["shape"], dtype = meta["torch_dtype"], device = "meta")
                    self.register_buffer(subkey, t)
                else:
                    self.register_buffer(subkey, None)

    def patched_finalize(self):
        original_finalize(self)
        # Re-create the inner layer with the codebook tensors wired in.
        mcg = getattr(self, "mcg", None)
        mul1 = getattr(self, "mul1", None)
        if (mcg is not None or mul1 is not None) and self.inner is not None:
            from exllamav3.modules.quant.exl3 import LinearEXL3
            self.inner = LinearEXL3(
                config = None,
                in_features = self.in_features,
                out_features = self.out_features,
                trellis = self.trellis,
                suh = self.suh,
                svh = self.svh,
                su = self.su,
                sv = self.sv,
                mcg = mcg,
                mul1 = mul1,
                bias = self.bias,
                out_dtype = torch.float16,
                transformers_fix = True,
            )

    hf_linear_cls.__init__ = patched_init
    hf_linear_cls.finalize = patched_finalize
    hf_linear_cls._unsloth_weight_patched = True


class _MissingKeyTolerantDict(dict):
    """A dict whose ``del`` is a no-op for absent keys.

    Used to neutralise exllamav3's unconditional ``del model_classes["mtp"]`` on
    Qwen3.5 causal-LM configs that never registered an "mtp" entry.
    """

    def __delitem__(self, key):
        if key in self:
            super().__delitem__(key)


def _patch_qwen35_mtp_keyerror() -> None:
    """Guard exllamav3's Qwen3.5 config against the MTP-free KeyError.

    exllamav3 (<=0.0.43) does ``del self.model_classes["mtp"]`` whenever a
    Qwen3.5 model reports ``mtp_num_hidden_layers == 0``, but the causal-LM
    config variants never add that key, so the delete raises KeyError and the
    model cannot be quantized/loaded. We wrap the affected config classes'
    ``__init__`` so ``model_classes`` tolerates the missing-key delete.
    """
    # The del happens inside a parent ``__init__``, so wrapping subclasses is
    # too late. Instead patch the base ``Config.__setattr__`` so that any
    # ``model_classes`` dict it stores becomes missing-key tolerant.
    try:
        from exllamav3.model.config import Config as _Config
        if not getattr(_Config, "_unsloth_mtp_patched", False):
            _orig_setattr = _Config.__setattr__

            def _patched_setattr(self, key, value):
                if (
                    key == "model_classes"
                    and isinstance(value, dict)
                    and not isinstance(value, _MissingKeyTolerantDict)
                ):
                    value = _MissingKeyTolerantDict(value)
                _orig_setattr(self, key, value)

            _Config.__setattr__ = _patched_setattr
            _Config._unsloth_mtp_patched = True
    except Exception:
        pass


def _patch_grouped_mm_fallback() -> None:
    """Make transformers' grouped-MoE matmul fall back on non-Hopper GPUs.

    transformers 5's default MoE experts forward uses ``torch._grouped_mm``,
    which only runs on CUDA compute capability >= 9.0 (Hopper). On consumer GPUs
    it raises at runtime, breaking EXL3 MoE inference/training. We wrap the
    module-level ``_grouped_mm`` so that, when the fused kernel is unavailable,
    it computes the same result with a portable per-group matmul.
    """
    try:
        import torch
        import transformers.integrations.moe as _moe
    except Exception:
        return
    if getattr(_moe, "_unsloth_grouped_mm_patched", False):
        return

    _orig_grouped_mm = getattr(_moe, "_grouped_mm", None)
    if _orig_grouped_mm is None:
        return

    def _portable_grouped_mm(
        input,
        weight,
        offs = None,
        **kwargs,
    ):
        # weight: [num_groups, K, N]; input: [total_tokens, K]; offs: cumulative
        # row counts per group. Compute each group's matmul with a plain mm.
        try:
            return _orig_grouped_mm(input, weight, offs = offs, **kwargs)
        except RuntimeError as e:
            if "compute capability" not in str(e) and "_grouped_mm" not in str(e):
                raise
        out = torch.empty(
            (input.shape[0], weight.shape[-1]),
            dtype = input.dtype,
            device = input.device,
        )
        start = 0
        offs_list = offs.tolist() if offs is not None else [input.shape[0]]
        for g, end in enumerate(offs_list):
            if end > start:
                out[start:end] = torch.matmul(input[start:end], weight[g].to(input.dtype))
            start = end
        return out

    _moe._grouped_mm = _portable_grouped_mm
    _moe._unsloth_grouped_mm_patched = True

    # Also patch torch._grouped_mm globally so *every* caller (transformers AND
    # unsloth_zoo's compiled grouped-MoE LoRA kernels) falls back on non-Hopper
    # GPUs. The fallback returns identical results via per-group matmuls.
    _patch_torch_grouped_mm()


def _patch_torch_grouped_mm() -> None:
    """Wrap ``torch._grouped_mm`` to fall back to per-group matmuls off Hopper."""
    try:
        import torch
    except Exception:
        return
    if getattr(torch, "_unsloth_grouped_mm_patched", False):
        return
    _orig = getattr(torch, "_grouped_mm", None)
    if _orig is None:
        return

    def _fallback(
        input,
        weight,
        offs = None,
        bias = None,
        out_dtype = None,
        **kwargs,
    ):
        try:
            return _orig(input, weight, offs = offs, bias = bias, out_dtype = out_dtype, **kwargs)
        except RuntimeError as e:
            if "compute capability" not in str(e):
                raise
        # input: [total_tokens, K]; weight: [num_groups, K, N]; offs: cumulative
        # per-group row counts. Emulate with per-group matmuls.
        dtype = out_dtype or input.dtype
        n = weight.shape[-1]
        out = torch.empty((input.shape[0], n), dtype = dtype, device = input.device)
        offs_list = offs.tolist() if offs is not None else [input.shape[0]]
        start = 0
        for g, end in enumerate(offs_list):
            if end > start:
                out[start:end] = torch.matmul(input[start:end].to(dtype), weight[g].to(dtype))
            start = end
        if bias is not None:
            out = out + bias
        return out

    torch._grouped_mm = _fallback
    torch._unsloth_grouped_mm_patched = True


def exllama_supported_architectures() -> set:
    """Return the set of architecture strings ExLlamaV3 can quantize/load.

    Reads ExLlamaV3's own architecture registry so the list stays correct as the
    library adds models. Returns an empty set if exllamav3 is unavailable OR if
    the registry could not be read (older/newer layout, import failure). Callers
    must treat an empty set as "unknown", not "nothing is supported" - see
    :func:`exllama_supports_arch`.
    """
    if not is_exllama_available():
        return set()
    try:
        from exllamav3.architecture.architectures import get_architectures
        return set(get_architectures().keys())
    except Exception:
        return set()


def exllama_supports_arch(model_dir_or_arch) -> bool:
    """True if ExLlamaV3 (probably) supports this model's architecture.

    Accepts either a local model directory (reads ``architectures`` from
    ``config.json``, including a nested ``text_config``) or an architecture
    string directly. Used to decide up front whether an EXL3 load can succeed,
    so the loader can fall back to bitsandbytes / 16-bit for architectures
    ExLlamaV3 does not yet implement (e.g. ``OlmoeForCausalLM``) - preserving
    Unsloth's near-universal model support instead of crashing.

    Fails **open**: if exllamav3's architecture registry cannot be read (empty
    set), we return ``True`` so a genuinely-supported model (Llama, Qwen, ...)
    is *not* silently downgraded to bnb, and an explicit EXL3 request is not
    spuriously rejected. A truly unsupported arch is then caught later by the
    converter. We only return ``False`` when the registry was read successfully
    and the model's architecture is genuinely absent from it.
    """
    supported = exllama_supported_architectures()
    if not supported:
        # Registry unreadable: fail open (don't block a supported model).
        return True
    if isinstance(model_dir_or_arch, str) and not os.path.isdir(model_dir_or_arch):
        return model_dir_or_arch in supported

    cfg = _read_json(os.path.join(str(model_dir_or_arch), "config.json"))
    if not isinstance(cfg, dict):
        # Config unreadable - cannot prove it is unsupported; let the converter
        # decide rather than silently downgrading.
        return True
    archs = list(cfg.get("architectures") or [])
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        archs += list(tc.get("architectures") or [])
    if not archs:
        # No architectures declared - cannot prove unsupported; defer to loader.
        return True
    # A model is loadable if ANY of its declared architectures is supported.
    return any(a in supported for a in archs)


def patch_transformers_exl3(force: bool = False) -> bool:
    """Register EXL3 with transformers' quantizer registries. Idempotent.

    Returns True if the patch is active (already or newly applied), False if
    exllamav3 is unavailable.
    """
    global _PATCHED
    if _PATCHED and not force:
        return True
    if not is_exllama_available():
        return False

    with _PATCH_LOCK:
        # Re-check under the lock: another thread may have patched while we
        # waited to acquire it.
        if _PATCHED and not force:
            return True
        try:
            from exllamav3.integration.transformers import (
                patch_transformers as _exl3_patch_transformers,
                Exl3HfQuantizer,
            )
        except Exception:
            return False

        # Apply exllamav3's own registry injection.
        _exl3_patch_transformers()
        # Allow LoRA/QLoRA training on top of the frozen EXL3 base.
        _relax_trainable(Exl3HfQuantizer)
        # Fix EXL3 layer discovery for nested (VLM) module names, so Qwen3.5 and
        # other model.language_model.* checkpoints get their layers quantized.
        _patch_get_modules_to_replace(Exl3HfQuantizer)
        # Replace fused MoE experts with quantized experts before weight loading
        # so a large MoE's dense expert params never OOM the GPU during load.
        _patch_moe_expert_replacement(Exl3HfQuantizer)
        # Make Exl3HfLinear.weight an nn.Parameter so newer transformers'
        # tied-weight finalization does not reject it, and wire the codebook.
        try:
            from exllamav3.integration.transformers import Exl3HfLinear
            _patch_hf_linear_weight(Exl3HfLinear)
        except Exception:
            pass

        # Guard an exllamav3 (<=0.0.43) unconditional del of a missing "mtp"
        # key that KeyErrors on MTP-free Qwen3.5 configs.
        _patch_qwen35_mtp_keyerror()

        # Fall back from the Hopper-only grouped_mm MoE kernel to a portable
        # matmul so EXL3 MoEs run on consumer GPUs.
        _patch_grouped_mm_fallback()

        _PATCHED = True
        return True


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding = "utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_exl3_model_dir(path: str) -> bool:
    """Return True if ``path`` is a local directory holding an EXL3 checkpoint.

    An EXL3 checkpoint is identified by a ``quantization_config`` with
    ``quant_method == "exl3"`` in ``config.json`` (or a standalone
    ``quantization_config.json`` that ExLlamaV3 writes), or, defensively, by the
    presence of ``.trellis`` tensors in the safetensors index.
    """
    if not isinstance(path, str) or not os.path.isdir(path):
        return False

    cfg = _read_json(os.path.join(path, "config.json"))
    if cfg is not None:
        qc = cfg.get("quantization_config")
        if isinstance(qc, dict):
            method = str(qc.get("quant_method", "")).lower()
            if method == "exl3":
                return True

    # ExLlamaV3 also writes a compiled quantization_config.json.
    qc_file = os.path.join(path, "quantization_config.json")
    if os.path.isfile(qc_file):
        qc = _read_json(qc_file)
        if isinstance(qc, dict):
            method = str(qc.get("quant_method", "exl3")).lower()
            if method in ("exl3", ""):
                return True

    # Fallback: scan the safetensors index for a trellis tensor.
    index = _read_json(os.path.join(path, "model.safetensors.index.json"))
    if isinstance(index, dict):
        weight_map = index.get("weight_map", {})
        if any(k.endswith(".trellis") for k in weight_map):
            return True

    return False


def read_exl3_bitrate(path: str) -> Optional[float]:
    """Best-effort read of a checkpoint's stored EXL3 decoder bitrate."""
    for fname in ("quantization_config.json", "config.json"):
        cfg = _read_json(os.path.join(path, fname))
        if cfg is None:
            continue
        qc = cfg.get("quantization_config", cfg) if fname == "config.json" else cfg
        if isinstance(qc, dict):
            for key in ("bits", "bpw", "final_bits", "final_bpw"):
                val = qc.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
    return None


def is_nonexl3_quantized_dir(path: str) -> bool:
    """True if ``path`` is already quantized by a non-EXL3 method (bnb/gptq/awq/
    hqq/...); such a model must load with its own backend, not be re-quantized."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    cfg = _read_json(os.path.join(path, "config.json"))
    if not isinstance(cfg, dict):
        return False
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    method = str(qc.get("quant_method", "")).lower()
    return method not in ("", "exl3")


def is_peft_adapter_dir(path: str) -> bool:
    """True if ``path`` is a PEFT adapter repo (adapter_config.json, no base
    config.json); it must load its base model, not be quantized as a base."""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    has_adapter = os.path.isfile(os.path.join(path, "adapter_config.json"))
    has_base = os.path.isfile(os.path.join(path, "config.json"))
    return has_adapter and not has_base
