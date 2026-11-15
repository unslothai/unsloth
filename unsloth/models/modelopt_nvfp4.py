# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""ModelOpt NVFP4 checkpoints (nvidia/*-NVFP4) load as compressed-tensors NVFP4A16, weights only.

No transformers release has a ModelOpt quantizer, so the config is rewritten to compressed-tensors
``nvfp4-pack-quantized``. ModelOpt decodes e2m1 * scale * weight_scale_2, compressed-tensors e2m1 * (scale /
global_scale): Linears expose ModelOpt names while loading, then global_scale = 1 / weight_scale_2 (fp32, raw shards).
"""

import fnmatch
import functools
import glob
import json
import os
import re
from typing import Optional

import torch

__all__ = [
    "modelopt_nvfp4_plan",
    "arm_modelopt_nvfp4_loading",
    "install_modelopt_nvfp4_loading",
    "is_modelopt_nvfp4_config",
    "refuse_modelopt_nvfp4_merged_save",
    "MODELOPT_NVFP4_ATTR",
]

MODELOPT_NVFP4_ATTR = "_unsloth_modelopt_nvfp4"
_PLAN_ATTR = "_unsloth_modelopt_nvfp4_plan"
_DROPPED = (r"\.input_scale$", r"\.k_scale$", r"\.v_scale$")


def _as_dict(quant):
    from .modelopt_fp8 import _as_dict as as_dict
    return as_dict(quant)


def _quant_block(quant):
    inner = quant.get("quantization")
    return inner if isinstance(inner, dict) else {}


def is_modelopt_nvfp4_config(quant) -> bool:
    quant = _as_dict(quant)
    if quant is None or str(quant.get("quant_method", "")).lower() != "modelopt":
        return False
    algo = quant.get("quant_algo") or _quant_block(quant).get("quant_algo")
    return str(algo).upper() == "NVFP4"


def _group_ok(group) -> bool:
    if not isinstance(group, dict):
        return False
    w = group.get("weights")
    if not isinstance(w, dict):
        return False
    return (
        str(w.get("type", "float")).lower() == "float"
        and int(w.get("num_bits", 4) or 0) == 4
        and int(w.get("group_size", 16) or 0) == 16
        and not w.get("dynamic", False)
        and list(group.get("targets") or ["Linear"]) == ["Linear"]
    )


def _ct_ignore(names, config) -> list:
    from .modelopt_fp8 import _checkpoint_renames, _renamed

    renames = _checkpoint_renames(config)
    out = []
    for name in names:
        variants = [str(name), _renamed(str(name), renames)]
        variants += ["model." + v for v in variants if not v.startswith(("*", "model."))]
        for v in variants:
            # compressed-tensors matches exact names, or `re:` patterns with re.match.
            entry = "re:" + fnmatch.translate(v) if any(c in v for c in "*?[") else v
            if entry not in out:
                out.append(entry)
    return out


def modelopt_nvfp4_plan(config) -> Optional[dict]:
    """compressed-tensors NVFP4A16 quantization_config for a ModelOpt NVFP4 config, else None."""
    quant = _as_dict(getattr(config, "quantization_config", None))
    if quant is None or not is_modelopt_nvfp4_config(quant):
        return None
    block = _quant_block(quant)
    if int(block.get("group_size", 16) or 0) != 16:
        return None
    groups = quant.get("config_groups")
    if groups is not None:
        if not isinstance(groups, dict) or not groups or not all(map(_group_ok, groups.values())):
            return None
    ignore = quant.get("ignore")
    if ignore is None:
        ignore = quant.get("exclude_modules")
    if ignore is None:
        ignore = block.get("exclude_modules")
    weights = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": False,
    }
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": None,
                "output_activations": None,
                "format": "nvfp4-pack-quantized",
            }
        },
        "ignore": _ct_ignore(ignore or [], config),
        "kv_cache_scheme": None,
    }


def arm_modelopt_nvfp4_loading(config, verbose: bool = True) -> Optional[dict]:
    plan = modelopt_nvfp4_plan(config)
    if plan is None or not install_modelopt_nvfp4_loading():
        return None
    try:
        config.quantization_config = plan
        setattr(config, MODELOPT_NVFP4_ATTR, True)
    except Exception:
        return None
    from .packed_expert_loader import install_packed_expert_loading

    install_packed_expert_loading()
    refuse_modelopt_nvfp4_merged_save()
    if verbose:
        print(
            "Unsloth: NVIDIA ModelOpt NVFP4 checkpoint detected; loading its weights as compressed-tensors "
            "NVFP4 (W4A16: the activation and kv-cache scales are not used)."
        )
    return plan


def _header_keys(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def _files_from_name(config):
    """Safetensors shards of an already resolved local dir or hub snapshot (transformers 4 passes no file list)."""
    name = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if not name:
        return []
    if os.path.isdir(name):
        index = os.path.join(name, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, encoding = "utf-8") as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            return [os.path.join(name, s) for s in shards]
        return sorted(glob.glob(os.path.join(name, "*.safetensors")))
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return []
    rev = getattr(config, "_commit_hash", None)
    index = try_to_load_from_cache(name, "model.safetensors.index.json", revision = rev)
    if isinstance(index, str):
        with open(index, encoding = "utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        paths = [try_to_load_from_cache(name, s, revision = rev) for s in shards]
        return paths if all(isinstance(p, str) for p in paths) else []
    single = try_to_load_from_cache(name, "model.safetensors", revision = rev)
    return [single] if isinstance(single, str) else []


def _global_scale_index(files) -> dict:
    """{checkpoint module prefix: (file, key)} for every ModelOpt ``weight_scale_2``."""
    out = {}
    for path in files:
        path = str(path)
        if not path.endswith(".safetensors"):
            return {}
        for key in _header_keys(path):
            if key.endswith(".weight_scale_2"):
                out[key[: -len(".weight_scale_2")]] = (path, key)
    return out


def _lookup(name, index, tails):
    if name in index:
        return index[name]
    if not tails:
        for prefix in index:
            parts = prefix.split(".")
            for i in range(len(parts)):
                tails.setdefault(".".join(parts[i:]), []).append(prefix)
    parts = name.split(".")
    for i in range(len(parts)):
        hits = tails.get(".".join(parts[i:]), [])
        if len(hits) == 1:
            return index[hits[0]]
        if hits:
            return None
    return None


def _quantized_linears(model):
    for name, module in model.named_modules():
        if getattr(module, "quantization_scheme", None) is None:
            continue
        params = module._parameters
        if "weight_packed" in params and "weight_global_scale" in params:
            yield name, module


def _before_loading(model, checkpoint_files):
    config = getattr(model, "config", None)
    if config is None or not config.__dict__.pop(MODELOPT_NVFP4_ATTR, False):
        return
    index = _global_scale_index(checkpoint_files) if checkpoint_files else None
    if checkpoint_files and not index:
        return
    renamed = []
    for name, module in _quantized_linears(model):
        params = module._parameters
        packed = params.pop("weight_packed")
        glob_scale = params.pop("weight_global_scale")
        module.register_parameter("weight", torch.nn.Parameter(packed, requires_grad = False))
        module.register_parameter(
            "weight_scale_2",
            torch.nn.Parameter(
                torch.empty((), dtype = torch.float32, device = glob_scale.device),
                requires_grad = False,
            ),
        )
        renamed.append((name, module))
    current = getattr(model, "_keys_to_ignore_on_load_unexpected", None)
    if isinstance(current, set):
        current.update(_DROPPED)
    else:
        model._keys_to_ignore_on_load_unexpected = list(current or []) + list(_DROPPED)
    model.__dict__[_PLAN_ATTR] = {"modules": renamed, "index": index}


def _after_loading(model):
    plan = model.__dict__.pop(_PLAN_ATTR, None)
    if not plan:
        return 0
    index = plan["index"]
    if index is None:
        files = _files_from_name(model.config)
        index = _global_scale_index(files) if files else {}
    tails = {}
    handles = {}
    from safetensors import safe_open

    with torch.no_grad():
        for name, module in plan["modules"]:
            params = module._parameters
            packed = params.pop("weight")
            loaded = params.pop("weight_scale_2")
            where = _lookup(name, index, tails) if index else None
            if where is not None:
                path, key = where
                if path not in handles:
                    handles[path] = safe_open(path, framework = "pt", device = "cpu")
                ws2 = handles[path].get_tensor(key)
            elif loaded.dtype == torch.float32 and loaded.device.type != "meta":
                ws2 = loaded.data
            else:
                raise RuntimeError(
                    f"Unsloth: could not read the fp32 weight_scale_2 of `{name}` from the ModelOpt NVFP4 "
                    "checkpoint shards; refusing to load it with a rounded global scale."
                )
            ws2 = ws2.to(device = packed.device, dtype = torch.float32).reshape(1)
            params["weight_packed"] = torch.nn.Parameter(packed.data, requires_grad = False)
            params["weight_global_scale"] = torch.nn.Parameter(
                torch.reciprocal(ws2), requires_grad = False
            )
    return len(plan["modules"])


_INSTALLED = False


def install_modelopt_nvfp4_loading() -> bool:
    """Wrap the compressed-tensors quantizer's before / after weight loading hooks; idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from transformers.quantizers.quantizer_compressed_tensors import (
            CompressedTensorsHfQuantizer,
        )
    except Exception:
        return False
    before = CompressedTensorsHfQuantizer._process_model_before_weight_loading
    after = CompressedTensorsHfQuantizer._process_model_after_weight_loading

    @functools.wraps(before)
    def _process_model_before_weight_loading(self, model, **kwargs):
        result = before(self, model, **kwargs)
        _before_loading(model, kwargs.get("checkpoint_files"))
        return result

    @functools.wraps(after)
    def _process_model_after_weight_loading(self, model, **kwargs):
        _after_loading(model)
        return after(self, model, **kwargs)

    CompressedTensorsHfQuantizer._process_model_before_weight_loading = (
        _process_model_before_weight_loading
    )
    CompressedTensorsHfQuantizer._process_model_after_weight_loading = (
        _process_model_after_weight_loading
    )
    _INSTALLED = True
    return True


def _source_is_modelopt_nvfp4(name, token = None) -> bool:
    from .modelopt_fp8 import _hf_quant_config_path, _modelopt_block_from_hf_quant_config

    if not name:
        return False
    try:
        if os.path.isdir(name):
            path = os.path.join(name, "config.json")
            quant = None
            if os.path.isfile(path):
                with open(path, encoding = "utf-8") as f:
                    quant = json.load(f).get("quantization_config")
        else:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(name, "config.json", token = token), encoding = "utf-8") as f:
                quant = json.load(f).get("quantization_config")
    except Exception:
        quant = None
    if quant is None:
        path = _hf_quant_config_path(name, token = token)
        quant = _modelopt_block_from_hf_quant_config(path) if path else None
    return is_modelopt_nvfp4_config(quant)


def refuse_modelopt_nvfp4_merged_save() -> None:
    """unsloth_zoo merges have no ModelOpt NVFP4 decode (uint8 codes would read as 16-bit weights): refuse."""
    try:
        import unsloth_zoo.saving_utils as zoo_saving
    except Exception:
        return
    determine = getattr(zoo_saving, "determine_base_model_source", None)
    if determine is None or getattr(determine, "_unsloth_modelopt_nvfp4", False):
        return

    @functools.wraps(determine)
    def determine_base_model_source(
        model_name,
        token = None,
        *args,
        **kwargs,
    ):
        result = determine(model_name, token, *args, **kwargs)
        source = result[0] if isinstance(result, tuple) and result else model_name
        if _source_is_modelopt_nvfp4(str(source), token):
            raise RuntimeError(
                f"Unsloth: `{source}` is a ModelOpt NVFP4 checkpoint, and merging LoRA into it is not supported "
                "yet (the merge would read its packed 4-bit codes as 16-bit weights). Save the LoRA adapter "
                'instead (`save_method = "lora"` / `model.save_pretrained`), or merge into the original '
                "16-bit model."
            )
        return result

    determine_base_model_source._unsloth_modelopt_nvfp4 = True
    zoo_saving.determine_base_model_source = determine_base_model_source
