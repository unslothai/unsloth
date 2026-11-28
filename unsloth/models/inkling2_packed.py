# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Inkling 2 routed experts kept packed as released (NVFP4 / MXFP8), opt-in: ``UNSLOTH_INKLING2_PACKED_EXPERTS=1``.

The released checkpoints store each layer's experts as stacks: ``mlp.experts.w13_weight`` (gate / up rows
interleaved) and ``mlp.experts.w2_weight``, with ``_weight_scale`` block scales and, for NVFP4, a per-expert
``_weight_amax`` and a scalar ``_weight_outer_scale_denominator``:

- NVFP4: U8 codes ``(E, N, K / 2)`` (even element in the low nibble), F8_E4M3 scales ``(E, N, K / 16)``,
  value = e2m1 * scale * (amax / outer_scale_denominator).
- MXFP8: F8_E4M3 values ``(E, N, K)``, U8 e8m0 scales ``(E, N, K / 32)``, value = e4m3 * 2 ** (scale - 127).

transformers dequantizes them while loading (``Inkling2LoopDequantizeExperts``); by default Unsloth keeps that, and
bitsandbytes then rounds them again to 4 bit. With the flag the experts become unsloth_zoo's frozen
``Nvfp4ExpertParam`` (multiply mode, global = amax / denominator per expert) and ``Mxfp8ExpertParam`` holding the
exact checkpoint bytes (only the w13 rows are de-interleaved, a row permutation applied to codes and scales alike),
and LoRA trains on the released weights. Under expert parallel loading each rank reads only its own experts
[lo, hi) of every stack from the checkpoint.
"""

import os
import re
from typing import Optional

import torch

__all__ = [
    "INKLING2_PACKED_EXPERTS_ENV",
    "inkling2_packed_experts_enabled",
    "install_inkling2_packed_expert_loading",
    "inkling2_expert_format",
    "warn_if_inkling2_packed_experts_inert",
]

INKLING2_PACKED_EXPERTS_ENV = "UNSLOTH_INKLING2_PACKED_EXPERTS"
_KEY = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts)\.(?P<proj>w13|w2)_(?P<suffix>weight|weight_scale|weight_amax|"
    r"weight_outer_scale_denominator)$"
)
_STACK = {"w13": "gate_up_proj", "w2": "down_proj"}
_ST_DTYPES = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}


def inkling2_packed_experts_enabled() -> bool:
    """Opt-in; full finetuning trains the experts, so they load dequantized as before."""
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        return False
    return os.environ.get(INKLING2_PACKED_EXPERTS_ENV, "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


_INERT_WARNED = False


def _quant_method(config) -> str:
    text = (
        config.get("text_config")
        if isinstance(config, dict)
        else getattr(config, "text_config", None)
    )
    for cfg in (config, text):
        quant = (
            cfg.get("quantization_config")
            if isinstance(cfg, dict)
            else getattr(cfg, "quantization_config", None)
        )
        if quant is None:
            continue
        if not isinstance(quant, dict):
            quant = quant.to_dict() if hasattr(quant, "to_dict") else vars(quant)
        return str(quant.get("quant_method", "")).lower()
    return ""


def warn_if_inkling2_packed_experts_inert(
    config,
    model_name = None,
    load_in_4bit = False,
) -> bool:
    """Say once that the flag does nothing for an Inkling 2 checkpoint without the released packed experts, e.g. the
    bitsandbytes 4bit copy ``load_in_4bit = True`` resolves to (its experts are already NF4). Returns True if it
    printed. Not called for the released checkpoint, where the flag engages."""
    global _INERT_WARNED
    if _INERT_WARNED or not inkling2_packed_experts_enabled():
        return False
    from .inkling2 import is_inkling2_config

    if not is_inkling2_config(config):
        return False
    _INERT_WARNED = True
    name = f"`{model_name}`" if model_name else "this checkpoint"
    if _quant_method(config) == "bitsandbytes":
        why = (
            " (the bitsandbytes 4bit copy that load_in_4bit = True loads stores its experts in NF4)"
        )
    elif load_in_4bit:
        why = " (load_in_4bit = True)"
    else:
        why = ""
    print(
        f"Unsloth: {INKLING2_PACKED_EXPERTS_ENV}=1 has no effect: {name} has no packed NVFP4 / MXFP8 experts"
        + why
        + ". To train on the packed experts, load the released Inkling 2 checkpoint with load_in_4bit = False; "
        "the 4bit load keeps using less memory."
    )
    return True


def _zoo():
    try:
        from unsloth_zoo import packed_experts
    except Exception:
        return None
    return packed_experts if hasattr(packed_experts, "Nvfp4ExpertParam") else None


def inkling2_expert_format(parts) -> Optional[str]:
    """ "nvfp4", "mxfp8" or None for one stack's checkpoint entries ``{suffix: (shape, dtype)}``."""
    weight, scale = parts.get("weight"), parts.get("weight_scale")
    if weight is None or scale is None or len(weight[0]) != 3 or len(scale[0]) != 3:
        return None
    (E, N, K), (sshape, sdtype) = weight[0], scale
    if (
        weight[1] == torch.uint8
        and sdtype == torch.float8_e4m3fn
        and tuple(sshape) == (E, N, 2 * K // 16)
        and tuple(parts.get("weight_amax", ((), None))[0]) == (E,)
        and parts.get("weight_outer_scale_denominator", ((), None))[0] is not None
    ):
        return "nvfp4"
    if (
        weight[1] == torch.float8_e4m3fn
        and sdtype == torch.uint8
        and tuple(sshape) == (E, N, K // 32)
    ):
        return "mxfp8"
    return None


def _scan(checkpoint_files):
    """{checkpoint experts prefix: {proj: {suffix: ((shape), dtype, file, key)}}} from the shard headers."""
    from safetensors import safe_open

    found = {}
    for path in checkpoint_files or []:
        path = str(path)
        if not path.endswith(".safetensors"):
            return {}
        with safe_open(path, framework = "pt") as f:
            for key in f.keys():
                m = _KEY.match(key)
                if m is None:
                    continue
                piece = f.get_slice(key)
                found.setdefault(m["prefix"], {}).setdefault(m["proj"], {})[m["suffix"]] = (
                    tuple(piece.get_shape()),
                    _ST_DTYPES.get(piece.get_dtype()),
                    path,
                    key,
                )
    return found


def _layer_tail(name):
    """``...layers.7.mlp.experts`` -> ``layers.7.mlp.experts`` (checkpoint ``model.llm`` vs model ``language_model``)."""
    m = re.search(r"(layers\.\d+\.mlp\.experts)$", name)
    return m.group(1) if m else None


def _plan(model, checkpoint_files):
    """All or nothing: every Inkling 2 experts module maps to a complete packed w13 / w2 set of one format each."""
    if _zoo() is None:
        return None
    experts = [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ in ("Inkling2Experts", "Inkling2LoopExperts")
        and getattr(module, "gate_up_proj", None) is not None
        and getattr(module, "down_proj", None) is not None
    ]
    if not experts:
        return None
    scanned = _scan(checkpoint_files)
    by_tail = {}
    for prefix in scanned:
        tail = _layer_tail(prefix)
        if tail is None or tail in by_tail:
            return None
        by_tail[tail] = prefix
    if len(by_tail) != len(experts):
        return None
    plan = []
    for name, module in experts:
        prefix = by_tail.get(_layer_tail(name))
        if prefix is None:
            return None
        E, two_i, H = module.gate_up_proj.shape
        if tuple(module.down_proj.shape) != (E, H, two_i // 2):
            return None
        item = {"name": name, "module": module, "prefix": prefix, "stacks": {}}
        for proj, (rows, cols) in (("w13", (two_i, H)), ("w2", (H, two_i // 2))):
            parts = scanned[prefix].get(proj, {})
            fmt = inkling2_expert_format({k: (v[0], v[1]) for k, v in parts.items()})
            if fmt is None:
                return None
            code_cols = cols // 2 if fmt == "nvfp4" else cols
            if tuple(parts["weight"][0]) != (E, rows, code_cols):
                return None
            item["stacks"][_STACK[proj]] = (proj, fmt, parts)
        plan.append(item)
    return plan


def _deinterleave_rows(t):
    """transformers' Interleave(dim = 1) on an (E, 2I, *) stack: [g0, u0, g1, u1, ...] -> [g..., u...]."""
    E, N = t.shape[:2]
    return t.reshape(E, N // 2, 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape).contiguous()


def _read(parts, suffix, lo, hi, files):
    from safetensors import safe_open

    from ._windows_safetensors import pread_backend_kwargs

    shape, _, path, key = parts[suffix]
    if path not in files:
        files[path] = safe_open(path, framework = "pt", device = "cpu", **pread_backend_kwargs())
    piece = files[path].get_slice(key)
    if lo is None or len(shape) == 0 or shape[0] == 1 and suffix.endswith("denominator"):
        return piece[:]
    return piece[lo:hi]


def _build(item, stack, cut, files, dtype, device):
    pe = _zoo()
    proj, fmt, parts = item["stacks"][stack]
    E = parts["weight"][0][0]
    lo, hi = (cut[0], cut[1]) if cut is not None else (None, None)
    n = E if cut is None else hi - lo
    codes = _read(parts, "weight", lo, hi, files).view(torch.uint8)
    scales = _read(parts, "weight_scale", lo, hi, files).view(torch.uint8)
    if proj == "w13":
        codes, scales = _deinterleave_rows(codes), _deinterleave_rows(scales)
    codes, scales = codes.to(device).contiguous(), scales.to(device).contiguous()
    if fmt == "mxfp8":
        return pe.Mxfp8ExpertParam(codes, packed_scales = scales, packed_dtype = dtype)
    amax = _read(parts, "weight_amax", lo, hi, files).float().reshape(n)
    denominator = (
        _read(parts, "weight_outer_scale_denominator", None, None, files).float().reshape(-1)
    )
    if denominator.numel() not in (1, E):
        raise RuntimeError(
            f"Unsloth: {item['name']} {proj} has {denominator.numel()} outer scale denominators, expected 1."
        )
    if denominator.numel() == E and cut is not None:
        denominator = denominator[lo:hi]
    glob = (amax / denominator).reshape(n, 1).to(device).contiguous()
    return pe.Nvfp4ExpertParam(
        codes,
        packed_scales = scales,
        packed_global = glob,
        packed_dtype = dtype,
        packed_global_mode = "multiply",
    )


def _ep_cut(model, module):
    import sys

    tpn = sys.modules.get(__name__.rpartition(".")[0] + ".tp_native")
    cut = getattr(tpn, "ep_packed_load_cut", None)
    return cut(model, module) if callable(cut) else None


def _device_of(model, name, device_map):
    """Where the device map puts the experts module `name` (its own stacks are already taken out of the model)."""
    if isinstance(device_map, dict) and device_map:
        try:
            from transformers.integrations.accelerate import get_device
            device = get_device(device_map, f"{name}.gate_up_proj", valid_torch_device = True)
            return torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        except Exception:
            pass
    parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
    for p in parent.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _offloads_to_disk(device_map):
    return isinstance(device_map, dict) and any(
        str(v) in ("disk", "meta") for v in device_map.values()
    )


def _drop_expert_converters(weight_mapping):
    from .inkling2 import _has_dequant_op, _is_expert_transform
    kept = [t for t in weight_mapping or [] if not (_is_expert_transform(t) and _has_dequant_op(t))]
    return kept if len(kept) < len(weight_mapping or []) else None


def _fill(
    model,
    plan,
    device_map = None,
):
    from .packed_expert_loader import _mark_cut
    files = {}
    for item in plan:
        module = item["module"]
        device = _device_of(model, item["name"], device_map)
        for stack in ("gate_up_proj", "down_proj"):
            param = _build(item, stack, item["cut"], files, item["dtype"], device)
            param._is_hf_initialized = (
                True  # the checkpoint's bytes: transformers must not re-initialize them
            )
            module.__dict__.pop(stack, None)
            module._parameters[stack] = param
            if item["cut"] is not None:
                _mark_cut(module, stack, param, item["cut"])
        module._unsloth_packed_expert_format = "inkling2"


_INSTALLED = False


def install_inkling2_packed_expert_loading() -> bool:
    """Patch transformers' checkpoint load (5.x weight converters) for Inkling 2 models; idempotent. Inert unless
    UNSLOTH_INKLING2_PACKED_EXPERTS=1 and every experts module maps to a complete packed set."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import transformers.core_model_loading  # noqa: F401  (5.x only)
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return False
    if _zoo() is None:
        return False
    original = PreTrainedModel._load_pretrained_model
    if getattr(original, "_unsloth_inkling2_packed", False):
        _INSTALLED = True
        return True

    def _load_pretrained_model(model, state_dict, checkpoint_files, load_config, *args, **kwargs):
        plan = None
        if (
            inkling2_packed_experts_enabled()
            and state_dict is None
            and checkpoint_files
            # disk offload dispatches before the fill could place the stacks; transformers' path is kept there
            and not _offloads_to_disk(getattr(load_config, "device_map", None))
        ):
            try:
                plan = _plan(model, checkpoint_files)
                kept = (
                    _drop_expert_converters(getattr(load_config, "weight_mapping", None))
                    if plan
                    else None
                )
                if plan and kept is None:
                    plan = None
            except Exception as exc:
                print(
                    f"Unsloth: could not plan packed Inkling 2 experts ({exc}); loading them dequantized."
                )
                plan = None
        if not plan:
            return original(model, state_dict, checkpoint_files, load_config, *args, **kwargs)
        # The packed expert keys are read below, not by transformers (its loading report runs after this call).
        ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", None)
        model.__dict__["_keys_to_ignore_on_load_unexpected"] = set(ignore or ()) | {
            r"\.mlp\.experts\.w(13|2)_weight"
        }
        import dataclasses

        full_mapping = list(load_config.weight_mapping or [])
        load_config = dataclasses.replace(load_config, weight_mapping = kept)
        originals = []
        for item in plan:
            module = item["module"]
            item["dtype"] = module.gate_up_proj.dtype
            # Under expert parallel loading only this rank's experts are read (decided on the meta model).
            item["cut"] = _ep_cut(model, module)
            for stack in ("gate_up_proj", "down_proj"):
                param = module._parameters.pop(stack)
                originals.append((module, stack, param))
                module.__dict__[stack] = torch.empty(0, dtype = param.dtype, device = "meta")
        try:
            result = original(model, state_dict, checkpoint_files, load_config, *args, **kwargs)
            with torch.no_grad():
                _fill(model, plan, getattr(load_config, "device_map", None))
            # transformers records the converters it used for save_pretrained's reverse mapping: the dropped
            # expert converters still describe this checkpoint's layout, so a save writes w13 / w2 as before.
            used = {id(c) for c in getattr(model, "_weight_conversions", None) or ()}
            dropped = {id(c) for c in full_mapping} - {id(c) for c in kept}
            model._weight_conversions = [
                c for c in full_mapping if id(c) in used or id(c) in dropped
            ]
        except BaseException:
            for module, stack, param in originals:
                module.__dict__.pop(stack, None)
                module._parameters[stack] = param
            raise
        print(
            f"Unsloth: kept {2 * len(plan)} Inkling 2 routed-expert stacks packed as released (NVFP4 / MXFP8)"
            + (", this rank's experts only." if plan[0]["cut"] is not None else ".")
        )
        return result

    _load_pretrained_model._unsloth_inkling2_packed = True
    PreTrainedModel._load_pretrained_model = staticmethod(_load_pretrained_model)
    _INSTALLED = True
    return True
