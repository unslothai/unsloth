# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Keep packed per-expert NVFP4 / MXFP8 checkpoint tensors packed in transformers 5 fused expert stacks.

transformers 5 dequantizes them while fusing: NVFP4 comes out multiplied by its global scale (5.18 to 5.23) and
MXFP8 fails. So fused stacks become packed placeholders, their converters are dropped, and after loading they are
filled with exact checkpoint bytes. ``UNSLOTH_PACKED_EXPERTS=0`` installs them dequantized to 16 bit instead.
"""

import os
import re
from typing import Optional

import torch

__all__ = [
    "install_packed_expert_loading",
    "packed_experts_enabled",
    "quantize_moe_experts_if_requested",
    "PACKED_EXPERTS_ENV",
]

PACKED_EXPERTS_ENV = "UNSLOTH_PACKED_EXPERTS"
_PLAN_ATTR = "_unsloth_packed_expert_plan"
_PLACEHOLDER = "unsloth_packed_"

_EXPERT_KEY = re.compile(
    r"^(?P<prefix>.+)\.experts\.(?P<idx>\d+)\.(?P<proj>[A-Za-z0-9_]+)\."
    r"(?P<suffix>weight_packed|weight_scale_2|weight_scale|weight_global_scale|weight|input_scale|weight_shape)$"
)
_ST_DTYPES = {
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "I64": torch.int64,
    "I32": torch.int32,
}


def packed_experts_enabled() -> bool:
    """Packed by default; 0 or full finetuning installs them dequantized to 16 bit."""
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        return False
    return os.environ.get(PACKED_EXPERTS_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _zoo():
    try:
        from unsloth_zoo import packed_experts
    except Exception:
        return None
    return packed_experts


def _scan(checkpoint_files) -> dict:
    """{ckpt prefix: {(idx, proj, suffix): (file, shape, dtype)}} for per-expert keys, from shard headers."""
    from safetensors import safe_open

    found: dict = {}
    for path in checkpoint_files or []:
        path = str(path)
        if not path.endswith(".safetensors"):
            return {}
        with safe_open(path, framework = "pt") as f:
            for key in f.keys():
                m = _EXPERT_KEY.match(key)
                if m is None:
                    continue
                piece = f.get_slice(key)
                dtype = _ST_DTYPES.get(piece.get_dtype())
                found.setdefault(m["prefix"], {})[(int(m["idx"]), m["proj"], m["suffix"])] = (
                    path,
                    tuple(piece.get_shape()),
                    dtype,
                    key,
                )
    return found


def _format_of(entries) -> Optional[str]:
    """nvfp4 (compressed-tensors), nvfp4_modelopt or mxfp8 when every expert projection is packed alike, else None."""
    by_proj: dict = {}
    for (idx, proj, suffix), (_, shape, dtype, _) in entries.items():
        by_proj.setdefault((idx, proj), {})[suffix] = (shape, dtype)
    kinds = set()
    for parts in by_proj.values():
        if (
            "weight_packed" in parts
            and parts["weight_packed"][1] == torch.uint8
            and parts.get("weight_scale", (None, None))[1] == torch.float8_e4m3fn
            and "weight_global_scale" in parts
        ):
            kinds.add("nvfp4")
        elif (
            parts.get("weight", (None, None))[1] == torch.uint8
            and "weight_scale_2" in parts
            and parts.get("weight_scale", (None, None))[1] == torch.float8_e4m3fn
        ):
            kinds.add("nvfp4_modelopt")
        elif (
            parts.get("weight", (None, None))[1] == torch.float8_e4m3fn
            and parts.get("weight_scale", (None, None))[1] == torch.uint8
        ):
            shape, _ = parts["weight"]
            sshape, _ = parts["weight_scale"]
            kinds.add(
                "mxfp8"
                if len(shape) == 2 and tuple(sshape) == (shape[0], shape[1] // 32)
                else "other"
            )
        else:
            kinds.add("other")
    return kinds.pop() if len(kinds) == 1 and "other" not in kinds else None


def _tails(name):
    parts = name.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]


def _match_prefix(module_name, prefixes):
    """Checkpoint prefix of an experts module (``a.b.experts`` vs ``x.a.b`` / ``a.b``), via the longest shared tail."""
    if not module_name.endswith(".experts"):
        return None
    for tail in _tails(module_name[: -len(".experts")]):
        hits = [p for p in prefixes if p == tail or p.endswith("." + tail)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            continue
        return None
    return None


def _fused_experts(model):
    out = []
    for name, module in model.named_modules():
        params = module._parameters
        gu, dn = params.get("gate_up_proj"), params.get("down_proj")
        if gu is not None and dn is not None and gu.dim() == 3 and dn.dim() == 3:
            out.append((name, module))
    return out


def _orientation_ok(module):
    """Packed rows are (E, out, in): only an F.linear-layout stack (gate_up (E, 2I, H), down (E, H, I)) fits them."""
    pe = _zoo()
    if pe is None:
        return False
    try:
        return (
            pe.stack_rows_are_transposed(module, "gate_up_proj", "gate_up") is False
            and pe.stack_rows_are_transposed(module, "down_proj", "down") is False
        )
    except Exception:
        return False


def _placeholder_shapes(fmt, E, N, K, parts):
    if fmt == "mxfp8":
        return {"codes": ((E, N, K), torch.uint8), "scales": ((E, N, K // 32), torch.uint8)}
    return {
        "codes": ((E, N, K // 2), torch.uint8),
        "scales": ((E, N, K // 16), torch.uint8),
        "global": ((E, parts), torch.float32),
    }


def _plan(model, checkpoint_files):
    """All-or-nothing: every fused experts module maps to a complete packed per-expert set of one format."""
    if _zoo() is None or not hasattr(_zoo(), "Nvfp4ExpertParam"):
        return None
    experts = _fused_experts(model)
    if not experts:
        return None
    try:
        scanned = _scan(checkpoint_files)
    except Exception:
        return None
    if not scanned:
        return None
    plan = []
    for name, module in experts:
        prefix = _match_prefix(name, list(scanned))
        if prefix is None:
            return None
        entries = scanned[prefix]
        fmt = _format_of(entries)
        if fmt is None or not _orientation_ok(module):
            return None
        E, two_i, H = module._parameters["gate_up_proj"].shape
        if module._parameters["down_proj"].shape != (E, H, two_i // 2):
            return None
        idxs = {idx for idx, _, _ in entries}
        if idxs != set(range(E)):
            return None
        plan.append(
            {"name": name, "module": module, "prefix": prefix, "entries": entries, "fmt": fmt}
        )
    if (
        len({item["prefix"] for item in plan}) != len(plan)
        or len({item["fmt"] for item in plan}) != 1
    ):
        return None
    return plan


def _swap_in_placeholders(plan):
    for item in plan:
        module = item["module"]
        originals = {}
        for stack in ("gate_up_proj", "down_proj"):
            param = module._parameters.pop(stack)
            originals[stack] = param
            E, N, K = param.shape
            parts = 2 if stack == "gate_up_proj" else 1
            for kind, (shape, dtype) in _placeholder_shapes(item["fmt"], E, N, K, parts).items():
                module.register_parameter(
                    f"{_PLACEHOLDER}{stack}_{kind}",
                    torch.nn.Parameter(
                        torch.empty(shape, dtype = dtype, device = param.device), requires_grad = False
                    ),
                )
            module.__dict__[stack] = torch.empty(0, dtype = param.dtype, device = "meta")
        item["originals"] = originals
        item["dtype"] = originals["gate_up_proj"].dtype


def _restore(plan, model = None):
    if model is not None:
        for attr, value in model.__dict__.pop("_unsloth_packed_ignore_backup", {}).items():
            if value is _UNSET:
                model.__dict__.pop(attr, None)
            else:
                setattr(model, attr, value)
    for item in plan:
        module = item["module"]
        for stack, param in item.get("originals", {}).items():
            module.__dict__.pop(stack, None)
            for key in [k for k in module._parameters if k.startswith(f"{_PLACEHOLDER}{stack}_")]:
                del module._parameters[key]
            module._parameters[stack] = param


_UNSET = object()


def _ignore_keys(model):
    backup = model.__dict__.setdefault("_unsloth_packed_ignore_backup", {})
    for attr, pattern in (
        ("_keys_to_ignore_on_load_missing", rf"\.{_PLACEHOLDER}"),
        ("_keys_to_ignore_on_load_unexpected", r"\.experts\.\d+\."),
    ):
        current = getattr(model, attr, None)
        backup.setdefault(
            attr, model.__dict__.get(attr, _UNSET) if not isinstance(current, set) else set(current)
        )
        if isinstance(current, set):
            current.add(pattern)
        else:
            setattr(model, attr, list(current or []) + [pattern])


def _converter_order(conversions, stack):
    """(kept conversions, projection order of the fused ``stack``) or None when its converter is not a plain merge."""
    kept, order = [], None
    for conv in conversions:
        targets = list(
            getattr(conv, "_original_target_patterns", None)
            or getattr(conv, "target_patterns", None)
            or []
        )
        if any(str(t).rstrip("$").endswith("experts." + stack) for t in targets):
            ops = [type(op).__name__ for op in getattr(conv, "operations", None) or []]
            if any(
                op
                not in (
                    "MergeModulelist",
                    "Concatenate",
                    "CompressedTensorsDequantize",
                    "DecompressExperts",
                )
                for op in ops
            ):
                return None
            sources = list(
                getattr(conv, "_original_source_patterns", None)
                or getattr(conv, "source_patterns", None)
                or []
            )
            projs = []
            for src in sources:
                m = re.search(r"experts\.\*\.([A-Za-z0-9_]+)\.weight", str(src))
                if m and m.group(1) not in projs:
                    projs.append(m.group(1))
            if not projs:
                return None
            order = projs
            continue
        kept.append(conv)
    return kept, order


class _Shards:
    def __init__(self):
        self.files = {}

    def get(self, entries, idx, proj, suffix):
        from safetensors import safe_open

        path, _, _, key = entries[(idx, proj, suffix)]
        if path not in self.files:
            self.files[path] = safe_open(path, framework = "pt", device = "cpu")
        return self.files[path].get_tensor(key)


def _fill(item, shards):
    pe = _zoo()
    module, entries, fmt = item["module"], item["entries"], item["fmt"]
    keep = packed_experts_enabled()
    code_key = "weight_packed" if fmt == "nvfp4" else "weight"
    glob_key = "weight_global_scale" if fmt == "nvfp4" else "weight_scale_2"
    for stack in ("gate_up_proj", "down_proj"):
        projs = item["order"][stack]
        holders = {
            k: module._parameters.pop(f"{_PLACEHOLDER}{stack}_{k}")
            for k in ("codes", "scales", "global")
            if f"{_PLACEHOLDER}{stack}_{k}" in module._parameters
        }
        device = holders["codes"].device
        if device.type == "meta":
            raise RuntimeError(
                f"Unsloth: the packed experts of {item['name']} were offloaded to disk; packed expert stacks "
                "need a GPU or CPU device map. Load with an explicit device_map that keeps the experts off disk."
            )
        # Fresh buffers in the storage dtypes: a load-time cast of the placeholders must not round the scales.
        codes = torch.empty(tuple(holders["codes"].shape), dtype = torch.uint8, device = device)
        scales = torch.empty(tuple(holders["scales"].shape), dtype = torch.uint8, device = device)
        glob = (
            torch.empty(tuple(holders["global"].shape), dtype = torch.float32, device = device)
            if "global" in holders
            else None
        )
        del holders
        E, rows = codes.shape[0], codes.shape[1] // len(projs)

        def _exact(tensor, target, what):
            # copy_ would broadcast a malformed (rows, 1) scale over the whole slice.
            if tuple(tensor.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Unsloth: {item['name']} expert {what} has shape {tuple(tensor.shape)}, expected "
                    f"{tuple(target.shape)}; the checkpoint does not match the model."
                )
            target.copy_(tensor)

        for e in range(E):
            for p, proj in enumerate(projs):
                rs = slice(p * rows, (p + 1) * rows)
                _exact(
                    shards.get(entries, e, proj, code_key).view(torch.uint8),
                    codes[e, rs],
                    f"{e}.{proj} codes",
                )
                _exact(
                    shards.get(entries, e, proj, "weight_scale").view(torch.uint8),
                    scales[e, rs],
                    f"{e}.{proj} scales",
                )
                if glob is not None:
                    g = shards.get(entries, e, proj, glob_key)
                    if g.numel() != 1:
                        raise RuntimeError(
                            f"Unsloth: {item['name']} expert {e}.{proj} has {g.numel()} global scales, expected 1."
                        )
                    glob[e, p] = g.to(torch.float32).reshape(())
        dtype = item["dtype"]
        if fmt == "mxfp8":
            param = pe.Mxfp8ExpertParam(codes, packed_scales = scales, packed_dtype = dtype)
        else:
            param = pe.Nvfp4ExpertParam(
                codes,
                packed_scales = scales,
                packed_global = glob,
                packed_dtype = dtype,
                packed_global_mode = "multiply" if fmt == "nvfp4_modelopt" else "divide",
            )
        module.__dict__.pop(stack, None)
        if keep:
            module._parameters[stack] = param
        else:
            module._parameters[stack] = torch.nn.Parameter(
                param.dequantize(dtype), requires_grad = False
            )
    module._unsloth_packed_expert_format = fmt if keep else None


def finalize_packed_experts(model) -> int:
    plan = model.__dict__.pop(_PLAN_ATTR, None)
    if not plan:
        return 0
    if any("order" not in item for item in plan):
        _restore(plan, model)
        return 0
    model.__dict__.pop("_unsloth_packed_ignore_backup", None)
    shards = _Shards()
    with torch.no_grad():
        for item in plan:
            _fill(item, shards)
    fmt = plan[0]["fmt"]
    kind = {"nvfp4": "NVFP4", "nvfp4_modelopt": "ModelOpt NVFP4", "mxfp8": "MXFP8"}[fmt]
    if packed_experts_enabled():
        print(
            f"Unsloth: kept {2 * len(plan)} {kind} routed-expert stacks packed ({PACKED_EXPERTS_ENV}=0 dequantizes "
            "them to 16 bit)."
        )
    return len(plan)


def _offloads(device_map) -> bool:
    if not isinstance(device_map, dict):
        return False
    return any(str(v) in ("cpu", "disk", "meta") for v in device_map.values())


def _before_loading(
    quantizer,
    model,
    checkpoint_files,
    device_map = None,
):
    # dequantize=True asks transformers for 16-bit experts; its own (lossy) path is left as it is.
    if getattr(getattr(quantizer, "quantization_config", None), "dequantize", False):
        return
    # transformers' own tensor / expert parallel load shards the stacks per rank; the placeholders are whole stacks.
    if (
        getattr(getattr(model, "config", None), "distributed_config", None) is not None
        or getattr(model, "_device_mesh", None) is not None
        or (getattr(model, "tp_size", None) or 1) > 1
    ):
        return
    # cpu / disk offload dispatches before the fill could read the shards; transformers' own path is kept there.
    if _offloads(device_map):
        return
    plan = _plan(model, checkpoint_files)
    if not plan:
        return
    _swap_in_placeholders(plan)
    _ignore_keys(model)
    model.__dict__[_PLAN_ATTR] = plan
    quantizer._unsloth_packed_expert_model = model


def _filter_conversions(quantizer, conversions):
    model = getattr(quantizer, "_unsloth_packed_expert_model", None)
    plan = model.__dict__.get(_PLAN_ATTR) if model is not None else None
    if not plan:
        return conversions
    kept = list(conversions)
    order = {}
    for stack in ("gate_up_proj", "down_proj"):
        result = _converter_order(kept, stack)
        if result is None or result[1] is None:
            _restore(plan, model)
            model.__dict__.pop(_PLAN_ATTR, None)
            return conversions
        kept, order[stack] = result
    entries = plan[0]["entries"]
    known = {proj for _, proj, _ in entries}
    if not set(order["gate_up_proj"]) | set(order["down_proj"]) <= known:
        _restore(plan, model)
        model.__dict__.pop(_PLAN_ATTR, None)
        return conversions
    for item in plan:
        item["order"] = order
    return kept


_INSTALLED = False


def install_packed_expert_loading() -> bool:
    """Patch the compressed-tensors quantizer (transformers 5 weight converters only); idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from transformers.quantizers.quantizer_compressed_tensors import (
            CompressedTensorsHfQuantizer,
        )
        import transformers.core_model_loading  # noqa: F401  (5.x only)
    except Exception:
        return False
    if _zoo() is None or not hasattr(_zoo(), "Nvfp4ExpertParam"):
        return False
    before = CompressedTensorsHfQuantizer._process_model_before_weight_loading
    after = CompressedTensorsHfQuantizer._process_model_after_weight_loading
    update = getattr(CompressedTensorsHfQuantizer, "update_weight_conversions", None)
    if update is None:
        return False

    def _process_model_before_weight_loading(self, model, **kwargs):
        result = before(self, model, **kwargs)
        try:
            _before_loading(self, model, kwargs.get("checkpoint_files"), kwargs.get("device_map"))
        except Exception as exc:
            plan = model.__dict__.pop(_PLAN_ATTR, None)
            if plan:
                _restore(plan, model)
            print(
                f"Unsloth: could not plan packed expert stacks ({exc}); loading them as transformers does."
            )
        return result

    def update_weight_conversions(self, weight_conversions, *args, **kwargs):
        out = update(self, weight_conversions, *args, **kwargs)
        return _filter_conversions(self, out)

    def _process_model_after_weight_loading(self, model, **kwargs):
        finalize_packed_experts(model)
        return after(self, model, **kwargs)

    _process_model_before_weight_loading._unsloth_packed_experts = True
    CompressedTensorsHfQuantizer._process_model_before_weight_loading = (
        _process_model_before_weight_loading
    )
    CompressedTensorsHfQuantizer._process_model_after_weight_loading = (
        _process_model_after_weight_loading
    )
    CompressedTensorsHfQuantizer.update_weight_conversions = update_weight_conversions
    _INSTALLED = True
    return True


def quantize_moe_experts_if_requested(model, full_finetuning = False) -> int:
    """UNSLOTH_MOE_EXPERTS_QUANTIZE=nvfp4|mxfp4|nf4: round-to-nearest the 16-bit routed-expert stacks after load."""
    pe = _zoo()
    if pe is None or not hasattr(pe, "experts_quantize_format"):
        if os.environ.get("UNSLOTH_MOE_EXPERTS_QUANTIZE", "").strip() not in ("", "0"):
            print(
                "Unsloth: UNSLOTH_MOE_EXPERTS_QUANTIZE needs a newer unsloth_zoo; experts stay 16 bit."
            )
        return 0
    fmt = pe.experts_quantize_format()
    if fmt is None:
        return 0
    if full_finetuning or os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        print(
            "Unsloth: UNSLOTH_MOE_EXPERTS_QUANTIZE is ignored for full finetuning (the experts train)."
        )
        return 0
    count = pe.quantize_moe_expert_stacks(model, fmt)
    if count:
        print(
            f"Unsloth: quantized {count} 16-bit routed-expert stacks to {fmt.upper()} (round to nearest)."
        )
    return count
