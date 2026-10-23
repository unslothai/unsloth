# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Original-layout (ComfyUI / reference repo) single files -> diffusers state dicts for Krea 2 and
HunyuanImage 2.1, which diffusers ships without a single-file converter.

Rules only rename, split rows or reorder whole rows, never columns: ``diffusion_comfy_quant`` traces
each int8 / fp8 row (and GGUF block) back to its source and refuses anything else. Unknown keys raise.
No module-level torch import: ``diffusion.py`` imports this on every load.
"""

from __future__ import annotations

import contextvars
import os
import re
from typing import Any, Optional

_CONTAINER_PREFIXES = ("model.diffusion_model.", "model.model.", "diffusion_model.")


def _strip_prefix(key: str) -> str:
    for prefix in _CONTAINER_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _plain_1d(value: Any) -> Any:
    """Dequantise a 1-D GGUF tensor: diffusers only dequantises inside swapped Linears, so a packed
    norm / modulation vector would reach the forward as raw bytes."""
    quant_shape = getattr(value, "quant_shape", None)
    if quant_shape is not None and len(quant_shape) == 1:
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
        return dequantize_gguf_tensor(value)
    return value


def _split_rows(key: str, value: Any, sizes: tuple) -> list:
    rows = int(value.shape[0])
    if rows != sum(sizes):
        raise ValueError(
            f"{key}: {rows} rows cannot be split into {list(sizes)}; this is not the expected layout"
        )
    out, start = [], 0
    for size in sizes:
        out.append(value[start : start + size])
        start += size
    return out


_KREA2_TOP = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
}

_KREA2_BLOCK = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
}

_KREA2_BLOCK_RE = re.compile(
    r"^(blocks|txtfusion\.layerwise_blocks|txtfusion\.refiner_blocks)\.(\d+)\.(.+)$"
)
_KREA2_BLOCK_STEM = {
    "blocks": "transformer_blocks",
    "txtfusion.layerwise_blocks": "text_fusion.layerwise_blocks",
    "txtfusion.refiner_blocks": "text_fusion.refiner_blocks",
}


def krea2_checkpoint_to_diffusers(checkpoint: Any = None, **kwargs: Any) -> dict:
    """Krea 2 original layout -> ``Krea2Transformer2DModel``. Renames, plus ``blocks.N.mod.lin``
    ``[6 * hidden]`` viewed as ``scale_shift_table`` ``[6, hidden]`` (same six chunks, row-major)."""
    converted: dict = {}
    for raw_key, value in (checkpoint or {}).items():
        key = _strip_prefix(raw_key)
        if value is not None and getattr(value, "dim", lambda: 2)() == 1:
            value = _plain_1d(value)
        new = _KREA2_TOP.get(key)
        if new is not None:
            converted[new] = value
            continue
        match = _KREA2_BLOCK_RE.match(key)
        if match is None:
            raise ValueError(f"{raw_key}: not a Krea 2 transformer key this converter knows")
        group, index, rest = match.groups()
        stem = f"{_KREA2_BLOCK_STEM[group]}.{index}."
        if group == "blocks" and rest == "mod.lin":
            if value.dim() != 1 or value.shape[0] % 6:
                raise ValueError(f"{raw_key}: expected a flat [6 * hidden] modulation table")
            converted[stem + "scale_shift_table"] = value.reshape(6, -1)
            continue
        new = _KREA2_BLOCK.get(rest)
        if new is None:
            raise ValueError(f"{raw_key}: not a Krea 2 transformer key this converter knows")
        converted[stem + new] = value
    return converted


_HYIMG_TOP = {
    "img_in.proj": "x_embedder.proj",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    # Guidance-distilled (distilled_guidance_scale) and MeanFlow (timestep_r) embedders: distilled only.
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "time_r_in.in_layer": "time_guidance_embed.timestep_embedder_r.linear_1",
    "time_r_in.out_layer": "time_guidance_embed.timestep_embedder_r.linear_2",
    "txt_in.input_embedder": "context_embedder.proj_in",
    "txt_in.t_embedder.in_layer": "context_embedder.time_text_embed.timestep_embedder.linear_1",
    "txt_in.t_embedder.out_layer": "context_embedder.time_text_embed.timestep_embedder.linear_2",
    "txt_in.c_embedder.in_layer": "context_embedder.time_text_embed.text_embedder.linear_1",
    "txt_in.c_embedder.out_layer": "context_embedder.time_text_embed.text_embedder.linear_2",
    "byt5_in.layernorm": "context_embedder_2.norm",
    "byt5_in.fc1": "context_embedder_2.linear_1",
    "byt5_in.fc2": "context_embedder_2.linear_2",
    "byt5_in.fc3": "context_embedder_2.linear_3",
    "final_layer.linear": "proj_out",
}

_HYIMG_REFINER = {
    "norm1": "norm1",
    "norm2": "norm2",
    "self_attn.proj": "attn.to_out.0",
    "mlp.0": "ff.net.0.proj",
    "mlp.2": "ff.net.2",
    "adaLN_modulation.1": "norm_out.linear",
}

_HYIMG_DOUBLE = {
    "img_mod.lin": "norm1.linear",
    "txt_mod.lin": "norm1_context.linear",
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.proj": "attn.to_add_out",
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "img_mlp.0": "ff.net.0.proj",
    "img_mlp.2": "ff.net.2",
    "txt_mlp.0": "ff_context.net.0.proj",
    "txt_mlp.2": "ff_context.net.2",
}

_HYIMG_SINGLE = {
    "linear2": "proj_out",
    "modulation.lin": "norm.linear",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}

_HYIMG_QKV = {
    # Fused rows are q, then k, then v.
    "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
    "self_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
}

_HYIMG_REFINER_RE = re.compile(
    r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.+)\.(weight|bias|scale)$"
)
_HYIMG_DOUBLE_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)\.(weight|bias|scale)$")
_HYIMG_SINGLE_RE = re.compile(r"^single_blocks\.(\d+)\.(.+)\.(weight|bias|scale)$")
_HYIMG_TOP_RE = re.compile(r"^(.+)\.(weight|bias|scale)$")


# Reference-repo names (Comfy-Org's bf16 file) -> the ComfyUI names the tables above use.
_HYIMG_REFERENCE_NAMES = (
    (re.compile(r"^((?:double_blocks\.\d+\.)(?:img|txt))_attn_(qkv|proj)\."), r"\1_attn.\2."),
    (
        re.compile(r"^((?:double_blocks\.\d+\.)(?:img|txt))_attn_q_norm\.weight$"),
        r"\1_attn.norm.query_norm.scale",
    ),
    (
        re.compile(r"^((?:double_blocks\.\d+\.)(?:img|txt))_attn_k_norm\.weight$"),
        r"\1_attn.norm.key_norm.scale",
    ),
    (re.compile(r"^(single_blocks\.\d+\.)q_norm\.weight$"), r"\1norm.query_norm.scale"),
    (re.compile(r"^(single_blocks\.\d+\.)k_norm\.weight$"), r"\1norm.key_norm.scale"),
    (
        re.compile(r"^(txt_in\.individual_token_refiner\.blocks\.\d+\.)self_attn_(qkv|proj)\."),
        r"\1self_attn.\2.",
    ),
    (re.compile(r"\.(img_mlp|txt_mlp|mlp)\.fc1\."), r".\1.0."),
    (re.compile(r"\.(img_mlp|txt_mlp|mlp)\.fc2\."), r".\1.2."),
    (re.compile(r"\.(img_mod|txt_mod|modulation)\.linear\."), r".\1.lin."),
    (re.compile(r"^(time_in|time_r_in|guidance_in|txt_in\.t_embedder)\.mlp\.0\."), r"\1.in_layer."),
    (
        re.compile(r"^(time_in|time_r_in|guidance_in|txt_in\.t_embedder)\.mlp\.2\."),
        r"\1.out_layer.",
    ),
    (re.compile(r"^txt_in\.c_embedder\.linear_1\."), "txt_in.c_embedder.in_layer."),
    (re.compile(r"^txt_in\.c_embedder\.linear_2\."), "txt_in.c_embedder.out_layer."),
)


def _hyimg_comfy_name(key: str) -> str:
    for pattern, repl in _HYIMG_REFERENCE_NAMES:
        key = pattern.sub(repl, key)
    return key


def _hyimg_param(suffix: str) -> str:
    return "weight" if suffix == "scale" else suffix


def hunyuanimage_checkpoint_to_diffusers(
    checkpoint: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> dict:
    """HunyuanImage 2.1 original layout (reference or ComfyUI naming, base or distilled) ->
    ``HunyuanImageTransformer2DModel``. Fused ``qkv`` and single-block ``linear1`` (q, k, v, MLP in)
    split by rows; ``final_layer.adaLN_modulation.1`` is (shift, scale), diffusers reads (scale, shift)."""
    hidden = None
    if config is not None:
        try:
            hidden = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
        except Exception:  # noqa: BLE001 - derived from the tensors below instead
            hidden = None
    stripped = {_hyimg_comfy_name(_strip_prefix(k)): (k, v) for k, v in (checkpoint or {}).items()}
    if hidden is None:
        probe = stripped.get("img_in.proj.weight") or stripped.get("img_in.proj.bias")
        if probe is None:
            raise ValueError("not a HunyuanImage 2.1 checkpoint: no img_in.proj")
        hidden = int(probe[1].shape[0])

    converted: dict = {}
    for key, (raw_key, value) in stripped.items():
        if value is not None and getattr(value, "dim", lambda: 2)() == 1:
            value = _plain_1d(value)

        match = _HYIMG_REFINER_RE.match(key)
        if match:
            index, layer, suffix = match.groups()
            stem = f"context_embedder.token_refiner.refiner_blocks.{index}."
            if layer in _HYIMG_QKV:
                for name, part in zip(
                    _HYIMG_QKV[layer], _split_rows(raw_key, value, (hidden,) * 3)
                ):
                    converted[f"{stem}{name}.{suffix}"] = part
                continue
            new = _HYIMG_REFINER.get(layer)
            if new is None:
                raise ValueError(
                    f"{raw_key}: not a HunyuanImage 2.1 transformer key this converter knows"
                )
            converted[f"{stem}{new}.{_hyimg_param(suffix)}"] = value
            continue

        match = _HYIMG_DOUBLE_RE.match(key)
        if match:
            index, layer, suffix = match.groups()
            stem = f"transformer_blocks.{index}."
            if layer in _HYIMG_QKV:
                for name, part in zip(
                    _HYIMG_QKV[layer], _split_rows(raw_key, value, (hidden,) * 3)
                ):
                    converted[f"{stem}{name}.{suffix}"] = part
                continue
            new = _HYIMG_DOUBLE.get(layer)
            if new is None:
                raise ValueError(
                    f"{raw_key}: not a HunyuanImage 2.1 transformer key this converter knows"
                )
            converted[f"{stem}{new}.{_hyimg_param(suffix)}"] = value
            continue

        match = _HYIMG_SINGLE_RE.match(key)
        if match:
            index, layer, suffix = match.groups()
            stem = f"single_transformer_blocks.{index}."
            if layer == "linear1":
                mlp = int(value.shape[0]) - 3 * hidden
                if mlp <= 0:
                    raise ValueError(f"{raw_key}: too few rows for q, k, v and the MLP input")
                names = ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp")
                for name, part in zip(
                    names, _split_rows(raw_key, value, (hidden, hidden, hidden, mlp))
                ):
                    converted[f"{stem}{name}.{suffix}"] = part
                continue
            new = _HYIMG_SINGLE.get(layer)
            if new is None:
                raise ValueError(
                    f"{raw_key}: not a HunyuanImage 2.1 transformer key this converter knows"
                )
            converted[f"{stem}{new}.{_hyimg_param(suffix)}"] = value
            continue

        match = _HYIMG_TOP_RE.match(key)
        if match:
            layer, suffix = match.groups()
            if layer == "final_layer.adaLN_modulation.1":
                rows = int(value.shape[0])
                if rows % 2:
                    raise ValueError(f"{raw_key}: an odd row count cannot hold (shift, scale)")
                shift, scale = value[: rows // 2], value[rows // 2 :]
                converted[f"norm_out.linear.{suffix}"] = _cat_rows(scale, shift)
                continue
            new = _HYIMG_TOP.get(layer)
            if new is not None:
                converted[f"{new}.{_hyimg_param(suffix)}"] = value
                continue
        raise ValueError(f"{raw_key}: not a HunyuanImage 2.1 transformer key this converter knows")
    return converted


def _cat_rows(first: Any, second: Any) -> Any:
    """Concatenate two row blocks, keeping a ``GGUFParameter``'s quant type (whole blocks per row)."""
    import torch

    charge_host_ram(first.numel() * first.element_size() + second.numel() * second.element_size())
    out = torch.cat([first, second], dim = 0)
    quant_type = getattr(first, "quant_type", None)
    if quant_type is not None and getattr(out, "quant_type", None) is None:
        from diffusers.quantizers.gguf.utils import GGUFParameter
        out = GGUFParameter(out, quant_type = quant_type)
    return out


CONVERTERS: dict = {
    "Krea2Transformer2DModel": krea2_checkpoint_to_diffusers,
    "HunyuanImageTransformer2DModel": hunyuanimage_checkpoint_to_diffusers,
}


_ST_ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}  # fmt: skip
_ST_FLOATS = ("F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "F8_E4M3FNUZ", "F8_E5M2FNUZ")
HOST_RAM_CHECK_MIN_MIB = 1024
_MIB = 1024 * 1024


class HostRamShortfallError(RuntimeError):
    """A single-file load whose converted weights would not fit in the host RAM left."""


def _torch_dtype_tag(dtype: Any) -> str:
    return {
        "torch.float32": "F32",
        "torch.float16": "F16",
        "torch.bfloat16": "BF16",
    }.get(str(dtype), "BF16")


def converted_host_bytes(
    path: str,
    dtype: Any,
    *,
    keep_fp32: Any = (),
    dequantized: Any = None,
    keep_key: Any = None,
) -> int:
    """Anonymous host RAM bytes loading ``path`` at ``dtype`` writes, from the header alone.

    ``keep_key(key)``: the loader's own key filter (a file bundling other components), so tensors it never
    reads are not priced.

    Tensors already at the target dtype stay memory-mapped (droppable page cache) and cost 0.
    ``dequantized(key)``: None = not quantized, 0 / False = kept as codes, else the dense elements each
    stored element expands to (True = 1; 2 for nvfp4, which packs two columns per byte) when widened
    to ``dtype``. Unreadable header returns 0 (no claim)."""
    from .diffusion_comfy_quant import _read_header

    try:
        header, _base = _read_header(str(path))
    except Exception:  # noqa: BLE001 -- the loader reports a bad file itself
        return 0
    header.pop("__metadata__", None)
    target = _torch_dtype_tag(dtype)
    keep = tuple(keep_fp32 or ())
    total = 0
    for key, meta in header.items():
        if keep_key is not None and not keep_key(key):
            continue
        try:
            stored = str(meta.get("dtype"))
            numel = 1
            for dim in meta.get("shape") or ():
                numel *= int(dim)
        except Exception:  # noqa: BLE001
            continue
        if dequantized is None:
            verdict = None
        elif getattr(dequantized, "wants_meta", False):
            verdict = dequantized(key, meta)
        else:
            verdict = dequantized(key)
        if verdict is not None:
            total += numel * int(verdict) * _ST_ITEMSIZE[target]
            continue
        if stored not in _ST_FLOATS:
            continue
        want = "F32" if keep and any(m in key.split(".") for m in keep) else target
        if stored != want:
            total += numel * _ST_ITEMSIZE[want]
    return total


class HostRamBudget:
    """The host RAM check of one single-file load: an up-front header estimate, then a check before each
    large allocation against what is free right now.

    ``converted_host_bytes`` prices a load from the header before any read, but cannot see the per-layer
    choices made later (runtime filters, nibble swaps, row reorders). Every allocating helper therefore
    calls ``charge_host_ram(nbytes)`` (a no-op outside an active budget) before it allocates, and the load
    is refused when that allocation plus the small-host reserve exceeds the host's LIVE available memory,
    instead of the OOM that ends the whole server. Live, not a running total: buffers the load has
    already freed are back in the next reading, and copies made where nothing checks still show up in
    it. ``UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0`` disables both checks."""

    def __init__(
        self,
        path: str,
        why: str,
        logger: Any = None,
    ) -> None:
        from .diffusion_small_host import (
            host_ram_check_disabled,
            host_ram_mib,
            host_reserve_mib,
        )

        self.path, self.why, self.logger = path, why, logger
        self._token = None
        self.available_mib: Optional[int] = None
        self.reserve_mib = 0
        self.armed = False
        self.requested = (
            0  # only to arm a load that started under the floor; the check reads live memory
        )
        if host_ram_check_disabled():
            return
        try:
            total, available = host_ram_mib()
        except Exception:  # noqa: BLE001 -- unreadable: no claim, as the small-host route does
            return
        if available is None:
            return
        self.available_mib = int(available)
        self.reserve_mib = host_reserve_mib(total)
        # Armed by the file size here, and by the widened estimate in ``check``: a 900 MiB nvfp4 file widens
        # to 3.6 GiB of bf16. Only a load small on disk AND in its estimate is never checked.
        try:
            size_mib = os.path.getsize(str(path)) // _MIB
        except OSError:
            size_mib = HOST_RAM_CHECK_MIN_MIB
        self.armed = size_mib >= HOST_RAM_CHECK_MIN_MIB

    def _refuse_if_over(self, need_bytes: int, available_mib: int, at_least: bool) -> None:
        from .diffusion_small_host import HOST_RAM_CHECK_ENV

        need_mib = -(-int(need_bytes) // _MIB)
        if need_mib + self.reserve_mib <= available_mib:
            return
        reserve = self.reserve_mib
        name = os.path.basename(str(self.path))
        message = (
            f"Not enough system RAM to load {name}: {self.why} needs "
            f"{'at least' if at_least else 'about'} {(need_mib + reserve) / 1024:.1f} GB of free host "
            f"memory ({need_mib / 1024:.1f} GB of weights + {reserve / 1024:.1f} GB reserve) and "
            f"{available_mib / 1024:.1f} GB is free. Close other applications, use a GGUF or a "
            f"smaller quant of this model, or set {HOST_RAM_CHECK_ENV}=0 to try anyway."
        )
        if self.logger is not None:
            self.logger.error("diffusion.single_file: refusing load: %s", message)
        raise HostRamShortfallError(message)

    def check(self, need_bytes: int) -> None:
        """Up-front refusal from a header estimate, against free memory when the load started."""
        if self.available_mib is None or -(-int(need_bytes) // _MIB) < HOST_RAM_CHECK_MIN_MIB:
            return
        self.armed = True
        self._refuse_if_over(need_bytes, self.available_mib, at_least = False)

    def charge(self, nbytes: int) -> None:
        """Refuse before allocating ``nbytes`` when they and the reserve exceed live available memory."""
        if self.available_mib is None or nbytes <= 0:
            return
        if not self.armed:
            # A load small on disk and in its estimate can still write a lot (a runtime filter rejecting
            # layers the estimate assumed kept): arm once the checked allocations reach the floor.
            self.requested += int(nbytes)
            if self.requested < HOST_RAM_CHECK_MIN_MIB * _MIB:
                return
            self.armed = True
        from .diffusion_small_host import host_ram_mib

        try:
            available = host_ram_mib()[1]
        except Exception:  # noqa: BLE001 -- an unreadable reading makes no claim
            return
        if available is not None:
            self._refuse_if_over(nbytes, int(available), at_least = True)

    def activate(self) -> "HostRamBudget":
        """Make this the budget the allocating helpers charge, until the enclosing
        ``host_ram_budget_scope`` returns."""
        if self.available_mib is not None:  # check disabled / RAM unreadable: nothing is charged
            _ACTIVE_BUDGET.set(self)
        return self

    def __enter__(self) -> "HostRamBudget":
        self._token = _ACTIVE_BUDGET.set(self if self.available_mib is not None else None)
        return self

    def __exit__(self, *exc: Any) -> None:
        _ACTIVE_BUDGET.reset(self._token)


_ACTIVE_BUDGET: "contextvars.ContextVar[Optional[HostRamBudget]]" = contextvars.ContextVar(
    "unsloth_single_file_host_ram_budget", default = None
)


def host_ram_budget_scope(fn: Any) -> Any:
    """Run ``fn`` with no budget inherited and none left behind: a budget it ``activate()``s lives
    exactly as long as the call."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _ACTIVE_BUDGET.set(None)
        try:
            return fn(*args, **kwargs)
        finally:
            _ACTIVE_BUDGET.reset(token)

    return wrapper


def charge_host_ram(nbytes: int) -> None:
    """Check ``nbytes`` about to be allocated against the active ``HostRamBudget``, if any."""
    budget = _ACTIVE_BUDGET.get()
    if budget is not None and nbytes > 0:
        budget.charge(nbytes)


def charge_cast(value: Any, dtype: Any) -> None:
    """Charge the copy ``value.to(dtype)`` makes; nothing when it is already ``dtype``."""
    if getattr(value, "dtype", dtype) != dtype:
        import torch
        charge_host_ram(value.numel() * torch.empty((), dtype = dtype).element_size())


def load_original_layout_transformer(
    transformer_cls: Any,
    path: str,
    sf_kwargs: dict,
    logger: Any = None,
    *,
    fp8_storage: bool = False,
) -> Any:
    """Load a plain original-layout safetensors file for a class without ``from_single_file``
    (Krea 2 on diffusers 0.40). ``sf_kwargs`` = the ``from_single_file`` kwargs. Cast like
    ``from_pretrained`` (``_keep_in_fp32_modules`` stay fp32); strict on keys and shapes. ``fp8_storage``
    keeps fp8 Linears as fp8 with unit scales."""
    import torch
    from diffusers.loaders import single_file_model as sfm

    kwargs = dict(sf_kwargs)
    dtype = kwargs.pop("torch_dtype", None) or torch.bfloat16
    config_repo = kwargs.pop("config", None)
    if not isinstance(config_repo, str):
        raise ValueError("an original-layout single file needs the base repo's transformer config")
    config = transformer_cls.load_config(
        config_repo,
        subfolder = kwargs.pop("subfolder", None),
        token = kwargs.pop("token", None),
        cache_dir = kwargs.pop("cache_dir", None),
        local_files_only = bool(kwargs.pop("local_files_only", False)),
    )
    expected, optional = transformer_cls._get_signature_keys(transformer_cls)
    config.update({k: v for k, v in kwargs.items() if k in expected or k in optional})
    entry = sfm.SINGLE_FILE_LOADABLE_CLASSES.get(transformer_cls.__name__) or {}
    mapping_fn = CONVERTERS.get(transformer_cls.__name__) or entry.get("checkpoint_mapping_fn")
    if mapping_fn is None:
        raise ValueError(f"{transformer_cls.__name__} has no single-file converter")
    keep = getattr(transformer_cls, "_keep_in_fp32_modules", None) or []
    if isinstance(keep, str):
        keep = [keep]
    # Before any read: fp8 widened to bf16 (13B Krea 2: ~26 GB) used to OOM the whole app.
    budget = HostRamBudget(
        path, f"widening its weights to {str(dtype).replace('torch.', '')}", logger
    )
    budget.check(
        converted_host_bytes(
            path,
            dtype,
            keep_fp32 = keep,
            dequantized = fp8_stored_estimate(keep) if fp8_storage else None,
        )
        + (_fp8_stored_scale_bytes(path, keep) if fp8_storage else 0)
    )
    with budget:
        return _load_original_layout(
            transformer_cls, path, config, mapping_fn, keep, dtype, logger, fp8_storage
        )


def fp8_stored_estimate(keep: Any = ()) -> Any:
    """``dequantized`` for fp8 storage: 2-D fp8 ``.weight`` outside ``keep`` stays mmapped (0); later widens are
    charged live."""
    keep = tuple(keep or ())

    def verdict(key: str, meta: Optional[dict] = None) -> Any:
        meta = meta or {}
        if (
            str(meta.get("dtype")) in ("F8_E4M3", "F8_E5M2")
            and len(meta.get("shape") or ()) == 2
            and key.endswith(".weight")
            and not any(m in key.split(".") for m in keep)
        ):
            return 0
        return None

    verdict.wants_meta = True
    return verdict


def _fp8_stored_scale_bytes(path: str, keep: Any = ()) -> int:
    from .diffusion_comfy_quant import _read_header

    try:
        header, _base = _read_header(str(path))
    except Exception:  # noqa: BLE001
        return 0
    header.pop("__metadata__", None)
    verdict = fp8_stored_estimate(keep)
    return sum(
        4 * int((meta.get("shape") or (0,))[0])
        for key, meta in header.items()
        if verdict(key, meta) == 0
    )


def _load_original_layout(
    transformer_cls,
    path,
    config,
    mapping_fn,
    keep,
    dtype,
    logger,
    fp8_storage = False,
) -> Any:
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    with init_empty_weights():
        model = transformer_cls.from_config(config)
    wanted = model.state_dict()
    state = load_file(str(path))
    converted = state if set(state) == set(wanted) else mapping_fn(checkpoint = state, config = config)
    del state
    missing = sorted(set(wanted) - set(converted))
    unexpected = sorted(set(converted) - set(wanted))
    if missing or unexpected:
        import os
        name = os.path.basename(str(path))
        raise ValueError(
            f"{name} does not match {transformer_cls.__name__}: {len(missing)} missing "
            f"(e.g. {missing[:1]}), {len(unexpected)} unused (e.g. {unexpected[:1]})"
        )
    fp8_dtypes = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}
    modules = dict(model.named_modules()) if fp8_storage else {}
    stored: list = []
    for key, value in list(converted.items()):
        want = torch.float32 if any(m in key.split(".") for m in keep) else dtype
        if not value.is_floating_point():
            continue
        if tuple(value.shape) != tuple(wanted[key].shape):
            raise ValueError(
                f"{key}: {tuple(value.shape)} in the file, {tuple(wanted[key].shape)} in the model"
            )
        fqn = key[: -len(".weight")]
        if (
            value.dtype in fp8_dtypes
            and want == dtype
            and key.endswith(".weight")
            and isinstance(modules.get(fqn), torch.nn.Linear)
        ):
            stored.append(fqn)
            continue
        if value.dtype != want:
            charge_cast(value, want)
            converted[key] = value.to(want)
    if stored:
        from .diffusion_native_quant import NATIVE_FP8, native_linear_class
        cls = native_linear_class()
        for fqn in stored:
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            codes = converted.pop(fqn + ".weight")
            charge_host_ram(codes.shape[0] * 4)  # the unit scale; the codes stay as read
            layer = cls(
                getattr(parent, leaf).to(dtype),
                NATIVE_FP8,
                codes = codes,
                scale = torch.ones(codes.shape[0], 1, dtype = torch.float32),
            )
            setattr(parent, leaf, layer)
            converted[fqn + ".weight_q"] = layer.weight_q
            converted[fqn + ".weight_scale"] = layer.weight_scale
    model.load_state_dict(converted, strict = True, assign = True)
    if stored:
        from .diffusion_comfy_quant import FP8_STORED_ATTR
        setattr(model, FP8_STORED_ATTR, len(stored))
    model.eval()
    if logger is not None:
        logger.info(
            "diffusion.single_file: %s loaded from an original-layout file (%d fp8 layers stored as fp8, "
            "widened per layer)",
            transformer_cls.__name__,
            len(stored),
        )
    return model
