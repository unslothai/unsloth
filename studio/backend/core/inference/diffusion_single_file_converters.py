# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Original-layout (ComfyUI / reference repo) single files -> diffusers state dicts, for transformer
classes diffusers ships without a single-file converter: Krea 2 and HunyuanImage 2.1.

Every rule is a rename, a split along dim 0 (rows), or a reorder of whole rows. Nothing mixes or
slices columns, which is what lets the ComfyUI quantized loader (``diffusion_comfy_quant``) keep
int8 / fp8 codes and their per-row scales exact through the conversion: it tags each source row,
runs the converter, and refuses any output whose rows it cannot trace back to whole source rows.
A GGUF tensor survives the same way, since GGML packs blocks along the last (input) dimension.

Both converters are strict about what they understand: a key they have no rule for raises
instead of passing through, so a different architecture or a new upstream layout fails here
with the key named, not later as a model that silently left weights at their init values.

No torch import at module level: the registry in ``diffusion.py`` imports this on every load.
"""

from __future__ import annotations

import re
from typing import Any

# Container prefixes the original-layout files are stored under. ComfyUI's HunyuanImage 2.1 files
# nest the model twice (``model.model.``), sd.cpp / ComfyUI checkpoints use ``model.diffusion_model.``.
_CONTAINER_PREFIXES = ("model.diffusion_model.", "model.model.", "diffusion_model.")


def _strip_prefix(key: str) -> str:
    for prefix in _CONTAINER_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _plain_1d(value: Any) -> Any:
    """A one-dimensional GGUF tensor as real values; anything else unchanged.

    diffusers only dequantises inside the Linears its GGUF quantizer swaps in, so a norm or
    modulation vector stored in a packed type would otherwise reach the forward as raw bytes.
    These are a few KB per model, so there is nothing to save by keeping them packed."""
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


# --------------------------------------------------------------------------------------------- Krea 2

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

# Inside a DiT block or a text-fusion block (both use the same sub-layout).
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
    """Krea 2 original layout (the ComfyUI ``diffusion_models/krea2_*`` files) -> diffusers
    ``Krea2Transformer2DModel``.

    Pure renames except one: a DiT block's AdaLN-single table ``blocks.N.mod.lin`` is stored flat,
    ``[6 * hidden]``, where diffusers keeps ``scale_shift_table`` as ``[6, hidden]``. Both sides add
    it to the projected timestep split into six equal chunks along the last dim, so the flat vector
    is that same six-chunk table in row-major order and a view is exact (it is never quantized: a
    1-D tensor holds no Linear)."""
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


# ---------------------------------------------------------------------------------- HunyuanImage 2.1

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
    # (original fused layer, diffusers q/k/v layers); the fused rows are q, then k, then v.
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


# The reference repo's own names (Comfy-Org's bf16 file keeps them) -> the ComfyUI names the tables above use.
# Each rewrites only a name, so the two layouts share every row rule below.
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
    # The original QK norms are RMSNorms holding ``scale``; diffusers' hold ``weight``.
    return "weight" if suffix == "scale" else suffix


def hunyuanimage_checkpoint_to_diffusers(
    checkpoint: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> dict:
    """HunyuanImage 2.1 original layout -> diffusers ``HunyuanImageTransformer2DModel``. Both original
    namings load: the reference repo's (``img_attn_qkv``, ``mlp.fc1``, ``mod.linear``; Comfy-Org's bf16
    file) and ComfyUI's (``img_attn.qkv``, ``mlp.0``, ``mod.lin``, under ``model.model.``; its fp8 and
    distilled files), the base model or the guidance-distilled MeanFlow one.

    Beyond renames, three row-only rules:

    * every fused ``qkv`` is split into q, k, v by rows (three equal parts);
    * a single-stream block's ``linear1`` is split by rows into q, k, v (``hidden`` each) and the
      MLP input projection (the remainder);
    * ``final_layer.adaLN_modulation.1`` holds (shift, scale) where diffusers' continuous AdaLN norm
      reads (scale, shift), so its two row halves swap.
    """
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

    out = torch.cat([first, second], dim = 0)
    quant_type = getattr(first, "quant_type", None)
    if quant_type is not None and getattr(out, "quant_type", None) is None:
        from diffusers.quantizers.gguf.utils import GGUFParameter
        out = GGUFParameter(out, quant_type = quant_type)
    return out


# Transformer class name -> converter, for diffusion.py's registry of classes diffusers lacks.
CONVERTERS: dict = {
    "Krea2Transformer2DModel": krea2_checkpoint_to_diffusers,
    "HunyuanImageTransformer2DModel": hunyuanimage_checkpoint_to_diffusers,
}


def load_original_layout_transformer(
    transformer_cls: Any,
    path: str,
    sf_kwargs: dict,
    logger: Any = None,
) -> Any:
    """``transformer_cls`` from a plain (unquantized) original-layout safetensors file, for a class
    diffusers gives no ``from_single_file`` at all (Krea 2: no ``FromOriginalModelMixin``).

    ``sf_kwargs`` are the kwargs the caller would have passed to ``from_single_file`` (``config`` = the
    base repo, ``subfolder``, ``torch_dtype``, ``token``, ``cache_dir``, ``local_files_only``, plus any
    config overrides). The weights are cast the way ``from_pretrained`` casts the base repo's own
    shards: ``torch_dtype``, except the class's ``_keep_in_fp32_modules``, which stay float32. So a
    bf16 ComfyUI file of the base model loads into the same tensors as the base repo does. Strict:
    a missing or unused key raises with the key named."""
    import torch
    from accelerate import init_empty_weights
    from diffusers.loaders import single_file_model as sfm
    from safetensors.torch import load_file

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
    mapping_fn = entry.get("checkpoint_mapping_fn") or CONVERTERS.get(transformer_cls.__name__)
    if mapping_fn is None:
        raise ValueError(f"{transformer_cls.__name__} has no single-file converter")
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
    keep = getattr(transformer_cls, "_keep_in_fp32_modules", None) or []
    if isinstance(keep, str):
        keep = [keep]
    for key, value in converted.items():
        want = torch.float32 if any(m in key.split(".") for m in keep) else dtype
        if not value.is_floating_point():
            continue
        if tuple(value.shape) != tuple(wanted[key].shape):
            raise ValueError(
                f"{key}: {tuple(value.shape)} in the file, {tuple(wanted[key].shape)} in the model"
            )
        if value.dtype != want:
            converted[key] = value.to(want)
    model.load_state_dict(converted, strict = True, assign = True)
    model.eval()
    if logger is not None:
        logger.info(
            "diffusion.single_file: %s loaded from an original-layout file",
            transformer_cls.__name__,
        )
    return model
