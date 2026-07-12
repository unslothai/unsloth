# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 pipeline assembly for a transformers-4.x runtime.

The ideogram-ai repos ship the transformers-5.x Qwen text stack (like the krea repos), breaking
``Ideogram4Pipeline.from_pretrained`` on 4.x twice: (1) ``text_encoder/config.json`` keeps rope
under ``rope_parameters`` (5.x), which 4.x Qwen3-VL crashes on -- fixed by the shared krea remap
shim; (2) ``model_index.json`` pins the SLOW ``Qwen2Tokenizer`` but the repo ships only
``tokenizer.json``, so neither the slow class (can't construct) nor the fast class (type-gate
rejected) loads. So the pipeline is assembled per-component (no from_pretrained type gate).

The two DiTs need one more fix on the ``-fp8`` base repo, whose shards store the vendor's float8
layout diffusers can't read: attention is FUSED as ``attention.qkv.weight`` [3*hidden, hidden]
(Q/K/V stacked) + ``attention.o.weight``, vs diffusers' SPLIT ``to_q``/``to_k``/``to_v`` +
``to_out.0`` (from_pretrained maps neither -> random weights on meta); and each ``*.weight`` is
float8_e4m3 with a per-channel ``*.weight_scale`` (real weight = ``fp8.float() * scale[:, None]``)
that diffusers drops. So ``load_ideogram4_transformer`` reads the shards, dequantizes, splits qkv
and renames o -> to_out.0, then loads into a config-constructed model (verified vs the split
``-nf4`` repo: cosine ~0.997, quant noise apart). The already-split ``-nf4`` repos carry a
``quantization_config`` and use the stock path, so the conversion is gated on the ``*.weight_scale``
marker. VAE via ``AutoencoderKLFlux2``, scheduler ``FlowMatchEulerDiscreteScheduler``.

One last incompat is in the diffusers pipeline: it calls ``create_causal_mask(inputs_embeds=...)``
with no ``cache_position``, but 4.x/5.0 spell it ``input_embeds`` and require ``cache_position``.
``_patch_create_causal_mask`` installs a signature-aware wrapper that renames the kwarg and derives
``cache_position``; self-disabling where the installed function already accepts the exact kwargs.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

from .diffusion_krea2 import load_krea2_text_encoder, load_krea2_tokenizer

logger = get_logger(__name__)

_CAUSAL_MASK_PATCHED = False


def _patch_create_causal_mask() -> None:
    """Adapt the diffusers Ideogram4 pipeline's ``create_causal_mask`` call to the
    installed transformers signature (see module doc). Idempotent and self-disabling.
    """
    global _CAUSAL_MASK_PATCHED
    if _CAUSAL_MASK_PATCHED:
        return
    import torch
    from diffusers.pipelines.ideogram4 import pipeline_ideogram4 as pipe_mod

    original = pipe_mod.create_causal_mask
    params = inspect.signature(original).parameters

    def create_causal_mask_compat(*args, **kwargs):
        # The pipeline calls this by keyword. Rename inputs_embeds -> input_embeds for the 5.x spelling.
        if "inputs_embeds" in kwargs and "inputs_embeds" not in params and "input_embeds" in params:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        # Supply cache_position when required and omitted: past_key_values is None, so positions
        # run 0..seq_len-1.
        if "cache_position" in params and "cache_position" not in kwargs:
            embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
            if embeds is not None:
                kwargs["cache_position"] = torch.arange(embeds.shape[1], device = embeds.device)
        return original(*args, **kwargs)

    pipe_mod.create_causal_mask = create_causal_mask_compat
    _CAUSAL_MASK_PATCHED = True


# The fp8 attention is a fused ``qkv`` matrix (Q/K/V stacked, each ``hidden_size`` rows).
# hidden_size = attention_head_dim * num_attention_heads, read from config so a future change
# can't mis-split it.
_QKV_SPLIT = ("to_q", "to_k", "to_v")


def _transformer_shard_paths(repo_id: str, subfolder: str, token: Optional[str]) -> list[str]:
    """The local safetensors shard paths for ``repo_id/subfolder``.

    Prefers the sharded index; falls back to the single-file name when the subfolder
    ships one file. Resolves through a local dir when ``repo_id`` is a path, else the
    Hub cache.
    """
    from huggingface_hub import hf_hub_download

    local_root = Path(repo_id).expanduser()
    if local_root.is_dir():
        sub = local_root / subfolder
        index = sub / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            return [str(sub / name) for name in sorted(set(weight_map.values()))]
        single = sub / "diffusion_pytorch_model.safetensors"
        if single.is_file():
            return [str(single)]
        raise FileNotFoundError(f"no transformer safetensors under {sub}")

    index_name = f"{subfolder}/diffusion_pytorch_model.safetensors.index.json"
    try:
        index_path = hf_hub_download(repo_id, index_name, token = token)
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    except Exception:  # noqa: BLE001 -- single-file subfolder has no index
        shards = ["diffusion_pytorch_model.safetensors"]
    return [hf_hub_download(repo_id, f"{subfolder}/{name}", token = token) for name in shards]


def _read_transformer_config(repo_id: str, subfolder: str, token: Optional[str]) -> dict[str, Any]:
    """``subfolder/config.json`` as a dict, from a local path or the Hub cache."""
    local = Path(repo_id).expanduser() / subfolder / "config.json"
    if local.is_file():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, f"{subfolder}/config.json", token = token)
    return json.loads(Path(path).read_text())


def _convert_fp8_state_dict(raw: dict, hidden_size: int, dtype) -> dict:
    """Dequantize + rename the vendor fp8 shards into the diffusers split layout.

    A ``*.weight`` with a companion ``*.weight_scale`` is float8 per-channel (real weight =
    ``fp8.float() * weight_scale[:, None]``). Fused ``attention.qkv`` -> ``to_q``/``to_k``/``to_v``
    (Q/K/V order), ``attention.o`` -> ``to_out.0``. Dense tensors pass through cast to ``dtype``.
    """
    import torch

    def dequantize(name: str):
        weight = raw[name].to(torch.float32)
        scale = raw[name + "_scale"].to(torch.float32)
        # Per-channel scale, rank-aware broadcast (correct for a future non-2D quantized tensor).
        return (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)

    converted: dict = {}
    for key, value in raw.items():
        if key.endswith("_scale"):
            continue
        if key + "_scale" not in raw:
            # Dense tensor (norms/biases/embeddings): load as-is.
            converted[key] = value.to(dtype)
            continue
        if key.endswith("attention.qkv.weight"):
            fused = dequantize(key)  # [3 * hidden_size, hidden_size]
            if fused.shape[0] != 3 * hidden_size:
                # Equal-thirds only holds for full multi-head attention; a GQA export must fail loudly.
                raise RuntimeError(
                    f"fused qkv at {key} has {fused.shape[0]} rows, expected "
                    f"{3 * hidden_size}; cannot split into equal Q/K/V blocks"
                )
            base = key[: -len("qkv.weight")]
            for index, proj in enumerate(_QKV_SPLIT):
                block = fused[index * hidden_size : (index + 1) * hidden_size]
                converted[f"{base}{proj}.weight"] = block.clone()
        elif key.endswith("attention.o.weight"):
            converted[key[: -len("o.weight")] + "to_out.0.weight"] = dequantize(key)
        else:
            converted[key] = dequantize(key)
    return converted


def _text_encoder_shard_paths(repo_id: str, token: Optional[str]) -> list[str]:
    """The local safetensors shard paths for ``repo_id/text_encoder`` (index or single file)."""
    from huggingface_hub import hf_hub_download

    local_root = Path(repo_id).expanduser()
    if local_root.is_dir():
        sub = local_root / "text_encoder"
        index = sub / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            return [str(sub / name) for name in sorted(set(weight_map.values()))]
        single = sub / "model.safetensors"
        if single.is_file():
            return [str(single)]
        raise FileNotFoundError(f"no text_encoder safetensors under {sub}")

    try:
        index_path = hf_hub_download(
            repo_id, "text_encoder/model.safetensors.index.json", token = token
        )
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    except Exception:  # noqa: BLE001 -- single-file text encoder has no index
        shards = ["model.safetensors"]
    return [hf_hub_download(repo_id, f"text_encoder/{name}", token = token) for name in shards]


def _text_encoder_is_fp8(repo_id: str, token: Optional[str]) -> bool:
    """True when the text_encoder ships the vendor fp8 layout (a ``*.weight_scale`` key)."""
    from huggingface_hub import hf_hub_download

    local_root = Path(repo_id).expanduser()
    if local_root.is_dir():
        index = local_root / "text_encoder" / "model.safetensors.index.json"
        if index.is_file():
            return any(k.endswith("_scale") for k in json.loads(index.read_text())["weight_map"])
    else:
        try:
            index_path = hf_hub_download(
                repo_id, "text_encoder/model.safetensors.index.json", token = token
            )
            weight_map = json.loads(Path(index_path).read_text())["weight_map"]
            return any(k.endswith("_scale") for k in weight_map)
        except Exception:  # noqa: BLE001 -- single-file (nf4) text encoder, not fp8
            return False
    # Single-file local text encoder: peek the header keys.
    import safetensors

    single = local_root / "text_encoder" / "model.safetensors"
    if single.is_file():
        with safetensors.safe_open(str(single), "pt") as handle:
            return any(k.endswith("_scale") for k in handle.keys())
    return False


def load_ideogram4_text_encoder(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
):
    """The Qwen3-VL text encoder for ``repo_id``.

    The ``-fp8`` repo stores it in the same float8-plus-per-channel-scale layout as its DiTs, but
    its keys already match transformers Qwen3-VL (no fused qkv rename needed), so only the float8
    dequant is required. The ``-nf4`` and dense repos fall through to the shared krea shim (which
    also applies the rope_parameters remap).
    """
    token = hf_token or None
    if not _text_encoder_is_fp8(repo_id, token):
        return load_krea2_text_encoder(repo_id, dtype, hf_token = token)

    import safetensors
    import torch
    from transformers import AutoConfig, Qwen3VLModel

    from .diffusion_krea2 import remap_rope_parameters

    config_kwargs: dict[str, Any] = {"subfolder": "text_encoder"}
    if token:
        config_kwargs["token"] = token
    config = AutoConfig.from_pretrained(repo_id, **config_kwargs)
    remap_rope_parameters(getattr(config, "text_config", config))

    raw: dict = {}
    for path in _text_encoder_shard_paths(repo_id, token):
        with safetensors.safe_open(path, "pt") as handle:
            for key in handle.keys():
                raw[key] = handle.get_tensor(key)

    state_dict: dict = {}
    for key, value in raw.items():
        if key.endswith("_scale"):
            continue
        if key + "_scale" in raw:
            weight = value.to(torch.float32)
            scale = raw[key + "_scale"].to(torch.float32)
            # Rank-aware broadcast (matches _convert_fp8_state_dict).
            state_dict[key] = (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)
        else:
            state_dict[key] = value.to(dtype)

    # Construct normally (so __init__ computes the non-persistent rotary inv_freq the checkpoint
    # omits) then copy the dequantized weights in. Build at the target dtype: this ~8B Qwen3-VL
    # scaffold is ~2x at the fp32 default (~33 vs ~16 GB) and loads FIRST, so fp32 can OOM a 64 GB
    # host. inv_freq is computed in explicit fp32, so a bf16 default leaves it correct.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen3VLModel(config).to(dtype)
    finally:
        torch.set_default_dtype(default_dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"ideogram4 fp8 text_encoder remap left keys unmatched for {repo_id}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


def ideogram4_repo_is_fp8(repo_id: str, hf_token: Optional[str] = None) -> bool:
    """True when ``repo_id``'s transformer ships the vendor fp8 layout (a ``*.weight_scale`` key).

    Those weights dequantize to a WIDER resident dtype, so on-disk bytes undershoot the bf16
    footprint; memory planning uses this to reserve the real size for a LOCAL fp8 mirror (whose
    path can't string-match ``base_repo``; ``-nf4`` mirrors have no marker and stay compressed).
    Reads shard HEADERS only. Any failure resolves to False (caller uses the file-size estimate).
    """
    try:
        shard_paths = _transformer_shard_paths(repo_id, "transformer", hf_token or None)
        import safetensors
    except Exception:  # noqa: BLE001 -- treat an unreadable / absent transformer as not fp8
        return False
    for path in shard_paths:
        with safetensors.safe_open(path, "pt") as handle:
            if any(key.endswith("_scale") for key in handle.keys()):
                return True
    return False


def load_ideogram4_transformer(
    repo_id: str,
    subfolder: str,
    dtype,
    hf_token: Optional[str] = None,
):
    """An ``Ideogram4Transformer2DModel`` for ``repo_id/subfolder`` (still on CPU).

    If the shards carry the vendor fp8 layout, dequantizes + renames into the diffusers split
    layout and loads into a config-constructed model. Already-split ``-nf4`` repos (with a
    ``quantization_config``) delegate to stock ``from_pretrained`` so bnb re-applies the 4-bit weights.
    """
    import diffusers
    import safetensors
    import torch

    token = hf_token or None
    config = _read_transformer_config(repo_id, subfolder, token)
    shard_paths = _transformer_shard_paths(repo_id, subfolder, token)

    # Detect fp8 from shard HEADERS (keys() reads metadata only). All shards checked so a
    # dense-first multi-shard export still routes to the dequant path. Only the fp8 path
    # materializes tensors; -nf4 goes straight to from_pretrained.
    is_fp8 = False
    for path in shard_paths:
        with safetensors.safe_open(path, "pt") as handle:
            if any(key.endswith("_scale") for key in handle.keys()):
                is_fp8 = True
                break
    if not is_fp8:
        # Already the diffusers split layout (-nf4): let from_pretrained re-apply its
        # quantization_config.
        model_kwargs: dict[str, Any] = {"subfolder": subfolder, "torch_dtype": dtype}
        if token:
            model_kwargs["token"] = token
        return diffusers.Ideogram4Transformer2DModel.from_pretrained(repo_id, **model_kwargs)

    raw: dict = {}
    for path in shard_paths:
        with safetensors.safe_open(path, "pt") as handle:
            for key in handle.keys():
                raw[key] = handle.get_tensor(key)

    config.pop("quantization_config", None)
    hidden_size = int(config["attention_head_dim"]) * int(config["num_attention_heads"])
    # from_config materializes the full ~9B module before the weights copy in. At fp32 that's ~2x
    # the bf16 model (~37 vs ~18 GB), and the second DiT builds while the first + encoder are
    # resident, so fp32 can OOM smaller hosts. Build at the target dtype; only rotary_emb.inv_freq
    # is absent from the checkpoint (computed in explicit fp32), so a bf16 default leaves it correct.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = diffusers.Ideogram4Transformer2DModel.from_config(config)
    finally:
        torch.set_default_dtype(default_dtype)
    state_dict = _convert_fp8_state_dict(raw, hidden_size, dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    # rotary_emb.inv_freq is the only expected "missing" key (built in __init__); a real gap or
    # leftover key must fail loudly rather than ship a partly random model.
    real_missing = [k for k in missing if not k.endswith("rotary_emb.inv_freq")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"ideogram4 fp8 remap left keys unmatched for {repo_id}/{subfolder}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    model.to(dtype)
    return model


def load_ideogram4_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
):
    """Assemble Ideogram4Pipeline from ``repo_id`` per-component (see module doc)."""
    import diffusers

    # The pipeline's text-encoder call uses a 5.x create_causal_mask signature; adapt it first.
    _patch_create_causal_mask()

    token = hf_token or None
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if token:
        model_kwargs["token"] = token

    text_encoder = load_ideogram4_text_encoder(repo_id, dtype, hf_token = token)
    tokenizer = load_krea2_tokenizer(repo_id, hf_token = token)
    transformer = load_ideogram4_transformer(repo_id, "transformer", dtype, hf_token = token)
    # The second DiT drives the unconditional branch of Ideogram's dual-branch CFG (same class/size,
    # always required).
    unconditional_transformer = load_ideogram4_transformer(
        repo_id, "unconditional_transformer", dtype, hf_token = token
    )
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(repo_id, subfolder = "vae", **model_kwargs)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id, subfolder = "scheduler", token = token
    )
    logger.info("diffusion.ideogram4: assembled pipeline from %s per-component", repo_id)
    return diffusers.Ideogram4Pipeline(
        scheduler = scheduler,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        transformer = transformer,
        unconditional_transformer = unconditional_transformer,
    )
