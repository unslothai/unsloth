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
from typing import Any, Callable, Optional

from loggers import get_logger

from .diffusion_krea2 import load_krea2_text_encoder, load_krea2_tokenizer
from .diffusion_transformer_quant import mark_source_precision

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
        if "inputs_embeds" in kwargs and "inputs_embeds" not in params and "input_embeds" in params:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        if "cache_position" in params and "cache_position" not in kwargs:
            embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
            if embeds is not None:
                kwargs["cache_position"] = torch.arange(embeds.shape[1], device = embeds.device)
        return original(*args, **kwargs)

    pipe_mod.create_causal_mask = create_causal_mask_compat
    _CAUSAL_MASK_PATCHED = True


_QKV_SPLIT = ("to_q", "to_k", "to_v")


def _transformer_shard_paths(
    repo_id: str,
    subfolder: str,
    token: Optional[str],
    check_cancelled: Optional[Callable[[], None]] = None,
) -> list[str]:
    """The local safetensors shard paths for ``repo_id/subfolder``.

    Prefers the sharded index; falls back to the single-file name when the subfolder
    ships one file. Resolves through a local dir when ``repo_id`` is a path, else the
    Hub cache.
    """
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    from huggingface_hub import hf_hub_download

    local_root = Path(repo_id).expanduser()
    if local_root.is_dir():
        sub = local_root / subfolder
        index = sub / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding = "utf-8"))["weight_map"]
            return [str(sub / name) for name in sorted(set(weight_map.values()))]
        single = sub / "diffusion_pytorch_model.safetensors"
        if single.is_file():
            return [str(single)]
        raise FileNotFoundError(f"no transformer safetensors under {sub}")

    index_name = f"{subfolder}/diffusion_pytorch_model.safetensors.index.json"
    try:
        index_path = hf_hub_download(repo_id, index_name, token = token)
        weight_map = json.loads(Path(index_path).read_text(encoding = "utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    except Exception:  # noqa: BLE001 -- single-file subfolder has no index
        shards = ["diffusion_pytorch_model.safetensors"]
    paths = []
    for name in shards:
        check_cancelled()
        paths.append(hf_hub_download(repo_id, f"{subfolder}/{name}", token = token))
    return paths


def _read_transformer_config(repo_id: str, subfolder: str, token: Optional[str]) -> dict[str, Any]:
    """``subfolder/config.json`` as a dict, from a local path or the Hub cache."""
    local = Path(repo_id).expanduser() / subfolder / "config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding = "utf-8"))
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, f"{subfolder}/config.json", token = token)
    return json.loads(Path(path).read_text(encoding = "utf-8"))


def _convert_fp8_state_dict(
    raw: dict,
    hidden_size: int,
    dtype,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> dict:
    """Dequantize + rename the vendor fp8 shards into the diffusers split layout.

    A ``*.weight`` with a companion ``*.weight_scale`` is float8 per-channel (real weight =
    ``fp8.float() * weight_scale[:, None]``). Fused ``attention.qkv`` -> ``to_q``/``to_k``/``to_v``
    (Q/K/V order), ``attention.o`` -> ``to_out.0``. Dense tensors pass through cast to ``dtype``.
    """
    check_cancelled = check_cancelled or (lambda: None)
    import torch

    def dequantize(name: str):
        weight = raw[name].to(torch.float32)
        scale = raw[name + "_scale"].to(torch.float32)
        return (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)

    converted: dict = {}
    for key, value in raw.items():
        check_cancelled()
        if key.endswith("_scale"):
            continue
        if key + "_scale" not in raw:
            converted[key] = value.to(dtype)
            continue
        if key.endswith("attention.qkv.weight"):
            fused = dequantize(key)
            if fused.shape[0] != 3 * hidden_size:
                # equal thirds only holds for full multi-head attention; a GQA export must fail loudly
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


def _text_encoder_shard_paths(
    repo_id: str,
    token: Optional[str],
    check_cancelled: Optional[Callable[[], None]] = None,
) -> list[str]:
    """The local safetensors shard paths for ``repo_id/text_encoder`` (index or single file)."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    from huggingface_hub import hf_hub_download

    local_root = Path(repo_id).expanduser()
    if local_root.is_dir():
        sub = local_root / "text_encoder"
        index = sub / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding = "utf-8"))["weight_map"]
            return [str(sub / name) for name in sorted(set(weight_map.values()))]
        single = sub / "model.safetensors"
        if single.is_file():
            return [str(single)]
        raise FileNotFoundError(f"no text_encoder safetensors under {sub}")

    try:
        index_path = hf_hub_download(
            repo_id, "text_encoder/model.safetensors.index.json", token = token
        )
        weight_map = json.loads(Path(index_path).read_text(encoding = "utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    except Exception:  # noqa: BLE001 -- single-file text encoder has no index
        shards = ["model.safetensors"]
    paths = []
    for name in shards:
        check_cancelled()
        paths.append(hf_hub_download(repo_id, f"text_encoder/{name}", token = token))
    return paths


def _text_encoder_is_fp8(repo_id: str, token: Optional[str]) -> bool:
    """True when the text_encoder ships the vendor fp8 layout (a ``*.weight_scale`` key)."""
    from huggingface_hub import hf_hub_download

    local_root = Path(repo_id).expanduser()
    if local_root.is_dir():
        index = local_root / "text_encoder" / "model.safetensors.index.json"
        if index.is_file():
            return any(
                k.endswith("_scale")
                for k in json.loads(index.read_text(encoding = "utf-8"))["weight_map"]
            )
    else:
        try:
            index_path = hf_hub_download(
                repo_id, "text_encoder/model.safetensors.index.json", token = token
            )
            weight_map = json.loads(Path(index_path).read_text(encoding = "utf-8"))["weight_map"]
            return any(k.endswith("_scale") for k in weight_map)
        except Exception:  # noqa: BLE001 -- single-file (nf4) text encoder, not fp8
            return False
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
    check_cancelled: Optional[Callable[[], None]] = None,
):
    """The Qwen3-VL text encoder for ``repo_id``.

    The ``-fp8`` repo stores it in the same float8-plus-per-channel-scale layout as its DiTs, but
    its keys already match transformers Qwen3-VL (no fused qkv rename needed), so only the float8
    dequant is required. The ``-nf4`` and dense repos fall through to the shared krea shim (which
    also applies the rope_parameters remap).
    """
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    token = hf_token or None
    is_fp8 = _text_encoder_is_fp8(repo_id, token)
    check_cancelled()
    if not is_fp8:
        return load_krea2_text_encoder(
            repo_id, dtype, hf_token = token, check_cancelled = check_cancelled
        )

    import safetensors
    import torch
    from transformers import AutoConfig, Qwen3VLModel

    from .diffusion_krea2 import remap_rope_parameters

    config_kwargs: dict[str, Any] = {"subfolder": "text_encoder"}
    if token:
        config_kwargs["token"] = token
    config = AutoConfig.from_pretrained(repo_id, **config_kwargs)
    check_cancelled()
    remap_rope_parameters(getattr(config, "text_config", config))

    raw: dict = {}
    for path in _text_encoder_shard_paths(repo_id, token, check_cancelled = check_cancelled):
        check_cancelled()
        with safetensors.safe_open(path, "pt") as handle:
            for key in handle.keys():
                check_cancelled()
                raw[key] = handle.get_tensor(key)

    state_dict: dict = {}
    for key, value in raw.items():
        check_cancelled()
        if key.endswith("_scale"):
            continue
        if key + "_scale" in raw:
            weight = value.to(torch.float32)
            scale = raw[key + "_scale"].to(torch.float32)
            state_dict[key] = (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)
        else:
            state_dict[key] = value.to(dtype)

    check_cancelled()
    # Build at the target dtype: the fp32 default can OOM a 64 GB host.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen3VLModel(config)
        check_cancelled()
        model.to(dtype)
    finally:
        torch.set_default_dtype(default_dtype)
    check_cancelled()
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
    check_cancelled: Optional[Callable[[], None]] = None,
):
    """An ``Ideogram4Transformer2DModel`` for ``repo_id/subfolder`` (still on CPU).

    If the shards carry the vendor fp8 layout, dequantizes + renames into the diffusers split
    layout and loads into a config-constructed model. Already-split ``-nf4`` repos (with a
    ``quantization_config``) delegate to stock ``from_pretrained`` so bnb re-applies the 4-bit weights.
    """
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    import diffusers
    import safetensors
    import torch

    token = hf_token or None
    config = _read_transformer_config(repo_id, subfolder, token)
    check_cancelled()
    shard_paths = _transformer_shard_paths(
        repo_id, subfolder, token, check_cancelled = check_cancelled
    )

    # Check every shard header so a dense-first multi-shard fp8 export still dequantizes.
    is_fp8 = False
    for path in shard_paths:
        check_cancelled()
        with safetensors.safe_open(path, "pt") as handle:
            if any(key.endswith("_scale") for key in handle.keys()):
                is_fp8 = True
                break
    check_cancelled()
    if not is_fp8:
        model_kwargs: dict[str, Any] = {"subfolder": subfolder, "torch_dtype": dtype}
        if token:
            model_kwargs["token"] = token
        return diffusers.Ideogram4Transformer2DModel.from_pretrained(repo_id, **model_kwargs)

    raw: dict = {}
    for path in shard_paths:
        check_cancelled()
        with safetensors.safe_open(path, "pt") as handle:
            for key in handle.keys():
                check_cancelled()
                raw[key] = handle.get_tensor(key)

    check_cancelled()
    config.pop("quantization_config", None)
    hidden_size = int(config["attention_head_dim"]) * int(config["num_attention_heads"])
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = diffusers.Ideogram4Transformer2DModel.from_config(config)
    finally:
        torch.set_default_dtype(default_dtype)
    check_cancelled()
    state_dict = _convert_fp8_state_dict(raw, hidden_size, dtype, check_cancelled = check_cancelled)
    check_cancelled()
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    # inv_freq is built in __init__; any other missing or leftover key must fail loudly.
    real_missing = [k for k in missing if not k.endswith("rotary_emb.inv_freq")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"ideogram4 fp8 remap left keys unmatched for {repo_id}/{subfolder}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    check_cancelled()
    model.to(dtype)
    return mark_source_precision(model, "fp8")


def _hidden_states_on_mask_device(original: Callable[..., Any]) -> Callable[..., Any]:
    """Return the tapped states on ``attention_mask``'s device: under group offload ``text_encoder.device`` reads CPU
    (so the mask lands there) while the states come back on the GPU, and ``encode_prompt``'s multiply raised."""

    def get_text_encoder_hidden_states(text_encoder, token_ids, attention_mask, pos_2d):
        states = original(text_encoder, token_ids, attention_mask, pos_2d)
        device = getattr(attention_mask, "device", None)
        if device is None:
            return states
        return [s.to(device) if getattr(s, "device", device) != device else s for s in states]

    get_text_encoder_hidden_states._unsloth_mask_device = True  # type: ignore[attr-defined]
    return get_text_encoder_hidden_states


def install_text_encoder_device_guard(pipe: Any) -> bool:
    """Shadow the pipeline's static ``_get_text_encoder_hidden_states`` on this instance. Idempotent."""
    original = getattr(pipe, "_get_text_encoder_hidden_states", None)
    if not callable(original):
        return False
    if getattr(original, "_unsloth_mask_device", False):
        return True
    try:
        pipe._get_text_encoder_hidden_states = _hidden_states_on_mask_device(original)
    except Exception:  # noqa: BLE001 - a frozen / slotted pipeline keeps the stock path
        return False
    return True


def load_ideogram4_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
):
    """Assemble Ideogram4Pipeline from ``repo_id`` per-component (see module doc)."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    import diffusers

    _patch_create_causal_mask()
    check_cancelled()

    token = hf_token or None
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if token:
        model_kwargs["token"] = token

    text_encoder = load_ideogram4_text_encoder(
        repo_id, dtype, hf_token = token, check_cancelled = check_cancelled
    )
    check_cancelled()
    tokenizer = load_krea2_tokenizer(repo_id, hf_token = token, check_cancelled = check_cancelled)
    check_cancelled()
    transformer = load_ideogram4_transformer(
        repo_id, "transformer", dtype, hf_token = token, check_cancelled = check_cancelled
    )
    check_cancelled()
    unconditional_transformer = load_ideogram4_transformer(
        repo_id, "unconditional_transformer", dtype, hf_token = token, check_cancelled = check_cancelled
    )
    check_cancelled()
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(repo_id, subfolder = "vae", **model_kwargs)
    check_cancelled()
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id, subfolder = "scheduler", token = token
    )
    check_cancelled()
    logger.info("diffusion.ideogram4: assembled pipeline from %s per-component", repo_id)
    pipe = diffusers.Ideogram4Pipeline(
        scheduler = scheduler,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        transformer = transformer,
        unconditional_transformer = unconditional_transformer,
    )
    install_text_encoder_device_guard(pipe)
    return pipe
