# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 pipeline assembly for a transformers-4.x runtime.

The ideogram-ai repos ship the same transformers-5.x style Qwen text stack as the
krea repos, which breaks ``Ideogram4Pipeline.from_pretrained`` twice on the 4.x
line:

- ``text_encoder/config.json`` keeps rope settings under ``rope_parameters`` (the
  5.x name); 4.x's Qwen3-VL rotary embedding reads ``config.rope_scaling`` and
  crashes on None. Fixed by ``diffusion_krea2.load_krea2_text_encoder`` (the shared
  remap shim).
- ``model_index.json`` pins the SLOW ``Qwen2Tokenizer`` while the repo ships only
  ``tokenizer.json`` (no vocab.json/merges.txt), so the slow class cannot even
  construct -- and diffusers' passed-component type gate rejects the fast class
  against the slow pin, so the fast tokenizer cannot be handed to from_pretrained
  either.

So the pipeline is assembled per-component (the constructor registers modules
without from_pretrained's type gate), mirroring ``diffusion_krea2``.

The two DiTs need one more fix on the ``-fp8`` base repo. Its transformer shards
store the vendor's OWN float8 layout, which diffusers 0.39.0 cannot read:

- attention is stored FUSED as ``attention.qkv.weight`` (shape ``[3*hidden, hidden]``,
  the Q/K/V rows stacked in that order) plus ``attention.o.weight``, whereas the
  diffusers ``Ideogram4Transformer2DModel`` has SPLIT ``to_q`` / ``to_k`` / ``to_v``
  and ``to_out.0`` projections. from_pretrained can map neither name, so it leaves
  every attention projection randomly initialized (garbage images) AND on the meta
  device (a later ``pipe.to(device)`` then dies with "Cannot copy out of meta tensor").
- each quantized ``*.weight`` is float8_e4m3 with a companion per-output-channel
  ``*.weight_scale`` (float32); the real weight is ``fp8.float() * weight_scale[:, None]``.
  diffusers 0.39.0 has no float8 dequant path here, so it drops the scales entirely
  and loads the raw fp8 values (range +-448) as if they were the weights.

diffusers ``main`` still ships neither the fused->split rename nor the float8 dequant
(the attention module is split-only and there is no ideogram single-file converter),
so ``load_ideogram4_transformer`` does the conversion here: it reads the shards,
dequantizes every scaled weight, splits the fused ``qkv`` into ``to_q``/``to_k``/``to_v``
and renames ``o`` -> ``to_out.0``, then loads the result into a config-constructed model
(verified against the byte-identical ``-nf4`` repo, whose transformer is ALREADY exported
in the diffusers split layout: the dequantized fp8 projections match its bnb-4bit weights
to cosine ~0.997, i.e. only quant noise apart). The already-split ``-nf4`` repos carry a
``quantization_config`` and load through the stock diffusers path, so the conversion is
gated on the fp8 marker (a ``*.weight_scale`` key) and is a no-op for them.

The VAE loads through ``AutoencoderKLFlux2`` and the scheduler is stock
``FlowMatchEulerDiscreteScheduler``.

One last 4.x incompatibility is in the diffusers pipeline itself, not the repo:
``Ideogram4Pipeline._get_text_encoder_hidden_states`` calls transformers'
``create_causal_mask(inputs_embeds = ...)`` with no ``cache_position``, but on the
4.x line (and even transformers 5.0) the parameter is spelled ``input_embeds`` and
``cache_position`` is required. ``_patch_create_causal_mask`` installs a signature-aware
wrapper over the name the pipeline module imported, which renames the kwarg and derives
``cache_position`` when the installed function needs one. It is self-disabling: on a
transformers whose ``create_causal_mask`` already accepts the pipeline's exact kwargs the
wrapper forwards them unchanged.
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
        # The pipeline always calls this by keyword. Rename inputs_embeds -> input_embeds
        # when the installed function uses the (older/5.x) spelling.
        if "inputs_embeds" in kwargs and "inputs_embeds" not in params and "input_embeds" in params:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        # Supply a cache_position when the function requires one and the caller omitted it:
        # past_key_values is None here, so positions run 0..seq_len-1 over the text region.
        if "cache_position" in params and "cache_position" not in kwargs:
            embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
            if embeds is not None:
                kwargs["cache_position"] = torch.arange(embeds.shape[1], device = embeds.device)
        return original(*args, **kwargs)

    pipe_mod.create_causal_mask = create_causal_mask_compat
    _CAUSAL_MASK_PATCHED = True


# The fp8 attention is stored as a single fused ``qkv`` matrix with the Q, K and V
# rows stacked in that order; each block is ``hidden_size`` rows tall. hidden_size =
# attention_head_dim * num_attention_heads, read from the transformer config so a
# future config change cannot silently mis-split the matrix.
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

    A ``*.weight`` with a companion ``*.weight_scale`` is float8 stored per-output-channel:
    the real weight is ``fp8.float() * weight_scale[:, None]``. The fused ``attention.qkv``
    is split into ``to_q``/``to_k``/``to_v`` (``hidden_size`` rows each, Q/K/V order) and
    ``attention.o`` is renamed ``to_out.0``. Everything else (norms, biases, embeddings) is
    stored dense and passes through cast to ``dtype``.
    """
    import torch

    def dequantize(name: str):
        weight = raw[name].to(torch.float32)
        scale = raw[name + "_scale"].to(torch.float32)
        # Per-output-channel scale, broadcast over the remaining dims. Every scaled
        # tensor in the shipped repos is 2D; the rank-aware view keeps a future
        # non-2D quantized tensor correct instead of silently mis-broadcasting.
        return (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)

    converted: dict = {}
    for key, value in raw.items():
        if key.endswith("_scale"):
            continue
        if key + "_scale" not in raw:
            # Dense (non-fp8) tensor: norms, biases, embeddings -- load as-is.
            converted[key] = value.to(dtype)
            continue
        if key.endswith("attention.qkv.weight"):
            fused = dequantize(key)  # [3 * hidden_size, hidden_size]
            if fused.shape[0] != 3 * hidden_size:
                # Equal-thirds is only correct for full multi-head attention; a GQA
                # export (fewer K/V rows) must fail loudly, not split into garbage.
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

    The ``-fp8`` repo stores this encoder in the SAME float8-plus-per-channel-scale
    layout as its DiTs, and its keys already match the transformers Qwen3-VL module
    (only the DiTs used the fused ``qkv``; Qwen3-VL's own attention is already split
    and its visual tower's fused ``qkv`` matches transformers), so it needs no rename
    -- only the float8 dequant diffusers/transformers skip. So the fp8 encoder is
    dequantized and loaded into a config-constructed model; the ``-nf4`` (bnb-4bit)
    and any dense repo fall through to the shared krea shim (which also applies the
    rope_parameters remap).
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
            # Rank-aware broadcast, matching _convert_fp8_state_dict.
            state_dict[key] = (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)
        else:
            state_dict[key] = value.to(dtype)

    # Construct normally (so __init__ computes the non-persistent rotary inv_freq
    # buffers the checkpoint omits) then copy the dequantized weights in with
    # assign=False. Host RAM is ample, so the transient dense init is fine.
    model = Qwen3VLModel(config).to(dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"ideogram4 fp8 text_encoder remap left keys unmatched for {repo_id}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


def load_ideogram4_transformer(
    repo_id: str,
    subfolder: str,
    dtype,
    hf_token: Optional[str] = None,
):
    """An ``Ideogram4Transformer2DModel`` for ``repo_id/subfolder`` (still on CPU).

    Reads the transformer config, and if the shards carry the vendor fp8 layout
    (a ``*.weight_scale`` key), dequantizes + renames them into the diffusers split
    layout and loads that into a config-constructed model. When the shards are already
    in the diffusers layout (the ``-nf4`` repos, which carry a ``quantization_config``),
    delegates to the stock ``from_pretrained`` so bnb re-applies the 4-bit weights.
    """
    import diffusers
    import safetensors

    token = hf_token or None
    config = _read_transformer_config(repo_id, subfolder, token)
    shard_paths = _transformer_shard_paths(repo_id, subfolder, token)

    # Detect the fp8 layout from the shard HEADERS (safe_open.keys() reads metadata only,
    # not the multi-GB tensor bodies). All shards are checked so a multi-shard export
    # whose first shard happens to hold only dense tensors still routes to the dequant
    # path. Only the fp8 path then materializes the tensors; the -nf4 path goes straight
    # to from_pretrained without a wasteful full-shard read.
    is_fp8 = False
    for path in shard_paths:
        with safetensors.safe_open(path, "pt") as handle:
            if any(key.endswith("_scale") for key in handle.keys()):
                is_fp8 = True
                break
    if not is_fp8:
        # Already the diffusers split layout (the quantized -nf4 exports). Let
        # from_pretrained re-apply the embedded quantization_config unchanged.
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
    model = diffusers.Ideogram4Transformer2DModel.from_config(config)
    state_dict = _convert_fp8_state_dict(raw, hidden_size, dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    # rotary_emb.inv_freq is a non-persistent buffer built in __init__, so it is
    # (correctly) absent from the checkpoint and the only expected "missing" key; a
    # real gap (an unmapped weight) or any leftover checkpoint key must fail loudly
    # rather than ship a partly random model.
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

    # The pipeline's text-encoder call uses a transformers-5.x create_causal_mask
    # signature; adapt it to the installed one before any generate runs.
    _patch_create_causal_mask()

    token = hf_token or None
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if token:
        model_kwargs["token"] = token

    text_encoder = load_ideogram4_text_encoder(repo_id, dtype, hf_token = token)
    tokenizer = load_krea2_tokenizer(repo_id, hf_token = token)
    transformer = load_ideogram4_transformer(repo_id, "transformer", dtype, hf_token = token)
    # The second DiT drives the unconditional branch of Ideogram's dual-branch CFG;
    # it is the same class and size as the conditional one and always required.
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
