# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 assembled per component for transformers 4.x; -fp8 DiTs need fp8 dequant and qkv
split, gated on weight_scale (-nf4 keeps stock path); create_causal_mask kwarg patch self-
disables where supported."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
from collections.abc import Mapping
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
        # The pipeline calls this by keyword. Rename inputs_embeds -> input_embeds for the 5.x spelling.
        if "inputs_embeds" in kwargs and "inputs_embeds" not in params and "input_embeds" in params:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        # Supply cache_position when required and omitted: past_key_values is None, so positions run 0..seq_len-1.
        if "cache_position" in params and "cache_position" not in kwargs:
            embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
            if embeds is not None:
                kwargs["cache_position"] = torch.arange(embeds.shape[1], device = embeds.device)
        return original(*args, **kwargs)

    pipe_mod.create_causal_mask = create_causal_mask_compat
    _CAUSAL_MASK_PATCHED = True


# The fp8 attention is a fused ``qkv`` matrix (Q/K/V stacked, each ``hidden_size`` rows), read from config so a future
# change cannot mis-split it.
_QKV_SPLIT = ("to_q", "to_k", "to_v")

# "1" stores the vendor fp8 DiT codes even when the widened load fits host RAM, "0" always widens at load.
FP8_STORAGE_ENV = "UNSLOTH_DIFFUSION_IDEOGRAM4_FP8_STORAGE"
# Two ~9.3B DiTs per pipeline. The stored route keeps every fp8 Linear as its 1-byte codes (plus a float32 row
# scale): half the bf16 bytes whatever the compute dtype. Norms, biases and embeddings still widen, about 2% of the
# bf16 DiT, scaled by the compute width.
_STORED_CODES_FRACTION = 0.50
_STORED_WIDENED_FRACTION = 0.02
_MIB = 1024 * 1024


class _LazyTensors(Mapping):
    """Read-on-access view over open safetensors handles: a key's tensor exists only while its caller holds it,
    so converting a checkpoint never keeps the whole stored copy beside the converted one."""

    def __init__(self, handles: list):
        self._where: dict = {}
        for handle in handles:
            for key in handle.keys():
                self._where[key] = handle

    def __getitem__(self, key: str):
        return self._where[key].get_tensor(key)

    def __contains__(self, key: object) -> bool:
        return key in self._where

    def __iter__(self):
        return iter(sorted(self._where))

    def __len__(self) -> int:
        return len(self._where)


def _open_shards(
    stack: contextlib.ExitStack, paths: list[str], check_cancelled: Callable[[], None]
):
    import safetensors

    handles = []
    for path in paths:
        check_cancelled()
        handles.append(stack.enter_context(safetensors.safe_open(path, "pt")))
    return _LazyTensors(handles)


def _empty_weights():
    """Parameters on ``meta`` while the module is built (buffers such as rotary ``inv_freq`` stay real), so a
    checkpoint assigned in afterwards is the only host copy; a plain build when accelerate is missing."""
    try:
        from accelerate import init_empty_weights
    except Exception:  # noqa: BLE001 -- no accelerate: build materialised, the assign below still replaces it
        return contextlib.nullcontext()
    return init_empty_weights(include_buffers = False)


def _assign_state_dict(model: Any, state_dict: dict, skip_missing: Callable[[str], bool]) -> tuple:
    """``load_state_dict(assign=True)``: the converted tensors become the parameters (no second copy). Returns
    (missing, unexpected), missing filtered by ``skip_missing``; a parameter still on ``meta`` counts as missing."""
    missing, unexpected = model.load_state_dict(state_dict, strict = False, assign = True)
    real_missing = [k for k in missing if not skip_missing(k)]
    for name, param in model.named_parameters():
        if param.device.type == "meta" and name not in real_missing:
            real_missing.append(name)
    return real_missing, list(unexpected)


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
    raw: Mapping,
    hidden_size: int,
    dtype,
    check_cancelled: Optional[Callable[[], None]] = None,
    stored: Optional[dict] = None,
    storable: Optional[Callable[[str], bool]] = None,
) -> dict:
    """Dequantises fp8 weights by their per-channel scale and splits fused qkv into to_q/to_k/to_v.

    ``raw`` may be lazy (``_LazyTensors``): each tensor is read once, converted, and dropped. With ``stored`` (a dict)
    and ``storable`` (target weight key -> bool), a storable 2-D fp8 weight is not widened: its codes and float32
    row scale land in ``stored[module fqn]`` instead, which widen per call to exactly the values dequantised here."""
    check_cancelled = check_cancelled or (lambda: None)
    import torch

    fp8_dtypes = tuple(
        getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)
    )

    def emit(target: str, codes, scale) -> None:
        # Row-wise op, so a row slice of the codes widens to the same values as that slice of the widened whole.
        if (
            stored is not None
            and storable is not None
            and codes.ndim == 2
            and codes.dtype in fp8_dtypes
            and target.endswith(".weight")
            and storable(target)
        ):
            stored[target[: -len(".weight")]] = (codes, scale.reshape(-1))
            return
        weight = codes.to(torch.float32)
        # Per-channel scale, rank-aware broadcast (correct for a future non-2D quantized tensor).
        converted[target] = (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)

    converted: dict = {}
    for key in list(raw.keys()):
        check_cancelled()
        if key.endswith("_scale"):
            continue
        if key + "_scale" not in raw:
            # Dense tensor (norms/biases/embeddings): load as-is.
            converted[key] = raw[key].to(dtype)
            continue
        codes = raw[key]
        scale = raw[key + "_scale"].to(torch.float32)
        if key.endswith("attention.qkv.weight"):
            if codes.shape[0] != 3 * hidden_size:
                # equal thirds only holds for full multi-head attention; a GQA export must fail loudly
                raise RuntimeError(
                    f"fused qkv at {key} has {codes.shape[0]} rows, expected "
                    f"{3 * hidden_size}; cannot split into equal Q/K/V blocks"
                )
            base = key[: -len("qkv.weight")]
            for index, proj in enumerate(_QKV_SPLIT):
                rows = slice(index * hidden_size, (index + 1) * hidden_size)
                emit(f"{base}{proj}.weight", codes[rows], scale.reshape(-1)[rows])
        elif key.endswith("attention.o.weight"):
            emit(key[: -len("o.weight")] + "to_out.0.weight", codes, scale)
        else:
            emit(key, codes, scale)
        del codes, scale
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
    """The -fp8 encoder needs only the float8 dequant: its keys already match Qwen3-VL, so no qkv rename."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    token = hf_token or None
    is_fp8 = _text_encoder_is_fp8(repo_id, token)
    check_cancelled()
    if not is_fp8:
        return load_krea2_text_encoder(
            repo_id, dtype, hf_token = token, check_cancelled = check_cancelled
        )

    import torch
    from transformers import AutoConfig, Qwen3VLModel

    from .diffusion_krea2 import remap_rope_parameters

    config_kwargs: dict[str, Any] = {"subfolder": "text_encoder"}
    if token:
        config_kwargs["token"] = token
    config = AutoConfig.from_pretrained(repo_id, **config_kwargs)
    check_cancelled()
    remap_rope_parameters(getattr(config, "text_config", config))

    shard_paths = _text_encoder_shard_paths(repo_id, token, check_cancelled = check_cancelled)
    with contextlib.ExitStack() as stack:
        # Read lazily: holding every fp8 shard (~9 GB) beside the widened copy (~16 GB) was the old load's peak.
        raw = _open_shards(stack, shard_paths, check_cancelled)
        state_dict: dict = {}
        for key in list(raw.keys()):
            check_cancelled()
            if key.endswith("_scale"):
                continue
            if key + "_scale" in raw:
                weight = raw[key].to(torch.float32)
                scale = raw[key + "_scale"].to(torch.float32)
                # Rank-aware broadcast (matches _convert_fp8_state_dict).
                state_dict[key] = (weight * scale.view(-1, *([1] * (weight.ndim - 1)))).to(dtype)
                del weight, scale
            else:
                state_dict[key] = raw[key].to(dtype)

    check_cancelled()
    # build at the target dtype with meta parameters: the widened tensors are then assigned in, not copied over
    # a second ~16 GB scaffold beside them (this ~8B encoder loads FIRST, before both DiTs)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with _empty_weights():
            model = Qwen3VLModel(config)
        check_cancelled()
    finally:
        torch.set_default_dtype(default_dtype)
    check_cancelled()
    real_missing, unexpected = _assign_state_dict(
        model, state_dict, lambda k: k.endswith("inv_freq")
    )
    del state_dict
    model.to(dtype)
    if real_missing or unexpected:
        raise RuntimeError(
            f"ideogram4 fp8 text_encoder remap left keys unmatched for {repo_id}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


def ideogram4_repo_is_fp8(repo_id: str, hf_token: Optional[str] = None) -> bool:
    """Checks shard headers for a *.weight_scale key (wider in memory than on disk); any failure returns False."""
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


def _fp8_storage_env() -> Optional[bool]:
    value = (os.environ.get(FP8_STORAGE_ENV) or "").strip().lower()
    if value in ("1", "on", "true", "yes", "force"):
        return True
    if value in ("0", "off", "false", "no"):
        return False
    return None


def ideogram4_host_plan(
    *,
    dit_bf16_mib: int,
    text_encoder_mib: int,
    vae_mib: int,
    host_total_mib: Optional[int],
    host_available_mib: Optional[int],
    lora_active: bool = False,
    compute_width: float = 1.0,
) -> tuple[bool, Optional[str], str]:
    """(fp8_storage, refusal, reason) for assembling an fp8 Ideogram 4 pipeline on this host.

    ``dit_bf16_mib`` is the DiTs' bf16 size; ``compute_width`` is the compute dtype's bytes over bf16's (2.0 for
    float32), applied only to what widens. ``text_encoder_mib`` / ``vae_mib`` are already at the compute dtype.

    The vendor DiTs are float8 on disk, so unlike a bf16 checkpoint (which stays memory-mapped, reclaimable page
    cache) the dense load widens both ~9.3B DiTs and the encoder into anonymous host memory: ~53 GB, which the
    memcg OOM killer ends on a 53 GB host. Widen when that fits with the small-host reserve free (every host where it
    loads today keeps its load); else keep the DiT Linears as their fp8 codes (half the DiT bytes, the same values
    per call); refuse up front when even that does not fit, or when LoRA adapters need the widened Linears."""
    from .diffusion_small_host import (
        HOST_RAM_CHECK_ENV,
        _refusal,
        host_ram_check_disabled,
        host_reserve_mib,
    )

    width = float(compute_width)
    dense = int(dit_bf16_mib * width) + int(text_encoder_mib) + int(vae_mib)
    stored_dit = dit_bf16_mib * (_STORED_CODES_FRACTION + _STORED_WIDENED_FRACTION * width)
    stored = int(stored_dit) + int(text_encoder_mib) + int(vae_mib)
    reserve = host_reserve_mib(host_total_mib)
    forced = _fp8_storage_env()
    if forced is not None and not (forced and lora_active):
        return forced, None, f"{FP8_STORAGE_ENV}={'1' if forced else '0'}"
    if host_available_mib is None:
        return False, None, "host RAM unreadable: widening at load"
    available = int(host_available_mib)
    if dense + reserve <= available:
        return (
            False,
            None,
            f"widened load (~{dense} MiB + {reserve} MiB reserve) fits {available} MiB available host RAM",
        )
    check = not host_ram_check_disabled()
    if lora_active:
        refuse = (
            _refusal(dense, reserve, available, "the widened load LoRA adapters need")
            if check
            else None
        )
        return (
            False,
            refuse,
            (
                f"LoRA adapters need the widened DiTs (~{dense} MiB + {reserve} MiB reserve, {available} MiB available)"
            ),
        )
    refuse = None
    if check and stored + reserve > available:
        refuse = _refusal(stored, reserve, available, "even the fp8-stored load")
    return (
        True,
        refuse,
        (
            f"widened load needs ~{dense} MiB + {reserve} MiB reserve of host RAM, {available} MiB available: "
            f"keeping the DiTs' fp8 Linears stored (~{stored} MiB; {HOST_RAM_CHECK_ENV}=0 skips the refusal)"
        ),
    )


def load_ideogram4_transformer(
    repo_id: str,
    subfolder: str,
    dtype,
    hf_token: Optional[str] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
    fp8_storage: bool = False,
):
    """Vendor fp8 shards are dequantised and renamed; bnb -nf4 repos load via stock from_pretrained.

    ``fp8_storage``: keep every fp8 Linear as its stored codes and row scale (``NativeWeightOnlyLinear``), widened
    per call to the same values the dequantised load holds, so the DiT costs about half its bf16 size in host RAM
    (see ``ideogram4_host_plan``)."""
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
        # Already the diffusers split layout (-nf4): let from_pretrained re-apply its quantization_config.
        model_kwargs: dict[str, Any] = {"subfolder": subfolder, "torch_dtype": dtype}
        if token:
            model_kwargs["token"] = token
        return diffusers.Ideogram4Transformer2DModel.from_pretrained(repo_id, **model_kwargs)

    check_cancelled()
    config.pop("quantization_config", None)
    hidden_size = int(config["attention_head_dim"]) * int(config["num_attention_heads"])
    # Build at target dtype with meta parameters: the converted tensors are assigned in, so the ~9B scaffold is
    # never materialised beside them (that, plus every stored shard held at once, was ~2.5x the DiT at load).
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with _empty_weights():
            model = diffusers.Ideogram4Transformer2DModel.from_config(config)
    finally:
        torch.set_default_dtype(default_dtype)
    check_cancelled()

    def storable(weight_key: str) -> bool:
        try:
            module = model.get_submodule(weight_key[: -len(".weight")])
        except AttributeError:
            return False
        return type(module) is torch.nn.Linear

    stored: dict = {}
    with contextlib.ExitStack() as stack:
        raw = _open_shards(stack, shard_paths, check_cancelled)
        state_dict = _convert_fp8_state_dict(
            raw,
            hidden_size,
            dtype,
            check_cancelled = check_cancelled,
            stored = stored if fp8_storage else None,
            storable = storable if fp8_storage else None,
        )
    check_cancelled()
    if stored:
        from .diffusion_comfy_quant import FP8_STORED_ATTR
        from .diffusion_native_quant import NATIVE_FP8, native_linear_class

        cls = native_linear_class()
        for fqn, (codes, scale) in stored.items():
            check_cancelled()
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            # The meta Linear gives shapes and compute dtype; its bias is assigned from the checkpoint below.
            layer = cls(getattr(parent, leaf), NATIVE_FP8, codes = codes.contiguous(), scale = scale)
            setattr(parent, leaf, layer)
        stored_names = {f"{fqn}.{b}" for fqn in stored for b in ("weight_q", "weight_scale")}
    else:
        stored_names = set()
    # inv_freq is built in __init__; any other missing or leftover key must fail loudly.
    real_missing, unexpected = _assign_state_dict(
        model,
        state_dict,
        lambda k: k.endswith("rotary_emb.inv_freq") or k in stored_names,
    )
    del state_dict
    if real_missing or unexpected:
        raise RuntimeError(
            f"ideogram4 fp8 remap left keys unmatched for {repo_id}/{subfolder}: "
            f"missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    check_cancelled()
    # Integer-view buffers: a module-wide cast leaves the stored codes and scales untouched.
    model.to(dtype)
    if stored:
        setattr(model, FP8_STORED_ATTR, len(stored))
        logger.info(
            "diffusion.ideogram4: %s/%s keeps %d fp8 Linears as stored codes, widened per call to %s",
            repo_id,
            subfolder,
            len(stored),
            str(dtype).replace("torch.", ""),
        )
    # Preserve the published source precision after widening the tensors to bf16.
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
    text_encoder: Any = None,
    fp8_storage: bool = False,
):
    """Assemble Ideogram4Pipeline from ``repo_id`` per-component (see module doc).

    ``text_encoder`` is a ready encoder (the hosted pre-cast fp8 one); the dense vendor-fp8 encoder is read and
    dequantized only when it is None, so a pre-cast load never touches the ~9 GB of text_encoder shards.
    ``fp8_storage`` keeps both DiTs' fp8 Linears stored (``load_ideogram4_transformer``)."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    import diffusers

    # The pipeline's text-encoder call uses a 5.x create_causal_mask signature; adapt it first.
    _patch_create_causal_mask()
    check_cancelled()

    token = hf_token or None
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if token:
        model_kwargs["token"] = token

    if text_encoder is None:
        text_encoder = load_ideogram4_text_encoder(
            repo_id, dtype, hf_token = token, check_cancelled = check_cancelled
        )
    check_cancelled()
    tokenizer = load_krea2_tokenizer(repo_id, hf_token = token, check_cancelled = check_cancelled)
    check_cancelled()
    # Pass fp8_storage only when set, so a loader stub with the old signature keeps working.
    storage_kwargs = {"fp8_storage": True} if fp8_storage else {}
    transformer = load_ideogram4_transformer(
        repo_id,
        "transformer",
        dtype,
        hf_token = token,
        check_cancelled = check_cancelled,
        **storage_kwargs,
    )
    check_cancelled()
    # A local checkpoint dir without the file must fail clearly here, else hf_hub_download dies with an opaque HFValidationError
    unconditional_transformer = load_ideogram4_transformer(
        repo_id,
        "unconditional_transformer",
        dtype,
        hf_token = token,
        check_cancelled = check_cancelled,
        **storage_kwargs,
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
