# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3 pipeline assembly for diffusers 0.39.

diffusers 0.39 ships every LTX-2.3 model class but its single-file loader maps every LTX-2
checkpoint to the 2.0 config, so 2.3 checkpoints fail a shape check at load. The community
transformer-only GGUFs also carry the DiT + connectors but NOT the text projections, VAEs, or
vocoder that 2.3 moved out of the transformer. This assembles the full 2.3 pipeline:

- transformer: from the checkpoint via ``from_single_file`` with the 2.3 config overrides and the
  ``prompt_adaln_single`` keys pre-renamed (the library converter doesn't know them).
- connectors: from the checkpoint's connector keys plus the ``text_embedding_projection`` tensors,
  fetched from the companion file in ``unsloth/LTX-2.3-GGUF`` when not bundled.
- video/audio VAE, vocoder: from the checkpoint when bundled, else the companion files.
- scheduler, text encoder (Gemma3), tokenizer: from the LTX-2.0 base repo, which 2.3 shares.

Every config and rename table mirrors diffusers' ``scripts/convert_ltx2_to_diffusers.py`` (the
authoritative 2.3 mapping the loader hasn't absorbed). Assembled through the constructor, not
``from_pretrained``, because the vocoder class differs from the base pin (``LTX2VocoderWithBWE`` vs
``LTX2Vocoder``) and the type gate would reject it.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Companion files (projections, VAEs, vocoder) beside the quants in unsloth's GGUF repo.
LTX23_EXTRAS_REPO = "unsloth/LTX-2.3-GGUF"


def _live_cache_dir() -> str:
    """Unsloth's LIVE hub cache root. Read from utils rather than ``diffusion.hub_cache_dir`` to
    avoid a circular import, the same way diffusion_auto_policy does."""
    from utils.hf_cache_settings import active_hf_hub_cache
    return active_hf_hub_cache()


_EXTRAS_TEXT_PROJ = "text_encoders/ltx-2.3-22b-{variant}_embeddings_connectors.safetensors"
_EXTRAS_VIDEO_VAE = "vae/ltx-2.3-22b-{variant}_video_vae.safetensors"
_EXTRAS_AUDIO_VAE = "vae/ltx-2.3-22b-{variant}_audio_vae.safetensors"


LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES: dict[str, Any] = {
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "use_prompt_embeddings": False,
    "perturbed_attn": True,
}

_TRANSFORMER_PRERENAME = (
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
)

_CONNECTOR_KEY_PREFIXES = (
    "video_embeddings_connector",
    "audio_embeddings_connector",
    "transformer_1d_blocks",
    "text_embedding_projection",
    "connectors.",
    "video_connector",
    "audio_connector",
    "text_proj_in",
)

_CONNECTORS_RENAME = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.audio_aggregate_embed": "audio_text_proj_in",
    "text_embedding_projection.video_aggregate_embed": "video_text_proj_in",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

_CONNECTORS_CONFIG: dict[str, Any] = {
    "caption_channels": 3840,
    "text_proj_in_factor": 49,
    "video_connector_num_attention_heads": 32,
    "video_connector_attention_head_dim": 128,
    "video_connector_num_layers": 8,
    "video_connector_num_learnable_registers": 128,
    "video_gated_attn": True,
    "audio_connector_num_attention_heads": 32,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_layers": 8,
    "audio_connector_num_learnable_registers": 128,
    "audio_gated_attn": True,
    "connector_rope_base_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_temporal_positioning": False,
    "rope_type": "split",
    "per_modality_projections": True,
    "video_hidden_dim": 4096,
    "audio_hidden_dim": 2048,
    "proj_bias": True,
}

_VIDEO_VAE_RENAME = {
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
    "last_time_embedder": "time_embedder",
    "last_scale_shift_table": "scale_shift_table",
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

_VIDEO_VAE_REMOVE_SUFFIXES = (
    "per_channel_statistics.channel",
    "per_channel_statistics.mean-of-stds",
)

_VIDEO_VAE_CONFIG: dict[str, Any] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 128,
    "block_out_channels": (256, 512, 1024, 1024),
    "down_block_types": (
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
        "LTX2VideoDownBlock3D",
    ),
    "decoder_block_out_channels": (256, 512, 512, 1024),
    "layers_per_block": (4, 6, 4, 2, 2),
    "decoder_layers_per_block": (4, 6, 4, 2, 2),
    "spatio_temporal_scaling": (True, True, True, True),
    "decoder_spatio_temporal_scaling": (True, True, True, True),
    "decoder_inject_noise": (False, False, False, False, False),
    "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    "upsample_type": ("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    "upsample_residual": (False, False, False, False),
    "upsample_factor": (2, 2, 1, 2),
    "timestep_conditioning": False,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-6,
    "encoder_causal": True,
    "decoder_causal": False,
    "encoder_spatial_padding_mode": "zeros",
    "decoder_spatial_padding_mode": "zeros",
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
}

_AUDIO_VAE_RENAME = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

_AUDIO_VAE_CONFIG: dict[str, Any] = {
    "base_channels": 128,
    "output_channels": 2,
    "ch_mult": (1, 2, 4),
    "num_res_blocks": 2,
    "attn_resolutions": None,
    "in_channels": 2,
    "resolution": 256,
    "latent_channels": 8,
    "norm_type": "pixel",
    "causality_axis": "height",
    "dropout": 0.0,
    "mid_block_add_attention": False,
    "sample_rate": 16000,
    "mel_hop_length": 160,
    "is_causal": True,
    "mel_bins": 64,
    "double_z": True,
}

_VOCODER_RENAME = {
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
    "act_post": "act_out",
    "downsample.lowpass": "downsample",
}

_VOCODER_CONFIG: dict[str, Any] = {
    "in_channels": 128,
    "hidden_channels": 1536,
    "out_channels": 2,
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
    "upsample_factors": [5, 2, 2, 2, 2, 2],
    "resnet_kernel_sizes": [3, 7, 11],
    "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "act_fn": "snakebeta",
    "leaky_relu_negative_slope": 0.1,
    "antialias": True,
    "antialias_ratio": 2,
    "antialias_kernel_size": 12,
    "final_act_fn": None,
    "final_bias": False,
    "bwe_in_channels": 128,
    "bwe_hidden_channels": 512,
    "bwe_out_channels": 2,
    "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
    "bwe_upsample_factors": [6, 5, 2, 2, 2],
    "bwe_resnet_kernel_sizes": [3, 7, 11],
    "bwe_resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "bwe_act_fn": "snakebeta",
    "bwe_leaky_relu_negative_slope": 0.1,
    "bwe_antialias": True,
    "bwe_antialias_ratio": 2,
    "bwe_antialias_kernel_size": 12,
    "bwe_final_act_fn": None,
    "bwe_final_bias": False,
    "filter_length": 512,
    "hop_length": 80,
    "window_length": 512,
    "num_mel_channels": 64,
    "input_sampling_rate": 16000,
    "output_sampling_rate": 48000,
}

_DIT_PREFIX = "model.diffusion_model."


def read_checkpoint_header(checkpoint_path: Path | str) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape from the checkpoint HEADER only (no weight data). GGUF shapes come back
    in GGML (reversed) order, so callers should membership-test, not assume a dimension position."""
    names_shapes: dict[str, tuple[int, ...]] = {}
    path = str(checkpoint_path)
    if path.lower().endswith(".gguf"):
        from gguf import GGUFReader
        for tensor in GGUFReader(path).tensors:
            names_shapes[str(tensor.name)] = tuple(int(x) for x in tensor.shape)
    else:
        from safetensors import safe_open
        with safe_open(path, framework = "pt") as handle:
            for name in handle.keys():
                names_shapes[name] = tuple(handle.get_slice(name).get_shape())
    return names_shapes


def is_ltx23_checkpoint(checkpoint_path: Path | str) -> bool:
    """True when the checkpoint carries the 9-row LTX-2.3 modulation tables (2.0 has 6-row
    per-block scale/shift tables; 2.3 widens them to 9). An unreadable header returns False so the
    caller falls back to the stock 2.0 path."""
    try:
        header = read_checkpoint_header(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("video.ltx2_header_probe_failed: %s", exc)
        return False
    for name, shape in header.items():
        if name.endswith("transformer_blocks.0.scale_shift_table"):
            return 9 in shape
    return False


def _apply_rename(state: dict[str, Any], rename: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        new_key = key
        for old, new in rename.items():
            new_key = new_key.replace(old, new)
        out[new_key] = value
    return out


def _to_plain_dtype(state: dict[str, Any], torch_dtype: Any) -> dict[str, Any]:
    """Materialise every tensor as a plain torch tensor in torch_dtype. GGUF tensors arrive as
    block-quantized GGUFParameter; the small non-DiT components run dense, so dequantize here."""
    import torch

    try:
        from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    except Exception:  # noqa: BLE001 -- gguf support not installed; plain tensors only
        GGUFParameter, dequantize_gguf_tensor = (), None

    out: dict[str, Any] = {}
    for key, value in state.items():
        if dequantize_gguf_tensor is not None and isinstance(value, GGUFParameter):
            value = dequantize_gguf_tensor(value)
        out[key] = value.to(torch_dtype) if isinstance(value, torch.Tensor) else value
    return out


def _split_checkpoint(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Partition a combined LTX checkpoint into per-component state dicts. Handles both layouts: the
    official combined single file (``vae.*`` / ``audio_vae.*`` / ``vocoder.*`` / DiT + projections)
    and transformer-only GGUFs (bare DiT + connector keys)."""
    groups: dict[str, dict[str, Any]] = {
        "dit": {},
        "connectors": {},
        "vae": {},
        "audio_vae": {},
        "vocoder": {},
    }
    for key, value in state.items():
        group, name = _checkpoint_group(key)
        groups[group][name] = value
    return groups


def _checkpoint_group(key: str) -> tuple[str, str]:
    """(component group, key within it) for one combined-checkpoint key; see ``_split_checkpoint``."""
    bare = key[len(_DIT_PREFIX) :] if key.startswith(_DIT_PREFIX) else key
    for prefix, group in (("vae.", "vae"), ("audio_vae.", "audio_vae"), ("vocoder.", "vocoder")):
        if bare.startswith(prefix):
            return group, bare[len(prefix) :]
    if bare.startswith(_CONNECTOR_KEY_PREFIXES):
        return "connectors", bare
    return "dit", bare


def _load_checkpoint_without_dit(checkpoint_path: Path | str) -> Optional[dict[str, Any]]:
    """The non-DiT tensors of a combined safetensors checkpoint, read selectively (the DiT is ~44 GB of
    the 46 GB file); None when the file is not safetensors, so the caller reads it whole."""
    if not str(checkpoint_path).lower().endswith(".safetensors"):
        return None
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework = "pt", device = "cpu") as handle:
        return {
            key: handle.get_tensor(key)
            for key in handle.keys()
            if _checkpoint_group(key)[0] != "dit"
        }


DIRECT_LOAD_ENV = "UNSLOTH_VIDEO_DIRECT_LOAD"
_DIRECT_CHUNK_BYTES = 16 << 20
_DIRECT_BUFFERS = 16
_DIRECT_THREADS = 8
_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


_SWITCH_OFF = ("0", "off", "false", "no")


def direct_load_enabled() -> bool:
    import os
    return (os.environ.get(DIRECT_LOAD_ENV) or "").strip().lower() not in _SWITCH_OFF


def direct_load_device(device: Any) -> Optional[Any]:
    """The device a resident LTX-2.3 checkpoint is read straight onto, or None."""
    if not direct_load_enabled() or device is None:
        return None
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        return dev
    except Exception:  # noqa: BLE001 - an unparseable device keeps the host load
        return None


def _read_safetensors_header(path: str) -> tuple[int, dict[str, Any]]:
    import json
    import struct

    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return 8 + size, header


def read_safetensors_to_device(
    checkpoint_path: Path | str,
    device: Any,
    keep: Any = None,
    *,
    chunk_bytes: int = _DIRECT_CHUNK_BYTES,
    buffers: int = _DIRECT_BUFFERS,
    threads: int = _DIRECT_THREADS,
) -> Optional[dict[str, Any]]:
    """Read the safetensors tensors passing ``keep`` straight into ``device`` memory via a pinned ring (46 GB: 13.5 s
    host path vs 2.5 s). None for a dtype this reader does not map."""
    import torch

    path = str(checkpoint_path)
    data_start, header = _read_safetensors_header(path)
    entries = []
    for name, info in header.items():
        if keep is not None and not keep(name):
            continue
        dtype_name = _SAFETENSORS_DTYPES.get(str(info.get("dtype")))
        dtype = getattr(torch, dtype_name, None) if dtype_name else None
        if dtype is None:
            return None
        begin, end = (int(x) for x in info["data_offsets"])
        entries.append((begin, end, name, dtype, tuple(int(x) for x in info["shape"])))
    entries.sort()
    device = torch.device(device)
    out: dict[str, Any] = {}
    jobs: list[tuple[int, int, Any, int]] = []
    for begin, end, name, dtype, shape in entries:
        tensor = torch.empty(shape, dtype = dtype, device = device)
        out[name] = tensor
        nbytes = end - begin
        if nbytes != tensor.numel() * tensor.element_size():
            raise ValueError(f"safetensors entry {name} has {nbytes} bytes for shape {shape}")
        if nbytes == 0:
            continue
        flat = tensor.view(-1).view(torch.uint8)
        offset = 0
        while offset < nbytes:
            size = min(chunk_bytes, nbytes - offset)
            jobs.append((data_start + begin + offset, size, flat, offset))
            offset += size
    if not jobs:
        return out
    _copy_file_chunks(path, jobs, device, chunk_bytes, buffers, threads)
    return out


def _copy_file_chunks(
    path: str,
    jobs: list[tuple[int, int, Any, int]],
    device: Any,
    chunk_bytes: int,
    buffers: int,
    threads: int,
) -> None:
    """Copy ``(file offset, size, uint8 view, view offset)`` jobs; reads overlap the async uploads."""
    import queue
    from concurrent.futures import ThreadPoolExecutor

    import torch

    if torch.device(device).type != "cuda":
        with open(path, "rb", buffering = 0) as handle:
            for offset, size, flat, at in jobs:
                view = memoryview(flat.numpy())[at : at + size]
                handle.seek(offset)
                got = 0
                while got < size:
                    read = handle.readinto(view[got:])
                    if not read:
                        raise EOFError(f"{path}: short read at {offset + got}")
                    got += read
        return
    buffers = max(1, min(buffers, len(jobs)))
    staging = [torch.empty(chunk_bytes, dtype = torch.uint8, pin_memory = True) for _ in range(buffers)]
    done_events: list[Any] = [None] * buffers
    handles: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
    opened: list[Any] = []

    def _read(slot: int, offset: int, size: int) -> None:
        event = done_events[slot]
        if event is not None:
            event.synchronize()
        try:
            handle = handles.get_nowait()
        except queue.Empty:
            handle = open(path, "rb", buffering = 0)
            opened.append(handle)
        try:
            view = memoryview(staging[slot].numpy())[:size]
            handle.seek(offset)
            got = 0
            while got < size:
                read = handle.readinto(view[got:])
                if not read:
                    raise EOFError(f"{path}: short read at {offset + got}")
                got += read
        finally:
            handles.put(handle)

    stream = torch.cuda.Stream(device = device)
    try:
        with ThreadPoolExecutor(
            max_workers = max(1, threads), thread_name_prefix = "unsloth-direct-load"
        ) as pool:
            pending = {}
            for index in range(min(buffers, len(jobs))):
                offset, size, _flat, _at = jobs[index]
                pending[index] = pool.submit(_read, index % buffers, offset, size)
            with torch.cuda.stream(stream):
                for index, (offset, size, flat, at) in enumerate(jobs):
                    slot = index % buffers
                    pending.pop(index).result()
                    flat[at : at + size].copy_(staging[slot][:size], non_blocking = True)
                    event = torch.cuda.Event()
                    event.record(stream)
                    done_events[slot] = event
                    following = index + buffers
                    if following < len(jobs):
                        next_offset, next_size, _f, _a = jobs[following]
                        pending[following] = pool.submit(_read, slot, next_offset, next_size)
            stream.synchronize()
    finally:
        try:
            stream.synchronize()
        except Exception:  # noqa: BLE001
            pass
        for handle in opened:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        # The side stream is synchronised above, so the default stream sees finished bytes.
        del staging
        try:
            torch._C._host_emptyCache()
        except Exception:  # noqa: BLE001 - older torch keeps the 256 MiB ring cached
            pass


def _load_checkpoint_without_dit_to_device(
    checkpoint_path: Path | str, device: Any
) -> Optional[dict[str, Any]]:
    """``_load_checkpoint_without_dit`` onto ``device``; None falls back to the host read."""
    if not str(checkpoint_path).lower().endswith(".safetensors"):
        return None
    try:
        return read_safetensors_to_device(
            checkpoint_path, device, keep = lambda key: _checkpoint_group(key)[0] != "dit"
        )
    except Exception as exc:  # noqa: BLE001 - a failed direct read degrades to the host load
        logger.warning("video.ltx23_direct_load: falling back to the host load (%s)", exc)
        _release_device_cache()
        return None


def _release_device_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def load_checkpoint_to_device(checkpoint_path: Path | str, device: Any) -> Optional[dict[str, Any]]:
    """Every tensor of a safetensors checkpoint on ``device``; None falls back to the host read."""
    if not str(checkpoint_path).lower().endswith(".safetensors"):
        return None
    try:
        return read_safetensors_to_device(checkpoint_path, device)
    except Exception as exc:  # noqa: BLE001 - a failed direct read degrades to the host load
        logger.warning("video.ltx23_direct_load: falling back to the host load (%s)", exc)
        _release_device_cache()
        return None


def _load_extras_file(
    filename: str,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

    path = hf_hub_download_with_xet_fallback(
        LTX23_EXTRAS_REPO,
        filename,
        hf_token,
        # Resolve extras under EITHER cache root, else re-pull what the planner skipped.
        reuse_other_cache_root = True,
        local_files_only = local_files_only,
        gguf_header_delta = True,
    )
    return load_file(path)


# sha256[:16] of ``proj_out.bias`` as bf16 bits: headers match across 2.3 variants, but this bias is retrained per
# release and never quantized, so fp8 / int8 / GGUF repacks carry the release's values.
LTX23_PROJ_OUT_BIAS_SHA = {
    "b6a06f88015c612b": "distilled",  # ltx-2.3-22b-distilled (and -distilled-fp8)
    "140a7d077a306ea9": "distilled",  # ltx-2.3-22b-distilled-1.1
    "3d334d94df5daf30": "dev",  # ltx-2.3-22b-dev (and -dev-fp8)
}
_PROJ_OUT_BIAS_NAMES = (
    "proj_out.bias",
    "model.diffusion_model.proj_out.bias",
    "diffusion_model.proj_out.bias",
)
_CONTENT_VARIANT_CACHE: dict[tuple[str, int, int], Optional[str]] = {}


def _safetensors_tensor_bf16_bits(path: str, names: tuple[str, ...]) -> Optional[bytes]:
    """One small tensor's values as bfloat16 bits, read through the header offsets (no other weight byte)."""
    import json
    import struct

    import torch

    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        if size <= 0 or size > 256 * 1024 * 1024:
            return None
        header = json.loads(fh.read(size))
        entry = next((header[n] for n in names if isinstance(header.get(n), dict)), None)
        if entry is None:
            return None
        dtype = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}.get(
            entry.get("dtype")
        )
        start, end = entry["data_offsets"]
        if dtype is None or not 0 < end - start <= 1 << 20:
            return None
        fh.seek(8 + size + start)
        raw = bytearray(fh.read(end - start))
    values = torch.frombuffer(raw, dtype = dtype)
    return values.to(torch.bfloat16).view(torch.int16).numpy().tobytes()


def _gguf_tensor_bf16_bits(path: str, names: tuple[str, ...]) -> Optional[bytes]:
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, GGUFReader

    for tensor in GGUFReader(path).tensors:
        if str(tensor.name) not in names:
            continue
        data = np.asarray(tensor.data)
        if tensor.tensor_type == GGMLQuantizationType.F32:
            values = torch.from_numpy(data.astype(np.float32).reshape(-1).copy())
        elif tensor.tensor_type == GGMLQuantizationType.F16:
            values = torch.from_numpy(data.astype(np.float16).reshape(-1).copy())
        elif tensor.tensor_type == GGMLQuantizationType.BF16:
            values = torch.from_numpy(data.reshape(-1).view(np.int16).copy()).view(torch.bfloat16)
        else:
            return None
        return values.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    return None


def ltx23_checkpoint_variant(checkpoint_path: Path | str | None) -> Optional[str]:
    """``"distilled"`` / ``"dev"`` from the checkpoint's CONTENT (a 128-value bias fingerprint), or None when the file
    is not on disk, unreadable, or not a known LTX-2.3 release (a finetune): callers then fall back to the name. Reads
    the header plus 256 bytes; cached on (path, size, mtime)."""
    if not checkpoint_path:
        return None
    path = str(checkpoint_path)
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key in _CONTENT_VARIANT_CACHE:
        return _CONTENT_VARIANT_CACHE[key]
    variant: Optional[str] = None
    try:
        import hashlib

        reader = (
            _gguf_tensor_bf16_bits
            if path.lower().endswith(".gguf")
            else _safetensors_tensor_bf16_bits
        )
        bits = reader(path, _PROJ_OUT_BIAS_NAMES)
        if bits is not None:
            variant = LTX23_PROJ_OUT_BIAS_SHA.get(hashlib.sha256(bits).hexdigest()[:16])
    except Exception as exc:  # noqa: BLE001 -- the name decides instead
        logger.debug("video.ltx23_variant_probe_failed: %s", exc)
    _CONTENT_VARIANT_CACHE[key] = variant
    return variant


def ltx23_variant_identifier(checkpoint_path: Path | str | None) -> Optional[str]:
    """An identifier naming the content-detected variant, to put FIRST in the id lists the generation defaults and the
    distilled recipe read (``default_video_generation_params`` / ``ltx2_distilled_ids``), so the file's weights outrank
    its name. None when the content is inconclusive, leaving the name to decide as before."""
    variant = ltx23_checkpoint_variant(checkpoint_path)
    return f"ltx-2.3-22b-{variant}" if variant else None


def checkpoint_variant(checkpoint_path: Path | str) -> str:
    """Which companion-weight set a checkpoint pairs with ("dev"/"distilled"): the file's content when it is a known
    release, else its name. The distilled-1.1 refresh only retrained the DiT, so it shares the distilled companions."""
    by_content = ltx23_checkpoint_variant(checkpoint_path)
    if by_content is not None:
        return by_content
    return "dev" if "dev" in Path(checkpoint_path).name.lower() else "distilled"


def ltx23_extras_files(checkpoint_path: Path | str) -> tuple[str, ...]:
    """The companion files in ``LTX23_EXTRAS_REPO`` a 2.3 checkpoint loads alongside itself.

    Same variant rule as the assembly, so the download plan stages exactly what the load reads
    (they are otherwise fetched inline, outside the panel's progress, cancel and disk preflight)."""
    variant = checkpoint_variant(checkpoint_path)
    return tuple(
        template.format(variant = variant)
        for template in (_EXTRAS_TEXT_PROJ, _EXTRAS_VIDEO_VAE, _EXTRAS_AUDIO_VAE)
    )


# Upstream ltx_core DISTILLED_SIGMA_VALUES: the fixed 8-step curve the distilled DiT expects.
LTX23_DISTILLED_SIGMAS: tuple[float, ...] = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
)


def ltx2_distilled_ids(*ids: Optional[str]) -> bool:
    """Distilled DiT, by the generation defaults' precedence (selected file before repo): a dev file under a
    '...distilled...' folder stays dev."""
    from .video_families import video_generation_variant
    return (
        video_generation_variant(*(str(i) if i is not None else None for i in ids)) == "distilled"
    )


# LTX2Pipeline guidance kwargs and their off values: diffusers #14447 moved defaults to the dev recipe (STG + modality passes).
_LTX2_GUIDANCE_OFF: dict[str, float] = {
    "stg_scale": 0.0,
    "modality_scale": 1.0,
    "guidance_rescale": 0.0,
    "audio_stg_scale": 0.0,
    "audio_modality_scale": 1.0,
    "audio_guidance_rescale": 0.0,
}


def ltx2_distilled_guidance_kwargs(call_params: Any, guidance: Optional[float]) -> dict[str, float]:
    """Distilled LTX-2/2.3 guidance: STG, modality and rescale off, audio CFG = video CFG (one unguided forward per
    step). Only kwargs the installed pipeline accepts, so older diffusers get the call they always did."""
    kwargs = {k: v for k, v in _LTX2_GUIDANCE_OFF.items() if k in call_params}
    if "audio_guidance_scale" in call_params:
        kwargs["audio_guidance_scale"] = float(guidance if guidance is not None else 1.0)
    return kwargs


# Hosted 2.3 checkpoints were baked from this file only (not distilled-1.1 or dev DiTs).
LTX23_PREQUANT_BASE = "Lightricks/LTX-2.3"
LTX23_PREQUANT_SOURCE_FILES = frozenset({"ltx-2.3-22b-distilled.safetensors"})
# Resident size of LTX-2.3-FP8.pt (19,057,628,489 bytes); companions are priced separately.
LTX23_PREQUANT_RESIDENT_GB = 19.06


# The hosted DiT REPLACES the file's own, so the file must be the official one (a same-named fine-tune would be swapped silently).
LTX23_PREQUANT_SOURCE_REPOS = frozenset({"lightricks/ltx-2.3"})
_LTX23_HUB_REPO_DIRS = frozenset(
    "models--" + r.replace("/", "--") for r in LTX23_PREQUANT_SOURCE_REPOS
)
LTX23_PREQUANT_SOURCE_SIZE = 46_149_345_038
# LFS sha256, also the Hub cache blob name.
LTX23_PREQUANT_SOURCE_SHA256 = "14409a4d1337a8ded02fa87fb895b17a91ab2c6588f7cc3352e624ff18a689bf"
# Elsewhere the whole file is hashed once (~30 s), persisted per (realpath, size, mtime_ns, inode); a sample misses partial fine-tunes.
_LTX23_HASH_CHUNK = 16 << 20
_LTX23_VERDICTS_FILE = "ltx23-source-verdicts.json"
_LTX23_VERDICTS_VERSION = 1
_LTX23_VERIFY_LOCK = threading.Lock()
_LTX23_NO_HASH: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ltx23_no_hash", default = False
)


@contextlib.contextmanager
def ltx23_identity_without_hashing() -> Iterator[None]:
    """For planning (the download plan and its precision check): a local file with the official size and no stored
    verdict reads as official instead of being hashed there, like an uncached Hub pick; the load hashes it."""
    token = _LTX23_NO_HASH.set(True)
    try:
        yield
    finally:
        _LTX23_NO_HASH.reset(token)


def _ltx23_verdicts_path() -> Path:
    from utils.paths.storage_roots import cache_root
    return cache_root() / _LTX23_VERDICTS_FILE


def _ltx23_read_verdicts() -> dict[str, Any]:
    import json

    try:
        data = json.loads(_ltx23_verdicts_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(data, dict)
        or data.get("version") != _LTX23_VERDICTS_VERSION
        or data.get("sha256") != LTX23_PREQUANT_SOURCE_SHA256
        or not isinstance(data.get("files"), dict)
    ):
        return {}
    return data["files"]


def _ltx23_write_verdict(real: str, record: dict[str, Any]) -> None:
    import json
    import os
    import uuid

    files = _ltx23_read_verdicts()
    files[real] = record
    path = _ltx23_verdicts_path()
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    payload = {
        "version": _LTX23_VERDICTS_VERSION,
        "sha256": LTX23_PREQUANT_SOURCE_SHA256,
        "files": files,
    }
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        with tmp.open("w", encoding = "utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not persist the LTX-2.3 source verdict: %s", exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass


def _ltx23_in_hub_cache_root(root: Path) -> bool:
    """Whether *root* is one of the Hugging Face hub caches Studio downloads into (a look-alike tree elsewhere is not)."""
    try:
        from hub.utils.hf_cache_state import hf_cache_roots
        return any(root == Path(r).resolve() for r in hf_cache_roots())
    except Exception:  # noqa: BLE001 - unknown roots: hash instead
        return False


def _ltx23_stat_key(stat: Any) -> dict[str, int]:
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def ltx23_source_sha256(path: Path | str) -> str:
    """The full sha256 of *path*. Raises OSError on a read failure."""
    import hashlib

    digest = hashlib.sha256()
    buf = bytearray(_LTX23_HASH_CHUNK)
    view = memoryview(buf)
    with open(path, "rb", buffering = 0) as fh:
        while n := fh.readinto(buf):
            digest.update(view[:n])
    return digest.hexdigest()


def ltx23_source_file_verified(checkpoint_path: Path | str) -> bool:
    """Whether the file on disk is the official ``ltx-2.3-22b-distilled.safetensors``: its size, then the Hub cache's
    content-addressed blob name, else the full sha256 (persisted per realpath, size, mtime and inode). Never raises."""
    try:
        path = Path(str(checkpoint_path)).expanduser()
        if path.name.lower() not in LTX23_PREQUANT_SOURCE_FILES:
            return False
        real = path.resolve()
        stat = real.stat()
        if stat.st_size != LTX23_PREQUANT_SOURCE_SIZE:
            return False
        if (
            real.name == LTX23_PREQUANT_SOURCE_SHA256
            and real.parent.name == "blobs"
            and real.parent.parent.name.lower() in _LTX23_HUB_REPO_DIRS
            and _ltx23_in_hub_cache_root(real.parent.parent.parent)
        ):
            return True
        key = _ltx23_stat_key(stat)
        stored = _ltx23_read_verdicts().get(str(real))
        if isinstance(stored, dict) and {k: stored.get(k) for k in key} == key:
            return stored.get("verified") is True
        if _LTX23_NO_HASH.get():
            return True
        with _LTX23_VERIFY_LOCK:
            stored = _ltx23_read_verdicts().get(str(real))
            if isinstance(stored, dict) and {k: stored.get(k) for k in key} == key:
                return stored.get("verified") is True
            logger.info(
                "video.ltx23_prequant: hashing %s once to confirm it is the official file", real
            )
            verified = ltx23_source_sha256(real) == LTX23_PREQUANT_SOURCE_SHA256
            if _ltx23_stat_key(real.stat()) != key:
                return False
            _ltx23_write_verdict(str(real), {**key, "verified": verified})
            return verified
    except Exception:  # noqa: BLE001 -- unverifiable is not official
        return False


def _ltx23_hub_cached_file(repo_id: str, filename: str) -> Optional[Path]:
    """The cached copy of *filename* in *repo_id* under either cache root, network-free; None when not cached."""
    try:
        from huggingface_hub import try_to_load_from_cache

        from .diffusion import hub_cache_dir
        for cache_dir in dict.fromkeys((None, hub_cache_dir())):
            hit = try_to_load_from_cache(repo_id, filename, cache_dir = cache_dir)
            if isinstance(hit, str):
                return Path(hit)
    except Exception:  # noqa: BLE001 -- a lookup failure reads as not cached
        return None
    return None


def ltx23_prequant_eligible(
    checkpoint_filename: Optional[str], repo_id: Optional[str] = None
) -> bool:
    """Whether the hosted DiT may replace this pick's own: the official file by name AND identity (local content, or
    the official Hub repo, its cached copy verified by content; the load re-checks)."""
    if not checkpoint_filename or not repo_id:
        return False
    if Path(str(checkpoint_filename)).name.lower() not in LTX23_PREQUANT_SOURCE_FILES:
        return False
    try:
        root = Path(str(repo_id)).expanduser()
        if root.is_file():
            return ltx23_source_file_verified(root)
        if root.is_dir():
            from .diffusion_families import resolve_local_gguf_child
            return ltx23_source_file_verified(
                resolve_local_gguf_child(root, str(checkpoint_filename))
            )
    except Exception:  # noqa: BLE001 -- an unresolvable local pick is not verified
        return False
    if str(repo_id).strip().lower() not in LTX23_PREQUANT_SOURCE_REPOS:
        return False
    cached = _ltx23_hub_cached_file(str(repo_id).strip(), str(checkpoint_filename))
    return True if cached is None else ltx23_source_file_verified(cached)


class _LTX23PrequantConfig:
    """Transformer "class" for ``load_prequantized_transformer``: the 2.0 config + 2.3 overrides, since the
    single-file 2.3 repo has no ``transformer/``."""

    def __init__(self, config_repo: str):
        self.config_repo = config_repo

    def load_config(self, _base: str, **kwargs: Any) -> dict[str, Any]:
        from diffusers import LTX2VideoTransformer3DModel

        config = dict(LTX2VideoTransformer3DModel.load_config(self.config_repo, **kwargs))
        config.update(LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES)
        return config

    @staticmethod
    def from_config(config: Any) -> Any:
        from diffusers import LTX2VideoTransformer3DModel
        return LTX2VideoTransformer3DModel.from_config(config)


def load_ltx23_prequant_transformer(
    fam: Any,
    scheme: str,
    checkpoint_path: Path | str,
    *,
    config_repo: str,
    device: str,
    dtype: Any,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> Optional[tuple[Any, Any]]:
    """``(transformer, source)`` from the hosted pre-quantized 2.3 distilled DiT, or None (the
    caller keeps the dense DiT). Never raises."""
    try:
        if not ltx23_source_file_verified(checkpoint_path):
            if logger is not None:
                logger.warning(
                    "video.ltx23_prequant: %s is not the official LTX-2.3 distilled file, keeping its own DiT",
                    checkpoint_path,
                )
            return None
        from .diffusion_prequant import load_prequantized_transformer, resolve_prequant_source
        from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES

        source = resolve_prequant_source(fam, scheme, base_repo = LTX23_PREQUANT_BASE)
        if source is None:
            return None
        module = load_prequantized_transformer(
            _LTX23PrequantConfig(config_repo),
            LTX23_PREQUANT_BASE,
            source,
            device = device,
            dtype = dtype,
            hf_token = hf_token,
            scheme = scheme,
            min_features = DEFAULT_MIN_LINEAR_FEATURES,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            logger = logger,
        )
        return None if module is None else (module, source)
    except Exception as exc:  # noqa: BLE001 -- a hosted checkpoint is an optimisation, never a blocker
        if logger is not None:
            logger.warning(
                "video.ltx23_prequant: %s failed, keeping the dense DiT: %s", scheme, exc
            )
        return None


def disable_cudnn_benchmark() -> bool:
    """Switch the process-wide cudnn.benchmark off (True when it changed)."""
    try:
        import torch

        if not torch.backends.cudnn.benchmark:
            return False
        torch.backends.cudnn.benchmark = False
        return True
    except Exception:  # noqa: BLE001 -- optimisation only
        return False


# Static compile: 1 graph per shape unguided, 4 guided; dynamo's default limit of 8 fails the third guided resolution.
LTX2_RECOMPILE_LIMIT = 64


def ensure_recompile_limit(limit: int = LTX2_RECOMPILE_LIMIT) -> None:
    """Raise dynamo's recompile limit in the calling thread: torch >= 2.12 keeps config per thread context, so the
    render thread would otherwise see the default 8 despite the load-time raise."""
    try:
        import torch._dynamo.config as dynamo_cfg
    except Exception:  # noqa: BLE001 -- no dynamo, nothing compiled
        return
    for attr in ("recompile_limit", "cache_size_limit"):  # name varies by torch version
        try:
            if hasattr(dynamo_cfg, attr) and (getattr(dynamo_cfg, attr) or 0) < limit:
                setattr(dynamo_cfg, attr, limit)
        except Exception:  # noqa: BLE001 -- optimisation only
            pass


def _recompile_limit_hit(exc: BaseException) -> bool:
    try:
        import torch
        kind = getattr(torch._dynamo.exc, "FailOnRecompileLimitHit", None)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(kind, type) and isinstance(exc, kind)


def install_stg_compile_adapter(transformer: Any) -> int:
    """Convert the STG pass's 0-d ``all_perturbed`` tensor to a bool before the compiled block (LTX2Attention branches
    on it, which fails fullgraph and drops the DiT to eager). Keeps the guard's marker so ``guard_compiled_blocks``
    stays idempotent; past the recompile limit drops to eager instead of failing. Returns blocks adapted."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return 0
    count = 0
    for module in getattr(transformer, "modules", lambda: ())():
        if type(module).__name__ != "LTX2VideoTransformerBlock":
            continue
        inner = getattr(module, "_compiled_call_impl", None)
        if inner is None or getattr(inner, "_unsloth_stg_adapter", False):
            continue

        guard = getattr(inner, "_unsloth_compile_guard", None)

        def adapted(
            *args: Any,
            _inner: Any = inner,
            _guard: Any = guard,
            **kwargs: Any,
        ) -> Any:
            flag = kwargs.get("all_perturbed")
            if isinstance(flag, torch.Tensor):
                kwargs["all_perturbed"] = bool(flag)
            try:
                return _inner(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- reraised unless the recompile limit was hit
                if _guard is None or _guard.error is not None or not _recompile_limit_hit(exc):
                    raise
                # Past the recompile limit fullgraph raises unclassified by the guard: drop this DiT to eager.
                _guard.fail(exc, transformer)
                exc.__traceback__ = None
                return _inner(*args, **kwargs)

        adapted._unsloth_stg_adapter = True
        if guard is not None:
            adapted._unsloth_compile_guard = guard
        module._compiled_call_impl = adapted
        count += 1
    return count


def ltx23_verbatim_sigmas(pipe: Any) -> Any:
    """Context manager neutralising the scheduler transforms that re-shape even explicit
    ``sigmas`` (FlowMatchEulerDiscreteScheduler applies dynamic time-shift and the
    shift_terminal stretch to caller-provided lists): dynamic shifting off, shift 1.0
    (identity), no terminal stretch, restored on exit. Without this the calibrated curve
    above would arrive at the DiT distorted (its 0.421875 tail clamped to 0.1)."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        sched = getattr(pipe, "scheduler", None)
        cfg = getattr(sched, "config", None)
        register = getattr(sched, "register_to_config", None)
        if cfg is None or not callable(register):
            yield
            return
        saved = {
            "use_dynamic_shifting": cfg.get("use_dynamic_shifting", False),
            "shift": cfg.get("shift", 1.0),
            "shift_terminal": cfg.get("shift_terminal", None),
        }
        register(use_dynamic_shifting = False, shift = 1.0, shift_terminal = None)
        try:
            yield
        finally:
            register(**saved)

    return _ctx()


def _build_from_config(
    model_cls: Any,
    config: dict[str, Any],
    state: dict[str, Any],
    rename: dict[str, str],
    torch_dtype: Any,
    remove_suffixes: tuple[str, ...] = (),
) -> Any:
    from accelerate import init_empty_weights

    state = _apply_rename(_to_plain_dtype(state, torch_dtype), rename)
    for key in [k for k in state if k.endswith(remove_suffixes)] if remove_suffixes else []:
        state.pop(key)
    with init_empty_weights():
        model = model_cls.from_config(config)
    model.load_state_dict(state, strict = True, assign = True)
    return model.to(torch_dtype)


def ltx23_is_dit_key(key: str) -> bool:
    """Whether a combined-checkpoint key belongs to the DiT (not the connectors, VAEs or vocoder)."""
    return _checkpoint_group(key)[0] == "dit"


def ltx23_is_dit_or_connector_key(key: str) -> bool:
    """The keys a single-file DiT plan prices: the connectors stay in, as for a bf16 file."""
    return _checkpoint_group(key)[0] in ("dit", "connectors")


def _ltx23_pre_convert(state: dict[str, Any]) -> dict[str, Any]:
    """Bare DiT keys with the 2.3-only prefixes renamed, as ``load_ltx23_transformer`` does before the converter."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        bare = key[len(_DIT_PREFIX) :] if key.startswith(_DIT_PREFIX) else key
        for old, new in _TRANSFORMER_PRERENAME:
            if bare.startswith(old):
                bare = new + bare[len(old) :]
                break
        out[bare] = value
    return out


def load_ltx23_comfy_transformer(
    checkpoint_path: Path | str,
    scan: Any,
    *,
    base_repo: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    int8_backend: Optional[str] = None,
    fp8_backend: Optional[str] = None,
    family: Optional[str] = None,
    target: Any = None,
    logger: Any = None,
) -> Any:
    """The LTX-2.3 DiT of a ComfyUI-quantized single file, for ``load_ltx23_pipeline(transformer_override=...)``.

    Reads only the DiT keys (the assembly reads the connectors, VAEs and vocoder from the same file), applies the
    2.3 pre-rename and the stock converter, and keeps int8 / fp8 codes where ``load_comfy_quant_transformer`` can."""
    from diffusers import LTX2VideoTransformer3DModel

    from .diffusion_comfy_quant import load_comfy_quant_transformer

    return load_comfy_quant_transformer(
        LTX2VideoTransformer3DModel,
        str(checkpoint_path),
        scan,
        {
            "torch_dtype": torch_dtype,
            "config": base_repo,
            "subfolder": "transformer",
            "token": hf_token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            **LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES,
        },
        int8_backend = int8_backend,
        fp8_backend = fp8_backend,
        family = family,
        target = target,
        logger = logger,
        keep_key = ltx23_is_dit_key,
        pre_convert = _ltx23_pre_convert,
    )


def load_ltx23_transformer(
    dit_state: dict[str, Any],
    *,
    base_repo: str,
    torch_dtype: Any,
    is_gguf: bool,
    hf_token: Optional[str],
    local_files_only: bool = False,
    device: Optional[Any] = None,
) -> Any:
    import diffusers
    from diffusers import LTX2VideoTransformer3DModel

    # Pre-rename 2.3-only keys the converter does not know before from_single_file.
    for old, new in _TRANSFORMER_PRERENAME:
        for key in [k for k in dit_state if k.startswith(old)]:
            dit_state[new + key[len(old) :]] = dit_state.pop(key)
    kwargs: dict[str, Any] = {
        "config": base_repo,
        "subfolder": "transformer",
        "torch_dtype": torch_dtype,
        "token": hf_token,
        "local_files_only": local_files_only,
        **LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES,
    }
    if is_gguf:
        kwargs["quantization_config"] = diffusers.GGUFQuantizationConfig(compute_dtype = torch_dtype)
    elif device is not None:
        # Else the meta load copies every tensor back to the host.
        kwargs["device"] = device
    return LTX2VideoTransformer3DModel.from_single_file(dit_state, **kwargs)


def load_ltx23_connectors(
    connector_state: dict[str, Any],
    *,
    variant: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> Any:
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    # Transformer-only checkpoints lack per-modality text projections; fetch from the companion.
    if not any(k.startswith("text_embedding_projection") for k in connector_state):
        connector_state = dict(connector_state)
        connector_state.update(
            _load_extras_file(_EXTRAS_TEXT_PROJ.format(variant = variant), hf_token, local_files_only)
        )
    return _build_from_config(
        LTX2TextConnectors,
        _CONNECTORS_CONFIG,
        connector_state,
        _CONNECTORS_RENAME,
        torch_dtype,
    )


def load_ltx23_vae(
    vae_state: dict[str, Any],
    *,
    variant: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> Any:
    from diffusers import AutoencoderKLLTX2Video
    if not vae_state:
        vae_state = _load_extras_file(
            _EXTRAS_VIDEO_VAE.format(variant = variant), hf_token, local_files_only
        )
    return _build_from_config(
        AutoencoderKLLTX2Video,
        _VIDEO_VAE_CONFIG,
        vae_state,
        _VIDEO_VAE_RENAME,
        torch_dtype,
        remove_suffixes = _VIDEO_VAE_REMOVE_SUFFIXES,
    )


def load_ltx23_audio_vae_and_vocoder(
    audio_vae_state: dict[str, Any],
    vocoder_state: dict[str, Any],
    *,
    variant: str,
    torch_dtype: Any,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    from diffusers import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    if not audio_vae_state or not vocoder_state:
        combined = _load_extras_file(
            _EXTRAS_AUDIO_VAE.format(variant = variant), hf_token, local_files_only
        )
        audio_vae_state = {
            k[len("audio_vae.") :]: v for k, v in combined.items() if k.startswith("audio_vae.")
        }
        vocoder_state = {
            k[len("vocoder.") :]: v for k, v in combined.items() if k.startswith("vocoder.")
        }
    audio_vae = _build_from_config(
        AutoencoderKLLTX2Audio,
        _AUDIO_VAE_CONFIG,
        audio_vae_state,
        _AUDIO_VAE_RENAME,
        torch_dtype,
    )
    # 2.3 vocoder is composite; keys line up module-for-module after the renames.
    vocoder_state = _apply_rename(_to_plain_dtype(vocoder_state, torch_dtype), _VOCODER_RENAME)
    for key in [k for k in vocoder_state if ".ups." in k]:
        vocoder_state[key.replace(".ups.", ".upsamplers.")] = vocoder_state.pop(key)
    from accelerate import init_empty_weights

    with init_empty_weights():
        vocoder = LTX2VocoderWithBWE.from_config(_VOCODER_CONFIG)
    vocoder.load_state_dict(vocoder_state, strict = True, assign = True)
    return audio_vae, vocoder.to(torch_dtype)


PREFETCH_ENV = "UNSLOTH_VIDEO_PREFETCH"
_PREFETCH_THREADS = 8
_PREFETCH_CHUNK_BYTES = 16 << 20


def prefetch_enabled() -> bool:
    import os
    return (os.environ.get(PREFETCH_ENV) or "").strip().lower() not in _SWITCH_OFF


def _uncached_bytes(
    path: str,
    stride: int = 64 << 20,
    window: int = 1 << 20,
) -> int:
    """Estimated bytes of ``path`` not in the page cache (sampled Linux ``mincore``); 0 when it cannot tell."""
    import ctypes
    import mmap
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return 0
    size = os.path.getsize(path)
    if size == 0:
        return 0
    page = mmap.PAGESIZE
    window = max(page, window - window % page)
    stride = max(window, stride - stride % page)
    with open(path, "rb") as handle:
        mapped = mmap.mmap(handle.fileno(), size, access = mmap.ACCESS_COPY)
    try:
        anchor = ctypes.c_char.from_buffer(mapped)
        try:
            base = ctypes.addressof(anchor)
            libc = ctypes.CDLL(None, use_errno = True)
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            vector = (ctypes.c_ubyte * (window // page))()
            sampled = missing = 0
            for offset in range(0, size, stride):
                length = min(window, size - offset)
                pages = (length + page - 1) // page
                if libc.mincore(base + offset, ctypes.c_size_t(length), vector) != 0:
                    return 0
                missing += bytes(vector)[:pages].count(0)
                sampled += pages
            return int(size * missing / sampled) if sampled else 0
        finally:
            del anchor
    finally:
        mapped.close()


def _text_encoder_files(base_repo: str, cache_dir: Optional[str]) -> list[str]:
    """The cached text encoder shards of ``base_repo`` (cache only, never a download)."""
    import json
    import os

    try:
        from huggingface_hub import try_to_load_from_cache

        index = try_to_load_from_cache(
            base_repo, "text_encoder/model.safetensors.index.json", cache_dir = cache_dir
        )
        if not isinstance(index, str):
            return []
        with open(index, "r", encoding = "utf-8") as handle:
            shards = sorted(set(json.load(handle).get("weight_map", {}).values()))
    except Exception:  # noqa: BLE001 - not cached (the load downloads it) or an unreadable cache
        return []
    folder = os.path.dirname(index)
    return [p for p in (os.path.join(folder, name) for name in shards) if os.path.isfile(p)]


def start_prefetch(paths: list[str]) -> Optional[Any]:
    """Warm the uncached parts of ``paths`` in the page cache on worker threads; a stop Event, or None when nothing
    needs it or host RAM is too tight. Cold B200 cache: the Gemma3 shards otherwise read at ~1.7 GB/s after the checkpoint."""
    import os
    import threading

    try:
        uncached = [(path, _uncached_bytes(path)) for path in paths]
    except Exception:  # noqa: BLE001 - residency unknown: leave the reads to the loader
        return None
    pending = [path for path, missing in uncached if missing > 0]
    need = sum(missing for _path, missing in uncached)
    if not pending or need < 1 << 30:
        return None
    try:
        from .diffusion_memory import _available_system_memory_mib
        available = _available_system_memory_mib()
    except Exception:  # noqa: BLE001
        available = None
    if available is None or need > (int(available) << 20) // 2:
        return None
    stop = threading.Event()
    jobs = []
    for path in pending:
        size = os.path.getsize(path)
        jobs += [(path, offset) for offset in range(0, size, _PREFETCH_CHUNK_BYTES)]
    lock = threading.Lock()
    queue = iter(jobs)

    def _worker() -> None:
        buffer = bytearray(_PREFETCH_CHUNK_BYTES)
        view = memoryview(buffer)
        handles: dict[str, Any] = {}
        try:
            while not stop.is_set():
                with lock:
                    job = next(queue, None)
                if job is None:
                    return
                path, offset = job
                handle = handles.get(path)
                if handle is None:
                    handle = handles[path] = open(path, "rb", buffering = 0)
                handle.seek(offset)
                handle.readinto(view)
        except Exception:  # noqa: BLE001 - a prefetch is only a hint
            pass
        finally:
            for handle in handles.values():
                handle.close()

    for index in range(_PREFETCH_THREADS):
        threading.Thread(target = _worker, name = f"unsloth-prefetch-{index}", daemon = True).start()
    logger.info(
        "video.ltx23_prefetch: warming %.1f GiB of %d text encoder file(s)",
        need / 2**30,
        len(pending),
    )
    return stop


def load_ltx23_pipeline(
    checkpoint_path: Path | str,
    *,
    base_repo: str,
    text_encoder: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Full LTX-2.3 pipeline (``_assemble_ltx23_pipeline``), warming the text encoder files beside the checkpoint read."""
    prefetch = None
    if text_encoder is None and prefetch_enabled():
        prefetch = start_prefetch(_text_encoder_files(base_repo, _live_cache_dir()))
    try:
        return _assemble_ltx23_pipeline(
            checkpoint_path, base_repo = base_repo, text_encoder = text_encoder, **kwargs
        )
    finally:
        if prefetch is not None:
            prefetch.set()


def _assemble_ltx23_pipeline(
    checkpoint_path: Path | str,
    *,
    base_repo: str,
    torch_dtype: Any,
    is_gguf: bool,
    hf_token: Optional[str] = None,
    text_encoder: Optional[Any] = None,
    local_files_only: bool = False,
    transformer_override: Optional[Any] = None,
    device: Optional[Any] = None,
    text_encoder_device: Optional[Any] = None,
) -> Any:
    """Full LTX-2.3 pipeline from a single-file/GGUF checkpoint. Assembled per-component
    (constructor, not from_pretrained) because the base model_index pins LTX2Vocoder while 2.3
    needs LTX2VocoderWithBWE, which the type gate would reject.

    ``text_encoder`` supplies an already-built encoder (the caller's pre-cast fp8 Gemma3);
    None builds it dense from the base repo. Because the assembly bypasses
    ``from_pretrained``, this is the only way an fp8 request reaches the 2.3 path.

    ``local_files_only`` is a load nobody asked for. Because the assembly bypasses
    ``from_pretrained`` it also bypasses the caller's guarded ``pipe_kwargs``, and it is handed the
    base REPO ID rather than a staged snapshot (the 2.3 snapshot lacks the base VAEs, so
    ``_base_local_dir`` is deliberately None here), so without the flag the base config, the
    scheduler, the tokenizer, the dense Gemma3 encoder and the companion VAE/vocoder artifacts are
    all fetched by a load that promised to fetch nothing.

    ``device`` / ``text_encoder_device`` (resident plans only) read the checkpoint / build the Gemma3 encoder there."""
    import transformers

    from .ltx2_import_compat import ensure_ltx2_pipelines_importable

    ensure_ltx2_pipelines_importable(logger)
    from diffusers import LTX2Pipeline
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint

    variant = checkpoint_variant(checkpoint_path)
    logger.info(
        "video.ltx23_assembly: variant=%s gguf=%s extras=%s",
        variant,
        is_gguf,
        LTX23_EXTRAS_REPO,
    )
    state = None
    on_device = False
    if device is not None and not is_gguf:
        import time

        began = time.perf_counter()
        if transformer_override is not None:
            state = _load_checkpoint_without_dit_to_device(checkpoint_path, device)
        else:
            state = load_checkpoint_to_device(checkpoint_path, device)
        on_device = state is not None
        if on_device:
            logger.info(
                "video.ltx23_direct_load: read %.1f GiB onto %s in %.1f s",
                sum(t.numel() * t.element_size() for t in state.values()) / 2**30,
                device,
                time.perf_counter() - began,
            )
    if state is None and transformer_override is not None and not is_gguf:
        state = _load_checkpoint_without_dit(checkpoint_path)
    if state is None:
        state = load_single_file_checkpoint(str(checkpoint_path))
    groups = _split_checkpoint(state)
    del state

    # Lightricks fp8 files are SCALED float8; casting without scales corrupts weights, so refuse.
    if any(k.endswith((".weight_scale", ".input_scale")) for k in groups["dit"]):
        raise ValueError(
            "This LTX checkpoint stores scaled fp8 weights, which this loader does "
            "not dequantize yet. Use the GGUF quants from unsloth/LTX-2.3-GGUF "
            "instead (Q8_0 for the highest fidelity) or the official bf16 checkpoint."
        )

    if transformer_override is not None:
        transformer = transformer_override
        groups.pop("dit", None)
    else:
        transformer = load_ltx23_transformer(
            groups["dit"],
            base_repo = base_repo,
            torch_dtype = torch_dtype,
            is_gguf = is_gguf,
            hf_token = hf_token,
            local_files_only = local_files_only,
            device = device if on_device else None,
        )
    connectors = load_ltx23_connectors(
        groups["connectors"],
        variant = variant,
        torch_dtype = torch_dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
    )
    vae = load_ltx23_vae(
        groups["vae"],
        variant = variant,
        torch_dtype = torch_dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
    )
    audio_vae, vocoder = load_ltx23_audio_vae_and_vocoder(
        groups["audio_vae"],
        groups["vocoder"],
        variant = variant,
        torch_dtype = torch_dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
    )

    # Pin to the LIVE hub root (a setting), not the import-time constant, to match the locality gate.
    cache_dir = _live_cache_dir()
    index = LTX2Pipeline.load_config(
        base_repo, token = hf_token, local_files_only = local_files_only, cache_dir = cache_dir
    )

    def _sub(name: str, **extra: Any) -> Any:
        library, class_name = index[name]
        module = transformers if library == "transformers" else __import__("diffusers")
        return getattr(module, class_name).from_pretrained(
            base_repo,
            subfolder = name,
            token = hf_token,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
            **extra,
        )

    scheduler = _sub("scheduler")
    tokenizer = _sub("tokenizer")
    if text_encoder is None:
        encoder_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if text_encoder_device is not None:
            encoder_kwargs["device_map"] = {"": text_encoder_device}
        text_encoder = _sub("text_encoder", **encoder_kwargs)

    return LTX2Pipeline(
        scheduler = scheduler,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        connectors = connectors,
        transformer = transformer,
        vae = vae,
        audio_vae = audio_vae,
        vocoder = vocoder,
    )
