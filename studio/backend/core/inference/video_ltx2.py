# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3 pipeline is built via the constructor; from_pretrained's type gate rejects the 2.3 vocoder."""

from __future__ import annotations

import contextlib
import contextvars
import os
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


# from_single_file config overrides on top of the base 2.0 transformer config.
LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES: dict[str, Any] = {
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "use_prompt_embeddings": False,
    "perturbed_attn": True,
}

# Keys the 2.0-era converter doesn't know; renamed before from_single_file. Audio prefix first.
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
    # Decoder (2.3 adds up_blocks.7/8: a 4th decoder stage)
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

# Same config as LTX-2.0 (upstream's comment); the weights are still 2.3-specific.
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


# Keyed by file identity + stamps: an edited or replaced file is re-read; failures never kept.
_LTX23_VERDICTS: dict[tuple, bool] = {}
_LTX23_VERDICTS_LOCK = threading.Lock()
_LTX23_VERDICTS_MAX = 256


def _checkpoint_stamp(checkpoint_path: Path | str) -> Optional[tuple]:
    try:
        path = os.fspath(checkpoint_path)
        st = os.stat(path)
        return (
            os.path.realpath(path),
            st.st_dev,
            st.st_ino,
            st.st_size,
            st.st_mtime_ns,
            st.st_ctime_ns,
        )
    except (OSError, TypeError, ValueError):
        return None


def is_ltx23_checkpoint(checkpoint_path: Path | str) -> bool:
    """True for 9-row LTX-2.3 modulation tables (2.0 has 6 rows); an unreadable header gives False."""
    stamp = _checkpoint_stamp(checkpoint_path)
    if stamp is not None:
        with _LTX23_VERDICTS_LOCK:
            verdict = _LTX23_VERDICTS.get(stamp)
        if verdict is not None:
            return verdict
    try:
        header = read_checkpoint_header(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("video.ltx2_header_probe_failed: %s", exc)
        return False
    verdict = False
    for name, shape in header.items():
        if name.endswith("transformer_blocks.0.scale_shift_table"):
            verdict = 9 in shape
            break
    if stamp is not None:
        with _LTX23_VERDICTS_LOCK:
            if len(_LTX23_VERDICTS) >= _LTX23_VERDICTS_MAX:
                _LTX23_VERDICTS.clear()
            _LTX23_VERDICTS[stamp] = verdict
    return verdict


def assert_not_ltx25_checkpoint(checkpoint_path: Path | str) -> None:
    """Catch a renamed LTX-2.5 file by header: ``is_ltx23_checkpoint`` counts modulation rows, and 2.5 keeps 2.3's nine."""
    from .diffusion_content import is_ltx25_checkpoint
    from .video_families import LTX2_FAMILY, wrong_ltx_family_message

    if is_ltx25_checkpoint(str(checkpoint_path)):
        raise ValueError(
            wrong_ltx_family_message(
                "2.5", LTX2_FAMILY, f"'{Path(str(checkpoint_path)).name}' (by its tensors)"
            )
        )


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
    """Splits a combined LTX file (vae.*, audio_vae.*, vocoder.* keys) or a bare-DiT GGUF by key prefix."""
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
    from .host_pinning import host_pinning_allowed

    # Pageable on an integrated GPU: the copy then completes before copy_ returns, so slot reuse stays safe.
    pin = host_pinning_allowed(device)
    staging = [torch.empty(chunk_bytes, dtype = torch.uint8, pin_memory = pin) for _ in range(buffers)]
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
            # A failed read leaves uploads queued from the staging ring.
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
        # the switch's locality gate cleared these three artifacts by name
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
    """Distilled or dev from the checkpoint content (128-value bias fingerprint); None if
    unreadable, missing or a finetune, and callers then fall back to the name"""
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
    """Companion files a 2.3 checkpoint loads, picked by the assembly's variant rule so the plan matches."""
    variant = checkpoint_variant(checkpoint_path)
    return tuple(
        template.format(variant = variant)
        for template in (_EXTRAS_TEXT_PROJ, _EXTRAS_VIDEO_VAE, _EXTRAS_AUDIO_VAE)
    )


# Upstream ltx_core DISTILLED_SIGMA_VALUES: fixed 8-step curve the distilled DiT expects; pass verbatim, base scheduler spacing never lands near it.
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
        # Pre-#14447: audio follows the video CFG.
        kwargs["audio_guidance_scale"] = float(guidance if guidance is not None else 1.0)
    return kwargs


# Hosted 2.3 checkpoints were baked from this file only; distilled-1.1 and dev DiTs may not take them.
LTX23_PREQUANT_BASE = "Lightricks/LTX-2.3"
LTX23_PREQUANT_SOURCE_FILES = frozenset({"ltx-2.3-22b-distilled.safetensors"})
# Resident size of LTX-2.3-FP8.pt (19,057,628,489 bytes); companions are priced separately.
LTX23_PREQUANT_RESIDENT_GB = 19.06


# The hosted DiT REPLACES the file's own, so the file must be the official one (a same-named fine-tune would be swapped silently).
# The official repo and the unsloth repo carrying its byte-identical files (diffusion_families._VIDEO_MIRROR_PAIRS): a
# pick of Lightricks/LTX-2.3 is FETCHED from there, so the file it lands on sits under that repo's cache dir.
LTX23_PREQUANT_SOURCE_REPOS = frozenset({"lightricks/ltx-2.3", "unsloth/ltx-2.3-fp8"})
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
            # Changed while being read: neither stored nor trusted.
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
    cached = None
    from .diffusion_families import canonical_base, mirror_repo

    upstream = canonical_base(str(repo_id).strip())
    for candidate in dict.fromkeys(
        (str(repo_id).strip(), upstream, mirror_repo(upstream) or upstream)
    ):
        cached = _ltx23_hub_cached_file(candidate, str(checkpoint_filename))
        if cached is not None:
            break
    return True if cached is None else ltx23_source_file_verified(cached)


class _LTX23PrequantConfig:
    """Transformer "class" for ``load_prequantized_transformer``: the 2.0 config + 2.3 overrides, since the
    single-file 2.3 repo has no ``transformer/``."""

    def __init__(self, config_repo: str):
        self.config_repo = config_repo

    def load_config(self, _base: str, **kwargs: Any) -> dict[str, Any]:
        from diffusers import LTX2VideoTransformer3DModel

        from .diffusion_families import pinned_revision_kwargs

        # Pinned on the repo actually read, which need not be the caller's ``_base``.
        kwargs.pop("revision", None)
        config = dict(
            LTX2VideoTransformer3DModel.load_config(
                self.config_repo, **kwargs, **pinned_revision_kwargs(self.config_repo)
            )
        )
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


def _stg_adapted(inner: Any, guard: Any, transformer: Any, torch: Any) -> Any:
    def adapted(*args: Any, **kwargs: Any) -> Any:
        flag = kwargs.get("all_perturbed")
        if isinstance(flag, torch.Tensor):
            kwargs["all_perturbed"] = bool(flag)
        try:
            return inner(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- reraised unless the recompile limit was hit
            if guard is None or guard.error is not None or not _recompile_limit_hit(exc):
                raise
            # Past the recompile limit fullgraph raises unclassified by the guard: drop this DiT to eager.
            guard.fail(exc, transformer)
            exc.__traceback__ = None
            return inner(*args, **kwargs)

    adapted._unsloth_stg_adapter = True
    if guard is not None:
        adapted._unsloth_compile_guard = guard
    # A compile moved below an offload hook rebuilds the block's callable: re-apply the adapter around it.
    adapted._unsloth_rewrap = lambda new_inner: _stg_adapted(new_inner, guard, transformer, torch)
    return adapted


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
        module._compiled_call_impl = _stg_adapted(
            inner, getattr(inner, "_unsloth_compile_guard", None), transformer, torch
        )
        count += 1
    return count


# LTX-2.x text masks: the connectors replace padding with learned registers and return an all-ones mask, which the DiT
# turns into an all-zero additive bias. That bias is a no-op, but it sends every text cross-attention to the
# mem-efficient kernel, and Inductor materialises it at [batch, heads, video tokens, text tokens] (0.9 GB per call at
# 1280x704x121). Dropping an all-ones mask keeps the maths and lets SDPA pick flash. =0 keeps the mask.
LTX2_FULL_MASK_ENV = "UNSLOTH_LTX2_DROP_FULL_TEXT_MASK"
_LTX2_TRANSFORMER_CLS = "LTX2VideoTransformer3DModel"
_LTX2_MASK_KEYS = ("encoder_attention_mask", "audio_encoder_attention_mask")
_LTX2_MASK_MEMO = "_unsloth_full_mask_memo"
_LTX2_MASK_HOOK = "_unsloth_full_mask_hook"
_LTX2_MASK_DROPS = "_unsloth_full_mask_drops"


def ltx2_full_mask_drop_disabled() -> bool:
    raw = (os.environ.get(LTX2_FULL_MASK_ENV) or "").strip().lower()
    return raw in ("0", "off", "false", "no")


def _ltx2_mask_all_ones(module: Any, mask: Any) -> bool:
    """Whether a 2-D multiplicative text mask is 1 everywhere, memoised per tensor and version (one host read per
    render: the pipeline passes the same mask to every step)."""
    import torch

    version = "inference" if mask.is_inference() else mask._version
    memo = module.__dict__.setdefault(_LTX2_MASK_MEMO, [])
    for held, held_version, verdict in memo:
        if held is mask and held_version == version:
            return verdict
    verdict = bool(mask.numel() > 0 and bool(torch.all(mask == 1).item()))
    memo.insert(0, (mask, version, verdict))
    del memo[4:]
    return verdict


def _ltx2_full_mask_pre_hook(module: Any, args: Any, kwargs: dict) -> Any:
    """Replace an all-ones text mask by None; any other mask, or a positional one, is passed through unchanged."""
    try:
        import torch

        dropped = 0
        for key in _LTX2_MASK_KEYS:
            mask = kwargs.get(key)
            # 2-D only: that is the multiplicative mask the DiT turns into a bias; 3-D+ is already a bias.
            if torch.is_tensor(mask) and mask.dim() == 2 and _ltx2_mask_all_ones(module, mask):
                kwargs[key] = None
                dropped += 1
        if dropped:
            setattr(module, _LTX2_MASK_DROPS, getattr(module, _LTX2_MASK_DROPS, 0) + dropped)
        return args, kwargs
    except Exception:  # noqa: BLE001 - optimisation only; never break the forward
        return None


def install_ltx2_full_mask_drop(pipe: Any, logger: Any = None) -> int:
    """Hook every LTX-2.x DiT of ``pipe`` once; returns the DiTs hooked (0 when switched off or not LTX-2.x)."""
    if ltx2_full_mask_drop_disabled():
        return 0
    try:
        from .diffusion_autoopt.targets import denoiser_dits
        dits = denoiser_dits(pipe)
    except Exception:  # noqa: BLE001
        dits = [getattr(pipe, "transformer", None)]
    count = 0
    for dit in dits:
        if dit is None or type(dit).__name__ != _LTX2_TRANSFORMER_CLS:
            continue
        if getattr(dit, _LTX2_MASK_HOOK, None) is None:
            try:
                handle = dit.register_forward_pre_hook(_ltx2_full_mask_pre_hook, with_kwargs = True)
                setattr(dit, _LTX2_MASK_HOOK, handle)
            except Exception:  # noqa: BLE001 - optimisation only
                continue
        count += 1
    if count and logger is not None:
        logger.info(
            "video.ltx2: all-ones text masks dropped, text cross-attention runs without a bias"
        )
    return count


# LTX-2.3 / 2.5 connectors: diffusers keeps the masked norm and both scaled copies ([B, 1024, 49 * 3840], 368 MiB each
# in bf16) alive while the 1.44 GiB video projection onloads; the lean forward builds each input when its projection
# runs and frees it after (same ops, bit-identical). =0 keeps the diffusers forward.
LTX2_LEAN_CONNECTOR_ENV = "UNSLOTH_LTX2_LEAN_CONNECTOR_FEATURES"
_LTX2_CONNECTORS_CLS = "LTX2TextConnectors"
_LTX2_LEAN_CONNECTOR_ATTR = "_unsloth_lean_connector_features"
# sha256 of ``inspect.getsource(LTX2TextConnectors.forward)`` this copy matches (diffusers 0.42.1); any other source
# keeps the diffusers forward.
_LTX2_CONNECTORS_FORWARD_SHA256 = frozenset(
    {"9764cf89cec9603d1190a95d5e3daa95cde08eb567ce19e2eee31644ed15f6dd"}
)


def ltx2_lean_connector_disabled() -> bool:
    raw = (os.environ.get(LTX2_LEAN_CONNECTOR_ENV) or "").strip().lower()
    return raw in ("0", "off", "false", "no")


def _ltx2_connectors_forward_matches(cls: Any) -> bool:
    try:
        import hashlib
        import inspect
        source = inspect.getsource(cls.forward)
    except Exception:  # noqa: BLE001 - unreadable source: keep the diffusers forward
        return False
    return hashlib.sha256(source.encode()).hexdigest() in _LTX2_CONNECTORS_FORWARD_SHA256


def _ltx2_lean_connectors_forward(
    self: Any,
    text_encoder_hidden_states: Any,
    attention_mask: Any,
    padding_side: str = "left",
    scale_factor: int = 8,
) -> Any:
    """``LTX2TextConnectors.forward`` (diffusers 0.42.1) with the per-modality inputs built one at a time."""
    import math

    import torch
    from diffusers.pipelines.ltx2.connectors import per_token_rms_norm

    if not self.config.per_modality_projections:
        return type(self).forward(
            self,
            text_encoder_hidden_states,
            attention_mask,
            padding_side = padding_side,
            scale_factor = scale_factor,
        )
    if text_encoder_hidden_states.ndim == 3:
        text_encoder_hidden_states = text_encoder_hidden_states.unflatten(
            2, (self.config.caption_channels, -1)
        )
    keep = attention_mask.bool().unsqueeze(-1)

    def scaled_input(hidden_dim: int) -> Any:
        # torch.where(keep, norm, 0) * scale, in place on the norm this call owns
        features = per_token_rms_norm(text_encoder_hidden_states).flatten(2, 3)
        features.masked_fill_(keep.logical_not(), 0.0)
        return features.mul_(math.sqrt(hidden_dim / self.config.caption_channels))

    features = scaled_input(self.config.video_hidden_dim)
    video_text_emb_proj = self.video_text_proj_in(features)
    del features
    features = scaled_input(self.config.audio_hidden_dim)
    audio_text_emb_proj = self.audio_text_proj_in(features)
    del features
    if not torch.compiler.is_compiling():
        # eager-only engagement count: a traced attribute write would add a Dynamo guard and a recompile
        self.__dict__["_unsloth_lean_connector_runs"] = (
            self.__dict__.get("_unsloth_lean_connector_runs", 0) + 1
        )

    text_dtype = video_text_emb_proj.dtype
    attention_mask = (attention_mask.to(torch.int64) - 1).to(text_dtype)
    attention_mask = attention_mask.reshape(
        attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
    )
    add_attn_mask = attention_mask * torch.finfo(text_dtype).max

    video_text_embedding, video_attn_mask = self.video_connector(video_text_emb_proj, add_attn_mask)

    binary_attn_mask = (video_attn_mask < 1e-6).to(torch.int64)
    binary_attn_mask = binary_attn_mask.reshape(
        video_text_embedding.shape[0], video_text_embedding.shape[1], 1
    )
    video_text_embedding = video_text_embedding * binary_attn_mask

    audio_text_embedding, _ = self.audio_connector(audio_text_emb_proj, add_attn_mask)

    return video_text_embedding, audio_text_embedding, binary_attn_mask.squeeze(-1)


def install_ltx2_lean_connector_features(pipe: Any, logger: Any = None) -> bool:
    """Give ``pipe.connectors`` the lean forward, before any offload hook wraps it. False (diffusers forward kept)
    when switched off, not per-modality connectors, a diffusers forward this copy does not match, or a forward
    already wrapped on the instance (an offload hook would be bypassed)."""
    if ltx2_lean_connector_disabled():
        return False
    connectors = getattr(pipe, "connectors", None)
    if connectors is None or type(connectors).__name__ != _LTX2_CONNECTORS_CLS:
        return False
    if getattr(connectors, _LTX2_LEAN_CONNECTOR_ATTR, False):
        return True
    try:
        if not bool(getattr(connectors.config, "per_modality_projections", False)):
            return False
        if "forward" in connectors.__dict__ or not _ltx2_connectors_forward_matches(
            type(connectors)
        ):
            return False
        import types

        connectors.forward = types.MethodType(_ltx2_lean_connectors_forward, connectors)
        setattr(connectors, _LTX2_LEAN_CONNECTOR_ATTR, True)
    except Exception:  # noqa: BLE001 - memory lever only; the diffusers forward stays
        return False
    if logger is not None:
        logger.info(
            "video.ltx2: connector inputs built one at a time (two fewer prompt-sized copies held)"
        )
    return True


# Streamed tiers: text-to-video never runs the LTX-2 VAE encoder (612 MiB on LTX-2.5), so it waits on the host and
# joins the card only for ``vae.encode``. =0 keeps it on the card.
LTX2_PARK_VAE_ENCODER_ENV = "UNSLOTH_LTX2_PARK_VAE_ENCODER"
_LTX2_VAE_CLS = "AutoencoderKLLTX2Video"
_LTX2_ENCODER_HOME = "_unsloth_encoder_home"
_LTX2_ENCODE_WRAPPED = "_unsloth_parked_encode"
# the tiers whose decoders stay resident on the card (not hooked per call)
_LTX2_PARK_TIERS = ("group", "streaming")


def ltx2_vae_encoder_parks(vae: Any, offload_policy: Any) -> bool:
    """Whether ``vae``'s encoder waits on the host under ``offload_policy``."""
    if (os.environ.get(LTX2_PARK_VAE_ENCODER_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return False
    return (
        str(offload_policy) in _LTX2_PARK_TIERS
        and type(vae).__name__ == _LTX2_VAE_CLS
        and getattr(vae, "encoder", None) is not None
    )


def _module_device(module: Any) -> Any:
    first = next(iter(module.parameters()), None)
    return None if first is None else first.device


def _module_devices(module: Any) -> set:
    """Every device ``module``'s submodules hold parameters on: a move that failed partway leaves more than one."""
    devices = set()
    for sub in module.modules():
        own = next(iter(sub.parameters(recurse = False)), None)
        if own is None:
            own = next(iter(sub.buffers(recurse = False)), None)
        if own is not None:
            device = _module_device(sub)
            devices.add(own.device if device is None else device)
    return devices


def park_ltx2_vae_encoder(
    pipe: Any,
    offload_policy: Any,
    logger: Any = None,
) -> int:
    """Move the LTX-2 VAE encoder to the host and give ``vae.encode`` a wrapper that brings it back for the call.
    Idempotent (a later ``vae.to(device)`` is undone on the next call); bytes moved off the card, else 0."""
    vae = getattr(pipe, "vae", None)
    if vae is None or not ltx2_vae_encoder_parks(vae, offload_policy):
        return 0
    try:
        import torch

        encoder = vae.encoder
        home = _module_device(getattr(vae, "decoder", None) or vae)
        if home is None or home.type == "cpu" or getattr(vae, "_hf_hook", None) is not None:
            return 0
        if not vae.__dict__.get(_LTX2_ENCODE_WRAPPED, False):
            inner = vae.encode

            def encode(*args: Any, **kwargs: Any) -> Any:
                target = vae.__dict__.get(_LTX2_ENCODER_HOME)
                devices = _module_devices(encoder)
                # parked (or split by a failed move): onto the card for the call, back to the host after
                moved = target is not None and bool(devices) and devices != {target}
                try:
                    if moved:
                        encoder.to(target)
                    return inner(*args, **kwargs)
                finally:
                    if moved:
                        # also after a partial onload: the whole encoder goes back to the host
                        encoder.to("cpu")

            vae.encode = encode
            vae.__dict__[_LTX2_ENCODE_WRAPPED] = True
        vae.__dict__[_LTX2_ENCODER_HOME] = home
        if _module_devices(encoder) == {torch.device("cpu")}:
            return 0
        moved = sum(
            int(t.numel()) * int(t.element_size())
            for t in list(encoder.parameters()) + list(encoder.buffers())
        )
        encoder.to("cpu")
    except Exception:  # noqa: BLE001 -- memory lever only: the encoder stays where it was
        return 0
    if moved and logger is not None:
        logger.info(
            "video.ltx2: VAE encoder (%d MiB) waits on the host; it joins the card only to encode an image",
            moved >> 20,
        )
    return moved


def unpark_ltx2_vae_encoder(
    pipe: Any,
    offload_policy: Any,
    logger: Any = None,
) -> int:
    """For an image-conditioned request: bring a parked encoder back to the card for the whole request (it encodes in
    both stages, so per-call moves would copy it four times), and return its MiB for the floor; 0 when it does not
    park. The next unconditioned request parks it again."""
    vae = getattr(pipe, "vae", None)
    if vae is None or not ltx2_vae_encoder_parks(vae, offload_policy):
        return 0
    encoder = vae.encoder
    mib = (
        sum(
            int(t.numel()) * int(t.element_size())
            for t in list(encoder.parameters()) + list(encoder.buffers())
        )
        >> 20
    )
    home = vae.__dict__.get(_LTX2_ENCODER_HOME)
    if home is None or _module_devices(encoder) in (set(), {home}):
        return mib
    try:
        encoder.to(home)
    except Exception:  # noqa: BLE001 -- e.g. an OOM partway: the whole encoder back on the host, moved per call
        try:
            encoder.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        return mib
    if logger is not None:
        logger.info(
            "video.ltx2: VAE encoder (%d MiB) on the card for an image-conditioned request", mib
        )
    return mib


def ltx2_memory_levers(pipe: Any, offload_policy: Any) -> str:
    """The memory levers engaged on ``pipe`` ("" when none): part of the measured-peak key, so a peak recorded with a
    lever on never lowers the headroom of a load or request running with it off."""
    levers = []
    try:
        if getattr(getattr(pipe, "connectors", None), _LTX2_LEAN_CONNECTOR_ATTR, False):
            levers.append("lean_connectors")
        from .video_ltx25 import compact_conditioning_enabled, release_prompt_rows_enabled

        from .video_ltx25 import prompt_rows_release_supported, two_stage_enabled

        if (
            getattr(pipe, "_unsloth_ltx25_encoder", None) is not None
            and not bool(getattr(pipe, "_unsloth_ltx25_dev", False))
            and two_stage_enabled()
            and release_prompt_rows_enabled()
            and compact_conditioning_enabled()
            and prompt_rows_release_supported()
        ):
            # only the two-stage render releases them (CFG, which batches the rows, has its own key entry)
            levers.append("released_prompt_rows")
        vae = getattr(pipe, "vae", None)
        if vae is not None and ltx2_vae_encoder_parks(vae, offload_policy):
            levers.append("parked_vae_encoder")
    except Exception:  # noqa: BLE001
        pass
    return ",".join(levers)


def ltx23_verbatim_sigmas(pipe: Any) -> Any:
    """Disable scheduler sigma rewrites (dynamic shift, terminal stretch) so sigmas pass through intact."""
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
    nvfp4_backend: Optional[str] = None,
    mxfp8_backend: Optional[str] = None,
    fp8_storage: Optional[bool] = None,
    fp8_widen_chosen: bool = False,
    family: Optional[str] = None,
    target: Any = None,
    logger: Any = None,
) -> Any:
    """LTX-2.3 DiT from a ComfyUI-quantized file: DiT keys only; int8 / fp8 codes kept where supported"""
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
        nvfp4_backend = nvfp4_backend,
        mxfp8_backend = mxfp8_backend,
        fp8_storage = fp8_storage,
        fp8_widen_chosen = fp8_widen_chosen,
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
        # ``config`` is the BASE REPO, so the 2.0 transformer config is a hub read here.
        "local_files_only": local_files_only,
        **LTX_2_3_TRANSFORMER_CONFIG_OVERRIDES,
    }
    from .diffusion_families import pinned_revision_kwargs

    # The load plan's commit for that hub-id config (absent for an unpinned load).
    kwargs.update(pinned_revision_kwargs(base_repo))
    if is_gguf:
        from .diffusion_gguf_exact import install_exact_dequant
        from .diffusion_gguf_lowbit import install_lowbit_iq
        from .diffusion_gguf_mmap import install_gguf_mmap

        install_lowbit_iq()
        install_exact_dequant()
        install_gguf_mmap()
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
    vocoder = vocoder.to(torch_dtype)
    install_vocoder_fp32_forward(vocoder)
    return audio_vae, vocoder


# Lightricks' LTX-2.3+ VocoderWithBWE runs its whole forward in float32 on bf16 weights (autocast on CUDA, weights
# upcast for the call elsewhere): bf16 accumulation through BigVGAN's ~108 convs degrades mel / MRSTFT error 40-90%.
# diffusers' LTX2VocoderWithBWE has no upcast. "0" runs the stored dtype again.
VOCODER_FP32_ENV = "UNSLOTH_LTX_VOCODER_FP32"
_STORED_FILTER = "_unsloth_stored_filter"


def vocoder_fp32_enabled() -> bool:
    return (os.environ.get(VOCODER_FP32_ENV) or "").strip().lower() not in _SWITCH_OFF


# cuDNN's atomic transposed-conv algos make the waveform vary per process; CUDA only (MIOpen lacks the flag). "0" = off.
VOCODER_DETERMINISTIC_ENV = "UNSLOTH_LTX_VOCODER_DETERMINISTIC"


def vocoder_deterministic_enabled() -> bool:
    return (os.environ.get(VOCODER_DETERMINISTIC_ENV) or "").strip().lower() not in _SWITCH_OFF


# The flag is process-global, so overlapping scopes share it: the first entry saves it, the last exit restores it.
_DETERMINISTIC_LOCK = threading.Lock()
_deterministic_depth = 0
_deterministic_saved = False


@contextlib.contextmanager
def _deterministic_cudnn(mel_spec: Any) -> Iterator[None]:
    global _deterministic_depth, _deterministic_saved
    import torch

    from .rocm_bf16 import is_rocm_torch

    cudnn = torch.backends.cudnn
    if (
        getattr(getattr(mel_spec, "device", None), "type", None) != "cuda"
        or is_rocm_torch(torch)  # AMD SDK / Radeon wheels leave torch.version.hip unset
        or _is_compiling()
        or not vocoder_deterministic_enabled()
    ):
        yield
        return
    with _DETERMINISTIC_LOCK:
        if _deterministic_depth == 0:
            _deterministic_saved = bool(cudnn.deterministic)
            cudnn.deterministic = True
        _deterministic_depth += 1
    try:
        yield
    finally:
        with _DETERMINISTIC_LOCK:
            _deterministic_depth -= 1
            if _deterministic_depth == 0:
                cudnn.deterministic = _deterministic_saved


def _is_compiling() -> bool:
    try:
        import torch
        check = getattr(getattr(torch, "compiler", None), "is_compiling", None)
        return bool(check()) if check is not None else False
    except Exception:  # noqa: BLE001
        return False


def _no_autocast(device_type: str) -> Any:
    """The caller's autocast switched off for ``device_type``: the float32 pass must not be re-lowered."""
    import torch
    try:
        return torch.autocast(device_type = device_type, enabled = False)
    except Exception:  # noqa: BLE001 -- a backend without autocast has none to override
        return contextlib.nullcontext()


def _stage_fp32(module: Any) -> list:
    """Float32 copies of every floating parameter / buffer not already float32, as (owner, name, is_param, copy).
    Allocates only: the module is untouched, so an out-of-memory here leaves nothing to roll back."""
    import torch

    staged = []
    for owner in module.modules():
        for name, p in owner._parameters.items():
            if p is not None and p.is_floating_point() and p.dtype != torch.float32:
                staged.append((owner, name, True, p.detach().float()))
        for name, b in owner._buffers.items():
            if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                staged.append((owner, name, False, b.float()))
    return staged


@contextlib.contextmanager
def _swapped_in(staged: list) -> Iterator[None]:
    """Point the module at the staged float32 tensors for the call and back after. Reference swaps only (``.data``
    on the existing Parameters, buffer dict entries): nothing allocates, so the restore cannot fail half way."""
    saved = []
    try:
        for owner, name, is_param, copy in staged:
            if is_param:
                p = owner._parameters[name]
                saved.append((owner, name, True, p.data))
                p.data = copy
            else:
                saved.append((owner, name, False, owner._buffers[name]))
                owner._buffers[name] = copy
        yield
    finally:
        for owner, name, is_param, old in reversed(saved):
            if is_param:
                owner._parameters[name].data = old
            else:
                owner._buffers[name] = old


def _fp32_resampler_filter(vocoder: Any) -> Any:
    """The BWE skip resampler's sinc filter rebuilt in float32, or None. It is a non-persistent buffer, so no
    checkpoint holds it: the reference builds it in float32 beside bf16 weights, while a bf16 load either rounds it
    (``.to(bf16)``) or computes it in bf16 (diffusers' from_pretrained sets the default dtype). Returned only when it
    matches the stored filter to bf16 accuracy, so a changed upstream filter is left alone."""
    import torch

    res = getattr(vocoder, "resampler", None)
    stored = getattr(res, "_buffers", {}).get("filter") if res is not None else None
    if (
        stored is None
        or "filter" not in res._non_persistent_buffers_set
        or stored.dtype == torch.float32
    ):
        return None
    with torch.device("cpu"):
        fresh = type(res)(
            ratio = res.ratio,
            window_type = "hann",
            padding_mode = getattr(res, "padding_mode", "replicate"),
            persistent = False,
        ).filter.float()
    old = stored.detach().float().cpu()
    if fresh.shape != old.shape or not torch.allclose(
        fresh, old, rtol = 0, atol = 0.02 * float(fresh.abs().max())
    ):
        return None
    return fresh


def _install_fp32_filter(res: Any, fp32_filter: Any) -> None:
    """The float32 filter becomes the resampler's ``filter`` buffer (so the float32 forward reads it with no swap and
    stays traceable); the loaded one is kept beside it as a buffer for the stored-dtype path."""
    stored = res._buffers["filter"]
    res.register_buffer(_STORED_FILTER, stored, persistent = False)
    res.register_buffer("filter", fp32_filter.to(stored.device), persistent = False)


@contextlib.contextmanager
def _stored_filter(res: Any) -> Iterator[None]:
    """The filter the vocoder was loaded with, for a stored-dtype (main) call: that path stays bit-identical."""
    stored = getattr(res, "_buffers", {}).get(_STORED_FILTER) if res is not None else None
    if stored is None:
        yield
        return
    current = res._buffers["filter"]
    res._buffers["filter"] = stored
    try:
        yield
    finally:
        res._buffers["filter"] = current


def _refresh_fp32_filter(res: Any, fp32_filter: Any) -> None:
    """A cast of the whole pipe (``.to(bf16)``) rounds the float32 filter: put the exact one back (eager only)."""
    import torch

    current = res._buffers.get("filter")
    if current is not None and current.dtype != torch.float32:
        res._buffers["filter"] = fp32_filter.to(current.device)


# Extra peak of the float32 forward over bf16, measured on an L4 (LTX-2.5 vocoder, 5.04 s clip = 501 mel frames):
# +104.5 MiB, linear in the clip length. Below it (with margin) free on the card, the call runs the stored dtype.
_FP32_EXTRA_MIB_PER_MEL_FRAME = 105.0 / 501
_FP32_MARGIN = 1.5
_FP32_MARGIN_MIB = 64
_fallback_logged = [False]
# Stored-dtype fallbacks taken so far: a request whose vocoder fell back records no float32 render peak.
_fallbacks = [0]


def vocoder_fallback_count() -> int:
    return _fallbacks[0]


def vocoder_precision(vocoder: Any) -> Optional[str]:
    """The precision this vocoder decodes in, for the render-peak calibration key: "fp32" (the float32 pass is on, or
    the weights are float32), "stored" (switched off), None for any other vocoder."""
    if not getattr(vocoder, "_unsloth_fp32_forward", False):
        return None
    try:
        param = next(vocoder.parameters(), None)
        if param is not None and str(param.dtype) == "torch.float32":
            return "fp32"
    except Exception:  # noqa: BLE001
        return None
    return "fp32" if vocoder_fp32_enabled() else "stored"


def vocoder_fp32_extra_bytes(mel_spec: Any) -> int:
    """Bytes the float32 forward needs beyond the bf16 one for this mel input, margin included."""
    frames = int(mel_spec.shape[-2]) if getattr(mel_spec, "dim", lambda: 0)() >= 2 else 0
    batch = int(mel_spec.shape[0]) if getattr(mel_spec, "dim", lambda: 0)() >= 4 else 1
    mib = _FP32_EXTRA_MIB_PER_MEL_FRAME * frames * batch * _FP32_MARGIN + _FP32_MARGIN_MIB
    return int(mib * (1 << 20))


def _cuda_room(device: Any) -> Optional[int]:
    """Free bytes on ``device``: the driver's free memory plus the caching allocator's unused reserve."""
    try:
        import torch
        free, _ = torch.cuda.mem_get_info(device)
        return (
            int(free)
            + int(torch.cuda.memory_reserved(device))
            - int(torch.cuda.memory_allocated(device))
        )
    except Exception:  # noqa: BLE001
        return None


def _is_oom(exc: BaseException) -> bool:
    import torch

    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _note_fallback(reason: str) -> None:
    _fallbacks[0] += 1
    if not _fallback_logged[0]:
        _fallback_logged[0] = True
        logger.warning(
            "LTX vocoder: %s; this decode runs in the stored dtype (bf16) instead of float32",
            reason,
        )


def install_vocoder_fp32_forward(vocoder: Any) -> None:
    """Run this vocoder instance's forward in float32 and return a float32 waveform; idempotent, never raises.

    Eager calls never fail a render: short of room on the card, or out of memory (while staging float32 weights or
    in the pass), the call runs the stored dtype on an unchanged module. Under torch.compile the forward is a plain
    traceable graph: CUDA autocast in float32, or the stored dtype where float32 needs eager-only weight staging."""
    try:
        import torch
        import types

        if getattr(vocoder, "_unsloth_fp32_forward", False):
            return
        inner = vocoder.forward
        fp32_filter = _fp32_resampler_filter(vocoder)
        res = getattr(vocoder, "resampler", None)
        if fp32_filter is not None:
            try:
                _install_fp32_filter(res, fp32_filter)
            except Exception as exc:  # noqa: BLE001 -- the loaded filter stays: float32 weights, rounded filter
                logger.debug("LTX vocoder float32 filter not installed: %s", exc)
                fp32_filter = None

        def stored_call(self, mel_spec, args, kwargs):
            if _is_compiling():
                return inner(mel_spec, *args, **kwargs)
            with _stored_filter(self.resampler if fp32_filter is not None else None):
                return inner(mel_spec, *args, **kwargs)

        def fallback(self, mel_spec, args, kwargs, reason):
            _note_fallback(reason)
            if mel_spec.device.type == "cuda":
                torch.cuda.empty_cache()
            return stored_call(self, mel_spec, args, kwargs)

        def forward(self, mel_spec, *args, **kwargs):
            with _deterministic_cudnn(mel_spec):
                return fp32_forward(self, mel_spec, *args, **kwargs)

        def fp32_forward(self, mel_spec, *args, **kwargs):
            param = next(self.parameters(), None)
            compiling = _is_compiling()
            if (
                param is None
                or not torch.is_floating_point(mel_spec)
                or (not compiling and not vocoder_fp32_enabled())
            ):
                return stored_call(self, mel_spec, args, kwargs)
            device_type = mel_spec.device.type
            if param.dtype == torch.float32:
                with _no_autocast(device_type):
                    return inner(mel_spec.float(), *args, **kwargs)
            if compiling:
                if device_type == "cuda":
                    with torch.autocast(device_type = "cuda", dtype = torch.float32):
                        return inner(mel_spec.float(), *args, **kwargs)
                return inner(mel_spec, *args, **kwargs)
            if fp32_filter is not None:
                _refresh_fp32_filter(self.resampler, fp32_filter)
            if device_type == "cuda":
                room = _cuda_room(mel_spec.device)
                need = vocoder_fp32_extra_bytes(mel_spec)
                if room is not None and room < need:
                    return fallback(
                        self,
                        mel_spec,
                        args,
                        kwargs,
                        f"{room >> 20} MiB free, the float32 pass needs about {need >> 20} MiB",
                    )
                failed = False
                try:
                    # CUDA (and ROCm) autocast upcasts bf16 weights per op: bit-equal to a float32 build, at a
                    # fraction of the memory of .float(). It overrides an enclosing autocast.
                    with torch.autocast(device_type = "cuda", dtype = torch.float32):
                        return inner(mel_spec.float(), *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc):
                        raise
                    failed = True
                if (
                    failed
                ):  # outside the except block, so the failed pass's tensors are released first
                    return fallback(
                        self, mel_spec, args, kwargs, "out of memory in the float32 pass"
                    )
            # Other backends refuse float32 autocast: float32 copies of the weights for the call, staged before the
            # module is touched and swapped by reference, so any failure leaves the stored weights in place.
            failed = False
            try:
                staged = _stage_fp32(self)
                with _swapped_in(staged), _no_autocast(device_type):
                    return inner(mel_spec.float(), *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    raise
                failed = True
            staged = None
            if failed:
                return fallback(self, mel_spec, args, kwargs, "out of memory in the float32 pass")

        vocoder.forward = types.MethodType(forward, vocoder)
        vocoder._unsloth_fp32_forward = True
    except Exception as exc:  # the render keeps the stored-dtype vocoder
        logger.debug("LTX vocoder float32 forward not installed: %s", exc)


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
    revision: Optional[str] = None,
) -> Any:
    """Assembled per component, bypassing from_pretrained, so local_files_only must be passed explicitly."""
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
        if is_gguf:
            from .diffusion_gguf_exact import install_exact_dequant
            from .diffusion_gguf_lowbit import install_lowbit_iq
            from .diffusion_gguf_mmap import install_gguf_mmap

            install_lowbit_iq()  # before the read: diffusers' reader refuses unknown types
            install_exact_dequant()
            install_gguf_mmap()
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
        # Pre-built (hosted) DiT: the file contributes only connectors / VAEs / vocoder.
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
    # ``revision``: the load plan's commit for the base, so the components come from the snapshot it staged (passed only
    # when set, so an unpinned load calls exactly as before).
    pinned: dict[str, Any] = {"revision": revision} if revision else {}
    index = LTX2Pipeline.load_config(
        base_repo,
        token = hf_token,
        local_files_only = local_files_only,
        cache_dir = cache_dir,
        **pinned,
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
            **pinned,
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
