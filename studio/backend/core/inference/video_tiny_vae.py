# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# The decoder below follows TAEHV (https://github.com/madebyollin/taehv, MIT License,
# Copyright (c) 2025 Ollin Boer Bohan): same layer layout, so its released checkpoints load as-is.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files, to deal in the Software without restriction, subject to the following conditions: the above
# copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

"""Tiny Wan video decoder (TAEHV layout, ~11M params): sharp live previews for every Wan family and an opt-in draft
decode of the final latent. Reads the denoiser's own (normalised) latents, no ``latents_mean`` / ``latents_std``.
Weights: ``lightx2v/Autoencoders`` (Apache-2.0), fetched once in the background; until they are cached the
previews keep their linear map (or none) and the full VAE decodes, so nothing here can fail a render."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

TINY_VAE_ENV = "UNSLOTH_VIDEO_TINY_VAE"
DRAFT_DECODE_ENV = "UNSLOTH_VIDEO_DRAFT_DECODE"
TINY_VAE_REPO = "lightx2v/Autoencoders"
# latent channels -> (checkpoint, pixel-shuffle patch). Wan2.1 VAE z=16 (1.3B, 14B, A14B), Wan2.2 VAE z=48 (TI2V-5B).
_CHECKPOINTS = {16: ("taew2_1.safetensors", 1), 48: ("taew2_2.safetensors", 2)}
_WAN_PREFIX = "wan"


def _env_on(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "off", "no")


def enabled() -> bool:
    """Kill switch for both uses: ``UNSLOTH_VIDEO_TINY_VAE=0`` keeps the linear previews and the full decode only."""
    return _env_on(TINY_VAE_ENV, True)


def family_supported(family: Optional[str]) -> bool:
    return bool(family) and str(family).strip().lower().startswith(_WAN_PREFIX)


def checkpoint_for(
    family: Optional[str], latent_channels: Optional[int]
) -> Optional[tuple[str, int]]:
    """``(file, patch)`` for a Wan family's latent width, else None (another family's latents would decode to noise)."""
    if not family_supported(family):
        return None
    try:
        return _CHECKPOINTS.get(int(latent_channels))
    except (TypeError, ValueError):
        return None


def family_latent_channels(family: Optional[str]) -> Optional[int]:
    """Wan2.2-TI2V-5B is the only z=48 Wan; every other Wan family decodes through the Wan2.1 VAE (z=16)."""
    if not family_supported(family):
        return None
    return 48 if "5b" in str(family).lower() else 16


def draft_decode_wanted(requested: Optional[bool]) -> bool:
    """Off unless asked (request flag, else ``UNSLOTH_VIDEO_DRAFT_DECODE=1``): the final clip is the full VAE's."""
    if not enabled():
        return False
    if requested is not None:
        return bool(requested)
    return _env_on(DRAFT_DECODE_ENV, False)


def _build(latent_channels: int, patch: int) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv(n_in: int, n_out: int, **kw: Any) -> Any:
        return nn.Conv2d(n_in, n_out, 3, padding = 1, **kw)

    class Clamp(nn.Module):
        def forward(self, x):
            return torch.tanh(x / 3) * 3

    class MemBlock(nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.conv = nn.Sequential(
                conv(n * 2, n), nn.ReLU(inplace = True), conv(n, n), nn.ReLU(inplace = True), conv(n, n)
            )
            self.skip = nn.Identity()
            self.act = nn.ReLU(inplace = True)

        def forward(self, x, past):
            return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

    class TGrow(nn.Module):
        def __init__(self, n: int, stride: int):
            super().__init__()
            self.stride = stride
            self.conv = nn.Conv2d(n, n * stride, 1, bias = False)

        def forward(self, x):
            c = x.shape[1]
            return self.conv(x).reshape(-1, c, x.shape[2], x.shape[3])

    class TinyWanDecoder(nn.Module):
        """``decode(latents)``: (B, C, T, h, w) normalised latents -> (B, 3, 4T-3, H, W) RGB in [0, 1]."""

        def __init__(self):
            super().__init__()
            self.latent_channels, self.patch = latent_channels, patch
            n = (256, 128, 64, 64)
            # Time upscale per stage (False, True, True): 4x, so 4 frames per latent with the first 3 trimmed.
            self.decoder = nn.Sequential(
                Clamp(), conv(latent_channels, n[0]), nn.ReLU(inplace = True),
                MemBlock(n[0]), MemBlock(n[0]), MemBlock(n[0]), nn.Upsample(scale_factor = 2), TGrow(n[0], 1),
                conv(n[0], n[1], bias = False),
                MemBlock(n[1]), MemBlock(n[1]), MemBlock(n[1]), nn.Upsample(scale_factor = 2), TGrow(n[1], 2),
                conv(n[1], n[2], bias = False),
                MemBlock(n[2]), MemBlock(n[2]), MemBlock(n[2]), nn.Upsample(scale_factor = 2), TGrow(n[2], 2),
                conv(n[2], n[3], bias = False),
                nn.ReLU(inplace = True), conv(n[3], 3 * patch * patch),
            )  # fmt: skip
            self.frames_to_trim = 3

        @torch.no_grad()
        def decode(
            self,
            latents,
            chunk_frames: int = 2,
        ):
            """Causal over time, so ``chunk_frames`` latent frames at a time with each MemBlock's last input carried
            over: the same frames as one pass (to float rounding) at a fraction of the peak memory."""
            b, t = latents.shape[0], latents.shape[2]
            step = max(1, int(chunk_frames or t))
            carry: dict = {}
            outs = []
            for start in range(0, t, step):
                piece = latents[:, :, start : start + step]
                x = piece.transpose(1, 2).reshape(-1, *piece.shape[1:2], *piece.shape[3:])
                for i, block in enumerate(self.decoder):
                    if isinstance(block, MemBlock):
                        seq = x.reshape(b, -1, *x.shape[1:])
                        # each frame's memory is the previous frame's input to this block (zeros for the first)
                        first = carry.get(i)
                        first = torch.zeros_like(seq[:, :1]) if first is None else first
                        past = torch.cat([first, seq[:, :-1]], 1).reshape(x.shape)
                        carry[i] = seq[:, -1:].clone()
                        x = block(x, past)
                    else:
                        x = block(x)
                if self.patch > 1:
                    x = F.pixel_shuffle(x, self.patch)
                outs.append(x.clamp_(0, 1).reshape(b, -1, *x.shape[1:]))
            video = torch.cat(outs, 1) if len(outs) > 1 else outs[0]
            return video[:, self.frames_to_trim :].transpose(1, 2)

    return TinyWanDecoder()


_LOCK = threading.Lock()
_DECODERS: dict = {}
_FETCHING: dict = {}
_FAILED: set = set()


def _hub_cache() -> Optional[str]:
    """Studio's live Hugging Face cache root (Settings can move it); huggingface_hub's import-time default otherwise
    lands the checkpoint where no other Studio loader looks."""
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        return str(active_hf_hub_cache())
    except Exception:  # noqa: BLE001 - outside Studio: the library default
        return None


def _cached_path(filename: str) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(TINY_VAE_REPO, filename, cache_dir = _hub_cache())
        return hit if isinstance(hit, str) and os.path.isfile(hit) else None
    except Exception:  # noqa: BLE001
        return None


def _fetch(filename: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(TINY_VAE_REPO, filename, cache_dir = _hub_cache())
    except Exception as exc:  # noqa: BLE001 - offline / blocked: previews stay linear
        with _LOCK:
            _FAILED.add(filename)
        logger.info(
            "video.tiny_vae: %s not fetched (%s); previews keep the linear map", filename, exc
        )
    finally:
        with _LOCK:
            _FETCHING.pop(filename, None)


def prefetch(family: Optional[str], *, local_files_only: bool = False) -> None:
    """Start the one-off background download (23 MB) of this family's checkpoint; never blocks, never raises."""
    if not enabled():
        return
    hit = checkpoint_for(family, family_latent_channels(family))
    if hit is None or _cached_path(hit[0]) is not None or local_files_only:
        return
    with _LOCK:
        if hit[0] in _FETCHING or hit[0] in _FAILED:
            return
        worker = threading.Thread(target = _fetch, args = (hit[0],), name = "tiny-vae-fetch", daemon = True)
        _FETCHING[hit[0]] = worker
    worker.start()


def load_from_file(path: str, latent_channels: int, patch: int, *, device: Any, dtype: Any) -> Any:
    """Decoder half of a TAEHV checkpoint (encoder keys dropped), strict, eval, on ``device`` in ``dtype``."""
    from safetensors.torch import load_file

    model = _build(latent_channels, patch)
    state = {k: v for k, v in load_file(path).items() if k.startswith("decoder.")}
    model.load_state_dict(state, strict = True)
    return model.eval().requires_grad_(False).to(device = device, dtype = dtype)


def decode_dtype(device: Any) -> Any:
    """fp16 on CUDA / ROCm / MPS (what the checkpoints are), fp32 on CPU (no fast fp16 convs there)."""
    import torch
    return torch.float32 if torch.device(device).type == "cpu" else torch.float16


def get_decoder(family: Optional[str], latent_channels: int, device: Any) -> Optional[Any]:
    """The cached decoder for this family / latent width / device, or None when it is not usable (yet)."""
    if not enabled():
        return None
    hit = checkpoint_for(family, latent_channels)
    if hit is None:
        return None
    filename, patch = hit
    import torch

    dev = torch.device(device)
    key = (filename, str(dev))
    with _LOCK:
        model = _DECODERS.get(key)
        if model is not None or key in _FAILED:
            return model
    path = _cached_path(filename)
    if path is None:
        prefetch(family)
        return None
    try:
        model = load_from_file(
            path, int(latent_channels), patch, device = dev, dtype = decode_dtype(dev)
        )
    except Exception as exc:  # noqa: BLE001 - a bad file means no tiny decode, never a failed render
        logger.warning(
            "video.tiny_vae: %s unusable (%s); keeping the linear previews", filename, exc
        )
        with _LOCK:
            _FAILED.add(key)
        return None
    with _LOCK:
        _DECODERS[key] = model
    return model


class TinyVaeDraftDecode:
    """Context manager that swaps ``vae.decode`` for the tiny decoder for one render, mapping the pipeline's
    un-normalised latents back with the VAE's own ``latents_mean`` / ``latents_std``. ``engaged`` says whether it did."""

    def __init__(self, vae: Any, decoder: Any):
        self.vae, self.decoder, self.engaged, self.calls = vae, decoder, False, 0
        self._had_own = False
        self._original = None

    def _draft(
        self,
        z: Any,
        return_dict: bool = True,
        **_kw: Any,
    ) -> Any:
        import torch

        cfg = getattr(self.vae, "config", None)
        mean = torch.tensor(cfg.latents_mean, device = z.device, dtype = torch.float32).view(
            1, -1, 1, 1, 1
        )
        std = torch.tensor(cfg.latents_std, device = z.device, dtype = torch.float32).view(
            1, -1, 1, 1, 1
        )
        p = next(self.decoder.parameters())
        x = ((z.float() - mean) / std).to(p.device, p.dtype)
        video = self.decoder.decode(x).to(z.device, torch.float32).mul_(2.0).sub_(1.0)
        self.calls += 1
        if return_dict:
            from diffusers.models.autoencoders.vae import DecoderOutput
            return DecoderOutput(sample = video)
        return (video,)

    def __enter__(self) -> "TinyVaeDraftDecode":
        cfg = getattr(self.vae, "config", None)
        if self.vae is None or self.decoder is None or cfg is None:
            return self
        if getattr(cfg, "latents_mean", None) is None or getattr(cfg, "latents_std", None) is None:
            return self
        if int(getattr(cfg, "z_dim", -1)) != int(self.decoder.latent_channels):
            return self
        self._had_own = "decode" in self.vae.__dict__
        self._original = self.vae.__dict__.get("decode")
        self.vae.decode = self._draft
        self.engaged = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self.engaged:
            return
        if self._had_own:
            self.vae.decode = self._original
        else:
            try:
                del self.vae.decode
            except AttributeError:
                pass
