# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3 (nvidia/Cosmos3-Nano, nvidia/Cosmos3-Edge) on diffusers' ``Cosmos3OmniPipeline``.

The guardrail (``cosmos_guardrail``) is built whenever ``model_index.json`` enables it and raises ``ImportError``
without the package, so Studio refuses the load up front; ``UNSLOTH_COSMOS3_SAFETY_CHECKER=0`` opts out.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

SAFETY_CHECKER_ENV = "UNSLOTH_COSMOS3_SAFETY_CHECKER"
# Passed as None to from_pretrained: Studio renders video only.
NONE_COMPONENTS = ("sound_tokenizer",)
GUARDRAIL_PACKAGE = "cosmos_guardrail"
# Fallback when the index is not on disk yet (validation is network-free).
_BASE_SAFETY_CHECKER = {
    "nvidia/cosmos3-nano": True,
    "nvidia/cosmos3-edge": False,
}


def is_cosmos3_listing(names: Iterable[str]) -> bool:
    """Whether a Diffusers repo listing has Cosmos3's layout: a ``text_tokenizer/`` and no ``text_encoder/``."""
    names = list(names)
    return any(n.startswith("text_tokenizer/") for n in names) and not any(
        n.startswith("text_encoder/") for n in names
    )


def safety_checker_opted_out() -> bool:
    return os.environ.get(SAFETY_CHECKER_ENV, "").strip().lower() in ("0", "false", "no", "off")


def guardrail_installed() -> bool:
    try:
        return importlib.util.find_spec(GUARDRAIL_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def _cached_model_index(repo_id: str) -> Optional[dict]:
    """The pipeline index from a local folder or the Hub cache; never the network."""
    path: Optional[Path] = None
    local = Path(repo_id).expanduser()
    if local.is_dir():
        path = local / "model_index.json"
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
            found = try_to_load_from_cache(repo_id, "model_index.json")
            path = Path(found) if isinstance(found, str) else None
        except Exception:  # noqa: BLE001 -- a cache miss falls back to the base repo's known setting
            path = None
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def checkpoint_enables_safety_checker(repo_id: str, index: Optional[dict] = None) -> bool:
    """``enable_safety_checker`` from the model_index.json, else the base repo's shipped value; on when unknown."""
    if index is None:
        index = _cached_model_index(repo_id)
    if index is not None:
        value = index.get("enable_safety_checker", True)
        return bool(value) if isinstance(value, bool) else True
    return _BASE_SAFETY_CHECKER.get(str(repo_id or "").strip().lower(), True)


_TE_QUANT_UNSET = frozenset({"", "auto", "none", "off"})


def load_request_refusal(
    family: str, kind: str, text_encoder_quant: Optional[str]
) -> Optional[str]:
    """Refuse non-pipeline picks (no ``from_single_file``) and an explicit text-encoder precision (none exists)."""
    if kind != "pipeline":
        return (
            f"{family} loads only as its full Diffusers pipeline (nvidia/Cosmos3-Nano or nvidia/Cosmos3-Edge): "
            "Cosmos3OmniTransformer has no single-file loader, so GGUF and single .safetensors checkpoints cannot run."
        )
    scheme = str(text_encoder_quant or "").strip().lower()
    if scheme not in _TE_QUANT_UNSET:
        return (
            f"text_encoder_quant '{text_encoder_quant}' does not apply to {family}: Cosmos3 has no separate text "
            "encoder (its transformer reads the prompt tokens itself). Leave text_encoder_quant unset."
        )
    return None


def guardrail_refusal(repo_id: str, index: Optional[dict] = None) -> Optional[str]:
    """Why this load cannot start, or None: the checkpoint enables the guardrail and the package is missing."""
    if safety_checker_opted_out() or guardrail_installed():
        return None
    if not checkpoint_enables_safety_checker(repo_id, index):
        return None
    return (
        f"'{repo_id}' runs NVIDIA's Cosmos guardrail (its model_index.json does not turn enable_safety_checker off, and "
        f"the pipeline's default is on), which needs the cosmos_guardrail package: pip install cosmos-guardrail into "
        f"Studio's Python environment, restart Studio, and retry. To run without it, set {SAFETY_CHECKER_ENV}=0; the "
        f"model card's terms and safe use stay your responsibility."
    )


def pipeline_kwargs(repo_id: str, index: Optional[dict] = None) -> dict[str, Any]:
    """``from_pretrained`` kwargs: no sound tokenizer; guardrail off only on the explicit opt-out."""
    kwargs: dict[str, Any] = dict.fromkeys(NONE_COMPONENTS)
    if safety_checker_opted_out():
        kwargs["enable_safety_checker"] = False
        if checkpoint_enables_safety_checker(repo_id, index):
            logger.warning(
                "video.cosmos3: %s=0, so '%s' loads without the Cosmos guardrail its checkpoint runs by default",
                SAFETY_CHECKER_ENV,
                repo_id,
            )
    return kwargs


def install_untiled_encode(pipe: Any, logger: Any = None) -> bool:
    """Run ``pipe.vae.encode`` untiled, tiled only on out-of-memory: every step attends to the encoded i2v clip,
    so tiling's small shift grows across it (Nano i2v vs diffusers: 24.7 dB tiled)."""
    vae = getattr(pipe, "vae", None)
    encode = getattr(vae, "encode", None)
    if vae is None or not callable(encode):
        return False
    if getattr(encode, "_unsloth_untiled_encode", False):
        return True

    def untiled_encode(*args: Any, **kwargs: Any) -> Any:
        tiling = bool(getattr(vae, "use_tiling", False))
        if not tiling:
            return encode(*args, **kwargs)
        from .diffusion_batched import is_oom_error

        vae.use_tiling = False
        try:
            return encode(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if not is_oom_error(exc):
                raise
        finally:
            vae.use_tiling = tiling
        # Retried outside the handler so the failed attempt's tensors (held by the traceback) are gone first.
        import torch

        torch.cuda.empty_cache()
        if logger is not None:
            logger.warning("video.cosmos3: untiled VAE encode ran out of memory; encoding tiled")
        return encode(*args, **kwargs)

    untiled_encode._unsloth_untiled_encode = True
    vae.encode = untiled_encode
    if logger is not None:
        logger.info(
            "video.cosmos3: VAE encodes the conditioning clip untiled (tiled on out-of-memory)"
        )
    return True


# Wan@16x seam audit floor (test_vae_tile_geometry): >= 4-latent overlaps, >= 8-latent edge tiles.
_MIN_OVERLAP_LATENTS = 4
_MIN_EDGE_LATENTS = 8


def seam_safe_stride(length: int, tile: int, stride: int) -> int:
    """Largest stride <= ``stride`` (latents) whose last tile over ``length`` keeps >= 8 latents; stock if none."""
    if length <= tile:
        return stride
    for s in range(stride, tile // 2 - 1, -1):
        if tile - s >= _MIN_OVERLAP_LATENTS and length - (length - 1) // s * s >= _MIN_EDGE_LATENTS:
            return s
    return stride


@contextlib.contextmanager
def seam_safe_tile_strides(pipe: Any, height: int, width: int):
    """Tile strides for this canvas while tiling runs: stock Wan@16x tiles leave a 4-latent edge tile at 480 / 832 px."""
    vae = getattr(pipe, "vae", None)
    keys = ("tile_sample_stride_height", "tile_sample_stride_width")
    try:
        ratio = int(vae.spatial_compression_ratio)
        old = tuple(int(getattr(vae, k)) for k in keys)
        tiles = (int(vae.tile_sample_min_height) // ratio, int(vae.tile_sample_min_width) // ratio)
    except (AttributeError, TypeError, ValueError):
        yield
        return
    for key, side, tile, stride in zip(keys, (height, width), tiles, old):
        setattr(vae, key, seam_safe_stride(int(side) // ratio, tile, stride // ratio) * ratio)
    try:
        yield
    finally:
        for key, value in zip(keys, old):
            setattr(vae, key, value)
