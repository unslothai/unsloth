# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 pipeline loader: assembles ``Krea2Pipeline`` from per-component loads.

Why not ``from_pretrained``: the ``krea/Krea-2-Turbo`` repo was exported with transformers 5.2 and
two configs use 5.x-only conventions 4.x can't parse:

- ``tokenizer_config.json`` declares slow ``Qwen2Tokenizer`` but ships only ``tokenizer.json``.
  4.x's slow class needs vocab.json/merges.txt (absent), and its fast class trips over
  ``extra_special_tokens`` stored as a LIST. Loading the fast class with ``extra_special_tokens={}``
  is id-identical (every token is already an added special token, and the pipeline templates prompts
  manually).
- ``text_encoder/config.json`` keeps rope under ``rope_parameters`` (5.x); 4.x reads
  ``rope_scaling`` + ``rope_theta`` and crashes. The values are copied verbatim and equal 4.x's
  Qwen3-VL defaults, so the rotary embedding is numerically identical.

``from_pretrained`` also type-checks a passed ``tokenizer`` against the SLOW class, so the pipeline
is built through its constructor, forwarding the ``is_distilled`` / ``text_encoder_select_layers`` /
``patch_size`` init config (Turbo's mu=1.15 shift rides on ``is_distilled``).

Both workarounds self-disable on transformers 5.x (the plain tokenizer load succeeds, rope_scaling
parses non-None).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

KREA2_FAMILY_NAME = "krea-2"


def _live_cache_dir() -> str:
    """Live hub root, since the import-time constant goes stale after a cache-folder change."""
    from utils.hf_cache_settings import active_hf_hub_cache
    return active_hf_hub_cache()


def load_krea2_tokenizer(
    repo_id: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
):
    """The Krea 2 tokenizer, tolerating the repo's transformers-5.x tokenizer config."""
    if check_cancelled is not None:
        check_cancelled()
    from transformers import AutoTokenizer

    from .diffusion_offline_source import offline_snapshot_source

    cache_dir = _live_cache_dir()
    kwargs: dict[str, Any] = {
        "subfolder": "tokenizer",
        "local_files_only": local_files_only,
        "cache_dir": cache_dir,
    }
    if hf_token:
        kwargs["token"] = hf_token
    # Offline by repo id, transformers 5.x fails both attempts below (see diffusion_offline_source).
    source = offline_snapshot_source(
        repo_id, "tokenizer", local_files_only = local_files_only, cache_dir = cache_dir
    )
    try:
        return AutoTokenizer.from_pretrained(source, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- 4.x config-parse failure, retry with override
        if check_cancelled is not None:
            check_cancelled()
        logger.info("diffusion.krea2 tokenizer compat fallback: %s", exc)
        return AutoTokenizer.from_pretrained(source, extra_special_tokens = {}, **kwargs)


def remap_rope_parameters(text_config) -> None:
    """Copy 5.x ``rope_parameters`` onto the 4.x ``rope_scaling`` / ``rope_theta`` slots in place.
    No-op on a 5.x runtime (rope_scaling already non-None) or when there is no ``rope_parameters``."""
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if getattr(text_config, "rope_scaling", None) is None and isinstance(rope_parameters, dict):
        text_config.rope_scaling = {k: v for k, v in rope_parameters.items() if k != "rope_theta"}
        if "rope_theta" in rope_parameters:
            text_config.rope_theta = rope_parameters["rope_theta"]


def load_krea2_text_encoder(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
):
    """The Qwen3-VL text encoder, remapping 5.x ``rope_parameters`` for a 4.x runtime."""
    if check_cancelled is not None:
        check_cancelled()
    from transformers import AutoConfig, Qwen3VLModel

    kwargs: dict[str, Any] = {
        "subfolder": "text_encoder",
        "local_files_only": local_files_only,
        "cache_dir": _live_cache_dir(),
    }
    if hf_token:
        kwargs["token"] = hf_token
    config = AutoConfig.from_pretrained(repo_id, **kwargs)
    if check_cancelled is not None:
        check_cancelled()
    remap_rope_parameters(getattr(config, "text_config", config))
    return Qwen3VLModel.from_pretrained(repo_id, config = config, dtype = dtype, **kwargs)


def _read_model_index(path: Path, source: str) -> dict[str, Any]:
    try:
        model_index = json.loads(path.read_text(encoding = "utf-8-sig"))
    # RecursionError (nesting bomb) is not a ValueError.
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            f"Unable to read valid model_index.json from {source} at {path}: {exc}"
        ) from exc
    if not isinstance(model_index, dict):
        raise ValueError(f"model_index.json from {source} at {path} must contain a JSON object")
    return model_index


def _load_model_index(
    repo_id: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """model_index.json as a dict, from a local path or the Hub cache."""
    is_local_dir = False
    try:
        root = Path(repo_id).expanduser()
        is_local_dir = root.is_dir()
        local = root / "model_index.json"
        if local.is_file():
            return _read_model_index(local, f"local model directory {root}")
    except OSError:
        pass
    if is_local_dir:
        raise FileNotFoundError(f"model_index.json not found in local model dir {repo_id}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id,
        "model_index.json",
        token = hf_token or None,
        local_files_only = local_files_only,
        cache_dir = _live_cache_dir(),
    )
    return _read_model_index(Path(path), f"Hub/cache for {repo_id}")


def load_krea2_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    transformer = None,
    with_transformer: bool = True,
    text_encoder = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
):
    """local_files_only raises rather than fetching a component after the resident pipeline was evicted."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    import diffusers

    if not hasattr(diffusers, "Krea2Pipeline"):
        from .diffusion_families import DIFFUSERS_UPDATE_REMEDY
        raise RuntimeError(
            f"Krea 2 needs diffusers >= 0.39.0 (Krea2Pipeline); this environment has "
            f"diffusers {getattr(diffusers, '__version__', 'unknown')}. "
            f"{DIFFUSERS_UPDATE_REMEDY} On a plain pip install: pip install -U diffusers"
        )

    token = hf_token or None
    cache_dir = _live_cache_dir()
    model_index = _load_model_index(repo_id, hf_token = token, local_files_only = local_files_only)
    check_cancelled()
    tokenizer = load_krea2_tokenizer(
        repo_id,
        hf_token = token,
        local_files_only = local_files_only,
        check_cancelled = check_cancelled,
    )
    check_cancelled()
    if text_encoder is None:
        text_encoder = load_krea2_text_encoder(
            repo_id,
            dtype,
            hf_token = token,
            local_files_only = local_files_only,
            check_cancelled = check_cancelled,
        )
        check_cancelled()
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id,
        subfolder = "scheduler",
        token = token,
        local_files_only = local_files_only,
        cache_dir = cache_dir,
    )
    check_cancelled()
    vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        repo_id,
        subfolder = "vae",
        torch_dtype = dtype,
        token = token,
        local_files_only = local_files_only,
        cache_dir = cache_dir,
    )
    check_cancelled()
    if transformer is None and with_transformer:
        transformer = diffusers.Krea2Transformer2DModel.from_pretrained(
            repo_id,
            subfolder = "transformer",
            torch_dtype = dtype,
            token = token,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
        check_cancelled()
    return diffusers.Krea2Pipeline(
        scheduler = scheduler,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        transformer = transformer,
        text_encoder_select_layers = model_index.get("text_encoder_select_layers"),
        is_distilled = bool(model_index.get("is_distilled", False)),
        patch_size = int(model_index.get("patch_size", 2)),
    )
