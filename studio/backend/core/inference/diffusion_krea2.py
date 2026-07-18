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
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

KREA2_FAMILY_NAME = "krea-2"


def load_krea2_tokenizer(repo_id: str, hf_token: Optional[str] = None):
    """The Krea 2 tokenizer, tolerating the repo's transformers-5.x tokenizer config."""
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"subfolder": "tokenizer"}
    if hf_token:
        kwargs["token"] = hf_token
    try:
        return AutoTokenizer.from_pretrained(repo_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- 4.x config-parse failure, retry with override
        logger.info("diffusion.krea2 tokenizer compat fallback: %s", exc)
        return AutoTokenizer.from_pretrained(repo_id, extra_special_tokens = {}, **kwargs)


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
):
    """The Qwen3-VL text encoder, remapping 5.x ``rope_parameters`` for a 4.x runtime."""
    from transformers import AutoConfig, Qwen3VLModel

    kwargs: dict[str, Any] = {"subfolder": "text_encoder"}
    if hf_token:
        kwargs["token"] = hf_token
    config = AutoConfig.from_pretrained(repo_id, **kwargs)
    remap_rope_parameters(getattr(config, "text_config", config))
    return Qwen3VLModel.from_pretrained(repo_id, config = config, dtype = dtype, **kwargs)


def _load_model_index(repo_id: str, hf_token: Optional[str] = None) -> dict[str, Any]:
    """model_index.json as a dict, from a local path or the Hub cache."""
    is_local_dir = False
    try:
        root = Path(repo_id).expanduser()
        is_local_dir = root.is_dir()
        local = root / "model_index.json"
        if local.is_file():
            return json.loads(local.read_text())
    except OSError:
        pass
    if is_local_dir:
        # A local checkpoint dir without the file must fail clearly here: falling through
        # to hf_hub_download with a filesystem path as the repo id would die with an
        # opaque HFValidationError instead.
        raise FileNotFoundError(f"model_index.json not found in local model dir {repo_id}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "model_index.json", token = hf_token or None)
    return json.loads(Path(path).read_text())


def load_krea2_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    transformer = None,
    with_transformer: bool = True,
    text_encoder = None,
):
    """A ready ``Krea2Pipeline`` for ``repo_id`` (still on CPU; caller places it).

    ``transformer`` lets the single-file/quant paths hand in a prebuilt denoiser;
    ``with_transformer = False`` skips the (26 GB) denoiser entirely for a
    conditioning-only pipeline (the trainer's phased load). ``text_encoder`` lets the
    pre-cast TE path (diffusion_te_prequant) hand in an already-built encoder, skipping
    the dense Qwen3-VL download. The remaining components (VAE, tokenizer, scheduler)
    come from the repo.
    """
    import diffusers

    # diffusers gained Krea2Pipeline in 0.39; on an older install the getattr chain below
    # would die with a bare AttributeError mid-load, so fail first with the actionable fix.
    if not hasattr(diffusers, "Krea2Pipeline"):
        raise RuntimeError(
            f"Krea 2 needs diffusers >= 0.39.0 (Krea2Pipeline); this environment has "
            f"diffusers {getattr(diffusers, '__version__', 'unknown')}. "
            f"Upgrade with: pip install -U diffusers"
        )

    token = hf_token or None
    tokenizer = load_krea2_tokenizer(repo_id, hf_token = token)
    if text_encoder is None:
        text_encoder = load_krea2_text_encoder(repo_id, dtype, hf_token = token)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id, subfolder = "scheduler", token = token
    )
    vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        repo_id, subfolder = "vae", torch_dtype = dtype, token = token
    )
    if transformer is None and with_transformer:
        transformer = diffusers.Krea2Transformer2DModel.from_pretrained(
            repo_id, subfolder = "transformer", torch_dtype = dtype, token = token
        )
    model_index = _load_model_index(repo_id, hf_token = token)
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
