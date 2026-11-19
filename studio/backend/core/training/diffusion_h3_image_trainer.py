# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRA training for MiniMax-H3-Image (33B DiT, Qwen3-VL-32B conditioner).

Own loop (shares ``diffusion_train_common``): the release ships a native key layout with no diffusers pipeline, so it
reuses Studio's inference loaders. The checkpoint is distilled to 6 steps with one head per step and an AdaLN lookup
table over exactly those timesteps, so training draws one of the 6 sampler sigmas and runs that step's frozen head.
The head predicts a DATA-ward velocity (``x0 = x_t + sigma * v``): target ``latents - noise``.
"""

from __future__ import annotations

import gc
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from core.training.diffusion_train_common import (
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    EventCb,
    PermutationBatchSampler,
    StopCb,
    _apply_perf_flags,
    _assert_trusted_base_model,
    _emit,
    _publish_to_lora_catalog,
    _restore_perf_flags,
    discover_image_caption_pairs,
    enable_diffusion_gradient_checkpointing,
    flow_bf16_trainable,
    h3_train_unsupported_reason,
    native_bf16_supported_xpu,
    reset_training_peak_memory,
    resolve_train_device,
    resolve_train_steps,
    train_recipe_overrides,
    training_fetch_repo,
    training_peak_memory_gb,
)
from core.training.diffusion_train_extras import (
    LoRAEMA,
    accumulate_loss,
    loss_value,
    save_ema_adapter,
)

H3_IMAGE_TRAIN_FAMILY = "minimax-h3-image"

# A REGEX: PEFT suffix-matches a list and would also adapt the token refiner.
H3_IMAGE_TARGET_LEAVES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
H3_IMAGE_TARGETS = (
    r"transformer_blocks\.\d+\.(?:"
    + "|".join(leaf.replace(".", r"\.") for leaf in H3_IMAGE_TARGET_LEAVES)
    + ")"
)

# Free GB above which ``auto`` trains on the bf16 base.
H3_IMAGE_BF16_MIN_FREE_GB = 56.0

# "grid" (default): the 6 distilled sigmas. "continuous" (diagnostic): needs the dense AdaLN projections (+13B params).
SIGMA_MODE_ENV = "UNSLOTH_H3_IMAGE_TRAIN_SIGMAS"

# The conditioner the run encodes with: "int8" (Studio's hosted ConvRot copy, default) or "bf16" (66.7 GB).
TEXT_ENCODER_ENV = "UNSLOTH_H3_IMAGE_TRAIN_TEXT_ENCODER"


# De-distillation adapter merged into the FROZEN base for training only (never saved or used at inference), as
# ai-toolkit does. "auto" | "ostris" | "off" | path. A/B: without it, outputs came out blotchy.
TRAINING_ADAPTER_ENV = "UNSLOTH_H3_IMAGE_TRAINING_ADAPTER"
EVAL_ENV = "UNSLOTH_H3_IMAGE_TRAIN_EVAL"
OSTRIS_ADAPTER = (
    "ostris/minimax_h3_training_adapter",
    "minimax_h3_image_training_adapter_v1.safetensors",
)
# Pinned: the repo keeps receiving uploads, so a run fetches the exact file the defaults were tuned with
# (sha256 45a5d60d...68bb60, the file ai-toolkit's minimax_h3_image preset names).
OSTRIS_ADAPTER_REVISION = "e4e119caa391053286145e48ebbaf0e0c00293d3"


def training_adapter_path(hf_token: Optional[str] = None, on_note = None) -> Optional[str]:
    value = (os.environ.get(TRAINING_ADAPTER_ENV) or "auto").strip()
    if value.lower() in ("0", "off", "none", "false", "no"):
        return None
    if value.lower() in ("auto", "ostris"):
        from huggingface_hub import hf_hub_download

        from core.inference.diffusion import hub_cache_dir

        repo, filename = OSTRIS_ADAPTER
        try:
            return hf_hub_download(
                repo,
                filename,
                revision = OSTRIS_ADAPTER_REVISION,
                token = hf_token,
                cache_dir = hub_cache_dir(),
            )
        except Exception as exc:  # noqa: BLE001 - auto trains without it
            if value.lower() == "ostris":
                raise
            if on_note is not None:
                on_note(
                    f"Training adapter {repo}/{filename} unavailable ({exc}); training without it."
                )
            return None
    if not Path(value).expanduser().is_file():
        raise ValueError(
            f"{TRAINING_ADAPTER_ENV}={value!r} is not 'auto', 'ostris', 'off' or a LoRA file"
        )
    return str(Path(value).expanduser())


def sigma_mode() -> str:
    value = (os.environ.get(SIGMA_MODE_ENV) or "grid").strip().lower()
    return "continuous" if value == "continuous" else "grid"


def resolve_h3_image_base_precision(
    requested: str,
    free_gb: Optional[float],
    int8_ok: bool = True,
) -> str:
    """bf16 or int8; ``auto`` (and nf4) picks by free memory."""
    mode = (requested or "auto").strip().lower()
    int8_missing = "MiniMax-H3-Image int8 training needs a CUDA GPU with a working torchao, which this machine does not have"
    if mode == "bf16":
        return mode
    if mode == "int8":
        if not int8_ok:
            raise ValueError(int8_missing + ".")
        return mode
    if mode in ("fp8", "mxfp8"):
        raise ValueError(
            f"base_precision={mode!r} is not supported for MiniMax-H3-Image training: use 'bf16', 'int8' or 'auto'."
        )
    if free_gb is not None and free_gb >= H3_IMAGE_BF16_MIN_FREE_GB:
        return "bf16"
    if not int8_ok:
        raise ValueError(
            f"{int8_missing}, and a bf16 base needs about {H3_IMAGE_BF16_MIN_FREE_GB:g} GB of free GPU memory."
        )
    return "int8"


def training_sigmas():
    """``[(sampler sigma, model timestep, head index)]`` for the 6 distilled steps, exactly as the sampler runs them."""
    from core.inference.diffusion_minimax_h3_image import (
        H3_IMAGE_SIGMAS,
        H3_IMAGE_STEPS,
        MiniMaxH3ImageScheduler,
        pdd_head_span,
    )

    sched = MiniMaxH3ImageScheduler(sample_sigmas = list(H3_IMAGE_SIGMAS))
    sched.set_timesteps(H3_IMAGE_STEPS, device = "cpu")
    sample_sigmas = sched.sigmas.cpu()
    out = []
    for k in range(H3_IMAGE_STEPS):
        start, stop = pdd_head_span(sched.model_sigmas[k], sample_sigmas, sched.head_grid)
        if (start, stop) != (k, k + 1):
            raise ValueError(
                f"step {k} of the 6-step schedule spans heads {start}..{stop - 1}, not one head"
            )
        out.append((float(sample_sigmas[k]), float(sched.timesteps[k]), k))
    return out


def canvas_for(width: int, height: int, resolution: int) -> tuple[int, int]:
    """The training canvas of one picture: ``resolution ** 2`` pixels at its aspect, sides multiples of 32."""
    from core.inference.diffusion_minimax_h3_image import H3_IMAGE_CANVAS_MULTIPLE as m

    scale = math.sqrt(float(resolution) * float(resolution) / float(width * height))
    w = max(m, int(round(width * scale / m)) * m)
    h = max(m, int(round(height * scale / m)) * m)
    return w, h


def load_picture(path: str, resolution: int):
    """Upright RGB, cover-resized (LANCZOS) and centre-cropped onto its canvas; float ``[1, H, W, 3]`` in [0, 1]."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    w, h = canvas_for(image.width, image.height, resolution)
    scale = max(w / image.width, h / image.height)
    sw, sh = max(w, math.ceil(image.width * scale)), max(h, math.ceil(image.height * scale))
    image = image.resize((sw, sh), resample = Image.LANCZOS)
    left, top = (sw - w) // 2, (sh - h) // 2
    image = image.crop((left, top, left + w, top + h))
    array = np.asarray(image, dtype = np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _hub_root(cfg, patterns) -> Path:
    from core.inference.diffusion import hub_cache_dir
    from core.inference.diffusion_minimax_h3_image import _hub_files

    repo = getattr(cfg, "fetch_base_model", None) or cfg.base_model
    return _hub_files(
        repo, patterns, hf_token = cfg.hf_token, cache_dir = hub_cache_dir(), local_files_only = False
    )


def text_encoder_repo(root: Path) -> str:
    """The conditioner repo the base's model_index names (the released default when it names none)."""
    from core.inference.diffusion_minimax_h3_image import H3_IMAGE_TEXT_ENCODER_REPO, _read_json

    index = _read_json(root / "model_index.json") if (root / "model_index.json").is_file() else {}
    entry = index.get("text_encoder")
    te_repo = None
    if isinstance(entry, list) and len(entry) > 2 and isinstance(entry[2], dict):
        te_repo = entry[2].get("pretrained_model_name_or_path")
    return te_repo or H3_IMAGE_TEXT_ENCODER_REPO


def text_encoder_scheme() -> str:
    """int8 (default) or bf16: the two encode different embeddings, so the cache keys them apart."""
    scheme = (os.environ.get(TEXT_ENCODER_ENV) or "int8").strip().lower()
    return "int8" if scheme == "int8" else "bf16"


def load_text_encoder(cfg, root: Path):
    """(encoder on the host, tokenizer, processor). Module-level so tests hand in a tiny conditioner."""
    import torch
    from transformers import AutoTokenizer

    from core.inference.diffusion import hub_cache_dir
    from core.inference.diffusion_minimax_h3_image import load_h3_image_text_encoder

    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    processor = None
    if (root / "processor").is_dir():
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(root / "processor"))
        except Exception:  # noqa: BLE001 - text-only captions do not need it on transformers 4.x
            processor = None
    encoder, _engaged = load_h3_image_text_encoder(
        "int8" if text_encoder_scheme() == "int8" else None,
        dtype = torch.bfloat16,
        text_encoder_repo = text_encoder_repo(root),
        hf_token = cfg.hf_token,
        cache_dir = hub_cache_dir(),
        local_files_only = False,
    )
    return encoder, tokenizer, processor


def release_text_encoder(encoder) -> None:
    """Free the conditioner's storage in place (it is never needed again in this run)."""
    from core.inference.diffusion_minimax_h3_image import free_module_storage
    free_module_storage(encoder)


def encode_captions(encoder, tokenizer, processor, captions, device) -> dict:
    """caption -> ``hidden_states[50]`` ``(1, L, text_dim)`` float32 on the host, tokenized as the pipeline does."""
    import torch

    from core.inference.diffusion_minimax_h3_image import (
        H3_IMAGE_EMPTY_TOKEN,
        H3_IMAGE_TEXT_ENCODER_LAYER,
        qwen3vl_hidden_state,
    )

    out = {}
    with torch.no_grad():
        for caption in captions:
            ids = tokenizer(caption, add_special_tokens = False)["input_ids"] if caption else []
            if not ids:
                ids = [int(tokenizer.convert_tokens_to_ids(H3_IMAGE_EMPTY_TOKEN))]
            hidden = qwen3vl_hidden_state(
                encoder, processor, ids, {}, layer = H3_IMAGE_TEXT_ENCODER_LAYER, device = device
            )
            out[caption] = hidden.to("cpu", torch.float32)
    return out


def encode_latent_stats(vae, picture, device) -> tuple[Any, Any]:
    """Normalised posterior (mean, std) ``(1, C, h, w)`` float32 on the host: each step draws a fresh sample."""
    import torch

    from core.inference.diffusion_minimax_h3_image import (
        H3_IMAGE_PIXEL_MEAN,
        H3_IMAGE_PIXEL_STD,
        reference_vae_pixels,
    )

    mean = torch.tensor(H3_IMAGE_PIXEL_MEAN, device = device).view(1, -1, 1, 1)
    std = torch.tensor(H3_IMAGE_PIXEL_STD, device = device).view(1, -1, 1, 1)
    lmean = torch.tensor(vae.config.latents_mean, device = device).view(1, -1, 1, 1)
    lstd = torch.tensor(vae.config.latents_std, device = device).view(1, -1, 1, 1)
    with torch.no_grad():
        posterior = vae.encode(
            reference_vae_pixels(picture.to(device), torch.float32, mean, std), return_dict = False
        )[0]
        moments = posterior.parameters.float()
        mu, logvar = moments.chunk(2, dim = 1)
        sd = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
    return ((mu - lmean) / lstd).cpu(), (sd / lstd).cpu()


def load_transformer(
    root: Path,
    device,
    base_precision: str,
    table: bool,
    lora_deltas: Optional[dict] = None,
):
    """int8 stays on the host until quantized after the adapter attaches."""
    import torch

    from core.inference.diffusion_minimax_h3_image import (
        install_h3_image_forward,
        install_pdd_head_bank,
        load_h3_image_transformer,
    )

    keep = base_precision == "bf16" and device != "cpu"
    transformer, _stats = load_h3_image_transformer(
        root,
        dtype = torch.bfloat16 if device != "cpu" else torch.float32,
        scheme = None,
        device = device,
        keep_on_device = keep,
        table = table,
        lora_deltas = lora_deltas,
    )
    install_h3_image_forward(transformer)
    install_pdd_head_bank(transformer)
    if device == "cpu":
        transformer.to("cpu")
    return transformer


def _save_lora(
    out_dir: str,
    layers: dict,
    adapter_config: Optional[dict] = None,
) -> None:
    """diffusers' single-file layout (``transformer.`` keys, adapter config in the metadata so alpha survives)."""
    import json

    from safetensors.torch import save_file

    Path(out_dir).mkdir(parents = True, exist_ok = True)
    state = {f"transformer.{k}": v.to("cpu").contiguous() for k, v in layers.items()}
    metadata = {"format": "pt", "unsloth_family": H3_IMAGE_TRAIN_FAMILY}
    if adapter_config:
        packed = {
            f"transformer.{key}": list(value) if isinstance(value, set) else value
            for key, value in adapter_config.items()
        }
        metadata["lora_adapter_metadata"] = json.dumps(
            packed, indent = 2, sort_keys = True, default = str
        )
    save_file(state, str(Path(out_dir) / DEFAULT_LORA_FILENAME), metadata = metadata)


class _Layouts:
    """Packed ``[text | image]`` layout per (text rows, latent h, latent w), host copy + device copy."""

    def __init__(self, patch, device) -> None:
        self.patch = patch
        self.device = device
        self._cache: dict = {}

    def get(self, num_text: int, lh: int, lw: int) -> dict:
        import torch

        from core.inference.diffusion_minimax_h3_image import H3_IMAGE_TEXT_TAG, build_packed_layout

        key = (num_text, lh, lw)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        tags = torch.full((num_text,), H3_IMAGE_TEXT_TAG, dtype = torch.long)
        pos, token_tags, image_idx, text_idx, _n_ref = build_packed_layout(
            tags, [], lh, lw, self.patch
        )
        dev = torch.device(self.device)
        pos_dtype = torch.float64 if dev.type in ("cpu", "cuda") else torch.float32
        layout = {
            "position_ids": pos.to(dev, pos_dtype),
            "token_tags": token_tags.to(dev),
            "image_indices": image_idx.to(dev),
            "text_indices": text_idx.to(dev),
            "seq": int(pos.shape[0]),
        }
        self._cache[key] = layout
        return layout


def denoise_velocity(
    transformer,
    layout,
    rows,
    embeds,
    text_embeds,
    t: float,
    head_index: int,
    head = None,
):
    """One DiT call at timestep ``t`` on the generated rows only: the velocity ``(n_rows, C * patch)``."""
    import torch

    timestep = torch.tensor([float(t)], dtype = torch.float32, device = rows.device)
    index = torch.zeros(layout["seq"], dtype = torch.long, device = rows.device)
    extra = {"output_indices": layout["image_indices"], "text_embeds": text_embeds}
    if head is not None:
        extra["head"] = head
    out = transformer(
        hidden_states = rows[None],
        encoder_hidden_states = embeds,
        timestep = timestep,
        timestep_indices = index,
        token_tags = layout["token_tags"],
        position_ids = layout["position_ids"],
        image_indices = layout["image_indices"],
        text_indices = layout["text_indices"],
        pdd_step_index = int(head_index),
        return_dict = False,
        **extra,
    )[0]
    return out[0]


def run_h3_image_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    cfg = config.normalized()
    if cfg.resolved_family != H3_IMAGE_TRAIN_FAMILY:
        raise ValueError(
            f"This trainer is for {H3_IMAGE_TRAIN_FAMILY}, not {cfg.resolved_family!r}."
        )
    reason = h3_train_unsupported_reason(cfg)
    if reason:
        raise ValueError(reason)
    cfg = replace(cfg, **train_recipe_overrides(cfg))

    import torch

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    save_on_stop = True

    def _check_stop() -> bool:
        nonlocal save_on_stop
        if should_stop is None:
            return False
        sig = should_stop()
        if not sig:
            return False
        if isinstance(sig, dict) and sig.get("save") is False:
            save_on_stop = False
        return True

    device = resolve_train_device()
    if device == "cuda" and not flow_bf16_trainable():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU (Ampere or newer); this CUDA device does not support bf16."
        )
    if device == "xpu" and not native_bf16_supported_xpu():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU; this XPU device does not support bf16 natively."
        )
    _assert_trusted_base_model(cfg.base_model, family = H3_IMAGE_TRAIN_FAMILY)
    pairs = discover_image_caption_pairs(
        cfg.data_dir,
        instance_prompt = cfg.instance_prompt,
        caption_column = cfg.caption_column,
        verify_images = True,
    )
    cfg = replace(cfg, train_steps = resolve_train_steps(cfg, len(pairs)), num_epochs = 0)
    _emit(
        on_event, "model_load_started", num_images = len(pairs), fetch_repo = training_fetch_repo(cfg)
    )
    out_dir = Path(cfg.output_dir).expanduser()
    if _check_stop():
        _emit(
            on_event, "complete", output_dir = str(out_dir), lora_path = None, stopped = True, steps_run = 0
        )
        return str(out_dir)
    perf_snap = _apply_perf_flags(cfg, device)
    try:
        return _train(cfg, pairs, rng, device, on_event, _check_stop, lambda: save_on_stop)
    finally:
        _restore_perf_flags(perf_snap)


def _empty(device) -> None:
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "xpu":
        torch.xpu.empty_cache()


def _free_gb(device) -> Optional[float]:
    import torch
    try:
        if device == "cuda":
            free, _total = torch.cuda.mem_get_info()
            return free / 1024**3
        if device == "xpu":
            props = torch.xpu.get_device_properties()
            return (props.total_memory - torch.xpu.memory_reserved()) / 1024**3
    except Exception:  # noqa: BLE001 - unknown: int8
        return None
    return None


# Bump when what a picture entry holds (or how it is encoded) changes.
_H3_IMAGE_LATENT_CACHE_VERSION = "1"


def _open_h3_image_cache(cfg, root: Path, on_event):
    """The persistent conditioning cache for this run, or None (switched off, or unusable: never fatal)."""
    from core.inference.diffusion_minimax_h3_image import H3_IMAGE_TEXT_ENCODER_LAYER
    from core.training.diffusion_train_extras import (
        PersistentConditioningCache,
        compact_cache_namespace,
        effective_cond_cache_dir,
        source_revision,
    )

    cache_dir = effective_cond_cache_dir(getattr(cfg, "cond_cache_dir", None))
    if not cache_dir:
        return None
    try:
        repo = getattr(cfg, "fetch_base_model", None) or cfg.base_model
        namespace = compact_cache_namespace(
            H3_IMAGE_TRAIN_FAMILY, cfg.base_model, source_revision(repo)
        )
        # The conditioner is its own repo: its revision, the int8 / bf16 scheme and the read layer key captions.
        te_repo = text_encoder_repo(root)
        encoder = compact_cache_namespace(
            "qwen3vl",
            te_repo,
            source_revision(te_repo),
            text_encoder_scheme(),
            f"hs{H3_IMAGE_TEXT_ENCODER_LAYER}",
        )
        return PersistentConditioningCache(
            cache_dir, namespace, cfg.resolution, text_encoder = encoder
        )
    except Exception as exc:  # noqa: BLE001 -- the cache is an optimisation, never fatal
        _emit(on_event, "warning", message = f"conditioning cache disabled: {exc}")
        return None


def _picture_key(pcache, path: str, resolution: int) -> Optional[str]:
    """The canvas follows from the picture's own size (in its bytes) and the resolution; None when uncached."""
    if pcache is None:
        return None
    try:
        return pcache.entry_key(
            "h3img", _H3_IMAGE_LATENT_CACHE_VERSION, str(path), f"r{int(resolution)}"
        )
    except OSError:
        # load_picture raises the real error for an unreadable picture.
        return None


def _cache_get(pcache, key):
    if pcache is None or key is None:
        return None
    try:
        return pcache.get(key)
    except Exception:  # noqa: BLE001 -- an unreadable entry is a miss
        return None


def _cache_put(pcache, key, tensors) -> None:
    if pcache is None or key is None:
        return
    try:
        pcache.put(key, tensors)
    except Exception:  # noqa: BLE001 -- disk-full etc.: the run still trains
        pass


def _prepare_h3_image_conditioning(
    cfg, root, paths, texts, device, on_event, _check_stop, *, need_vae
):
    """(embeds_by_text, per-picture (mean, std), vae or None), or None on a stop request.

    Captions and sample prompts are served from the persistent cache when present; the conditioner loads only for
    the misses, so a warm run never loads it. The VAE loads for latent misses and for the preview decode. Loads and
    encodes run RNG-isolated, so a warm run and a cold run hand the same training streams on."""
    from core.inference import diffusion_minimax_h3_image as h3i
    from core.training.diffusion_samples import isolated_sampling

    pcache = _open_h3_image_cache(cfg, root, on_event)
    embeds_by_text: dict = {}
    if pcache is not None:
        for text in texts:
            hit = _cache_get(pcache, pcache.text_key(text))
            if hit is not None:
                embeds_by_text[text] = hit[0]
    missing = [t for t in texts if t not in embeds_by_text]
    if missing:
        with isolated_sampling(device, False):
            encoder, tokenizer, processor = load_text_encoder(cfg, root)
            encoder.to(device)
            # A first run reads the conditioner revision before its download: write under the real one,
            # dropping anything served under a namespace the load has moved away from.
            if pcache is not None:
                fresh = _open_h3_image_cache(cfg, root, on_event)
                if fresh is not None and (fresh.family, fresh.text_encoder) != (
                    pcache.family,
                    pcache.text_encoder,
                ):
                    pcache = fresh
                    embeds_by_text.clear()
                    for text in texts:
                        hit = _cache_get(pcache, pcache.text_key(text))
                        if hit is not None:
                            embeds_by_text[text] = hit[0]
                    missing = [t for t in texts if t not in embeds_by_text]
            encoded = encode_captions(encoder, tokenizer, processor, missing, device)
        for text in missing:
            embeds_by_text[text] = encoded[text]
            if pcache is not None:
                _cache_put(pcache, pcache.text_key(text), (encoded[text],))
        release_text_encoder(encoder)
        del encoder
        _empty(device)
    embeds_by_text = {t: embeds_by_text[t] for t in texts}
    _emit(on_event, "preparing", stage = "encode_prompts", done = len(texts), total = len(texts))
    if _check_stop():
        return None

    # Phase 2: VAE posterior stats per picture (the VAE stays on the host afterwards for the preview decode).
    entries: dict = {}
    if pcache is not None:
        for i, path in enumerate(paths):
            hit = _cache_get(pcache, _picture_key(pcache, path, cfg.resolution))
            if hit is not None:
                entries[i] = tuple(hit)
    vae = None
    if len(entries) < len(paths) or need_vae:
        with isolated_sampling(device, False):
            vae = h3i.load_h3_image_vae(root)
    cache = []
    if len(entries) < len(paths):
        vae.to(device)
    for i, path in enumerate(paths):
        entry = entries.get(i)
        if entry is None:
            picture = load_picture(path, cfg.resolution)
            with isolated_sampling(device, False):
                entry = encode_latent_stats(vae, picture, device)
            _cache_put(pcache, _picture_key(pcache, path, cfg.resolution), entry)
        cache.append(entry)
        _emit(on_event, "preparing", stage = "cache_latents", done = i + 1, total = len(paths))
        if _check_stop():
            return None
    if vae is not None:
        vae.to("cpu")
    if not need_vae:
        del vae
        vae = None
        gc.collect()
    _empty(device)
    return embeds_by_text, cache, vae


def _train(cfg, pairs, rng, device, on_event, _check_stop, _save_on_stop):
    import torch
    import torch.nn.functional as F
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from core.inference import diffusion_minimax_h3_image as h3i
    from core.training.diffusion_dit_trainer import _int8_quantize_base, _make_optimizer
    from core.training.diffusion_samples import isolated_sampling, plan_samples, run_sample_round

    # Resolved before any download: a precision this machine cannot run fails here, not after 90 GB of fetches.
    if device in ("cuda", "xpu"):
        from core.training.diffusion_train_common import train_precision_modes
        base_precision = resolve_h3_image_base_precision(
            cfg.base_precision, _free_gb(device), int8_ok = "int8" in train_precision_modes()[0]
        )
    else:
        base_precision = "bf16"  # CPU (tests): torchao int8 is a CUDA path
    out_dir = Path(cfg.output_dir).expanduser()
    paths = [p for p, _ in pairs]
    captions = [c for _, c in pairs]
    sample_plan = plan_samples(cfg, H3_IMAGE_TRAIN_FAMILY, captions)
    mode = sigma_mode()

    def _stopped_early() -> str:
        _emit(
            on_event, "complete", output_dir = str(out_dir), lora_path = None, stopped = True, steps_run = 0
        )
        return str(out_dir)

    root = _hub_root(cfg, h3i.H3_IMAGE_BASE_PATTERNS)
    texts = list(dict.fromkeys(captions + list(sample_plan.prompts if sample_plan else ())))
    prepared = _prepare_h3_image_conditioning(
        cfg, root, paths, texts, device, on_event, _check_stop, need_vae = sample_plan is not None
    )
    if prepared is None:
        return _stopped_early()
    embeds_by_text, cache, vae = prepared

    # Phase 3: the DiT, frozen, with the adapter on the trunk.
    dense_adaln = mode == "continuous"
    _emit(on_event, "model_load_started", fetch_repo = training_fetch_repo(cfg))
    dit_root = _hub_root(
        cfg, tuple(h3i.H3_IMAGE_BASE_PATTERNS) + tuple(h3i.H3_IMAGE_DENSE_DIT_PATTERNS)
    )
    adapter = training_adapter_path(
        cfg.hf_token, on_note = lambda m: _emit(on_event, "warning", message = m)
    )
    adapter_deltas = h3i.read_lora_deltas([(adapter, 1.0)]) if adapter else None
    transformer = load_transformer(
        dit_root, device, base_precision, table = not dense_adaln, lora_deltas = adapter_deltas
    )
    transformer.requires_grad_(False)
    targets: Any = (
        H3_IMAGE_TARGETS
        if tuple(cfg.lora_target_modules) == DEFAULT_LORA_TARGETS
        else list(cfg.lora_target_modules)
    )
    lora_config = LoraConfig(
        r = cfg.lora_rank,
        lora_alpha = cfg.lora_alpha,
        lora_dropout = cfg.lora_dropout,
        init_lora_weights = "gaussian",
        target_modules = targets,
    )
    if hasattr(transformer, "add_adapter"):
        transformer.add_adapter(lora_config)
    else:
        # The vendored class (diffusers without MiniMax-H3-Image) carries no PeftAdapterMixin.
        from peft import inject_adapter_in_model
        inject_adapter_in_model(lora_config, transformer, adapter_name = "default")
        if not isinstance(getattr(transformer, "peft_config", None), dict):
            transformer.peft_config = {"default": lora_config}
    if cfg.gradient_checkpointing:
        enable_diffusion_gradient_checkpointing(transformer)
    cast_training_params(transformer, dtype = torch.float32)
    if base_precision == "int8":
        _int8_quantize_base(transformer, H3_IMAGE_TRAIN_FAMILY, device = device)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if not lora_params:
        raise ValueError(
            f"No LoRA target matched the MiniMax-H3-Image transformer for {targets!r}."
        )
    # A refiner the adapter does not touch gives the same rows every step: refine each caption once.
    refiner_adapted = any(
        p.requires_grad
        for n, p in transformer.named_parameters()
        if n.startswith(("token_refiner.", "context_embedder."))
    )
    ema = LoRAEMA(transformer, decay = cfg.ema_decay) if getattr(cfg, "ema_decay", 0.0) else None
    optimizer = _make_optimizer(lora_params, cfg.learning_rate)
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = cfg.lr_warmup_steps,
        num_training_steps = cfg.train_steps,
    )
    compute_dtype = h3i._module_dtype(transformer, skip_fp32 = True)
    patch = tuple(transformer.config.patch_size)
    layouts = _Layouts(patch, device)
    grid_steps = training_sigmas()
    bank = getattr(transformer, h3i.PDD_BANK_ATTR, None)
    if mode == "continuous":
        bank = tuple(t.to(device) for t in bank)
        sample_sigmas = torch.tensor([s for s, _t, _k in grid_steps] + [0.0])

    def _embeds(text: str):
        return embeds_by_text[text].to(device, compute_dtype)

    refined: dict = {}

    def _refined(text: str):
        if refiner_adapted:
            return h3i.refined_text(transformer, _embeds(text))
        hit = refined.get(text)
        if hit is None:
            with torch.no_grad():
                hit = h3i.refined_text(transformer, _embeds(text)).detach()
            refined[text] = hit
        return hit

    _emit(
        on_event,
        "model_load_completed",
        compiled = False,
        base_precision = base_precision,
        sigma_mode = mode,
        training_adapter = Path(adapter).name if adapter else None,
        train_sigmas = [round(sv, 6) for sv, _t, _k in grid_steps],
        sequence_length = max(
            layouts.get(int(embeds_by_text[c].shape[1]), a.shape[-2], a.shape[-1])["seq"]
            for c, (a, _b) in zip(captions, cache)
        ),
    )

    def _render(index: int, prompt: str):
        """One 6-step preview at the plan's canvas, decoded to [-1, 1]."""
        from core.training.diffusion_samples import initial_noise

        side = max(
            256,
            (int(sample_plan.resolution) // h3i.H3_IMAGE_CANVAS_MULTIPLE)
            * h3i.H3_IMAGE_CANVAS_MULTIPLE,
        )
        lh = lw = side // h3i.H3_IMAGE_VAE_SCALE
        channels = int(vae.config.latent_channels)
        x = h3i.patchify(initial_noise(sample_plan, index, (1, channels, 1, lh, lw), device), patch)
        layout = layouts.get(int(embeds_by_text[prompt].shape[1]), lh, lw)
        text_embeds = _refined(prompt)
        sched = h3i.MiniMaxH3ImageScheduler(sample_sigmas = list(h3i.H3_IMAGE_SIGMAS))
        sched.set_timesteps(h3i.H3_IMAGE_STEPS, device = "cpu")
        for k, t in enumerate(sched.timesteps):
            v = denoise_velocity(
                transformer, layout, x.to(compute_dtype), _embeds(prompt), text_embeds, float(t), k
            )
            x = sched.step(v.to(compute_dtype).float(), t, x)[0]
        vae.to(device)
        try:
            lmean = torch.tensor(vae.config.latents_mean, device = device).view(1, -1, 1, 1)
            lstd = torch.tensor(vae.config.latents_std, device = device).view(1, -1, 1, 1)
            z = h3i.unpatchify(x, lh, lw, channels, patch) * lstd + lmean
            image = vae.decode(z.to(torch.float32), return_dict = False)[0].float()
        finally:
            vae.to("cpu")
        mean = torch.tensor(h3i.H3_IMAGE_PIXEL_MEAN, device = image.device).view(1, -1, 1, 1)
        std = torch.tensor(h3i.H3_IMAGE_PIXEL_STD, device = image.device).view(1, -1, 1, 1)
        return ((image * std + mean).clamp(0, 1) * 2.0 - 1.0).cpu()

    def _sample_round(step: int) -> None:
        if sample_plan is None:
            return
        try:
            with isolated_sampling(device, False):
                run_sample_round(
                    sample_plan, step, _render, on_event, _emit, stop_requested = _check_stop
                )
        except Exception as exc:  # noqa: BLE001 - a preview never ends the run
            _emit(on_event, "warning", message = f"Sample images skipped: {exc}")

    def _target_error(tag: str) -> None:
        """Diagnostic (UNSLOTH_H3_IMAGE_TRAIN_EVAL=1): per-head x0-estimate error before and after training."""
        if (os.environ.get(EVAL_ENV) or "").strip() != "1":
            return
        gen = torch.Generator().manual_seed(1234)
        was_training = transformer.training
        transformer.eval()
        per_head = {}
        with torch.no_grad():
            for i, (mean, _std) in enumerate(cache):
                x0 = mean.to(device)[:, :, None]
                noise = torch.randn(tuple(x0.shape), generator = gen).to(device)
                layout = layouts.get(
                    int(embeds_by_text[captions[i]].shape[1]), mean.shape[-2], mean.shape[-1]
                )
                for sigma, t, head_index in grid_steps:
                    x_t = (1.0 - sigma) * x0 + sigma * noise
                    v = denoise_velocity(
                        transformer,
                        layout,
                        h3i.patchify(x_t, patch).to(compute_dtype),
                        _embeds(captions[i]),
                        _refined(captions[i]),
                        t,
                        head_index,
                    ).float()
                    x0_hat = h3i.patchify(x_t, patch) + sigma * v
                    err = float((x0_hat - h3i.patchify(x0, patch)).pow(2).mean())
                    per_head.setdefault(head_index, []).append(err)
        if was_training:
            transformer.train()
        _emit(
            on_event,
            "eval",
            tag = tag,
            x0_error_by_head = {
                str(k): round(sum(v) / len(v), 6) for k, v in sorted(per_head.items())
            },
        )

    _target_error("before")
    _sample_round(0)
    index_sampler = PermutationBatchSampler(len(paths), rng)
    transformer.train()
    stopped = False
    running_loss = 0.0
    t_start = time.time()
    t_steady = None
    done = 0
    per_head: dict = {}
    reset_training_peak_memory(device)
    for opt_step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none = True)
        step_acc = None
        for _ in range(cfg.gradient_accumulation_steps):
            index = index_sampler.next_batch(1)[0]
            mean, std = cache[index]
            mean = mean.to(device, non_blocking = True)
            std = std.to(device, non_blocking = True)
            clean = (mean + std * torch.randn_like(mean))[:, :, None]
            head = None
            if mode == "grid":
                sigma, t, head_index = grid_steps[rng.randrange(len(grid_steps))]
            else:
                sigma = 1.0 - rng.random() * 0.999
                t = float(torch.tensor(1.0 - sigma, dtype = torch.float32))
                head_index = h3i.pdd_head_span(sigma, sample_sigmas)[0]
                head = h3i.pdd_head_weights(bank, sigma, sample_sigmas)
            noise = torch.randn_like(clean)
            noisy = (1.0 - sigma) * clean + sigma * noise
            rows = h3i.patchify(noisy, patch).to(compute_dtype)
            # DATA-ward velocity: x0 = x_t + sigma * v.
            target = h3i.patchify(clean - noise, patch)
            caption = captions[index]
            layout = layouts.get(
                int(embeds_by_text[caption].shape[1]), mean.shape[-2], mean.shape[-1]
            )
            pred = denoise_velocity(
                transformer, layout, rows, _embeds(caption), _refined(caption), t, head_index, head
            )
            loss = F.mse_loss(pred.float(), target.float())
            (loss / cfg.gradient_accumulation_steps).backward()
            step_acc = accumulate_loss(step_acc, loss, cfg.gradient_accumulation_steps)
            stats = per_head.setdefault(head_index, [0.0, 0])
            stats[0] += float(loss.detach())
            stats[1] += 1
        grad_norm = None
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm))
        step_loss = loss_value(step_acc)
        optimizer.step()
        lr_sched.step()
        if ema is not None:
            ema.update(transformer)
        running_loss += step_loss
        done = opt_step + 1
        now = time.time()
        if done == 1:
            t_steady = now
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            per_step = cfg.gradient_accumulation_steps
            if t_steady is not None and done > 1:
                sps = round((done - 1) * per_step / max(now - t_steady, 1e-6), 3)
            else:
                sps = round(done * per_step / max(now - t_start, 1e-6), 3)
            peak_gb = training_peak_memory_gb(device)
            _emit(
                on_event,
                "progress",
                step = done,
                total_steps = cfg.train_steps,
                loss = round(step_loss, 5),
                avg_loss = round(running_loss / done, 5),
                learning_rate = lr_sched.get_last_lr()[0],
                grad_norm = round(grad_norm, 5) if grad_norm is not None else None,
                samples_per_second = sps,
                peak_memory_gb = peak_gb or None,
                head_losses = {
                    str(k): round(v[0] / max(1, v[1]), 5) for k, v in sorted(per_head.items())
                },
            )
        if sample_plan is not None and sample_plan.due(done, cfg.train_steps):
            transformer.eval()
            _sample_round(done)
            transformer.train()
        if _check_stop():
            stopped = True
            break

    _target_error("after")
    lora_path: Optional[str] = None
    catalog_path: Optional[str] = None
    ema_path: Optional[str] = None
    if not (stopped and not _save_on_stop()):
        layers = get_peft_model_state_dict(transformer)
        adapter_config = dict(transformer.peft_config["default"].to_dict())
        _save_lora(str(out_dir), layers, adapter_config)
        lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
        # The Images LoRA picker, Deploy and Export read the catalog copy.
        catalog_path = _publish_to_lora_catalog(lora_path, cfg, done)
        if ema is not None and ema.updates > 0:
            try:
                ema_dir = save_ema_adapter(
                    ema,
                    transformer,
                    lambda _pipe, d, l: _save_lora(d, l, adapter_config),
                    str(out_dir),
                )
                ema_path = str(Path(ema_dir) / DEFAULT_LORA_FILENAME)
            except Exception as exc:  # noqa: BLE001 -- the primary adapter is already saved
                _emit(on_event, "warning", message = f"EMA adapter save failed: {exc}")
    _emit(
        on_event,
        "complete",
        output_dir = str(out_dir),
        lora_path = lora_path,
        ema_path = ema_path,
        catalog_path = catalog_path,
        family = cfg.resolved_family,
        base_model = cfg.base_model,
        stopped = stopped,
        steps_run = done if cfg.train_steps else 0,
        wall_seconds = round(time.time() - t_start, 1),
        base_precision = base_precision,
    )
    return str(out_dir)
