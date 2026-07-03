# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion LoRA training for Unsloth Studio (text-to-image, SDXL).

Trains a LoRA adapter on the U-Net of an SDXL pipeline from an image + caption dataset
and exports it as a diffusers-format ``.safetensors`` that the Studio diffusion backend
(and any diffusers pipeline via ``load_lora_weights``) can load.

Design:
- The pure helpers (dataset discovery, config normalisation, SDXL add-time-ids) have no
  torch/diffusers dependency at call time and are unit-tested without a GPU.
- ``run_diffusion_lora_training`` is the training loop. It reports progress through an
  ``on_event`` callback whose payloads match the training worker's event protocol
  (``{"type": ..., "ts": ...}``) so it can be spawned as a subprocess and streamed to the
  UI, and it polls a ``should_stop`` callback so a stop request ends it cleanly (with a
  partial save).
- ``run_diffusion_training_process`` is the thin mp.Queue adapter, and ``main`` is a CLI.

Only the SDXL (U-Net) architecture is trained here; DiT families (FLUX / Qwen-Image /
Z-Image) are a follow-up. The exported adapter is loaded by the existing LoRA path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

# Default LoRA target modules: the U-Net attention projections. These are the standard
# SDXL LoRA targets (the diffusers/kohya convention) and keep the adapter small.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_CAPTION_EXTS = (".txt", ".caption")
# diffusers' canonical single-file LoRA name, so load_lora_weights(dir) finds it.
DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"

# Families Studio can LOAD but not train (DiT architectures). A base-model name that
# clearly belongs to one is refused in normalized(), so a wrong pick fails at start
# (an instant HTTP 400 through the API) instead of minutes later inside
# StableDiffusionXLPipeline.from_pretrained. Single tokens match on word boundaries;
# hyphenated markers match as substrings of the hyphen-condensed name.
_NON_SDXL_TOKENS = frozenset({"flux", "sd3", "kontext", "pixart", "sana", "lumina", "cogview"})
_NON_SDXL_PHRASES = ("qwen-image", "z-image", "stable-diffusion-3", "hunyuan-dit")
_ONLY_SDXL_HINT = (
    "Only SDXL bases can be trained right now (e.g. stabilityai/stable-diffusion-xl-base-1.0 "
    "or stabilityai/sdxl-turbo). Other families can load LoRAs but not train them yet."
)

EventCb = Callable[[dict[str, Any]], None]
# Returns a falsy value to keep training, or a truthy stop signal: bare True, or a dict
# that may carry ``save=False`` to cancel without saving a partial adapter.
StopCb = Callable[[], Any]


@dataclass
class DiffusionLoraConfig:
    """Everything a diffusion LoRA training run needs. Only ``base_model`` /
    ``data_dir`` / ``output_dir`` are required; the rest have sensible defaults."""

    base_model: str
    data_dir: str
    output_dir: str
    # Dreambooth-style caption applied to any image without its own caption. Required if
    # the dataset has no captions.jsonl / sidecar files.
    instance_prompt: Optional[str] = None
    resolution: int = 1024
    train_steps: int = 500
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lora_rank: int = 16
    lora_alpha: Optional[int] = None  # defaults to lora_rank
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    seed: int = 42
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    snr_gamma: Optional[float] = 5.0  # min-SNR loss weighting; None disables
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    center_crop: bool = False
    random_flip: bool = True
    caption_column: str = "text"  # column in metadata.jsonl
    adapter_name: str = "default"
    hf_token: Optional[str] = None
    # How often to emit a progress event (in optimizer steps).
    log_every: int = 1

    def normalized(self) -> "DiffusionLoraConfig":
        """Return a copy with derived/validated fields filled in. Raises ValueError on a
        request that cannot train (bad numbers, or no caption source).

        Also coerces values that arrive as strings/blanks through the Studio config path
        (``learning_rate`` is preserved as a string there; ``hf_token`` defaults to "")."""
        assert_trainable_base_model(self.base_model)
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.lora_alpha is not None and self.lora_alpha < 1:
            raise ValueError(
                "lora_alpha must be >= 1 (a zero/negative alpha scales the adapter to nothing)"
            )
        if self.resolution < 64 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a multiple of 8 and >= 64")
        if self.mixed_precision not in ("bf16", "fp16", "no"):
            raise ValueError("mixed_precision must be one of bf16 / fp16 / no")
        # learning_rate can arrive as a string ("1e-4") from the Studio config path, which
        # preserves it as a string after validation; coerce so AdamW receives a float.
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"learning_rate must be a number, got {self.learning_rate!r}") from exc
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        alpha = self.lora_alpha if self.lora_alpha is not None else self.lora_rank
        targets = tuple(self.lora_target_modules) or DEFAULT_LORA_TARGETS
        # A blank Hub token (the Studio default when none is configured) must load
        # anonymously, not as an explicit empty credential.
        token = self.hf_token.strip() if isinstance(self.hf_token, str) else self.hf_token
        return replace(
            self,
            learning_rate = learning_rate,
            lora_alpha = alpha,
            lora_target_modules = targets,
            max_grad_norm = float(self.max_grad_norm),
            hf_token = token or None,
        )


def discover_image_caption_pairs(
    data_dir: str | os.PathLike[str],
    *,
    instance_prompt: Optional[str] = None,
    caption_column: str = "text",
) -> list[tuple[str, str]]:
    """Resolve ``(image_path, caption)`` pairs from a dataset directory.

    Caption sources, in priority order per image:
      1. a ``metadata.jsonl`` / ``captions.jsonl`` row keyed by ``file_name`` (or ``image``)
         carrying the caption in ``caption_column`` (default ``text``),
      2. a per-image sidecar ``<stem>.txt`` / ``<stem>.caption``,
      3. ``instance_prompt`` (dreambooth) for any remaining image.

    Images with no caption from any source are skipped. Pure filesystem + JSON, so it is
    unit-testable without torch. Raises FileNotFoundError for a missing dir and ValueError
    when nothing is captionable.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")

    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    # 1. metadata.jsonl / captions.jsonl (either name accepted).
    meta_caption: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = root / meta_name
        if not meta_path.is_file():
            continue
        for line in meta_path.read_text(encoding = "utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("file_name") or row.get("image") or row.get("file")
            cap = row.get(caption_column)
            if isinstance(key, str) and isinstance(cap, str) and cap.strip():
                meta_caption[Path(key).name] = cap.strip()
        break  # first present metadata file wins

    pairs: list[tuple[str, str]] = []
    for img in images:
        caption = meta_caption.get(img.name)
        if caption is None:
            for ext in _CAPTION_EXTS:
                sidecar = img.with_suffix(ext)
                if sidecar.is_file():
                    text = sidecar.read_text(encoding = "utf-8").strip()
                    if text:
                        caption = text
                        break
        if caption is None and instance_prompt and instance_prompt.strip():
            caption = instance_prompt.strip()
        if caption:
            pairs.append((str(img), caption))

    if not pairs:
        raise ValueError(
            f"No captioned images found under {data_dir}. Provide captions via a "
            f"metadata.jsonl, per-image .txt sidecars, or an instance_prompt."
        )
    return pairs


def compute_sdxl_add_time_ids(resolution: int) -> tuple[int, int, int, int, int, int]:
    """SDXL micro-conditioning ``add_time_ids`` for a square ``resolution`` train crop:
    (original_h, original_w, crop_top, crop_left, target_h, target_w). Pure; the trainer
    turns it into a tensor. No crop offset is applied (top-left = 0). The training loop
    derives per-image time-ids from the actual crop instead; this is the square default."""
    return (resolution, resolution, 0, 0, resolution, resolution)


def _emit(on_event: Optional[EventCb], type_: str, **kw: Any) -> None:
    if on_event is not None:
        on_event({"type": type_, "ts": time.time(), **kw})


def _load_image_tensor(
    path: str, resolution: int, center_crop: bool, random_flip: bool, rng: random.Random
) -> tuple[Any, tuple[int, int, int, int, int, int]]:
    """Load an image to a normalised CxHxW tensor in [-1, 1] (resize shorter side to
    ``resolution``, crop to a square, optional horizontal flip). No torchvision.

    Returns ``(tensor, add_time_ids)`` where add_time_ids is the SDXL micro-conditioning
    (original_h, original_w, crop_top, crop_left, target_h, target_w) for THIS sample, so
    the U-Net is told the real original size and crop offset (not a fixed uncropped
    square). EXIF orientation is applied first so rotated phone photos train upright."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    # Honour EXIF orientation before any geometry, or rotated camera/phone photos would
    # train in their stored (sideways) orientation, mismatched to their captions.
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    original_w, original_h = img.size
    scale = resolution / min(original_w, original_h)
    resized_w = max(resolution, round(original_w * scale))
    resized_h = max(resolution, round(original_h * scale))
    img = img.resize((resized_w, resized_h), Image.LANCZOS)
    if center_crop:
        left, top = (resized_w - resolution) // 2, (resized_h - resolution) // 2
    else:
        left = rng.randint(0, max(0, resized_w - resolution))
        top = rng.randint(0, max(0, resized_h - resolution))
    img = img.crop((left, top, left + resolution, top + resolution))
    crop_left = left
    if random_flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # A horizontal flip mirrors the crop's left origin, so report the mirrored offset
        # (diffusers' SDXL training scripts do the same) to keep the conditioning honest.
        crop_left = max(0, resized_w - resolution - left)
    arr = np.asarray(img, dtype = np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
    time_ids = (original_h, original_w, top, crop_left, resolution, resolution)
    return tensor, time_ids


def _encode_sdxl_prompts(
    prompts: list[str], tokenizers: list, text_encoders: list, device: Any
) -> tuple:
    """Encode a batch of prompts with both SDXL text encoders. Returns
    (prompt_embeds [B, T, 2048], pooled_prompt_embeds [B, 1280]). Text encoders are
    frozen, so this runs without grad."""
    import torch

    embeds_list = []
    pooled = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        tokens = tokenizer(
            prompts,
            padding = "max_length",
            max_length = tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt",
        ).input_ids.to(device)
        with torch.no_grad():
            out = text_encoder(tokens, output_hidden_states = True)
        # The pooled embed always comes from the second (bigG) text encoder's [0] output.
        pooled = out[0]
        embeds_list.append(out.hidden_states[-2])
    prompt_embeds = torch.concat(embeds_list, dim = -1)
    return prompt_embeds, pooled


def assert_trainable_base_model(base_model: str) -> None:
    """Refuse base models that are recognisably not SDXL, before anything is downloaded.

    Purely name-based: a GGUF filename or a known DiT-family name (FLUX / Qwen-Image /
    Z-Image / SD3 / ...) can never train on the SDXL U-Net trainer, so failing here turns
    a confusing mid-run crash into an immediate, actionable error. Names this cannot
    classify pass through; from_pretrained still fails cleanly on a genuinely wrong pick."""
    name = str(base_model or "").strip().lower()
    if name.endswith(".gguf"):
        raise ValueError(
            f"'{base_model}' is a GGUF checkpoint, which can't be trained. {_ONLY_SDXL_HINT}"
        )
    condensed = re.sub(r"[^a-z0-9]+", "-", name)
    hit = next(
        (p for p in _NON_SDXL_PHRASES if p in condensed),
        None,
    ) or next(
        (t for t in condensed.split("-") if t in _NON_SDXL_TOKENS),
        None,
    )
    if hit:
        raise ValueError(
            f"'{base_model}' looks like a {hit} model, which isn't trainable. {_ONLY_SDXL_HINT}"
        )


def _assert_trusted_base_model(base_model: str) -> None:
    """Gate the training base model the same way the inference backend gates non-GGUF loads:
    a local path or a trusted repo (``unsloth/*`` or an allowlisted official base). This runs
    BEFORE ``from_pretrained`` so an untrusted remote repo (which could ship pickle weights)
    is never fetched or deserialised."""
    from core.inference.diffusion import _is_trusted_diffusion_repo
    if not _is_trusted_diffusion_repo(base_model):
        raise ValueError(
            f"Refusing to train from untrusted base model '{base_model}'. Use a local path or "
            f"a trusted repo (an unsloth/* repo or an official SDXL base)."
        )


def run_diffusion_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    """Train an SDXL U-Net LoRA and export it. Returns the output directory.

    Emits ``model_load_started`` / ``model_load_completed`` / ``progress`` (step, loss) /
    ``complete`` (output_dir, lora_path) events via ``on_event``; ``error`` is emitted by
    the process adapter. Honours ``should_stop`` (checked before model load and between
    optimizer steps); a stop saves a partial adapter unless it carries ``save=False``."""
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.training_utils import cast_training_params, compute_snr
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    cfg = config.normalized()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    # A stop signal may be a bare truthy value or a dict carrying save=False (cancel without
    # saving a partial adapter). ``save_on_stop`` records that decision for the export step.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = cfg.mixed_precision if device == "cuda" else "no"
    if precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        # The default is bf16, but pre-Ampere GPUs (T4 / V100 / RTX 20xx) have no
        # bf16 compute; fall back to fp16 there instead of failing at load/forward.
        precision = "fp16"
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[precision]

    # Preflight the base model against the same trust gate as inference, before any fetch.
    _assert_trusted_base_model(cfg.base_model)

    pairs = discover_image_caption_pairs(
        cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
    )
    _emit(on_event, "model_load_started", num_images = len(pairs))

    # Honour a stop requested before the (potentially large / slow) base model loads, the
    # same way the LLM training worker checks its stop thread around model load.
    if _check_stop():
        out_dir = Path(cfg.output_dir).expanduser()
        _emit(
            on_event,
            "complete",
            output_dir = str(out_dir),
            lora_path = None,
            stopped = True,
            steps_run = 0,
        )
        return str(out_dir)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.base_model, torch_dtype = weight_dtype, token = cfg.hf_token, add_watermarker = False
    )
    unet, vae = pipe.unet, pipe.vae
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze the base; only the LoRA trains. The SDXL VAE overflows fp16, so keep it fp32.
    for m in (unet, vae, *text_encoders):
        m.requires_grad_(False)
    vae.to(device, dtype = torch.float32)
    for m in (unet, *text_encoders):
        m.to(device, dtype = weight_dtype)

    unet.add_adapter(
        LoraConfig(
            r = cfg.lora_rank,
            lora_alpha = cfg.lora_alpha,
            lora_dropout = cfg.lora_dropout,
            init_lora_weights = "gaussian",
            target_modules = list(cfg.lora_target_modules),
        )
    )
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    # LoRA params must be fp32 for a stable optimizer under mixed precision.
    if weight_dtype != torch.float32:
        cast_training_params(unet, dtype = torch.float32)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr = cfg.learning_rate)
    # The scheduler advances once per optimizer update: lr_sched.step() runs a single
    # time per outer opt_step (after the accumulation inner loop), for cfg.train_steps
    # total. Count warmup/decay in those optimizer steps -- multiplying by the
    # accumulation factor would stretch warmup past the run and never reach the decay.
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = cfg.lr_warmup_steps,
        num_training_steps = cfg.train_steps,
    )

    vae_scale = vae.config.scaling_factor
    prediction_type = noise_scheduler.config.prediction_type

    _emit(on_event, "model_load_completed")

    def _next_batch() -> tuple[list[str], list[str]]:
        idx = rng.sample(range(len(pairs)), k = min(cfg.train_batch_size, len(pairs)))
        chosen = [pairs[i] for i in idx]
        return [c[0] for c in chosen], [c[1] for c in chosen]

    unet.train()
    stopped = False
    micro = 0
    running_loss = 0.0
    for opt_step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none = True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            img_paths, captions = _next_batch()
            loaded = [
                _load_image_tensor(p, cfg.resolution, cfg.center_crop, cfg.random_flip, rng)
                for p in img_paths
            ]
            pixel_values = torch.stack([t for t, _ in loaded]).to(device, dtype = torch.float32)
            # Per-sample SDXL micro-conditioning from the actual crop (original size + offset).
            batch_time_ids = torch.tensor(
                [tid for _, tid in loaded], device = device, dtype = weight_dtype
            )

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae_scale
            latents = latents.to(dtype = weight_dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device = device
            ).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, pooled = _encode_sdxl_prompts(
                captions, tokenizers, text_encoders, device
            )
            prompt_embeds = prompt_embeds.to(dtype = weight_dtype)
            pooled = pooled.to(dtype = weight_dtype)
            added = {"text_embeds": pooled, "time_ids": batch_time_ids}

            model_pred = unet(
                noisy, timesteps, prompt_embeds, added_cond_kwargs = added, return_dict = False
            )[0]

            if prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            if cfg.snr_gamma is not None:
                snr = compute_snr(noise_scheduler, timesteps)
                w = torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim = 1).min(
                    dim = 1
                )[0]
                w = w / snr if prediction_type != "v_prediction" else w / (snr + 1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction = "none")
                loss = loss.mean(dim = list(range(1, loss.ndim))) * w
                loss = loss.mean()
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction = "mean")

            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps
            micro += 1

        # max_grad_norm <= 0 means "disable clipping" (the Studio payload sends 0.0 for that);
        # passing 0.0 to clip_grad_norm_ would scale every gradient to zero (no learning).
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm)
        optimizer.step()
        lr_sched.step()

        running_loss += step_loss
        done = opt_step + 1
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            # ``learning_rate`` (not ``lr``) is the field the Studio training pump reads, so
            # these progress events are directly consumable by the existing training
            # status/SSE machinery when the diffusion trainer is wired into the worker.
            _emit(
                on_event,
                "progress",
                step = done,
                total_steps = cfg.train_steps,
                loss = round(step_loss, 5),
                avg_loss = round(running_loss / done, 5),
                learning_rate = lr_sched.get_last_lr()[0],
            )

        if _check_stop():
            stopped = True
            break

    # Export the trained LoRA in diffusers format (loadable via load_lora_weights), unless
    # the run was cancelled with save disabled -- then leave no partial adapter behind.
    out_dir = Path(cfg.output_dir).expanduser()
    lora_path: Optional[str] = None
    catalog_path: Optional[str] = None
    if not (stopped and not save_on_stop):
        out_dir.mkdir(parents = True, exist_ok = True)
        unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory = str(out_dir),
            unet_lora_layers = unet_lora,
            safe_serialization = True,
            weight_name = DEFAULT_LORA_FILENAME,
        )
        lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
        # Mirror into the Studio diffusion LoRA directory so the Images picker discovers it
        # (its scan lists only files directly under loras/diffusion, not subdirectories).
        catalog_path = _publish_to_lora_catalog(lora_path, cfg)
    _emit(
        on_event,
        "complete",
        output_dir = str(out_dir),
        lora_path = lora_path,
        catalog_path = catalog_path,
        stopped = stopped,
        steps_run = done if cfg.train_steps else 0,
    )
    return str(out_dir)


def _publish_to_lora_catalog(lora_path: str, cfg: DiffusionLoraConfig) -> Optional[str]:
    """Best-effort copy of the trained adapter into the Studio diffusion LoRA directory so
    the Images LoRA picker (which scans only files directly under ``loras/diffusion``) finds
    it without the user moving files. Returns the published path, or None on any failure."""
    try:
        import shutil

        from core.inference.diffusion_lora import loras_dir, sanitize_alias

        base = (
            cfg.adapter_name
            if cfg.adapter_name and cfg.adapter_name != "default"
            else Path(cfg.output_dir).name
        )
        dest = loras_dir() / f"{sanitize_alias(base)}.safetensors"
        if Path(lora_path).resolve() != dest.resolve():
            shutil.copy2(lora_path, dest)
        return str(dest)
    except Exception:  # noqa: BLE001 -- the catalog mirror is best-effort, never fatal
        return None


def run_diffusion_training_process(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    """mp.Queue subprocess adapter: run a training job, translating the ``on_event``
    callback to ``event_queue`` and a ``stop_queue`` poll to ``should_stop``. Any
    unexpected exception is reported as an ``error`` event rather than crashing silently."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def on_event(ev: dict) -> None:
        event_queue.put(ev)

    def should_stop() -> Any:
        # Drain the queue and return the last stop message (bool True, or a dict that may
        # carry save=False for cancel-without-save); False when nothing was requested.
        got: Any = None
        saw = False
        try:
            while not stop_queue.empty():
                got = stop_queue.get_nowait()
                saw = True
        except Exception:  # noqa: BLE001 -- an empty/closed queue just means "keep going"
            pass
        return got if saw else False

    try:
        cfg = _config_from_dict(config)
        run_diffusion_lora_training(cfg, on_event = on_event, should_stop = should_stop)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as an error event
        # Emit both keys: the diffusion service reads ``message``, but the generic Studio
        # training worker reads ``error``; carrying both keeps the real failure visible on
        # either path instead of surfacing as "Unknown error".
        event_queue.put(
            {"type": "error", "message": str(exc), "error": str(exc), "ts": time.time()}
        )


# Aliases from the generic Studio training payload onto DiffusionLoraConfig fields, so the
# diffusion trainer can also be driven by the shared training request shape (not only its
# own request model whose keys already match).
_CONFIG_ALIASES = {
    "model_name": "base_model",
    "max_steps": "train_steps",
    "batch_size": "train_batch_size",
    "lora_r": "lora_rank",
    "lr_scheduler_type": "lr_scheduler",
    "random_seed": "seed",
    "lr": "learning_rate",
}


def _coerce_gradient_checkpointing(value: Any) -> bool:
    """Studio sends gradient_checkpointing as a string ("none" / "true" / "unsloth"); the
    disable words are False, anything else truthy is True. A real bool passes through."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no", "off")
    return bool(value)


def _config_from_dict(config: dict) -> DiffusionLoraConfig:
    """Build a DiffusionLoraConfig from a plain dict. Unknown keys are ignored so a richer
    request payload (UI form) does not break construction; a small set of generic Studio
    training keys are aliased onto the diffusion field names, and string flags are coerced."""
    valid = DiffusionLoraConfig.__dataclass_fields__.keys()
    kwargs: dict[str, Any] = {}
    # Aliases first (lowest priority); a canonical key present in the payload overrides.
    for src, dst in _CONFIG_ALIASES.items():
        if src in config and config[src] is not None and dst in valid:
            kwargs[dst] = config[src]
    for k, v in config.items():
        if k in valid:
            kwargs[k] = v
    if kwargs.get("lora_target_modules"):
        kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
    if "gradient_checkpointing" in kwargs:
        kwargs["gradient_checkpointing"] = _coerce_gradient_checkpointing(
            kwargs["gradient_checkpointing"]
        )
    return DiffusionLoraConfig(**kwargs)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description = "Train an SDXL LoRA and export it.")
    p.add_argument("--base-model", required = True, help = "HF repo or local path to an SDXL pipeline")
    p.add_argument("--data-dir", required = True, help = "Folder of images (+ captions)")
    p.add_argument("--output-dir", required = True, help = "Where to write the LoRA .safetensors")
    p.add_argument("--instance-prompt", default = None)
    p.add_argument("--resolution", type = int, default = 1024)
    p.add_argument("--train-steps", type = int, default = 500)
    p.add_argument("--learning-rate", type = float, default = 1e-4)
    p.add_argument("--train-batch-size", type = int, default = 1)
    p.add_argument("--gradient-accumulation-steps", type = int, default = 1)
    p.add_argument("--lora-rank", type = int, default = 16)
    p.add_argument("--lora-alpha", type = int, default = None)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--mixed-precision", default = "bf16", choices = ["bf16", "fp16", "no"])
    p.add_argument("--snr-gamma", type = float, default = 5.0)
    p.add_argument("--no-snr", action = "store_true", help = "Disable min-SNR loss weighting")
    p.add_argument("--no-gradient-checkpointing", action = "store_true")
    p.add_argument("--lr-scheduler", default = "constant")
    p.add_argument("--lr-warmup-steps", type = int, default = 0)
    args = p.parse_args(argv)

    cfg = DiffusionLoraConfig(
        base_model = args.base_model,
        data_dir = args.data_dir,
        output_dir = args.output_dir,
        instance_prompt = args.instance_prompt,
        resolution = args.resolution,
        train_steps = args.train_steps,
        learning_rate = args.learning_rate,
        train_batch_size = args.train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        lora_rank = args.lora_rank,
        lora_alpha = args.lora_alpha,
        seed = args.seed,
        mixed_precision = args.mixed_precision,
        snr_gamma = None if args.no_snr else args.snr_gamma,
        gradient_checkpointing = not args.no_gradient_checkpointing,
        lr_scheduler = args.lr_scheduler,
        lr_warmup_steps = args.lr_warmup_steps,
    )

    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "progress":
            print(
                f"step {ev['step']}/{ev['total_steps']} loss={ev['loss']} "
                f"avg={ev['avg_loss']} lr={ev['learning_rate']:.2e}",
                flush = True,
            )
        elif t == "complete":
            print(f"done: {ev['lora_path']} (stopped={ev['stopped']})", flush = True)
        else:
            print(ev, flush = True)

    run_diffusion_lora_training(cfg, on_event = on_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
