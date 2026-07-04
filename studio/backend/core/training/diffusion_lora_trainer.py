# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion LoRA training for Unsloth Studio (text-to-image, SDXL).

Trains a LoRA adapter on the U-Net of an SDXL pipeline from an image + caption dataset
and exports it as a diffusers-format ``.safetensors`` that the Studio diffusion backend
(and any diffusers pipeline via ``load_lora_weights``) can load.

Design:
- Family-agnostic building blocks (dataset discovery, config normalisation + validation,
  event emission, the stop protocol, adapter publishing, and the family/trainer registry)
  live in ``diffusion_train_common`` and are shared with the DiT trainers. They are
  re-exported here so existing import paths keep working.
- ``run_diffusion_lora_training`` is the SDXL training loop. It reports progress through an
  ``on_event`` callback whose payloads match the training worker's event protocol
  (``{"type": ..., "ts": ...}``) so it can be spawned as a subprocess and streamed to the
  UI, and it polls a ``should_stop`` callback so a stop request ends it cleanly (with a
  partial save).
- ``run_diffusion_training_process`` is the thin mp.Queue adapter; it dispatches to the
  trainer registered for the resolved family (SDXL here, DiT families in a follow-up).
  ``main`` is a CLI.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

# Shared, family-agnostic building blocks. Re-exported so callers/tests that import them
# from this module (its historical home) keep working unchanged.
from core.training.diffusion_train_common import (  # noqa: F401
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    EventCb,
    StopCb,
    DiffusionLoraConfig,
    _assert_trusted_base_model,
    _coerce_gradient_checkpointing,
    _config_from_dict,
    _CONFIG_ALIASES,
    _emit,
    _publish_to_lora_catalog,
    discover_image_caption_pairs,
    get_trainer,
)


def compute_sdxl_add_time_ids(resolution: int) -> tuple[int, int, int, int, int, int]:
    """SDXL micro-conditioning ``add_time_ids`` for a square ``resolution`` train crop:
    (original_h, original_w, crop_top, crop_left, target_h, target_w). Pure; the trainer
    turns it into a tensor. No crop offset is applied (top-left = 0). The training loop
    derives per-image time-ids from the actual crop instead; this is the square default."""
    return (resolution, resolution, 0, 0, resolution, resolution)


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

    # An empty (unset) config means "use the family default": the SDXL attention projections.
    unet_targets = list(cfg.lora_target_modules) or list(DEFAULT_LORA_TARGETS)
    unet.add_adapter(
        LoraConfig(
            r = cfg.lora_rank,
            lora_alpha = cfg.lora_alpha,
            lora_dropout = cfg.lora_dropout,
            init_lora_weights = "gaussian",
            target_modules = unet_targets,
        )
    )
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    # LoRA params must be fp32 for a stable optimizer under mixed precision.
    if weight_dtype != torch.float32:
        cast_training_params(unet, dtype = torch.float32)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = _make_lora_optimizer(lora_params, cfg.learning_rate)
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

    # Precompute text embeddings once per unique caption, then free the CLIP text encoders.
    # SDXL re-encoded captions every step (pure waste: captions are constant) and kept both
    # text encoders (~1.5 GB) resident. Embeddings are deterministic and this consumes no
    # torch RNG, so the training math is bit-identical to in-loop encoding -- only faster and
    # lighter. The env toggle exists purely so the accuracy guard can A/B the two paths.
    precompute = os.environ.get("UNSLOTH_DIFFUSION_NO_PRECOMPUTE", "") not in ("1", "true")
    caption_embeds: dict[str, tuple] = {}
    if precompute:
        for cap in sorted({c for _, c in pairs}):
            pe, pooled_c = _encode_sdxl_prompts([cap], tokenizers, text_encoders, device)
            caption_embeds[cap] = (pe.cpu(), pooled_c.cpu())
        for te in text_encoders:
            te.to("cpu")
        text_encoders = []
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    _emit(on_event, "model_load_completed")

    def _next_batch() -> tuple[list[str], list[str]]:
        idx = rng.sample(range(len(pairs)), k = min(cfg.train_batch_size, len(pairs)))
        chosen = [pairs[i] for i in idx]
        return [c[0] for c in chosen], [c[1] for c in chosen]

    unet.train()
    stopped = False
    micro = 0
    running_loss = 0.0
    peak_gb = 0.0
    t_start = time.time()
    done = 0
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

            if precompute:
                prompt_embeds = torch.cat([caption_embeds[c][0] for c in captions]).to(device)
                pooled = torch.cat([caption_embeds[c][1] for c in captions]).to(device)
            else:
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
            if device == "cuda":
                peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            samples_per_second = round(
                (done * cfg.train_batch_size * cfg.gradient_accumulation_steps)
                / max(time.time() - t_start, 1e-6),
                3,
            )
            _emit(
                on_event,
                "progress",
                step = done,
                total_steps = cfg.train_steps,
                loss = round(step_loss, 5),
                avg_loss = round(running_loss / done, 5),
                learning_rate = lr_sched.get_last_lr()[0],
                samples_per_second = samples_per_second,
                peak_memory_gb = peak_gb or None,
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
        family = cfg.resolved_family,
        base_model = cfg.base_model,
        stopped = stopped,
        steps_run = done if cfg.train_steps else 0,
    )
    return str(out_dir)


def _make_lora_optimizer(params: list, lr: float) -> Any:
    """8-bit AdamW (bitsandbytes) by default -- half the optimizer state, no meaningful
    quality cost for LoRA -- falling back to fp32 AdamW when unavailable or when
    UNSLOTH_DIFFUSION_FP32_OPTIM is set (used by the accuracy guard)."""
    import torch

    if os.environ.get("UNSLOTH_DIFFUSION_FP32_OPTIM", "") not in ("1", "true"):
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr = lr)
        except Exception:  # noqa: BLE001 -- bnb missing / no CUDA: fall back to torch AdamW
            pass
    return torch.optim.AdamW(params, lr = lr)


def run_diffusion_training_process(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    """mp.Queue subprocess adapter: run a training job, translating the ``on_event``
    callback to ``event_queue`` and a ``stop_queue`` poll to ``should_stop``. Dispatches to
    the trainer registered for the resolved family. Any unexpected exception is reported as
    an ``error`` event rather than crashing silently."""
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
        # normalized() resolves + validates the family; dispatch through the registry so a
        # DiT family runs its own trainer while SDXL keeps this module's loop.
        cfg = _config_from_dict(config).normalized()
        trainer = get_trainer(cfg.resolved_family)
        trainer(cfg, on_event = on_event, should_stop = should_stop)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as an error event
        # Emit both keys: the diffusion service reads ``message``, but the generic Studio
        # training worker reads ``error``; carrying both keeps the real failure visible on
        # either path instead of surfacing as "Unknown error".
        event_queue.put(
            {"type": "error", "message": str(exc), "error": str(exc), "ts": time.time()}
        )


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
