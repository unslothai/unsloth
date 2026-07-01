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

EventCb = Callable[[dict[str, Any]], None]
StopCb = Callable[[], bool]


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
        request that cannot train (bad numbers, or no caption source)."""
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.resolution < 64 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a multiple of 8 and >= 64")
        if self.mixed_precision not in ("bf16", "fp16", "no"):
            raise ValueError("mixed_precision must be one of bf16 / fp16 / no")
        alpha = self.lora_alpha if self.lora_alpha is not None else self.lora_rank
        targets = tuple(self.lora_target_modules) or DEFAULT_LORA_TARGETS
        return replace(self, lora_alpha=alpha, lora_target_modules=targets)


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

    images = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )

    # 1. metadata.jsonl / captions.jsonl (either name accepted).
    meta_caption: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = root / meta_name
        if not meta_path.is_file():
            continue
        for line in meta_path.read_text(encoding="utf-8").splitlines():
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
                    text = sidecar.read_text(encoding="utf-8").strip()
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
    turns it into a tensor. No crop offset is applied (top-left = 0)."""
    return (resolution, resolution, 0, 0, resolution, resolution)


def _emit(on_event: Optional[EventCb], type_: str, **kw: Any) -> None:
    if on_event is not None:
        on_event({"type": type_, "ts": time.time(), **kw})


def _load_image_tensor(
    path: str, resolution: int, center_crop: bool, random_flip: bool, rng: random.Random
) -> Any:
    """Load an image to a normalised CxHxW tensor in [-1, 1] (resize shorter side to
    ``resolution``, crop to a square, optional horizontal flip). No torchvision."""
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = resolution / min(w, h)
    img = img.resize((max(resolution, round(w * scale)), max(resolution, round(h * scale))), Image.LANCZOS)
    w, h = img.size
    if center_crop:
        left, top = (w - resolution) // 2, (h - resolution) // 2
    else:
        left = rng.randint(0, max(0, w - resolution))
        top = rng.randint(0, max(0, h - resolution))
    img = img.crop((left, top, left + resolution, top + resolution))
    if random_flip and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def _encode_sdxl_prompts(prompts: list[str], tokenizers: list, text_encoders: list, device: Any) -> tuple:
    """Encode a batch of prompts with both SDXL text encoders. Returns
    (prompt_embeds [B, T, 2048], pooled_prompt_embeds [B, 1280]). Text encoders are
    frozen, so this runs without grad."""
    import torch

    embeds_list = []
    pooled = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        tokens = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.no_grad():
            out = text_encoder(tokens, output_hidden_states=True)
        # The pooled embed always comes from the second (bigG) text encoder's [0] output.
        pooled = out[0]
        embeds_list.append(out.hidden_states[-2])
    prompt_embeds = torch.concat(embeds_list, dim=-1)
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
    the process adapter. Honours ``should_stop`` between optimizer steps (partial save)."""
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        cfg.mixed_precision if device == "cuda" else "no"
    ]

    pairs = discover_image_caption_pairs(
        cfg.data_dir, instance_prompt=cfg.instance_prompt, caption_column=cfg.caption_column
    )
    _emit(on_event, "model_load_started", num_images=len(pairs))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.base_model, torch_dtype=weight_dtype, token=cfg.hf_token, add_watermarker=False
    )
    unet, vae = pipe.unet, pipe.vae
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze the base; only the LoRA trains. The SDXL VAE overflows fp16, so keep it fp32.
    for m in (unet, vae, *text_encoders):
        m.requires_grad_(False)
    vae.to(device, dtype=torch.float32)
    for m in (unet, *text_encoders):
        m.to(device, dtype=weight_dtype)

    unet.add_adapter(
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=list(cfg.lora_target_modules),
        )
    )
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    # LoRA params must be fp32 for a stable optimizer under mixed precision.
    if weight_dtype != torch.float32:
        cast_training_params(unet, dtype=torch.float32)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=cfg.learning_rate)
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.train_steps * cfg.gradient_accumulation_steps,
    )

    add_time_ids = torch.tensor(
        [compute_sdxl_add_time_ids(cfg.resolution)], device=device, dtype=weight_dtype
    )
    vae_scale = vae.config.scaling_factor
    prediction_type = noise_scheduler.config.prediction_type

    _emit(on_event, "model_load_completed")

    def _next_batch() -> tuple[list[str], list[str]]:
        idx = rng.sample(range(len(pairs)), k=min(cfg.train_batch_size, len(pairs)))
        chosen = [pairs[i] for i in idx]
        return [c[0] for c in chosen], [c[1] for c in chosen]

    unet.train()
    stopped = False
    micro = 0
    running_loss = 0.0
    for opt_step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            img_paths, captions = _next_batch()
            pixel_values = torch.stack(
                [
                    _load_image_tensor(p, cfg.resolution, cfg.center_crop, cfg.random_flip, rng)
                    for p in img_paths
                ]
            ).to(device, dtype=torch.float32)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae_scale
            latents = latents.to(dtype=weight_dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds, pooled = _encode_sdxl_prompts(captions, tokenizers, text_encoders, device)
            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            pooled = pooled.to(dtype=weight_dtype)
            added = {"text_embeds": pooled, "time_ids": add_time_ids.repeat(bsz, 1)}

            model_pred = unet(
                noisy, timesteps, prompt_embeds, added_cond_kwargs=added, return_dict=False
            )[0]

            if prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            if cfg.snr_gamma is not None:
                snr = compute_snr(noise_scheduler, timesteps)
                w = torch.stack(
                    [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                w = w / snr if prediction_type != "v_prediction" else w / (snr + 1)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, loss.ndim))) * w
                loss = loss.mean()
            else:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps
            micro += 1

        torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm)
        optimizer.step()
        lr_sched.step()

        running_loss += step_loss
        done = opt_step + 1
        if done % cfg.log_every == 0 or done == cfg.train_steps:
            _emit(
                on_event,
                "progress",
                step=done,
                total_steps=cfg.train_steps,
                loss=round(step_loss, 5),
                avg_loss=round(running_loss / done, 5),
                lr=lr_sched.get_last_lr()[0],
            )

        if should_stop is not None and should_stop():
            stopped = True
            break

    # Export the trained LoRA in diffusers format (loadable via load_lora_weights).
    out_dir = Path(cfg.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=unet_lora,
        safe_serialization=True,
        weight_name=DEFAULT_LORA_FILENAME,
    )
    lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
    _emit(
        on_event,
        "complete",
        output_dir=str(out_dir),
        lora_path=lora_path,
        stopped=stopped,
        steps_run=done if cfg.train_steps else 0,
    )
    return str(out_dir)


def run_diffusion_training_process(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    """mp.Queue subprocess adapter: run a training job, translating the ``on_event``
    callback to ``event_queue`` and a ``stop_queue`` poll to ``should_stop``. Any
    unexpected exception is reported as an ``error`` event rather than crashing silently."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def on_event(ev: dict) -> None:
        event_queue.put(ev)

    def should_stop() -> bool:
        try:
            while not stop_queue.empty():
                stop_queue.get_nowait()
                return True
        except Exception:  # noqa: BLE001 -- an empty/closed queue just means "keep going"
            pass
        return False

    try:
        cfg = _config_from_dict(config)
        run_diffusion_lora_training(cfg, on_event=on_event, should_stop=should_stop)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as an error event
        event_queue.put({"type": "error", "message": str(exc), "ts": time.time()})


def _config_from_dict(config: dict) -> DiffusionLoraConfig:
    """Build a DiffusionLoraConfig from a plain dict, ignoring unknown keys so a richer
    request payload (UI form) does not break construction."""
    valid = DiffusionLoraConfig.__dataclass_fields__.keys()
    kwargs = {k: v for k, v in config.items() if k in valid}
    if "lora_target_modules" in kwargs and kwargs["lora_target_modules"]:
        kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
    return DiffusionLoraConfig(**kwargs)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train an SDXL LoRA and export it.")
    p.add_argument("--base-model", required=True, help="HF repo or local path to an SDXL pipeline")
    p.add_argument("--data-dir", required=True, help="Folder of images (+ captions)")
    p.add_argument("--output-dir", required=True, help="Where to write the LoRA .safetensors")
    p.add_argument("--instance-prompt", default=None)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "no"])
    p.add_argument("--snr-gamma", type=float, default=5.0)
    p.add_argument("--no-snr", action="store_true", help="Disable min-SNR loss weighting")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--lr-scheduler", default="constant")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    args = p.parse_args(argv)

    cfg = DiffusionLoraConfig(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        instance_prompt=args.instance_prompt,
        resolution=args.resolution,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        snr_gamma=None if args.no_snr else args.snr_gamma,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
    )

    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "progress":
            print(
                f"step {ev['step']}/{ev['total_steps']} loss={ev['loss']} "
                f"avg={ev['avg_loss']} lr={ev['lr']:.2e}",
                flush=True,
            )
        elif t == "complete":
            print(f"done: {ev['lora_path']} (stopped={ev['stopped']})", flush=True)
        else:
            print(ev, flush=True)

    run_diffusion_lora_training(cfg, on_event=on_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
