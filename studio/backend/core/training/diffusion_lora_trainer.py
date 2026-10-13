# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion LoRA training for Unsloth Studio (text-to-image, SDXL).

Trains a LoRA adapter on the U-Net of an SDXL pipeline from an image + caption dataset and exports
it as a diffusers-format ``.safetensors`` the Unsloth diffusion backend (and any diffusers pipeline
via ``load_lora_weights``) can load.

Family-agnostic building blocks (dataset discovery, config normalisation and validation, event
emission, the stop protocol, adapter publishing, and the family/trainer registry) live in
``diffusion_train_common`` and are shared with the DiT trainers; they are re-exported here so
existing import paths keep working. ``run_diffusion_lora_training`` is the SDXL training loop: it
reports progress through an ``on_event`` callback whose payloads match the training worker's event
protocol so it can be spawned as a subprocess and streamed to the UI, and it polls a ``should_stop``
callback so a stop request ends it cleanly with a partial save. ``run_diffusion_training_process``
is the thin mp.Queue adapter, dispatching to the trainer registered for the resolved family, and
``main`` is a CLI.

Memory/perf: captions are encoded once up front and the CLIP text encoders freed; VAE latents are
likewise precomputed into a small CPU cache (``cache_latents``) and the VAE freed. The cache stores
the posterior's affine pair (mean/std, scale folded in), so every step still draws a fresh VAE
sample, distribution-identical to encoding in the loop without keeping the VAE resident. TF32
matmuls and cudnn autotuning are enabled under ``cfg.enable_tf32``, and the U-Net's repeated
transformer blocks are regionally torch.compiled (``cfg.compile_transformer``, never fatal).
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

# Re-exported so callers/tests importing from this module keep working.
from core.training.diffusion_train_common import (  # noqa: F401
    DEFAULT_LORA_FILENAME,
    DEFAULT_LORA_TARGETS,
    EventCb,
    StopCb,
    DiffusionLoraConfig,
    LATENT_CACHE_OVER_BUDGET,
    bitsandbytes_optimizer_supported,
    native_bf16_supported,
    _apply_perf_flags,
    _assert_trusted_base_model,
    _coerce_gradient_checkpointing,
    _config_from_dict,
    _CONFIG_ALIASES,
    _emit,
    resolve_train_device,
    _latent_cache_forced,
    _latent_cache_over_budget,
    _plan_cache_variants,
    _publish_to_lora_catalog,
    _restore_perf_flags,
    bucket_groups,
    discover_image_caption_pairs,
    get_trainer,
    PermutationBatchSampler,
    plan_image_canvases,
    resolve_bucketing,
    resolve_train_steps,
    restore_resume_state,
    write_resume_checkpoint,
)
from core.training.diffusion_samples import (
    discard_samples,
    initial_noise,
    isolated_sampling,
    plan_samples,
    run_sample_round,
    sample_seed,
    SampleRoundStopped,
)
from core.training.diffusion_train_extras import BucketBatchSampler, cover_resize_dims
from core.training.diffusion_checkpoint import (
    clear_own_checkpoints,
    discard_preexisting_checkpoints,
    retire_own_checkpoints,
    resumed_into_this_dir,
    snapshot_checkpoints,
    identity_for_config,
    with_cache_mode,
    with_resolved_revision,
    preflight_resume,
)


def compute_sdxl_add_time_ids(resolution: int) -> tuple[int, int, int, int, int, int]:
    """Square default with no crop offset; the training loop derives per-image ids from the actual crop."""
    return (resolution, resolution, 0, 0, resolution, resolution)


def _canvas(resolution: Any) -> tuple[int, int]:
    """(w, h) of the training canvas: a bucket tuple, or the legacy square ``resolution``."""
    if isinstance(resolution, (tuple, list)):
        return int(resolution[0]), int(resolution[1])
    return int(resolution), int(resolution)


def _crop_to_canvas(
    path: str, resolution: Any, center_crop: bool, pick: Any, flip: Any
) -> tuple[Any, tuple[int, int, int, int, int, int]]:
    """Cover-resize to the canvas, crop at ``pick(range_x, range_y)``, optionally mirror. Returns
    CxHxW pixels in [-1, 1] and SDXL time ids (original_h, original_w, crop_top, crop_left,
    target_h, target_w); a square canvas reproduces the legacy short-side geometry exactly."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    from core.inference.mcp_images import flattened_rgb

    cw, ch = _canvas(resolution)
    # apply EXIF orientation before geometry so rotated photos train upright.
    img = flattened_rgb(ImageOps.exif_transpose(Image.open(path)), background = (255, 255, 255))
    original_w, original_h = img.size
    resized_w, resized_h = cover_resize_dims(original_w, original_h, cw, ch)
    img = img.resize((resized_w, resized_h), Image.LANCZOS)
    if center_crop:
        left, top = (resized_w - cw) // 2, (resized_h - ch) // 2
    else:
        left, top = pick(resized_w - cw, resized_h - ch)
    img = img.crop((left, top, left + cw, top + ch))
    crop_left = left
    if flip():
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # mirror the crop origin after a flip to match diffusers.
        crop_left = max(0, resized_w - cw - left)
    arr = np.asarray(img, dtype = np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
    time_ids = (original_h, original_w, top, crop_left, ch, cw)
    return tensor, time_ids


def _load_image_tensor(
    path: str, resolution: Any, center_crop: bool, random_flip: bool, rng: random.Random
) -> tuple[Any, tuple[int, int, int, int, int, int]]:
    """CxHxW pixels in [-1, 1] with SDXL time ids for the EXIF-oriented crop."""
    return _crop_to_canvas(
        path,
        resolution,
        center_crop,
        lambda rx, ry: (rng.randint(0, max(0, rx)), rng.randint(0, max(0, ry))),
        lambda: random_flip and rng.random() < 0.5,
    )


def _load_image_tensor_planned(
    path: str, resolution: Any, center_crop: bool, u_left: float, u_top: float, flip: bool
) -> tuple[Any, tuple[int, int, int, int, int, int]]:
    """cache loader preserving ``_load_image_tensor`` crop, flip, and legacy center semantics."""
    return _crop_to_canvas(
        path,
        resolution,
        center_crop,
        lambda rx, ry: (
            min(int(u_left * (rx + 1)), max(0, rx)),
            min(int(u_top * (ry + 1)), max(0, ry)),
        ),
        lambda: flip,
    )


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
        # Pooled embed always comes from the second (bigG) encoder's [0] output.
        pooled = out[0]
        embeds_list.append(out.hidden_states[-2])
    prompt_embeds = torch.concat(embeds_list, dim = -1)
    return prompt_embeds, pooled


def _build_sdxl_latent_cache(
    vae,
    vae_scale,
    image_paths,
    cfg,
    device,
    weight_dtype,
    on_event,
    check_stop,
    canvases = None,
    plan = None,
):
    """Precompute the per-image latent posterior cache: for each planned crop/flip variant,
    encode once and store ``(A, B, time_ids)`` on CPU in fp32. ``A`` and ``B`` are the affine
    posterior parameters (mean/std with the VAE scale folded in) so a per-step sample is
    ``A + B * randn`` -- distribution-identical to an in-loop ``latent_dist.sample()`` -- and
    ``time_ids`` is the SDXL micro-conditioning for the crop. The stats stay fp32 so the
    per-step sample happens in fp32 and only the RESULT is cast to weight_dtype, matching the
    in-loop path (encode fp32 -> sample fp32 -> scale -> .to(weight_dtype)); fp32 doubles the
    cache RAM over bf16 but the cache is tiny (a handful of latents per image). Returns None if
    the build was interrupted by a stop request. ``vae_scale`` is read before the VAE is freed.
    ``canvases`` holds each image's bucket for a bucketed run (None: the square resolution)."""
    import torch

    if plan is None:
        plan = _plan_cache_variants(
            len(image_paths), cfg.cache_variants, cfg.center_crop, cfg.random_flip, cfg.seed
        )

    def _hold(t):
        t = t.to(torch.float32).cpu()
        if device == "cuda":
            try:
                t = t.pin_memory()
            except RuntimeError:
                pass
        return t

    cache: list[list[tuple]] = []
    total = len(image_paths)
    total_variants = sum(len(v) for v in plan)
    forced = _latent_cache_forced()
    gated = False
    for i, path in enumerate(image_paths):
        variants = []
        for u_left, u_top, flip in plan[i]:
            tensor, time_ids = _load_image_tensor_planned(
                path,
                cfg.resolution if canvases is None else canvases[i],
                cfg.center_crop,
                u_left,
                u_top,
                flip,
            )
            pixel_values = tensor.unsqueeze(0).to(device, dtype = torch.float32)
            with torch.no_grad():
                dist = vae.encode(pixel_values).latent_dist
            a = _hold(dist.mean * vae_scale)
            b = _hold(dist.std * vae_scale)
            if not forced and not gated:
                # Size-gate the latent cache: it can exhaust pinned RAM.
                per_variant = a.numel() * a.element_size() + b.numel() * b.element_size()
                if _latent_cache_over_budget(per_variant, total_variants):
                    _emit(
                        on_event,
                        "warning",
                        message = (
                            "Latent cache disabled: estimated "
                            f"{per_variant * total_variants / 1024 ** 3:.1f} GiB over the "
                            "budget; encoding latents per step instead. Set "
                            "UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE=1 to keep it."
                        ),
                    )
                    return LATENT_CACHE_OVER_BUDGET
                gated = True
            variants.append((a, b, tuple(time_ids)))
        cache.append(variants)
        if (i + 1) % 4 == 0 or i + 1 == total:
            _emit(on_event, "preparing", stage = "cache_latents", done = i + 1, total = total)
        if check_stop():
            return None
    return cache


def _sample_sdxl_cached_latents(cache, idxs, variant_rng, device, weight_dtype):
    """Draws fresh noise each step from the cached A and B, as an in-loop latent_dist.sample() would."""
    import torch

    parts_a, parts_b, tid_rows = [], [], []
    for i in idxs:
        variants = cache[i]
        a, b, time_ids = (
            variants[variant_rng.randrange(len(variants))] if len(variants) > 1 else variants[0]
        )
        parts_a.append(a)
        parts_b.append(b)
        tid_rows.append(time_ids)
    lat_a = torch.cat(parts_a).to(device, non_blocking = True)
    lat_b = torch.cat(parts_b).to(device, non_blocking = True)
    latents = (lat_a + lat_b * torch.randn_like(lat_a)).to(dtype = weight_dtype)
    batch_time_ids = torch.tensor(tid_rows, device = device, dtype = weight_dtype)
    return latents, batch_time_ids


def _mps_available() -> bool:
    import torch
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 -- an unprobeable backend is not one we train on
        return False


def run_diffusion_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    """A stop saves a partial adapter unless it carries save=False; train_steps is the target total."""
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.training_utils import cast_training_params, compute_snr
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    # diffusers hard-codes _tqdm_active = True; this is the earliest point to silence it.
    try:
        from loggers.config import quiet_third_party_progress_bars
        quiet_third_party_progress_bars()
    except Exception:  # noqa: BLE001 - never let log tidying stop a training run
        pass

    cfg = resolve_bucketing(config.normalized())
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = cfg.mixed_precision if device == "cuda" else "no"
    if precision == "bf16" and device == "cuda" and not native_bf16_supported():
        # is_bf16_supported() counts emulation; use the capability probe, fall back to fp16.
        precision = "fp16"
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[precision]

    snap = _apply_perf_flags(cfg, device)
    try:
        _assert_trusted_base_model(cfg.base_model)

        pairs = discover_image_caption_pairs(
            cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
        )
        # Resolve num_epochs into a concrete train_steps now the dataset size is known, and rebind cfg so every
        # downstream read agrees.
        cfg = replace(
            cfg,
            train_steps = resolve_train_steps(cfg, len(pairs), [p for p, _ in pairs]),
            num_epochs = 0,
        )
        # Validate a resume request against this run's identity BEFORE the multi-GB base model load, so a
        # mismatch fails in seconds instead of after the download.
        identity = identity_for_config(cfg, dataset_pairs = pairs)
        if cfg.resume_from_checkpoint:
            preflight_resume(
                cfg.resume_from_checkpoint,
                identity = identity,
                target_steps = cfg.train_steps,
            )
        if device == "cpu" and (resolve_train_device() == "xpu" or _mps_available()):
            _emit(
                on_event,
                "warning",
                message = (
                    "SDXL training runs on the CPU in fp32 on this host: the SDXL trainer uses CUDA / ROCm "
                    "GPUs only, so expect slow steps and previews."
                ),
            )
        _emit(on_event, "model_load_started", num_images = len(pairs))

        if _check_stop():
            out_dir = Path(cfg.output_dir).expanduser()
            _emit(
                on_event,
                "complete",
                output_dir = str(out_dir),
                lora_path = None,
                stopped = True,
                steps_run = 0,
                discarded = not save_on_stop,
            )
            return str(out_dir)

        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.base_model, torch_dtype = weight_dtype, token = cfg.hf_token, add_watermarker = False
        )
        unet, vae = pipe.unet, pipe.vae
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # The SDXL VAE overflows fp16, so keep it fp32.
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
        if weight_dtype != torch.float32:
            cast_training_params(unet, dtype = torch.float32)

        from core.training.diffusion_dit_trainer import _maybe_compile_transformer

        compiled = _maybe_compile_transformer(
            unet, cfg, False, device, on_event, base_precision = "bf16"
        )

        lora_params = [p for p in unet.parameters() if p.requires_grad]
        optimizer = _make_lora_optimizer(lora_params, cfg.learning_rate)
        # A frozen fp16 UNet (pre-Ampere / RDNA2 fallback) underflows the LoRA gradients without loss scaling.
        scaler = torch.amp.GradScaler("cuda", enabled = weight_dtype is torch.float16)
        # The scheduler advances once per optimizer update, so count warmup/decay in optimizer steps; the
        # accumulation factor would stretch warmup.
        lr_sched = get_scheduler(
            cfg.lr_scheduler,
            optimizer = optimizer,
            num_warmup_steps = cfg.lr_warmup_steps,
            num_training_steps = cfg.train_steps,
        )

        vae_scale = vae.config.scaling_factor
        prediction_type = noise_scheduler.config.prediction_type

        # Deterministic, so the precomputed embeddings are bit-identical to in-loop encoding; the ~1.5 GB
        # CLIP encoders are freed afterwards.
        precompute = os.environ.get("UNSLOTH_DIFFUSION_NO_PRECOMPUTE", "") not in ("1", "true")
        caption_embeds: dict[str, tuple] = {}
        sample_plan = plan_samples(cfg, "sdxl", [c for _, c in pairs])
        # Kept past a failed round (which drops sample_plan) so a discard still removes what was written.
        sample_owner = sample_plan
        sample_embeds: dict[str, tuple] = {}
        if sample_plan is not None:
            for text in sample_plan.encode_texts:
                pe, pooled_c = _encode_sdxl_prompts([text], tokenizers, text_encoders, device)
                sample_embeds[text] = (pe.cpu(), pooled_c.cpu())
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

        # The cache holds the posterior affine pair, so per-step sampling noise is preserved once the VAE is freed.
        use_cache = cfg.cache_latents and os.environ.get(
            "UNSLOTH_DIFFUSION_NO_LATENT_CACHE", ""
        ) not in ("1", "true")
        latent_cache = None
        canvases, crop_room = plan_image_canvases(cfg, [p for p, _ in pairs])
        if use_cache:
            latent_cache = _build_sdxl_latent_cache(
                vae,
                vae_scale,
                [p for p, _ in pairs],
                cfg,
                device,
                weight_dtype,
                on_event,
                _check_stop,
                canvases = canvases,
                plan = _plan_cache_variants(
                    len(pairs),
                    cfg.cache_variants,
                    cfg.center_crop,
                    cfg.random_flip,
                    cfg.seed,
                    crop_room = crop_room,
                ),
            )
            if latent_cache is LATENT_CACHE_OVER_BUDGET:
                latent_cache = None
            elif latent_cache is None:
                out_dir = Path(cfg.output_dir).expanduser()
                _emit(
                    on_event,
                    "complete",
                    output_dir = str(out_dir),
                    lora_path = None,
                    stopped = True,
                    steps_run = 0,
                    discarded = not save_on_stop,
                )
                return str(out_dir)
            else:
                if sample_plan is not None:
                    sample_vae = vae.to("cpu")
                try:
                    pipe.vae = None
                except Exception:  # noqa: BLE001 -- a pipeline without a settable vae keeps it
                    pass
                del vae
                vae = None
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        if sample_plan is not None and vae is not None:
            sample_vae = vae
        # Variant picks use their own stream so the loop index/noise draws stay seed-deterministic whether
        # or not the cache is enabled.
        variant_rng = random.Random(cfg.seed + 1)

        _emit(on_event, "model_load_completed", compiled = compiled)
        # Identity was built before the load; resolve the revision now the base is on disk.
        identity = with_resolved_revision(identity, cfg.base_model)
        identity = with_cache_mode(identity, latent_cache is not None)

        # Permutation-cycle index sampler: each image is visited once per cycle before any repeat, so a
        # short run does not leave a small dataset partly unseen.
        # Bucketed runs draw each batch from one bucket so its latents stack into one shape.
        index_sampler = (
            PermutationBatchSampler(len(pairs), rng)
            if canvases is None
            else BucketBatchSampler(bucket_groups(canvases), rng)
        )

        def _next_batch() -> tuple[list[int], list[str], list[str]]:
            # Draw the full configured batch, not min(batch, n): the sampler refills across cycles so a small
            # dataset still yields that many indices.
            idx = index_sampler.next_batch(cfg.train_batch_size)
            if canvases is not None:
                idx = idx[1]
            chosen = [pairs[i] for i in idx]
            return idx, [c[0] for c in chosen], [c[1] for c in chosen]

        # Restore a previous run's state BEFORE the loop, so resumed is the last completed step and the loop
        # picks up at resumed + 1; a mismatch raises out to the adapter.
        rng_streams = {"loop": rng, "variant": variant_rng}
        restored = restore_resume_state(
            cfg,
            model = unet,
            optimizer = optimizer,
            lr_scheduler = lr_sched,
            identity = identity,
            on_event = on_event,
            sampler = index_sampler,
            rng_streams = rng_streams,
        )
        resumed = restored.step if restored is not None else 0
        scaler_state = restored.progress.get("grad_scaler") if restored is not None else None
        if scaler.is_enabled() and isinstance(scaler_state, dict) and scaler_state:
            # Continue at the scale the run had backed off to, not 2**16, which would overflow again.
            scaler.load_state_dict(scaler_state)
        # Bound HERE: the checkpoint scan and the periodic saves both need it, and the two earlier
        # assignments live inside early-return branches.
        out_dir = Path(cfg.output_dir).expanduser()
        # Foreign bundles in a reused output_dir must be cleared or they outrank this run.
        resumed_here = bool(resumed) and resumed_into_this_dir(cfg, out_dir)
        preexisting_checkpoints = snapshot_checkpoints(out_dir)

        unet.train()
        stopped = False
        micro = 0
        running_loss = restored.running_loss if restored is not None else 0.0
        peak_gb = 0.0
        t_start = time.time()
        t_steady = None
        t_rate = t_start
        # Starts at the resumed step so a no-op resume (nothing left to train) still reports the real step.
        done = resumed

        def _save_checkpoint(step: int) -> None:
            # write_resume_checkpoint reports its own checkpoint_saved / checkpoint_failed events, so a run
            # that crashes after a save is still known to be resumable and one whose save failed is still
            # known to be blocked.
            _written, _error = write_resume_checkpoint(
                cfg,
                step = step,
                model = unet,
                optimizer = optimizer,
                lr_scheduler = lr_sched,
                identity = identity,
                on_event = on_event,
                sampler = index_sampler,
                rng_streams = rng_streams,
                progress = {"running_loss": running_loss, "grad_scaler": scaler.state_dict()},
                # NOT discard_existing: deleting the previous run's bundles at the FIRST periodic save spends them
                # before this run produced anything, so cancelling a retrain destroyed the thing being retrained.
                discard_existing = False,
                preexisting = preexisting_checkpoints,
            )

        def _interrupt_on_stop(p, _i, _t, _kwargs):
            # The pipeline skips its remaining steps once _interrupt is set (pipeline_stable_diffusion_xl.py).
            if _stop_during_sample():
                p._interrupt = True
            return {}

        def _render_sdxl(index, prompt):
            res = sample_plan.resolution
            pe, pooled_c = (t.to(device, dtype = weight_dtype) for t in sample_embeds[prompt])
            npe = npooled = None
            if sample_plan.uses_cfg:
                npe, npooled = (t.to(device, dtype = weight_dtype) for t in sample_embeds[""])
                # Studio inference sends no negative, so the pipeline zeros it on SDXL base (force_zeros_for_empty_prompt).
                if pipe.config.get("force_zeros_for_empty_prompt", False):
                    npe, npooled = torch.zeros_like(npe), torch.zeros_like(npooled)
            noise = initial_noise(sample_plan, index, (1, 4, res // 8, res // 8), device)
            latents = pipe(
                prompt_embeds = pe,
                pooled_prompt_embeds = pooled_c,
                negative_prompt_embeds = npe,
                negative_pooled_prompt_embeds = npooled,
                num_inference_steps = sample_plan.steps,
                guidance_scale = sample_plan.guidance,
                height = res,
                width = res,
                latents = noise.to(weight_dtype),
                generator = torch.Generator(device = "cpu").manual_seed(
                    sample_seed(sample_plan, index)
                ),
                output_type = "latent",
                return_dict = False,
                callback_on_step_end = _interrupt_on_stop,
            )[0]
            if stop_latched:
                raise SampleRoundStopped()
            return sample_vae.decode(latents.float() / vae_scale).sample

        # The stop poll drains the request, so one seen mid-round is latched here for the loop to act on.
        stop_latched = False

        def _stop_during_sample() -> bool:
            nonlocal stop_latched
            if not stop_latched and _check_stop():
                stop_latched = True
            return stop_latched

        def _sample(step: int) -> None:
            nonlocal sample_plan, t_rate, t_steady
            if sample_plan is None:
                return
            t_round = time.time()
            parked = next(sample_vae.parameters()).device.type == "cpu"
            try:
                with isolated_sampling(device, compiled):
                    unet.eval()
                    if parked:
                        sample_vae.to(device)
                    run_sample_round(
                        sample_plan, step, _render_sdxl, on_event, _emit, _stop_during_sample
                    )
            except Exception as exc:  # noqa: BLE001 -- previews never cost the run
                _emit(on_event, "warning", message = f"Sample images disabled after an error: {exc}")
                sample_plan = None
            finally:
                unet.train()
                if parked:
                    sample_vae.to("cpu")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                spent = time.time() - t_round
                t_rate += spent
                if t_steady is not None:
                    t_steady += spent

        if sample_plan is not None:
            try:
                pipe.set_progress_bar_config(disable = True)
            except Exception:  # noqa: BLE001
                pass
            if resumed < cfg.train_steps:
                _sample(resumed)

        for opt_step in range(resumed, cfg.train_steps):
            if stop_latched:
                stopped = True
                break
            optimizer.zero_grad(set_to_none = True)
            step_loss = 0.0
            for _ in range(cfg.gradient_accumulation_steps):
                idx, img_paths, captions = _next_batch()
                if latent_cache is not None:
                    latents, batch_time_ids = _sample_sdxl_cached_latents(
                        latent_cache, idx, variant_rng, device, weight_dtype
                    )
                else:
                    loaded = [
                        _load_image_tensor(
                            p,
                            cfg.resolution if canvases is None else canvases[i],
                            cfg.center_crop,
                            cfg.random_flip,
                            rng,
                        )
                        for i, p in zip(idx, img_paths)
                    ]
                    pixel_values = torch.stack([t for t, _ in loaded]).to(
                        device, dtype = torch.float32
                    )
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

                scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
                step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps
                micro += 1

            # max_grad_norm <= 0 disables clipping; clip_grad_norm_(0.0) would zero all grads.
            grad_norm = None
            scaler.unscale_(optimizer)
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm))
                if not math.isfinite(grad_norm):
                    grad_norm = None
            scale_before = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            # An overflow step lowers the scale and skips the update; the schedule must not run ahead of the weights.
            if scale_before is None or scaler.get_scale() >= scale_before:
                lr_sched.step()

            running_loss += step_loss
            done = opt_step + 1
            now = time.time()
            # Only steps this process ran count toward rates.
            ran_here = done - resumed
            if ran_here == 1:
                # Step 1 pays the one-time costs (cudnn autotune, compile warmup), so the rate starts after it.
                t_steady = now
            if done % cfg.log_every == 0 or done == cfg.train_steps:
                # ``learning_rate`` (not ``lr``) is the field the Unsloth training pump reads.
                if device == "cuda":
                    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                per_step = cfg.train_batch_size * cfg.gradient_accumulation_steps
                if t_steady is not None and ran_here > 1:
                    samples_per_second = round(
                        (ran_here - 1) * per_step / max(now - t_steady, 1e-6), 3
                    )
                else:
                    samples_per_second = round(ran_here * per_step / max(now - t_rate, 1e-6), 3)
                _emit(
                    on_event,
                    "progress",
                    step = done,
                    total_steps = cfg.train_steps,
                    loss = round(step_loss, 5),
                    avg_loss = round(running_loss / done, 5),
                    learning_rate = lr_sched.get_last_lr()[0],
                    grad_norm = round(grad_norm, 5) if grad_norm is not None else None,
                    samples_per_second = samples_per_second,
                    peak_memory_gb = peak_gb or None,
                )

            stop_now = _check_stop()
            if not stop_now and sample_plan is not None and sample_plan.due(done, cfg.train_steps):
                _sample(done)
                stop_now = stop_latched
            # Skipped on the final step and when stopping, since the stop path writes one at the exact step.
            if (
                not stop_now
                and cfg.save_steps
                and done % cfg.save_steps == 0
                and done < cfg.train_steps
            ):
                _save_checkpoint(done)
            if stop_now:
                stopped = True
                break

        lora_path: Optional[str] = None
        catalog_path: Optional[str] = None
        if not (stopped and not save_on_stop):
            out_dir.mkdir(parents = True, exist_ok = True)
            # Write the bundle before the adapter export so a failed export still leaves it.
            if stopped and done > 0:
                _save_checkpoint(done)
            unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory = str(out_dir),
                unet_lora_layers = unet_lora,
                safe_serialization = True,
                weight_name = DEFAULT_LORA_FILENAME,
            )
            lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
            # ``done``, not cfg.train_steps: a stop at 11/500 must not advertise 500.
            catalog_path = _publish_to_lora_catalog(lora_path, cfg, done)
            if not stopped:
                # Retire this run's own bundles so a later resume does not roll back to them.
                retire_own_checkpoints(out_dir, preexisting_checkpoints, resumed_here = resumed_here)
            elif not resumed_here:
                # A lower-step partial would be outranked by the earlier run's leftovers.
                discard_preexisting_checkpoints(out_dir, preexisting_checkpoints)
        else:
            # save_steps writes bundles as the run goes, so without this a discard leaves up to
            # save_total_limit copies of the optimizer state in a directory the user got no artifact from:
            # invisible to every scanner, unresumable, with no delete path in the UI.
            clear_own_checkpoints(out_dir, preexisting_checkpoints)
            discard_samples(sample_owner)
            try:
                out_dir.rmdir()
            except OSError:
                pass
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
            resumed_from_step = resumed or None,
            discarded = bool(stopped and not save_on_stop),
        )
        return str(out_dir)
    finally:
        _restore_perf_flags(snap)


def _make_lora_optimizer(params: list, lr: float) -> Any:
    """UNSLOTH_DIFFUSION_FP32_OPTIM forces plain non-fused AdamW, as the accuracy guard requires."""
    import torch

    if os.environ.get("UNSLOTH_DIFFUSION_FP32_OPTIM", "") in ("1", "true"):
        return torch.optim.AdamW(params, lr = lr)
    # Check before construction: on XPU the 8-bit optimizer only fails at the first step().
    if bitsandbytes_optimizer_supported():
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr = lr)
        except Exception:  # noqa: BLE001 -- bnb missing / no CUDA: fall back to torch AdamW
            pass
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, lr = lr, fused = True)
        except Exception:  # noqa: BLE001 -- fused unsupported on this build/device
            pass
    return torch.optim.AdamW(params, lr = lr)


def run_diffusion_training_process(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Unexpected exceptions become an error event on the queue, not a silent subprocess crash."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def on_event(ev: dict) -> None:
        event_queue.put(ev)

    def should_stop() -> Any:
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
        cfg = _config_from_dict(config).normalized()
        trainer = get_trainer(cfg.resolved_family)
        trainer(cfg, on_event = on_event, should_stop = should_stop)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as an error event
        # Emit both keys: the diffusion service reads ``message``, the generic worker ``error``.
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
