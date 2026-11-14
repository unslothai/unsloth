# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 on Diffusers, made to follow the reference implementation's sampling and conditioning.

Installed per render by ``install_h3_reference_recipe(pipe, ...)``, which swaps classes in place on the pipeline's own
blocks and schedulers (as ``keep_h3_reference_image_sizes`` does); nothing global is patched.

- Sigma grid: the ``simple`` schedule read off the 1000-entry shifted table, so N steps that do not divide 1000 floor
  to it; the transformer's video and audio timesteps are derived the way the reference does it (the audio one from
  the video sigma, not from its own grid).
- Sampler: ``res_multistep`` (second-order multistep, eta 0, one evaluation per step) by default. The audio stream is
  stepped in the carried variable ``(sigma_v / sigma_a) * x_a`` on the video sigmas, which is what makes it equal the
  reference rather than a second-order step in the audio space. Euler stays for PDD (turbo) banks.
  ``UNSLOTH_H3_SAMPLER=euler`` brings back Euler for every checkpoint.
- Noise: video then audio from one CPU generator seeded with the request seed, audio drawn as ``[1, 32, 2, T]``;
  conditioning noise from its own CPU generator re-seeded with the seed for every condition, in packed-row shape.
- Ref2VA vision input: each two-frame block sized on its own (3136 to 12,845,056 pixels, factor 32, bilinear), from
  the frames after the ``17 n + 5`` snap; image references get the same limits. ``UNSLOTH_H3_VISION_BUDGET=clip``
  brings back the processor's whole-clip budget for videos (fewer tokens on long references).
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any, Optional

H3_SAMPLER_ENV = "UNSLOTH_H3_SAMPLER"
H3_SAMPLER_RES_MULTISTEP = "res_multistep"
H3_SAMPLER_EULER = "euler"
H3_VISION_BUDGET_ENV = "UNSLOTH_H3_VISION_BUDGET"

# Qwen3-VL input limits of the reference conditioner, per image or per two-frame block.
H3_VISION_MIN_PIXELS = 3136
H3_VISION_MAX_PIXELS = 12_845_056
H3_VISION_MEAN = 0.5
H3_VISION_STD = 0.5
# The reference reads a reference video at 2 fps.
H3_VISION_SAMPLE_FPS = 2.0
H3_VISION_MIN_FRAMES = 5

# The shifted sigma table the ``simple`` schedule indexes.
H3_SIGMA_TABLE_SIZE = 1000


def h3_sampler(pdd: bool) -> str:
    """The sampler for a render: Euler for PDD (turbo) banks, which are distilled for it, else res_multistep."""
    if pdd:
        return H3_SAMPLER_EULER
    raw = (os.environ.get(H3_SAMPLER_ENV) or "").strip().lower()
    return H3_SAMPLER_EULER if raw == H3_SAMPLER_EULER else H3_SAMPLER_RES_MULTISTEP


def h3_vision_per_block() -> bool:
    return (os.environ.get(H3_VISION_BUDGET_ENV) or "").strip().lower() != "clip"


# ── sigma grid ──


def time_snr_shift(alpha: float, t: Any) -> Any:
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def time_shift_sigma(sigma: Any, from_shift: float, to_shift: float) -> Any:
    """Move a sigma from one shift to another through the shared base grid."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def h3_simple_sigmas(shift: float, steps: int) -> Any:
    """``steps`` sigmas plus a terminal 0, float32: entry ``int(i * 1000 / steps)`` from the top of the 1000-entry
    shifted table. Equal to the shifted ``linspace(1, 0, steps + 1)`` when ``steps`` divides 1000, floored to the
    table otherwise."""
    import torch

    multiplier = H3_SIGMA_TABLE_SIZE
    table = time_snr_shift(
        float(shift),
        ((torch.arange(1, H3_SIGMA_TABLE_SIZE + 1, 1) / H3_SIGMA_TABLE_SIZE) * multiplier)
        / multiplier,
    )
    stride = len(table) / int(steps)
    values = [float(table[-(1 + int(x * stride))]) for x in range(int(steps))] + [0.0]
    return torch.tensor(values, dtype = torch.float32)


def h3_model_sigma(sigmas: Any) -> Any:
    """The sigma the denoiser is conditioned on: the sampler hands it ``sigma * 1000`` and reads it back."""
    return (sigmas * H3_SIGMA_TABLE_SIZE / float(H3_SIGMA_TABLE_SIZE)).float().clamp(min = 1e-6)


# ── res_multistep ──


def res_multistep_coefficients(sigmas: Any, i: int) -> Optional[tuple]:
    """``(exp(-h), h * b1, h * b2)`` pieces for step ``i`` of a second-order res_multistep (eta 0), or None where it
    takes an Euler step (the first step and the step onto sigma 0). ``sigmas`` is the float32 CPU grid."""
    import torch

    sigma, sigma_next = sigmas[i], sigmas[i + 1]
    if i == 0 or float(sigma_next) == 0.0:
        return None
    t_fn = lambda s: s.log().neg()  # noqa: E731
    phi1_fn = lambda t: torch.expm1(t) / t  # noqa: E731
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t  # noqa: E731
    # eta 0: sigma_down is the next sigma, and the previous sigma_down is this one
    t, t_old, t_next, t_prev = t_fn(sigma), t_fn(sigma), t_fn(sigma_next), t_fn(sigmas[i - 1])
    h = t_next - t
    c2 = (t_prev - t_old) / h
    phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)
    b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan = 0.0)
    b2 = torch.nan_to_num(phi2_val / c2, nan = 0.0)
    return h, b1, b2


def res_multistep_update(
    x: Any, denoised: Any, old_denoised: Optional[Any], sigmas: Any, i: int
) -> Any:
    """One res_multistep step (eta 0) from ``x`` at ``sigmas[i]`` given this step's and the last step's x0."""
    coefficients = None if old_denoised is None else res_multistep_coefficients(sigmas, i)
    sigma, sigma_next = sigmas[i], sigmas[i + 1]
    if coefficients is None:
        d = (x - denoised) / sigma
        return x + d * (sigma_next - sigma)
    h, b1, b2 = coefficients
    return h.neg().exp() * x + h * (b1 * denoised + b2 * old_denoised)


# ── schedulers ──


@functools.lru_cache(maxsize = None)
def _scheduler_class(base: type) -> type:
    class StudioMiniMaxH3Scheduler(base):
        """The stock scheduler on the reference grid, with res_multistep or Euler. ``_studio_stream`` is "video" or
        "audio"; the audio one reads its sigmas off the video scheduler it is bound to."""

        def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device: Any = None,
            sigmas: Any = None,
        ) -> None:
            import torch

            state = self.__dict__.setdefault("_studio", {})
            if sigmas is not None or num_inference_steps is None or int(num_inference_steps) < 2:
                state.clear()
                return super().set_timesteps(num_inference_steps, device = device, sigmas = sigmas)
            if state.get("stream") == "audio":
                video = state["video"]
                video_state = video.__dict__.get("_studio") or {}
                if "sigmas_cpu" not in video_state or video.num_inference_steps + 1 != int(
                    num_inference_steps
                ):
                    raise RuntimeError("MiniMax-H3: set the video schedule before the audio one.")
                raw = video_state["sigmas_cpu"]
                model_sigma = time_shift_sigma(
                    h3_model_sigma(raw[:-1]), float(video._shift), float(self._shift)
                )
                grid = torch.cat([model_sigma, raw.new_zeros(1)])
                state.update(
                    sigmas_cpu = raw,  # the carried variable steps on the video sigmas
                    audio_sigmas_cpu = grid,
                    scale = float(video._shift) / float(self._shift),
                )
                self.sigmas = grid.to(device = device)
                self.timesteps = (1.0 - model_sigma).to(device = device)
            else:
                raw = h3_simple_sigmas(float(self._shift), int(num_inference_steps) - 1)
                state.update(sigmas_cpu = raw)
                self.sigmas = raw.to(device = device)
                self.timesteps = (1.0 - h3_model_sigma(raw[:-1])).to(device = device)
            state.update(old_denoised = None, carried = None)
            self.num_inference_steps = int(self.timesteps.numel())
            self._step_index = None
            self._begin_index = None

        def step(
            self,
            model_output: Any,
            timestep: Any,
            sample: Any,
            return_dict: bool = True,
        ) -> Any:
            state = self.__dict__.get("_studio") or {}
            if state.get("sampler") != H3_SAMPLER_RES_MULTISTEP or "sigmas_cpu" not in state:
                return super().step(model_output, timestep, sample, return_dict = return_dict)
            import torch

            if self._step_index is None:
                self._step_index = (
                    self.index_for_timestep(timestep)
                    if self._begin_index is None
                    else self._begin_index
                )
            i = self._step_index
            sigmas = state["sigmas_cpu"]
            dtype = sample.dtype
            x = sample.to(torch.float32)
            velocity = model_output.to(torch.float32)
            sigma = sigmas[i]
            if state.get("stream") == "audio":
                # The carried variable y = (sigma_v / sigma_a) * x_a is an ordinary flow on the video sigmas whose
                # target is scale * x0_a; its x0 is y - sigma_v * dy/dsigma_v, written as the reference writes it.
                scale = state["scale"]
                sigma_a = state["audio_sigmas_cpu"][i]
                carried = state.get("carried")
                if carried is None or i == 0 or carried.shape != x.shape:
                    carried = x * (h3_model_sigma(sigma) / sigma_a)
                dy = (1.0 - scale) * x + (1.0 + (scale - 1.0) * sigma_a) * velocity.neg()
                denoised = carried - dy * sigma
                y_next = res_multistep_update(
                    carried, denoised, state.get("old_denoised"), sigmas, i
                )
                sigma_next = sigmas[i + 1]
                if float(sigma_next) == 0.0:
                    out = y_next / scale
                else:
                    out = y_next * (state["audio_sigmas_cpu"][i + 1] / h3_model_sigma(sigma_next))
                state["carried"] = y_next
            else:
                denoised = x + velocity * sigma
                out = res_multistep_update(x, denoised, state.get("old_denoised"), sigmas, i)
            state["old_denoised"] = denoised
            self._step_index += 1
            out = out.to(dtype)
            if not return_dict:
                return (out,)
            from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput

            return MiniMaxH3SchedulerOutput(prev_sample = out)

    StudioMiniMaxH3Scheduler.__qualname__ = StudioMiniMaxH3Scheduler.__name__
    return StudioMiniMaxH3Scheduler


def install_h3_schedulers(pipe: Any, sampler: str) -> bool:
    """Put the pipeline's video and audio schedulers on the reference grid with ``sampler``; False when the pipeline
    has no MiniMax-H3 scheduler pair."""
    try:
        from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    except Exception:  # noqa: BLE001 - diffusers wraps a failed lazy import in RuntimeError
        return False
    video = getattr(pipe, "scheduler", None)
    audio = getattr(pipe, "audio_scheduler", None)
    if not isinstance(video, MiniMaxH3Scheduler) or not isinstance(audio, MiniMaxH3Scheduler):
        return False
    cls = _scheduler_class(MiniMaxH3Scheduler)
    for scheduler, stream in ((video, "video"), (audio, "audio")):
        if not isinstance(scheduler, cls):
            scheduler.__class__ = cls
        state = scheduler.__dict__.setdefault("_studio", {})
        state.clear()
        state.update(stream = stream, sampler = sampler)
    audio.__dict__["_studio"]["video"] = video
    return True


# ── noise ──


def _request_seed(generator: Any) -> Optional[int]:
    import torch
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        return None
    return int(generator.initial_seed())


def h3_reference_noise(seed: int, video_shape: tuple, audio_shape: tuple) -> tuple:
    """Video noise, then audio noise ``[1, 32, 2, T]``, from one CPU generator seeded with ``seed``."""
    import torch

    generator = torch.Generator("cpu").manual_seed(int(seed))
    video = torch.randn(video_shape, generator = generator, dtype = torch.float32)
    audio = torch.randn(audio_shape, generator = generator, dtype = torch.float32)
    return video, audio


def h3_condition_rows(condition: Any, patch_size: Any, seed: int, noise_aug: float) -> Any:
    """One visual condition patchified to rows and noised to ``noise_aug``, with noise drawn in row shape from a CPU
    generator re-seeded with ``seed`` (every condition restarts the same stream). Computed on the CPU."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    rows = patchify_video_latents(condition.detach().to("cpu", torch.float32), patch_size)
    if noise_aug < 1.0:
        generator = torch.Generator("cpu").manual_seed(int(seed))
        noise = torch.randn(rows.shape, generator = generator, dtype = torch.float32)
        rows = noise_aug * rows + (1.0 - noise_aug) * noise
    return rows


@functools.lru_cache(maxsize = None)
def _prepare_latents_class(base: type) -> type:
    class StudioMiniMaxH3PrepareLatentsStep(base):
        def __call__(self, components: Any, state: Any) -> Any:
            seed = _request_seed(state.get("generator"))
            if (
                seed is None
                or state.get("latents") is not None
                or state.get("audio_latents") is not None
            ):
                return super().__call__(components, state)
            video, audio = h3_reference_noise(
                seed,
                (
                    1,
                    components.vae_latent_channels,
                    state.get("num_latent_frames"),
                    state.get("latent_height"),
                    state.get("latent_width"),
                ),
                (
                    1,
                    components.audio_latent_channels,
                    components.audio_channels,
                    state.get("num_audio_latents"),
                ),
            )
            state.set("latents", video)
            # the block takes (channels, 32, T) and packs it channel-major
            state.set("audio_latents", audio[0].permute(1, 0, 2))
            return super().__call__(components, state)

    StudioMiniMaxH3PrepareLatentsStep.__qualname__ = StudioMiniMaxH3PrepareLatentsStep.__name__
    return StudioMiniMaxH3PrepareLatentsStep


@functools.lru_cache(maxsize = None)
def _prepare_condition_class(base: type) -> type:
    class StudioMiniMaxH3PrepareConditionLatentsStep(base):
        def __call__(self, components: Any, state: Any) -> Any:
            import torch

            seed = _request_seed(state.get("generator"))
            if seed is None:
                return super().__call__(components, state)
            block_state = self.get_block_state(state)
            device = components._execution_device
            aug = float(components.keyframe_noise_aug)
            packed = [
                h3_condition_rows(condition, components.patch_size, seed, aug)
                for condition in block_state.condition_latents
            ]
            block_state.condition_rows = torch.cat(packed).to(device)
            if block_state.condition_rows.shape[0] != block_state.num_condition_video_rows:
                raise ValueError(
                    f"The layout reserved {block_state.num_condition_video_rows} conditioning rows but the encoded "
                    f"conditioning latents pack into {block_state.condition_rows.shape[0]}."
                )
            self.set_block_state(state, block_state)
            return components, state

    StudioMiniMaxH3PrepareConditionLatentsStep.__qualname__ = (
        StudioMiniMaxH3PrepareConditionLatentsStep.__name__
    )
    return StudioMiniMaxH3PrepareConditionLatentsStep


# ── Qwen3-VL vision input ──


def h3_smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple:
    """Qwen2-VL's target size: multiples of ``factor``, pixel count within ``[min_pixels, max_pixels]``."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def h3_vision_patches(
    frames: Any,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> tuple:
    """``[T, H, W, 3]`` uint8 frames (one image repeated, or a two-frame block) to Qwen3-VL's flattened patches and
    their ``(grid_h, grid_w)``: per-frame limits, bilinear float resize, (x - 0.5) / 0.5."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32) / 255.0
    _, height, width, _ = pixels.shape
    pixels = pixels.permute(0, 3, 1, 2)
    h_bar, w_bar = h3_smart_resize(
        height, width, patch_size * merge_size, H3_VISION_MIN_PIXELS, H3_VISION_MAX_PIXELS
    )
    pixels = F.interpolate(pixels, size = (h_bar, w_bar), mode = "bilinear", align_corners = False)
    pixels = (pixels - H3_VISION_MEAN) / H3_VISION_STD
    grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
    grid_t = pixels.shape[0] // temporal_patch_size
    patches = pixels.reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(
        grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size
    )
    return flat, (grid_h, grid_w)


def h3_snap_reference_frames(num_frames: int) -> int:
    """A reference clip's frame count after the snap down to ``17 n + 5`` (at least 5 frames)."""
    n = int(num_frames)
    if n < H3_VISION_MIN_FRAMES:
        raise ValueError(
            "MiniMax-H3 reference videos need at least 5 frames (about 0.2 s at 24 fps)."
        )
    while n % 17 != 5:
        n -= 1
    return n


def h3_vision_frame_indices(num_frames: int, fps: float) -> tuple[list, list]:
    """Indices of the 2 fps frames the conditioner reads, after the snap, and one timestamp per two-frame block."""
    stride = int(round(fps / H3_VISION_SAMPLE_FPS))
    indices = list(range(0, h3_snap_reference_frames(num_frames), stride))
    timestamps = [i / H3_VISION_SAMPLE_FPS for i in range(len(indices))]
    if len(indices) % 2:
        indices.append(indices[-1])
        timestamps.append(timestamps[-1])
    blocks = [(timestamps[i] + timestamps[i + 1]) / 2.0 for i in range(0, len(timestamps), 2)]
    return indices, blocks


@functools.lru_cache(maxsize = None)
def _ref2va_text_encoder_class(base: type) -> type:
    class StudioMiniMaxH3Ref2VATextEncoderStep(base):
        def _gather_vision_features(self, processor: Any, references: list, fps: float) -> tuple:
            import numpy as np
            import torch

            if not h3_vision_per_block():
                # Whole-clip budget for videos, but still from the frames after the snap.
                snapped = [_snapped_reference(ref) for ref in references]
                return super()._gather_vision_features(processor, snapped, fps)
            image_processor = processor.image_processor
            patch, merge = int(image_processor.patch_size), int(image_processor.merge_size)
            temporal = int(processor.video_processor.temporal_patch_size)
            per_token = merge * merge
            vision_inputs: dict = {}

            image_pixels, image_grids, image_tokens = [], [], []
            for reference in references:
                if reference.kind != "image":
                    continue
                pixels = np.asarray(reference.image.convert("RGB"))[None].repeat(temporal, axis = 0)
                flat, (gh, gw) = h3_vision_patches(
                    pixels, patch_size = patch, temporal_patch_size = temporal, merge_size = merge
                )
                image_pixels.append(flat)
                image_grids.append([1, gh, gw])
                image_tokens.append(gh * gw // per_token)
            if image_pixels:
                vision_inputs["pixel_values"] = torch.cat(image_pixels)
                vision_inputs["image_grid_thw"] = torch.tensor(image_grids, dtype = torch.long)

            video_pixels, video_grids, video_tokens, video_timestamps = [], [], [], []
            for reference in references:
                if reference.kind != "video":
                    continue
                frames = reference.frames
                indices, blocks = h3_vision_frame_indices(frames.shape[0], fps)
                grid = None
                for start in range(0, len(indices), temporal):
                    flat, block_grid = h3_vision_patches(
                        frames[indices[start : start + temporal]],
                        patch_size = patch,
                        temporal_patch_size = temporal,
                        merge_size = merge,
                    )
                    grid = block_grid
                    video_pixels.append(flat)
                # One video grid of len(blocks) frames: Qwen3-VL attends and positions each frame on its own,
                # exactly as for that many single-frame grids.
                video_grids.append([len(blocks), grid[0], grid[1]])
                video_tokens.append(grid[0] * grid[1] // per_token)
                video_timestamps.append(blocks)
            if video_pixels:
                vision_inputs["pixel_values_videos"] = torch.cat(video_pixels)
                vision_inputs["video_grid_thw"] = torch.tensor(video_grids, dtype = torch.long)
            return vision_inputs, image_tokens, video_tokens, video_timestamps

    StudioMiniMaxH3Ref2VATextEncoderStep.__qualname__ = (
        StudioMiniMaxH3Ref2VATextEncoderStep.__name__
    )
    return StudioMiniMaxH3Ref2VATextEncoderStep


class _ReferenceImageProcessor:
    """Stands in for Qwen3-VL's image processor inside one call: the reference limits and arithmetic for pictures,
    everything else read through to the processor it wraps."""

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(
        self,
        images: Any = None,
        return_tensors: Any = None,
        **kwargs: Any,
    ) -> dict:
        import numpy as np
        import torch

        patch, merge = int(self._inner.patch_size), int(self._inner.merge_size)
        temporal = int(getattr(self._inner, "temporal_patch_size", 2) or 2)
        pixels, grids = [], []
        for image in images:
            frames = np.asarray(image.convert("RGB"))[None].repeat(temporal, axis = 0)
            flat, (gh, gw) = h3_vision_patches(
                frames, patch_size = patch, temporal_patch_size = temporal, merge_size = merge
            )
            pixels.append(flat)
            grids.append([1, gh, gw])
        return {
            "pixel_values": torch.cat(pixels),
            "image_grid_thw": torch.tensor(grids, dtype = torch.long),
        }


@functools.lru_cache(maxsize = None)
def _fl2va_text_encoder_class(base: type) -> type:
    class StudioMiniMaxH3FL2VATextEncoderStep(base):
        def __call__(self, components: Any, state: Any) -> Any:
            processor = components.processor
            stock = processor.image_processor
            if isinstance(stock, _ReferenceImageProcessor):
                return super().__call__(components, state)
            processor.image_processor = _ReferenceImageProcessor(stock)
            try:
                return super().__call__(components, state)
            finally:
                processor.image_processor = stock

    StudioMiniMaxH3FL2VATextEncoderStep.__qualname__ = StudioMiniMaxH3FL2VATextEncoderStep.__name__
    return StudioMiniMaxH3FL2VATextEncoderStep


def _snapped_reference(reference: Any) -> Any:
    if getattr(reference, "kind", None) != "video":
        return reference
    import dataclasses

    n = h3_snap_reference_frames(reference.frames.shape[0])
    return dataclasses.replace(reference, frames = reference.frames[:n])


# ── condition encode ──


@functools.lru_cache(maxsize = None)
def _fixed_conv_encode_class(base: type) -> type:
    class StudioMiniMaxH3ConditionEncodeStep(base):
        """The condition VAE encode with ``cudnn.benchmark`` held off: a timed search picks the convolution
        algorithm per process, so the same keyframe or reference encoded to different latents in every fresh
        process (and the whole render followed). The heuristic pick is the same every time."""

        def __call__(self, components: Any, state: Any) -> Any:
            from .diffusion_speed import cudnn_benchmark_scope
            with cudnn_benchmark_scope(False):
                return super().__call__(components, state)

    StudioMiniMaxH3ConditionEncodeStep.__qualname__ = StudioMiniMaxH3ConditionEncodeStep.__name__
    return StudioMiniMaxH3ConditionEncodeStep


# ── install ──


def _swap_blocks(pipe: Any, swaps: dict) -> dict:
    """Swap every block of a listed class (in ``pipe._blocks``, which the pipeline runs) for its subclass."""
    counts = {base.__name__: 0 for base in swaps}
    stack, seen = [getattr(pipe, "_blocks", None)], set()
    while stack:
        block = stack.pop()
        if block is None or id(block) in seen:
            continue
        seen.add(id(block))
        for base, sub in swaps.items():
            if type(block) is base or type(block) is sub:
                if type(block) is base:
                    block.__class__ = sub
                counts[base.__name__] += 1
        stack.extend((getattr(block, "sub_blocks", None) or {}).values())
    return counts


def install_h3_reference_recipe(pipe: Any, *, pdd: bool = False) -> dict:
    """Put one MiniMax-H3 Diffusers pipeline on the reference recipe for the next render. Idempotent; returns what it
    switched (block class -> count, plus the sampler)."""
    try:
        from diffusers.modular_pipelines.minimax_h3 import before_denoise, encoders
    except Exception:  # noqa: BLE001 - diffusers wraps a failed lazy import in RuntimeError
        return {}
    swaps = {
        before_denoise.MiniMaxH3PrepareLatentsStep: _prepare_latents_class(
            before_denoise.MiniMaxH3PrepareLatentsStep
        ),
        before_denoise.MiniMaxH3PrepareConditionLatentsStep: _prepare_condition_class(
            before_denoise.MiniMaxH3PrepareConditionLatentsStep
        ),
        encoders.MiniMaxH3Ref2VATextEncoderStep: _ref2va_text_encoder_class(
            encoders.MiniMaxH3Ref2VATextEncoderStep
        ),
        encoders.MiniMaxH3FL2VATextEncoderStep: _fl2va_text_encoder_class(
            encoders.MiniMaxH3FL2VATextEncoderStep
        ),
        encoders.MiniMaxH3KeyframeVaeEncoderStep: _fixed_conv_encode_class(
            encoders.MiniMaxH3KeyframeVaeEncoderStep
        ),
        encoders.MiniMaxH3Ref2VAReferenceEncoderStep: _fixed_conv_encode_class(
            encoders.MiniMaxH3Ref2VAReferenceEncoderStep
        ),
    }
    report: dict = dict(_swap_blocks(pipe, swaps))
    sampler = h3_sampler(pdd)
    report["sampler"] = sampler if install_h3_schedulers(pipe, sampler) else None
    return report
