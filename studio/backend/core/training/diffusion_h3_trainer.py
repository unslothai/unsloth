# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Joint video + audio LoRA training for MiniMax-H3.

MiniMax-H3 is rectified flow like every family in ``diffusion_dit_trainer``, but three things
put it outside that trainer's ``_FamilySpec`` seams rather than inside them, so it gets its
own loop here and shares ``diffusion_train_common`` (config, events, stop, trust gate,
publishing, EMA, the 8-bit optimizer) with everything else:

1. **One packed sequence, two modalities.** H3 runs a single block stack over one 1-D
   sequence holding ``[text | audio | video]`` rows with full self-attention. There is no
   cross-attention, no per-modality block weights and no ``isolate_modalities`` escape hatch
   (LTX-2 has all three). So a LoRA on the attention/MLP projections is applied to the audio
   rows as well as the video ones, and the audio rows are keys and values for every video
   row. Audio therefore cannot be excluded from the adapter, and the only honest objective is
   the model's own: regress BOTH velocities. That is why the dataset unit is a clip with
   sound (``diffusion_h3_clips``) and why there is no still-image milestone here.

2. **Two coupled schedules.** Video is noised through an exponential shift of 12.0 and audio
   through 3.0. At inference both are indexed by the same step, so the pair
   ``(sigma_video, sigma_audio)`` traverses one curve. Training draws ONE base ``u`` and
   pushes it through both shifts, which reproduces that curve exactly; drawing the two
   independently would train pairs the sampler never visits. ``MiniMaxH3Scheduler`` is also
   not a ``FlowMatchEulerDiscreteScheduler`` (reversed velocity sign, ``t = 1 - sigma`` in
   [0, 1], a different sigma grid), so the shared trainer's sigma table does not apply.

3. **It is modular-only.** There is no ``MiniMaxH3Pipeline``; the integration is a
   ``ModularPipeline`` plus blocks. The trainer never builds one: the two layout builders it
   needs are ``@staticmethod``s, and the components load individually -- which is a better fit
   for the phased load than the ``transformer = None`` trick, since ``load_components`` takes
   the component names outright.

Velocity sign: H3 predicts a DATA-ward velocity (``x0 = x_t + sigma * v``), i.e. the target is
``latents - noise``, the negation of the convention in ``diffusion_dit_trainer``.

Memory: the Qwen3-VL-32B conditioner is 63 GiB on disk, so captions are encoded once up front
and it is freed before the VAEs encode and long before the 66 GB transformer loads -- the same
phased load the DiT trainer uses, expressed through ``load_components``.
"""

from __future__ import annotations

import gc
import random
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from core.training.diffusion_h3_clips import (
    H3_AUDIO_CHANNELS,
    H3_AUDIO_LATENT_CHANNELS,
    H3_AUDIO_TAG,
    H3_CANVAS_MULTIPLE,
    H3_FRAMES_PER_CHUNK,
    H3_LATENTS_PER_CHUNK,
    H3_PIXEL_MEAN,
    H3_PIXEL_STD,
    H3_SPATIAL_COMPRESSION,
    H3_TEXT_TAG,
    H3_TRAIN_NUM_FRAMES,
    H3_VIDEO_LATENT_CHANNELS,
    H3_VIDEO_TAG,
    decode_clip,
    discover_clip_caption_pairs,
    h3_audio_latent_count,
    h3_packed_sequence_length,
    h3_train_canvas,
    h3_video_latent_frames,
)
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
    _restore_perf_flags,
    native_bf16_supported,
    resolve_train_steps,
)
from core.training.diffusion_train_extras import LoRAEMA, save_ema_adapter

# The video-stream and audio-stream projections of every block, fully qualified.
#
# There is nothing to exclude: H3 has ONE set of block weights serving all three modalities,
# so this is the whole adapter surface and the audio rows are trained through the same
# matrices as the video rows. What IS excluded is deliberate:
#   - ``adaln_proj.linear`` projects the timestep embedding into the modulation table. It is
#     40% of the checkpoint's parameters, but its input is a (num_timesteps, 2688) tensor with
#     two or three rows, so a rank-r adapter there sees a handful of samples per step and
#     learns noise.
#   - ``proj_in`` / ``audio_proj_in`` / ``proj_out`` / ``audio_proj_out`` / ``context_embedder``
#     are the patch and text projections, kept in fp32 by the checkpoint's own
#     ``_keep_in_fp32_modules``; adapting them mixes precisions for no benefit.
#   - the ``token_refiner`` blocks carry ``attn`` and ``ff`` under the SAME leaf names
#     (``token_refiner.refiner_blocks.0.attn.to_q``), so PEFT's suffix rule for a LIST of
#     target names would adapt the text refiner as well as the denoiser stack.
#
# That last point is why the targets are a REGEX and not a list: PEFT globs nothing, it either
# suffix-matches a list or ``re.fullmatch``es a string, so qualifying a list entry with
# ``transformer_blocks.*.`` matches nothing at all and the adapter silently trains zero
# parameters (the trainer also refuses an empty adapter, so both halves of that mistake are
# caught).
_H3_TARGET_LEAVES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_H3_TARGETS = r"transformer_blocks\.\d+\.(?:" + "|".join(
    leaf.replace(".", r"\.") for leaf in _H3_TARGET_LEAVES
) + ")"

# Modules whose weights the transformer reads a DTYPE off to align an activation with
# (``x.to(self.linear.weight.dtype)``). A bitsandbytes ``Params4bit`` reports ``uint8``, so
# quantizing one of these casts the activation to Byte and the next norm dies with
# ``"rms_norm" not implemented for 'Byte'``. Keeping them dense is the whole fix; it costs the
# nf4 saving on ``adaln_proj`` (36.6 GB resident instead of ~17 GB), which is still well under
# the 66.3 GB dense weight. torchao int8 is unaffected -- its tensor subclass reports the
# logical bfloat16 -- so ``base_precision="int8"`` quantizes everything.
_H3_NF4_SKIP_MODULES = ("context_embedder", "adaln_proj", "norm_out")

# The exponential sigma shifts of the two schedules, from the released scheduler configs.
_H3_VIDEO_SHIFT = 12.0
_H3_AUDIO_SHIFT = 3.0

# Which Qwen3-VL hidden state conditions the transformer. The last layer is post-norm and is
# not what the released weights were trained against.
_H3_TEXT_ENCODER_LAYER = 50

# Components the conditioning phase needs, and the one the training phase needs. Naming them
# is what keeps the 66 GB transformer off the device while the 63 GiB conditioner is on it.
_H3_TEXT_COMPONENTS = ("text_encoder", "tokenizer", "processor")
# The VAEs load in fp32: both carry modules diffusers refuses to cast (their encoders,
# decoders and projection heads), so a bf16 load followed by a .to(float32) both warns and
# leaves the cast half-applied.
_H3_VAE_COMPONENTS = ("vae", "audio_vae")
_H3_CONDITIONING_COMPONENTS = _H3_TEXT_COMPONENTS + _H3_VAE_COMPONENTS


def _shifted_sigma(u: float, shift: float) -> float:
    """The exponential shift both H3 schedules apply: ``s*u / (1 + (s - 1) * u)``."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def _assert_component_grid(pipe: Any) -> None:
    """Check the constants in ``diffusion_h3_clips`` against the loaded components.

    Those constants drive the packed layout, and a wrong one produces a silently misaligned
    sequence rather than an exception, so they are asserted against the checkpoint that is
    actually loaded instead of trusted."""
    vae = getattr(pipe, "vae", None)
    audio_vae = getattr(pipe, "audio_vae", None)
    actual = {}
    if vae is not None:
        actual["spatial compression"] = (vae.spatial_compression_ratio, H3_SPATIAL_COMPRESSION)
        actual["VAE frames per chunk"] = (vae.config.clip_length, H3_FRAMES_PER_CHUNK)
        actual["VAE latents per chunk"] = (vae.tokens_chunk_size, H3_LATENTS_PER_CHUNK)
        actual["video latent channels"] = (vae.config.latent_channels, H3_VIDEO_LATENT_CHANNELS)
    if audio_vae is not None:
        actual["audio latent channels"] = (
            audio_vae.config.latent_channels,
            H3_AUDIO_LATENT_CHANNELS,
        )
        actual["audio sampling rate"] = (audio_vae.config.sampling_rate, 32000)
    mismatched = [f"{k}: checkpoint {a}, trainer {b}" for k, (a, b) in actual.items() if a != b]
    if mismatched:
        raise ValueError(
            "This MiniMax-H3 checkpoint does not match the grid the trainer builds its packed "
            "sequence on (" + "; ".join(mismatched) + ")."
        )


def _load_conditioners(cfg, device):
    """Build the modular pipeline and load ONLY the conditioning components.

    ``ModularPipeline.from_pretrained`` without ``workflow=`` keeps the whole block graph (a
    ``workflow=`` prunes it statically and cannot be undone), and ``load_components(names=...)``
    then fetches exactly the components named -- so the transformer is never even downloaded at
    this phase, let alone resident."""
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(cfg.base_model, token = cfg.hf_token)
    pipe.load_components(names = list(_H3_TEXT_COMPONENTS), torch_dtype = torch.bfloat16)
    pipe.load_components(names = list(_H3_VAE_COMPONENTS), torch_dtype = torch.float32)
    _assert_component_grid(pipe)
    # ``load_components`` builds every component on the CPU, so place them explicitly.
    pipe.text_encoder.to(device)
    pipe.vae.to(device)
    pipe.audio_vae.to(device)
    return pipe


def _encode_prompt(pipe, caption: str, device) -> Any:
    """One caption -> the ``(1, num_tokens, 5120)`` hidden state H3 conditions on.

    The presentation is the prompt verbatim: no chat template, no special tokens. The stack is
    called through ``text_encoder.model`` because H3 reads ``hidden_states[50]`` and never uses
    the language-model head, whose vocabulary-wide projection is all the top-level forward would
    add."""
    import torch

    token_ids = pipe.tokenizer(caption, add_special_tokens = False)["input_ids"]
    if not token_ids:
        raise ValueError(
            "A MiniMax-H3 caption cannot be empty: the packed sequence needs text rows."
        )
    input_ids = torch.tensor([token_ids], dtype = torch.long, device = device)
    mm_token_type_ids = torch.tensor(
        pipe.processor.create_mm_token_type_ids([token_ids]), dtype = torch.long, device = device
    )
    encoder = pipe.text_encoder
    num_layers = encoder.config.text_config.num_hidden_layers
    if num_layers <= _H3_TEXT_ENCODER_LAYER:
        raise ValueError(
            f"MiniMax-H3 conditions on hidden_states[{_H3_TEXT_ENCODER_LAYER}] of its Qwen3-VL "
            f"conditioner, which needs more than {_H3_TEXT_ENCODER_LAYER} decoder layers; this "
            f"one has {num_layers}."
        )
    with torch.no_grad():
        out = encoder.model(
            input_ids = input_ids,
            attention_mask = torch.ones_like(input_ids),
            mm_token_type_ids = mm_token_type_ids,
            use_cache = False,
            output_hidden_states = True,
        )
    return out.hidden_states[_H3_TEXT_ENCODER_LAYER].to("cpu", torch.float32)


def _encode_video_stats(vae, frames, device) -> tuple[Any, Any]:
    """Encode one clip's frames to the normalised posterior's affine ``(A, B)`` pair.

    ``frames`` is the uint8 ``(F, H, W, 3)`` array ``decode_clip`` returns. The recipe is the
    released model's: ImageNet-normalise the pixels, encode, and normalise the latent by the
    VAE's ``latents_mean`` / ``latents_std``. Caching ``(A, B)`` rather than a sample keeps a
    fresh posterior draw available at every step without the VAE resident, exactly as the DiT
    trainer's latent cache does. The reference's float16 rounding of the SAMPLE is not applied:
    it is a conditioning-anchor reproducibility device (it makes a keyframe encode
    bit-identical), and rounding a training target only removes signal."""
    import torch

    pixels = torch.from_numpy(frames).to(device)
    # (F, H, W, 3) uint8 -> (1, 3, F, H, W) float32
    pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).to(torch.float32).div_(255.0)
    pixel_mean = torch.tensor(H3_PIXEL_MEAN, device = device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(H3_PIXEL_STD, device = device).view(1, -1, 1, 1, 1)
    pixels = (pixels - pixel_mean) / pixel_std
    with torch.no_grad():
        posterior = vae.encode(pixels, return_dict = False)[0]
    latents_mean = torch.tensor(vae.config.latents_mean, device = device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device = device).view(1, -1, 1, 1, 1)
    return (
        ((posterior.mean - latents_mean) / latents_std).float().cpu(),
        (posterior.std / latents_std).float().cpu(),
    )


def _encode_audio_latents(audio_vae, waveform, device) -> Any:
    """Encode one clip's stereo soundtrack to normalised audio latents ``(2, 32, n)``.

    The audio VAE is mono, so the two stereo channels go through as two batch items -- the same
    boundary the decoder crosses in reverse. MiniMax-H3 consumes the posterior MEAN and never
    evaluates the ``logs_proj`` head, so the audio target is deterministic and no affine pair is
    cached for it."""
    import torch

    samples = torch.from_numpy(waveform).to(device).unsqueeze(1)  # (2, 1, samples)
    with torch.no_grad():
        posterior = audio_vae.encode(samples, return_dict = False)[0]
    latents = posterior.mode()
    mean = torch.tensor(audio_vae.config.latents_mean, device = device).view(1, -1, 1)
    std = torch.tensor(audio_vae.config.latents_std, device = device).view(1, -1, 1)
    return ((latents - mean) / std).float().cpu()


def _load_transformer(cfg, device, base_precision):
    """Load the denoiser alone, in the resolved precision. See ``_H3_NF4_SKIP_MODULES``."""
    import torch
    from diffusers import MiniMaxH3Transformer3DModel

    if base_precision == "nf4":
        from diffusers import BitsAndBytesConfig as DiffusersBnb

        quant = DiffusersBnb(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
            llm_int8_skip_modules = list(_H3_NF4_SKIP_MODULES),
        )
        return MiniMaxH3Transformer3DModel.from_pretrained(
            cfg.base_model,
            subfolder = "transformer",
            quantization_config = quant,
            torch_dtype = torch.bfloat16,
            token = cfg.hf_token,
        )
    return MiniMaxH3Transformer3DModel.from_pretrained(
        cfg.base_model,
        subfolder = "transformer",
        torch_dtype = torch.bfloat16,
        token = cfg.hf_token,
    ).to(device)


def _patchify(latents, patch: tuple[int, int, int]):
    """``(1, C, F, H, W)`` video latents -> the transformer's rows, frame-major then row-major."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    return patchify_video_latents(latents, patch)


def _build_layout(num_text_tokens: int, latent_frames: int, latent_h: int, latent_w: int,
                  num_audio_latents: int, patch: tuple[int, int, int], device):
    """The packed ``[text | audio | video]`` layout for one training sample.

    Built by the pipeline's own ``@staticmethod`` so the trainer and the sampler cannot drift:
    the rotary grid, the row order and the modality tags are a checkpoint contract, and a
    reimplementation that agreed today would not stay agreeing."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    text_token_tags = torch.full((num_text_tokens,), H3_TEXT_TAG, dtype = torch.long)
    (position_ids, token_tags, video_indices, audio_indices, text_indices,
     n_cond_video, n_cond_audio) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags,
        latent_frames,
        latent_h,
        latent_w,
        num_audio_latents,
        patch,
        H3_AUDIO_CHANNELS,
        H3_AUDIO_TAG,
        H3_VIDEO_TAG,
        (),  # the trainer trains the t2va layout: no keyframe conditioning rows
    )
    return {
        "position_ids": position_ids.to(device),
        "token_tags": token_tags.to(device),
        "video_indices": video_indices.to(device),
        "audio_indices": audio_indices.to(device),
        "text_indices": text_indices.to(device),
        "num_condition_video_rows": n_cond_video,
        "num_condition_audio_rows": n_cond_audio,
    }


def _row_timesteps(layout, num_text_tokens: int, t_video: float, t_audio: float, device):
    """The transformer's ``(timestep, timestep_indices)`` pair for one step.

    Also the pipeline's own ``@staticmethod``: one forward serves rows at different noise
    levels, and which row is at which level is the layout's business, not the loop's."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        layout["video_indices"],
        layout["audio_indices"],
        layout["num_condition_video_rows"],
        layout["num_condition_audio_rows"],
        num_text_tokens,
        t_video,
        t_audio,
        # No conditioning rows exist in this layout, so the two condition timesteps are
        # unreachable; pass the generated ones so no phantom value enters ``torch.unique``.
        t_video,
        t_audio,
    )
    return timestep.to(device), timestep_indices.to(device)


def _save_lora(out_dir: str, layers: dict) -> None:
    """Write the adapter as ``pytorch_lora_weights.safetensors``.

    Diffusers ships no ``MiniMaxH3LoraLoaderMixin``, so there is no
    ``MiniMaxH3Pipeline.save_lora_weights`` to route through and no
    ``pipe.load_lora_weights`` to read it back. The file this writes is nonetheless the
    ordinary diffusers single-file layout with the ``transformer.`` prefix every mixin uses, so
    it loads with ``transformer.load_lora_adapter(path, prefix="transformer")`` today and will
    load with ``load_lora_weights`` unchanged the day that mixin lands."""
    from safetensors.torch import save_file

    Path(out_dir).mkdir(parents = True, exist_ok = True)
    state = {f"transformer.{k}": v.to("cpu").contiguous() for k, v in layers.items()}
    save_file(state, str(Path(out_dir) / DEFAULT_LORA_FILENAME))


def run_h3_lora_training(
    config: DiffusionLoraConfig,
    *,
    on_event: Optional[EventCb] = None,
    should_stop: Optional[StopCb] = None,
) -> str:
    """Train a MiniMax-H3 joint video + audio LoRA from a clip dataset and export it."""
    cfg = config.normalized()
    if cfg.resolved_family != "minimax-h3":
        raise ValueError(f"This trainer is for minimax-h3, not {cfg.resolved_family!r}.")
    if cfg.mixed_precision == "fp16":
        raise ValueError(
            "MiniMax-H3 LoRA training requires bf16: its checkpoint keeps the patch "
            "projections, the timestep MLP and the output heads in fp32 and fp16 overflows "
            "them. Set mixed precision to bf16."
        )
    if cfg.resolution % H3_CANVAS_MULTIPLE:
        raise ValueError(
            f"MiniMax-H3 trains on a canvas whose edges are multiples of {H3_CANVAS_MULTIPLE} "
            f"(a 16x VAE compression and a 2x patch); got resolution {cfg.resolution}."
        )
    if float(getattr(cfg, "cfg_dropout", 0.0) or 0.0) > 0:
        raise ValueError(
            "MiniMax-H3 is guidance-distilled: it has no unconditional branch and no negative "
            "prompt, so a classifier-free-guidance dropout trains a path the sampler never "
            "takes. Set cfg_dropout to 0."
        )
    if str(getattr(cfg, "weighting_scheme", "none") or "none") != "none":
        raise ValueError(
            "MiniMax-H3 has no timestep-weighted loss yet: its two schedules put video and "
            "audio at different sigmas in the same step, so a single weight over 'the' "
            "timestep is ambiguous. Use weighting_scheme='none'."
        )
    # The batch axis of an H3 forward is a pure replication axis: the layout, the rotary grid
    # and the row timesteps describe ONE packed sequence that every batch item shares. Two
    # clips with different captions have different text lengths and therefore different
    # layouts, so a batch > 1 cannot be formed without padding the model has no mask for.
    if cfg.train_batch_size != 1:
        raise ValueError(
            "MiniMax-H3 trains at batch size 1: one forward covers one packed sequence, whose "
            "row layout is set by the clip's own geometry and its caption's length. Use "
            "gradient_accumulation_steps to raise the effective batch."
        )

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not native_bf16_supported():
        raise ValueError(
            "This trainer requires a bfloat16-capable GPU (Ampere or newer); "
            "this CUDA device does not support bf16."
        )
    weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    _assert_trusted_base_model(cfg.base_model)
    pairs = discover_clip_caption_pairs(
        cfg.data_dir, instance_prompt = cfg.instance_prompt, caption_column = cfg.caption_column
    )
    cfg = replace(cfg, train_steps = resolve_train_steps(cfg, len(pairs)), num_epochs = 0)
    _emit(on_event, "model_load_started", num_images = len(pairs))
    if _check_stop():
        out_dir = Path(cfg.output_dir).expanduser()
        _emit(on_event, "complete", output_dir = str(out_dir), lora_path = None,
              stopped = True, steps_run = 0)
        return str(out_dir)

    perf_snap = _apply_perf_flags(cfg, device)
    try:
        return _train_h3(cfg, pairs, rng, device, weight_dtype, on_event, _check_stop,
                         lambda: save_on_stop)
    finally:
        _restore_perf_flags(perf_snap)


def _train_h3(cfg, pairs, rng, device, weight_dtype, on_event, _check_stop, _save_on_stop):
    import torch
    import torch.nn.functional as F
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from core.training.diffusion_dit_trainer import _int8_quantize_base, _make_optimizer

    out_dir = Path(cfg.output_dir).expanduser()
    clip_paths = [p for p, _ in pairs]
    captions = [c for _, c in pairs]
    num_frames = H3_TRAIN_NUM_FRAMES
    latent_frames = h3_video_latent_frames(num_frames)
    num_audio_latents = h3_audio_latent_count(num_frames)

    to_encode = sorted(set(captions))

    # ── phase 1: conditioning. The 63 GiB Qwen3-VL conditioner and both VAEs are resident
    # here and nowhere else; the 66 GB transformer has not been fetched yet.
    pipe = _load_conditioners(cfg, device)
    caption_embeds = {cap: _encode_prompt(pipe, cap, device) for cap in to_encode}
    _emit(on_event, "preparing", stage = "encode_prompts", done = len(to_encode),
          total = len(to_encode))
    pipe.text_encoder = None
    if getattr(pipe, "processor", None) is not None:
        pipe.processor = None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── phase 2: the clip cache. One canvas for the run, taken from the FIRST clip's aspect
    # ratio: every other clip is cover-cropped onto it, so a mixed-aspect dataset trains on one
    # geometry rather than reshaping the packed sequence per step.
    width, height = _dataset_canvas(clip_paths[0], cfg.resolution)
    latent_h, latent_w = height // H3_SPATIAL_COMPRESSION, width // H3_SPATIAL_COMPRESSION
    cache: list[tuple[Any, Any, Any]] = []
    for i, path in enumerate(clip_paths):
        frames, waveform = decode_clip(
            path, num_frames = num_frames, width = width, height = height
        )
        video_a, video_b = _encode_video_stats(pipe.vae, frames, device)
        audio = _encode_audio_latents(pipe.audio_vae, waveform, device)
        if audio.shape[-1] != num_audio_latents:
            raise ValueError(
                f"{Path(path).name} encoded to {audio.shape[-1]} audio latents, but the packed "
                f"layout reserves {num_audio_latents} rows per channel."
            )
        cache.append((video_a, video_b, audio))
        _emit(on_event, "preparing", stage = "cache_latents", done = i + 1, total = len(clip_paths))
        if _check_stop():
            _emit(on_event, "complete", output_dir = str(out_dir), lora_path = None,
                  stopped = True, steps_run = 0)
            return str(out_dir)
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── phase 3: the denoiser.
    base_precision = cfg.base_precision if cfg.base_precision != "auto" else "nf4"
    if base_precision in ("fp8", "mxfp8"):
        raise ValueError(
            "MiniMax-H3 has no fp8 training path: its packed sequence mixes three modalities "
            "through one set of linears, so the activation range is not the per-family range "
            "the fp8 filter was measured against. Use nf4, int8 or bf16."
        )
    transformer = _load_transformer(cfg, device, base_precision)
    transformer.requires_grad_(False)
    # ``normalized()`` fills lora_target_modules with the generic DEFAULT_LORA_TARGETS when a
    # caller does not set it, so that value means "unset" and the family's own regex wins. Any
    # other explicit tuple is a deliberate override and is passed through as a list.
    targets: Any = (
        _H3_TARGETS
        if tuple(cfg.lora_target_modules) == DEFAULT_LORA_TARGETS
        else list(cfg.lora_target_modules)
    )
    transformer.add_adapter(
        LoraConfig(
            r = cfg.lora_rank,
            lora_alpha = cfg.lora_alpha,
            lora_dropout = cfg.lora_dropout,
            init_lora_weights = "gaussian",
            target_modules = targets,
        )
    )
    if cfg.gradient_checkpointing:
        import functools
        import torch.utils.checkpoint as _ckpt

        transformer.enable_gradient_checkpointing(
            gradient_checkpointing_func = functools.partial(_ckpt.checkpoint, use_reentrant = False)
        )
    cast_training_params(transformer, dtype = torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if not lora_params:
        raise ValueError(
            f"No LoRA target matched the MiniMax-H3 transformer for {targets!r}; the adapter "
            f"would train nothing."
        )
    if base_precision == "int8":
        _int8_quantize_base(transformer)

    ema = LoRAEMA(transformer, decay = cfg.ema_decay) if getattr(cfg, "ema_decay", 0.0) else None
    optimizer = _make_optimizer(lora_params, cfg.learning_rate)
    lr_sched = get_scheduler(
        cfg.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = cfg.lr_warmup_steps,
        num_training_steps = cfg.train_steps,
    )
    video_shift = (
        float(cfg.flow_shift)
        if isinstance(cfg.flow_shift, (int, float))
        else _H3_VIDEO_SHIFT
    )
    _emit(on_event, "model_load_completed", compiled = False, base_precision = base_precision,
          sequence_length = h3_packed_sequence_length(
              max(e.shape[1] for e in caption_embeds.values()), num_frames, height, width
          ))

    transformer.train()
    patch = tuple(transformer.config.patch_size)
    index_sampler = PermutationBatchSampler(len(clip_paths), rng)
    autocast = (
        torch.autocast(device_type = "cuda", dtype = torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    stopped = False
    running_loss = 0.0
    peak_gb = 0.0
    t_start = time.time()
    t_steady = None
    done = 0
    for opt_step in range(cfg.train_steps):
        optimizer.zero_grad(set_to_none = True)
        step_loss = 0.0
        step_video = 0.0
        step_audio = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            index = index_sampler.next_batch(1)[0]
            video_a, video_b, audio_latents = cache[index]
            video_a = video_a.to(device)
            video_b = video_b.to(device)
            # A fresh posterior draw per step, exactly like encoding in the loop would give.
            clean_video = video_a + video_b * torch.randn_like(video_a)
            clean_audio = audio_latents.to(device)

            # ONE base u through BOTH shifts: the (video, audio) sigma pair the sampler visits.
            u = rng.random()
            sigma_video = _shifted_sigma(u, video_shift)
            sigma_audio = _shifted_sigma(u, _H3_AUDIO_SHIFT)

            noise_video = torch.randn_like(clean_video)
            noise_audio = torch.randn_like(clean_audio)
            noisy_video = (1.0 - sigma_video) * clean_video + sigma_video * noise_video
            noisy_audio = (1.0 - sigma_audio) * clean_audio + sigma_audio * noise_audio

            video_rows = _patchify(noisy_video, patch)
            audio_rows = noisy_audio.permute(0, 2, 1).reshape(-1, H3_AUDIO_LATENT_CHANNELS)
            # DATA-ward velocity: x0 = x_t + sigma * v, so the target is latents - noise.
            target_video = _patchify(clean_video - noise_video, patch)
            target_audio = (clean_audio - noise_audio).permute(0, 2, 1).reshape(
                -1, H3_AUDIO_LATENT_CHANNELS
            )

            embeds = caption_embeds[captions[index]].to(device, weight_dtype)
            num_text_tokens = embeds.shape[1]
            layout = _build_layout(num_text_tokens, latent_frames, latent_h, latent_w,
                                   num_audio_latents, patch, device)
            timestep, timestep_indices = _row_timesteps(
                layout, num_text_tokens, 1.0 - sigma_video, 1.0 - sigma_audio, device
            )
            with autocast:
                pred_video, pred_audio = transformer(
                    hidden_states = video_rows[None],
                    audio_hidden_states = audio_rows[None],
                    encoder_hidden_states = embeds,
                    timestep = timestep,
                    timestep_indices = timestep_indices,
                    token_tags = layout["token_tags"],
                    position_ids = layout["position_ids"],
                    video_indices = layout["video_indices"],
                    audio_indices = layout["audio_indices"],
                    text_indices = layout["text_indices"],
                    return_dict = False,
                )
                loss_video = F.mse_loss(pred_video[0].float(), target_video.float())
                loss_audio = F.mse_loss(pred_audio[0].float(), target_audio.float())
                # Unweighted sum, the model's own objective: the two streams share every
                # matrix the adapter touches, so down-weighting audio would not protect it, it
                # would only stop the loss reporting that it had drifted.
                loss = loss_video + loss_audio
            (loss / cfg.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) / cfg.gradient_accumulation_steps
            step_video += float(loss_video.detach()) / cfg.gradient_accumulation_steps
            step_audio += float(loss_audio.detach()) / cfg.gradient_accumulation_steps

        grad_norm = None
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm))
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
            if device == "cuda":
                peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            per_step = cfg.gradient_accumulation_steps
            if t_steady is not None and done > 1:
                sps = round((done - 1) * per_step / max(now - t_steady, 1e-6), 3)
            else:
                sps = round(done * per_step / max(now - t_start, 1e-6), 3)
            _emit(
                on_event,
                "progress",
                step = done,
                total_steps = cfg.train_steps,
                loss = round(step_loss, 5),
                avg_loss = round(running_loss / done, 5),
                video_loss = round(step_video, 5),
                audio_loss = round(step_audio, 5),
                learning_rate = lr_sched.get_last_lr()[0],
                grad_norm = round(grad_norm, 5) if grad_norm is not None else None,
                samples_per_second = sps,
                peak_memory_gb = peak_gb or None,
            )
        if _check_stop():
            stopped = True
            break

    lora_path: Optional[str] = None
    ema_path: Optional[str] = None
    if not (stopped and not _save_on_stop()):
        layers = get_peft_model_state_dict(transformer)
        _save_lora(str(out_dir), layers)
        lora_path = str(out_dir / DEFAULT_LORA_FILENAME)
        if ema is not None and ema.updates > 0:
            try:
                ema_dir = save_ema_adapter(ema, transformer, lambda _pipe, d, l: _save_lora(d, l),
                                           str(out_dir))
                ema_path = str(Path(ema_dir) / DEFAULT_LORA_FILENAME)
            except Exception as exc:  # noqa: BLE001 -- the primary adapter is already saved
                _emit(on_event, "warning", message = f"EMA adapter save failed: {exc}")
    _emit(
        on_event,
        "complete",
        output_dir = str(out_dir),
        lora_path = lora_path,
        ema_path = ema_path,
        catalog_path = None,
        family = cfg.resolved_family,
        base_model = cfg.base_model,
        stopped = stopped,
        steps_run = done if cfg.train_steps else 0,
        wall_seconds = round(time.time() - t_start, 1),
    )
    return str(out_dir)


def _dataset_canvas(clip_path: str, short_edge: int) -> tuple[int, int]:
    """The one canvas the run trains on, from the first clip's stored aspect ratio."""
    import av

    with av.open(str(clip_path)) as container:
        stream = container.streams.video[0]
        source_w = int(stream.codec_context.width)
        source_h = int(stream.codec_context.height)
    return h3_train_canvas(source_w, source_h, short_edge = short_edge)
