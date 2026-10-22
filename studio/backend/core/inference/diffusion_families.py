# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure helpers for diffusion model identification. No torch/diffusers imports: everything here is a
pure function of its string/path arguments, so it can be unit-tested without the heavy runtime. A
diffusion checkpoint published as a single-file GGUF only carries the transformer weights; the
matching VAE / text encoders / scheduler come from a companion ``diffusers`` base repo, and
``DiffusionFamily`` maps a checkpoint to the diffusers classes and base repo needed to assemble
the full pipeline."""

from __future__ import annotations

import importlib.util
import json
import contextvars
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence
from utils.paths.path_utils import is_appledouble_metadata

from .diffusion_flow_shift import flux_mu_shift
from .family_name_match import (
    model_identity,
    name_key_in,
    normalize_family_name,
    token_in_name,
    token_length,
)
from .diffusion_nvfp4_flag import nvfp4_blocked
from .family_profile import AutoPrefer, ConvRotSpec, FamilyProfile


# Runtime->route contract: the /images/generate route matches these messages EXACTLY for a 409 (vs a 500), so both
# engines raise them verbatim.
DIFFUSION_NOT_LOADED_MSG = "No diffusion model is loaded."
DIFFUSION_CANCELLED_MSG = "Diffusion generation was cancelled."


@dataclass(frozen = True)
class LoadIdentity:
    """What a caller's derived request parameters depend on, as one comparable value.

    ``repo_id`` alone is not one: /images/load takes ``base_repo`` and ``family_override``
    independently of the path, so a local checkpoint reloads as a different model while the path
    stays put, and the images route derives steps/guidance from ``base_repo`` and its edit-only
    verdict from the family (#9448). Loads agreeing on all three derive identical parameters, which
    is exactly when accepting one for the other is correct.

    A type rather than a tuple, so pinning a bare repo id compares unequal and is refused instead of
    matching some other shape by accident.
    """

    repo_id: str
    base_repo: str
    family: str


def load_identity(repo_id, base_repo, family) -> LoadIdentity:
    """``LoadIdentity`` for one load. None and "" describe the same absent field."""
    return LoadIdentity(str(repo_id or ""), str(base_repo or ""), str(family or ""))


class DiffusionModelReplacedError(RuntimeError):
    """Both engines' ``generate`` refusing a ``LoadIdentity`` that is no longer loaded. Keeps a
    caller's per-model steps/guidance and workflow verdict, taken from an earlier ``status()``
    read, off a model they never validated (#9448). Here rather than in ``diffusion`` so the
    native engine can raise it without importing the torch backend."""

    def __init__(self, expected: LoadIdentity, actual: LoadIdentity):
        super().__init__(
            f"The image model was replaced while this request waited "
            f"(expected {expected.repo_id!r}, loaded {actual.repo_id!r}); "
            "retry with fresh parameters."
        )
        self.expected = expected
        self.actual = actual


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Pipeline kwarg carrying guidance. Most use "guidance_scale"; Qwen-Image real CFG is "true_cfg_scale".
    cfg_kwarg: str = "guidance_scale"
    # False when the diffusers pipeline ignores a negative prompt; the native engine decides on its own.
    uses_negative_prompt: bool = True
    # False when the diffusers pipeline takes no guidance (FLUX.3 distilled); FLUX.2 klein is per checkpoint.
    uses_guidance: bool = True
    # One or two sentences on how this family reads prompts; None when it has no quirks.
    prompt_hint: Optional[str] = None
    # The pipe attribute holding the denoiser: ``pipe.transformer`` for DiT families, ``pipe.unet`` for SDXL.
    denoiser_attr: str = "transformer"
    # True when a single-file ``.safetensors`` is the WHOLE pipeline (SDXL), so the loader calls ``from_single_file``.
    single_file_is_pipeline: bool = False
    # True for families needing MULTIPLE denoisers no single file carries (Ideogram 4), so only a full pipeline load
    # is valid.
    pipeline_only: bool = False
    # Transformer-only files (GGUF) load only via sd.cpp; diffusers has no single-file loader (MOSS-Image).
    native_transformer_files_only: bool = False
    # Optional diffusers pipeline classes for image-conditioned workflows, built via ``Pipeline.from_pipe`` (no
    # reload). None = unsupported.
    img2img_pipeline_class: Optional[str] = None
    inpaint_pipeline_class: Optional[str] = None
    # ControlNet pipeline + model classes: the model loads via from_pretrained, the pipeline via ``from_pipe``. None
    # on both = no support.
    controlnet_pipeline_class: Optional[str] = None
    controlnet_model_class: Optional[str] = None
    # True when the inpaint pipeline keeps the canvas size, so it can also drive outpaint. False for FLUX.2 (it scales
    # >1MP inputs to ~1MP).
    inpaint_preserves_size: bool = True
    # True for instruction-editing families (Qwen-Image-Edit / FLUX Kontext): the pipeline IS the edit pipeline (image
    # + instruction, no plain text-to-image). ``base_repo`` supplies the VAE / text-encoder / processor / scheduler.
    edit: bool = False
    # True for families whose text-to-image pipeline ALSO accepts reference image(s) (FLUX.2 ``image``): no
    # ``strength``, size from width/height.
    reference: bool = False
    # True when the text-to-image pipeline also follows edit instructions over ``image`` (Qwen-Image-2.1), at the
    # requested size. Never inferred from ``reference``: a reference family is not necessarily trained to edit.
    unified_edit: bool = False
    # Qwen-Image-Layered: RGBA layers per input (each an output image); 0 = not layered.
    layer_count: int = 0
    # The pipeline's ``resolution`` area bucket (640 or 1024).
    layer_resolution: int = 640
    # Condition images per call, INCLUDING the init image; overflow is refused, never sliced.
    max_condition_images: int = 4
    condition_image_mode: str = "RGB"
    dimension_multiple: int = 16
    max_output_side: int = 2048
    max_output_pixels: int = 2048 * 2048
    # Accepted condition-image preprocessing resolutions (square side, by area); empty = no such control.
    reference_resolutions: tuple[int, ...] = field(default_factory = tuple)
    # ComfyUI's static sigma shift; None = keep the shipped scheduler.
    comfy_flow_shift: Optional[float] = None
    # (lowercased id substring, shift or None = shipped) for checkpoints whose template differs; first match wins.
    comfy_flow_shift_variants: tuple[tuple[str, Optional[float]], ...] = field(
        default_factory = tuple
    )
    # ComfyUI shift node (diffusion_flow_shift SHIFT_NODE_*); None = no Shift control. ``_variants``: (id substring, node), first match wins.
    comfy_shift_node: Optional[str] = None
    comfy_shift_node_variants: tuple[tuple[str, Optional[str]], ...] = field(default_factory = tuple)
    # Textual-inversion format (diffusion_embeddings KIND_*); None = no embeddings.
    textual_inversion: Optional[str] = None
    # (lowercased id substring, ((key, value), ...)) overriding ``base_repo``'s transformer config; first match wins.
    transformer_config_variants: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = field(
        default_factory = tuple
    )
    # (lowercased id substring, max_condition_images) for a family whose checkpoints take different counts; first
    # match wins, none keeps ``max_condition_images``.
    max_condition_images_variants: tuple[tuple[str, int], ...] = field(default_factory = tuple)
    # Same config, different weights: a GGUF of one must never get the base's transformer of another.
    checkpoint_variants: tuple[str, ...] = field(default_factory = tuple)
    # Activation-guard cost of one condition pixel relative to one output pixel.
    condition_pixel_weight: float = 1.0
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True for families whose activations overflow float16 (-> black image); the backend promotes a resolved float16
    # to float32.
    fp16_incompatible: bool = False
    # diffusion_fp16_guard recipe keeping an fp16_incompatible family in float16 on fp16-only cards; None = promote.
    fp16_guard: Optional[str] = None
    # false only for a family whose denoiser block does not compile cleanly with regional torch.compile
    supports_torch_compile: bool = True
    # False keeps cudnn.benchmark off (as VideoFamily.cudnn_benchmark): its per-process conv pick changes pixels.
    cudnn_benchmark: bool = True
    # Pin inductor's reduction-config filter on every arch (``filter_reduction_configs``) or only on the listed (major,
    # minor) archs (``filter_reduction_configs_archs``); both unset = inductor's pick. Same two fields on VideoFamily.
    filter_reduction_configs: bool = False
    filter_reduction_configs_archs: tuple[tuple[int, int], ...] = field(default_factory = tuple)
    # Optional pre-quantized transformer checkpoints as (scheme, repo_id): fetched instead of the dense bf16 (lower
    # load VRAM + download).
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Hosted checkpoints for NON-DEFAULT bases as (base_repo, scheme, repo_id), base_repo lowercased: one family entry
    # covers variants whose weights differ. Resolution prefers an exact variant match, then falls back to
    # ``prequant_repos``.
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Lowercased bases with different weights: never inherit ``prequant_repos``; their variant rows still win.
    prequant_excluded_bases: tuple[str, ...] = field(default_factory = tuple)
    # Variant bases a repo id or GGUF name can select when no card ``base_model`` tag resolves one.
    named_variant_bases: tuple[str, ...] = field(default_factory = tuple)
    # Preferred checkpoint FILENAME for a scheme, as (scheme, filename), overriding the ``<Model>-<SCHEME>.pt`` name
    # ``prequant_repo_filename`` derives. The derived name stays on as the fallback, so a repo hosting BOTH an old and
    # a new artifact serves the new one to a build that asks for it by name and the old one to every build that does
    # not. That is what lets a rotated (v2) checkpoint ship without regressing an already-installed Unsloth, which
    # would otherwise refuse the v2 tag and fall all the way back to the dense download. A row may also be (scheme,
    # task, filename), which names the artifact for ONE task and beats the task-agnostic row; see
    # ``family_prequant_filename``.
    prequant_filenames: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
    # Hosted PRE-CAST text-encoder checkpoints as (scheme, component, repo_id). Layerwise-fp8 only: the cast is
    # deterministic, so the artifact is bit-identical while skipping the dense TE download. An optional fourth element
    # names the component the hosted file was built as, when this family loads another family's bytes into a
    # different slot (HiDream's text_encoder_3 is FLUX.1's text_encoder_2).
    te_prequant_repos: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
    # The same rows for a VARIANT base whose encoder is a different model, as (base_repo, scheme, component, repo_id):
    # FLUX.2 klein 9B carries Qwen3-8B where the family base carries Qwen3-4B. A load of that base (or one in its
    # equivalence group) resolves these rows only; ``te_prequant_repos`` keeps serving the family base.
    te_prequant_variant_repos: tuple[tuple[str, str, str, str], ...] = field(default_factory = tuple)
    # The text-encoder scheme an UNSET request resolves to on this family, or None to keep the released bf16 encoder.
    # Opt-in per family rather than implied by ``te_prequant_repos``, because hosting an artifact says the cast is
    # reproducible, not that its conditioning is good enough to hand someone who asked for nothing. These encoders are
    # different models (T5, CLIP, Qwen3-VL, Gemma) and fp8 tolerance does not transfer between them, so a family joins
    # this list only once the fp8-vs-bf16 delta has been measured on it.
    te_quant_auto: Optional[str] = None
    # Native (sd.cpp) single-file assets, used only on the no-GPU sd.cpp engine. The transformer GGUF is shared with
    # diffusers; sd-cli also needs a (repo_id, filename) VAE + text encoder(s), each with a trailing SdCppModelFiles
    # field name.
    sd_cpp_vae: Optional[tuple[str, str]] = None
    # VAE latent-format override for sd-cli (--vae-format): "flux2" for FLUX.2, None otherwise
    sd_cpp_vae_format: Optional[str] = None
    sd_cpp_text_encoders: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # (repo_id, filename) of a tokenizer.json for sd-cli --tokenizer (MOSS-Image requires it).
    sd_cpp_tokenizer: Optional[tuple[str, str]] = None
    # mflux ModelConfig name (None = no MLX); ``_by_base`` maps a lowercase base repo to another config.
    mlx_model_config: Optional[str] = None
    mlx_model_config_by_base: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Family-specific sd-cli sampler settings so native output matches the model's supported invocation. None leaves
    # sd-cli defaults.
    sd_cpp_sampling_method: Optional[str] = None
    sd_cpp_flow_shift: Optional[float] = None
    # A literal this family's ARCHITECTURE puts in an sd.cpp build that can run it. Set it for a
    # family whose support landed upstream after builds were already in the wild: the asset mapping
    # above says what to hand sd-cli, not whether the sd-cli on this disk understands the model, and
    # a binary installed before the support existed is reused as-is (nothing upgrades a runnable
    # build of the right accelerator). Without a gate that is a load that reports ready and dies on
    # the first generation, where falling back to diffusers is both possible and correct.
    #
    # Matched against the binary's BYTES rather than the install record's tag: every mirror release
    # carries the upstream base in its tag name whatever tree was built, so the tag cannot answer
    # this, and a custom SD_CLI_PATH build has no record at all.
    sd_cpp_arch_marker: Optional[str] = None
    # Literal an sd.cpp build carries once it can EDIT this family; the arch marker proves text-to-image only.
    sd_cpp_edit_marker: Optional[str] = None
    # sd.cpp feeds this family's reference images through the VAE (no vision projector), so any build the arch marker
    # admits runs the reference workflow natively.
    sd_cpp_reference: bool = False
    # True when Unsloth can TRAIN a LoRA on this family; the training-start path refuses a non-trainable family up
    # front.
    trainable: bool = False
    # Recommended base repos to train FROM, most-preferred first (e.g. a QLoRA prequant repo, then bf16). Surfaced by
    # the Train UI.
    train_base_repos: tuple[str, ...] = field(default_factory = tuple)
    # When set, deploying a LoRA trained on this family loads THIS repo instead (Krea: train on Raw, preview on
    # Turbo). Same precision both sides.
    deploy_base_repo: Optional[str] = None
    # Variant-specific training-base to inference-base mappings. FLUX.2 Klein trains on an undistilled base and runs
    # the adapter on the matching 4-step checkpoint, so its 4B and 9B variants cannot share the single family-wide
    # deploy_base_repo above.
    deploy_base_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Display label (Train UI); None = the family name.
    label: Optional[str] = None
    # Per-family policy data (quant excludes / ConvRot / denies, TE int8 schedule, preview, step skip, trusted
    # repos); see family_profile.FamilyProfile.
    profile: FamilyProfile = field(default_factory = FamilyProfile)

    def deploy_base_for(self, trained_base: str) -> str:
        """The inference checkpoint paired with ``trained_base``, or the input unchanged."""
        key = canonical_base(trained_base).lower()
        for training_repo, inference_repo in self.deploy_base_repos:
            if canonical_base(training_repo).lower() == key:
                return inference_repo
        return self.deploy_base_repo or trained_base


# Fresh servers rendered 2-4 variants of one seed here (FLUX.1, Z-Image, Qwen-Image; the edit siblings share their DiT):
# near-tied norm-reduction configs, benchmarked per process, sum in different orders. B200: FLUX.1 measured deterministic.
_REDUCTION_RACE_ARCHS: tuple[tuple[int, int], ...] = ((8, 0), (8, 9), (12, 0))
# Qwen-Image races on B200 too (torch 2.14): fresh processes rendered one seed 26 dB apart; with the filter 6/6 matched bitwise.
_QWEN_REDUCTION_RACE_ARCHS: tuple[tuple[int, int], ...] = _REDUCTION_RACE_ARCHS + ((10, 0),)

# --- Shared FamilyProfile data (see family_profile.FamilyProfile) ---

# int8 PER-FAMILY exclusions, on top of diffusion_transformer_quant._INT8_EXCLUDE_NAME_TOKENS. Qwen-Image MMDiT runs
# every TEXT-stream Linear at M = actual prompt tokens (unpadded, unlike FLUX's 512-token T5), so a short prompt falls
# under _int_mm's M floor of 16 and the denoise crashes. bf16 there costs ~nothing. Baked into the published int8
# artifacts of qwen-image / qwen-image-edit, so it moves only with a rebuild and republish.
_QWENIMAGE_INT8_EXCLUDES = (
    "txt_in",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "txt_mlp",
)
# Families whose activation ranges break specific schemes at the MODEL level (the smoke probe only proves the GEMM
# runs). Measured with the 28-pair prequant accuracy gate on B200: qwen-image + mxfp8 does semantic damage at 1024px,
# + nvfp4 is unusable (LPIPS 0.51). Neither has a known fix, so both stay denied and auto falls through to int8
# (excellent on Qwen). The deny also applies to an EXPLICIT request.
# fp8 was denied here too, for black frames. That cause is gone: it was torchao's fp8 scale chooser having no eps
# clamp, so qwen's all-zero text rows gave scale 0 and NaN, and ``_make_quant_config`` now floors it with
# ``activation_value_lb``. Re-measured on B200 with the floor, and BOTH paths the deny governs clear the 0.10 LPIPS
# bar: prequant checkpoint, full 28-pair gate -> 28/28 PASS (SSIM 0.87-0.99, LPIPS 0.027-0.228, CLIP delta <= 0.019);
# on-the-fly quantize_, gate's simple_object prompt at its own seeds -> LPIPS 0.048 / 0.033 / 0.027 / 0.063, against
# pre-floor plain fp8 at LPIPS 0.712 with SSIM 0.016 and mean luma 0. Keeping the deny cost real speed: it pinned qwen
# to int8 at 1.10x bf16 when compiled fp8 measures 1.21x.
_QWENIMAGE_SCHEME_DENY = frozenset({"mxfp8", "nvfp4"})
# FLUX.3 Klein transformer-quant keep-list, every scheme (FamilyProfile.quant_exclude_tokens).
_FLUX3_QUANT_KEEP = (
    "modulation",
    "timestep_embed",
    "norm_out",
    "txt_mode_blocks",
    "context_embedder",
)
# Schemes denied for TRAINING on top of the inference deny. Training holds a stricter bar because the evidence above is
# rendering evidence: it says a frozen fp8 forward reconstructs the bf16 image, not that a LoRA converges when its
# frozen linears are fp8. Nobody has run that, so qwen fp8 stays out of the Train UI until someone does. Delete the
# entry once a training run is measured.
_QWENIMAGE_TRAIN_SCHEME_DENY = frozenset({"fp8"})
# Text-encoder int8 keep-bf16 schedule (skip_first, skip_last) from measured hidden-state fidelity: qwen-image
# (Qwen2.5-VL-7B) first+last 6 gives ~0.997 cosine; flux.2-dev (Mistral-Small-24B) first 3 gives ~0.98 (early-layer
# seeding). A family without one has no schedule clearing the bar, so int8 falls back to fp8.
_QWENIMAGE_TE_INT8_SKIP = (6, 6)
# Gated NVFP4 auto head (int8 first; nvfp4 only with a passing gate record for this base and device).
_GATED_AUTO_PREFER = AutoPrefer(
    floor = (10, 0), schemes = ("int8", "fp8", "nvfp4", "mxfp8"), gated = True
)
# AUTO keeps bf16 while it fits for these (measured with both sides compiled: INT8 was at most ~5% faster and failed
# the default-on LPIPS bar); when bf16 would offload, INT8 is still picked.
_LUMINA2_BF16_WHEN_RESIDENT = (
    "INT8 is no faster than compiled bf16 on a card that holds bf16 and changes image detail"
)
_HIDREAM_BF16_WHEN_RESIDENT = (
    "INT8 is barely faster than compiled bf16 on a card that holds bf16 and changes image detail"
)
# Small DiTs (<= ~6B) stay bf16 while it fits: quantising moves detail (L4 vs bf16 LPIPS: FP8 FLUX.2-klein-4B 0.065,
# Z-Image-Turbo 0.103; FLUX.3-klein INT8 0.027). INT8 still when bf16 would offload; such loads compile from render 1.
SMALL_DIT_BF16_WHEN_RESIDENT = (
    "a small denoiser stays bf16 while it fits (INT8 / FP8 change image detail: FP8 LPIPS 0.065 on FLUX.2-klein-4B, "
    "0.103 on Z-Image-Turbo)"
)
# Bases of a small-DiT family that are not small: FLUX.2 klein 9B / base 9B (18.2 GB bf16 transformer, unmeasured)
# keep the quantized auto default. Upstream ids; mirrors fold back through ``canonical_base``.
_SMALL_DIT_LARGE_BASES = frozenset(
    {
        "black-forest-labs/flux.2-klein-9b",
        "black-forest-labs/flux.2-klein-base-9b",
    }
)


def small_dit_rule_excludes(base_repo: Optional[str]) -> bool:
    """True when ``base_repo`` (vendor id or mirror) is a large variant the small-DiT bf16 rule must not cover."""
    return canonical_base(base_repo).lower() in _SMALL_DIT_LARGE_BASES


# FLUX.1's sd-cli encoders are BFL's own diffusers files, read as-is: sd.cpp loads the CLIP-L text_encoder and the
# sharded T5 through its index (the shards sit beside it in the snapshot). The shards are listed so every fetch,
# size and cache probe sees them as plain files; ``*_shard`` kinds are fetched but never handed to sd-cli. dev, Krea
# and Kontext reuse schnell's files, as they shared one repack before.
_FLUX1_SD_CPP_TEXT_ENCODERS: tuple[tuple[str, str, str], ...] = (
    ("black-forest-labs/FLUX.1-schnell", "text_encoder/model.safetensors", "clip_l"),
    ("black-forest-labs/FLUX.1-schnell", "text_encoder_2/model.safetensors.index.json", "t5xxl"),
    (
        "black-forest-labs/FLUX.1-schnell",
        "text_encoder_2/model-00001-of-00002.safetensors",
        "t5xxl_shard",
    ),
    (
        "black-forest-labs/FLUX.1-schnell",
        "text_encoder_2/model-00002-of-00002.safetensors",
        "t5xxl_shard",
    ),
)
# FLUX.3 Klein's encoder is Qwen3-VL-4B-Instruct (byte-identical), read by sd.cpp through --llm with hidden states 4..32.
# References go through the VAE, not the vision tower, so editing needs no projector. Q8_0 rather than Q4_K_M: the
# encoder quant dominates the error here. Against a BF16 DiT + BF16 encoder render (sd.cpp, 4 prompts, 1024, same
# seeds, B200), LPIPS-alex mean with the Q8_0 DiT is 0.031 (Q8_0 encoder) vs 0.080 (Q4_K_M) on the distilled model and
# 0.008 vs 0.082 on the base; with UD-Q4_K_XL 0.068 vs 0.090 and 0.110 vs 0.109. 4.28 GB against 2.50 GB.
_FLUX3_SD_CPP_TEXT_ENCODERS = (
    (
        "unsloth/Qwen3-VL-4B-Instruct-GGUF",
        "Qwen3-VL-4B-Instruct-Q8_0.gguf",
        "llm",
    ),
)
# FLUX.3 landed upstream in master-968 (leejet/stable-diffusion.cpp#2143), after builds were in the wild. The model
# version name is in every sd-cli / sd-server that can run it and in none before.
_FLUX3_SD_CPP_ARCH_MARKER = "Flux.3 Image Klein"


# Keyed by architecture, not per variant: the base repo is read from the HF base_model tag at load time, so one entry
# covers Turbo/full, schnell/dev.
MOSS_IMAGE_FAMILY_NAME = "moss-image"
# MOSS-Image keep-list (plans/moss_quant.md): text_embedder, timestep MLP, AdaLN stay bf16 (LPIPS 0.058 vs 0.077 generic).
_MOSS_IMAGE_QUANT_KEEP: tuple[str, ...] = ("text_embedder", "time_embedding", "modulation")
_MOSS_IMAGE_CONVROT_SUFFIXES = (
    "proj.q",
    "proj.k",
    "proj.v",
    "proj.o",
    "proj.ffn_up_gate",
    "proj.ffn_down",
)

_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.1",
        comfy_shift_node = "flux",
        # FLUX.1-schnell is ComfyUI's FLOW-type FluxSchnell (ModelSamplingDiscreteFlow, shift 1.0).
        comfy_shift_node_variants = tuple(
            (f"{prefix}-{model}", node)
            for model, node in (("schnell", "sd3"), ("krea-dev", "flux"), ("dev", "flux"))
            for prefix in ("flux.1", "flux1", "flux-1", "flux")
        ),
        label = "FLUX.1-dev",
        profile = FamilyProfile(
            auto_prefer = _GATED_AUTO_PREFER,
            preview_factors = "flux1",
            # Auto static step skip, measured vs the no-skip render (PR #12652); dev: every 2 is 0.29 LPIPS on a
            # dense-texture prompt, so max only.
            auto_static_skip = (
                ("black-forest-labs/flux.1-krea-dev", (("default", 2), ("max", 3), ("steps", 28))),
                ("black-forest-labs/flux.1-dev", (("max", 3), ("steps", 28))),
            ),
        ),
        checkpoint_variants = ("flux1-schnell", "flux1-krea-dev", "flux1-dev"),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        cudnn_benchmark = False,
        pipeline_class = "FluxPipeline",
        uses_negative_prompt = False,
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        # ComfyUI fixed mu 1.15 for dev / Krea; schnell first (a dev GGUF may resolve to its base). Keys name the model: paths match too.
        comfy_flow_shift_variants = tuple(
            (f"{prefix}-{model}", shift)
            for model, shift in (
                ("schnell", None),
                ("krea-dev", flux_mu_shift(1.15)),
                ("dev", flux_mu_shift(1.15)),
            )
            for prefix in ("flux.1", "flux1", "flux-1", "flux")
        ),
        prequant_repos = (
            ("int8", "unsloth/FLUX.1-schnell-FP8"),
            ("fp8", "unsloth/FLUX.1-schnell-FP8"),
        ),
        # Checkpoints baked from the dev / Krea-dev weights (same arch, different weights); without these every
        # int8/fp8 load pays the dense download + on-the-fly quantise.
        prequant_variant_repos = (
            ("black-forest-labs/flux.1-dev", "int8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-dev", "fp8", "unsloth/FLUX.1-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "int8", "unsloth/FLUX.1-Krea-dev-FP8"),
            ("black-forest-labs/flux.1-krea-dev", "fp8", "unsloth/FLUX.1-Krea-dev-FP8"),
            # schnell ONLY: dev and Krea-dev would download it just for _validate_checkpoint to refuse.
            ("black-forest-labs/flux.1-schnell", "nvfp4", "unsloth/FLUX.1-schnell-NVFP4"),
        ),
        # Pre-cast T5-XXL (9.52 -> 5.90 GB; CLIP-L stays dense). One artifact serves schnell/dev/Krea-dev (T5 shards
        # are byte-identical).
        te_prequant_repos = (("fp8", "text_encoder_2", "unsloth/FLUX.1-schnell-FP8"),),
        # No te_quant_auto: a small encoder (T5-XXL, 4.8B) stays bf16 unless ComfyUI's template quantizes it, and the
        # FLUX.1 templates load t5xxl_fp16. Measured against the bf16 encoder (bf16 DiT, schnell 4 steps, 24 prompts,
        # bf16 A/A repeat bit-identical): the hosted int8 ConvRot T5 scores LPIPS-alex mean 0.055, SSIM 0.927, layerwise
        # fp8 0.096 on 12; both stay one explicit pick away (T5 resident 8.87 -> 4.56 GiB).
        aliases = ("flux1", "flux-1"),
        # LoRA training targets FLUX.1-dev via the DiT trainer (QLoRA nf4); the dev repo is gated
        trainable = True,
        train_base_repos = ("black-forest-labs/FLUX.1-dev",),
        img2img_pipeline_class = "FluxImg2ImgPipeline",
        inpaint_pipeline_class = "FluxInpaintPipeline",
        controlnet_pipeline_class = "FluxControlNetPipeline",
        controlnet_model_class = "FluxControlNetModel",
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        sd_cpp_text_encoders = _FLUX1_SD_CPP_TEXT_ENCODERS,
    ),
    # FLUX.2-klein is Flux2KleinPipeline (Qwen3 encoder), not the Mistral Flux2Pipeline, so it must precede a generic
    # flux match.
    DiffusionFamily(
        name = "flux.2-klein",
        comfy_shift_node = "flux",
        label = "FLUX.2 Klein",
        profile = FamilyProfile(
            auto_bf16_when_resident = SMALL_DIT_BF16_WHEN_RESIDENT,
            preview_factors = "flux2",
            auto_static_skip = (
                (
                    "black-forest-labs/flux.2-klein-base-4b",
                    (("default", 2), ("max", 3), ("steps", 50)),
                ),
            ),
        ),
        pipeline_class = "Flux2KleinPipeline",
        # B200: a per-process XBLOCK pick for the q/k RMSNorm reduction changed the latent and stuck to the compile home.
        filter_reduction_configs_archs = ((10, 0),),
        uses_negative_prompt = False,
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        prequant_repos = (
            ("int8", "unsloth/FLUX.2-klein-4B-FP8"),
            ("fp8", "unsloth/FLUX.2-klein-4B-FP8"),
        ),
        # Stock Qwen3-4B pre-cast fp8, the int8 ConvRot encoder's fallback (in
        # diffusion_te_prequant.TE_INT8_CONVROT_FILES); the distilled and base 4B repos ship the same bytes.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/FLUX.2-klein-4B-FP8"),),
        # The 9B pair carries stock Qwen3-8B, a different model, so it has its own files (int8 ConvRot in
        # diffusion_te_prequant.TE_INT8_CONVROT_VARIANT_FILES).
        te_prequant_variant_repos = (
            (
                "black-forest-labs/FLUX.2-klein-9B",
                "fp8",
                "text_encoder",
                "unsloth/FLUX.2-klein-9B-FP8",
            ),
        ),
        # No te_quant_auto: an unset request keeps the bf16 encoder. ComfyUI loads qwen_3_4b bf16 for the 4B pair and
        # qwen_3_8b_fp8mixed for the 9B pair; against the bf16 encoder (bf16 DiT) the hosted int8 ConvRot files score
        # LPIPS-alex mean 0.078 (4B, 4 steps, 24 prompts), 0.103 (base 4B, 20 steps CFG 5) and 0.047 (9B), layerwise fp8
        # 0.115 / 0.214 / 0.072. The encoder itself is close (int8 hidden-state cosine 0.9997 at the read layers), but
        # the distilled 4B amplifies it past what a silent default should cost; the files stay for explicit picks.
        aliases = ("flux2-klein",),
        # A local ComfyUI file has no card tag, so without these a 9B pick got the 4B base and was refused.
        named_variant_bases = (
            "black-forest-labs/FLUX.2-klein-9B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        # Train the undistilled bases, then preview their adapters on the matching 4-step models. Both vendor ids
        # resolve through the ungated mirrors at fetch time.
        trainable = True,
        train_base_repos = (
            "black-forest-labs/FLUX.2-klein-base-4B",
            "black-forest-labs/FLUX.2-klein-base-9B",
        ),
        deploy_base_repos = (
            (
                "black-forest-labs/FLUX.2-klein-base-4B",
                "black-forest-labs/FLUX.2-klein-4B",
            ),
            (
                "black-forest-labs/FLUX.2-klein-base-9B",
                "black-forest-labs/FLUX.2-klein-9B",
            ),
        ),
        # Flux2KleinPipeline takes reference image(s) via `image`, so it exposes a "reference" workflow atop
        # text-to-image. Inpaint but no img2img.
        reference = True,
        inpaint_pipeline_class = "Flux2KleinInpaintPipeline",
        # FLUX.2 scales >1MP inputs to ~1MP, so outpaint can't grow.
        inpaint_preserves_size = False,
        # BFL's own diffusers VAE file, byte-identical to the single-file repack this used to name.
        sd_cpp_vae = ("black-forest-labs/FLUX.2-dev", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_vae_format = "flux2",
        # Klein 4B's text encoder IS Qwen3-4B (byte-identical shards to unsloth/Qwen3-4B; the BF16 GGUF matches it
        # tensor for tensor and renders pixel-identical to the old single-file encoder), so sd-cli reads unsloth's
        # own GGUF of it. Q8_0, not Q4_K_M: on klein 4B the Q4 encoder moved renders by LPIPS 0.15, Q8 by 0.06.
        sd_cpp_text_encoders = (("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", "llm"),),
    ),
    # FLUX.3 Image Klein: a 5.6B flow-map DiT with a Qwen3-VL-4B encoder (layers 4..32 stacked) and the FLUX.2 VAE.
    # Diffusers ships it only as modular blocks, so diffusion_flux3 assembles it per component behind a classic
    # pipeline face (no model_index.json in the repo). The distilled checkpoint runs 16 steps without CFG and edits
    # with up to 10 reference images through the same text-to-image call. Base and distilled are separate entries:
    # they run different block graphs (the base adds a CFG guider and a negative prompt). Base first in the table only
    # for reading order; the longest-alias match is what keeps "...-klein-base" off the distilled entry.
    DiffusionFamily(
        name = "flux.3-klein-base",
        comfy_shift_node = "flux",
        label = "FLUX.3 Klein Base",
        profile = FamilyProfile(
            auto_bf16_when_resident = SMALL_DIT_BF16_WHEN_RESIDENT,
            # Left dense for EVERY transformer-quant scheme: per-token modulations, timestep embedders, norm_out, the
            # text-stream per-mode blocks and the 20480-wide text projection. Measured on B200 against bf16 (11-case
            # recipe set, LPIPS mean): int8 0.097 -> 0.027, fp8 0.063 -> 0.033, mxfp8 0.024 with it; without it mxfp8
            # and nvfp4 crash at the first step (the modulation inputs are a stride-0 ``temb.expand``). Spelled out
            # in full because the generic int8 "_mod" token only matches txt_mode_blocks by accident.
            quant_exclude_tokens = _FLUX3_QUANT_KEEP,
            # Every keep-set measured 0.09-0.12 LPIPS against bf16 on NVFP4, as did BFL's own nvfp4 package. A passing
            # NVFP4 gate record for a base still lifts it.
            scheme_deny = frozenset({"nvfp4"}),
            # The FLUX.2 VAE and the same 2x2 packed 128-channel tokens, so FLUX.2's fitted map applies.
            preview_factors = "flux2",
        ),
        pipeline_class = "Flux3ModularPipeline",
        # Same layout as the distilled repo. The INT8 file is BFL's own ComfyUI int8_convrot denoiser
        # (black-forest-labs/flux-3-image-klein-comfy, byte-identical), which scored better than an INT8 built with
        # the keep-list, so the prequant loader reads it through the ComfyUI quant path. The FP8 file is Studio's own.
        prequant_repos = (
            ("int8", "unsloth/flux-3-image-klein-base-FP8"),
            ("fp8", "unsloth/flux-3-image-klein-base-FP8"),
        ),
        transformer_class = "Flux3Transformer2DModel",
        # Qwen3-VL-4B pre-cast fp8, the int8 ConvRot encoder's fallback (diffusion_te_prequant.TE_INT8_CONVROT_FILES).
        # Byte-identical to the distilled repo's: the two bases ship the same text_encoder/ shards.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/flux-3-image-klein-base-FP8"),),
        # int8 ConvRot by default, as BFL's ComfyUI templates load qwen3vl_4b_int8_convrot (4 of 5 workflows in
        # black-forest-labs/flux-3-image-klein-comfy). Against the bf16 encoder (bf16 DiT, 19 cases, 25 steps CFG 3.5,
        # B200, same-GPU A/A bit-identical) the hosted file scores LPIPS-alex mean 0.023 / max 0.121, SSIM 0.973;
        # layerwise fp8, the fallback when the file cannot be loaded, 0.035 / 0.132, SSIM 0.960. Resident encoder
        # 8.27 -> 4.89 GiB, load 18.4 -> 1.8 s.
        te_quant_auto = "int8",
        base_repo = "black-forest-labs/flux-3-image-klein-base",
        aliases = ("flux-3-image-klein-base", "flux3-image-klein-base", "flux3-klein-base"),
        reference = True,
        max_condition_images = 10,
        # Two DiT patch grids (VAE /8, patch 2): sizes snap to 16 like FLUX.2.
        dimension_multiple = 16,
        # Each reference (resized to 1 MP) adds about 700 MiB of denoise activations in bf16 and 900 MiB with an int8 /
        # fp8 denoiser, measured on B200, against the estimator's 8192 MiB per megapixel: 0.11 covers the quantised
        # figure.
        condition_pixel_weight = 0.11,
        # Native (sd.cpp) assets, the same set as the distilled entry: see there.
        sd_cpp_vae = (
            "unsloth/flux-3-image-klein-base-FP8",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        sd_cpp_text_encoders = _FLUX3_SD_CPP_TEXT_ENCODERS,
        sd_cpp_sampling_method = "euler",
        sd_cpp_arch_marker = _FLUX3_SD_CPP_ARCH_MARKER,
        sd_cpp_reference = True,
    ),
    DiffusionFamily(
        name = "flux.3-klein",
        comfy_shift_node = "flux",
        label = "FLUX.3 Klein",
        profile = FamilyProfile(
            auto_bf16_when_resident = SMALL_DIT_BF16_WHEN_RESIDENT,
            # Left dense for EVERY transformer-quant scheme: per-token modulations, timestep embedders, norm_out, the
            # text-stream per-mode blocks and the 20480-wide text projection. Measured on B200 against bf16 (11-case
            # recipe set, LPIPS mean): int8 0.097 -> 0.027, fp8 0.063 -> 0.033, mxfp8 0.024 with it; without it mxfp8
            # and nvfp4 crash at the first step (the modulation inputs are a stride-0 ``temb.expand``). Spelled out
            # in full because the generic int8 "_mod" token only matches txt_mode_blocks by accident.
            quant_exclude_tokens = _FLUX3_QUANT_KEEP,
            # Every keep-set measured 0.09-0.12 LPIPS against bf16 on NVFP4, as did BFL's own nvfp4 package. A passing
            # NVFP4 gate record for a base still lifts it.
            scheme_deny = frozenset({"nvfp4"}),
            # The FLUX.2 VAE and the same 2x2 packed 128-channel tokens, so FLUX.2's fitted map applies.
            preview_factors = "flux2",
        ),
        pipeline_class = "Flux3DistilledModularPipeline",
        # Pre-quantized distilled denoisers (both schemes in one repo, as FLUX.2 klein), built with the keep-list above;
        # the loader checks the stored exclude set against it. The base has its own repo.
        prequant_repos = (
            ("int8", "unsloth/flux-3-image-klein-FP8"),
            ("fp8", "unsloth/flux-3-image-klein-FP8"),
        ),
        uses_negative_prompt = False,
        uses_guidance = False,
        transformer_class = "Flux3Transformer2DModel",
        # Qwen3-VL-4B pre-cast fp8, the int8 ConvRot encoder's fallback (diffusion_te_prequant.TE_INT8_CONVROT_FILES).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/flux-3-image-klein-FP8"),),
        # int8 ConvRot by default, as on the base and in BFL's ComfyUI templates: 0.017 / 0.098 LPIPS-alex, SSIM 0.974,
        # fp8 fallback 0.038 / 0.220, SSIM 0.945 against the bf16 encoder (19 cases, 16 steps).
        te_quant_auto = "int8",
        base_repo = "black-forest-labs/flux-3-image-klein",
        aliases = ("flux-3-image-klein", "flux3-image-klein", "flux3-klein"),
        reference = True,
        max_condition_images = 10,
        dimension_multiple = 16,
        condition_pixel_weight = 0.11,
        # Native (sd.cpp) assets. sd.cpp reads the unsloth/flux-3-image-klein[-base]-GGUF denoisers as published
        # (BFL tensor names, detected from emb_in.image + txt_mode_blocks). The VAE is the FLUX.2 VAE, byte-identical
        # in both FP8 repos and in unsloth/FLUX.2-VAE (sha256 d64f3a68...), taken from this family's own FP8 repo,
        # which prequant_repos keeps loadable. sd.cpp detects its layout itself, so no --vae-format.
        sd_cpp_vae = ("unsloth/flux-3-image-klein-FP8", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_text_encoders = _FLUX3_SD_CPP_TEXT_ENCODERS,
        # sd.cpp's documented invocation (docs/flux3.md). No flow shift: its scheduler already applies FLUX.3's 6.93.
        sd_cpp_sampling_method = "euler",
        sd_cpp_arch_marker = _FLUX3_SD_CPP_ARCH_MARKER,
        sd_cpp_reference = True,
    ),
    # FLUX.2-dev: full (non-distilled) FLUX.2 on the Mistral Flux2Pipeline, so its own entry. Gated base,
    # text-to-image only; sd-cli takes the Mistral encoder from unsloth's Mistral-Small-3.2 GGUF and the VAE from
    # BFL's own diffusers file.
    DiffusionFamily(
        name = "flux.2-dev",
        comfy_shift_node = "flux",
        label = "FLUX.2-dev",
        profile = FamilyProfile(te_int8_skip = (3, 0), preview_factors = "flux2"),
        pipeline_class = "Flux2Pipeline",
        # cudnn.benchmark's per-process VAE conv pick rendered 5 variants of one seed over 7 fresh B200 processes.
        cudnn_benchmark = False,
        uses_negative_prompt = False,
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        prequant_repos = (
            ("int8", "unsloth/FLUX.2-dev-FP8"),
            ("fp8", "unsloth/FLUX.2-dev-FP8"),
        ),
        # Pre-cast Mistral-Small-24B conditioner (bf16 ~48 GB dense, ~24.7 GB pre-cast).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/FLUX.2-dev-FP8"),),
        # int8 ConvRot by default. ComfyUI's FLUX.2-dev templates load mistral_3_small_flux2_bf16 (2 of 3; the third
        # fp8); against the bf16 encoder (bf16 DiT, 20 steps guidance 4, 24 prompts, bf16 A/A repeat bit-identical) the
        # hosted int8 ConvRot file scores LPIPS-alex mean 0.057, SSIM 0.929, layerwise fp8 0.266 / SSIM 0.730 on 12. The
        # Pixtral vision tower stays bf16. Resident 44.7 -> 22.8 GiB, load 9.3 -> 3.6 s.
        te_quant_auto = "int8",
        aliases = ("flux2-dev", "flux2dev"),
        trainable = True,
        train_base_repos = ("black-forest-labs/FLUX.2-dev",),
        # BFL's own diffusers VAE file, byte-identical to the single-file repack this used to name.
        sd_cpp_vae = ("black-forest-labs/FLUX.2-dev", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_vae_format = "flux2",
        # FLUX.2-dev's text encoder is Mistral-Small-3.2-24B-Instruct-2506 (byte-identical shards to the unsloth
        # copy; the BF16 GGUF matches every text tensor), so sd-cli reads unsloth's GGUF of it.
        sd_cpp_text_encoders = (
            (
                "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
                "llm",
            ),
        ),
    ),
    DiffusionFamily(
        # FLUX instruction editing: FluxKontextPipeline takes an image + instruction. Specific aliases first so
        # detect_family prefers this over "flux.1".
        name = "flux.1-kontext",
        comfy_shift_node = "flux",
        profile = FamilyProfile(preview_factors = "flux1"),
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS,
        comfy_flow_shift = flux_mu_shift(
            1.15
        ),  # ComfyUI ModelSamplingFlux fixed mu 1.15 (Kontext template)
        pipeline_class = "FluxKontextPipeline",
        uses_negative_prompt = False,
        prompt_hint = (
            'Write the change you want as an instruction, such as "make the sky orange", not a'
            " description of the whole picture."
        ),
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        # FLUX.1's pre-cast T5-XXL: Kontext ships the same text_encoder_2 shards (sha256 compared 2026-10-16).
        te_prequant_repos = (("fp8", "text_encoder_2", "unsloth/FLUX.1-schnell-FP8"),),
        # fp8 by default, as ComfyUI's Kontext template (t5xxl_fp8_e4m3fn_scaled): against the bf16 encoder (bf16 DiT, 8
        # edits, 20 steps, bf16 A/A repeat bit-identical) LPIPS-alex mean 0.034 / max 0.143, SSIM 0.945. The shared int8
        # ConvRot T5 scores 0.021 and is one explicit pick away.
        te_quant_auto = "fp8",
        aliases = ("flux.1-kontext-dev", "flux1-kontext", "flux-kontext", "kontext"),
        edit = True,
        # FluxKontextPipeline conditions on one image.
        max_condition_images = 1,
        # Native: FLUX.1's VAE + CLIP-L / T5-XXL, the source as a reference image (sd.cpp docs/kontext.md).
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        sd_cpp_text_encoders = _FLUX1_SD_CPP_TEXT_ENCODERS,
    ),
    DiffusionFamily(
        # Qwen instruction editing: the 2511 checkpoint ships as QwenImageEditPlusPipeline. Specific aliases first so
        # detect_family prefers this over "qwen-image".
        name = "qwen-image-edit",
        comfy_shift_node = "sd3",
        # Same DiT class + unpadded text stream as qwen-image.
        profile = FamilyProfile(
            int8_exclude_tokens = _QWENIMAGE_INT8_EXCLUDES,
            scheme_deny = _QWENIMAGE_SCHEME_DENY,
            train_scheme_deny = _QWENIMAGE_TRAIN_SCHEME_DENY,
            te_int8_skip = _QWENIMAGE_TE_INT8_SKIP,
            preview_factors = "qwen_image",
        ),
        filter_reduction_configs_archs = _QWEN_REDUCTION_RACE_ARCHS,
        comfy_flow_shift = 3.1,  # ComfyUI ModelSamplingAuraFlow 3.1 (Qwen-Image-Edit 2511 template)
        comfy_flow_shift_variants = (
            ("qwen-image-edit-2511", 3.1),
            ("qwen-image-edit-2509", 3.0),
            ("qwen-image-edit", 3.0),
        ),
        # Only the 2511 config sets zero_cond_t; on 2509 / original Edit it renders oversaturated, off-identity images.
        transformer_config_variants = (
            ("qwen-image-edit-2511", ()),
            ("qwen-image-edit-2509", (("zero_cond_t", False),)),
            ("qwen-image-edit", (("zero_cond_t", False),)),
        ),
        pipeline_class = "QwenImageEditPlusPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        # Qwen-Image's pre-cast Qwen2.5-VL-7B: every edit release (original, 2509, 2511) ships the same text_encoder/
        # shards. The vision tower is cast too, and the edit path runs it.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        # int8 ConvRot by default. ComfyUI loads qwen_2.5_vl_7b_fp8_scaled; our layerwise fp8 (no scales) sits at the
        # limit against the bf16 encoder (bf16 DiT, 8 edits, 40 steps: LPIPS-alex mean 0.060, max 0.330), the shared
        # int8 ConvRot file at 0.017 / max 0.033, SSIM 0.970. The vision tower stays bf16 in it.
        te_quant_auto = "int8",
        cfg_kwarg = "true_cfg_scale",
        aliases = (
            "qwen-image-edit-2511",
            "qwen-image-edit-2509",
            "qwen-image-edit",
            "qwen_image_edit",
            "qwenimageedit",
        ),
        edit = True,
        # The source plus up to two more images in one edit (2509 / 2511 multi-image editing): the Plus pipeline reads
        # each through the VLM as "Picture N" and VAE-encodes it beside the source. ComfyUI's TextEncodeQwenImageEditPlus
        # node takes image1..image3, so 3 in total.
        max_condition_images = 3,
        # The original Qwen-Image-Edit was trained on the source alone: it keeps the single-image edit.
        max_condition_images_variants = (
            ("qwen-image-edit-2511", 3),
            ("qwen-image-edit-2509", 3),
            ("qwen-image-edit", 1),
        ),
        prompt_hint = (
            "Write the change you want as an instruction, not a description of the whole picture."
            " With several input images, refer to them as Image 1, Image 2 and so on."
        ),
        # same DiT as qwen-image
        fp16_incompatible = True,
        # Native: qwen-image's VAE + Qwen2.5-VL, plus its projector (edits read the source through the VLM; a GGUF
        # encoder has no vision weights). 2511 needs no flag: sd.cpp keys zero_cond_t off __index_timestep_zero__.
        # The official diffusers VAE; sd.cpp maps its names itself, as it does for Layered's.
        sd_cpp_vae = ("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
            ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision"),
        ),
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 3.0,
    ),
    DiffusionFamily(
        # qwen-image's DiT plus use_additional_t_cond / use_layer3d_rope (read from the base config for a GGUF) and an
        # RGBA VAE. Listed before qwen-image so the name outranks it.
        name = "qwen-image-layered",
        comfy_shift_node = "sd3",
        # Same DiT class + unpadded text stream as qwen-image.
        profile = FamilyProfile(
            int8_exclude_tokens = _QWENIMAGE_INT8_EXCLUDES,
            scheme_deny = _QWENIMAGE_SCHEME_DENY,
            te_int8_skip = _QWENIMAGE_TE_INT8_SKIP,
        ),
        filter_reduction_configs_archs = _QWEN_REDUCTION_RACE_ARCHS,
        # ComfyUI's Image to Layers template: ModelSamplingAuraFlow 1, 2 layers, the input scaled to 640.
        comfy_flow_shift = 1.0,
        layer_count = 2,
        layer_resolution = 640,
        pipeline_class = "QwenImageLayeredPipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image-Layered",
        # Qwen-Image's pre-cast Qwen2.5-VL-7B: the layered release ships the same text_encoder/ shards.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        # fp8 by default, as ComfyUI's layered templates (qwen_2.5_vl_7b_fp8_scaled): LPIPS-alex mean 0.0055 / max
        # 0.034, SSIM 0.993 against the bf16 encoder (bf16 DiT, 8 inputs, 20 steps CFG 2.5, bf16 A/A repeat
        # bit-identical).
        te_quant_auto = "fp8",
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image_layered", "qwenimagelayered"),
        edit = True,
        max_condition_images = 1,
        condition_image_mode = "RGBA",
        # same DiT as qwen-image
        fp16_incompatible = True,
        # Native: its own 4-channel VAE (the RGB one cannot decode layers; pixel-identical to the ComfyUI repack) and
        # qwen-image's encoder, no projector (sd.cpp: enable_vision = version != VERSION_QWEN_IMAGE_LAYERED).
        sd_cpp_vae = ("Qwen/Qwen-Image-Layered", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
        ),
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 1.0,
        # Layered support landed upstream in master-744 (556f04b); an older reused build has no such literal.
        sd_cpp_arch_marker = "qwen_image_layers",
    ),
    DiffusionFamily(
        name = "qwen-image",
        comfy_shift_node = "sd3",
        label = "Qwen-Image",
        profile = FamilyProfile(
            int8_exclude_tokens = _QWENIMAGE_INT8_EXCLUDES,
            scheme_deny = _QWENIMAGE_SCHEME_DENY,
            train_scheme_deny = _QWENIMAGE_TRAIN_SCHEME_DENY,
            auto_prefer = _GATED_AUTO_PREFER,
            te_int8_skip = _QWENIMAGE_TE_INT8_SKIP,
            preview_factors = "qwen_image",
            auto_static_skip = (("qwen/qwen-image", (("default", 2), ("max", 3), ("steps", 20))),),
        ),
        filter_reduction_configs_archs = _QWEN_REDUCTION_RACE_ARCHS,
        comfy_flow_shift = 3.1,  # ComfyUI ModelSamplingAuraFlow 3.1 (Qwen-Image templates)
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        # Qwen-Image-FP8.pt was rebuilt with the activation scale floor (LPIPS 0.044 against the bf16
        # render at 1024, seed 20260914), so an fp8 host seeds it instead of quantising bf16 in memory.
        prequant_repos = (
            ("int8", "unsloth/Qwen-Image-FP8"),
            ("fp8", "unsloth/Qwen-Image-FP8"),
        ),
        # 2512 is a different checkpoint with its own baked artifacts; without these rows a 2512 pick
        # plans the Qwen-Image artifact, refused by base_model_id once the shards are already dropped.
        prequant_variant_repos = (
            ("qwen/qwen-image-2512", "int8", "unsloth/Qwen-Image-2512-FP8"),
            ("qwen/qwen-image-2512", "fp8", "unsloth/Qwen-Image-2512-FP8"),
            # Policy ``qwen2512_m120_attn8_v1``, 2512 only: Qwen/Qwen-Image keeps its nvfp4 deny.
            ("qwen/qwen-image-2512", "nvfp4", "unsloth/Qwen-Image-2512-NVFP4"),
        ),
        # Pre-cast Qwen2.5-VL-7B (16.6 -> 8.8 GB). Always was independent of the DiT scheme rules.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        # int8 ConvRot by default. ComfyUI loads qwen_2.5_vl_7b_fp8_scaled, but our layerwise fp8 (no per-tensor scale)
        # moves too much: against the bf16 encoder (bf16 DiT, 20 steps CFG 4, bf16 A/A repeat bit-identical) LPIPS-alex
        # mean 0.138 / SSIM 0.844 on 12 prompts, int8 ConvRot 0.057 / SSIM 0.938 on 24 (one 0.47 is a cone moved to the
        # side the prompt asks for). Text renders correctly in both. Resident 15.45 -> 8.36 GiB, load 7.3 -> 2.6 s.
        te_quant_auto = "int8",
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image", "qwenimage"),
        # fp16 overflows to NaN latents (black images)
        fp16_incompatible = True,
        cudnn_benchmark = False,
        trainable = True,
        train_base_repos = ("unsloth/Qwen-Image-2512-unsloth-bnb-4bit", "Qwen/Qwen-Image"),
        img2img_pipeline_class = "QwenImageImg2ImgPipeline",
        inpaint_pipeline_class = "QwenImageInpaintPipeline",
        controlnet_pipeline_class = "QwenImageControlNetPipeline",
        controlnet_model_class = "QwenImageControlNetModel",
        # The official diffusers VAE; sd.cpp maps its names itself, as it does for Layered's.
        sd_cpp_vae = ("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors"),
        # Qwen2.5-VL as a Q4_K_M GGUF keeps the CPU RAM win (bf16 encoder is ~15 GB). sd-cli --qwen2vl aliases --llm.
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "qwen2vl",
            ),
        ),
        # Qwen-Image's supported sd.cpp invocation (docs/qwen_image.md).
        sd_cpp_sampling_method = "euler",
        sd_cpp_flow_shift = 3.0,
    ),
    DiffusionFamily(
        # Qwen-Image-2.1 is a different ARCHITECTURE, not a refreshed checkpoint, so it is its own
        # family rather than a prequant_variant_repos row on qwen-image: 32 single-stream blocks
        # against 60 dual-stream MMDiT ones, nine weights per block, no bias tensors anywhere, one
        # global modulation projection, a Qwen3-VL text encoder and its own VAE class. Sharing the
        # qwen-image entry would hand it that family's pipeline, transformer, VAE and exclusion
        # rules, none of which fit.
        name = "qwen-image-2.1",
        comfy_shift_node = "flux",
        profile = FamilyProfile(
            # 2.1 is a 32-block SINGLE-stream DiT: no add_* projections, no txt_mlp, so the 20B MMDiT's exclusion list
            # does not apply and this one was measured rather than inherited. ``txt_in`` is here as a QUALITY lever,
            # not the small-M crash guard it is on qwen-image: all four policy arms rendered a deliberately 2-token
            # prompt without tripping ``_int_mm``'s M floor, so nothing crashes either way. Excluding it scored LPIPS
            # 0.0642 against 0.1032 with the generic tokens alone, paired over 16 calibration prompts, -0.0390 CI95
            # [-0.0708, -0.0072], better on 14/16.
            int8_exclude_tokens = ("txt_in",),
            int8_convrot = ConvRotSpec(
                group = 256,
                suffixes = (
                    "attn.to_q",
                    "attn.to_k",
                    "attn.to_v",
                    "attn.to_out.0",
                    "img_mlp.gate_layer",
                    "img_mlp.proj",
                    "img_mlp.out",
                ),
                filename = "Qwen-Image-2.1-INT8-ConvRot.safetensors",
                repo = "unsloth/Qwen-Image-2.1-FP8",
                variant_repos = ("unsloth/Qwen-Image-2.1-Turbo-FP8",),
                default_on = True,
            ),
            auto_prefer = _GATED_AUTO_PREFER,
            preview_factors = "qwen_image_21",
            auto_static_skip = (
                ("qwen/qwen-image-2.1", (("default", 3), ("max", 3), ("steps", 25))),
            ),
        ),
        # ComfyUI QwenImage21: ModelSamplingFlux fixed mu 0.69, no terminal stretch.
        comfy_flow_shift = flux_mu_shift(0.69),
        pipeline_class = "QwenImage21Pipeline",
        transformer_class = "QwenImage21Transformer2DModel",
        base_repo = "Qwen/Qwen-Image-2.1",
        prequant_repos = (
            ("int8", "unsloth/Qwen-Image-2.1-FP8"),
            ("fp8", "unsloth/Qwen-Image-2.1-FP8"),
            ("nvfp4", "unsloth/Qwen-Image-2.1-NVFP4"),
        ),
        # Turbo's own checkpoints (a different distill); derived names (2.1's plus -Turbo) resolve them.
        prequant_variant_repos = (
            ("qwen/qwen-image-2.1-turbo", "int8", "unsloth/Qwen-Image-2.1-Turbo-FP8"),
            ("qwen/qwen-image-2.1-turbo", "fp8", "unsloth/Qwen-Image-2.1-Turbo-FP8"),
        ),
        # 2.1's artifacts are baked from 2.1's denoiser: for nvfp4 Turbo quantizes its own weights.
        prequant_excluded_bases = ("qwen/qwen-image-2.1-turbo",),
        named_variant_bases = ("Qwen/Qwen-Image-2.1-Turbo",),
        # The artifacts are safetensors, not the historical torch.save pickle, so the family has to
        # NAME them: every derived fallback ends in .pt, and without these rows the loader would ask
        # the Hub for a file that is not there and silently fall back to the dense bf16 download.
        prequant_filenames = (
            ("fp8", "Qwen-Image-2.1-FP8.safetensors"),
            ("int8", "Qwen-Image-2.1-INT8.safetensors"),
        ),
        # Qwen3-VL 8B pre-cast fp8, the int8 ConvRot encoder's fallback (diffusion_te_prequant.TE_INT8_CONVROT_FILES).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-2.1-FP8"),),
        # The encoder is the BIG component here, not the denoiser: Qwen3-VL-8B is 16.33 GiB dense against 6.76 GiB for
        # the INT8 transformer, so a quantised denoiser alone still costs ~26 GB and the hosted pre-cast encoder is
        # what makes the family fit a 24 GB card. Measured on this family at 1024/40 steps: 23.74 -> 16.16 GiB resident
        # and 11.6 -> 1.5 s to load the encoder, with per-image render time unchanged.
        # int8 ConvRot weight-only, as ComfyUI's Qwen-Image-2.1 template (fp8 misspells rendered text).
        te_quant_auto = "int8",
        cfg_kwarg = "true_cfg_scale",
        # 2.1 is UNIFIED: one pipeline, and QwenImage21Pipeline.__call__ takes ``image`` as condition
        # images alongside the prompt, so this is the FLUX.2 shape rather than the Qwen-Image-Edit
        # one. Not ``edit``, which means the pipeline IS the edit pipeline and there is no plain
        # text-to-image: here text-to-image is the default and the images are optional context. The
        # encoder reads each one as vision context and the VAE turns it into latent tokens prepended
        # to the noise, with no ``strength`` anywhere, which is exactly what the reference workflow
        # passes. Without this flag diffusion.py refuses reference images for this family and the
        # capability ships dark.
        reference = True,
        # Model card: ten inputs, alpha kept for the VAE, 32 px grid (16x VAE, 2x2 token groups), 2K presets up to
        # 2400 x 1792. ``reference_resolutions`` maps to the pipeline's ``output_resolution``.
        unified_edit = True,
        max_condition_images = 10,
        condition_image_mode = "RGBA",
        dimension_multiple = 32,
        max_output_side = 2752,
        max_output_pixels = 2400 * 1792,
        reference_resolutions = (512, 1024, 2048),
        # Condition tokens are prefilled once into a KV cache, not denoised per step. Measured on an RTX 3090: about
        # 2.5-2.7 GiB per 1024-square condition image against the estimator's 8 GiB per output megapixel.
        condition_pixel_weight = 0.32,
        aliases = ("qwen_image_21", "qwenimage21", "qwen-image-21"),
        prompt_hint = (
            "Describe the image, or give an edit instruction over the input images and refer to them"
            ' as Image 1, Image 2. For a transparent background, start with "This is an RGBA image'
            ' with transparency."'
        ),
        # Built by us from Qwen/Qwen-Image-2.1 itself: the same 238 tensors under upstream's own
        # names, cast fp32 -> bf16 and written as one file, because sd-cli takes --vae as a single
        # file rather than a diffusers subfolder. That is 0.63 GiB against upstream's 1.26 GiB of
        # fp32, and a fixed-seed 1024 render through it is PIXEL-identical to one made with the
        # fp32 original. No third-party repack is in the chain; the file is nonetheless
        # value-identical to Comfy-Org's bf16 repack, all 238 tensors, so either works.
        #
        # It lives beside the pre-cast text encoder in the FP8 repo rather than in a companion repo
        # of its own, and NOT in the GGUF repo: sd_cpp_companion_only_repo_ids() is companions
        # minus loadable, and the GGUF repo appears in no family field, so pointing here at it
        # would classify the repo that holds every denoiser as fetch-only and hide it from the
        # catalog. The FP8 repo is in prequant_repos, so it is subtracted and stays loadable.
        # FP8 and INT8 picks do not use this file at all; they take the VAE from the base repo
        # through diffusers, as every prequant family does.
        #
        # 2.1 has its own VAE class, so the qwen-image or Wan 2.2 file decodes to noise here
        # rather than failing.
        sd_cpp_vae = ("unsloth/Qwen-Image-2.1-FP8", "vae/qwen_image_2.1_vae_bf16.safetensors"),
        # Qwen3-VL 8B, not Qwen2.5-VL, and supplied through --llm rather than --qwen2vl: the
        # qwen2vl flag carries Qwen2-VL's vision preprocessing, which this encoder does not want.
        # UD-Q4_K_XL, the Dynamic 2.0 rung, rather than the uniform Q4_K_M: same 4-bit class and
        # the same CPU RAM win the no-GPU route exists for (bf16 is 16.4 GB, this is 5.1 against
        # 5.0), with the per-tensor precision the dynamic recipe assigns. Measured on this host
        # against the 2.1-capable build, one prompt at 1024 with 20 steps and a shared seed:
        # LPIPS 0.02894 / SSIM 0.95908 between the two, 36.5 s against 39.0 s, both coherent.
        # The vision projector (--llm_vision) is what native editing needs; text-to-image never reads it (a fixed-seed
        # render is byte-identical with and without it). Listed here so download, readiness and delete guards see it.
        sd_cpp_text_encoders = (
            (
                "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                "llm",
            ),
            ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision"),
        ),
        sd_cpp_sampling_method = "euler",
        # No flow shift on purpose. Qwen-Image pins 3.0, but upstream sd.cpp selects a
        # resolution-dependent schedule for qwen_image_2_1 itself, and passing a fixed shift
        # overrides that silently: the render still succeeds and is simply worse off-square.
        #
        # Support landed upstream on 2026-09-20, long after builds shipped, so the gate matters
        # here more than anywhere: measured across six builds on this host, only the pinned
        # release (both sd-cli and sd-server, CPU and CUDA) carries this literal, and the five
        # older ones, including the previous pin, do not.
        sd_cpp_arch_marker = "qwen_image_2_1",
        # The edit path's no-projector refusal: in the pinned build, absent from the one before it.
        sd_cpp_edit_marker = "Qwen Image 2.1 editing requires Qwen3-VL vision weights",
    ),
    DiffusionFamily(
        name = "z-image",
        comfy_shift_node = "sd3",
        label = "Z-Image",
        profile = FamilyProfile(
            # exactly the Linears the hosted INT8 checkpoint quantizes (all input axes 256-divisible)
            int8_convrot = ConvRotSpec(
                group = 256,
                suffixes = (
                    "attention.to_q",
                    "attention.to_k",
                    "attention.to_v",
                    "attention.to_out.0",
                    "feed_forward.w1",
                    "feed_forward.w2",
                    "feed_forward.w3",
                    "cap_embedder.1",
                ),
                filename = "Z-Image-Turbo-INT8-ConvRot.safetensors",
                repo = "unsloth/Z-Image-Turbo-FP8",
                default_on = True,
            ),
            auto_prefer = _GATED_AUTO_PREFER,
            auto_bf16_when_resident = SMALL_DIT_BF16_WHEN_RESIDENT,
            preview_factors = "zimage",
        ),
        # B200 too: 1 of 4 fresh servers picked R0_BLOCK 1024 (others 4096) and rendered other latents.
        filter_reduction_configs_archs = _REDUCTION_RACE_ARCHS + ((10, 0),),
        cudnn_benchmark = False,
        comfy_flow_shift = 3.0,  # ComfyUI shift 3 for Turbo and base (Turbo already ships 3.0)
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        prequant_repos = (
            ("int8", "unsloth/Z-Image-Turbo-FP8"),
            ("fp8", "unsloth/Z-Image-Turbo-FP8"),
            ("nvfp4", "unsloth/Z-Image-Turbo-NVFP4"),
        ),
        # Both hosted checkpoints are baked from the distilled Turbo transformer, so the undistilled base has none and
        # must quantize its own dense weights.
        prequant_excluded_bases = ("tongyi-mai/z-image",),
        # Pre-cast Qwen3-4B TE (8.04 -> 4.41 GB), shared with the undistilled base (byte-identical encoder). NOT
        # shared with flux.2-klein-4B: klein ships stock Qwen3-4B (sha256-identical to unsloth/Qwen3-4B), while this
        # encoder is Z-Image's own variant of it (layer 35 MLP up/down_proj maxdiff 0.86 against klein's).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Z-Image-Turbo-FP8"),),
        # No te_quant_auto: a small encoder (Qwen3-4B) stays bf16 unless ComfyUI's template quantizes it, and 3 of its
        # 4 templates load qwen_3_4b bf16. Measured against the bf16 encoder (bf16 DiT, 8 steps, 12 prompts, bf16 A/A
        # repeat bit-identical): int8 ConvRot LPIPS-alex mean 0.049, SSIM 0.934, layerwise fp8 0.105; both stay one
        # explicit pick away (resident 7.49 -> 4.11 GiB).
        aliases = ("zimage", "z_image"),
        # LoRA training via the DiT trainer (bf16); defaults to the prequant nf4 repo for QLoRA
        trainable = True,
        # The undistilled base is what the upstream DreamBooth recipe trains on. No deploy pairing: its adapters
        # preview on the base itself at the 20-step / guidance 4 recipe.
        train_base_repos = (
            "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
            "Tongyi-MAI/Z-Image-Turbo",
            "Tongyi-MAI/Z-Image",
        ),
        img2img_pipeline_class = "ZImageImg2ImgPipeline",
        inpaint_pipeline_class = "ZImageInpaintPipeline",
        fp16_incompatible = True,
        # The attention / FFN branches overflow float16 before their post-norm.
        fp16_guard = "rescale_post_norm",
        mlx_model_config = "z-image-turbo",
        mlx_model_config_by_base = (("Tongyi-MAI/Z-Image", "z-image"),),
        # Z-Image decodes with the FLUX.1 autoencoder: BFL's ae.safetensors, byte-identical to the repack's file.
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        # Z-Image's encoder differs from stock Qwen3-4B only in the LAST layer's MLP, which neither diffusers
        # (hidden_states[-2]) nor sd.cpp (out_layers {35}) reads, so the conditioning is the same. Same Q8_0 file
        # klein 4B reads, so one download serves both.
        sd_cpp_text_encoders = (("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", "llm"),),
    ),
    DiffusionFamily(
        name = "krea-2",
        comfy_shift_node = "flux",
        label = "Krea 2",
        pipeline_class = "Krea2Pipeline",
        transformer_class = "Krea2Transformer2DModel",
        base_repo = "krea/Krea-2-Turbo",
        prequant_repos = (
            ("int8", "unsloth/Krea-2-Turbo-FP8"),
            ("fp8", "unsloth/Krea-2-Turbo-FP8"),
        ),
        # Both are baked from the distilled Turbo denoiser: Raw quantizes its own dense weights.
        prequant_excluded_bases = ("krea/krea-2-raw",),
        # Pre-cast Qwen3-VL-4B TE (8.88 -> 4.83 GB); handed into load_krea2_pipeline directly (assembly never sees
        # pipe_kwargs).
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Krea-2-Turbo-FP8"),),
        # int8 ConvRot by default. ComfyUI loads qwen3vl_4b_fp8_scaled; our layerwise fp8 scores LPIPS-alex mean 0.110
        # against the bf16 encoder (bf16 DiT, Turbo 8 steps, 12 prompts, bf16 A/A repeat bit-identical), int8 ConvRot
        # (the Comfy-Org/Krea-2 qwen3vl_4b_int8_convrot scheme) 0.047 / SSIM 0.934. Resident 8.27 -> 4.79 GiB.
        te_quant_auto = "int8",
        aliases = ("krea2",),
        # no prequant repo yet, so nf4 quantizes on the fly
        trainable = True,
        train_base_repos = ("krea/Krea-2-Raw", "krea/Krea-2-Turbo"),
        # Adapters trained on Raw run on Turbo; deploy previews them there (same bf16 precision).
        deploy_base_repo = "krea/Krea-2-Turbo",
        fp16_incompatible = True,
    ),
    # Lumina Image 2.0: a 2.6B single-stream DiT with a Gemma2-2B encoder and a standard AutoencoderKL, so the generic
    # from_pretrained path loads it. NOT aliased to bare "lumina": Lumina-Next checkpoints are a different arch.
    DiffusionFamily(
        name = "lumina-2",
        comfy_shift_node = "sd3",
        profile = FamilyProfile(auto_bf16_when_resident = _LUMINA2_BF16_WHEN_RESIDENT),
        pipeline_class = "Lumina2Pipeline",
        transformer_class = "Lumina2Transformer2DModel",
        base_repo = "Alpha-VLLM/Lumina-Image-2.0",
        # gate-validated hosted checkpoints (28/28 pairs each; LPIPS mean 0.146 int8 / 0.116 fp8)
        prequant_repos = (
            ("int8", "unsloth/Lumina-Image-2.0-FP8"),
            ("fp8", "unsloth/Lumina-Image-2.0-FP8"),
        ),
        # pre-cast Gemma2-2B TE; the Hub stores it fp32 (10.46 GB), so the 3.20 GB artifact is a 3.3x download cut
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Lumina-Image-2.0-FP8"),),
        # No te_quant_auto: a small encoder (Gemma-2-2B) stays bf16 unless ComfyUI's template quantizes it, and ComfyUI
        # ships gemma_2_2b_fp16 only. Measured against the bf16 encoder (bf16 DiT, 50 steps, 8 prompts, bf16 A/A repeat
        # bit-identical): int8 ConvRot LPIPS-alex mean 0.015, SSIM 0.986, layerwise fp8 0.177; both stay one explicit
        # pick away (resident 4.87 -> 2.99 GiB, download 10.46 GB fp32 -> 3.21 GB).
        aliases = ("lumina-image-2.0", "lumina-image-2", "lumina2"),
        fp16_incompatible = True,
    ),
    # HunyuanImage 2.1: a 17B dual-stream DiT with a Qwen2.5-VL encoder, a ByT5 glyph encoder and the 32x HunyuanImage
    # VAE.
    # HiDream-I1: a 17B MoE DiT with FOUR text encoders. The repos ship CLIP-L/CLIP-G/T5-XXL but NOT the Llama-3.1-8B
    # text_encoder_4 their model_index names, so the loader assembles it from the open unsloth mirror. One family
    # covers Full/Dev/Fast.
    DiffusionFamily(
        name = "hunyuanimage-2.1",
        comfy_shift_node = "sd3",
        profile = FamilyProfile(
            auto_static_skip = (
                ("hunyuanvideo-community/hunyuanimage-2.1-diffusers", (("max", 2), ("steps", 50))),
            ),
        ),
        # hosted checkpoints, verified bit-identical to on-the-fly quantize (the guider pipeline is not run-to-run
        # deterministic, so same-seed LPIPS mixes trajectory divergence with harness noise)
        prequant_repos = (
            ("int8", "unsloth/HunyuanImage-2.1-FP8"),
            ("fp8", "unsloth/HunyuanImage-2.1-FP8"),
        ),
        # the Qwen2.5-VL TE is byte-identical to Qwen-Image (verified sha256), so reuse that artifact: 16.58 -> 8.84 GB.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),),
        # int8 ConvRot by default (Comfy-Org's HunyuanImage repo offers bf16 and fp8_scaled, no template): Qwen-Image's
        # shared file scores LPIPS-alex mean 0.063, SSIM 0.917 against the bf16 encoder (bf16 DiT, 50 steps, 24 prompts,
        # bf16 A/A repeat bit-identical); layerwise fp8 0.165. ByT5 stays bf16. Resident 15.45 -> 8.36 GiB.
        te_quant_auto = "int8",
        pipeline_class = "HunyuanImagePipeline",
        transformer_class = "HunyuanImageTransformer2DModel",
        base_repo = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        cfg_kwarg = "distilled_guidance_scale",
        aliases = ("hunyuanimage-2.1-diffusers", "hunyuanimage2.1"),
        fp16_incompatible = True,
        # Distilled MeanFlow files: two extra embedders, shift 4. The catch-all row marks the base as another variant.
        transformer_config_variants = (
            ("distilled", (("guidance_embeds", True), ("use_meanflow", True))),
            ("hunyuanimage", ()),
        ),
        comfy_flow_shift_variants = (("distilled", 4.0),),
    ),
    DiffusionFamily(
        name = "hidream-i1",
        comfy_shift_node = "sd3",
        profile = FamilyProfile(
            auto_bf16_when_resident = _HIDREAM_BF16_WHEN_RESIDENT,
            trusted_repos = ("hidream-ai/hidream-i1-dev", "hidream-ai/hidream-i1-fast"),
        ),
        # Hosted checkpoints: 28/28 per-case gate pairs per scheme (LPIPS 0.291 int8 / 0.278 fp8); int8 bit-identical
        # to on-the-fly quantize.
        prequant_repos = (
            ("int8", "unsloth/HiDream-I1-Full-FP8"),
            ("fp8", "unsloth/HiDream-I1-Full-FP8"),
        ),
        # Dev and Fast are distillations of Full, so the hosted Full checkpoint is baked from other
        # weights and neither has its own artifact. Excluded rather than inherited: base_model_id
        # refuses the Full artifact only after the plan has dropped their released shards.
        prequant_excluded_bases = ("hidream-ai/hidream-i1-dev", "hidream-ai/hidream-i1-fast"),
        # Pre-cast Llama-3.1-8B TE4 (16.1 -> 8.1 GB). The generic TE pass only covers text_encoder.._3, so TE4 engages
        # via hidream_te4_kwargs. text_encoder_3 is FLUX.1's T5-XXL byte for byte (sha256 compared 2026-10-16, Full and
        # Dev), so it loads FLUX.1's hosted text_encoder_2 file: 9.52 -> 5.90 GB.
        te_prequant_repos = (
            ("fp8", "text_encoder_4", "unsloth/HiDream-I1-Full-FP8"),
            ("fp8", "text_encoder_3", "unsloth/FLUX.1-schnell-FP8", "text_encoder_2"),
        ),
        # No te_quant_auto. ComfyUI loads t5xxl_fp8_e4m3fn_scaled + llama_3.1_8b_instruct_fp8_scaled; against the bf16
        # encoders (bf16 DiT, Full 50 steps CFG 5, 12 prompts, bf16 A/A repeat bit-identical) the hosted int8 ConvRot T5
        # + Llama score LPIPS-alex mean 0.103 / SSIM 0.890 and layerwise fp8 0.225 / SSIM 0.763: this DiT reads every
        # Llama layer, and neither is close enough for a silent default. Both stay one explicit pick away (CLIPs stay
        # bf16).
        pipeline_class = "HiDreamImagePipeline",
        transformer_class = "HiDreamImageTransformer2DModel",
        base_repo = "HiDream-ai/HiDream-I1-Full",
        aliases = ("hidream", "hidream-i1-full", "hidream-i1-dev", "hidream-i1-fast"),
        fp16_incompatible = True,
    ),
    # Ideogram 4: a 34-layer DiT PAIR (conditional + unconditional, ~9B each, so planning counts two) with a Qwen3-VL
    # encoder
    DiffusionFamily(
        name = "ideogram-4",
        profile = FamilyProfile(
            trusted_repos = ("ideogram-ai/ideogram-4-nf4", "ideogram-ai/ideogram-4-nf4-diffusers"),
        ),
        pipeline_class = "Ideogram4Pipeline",
        uses_negative_prompt = False,
        prompt_hint = (
            "Ideogram 4 follows long, descriptive prompts, and text in quotes renders as lettering."
        ),
        transformer_class = "Ideogram4Transformer2DModel",
        base_repo = "ideogram-ai/ideogram-4-fp8",
        # Qwen3-VL-8B, built from the vendor's per-channel-scaled fp8 shards dequantized to bf16 (what a dense load
        # runs): layerwise fp8 here, int8 ConvRot in diffusion_te_prequant.TE_INT8_CONVROT_FILES.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/ideogram-4-FP8"),),
        # No te_quant_auto. ComfyUI loads qwen3vl_8b_fp8_scaled; against the dequantized vendor encoder (bf16 DiT, 20
        # steps CFG 7, 12 prompts, bf16 A/A repeat bit-identical) int8 ConvRot scores LPIPS-alex mean 0.081 and
        # layerwise fp8 0.114, both past what a silent default should cost, although text survives and the encoder loads
        # in 2.1 s instead of 153 s (8.7 instead of 15.2 GiB). The files stay for explicit picks.
        aliases = ("ideogram4", "ideogram-v4", "ideogram"),
        # Two DiTs assembled per-component, so no transformer-only single-file / GGUF load.
        pipeline_only = True,
    ),
    # MOSS-Image: remote-code repos, so the pipeline is the in-tree port in diffusion_moss_image; encoder needs transformers >= 5.3.
    DiffusionFamily(
        name = MOSS_IMAGE_FAMILY_NAME,
        label = "MOSS-Image",
        # ComfyUI's MOSSImage is ModelType.FLOW (ModelSamplingDiscreteFlow, shift 5), the node ModelSamplingSD3 sets.
        comfy_shift_node = "sd3",
        profile = FamilyProfile(
            quant_exclude_tokens = _MOSS_IMAGE_QUANT_KEEP,
            # Required, not optional: plain int8 with the same keep-list scored 0.111 against ConvRot's 0.058 (SFT).
            int8_convrot = ConvRotSpec(
                group = 256,
                suffixes = _MOSS_IMAGE_CONVROT_SUFFIXES,
                filename = "MOSS-Image-SFT-INT8-ConvRot.safetensors",
                repo = "unsloth/MOSS-Image-SFT-FP8",
                variant_repos = ("unsloth/MOSS-Image-Base-FP8",),
                default_on = True,
            ),
            quant_names = ("moss-image-base",),
            # fp8 scored LPIPS 0.163 (int8 ConvRot 0.058): explicit picks only.
            auto_scheme_skip = frozenset({"fp8", "mxfp8"}),
            # NVFP4 (torchao W4A4, same keep-list) scored LPIPS 0.383 / max 0.546 against bf16 (12 prompts, SFT);
            # INT8-ConvRot 0.058. Same as FLUX.3: a passing NVFP4 gate record for a base still lifts it.
            scheme_deny = frozenset({"nvfp4"}),
            # A 5.08B DiT, inside the small-DiT rule: INT8-ConvRot LPIPS 0.058 vs bf16 (SFT; one-ulp floor 0.041),
            # Base 0.096. bf16 while it fits, INT8-ConvRot when bf16 would offload.
            auto_bf16_when_resident = SMALL_DIT_BF16_WHEN_RESIDENT,
            preview_factors = "flux2",
        ),
        pipeline_class = "MOSSImagePipeline",
        transformer_class = "MOSSImageTransformer",
        base_repo = "OpenMOSS-Team/MOSS-Image-SFT",
        named_variant_bases = ("OpenMOSS-Team/MOSS-Image-Base",),
        prequant_repos = (
            ("int8", "unsloth/MOSS-Image-SFT-FP8"),
            ("fp8", "unsloth/MOSS-Image-SFT-FP8"),
        ),
        prequant_filenames = (
            ("int8", "MOSS-Image-SFT-INT8.safetensors"),
            ("fp8", "MOSS-Image-SFT-FP8.safetensors"),
        ),
        prequant_variant_repos = (
            ("openmoss-team/moss-image-base", "int8", "unsloth/MOSS-Image-Base-FP8"),
            ("openmoss-team/moss-image-base", "fp8", "unsloth/MOSS-Image-Base-FP8"),
        ),
        # Other schemes on Base quantise Base's own weights, never SFT's file.
        prequant_excluded_bases = ("openmoss-team/moss-image-base",),
        # Qwen3.5-9B int8 ConvRot: LPIPS 0.059 vs bf16; without the hosted file auto keeps bf16.
        te_quant_auto = "int8",
        aliases = ("moss-image-sft", "moss-image-base", "mossimage", "moss_image"),
        prompt_hint = (
            "MOSS-Image reads plain captions, tag lists or JSON layouts; raise CFG (3 to 6) for short prompts."
        ),
        dimension_multiple = 16,
        # ComfyUI lists bf16 / fp32 only for MOSS-Image; the residual stream overflows float16.
        fp16_incompatible = True,
        # cudnn.benchmark on the fp32 VAE decode took 254 s / 73 GiB and picked a 4x slower plan: keep the heuristic.
        cudnn_benchmark = False,
        native_transformer_files_only = True,
        # MOSS's own small-decoder FLUX.2 VAE, not FLUX.2-dev's.
        sd_cpp_vae = ("unsloth/MOSS-Image-SFT-FP8", "vae/diffusion_pytorch_model.safetensors"),
        sd_cpp_text_encoders = (("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q8_0.gguf", "llm"),),
        sd_cpp_tokenizer = ("unsloth/MOSS-Image-SFT-FP8", "tokenizer/tokenizer.json"),
        # sd.cpp master-975 model_version_to_str name; its default flow shift is already 5.
        sd_cpp_arch_marker = "MOSS-Image",
    ),
    # SDXL is the one U-Net family: the denoiser is ``pipe.unet`` and a single-file ``.safetensors`` is the WHOLE
    # pipeline. img2img / inpaint / ControlNet are the standard SDXL pipelines. No GGUF path.
    DiffusionFamily(
        name = "sdxl",
        textual_inversion = "sdxl",
        label = "SDXL",
        profile = FamilyProfile(preview_factors = "sdxl"),
        # ~1.8% slower warm, but compiled-UNet renders then match across servers (ComfyUI's default)
        cudnn_benchmark = False,
        pipeline_class = "StableDiffusionXLPipeline",
        transformer_class = "UNet2DConditionModel",
        base_repo = "stabilityai/stable-diffusion-xl-base-1.0",
        aliases = ("stable-diffusion-xl", "sd-xl", "sd_xl", "sdxl-turbo", "sdxl-base"),
        denoiser_attr = "unet",
        single_file_is_pipeline = True,
        img2img_pipeline_class = "StableDiffusionXLImg2ImgPipeline",
        inpaint_pipeline_class = "StableDiffusionXLInpaintPipeline",
        controlnet_pipeline_class = "StableDiffusionXLControlNetPipeline",
        controlnet_model_class = "ControlNetModel",
        trainable = True,
        train_base_repos = (
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
        ),
    ),
)


def trainable_family_names() -> tuple[str, ...]:
    """Names of families Unsloth can train a LoRA on, in registry order."""
    return tuple(fam.name for fam in _FAMILIES if fam.trainable)


# the family whose CFG uses a guidance_scale/guidance_schedule pair, named here so the two modules cannot drift
IDEOGRAM4_FAMILY_NAME = "ideogram-4"

# the family whose generate call carries the card CFG-truncation ratio, named here so the two modules cannot drift
LUMINA2_FAMILY_NAME = "lumina-2"


# Models Unsloth deliberately does NOT support, reason surfaced verbatim in the load error, keyed by lowercase repo-id
# substring. The bar is a diffusers pipeline.
_EXCLUDED_MODELS: tuple[tuple[str, str], ...] = (
    (
        # "-3" scoped so a future HunyuanImage 2.x with a diffusers pipeline falls through normally
        "hunyuanimage-3",
        "HunyuanImage-3.0 has no diffusers pipeline (it is an 80B autoregressive MoE "
        "that requires trust_remote_code), so Unsloth does not support it.",
    ),
)


def excluded_model_reason(repo_id: str) -> Optional[str]:
    """The stated reason ``repo_id`` is unsupported, or None when it is simply unknown."""
    needle = (repo_id or "").lower()
    for token, reason in _EXCLUDED_MODELS:
        if _token_in_needle(token, needle):
            return reason
    return None


# Editing / inpaint checkpoints share an arch keyword but need a different pipeline + input image. "layered" rejects a
# layered checkpoint of any family without a layered entry of its own; Qwen-Image-Layered has one, so it matches that.
_EDIT_KEYWORDS = ("edit", "kontext", "inpaint", "layered")
# Repos named after a family that are not image models: FLUX.3's prompt upsampler (``flux-3-image-klein-pup``) is a
# Qwen3.5 LLM, not a checkpoint any pipeline loads.
_NON_IMAGE_SIBLING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "flux.3-klein": ("pup",),
    "flux.3-klein-base": ("pup",),
}


def _token_in_needle(token: str, needle: str) -> bool:
    """True when ``token`` appears in ``needle`` as a whole segment (delimited by ``- _ . / \\`` or
    a boundary), not a raw substring, so 'qwen-image-edit' matches '...-2511' but 'kontext'
    doesn't match 'kontextual'. Separator-insensitive."""
    return token_in_name(token, needle)


def _best_family_match(needle: str) -> Optional[DiffusionFamily]:
    """The family whose name/alias is the LONGEST whole-segment token of ``needle`` (longest = most
    specific, so '...qwen-image-edit-2511...' matches 'qwen-image-edit', not 'qwen-image')."""
    best: Optional[tuple[DiffusionFamily, int]] = None
    for fam in _FAMILIES:
        for token in (fam.name, *fam.aliases):
            if _token_in_needle(token, needle) and (best is None or token_length(token) > best[1]):
                best = (fam, token_length(token))
    return best[0] if best else None


def detect_family(repo_id: str, override: Optional[str] = None) -> Optional[DiffusionFamily]:
    """Resolve a ``DiffusionFamily`` from a repo id, or an explicit override. ``override`` matches a
    family ``name``/alias exactly; otherwise the most-specific family whose name/alias is a
    substring of the repo id wins. Supported editing families match here; unsupported
    editing/inpaint/layered checkpoints sharing only an arch keyword are rejected (None)."""
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        norm = normalize_family_name(key)
        for fam in _FAMILIES:
            if any(normalize_family_name(t) == norm for t in (fam.name, *fam.aliases)):
                return fam
        return None
    needle = model_identity(repo_id).lower()
    match = _best_family_match(needle)
    if match is not None:
        # Do not let a generic family (qwen-image) swallow a variant it cannot run (qwen-image-LAYERED). Scoped to the
        # LAST path component so a parent folder named `edit` does not reject a valid file.
        basename = re.split(r"[/\\]+", needle)[-1]
        matched_tokens = (match.name, *match.aliases)
        if any(
            _token_in_needle(kw, basename) and not any(kw in tok for tok in matched_tokens)
            for kw in _EDIT_KEYWORDS
        ):
            return None
        if any(
            _token_in_needle(kw, basename) for kw in _NON_IMAGE_SIBLING_KEYWORDS.get(match.name, ())
        ):
            return None
        return match
    return None


def supported_family_names() -> tuple[str, ...]:
    """Family names accepted as ``family_override`` and shown in the unknown-model error (registry
    order)."""
    return tuple(fam.name for fam in _FAMILIES)


def detect_family_by_pipeline_class(class_name: Optional[str]) -> Optional[DiffusionFamily]:
    """The family a saved pipeline's ``model_index.json`` ``_class_name`` names, or None.

    Evidence out of the checkpoint rather than out of its name, the counterpart of a GGUF's
    ``general.architecture``: an HF cache snapshot's leaf is a commit hash, so the listing had no
    name to match and hid a model the load path accepts (#8407).

    Only the BASE class matches. The loader instantiates ``fam.pipeline_class``
    (``diffusion.py:2946``), never the declared class, so tagging an inpaint or img2img checkpoint
    would list it and then load it through the wrong pipeline (its UNet input shape differs), which
    is the listing-versus-loader split this exists to close. A variant stays untagged, the answer it
    got before the index was read at all.
    """
    key = (class_name or "").strip()
    if not key:
        return None
    for fam in _FAMILIES:
        if fam.pipeline_class and fam.pipeline_class == key:
            return fam
    return None


def pipeline_class_from_index(path: Optional[str]) -> Optional[str]:
    """The ``_class_name`` the diffusers pipeline saved at ``path`` declares, or None.

    Size-capped and schema-free: neither a listing nor a load may be held up by whatever a scan
    folder contains. ``_class_name`` is a LIST for a remote-code community pipeline, which Unsloth
    cannot load, so only a plain string answers.

    ``utf-8-sig`` because PowerShell writes JSON with a BOM and a hand-authored index is ordinary
    beside a converted checkpoint; read as ``utf-8`` it raises and the model stays hidden, #8407
    again. ``RecursionError`` (a nesting bomb, not a ``ValueError``) is caught too: both callers
    wrap this in a blanket except that reads a raise as detection having succeeded.
    """
    root = Path(path or "")
    if not str(root):
        return None
    for name in ("model_index.json", "modular_model_index.json"):
        try:
            index = root / name
            if not index.is_file() or index.stat().st_size > 1_000_000:
                continue
            payload = json.loads(index.read_text(encoding = "utf-8-sig"))
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(payload, dict):
            value = payload.get("_class_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def detect_family_by_pipeline_index(path: Optional[str]) -> Optional[DiffusionFamily]:
    """The family the pipeline saved at ``path`` declares in its ``model_index.json``, or None. The
    path-shaped counterpart of ``detect_family_by_pipeline_class``, used by BOTH the listing and
    the loader so the picker and ``validate_load_request`` answer off the same evidence (#8407).
    Carries over ``detect_family``'s variant guard: a directory NAMED for a checkpoint the
    matched family cannot run (``...-layered``) is still refused, so the index only adds models
    whose name said nothing, never overrides a name that said no."""
    fam = detect_family_by_pipeline_class(pipeline_class_from_index(path))
    if fam is None or _index_family_ruled_out(fam, path):
        return None
    return fam


def _index_family_ruled_out(fam: DiffusionFamily, path: Optional[str]) -> bool:
    """True when the directory's NAME carries a variant keyword the index's family cannot run."""
    basename = re.split(r"[/\\]+", str(path).lower())[-1]
    matched_tokens = (fam.name, *fam.aliases)
    return any(
        _token_in_needle(kw, basename) and not any(kw in tok for tok in matched_tokens)
        for kw in _EDIT_KEYWORDS
    )


def pipeline_index_contradicts_name(path: Optional[str]) -> bool:
    """True when a local pipeline's ``model_index.json`` declares a family its directory name rules out (a
    ``qwen-image-layered`` folder holding a plain ``QwenImagePipeline``). The name-based fallback must not answer
    for such a directory: the name would pick the variant's family and the loader would then build that pipeline
    over a checkpoint saved as another. The listing and the loader both refuse it."""
    fam = detect_family_by_pipeline_class(pipeline_class_from_index(path))
    return fam is not None and _index_family_ruled_out(fam, path)


def detect_family_for_pick(
    repo_id: str,
    gguf_filename: Optional[str] = None,
    override: Optional[str] = None,
) -> Optional[DiffusionFamily]:
    """``detect_family``, falling back to the combined path/filename for a local ``.gguf`` pick where
    the family keyword lives only in the filename, and then to the saved pipeline class of a local
    diffusers pipeline directory. Only fallbacks, so remote picks and overrides behave exactly as
    ``detect_family``. Shared by both engines.

    The index keeps the listing and the loader on one answer: the listing classifies a moved
    pipeline from its ``model_index.json`` (its directory name is a commit hash), and the pick sent
    back is that same opaque path with no family_override, so without this the model is shown as
    text-to-image and then refused as an unsupported family (#8407).
    """
    fam = None
    if not override:
        # The checkpoint's own declaration outranks any guess made from its path: a family keyword in ANY ancestor
        # segment otherwise shadows the index (a QwenImagePipeline under `.../flux.1/checkpoint` matched FLUX and
        # never reached it), so the listing, which reads the index, named one family and the loader another for one
        # directory. Remote picks are unaffected: with no local index this is None and the name-based paths below run
        # as before.
        fam = detect_family_by_pipeline_index(repo_id)
        if fam is None and pipeline_index_contradicts_name(repo_id):
            return None
    if fam is None:
        fam = detect_family(repo_id, override)
    if fam is None and gguf_filename and not override:
        fam = detect_family(f"{repo_id}/{gguf_filename}", override)
    if not override:
        fam = _family_from_content(fam, repo_id, gguf_filename)
    return fam


def _family_from_content(
    fam: Optional[DiffusionFamily], repo_id: str, gguf_filename: Optional[str]
) -> Optional[DiffusionFamily]:
    """Reconcile the name verdict with a LOCAL file's header: non-DiT / video DiT -> None, renamed DiT
    -> its header family; the name still picks same-architecture variants. Remote picks untouched."""
    from .diffusion_content import local_pick_file, resolve_family_with_content

    path = local_pick_file(repo_id, gguf_filename)
    if not path:
        return fam
    needles = [repo_id] + ([f"{repo_id}/{gguf_filename}"] if gguf_filename else [])
    vetoed = fam is None and any(
        _best_family_match(model_identity(n).lower()) is not None for n in needles
    )
    name, _ = resolve_family_with_content(fam.name if fam else None, path, "image", vetoed)
    if name is None:
        return None
    if fam is not None and fam.name == name:
        return fam
    return detect_family("", override = name)


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


# Byte-identical unsloth mirrors of the vendor bases: a GGUF/FP8 pick ships only the denoiser, so companions come from
# the base, which is a 401 for gated vendors and a third-party fetch for the rest. Swapped at the fetch sites only,
# never in ``resolve_base_repo``, whose result keys the UPSTREAM-id tables below (see ``canonical_base``). A mirror
# stands in for the WHOLE base, bf16 pipeline loads included, so it must be a complete copy: a companions-only repo
# breaks every pick that needs the transformer.
_GATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("black-forest-labs/FLUX.1-dev", "unsloth/FLUX.1-dev"),
    ("black-forest-labs/FLUX.1-schnell", "unsloth/FLUX.1-schnell"),
    ("black-forest-labs/FLUX.1-Kontext-dev", "unsloth/FLUX.1-Kontext-dev"),
    ("black-forest-labs/FLUX.1-Krea-dev", "unsloth/FLUX.1-Krea-dev"),
    ("black-forest-labs/FLUX.2-dev", "unsloth/FLUX.2-dev"),
    ("black-forest-labs/FLUX.2-klein-9B", "unsloth/FLUX.2-klein-9B"),
    ("black-forest-labs/FLUX.2-klein-base-9B", "unsloth/FLUX.2-klein-base-9B"),
    # FLUX.3 keeps the upstream diffusers folders (modular manifest, text_encoder, transformer, vae; byte-identical)
    # inside its FP8 repo beside the pre-quantized files, so no separate mirror repo is needed.
    ("black-forest-labs/flux-3-image-klein", "unsloth/flux-3-image-klein-FP8"),
    ("black-forest-labs/flux-3-image-klein-base", "unsloth/flux-3-image-klein-base-FP8"),
    ("krea/Krea-2-Turbo", "unsloth/Krea-2-Turbo"),
    ("krea/Krea-2-Raw", "unsloth/Krea-2-Raw"),
    ("ideogram-ai/ideogram-4-fp8", "unsloth/ideogram-4-fp8"),
    ("ideogram-ai/ideogram-4-nf4", "unsloth/ideogram-4-nf4"),
    ("ideogram-ai/ideogram-4-nf4-diffusers", "unsloth/ideogram-4-nf4-diffusers"),
)

# Mirrored to drop the third-party fetch, NOT to route around a gate. Every licence here permits redistribution, and
# each mirror carries the upstream licence text plus the notice that licence prescribes. Qwen-Image-2512 is the one no
# other redirect could reach: its companions are named by the artifact repo's base_model card tag, not the family
# table. Kept apart from the gated pairs because the two answer different questions: both redirect a fetch, but only a
# GATED upstream justifies overriding a user's existing cache, and for these the upstream is reachable without
# credentials, so a complete local snapshot must keep being used rather than re-pulled from the mirror.
_UNGATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Qwen/Qwen-Image-2512", "unsloth/Qwen-Image-2512"),
    ("Qwen/Qwen-Image-2.1", "unsloth/Qwen-Image-2.1"),
    ("Qwen/Qwen-Image", "unsloth/Qwen-Image"),
    ("Qwen/Qwen-Image-Edit-2511", "unsloth/Qwen-Image-Edit-2511"),
    # Variant bases reached only through a GGUF's base_model card tag (Edit, Edit-2509, Z-Image) or a named variant
    # (2.1 Turbo, also a direct bf16 pick), plus Layered, whose sd-cli VAE is read straight from its base. No separate
    # mirror repo: the official diffusers folders (model_index.json, transformer/, text_encoder/, tokenizer/,
    # processor/, scheduler/, vae/) sit byte-identical at their upstream paths inside the model's EXISTING unsloth
    # repo, its -FP8 repo where one exists, else its -GGUF repo, beside the quantised files that repo already ships.
    ("Qwen/Qwen-Image-2.1-Turbo", "unsloth/Qwen-Image-2.1-Turbo-FP8"),
    ("Qwen/Qwen-Image-Edit", "unsloth/Qwen-Image-Edit-GGUF"),
    ("Qwen/Qwen-Image-Edit-2509", "unsloth/Qwen-Image-Edit-2509-GGUF"),
    ("Qwen/Qwen-Image-Layered", "unsloth/Qwen-Image-Layered-GGUF"),
    # MOSS-Image, as FLUX.3: upstream diffusers folders are mirrored byte-identical into each variant's -FP8 repo.
    ("OpenMOSS-Team/MOSS-Image-SFT", "unsloth/MOSS-Image-SFT-FP8"),
    ("OpenMOSS-Team/MOSS-Image-Base", "unsloth/MOSS-Image-Base-FP8"),
    ("black-forest-labs/FLUX.2-klein-4B", "unsloth/FLUX.2-klein-4B"),
    # Lookup is by exact id, so every VARIANT a pick can resolve to needs its own row: the base-4B is what
    # `unsloth/FLUX.2-klein-base-4B-GGUF` resolves to from its card tag, and Dev / Fast are offered directly as bf16
    # pipeline picks. Without these three the table silently misses them.
    ("black-forest-labs/FLUX.2-klein-base-4B", "unsloth/FLUX.2-klein-base-4B"),
    ("Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo"),
    ("Tongyi-MAI/Z-Image", "unsloth/Z-Image-GGUF"),
    ("Alpha-VLLM/Lumina-Image-2.0", "unsloth/Lumina-Image-2.0"),
    ("HiDream-ai/HiDream-I1-Full", "unsloth/HiDream-I1-Full"),
    ("HiDream-ai/HiDream-I1-Dev", "unsloth/HiDream-I1-Dev"),
    ("HiDream-ai/HiDream-I1-Fast", "unsloth/HiDream-I1-Fast"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "unsloth/stable-diffusion-xl-base-1.0"),
    ("stabilityai/sdxl-turbo", "unsloth/sdxl-turbo"),
)
_MIRROR_PAIRS: tuple[tuple[str, str], ...] = _GATED_MIRROR_PAIRS + _UNGATED_MIRROR_PAIRS

# The VIDEO bases (video_families). No separate mirror repo: the official files Studio loads (the diffusers folders,
# bf16 transformer included, or the single files of a split release) are copied byte-identical, at their upstream
# paths, into the EXISTING unsloth repo of that model -- its -FP8 repo, or -GGUF when there is no -FP8 -- beside the
# upstream LICENSE and the NOTICE the licence asks for. The listing selectors skip that repo's own root checkpoints
# (.pt, root .safetensors, .gguf, samples/), so it serves as the base. MiniMax-H3 already has a complete copy.
# Kept OUT of ``_MIRROR_PAIRS``: that tuple is the IMAGE companion table (hub/utils/companion_assets reads it to decide
# which cached repos an image checkpoint can strand), while these feed only the id lookups below, so mirror_repo /
# canonical_base / prefer_ungated_mirror answer for video ids and nothing image-side changes.
# Absent on purpose: HunyuanVideo-1.5, whose Tencent Hunyuan Community licence is limited to a Territory that excludes
# the EU, UK and South Korea, which a public Hub repo cannot honour (the reason HunyuanImage-2.1 has no mirror either);
# and Wan2.1-T2V-1.3B / -14B, for which no unsloth repo exists to carry the files.
_VIDEO_GATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Lightricks/LTX-2.5-Diffusers", "unsloth/LTX-2.5-FP8"),
    # The split single-file release (text_encoders/, vae/, latent_upscale_models/, diffusion_models/ bf16 DiTs). In the
    # GGUF repo because the FP8 repo already carries the Diffusers pack, and one repo cannot stand for two upstreams.
    ("Lightricks/LTX-2.5", "unsloth/LTX-2.5-GGUF"),
)
_VIDEO_UNGATED_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "unsloth/Wan2.2-TI2V-5B-FP8"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "unsloth/Wan2.2-T2V-A14B-FP8"),
    # LTX-2 is the base of the ltx-2 family AND the source of LTX-2.3's scheduler, Gemma3 text encoder and tokenizer.
    ("Lightricks/LTX-2", "unsloth/LTX-2-FP8"),
    # LTX-2.3 ships single files only (the bf16 dev / distilled / distilled-1.1 DiTs a pick names).
    ("Lightricks/LTX-2.3", "unsloth/LTX-2.3-FP8"),
    ("MiniMaxAI/MiniMax-H3", "unsloth/MiniMax-H3"),
)
_VIDEO_MIRROR_PAIRS: tuple[tuple[str, str], ...] = (
    _VIDEO_GATED_MIRROR_PAIRS + _VIDEO_UNGATED_MIRROR_PAIRS
)

# Mirror targets that are ALSO a model's own artifact repo (its GGUF / prequant checkpoints sit at the root beside the
# official diffusers folders). A user picks these ids directly, so they are never "just a copy of the upstream": no
# opt-out may rewrite them to the upstream (which has no such checkpoint), a cached one is its own checkpoint rather
# than a stand-in for the base, and only the folder part of the snapshot is the pipeline.
# Listed by hand, never inferred from a -GGUF / -FP8 suffix: unsloth/ideogram-4-fp8 is an ordinary complete copy of
# ideogram-ai/ideogram-4-fp8 (the FP8 is in the upstream name), so it must keep canonicalising like every full mirror.
_ARTIFACT_MIRRORS: frozenset[str] = frozenset(
    {
        "unsloth/z-image-gguf",
        "unsloth/qwen-image-2.1-turbo-fp8",
        "unsloth/qwen-image-edit-gguf",
        "unsloth/qwen-image-edit-2509-gguf",
        "unsloth/qwen-image-layered-gguf",
        "unsloth/moss-image-sft-fp8",
        "unsloth/moss-image-base-fp8",
    }
)
_ALL_MIRROR_PAIRS: tuple[tuple[str, str], ...] = _MIRROR_PAIRS + _VIDEO_MIRROR_PAIRS
_GATED_MIRRORS: dict[str, str] = {u.lower(): m for u, m in _ALL_MIRROR_PAIRS}
_MIRROR_UPSTREAM: dict[str, str] = {m.lower(): u for u, m in _ALL_MIRROR_PAIRS}
_GATED_UPSTREAMS: frozenset[str] = frozenset(
    u.lower() for u, _m in _GATED_MIRROR_PAIRS + _VIDEO_GATED_MIRROR_PAIRS
)

# Retired full copies of a gated base: ``unsloth/flux-3-image-klein[-base]`` hold the whole BFL repo (single files and
# diffusers folders). They stay up for existing links, but Studio's copy of those files now lives in the -FP8 repo
# (``_GATED_MIRROR_PAIRS``), so a load that names one is the same load as a pick of the base it copies.
_RETIRED_FULL_COPIES: dict[str, str] = {
    "unsloth/flux-3-image-klein": "black-forest-labs/flux-3-image-klein",
    "unsloth/flux-3-image-klein-base": "black-forest-labs/flux-3-image-klein-base",
}


_RETIRED_COPY_SNAPSHOT = re.compile(
    r"(?:^|[\\/])models--unsloth--(flux-3-image-klein(?:-base)?)[\\/]snapshots[\\/][^\\/]+[\\/]?$",
    re.IGNORECASE,
)


def retired_full_copy_base(repo_id: Optional[str]) -> Optional[str]:
    """The base a retired full copy stands for (Studio loads it through that base's mirror), else None. Its id, or an
    HF-cache snapshot of it (``.../models--unsloth--flux-3-image-klein/snapshots/<sha>``)."""
    key = (repo_id or "").strip()
    base = _RETIRED_FULL_COPIES.get(key.lower())
    if base is None:
        snapshot = _RETIRED_COPY_SNAPSHOT.search(key)
        base = (
            _RETIRED_FULL_COPIES.get(f"unsloth/{snapshot.group(1).lower()}") if snapshot else None
        )
    return base


# Repo id (lowercased) -> the commit a load's plan pinned it to, for the duration of that load's assembly. Set only by
# ``pinned_repo_revision`` around a video load; every hub-id CONFIG read of a base repo (encoder / VAE / transformer
# configs, model_index, the LTX-2.5 companions) asks ``pinned_revision_kwargs`` so it reads the planned commit. Empty
# outside such a block, so unpinned callers (image loads, direct calls) pass no revision and behave as before.
_PINNED_REVISIONS: "contextvars.ContextVar[Optional[dict[str, str]]]" = contextvars.ContextVar(
    "unsloth_pinned_repo_revisions", default = None
)


def _pins() -> "contextvars.ContextVar[Optional[dict[str, str]]]":
    return _PINNED_REVISIONS


class pinned_repo_revision:
    """``with pinned_repo_revision(repo, revision):`` pins hub-id reads of ``repo`` to ``revision`` in this context.
    A no-op when either is missing or ``repo`` is a local path."""

    def __init__(self, repo: Optional[str], revision: Optional[str]):
        self._key = (
            (repo or "").strip().lower() if repo and revision and not _is_local_path(repo) else ""
        )
        self._revision = revision
        self._token = None

    def __enter__(self) -> "pinned_repo_revision":
        if self._key:
            var = _pins()
            self._token = var.set({**(var.get() or {}), self._key: str(self._revision)})
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._token is not None:
            _pins().reset(self._token)
            self._token = None


def pinned_revision_kwargs(repo: Any) -> dict[str, str]:
    """``{"revision": <commit>}`` when ``repo`` is pinned in this context, else ``{}``."""
    if not isinstance(repo, str) or not repo:
        return {}
    revision = (_pins().get() or {}).get(repo.strip().lower())
    return {"revision": revision} if revision else {}


def mirror_repo(repo_id: Optional[str]) -> Optional[str]:
    """The unsloth mirror of ``repo_id``, or None when it is not a mirrored vendor base."""
    return _GATED_MIRRORS.get((repo_id or "").strip().lower())


def mirror_hosts_artifacts(repo_id: Optional[str]) -> bool:
    """True when ``repo_id`` is a mirror target that also ships the model's own GGUF / prequant checkpoints."""
    return (repo_id or "").strip().lower() in _ARTIFACT_MIRRORS


def upstream_is_gated(repo_id: Optional[str]) -> bool:
    """True when ``repo_id`` is a vendor base the Hub refuses without accepted terms. Distinct from
    "has a mirror": most of the mirror table is ungated and exists only to keep the fetch inside
    ``unsloth/*``. Only the gated half justifies overriding a user's cache."""
    return (repo_id or "").strip().lower() in _GATED_UPSTREAMS


def named_variant_base(fam: "DiffusionFamily", *names: Optional[str]) -> Optional[str]:
    """The longest ``fam.named_variant_bases`` entry ``names`` spell (separators folded), or None. The last, most
    specific name decides first; one spelling the plain base stops the search (a 2.1 file in a Turbo repo stays 2.1)."""

    def fold(text: Optional[str]) -> str:
        return "".join(c for c in (text or "").lower() if c.isalnum())

    variants = [
        (b, fold(b.rsplit("/", 1)[-1])) for b in getattr(fam, "named_variant_bases", ()) or ()
    ]
    plain = fold((getattr(fam, "base_repo", "") or "").rsplit("/", 1)[-1])
    for name in reversed([n for n in names if n]):
        identity = fold(name)
        hits = [b for b, key in variants if key and key in identity]
        if hits:
            return max(hits, key = len)
        if plain and plain in identity:
            return None
    return None


def canonical_base(repo_id: Optional[str]) -> str:
    """A mirror id mapped back to the upstream it copies, else ``repo_id`` unchanged. Base-keyed
    tables hold UPSTREAM ids, so every lookup normalises here: a mirror reaching
    ``_FLUX2_BASE_INNER_DIM`` misses, and that shape guard fails OPEN, so it goes silent."""
    base = (repo_id or "").strip()
    return _MIRROR_UPSTREAM.get(base.lower(), base)


# what counts as a real weight file, vs the stray config an interrupted pull leaves behind
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".ckpt", ".gguf"})


def _cached_revisions(root: Path, repo_id: str) -> list[Path]:
    """The snapshot dirs a fetch pinned to ``root`` could resolve for ``repo_id``. Never raises.
    ``refs/main`` alone when it exists, the one commit a branch fetch falls back to on a failed
    HEAD; every snapshot when it does not, since a commit-pinned download leaves no ref."""
    try:
        repo = root / f"models--{repo_id.replace('/', '--')}"
        ref = repo / "refs" / "main"
        revs = (
            [repo / "snapshots" / ref.read_text(encoding = "utf-8").strip()]
            if ref.is_file()
            else sorted((repo / "snapshots").iterdir())
        )
        return [rev for rev in revs if rev.is_dir()]
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return []


def _root_revision_coverage(root: Path, repo_id: str, wanted: Sequence[str]) -> set[frozenset[str]]:
    """Which subsets of ``wanted`` ONE root can serve, one entry per revision it could resolve. A
    fetch that lands in a root lands in a SINGLE revision of it, so a name may not be borrowed
    from a superseded snapshot to complete a newer one. The empty set is always an option: a root
    is allowed to contribute nothing."""
    covers: set[frozenset[str]] = {frozenset()}
    for rev in _cached_revisions(root, repo_id):
        covers.add(frozenset(name for name in wanted if (rev / name).exists()))
    return covers


def _root_holds_upstream(root: Path, repo_id: str, wanted: Sequence[str]) -> bool:
    """``_upstream_is_cached`` for ONE cache root. Never raises."""
    try:
        for rev in _cached_revisions(root, repo_id):
            if wanted:
                if all((rev / name).exists() for name in wanted):
                    return True
            elif any(
                p.suffix.lower() in _WEIGHT_SUFFIXES
                and p.is_file()
                and not is_appledouble_metadata(p)
                for p in rev.rglob("*")
            ):
                return True
        return False
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return False


def _upstream_is_cached(
    repo_id: str,
    files: Optional[Sequence[str]] = None,
    *,
    other_root: bool = False,
) -> bool:
    """Whether the upstream load is SATISFIABLE from the local cache.

    Not "has any blob": one config left by an interrupted or previously-tokened pull would pin every
    later load to the gated upstream and re-raise the 401 the mirror exists to avoid. So the
    revision must hold ``files``, else a real weight file. ``.incomplete`` downloads have no
    snapshot symlink, so they count as absent.

    Only the revision ``refs/main`` names counts, the one a gated fetch falls back to: on a 401 the
    HEAD fails and ``hf_hub_download`` resolves the ref to ONE commit, so a complete but superseded
    revision would read as cached and hand the loader a repo it cannot fetch from. With no ref (a
    commit-pinned download) any revision counts, as before.

    Reads the LIVE cache root; huggingface_hub's import-time constant goes stale after a
    cache-folder change. ``other_root`` adds that constant back, for the callers whose fetch passes
    ``reuse_other_cache_root``: those resolve each file through whichever root holds it, so bytes
    left in the pre-change root really do satisfy the load. OFF by default, because a
    ``from_pretrained`` is pinned to the live root and cannot see the other one -- counting those
    bytes there would send a gated base back to the 401 the mirror exists to avoid.

    With BOTH roots the set is answered per FILE, because that is what those callers then do: a pair
    split across a cache-folder change (one file fetched before it, one after) is held by neither
    root alone, and asking each root for the whole set calls it absent and re-pulls bytes the two
    roots already have between them. Split across ROOTS only: within a root a name still has to come
    from the one revision that root's fetch resolves, so a superseded snapshot cannot complete a
    newer one. Any single root answering the whole set is the case above unchanged.
    """
    try:
        from utils.hf_cache_settings import active_hf_hub_cache

        roots = [Path(active_hf_hub_cache())]
        if other_root:
            from huggingface_hub import constants

            # Exactly what ``cache_dir = None`` resolves to, i.e. the root the per-file reuse probe falls back to
            # (file_download reads constants.HF_HUB_CACHE per call).
            fallback = Path(constants.HF_HUB_CACHE)
            if fallback != roots[0]:
                roots.append(fallback)
        wanted = tuple(files or ())
        if wanted and len(roots) > 1:
            # One revision per root, unioned across roots: a name may come from either root, but within a root it has
            # to come from the revision that root's fetch would resolve.
            reachable = {frozenset()}
            for root in roots:
                covers = _root_revision_coverage(root, repo_id, wanted)
                reachable = {have | cover for have in reachable for cover in covers}
            return any(set(wanted) <= have for have in reachable)
        return any(_root_holds_upstream(root, repo_id, wanted) for root in roots)
    except Exception:  # noqa: BLE001 -- an unreadable/absent cache just means "not cached"
        return False


def cache_holds_files(repo_id: str, files: Sequence[str]) -> bool:
    """Whether ``repo_id``'s local cache holds EVERY name in ``files``.

    The same revision rule ``_upstream_is_cached`` applies, exposed for callers that need to know a
    component is complete rather than merely started: a partial pull leaves some shards resident,
    and "some" is not a cache hit for anything that then decides not to download the rest.

    The LIVE root only. It is tempting to count the import-time root as well, since
    ``_prefetch_files`` passes ``reuse_other_cache_root`` and would not re-fetch from it, but the
    prefetch is not the consumer that matters here: the dense fast path this verdict unlocks calls
    ``from_pretrained(cache_dir = hub_cache_dir())``, which is pinned to the live root and cannot
    see the other one. A hit there would widen the plan and then download the whole transformer
    again after eviction, which is the exact outcome the check exists to prevent.
    """
    return bool(files) and _upstream_is_cached(repo_id, tuple(files))


# Lowercased unsloth mirror -> the community repack the tables named before it. The mirrors are byte identical, so an
# existing install that already pulled the repack holds the very same bytes under the OLD repo key: the HF cache is
# keyed by repo id, so re-pointing the table alone would re-download up to tens of GB on upgrade and fail outright
# offline. Empty now: every image sd-cli asset is an official upstream file or an unsloth GGUF, and a video load never
# reads from a Comfy or third-party repack (the MiniMax-H3 and Wan2.2-TI2V-5B-GGUF rows are gone), so no load reads a
# repack, not even bytes already cached under one. Kept as the hook ``prefer_cached_legacy_source`` reads.
_SD_CPP_LEGACY_SOURCES: dict[str, str] = {}


def legacy_source_repo(repo_id: Optional[str]) -> Optional[str]:
    """The community repack ``repo_id`` mirrors, or None when it is not one of the mirrors."""
    return _SD_CPP_LEGACY_SOURCES.get((repo_id or "").strip().lower())


def prefer_cached_legacy_source(repo_id: str, files: Optional[Sequence[str]] = None) -> str:
    """``repo_id``, or the community repack it mirrors when THAT already satisfies ``files``.

    The mirror stays the preferred source for a fresh install; this only spares an existing one from
    re-fetching bytes it already has under the old repo key. Same ``_upstream_is_cached`` probe the
    gated swap uses, so an interrupted or partial repack does not win.

    Both cache roots count: the sd.cpp fetch passes ``reuse_other_cache_root``, so a repack left
    behind by a cache-folder change is still reusable, and only the repo id can reach it -- once the
    id has become the mirror those bytes are unreachable and the load re-pulls several GB (offline,
    it fails outright).

    PURE: table lookup + local stat, no network, so the staging plan and the fetch agree.
    """
    legacy = _SD_CPP_LEGACY_SOURCES.get((repo_id or "").strip().lower())
    if not legacy:
        return repo_id
    return legacy if _upstream_is_cached(legacy, files, other_root = True) else repo_id


def _is_local_path(base: str) -> bool:
    """Whether ``base`` exists on disk, i.e. a local dir rather than a Hub id. A user can clone a
    base into a relative dir named exactly like the vendor id (``black-forest-labs/FLUX.1-dev``),
    which the loaders deliberately treat as local. OSError from an id with invalid path
    characters just means "not a local path"."""
    try:
        return Path(base or "").expanduser().exists()
    except OSError:
        return False


def prefer_ungated_mirror(
    base: str,
    hf_token: Optional[str] = None,
    *,
    files: Optional[Sequence[str]] = None,
    other_root: bool = False,
) -> str:
    """``base``, or its ungated unsloth mirror when that is the better repo to FETCH from.

    A GGUF/FP8 pick carries only the denoiser, so a gated base 401s on the companions; the mirrors
    are byte identical, so this drops the gate without changing a weight. Fetch only: the upstream
    id stays what the picker and status() show, what saved configs hold and what a trained LoRA's
    base_model tag records. The one visible swap is the download manager row, which must name the
    repo actually being pulled -- staging the gated id there is the 401 this exists to remove.

    Declines to today's behaviour under ``UNSLOTH_DIFFUSION_NO_MIRROR``, for a local path, or when
    the upstream already satisfies the load from cache and switching would re-pull tens of GiB.
    Under that opt-out a mirror id picked directly maps back to its upstream, cached or not: even
    a cached mirror is listed on the Hub before it loads.
    ``files`` sharpens that last test to the names about to be fetched; without it any weight
    counts.

    PURE: table lookup, env read, local stat, NO network. This runs inside pipeline assembly, where
    an earlier ``model_info`` probe of the mirror put a Hub round trip on the load path and broke
    four download-plan tests. A missing mirror surfaces as the ordinary download error. ``hf_token``
    is unused, kept so callers need not care.
    """
    del hf_token  # noqa: F841 -- signature stability only
    if os.environ.get("UNSLOTH_DIFFUSION_NO_MIRROR", "").strip():
        # An artifact repo stays itself: its GGUF / FP8 checkpoint does not exist under the upstream id.
        if _is_local_path(base) or mirror_hosts_artifacts(base):
            return base
        return canonical_base(base)
    mirror = mirror_repo(base)
    if not mirror:
        return base
    # a local path is never a Hub id: rewriting one sends loads the other sites resolve on disk to the Hub, skipping
    # the copy already downloaded
    if _is_local_path(base):
        return base
    # ``other_root`` for a fetch that resolves each file through either cache root (``reuse_other_cache_root``):
    # an upstream copy left in the pre-change root then serves the load, so it must keep the upstream id.
    cached = (
        _upstream_is_cached(base, files, other_root = True)
        if other_root
        else _upstream_is_cached(base, files)
    )
    return base if cached else mirror


# Default (steps, guidance) per model for callers that cannot pass them. Matched by substring, most specific first;
# same values as the UI MODEL_DEFAULTS table, keep in sync.
_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    # Values follow ComfyUI's official templates for the same model (our baseline).
    ("z-image-turbo", 8, 0.0),
    # FLUX.1 Krea dev is a FLUX.1-dev finetune, NOT a Krea-2; must precede the generic "krea" key.
    ("flux.1-krea", 20, 3.5),
    # Krea 2 Raw (undistilled): 52 steps / guidance 3.5. Must precede the generic "krea" key.
    ("krea-2-raw", 52, 3.5),
    # Krea 2 Turbo (distilled): 8 steps, no CFG. "krea" then covers Turbo and other krea ids but Raw.
    ("flux1-krea", 20, 3.5),  # before Krea-2's generic row
    ("krea", 8, 0.0),
    ("flux.1-schnell", 4, 0.0),
    ("flux1-schnell", 4, 0.0),
    ("kontext", 20, 2.5),  # editing: before the generic flux.1
    ("flux.1", 20, 3.5),
    ("flux1", 20, 3.5),
    # FLUX.3 Image Klein, ComfyUI's templates (Comfy-Org/workflow_templates 0.11.87): the base t2i and edit templates
    # run 50 steps at CFG 3.5 (euler / simple), raised from the launch templates' 25, which is also the BFL reference
    # setting (diffusers' FLUX.3 pipeline defaults: 50 steps, guidance 3.5). Both distilled templates run 16 steps at
    # CFG 1, as does the BFL reference. Base first.
    # Every registered alias plus the canonical family names (an opaque path resolves through the family).
    ("flux.3-klein-base", 50, 3.5),
    ("flux-3-image-klein-base", 50, 3.5),
    ("flux3-image-klein-base", 50, 3.5),
    ("flux3-klein-base", 50, 3.5),
    ("flux.3-klein", 16, 1.0),
    ("flux-3-image-klein", 16, 1.0),
    ("flux3-image-klein", 16, 1.0),
    ("flux3-klein", 16, 1.0),
    # Undistilled base runs real CFG; keep before the generic distilled key.
    ("flux.2-klein-base", 20, 5.0),
    ("flux.2-klein", 4, 1.0),
    ("flux.2-dev", 20, 4.0),  # full (non-distilled)
    # Before the generic qwen-image key, Turbo before 2.1; qwenimage21 has no separator to fold, hence its own row.
    ("qwen-image-2.1-turbo", 8, 1.0),
    ("qwen-image-21-turbo", 8, 1.0),
    ("qwenimage21-turbo", 8, 1.0),
    ("qwenimage21turbo", 8, 1.0),
    ("qwen-image-2.1", 25, 1.0),
    ("qwen-image-21", 25, 1.0),
    ("qwen_image_21", 25, 1.0),
    ("qwenimage21", 25, 1.0),
    # ComfyUI's Image to Layers template: 20 steps, cfg 2.5.
    ("qwen-image-layered", 20, 2.5),
    ("qwen_image_layered", 20, 2.5),
    ("qwenimagelayered", 20, 2.5),
    # 2509 template: 20 / 4; the generic key is the 2511 recipe (the family base).
    ("qwen-image-edit-2509", 20, 4.0),
    ("qwen-image-edit", 40, 4.0),
    ("qwen-image-2512", 50, 4.0),
    ("qwen-image", 20, 4.0),
    # diffusers Z-Image g = ComfyUI cfg - 1 (pos + g * (pos - neg)).
    ("z-image", 25, 3.0),
    # Lumina Image 2.0 card: 50 steps, guidance 4 (plus cfg_trunc_ratio 0.25, which the loader passes itself).
    ("lumina", 50, 4.0),
    # HunyuanImage 2.1 card: 50 steps; guidance feeds distilled_guidance_scale, while real CFG runs inside the
    # guiders.
    ("hunyuanimage", 50, 3.25),
    # HiDream-I1 upstream: Full 50 steps / guidance 5; the distilled Dev (28) and Fast (16) run guidance-free.
    ("hidream-i1-dev", 28, 0.0),
    ("hidream-i1-fast", 16, 0.0),
    ("hidream", 50, 5.0),
    # Ideogram 4: an explicit 48 steps at 7 still runs the card's tapered schedule.
    ("ideogram", 20, 7.0),
    # MOSS-Image: ComfyUI workflow, 50 steps at CFG 3.
    ("moss-image", 50, 3.0),
    ("moss_image", 50, 3.0),
    ("mossimage", 50, 3.0),
    # SDXL: Turbo distilled; base wants ~30 steps + CFG ~7. "sdxl-turbo" precedes "sdxl".
    ("sdxl-turbo", 3, 0.0),
    ("stable-diffusion-xl", 25, 7.0),
    ("sdxl", 25, 7.0),
)
# Unrecognised model: distilled few-step / no-CFG shape, matching the UI fallback.
_GENERATION_DEFAULT_FALLBACK = (9, 0.0)


def _first_variant(rows: Any, identifiers: tuple[Optional[str], ...]) -> Optional[tuple]:
    """First ``(key, ...)`` row whose key is in an identifier; ``_`` reads as ``-`` (ComfyUI file names)."""
    for identifier in identifiers:
        needle = (identifier or "").lower().replace("_", "-")
        for row in rows or ():
            if row[0] in needle:
                return row
    return None


def family_for_checkpoint(fam: Any, *identifiers: Optional[str]) -> Any:
    """``fam`` with the condition-image limit of the checkpoint ``identifiers`` name (``max_condition_images_variants``),
    so status, admission and both engines read the loaded checkpoint's limit off the family they hold."""
    row = _first_variant(getattr(fam, "max_condition_images_variants", ()), identifiers)
    if row is None or row[1] == getattr(fam, "max_condition_images", None):
        return fam
    return replace(fam, max_condition_images = row[1])


def comfy_flow_shift_for(fam: Any, *identifiers: Optional[str]) -> Optional[float]:
    row = _first_variant(getattr(fam, "comfy_flow_shift_variants", ()), identifiers)
    return row[1] if row else getattr(fam, "comfy_flow_shift", None)


# FLUX.3 single files / GGUFs take their transformer config from the companion repo, which may be the other
# variant's (the TE and VAE are shared, and the base repo is gated). The variant check has already matched the
# checkpoint to the family, so the family sets flow_map: a base file on the distilled config would otherwise build the
# target-time embedder the file has no weights for and fail on meta tensors, and the reverse would drop its weights.
_FLUX3_FLOW_MAP = {"flux.3-klein": True, "flux.3-klein-base": False}


def comfy_shift_node_for(fam: Any, *identifiers: Optional[str]) -> Optional[str]:
    """The ComfyUI shift node (diffusion_flow_shift SHIFT_NODE_*) for this checkpoint, or None."""
    row = _first_variant(getattr(fam, "comfy_shift_node_variants", ()), identifiers)
    return row[1] if row else getattr(fam, "comfy_shift_node", None)


def transformer_config_overrides_for(fam: Any, *identifiers: Optional[str]) -> dict[str, Any]:
    row = _first_variant(getattr(fam, "transformer_config_variants", ()), identifiers)
    overrides = dict(row[1]) if row else {}
    flow_map = _FLUX3_FLOW_MAP.get(getattr(fam, "name", None))
    if flow_map is not None:
        overrides["flow_map"] = flow_map
    return overrides


def _first_checkpoint_variant(
    keys: tuple[str, ...], identifiers: tuple[Optional[str], ...]
) -> Optional[str]:
    # flux.1-dev / flux1_dev / FLUX-1-dev all read flux1-dev
    for identifier in identifiers:
        needle = re.sub(r"(?<=[a-z])-(?=\d)", "", normalize_family_name(identifier or ""))
        for key in keys:
            if key in needle:
                return key
    return None


def transformer_variant_differs_from_base(
    fam: Any, base: Optional[str], *identifiers: Optional[str]
) -> bool:
    """Checkpoint and ``base`` name different variants; a base naming none (a local dir) is unknown."""
    keys = getattr(fam, "checkpoint_variants", ())
    if keys:
        base_key = _first_checkpoint_variant(keys, (base,))
        return base_key is not None and _first_checkpoint_variant(keys, identifiers) not in (
            None,
            base_key,
        )
    rows = getattr(fam, "transformer_config_variants", ())
    base_row = _first_variant(rows, (base,))
    return base_row is not None and _first_variant(rows, identifiers) not in (None, base_row)


def named_generation_params(*identifiers: Optional[str]) -> Optional[tuple[int, float]]:
    """``(steps, guidance)`` of the first identifier naming a known model (a local path resolves via its base), or None."""
    for identifier in identifiers:
        needle = model_identity(identifier).lower()
        for key, steps, guidance in _GENERATION_DEFAULTS:
            if name_key_in(key, needle):
                return steps, guidance
    return None


def default_generation_params(*identifiers: Optional[str]) -> tuple[int, float]:
    """:func:`named_generation_params`, else the generic fallback."""
    return named_generation_params(*identifiers) or _GENERATION_DEFAULT_FALLBACK


def generation_params_with_grid(
    grid: Optional[tuple[float, ...]], *identifiers: Optional[str]
) -> tuple[int, float]:
    """:func:`default_generation_params`, but an unnamed load with a shipped grid runs the grid's own step count."""
    named = named_generation_params(*identifiers)
    if named is not None:
        return named
    if grid:
        return len(grid), _GENERATION_DEFAULT_FALLBACK[1]
    return _GENERATION_DEFAULT_FALLBACK


def family_prequant_repo(
    fam: DiffusionFamily,
    scheme: str,
    base_repo: Optional[str] = None,
) -> Optional[str]:
    """The hosted pre-quantized transformer repo for ``scheme`` in this family, or None.
    ``base_repo`` (when known) selects a variant-specific checkpoint first: a checkpoint is baked
    from ONE base's weights and the loader refuses it for any other base, so a variant without
    its own entry still returns the family default. That is harmless only while the default is
    close enough that planning around it costs nothing, since the base_model_id validation
    refuses the artifact well after the plan was made. A base whose weights really differ belongs
    in ``prequant_excluded_bases``, which returns None here instead."""
    if nvfp4_blocked(scheme):
        return None
    # Both tables are keyed on lowercased upstream ids.
    base = canonical_base(base_repo).lower()
    named = named_variant_base(fam, base_repo) if base else None
    if named and named.lower() != base:
        # A local copy naming a variant: its row if the loader's tail compare accepts it, else none (never 2.1's).
        if (
            base.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            != named.lower().rsplit("/", 1)[-1]
        ):
            return None
        base = named.lower()
    if base:
        # getattr, because the video loader calls this with a VideoFamily, which has no such field. A plain attribute
        # read raises AttributeError, resolve_prequant_source swallows it in its bare except and hands back None, and
        # every video family silently loses its hosted prequant checkpoint to the dense path whenever a base_repo is
        # passed.
        # A variant row beats the exclusion: an excluded base may host its own checkpoints for some schemes.
        for entry_base, entry_scheme, repo_id in fam.prequant_variant_repos:
            if entry_base == base and entry_scheme == scheme:
                return repo_id
        if base in (getattr(fam, "prequant_excluded_bases", ()) or ()):
            return None
    for entry_scheme, repo_id in fam.prequant_repos:
        if entry_scheme == scheme:
            return repo_id
    return None


def family_prequant_filename(
    fam: DiffusionFamily,
    scheme: str,
    task: Optional[str] = None,
) -> Optional[str]:
    """The preferred checkpoint filename this family declares for ``scheme``, or None.

    ``None`` means "use the derived ``<Model>-<SCHEME>.pt`` name", which is every family but the
    ones shipping a second artifact under the same repo and scheme. Not variant-keyed: the filename
    says WHICH artifact, the repo says which base.

    Rows come in two shapes. ``(scheme, filename)`` is the historical one and is TASK-AGNOSTIC.
    ``(scheme, task, filename)`` names an artifact for one task only and wins over the agnostic row
    when ``task`` matches. That distinction exists because a family can hold several denoiser
    PARTITIONS in one repo (MiniMax-H3: keyframe vs reference), whose checkpoints have identical key
    sets and identical metadata and so cannot be told apart by any later check -- picking the wrong
    one generates from the wrong partition rather than failing.

    ``task = None`` therefore sees only the agnostic rows, and a scheme with no row for the task
    asked for falls back to the agnostic one, i.e. exactly today's behaviour. Malformed rows are
    skipped rather than raising: this runs on a refusal path where a table typo must not 500.
    """
    wanted = (task or "").strip().lower()
    agnostic: Optional[str] = None
    for entry in getattr(fam, "prequant_filenames", ()) or ():
        if not isinstance(entry, (tuple, list)):
            continue
        if len(entry) == 2:
            entry_scheme, filename = entry
            if entry_scheme == scheme and agnostic is None and filename:
                agnostic = filename
        elif len(entry) == 3:
            entry_scheme, entry_task, filename = entry
            if (
                wanted
                and entry_scheme == scheme
                and (entry_task or "").strip().lower() == wanted
                and filename
            ):
                return filename
    return agnostic


# the release where diffusers' own requires-python went ">= 3.10.0", making 0.36.0 the newest a supported Python 3.9
# host can resolve
_DIFFUSERS_DROPPED_PY39 = "0.37.0"
_MAX_PIPELINE_MANIFEST_BYTES = 1 << 20

# First diffusers release exporting each pipeline class, read off ``src/diffusers/__init__.py`` at the upstream tags
# and cross-checked against each release's requires-python on PyPI. An unlisted class gets a version-free "a newer
# diffusers" instead of a number, since the ones left out are older than any release in play -- and
# ``family_pipeline_available`` reads one as available, so every class the listing probes belongs here. This exists so
# the remedy is true: telling a 3.9 host that Z-Image needs Python >= 3.10 sends it to upgrade the interpreter when
# ``pip install -U diffusers`` (0.36.0 there) would have been enough.
_PIPELINE_MIN_DIFFUSERS: dict[str, str] = {
    # MiniMax-H3 is judged by its transformer, first available in diffusers 0.40.0.
    "MiniMaxH3Transformer3DModel": "0.40.0",
    "Flux2Pipeline": "0.36.0",
    "ZImagePipeline": "0.36.0",
    "ZImageImg2ImgPipeline": "0.36.0",
    "HunyuanImagePipeline": "0.36.0",
    "HunyuanVideo15Pipeline": "0.36.0",
    "QwenImageControlNetPipeline": "0.36.0",
    "QwenImageEditPlusPipeline": "0.36.0",
    "Flux2KleinPipeline": "0.37.0",
    "ZImageInpaintPipeline": "0.37.0",
    "LTX2Pipeline": "0.37.0",
    # Absent from the 0.36.0 wheel's diffusers/__init__.py, exported by 0.37.0's.
    "QwenImageLayeredPipeline": "0.37.0",
    "Flux2KleinInpaintPipeline": "0.38.0",
    "Ideogram4Pipeline": "0.39.0",
    # Qwen-Image-2.1 merged upstream on 2026-09-18, four weeks after 0.40.0 was cut, so 0.41.0 is
    # the first release that can carry it. ``_version_tuple`` stops at the first non-numeric part,
    # so a 0.41.0.dev0 build from main reads as (0, 41, 0) and satisfies this too.
    "QwenImage21Pipeline": "0.41.0",
    # FLUX.3 Image Klein (huggingface/diffusers#15046) is in no release yet, so the first one that can carry it is
    # 0.42.0. Main and the PR branch both report 0.41.0.dev0, which reads as the 0.41.0 release that lacks it, so the
    # listing probes the module file instead (``_PIPELINE_PROBE_FILES``).
    "Flux3ModularPipeline": "0.42.0",
    "Flux3DistilledModularPipeline": "0.42.0",
    "Krea2Pipeline": "0.39.0",
    # Older than the 0.35 baseline, but listed anyway: the packaging leaves an UNCONSTRAINED diffusers installable
    # below 3.10, so an already-present ancient one satisfies the pin, and quoting the 0.39 floor at a family that has
    # existed since 0.30 is the same wrong remedy.
    "QwenImagePipeline": "0.35.0",
    "QwenImageImg2ImgPipeline": "0.35.0",
    "QwenImageInpaintPipeline": "0.35.0",
    "FluxKontextPipeline": "0.35.0",
    "HiDreamImagePipeline": "0.34.0",
    # WanPipeline arrived with Wan2.1 in 0.33.0. The shipped Wan2.2 family wants weights only 0.35 carries, but this
    # table answers class presence, like the attribute probe before it.
    "WanPipeline": "0.33.0",
    "Lumina2Pipeline": "0.33.0",
    "FluxPipeline": "0.30.0",
    "FluxImg2ImgPipeline": "0.30.0",
    "FluxInpaintPipeline": "0.30.0",
}


def _version_tuple(v: str) -> tuple[int, ...]:
    """``"0.37.0" -> (0, 37, 0)`` for ordering. Numeric so 0.9 sorts below 0.10, which a string
    compare gets backwards; a non-numeric part stops the parse rather than raising."""
    out: list[int] = []
    for part in str(v).split("."):
        if not part.isdigit():
            break
        out.append(int(part))
    return tuple(out)


def pipeline_class_requirement(pipeline_class: str) -> tuple[Optional[str], bool]:
    """``(minimum diffusers version, whether that minimum also needs Python >= 3.10)``. ``None`` for
    a class with no entry. That is deliberately not the packaging floor: an unlisted class is one
    old enough that no release in play lacks it (StableDiffusionXLPipeline goes back past 0.29),
    so naming 0.39 would send a supported Python 3.9 host to upgrade its interpreter for a class
    every diffusers it can install already has. Without an entry the refusal says "a newer
    diffusers" and stops there, which is true whatever the class."""
    minimum = _PIPELINE_MIN_DIFFUSERS.get(pipeline_class)
    if minimum is None:
        return None, False
    return minimum, _version_tuple(minimum) >= _version_tuple(_DIFFUSERS_DROPPED_PY39)


def _json_dict(path: Path, max_bytes: int = _MAX_PIPELINE_MANIFEST_BYTES) -> Optional[dict]:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        # PowerShell writes JSON with a UTF-8 BOM; match pipeline_class_from_index.
        payload = json.loads(path.read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


_CALLER_SUPPLIED_COMPONENTS = {"HiDreamImagePipeline": frozenset({"text_encoder_4", "tokenizer_4"})}
# Remote-code components Studio loads through an in-tree diffusers port, judged by the layout that port reads (MOSS-Image
# names its own ``pipeline_moss_image`` module, which would otherwise pass on any file: a config-only transformer/).
_IN_TREE_PORT_LIBRARIES = {
    ("pipeline_moss_image", "MOSSImageTransformer"): "diffusers",
    ("pipeline_moss_image", "MOSSImageScheduler"): "diffusers",
}
# safetensors first, default variant only: the load uses variant=None, which cannot open *.fp16.safetensors.
_LOCAL_PIPELINE_WEIGHT_FORMATS = (
    (("diffusion_pytorch_model", "model"), "safetensors"),
    (("diffusion_pytorch_model", "pytorch_model"), "bin"),
)
_MAX_PIPELINE_WEIGHT_INDEX_BYTES = 64 * 1024 * 1024
_LOCAL_PIPELINE_METADATA_CONFIGS = (
    (("tokenizer",), ("tokenizer_config.json",)),
    (("scheduler",), ("scheduler_config.json",)),
    (("guider", "guidance"), ("guider_config.json",)),
    (("featureextractor", "imageprocessor"), ("preprocessor_config.json",)),
    (("processor",), ("processor_config.json", "preprocessor_config.json")),
)
_SELF_CONTAINED_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "vocab.txt",
    "spiece.model",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _safe_relative_parts(text: str) -> Optional[tuple[str, ...]]:
    # Manifest paths are POSIX-relative: reject other separators and drive prefixes, or ..\\ / C: escape on Windows.
    relative = PurePosixPath(text)
    if "\\" in text or ":" in text or relative.is_absolute() or ".." in relative.parts:
        return None
    return relative.parts


def _local_weights_are_complete(component: Path, library_name: str) -> bool:
    # The first format with any weights present decides: a leftover .bin index cannot veto safetensors.
    for stems, ext in _LOCAL_PIPELINE_WEIGHT_FORMATS:
        # Transformers never reads diffusion_pytorch_model* (LTX-2's text_encoder ships both shard sets).
        if library_name == "transformers":
            stems = tuple(s for s in stems if s != "diffusion_pytorch_model")
            # Transformers opens a single checkpoint before a shard index (_get_resolved_checkpoint_files).
            if any(_nonempty_file(component / f"{s}.{ext}") for s in stems):
                return True
        index = next(
            (p for s in stems if (p := component / f"{s}.{ext}.index.json").exists()), None
        )
        if index is not None:
            weight_map = (_json_dict(index, _MAX_PIPELINE_WEIGHT_INDEX_BYTES) or {}).get(
                "weight_map"
            )
            shards = (
                {str(v) for v in weight_map.values() if v} if isinstance(weight_map, dict) else ()
            )
            parts = [_safe_relative_parts(shard) for shard in shards]
            return bool(parts) and all(
                p is not None and _nonempty_file(component.joinpath(*p)) for p in parts
            )
        sizes = [w.stat().st_size for s in stems if (w := component / f"{s}.{ext}").is_file()]
        if sizes:
            return any(sizes)
    return False


def _local_pipeline_component_is_complete(
    component: Path, library_name: str, class_name: str, config_only_model_components: bool
) -> bool:
    if not component.is_dir():
        return False
    library_name = _IN_TREE_PORT_LIBRARIES.get((library_name, class_name), library_name)
    if library_name not in {"diffusers", "transformers"}:
        return any(_nonempty_file(child) for child in component.iterdir())
    identity = class_name.replace("_", "").lower()
    for tokens, config_names in _LOCAL_PIPELINE_METADATA_CONFIGS:
        if not any(token in identity for token in tokens):
            continue
        # Configs can exceed the manifest cap: LTX-2's Gemma3 tokenizer_config.json is 1.1 MB.
        if not any(
            _json_dict(component / name, _MAX_PIPELINE_WEIGHT_INDEX_BYTES) is not None
            for name in config_names
        ):
            return False
        if tokens[0] not in ("tokenizer", "processor") or "byt5tokenizer" in identity:
            return True
        return any(_nonempty_file(component / a) for a in _SELF_CONTAINED_TOKENIZER_ASSETS) or all(
            _nonempty_file(component / a) for a in ("vocab.json", "merges.txt")
        )
    return _json_dict(component / "config.json") is not None and (
        config_only_model_components or _local_weights_are_complete(component, library_name)
    )


def local_pipeline_components_are_complete(
    root: Path | str,
    filename: str,
    *,
    excluded_components: Sequence[str] = (),
    config_only_model_components: bool = False,
    fixed_component_folders: Optional[Mapping[str, tuple[str, str, str]]] = None,
) -> bool:
    """Check local component presence and known Diffusers/Transformers serialization layouts.

    ``fixed_component_folders`` (``{component: (folder, library, class)}``) is for a loader that reads a fixed set
    of components from fixed folders of this root whatever the manifest lists (FLUX.3's assembler). Exactly those
    are checked, each at ``root/<folder>``, whether or not the manifest names them, ``excluded_components`` for their
    config only;
    every other manifest entry and every spec source is ignored, since none of it is loaded (FLUX.3's optional
    prompt upsampler lives in another repo). The manifest still has to be readable and name a pipeline class."""
    if filename not in {"model_index.json", "modular_model_index.json"}:
        return False
    base = Path(root).expanduser()
    payload = _json_dict(base / filename) or {}
    class_name = payload.get("_class_name")
    if not isinstance(class_name, str) or not class_name.strip():
        return False
    if fixed_component_folders is not None:
        # An excluded component's WEIGHTS come from elsewhere (a GGUF / single file), but the loader still reads its
        # config from this folder (FLUX.3 builds the denoiser with config=<base>, subfolder="transformer"), so it is
        # checked config-only rather than skipped.
        try:
            return bool(fixed_component_folders) and all(
                _local_pipeline_component_is_complete(
                    base / folder,
                    library,
                    cls,
                    config_only_model_components or name in excluded_components,
                )
                for name, (folder, library, cls) in fixed_component_folders.items()
            )
        except OSError:
            return False
    caller_supplied = _CALLER_SUPPLIED_COMPONENTS.get(class_name, frozenset())
    declared = False
    try:
        for name, spec in payload.items():
            if (
                name.startswith("_")
                or not isinstance(spec, (list, tuple))
                or len(spec) < 2
                or not (isinstance(spec[0], str) and isinstance(spec[1], str))
            ):
                continue
            if name in {"", ".."} or "\\" in name or Path(name).name != name:
                return False
            if name in excluded_components or (
                name in caller_supplied and not (base / name).exists()
            ):
                continue
            declared = True
            component = base / name
            source = spec[2] if filename == "modular_model_index.json" and len(spec) >= 3 else None
            source = source if isinstance(source, dict) else {}
            repo = source.get("pretrained_model_name_or_path") or source.get("repo")
            if isinstance(repo, str) and repo.strip():
                # A modular spec may load the component from another repo or folder instead.
                repo = repo.strip()
                subfolder = source.get("subfolder")
                parts = (
                    _safe_relative_parts((subfolder or "").strip())
                    if subfolder is None or isinstance(subfolder, str)
                    else None
                )
                if parts is None:
                    return False
                rooted = Path(repo).expanduser()
                rooted = rooted if rooted.is_absolute() else base / rooted
                if not rooted.exists():
                    if repo.startswith(("/", "\\", "~", ".")) or "\\" in repo or ":" in repo:
                        return False
                    continue
                component = rooted.joinpath(*parts)
            if not _local_pipeline_component_is_complete(
                component, spec[0], spec[1], config_only_model_components
            ):
                return False
    except OSError:
        return False
    return declared


# Minimums that name a release which does not EXIST yet. ``pip install -U 'diffusers>=0.41.0'`` has
# no candidate today, so quoting it as the remedy sends someone to a command that cannot succeed.
# Studio installs the pinned main build for exactly these classes (studio/backend/requirements/
# diffusers-main.txt), so the remedy is to put that back, not to chase a release. Delete an entry
# here the moment its version ships, which is the same moment diffusers-pin.txt moves to it.
_UNRELEASED_MIN_DIFFUSERS: frozenset = frozenset({"0.42.0"})
# The pinned main build's requires-python (the installer's DIFFUSERS_MAIN_MIN_PYTHON).
_DIFFUSERS_MAIN_MIN_PYTHON = (3, 11)

# Classes whose presence a version number cannot answer: an unreleased model on a diffusers main build that reports
# the same x.y.0.dev0 as the release without it. The listing checks the file that defines the class, relative to the
# installed diffusers package, which needs no import. Delete an entry once its release is the pin.
_PIPELINE_PROBE_FILES: dict[str, str] = {
    "Flux3ModularPipeline": "modular_pipelines/flux3/modular_pipeline.py",
    "Flux3DistilledModularPipeline": "modular_pipelines/flux3/modular_pipeline.py",
}


# In-tree pipelines: (diffusers VAE module, transformers floor, family label).
_IN_TREE_PIPELINES: dict[str, tuple[str, str, str]] = {
    "MOSSImagePipeline": ("models/autoencoders/autoencoder_kl_flux2.py", "5.3.0", "MOSS-Image"),
}


def in_tree_pipeline(pipeline_class: Optional[str]) -> bool:
    """Whether ``pipeline_class`` is one of Studio's in-tree pipelines (no diffusers class of that name)."""
    return str(pipeline_class or "") in _IN_TREE_PIPELINES


def _installed_transformers_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        installed = version("transformers")
    except Exception:  # noqa: BLE001 - not installed / unreadable metadata
        return None
    return installed.strip() if isinstance(installed, str) and installed.strip() else None


def in_tree_pipeline_refusal(pipeline_class: str, family_name: str) -> Optional[str]:
    """Why this environment cannot build the in-tree ``pipeline_class``, or None when it can. Import-free."""
    vae_file, min_transformers, label = _IN_TREE_PIPELINES[pipeline_class]
    installed = _installed_transformers_version()
    if installed is None:
        return f"'{family_name}' ({label}) needs transformers >= {min_transformers}, which is not installed."
    if not _installed_at_least(installed, min_transformers):
        return (
            f"'{family_name}' ({label}) needs transformers >= {min_transformers} for its Qwen3.5 text encoder "
            f"(transformers.Qwen3_5Model), but this environment has transformers {installed}. Update Unsloth Studio "
            f"(or run: pip install -U 'transformers>={min_transformers}') and restart it."
        )
    if _diffusers_package_has_file(vae_file) is False:
        return (
            f"'{family_name}' ({label}) needs a diffusers with the FLUX.2 VAE (AutoencoderKLFlux2). "
            f"{DIFFUSERS_UPDATE_REMEDY}"
        )
    return None


def _diffusers_package_has_file(relative: str) -> Optional[bool]:
    """Whether the installed diffusers package contains ``relative``; None when it cannot be located."""
    try:
        import importlib.util

        module = sys.modules.get("diffusers")
        spec = getattr(module, "__spec__", None) if module is not None else None
        if spec is None:
            spec = importlib.util.find_spec("diffusers")
        locations = list(getattr(spec, "submodule_search_locations", None) or ())
    except Exception:  # noqa: BLE001 -- an unlocatable package answers nothing
        return None
    if not locations:
        return None
    return any((Path(location) / relative).is_file() for location in locations)


_DIFFUSERS_MAIN_PIN = Path(__file__).resolve().parents[2] / "requirements" / "diffusers-main.txt"
_DIFFUSERS_MAIN_COMMIT_RE = re.compile(
    r"github\.com/(?P<repo>[^/\s]+/[^/@\s]+?)(?:\.git)?@(?P<commit>[0-9a-fA-F]{40})\b"
)


# Leads every refusal: a desktop install has no terminal, so pip spellings come second.
DIFFUSERS_UPDATE_REMEDY = (
    "Update Unsloth to install it (in the desktop app: Settings, Check for updates; from a "
    "terminal: unsloth studio update), then restart Unsloth."
)
DIFFUSERS_MAIN_RESTART_REMEDY = (
    "Restart Unsloth: on start it installs this pinned build by itself when it can reach "
    "github.com. If that still fails, update Unsloth (in the desktop app: Settings, Check for "
    "updates)."
)


def _diffusers_main_archive_remedy() -> str:
    """A remedy line a reader can actually run, naming the commit this build wants.

    The old text said "re-run the installer, or pip install -r diffusers-main.txt". Both resolve
    the same ``git+https`` requirement, so both fail for the one cause that produces this refusal
    most often, a host with no working git, and the reader is sent round the loop that put them
    here. The installer now falls back to the zip for that case, so re-running is once again real
    advice, and the zip is quoted beside it because it is the fix that needs no git and no
    installer run at all.

    Read out of the pin file rather than hardcoded, so the commit in the message cannot drift from
    the commit that is installed. An unreadable or unrecognised pin file degrades to advice that is
    still true, just less specific, because a refusal that says nothing is worse than one that
    cannot name the SHA.
    """
    generic = (
        f"{DIFFUSERS_MAIN_RESTART_REMEDY} On a pip or server install, re-run the Unsloth installer "
        "(leaving UNSLOTH_DIFFUSERS_MAIN unset), which installs it from a zip archive when git is "
        "missing."
    )
    try:
        text = _DIFFUSERS_MAIN_PIN.read_text(encoding = "utf-8-sig")
    except (OSError, ValueError):
        return generic
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        found = _DIFFUSERS_MAIN_COMMIT_RE.search(stripped)
        if found is None:
            continue
        url = (
            f"https://github.com/{found.group('repo')}/archive/"
            f"{found.group('commit').lower()}.zip"
        )
        return (
            f"{generic} To install it by hand from a terminal with no git at all: "
            f'pip install "diffusers @ {url}"'
        )
    return generic


def _too_old_message(pipeline_class: str, family_name: str, installed: str) -> str:
    """The refusal text: what is missing, what is installed, and a remedy this interpreter can
    actually carry out."""
    minimum, needs_py310 = pipeline_class_requirement(pipeline_class)
    if minimum is None:
        return (
            f"'{family_name}' needs a newer diffusers ({pipeline_class}); this environment has "
            f"diffusers {installed}. {DIFFUSERS_UPDATE_REMEDY} On a plain pip install: "
            "pip install -U diffusers."
        )
    if minimum in _UNRELEASED_MIN_DIFFUSERS and sys.version_info[:2] < _DIFFUSERS_MAIN_MIN_PYTHON:
        # The installer never tries the main build here (it declares requires-python >= 3.11), so no
        # reinstall or archive can help: only a newer interpreter, or the release that carries it.
        return (
            f"'{family_name}' needs diffusers >= {minimum} ({pipeline_class}), which has not been "
            f"released yet; this environment has diffusers {installed}. Unsloth installs a pinned "
            "build of diffusers main for this, and that build needs Python 3.11 or newer; this "
            f"is Python {sys.version_info[0]}.{sys.version_info[1]}. Reinstall Unsloth on Python "
            "3.11+, or wait for the diffusers release that includes it."
        )
    if minimum in _UNRELEASED_MIN_DIFFUSERS:
        remedy = _diffusers_main_archive_remedy()
        return (
            f"'{family_name}' needs diffusers >= {minimum} ({pipeline_class}), which has not been "
            f"released yet; this environment has diffusers {installed}. Unsloth installs a pinned "
            "build of diffusers main for this and that build is not here, which almost always "
            "means the install had no working git (check with: git --version) or could not reach "
            f"github.com. {remedy}"
        )
    remedy = (
        f"{DIFFUSERS_UPDATE_REMEDY} On a plain pip install: pip install -U 'diffusers>={minimum}'."
    )
    if needs_py310:
        remedy += (
            f" diffusers dropped Python 3.9 in {_DIFFUSERS_DROPPED_PY39}, so that release needs "
            f"Python >= 3.10 too."
        )
    return (
        f"'{family_name}' needs diffusers >= {minimum} ({pipeline_class}); this environment has "
        f"diffusers {installed}. {remedy}"
    )


def _dummy_required_backends(cls: object) -> tuple[str, ...]:
    """The backends diffusers says ``cls`` REQUIRES, when ``cls`` is one of its placeholders.

    Required, not missing: ``_backends`` is the class's full requirement list, so a placeholder
    standing in because transformers is absent still lists torch beside it. Naming them all as
    missing, and prescribing a reinstall, is how you tell someone with a working ROCm or CUDA build
    of torch to replace it.

    With a required backend absent (torch, transformers, ...), diffusers still EXPORTS every
    pipeline name, as a ``DummyObject``-metaclassed stand-in from ``diffusers.utils.dummy_*`` whose
    ``from_pretrained`` raises ``ImportError`` on the first call. ``hasattr`` therefore answers True
    for a class that cannot be used, which is exactly the "importable" answer the strict gate must
    not accept. Empty tuple for a real class.
    """
    if not str(getattr(cls, "__module__", "")).startswith("diffusers.utils.dummy"):
        return ()
    backends = getattr(cls, "_backends", None) or ()
    return tuple(str(b) for b in backends)


def assert_pipeline_class_available(
    pipeline_class: str,
    family_name: str,
    *,
    strict: bool = False,
) -> None:
    """Raise ``ValueError`` before any download when the installed diffusers has no ``pipeline_class``.

    The newer families (Flux2Klein, Z-Image, Krea 2, LTX-2, HunyuanImage) only exist from a
    diffusers newer than the 0.35 baseline, and the packaging leaves an older one installable on
    Python 3.9 -- diffusers dropped 3.9 in 0.37 and this project still supports it, so the 0.39
    floor has to be conditional or the whole extra becomes unresolvable. Which release a family
    needs differs per class, so the refusal reads it from ``_PIPELINE_MIN_DIFFUSERS`` rather than
    quoting the floor at everyone. Without this check the getattr chain died with a bare
    AttributeError deep in the load, after the checkpoint had already been fetched.

    ``strict`` decides what an *unimportable* diffusers means. Inference (the default) stays silent:
    it only answers "is the installed diffusers new enough", and the native sd.cpp engine serves
    GGUF picks on a CPU or Apple host that has no diffusers at all. Training passes ``strict =
    True``, because its child is an ``mp.get_context("spawn")`` process in the SAME interpreter --
    an import that fails here fails there too, only after the route has reserved the training slot
    and freed the resident GPU models.

    ``ValueError``, like every other unloadable-pick refusal ``validate_load_request`` raises, so
    the routes map it to 400 with the message intact. A ``RuntimeError`` instead reached
    ``/images/load``'s 409 (the code that otherwise means "a load is already in progress") and
    escaped ``/images/download-plan``, which catches only (ValueError, FileNotFoundError), as a bare
    500 with the message lost.
    """
    # Here rather than at the call sites, because this is where the import is. Every caller runs
    # on a request thread that can be racing the background torch warm: image validation, video
    # validation, and the training preflight via _assert_family_pipeline_available. `import
    # diffusers` pulls torch._dynamo in by itself (diffusers.hooks evaluates
    # @torch.compiler.disable() at class-body time), and so does the hasattr below, which is what
    # actually imports the pipeline's submodule (#10350, #10963). Guarded around the IMPORT as
    # well as the call: utils.torch_warmup reaches importlib._bootstrap._ModuleLockManager, a
    # private CPython name, and a build lacking it must not turn this check into a failure.
    try:
        from loggers import get_logger
        from utils.torch_warmup import close_dynamo_import_window
        close_dynamo_import_window(get_logger(__name__))
    except Exception:  # noqa: BLE001, S110 - optimisation only, and this module has no logger
        pass

    if in_tree_pipeline(pipeline_class):
        reason = in_tree_pipeline_refusal(pipeline_class, family_name)
        if reason is not None:
            raise ValueError(reason)
        return
    try:
        # diffusers' LTX-2 modules import a transformers class the pinned transformers lacks; see ltx2_import_compat.
        from .ltx2_import_compat import ensure_ltx2_pipelines_importable, is_ltx2_pipeline_class

        if is_ltx2_pipeline_class(pipeline_class):
            ensure_ltx2_pipelines_importable()
        import diffusers

        present = hasattr(diffusers, pipeline_class)
        dummy_backends = _dummy_required_backends(getattr(diffusers, pipeline_class, None))
    except Exception as exc:  # noqa: BLE001 -- see below: this check must never raise anything but its own ValueError
        # Not this check's business under the default: it answers "is the installed diffusers new enough for this
        # family", and with nothing importable there is no version to judge. Refusing would also break the native
        # sd.cpp engine, which serves GGUF picks on a CPU or Apple host without diffusers. The one thing that must not
        # happen is a raise of the wrong type: ModuleNotFoundError is not the ValueError the routes map to 400, so it
        # escapes /images/download-plan as a bare 500 with the message lost. The attribute probe is inside the try for
        # the same reason: diffusers' top level is a lazy module, so ``hasattr`` is what actually imports the
        # pipeline's submodule, and when that submodule's own dependencies are unsatisfiable it raises RuntimeError
        # ("Failed to import diffusers.pipelines..."), which hasattr does NOT swallow since it only absorbs
        # AttributeError.
        if strict:
            raise ValueError(
                f"'{family_name}' needs diffusers ({pipeline_class}), which this environment "
                f"cannot import: {exc}. {DIFFUSERS_UPDATE_REMEDY} On a plain pip install, repair it "
                "with: pip install -U diffusers."
            ) from None
        return

    if present and dummy_backends:
        # A placeholder, not the pipeline. Under the default this is left alone like every other unusable install;
        # strict refuses, because the trainer child imports the same placeholder and its from_pretrained raises only
        # after the GPU residents are gone.
        if not strict:
            return
        raise ValueError(
            f"'{family_name}' needs diffusers ({pipeline_class}), but this diffusers exports it as "
            f"a placeholder, which it does when a backend it requires is unavailable. That class "
            f"requires: {', '.join(dummy_backends)}. Check which of those this environment is "
            f"missing and install it."
        )

    if present:
        return
    raise ValueError(
        _too_old_message(
            pipeline_class, family_name, str(getattr(diffusers, "__version__", "unknown"))
        )
    )


def _module_namespace_is_unreadable(module: Any) -> bool:
    """Return whether probing attributes could import code or read a partial module."""
    if hasattr(type(module), "__getattr__"):
        return True
    if callable(getattr(module, "__getattr__", None)):
        return True
    return bool(getattr(getattr(module, "__spec__", None), "_initializing", False))


def _installed_diffusers_version() -> Optional[str]:
    """Read the installed diffusers version without importing it."""
    module = sys.modules.get("diffusers")
    if module is not None:
        try:
            installed = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001 -- a module that raises on __version__ just falls through
            installed = None
        if isinstance(installed, str) and installed.strip():
            return installed.strip()
    try:
        from importlib.metadata import version
        installed = version("diffusers")
    except Exception:  # noqa: BLE001 -- not installed / unreadable metadata: caller fails open
        return None
    return installed.strip() if isinstance(installed, str) and installed.strip() else None


def _installed_at_least(installed: str, minimum: str) -> bool:
    """Whether an INSTALLED version satisfies ``minimum``, judged on its release numbers. Not
    ``_version_tuple``, which is for the clean constants in the table above: a vendor build
    carries a PEP 440 local suffix (``0.40.0+dfsg``) that stops the numeric parse mid-version,
    and a git install carries ``.dev0``, which strict PEP 440 sorts below its own release. Both
    HAVE the class, so compare the release they were cut from. An unreadable version answers
    OPEN."""
    try:
        from packaging.version import Version
        return Version(Version(installed).base_version) >= Version(minimum)
    except Exception:  # noqa: BLE001 -- an unparseable version must not hide a model
        return True


def family_probe_class(fam: Any) -> str:
    """The class whose presence in the installed diffusers actually proves ``fam`` is loadable.
    Normally that is ``fam.pipeline_class``. ``ModularPipeline`` is the exception: it is the
    generic entry point for every Modular Diffusers workflow, not a family, and it has existed
    for several releases, so a diffusers that predates MiniMax-H3's own blocks still answers
    hasattr for it. Probe the family's own transformer class there instead, which is the thing
    the load actually needs. Shared by the listing probe and by both training gates so a family
    cannot be hidden from the picker and simultaneously accepted by /diffusion/start."""
    name = str(getattr(fam, "pipeline_class", "") or "")
    if name == "ModularPipeline":
        return str(getattr(fam, "transformer_class", None) or name)
    return name


def family_pipeline_available(fam: Optional[DiffusionFamily]) -> bool:
    """True when the installed diffusers actually has this family's pipeline class.

    The boolean twin of ``assert_pipeline_class_available``, for the listing routes: the newer
    families exist only from diffusers 0.39, and the packaging leaves an older diffusers installable
    on Python 3.9 (diffusers dropped 3.9 in 0.37, so the 0.39 floor has to be conditional or the
    extra becomes unresolvable). Advertising Z-Image or Krea 2 in the picker on such an environment
    offers a pick that can only fail, and no `pip install -U diffusers` can fix it without also
    upgrading Python. Fails OPEN (True) when diffusers cannot be imported at all, so a listing never
    hides a model over an unrelated import problem.

    Uses installed-version metadata because probing diffusers' lazy attributes imports pipeline
    dependencies. The load path remains the final availability check.
    """
    if fam is None:
        return False
    # A modular family is judged on its own transformer class, not on the generic ``ModularPipeline`` entry point --
    # see ``family_probe_class``.
    name = family_probe_class(fam)
    # No class name to probe means the record is not one this helper can judge, which is the same position as a
    # missing diffusers: answer OPEN. ``family_probe_class`` reads the attribute with a default, so a record without
    # ``pipeline_class`` reaches here as "" rather than raising into the guard below, and ``hasattr(diffusers, "")``
    # is False -- which would hide the model.
    if not name:
        return True
    if in_tree_pipeline(name):
        return in_tree_pipeline_refusal(name, getattr(fam, "name", name)) is None
    probe_file = _PIPELINE_PROBE_FILES.get(name)
    if probe_file is not None:
        has_file = _diffusers_package_has_file(probe_file)
        if has_file is not None:
            return has_file
    if "diffusers" in sys.modules:
        module = sys.modules["diffusers"]
        # A None entry blocks imports, so preserve the existing fail-open behavior.
        if module is None:
            return True
        if not _module_namespace_is_unreadable(module):
            try:
                return hasattr(module, name)
            except Exception:  # noqa: BLE001 -- a probe failure must not hide a model
                return True
    minimum, _needs_py310 = pipeline_class_requirement(name)
    if minimum is None:
        return True
    installed = _installed_diffusers_version()
    if installed is None:
        return True
    return _installed_at_least(installed, minimum)


def _family_override_resolved(family_override: Optional[str], fam) -> tuple:
    reason = "detected from the model" if family_override is None else "requested"
    return (family_override, fam.name, reason)


def family_selectable(fam) -> bool:
    """Whether a Family selector may offer ``fam``: diffusers installed and new enough.

    Import-free: a pipeline-class probe from a status poll raced the loader's own diffusers import;
    the load path keeps the strict class gate."""
    module = sys.modules.get("diffusers", False)
    if module is False:
        try:
            module = importlib.util.find_spec("diffusers")
        except (ImportError, ValueError):
            module = None
    return module is not None and family_pipeline_available(fam)


def modular_image_family_names(names: Iterable[str]) -> tuple[str, ...]:
    """Of ``names``, the families whose repos are Modular Diffusers pipelines (``modular_model_index.json``), so an
    opaque folder picked under an explicit family override is classed ``diffusers_modular_pipeline`` (FLUX.3)."""
    by_name = {f.name: f for f in _FAMILIES}
    return tuple(
        n
        for n in names
        if str(getattr(by_name.get(n), "pipeline_class", "")).endswith("ModularPipeline")
    )


def pipeline_available_family_names() -> tuple[str, ...]:
    return tuple(fam.name for fam in _FAMILIES if family_selectable(fam))


def family_gguf_loadable(fam: DiffusionFamily) -> bool:
    """True when a GGUF transformer can be assembled for this family. The two exclusions mirror the
    ones ``DiffusionBackend.validate_load_request`` raises on (which keep their own specific
    messages): a family whose single file IS the whole pipeline has no transformer-only GGUF, and
    a multi-denoiser family has no single transformer to swap. Exposed so the model-listing
    routes can classify a GGUF the same way the loader would, instead of keeping a second
    hand-maintained list that drifts."""
    return not fam.single_file_is_pipeline and not fam.pipeline_only


def family_mlx_supported(fam: Optional[DiffusionFamily]) -> bool:
    return fam is not None and bool(getattr(fam, "mlx_model_config", None))


def mlx_model_config_for(fam: DiffusionFamily, base_repo: Optional[str]) -> Optional[str]:
    if not family_mlx_supported(fam):
        return None
    key = (base_repo or "").strip().lower()
    for base, name in getattr(fam, "mlx_model_config_by_base", ()) or ():
        if key == base.lower():
            return name
    return fam.mlx_model_config


def family_sd_cpp_supported(fam: DiffusionFamily) -> bool:
    """True when the family has the single-file VAE + text-encoder mapping sd.cpp needs; without it
    the no-GPU route falls back to diffusers.

    Says nothing about the sd.cpp build on this disk. A family that also declares
    ``sd_cpp_arch_marker`` needs ``sd_cpp_binary_runs_family`` on top of this before the native
    route is really available."""
    return bool(fam.sd_cpp_vae and fam.sd_cpp_text_encoders)


# FLUX.2-klein 9B pairs with Qwen3-8B, the 4B with the family-default Qwen3-4B (a mismatched encoder fails deep in
# sd-cli), so pick per variant. Klein 9B's text encoder is byte-identical to unsloth/Qwen3-8B.
_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS = (("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "llm"),)


# sd-cli companion repos released Studio builds fetched and this one no longer reads: the unsloth copies of the
# ComfyUI-built repacks, each holding only VAE / text-encoder files (no denoiser). Nothing loads from them now, but an
# upgraded install still holds their bytes (up to ~70 GB), so they stay companion-only: hidden from the model pickers
# and offered by Free up space once nothing pins them. Never fetched again. The Comfy-Org originals are NOT listed:
# those repos also ship denoisers a user may load as single files, so they must stay ordinary cached repos.
_RETIRED_SD_CPP_COMPANION_REPOS: frozenset[str] = frozenset(
    {
        "unsloth/flux-text-encoders",
        "unsloth/qwen-image-comfyui",
        "unsloth/flux.2-vae",
        "unsloth/flux.2-dev-comfyui",
        "unsloth/z-image-turbo-comfyui",
        "unsloth/flux.2-klein-9b-comfyui",
    }
)


# The community repacks those retired copies stood in for. Released builds sent an sd-cli fetch back to one already
# cached, so a recorded companion link may name it. Ordinary cached repos otherwise (some also ship denoiser files), so
# this set only drops stale LINKS; it never makes one a companion.
_RETIRED_SD_CPP_LEGACY_SOURCES: frozenset[str] = frozenset(
    {
        "comfyanonymous/flux_text_encoders",
        "comfy-org/qwen-image_comfyui",
        "comfy-org/flux2-dev",
        "comfy-org/z_image_turbo",
        "comfy-org/vae-text-encorder-for-flux-klein-9b",
    }
)


def retired_companion_link_ids() -> frozenset[str]:
    """Lowercased repo ids no load reads any more, so a companion link naming one is stale: released sd.cpp loads
    recorded these asset repos, and an installed GGUF must not keep pinning them in Free up space forever."""
    return _RETIRED_SD_CPP_COMPANION_REPOS | _RETIRED_SD_CPP_LEGACY_SOURCES


def sd_cpp_companion_only_repo_ids() -> frozenset[str]:
    """Lowercased ids of repos that exist ONLY to hand sd.cpp a single-file VAE / text encoder.

    They carry no DENOISER, so no diffusion pipeline loads from one. Third-party repacks never
    cleared the cached-model trust gate, but their ``unsloth/*`` mirrors do, so they need excluding
    by hand or each becomes an un-loadable On Device row. Derived from the tables, minus repos that
    are also a real base: FLUX.1-schnell ships the FLUX.1 VAE and IS loadable.

    Diffusion-only, and callers must keep it that way: the set includes
    ``unsloth/Qwen2.5-VL-7B-Instruct-GGUF``, which is a perfectly good CHAT model that sd.cpp also
    borrows as a text encoder. Hiding this set from a chat or GGUF listing would take a real model
    away from a user who downloaded it to chat with. Today's only consumer is the non-GGUF
    cached-model listing, which never sees a GGUF-only repo.
    """
    companions: set[str] = set()
    loadable: set[str] = set()
    for fam in _FAMILIES:
        if fam.sd_cpp_vae:
            companions.add(fam.sd_cpp_vae[0])
        companions.update(repo for repo, _f, _k in fam.sd_cpp_text_encoders)
        loadable.add(fam.base_repo)
        loadable.update(fam.train_base_repos)
        if fam.deploy_base_repo:
            loadable.add(fam.deploy_base_repo)
        loadable.update(repo for _scheme, repo in fam.prequant_repos)
        loadable.update(repo for _base, _scheme, repo in fam.prequant_variant_repos)
        loadable.update(row[2] for row in fam.te_prequant_repos)
        loadable.update(row[3] for row in fam.te_prequant_variant_repos)
    companions.update(repo for repo, _f, _k in _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS)
    return (
        frozenset(r.strip().lower() for r in companions - loadable if r)
        | _RETIRED_SD_CPP_COMPANION_REPOS
    )


def prequant_only_repo_ids() -> frozenset[str]:
    """Repos a pick names for their prequant checkpoints, never a base or a full mirror. A prequant repo that also
    carries the official diffusers folders for its base (``mirror_hosts_artifacts``) stays here: those folders are
    what a pick of the BASE fetches, and picking the prequant repo itself as a pipeline is still the wrong request."""
    hosted: set[str] = set()
    bases: set[str] = set()
    for fam in _FAMILIES:
        hosted.update(repo for _scheme, repo in fam.prequant_repos)
        hosted.update(repo for _base, _scheme, repo in fam.prequant_variant_repos)
        hosted.update(row[2] for row in fam.te_prequant_repos)
        hosted.update(row[3] for row in fam.te_prequant_variant_repos)
        bases.add(fam.base_repo)
        bases.update(fam.train_base_repos)
        if fam.deploy_base_repo:
            bases.add(fam.deploy_base_repo)
    bases.update(rid for pair in _MIRROR_PAIRS for rid in pair if not mirror_hosts_artifacts(rid))
    lowered = {b.strip().lower() for b in bases if b}
    # Every artifact mirror too: the -GGUF ones host no prequant scheme, but their folders are the BASE's pipeline
    # all the same, and a pipeline pick of the repo id itself is the wrong request.
    return (
        frozenset(r.strip().lower() for r in hosted if r and r.strip().lower() not in lowered)
        | _ARTIFACT_MIRRORS
    )


def prequant_repo_role(
    fam: DiffusionFamily, repo_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    key = (repo_id or "").strip().lower()
    known = {
        b.lower(): b
        for b in (fam.base_repo, *fam.train_base_repos, fam.deploy_base_repo or "")
        if b
    }
    bases: list[str] = []
    schemes: list[str] = []
    for entry_base, scheme, repo in fam.prequant_variant_repos:
        if repo.strip().lower() == key:
            bases.append(known.get(entry_base.lower(), entry_base))
            schemes.append(scheme)
    for scheme, repo in fam.prequant_repos:
        if repo.strip().lower() == key:
            bases.append(fam.base_repo)
            schemes.append(scheme)
    te = sorted({row[0] for row in fam.te_prequant_repos if row[2].strip().lower() == key})
    te_variant = [row for row in fam.te_prequant_variant_repos if row[3].strip().lower() == key]
    te = sorted({*te, *(row[1] for row in te_variant)})
    if te_variant and not bases:
        bases.extend(known.get(row[0].lower(), row[0]) for row in te_variant)
    if te and not bases:
        bases.append(fam.base_repo)
    return tuple(dict.fromkeys(bases)), tuple(sorted(set(schemes))), tuple(te)


def sd_cpp_text_encoder_candidates(fam: DiffusionFamily) -> tuple[tuple[str, str, str], ...]:
    """EVERY text-encoder set an sd.cpp load of *fam* could pick, unioned. For the guard, not for a
    load: a load reads the GGUF header and picks one, while a guard reconstructing a checkpoint
    it cannot open has no header, and for FLUX.2-klein a renamed 9B file carries no size token
    either, so the string fallback answers 4B and the 9B encoder the load actually fetched is
    left unprotected. Naming both costs a delete that is refused and saves one that strands an
    installed model."""
    sets = [fam.sd_cpp_text_encoders]
    if fam.name == "flux.2-klein":
        sets.append(_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS)
    return tuple(dict.fromkeys(entry for group in sets for entry in group or ()))


def sd_cpp_text_encoders_for(
    fam: DiffusionFamily,
    repo_id: Optional[str] = None,
    gguf_filename: Optional[str] = None,
    inner_dim: Optional[int] = None,
) -> tuple[tuple[str, str, str], ...]:
    """The sd.cpp text encoders for a specific load. FLUX.2-klein picks by variant (9B needs
    Qwen3-8B, 4B the family default); every other family returns its static table. ``inner_dim``
    is the checkpoint's own answer, read from the GGUF header (``gguf_flux2_inner_dim``): it
    decides whenever the caller has it, because a renamed or hand-picked file makes the load
    identity say nothing. The repo id + filename string match is the fallback for the callers
    that have no header to read (the delete guard reconstructs a committed load; the plan runs
    before a byte is fetched)."""
    if fam.name == "flux.2-klein":
        # A dim this family does not have (a FLUX.2-dev file, a misread) falls through to the strings rather than
        # guessing, exactly as the base-size guard fails open on one.
        if inner_dim == _FLUX2_KLEIN_9B_INNER_DIM:
            return _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS
        if inner_dim == _FLUX2_KLEIN_4B_INNER_DIM:
            return fam.sd_cpp_text_encoders
        identity = f"{repo_id or ''}/{gguf_filename or ''}".lower()
        # Match the size token on its own: klein-BASE-9B is 9B too, and matching "klein-9b" literally handed it the 4B
        # text encoder.
        if _token_in_needle("9b", identity) or "klein9b" in identity:
            return _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS
    return fam.sd_cpp_text_encoders


# FLUX.2 sizes the adaLN modulation projection as (6 * inner_dim, inner_dim), and inner_dim is the only thing that
# differs between the klein sizes and dev. sd.cpp keys its own FLUX.2 version detection on this same tensor, and GGUF
# metadata cannot help: our files and leejet's carry no kv pairs at all, and city96/orabazes write
# general.architecture = "flux" for FLUX.1 and FLUX.2 alike.
_FLUX2_PROBE_TENSOR = "double_stream_modulation_img.lin.weight"
_FLUX2_INNER_DIMS = {
    3072: "FLUX.2-klein-4B / klein-base-4B",
    4096: "FLUX.2-klein-9B / klein-base-9B",
    6144: "FLUX.2-dev",
}
_FLUX2_KLEIN_4B_INNER_DIM = 3072
_FLUX2_KLEIN_9B_INNER_DIM = 4096
_FLUX2_BASE_INNER_DIM = {
    "black-forest-labs/flux.2-klein-4b": 3072,
    "black-forest-labs/flux.2-klein-base-4b": 3072,
    "black-forest-labs/flux.2-klein-9b": 4096,
    "black-forest-labs/flux.2-klein-base-9b": 4096,
    "black-forest-labs/flux.2-dev": 6144,
}


class _HeaderTensor(NamedTuple):
    """The two fields a FLUX.2 size probe reads off a GGUF tensor table entry."""

    name: str
    shape: Sequence[int]


def flux2_base_inner_dim(base_repo: Optional[str]) -> Optional[int]:
    """The ``inner_dim`` a FLUX.2 base config expects, or None when the repo is not one we map.
    Keyed on UPSTREAM ids, reached through ``canonical_base``: a known ungated mirror is
    byte-identical to what it copies, so it maps back and is checked exactly like its upstream.
    Anything else -- a local path, a third-party repack, a base we do not ship -- misses, and
    every caller fails OPEN on the None rather than guessing."""
    return _FLUX2_BASE_INNER_DIM.get(canonical_base(base_repo or "").lower())


def _flux2_inner_dim_from_tensors(tensors) -> Optional[int]:
    """``inner_dim`` from a parsed GGUF tensor table, or None when the probe tensor is absent."""
    for t in tensors:
        if t.name == _FLUX2_PROBE_TENSOR or t.name.endswith("." + _FLUX2_PROBE_TENSOR):
            # GGUF stores dims reversed relative to torch, so the input dim leads. A missing or non-positive dim is a
            # parse that went wrong, and "0" would be compared against the base as a real answer and refuse a valid
            # pick; say nothing instead.
            dim = int(t.shape[0]) if len(t.shape) else 0
            return dim if dim > 0 else None
    return None


def gguf_flux2_inner_dim(path) -> Optional[int]:
    """``inner_dim`` of a FLUX.2 GGUF, read from its header, or None if it cannot be determined."""
    try:
        from gguf import GGUFReader
        return _flux2_inner_dim_from_tensors(GGUFReader(str(path)).tensors)
    except Exception:
        return None


def gguf_flux2_inner_dim_from_header(header: bytes) -> Optional[int]:
    """``inner_dim`` from the leading bytes of a FLUX.2 GGUF, or None (see ``_gguf_header_tensors``)."""
    tensors = _gguf_header_tensors(header)
    return None if tensors is None else _flux2_inner_dim_from_tensors(tensors)


def gguf_tensor_names_from_header(header: bytes) -> Optional[list[str]]:
    """Every tensor name in the table at the start of a GGUF, or None when the prefix does not hold the whole
    table (so a missing name is never read as absent)."""
    tensors = _gguf_header_tensors(header)
    return None if tensors is None else [str(t.name) for t in tensors]


def _gguf_header_tensors(header: bytes) -> Optional[list]:
    """The tensor table (name and dims of every tensor) read from the leading bytes of a GGUF, or None.

    Lets the selection-time preflight range-read a few hundred KiB over HTTP instead of pulling the
    whole multi-GB checkpoint: the tensor table it needs sits in the first ~15 KiB. Same parser as
    ``gguf_flux2_inner_dim``, with the two things a PREFIX changes. ``_build_tensors`` is skipped,
    because the base class builds a numpy view over every tensor's DATA, which a prefix does not
    carry, and satisfying those reads would allocate the whole declared checkpoint (tens of GiB);
    only the name and shape from the table are wanted, and both are already parsed. And ``_get``
    REFUSES a read past the end instead of returning short or zero-filled data, since the table is
    read field by field and a prefix cutting between a tensor's name and its dims would otherwise
    hand back a zero shape, a wrong answer rather than a missing one, which would refuse a perfectly
    valid pick.

    None on anything unreadable: too short a prefix, a non-GGUF file, an absent probe tensor.
    """
    if not header:
        return None
    tmp_path = None
    reader = None
    try:
        import numpy as np
        from gguf import GGUFReader

        class _HeaderOnlyGGUFReader(GGUFReader):  # type: ignore[misc, valid-type]
            """``GGUFReader`` over a file that holds only the header."""

            def _get(
                self,
                offset,
                dtype,
                count = 1,
                override_order = None,
            ):
                itemsize = int(np.empty([], dtype = dtype).itemsize)
                if int(offset) < 0 or int(offset) + itemsize * int(count) > len(self.data):
                    raise ValueError("GGUF header is truncated")
                return super()._get(offset, dtype, count, override_order)

            def _build_tensors(self, start_offs, fields):
                # ``field.name`` is the decoded tensor name and parts[3] its dims, both already read from real bytes
                # by ``_get_tensor_info_field``.
                self.tensors = [_HeaderTensor(field.name, field.parts[3]) for field in fields]

        # GGUFReader memory-maps a path, so the prefix has to land on disk. Bounded by the caller's range request, so
        # this is KiB, never the checkpoint.
        fd, tmp_path = tempfile.mkstemp(suffix = ".gguf-header")
        with os.fdopen(fd, "wb") as fh:
            fh.write(header)
        reader = _HeaderOnlyGGUFReader(tmp_path)
        # Belt and braces on top of the refusing ``_get``: ``data_offset`` is where the table ends, so a table that
        # ran past the prefix is not one to read a shape out of. The alignment slack is for a header-only file, whose
        # padding was never written.
        if reader.data_offset > len(header) + min(int(reader.alignment), 4096):
            return None
        return list(reader.tensors)
    except Exception:  # noqa: BLE001 - an unreadable header is not a verdict
        return None
    finally:
        # Drop the mmap before unlinking: Windows refuses to delete a mapped file.
        del reader
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def flux2_mismatch_reason(
    gguf_name: str, base_repo: str, got: Optional[int], want: Optional[int]
) -> Optional[str]:
    """Why this FLUX.2 GGUF cannot load against this base, or None when they agree. One message for
    all three checks on the pairing (the plan, the pre-eviction preflight and the loader's
    backstop), so the user reads the same sentence wherever it is caught. ``gguf_name`` is a
    display name the caller has already reduced to a basename. Returns None on any unknown -- an
    unreadable header, an unmapped base -- so every caller fails open."""
    if want is None or got is None or want == got:
        return None
    return (
        f"'{gguf_name}' is a "
        f"{_FLUX2_INNER_DIMS.get(got, f'FLUX.2 variant with inner_dim {got}')} "
        f"checkpoint, but it is being loaded against '{base_repo}', which is "
        f"{_FLUX2_INNER_DIMS.get(want, f'inner_dim {want}')}. Pass base_repo for the matching "
        f"variant, or pick the GGUF that matches the selected model."
    )


def assert_flux2_gguf_matches_base(fam, base_repo: str, gguf_path) -> None:
    """Fail early, and legibly, when a FLUX.2 GGUF is paired with a different-size base config.
    Without this the mismatch surfaces from inside the GGUF quantizer as a bare shape error
    ("expected torch.Size([18432, 3072]), decodes to (24576, 4096)") that names neither the file
    nor the repo. Fail-open by construction: any unreadable file, non-FLUX.2 tensor set, or
    unmapped base leaves the load exactly as it was. The LAST of three checks on the same
    pairing, and the only one that opens the downloaded file: ``diffusion_compat`` runs the same
    comparison off a range-read header at plan time and again before the resident pipeline is
    torn down, so a mismatch normally never reaches here."""
    if gguf_path is None or not str(getattr(fam, "name", "")).startswith("flux.2"):
        return
    # Short-circuit on the free half: an unmapped base fails open anyway, so a mirror id must not cost a memory-map of
    # the whole checkpoint.
    want = flux2_base_inner_dim(base_repo)
    if want is None:
        return
    reason = flux2_mismatch_reason(
        Path(str(gguf_path)).name, base_repo, gguf_flux2_inner_dim(gguf_path), want
    )
    if reason is not None:
        raise ValueError(reason)


_FLUX3_FLOW_MAP_KEYS = ("ref_time_in.", "target_timestep_embedder.")


def _checkpoint_tensor_names(path) -> Optional[list[str]]:
    """Tensor names of a GGUF or safetensors file without loading any tensor; None when unreadable."""
    try:
        name = str(path)
        if name.lower().endswith(".gguf"):
            from gguf import GGUFReader
            return [t.name for t in GGUFReader(name, "r").tensors]
        from safetensors import safe_open

        with safe_open(name, "pt") as handle:
            return list(handle.keys())
    except Exception:  # noqa: BLE001 -- fail open: the loader reports its own error
        return None


def flux3_comfy_declared(names: Optional[list[str]], metadata: Optional[dict]) -> bool:
    """Whether a safetensors file declares ComfyUI quantization (per-layer ``.comfy_quant`` tensors or the
    ``_quantization_metadata`` header table): BFL's own ComfyUI int8 / nvfp4 files do, its ``variants/`` packages
    do not. Matches what diffusion_comfy_quant.scan_comfy_quant reads as a declaration."""
    if any(str(n).endswith(".comfy_quant") for n in names or ()):
        return True
    return isinstance(metadata, dict) and bool(metadata.get("_quantization_metadata"))


def _checkpoint_comfy_declared(path) -> bool:
    """:func:`flux3_comfy_declared` for a local safetensors file; False when unreadable or not safetensors."""
    name = str(path)
    if not name.lower().endswith(".safetensors"):
        return False
    try:
        from safetensors import safe_open
        with safe_open(name, "pt") as handle:
            return flux3_comfy_declared(list(handle.keys()), handle.metadata())
    except Exception:  # noqa: BLE001 -- fail open: the loader reports its own error
        return False


def flux3_variant_mismatch_reason(
    family_name: str,
    filename: str,
    names: Optional[list[str]],
    comfy_declared: bool = False,
) -> Optional[str]:
    """Why a FLUX.3 single file / GGUF cannot run under ``family_name``, or None. The distilled checkpoint is a flow
    map (it carries the target-time embedder ``ref_time_in``) and runs the 16-step distilled graph; the base has none
    and runs CFG. Same shapes otherwise, so a mix-up would load and render garbage instead of failing.
    ``comfy_declared`` (:func:`flux3_comfy_declared`): the scales belong to a ComfyUI quantized file, which the
    ComfyUI loader reads from its own per-layer declarations, so only BFL's undeclared packages are refused."""
    if not names or family_name not in ("flux.3-klein", "flux.3-klein-base"):
        return None
    stripped = [n.split("model.diffusion_model.", 1)[-1] for n in names]
    bfl_native = any(n.startswith(("emb_in.", "txt_in.")) for n in stripped)
    if (
        bfl_native
        and not comfy_declared
        and any(n.endswith((".weight_scale", ".weight_scale_2", ".input_scale")) for n in stripped)
    ):
        # BFL's variants/fp8r and variants/nvfp4: row-major scales in BFL's own layout. The NVFP4 one has the same
        # tensor sizes as a ComfyUI NVFP4 file whose scales are 128x4 tiles, so the ComfyUI loader would decode it
        # silently wrong; refuse instead.
        return (
            f"'{filename}' is one of BFL's quantized FLUX.3 packages (variants/fp8r or variants/nvfp4), whose scale "
            "layout Unsloth does not load. Load the bf16 checkpoint and pick a transformer precision instead."
        )
    flow_map = any(n.startswith(_FLUX3_FLOW_MAP_KEYS) for n in stripped)
    if flow_map and family_name == "flux.3-klein-base":
        return (
            f"'{filename}' is the distilled FLUX.3 Klein checkpoint (it has the flow-map target-time embedder), "
            "but it was loaded as flux.3-klein-base. Load it with family flux.3-klein."
        )
    if not flow_map and family_name == "flux.3-klein":
        return (
            f"'{filename}' is the FLUX.3 Klein base checkpoint (no flow-map target-time embedder), but it was "
            "loaded as the distilled flux.3-klein. Load it with family flux.3-klein-base."
        )
    return None


def assert_flux3_checkpoint_matches_family(fam, checkpoint_path) -> None:
    """Refuse a FLUX.3 single file / GGUF of the other variant before any pipeline is built. Fail-open."""
    name = str(getattr(fam, "name", ""))
    if checkpoint_path is None or not name.startswith("flux.3"):
        return
    reason = flux3_variant_mismatch_reason(
        name,
        Path(str(checkpoint_path)).name,
        _checkpoint_tensor_names(checkpoint_path),
        comfy_declared = _checkpoint_comfy_declared(checkpoint_path),
    )
    if reason is not None:
        raise ValueError(reason)


def resolve_local_gguf_child(repo_root: Path, gguf_filename: str) -> Path:
    """Resolve ``gguf_filename`` (user-supplied) to a file under ``repo_root``, rejecting absolute
    paths and ``..`` escapes."""
    if (
        Path(gguf_filename).is_absolute()
        or PurePosixPath(gguf_filename).is_absolute()
        or gguf_filename.startswith(("/", "\\"))
        or "\\" in gguf_filename
    ):
        raise ValueError("gguf_filename must be a relative path inside the repo.")
    rel = PurePosixPath(gguf_filename)
    if any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("gguf_filename must not contain '', '.', or '..' segments.")
    # resolve symlinks before the containment check (the lexical guards miss a symlink escape)
    repo_real = repo_root.resolve()
    child = repo_root.joinpath(*rel.parts).resolve()
    if child != repo_real and repo_real not in child.parents:
        raise ValueError("gguf_filename must resolve to a file inside the repo.")
    if not child.is_file():
        raise FileNotFoundError(f"'{gguf_filename}' is not a file under {repo_root}.")
    return child
