# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the Inference API."""

from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Annotated, Any, Dict, Literal, Optional, List, Union

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from core.inference.llama_server_args import (
    BATCH_MAX,
    BATCH_MIN,
    CACHE_RAM_MAX_MIB,
    CTX_CHECKPOINTS_MAX,
    PARALLEL_MAX,
    PARALLEL_MIN,
)
from core.inference.video_families import MAX_VIDEO_NUM_FRAMES
from picker.schemas import MAX_CHAT_TEMPLATE_BYTES


class LoadRequest(BaseModel):
    """Request to load a model for inference"""

    model_path: str = Field(..., description = "Model identifier or local path")
    load_request_id: Optional[str] = Field(
        None,
        min_length = 1,
        max_length = 128,
        pattern = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        description = "Opaque client attempt ID for scoped in-flight cancellation",
    )

    force_reload: bool = Field(
        False,
        description = "Start a fresh runtime even when the active settings already match",
    )
    native_path_lease: Optional[str] = Field(
        None, description = "Frontend-visible signed native path grant"
    )
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated models")
    max_seq_length: int = Field(
        0,
        ge = 0,
        le = 1048576,
        description = "Maximum sequence length (0 = model default for GGUF)",
    )
    load_in_4bit: bool = Field(True, description = "Load model in 4-bit quantization")
    is_lora: bool = Field(False, description = "Whether this is a LoRA adapter")
    gguf_variant: Optional[str] = Field(
        None, description = "GGUF quantization variant (e.g. 'Q4_K_M')"
    )
    trust_remote_code: bool = Field(
        False,
        description = "Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust.",
    )
    approved_remote_code_fingerprint: Optional[str] = Field(
        None,
        description = "sha256 fingerprint from the remote-code scan, pinning user approval of this exact custom-code version.",
    )
    chat_template_override: Optional[str] = Field(
        None,
        description = "Custom Jinja2 chat template to use instead of the model's default",
    )

    @field_validator("chat_template_override")
    @classmethod
    def normalize_blank_chat_template_override(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        # Char count is a lower bound on UTF-8 byte length: reject an oversized
        # template before spending work encoding it.
        if len(value) > MAX_CHAT_TEMPLATE_BYTES:
            raise ValueError(f"Chat template exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit.")
        if value.strip() == "":
            return None
        if len(value.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES:
            raise ValueError(f"Chat template exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit.")
        return value

    cache_type_kv: Optional[str] = Field(
        None,
        description = (
            "KV cache data type for both K and V "
            "(e.g. 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl', 'f32')"
        ),
    )
    mlx_kv_bits: Optional[int] = Field(
        None,
        description = (
            "MLX KV cache quantization bit width (8, 6, 5, 4, 3 or 2). MLX takes a bit "
            "width rather than a llama.cpp dtype name, so this is separate from "
            "cache_type_kv. Omit for an unquantized cache. Ignored by non-MLX "
            "backends; a model whose cache layout cannot be quantized reports "
            "the reason instead of applying it."
        ),
    )
    gpu_ids: Optional[List[int]] = Field(
        None,
        description = (
            "GPU placement pool, for example [0, 1]. Omit or pass [] to use "
            "automatic selection. CUDA/ROCm values are physical GPU indices; "
            "Vulkan values are ggml device ordinals. Explicit selection is not "
            "supported on XPU, and physical IDs are unsupported when the parent "
            "visibility mask uses "
            "non-numeric or subdevice entries, including CUDA_VISIBLE_DEVICES "
            "with UUID/MIG entries and ZE_AFFINITY_MASK with subdevice tokens "
            "(for example '0.0,0.1') or FLAT-hierarchy tile handles. For GGUF "
            "models the fitter may pin the smallest subset of this pool that fits."
        ),
    )
    speculative_type: Optional[str] = Field(
        None,
        description = (
            "Speculative decoding mode for GGUF models. Canonical values: "
            "'auto' (platform-aware: DSpark when the model ships a sidecar, "
            "else DFlash when it ships one, else MTP on MTP GGUFs, ngram-mod "
            "fallback for sub-3B), "
            "'mtp' (force draft-mtp only on both GPU and CPU), "
            "'dspark' (force a draft-dspark sidecar), "
            "'dflash' (force a draft-dflash sidecar), "
            "'ngram' (force ngram-mod only), 'mtp+ngram' (force "
            "ngram-mod+draft-mtp chain on both platforms), 'off' (disabled). "
            "Legacy values 'default' (-> auto), 'draft-mtp' (-> mtp), "
            "'draft-dspark' (-> dspark), 'draft-dflash' (-> dflash), "
            "'ngram-mod' (-> ngram), and 'ngram-simple' (kept as-is) are "
            "still accepted. Ignored for non-GGUF models."
        ),
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        ge = 1,
        le = 16,
        description = (
            "Max draft tokens per step for MTP, DSpark or DFlash speculative "
            "decoding (--spec-draft-n-max). Defaults to 2 on GPU and 3 on "
            "CPU/Mac when unset (upstream-bench sweet spot for dense Qwen3.6 "
            "MTP quants, and the measured sweet spot for DFlash too). Only "
            "applied when speculative_type resolves to 'mtp', 'mtp+ngram', "
            "'dspark' or 'dflash'."
        ),
    )
    n_parallel: Optional[int] = Field(
        None,
        ge = PARALLEL_MIN,
        le = PARALLEL_MAX,
        description = (
            "Parallel decode slots for llama-server (--parallel) for this "
            f"load ({PARALLEL_MIN}..{PARALLEL_MAX}). Omit for the server-wide "
            "default set at launch (the --parallel CLI flag). The VRAM fitter "
            "may launch fewer slots to keep the model fully on GPU. Ignored "
            "for non-GGUF models."
        ),
    )
    n_batch: Optional[int] = Field(
        None,
        ge = BATCH_MIN,
        le = BATCH_MAX,
        description = (
            "Logical prompt batch size for llama-server (--batch-size) for "
            f"this load ({BATCH_MIN}..{BATCH_MAX}). Omit for the llama.cpp "
            "default (2048). Ignored for non-GGUF models."
        ),
    )
    n_ubatch: Optional[int] = Field(
        None,
        ge = BATCH_MIN,
        le = BATCH_MAX,
        description = (
            "Physical prompt micro-batch size for llama-server (--ubatch-size) "
            f"for this load ({BATCH_MIN}..{BATCH_MAX}). Omit for the llama.cpp "
            "default (512). llama.cpp caps it at the batch size. Larger values "
            "speed up prompt processing at the cost of compute-buffer VRAM. "
            "Ignored for non-GGUF models."
        ),
    )
    load_mode: Optional[Literal["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]] = Field(
        None,
        description = (
            "How llama-server reads the weights off disk (--load-mode). 'auto' "
            "memory-maps unless a device cannot, 'mmap' forces the mapping, "
            "'mlock' keeps the model in RAM rather than letting it swap or "
            "compress, 'mmap+mlock' does both, 'dio' uses DirectIO where "
            "available and 'none' asks for no special mode. Omit for the "
            "llama.cpp default. The Model Memory settings own host placement, so "
            "'Keep model in GPU memory' replaces this with mmap+mlock and "
            "'Don't reserve system RAM' drops a mode that would hold a full host "
            "copy. Ignored for non-GGUF models."
        ),
    )
    spec_draft_cache_type: Optional[str] = Field(
        None,
        description = (
            "KV cache dtype for the DRAFT model's context "
            "(--spec-draft-type-k / --spec-draft-type-v), for example 'q8_0'. "
            "Separate from cache_type_kv, which is the target model's. Only "
            "reaches the command line when the load attaches a separate draft "
            "model; omit for the llama.cpp default (f16). Ignored for non-GGUF "
            "models."
        ),
    )
    ctx_checkpoints: Optional[int] = Field(
        None,
        ge = 0,
        le = CTX_CHECKPOINTS_MAX,
        description = (
            "Context checkpoints kept per slot (--ctx-checkpoints), which let a "
            "sliding-window model rewind instead of re-processing the prompt. "
            "Omit for the llama.cpp default (32); 0 disables them. "
            "Each costs host memory, and a model without a sliding window "
            "ignores it. Ignored for non-GGUF models."
        ),
    )
    cache_ram: Optional[int] = Field(
        None,
        ge = -1,
        le = CACHE_RAM_MAX_MIB,
        description = (
            "Host memory in MiB llama-server may spend caching prompt state it "
            "has evicted from a slot (--cache-ram). Omit for the llama.cpp "
            "default (8192); 0 disables the cache and -1 lifts the limit. "
            "Ignored for non-GGUF models."
        ),
    )
    tensor_parallel: bool = Field(
        False,
        description = (
            "Split the model across GPUs by tensor (--split-mode tensor) "
            "instead of by layer for GGUF models. Only affects multi-GPU "
            "setups, where it can make generation significantly faster. "
            "No effect on a single GPU. Ignored for non-GGUF models."
        ),
    )
    disable_vision: bool = Field(
        False,
        description = (
            "Load a vision-capable GGUF without its multimodal projector, as a "
            "text-only model. Frees the VRAM the projector would hold, at the cost "
            "of image input, which is off for the session; text generation is "
            "unaffected. Ignored for models with no vision projector, and for "
            "non-GGUF models."
        ),
    )
    gpu_memory_mode: Literal["auto", "manual"] = Field(
        "auto",
        description = (
            "GPU memory strategy for GGUF models. 'auto' (default): Unsloth "
            "selects GPUs and caps context to fit VRAM. 'manual': you own the "
            "offload. Leave gpu_layers at -1 (Auto) to hand memory management to "
            "llama.cpp's --fit (no device masking, no context auto-reduce, no "
            "gpu-layer/tensor-split planning); set gpu_layers >= 0 to pin layers "
            "and n_cpu_moe yourself (--fit off), with tensor_parallel still "
            "applying (split by free VRAM unless tensor_split is set, no planner). "
            "Ignored for non-GGUF."
        ),
    )
    gpu_layers: int = Field(
        -1,
        ge = -1,
        description = (
            "Manual mode only: number of layers to offload to the GPU "
            "(--gpu-layers, with --fit off). A value >= the model's layer count "
            "offloads all of them. -1 = Auto: hand layer + context sizing to "
            "llama.cpp's --fit. Ignored unless gpu_memory_mode is 'manual'."
        ),
    )
    cpu_fallback: bool = Field(
        False,
        description = (
            "Replay a previously recovered automatic Vulkan load in its managed CPU-only "
            "runtime. Used when restoring that model after a failed switch."
        ),
    )
    n_cpu_moe: int = Field(
        0,
        ge = 0,
        description = (
            "Manual mode only: keep the first N MoE expert layers on the CPU "
            "(--n-cpu-moe) to save VRAM on MoE models. 0 = none, N = number of "
            "MoE layers offloaded (the backend offsets past any leading dense "
            "layers). Ignored unless gpu_memory_mode is 'manual' with gpu_layers >= 0."
        ),
    )
    tensor_split: Optional[List[float]] = Field(
        None,
        description = (
            "Manual mode only: relative share of the model per GPU (--tensor-split), "
            "in the order of the GPUs in use, e.g. [2, 1] for 2:1. Omit it to let "
            "llama.cpp use its default, which splits by free VRAM. Any list given is "
            "passed through as-is, so send [1, 1] to force an even split. Ignored "
            "unless gpu_memory_mode is 'manual' with gpu_layers >= 0."
        ),
    )

    @field_validator("n_batch", "n_ubatch", "ctx_checkpoints", "cache_ram", mode = "before")
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        # bool subclasses int and pydantic parses non-strictly, so `true` arrives as 1 and
        # the load launches --batch-size 1, which llama-server aborts on: a 500 rather than
        # a 422. Mirrors ModelOverrideRequest._no_booleans so /load and /settings agree.
        # Kept off the annotation: an Annotated BeforeValidator stops the Field constraints
        # folding into the int core schema, and they leak into OpenAPI as ge/le.
        if isinstance(value, bool):
            raise ValueError("Expected a number, got a boolean.")
        return value

    @field_validator("tensor_split")
    @classmethod
    def _reject_degenerate_tensor_split(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        # A negative / non-finite / all-zero split is silently dropped at launch
        # (stored as None) yet still compared raw in the reload dedupe, so an
        # identical Apply reloads forever. Reject it up front; [] = no split.
        if not value:
            return value
        import math

        if any((not math.isfinite(v)) or v < 0 for v in value):
            raise ValueError("tensor_split entries must be finite and non-negative")
        if sum(value) <= 0:
            raise ValueError("tensor_split must have a positive total")
        return value

    llama_extra_args: Optional[List[str]] = Field(
        None,
        description = (
            "Extra arguments forwarded verbatim to llama-server for GGUF models. "
            "One token per list entry, e.g. ['--top-k', '20', '--seed', '42']. "
            "Unsloth-managed flags (model identity, port, context length, GPU placement, "
            "auth, UI/server mode) are rejected. Ignored for non-GGUF models."
        ),
    )
    force_cancel_active: bool = Field(
        False,
        description = (
            "Stop chats still generating instead of refusing with 409. A load "
            "replaces the llama-server every open conversation decodes on."
        ),
    )


class UnloadRequest(BaseModel):
    """Request to unload a model"""

    model_path: str = Field(..., description = "Model identifier to unload")
    cancel_load_request_id: Optional[str] = Field(
        None,
        min_length = 1,
        max_length = 128,
        pattern = r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        description = ("Cancel only this in-flight load attempt; never unload a resident model"),
    )
    force_cancel_active: bool = Field(
        False,
        description = (
            "Stop chats still generating instead of refusing with 409. An "
            "unload takes away the llama-server they are decoding on."
        ),
    )


class SearchImagesLookupRequest(BaseModel):
    subjects: list[str] = Field(
        ...,
        min_length = 1,
        max_length = 5,
        description = "Specific things to fetch one picture each for.",
    )


class TranscribeRequest(BaseModel):
    """Speech-to-text request for the dictation STT sidecar."""

    audio: str = Field(..., description = "Base64-encoded audio (any common format)")
    model: Optional[str] = Field(None, description = "STT model id; defaults server-side")
    language: Optional[str] = Field(None, description = "BCP-47 language, or 'auto'/None to detect")
    fast: bool = Field(
        False,
        description = "Use low-latency single-candidate decoding for dictation",
    )
    engine: Optional[str] = Field(
        None,
        description = "STT engine: 'transformers' (default) or 'gguf' (whisper.cpp)",
    )


class SttLoadRequest(BaseModel):
    """Warm the STT sidecar with a model without transcribing."""

    model: Optional[str] = Field(None, description = "STT model id; defaults server-side")
    engine: Optional[str] = Field(
        None,
        description = "STT engine: 'transformers' (default) or 'gguf' (whisper.cpp)",
    )


class ValidateModelRequest(BaseModel):
    """Check whether an identifier resolves to a ModelConfig; does NOT load weights."""

    model_path: str = Field(..., description = "Model identifier or local path")
    native_path_lease: Optional[str] = Field(
        None, description = "Frontend-visible signed native path grant"
    )
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated models")
    llama_extra_args: Optional[List[str]] = Field(
        None,
        description = (
            "Pass-through llama-server args the follow-up /load will send. Sized with, "
            "not validated here: a --ctx-size or cache override changes the memory this "
            "preflight estimates, so omitting it would approve a different command from "
            "the one that runs."
        ),
    )
    gguf_variant: Optional[str] = Field(
        None, description = "GGUF quantization variant (e.g. 'Q4_K_M')"
    )
    # Intended load settings so validate's coexistence check matches the follow-up
    # /load; defaults preserve old behavior for callers that omit them.
    max_seq_length: int = Field(0, ge = 0, le = 1048576)
    load_in_4bit: bool = Field(True)
    cache_type_kv: Optional[str] = Field(None)
    tensor_parallel: bool = Field(False)
    # Sized with, like the other intended load settings above: the follow-up /load
    # opens no projector when this is set, so a preflight that charges for one would
    # refuse a load that then fits.
    disable_vision: bool = Field(False)
    gpu_ids: Optional[List[int]] = Field(None)
    gpu_memory_mode: Literal["auto", "manual"] = Field(
        "auto",
        description = (
            "GGUF GPU-memory strategy intended for the follow-up load. Manual "
            "placement bypasses the training coexistence estimate: Auto layers "
            "delegate fitting to llama.cpp, while explicit layers are user-owned."
        ),
    )
    gpu_layers: int = Field(
        -1,
        ge = -1,
        description = (
            "Layer count intended for the follow-up load, so the coexistence estimate "
            "sizes like /load. Only 0 changes the verdict: a zero-layer DiffusionGemma "
            "split places no layers on any device, so it cannot compete with training "
            "for VRAM. -1 (Auto) keeps the previous behaviour for callers that omit it."
        ),
    )
    n_parallel: Optional[int] = Field(
        None,
        ge = PARALLEL_MIN,
        le = PARALLEL_MAX,
        description = (
            "Parallel decode slots intended for the follow-up load, so the "
            "coexistence estimate sizes the KV cache like /load. Omit for the "
            "server-wide --parallel default."
        ),
    )
    n_batch: Optional[int] = Field(
        None,
        ge = BATCH_MIN,
        le = BATCH_MAX,
        description = (
            "Batch size (--batch-size) intended for the follow-up load, so the "
            "coexistence estimate sizes the compute buffer like /load."
        ),
    )
    n_ubatch: Optional[int] = Field(
        None,
        ge = BATCH_MIN,
        le = BATCH_MAX,
        description = (
            "Micro-batch size (--ubatch-size) intended for the follow-up load, "
            "so the coexistence estimate sizes the compute buffer like /load."
        ),
    )
    speculative_type: Optional[str] = Field(
        None,
        description = (
            "Speculative mode intended for the follow-up load. The estimate is "
            "mode-dependent -- the drafter it charges differs by kind, and a "
            "DSpark sidecar is ~11 GB -- so omitting it makes this preflight "
            "disagree with /load in both directions."
        ),
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        ge = 1,
        le = 16,
        description = "Draft depth intended for the follow-up load; sizes the draft KV.",
    )
    spec_draft_cache_type: Optional[str] = Field(
        None,
        description = (
            "Draft KV cache dtype intended for the follow-up load. Sent so this "
            "preflight strips the same inherited draft-cache flags /load would, "
            "and so approves the command the load actually runs. The coexistence "
            "estimate prices the drafter's weights but not its KV, so the value "
            "itself does not move the number."
        ),
    )
    ctx_checkpoints: Optional[int] = Field(
        None,
        ge = 0,
        le = CTX_CHECKPOINTS_MAX,
        description = (
            "Checkpoints (--ctx-checkpoints) intended for the follow-up load, so "
            "the coexistence estimate sizes the SWA cache like /load. Each one is "
            "a per-slot snapshot that scales with the slot's context, so a load "
            "asking for them needs materially more memory than one that does not."
        ),
    )
    include_context_length: bool = Field(
        False,
        description = "Also read the native context length from the local GGUF header. "
        "Opt-in so the normal load preflight doesn't pay for a cache scan it doesn't need.",
    )
    include_chat_template: bool = Field(
        False,
        description = "Also read the embedded chat template from the local GGUF header, so a "
        "native (picked / drag-drop) file's default template can be shown before it is loaded. "
        "Opt-in and, like include_context_length, a metadata-only probe that skips the training "
        "guard. Only the leased file's own embedded template is read, never sibling sidecars.",
    )

    _no_booleans = field_validator("n_batch", "n_ubatch", "ctx_checkpoints", mode = "before")(
        LoadRequest._no_booleans.__func__
    )


class TransformersUpgradeInfo(BaseModel):
    """A model architecture no installed transformers ships, but a newer release does."""

    model_type: str = Field(
        ..., description = "config.json model_type unknown to every installed transformers"
    )
    pypi_version: Optional[str] = Field(
        None, description = "Latest transformers release on PyPI at check time"
    )
    supported_in_pypi: bool = Field(
        False,
        description = "True if the latest PyPI release ships this model_type; Unsloth can "
        "install it into a persistent sidecar after user consent.",
    )
    supported_in_main: bool = Field(
        False,
        description = "True if transformers GitHub main ships this model_type (dev-only; "
        "not installable through Unsloth yet).",
    )


class TransformersUpgradeCheckRequest(BaseModel):
    """Ask whether loading a model needs a newer transformers than any installed overlay."""

    model_name: str = Field(
        ...,
        min_length = 1,
        max_length = 1024,
        description = "Model identifier, local path or checkpoint directory to check.",
    )
    hf_token: Optional[str] = Field(
        None, description = "HuggingFace token, so gated repos resolve their config.json"
    )
    # Cache pin, in the same four fields /models/remote-code-scan takes and resolved by
    # the same precedence: a cached model loads from its pinned snapshot, whose
    # config.json can name a different architecture than the repo's current one.
    prefer_local_cache: bool = Field(
        False,
        description = "Inspect the cached snapshot rather than the Hub repo, when one is pinned.",
    )
    model_local_path: Optional[str] = Field(
        None,
        max_length = 4096,
        description = "Cache directory the caller selected for this model, if any.",
    )
    model_snapshot_path: Optional[str] = Field(
        None,
        max_length = 4096,
        description = "Exact snapshot the load is pinned to; takes precedence over "
        "model_local_path, as it does for the remote-code scan.",
    )
    model_snapshot_repo_id: Optional[str] = Field(
        None,
        max_length = 1024,
        description = "Repository the pinned snapshot belongs to, when it differs from model_name.",
    )
    resume_run_id: Optional[str] = Field(
        None,
        max_length = 128,
        description = "Run this check precedes a resume of. Lets the answer say whether "
        "installing would strand that checkpoint's exact 4-bit resume.",
    )


class TransformersUpgradeCheckResponse(BaseModel):
    """Upgrade + quantization preflight for a load that does not run /validate.

    /validate answers this for a chat load as part of a much larger check (GGUF
    placement, VRAM coexistence, security review). Training needs the same two answers
    on their own, before starting a worker that would die at model load with an
    unrecognized-architecture error.
    """

    model_name: str = Field(..., description = "The identifier that was checked")
    requires_transformers_upgrade: bool = Field(
        False,
        description = "True when the architecture is unknown to every installed transformers "
        "but a newer transformers ships it; the caller should raise the install consent "
        "dialog before starting the load.",
    )
    transformers_upgrade: Optional[TransformersUpgradeInfo] = Field(
        None,
        description = "Details for the consent dialog; set only when "
        "requires_transformers_upgrade is true.",
    )
    requires_trust_remote_code: bool = Field(
        False,
        description = "Whether the model can load on the CURRENT transformers through its own "
        "repo code, so a declined (or unavailable) install still has a path forward.",
    )
    latest_tier_active: bool = Field(
        False,
        description = "Whether the latest-transformers sidecar already routes this model.",
    )
    forces_16bit: bool = Field(
        False,
        description = "Whether a run started now would load 16-bit rather than bnb 4-bit: true "
        "when the latest sidecar already routes the model, and when an install-only upgrade "
        "would put it there. Lets the UI state the real precision before the run starts.",
    )
    install_breaks_exact_resume: bool = Field(
        False,
        description = "Set only for a resume_run_id: installing the offered release would "
        "activate the latest sidecar, which permanently refuses that checkpoint's attested "
        "4-bit model load mode. The caller must not offer the install when the run can "
        "start without it.",
    )


class ValidateModelResponse(BaseModel):
    """Result of model validation.

    valid == True means from_identifier() succeeded and GGUF/LoRA/vision flags are available.
    """

    valid: bool = Field(..., description = "Whether the model identifier looks valid")
    message: str = Field(..., description = "Human-readable validation message")
    identifier: Optional[str] = Field(None, description = "Resolved model identifier")
    display_name: Optional[str] = Field(None, description = "Display name derived from identifier")
    is_gguf: bool = Field(False, description = "Whether this is a GGUF model (llama.cpp)")
    is_diffusion: bool = Field(
        False, description = "Whether this is a block-diffusion model (DiffusionGemma)"
    )
    diffusion_unknown: bool = Field(
        False,
        description = "Whether the diffusion check came back inconclusive: the GGUF is not "
        "downloaded yet (or its header is unreadable) and its name does not carry the "
        "DiffusionGemma family, so is_diffusion == False here means 'not known to be "
        "diffusion', NOT 'known to be an ordinary GGUF'. Callers that choose a GPU-layer "
        "split before the load must treat this as possibly-diffusion.",
    )
    is_lora: bool = Field(False, description = "Whether this is a LoRA adapter")
    is_vision: bool = Field(False, description = "Whether this is a vision-capable model")
    requires_trust_remote_code: bool = Field(
        False,
        description = "Whether the model defaults require trust_remote_code to be enabled for loading.",
    )
    requires_security_review: bool = Field(
        False,
        description = "Whether Hugging Face's security scan flagged unsafe files (e.g. a "
        "malicious pickle), so the load is hard-blocked pending review.",
    )
    context_length: Optional[int] = Field(
        None,
        description = "Native training context length, read from the GGUF header when the file "
        "is already downloaded locally; None for non-GGUF, gated, or not-yet-downloaded models.",
    )
    layer_count: Optional[int] = Field(
        None,
        description = "Total layer count (GGUF block_count), the manual gpu-layers ceiling, read "
        "from the header alongside context_length; None when not read.",
    )
    moe_layer_count: Optional[int] = Field(
        None,
        description = "MoE expert-layer count (the manual --n-cpu-moe ceiling), read from the GGUF "
        "header alongside context_length; 0 for dense models, None when not read.",
    )
    chat_template: Optional[str] = Field(
        None,
        description = "Embedded GGUF chat template, read from the header when include_chat_template "
        "is set (native lease-backed picks); None for non-GGUF, over-cap, or not-read templates.",
    )
    # Additive fields; the consuming consent dialog ships in a follow-up frontend PR.
    requires_transformers_upgrade: bool = Field(
        False,
        description = "True when the model's architecture is unknown to every installed "
        "transformers but a newer transformers ships it; the UI should offer the "
        "install-latest-transformers consent dialog (or the dev-only notice).",
    )
    transformers_upgrade: Optional[TransformersUpgradeInfo] = Field(
        None,
        description = "Details for the transformers-upgrade dialog; set only when "
        "requires_transformers_upgrade is true.",
    )


class EstimateMemoryRequest(BaseModel):
    """Settings a Load-Model panel is about to submit, priced before it submits them.

    Every field mirrors the load request it previews, so the estimate answers for the
    command that would actually run. Header-only: nothing is read, touched or loaded.
    """

    model_path: str = Field(..., description = "Model identifier or local path")
    gguf_variant: Optional[str] = Field(
        None, description = "GGUF quantization to price (e.g. Q4_K_M); the picked variant."
    )
    hf_token: Optional[str] = Field(None, description = "Token for gated repositories")
    native_path_lease: Optional[str] = Field(
        None,
        description = "Lease for a picked / drag-dropped .gguf, as /validate takes one.",
    )
    n_ctx: Optional[int] = Field(
        None,
        ge = 0,
        description = "Context length to price (--ctx-size). 0 or omitted prices the "
        "model's native context, which is what an Auto load asks for.",
    )
    cache_type_kv: Optional[str] = Field(
        None,
        description = "KV cache dtype to price. The biggest lever on this estimate at "
        "long contexts, so omitting it prices a load nobody asked for.",
    )
    n_parallel: Optional[int] = Field(
        None,
        ge = PARALLEL_MIN,
        le = PARALLEL_MAX,
        description = "Serving slots (--parallel); scales both the cache and the buffers.",
    )
    n_batch: Optional[int] = Field(None, ge = BATCH_MIN, le = BATCH_MAX)
    n_ubatch: Optional[int] = Field(None, ge = BATCH_MIN, le = BATCH_MAX)
    ctx_checkpoints: Optional[int] = Field(None, ge = 0, le = CTX_CHECKPOINTS_MAX)
    speculative_type: Optional[str] = Field(
        None, description = "Speculative mode; decides which drafter's weights are charged."
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        ge = 1,
        le = 16,
        description = "Draft depth. Sizes the drafter's rollback state, so it moves the "
        "estimate for every mode that loads a separate drafter.",
    )
    spec_draft_cache_type: Optional[str] = Field(
        None,
        description = "Draft KV cache dtype. The drafter keeps its own cache, and at a "
        "long context that is GB, not a rounding error.",
    )
    tensor_parallel: bool = Field(False, description = "Whether tensor mode is requested")
    disable_vision: bool = Field(
        False, description = "Vision off, so an image projector's bytes are not charged."
    )
    gpu_memory_mode: Optional[str] = Field(
        None, description = "'auto' or 'manual'; manual splits the weights by gpu_layers."
    )
    gpu_layers: Optional[int] = Field(
        None, ge = 0, description = "Layers pinned to the GPU under manual placement."
    )
    n_cpu_moe: Optional[int] = Field(
        None,
        ge = 0,
        description = "Expert layers held on the CPU (--n-cpu-moe). Not priced; echoed "
        "back so the panel can say the GPU figure reads high.",
    )
    selected_gpu_ids: Optional[List[int]] = Field(
        None,
        description = "GPUs the load is pinned to. Tensor mode replicates its compute "
        "buffers per device, so the pinned count changes the footprint.",
    )
    llama_extra_args: Optional[List[str]] = Field(
        None,
        description = "Pass-through llama-server flags. Read, not just carried: -c, "
        "-nkvo, --swa-full and the cache-type flags all move this estimate.",
    )

    _no_booleans = field_validator(
        "n_batch", "n_ubatch", "ctx_checkpoints", "n_ctx", mode = "before"
    )(LoadRequest._no_booleans.__func__)


class EstimateMemoryResponse(BaseModel):
    """Itemized memory an inference load would occupy, or why it could not be sized."""

    available: bool = Field(..., description = "Whether a breakdown could be produced at all.")
    reason: Optional[str] = Field(
        None,
        description = "Cause when available is false: 'not_gguf', 'not_downloaded', "
        "'unsupported_source' or 'unsizable'.",
    )
    weights_bytes: int = Field(0, description = "Resident model files: weights, projector, drafter")
    kv_bytes: int = Field(0, description = "KV cache at the requested context and slots")
    compute_bytes: int = Field(0, description = "Compute / graph buffers, flat plus context-linear")
    drafter_runtime_bytes: int = Field(
        0,
        description = "A separate drafter's own KV cache and rollback state, on top of "
        "its file in weights_bytes. Included in total_bytes, and in gpu_bytes unless the "
        "drafter is CPU-pinned. Reported separately so the itemization accounts for it.",
    )
    drafter_runtime_gpu_bytes: int = Field(
        0,
        description = "The share of drafter_runtime_bytes that lands on the GPU. Not a "
        "placement flag, because the term is not placed as one piece: under MTP the "
        "target-side verification state follows the TARGET cache (so --no-kv-offload "
        "moves it) while the draft cache follows the drafter (so --spec-draft-ngl 0 "
        "moves that instead), and the two can go different ways in the same load.",
    )
    projector_runtime_bytes: int = Field(
        0,
        description = "The vision encoder's buffers, about 0.4x the projector file on top "
        "of it. Included in total_bytes, and in gpu_bytes unless --no-mmproj-offload "
        "keeps the projector in host RAM.",
    )
    drafter_kv_unsized: bool = Field(
        False,
        description = "True when a drafter is charged whose cache could not be sized: "
        "--spec-draft-hf names a repository whose header is not on this disk. Its weights "
        "are in weights_bytes but its context-scaled cache is missing, so total_bytes is a "
        "lower bound.",
    )
    adapters_unsized: bool = Field(
        False,
        description = "True when a pass-through --lora / --lora-scaled / --control-vector / "
        "--control-vector-scaled names a file that could not be stat'd. llama.cpp loads every "
        "one of them into resident tensors on top of the base model, so those bytes are "
        "missing from total_bytes and it is a lower bound.",
    )
    total_bytes: int = Field(0, description = "Weights + KV + compute, wherever they land")
    gpu_bytes: int = Field(
        0, description = "The share of total_bytes that lands on the GPU under this offload"
    )
    kv_estimable: bool = Field(
        True,
        description = "False when the GGUF header lacks the attention dims needed to "
        "size the cache. kv_bytes is then 0 meaning UNKNOWN and total_bytes is a lower "
        "bound; rendering it as a confident total is what this flag exists to prevent.",
    )
    kv_on_gpu: bool = Field(
        True, description = "False under --no-kv-offload, which moves the cache to host RAM"
    )
    n_ctx: int = Field(0, description = "Context length the estimate actually priced")
    cache_type_kv: Optional[str] = Field(
        None, description = "KV dtype the estimate priced, after flags and fallbacks resolve"
    )
    n_parallel: int = Field(1, description = "Slots the estimate priced, after the launch clamps")
    layer_count: Optional[int] = Field(None, description = "GGUF block_count, when readable")
    gpu_layers: Optional[int] = Field(
        None, description = "Layers charged to the GPU; null under automatic placement"
    )
    moe_offload_unmodelled: bool = Field(
        False,
        description = "True when --n-cpu-moe is set: experts move per-tensor, not "
        "per-block, so the GPU/host split above ignores it and reads high.",
    )


class MemoryEstimate(BaseModel):
    """The canonical answer to "what would this load occupy".

    Studio grew two routes that answer this -- ``POST /inference/estimate-memory``
    for the Load Model panel and ``GET /models/kv-cache-estimate`` for the Hub
    memory bar. They share the ``_gguf_memory_breakdown`` planner, so their
    arithmetic already agrees; this is the shared CONTRACT, so their vocabulary
    agrees too. Both legacy routes are projections of this model and keep their
    own shapes exactly.

    The one thing this model deliberately does not have is a field called
    ``weights_bytes``. That name means different things on the two legacy routes
    -- every resident file on one, the quant file alone on the other -- and it is
    the same type on both, so nothing catches a caller reading the wrong one.
    It is replaced here by two fields that each say which they are, and it is
    absent rather than redefined so that no future reader can pick it up and
    guess. See ``core/inference/memory_contract.py`` for the projections.
    """

    available: bool = Field(..., description = "Whether a breakdown could be produced at all.")
    reason: Optional[str] = Field(
        None,
        description = "Cause when available is false: 'not_gguf', 'not_downloaded', "
        "'unsupported_source' or 'unsizable'.",
    )

    # --- the two meanings that used to collide under `weights_bytes` ---
    quant_file_bytes: int = Field(
        0,
        description = "The selected GGUF quant file ALONE, as it sits on disk. This is "
        "the figure a download size or a weights segment should be drawn from, because "
        "it is the number the user already saw beside the model.",
    )
    resident_files_bytes: int = Field(
        0,
        description = "Every file this launch makes resident: the quant file PLUS "
        "whichever projector and drafter it opens. Always >= quant_file_bytes. This is "
        "the figure an itemized footprint should be drawn from.",
    )

    kv_bytes: int = Field(0, description = "KV cache at the requested context and slots")
    compute_bytes: int = Field(0, description = "Compute / graph buffers, flat plus context-linear")
    drafter_runtime_bytes: int = Field(
        0, description = "A separate drafter's own KV cache and rollback state"
    )
    drafter_runtime_gpu_bytes: int = Field(
        0, description = "The share of drafter_runtime_bytes that lands on the GPU"
    )
    projector_runtime_bytes: int = Field(
        0, description = "The vision encoder's buffers, on top of the projector file"
    )
    drafter_kv_unsized: bool = Field(
        False, description = "A charged drafter's cache could not be sized, so totals are a floor"
    )
    adapters_unsized: bool = Field(
        False, description = "A pass-through adapter could not be stat'd, so totals are a floor"
    )

    total_bytes: int = Field(0, description = "Weights + KV + compute, wherever they land")
    gpu_bytes: Optional[int] = Field(
        None,
        description = "The share of total_bytes that lands on the GPU. Optional because "
        "None and 0 are DIFFERENT answers here: None is 'the planner did not run', while "
        "0 is 'this launch puts nothing on the card', which inherited placement such as "
        "LLAMA_ARG_DEVICE=none really does produce. Collapsing the two sends a caller "
        "back to summing segments and drawing VRAM pressure for a load that touches no "
        "card at all.",
    )
    gpu_floor_bytes: Optional[int] = Field(
        None,
        description = "What still lands on the GPU at the SHORTEST context: drafter "
        "weights, flat compute buffers, recurrent rollback state. None of it shrinks "
        "when the context does, so it separates an overage a shorter context fixes from "
        "one it cannot. None when it was not computed.",
    )

    kv_estimable: bool = Field(True, description = "False when the header could not size the cache")
    kv_on_gpu: bool = Field(True, description = "Whether the target cache sits on the GPU")
    n_ctx: int = Field(0, description = "Context length the estimate actually priced")
    native_context: Optional[int] = Field(
        None, description = "The model's own trained context length, when readable"
    )
    cache_type_kv: Optional[str] = Field(None, description = "KV cache dtype the estimate priced")
    n_parallel: int = Field(1, description = "Slots the estimate priced, after the launch clamps")
    layer_count: Optional[int] = Field(None, description = "GGUF block_count, when readable")
    gpu_layers: Optional[int] = Field(None, description = "Layers placed on the GPU, when known")
    moe_offload_unmodelled: bool = Field(
        False, description = "--n-cpu-moe is set, so the GPU figure reads high"
    )
    context_is_pinned: bool = Field(
        True,
        description = "False only when the loader is free to shrink the context. A caller "
        "that softens its verdict for an auto-fitted row has to stop softening here.",
    )
    inherited_device_pin: bool = Field(
        False,
        description = "An inherited LLAMA_ARG_DEVICE confines the launch to the cards it "
        "names, so a budget aggregated over the visible inventory describes a pool the "
        "launch will not open.",
    )
    spec_unpriced: bool = Field(
        False, description = "A speculative term was charged that could not be sized"
    )


class InstallLatestTransformersRequest(BaseModel):
    """Consented request to install the latest transformers release into a sidecar."""

    version: str = Field(
        ...,
        min_length = 1,
        max_length = 64,
        description = "Exact transformers version to install; must match the current "
        "latest PyPI release reported by /validate.",
    )
    force_cancel_active: bool = Field(
        False,
        description = (
            "Stop chats still generating instead of refusing with 409. The install "
            "is a step of the model swap that raised the same prompt, so a client "
            "that already got consent for that swap can carry it through here."
        ),
    )


class InstallLatestTransformersResponse(BaseModel):
    """Result of the consented latest-transformers sidecar install."""

    success: bool = Field(..., description = "Whether the sidecar was provisioned")
    version: str = Field(..., description = "The requested transformers version")
    message: str = Field(..., description = "Human-readable result")
    model_unloaded: bool = Field(
        False,
        description = "Whether the active chat model was unloaded before the swap "
        "(reported even on failure, so the client can restore its state)",
    )
    latest_version: Optional[str] = Field(
        None,
        description = "On a version-mismatch failure: the release that superseded "
        "the requested one, so the client can retry with it",
    )


class GenerateRequest(BaseModel):
    """Request for text generation (legacy /generate/stream endpoint)"""

    messages: List[dict] = Field(..., description = "Chat messages in OpenAI format")
    system_prompt: str = Field("", description = "System prompt")
    temperature: float = Field(0.6, ge = 0.0, le = 2.0, description = "Sampling temperature")
    top_p: float = Field(0.95, ge = 0.0, le = 1.0, description = "Top-p sampling")
    top_k: int = Field(20, ge = -1, le = 100, description = "Top-k sampling")
    min_p: float = Field(0.0, ge = 0.0, le = 1.0, description = "Min-p sampling")
    max_new_tokens: int = Field(2048, ge = 1, le = 4096, description = "Maximum tokens to generate")
    repetition_penalty: float = Field(1.0, ge = 1.0, le = 2.0, description = "Repetition penalty")
    presence_penalty: float = Field(0.0, ge = 0.0, le = 2.0, description = "Presence penalty")
    image_base64: Optional[str] = Field(None, description = "Base64 encoded image for vision models")


class _InferenceRuntimeFields(BaseModel):
    """Runtime fields shared by load and status responses."""

    is_vision: bool = Field(False, description = "Whether model is a vision model")
    is_diffusion: bool = Field(
        False, description = "Whether model is a block-diffusion model (DiffusionGemma)"
    )
    diffusion_requested_ngl: Optional[int] = Field(
        None,
        description = "GPU-layer count the diffusion runner was ASKED for, when that differs "
        "from what it applied: an unsloth_zoo shim without --ngl drops the split and runs "
        "Auto, so gpu_layers reports -1 while this reports the standing request. None for "
        "non-diffusion models and whenever the ask and the applied split agree.",
    )
    is_audio: bool = Field(False, description = "Whether model is a TTS audio model")
    audio_type: Optional[str] = Field(None, description = "Audio codec type: snac, csm, bicodec, dac")
    has_audio_input: bool = Field(False, description = "Whether model accepts audio input (ASR)")
    has_video_input: bool = Field(
        False,
        description = (
            "Whether llama-server accepts video input for this model, from its /props "
            "modalities. False unless the mmproj, the build and ffmpeg all support it."
        ),
    )
    requires_trust_remote_code: bool = Field(
        False,
        description = "Whether the model defaults require trust_remote_code to be enabled for loading.",
    )
    context_length: Optional[int] = Field(
        None, description = "Runtime context length in tokens for the loaded model"
    )
    max_context_length: Optional[int] = Field(
        None, description = "Maximum context length currently available on this hardware"
    )
    native_context_length: Optional[int] = Field(
        None,
        description = "Model's native context length from GGUF metadata (not capped by VRAM)",
    )
    supports_reasoning: bool = Field(
        False,
        description = "Whether model supports thinking/reasoning mode (enable_thinking or reasoning_effort)",
    )
    reasoning_style: Literal["enable_thinking", "reasoning_effort", "enable_thinking_effort"] = (
        Field(
            "enable_thinking",
            description = "Reasoning control style: 'enable_thinking' (boolean), 'reasoning_effort' (low|medium|high), or 'enable_thinking_effort' (on/off gate plus an effort level, e.g. GLM-5.2 high|max)",
        )
    )
    reasoning_effort_levels: List[str] = Field(
        default_factory = list,
        description = "Discrete reasoning_effort levels the template offers when reasoning_style is 'enable_thinking_effort' (e.g. ['high', 'max']); empty otherwise",
    )
    reasoning_always_on: bool = Field(
        False,
        description = "Whether reasoning is always on (hardcoded <think> tags, not toggleable)",
    )
    supports_preserve_thinking: bool = Field(
        False,
        description = "Whether the template understands the optional preserve_thinking kwarg",
    )
    preserve_thinking_default: bool = Field(
        False,
        description = "Default preserve_thinking value resolved for the active model family",
    )
    supports_tools: bool = Field(
        False,
        description = "Whether model supports tool calling (web search, etc.)",
    )
    cache_type_kv: Optional[str] = Field(
        None,
        description = (
            "KV cache data type for K and V "
            "(e.g. 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl', 'f32')"
        ),
    )
    is_mlx: bool = Field(False, description = "Whether the active model is served by the MLX backend")
    mlx_kv_bits: Optional[int] = Field(
        None, description = "MLX KV quantization bit width actually applied, if any"
    )
    mlx_kv_bits_requested: Optional[int] = Field(
        None,
        description = (
            "MLX KV quantization bit width the load asked for. Differs from "
            "mlx_kv_bits when the model could not honor it, which is exactly "
            "when the reason matters."
        ),
    )
    chat_template_override: Optional[str] = Field(
        None,
        description = "Chat template override this model was loaded with. llama.cpp reports what it applied; MLX reports what was requested, since a refusal applies nothing and the reason accompanies it",
    )
    chat_template_override_reason: Optional[str] = Field(
        None,
        description = (
            "Why a requested chat template override was not applied: the "
            "runtime supplies the template as code, the model ships a named "
            "set rather than one template, it renders without a template, or "
            "the override could not render a conversation."
        ),
    )
    mlx_kv_quant_eligibility: Optional[str] = Field(
        None,
        description = (
            "Whether this model's KV cache could be quantized: full, partial, "
            "none or refused. Describes eligibility, not the exact set of "
            "converted layers, which the runtime decides."
        ),
    )
    mlx_kv_quant_reason: Optional[str] = Field(
        None,
        description = (
            "Why KV quantization was not applied in full: the model's cache "
            "layout, or the layers it could not cover when eligibility is partial"
        ),
    )
    mlx_kv_quant_note: Optional[str] = Field(
        None, description = "Caveat that applies when KV quantization is active"
    )
    chat_template: Optional[str] = Field(
        None,
        description = "Jinja2 chat template string (from GGUF metadata or tokenizer)",
    )
    speculative_type: Optional[str] = Field(
        None,
        description = (
            "Canonical UI-facing requested speculative decoding mode "
            "('auto' / 'mtp' / 'dspark' / 'ngram' / 'mtp+ngram' / 'off' / "
            "'ngram-simple'), round-tripped from the original LoadRequest "
            "via _canonicalize_spec_mode. None when no model is loaded."
        ),
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        description = (
            "Active --spec-draft-n-max for MTP or DSpark speculative decoding, or "
            "None when the platform default is in effect."
        ),
    )
    tensor_parallel: bool = Field(
        False,
        description = "Whether tensor-parallel split (--split-mode tensor) is active.",
    )
    disable_vision: bool = Field(
        False,
        description = (
            "Whether the load ran with the vision projector deliberately left "
            "unloaded. Echoes the request, so the Advanced Settings switch can "
            "reseed from it even on a GGUF that never had a projector."
        ),
    )
    vision_disabled_by_user: bool = Field(
        False,
        description = (
            "Whether image input is off because the user asked, rather than because "
            "the model has no usable mmproj. The two look identical to a client "
            "otherwise and need different guidance, so this stays False for a model "
            "that never had a projector to switch off."
        ),
    )
    gpu_memory_mode: Literal["auto", "manual"] = Field(
        "auto",
        description = "Active GPU memory strategy ('auto' or 'manual').",
    )
    gpu_layers: int = Field(
        -1,
        description = "Manual mode: requested --gpu-layers value (-1 = Auto/--fit, or when not manual).",
    )
    cpu_fallback_reason: Optional[Literal["vulkan_startup_crash"]] = Field(
        None,
        description = (
            "Why an automatic GGUF load was downgraded to CPU. "
            "'vulkan_startup_crash' means a managed, auto-selected Vulkan launch "
            "hard-crashed and the same launch became healthy with GPU devices disabled."
        ),
    )

    mmproj_fallback_reason: Optional[
        Literal["cpu_offload", "projector_incompatible", "projector_startup_failure"]
    ] = Field(
        None,
        description = (
            "How an automatic GGUF multimodal-projector recovery changed the load. "
            "'cpu_offload' keeps vision with the projector on CPU; "
            "'projector_incompatible' and 'projector_startup_failure' mean the "
            "model recovered text-only."
        ),
    )
    n_cpu_moe: int = Field(
        0,
        description = "Manual mode: MoE expert layers pinned to CPU (--n-cpu-moe); 0 = none.",
    )
    tensor_split: Optional[List[float]] = Field(
        None,
        description = "Manual mode: relative model share per GPU (--tensor-split); None = default (split by free VRAM).",
    )
    n_layers: Optional[int] = Field(
        None,
        description = "Model's layer count (GGUF block_count), for the manual gpu-layers ceiling.",
    )
    n_moe_layers: int = Field(
        0,
        description = "Model's MoE expert-layer count (the n_cpu_moe ceiling); 0 if not an MoE model.",
    )
    gpu_ids: Optional[List[int]] = Field(
        None,
        description = "Effective GPU indices the model is using after fit-time narrowing, or None for automatic selection.",
    )
    requested_gpu_ids: Optional[List[int]] = Field(
        None,
        description = (
            "GPU placement pool requested by the user before fit-time narrowing, "
            "or None for automatic selection."
        ),
    )
    requested_parallel_slots: Optional[int] = Field(
        None,
        description = (
            "Parallel decode slots the load was invoked with (per-load "
            "n_parallel, else the server-wide --parallel default). None for "
            "non-GGUF loads and for the diffusion runner, which ignores "
            "--parallel."
        ),
    )
    parallel_slots: Optional[int] = Field(
        None,
        description = (
            "Serving slots the active llama-server actually runs (--parallel "
            "after any fit-time slot reduction). None for non-GGUF loads and "
            "for the diffusion runner, which ignores --parallel."
        ),
    )
    requested_n_batch: Optional[int] = Field(
        None,
        description = (
            "Batch size (--batch-size) the load was invoked with, or None when "
            "the load left it at the llama.cpp default (or to extra args / env)."
        ),
    )
    requested_n_ubatch: Optional[int] = Field(
        None,
        description = (
            "Micro-batch size (--ubatch-size) the load was invoked with, or None "
            "when the load left it at the llama.cpp default (or to extra args / env)."
        ),
    )
    requested_load_mode: Optional[str] = Field(
        None,
        description = (
            "Load mode (--load-mode) the load was invoked with, or None when the "
            "load left it at the llama.cpp default. This is what was REQUESTED: "
            "the Model Memory settings can replace it, and what they emit is "
            "reported by the model-memory settings route instead."
        ),
    )
    requested_spec_draft_cache_type: Optional[str] = Field(
        None,
        description = (
            "Draft KV cache dtype the load was invoked with, or None when the "
            "load left it at the llama.cpp default (or attached no drafter)."
        ),
    )
    requested_ctx_checkpoints: Optional[int] = Field(
        None,
        description = (
            "Checkpoints (--ctx-checkpoints) the load was invoked with, or None "
            "when the load left it at the llama.cpp default."
        ),
    )
    requested_cache_ram: Optional[int] = Field(
        None,
        description = (
            "Host prompt cache size in MiB (--cache-ram) the load was invoked "
            "with, or None when the load left it at the llama.cpp default."
        ),
    )
    requested_llama_extra_args: Optional[List[str]] = Field(
        None,
        description = (
            "Pass-through llama-server arguments the running load was INVOKED "
            "with, or None for a non-GGUF load and for a load that passed none. "
            "Published so a client that attached to an already-running server "
            "(a fresh tab, or another browser) knows what it is running: without "
            "it, a rollback after a failed switch restores the previous model "
            "without the arguments it had."
        ),
    )


class LoadResponse(_InferenceRuntimeFields):
    """Response after loading a model"""

    status: str = Field(..., description = "Load status")
    model: str = Field(..., description = "Model identifier")
    display_name: str = Field(..., description = "Display name of the model")
    is_lora: bool = Field(False, description = "Whether model is a LoRA adapter")
    is_gguf: bool = Field(False, description = "Whether model is a GGUF model (llama.cpp)")
    is_local_model: bool = Field(
        False, description = "Whether the loaded model came from a local filesystem path"
    )
    inference: dict = Field(
        ..., description = "Inference parameters (temperature, top_p, top_k, min_p)"
    )
    memory_warning: Optional[str] = Field(
        None,
        description = "Non-blocking advisory about this load, or null. Set when the "
        "weights do not fit in free VRAM plus available system RAM, so llama.cpp pages "
        "them in from disk and generation will be slow. The model still loaded.",
    )


class UnloadResponse(BaseModel):
    """Response after unloading a model"""

    status: str = Field(..., description = "Unload status")
    model: str = Field(..., description = "Model identifier that was unloaded")


class LoadProgressResponse(BaseModel):
    """Progress of the active GGUF load, sampled on demand.

    Drives a real progress bar during the post-download warmup (mmap + CUDA upload)
    instead of a spinner that freezes for minutes on large MoE models.
    """

    phase: Optional[str] = Field(
        None,
        description = (
            "Load phase: 'mmap' (weights paging into RAM via mmap), "
            "'ready' (llama-server reported healthy), or null when no "
            "load is in flight."
        ),
    )
    bytes_loaded: int = Field(
        0,
        description = (
            "Bytes of the model already resident in the llama-server process (VmRSS on Linux)."
        ),
    )
    bytes_total: int = Field(
        0,
        description = "Total bytes across all GGUF shards for the active model.",
    )
    fraction: float = Field(0.0, description = "bytes_loaded / bytes_total, clamped to 0..1.")


class LlamaFlagCatalogResponse(BaseModel):
    """Every llama-server flag THIS build documents, for validating pass-through args.

    Read from the installed binary's ``--help`` rather than a list bundled with
    Unsloth: a custom or newer llama.cpp is exactly the case where a bundled list
    would reject a flag that works, or accept one that does not exist.
    """

    flags: dict[str, str] = Field(
        default_factory = dict,
        description = "Flag name -> its help text, e.g. {'--top-k': 'top-k sampling ...'}",
    )
    managed: list[str] = Field(
        default_factory = list,
        description = "Flags Unsloth Studio owns; validate_extra_args rejects these outright",
    )
    switch_flags: list[str] = Field(
        default_factory = list,
        description = (
            "Flags this build documents as taking no value, so an editor can tell "
            "'--verbose foo' (which llama-server refuses) from '--numa distribute'"
        ),
    )
    max_bytes: int = Field(
        0,
        description = (
            "Size limit validate_extra_args applies on THIS host; smaller on Windows, "
            "where the whole command line shares one 32767-character budget"
        ),
    )
    windows_command_budget: int = Field(
        0,
        description = (
            "Characters the quoted command may take on Windows, or 0 elsewhere. An "
            "editor mirrors it because the quoting can double a backslash-heavy value."
        ),
    )
    default_parallel_slots: int = Field(
        1,
        description = (
            "Serving slots a load gets when it names none, the server-wide "
            "--parallel. An editor needs it to judge a pass-through --batch-size: "
            "llama-server aborts on a batch below the slots it serves, and with the "
            "Slots field blank this is the number the launch will use."
        ),
    )
    parallel_slots_clamped: bool = Field(
        False,
        description = (
            "True when this build serves ONE slot however many are asked for, because "
            "it has no --kv-unified and load_model falls back. The default above is "
            "already effective, but an EXPLICIT Slots value is not: without this an "
            "editor sizes its batch floor from a count the launch will not use, and "
            "refuses a --batch-size the backend accepts."
        ),
    )
    probe_ok: bool = Field(
        False,
        description = (
            "False when --help could not be read. `flags` is then empty and callers "
            "must not report a flag as unknown, only as unverified."
        ),
    )


class InferenceStatusResponse(_InferenceRuntimeFields):
    """Current inference backend status"""

    active_model: Optional[str] = Field(
        None, description = "Currently active model display identifier"
    )
    model_identifier: Optional[str] = Field(
        None,
        description = "Loadable identifier for the active model.",
    )
    is_gguf: bool = Field(False, description = "Whether the active model is a GGUF model (llama.cpp)")
    is_local_model: bool = Field(
        False, description = "Whether the active model came from a local filesystem path"
    )
    gguf_variant: Optional[str] = Field(None, description = "GGUF quantization variant (e.g. Q4_K_M)")
    loading: List[str] = Field(default_factory = list, description = "Models currently being loaded")
    loaded: List[str] = Field(default_factory = list, description = "Models currently loaded")
    inference: Optional[Dict[str, Any]] = Field(
        None, description = "Recommended inference parameters for the active model"
    )
    requested_context_length: Optional[int] = Field(
        None,
        description = (
            "The n_ctx the active GGUF load was invoked with (0 = Auto). Lets the "
            "UI re-seed a Manual + Auto-layers context pin on hydration, where "
            "context_length only exposes the resolved value. None for non-GGUF."
        ),
    )
    llama_cpp_supports_mtp: bool = Field(
        True,
        description = (
            "Whether llama.cpp supports MTP (--spec-type mtp/draft-mtp). "
            "False -> recommend `unsloth studio update`."
        ),
    )
    spec_drafter_kind: Optional[str] = Field(
        None,
        description = (
            "Which drafter the resolution was about: 'mtp', 'dspark' or "
            "'dflash'. Needed "
            "because Auto resolves the kind itself, so speculative_type still "
            "reads 'auto', and a fallback leaves the engaged type at 'default': "
            "neither still says which file the UI should tell the user to fix."
        ),
    )
    spec_fallback_reason: Optional[str] = Field(
        None,
        description = (
            "Why a speculative drafter was disabled despite being requested. "
            "'binary_no_mtp' / 'binary_outdated' -> a newer prebuilt would "
            "re-enable it (show the update affordance); 'runtime_error' -> the "
            "current build could not run it; 'drafter_not_found' -> the model's "
            "separate MTP or DSpark drafter could not be resolved; "
            "'drafter_no_vram' -> an Auto-mode fit downgrade: the model pins on "
            "GPU but the drafter's reserve does not, and Auto keeps the context "
            "rather than shrink it; select the drafter in Settings to force it. "
            "'mla_mtp_disabled' -> "
            "an Auto-mode policy downgrade: the model is MLA (GLM-5.2 et al.) "
            "whose llama.cpp MTP path runs slower than no speculation, so Auto "
            "used ngram-mod or spec-off instead -- updating won't help; choose "
            "MTP in Settings (or set UNSLOTH_MLA_MTP_ENABLED=1) to force it. "
            "'mtp_partial_offload' -> an Auto-mode policy downgrade: the model "
            "has an embedded Hybrid Mamba MTP head and the placement offloads "
            "only part of it, where the recurrent rollback copies cost more "
            "layers than the drafting wins back -- updating won't help; choose "
            "MTP in Settings to force it. "
            "None when the requested strategy engaged or was not requested."
        ),
    )
    spec_fallback_binary_changed: Optional[bool] = Field(
        None,
        description = (
            "For a 'binary_no_mtp' / 'binary_outdated' stand-down only: whether a "
            "different llama-server is installed than the live one launched from, "
            "which is the necessary condition in "
            "LlamaCppBackend.spec_binary_fallback_can_retry. False -> an identical "
            "/load cannot repair the drafter and dedupes, so a client need not "
            "reload for it. None for every other reason, and on a backend too old "
            "to report it."
        ),
    )
    spec_probe_retry_pending: Optional[bool] = Field(
        None,
        description = (
            "The capability probe has started answering since a launch it degraded, so "
            "an identical /load is rejected once to re-derive the runtime "
            "(_runtime_matches_intent's _capability_probe_inconclusive arm). No "
            "speculative mode gates it. None on a backend too old to report it."
        ),
    )
    spec_dflash_retry_pending: Optional[bool] = Field(
        None,
        description = (
            "A DFlash sidecar fetch failed retryably, which under Auto records no "
            "spec_fallback_reason at all, so an identical /load is rejected to fetch "
            "again. Applies to the 'auto' and 'dflash' modes. None on a backend too old "
            "to report it."
        ),
    )
    spec_dspark_sidecar_absent: Optional[bool] = Field(
        None,
        description = (
            "The DSpark drafter is absent rather than transiently unfetchable, which is "
            "the permanent state of every repo but one. _runtime_matches_intent excludes "
            "it from the drafter_not_found retry arm, so an identical /load dedupes and a "
            "client need not reload for it. None on a backend too old to report it."
        ),
    )
    tensor_parallel_dropped_by_arch_gate: Optional[bool] = Field(
        None,
        description = (
            "The GPU architecture gate normalized a tensor-parallel request to layer "
            "mode, so tensor_parallel reads false while the request that produced it was "
            "true. _runtime_matches_intent accepts the same true request against this "
            "runtime. None on a backend too old to report it."
        ),
    )
    gpu_placement_paravirtual: Optional[bool] = Field(
        None,
        description = (
            "Metal is a virtualised Apple GPU, so paravirtual_normalized_request rewrites "
            "every GGUF request to manual / zero layers / no split / no MoE before the "
            "duplicate-load comparators run. Placement cannot tell two requests apart "
            "here. None on a backend too old to report it."
        ),
    )
    audio_probe_pending: bool = Field(
        False,
        description = (
            "The post-launch audio probe did not finish and has to be retried. The route "
            "refuses its own already-loaded answer while this is true so load_model can "
            "re-probe, and nothing else does, so a client must not skip /load for it."
        ),
    )
    diffusion_split_supported: Optional[bool] = Field(
        None,
        description = (
            "A diffusion launch right now would honour --ngl. _runtime_matches_intent "
            "rejects an otherwise identical request once this is true and gpu_layers "
            "differs from diffusion_requested_ngl, so a split an older shim dropped can "
            "be applied. None off a diffusion runner, and on a backend too old to report "
            "it."
        ),
    )
    llama_cpp_prebuilt_stale: bool = Field(
        False,
        description = (
            "Installed llama.cpp prebuilt is >=3 days behind the latest "
            "release. True -> show `unsloth studio update` banner."
        ),
    )
    llama_cpp_installed_tag: Optional[str] = Field(
        None,
        description = "Installed llama.cpp tag, or None if unknown.",
    )
    llama_cpp_latest_tag: Optional[str] = Field(
        None,
        description = "Latest published llama.cpp tag, or None if GitHub unreachable.",
    )


# =====================================================================
# OpenAI-Compatible Chat Completions Models
# =====================================================================


# ── Multimodal content parts (OpenAI vision format) ──────────────


class TextContentPart(BaseModel):
    """Text content part in a multimodal message."""

    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    """Image URL object — supports data URIs and remote URLs."""

    url: str = Field(..., description = "data:image/png;base64,... or https://...")
    detail: Optional[Literal["auto", "low", "high", "original"]] = "auto"


class ImageContentPart(BaseModel):
    """Image content part in a multimodal message."""

    type: Literal["image_url"]
    image_url: ImageUrl


class InputDocumentContentPart(BaseModel):
    """Document (PDF / file) content part in a multimodal message.

    Unsloth-normalised shape (file_data or file_url, plus optional filename/media_type).
    Mapped onto Anthropic ``document`` / OpenAI ``input_file`` for vision providers;
    dropped for non-vision providers.
    """

    type: Literal["input_document"]
    file_data: Optional[str] = Field(
        None,
        description = "data:<media_type>;base64,<DATA> URI for inline payloads. Either file_data or file_url must be set; otherwise the part is dropped.",
    )
    file_url: Optional[str] = Field(
        None,
        description = "Remote URL pointing to the document (https://...).",
    )
    filename: Optional[str] = Field(
        None,
        description = "Display filename, forwarded to providers as `title`/`filename`.",
    )
    media_type: Optional[str] = Field(
        None,
        description = 'Override the media type sniffed from the data URI (e.g. "application/pdf").',
    )


class OpenAIReasoningContentPart(BaseModel):
    """OpenAI Responses reasoning item paired with a tool output.

    Reasoning models may require this replayed before an ``image_generation_call``
    id. OpenAI-only; routes strip it for other providers before proxying.
    """

    type: Literal["reasoning"]
    id: str = Field(..., description = "OpenAI reasoning output item id.")
    summary: list[dict[str, Any]] = Field(default_factory = list)
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ImageGenerationCallContentPart(BaseModel):
    """OpenAI Responses image_generation call reference.

    Prior ``image_generation_call`` items let follow-up prompts edit a generated
    image without resending the payload. The frontend forwards it as a synthetic
    assistant part; ``external_provider`` maps it back to a top-level input item.
    """

    type: Literal["image_generation_call"]
    id: str = Field(..., description = "OpenAI image_generation_call output item id.")
    response_id: Optional[str] = Field(
        None,
        description = "OpenAI Responses response id to use as previous_response_id for follow-up edits.",
    )


class CompactionContentPart(BaseModel):
    """Anthropic server-side compaction state, round-tripped on the next turn.

    Anthropic returns a ``compaction`` block on the assistant message; the next
    request must forward it back so Anthropic reuses the compaction state instead
    of re-summarising. See ``external_provider._stream_anthropic`` and
    https://platform.claude.com/docs/en/build-with-claude/compaction
    """

    type: Literal["compaction"]
    content: str = Field(
        ...,
        description = "Anthropic-produced summary of the compacted-away conversation prefix.",
    )


def _content_part_discriminator(v):
    if isinstance(v, dict):
        return v.get("type")
    return getattr(v, "type", None)


ContentPart = Annotated[
    Union[
        Annotated[TextContentPart, Tag("text")],
        Annotated[ImageContentPart, Tag("image_url")],
        Annotated[InputDocumentContentPart, Tag("input_document")],
        Annotated[OpenAIReasoningContentPart, Tag("reasoning")],
        Annotated[ImageGenerationCallContentPart, Tag("image_generation_call")],
        Annotated[CompactionContentPart, Tag("compaction")],
    ],
    Discriminator(_content_part_discriminator),
]
"""Union type for multimodal content parts, discriminated by the 'type' field."""


# ── Messages ─────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """Single message in a chat conversation.

    ``content`` is a string or list of multimodal parts. Assistant messages with
    only ``tool_calls`` may set ``content=None``. Missing ``tool_call_id`` on
    ``role="tool"`` is resolved at the ``ChatCompletionRequest`` layer.
    """

    role: Literal["system", "user", "assistant", "tool", "developer"] = Field(
        ..., description = "Message role"
    )
    content: Optional[Union[str, list[ContentPart]]] = Field(
        None, description = "Message content (string or multimodal parts)"
    )
    reasoning_content: Optional[str] = Field(
        None,
        description = (
            "Assistant reasoning from an earlier turn, replayed to local chat templates "
            "that consume the OpenAI-compatible reasoning_content field."
        ),
    )
    tool_call_id: Optional[str] = Field(
        None,
        description = "OpenAI tool-result messages: id of the tool call this result belongs to.",
    )
    tool_calls: Optional[list[dict]] = Field(
        None,
        description = "OpenAI assistant messages: structured tool calls the model decided to make.",
    )
    name: Optional[str] = Field(
        None,
        description = "OpenAI tool-result messages: name of the tool whose result this is.",
    )
    extra_content: Optional[dict] = Field(
        None,
        description = (
            "Provider-specific extra fields the translator may read. "
            "Gemini reads `extra_content.google.thought_signature` "
            "from assistant messages to replay text-part signatures."
        ),
    )

    @field_validator("reasoning_content", mode = "before")
    @classmethod
    def _ignore_non_string_reasoning(cls, value):
        # This field used to be ignored as an unknown key. Some compatible
        # gateways send structured reasoning, so declaring the string form must
        # not turn those previously accepted requests into validation errors.
        return value if isinstance(value, str) else None

    @model_validator(mode = "after")
    def _validate_role_shape(self) -> "ChatMessage":
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError('"tool_calls" is only valid on role="assistant" messages.')
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError('"tool_call_id" is only valid on role="tool" messages.')
        if self.name is not None and self.role != "tool":
            raise ValueError('"name" is only valid on role="tool" messages.')

        if self.role == "tool":
            # tool_call_id resolution happens at ChatCompletionRequest scope.
            # OpenAI accepts empty tool results (commands with no output);
            # normalize to "" instead of a 400 agentic clients treat as fatal.
            if self.content is None or self.content == []:
                self.content = ""
        elif self.role == "assistant":
            # Post-Stop sentinel: collapse content="" / [] to None.
            if (self.content == "" or self.content == []) and not self.tool_calls:
                self.content = None
        else:  # "user" | "system"
            if self.content is None or self.content == []:
                raise ValueError(f'role="{self.role}" messages require "content".')
        return self


class ThinkingConfig(BaseModel):
    """Anthropic-compatible thinking/reasoning configuration.
    Use type='disabled' to turn off thinking, or type='enabled' to turn it on.
    Only type is read; extra fields (e.g. budget_tokens) are ignored, since
    Unsloth sets provider thinking budgets itself.
    """

    type: Literal["disabled", "enabled"] = "disabled"


# Recognized permission_mode values. The field accepts a plain string rather than
# a Literal so an unrecognized value from a newer UI/client degrades to the safest
# gate ("ask") instead of a 422. None stays unset at the request boundary: the tool
# loops normalize it to the product default "auto", while the route's confirm-gate
# derivation keeps an unset mode lenient (a non-streaming request cannot prompt, so
# it runs) to keep non-streaming clients and health checks working.
_KNOWN_PERMISSION_MODES = ("ask", "auto", "off", "full")


def _normalize_permission_mode(value: Any) -> Any:
    if value is None:
        return None
    if value not in _KNOWN_PERMISSION_MODES:
        return "ask"
    return value


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    Non-OpenAI extension fields are marked with 'x-unsloth'.
    """

    # Accept unknown fields so future OpenAI fields aren't dropped before route
    # code runs. Mirrors AnthropicMessagesRequest and ResponsesRequest.
    model_config = {"extra": "allow"}

    model: str = Field(
        "default",
        description = "Model identifier (informational; the active model is used)",
    )
    messages: list[ChatMessage] = Field(..., description = "Conversation messages")
    stream: bool = Field(
        False,
        description = (
            "Whether to stream the response via SSE. Default matches OpenAI's "
            "spec (`false`); opt into streaming by sending `stream: true`."
        ),
    )
    temperature: float = Field(0.6, ge = 0.0, le = 2.0)
    top_p: float = Field(0.95, ge = 0.0, le = 1.0)
    max_tokens: Optional[int] = Field(
        None, ge = 1, description = "Maximum tokens to generate (None = until EOS)"
    )
    # OpenAI's documented range is [-2, 2] on both penalties. Widening only
    # admits requests that used to be rejected, so nothing that works today
    # changes. A negative value boosts repetition wherever the backend applies
    # the penalty at all.
    presence_penalty: float = Field(
        0.0,
        ge = -2.0,
        le = 2.0,
        description = (
            "Presence penalty: charges a token once if it appears. MLX scores the completion"
            " only; llama-server also counts the prompt, within its own window."
        ),
    )
    frequency_penalty: float = Field(
        0.0,
        ge = -2.0,
        le = 2.0,
        description = (
            "Frequency penalty: charges a token per occurrence. MLX scores the completion"
            " only; llama-server also counts the prompt, within its own window."
        ),
    )
    # Ids are not range-checked here: MLX bounds-checks nothing either, so a
    # stray is dropped at the processors rather than failing the request.
    logit_bias: Optional[Dict[int, Annotated[float, Field(ge = -100.0, le = 100.0)]]] = Field(
        None,
        description = "Additive per-token logit bias keyed by token id, each in [-100, 100]. Ids past the model's logit width are ignored.",
    )
    stop: Optional[Union[str, list[str]]] = Field(
        None,
        description = "OpenAI stop sequences: a single string or list of strings at which generation halts.",
    )
    # Declared rather than left to model_extra so the schema documents it and a
    # backend that cannot constrain decoding can refuse it by name.
    response_format: Optional[Dict[str, Any]] = Field(
        None,
        description = (
            'Guided decoding contract, typically `{"type": "json_object"}` or'
            ' a `json_schema`. `{"type": "text"}` names the default and'
            " constrains nothing. Anything else needs a grammar engine, and a"
            " backend without one rejects the request rather than answering text"
            " that ignores the contract."
        ),
    )
    tools: Optional[list[dict]] = Field(
        None,
        description = (
            "OpenAI function-tool definitions. When provided without `enable_tools=true`, "
            "Unsloth forwards the tools to the backend so the model returns structured "
            "tool_calls for the client to execute (standard OpenAI function calling)."
        ),
    )
    tool_choice: Optional[Union[str, dict]] = Field(
        None,
        description = (
            "OpenAI tool choice: 'auto' | 'required' | 'none' | "
            "{'type': 'function', 'function': {'name': ...}}"
        ),
    )
    max_completion_tokens: Optional[int] = Field(
        None,
        ge = 1,
        description = "OpenAI upper bound on generated tokens (supersedes the deprecated max_tokens).",
    )
    n: Optional[int] = Field(
        None,
        ge = 1,
        le = 128,
        description = "Number of chat completion choices to generate.",
    )
    logprobs: Optional[bool] = Field(
        None, description = "Whether to return log probabilities of the output tokens."
    )
    top_logprobs: Optional[int] = Field(
        None,
        ge = 0,
        le = 20,
        description = "Number of most likely tokens (0-20) to return per position; requires logprobs=true.",
    )
    parallel_tool_calls: Optional[bool] = Field(
        None, description = "Whether to enable parallel function calling during tool use."
    )
    seed: Optional[int] = Field(None, description = "Best-effort deterministic sampling seed.")
    stream_options: Optional[dict] = Field(
        None,
        description = 'Streaming options, e.g. {"include_usage": true} to emit a final usage chunk.',
    )

    # ── Unsloth extensions (ignored by standard OpenAI clients) ──
    top_k: int = Field(20, ge = -1, le = 100, description = "[x-unsloth] Top-k sampling")
    min_p: float = Field(0.01, ge = 0.0, le = 1.0, description = "[x-unsloth] Min-p sampling threshold")
    repetition_penalty: float = Field(
        1.0, ge = 1.0, le = 2.0, description = "[x-unsloth] Repetition penalty"
    )
    image_base64: Optional[str] = Field(
        None, description = "[x-unsloth] Base64-encoded image for vision models"
    )
    audio_base64: Optional[str] = Field(
        None,
        description = "[x-unsloth] Base64-encoded audio (wav/mp3/ogg/flac/m4a) for audio-input models",
    )
    video_base64: Optional[str] = Field(
        None,
        description = (
            "[x-unsloth] Base64-encoded video (mp4/mov/webm/mkv/avi) for video-input "
            "models. GGUF only: llama-server samples frames with ffmpeg."
        ),
    )
    use_adapter: Optional[Union[bool, str]] = Field(
        None,
        description = (
            "[x-unsloth] Adapter control for compare mode. "
            "null = no change (default), "
            "false = disable adapters (base model), "
            "true = enable the current adapter, "
            "string = enable a specific adapter by name."
        ),
    )
    enable_thinking: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Enable/disable thinking/reasoning mode for supported models",
    )
    continue_final_message: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Continue the trailing assistant message instead of starting a new "
            "turn: the prompt ends mid-response so the model resumes token-exactly from "
            "where it stopped. Requires the last message to have role 'assistant'."
        ),
    )
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "max", "xhigh"]
    ] = Field(
        None,
        description = "[x-unsloth] Reasoning effort level ('none'|'minimal'|'low'|'medium'|'high'|'max'|'xhigh'). OpenAI `/v1/responses` accepts model-dependent subsets; Anthropic adaptive thinking uses `max` as the top tier on Claude 4.6 Opus/Sonnet (inbound `xhigh` is mapped to `max`) and `xhigh` on Claude 4.7 Opus; local Harmony/gpt-oss templates support low|medium|high.",
    )
    preserve_thinking: Optional[bool] = Field(
        None,
        description = "[x-unsloth] When true, keep historical <think> blocks from past assistant turns in the prompt (Qwen3.6 templates). Independent of enable_thinking / reasoning_effort.",
    )
    thinking: Optional[ThinkingConfig] = Field(
        None,
        description = "[Anthropic-compatible] Thinking configuration. "
        "Use {type: 'disabled'} to disable thinking, {type: 'enabled'} to enable.",
    )
    enable_tools: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Enable tool calling for supported models",
    )
    enabled_tools: Optional[list[str]] = Field(
        None,
        description = (
            "[x-unsloth] List of enabled tool names. Local GGUF/safetensors models "
            "accept ['web_search', 'python', 'terminal', 'edit_file', 'render_html']. External "
            "providers accept ['web_search', 'web_fetch', 'code_execution'] for "
            "Anthropic and ['web_search', 'code_execution', 'image_generation'] for "
            "OpenAI Responses. If None, all local tools are enabled and no "
            "server-side tools are forwarded."
        ),
    )
    mcp_enabled: Optional[bool] = Field(
        None,
        description = "[x-unsloth] When true, append tools from every enabled MCP server to this request's tool list.",
    )
    deep_research_armed: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Offer the deep_research handoff tool for this turn. Set when the "
            "composer has Deep Research armed; the model decides whether to use it."
        ),
    )
    confirm_tool_calls: Optional[bool] = Field(
        None,
        description = "[x-unsloth] When true, pause before each tool call and wait for the user to allow/deny it via POST /api/inference/tool-confirm.",
    )
    bypass_permissions: Optional[bool] = Field(
        False,
        description = "[x-unsloth] Bypass Permissions: when true, skip the tool-call confirmation gate AND disable the python/terminal execution sandbox (safety checks, command blocklist, resource limits). edit_file is likewise unconfined: an absolute path resolves as written and edits the real file there, instead of being held to the conversation's working directory. Secret env vars are still stripped. Takes precedence over confirm_tool_calls.",
    )
    permission_mode: Optional[str] = Field(
        None,
        description = (
            "[x-unsloth] Permission level for local tool calls. 'ask' pauses every "
            "call for approval; 'ask'/'auto' enable the confirmation gate on their "
            "own (needs a streaming request to deliver prompts). 'auto' ('Approve for "
            "me') only pauses calls detected as high risk (credential reads, privilege "
            "escalation, destructive/persistence, network exfil); ordinary calls run "
            "immediately, and the sandbox stays on. 'full' is equivalent to "
            "bypass_permissions=true (no confirmation, no sandbox). Unset defaults to "
            "'auto' for the per-call gate; a non-streaming request without an explicit "
            "mode cannot prompt and runs the loop. An unrecognized value (e.g. from a "
            "newer client) is treated as 'ask'."
        ),
    )
    auto_heal_tool_calls: Optional[bool] = Field(
        True,
        description = "[x-unsloth] Auto-detect and fix malformed tool calls from model output.",
    )
    nudge_tool_calls: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Opt-in tool-call recovery: when a model stalls with a short "
            "plan instead of calling an available tool, or passthrough healing cannot "
            "repair a malformed call, retry with a short nudge. Default off; "
            "UNSLOTH_TOOL_CALL_NUDGE=1 flips the process default."
        ),
    )
    context_overflow: Optional[Literal["error", "truncate_middle", "truncate_oldest"]] = Field(
        None,
        description = (
            "[x-unsloth] Local GGUF context-overflow behavior. 'error' (default) "
            "returns a 400 with code=context_length_exceeded. 'truncate_middle' is "
            "limited to client-tool or response_format passthrough and retries after "
            "keeping the first and recent turns. 'truncate_oldest' provides a rolling "
            "window for plain and Unsloth-tool chats by dropping complete oldest turns. "
            "Both truncation policies preserve system messages and tool-call groups."
        ),
    )
    context_policy: Optional[Literal["checkpoint", "rolling"]] = Field(
        None,
        description = (
            "[x-unsloth] How a local GGUF chat compacts once context_overflow is "
            "truncate_oldest. 'checkpoint' resets to the latest turn plus standing "
            "instructions (Studio default). 'rolling' drops oldest complete turns. "
            "Unset uses UNSLOTH_CONTEXT_POLICY."
        ),
    )
    compaction_headroom_ratio: Optional[float] = Field(
        None,
        ge = 0.0,
        le = 0.9,
        description = (
            "[x-unsloth] Extra share of the prompt budget to drop when a rolling "
            "compaction fires, so the boundary can stay put for a stretch of turns. "
            "0.25 is the process default (ROLLING_COMPACTION_HEADROOM_RATIO). Ignored "
            "for checkpoint compaction. Unset keeps the process default."
        ),
    )
    studio_tool_history: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] The replayed tool calls were produced by Studio's local "
            "tool loop rather than by an OpenAI-compatible client tool contract."
        ),
    )
    max_tool_calls_per_message: Optional[int] = Field(
        25,
        ge = 0,
        description = "[x-unsloth] Maximum number of tool call iterations per message (0 = disabled, 9999 = unlimited).",
    )
    tool_call_timeout: Optional[int] = Field(
        300,
        ge = 1,
        description = "[x-unsloth] Timeout in seconds for each tool call execution (9999 = no limit).",
    )
    run_tools_locally: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Execute the selected tools on the Unsloth host instead of "
            "asking the provider to run its own hosted builtins. Only meaningful "
            "for providers that ship hosted tools of the same name (OpenAI, "
            "Gemini, Kimi, OpenRouter), where 'web_search' alone is ambiguous: "
            "the same request means hosted search to a client written before "
            "Unsloth ran tools for external providers. Omitted keeps the hosted "
            "behaviour, so an older client is unaffected."
        ),
    )
    session_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Session/thread ID for scoping tool execution sandbox.",
    )
    thread_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Conversation ID for scoping stateful tool sessions (e.g. stdio MCP); stays per-thread where session_id may be shared project-wide.",
    )
    rag_scope: Optional[dict] = Field(
        None,
        description = (
            "[x-unsloth] Hidden RAG retrieval scope for the search_knowledge_base "
            "tool: {kb_id?, thread_id?, default_top_k?, mode?, autoinject?, "
            "autoinject_min_score?}. Candidate pools and the RRF constant come from "
            "server config. The model never sees this; the server resolves which "
            "documents to search."
        ),
    )
    cancel_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Per-request cancellation token. Frontend sends a fresh UUID per run so /inference/cancel matches one specific generation.",
    )

    # ── External provider routing (x-unsloth extensions) ──────────
    provider_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Saved provider config ID. Its stored key is used when encrypted_api_key is omitted.",
    )
    provider_type: Optional[str] = Field(
        None,
        description = "[x-unsloth] Provider type (e.g. 'openai', 'mistral'). Used if provider_id is not set.",
    )
    external_model: Optional[str] = Field(
        None,
        description = "[x-unsloth] Model ID at the external provider.",
    )
    encrypted_api_key: Optional[str] = Field(
        None,
        description = "[x-unsloth] RSA-encrypted, base64-encoded API key for the external provider.",
    )
    provider_base_url: Optional[str] = Field(
        None,
        description = "[x-unsloth] Override base URL for the external provider.",
    )
    enable_prompt_caching: Optional[Union[bool, str]] = Field(
        None,
        description = (
            "[x-unsloth] Opt in to provider-side prompt caching. On Anthropic, "
            "boolean true attaches cache_control={type:ephemeral} to the system "
            "block so the static prefix is reused across turns. On OpenAI cloud, "
            "caching is automatic for prompts >=1024 tokens and the boolean is "
            "informational. On Gemini, pass a string cache resource name such "
            "as `cachedContents/abc123` to attach `cachedContent` on the native "
            "request (boolean true is a no-op on Gemini because creating the "
            "cache requires a separate POST /cachedContents call). Ignored for "
            "every other provider. Treated as enabled when omitted."
        ),
    )

    @field_validator("enable_prompt_caching", mode = "before")
    @classmethod
    def _coerce_enable_prompt_caching(cls, value: Any) -> Any:
        """Coerce JSON bool strings back to bool. Widening to Union[bool, str] for
        Gemini cache names would let `"false"` read as truthy, so canonical bool
        literals are coerced to keep explicit opt-outs working."""
        if isinstance(value, str):
            lowered = value.strip().lower()
            # Match Pydantic v1's bool coercion table; anything else stays a
            # string for Gemini's cachedContent resource path.
            if lowered in ("true", "t", "1", "yes", "y", "on"):
                return True
            if lowered in ("false", "f", "0", "no", "n", "off"):
                return False
        return value

    prompt_cache_ttl: Optional[str] = Field(
        None,
        description = (
            "[x-unsloth] Anthropic cache_control TTL. Defaults to the 5-minute "
            "ephemeral pool when omitted. Pass `1h` to write into the 1-hour "
            "pool instead -- 1h writes are billed at 2x base input vs 1.25x "
            "for 5m, but reads stay at 0.1x for both, so 1h pays off the "
            "moment a single extra read lands more than 5 minutes after the "
            "write. Only `5m` and `1h` are forwarded; any other value is "
            "silently ignored downstream so a stale frontend can't make the "
            "API 422 on the request. No-op on every non-Anthropic provider."
        ),
    )
    compaction_threshold: Optional[int] = Field(
        None,
        ge = 1,
        le = 2_000_000,
        description = (
            "[x-unsloth] Server-side context compaction trigger, in tokens. "
            "Per-provider routing:\n"
            "  - Anthropic (Opus 4.6+, Sonnet 4.6, Mythos preview): attaches "
            "the `compact_20260112` edit and the `compact-2026-01-12` beta "
            "header. The upstream floor is 50k; `_stream_anthropic` clamps "
            "lower values up.\n"
            "  - OpenAI cloud (api.openai.com) and Azure OpenAI Foundry "
            "(*.openai.azure.com): attaches "
            "`context_management:[{type:'compaction', compact_threshold:N}]` "
            "to /v1/responses. Effective floor is around 200k (OpenAI's "
            "canonical example); values below it surface "
            "`compact_threshold is not enabled` 400s upstream.\n"
            "Schema floor stays at ge=1 (any positive int) so the field is a "
            "silent no-op on non-cloud OpenAI-compatible bases (ollama / "
            "llama.cpp / vLLM) and every non-compaction-capable provider "
            "rather than returning 422 at request validation time. Per-"
            "provider floors are enforced in the corresponding stream helpers."
        ),
    )
    openai_code_exec_container_id: Optional[str] = Field(
        None,
        description = (
            "[x-unsloth] OpenAI shell-tool container id from the prior response "
            "in the same chat thread. When set and `code_execution` is in "
            "`enabled_tools`, the next /v1/responses call uses "
            "environment.type='container_reference' so filesystem state "
            "persists across turns. Unset → environment.type='container_auto' "
            "and OpenAI creates a fresh container. Only meaningful for the "
            "OpenAI cloud + gpt-5.5 family path; ignored otherwise."
        ),
    )
    anthropic_code_exec_container_id: Optional[str] = Field(
        None,
        description = (
            "[x-unsloth] Anthropic code_execution container id from the prior "
            "response in the same chat thread. When set and `code_execution` "
            "is in `enabled_tools`, the next /v1/messages call carries a "
            "top-level `container` field so the model sees filesystem state "
            "from earlier turns. Unset → Anthropic auto-creates a fresh "
            "container. Stale ids surface a 4xx with a `container_expired` / "
            "`container_not_found` hint; the backend emits a synthetic "
            "`container_invalidated` _toolEvent so the next turn falls back "
            "to auto-create."
        ),
    )
    fast_mode: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Anthropic fast-mode toggle. On Claude Opus 4.6 / "
            "4.7 adds the `fast-mode-2026-02-01` beta header and sends "
            "`speed: 'fast'` for higher OTPS at premium pricing. Silently "
            "ignored on every other model + provider. See "
            "https://platform.claude.com/docs/en/build-with-claude/fast-mode"
        ),
    )

    @model_validator(mode = "after")
    def _resolve_missing_tool_call_ids(self) -> "ChatCompletionRequest":
        """Fill missing tool_call_id by walking back to the preceding assistant.

        OpenAI / Anthropic passthrough require the result id to match the
        assistant's tool_calls[].id. Prefer function.name match, else first
        unconsumed tool_call; synth a random id only if none exists. A user
        turn breaks the lookup.
        """
        # Both passes below were a backwards rescan per tool result, O(n^2) for one assistant
        # with n calls. Each is now one forward pass with an index -- the same search, since
        # the backward walk never left the current user-delimited segment.
        messages = self.messages
        # The first pass only feeds the second, so with every tool_call_id present there is
        # nothing to do (the common case).
        for msg in messages:
            if msg.role == "tool" and not msg.tool_call_id:
                break
        else:
            return self

        # Pre-mark explicit ids so a missing-id sibling can't steal a claimed one.
        consumed: set[tuple[int, int]] = set()

        # Newest assistant call per explicit id in this segment; within one assistant the
        # first index wins, matching the old first-match-nearest-assistant walk. Only
        # ``str`` ids are indexed: ``tool_call_id`` is a ``str``, so nothing else matches.
        latest_by_id: dict = {}
        for asst_idx, msg in enumerate(messages):
            role = msg.role
            if role == "user":
                latest_by_id.clear()
            elif role == "assistant":
                if not msg.tool_calls:
                    continue
                here: set = set()
                for tc_idx, tc in enumerate(msg.tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id not in here:
                        here.add(tc_id)
                        latest_by_id[tc_id] = (asst_idx, tc_idx)
            elif role == "tool" and msg.tool_call_id:
                claimed = latest_by_id.get(msg.tool_call_id)
                if claimed is not None:
                    consumed.add(claimed)

        # Assistants in this segment with an unclaimed call, oldest first, so the nearest is
        # on top. A drained assistant never refills, so popping it is permanent and the walk
        # past it happens once overall, not once per tool result. Each frame keeps its calls
        # in order plus the same indexes bucketed by function name; one consumed out of turn
        # is dropped when it reaches a queue front.
        stack: list = []
        for asst_idx, msg in enumerate(messages):
            role = msg.role
            if role == "user":
                stack.clear()
                continue
            if role == "assistant":
                if not msg.tool_calls:
                    continue
                in_order: deque = deque()
                by_name: dict = {}
                for tc_idx, tc in enumerate(msg.tool_calls):
                    if (asst_idx, tc_idx) in consumed or not isinstance(tc, dict):
                        continue
                    if not tc.get("id"):
                        continue
                    function = tc.get("function")
                    function_name = function.get("name") if isinstance(function, dict) else None
                    in_order.append(tc_idx)
                    # ``name`` is a ``str``, so only a ``str`` function name can match it.
                    if isinstance(function_name, str):
                        by_name.setdefault(function_name, deque()).append(tc_idx)
                if in_order:
                    stack.append((asst_idx, msg.tool_calls, in_order, by_name))
                continue
            if role != "tool" or msg.tool_call_id:
                continue
            picked = None
            while stack:
                frame_idx, tool_calls, in_order, by_name = stack[-1]
                while in_order and (frame_idx, in_order[0]) in consumed:
                    in_order.popleft()
                if not in_order:
                    stack.pop()
                    continue
                # Name match anywhere in this assistant, else its first remaining call,
                # exactly as the old in-order scan did.
                chosen = None
                if msg.name:
                    named = by_name.get(msg.name)
                    if named is not None:
                        while named and (frame_idx, named[0]) in consumed:
                            named.popleft()
                        if named:
                            chosen = named.popleft()
                if chosen is None:
                    chosen = in_order.popleft()
                consumed.add((frame_idx, chosen))
                picked = tool_calls[chosen].get("id")
                break
            if picked is None:
                import secrets as _secrets
                picked = f"call_{_secrets.token_hex(8)}"
            msg.tool_call_id = picked
        return self

    @model_validator(mode = "after")
    def _map_thinking_to_enable_thinking(self) -> "ChatCompletionRequest":
        """Map Anthropic-style ``thinking`` parameter to internal ``enable_thinking``.

        ``thinking: {type: 'enabled'}`` sets ``enable_thinking = True`` and
        ``thinking: {type: 'disabled'}`` sets ``enable_thinking = False``.
        ``enable_thinking`` takes precedence when both are provided so that
        callers who already use the internal field are unaffected. Invalid
        ``thinking`` shapes are rejected at validation time (422).
        """
        if self.thinking is not None and self.enable_thinking is None:
            self.enable_thinking = self.thinking.type == "enabled"
        return self

    @field_validator("permission_mode", mode = "before")
    @classmethod
    def _coerce_permission_mode(cls, value: Any) -> Any:
        # Accept any string so an unknown mode degrades to 'ask' instead of a
        # 422; mirrors the tool loops' unknown -> ask fallback.
        return _normalize_permission_mode(value)

    @model_validator(mode = "after")
    def _fold_full_permission_into_bypass(self) -> "ChatCompletionRequest":
        """permission_mode='full' is the documented equivalent of
        bypass_permissions=true, so fold it in before any route guard reads
        the flag (else a full request would trip the confirm-gate rejections)."""
        if self.permission_mode == "full":
            self.bypass_permissions = True
        elif self.bypass_permissions:
            # Legacy bypass callers map onto Full access (mirrors the tool loop).
            self.permission_mode = "full"
        elif self.permission_mode == "off":
            # "Off" never prompts, so route guards must see confirm disabled.
            self.confirm_tool_calls = False
        elif (
            self.permission_mode is None
            and self.confirm_tool_calls is True
            and not (self.provider_id or self.provider_type)
        ):
            # An explicit confirm_tool_calls=True with no mode opted into the
            # pre-permission-mode contract of gating every call, so resolve it to
            # "ask" rather than let the loop apply the "auto" default, which would
            # silently weaken that opt-in to high-risk calls only. Unlike the "ask"
            # branch below this only sets permission_mode, which is inert unless
            # Unsloth's own tool loop runs, so it needs no enable_tools/mcp gate --
            # deliberate, since a process-wide --enable-tools policy can force the
            # loop when the request sets neither flag. A bare unset request
            # (confirm_tool_calls is None) still defaults to auto.
            self.permission_mode = "ask"
        elif (
            self.permission_mode == "ask"
            and self.confirm_tool_calls is None
            and not (self.provider_id or self.provider_type)
            and (self.enable_tools is True or bool(self.mcp_enabled))
        ):
            # "Ask" gates every call, so a direct API caller that omits the legacy
            # confirm flag must still hit the confirmation gate for Unsloth's own
            # tool loop. An explicit confirm_tool_calls=False wins over the mode
            # (mirrors _permission_mode_confirm and the Anthropic pre-switch guard),
            # so only self-enable when the flag is unset. Only self-enable when that
            # loop is actually requested
            # (enable_tools / mcp_enabled) -- the router enters the loop on those
            # signals, not on enabled_tools alone (which merely filters which tools
            # run). A plain client-tool passthrough (client-supplied `tools` that
            # Unsloth does not execute) must route verbatim, and external-provider
            # routing rejects confirm_tool_calls with tools, so skip the fold there.
            #
            # "auto" is deliberately NOT folded: it only prompts for a call the
            # classifier flags, so leaving confirm_tool_calls unset lets the route's
            # _confirm_gate_needs_stream apply the safe-only exception (a safe-only
            # auto selection needs no stream) instead of an explicit-confirm forcing
            # stream=true. The mode still drives the loop's per-call gate.
            self.confirm_tool_calls = True
        return self


class ChatCountTokensRequest(BaseModel):
    """Count prompt tokens for a local GGUF chat without generating."""

    model_config = {"extra": "allow"}

    model: str = Field(
        "default",
        description = "Model identifier (informational; the active model is used)",
    )
    messages: list[ChatMessage] = Field(
        ...,
        description = "Conversation messages in OpenAI chat form",
    )
    tools: Optional[list[dict]] = Field(
        None,
        description = "Optional OpenAI tool definitions included in the prompt",
    )
    enable_thinking: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Render the template in thinking mode, as a completion would",
    )
    reasoning_effort: Optional[str] = Field(
        None,
        description = "[x-unsloth] Reasoning effort level the completion would request",
    )
    preserve_thinking: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Keep historical <think> blocks in the rendered prompt",
    )
    enable_tools: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Enable tool calling for supported models",
    )
    enabled_tools: Optional[list[str]] = Field(
        None,
        description = "[x-unsloth] List of enabled built-in tool names",
    )
    mcp_enabled: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Append tools from every enabled MCP server",
    )
    deep_research_armed: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Offer the deep_research handoff tool. Its schema is in the prompt "
            "whenever the composer armed research, so the count carries it too."
        ),
    )
    rag_scope: Optional[dict] = Field(
        None,
        description = "[x-unsloth] Hidden RAG retrieval scope for search_knowledge_base",
    )
    auto_heal_tool_calls: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Strip leaked tool-call markup from replayed history",
    )
    studio_tool_history: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Mirrors ChatCompletionRequest: the replayed tool calls came from "
            "Studio's local tool loop, so _takes_tool_passthrough routes the count the way "
            "it routes the completion. Declared rather than left to extra='allow', which "
            "coerces nothing and would read the string 'false' as a claim of ownership."
        ),
    )
    permission_mode: Optional[str] = Field(
        None,
        description = "[x-unsloth] Permission level the completion would send. Only 'full' changes "
        "the prompt: it swaps the python/terminal descriptions for the unsandboxed pair and adds a "
        "sentence to the tool nudge, so a count that omits it prices a prompt the completion will "
        "not send.",
    )
    bypass_permissions: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Equivalent of permission_mode='full'. Declared explicitly (not "
        "left to extra='allow') so an omitted flag reads as None instead of raising AttributeError.",
    )

    @field_validator("permission_mode", mode = "before")
    @classmethod
    def _coerce_permission_mode(cls, value: Any) -> Any:
        return _normalize_permission_mode(value)

    @model_validator(mode = "after")
    def _fold_full_permission_into_bypass(self) -> "ChatCountTokensRequest":
        """Mirrors ChatCompletionRequest: the prompt builders read only the
        bypass flag, so 'full' has to reach them the same way here."""
        if self.permission_mode == "full":
            self.bypass_permissions = True
        elif self.bypass_permissions:
            self.permission_mode = "full"
        return self


class ToolConfirmRequest(BaseModel):
    session_id: Optional[str] = None
    approval_id: Optional[str] = None
    decision: Literal["allow", "deny"] = "deny"


# ── OpenAI shell-tool container management ─────────────────────


class OpenAIContainerRequest(BaseModel):
    """Shared body for the OpenAI container endpoints (list / create / delete).

    Carries a saved provider ID or one-time encrypted API key plus base URL so
    the route can proxy to the user's account.
    """

    provider_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Saved OpenAI provider config whose stored key may be used.",
    )
    encrypted_api_key: Optional[str] = Field(
        None,
        description = "[x-unsloth] Optional RSA-encrypted OpenAI API key override.",
    )
    provider_base_url: Optional[str] = Field(
        None,
        description = "[x-unsloth] OpenAI base URL. Only api.openai.com is supported; non-cloud bases are rejected with 400.",
    )


class CreateOpenAIContainerBody(OpenAIContainerRequest):
    name: str = Field(
        ...,
        min_length = 1,
        max_length = 256,
        description = "Human-readable container name. Surfaces in the picker UI.",
    )
    ttl_minutes: int = Field(
        20,
        ge = 1,
        le = 20,
        description = (
            "Idle-timeout TTL the new container will inherit (anchor="
            "last_active_at). OpenAI hard-caps this at 20 minutes and "
            "rejects larger values with integer_above_max_value."
        ),
    )


class DeleteOpenAIContainerBody(OpenAIContainerRequest):
    container_id: str = Field(
        ...,
        description = "OpenAI container id (cntr_...) to delete.",
    )


class OpenAIContainerSummary(BaseModel):
    """One row from GET /v1/containers, reshaped for the UI."""

    id: str
    name: Optional[str] = None
    created_at: Optional[int] = None
    last_active_at: Optional[int] = None
    expires_after_minutes: Optional[int] = None
    status: Optional[str] = None


class ListOpenAIContainersResponse(BaseModel):
    containers: list[OpenAIContainerSummary]


# ── Streaming response chunks ────────────────────────────────────


class ChoiceDelta(BaseModel):
    """Delta content for a streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


OpenAIFinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class ChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[OpenAIFinishReason] = None
    logprobs: Optional[dict] = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk in OpenAI streaming format."""

    id: str = Field(default_factory = lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory = lambda: int(time.time()))
    model: str = "default"
    choices: list[ChunkChoice]
    usage: Optional[CompletionUsage] = None
    timings: Optional[dict] = None
    context_truncated: Optional[dict] = None


# ── Non-streaming response ───────────────────────────────────────


class CompletionMessage(BaseModel):
    """The assistant's complete response message."""

    role: Literal["assistant"] = "assistant"
    # ``None`` on a pure tool-call turn (OpenAI content=null); string otherwise.
    content: Optional[str] = None
    refusal: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class CompletionChoice(BaseModel):
    """A single choice in a non-streaming response."""

    index: int = 0
    message: CompletionMessage
    finish_reason: OpenAIFinishReason = "stop"
    logprobs: Optional[dict] = None


class CompletionUsage(BaseModel):
    """Token usage statistics (approximate)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[dict] = Field(
        default_factory = lambda: {"cached_tokens": 0, "audio_tokens": 0}
    )
    completion_tokens_details: Optional[dict] = Field(
        default_factory = lambda: {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }
    )


class ChatCompletion(BaseModel):
    """Non-streaming chat completion response."""

    id: str = Field(default_factory = lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory = lambda: int(time.time()))
    model: str = "default"
    choices: list[CompletionChoice]
    usage: CompletionUsage = Field(default_factory = CompletionUsage)
    system_fingerprint: Optional[str] = None


# =====================================================================
# OpenAI Responses API Models  (/v1/responses)
# =====================================================================


# ── Request models ──────────────────────────────────────────────


class ResponsesInputTextPart(BaseModel):
    """Text content part in a Responses API message (type=input_text)."""

    type: Literal["input_text"]
    text: str


class ResponsesInputImagePart(BaseModel):
    """Image content part in a Responses API message (type=input_image)."""

    type: Literal["input_image"]
    image_url: str = Field(..., description = "data:image/png;base64,... or https://...")
    detail: Optional[Literal["auto", "low", "high", "original"]] = "auto"


class ResponsesOutputTextPart(BaseModel):
    """Assistant ``output_text`` content part replayed on subsequent turns.

    Clients looping on a stateless Responses endpoint round-trip prior assistant
    messages as ``output_text`` parts; we keep the text and ignore the
    annotations/logprobs when flattening into Chat Completions.
    """

    type: Literal["output_text"]
    text: str
    annotations: Optional[list] = None
    logprobs: Optional[list] = None

    model_config = {"extra": "allow"}


class ResponsesUnknownContentPart(BaseModel):
    """Catch-all for unmodelled content-part types.

    Keeps validation green for newer part types (e.g. ``input_audio``); skipped
    during normalisation rather than rejected with a 422.
    """

    type: str

    model_config = {"extra": "allow"}


ResponsesContentPart = Union[
    ResponsesInputTextPart,
    ResponsesInputImagePart,
    ResponsesOutputTextPart,
    ResponsesUnknownContentPart,
]


class ResponsesInputMessage(BaseModel):
    """A single message in the Responses API input array."""

    type: Optional[Literal["message"]] = None
    role: Literal["system", "user", "assistant", "developer"]
    content: Union[str, list[ResponsesContentPart]]

    # Codex attaches a `phase` field to assistant messages and requires clients
    # to preserve it across turns; we round-trip it, llama-server ignores it.
    model_config = {"extra": "allow"}


class ResponsesFunctionCallInputItem(BaseModel):
    """A prior assistant function_call replayed in a multi-turn Responses input.

    Tool calls are top-level input items (not nested), correlated by ``call_id``.
    """

    type: Literal["function_call"]
    id: Optional[str] = Field(None, description = "Item id assigned by the server (e.g. fc_...)")
    call_id: str = Field(
        ...,
        description = "Correlation id matching a function_call_output on the next turn.",
    )
    name: str
    arguments: str = Field(..., description = "JSON string of the arguments the model produced.")
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ResponsesFunctionCallOutputInputItem(BaseModel):
    """A tool result supplied by the client for a prior function_call.

    Replaces Chat Completions' ``role="tool"`` message. Correlated to its
    originating call by ``call_id``.
    """

    type: Literal["function_call_output"]
    id: Optional[str] = None
    call_id: str
    output: Union[str, list] = Field(
        ..., description = "String or content-array result of the tool call."
    )
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ResponsesUnknownInputItem(BaseModel):
    """Catch-all for unmodelled Responses input item types.

    Covers ``reasoning`` items and future types. Dropped during normalisation
    (GGUFs can't consume them), but kept in the union so unrelated turns don't 422.
    """

    type: str

    model_config = {"extra": "allow"}


def _responses_input_item_discriminator(v: Any) -> str:
    """Route a Responses input item to the correct tagged variant.

    Pydantic's smart-union matching misreports errors when a strict-``Literal``
    variant doesn't match; an explicit discriminator makes routing deterministic
    and falls through to the catch-all.
    """
    if isinstance(v, dict):
        t = v.get("type")
        r = v.get("role")
    else:
        t = getattr(v, "type", None)
        r = getattr(v, "role", None)
    if t == "function_call":
        return "function_call"
    if t == "function_call_output":
        return "function_call_output"
    if r is not None or t == "message":
        return "message"
    return "unknown"


ResponsesInputItem = Annotated[
    Union[
        Annotated[ResponsesInputMessage, Tag("message")],
        Annotated[ResponsesFunctionCallInputItem, Tag("function_call")],
        Annotated[ResponsesFunctionCallOutputInputItem, Tag("function_call_output")],
        Annotated[ResponsesUnknownInputItem, Tag("unknown")],
    ],
    Discriminator(_responses_input_item_discriminator),
]


class ResponsesFunctionTool(BaseModel):
    """Flat function-tool definition for the Responses API request.

    Unlike Chat Completions (nested under a ``"function"`` key), this uses a flat
    shape with ``type``/``name``/``description``/``parameters``/``strict`` at top level.
    """

    type: Literal["function"]
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None
    strict: Optional[bool] = None


class ResponsesRequest(BaseModel):
    """OpenAI Responses API request."""

    model: str = Field("default", description = "Model identifier")
    input: Union[str, list[ResponsesInputItem]] = Field(
        default = [],
        description = "Input text or list of messages / function_call / function_call_output items",
    )
    instructions: Optional[str] = Field(None, description = "System / developer instructions")
    temperature: Optional[float] = Field(None, ge = 0.0, le = 2.0)
    top_p: Optional[float] = Field(None, ge = 0.0, le = 1.0)
    max_output_tokens: Optional[int] = Field(None, ge = 1)
    stream: bool = Field(False, description = "Whether to stream the response via SSE")

    # OpenAI function-calling fields, forwarded via the Chat Completions
    # pass-through. Plain list so built-in tool shapes round-trip without
    # validation errors; the translator forwards only ``type=="function"`` entries.
    tools: Optional[list[dict]] = Field(
        None,
        description = (
            "Responses-shape function tool definitions. Entries with "
            '`type="function"` are translated to the Chat Completions nested '
            "shape before being forwarded to llama-server; other tool types "
            "(built-in web_search, file_search, mcp, ...) are accepted for SDK "
            "compatibility but ignored on the llama-server passthrough."
        ),
    )
    tool_choice: Optional[Any] = Field(
        None,
        description = (
            "'auto' | 'required' | 'none' | {'type': 'function', 'name': ...} — "
            "the Responses-shape forcing object is translated to the Chat "
            "Completions nested shape internally."
        ),
    )
    parallel_tool_calls: Optional[bool] = None

    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    metadata: Optional[dict] = None
    truncation: Optional[Any] = None
    user: Optional[str] = None
    text: Optional[Any] = None
    reasoning: Optional[Any] = None

    model_config = {"extra": "allow"}


# ── Response models ─────────────────────────────────────────────


class ResponsesOutputTextContent(BaseModel):
    """A text content block inside an output message."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list = Field(default_factory = list)


class ResponsesOutputMessage(BaseModel):
    """An output message in the Responses API response."""

    type: Literal["message"] = "message"
    id: str = Field(default_factory = lambda: f"msg_{uuid.uuid4().hex[:12]}")
    status: Literal["completed", "in_progress"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[ResponsesOutputTextContent] = Field(default_factory = list)


class ResponsesOutputReasoningContent(BaseModel):
    """A reasoning text content block inside a reasoning output item."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ResponsesOutputReasoning(BaseModel):
    """A top-level reasoning output item in the Responses API response."""

    type: Literal["reasoning"] = "reasoning"
    id: str = Field(default_factory = lambda: f"rs_{uuid.uuid4().hex[:12]}")
    status: Literal["completed", "in_progress", "incomplete"] = "completed"
    summary: list = Field(default_factory = list)
    content: Optional[list[ResponsesOutputReasoningContent]] = None


class ResponsesOutputFunctionCall(BaseModel):
    """A function-call output item in the Responses API response.

    Each tool call is its own top-level ``output`` item, correlated via ``call_id``.
    """

    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory = lambda: f"fc_{uuid.uuid4().hex[:12]}")
    call_id: str
    name: str
    arguments: str = Field(..., description = "JSON string of the arguments the model produced.")
    status: Literal["completed", "in_progress", "incomplete"] = "completed"


ResponsesOutputItem = Union[
    ResponsesOutputMessage,
    ResponsesOutputReasoning,
    ResponsesOutputFunctionCall,
]


class ResponsesUsage(BaseModel):
    """Token usage for a Responses API response (input_tokens, not prompt_tokens)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    """Top-level Responses API response object."""

    id: str = Field(default_factory = lambda: f"resp_{uuid.uuid4().hex[:12]}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory = lambda: int(time.time()))
    status: Literal["completed", "in_progress", "failed"] = "completed"
    model: str = "default"
    output: list[ResponsesOutputItem] = Field(default_factory = list)
    usage: ResponsesUsage = Field(default_factory = ResponsesUsage)
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[str] = None
    metadata: dict = Field(default_factory = dict)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    text: Optional[Any] = None
    tool_choice: Optional[Any] = None
    tools: list = Field(default_factory = list)
    truncation: Optional[Any] = None


# =====================================================================
# Anthropic Messages API Models  (/v1/messages)
# =====================================================================


# ── Request models ─────────────────────────────────────────────


class AnthropicTextBlock(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicImageSource(BaseModel):
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class AnthropicImageBlock(BaseModel):
    type: Literal["image"]
    source: AnthropicImageSource


class AnthropicToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict


class AnthropicToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, list] = ""

    @field_validator("content", mode = "before")
    @classmethod
    def _coerce_null_content(cls, v):
        # Some clients send null content for an empty tool result; the str|list
        # union would 400 on it, so treat null as "".
        return "" if v is None else v


# Block types the converter translates explicitly. Anything else (thinking /
# redacted_thinking, a provider block a resumed session replays, or a future type)
# is accepted as an unknown block and dropped by the converter, rather than 400-ing
# the whole request on strict validation.
_KNOWN_ANTHROPIC_BLOCK_TYPES = frozenset(
    {"text", "image", "tool_use", "tool_result", "thinking", "redacted_thinking"}
)
# Thinking blocks are replayed only in assistant turns; the converter drops them
# from user content, so accepting them there would silently lose a user turn.
_USER_ANTHROPIC_BLOCK_TYPES = frozenset({"text", "image", "tool_use", "tool_result"})


class AnthropicUnknownBlock(BaseModel):
    type: str
    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def _only_unknown_types(cls, v):
        # Known types parse as their typed models above (so a malformed known block
        # still fails cleanly); this fallback only catches the rest.
        if v in _KNOWN_ANTHROPIC_BLOCK_TYPES:
            raise ValueError("known block type handled by its typed model")
        return v


class AnthropicThinkingBlock(BaseModel):
    # Clients replay thinking blocks with tool results (Anthropic's tool-use
    # protocol requires it), so the request model must accept them; conversion
    # drops them from the prompt.
    type: Literal["thinking"]
    thinking: str = ""
    signature: str = ""
    model_config = {"extra": "allow"}


class AnthropicRedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"]
    data: str = ""
    model_config = {"extra": "allow"}


AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
    AnthropicUnknownBlock,
]


def _anthropic_content_to_system_text(content: Any) -> str:
    """Convert misplaced system message content into Anthropic system text."""
    if content is None:  # null content must not become the literal "None"
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            if block is not None:
                parts.append(str(block))
        return "\n\n".join(part for part in parts if part)
    return str(content)


def _merge_anthropic_system(system: Any, additions: list[str]) -> Any:
    if not additions:
        return system

    addition_blocks = [{"type": "text", "text": text} for text in additions if text.strip()]
    if not addition_blocks:
        return system

    if system is None:
        return addition_blocks[0]["text"] if len(addition_blocks) == 1 else addition_blocks
    if isinstance(system, str):
        return "\n\n".join([system, *[block["text"] for block in addition_blocks]])
    if isinstance(system, list):
        return [*system, *addition_blocks]
    return system


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, list[AnthropicContentBlock]]

    @model_validator(mode = "before")
    @classmethod
    def _normalize_content(cls, data):
        # Role-aware leniency that never silently drops real user input:
        #  - assistant: a resumed tool-only turn's null content -> "" (str|list would
        #    400 on null; "" keeps the converter's `for block in content` safe).
        #    Unknown blocks (thinking / future types) validate via
        #    AnthropicUnknownBlock and are dropped by the converter.
        #  - user: keep strict. Null user content stays None so str|list rejects it
        #    (400) rather than forwarding an empty prompt; and reject block types the
        #    converter cannot translate, since it silently skips unknown user blocks
        #    -- a user turn made only of them would validate yet send no content
        #    (silent data loss).
        if not isinstance(data, dict):
            return data
        content = data.get("content")
        if data.get("role") == "assistant":
            # Coerce only an explicit null (resumed tool-only turn). A missing
            # content key stays malformed so the required-field check still 400s.
            if "content" in data and content is None:
                return {**data, "content": ""}
            return data
        if isinstance(content, list):
            for block in content:
                btype = (
                    block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                )
                # Guard the value: a non-string type is unsupported too, and a
                # membership test on an unhashable value would raise TypeError
                # (escaping as a 500 instead of a clean 400).
                if not isinstance(btype, str) or btype not in _USER_ANTHROPIC_BLOCK_TYPES:
                    raise ValueError(f"unsupported content block type {btype!r} in a user message")
        return data


class AnthropicTool(BaseModel):
    # User-defined client tools have input_schema; Anthropic-schema client tools
    # and server tools use type/name.
    type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[dict] = None
    model_config = {"extra": "allow"}


class AnthropicThinkingConfig(BaseModel):
    # Deliberately `str`, not a Literal. Anthropic ships thinking types beyond
    # enabled/disabled (adaptive tiers), and Claude Code sends them -- a strict
    # Literal turns an unrecognized value into a hard 400, which is worse than
    # the silent drop this replaced. Only "disabled" means off; treat anything
    # else as a request to think.
    type: str = "enabled"
    # Accepted for wire compatibility; llama-server has no thinking budget.
    budget_tokens: Optional[int] = None
    model_config = {"extra": "allow"}


class AnthropicMessagesRequest(BaseModel):
    model: str = "default"
    max_tokens: Optional[int] = None
    messages: list[AnthropicMessage]
    system: Optional[Union[str, list]] = None
    tools: Optional[list[AnthropicTool]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    metadata: Optional[dict] = None
    # [x-unsloth] extensions mirroring the OpenAI endpoint convenience fields
    min_p: Optional[float] = Field(
        None, ge = 0.0, le = 1.0, description = "[x-unsloth] Min-p sampling threshold"
    )
    repetition_penalty: Optional[float] = Field(
        None, ge = 1.0, le = 2.0, description = "[x-unsloth] Repetition penalty"
    )
    presence_penalty: Optional[float] = Field(
        None, ge = 0.0, le = 2.0, description = "[x-unsloth] Presence penalty"
    )
    enable_tools: Optional[bool] = None
    enabled_tools: Optional[list[str]] = None
    # Anthropic's native extended-thinking control. Only `type` is honored:
    # llama-server has no thinking-token budget, so `budget_tokens` is accepted
    # and ignored rather than 400'd (Claude Code always sends it alongside).
    thinking: Optional[AnthropicThinkingConfig] = None
    # [x-unsloth] reasoning controls mirroring the OpenAI endpoint. These win
    # over `thinking` when both are present, matching enable_tools precedence.
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "max", "xhigh"]
    ] = None
    preserve_thinking: Optional[bool] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Conversation ID for scoping stateful tool sessions (e.g. stdio MCP); stays per-thread where session_id may be shared project-wide.",
    )
    cancel_id: Optional[str] = None
    bypass_permissions: Optional[bool] = Field(
        False,
        description = "[x-unsloth] Bypass Permissions: when true, disable the python/terminal execution sandbox (safety checks, command blocklist, resource limits) for server-side tool calls. Secret env vars are still stripped. Declared explicitly (not relied on via extra='allow') so omitted requests default to False instead of raising AttributeError.",
    )
    permission_mode: Optional[str] = Field(
        None,
        description = "[x-unsloth] Permission level for local tool calls: 'ask' pauses every call, 'auto' ('Approve for me') only pauses calls detected as high risk, 'off' never pauses (sandbox stays on), 'full' equals bypass_permissions=true. Unset defaults to 'auto' for the per-call gate; a non-streaming request without an explicit mode runs the loop. An unrecognized value (e.g. from a newer client) is treated as 'ask'. Declared explicitly so omitted requests default to None instead of raising AttributeError.",
    )
    auto_heal_tool_calls: Optional[bool] = Field(
        True,
        description = "[x-unsloth] Auto-detect and fix malformed tool calls from model output (mirrors the Chat Completions field; applies to the client-tool passthrough).",
    )
    nudge_tool_calls: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Opt-in tool-call recovery; mirrors the Chat Completions nudge_tool_calls field and defaults off.",
    )
    model_config = {"extra": "allow"}

    def resolved_enable_thinking(self) -> Optional[bool]:
        """Effective on/off, preferring the x-unsloth field over `thinking`."""
        if self.enable_thinking is not None:
            return self.enable_thinking
        if self.thinking is not None:
            return self.thinking.type != "disabled"
        return None

    @model_validator(mode = "before")
    @classmethod
    def normalize_system_messages(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        messages = data.get("messages")
        if not isinstance(messages, list):
            return data

        normalized_messages: list[Any] = []
        system_additions: list[str] = []
        changed = False

        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                system_additions.append(
                    _anthropic_content_to_system_text(message.get("content", ""))
                )
                changed = True
                continue
            normalized_messages.append(message)

        if not changed:
            return data

        normalized = dict(data)
        normalized["messages"] = normalized_messages
        normalized["system"] = _merge_anthropic_system(normalized.get("system"), system_additions)
        return normalized

    @field_validator("permission_mode", mode = "before")
    @classmethod
    def _coerce_permission_mode(cls, value: Any) -> Any:
        # Accept any string so an unknown mode degrades to 'ask' instead of a
        # 422; mirrors the tool loops' unknown -> ask fallback.
        return _normalize_permission_mode(value)

    @model_validator(mode = "after")
    def _fold_full_permission_into_bypass(self) -> "AnthropicMessagesRequest":
        """permission_mode='full' equals bypass_permissions=true (mirrors the
        Chat Completions request)."""
        if self.permission_mode == "full":
            self.bypass_permissions = True
        elif self.bypass_permissions:
            # Legacy bypass callers map onto Full access (mirrors the tool loop).
            self.permission_mode = "full"
        elif self.permission_mode == "off":
            # "Off" never prompts, so route guards must see confirm disabled.
            self.confirm_tool_calls = False
        return self


# ── Response models ────────────────────────────────────────────


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponseTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicResponseToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class AnthropicResponseThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    # Anthropic signs thinking blocks so they can be replayed on a later turn.
    # Nothing local can produce a valid signature, so it stays empty; clients
    # that only render the trace do not check it.
    signature: str = ""


AnthropicResponseBlock = Union[
    AnthropicResponseTextBlock,
    AnthropicResponseToolUseBlock,
    AnthropicResponseThinkingBlock,
]


class AnthropicMessagesResponse(BaseModel):
    id: str = Field(default_factory = lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicResponseBlock] = Field(default_factory = list)
    model: str = "default"
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory = AnthropicUsage)


# ── Diffusion (local text-to-image) ──


class DiffusionLoadRequest(BaseModel):
    """Request to load a local diffusion (text-to-image) checkpoint."""

    model_path: str = Field(..., description = "Diffusion repo id or local path")
    gguf_filename: Optional[str] = Field(
        None,
        description = "The chosen single-file checkpoint (GGUF or safetensors) inside "
        "model_path. Required for the gguf / single_file kinds; omit for a full pipeline.",
    )
    model_kind: Optional[Literal["gguf", "single_file", "pipeline"]] = Field(
        None,
        description = "How to load the model (null = auto-detect from gguf_filename): gguf "
        "(single-file GGUF transformer, dequantised on-device), single_file (single-file "
        "safetensors transformer, e.g. fp8), or pipeline (a full diffusers repo via "
        "from_pretrained, embedded quant auto-applied). Non-GGUF kinds are restricted to "
        "unsloth/* repos (or a local path).",
    )
    base_repo: Optional[str] = Field(
        None, description = "Companion diffusers repo for VAE/text-encoders (default: family base)"
    )
    family_override: Optional[str] = Field(
        None, description = "Force a family when it can't be inferred from the repo id"
    )
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated repos")
    cpu_offload: bool = Field(False, description = "Enable model CPU offload to fit low-VRAM cards")
    memory_mode: Optional[Literal["auto", "fast", "balanced", "low_vram"]] = Field(
        None,
        description = "Memory policy: auto (measured), fast (resident), balanced "
        "(stream the transformer, near-resident speed, moderate VRAM "
        "cut), low_vram (offload every component, lowest VRAM, slower). "
        "Overrides cpu_offload when set.",
    )
    speed_mode: Optional[Literal["off", "eager", "default", "max"]] = Field(
        None,
        description = "Opt-in speed optims (default off -> bit-identical output): "
        "eager (channels_last + cudnn + attention + fused RMSNorm/AdaLayerNorm patches, "
        "NO torch.compile -> fast first image, no compile tax), "
        "default (also regional torch.compile where eligible), "
        "max (also TF32 + fused QKV).",
    )
    text_encoder_quant: Optional[Literal["fp8", "fp8_dynamic", "int8", "nvfp4"]] = Field(
        None,
        description = "Quantise the companion text encoder(s): fp8 (layerwise cast, ~2x smaller, "
        "CUDA cc>=8.9), fp8_dynamic (torchao compute fp8 on the tensor cores, ~2x + faster, "
        "cc>=8.9), int8 (torchao compute int8 with per-family keep-bf16 layers; falls back to "
        "fp8 where no schedule exists; cc>=8.0), or nvfp4 (~4x smaller, Blackwell sm_100+). A "
        "memory-vs-quality tradeoff (shifts fine detail), not free; pairs well with balanced mode. "
        "Fails CLOSED when NOTHING could be cast (409, or a load-progress error); an int8 request "
        "downgraded to fp8 loads and is reported through the status resolved record instead.",
    )
    transformer_quant: Optional[Literal["auto", "none", "off", "int8", "fp8", "nvfp4", "mxfp8"]] = (
        Field(
            None,
            description = "Transformer compute dtype. UNSET or auto (the default) picks the "
            "fastest precision the hardware supports: the DENSE bf16 transformer "
            "is loaded instead of the GGUF and torchao-quantised onto the "
            "low-precision tensor cores (data-center fp8, consumer/Ampere int8), "
            "falling back to the GGUF when the device, VRAM or disk cannot take "
            "it. none/off pins running the GGUF as-is; an explicit scheme forces "
            "that scheme. Dense path needs CUDA + bf16. An EXPLICIT scheme fails "
            "CLOSED: where it cannot be honored the load is refused (409, or an "
            "error phase on load-progress for the footprint-dependent declines) "
            "rather than silently running the GGUF at another precision. Only "
            "auto falls back.",
        )
    )
    transformer_quant_fast_accum: Optional[bool] = Field(
        None,
        description = "fp8 only: FP8 matmul accumulate. null auto-detects by GPU class "
        "(fast FP16 accumulate on consumer/workstation cards, where FP32 "
        "accumulate is ~2x slower; precise FP32 accumulate on data-center "
        "HBM cards, which are not nerfed). true/false force it. Negligible "
        "quality effect (below the fp8 quant noise floor); no overflow risk.",
    )
    transformer_prequant_path: Optional[str] = Field(
        None,
        description = "Local path to a pre-quantized transformer checkpoint (built by "
        "scripts/build_prequant_checkpoint.py) for the requested transformer_quant "
        "scheme. Loads the already-quantized weights with the dense bf16 never on the "
        "GPU (~half the load VRAM and a smaller download). null uses the family's hosted "
        "checkpoint if configured, else quantises the dense transformer at load time. "
        "A local path installs arbitrary weights into the served model, so it is "
        "ignored unless the path resolves inside a directory the operator allowlisted "
        "via UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH (one or more directories, separated by "
        "the OS path separator). A bare on/off value such as '1' is deliberately not "
        "accepted -- it must name an allowed directory.",
    )
    loras: Optional[list[LoraSpec]] = Field(
        None,
        max_length = 8,
        description = "LoRA adapters to BAKE into a torchao int8/fp8 build: they attach on "
        "the dense transformer before quantisation and compilation (the only ordering the "
        "quantized fast path supports), so they ride inside the compiled build. Weight "
        "changes and disabling apply live at generation time; CHANGING the adapter set "
        "needs a reload with the new selection. Ignored by every other load kind (bf16 / "
        "bnb-4bit loads take adapters at generation time; GGUF-as-is has no dense "
        "transformer). Also forces the dense build path: a baked-LoRA load skips the "
        "hosted pre-quantized checkpoint and pays the dense load peak.",
    )
    attention_backend: Optional[
        Literal[
            "auto",
            "native",
            "sdpa",
            "cudnn",
            "flash",
            "flash2",
            "flash3",
            "flash4",
            "sage",
            "xformers",
            "aiter",
        ]
    ] = Field(
        None,
        description = "Attention kernel via the diffusers dispatcher. auto picks the best "
        "exact backend for the device (cuDNN fused attention on NVIDIA, ~1.18x and "
        "near-lossless, when a speed profile is active; native SDPA elsewhere and when "
        "speed=off). native (alias sdpa) forces default SDPA; cudnn/flash/flash3/flash4 are exact "
        "(kernel/arch-gated); sage is INT8 attention (a small quality cost, consumer "
        "friendly); xformers/aiter are memory-efficient (NVIDIA) / AMD ROCm. An "
        "unavailable kernel falls back to the default.",
    )
    transformer_cache: Optional[Literal["off", "fbcache"]] = Field(
        None,
        description = "Opt-in step caching (off by default). fbcache = First-Block-Cache: "
        "reuse the transformer tail across denoise steps when the first block's residual "
        "barely changes (~1.4x on Flux 28-step at LPIPS ~0.08). For MANY-step models "
        "(Flux / Qwen-Image); leave off for few-step distilled models (e.g. Z-Image-Turbo), "
        "which have no caching headroom. Composes with compile (drops fullgraph "
        "automatically); incompatible models run uncached.",
    )
    transformer_cache_threshold: Optional[float] = Field(
        None,
        ge = 0.0,
        le = 1.0,
        description = "FBCache residual threshold (higher = skips more steps = faster, lower "
        "quality). null auto-picks 0.08 (0.12 when the transformer is quantised, which "
        "shifts the residual distribution).",
    )
    gpu_ids: Optional[List[int]] = Field(
        None,
        description = "CUDA / ROCm physical indices this load may use, or null for automatic. "
        "Neither engine shards a diffusion checkpoint, so a selection of several cards resolves "
        "to the one with the most free VRAM. Refused with a 400 when an index does not exist "
        "here; ignored on XPU / MPS / CPU, which have no applicator for a physical index.",
    )

    @field_validator("attention_backend", mode = "before")
    @classmethod
    def _normalize_attention_backend(cls, value):
        # The dispatcher accepts case/whitespace variants, but the Literal above is validated before any normaliser runs, so fold it here.
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("loras")
    @classmethod
    def _unique_lora_ids(cls, value: Optional[list["LoraSpec"]]) -> Optional[list["LoraSpec"]]:
        # Same guard DiffusionGenerateRequest carries, and it matters more here: _resolve_lora_set
        # suffixes colliding adapter names, so a repeated id resolves the SAME adapter twice and
        # set_adapters stacks both copies past the per-adapter weight bound. On the generation path
        # that is one bad image; on this path the adapters are baked into the quantized build
        # before compilation, so the unintended combination rides every image until a reload.
        if value:
            seen: set[str] = set()
            for spec in value:
                if spec.id in seen:
                    raise ValueError(
                        f"duplicate LoRA id '{spec.id}'; list each adapter at most once"
                    )
                seen.add(spec.id)
        return value


class LoraSpec(BaseModel):
    """One LoRA adapter to apply for a generation, referenced by its discovery id.

    The id is resolved against the backend's own LoRA catalog + local scan (see
    core/inference/diffusion_lora.py); the client never supplies a raw filesystem
    path, so an arbitrary file can't be loaded. Weight 0 disables the adapter.
    """

    id: str = Field(
        ..., min_length = 1, max_length = 512, description = "LoRA discovery id (repo id or local stem)"
    )
    weight: float = Field(
        1.0, ge = 0.0, le = 2.0, description = "Adapter strength; 0 disables, 1.0 is full strength"
    )


class ControlNetSpec(BaseModel):
    """A ControlNet to condition this generation on: a discovery id plus a control image.

    The id resolves against the backend's ControlNet catalog + local scan (see
    core/inference/diffusion_controlnet.py); the client never supplies a raw filesystem path.
    ``image`` is either an already-made control map (``control_type='passthrough'``) or a source
    image the backend turns into a map (``control_type='canny'``). strength 0 disables it.
    """

    id: str = Field(
        ...,
        min_length = 1,
        max_length = 512,
        description = "ControlNet discovery id (repo id or local name)",
    )
    image: str = Field(
        ...,
        min_length = 1,
        max_length = 32 * 1024 * 1024,
        description = "Base64/data-URL control image (a source image or a preprocessed map)",
    )
    control_type: str = Field(
        "passthrough",
        description = "How to derive the control map: 'passthrough' (already a map) or 'canny'",
    )
    strength: float = Field(
        1.0, ge = 0.0, le = 2.0, description = "ControlNet conditioning scale; 0 disables"
    )
    guidance_start: float = Field(
        0.0, ge = 0.0, le = 1.0, description = "Fraction of steps at which ControlNet begins"
    )
    guidance_end: float = Field(
        1.0, ge = 0.0, le = 1.0, description = "Fraction of steps at which ControlNet ends"
    )

    @model_validator(mode = "after")
    def _check_guidance_range(self) -> "ControlNetSpec":
        # An inverted range means "act over no steps"; reject it as a clean 422 instead of a 500 deep in the denoise.
        if self.guidance_start > self.guidance_end:
            raise ValueError("guidance_start must be <= guidance_end")
        return self


class DiffusionGenerateRequest(BaseModel):
    """Request to generate one image from the loaded diffusion model."""

    prompt: str = Field(..., min_length = 1, description = "Text prompt")
    negative_prompt: Optional[str] = Field(
        None, description = "What to avoid (if the model supports it)"
    )
    width: int = Field(1024, ge = 256, le = 2048, description = "Image width in pixels (multiple of 16)")
    height: int = Field(
        1024, ge = 256, le = 2048, description = "Image height in pixels (multiple of 16)"
    )
    steps: int = Field(9, ge = 1, le = 100, description = "Number of denoising steps")
    guidance: float = Field(0.0, ge = 0.0, le = 20.0, description = "Classifier-free guidance scale")
    # le = 2**53-1: seeds round-trip through JSON recipes, where JavaScript rounds larger integers and a restored recipe would differ.
    seed: Optional[int] = Field(
        None, ge = 0, le = 2**53 - 1, description = "Seed for reproducibility (random if omitted)"
    )
    batch_size: int = Field(
        1, ge = 1, le = 32, description = "Images generated in one forward pass (VRAM-heavy)"
    )
    # Batched multi-image generation: a prompt list renders one image per prompt (txt2img only), a seed list one per seed. Each image carries its OWN seed.
    prompts: Optional[list[str]] = Field(
        None,
        min_length = 1,
        max_length = 32,
        description = "Prompt list for batched generation: one image per prompt in a single "
        "forward pass (plain text-to-image only). Overrides `prompt` for the images; "
        "`prompt` is still required as the fallback/display value.",
    )
    seeds: Optional[list[int]] = Field(
        None,
        min_length = 1,
        max_length = 32,
        description = "Per-image seeds for batched generation: one image per seed (with "
        "`prompts`, lengths must match; alone, every image uses `prompt`). Each image is "
        "individually reproducible from its own seed.",
    )

    @field_validator("prompts")
    @classmethod
    def _non_empty_prompts(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is not None and any(not p.strip() for p in value):
            raise ValueError("every prompt in prompts must be non-empty")
        return value

    @field_validator("seeds")
    @classmethod
    def _seeds_json_safe(cls, value: Optional[list[int]]) -> Optional[list[int]]:
        # Same JSON safe-integer bound as `seed`, so every per-image seed survives the gallery recipe.
        if value is not None and any(s < 0 or s > 2**53 - 1 for s in value):
            raise ValueError("every seed must be between 0 and 2**53 - 1")
        return value

    @model_validator(mode = "after")
    def _prompts_seeds_lengths_match(self) -> "DiffusionGenerateRequest":
        if (
            self.prompts is not None
            and self.seeds is not None
            and len(self.prompts) != len(self.seeds)
        ):
            raise ValueError(
                f"prompts and seeds must have the same length (got {len(self.prompts)} "
                f"prompts, {len(self.seeds)} seeds)"
            )
        return self

    # Image-conditioned workflows (base64 or data-URL): init_image alone runs img2img, init_image + mask_image runs inpaint.
    # Cap each base64 string so one request cannot buffer a multi-GB payload; ~32 MiB fits a full 4096px image.
    init_image: Optional[str] = Field(
        None,
        max_length = 32 * 1024 * 1024,
        description = "Base64/data-URL source image for img2img or inpaint (omit for txt2img)",
    )
    mask_image: Optional[str] = Field(
        None,
        max_length = 32 * 1024 * 1024,
        description = "Base64/data-URL mask for inpaint (white = repaint, black = keep). "
        "Requires init_image.",
    )
    strength: Optional[float] = Field(
        None,
        # EXCLUSIVE lower bound: strength 0 leaves zero denoising steps, which raises in FLUX/Qwen/Z-Image and crashes SDXL img2img.
        gt = 0.0,
        le = 1.0,
        description = "img2img/inpaint denoise strength: low values stay close to the "
        "source, 1 fully redraws it. Must be greater than 0. Ignored for txt2img.",
    )
    upscale: Optional[float] = Field(
        None,
        ge = 1.0,
        le = 4.0,
        description = "Upscale (hires fix) factor for an init_image: enlarges the source "
        "by this multiple and re-denoises at low strength. Requires init_image; "
        "ignored for txt2img/inpaint/edit.",
    )
    reference_images: Optional[list[str]] = Field(
        None,
        max_length = 3,
        description = "Additional reference images (base64/data-URL) for the FLUX.2 reference "
        "workflow, combined with init_image. Up to 3; ignored by other workflows.",
    )
    loras: Optional[list[LoraSpec]] = Field(
        None,
        max_length = 8,
        description = "LoRA adapters to apply for this generation (by discovery id + weight). "
        "Omitted/empty applies none and behaves exactly as before. Rejected with a clear "
        "message when the loaded model or its quantisation can't apply LoRA.",
    )
    controlnet: Optional[ControlNetSpec] = Field(
        None,
        description = "ControlNet conditioning for this generation (id + control image + strength). "
        "Omitted applies none and behaves exactly as before. Rejected with a clear message when "
        "the loaded model or its quantisation can't apply ControlNet.",
    )

    @field_validator("loras")
    @classmethod
    def _unique_lora_ids(cls, value: Optional[list[LoraSpec]]) -> Optional[list[LoraSpec]]:
        # Both apply paths suffix colliding adapter names, so a repeated id would load the SAME adapter twice and stack its effect past the weight bound.
        if value:
            seen: set[str] = set()
            for spec in value:
                if spec.id in seen:
                    raise ValueError(
                        f"duplicate LoRA id '{spec.id}'; list each adapter at most once"
                    )
                seen.add(spec.id)
        return value

    @field_validator("reference_images")
    @classmethod
    def _bounded_reference_items(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        # Each reference is a base64 image; bound its length like init_image so several cannot buffer a multi-GB payload.
        if value is not None:
            for item in value:
                if len(item) > 32 * 1024 * 1024:
                    raise ValueError("each reference image must be at most 32 MiB (base64)")
        return value

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, value: int) -> int:
        # Z-Image requires dimensions divisible by 16 (8x VAE downsample + 2x patch); non-multiples crash deep in the pipeline.
        if value % 16 != 0:
            raise ValueError("must be a multiple of 16")
        return value

    @model_validator(mode = "after")
    def _batch_seeds_json_safe(self) -> "DiffusionGenerateRequest":
        # A batch derives seeds as seed..seed+batch_size-1, so a derived top-of-batch seed can exceed the 2**53-1 JSON-safe cap.
        if self.seed is not None and self.seed + self.batch_size - 1 > 2**53 - 1:
            raise ValueError(
                "seed + batch_size - 1 must not exceed 2**53 - 1 so every per-image seed "
                "stays JSON-safe (lower the seed or the batch_size)"
            )
        return self


class GalleryImage(BaseModel):
    """A persisted image's full generation recipe (embedded in the PNG too)."""

    id: str = Field(..., description = "Stable id (the on-disk filename stem)")
    url: str = Field(..., description = "Relative URL to fetch the PNG bytes")
    prompt: str = Field(..., description = "Prompt used")
    negative_prompt: Optional[str] = Field(None, description = "Negative prompt, if any")
    width: int = Field(..., description = "Image width")
    height: int = Field(..., description = "Image height")
    steps: int = Field(..., description = "Denoising steps")
    guidance: float = Field(..., description = "Guidance scale")
    seed: int = Field(..., description = "Seed used for THIS image")
    batch_seed: Optional[int] = Field(
        None,
        description = (
            "Seed restore must replay this image from. For a batch_size batch that is the base "
            "seed the batch launched with (the native engine derives per-image seeds as "
            "base + index, so the derived seed alone would not reproduce it); for a "
            "prompts/seeds list, where each image carries its own seed, it equals seed. "
            "Older records without it fall back to seed."
        ),
    )
    batch_index: int = Field(0, description = "Position within its batch (0-based)")
    batch_size: int = Field(
        1, description = "Batch size used; with batch_index it lets restore replay this image"
    )
    model: Optional[str] = Field(None, description = "Model repo id that produced it")
    # The load-time BUILD. The repo id alone does not identify a pipeline (quant choice, torchao scheme, baked adapters), so without these a recipe cannot be rebuilt.
    model_kind: Optional[str] = Field(
        None, description = "How the model was loaded: gguf, single_file or pipeline"
    )
    gguf_filename: Optional[str] = Field(
        None,
        description = "The single-file checkpoint the load committed, for a gguf/single_file load",
    )
    transformer_quant: Optional[str] = Field(
        None,
        description = "Transformer quantisation scheme actually engaged on the dense fast path "
        "(int8/fp8/nvfp4/mxfp8), or null when the GGUF ran as-is",
    )
    text_encoder_quant: Optional[str] = Field(
        None,
        description = "Text-encoder quantisation actually engaged (fp8/fp8_dynamic/int8/nvfp4), "
        "or null when the dense bf16 encoder ran. Absent on records written before this existed.",
    )
    memory_mode: Optional[str] = Field(
        None, description = "Memory mode the load ran under: auto | fast | balanced | low_vram"
    )
    offload_policy: Optional[str] = Field(
        None,
        description = "Offload policy actually engaged: none | group | model | sequential. Part of "
        "the build: an offloaded pipeline declines the torchao text-encoder modes.",
    )
    baked_loras: list[str] = Field(
        default_factory = list,
        description = "Adapter ids baked into the transformer AT LOAD TIME (before quantize + "
        "compile). Part of the build, not of this generation: reloading without them gives a "
        "different pipeline even though disabling them here contributes nothing to the image.",
    )
    loras: list[str] = Field(
        default_factory = list, description = "LoRA adapters applied, formatted as 'id:weight'"
    )
    controlnet: Optional[str] = Field(
        None, description = "ControlNet applied, formatted as 'id:control_type:strength'"
    )
    # Conditioned-workflow settings. The images themselves are NOT persisted (user uploads with their own lifetime), so these say what ran and let the client ask for them back.
    workflow: Optional[str] = Field(
        None,
        description = "Workflow that produced it: txt2img, img2img, inpaint, upscale, edit, "
        "reference or controlnet. Absent on records written before this was recorded.",
    )
    strength: Optional[float] = Field(
        None, description = "img2img/inpaint denoise strength, when the workflow used one"
    )
    upscale: Optional[float] = Field(None, description = "Upscale factor, for the upscale workflow")
    controlnet_guidance: Optional[str] = Field(
        None, description = "ControlNet guidance interval, formatted as 'start:end'"
    )
    reference_image_count: Optional[int] = Field(
        None, description = "How many reference images the reference workflow used"
    )
    created_at: float = Field(..., description = "Creation time (epoch seconds)")
    # Library state, not recipe: stored beside the PNG, so older files simply read as unset.
    pinned: bool = Field(False, description = "Pinned to the front of the gallery")
    archived: bool = Field(False, description = "Moved to the archived shelf, hidden from the strip")


class GalleryFlagsPatch(BaseModel):
    """Partial update of one gallery item's pin/archive flags; omitted fields are left alone."""

    pinned: Optional[bool] = Field(None, description = "Pin (True) or unpin (False) the item")
    archived: Optional[bool] = Field(None, description = "Archive (True) or restore (False) the item")


class DiffusionGenerateResponse(BaseModel):
    """The persisted gallery records for one generation call (a batch)."""

    images: list[GalleryImage] = Field(..., description = "Saved records, one per image in the batch")


class GalleryListResponse(BaseModel):
    """A newest-first page of persisted images, for infinite scroll."""

    images: list[GalleryImage] = Field(default_factory = list)
    has_more: bool = Field(False, description = "Whether older images remain past this page")


class DiffusionGenerateProgressResponse(BaseModel):
    """Live per-step progress for an in-flight generation."""

    active: bool = Field(False, description = "Whether a generation is running")
    step: int = Field(0, description = "Denoising steps completed so far")
    total_steps: int = Field(0, description = "Total denoising steps for this run")
    fraction: float = Field(0.0, description = "step / total_steps, clamped to [0,1]")
    eta_seconds: Optional[float] = Field(None, description = "Estimated seconds remaining")


class DiffusionLoadProgressResponse(BaseModel):
    """Download/finalize progress for an in-flight diffusion load."""

    phase: Optional[Literal["downloading", "finalizing", "ready", "error"]] = Field(
        None, description = "Load phase; null when idle"
    )
    bytes_downloaded: int = Field(0, description = "Bytes present in the HF cache so far")
    bytes_total: int = Field(0, description = "Estimated total bytes to download (0 = unknown)")
    fraction: float = Field(0.0, description = "bytes_downloaded / bytes_total, clamped to [0,1]")
    error: Optional[str] = Field(None, description = "Failure message when phase is 'error'")


class DiffusionResolvedControl(BaseModel):
    """One Advanced control's engaged value + provenance, for the "Auto: X" badges.

    ``value`` is what actually applied (a scheme string, a mode string, ``null`` when the
    control is off, or ``true``/``false`` for cpu_offload), so it is typed ``Any``.
    ``source`` is "auto" when this backend decided it or "explicit" when the caller did;
    ``reason`` is the short human-readable why the frontend shows as a tooltip.

    ``requested`` and ``status`` carry the OTHER half: what was asked for and whether it was
    honored. Without them a declined request was indistinguishable from an honored one -- the
    engaged value was reported truthfully, but the ask it replaced was gone, so the UI kept
    advertising a precision that never engaged.
    """

    value: Any = Field(
        None, description = "The engaged value: a string, a boolean (cpu_offload), or null."
    )
    requested: Any = Field(
        None,
        description = "What the caller asked for, verbatim (a string, a boolean for cpu_offload), "
        "or null when they left this control to the backend. Compare with ``value`` to see "
        "whether the request survived.",
    )
    source: str = Field(..., description = '"auto" (backend decided) or "explicit" (caller set it)')
    status: str = Field(
        "applied",
        description = 'Whether the request was honored: "applied" (it was, or there was no '
        'request), "fell_back" (an explicit request was declined and something else engaged) or '
        '"unsupported" (an explicit request cannot run on this host / model at all). Defaulted '
        "so a client reading an older backend's payload still parses.",
    )
    reason: str = Field("", description = "Short human-readable reason for the resolved value.")


class DiffusionDownloadPlanEntry(BaseModel):
    """One repo the pick needs, with the exact files to fetch from it."""

    repo_id: str = Field(..., description = "Repo to download from")
    files: List[str] = Field(
        default_factory = list,
        description = "Exact files the loader reads. Scoped on purpose: a full snapshot would "
        "also pull the packaged root single, transformer/ shards and fp16 twins the loader "
        "never opens (tens of GB on a FLUX repo).",
    )
    bytes: int = Field(
        0, description = "Declared size of the files still missing from cache, 0 when unknown"
    )
    gguf_filename: Optional[str] = Field(
        None, description = "Set when this entry is the single-file GGUF checkpoint"
    )
    checkpoint: bool = Field(
        False,
        description = "This entry holds the SELECTED model rather than a companion repo. The "
        "planner has to say so: a gated base is staged from its ungated mirror, so the entry's "
        "repo id is not the id the caller picked and the role cannot be recovered downstream.",
    )


class DiffusionDownloadPlanResponse(BaseModel):
    """What to download before a load, so the Hub download manager can fetch it with the
    same file scope the loader would. Empty entries mean nothing to download (local path)."""

    entries: List[DiffusionDownloadPlanEntry] = Field(default_factory = list)
    total_bytes: int = Field(
        0, description = "Sum of the remaining download entries, 0 when ready or unknown"
    )
    required_bytes: int = Field(
        0,
        description = "Full declared footprint of every file the load requires, including cached files",
    )
    checkpoint_bytes: int = Field(
        0, description = "Declared size of the selected checkpoint within required_bytes"
    )
    incompatible_reason: Optional[str] = Field(
        None,
        description = "Why this pick cannot load as selected (today: a FLUX.2 GGUF paired with a "
        "different-size base), so the picker can refuse at selection time instead of after a "
        "multi-GB download. Reported rather than raised: the images page falls back to "
        "/images/load on any plan failure, which would start that very download. Null means "
        "nothing is known to be wrong -- the check reads metadata only and stays silent on an "
        "unreadable header, an unmapped base or an offline host.",
    )


class DiffusionStatusResponse(BaseModel):
    """Current diffusion backend state."""

    loaded: bool = Field(False, description = "Whether a diffusion model is loaded")
    repo_id: Optional[str] = Field(None, description = "Loaded repo id or local path")
    family: Optional[str] = Field(None, description = "Detected diffusion family")
    base_repo: Optional[str] = Field(None, description = "Companion diffusers base repo")
    device: Optional[str] = Field(None, description = "Device the pipeline is on")
    dtype: Optional[str] = Field(None, description = "Compute dtype")
    model_kind: Optional[str] = Field(
        None, description = "Resolved load kind: gguf | single_file | pipeline (gates GGUF-only UI)"
    )
    gguf_variant: Optional[str] = Field(
        None, description = "Selected GGUF quantisation variant (for example Q8_0)"
    )
    cpu_offload: bool = Field(False, description = "Whether CPU offload is engaged")
    offload_policy: Optional[str] = Field(
        None, description = "Resolved offload policy: none | group | model | streaming | sequential"
    )
    vae_tiling: bool = Field(False, description = "Whether VAE tiling/slicing is enabled")
    memory_mode: Optional[str] = Field(None, description = "Requested memory mode")
    speed_mode: Optional[str] = Field(None, description = "Requested speed mode")
    speed_optims: list[str] = Field(
        default_factory = list, description = "Speed optimisations actually engaged"
    )
    text_encoder_quant: Optional[str] = Field(
        None, description = "Text-encoder quantisation engaged: fp8 | nvfp4 | null"
    )
    transformer_quant: Optional[str] = Field(
        None,
        description = "Transformer quant engaged on the dense fast path: int8 | fp8 | "
        "nvfp4 | mxfp8 | null (null = the GGUF transformer was loaded)",
    )
    attention_backend: Optional[str] = Field(
        None,
        description = "Attention backend engaged via the diffusers dispatcher (e.g. "
        "_native_cudnn), or null for the default SDPA",
    )
    transformer_cache: Optional[str] = Field(None, description = "Step cache engaged: fbcache | null")
    workflows: list[str] = Field(
        default_factory = list,
        description = "Image workflows the loaded family supports (drives UI tab gating): "
        "txt2img, img2img, inpaint. Empty when nothing is loaded or on the native engine.",
    )
    engine: Optional[str] = Field(None, description = "Active diffusion engine: diffusers | sd_cpp")
    native_mode: Optional[str] = Field(
        None,
        description = "Native sd.cpp execution mode: server (resident sd-server) | oneshot "
        "(per-image sd-cli) | null (diffusers engine)",
    )
    fallback_reason: Optional[str] = Field(
        None,
        description = "Why diffusers was chosen over the native sd.cpp engine (null when none)",
    )
    supports_lora: bool = Field(
        False,
        description = "Whether the loaded model + quantisation can apply LoRA adapters (drives the "
        "LoRA picker's enabled state). torchao int8/fp8 builds support LoRA via the load-time "
        "bake: select the adapters when loading; weight changes apply live, a different adapter "
        "set needs a reload. False on unsupported families/quant (e.g. nvfp4/mxfp8, "
        "GGUF-via-diffusers, or Qwen-Image on the native engine).",
    )
    supports_controlnet: bool = Field(
        False,
        description = "Whether the loaded model can apply a ControlNet (drives the ControlNet "
        "picker's enabled state). Diffusers only, for families with a ControlNet pipeline; False "
        "for the native engine, GGUF-via-diffusers, and torchao fp8/int8 dense.",
    )
    # Additive per-control provenance {control: {value, source, reason}}; null when nothing is loaded. Declared explicitly so pydantic extra='ignore' keeps it.
    resolved: Optional[Dict[str, DiffusionResolvedControl]] = Field(
        None,
        description = "Per-control resolved value + provenance (source auto|explicit + reason), "
        "keyed by Advanced control name; null when unloaded or unavailable.",
    )


class DiffusionInferenceInfo(BaseModel):
    """One family's bf16 component sizes + estimated resident footprint per quant scheme.

    Mirrors the dicts ``family_inference_infos()`` returns: the bf16-resident transformer /
    text-encoder / VAE sizes, and the estimated resident GB under bf16 and each dense
    transformer-quant scheme (transformer * factor + companions), rounded to 1 decimal."""

    family: str = Field(..., description = "Diffusion family name (auto-policy table key).")
    transformer_bf16_gb: float = Field(..., description = "bf16-resident transformer size in GB.")
    text_encoders_bf16_gb: float = Field(
        ..., description = "bf16-resident text encoder(s) size in GB."
    )
    vae_bf16_gb: float = Field(..., description = "bf16-resident VAE size in GB.")
    estimated_resident_gb: Dict[str, float] = Field(
        ...,
        description = "Estimated resident GB keyed by scheme: bf16, int8, fp8, mxfp8, nvfp4.",
    )


class DiffusionInferenceInfoResponse(BaseModel):
    """Static per-family footprint summary for the Advanced Dtype tradeoff (GET
    /api/inference/images/info). Hardware-independent: no GPU probing, so it is served
    from the pure auto-policy tables and is safe to fetch before anything is loaded."""

    families: List[DiffusionInferenceInfo] = Field(default_factory = list)


# ── OpenAI-compatible images API (POST /v1/images/generations) ──
# Shapes mirror OpenAI's CreateImageRequest / ImagesResponse. GPT-image-only knobs are accepted and ignored, like dall-e-2.
# The size string is parsed and `stream` rejected in the route; everything Pydantic can check declaratively is here.


class ImageGenerationRequest(BaseModel):
    """OpenAI ``CreateImageRequest`` for ``POST /v1/images/generations``.

    ``prompt`` is the only required field, per the spec. Unlisted OpenAI fields
    are ignored (Pydantic's default), matching dall-e-2's treatment of the
    GPT-image-only parameters."""

    prompt: str = Field(..., min_length = 1, description = "Text description of the image(s).")
    model: Optional[str] = Field(
        None, description = "Model id (informational; the loaded image model is used)."
    )
    n: int = Field(1, ge = 1, le = 10, description = "Number of images to generate (1-10).")
    size: str = Field(
        "auto", description = "'auto' or '<width>x<height>' (256-2048, each a multiple of 16)."
    )
    response_format: Literal["url", "b64_json"] = Field(
        "url", description = "Return each image as a URL or a base64-encoded PNG."
    )
    user: Optional[str] = Field(None, description = "End-user identifier (accepted, unused).")
    # gpt-image-only; declared so we can reject it clearly instead of returning JSON to a client that asked for an SSE stream.
    stream: Optional[bool] = Field(
        None, description = "Streaming image generation is not supported; omit or set false."
    )

    @field_validator("n", "size", "response_format", mode = "before")
    @classmethod
    def _null_means_default(cls, value, info):
        # OpenAI marks these nullable WITH a default, so an explicit null means "use the default": coalesce rather than 400.
        if value is None:
            return cls.model_fields[info.field_name].default
        return value


class ImageGenerationData(BaseModel):
    """One image in an ``ImagesResponse`` (OpenAI ``Image``). Exactly one of
    ``url`` / ``b64_json`` is set, per the request's ``response_format``; the
    route serializes with ``exclude_none`` so the unused key is omitted."""

    b64_json: Optional[str] = Field(
        None, description = "Base64-encoded PNG (response_format=b64_json)."
    )
    url: Optional[str] = Field(None, description = "URL to the PNG bytes (response_format=url).")


class ImageGenerationResponse(BaseModel):
    """OpenAI ``ImagesResponse``. dall-e-shaped: the GPT-image-only top-level
    fields (background/output_format/size/quality/usage) are omitted, since our
    sizes wouldn't satisfy their fixed enums and we report no token usage."""

    created: int = Field(..., description = "Unix timestamp (seconds) the images were created.")
    data: list[ImageGenerationData] = Field(..., description = "The generated images.")


# ── OpenAI-compatible audio API (POST /v1/audio/speech) ──


class AudioSpeechRequest(BaseModel):
    """OpenAI ``CreateSpeechRequest`` for ``POST /v1/audio/speech``.

    ``voice`` and ``speed`` are accepted for client compatibility but unused: no loaded
    TTS backend has voice or rate plumbing (CSM is fixed to speaker 0)."""

    input: str = Field(..., min_length = 1, description = "The text to synthesize.")
    model: Optional[str] = Field(
        None, description = "Model id (informational; the loaded audio model is used)."
    )
    voice: Optional[str] = Field(None, description = "Voice name (accepted, unused).")
    response_format: Optional[str] = Field(
        "wav", description = "Output container. Only 'wav' is supported."
    )
    speed: Optional[float] = Field(None, description = "Speech rate (accepted, unused).")
    provider_id: Optional[str] = Field(
        None,
        description = "[x-unsloth] Saved connection ID. When set, synthesis is proxied to that "
        "provider's /audio/speech and model/voice/speed are forwarded as sent.",
    )
    provider_base_url: Optional[str] = Field(
        None,
        description = "[x-unsloth] Browser-snapshotted connection base URL. Required with a "
        "legacy encrypted_api_key so an edit cannot route that key to another endpoint.",
    )
    encrypted_api_key: Optional[str] = Field(
        None,
        description = "[x-unsloth] Per-request key for a browser still holding a legacy "
        "provider key, used when the connection has none saved server side.",
    )

    @field_validator("response_format", mode = "before")
    @classmethod
    def _null_format_means_default(cls, value):
        # openai marks response_format nullable with a default, so an explicit null means wav
        return "wav" if value is None else value


class AudioGalleryItem(BaseModel):
    """One persisted TTS clip. ``url`` serves the WAV bytes (auth required)."""

    id: str
    url: str
    prompt: str
    model: str
    audio_type: str
    sample_rate: int
    duration_s: float
    created_at: str


class AudioGalleryListResponse(BaseModel):
    """A newest-first window of the audio gallery for infinite scroll."""

    audio: List[AudioGalleryItem] = Field(default_factory = list)
    has_more: bool = False
    next_before_mtime: Optional[float] = None
    next_before_id: Optional[str] = None


# ── Video (local text-to-video) ──


class VideoLoadRequest(BaseModel):
    """Request to load a local text-to-video checkpoint."""

    model_path: str = Field(..., description = "Video repo id or local path")
    gguf_filename: Optional[str] = Field(
        None,
        description = "The chosen single-file checkpoint (GGUF or safetensors) inside "
        "model_path. Required for the gguf / single_file kinds; omit for a full pipeline.",
    )
    model_kind: Optional[Literal["gguf", "single_file", "pipeline"]] = Field(
        None,
        description = "How to load the model (null = auto-detect from gguf_filename): gguf "
        "(single-file GGUF transformer, dequantised on-device), single_file (single-file "
        "safetensors transformer, e.g. fp8), or pipeline (a full diffusers repo via "
        "from_pretrained). Non-GGUF kinds are restricted to unsloth/* repos, the official "
        "family base repos, or a local path.",
    )
    base_repo: Optional[str] = Field(
        None,
        description = "Companion diffusers repo for VAE/text-encoders (default: family base)",
    )
    family_override: Optional[str] = Field(
        None, description = "Force a family when it can't be inferred from the repo id"
    )
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated repos")
    memory_mode: Optional[Literal["auto", "fast", "balanced", "low_vram"]] = Field(
        None,
        description = "Memory policy: auto (measured), fast (resident), balanced "
        "(stream the transformer, near-resident speed, moderate VRAM cut), low_vram "
        "(offload every component, lowest VRAM, slower).",
    )
    speed_mode: Optional[Literal["off", "eager", "default", "max"]] = Field(
        None,
        description = "Opt-in speed optims (default off -> bit-identical output): "
        "eager (channels_last + cudnn + attention + fused norm patches, NO torch.compile), "
        "default (also regional torch.compile where eligible), "
        "max (also TF32 + fused QKV). GGUF video loads default to the near-lossless "
        "compile profile.",
    )
    attention_backend: Optional[
        Literal[
            "auto",
            "native",
            "sdpa",
            "cudnn",
            "flash",
            "flash2",
            "flash3",
            "flash4",
            "sage",
            "xformers",
            "aiter",
        ]
    ] = Field(
        None,
        description = "Attention kernel via the diffusers dispatcher. auto picks the best "
        "exact backend for the device (cuDNN fused attention on NVIDIA when a speed profile "
        "is active; native SDPA elsewhere and when speed=off). native (alias sdpa) forces "
        "default SDPA; cudnn/flash/flash3/flash4 are exact (kernel/arch-gated); sage is INT8 "
        "attention; xformers/aiter are memory-efficient (NVIDIA) / AMD ROCm. An unavailable "
        "kernel falls back to the default.",
    )
    transformer_cache: Optional[Literal["off", "fbcache"]] = Field(
        None,
        description = "Opt-in step caching (off by default). fbcache = First-Block-Cache: "
        "reuse the transformer tail across denoise steps when the first block's residual "
        "barely changes. Engages on many-step schedules only; incompatible models run "
        "uncached.",
    )
    transformer_cache_threshold: Optional[float] = Field(
        None,
        ge = 0.0,
        le = 1.0,
        description = "FBCache residual threshold (higher = skips more steps = faster, lower "
        "quality). null auto-picks the family default.",
    )
    transformer_quant: Optional[Literal["auto", "none", "off", "int8", "fp8", "nvfp4", "mxfp8"]] = (
        Field(
            None,
            description = "Quantise the dense DiT(s) on a full-pipeline load. On a diffusers "
            "pipeline load the dense bf16 transformer(s) are torchao-quantised in place onto "
            "the low-precision tensor cores (data-center fp8, consumer/Ampere int8), which is "
            "faster than running dense bf16. For a dual-expert MoE family (Wan2.2-A14B) BOTH "
            "experts are quantised with the same scheme. null/none/off keeps the DiT(s) at "
            "their loaded precision; an explicit scheme forces it. Needs CUDA + bf16; ignored "
            "on gguf/single_file loads (they carry their own precision). Mirrors the image "
            "backend's transformer_quant field.",
        )
    )
    text_encoder_quant: Optional[
        Literal["auto", "none", "off", "fp8", "fp8_dynamic", "int8", "nvfp4"]
    ] = Field(
        None,
        description = "Quantise the dense companion text encoder (Gemma3 / UMT5 / Qwen2.5-VL), "
        "which loads bf16 from the base repo regardless of how the DiT was sourced and is often "
        "the largest resident component. fp8 = diffusers layerwise casting (memory only, cc >= "
        "8.9); fp8_dynamic = torchao per-row fp8 COMPUTE on the tensor cores (cc >= 8.9); int8 = "
        "torchao int8 COMPUTE with per-family keep-bf16 selection (cc >= 8.0; falls back to fp8 "
        "for a family without a measured schedule); nvfp4 = torchao 4-bit weight-only (Blackwell "
        "sm_100+). null/auto leaves the choice to the backend, which keeps the encoder dense on "
        "every family except MiniMax-H3, where it takes the hosted quantized conditioner; "
        "none/off always keeps the released bf16 encoder, which on MiniMax-H3 is the only way "
        "to ask for it. Mirrors the image backend's field.",
    )

    h3_task: Optional[Literal["fl2va", "ref2va"]] = Field(
        None,
        description = "Which MiniMax-H3 denoiser partition to bring up: fl2va (text-to-video "
        "and first/last-frame video, the default) or ref2va (omni-reference video). They are "
        "separate ~62 GB partitions, so a load serves one of them. Ignored for a GGUF pick, "
        "whose filename already names the partition; rejected if it contradicts that filename.",
    )
    gpu_ids: Optional[List[int]] = Field(
        None,
        description = "CUDA / ROCm physical indices this load may use, or null for automatic. "
        "Neither engine shards a video checkpoint, so a selection of several cards resolves to "
        "the one with the most free VRAM. Refused with a 400 when an index does not exist here; "
        "ignored on XPU / MPS / CPU, which have no applicator for a physical index. Mirrors the "
        "image backend's field.",
    )

    @field_validator("attention_backend", mode = "before")
    @classmethod
    def _normalize_attention_backend(cls, value):
        # The dispatcher accepts case/whitespace variants, but the Literal above is validated before any normaliser runs, so fold it here.
        return value.strip().lower() if isinstance(value, str) else value


class VideoReferenceVideo(BaseModel):
    """One reference video, with the soundtrack MiniMax-H3 conditions on alongside it."""

    video: str = Field(
        ...,
        max_length = 96 * 1024 * 1024,
        description = "Base64/data-URL video file, 2 to 15 seconds. Resampled onto the model's "
        "24 fps and onto the canvas its own aspect ratio resolves to.",
    )
    audio: Optional[str] = Field(
        None,
        max_length = 32 * 1024 * 1024,
        description = "Base64/data-URL soundtrack for THIS video. Omitted takes the track "
        "embedded in the file, if it has one; sent explicitly it replaces it.",
    )
    trim_start_seconds: Optional[float] = Field(
        None, ge = 0.0, description = "Inclusive start of an explicit video trim, in seconds."
    )
    trim_end_seconds: Optional[float] = Field(
        None, gt = 0.0, description = "Exclusive end of an explicit video trim, in seconds."
    )

    @model_validator(mode = "after")
    def _trim_is_a_complete_h3_interval(self) -> "VideoReferenceVideo":
        from core.inference.video_minimax_h3 import validate_h3_reference_trim
        validate_h3_reference_trim(self.trim_start_seconds, self.trim_end_seconds)
        return self


class VideoGenerateRequest(BaseModel):
    """Request to generate one clip from the loaded video model."""

    prompt: str = Field(..., min_length = 1, description = "Text prompt")
    negative_prompt: Optional[str] = Field(
        None, description = "What to avoid (if the model supports it)"
    )
    model: Optional[str] = Field(
        None,
        description = "Video model to generate on. Only read when media auto-switch is on, "
        "where a downloaded model that is not the resident one is loaded first; omit to use "
        "whatever is loaded. The Video page never sends it.",
    )
    # Width/height/num_frames/fps default per loaded family, so they are optional here. These bounds stay a COARSE outer
    # guard only -- they are family-agnostic, and a request that clears them can still be one no checkpoint can render. The
    # enforced rule is the LOADED family's own (its resolution presets and k * frame_step + frame_offset lattice), which the
    # route checks with validate_video_request_shape and rejects with a 422 naming the supported shapes. Nothing tighter
    # belongs here: with no model loaded there is no family to judge against, and that path must keep snapping as before.
    width: Optional[int] = Field(
        None,
        ge = 32,
        le = 2048,
        description = "Frame width in pixels (a resolution preset of the loaded family)",
    )
    height: Optional[int] = Field(
        None,
        ge = 32,
        le = 2048,
        description = "Frame height in pixels (a resolution preset of the loaded family)",
    )
    num_frames: Optional[int] = Field(
        None,
        ge = 1,
        le = MAX_VIDEO_NUM_FRAMES,
        description = "Number of frames; must lie on the family's temporal lattice (k * frame_step + frame_offset)",
    )
    fps: Optional[int] = Field(
        None, ge = 1, le = 120, description = "Playback frame rate (default per family)"
    )
    steps: Optional[int] = Field(
        None, ge = 1, le = 100, description = "Number of denoising steps (default per model)"
    )
    guidance: Optional[float] = Field(
        None, ge = 0.0, le = 20.0, description = "Classifier-free guidance scale (default per model)"
    )
    guidance_2: Optional[float] = Field(
        None,
        ge = 0.0,
        le = 20.0,
        description = "Low-noise-stage guidance scale for a dual-expert MoE family (Wan2.2-A14B): "
        "the guidance the second transformer uses on the low-noise denoise steps. null lets the "
        "pipeline default it to the main guidance. Ignored by single-DiT families (their pipeline "
        "signature has no second guidance kwarg).",
    )
    # le = 2**53-1: seeds round-trip through JSON recipes, where JavaScript rounds larger integers and a restored recipe would differ.
    seed: Optional[int] = Field(
        None, ge = 0, le = 2**53 - 1, description = "Seed for reproducibility (random if omitted)"
    )
    # Keyframe conditioning (MiniMax-H3). Bounded like the image backend's init_image so one
    # request cannot buffer a multi-GB payload; ~32 MiB fits a full 4096px source.
    first_frame: Optional[str] = Field(
        None,
        max_length = 32 * 1024 * 1024,
        description = "Base64/data-URL image the clip starts from (image-to-video). It sets the "
        "geometry and is stretched onto the canvas. Omit for text-to-video. Rejected by a "
        "family that takes no keyframes.",
    )
    last_frame: Optional[str] = Field(
        None,
        max_length = 32 * 1024 * 1024,
        description = "Base64/data-URL image the clip ends on. Valid on its own (generate up "
        "TO a frame) or with first_frame (first-and-last-frame video), in which case it "
        "follows the first and is centre cover-cropped onto the canvas.",
    )

    # Separate lists preserve the model's image, video, then audio packing order.
    reference_images: Optional[list[str]] = Field(
        None,
        max_length = 9,
        description = "Subject / style / scene reference images (base64/data-URL), at most 9. "
        "The prompt refers to them as <Picture 1>, <Picture 2>, ... in this order.",
    )
    reference_videos: Optional[list[VideoReferenceVideo]] = Field(
        None,
        max_length = 3,
        description = "Motion / camera reference videos, at most 3. The prompt refers to them "
        "as <Video 1>, <Video 2>, ... in this order.",
    )
    reference_audios: Optional[list[str]] = Field(
        None,
        max_length = 3,
        description = "Standalone reference audio (base64/data-URL), at most 3, for voice or "
        "score. The prompt refers to them as <Audio 1>, <Audio 2>, ... in this order. Audio "
        "cannot be the only kind of reference a request carries.",
    )
    flow_shift: Optional[float] = Field(
        None,
        gt = 0.0,
        le = 100.0,
        description = "Sigma shift of the video schedule (MiniMax-H3 ships 12.0). Higher spends "
        "more of the schedule at high noise, which reads as more motion and less detail. null "
        "keeps the released value.",
    )
    audio_flow_shift: Optional[float] = Field(
        None,
        gt = 0.0,
        le = 100.0,
        description = "Sigma shift of the audio schedule (MiniMax-H3 ships 3.0). Needs the "
        "Diffusers engine: stable-diffusion.cpp derives the audio schedule against a hardcoded "
        "3.0, so it has no flag to map this onto. null keeps the released value.",
    )
    reference_image_size: Optional[Literal["match", "max"]] = Field(
        None,
        description = "How reference images are sized: match (default) scales each down to the "
        "generation's pixel area; max uses the reference pipeline's 2048px short edge for "
        "stronger identity fidelity, several times slower. max needs the Diffusers engine -- "
        "stable-diffusion.cpp rescales every reference to the generation area regardless.",
    )

    @field_validator("reference_images", "reference_audios")
    @classmethod
    def _bounded_reference_media(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        # Bound each item like first_frame, so a list cannot buffer what one field may not.
        if value is not None:
            for item in value:
                if len(item) > 32 * 1024 * 1024:
                    raise ValueError("each reference must be at most 32 MiB (base64)")
        return value

    @model_validator(mode = "after")
    def _references_fit_the_models_budget(self) -> "VideoGenerateRequest":
        images = self.reference_images or []
        videos = self.reference_videos or []
        audios = self.reference_audios or []
        total = len(images) + len(videos) + len(audios)
        if total > 12:
            raise ValueError(f"MiniMax-H3 takes at most 12 references in total, got {total}")
        # Standalone audio must accompany an image or video reference.
        if audios and not images and not videos:
            raise ValueError(
                "reference audio needs at least one reference image or video to go with"
            )
        if (images or videos or audios) and (self.first_frame or self.last_frame):
            raise ValueError(
                "keyframes and references cannot be combined: MiniMax-H3 runs them against "
                "different denoiser partitions"
            )
        return self

    @model_validator(mode = "after")
    def _keyframe_canvas_needs_both_axes(self) -> "VideoGenerateRequest":
        # Omit both axes for "match source", or provide both for an explicit canvas.
        # KEYFRAME requests only. There a half-specified canvas is silently discarded:
        # _resolve_keyframes matches the source aspect whenever either axis is missing, so the
        # axis that was sent never reaches the render and the API would accept one recipe and
        # draw another. Without a keyframe the backend deliberately resolves the missing axis
        # from the family's default preset -- validate_video_request_shape and generate() both
        # document and implement that -- so applying the rule to every request would reject
        # half-specified LTX, Wan, Hunyuan and prompt-only H3 calls that have always been valid.
        if not (self.first_frame or self.last_frame):
            return self
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be sent together, or both omitted")
        return self


class GalleryVideo(BaseModel):
    """A persisted clip's full generation recipe (the JSON sidecar of the MP4)."""

    id: str = Field(..., description = "Stable id (the on-disk filename stem)")
    url: str = Field(..., description = "Relative URL to fetch the MP4 bytes")
    prompt: str = Field(..., description = "Prompt used")
    negative_prompt: Optional[str] = Field(None, description = "Negative prompt, if any")
    width: int = Field(..., description = "Frame width")
    height: int = Field(..., description = "Frame height")
    num_frames: int = Field(..., description = "Number of frames")
    fps: int = Field(..., description = "Playback frame rate")
    duration_s: float = Field(..., description = "Clip duration in seconds")
    steps: int = Field(..., description = "Denoising steps")
    guidance: float = Field(..., description = "Guidance scale")
    guidance_2: Optional[float] = Field(
        None, description = "Second-expert guidance scale (dual-expert families), if sent"
    )
    flow_shift: Optional[float] = Field(
        None, description = "Video-schedule sigma shift used, for families that expose it"
    )
    audio_flow_shift: Optional[float] = Field(
        None, description = "Audio-schedule sigma shift used, for families that expose it"
    )
    seed: int = Field(..., description = "Seed used")
    has_audio: bool = Field(False, description = "Whether the MP4 carries an audio track")
    conditioning: Optional[str] = Field(
        None,
        description = "How the clip was conditioned, in MiniMax-H3's own task names: t2va "
        "(prompt only), i2va (first frame), l2va (last frame) or fl2va (both). Absent on "
        "clips saved before keyframes existed.",
    )
    model: Optional[str] = Field(None, description = "Model repo id that produced it")
    # The load-time BUILD, mirroring GalleryImage: the repo id alone does not say which checkpoint
    # ran or at what precision, so a clip could not be told apart from one rendered at another. All
    # optional, so sidecars written before this existed still list.
    model_kind: Optional[str] = Field(
        None, description = "How the model was loaded: gguf, single_file or pipeline"
    )
    gguf_filename: Optional[str] = Field(
        None,
        description = "The single-file checkpoint the load committed, for a gguf/single_file load",
    )
    transformer_quant: Optional[str] = Field(
        None,
        description = "Dense DiT quantisation actually engaged (int8/fp8/nvfp4/mxfp8), or null "
        "when the DiT(s) ran at their loaded bf16 precision",
    )
    text_encoder_quant: Optional[str] = Field(
        None,
        description = "Text-encoder quantisation actually engaged (fp8/fp8_dynamic/int8/nvfp4), "
        "or null when the dense bf16 encoder ran",
    )
    memory_mode: Optional[str] = Field(
        None, description = "Memory mode the load ran under: auto | fast | balanced | low_vram"
    )
    offload_policy: Optional[str] = Field(
        None, description = "Offload policy actually engaged: none | group | model | sequential"
    )
    created_at: str = Field(..., description = "Creation time (ISO 8601 timestamp)")
    # Library state, not recipe: stored beside the clip, so older sidecars simply read as unset.
    pinned: bool = Field(False, description = "Pinned to the front of the gallery")
    archived: bool = Field(False, description = "Moved to the archived shelf, hidden from the strip")


class VideoGenerateResponse(BaseModel):
    """Acknowledgement that a generation was accepted and started.

    Generation runs as a background job (a clip takes minutes, and secure mode's
    tunnel caps the origin response window near 100 seconds, so the POST cannot
    span it). The saved gallery record arrives via GET /video/generate-progress
    when its phase reaches "completed"."""

    status: Literal["started"] = Field(
        "started", description = "Discriminator: the generation job was started"
    )
    video: Optional[GalleryVideo] = Field(
        None,
        description = "Always null (kept for response-shape compatibility); the saved "
        "record is delivered by generate-progress on completion",
    )


class VideoGalleryListResponse(BaseModel):
    """A newest-first page of persisted videos, for infinite scroll."""

    videos: list[GalleryVideo] = Field(default_factory = list)
    has_more: bool = Field(False, description = "Whether older videos remain past this page")


class VideoGenerateProgressResponse(BaseModel):
    """Live progress for an in-flight video generation, plus the terminal outcome
    of the background job POST /video/generate started."""

    active: bool = Field(False, description = "Whether a generation is running")
    phase: Optional[str] = Field(
        None,
        description = "Current phase: queued | denoise | export | completed | failed | null",
    )
    step: int = Field(0, description = "Denoising steps completed so far")
    total: int = Field(0, description = "Total denoising steps for this run")
    # Image-endpoint-compatible aliases so one poller works against both APIs.
    total_steps: int = Field(0, description = "Total denoising steps (alias of total)")
    fraction: float = Field(0.0, description = "step / total, clamped to [0,1]")
    eta_seconds: Optional[float] = Field(None, description = "Estimated seconds remaining")
    video: Optional[GalleryVideo] = Field(
        None, description = "Saved gallery record when phase is 'completed'"
    )
    error: Optional[str] = Field(
        None, description = "Client-safe failure detail when phase is 'failed'"
    )


class VideoLoadProgressResponse(BaseModel):
    """Download/finalize progress for an in-flight video load."""

    phase: Optional[Literal["downloading", "finalizing", "ready", "error"]] = Field(
        None, description = "Load phase; null when idle"
    )
    downloaded_bytes: int = Field(0, description = "Bytes present in the HF cache so far")
    expected_bytes: Optional[int] = Field(
        None, description = "Estimated total bytes to download (null = unknown)"
    )
    error: Optional[str] = Field(None, description = "Failure message when phase is 'error'")


class VideoGenerationDefaults(BaseModel):
    """Per-family generation defaults + shape constraints for the loaded video model."""

    steps: int = Field(..., description = "Default denoising steps")
    guidance: float = Field(..., description = "Default guidance scale")
    num_frames: int = Field(..., description = "Default frame count")
    fps: int = Field(..., description = "Default playback frame rate")
    frame_step: int = Field(..., description = "Temporal lattice stride")
    frame_offset: int = Field(
        1, description = "Temporal lattice offset: valid counts are k * frame_step + frame_offset"
    )
    duration_presets: list[float] = Field(
        default_factory = list, description = "Clip durations in seconds the UI offers"
    )
    resolution_multiple: int = Field(..., description = "Width/height must be divisible by this")
    resolution_presets: list[list[int]] = Field(
        default_factory = list, description = "(width, height) presets the UI offers, default first"
    )
    canvas_short_edge: Optional[int] = Field(
        None,
        description = "Short edge a source-derived canvas aims for, or null when the family "
        "has no keyframe canvas rule. With canvas_max_pixels and resolution_multiple this is "
        "the whole rule: short edge, then cap the area, then round both axes.",
    )
    canvas_max_pixels: Optional[int] = Field(
        None, description = "Area budget of a source-derived canvas, or null (see canvas_short_edge)"
    )
    flow_shift: Optional[float] = Field(
        None,
        description = "Released video-schedule sigma shift, or null when the family does not "
        "expose the control",
    )
    audio_flow_shift: Optional[float] = Field(
        None, description = "Released audio-schedule sigma shift, or null (see flow_shift)"
    )
    supports_audio_flow_shift: bool = Field(
        False,
        description = "Whether the ACTIVE engine can honour audio_flow_shift; false on "
        "stable-diffusion.cpp, which pins the audio schedule at its released value",
    )


class VideoStatusResponse(BaseModel):
    """Current video backend state."""

    loaded: bool = Field(False, description = "Whether a video model is loaded")
    repo_id: Optional[str] = Field(None, description = "Loaded repo id or local path")
    family: Optional[str] = Field(None, description = "Detected video family")
    base_repo: Optional[str] = Field(None, description = "Companion diffusers base repo")
    device: Optional[str] = Field(None, description = "Device the pipeline is on")
    dtype: Optional[str] = Field(None, description = "Compute dtype")
    model_kind: Optional[str] = Field(
        None, description = "Resolved load kind: gguf | single_file | pipeline (gates GGUF-only UI)"
    )
    engine: Optional[str] = Field(None, description = "Active video engine: diffusers | sd_cpp")
    gguf_variant: Optional[str] = Field(
        None, description = "Selected GGUF quantisation variant (for example Q8_0)"
    )
    offload_policy: Optional[str] = Field(
        None, description = "Resolved offload policy: none | group | model | sequential"
    )
    vae_tiling: bool = Field(False, description = "Whether VAE tiling is enabled")
    memory_mode: Optional[str] = Field(None, description = "Requested memory mode")
    speed_mode: Optional[str] = Field(None, description = "Requested speed mode")
    speed_optims: list[str] = Field(
        default_factory = list, description = "Speed optimisations actually engaged"
    )
    attention_backend: Optional[str] = Field(
        None,
        description = "Attention backend engaged via the diffusers dispatcher (e.g. "
        "_native_cudnn), or null for the default SDPA",
    )
    transformer_cache: Optional[str] = Field(None, description = "Step cache engaged: fbcache | null")
    transformer_quant: Optional[str] = Field(
        None,
        description = "Dense transformer quant engaged on a pipeline load: int8 | fp8 | nvfp4 | "
        "mxfp8 | null (null = the DiT(s) run at their loaded bf16 precision). For a dual-expert "
        "MoE family both experts share the reported scheme.",
    )
    text_encoder_quant: Optional[str] = Field(
        None,
        description = "Text-encoder quant engaged: fp8 | fp8_dynamic | int8 | nvfp4 | null "
        "(null = the dense bf16 encoder is loaded). An int8 request without a per-family "
        "keep-bf16 schedule is reported as the fp8 it fell back to.",
    )
    has_audio: bool = Field(
        False, description = "Whether the loaded family produces a synchronized audio track"
    )
    supports_keyframes: bool = Field(
        False,
        description = "Whether the LOADED checkpoint takes first/last-frame conditioning images "
        "(gates the image-to-video controls)",
    )
    supports_references: bool = Field(
        False,
        description = "Whether the LOADED checkpoint takes reference images / videos / audio "
        "(gates the reference-to-video controls). Never true at the same time as "
        "supports_keyframes: MiniMax-H3 serves the two from different denoiser partitions.",
    )
    h3_task: Optional[str] = Field(
        None,
        description = "The MiniMax-H3 denoiser partition that is up: fl2va | ref2va, or null "
        "for any other family",
    )
    supports_cfg: bool = Field(
        True, description = "Whether guidance and negative prompts apply to this family"
    )
    defaults: Optional[VideoGenerationDefaults] = Field(
        None, description = "Per-family generation defaults + shape constraints; null when unloaded"
    )
    # Additive per-control provenance, same shape as the diffusion status; null when nothing is loaded.
    resolved: Optional[Dict[str, DiffusionResolvedControl]] = Field(
        None,
        description = "Per-control resolved value + provenance (source auto|explicit + reason), "
        "keyed by Advanced control name; null when unloaded or unavailable.",
    )
