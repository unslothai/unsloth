# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the Inference API."""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Dict, Literal, Optional, List, Union

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)


class LoadRequest(BaseModel):
    """Request to load a model for inference"""

    model_path: str = Field(..., description = "Model identifier or local path")
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
        if value is not None and value.strip() == "":
            return None
        return value

    cache_type_kv: Optional[str] = Field(
        None,
        description = "KV cache data type for both K and V (e.g. 'f16', 'bf16', 'q8_0', 'q4_1', 'q5_1')",
    )
    gpu_ids: Optional[List[int]] = Field(
        None,
        description = "Physical GPU indices to use, for example [0, 1]. Omit or pass [] to use automatic selection. Explicit gpu_ids are unsupported when the parent CUDA_VISIBLE_DEVICES uses UUID/MIG entries. For GGUF models the picked devices are pinned via CUDA/HIP_VISIBLE_DEVICES.",
    )
    speculative_type: Optional[str] = Field(
        None,
        description = (
            "Speculative decoding mode for GGUF models. Canonical values: "
            "'auto' (platform-aware: MTP on MTP GGUFs, ngram-mod fallback "
            "for sub-3B), 'mtp' (force draft-mtp only on both GPU and CPU), "
            "'ngram' (force ngram-mod only), 'mtp+ngram' (force "
            "ngram-mod+draft-mtp chain on both platforms), 'off' (disabled). "
            "Legacy values 'default' (-> auto), 'draft-mtp' (-> mtp), "
            "'ngram-mod' (-> ngram), and 'ngram-simple' (kept as-is) are "
            "still accepted. Ignored for non-GGUF models."
        ),
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        ge = 1,
        le = 16,
        description = (
            "Max draft tokens per step for MTP speculative decoding "
            "(--spec-draft-n-max). Defaults to 2 on GPU and 3 on CPU/Mac "
            "when unset (upstream-bench sweet spot for dense Qwen3.6 MTP "
            "quants). Only applied when speculative_type resolves to "
            "'mtp' or 'mtp+ngram'."
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


class UnloadRequest(BaseModel):
    """Request to unload a model"""

    model_path: str = Field(..., description = "Model identifier to unload")


class ValidateModelRequest(BaseModel):
    """Check whether an identifier resolves to a ModelConfig; does NOT load weights."""

    model_path: str = Field(..., description = "Model identifier or local path")
    native_path_lease: Optional[str] = Field(
        None, description = "Frontend-visible signed native path grant"
    )
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated models")
    gguf_variant: Optional[str] = Field(
        None, description = "GGUF quantization variant (e.g. 'Q4_K_M')"
    )
    # Intended load settings so validate's coexistence check matches the follow-up
    # /load; defaults preserve old behavior for callers that omit them.
    max_seq_length: int = Field(0, ge = 0, le = 1048576)
    load_in_4bit: bool = Field(True)
    gpu_ids: Optional[List[int]] = Field(None)
    gpu_memory_mode: Literal["auto", "manual"] = Field(
        "auto",
        description = (
            "GGUF GPU-memory strategy intended for the follow-up load. Manual "
            "placement bypasses the training coexistence estimate: Auto layers "
            "delegate fitting to llama.cpp, while explicit layers are user-owned."
        ),
    )
    include_context_length: bool = Field(
        False,
        description = "Also read the native context length from the local GGUF header. "
        "Opt-in so the normal load preflight doesn't pay for a cache scan it doesn't need.",
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


class ValidateModelResponse(BaseModel):
    """Result of model validation.

    valid == True means from_identifier() succeeded and GGUF/LoRA/vision flags are available.
    """

    valid: bool = Field(..., description = "Whether the model identifier looks valid")
    message: str = Field(..., description = "Human-readable validation message")
    identifier: Optional[str] = Field(None, description = "Resolved model identifier")
    display_name: Optional[str] = Field(None, description = "Display name derived from identifier")
    is_gguf: bool = Field(False, description = "Whether this is a GGUF model (llama.cpp)")
    is_lora: bool = Field(False, description = "Whether this is a LoRA adapter")
    is_vision: bool = Field(False, description = "Whether this is a vision-capable model")
    is_audio: bool = Field(False, description = "Whether this is an audio-only/STT/TTS model")
    audio_type: Optional[str] = Field(None, description = "Audio type, when detected")
    has_audio_input: bool = Field(False, description = "Whether model accepts audio input")
    is_chat_capable: bool = Field(
        True, description = "Whether the model is suitable for automatic chat loading"
    )
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


class InstallLatestTransformersRequest(BaseModel):
    """Consented request to install the latest transformers release into a sidecar."""

    version: str = Field(
        ...,
        min_length = 1,
        max_length = 64,
        description = "Exact transformers version to install; must match the current "
        "latest PyPI release reported by /validate.",
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


class LoadResponse(BaseModel):
    """Response after loading a model"""

    status: str = Field(..., description = "Load status")
    model: str = Field(..., description = "Model identifier")
    display_name: str = Field(..., description = "Display name of the model")
    is_vision: bool = Field(False, description = "Whether model is a vision model")
    is_lora: bool = Field(False, description = "Whether model is a LoRA adapter")
    is_gguf: bool = Field(False, description = "Whether model is a GGUF model (llama.cpp)")
    is_diffusion: bool = Field(
        False, description = "Whether model is a block-diffusion model (DiffusionGemma)"
    )
    is_audio: bool = Field(False, description = "Whether model is a TTS audio model")
    audio_type: Optional[str] = Field(None, description = "Audio codec type: snac, csm, bicodec, dac")
    has_audio_input: bool = Field(False, description = "Whether model accepts audio input (ASR)")
    inference: dict = Field(
        ..., description = "Inference parameters (temperature, top_p, top_k, min_p)"
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
        description = "Whether the template understands the optional preserve_thinking kwarg (Qwen3.6-style)",
    )
    supports_tools: bool = Field(
        False,
        description = "Whether model supports tool calling (web search, etc.)",
    )
    cache_type_kv: Optional[str] = Field(
        None,
        description = "KV cache data type for K and V (e.g. 'f16', 'bf16', 'q8_0')",
    )
    chat_template: Optional[str] = Field(
        None,
        description = "Jinja2 chat template string (from GGUF metadata or tokenizer)",
    )
    speculative_type: Optional[str] = Field(
        None,
        description = (
            "Canonical UI-facing requested speculative decoding mode "
            "('auto' / 'mtp' / 'ngram' / 'mtp+ngram' / 'off' / "
            "'ngram-simple'), round-tripped from the original LoadRequest "
            "via _canonicalize_spec_mode. None when no model is loaded."
        ),
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        description = (
            "Active --spec-draft-n-max for MTP speculative decoding, or "
            "None when the platform default is in effect."
        ),
    )
    tensor_parallel: bool = Field(
        False,
        description = "Whether tensor-parallel split (--split-mode tensor) is active.",
    )
    gpu_memory_mode: Literal["auto", "manual"] = Field(
        "auto",
        description = "Active GPU memory strategy ('auto' or 'manual').",
    )
    gpu_layers: int = Field(
        -1,
        description = "Manual mode: requested --gpu-layers value (-1 = Auto/--fit, or when not manual).",
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
        description = "Physical GPU indices the model is pinned to, or None for automatic selection.",
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


class InferenceStatusResponse(BaseModel):
    """Current inference backend status"""

    active_model: Optional[str] = Field(
        None, description = "Currently active model display identifier"
    )
    model_identifier: Optional[str] = Field(
        None,
        description = "Loadable identifier for the active model.",
    )
    is_vision: bool = Field(False, description = "Whether the active model is a vision model")
    is_gguf: bool = Field(False, description = "Whether the active model is a GGUF model (llama.cpp)")
    is_diffusion: bool = Field(
        False, description = "Whether the active model is a block-diffusion model (DiffusionGemma)"
    )
    gguf_variant: Optional[str] = Field(None, description = "GGUF quantization variant (e.g. Q4_K_M)")
    is_audio: bool = Field(False, description = "Whether the active model is a TTS audio model")
    audio_type: Optional[str] = Field(None, description = "Audio codec type: snac, csm, bicodec, dac")
    has_audio_input: bool = Field(False, description = "Whether model accepts audio input (ASR)")
    loading: List[str] = Field(default_factory = list, description = "Models currently being loaded")
    loaded: List[str] = Field(default_factory = list, description = "Models currently loaded")
    inference: Optional[Dict[str, Any]] = Field(
        None, description = "Recommended inference parameters for the active model"
    )
    requires_trust_remote_code: bool = Field(
        False,
        description = "Whether the active model requires trust_remote_code to be enabled for loading.",
    )
    supports_reasoning: bool = Field(
        False, description = "Whether the active model supports reasoning/thinking mode"
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
        False, description = "Whether reasoning is always on (not toggleable)"
    )
    supports_preserve_thinking: bool = Field(
        False,
        description = "Whether the active model's template understands the optional preserve_thinking kwarg",
    )
    supports_tools: bool = Field(
        False, description = "Whether the active model supports tool calling"
    )
    context_length: Optional[int] = Field(None, description = "Context length of the active model")
    max_context_length: Optional[int] = Field(
        None,
        description = "Maximum context length currently available for the active model",
    )
    native_context_length: Optional[int] = Field(
        None,
        description = "Model's native context length from GGUF metadata (not capped by VRAM)",
    )
    cache_type_kv: Optional[str] = Field(
        None,
        description = "KV cache quantization dtype (e.g. 'q8_0'), or None for default",
    )
    chat_template: Optional[str] = Field(
        None, description = "Model's default chat template (Jinja2 source), if any"
    )
    chat_template_override: Optional[str] = Field(
        None,
        description = "Active chat template override applied at load time, or None if model is using its default",
    )
    speculative_type: Optional[str] = Field(
        None,
        description = (
            "Canonical UI-facing requested speculative decoding mode "
            "('auto' / 'mtp' / 'ngram' / 'mtp+ngram' / 'off' / "
            "'ngram-simple'), round-tripped from the original LoadRequest. "
            "None when no model is loaded."
        ),
    )
    spec_draft_n_max: Optional[int] = Field(
        None,
        description = (
            "Active --spec-draft-n-max for MTP speculative decoding, or "
            "None when the platform default is in effect."
        ),
    )
    tensor_parallel: bool = Field(
        False,
        description = "Whether tensor-parallel split (--split-mode tensor) is active.",
    )
    gpu_memory_mode: Literal["auto", "manual"] = Field(
        "auto",
        description = "Active GPU memory strategy ('auto' or 'manual').",
    )
    gpu_layers: int = Field(
        -1,
        description = "Manual mode: requested --gpu-layers value (-1 = Auto/--fit, or when not manual).",
    )
    n_cpu_moe: int = Field(
        0,
        description = "Manual mode: MoE expert layers pinned to CPU (--n-cpu-moe); 0 = none.",
    )
    tensor_split: Optional[List[float]] = Field(
        None,
        description = "Manual mode: relative model share per GPU (--tensor-split); None = default (split by free VRAM).",
    )
    requested_context_length: Optional[int] = Field(
        None,
        description = (
            "The n_ctx the active GGUF load was invoked with (0 = Auto). Lets the "
            "UI re-seed a Manual + Auto-layers context pin on hydration, where "
            "context_length only exposes the resolved value. None for non-GGUF."
        ),
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
        description = "Physical GPU indices the model is pinned to, or None for automatic selection.",
    )
    llama_cpp_supports_mtp: bool = Field(
        True,
        description = (
            "Whether llama.cpp supports MTP (--spec-type mtp/draft-mtp). "
            "False -> recommend `unsloth studio update`."
        ),
    )
    spec_fallback_reason: Optional[str] = Field(
        None,
        description = (
            "Why MTP was disabled on the loaded model despite being requested "
            "(auto on an MTP model, or forced mtp / mtp+ngram). "
            "'binary_no_mtp' / 'binary_outdated' -> a newer prebuilt would "
            "re-enable it (show the update affordance); 'runtime_error' -> the "
            "current build could not run it; 'drafter_not_found' -> the model's "
            "separate MTP drafter could not be resolved; 'mla_mtp_disabled' -> "
            "an Auto-mode policy downgrade: the model is MLA (GLM-5.2 et al.) "
            "whose llama.cpp MTP path runs slower than no speculation, so Auto "
            "used ngram-mod or spec-off instead -- updating won't help; choose "
            "MTP in Settings (or set UNSLOTH_MLA_MTP_ENABLED=1) to force it. "
            "None when MTP engaged or was not requested."
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
# a Literal so an unrecognized value from a newer UI/client degrades to the
# safest gate ("ask") instead of a 422; the tool loops apply the same unknown ->
# ask fallback, so normalizing here keeps that forward-compat path reachable at
# the API boundary. None stays unset ("behaves as 'ask'" without self-enabling
# the confirm gate).
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
    presence_penalty: float = Field(0.0, ge = 0.0, le = 2.0, description = "Presence penalty")
    stop: Optional[Union[str, list[str]]] = Field(
        None,
        description = "OpenAI stop sequences: a single string or list of strings at which generation halts.",
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
            "accept ['web_search', 'python', 'terminal', 'render_html']. External "
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
    confirm_tool_calls: Optional[bool] = Field(
        None,
        description = "[x-unsloth] When true, pause before each tool call and wait for the user to allow/deny it via POST /api/inference/tool-confirm.",
    )
    bypass_permissions: Optional[bool] = Field(
        False,
        description = "[x-unsloth] Bypass Permissions: when true, skip the tool-call confirmation gate AND disable the python/terminal execution sandbox (safety checks, command blocklist, resource limits). Secret env vars are still stripped. Takes precedence over confirm_tool_calls.",
    )
    permission_mode: Optional[str] = Field(
        None,
        description = (
            "[x-unsloth] Permission level for local tool calls. 'ask' pauses every "
            "call for approval; 'ask'/'auto' enable the confirmation gate on their "
            "own (needs a streaming request to deliver prompts). 'auto' ('Approve for "
            "me') only pauses calls detected as potentially unsafe (state-mutating "
            "terminal/python/MCP calls); read-only calls run immediately, and the "
            "sandbox stays on. 'full' is equivalent to bypass_permissions=true (no "
            "confirmation, no sandbox). Unset behaves as 'ask'. An unrecognized value "
            "(e.g. from a newer client) is treated as 'ask'."
        ),
    )
    auto_heal_tool_calls: Optional[bool] = Field(
        True,
        description = "[x-unsloth] Auto-detect and fix malformed tool calls from model output.",
    )
    nudge_tool_calls: Optional[bool] = Field(
        None,
        description = (
            "[x-unsloth] Opt-in, non-streaming client-tool passthrough only: when the "
            "model emitted a tool signal that healing could not repair, retry ONCE with "
            "a short nudge appended (the retry shares the full prompt prefix, so the "
            "server's KV cache is reused). Default off; UNSLOTH_TOOL_CALL_NUDGE=1 flips "
            "the process default."
        ),
    )
    context_overflow: Optional[Literal["error", "truncate_middle"]] = Field(
        None,
        description = (
            "[x-unsloth] Passthrough behavior when the prompt exceeds the real "
            "context window. 'error' (default) returns a 400 with "
            "code=context_length_exceeded. 'truncate_middle' drops middle "
            "turn-groups (system prompt, first turn, and recent turns kept; "
            "tool calls stay paired with their results) and retries."
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
        description = "[x-unsloth] Saved provider config ID. If set with encrypted_api_key, routes to external LLM.",
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
        # Pre-mark explicit ids so a missing-id sibling can't steal a claimed one.
        consumed: set[tuple[int, int]] = set()

        def _mark_consumed(start_idx: int, tool_call_id: str) -> None:
            for asst_idx in range(start_idx - 1, -1, -1):
                prev = self.messages[asst_idx]
                if prev.role == "user":
                    break
                if prev.role != "assistant" or not prev.tool_calls:
                    continue
                for tc_idx, tc in enumerate(prev.tool_calls):
                    if isinstance(tc, dict) and tc.get("id") == tool_call_id:
                        consumed.add((asst_idx, tc_idx))
                        return

        for tool_idx, msg in enumerate(self.messages):
            if msg.role == "tool" and msg.tool_call_id:
                _mark_consumed(tool_idx, msg.tool_call_id)

        for tool_idx, msg in enumerate(self.messages):
            if msg.role != "tool" or msg.tool_call_id:
                continue
            picked: str | None = None
            for asst_idx in range(tool_idx - 1, -1, -1):
                prev = self.messages[asst_idx]
                if prev.role != "assistant" or not prev.tool_calls:
                    if prev.role == "user":
                        break
                    continue
                name_match = None
                fallback = None
                for tc_idx, tc in enumerate(prev.tool_calls):
                    if (asst_idx, tc_idx) in consumed:
                        continue
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    if not tc_id:
                        continue
                    function = tc.get("function")
                    function_name = function.get("name") if isinstance(function, dict) else None
                    if msg.name and function_name == msg.name:
                        name_match = (tc_id, asst_idx, tc_idx)
                        break
                    if fallback is None:
                        fallback = (tc_id, asst_idx, tc_idx)
                chosen = name_match or fallback
                if chosen is not None:
                    picked, a, t = chosen
                    consumed.add((a, t))
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


class ToolConfirmRequest(BaseModel):
    session_id: Optional[str] = None
    approval_id: Optional[str] = None
    decision: Literal["allow", "deny"] = "deny"


# ── OpenAI shell-tool container management ─────────────────────


class OpenAIContainerRequest(BaseModel):
    """Shared body for the OpenAI container endpoints (list / create / delete).

    Carries the encrypted API key + base URL so the route can decrypt and proxy
    to the user's account, keeping the key off backend persistent storage.
    """

    encrypted_api_key: str = Field(
        ...,
        description = "[x-unsloth] RSA-encrypted, base64-encoded OpenAI API key.",
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
_KNOWN_ANTHROPIC_BLOCK_TYPES = frozenset({"text", "image", "tool_use", "tool_result"})


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


AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
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
                if not isinstance(btype, str) or btype not in _KNOWN_ANTHROPIC_BLOCK_TYPES:
                    raise ValueError(f"unsupported content block type {btype!r} in a user message")
        return data


class AnthropicTool(BaseModel):
    # Client tools have input_schema; server tools may only have type/name.
    type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[dict] = None
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
        description = "[x-unsloth] Permission level for local tool calls: 'ask' pauses every call, 'auto' only pauses calls detected as potentially unsafe, 'off' never pauses (sandbox stays on), 'full' equals bypass_permissions=true. Unset behaves as 'ask'; an unrecognized value (e.g. from a newer client) is treated as 'ask'. Declared explicitly so omitted requests default to None instead of raising AttributeError.",
    )
    auto_heal_tool_calls: Optional[bool] = Field(
        True,
        description = "[x-unsloth] Auto-detect and fix malformed tool calls from model output (mirrors the Chat Completions field; applies to the client-tool passthrough).",
    )
    nudge_tool_calls: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Opt-in, non-streaming only: retry once with a nudge when the model emitted a tool signal healing could not repair (mirrors the Chat Completions field).",
    )
    model_config = {"extra": "allow"}

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


AnthropicResponseBlock = Union[AnthropicResponseTextBlock, AnthropicResponseToolUseBlock]


class AnthropicMessagesResponse(BaseModel):
    id: str = Field(default_factory = lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicResponseBlock] = Field(default_factory = list)
    model: str = "default"
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory = AnthropicUsage)
