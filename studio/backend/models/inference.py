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
        description = "Physical GPU indices to use, for example [0, 1]. Omit or pass [] to use automatic selection. Explicit gpu_ids are unsupported when the parent CUDA_VISIBLE_DEVICES uses UUID/MIG entries. Not supported for GGUF models.",
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
    llama_extra_args: Optional[List[str]] = Field(
        None,
        description = (
            "Extra arguments forwarded verbatim to llama-server for GGUF models. "
            "One token per list entry, e.g. ['--top-k', '20', '--seed', '42']. "
            "Studio-managed flags (model identity, port, context length, GPU placement, "
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
    include_context_length: bool = Field(
        False,
        description = "Also read the native context length from the local GGUF header. "
        "Opt-in so the normal load preflight doesn't pay for a cache scan it doesn't need.",
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

    Studio-normalised shape (file_data or file_url, plus optional filename/media_type).
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
    Studio sets provider thinking budgets itself.
    """

    type: Literal["disabled", "enabled"] = "disabled"


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
            "Studio forwards the tools to the backend so the model returns structured "
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
    cancel_id: Optional[str] = None
    bypass_permissions: Optional[bool] = Field(
        False,
        description = "[x-unsloth] Bypass Permissions: when true, disable the python/terminal execution sandbox (safety checks, command blocklist, resource limits) for server-side tool calls. Secret env vars are still stripped. Declared explicitly (not relied on via extra='allow') so omitted requests default to False instead of raising AttributeError.",
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
        "memory-vs-quality tradeoff (shifts fine detail), not free; pairs well with balanced mode.",
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
            "that scheme. Dense path needs CUDA + bf16.",
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
        "Loading a local path unpickles the file (arbitrary code execution), so it is "
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

    @field_validator("attention_backend", mode = "before")
    @classmethod
    def _normalize_attention_backend(cls, value):
        # The dispatcher accepts case/whitespace variants ("CuDNN", " sage "), but the Literal
        # above is validated before any normaliser runs, so fold a string to its canonical
        # lower/stripped form here -- otherwise valid casing gets a 422.
        return value.strip().lower() if isinstance(value, str) else value


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
        # An inverted range (start > end) means "act over no steps"; reject it as a clean 422
        # instead of letting the diffusers pipeline 500 deep in the denoise.
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
    # le = 2**53-1: seeds round-trip through JSON gallery recipes, where JavaScript rounds
    # integers above Number.MAX_SAFE_INTEGER -- a restored recipe would then generate a
    # different image. Random seeds are already masked to this range.
    seed: Optional[int] = Field(
        None, ge = 0, le = 2**53 - 1, description = "Seed for reproducibility (random if omitted)"
    )
    batch_size: int = Field(
        1, ge = 1, le = 32, description = "Images generated in one forward pass (VRAM-heavy)"
    )
    # Batched multi-image generation (diffusers engine): a prompt list renders one image per
    # prompt in a single batched forward (txt2img only); a seed list renders one image per
    # seed. Each image carries its OWN generator seed, so any batch member replays alone.
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
        # Same JSON safe-integer bound as `seed`, so every per-image seed survives the
        # round-trip through the gallery recipe.
        if value is not None and any(s < 0 or s > 2**53 - 1 for s in value):
            raise ValueError("every seed must be between 0 and 2**53 - 1")
        return value

    @model_validator(mode = "after")
    def _prompts_seeds_lengths_match(self) -> "DiffusionGenerateRequest":
        if self.prompts is not None and self.seeds is not None and len(self.prompts) != len(
            self.seeds
        ):
            raise ValueError(
                f"prompts and seeds must have the same length (got {len(self.prompts)} "
                f"prompts, {len(self.seeds)} seeds)"
            )
        return self
    # Image-conditioned workflows (base64 or data-URL): init_image alone runs img2img,
    # init_image + mask_image runs inpaint. Both require a family with the matching pipeline or
    # the load is rejected. Cap each base64 string so one request can't buffer a multi-GB payload
    # (decoded dimensions are bounded separately); ~32 MiB fits a full 4096px image yet rejects abuse.
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
        ge = 0.0,
        le = 1.0,
        description = "img2img/inpaint denoise strength: 0 keeps the source, 1 fully "
        "redraws it. Ignored for txt2img.",
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
        # Both apply paths break alias collisions by suffixing the adapter name/file, so a
        # repeated id would load the SAME adapter several times and stack its effect past the
        # per-adapter weight bound. The UI already blocks duplicates; reject them for API clients
        # too so each adapter takes effect at most once.
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
        # Each reference is a base64 image; bound its length like init_image/mask_image so
        # several references can't buffer a multi-GB payload.
        if value is not None:
            for item in value:
                if len(item) > 32 * 1024 * 1024:
                    raise ValueError("each reference image must be at most 32 MiB (base64)")
        return value

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_16(cls, value: int) -> int:
        # Z-Image requires dimensions divisible by 16 (8x VAE downsample + 2x patch).
        # Non-multiples crash deep in the pipeline, so reject them here for a clean 422.
        if value % 16 != 0:
            raise ValueError("must be a multiple of 16")
        return value

    @model_validator(mode = "after")
    def _batch_seeds_json_safe(self) -> "DiffusionGenerateRequest":
        # A batch derives per-image seeds as seed .. seed+batch_size-1. The base seed is capped at
        # 2**53-1 to round-trip through the JSON recipe, but a derived top-of-batch seed near the cap
        # can exceed it, where the frontend rounds it and a restored recipe replays a different
        # image. Reject at the boundary so an API client can't persist an unreplayable seed.
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
            "Base seed the batch was launched with. The native engine derives per-image seeds as "
            "base + index, so restore must replay from this base, not from the derived per-image "
            "seed; older records without it fall back to seed."
        ),
    )
    batch_index: int = Field(0, description = "Position within its batch (0-based)")
    batch_size: int = Field(
        1, description = "Batch size used; with batch_index it lets restore replay this image"
    )
    model: Optional[str] = Field(None, description = "Model repo id that produced it")
    loras: list[str] = Field(
        default_factory = list, description = "LoRA adapters applied, formatted as 'id:weight'"
    )
    controlnet: Optional[str] = Field(
        None, description = "ControlNet applied, formatted as 'id:control_type:strength'"
    )
    created_at: float = Field(..., description = "Creation time (epoch seconds)")


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
    """

    value: Any = Field(
        None, description = "The engaged value: a string, a boolean (cpu_offload), or null."
    )
    source: str = Field(..., description = '"auto" (backend decided) or "explicit" (caller set it)')
    reason: str = Field("", description = "Short human-readable reason for the resolved value.")


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
    cpu_offload: bool = Field(False, description = "Whether CPU offload is engaged")
    offload_policy: Optional[str] = Field(
        None, description = "Resolved offload policy: none | group | model | sequential"
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
    # Additive: per-Advanced-control provenance {control: {value, source, reason}}. Present only
    # on backends that record it; null when nothing is loaded or on older backends. The frontend
    # renders an "Auto: X" badge next to each control whose source == "auto". Declared explicitly
    # so pydantic's extra='ignore' doesn't drop the resolved record.
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
#
# Shapes mirror OpenAI's CreateImageRequest / ImagesResponse so off-the-shelf clients work
# unchanged. The loaded image GGUF stands in for the model; GPT-image-only knobs (quality,
# style, background, output_format, ...) are accepted and ignored, like dall-e-2. The size
# string is parsed and `stream` is rejected in the route (where the diffusion backend is in
# reach); everything Pydantic can check declaratively lives here.


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
    # gpt-image-only; declared so we can reject it clearly instead of returning JSON to a
    # client that asked for an SSE stream.
    stream: Optional[bool] = Field(
        None, description = "Streaming image generation is not supported; omit or set false."
    )

    @field_validator("n", "size", "response_format", mode = "before")
    @classmethod
    def _null_means_default(cls, value, info):
        # OpenAI marks these nullable WITH a default, so an explicit null means "use the
        # default" -- coalesce it instead of 400-ing a spec-valid body.
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
    text_encoder_quant: Optional[Literal["fp8", "fp8_dynamic", "int8", "nvfp4"]] = Field(
        None,
        description = "Quantise the dense companion text encoder (Gemma3 / UMT5 / Qwen2.5-VL), "
        "which loads bf16 from the base repo regardless of how the DiT was sourced and is often "
        "the largest resident component. fp8 = diffusers layerwise casting (memory only, cc >= "
        "8.9); fp8_dynamic = torchao per-row fp8 COMPUTE on the tensor cores (cc >= 8.9); int8 = "
        "torchao int8 COMPUTE with per-family keep-bf16 selection (cc >= 8.0; falls back to fp8 "
        "for a family without a measured schedule); nvfp4 = torchao 4-bit weight-only (Blackwell "
        "sm_100+). null keeps the encoder dense. Mirrors the image backend's field.",
    )

    @field_validator("attention_backend", mode = "before")
    @classmethod
    def _normalize_attention_backend(cls, value):
        # The dispatcher accepts case/whitespace variants ("CuDNN", " sage "), but the
        # Literal above is validated before any normaliser runs, so fold a string to its
        # canonical lower/stripped form here -- otherwise valid casing gets a 422.
        return value.strip().lower() if isinstance(value, str) else value


class VideoGenerateRequest(BaseModel):
    """Request to generate one clip from the loaded video model."""

    prompt: str = Field(..., min_length = 1, description = "Text prompt")
    negative_prompt: Optional[str] = Field(
        None, description = "What to avoid (if the model supports it)"
    )
    # Width/height/num_frames/fps default per loaded family (the backend snaps them to its
    # required multiples/lattice), so they are optional here.
    width: Optional[int] = Field(
        None, ge = 32, le = 2048, description = "Frame width in pixels (family multiple)"
    )
    height: Optional[int] = Field(
        None, ge = 32, le = 2048, description = "Frame height in pixels (family multiple)"
    )
    num_frames: Optional[int] = Field(
        None,
        ge = 1,
        le = 1024,
        description = "Number of frames; snapped to the family's temporal lattice",
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
    # le = 2**53-1: seeds round-trip through JSON gallery recipes, where JavaScript rounds
    # integers above Number.MAX_SAFE_INTEGER -- a restored recipe would then generate a
    # different clip. Random seeds are already masked to this range.
    seed: Optional[int] = Field(
        None, ge = 0, le = 2**53 - 1, description = "Seed for reproducibility (random if omitted)"
    )


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
    seed: int = Field(..., description = "Seed used")
    has_audio: bool = Field(False, description = "Whether the MP4 carries an audio track")
    model: Optional[str] = Field(None, description = "Model repo id that produced it")
    created_at: str = Field(..., description = "Creation time (ISO 8601 timestamp)")


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
    frame_step: int = Field(
        ..., description = "Temporal lattice: valid counts are k * frame_step + 1"
    )
    resolution_multiple: int = Field(..., description = "Width/height must be divisible by this")
    resolution_presets: list[list[int]] = Field(
        default_factory = list, description = "(width, height) presets the UI offers, default first"
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
    defaults: Optional[VideoGenerationDefaults] = Field(
        None, description = "Per-family generation defaults + shape constraints; null when unloaded"
    )
    # Additive: per-Advanced-control provenance {control: {value, source, reason}}. Same shape
    # as the diffusion status; null when nothing is loaded. The frontend renders an "Auto: X"
    # badge next to each control whose source == "auto".
    resolved: Optional[Dict[str, DiffusionResolvedControl]] = Field(
        None,
        description = "Per-control resolved value + provenance (source auto|explicit + reason), "
        "keyed by Advanced control name; null when unloaded or unavailable.",
    )
