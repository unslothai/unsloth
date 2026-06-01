# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Inference API
"""

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
    hf_token: Optional[str] = Field(
        None, description = "HuggingFace token for gated models"
    )
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
    chat_template_override: Optional[str] = Field(
        None,
        description = "Custom Jinja2 chat template to use instead of the model's default",
    )

    @field_validator("chat_template_override")
    @classmethod
    def normalize_blank_chat_template_override(
        cls, value: Optional[str]
    ) -> Optional[str]:
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
            "still accepted. Ignored for non-GGUF and vision models."
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
    llama_extra_args: Optional[List[str]] = Field(
        None,
        description = (
            "Extra arguments forwarded verbatim to llama-server for GGUF models. "
            "One token per list entry, e.g. ['--top-k', '20', '--seed', '42']. "
            "Studio-managed flags (model identity, port, context length, GPU placement, "
            "auth, --flash-attn, --no-context-shift, --jinja) are rejected. Ignored for "
            "non-GGUF models."
        ),
    )


class UnloadRequest(BaseModel):
    """Request to unload a model"""

    model_path: str = Field(..., description = "Model identifier to unload")


class ValidateModelRequest(BaseModel):
    """
    Lightweight validation request to check whether a model identifier
    *can be resolved* into a ModelConfig.

    This does NOT actually load weights into GPU memory.
    """

    model_path: str = Field(..., description = "Model identifier or local path")
    native_path_lease: Optional[str] = Field(
        None, description = "Frontend-visible signed native path grant"
    )
    hf_token: Optional[str] = Field(
        None, description = "HuggingFace token for gated models"
    )
    gguf_variant: Optional[str] = Field(
        None, description = "GGUF quantization variant (e.g. 'Q4_K_M')"
    )


class ValidateModelResponse(BaseModel):
    """
    Result of model validation.

    valid == True means ModelConfig.from_identifier() succeeded and basic
    introspection (GGUF / LoRA / vision flags) is available.
    """

    valid: bool = Field(..., description = "Whether the model identifier looks valid")
    message: str = Field(..., description = "Human-readable validation message")
    identifier: Optional[str] = Field(None, description = "Resolved model identifier")
    display_name: Optional[str] = Field(
        None, description = "Display name derived from identifier"
    )
    is_gguf: bool = Field(False, description = "Whether this is a GGUF model (llama.cpp)")
    is_lora: bool = Field(False, description = "Whether this is a LoRA adapter")
    is_vision: bool = Field(False, description = "Whether this is a vision-capable model")
    requires_trust_remote_code: bool = Field(
        False,
        description = "Whether the model defaults require trust_remote_code to be enabled for loading.",
    )


class GenerateRequest(BaseModel):
    """Request for text generation (legacy /generate/stream endpoint)"""

    messages: List[dict] = Field(..., description = "Chat messages in OpenAI format")
    system_prompt: str = Field("", description = "System prompt")
    temperature: float = Field(0.6, ge = 0.0, le = 2.0, description = "Sampling temperature")
    top_p: float = Field(0.95, ge = 0.0, le = 1.0, description = "Top-p sampling")
    top_k: int = Field(20, ge = -1, le = 100, description = "Top-k sampling")
    max_new_tokens: int = Field(
        2048, ge = 1, le = 4096, description = "Maximum tokens to generate"
    )
    repetition_penalty: float = Field(
        1.0, ge = 1.0, le = 2.0, description = "Repetition penalty"
    )
    presence_penalty: float = Field(0.0, ge = 0.0, le = 2.0, description = "Presence penalty")
    image_base64: Optional[str] = Field(
        None, description = "Base64 encoded image for vision models"
    )


class LoadResponse(BaseModel):
    """Response after loading a model"""

    status: str = Field(..., description = "Load status")
    model: str = Field(..., description = "Model identifier")
    display_name: str = Field(..., description = "Display name of the model")
    is_vision: bool = Field(False, description = "Whether model is a vision model")
    is_lora: bool = Field(False, description = "Whether model is a LoRA adapter")
    is_gguf: bool = Field(
        False, description = "Whether model is a GGUF model (llama.cpp)"
    )
    is_audio: bool = Field(False, description = "Whether model is a TTS audio model")
    audio_type: Optional[str] = Field(
        None, description = "Audio codec type: snac, csm, bicodec, dac"
    )
    has_audio_input: bool = Field(
        False, description = "Whether model accepts audio input (ASR)"
    )
    inference: dict = Field(
        ..., description = "Inference parameters (temperature, top_p, top_k, min_p)"
    )
    requires_trust_remote_code: bool = Field(
        False,
        description = "Whether the model defaults require trust_remote_code to be enabled for loading.",
    )
    context_length: Optional[int] = Field(
        None, description = "Model's native context length (from GGUF metadata)"
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
    reasoning_style: Literal["enable_thinking", "reasoning_effort"] = Field(
        "enable_thinking",
        description = "Reasoning control style: 'enable_thinking' (boolean) or 'reasoning_effort' (low|medium|high)",
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


class UnloadResponse(BaseModel):
    """Response after unloading a model"""

    status: str = Field(..., description = "Unload status")
    model: str = Field(..., description = "Model identifier that was unloaded")


class LoadProgressResponse(BaseModel):
    """Progress of the active GGUF load, sampled on demand.

    Used by the UI to show a real progress bar during the
    post-download warmup window (mmap + CUDA upload), rather than a
    generic "Starting model..." spinner that freezes for minutes on
    large MoE models.
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
            "Bytes of the model already resident in the llama-server "
            "process (VmRSS on Linux)."
        ),
    )
    bytes_total: int = Field(
        0,
        description = "Total bytes across all GGUF shards for the active model.",
    )
    fraction: float = Field(
        0.0, description = "bytes_loaded / bytes_total, clamped to 0..1."
    )


class InferenceStatusResponse(BaseModel):
    """Current inference backend status"""

    active_model: Optional[str] = Field(
        None, description = "Currently active model display identifier"
    )
    model_identifier: Optional[str] = Field(
        None,
        description = "Loadable identifier for the active model.",
    )
    is_vision: bool = Field(
        False, description = "Whether the active model is a vision model"
    )
    is_gguf: bool = Field(
        False, description = "Whether the active model is a GGUF model (llama.cpp)"
    )
    gguf_variant: Optional[str] = Field(
        None, description = "GGUF quantization variant (e.g. Q4_K_M)"
    )
    is_audio: bool = Field(
        False, description = "Whether the active model is a TTS audio model"
    )
    audio_type: Optional[str] = Field(
        None, description = "Audio codec type: snac, csm, bicodec, dac"
    )
    has_audio_input: bool = Field(
        False, description = "Whether model accepts audio input (ASR)"
    )
    loading: List[str] = Field(
        default_factory = list, description = "Models currently being loaded"
    )
    loaded: List[str] = Field(
        default_factory = list, description = "Models currently loaded"
    )
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
    reasoning_style: Literal["enable_thinking", "reasoning_effort"] = Field(
        "enable_thinking",
        description = "Reasoning control style: 'enable_thinking' (boolean) or 'reasoning_effort' (low|medium|high)",
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
    context_length: Optional[int] = Field(
        None, description = "Context length of the active model"
    )
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
    llama_cpp_supports_mtp: bool = Field(
        True,
        description = (
            "Whether llama.cpp supports MTP (--spec-type mtp/draft-mtp). "
            "False -> recommend `unsloth studio update`."
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
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ImageContentPart(BaseModel):
    """Image content part in a multimodal message."""

    type: Literal["image_url"]
    image_url: ImageUrl


class InputDocumentContentPart(BaseModel):
    """Document (PDF / file) content part in a multimodal message.

    Studio-normalised shape. The frontend sends either
    ``{type:"input_document", file_data:"data:application/pdf;base64,..."}``
    or ``{type:"input_document", file_url:"https://..."}``, plus optional
    ``filename`` and ``media_type``. ``external_provider`` translates this
    onto Anthropic's ``document`` block or OpenAI Responses' ``input_file``
    block for vision-capable providers; non-vision providers drop the
    part entirely (handled in ``_build_external_messages``).
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

    Reasoning models can require the previous ``reasoning`` output item
    to be replayed immediately before an ``image_generation_call`` id
    when manually managing Responses context. This part is OpenAI-only;
    routes strip it for every other provider before proxying.
    """

    type: Literal["reasoning"]
    id: str = Field(..., description = "OpenAI reasoning output item id.")
    summary: list[dict[str, Any]] = Field(default_factory = list)
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ImageGenerationCallContentPart(BaseModel):
    """OpenAI Responses image_generation call reference.

    OpenAI accepts prior ``image_generation_call`` items in the next
    Responses ``input`` array so follow-up prompts can edit or refine a
    generated image without resending the base64 payload. The frontend
    forwards this as a synthetic assistant content part when building
    the next OpenAI Responses request; ``external_provider`` translates
    it back to the provider-specific top-level input item.
    """

    type: Literal["image_generation_call"]
    id: str = Field(..., description = "OpenAI image_generation_call output item id.")
    response_id: Optional[str] = Field(
        None,
        description = "OpenAI Responses response id to use as previous_response_id for follow-up edits.",
    )


class CompactionContentPart(BaseModel):
    """Anthropic server-side compaction state, attached to an assistant
    message for round-tripping on the next turn.

    When Anthropic runs compaction during a request, the response
    carries a ``{"type": "compaction", "content": "<summary>"}`` block
    on the assistant message. The chat-adapter persists it onto the
    stored message; the next turn's outbound request must forward it
    back so Anthropic recognises the existing compaction state and
    doesn't re-summarise the conversation from scratch. See
    ``external_provider._stream_anthropic`` for the wire-side handling
    and https://platform.claude.com/docs/en/build-with-claude/compaction
    for the upstream contract.
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

    ``content`` is a string or a list of multimodal content parts. Assistant
    messages with only ``tool_calls`` populated may set ``content=None``.
    Missing ``tool_call_id`` on ``role="tool"`` is resolved at the
    ``ChatCompletionRequest`` layer by walking back to the preceding assistant.
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(
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
            if not self.content:
                raise ValueError('role="tool" messages require non-empty "content".')
        elif self.role == "assistant":
            # Post-Stop sentinel: collapse content="" / [] to None.
            if (self.content == "" or self.content == []) and not self.tool_calls:
                self.content = None
        else:  # "user" | "system"
            if self.content is None or self.content == []:
                raise ValueError(f'role="{self.role}" messages require "content".')
        return self


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Extensions (non-OpenAI fields) are marked with 'x-unsloth'.
    """

    # Accept unknown fields defensively so future OpenAI fields (seed,
    # response_format, logprobs, frequency_penalty, etc.) don't get
    # silently dropped by Pydantic before route code runs. Mirrors
    # AnthropicMessagesRequest and ResponsesRequest.
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

    # ── Unsloth extensions (ignored by standard OpenAI clients) ──
    top_k: int = Field(20, ge = -1, le = 100, description = "[x-unsloth] Top-k sampling")
    min_p: float = Field(
        0.01, ge = 0.0, le = 1.0, description = "[x-unsloth] Min-p sampling threshold"
    )
    repetition_penalty: float = Field(
        1.0, ge = 1.0, le = 2.0, description = "[x-unsloth] Repetition penalty"
    )
    image_base64: Optional[str] = Field(
        None, description = "[x-unsloth] Base64-encoded image for vision models"
    )
    audio_base64: Optional[str] = Field(
        None, description = "[x-unsloth] Base64-encoded WAV for audio-input models (ASR)"
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
    auto_heal_tool_calls: Optional[bool] = Field(
        True,
        description = "[x-unsloth] Auto-detect and fix malformed tool calls from model output.",
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
            "tool: {kb_id?, thread_id?, default_top_k?, min_score?, mode?}. The "
            "model never sees this; the server resolves which documents to search."
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
        """Preserve the pre-PR coercion: the field used to be Optional[bool],
        so callers historically sent JSON strings `"true"` / `"false"` and
        Pydantic v1 coerced them. Widening to Optional[Union[bool, str]] for
        Gemini cache resource names lets `"false"` slip through as a truthy
        string. Coerce the canonical bool literals back so explicit opt-outs
        stay opt-out."""
        if isinstance(value, str):
            lowered = value.strip().lower()
            # Match Pydantic v1's BooleanField coercion table (yes/y/on/t/1
            # and no/n/off/f/0) so opt-outs that used to parse still parse.
            # Anything else is preserved as a string for Gemini's
            # cachedContent resource path.
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
        unconsumed tool_call; synth random id only if no candidate exists.
        Crossing a user turn breaks the lookup.
        """
        # Pre-mark explicit ids first so a sibling missing-id result does not
        # steal one already claimed by name.
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
                    function_name = (
                        function.get("name") if isinstance(function, dict) else None
                    )
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


# ── OpenAI shell-tool container management ─────────────────────


class OpenAIContainerRequest(BaseModel):
    """
    Shared body for the three OpenAI container endpoints (list / create
    / delete). Carries the encrypted API key + base URL so the route
    handler can decrypt it and proxy to the user's OpenAI account.
    Same pattern as the inference proxy endpoints — keeps the key off
    persistent storage on the backend.
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


class ChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


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
    content: str


class CompletionChoice(BaseModel):
    """A single choice in a non-streaming response."""

    index: int = 0
    message: CompletionMessage
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionUsage(BaseModel):
    """Token usage statistics (approximate)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    """Non-streaming chat completion response."""

    id: str = Field(default_factory = lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory = lambda: int(time.time()))
    model: str = "default"
    choices: list[CompletionChoice]
    usage: CompletionUsage = Field(default_factory = CompletionUsage)


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
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ResponsesOutputTextPart(BaseModel):
    """Assistant ``output_text`` content part replayed on subsequent turns.

    When a client (OpenAI Codex CLI, OpenAI Python SDK agents) loops on a
    stateless Responses endpoint, prior assistant messages are round-tripped
    as ``{"role":"assistant","content":[{"type":"output_text","text":...,
    "annotations":[],"logprobs":[]}]}``. We preserve the text and ignore
    the annotations/logprobs metadata when flattening into Chat Completions.
    """

    type: Literal["output_text"]
    text: str
    annotations: Optional[list] = None
    logprobs: Optional[list] = None

    model_config = {"extra": "allow"}


class ResponsesUnknownContentPart(BaseModel):
    """Catch-all for content-part types we don't model explicitly.

    Keeps validation green when a client sends newer part types (e.g.
    ``input_audio``, ``input_file``) we haven't mapped; these are silently
    skipped during normalisation rather than rejected with a 422.
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

    # Codex (gpt-5.3-codex+) attaches a `phase` field ("commentary" |
    # "final_answer") to assistant messages and requires clients to preserve
    # it on subsequent turns. We accept and round-trip it; llama-server does
    # not care about it.
    model_config = {"extra": "allow"}


class ResponsesFunctionCallInputItem(BaseModel):
    """A prior assistant function_call being replayed in a multi-turn Responses input.

    The Responses API represents tool calls as top-level input items (not
    nested inside assistant messages), correlated across turns by ``call_id``.
    """

    type: Literal["function_call"]
    id: Optional[str] = Field(
        None, description = "Item id assigned by the server (e.g. fc_...)"
    )
    call_id: str = Field(
        ...,
        description = "Correlation id matching a function_call_output on the next turn.",
    )
    name: str
    arguments: str = Field(
        ..., description = "JSON string of the arguments the model produced."
    )
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class ResponsesFunctionCallOutputInputItem(BaseModel):
    """A tool result supplied by the client for a prior function_call.

    Replaces Chat Completions' ``role="tool"`` message. Correlated to the
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
    """Catch-all for Responses input item types we don't model explicitly.

    Covers ``reasoning`` items (replayed from prior o-series / gpt-5 turns)
    and any future item types the client may send. These items are dropped
    during normalisation — llama-server-backed GGUFs cannot consume them —
    but keeping them in the request-model union stops unrelated turns from
    failing validation with a 422.
    """

    type: str

    model_config = {"extra": "allow"}


def _responses_input_item_discriminator(v: Any) -> str:
    """Route a Responses input item to the correct tagged variant.

    Pydantic's default smart-union matching fails when one variant in the
    union is tagged with a strict ``Literal`` (``function_call`` /
    ``function_call_output``) and the incoming dict uses a different
    ``type`` — the other variants' validation errors are hidden and the
    outer ``Union[str, list[...]]`` reports a misleading "Input should be a
    valid string" error. An explicit discriminator makes the routing
    deterministic and lets us fall through to the catch-all.
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
    """Flat function-tool definition used by the Responses API request.

    Unlike Chat Completions (which nests ``{"name": ..., "parameters": ...}``
    inside a ``"function"`` key), the Responses API uses a flat shape with
    ``type``, ``name``, ``description``, ``parameters``, and ``strict`` at the
    top level of each tool entry.
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
    instructions: Optional[str] = Field(
        None, description = "System / developer instructions"
    )
    temperature: Optional[float] = Field(None, ge = 0.0, le = 2.0)
    top_p: Optional[float] = Field(None, ge = 0.0, le = 1.0)
    max_output_tokens: Optional[int] = Field(None, ge = 1)
    stream: bool = Field(False, description = "Whether to stream the response via SSE")

    # OpenAI function-calling fields — forwarded to llama-server via the
    # Chat Completions pass-through (see routes/inference.py). Typed as a
    # plain list so built-in tool shapes (``web_search``, ``file_search``,
    # ``mcp``, ...) round-trip without validation errors — the translator
    # picks out only ``type=="function"`` entries for forwarding.
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


class ResponsesOutputFunctionCall(BaseModel):
    """A function-call output item in the Responses API response.

    Unlike Chat Completions (which nests tool calls inside the assistant
    message), the Responses API emits each tool call as its own top-level
    ``output`` item so clients can correlate results via ``call_id`` on the
    next turn.
    """

    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory = lambda: f"fc_{uuid.uuid4().hex[:12]}")
    call_id: str
    name: str
    arguments: str = Field(
        ..., description = "JSON string of the arguments the model produced."
    )
    status: Literal["completed", "in_progress", "incomplete"] = "completed"


ResponsesOutputItem = Union[ResponsesOutputMessage, ResponsesOutputFunctionCall]


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


AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
]


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, list[AnthropicContentBlock]]


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
    # [x-unsloth] extensions — mirror the OpenAI endpoint convenience fields
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
    model_config = {"extra": "allow"}


# ── Response models ────────────────────────────────────────────


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponseTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicResponseToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


AnthropicResponseBlock = Union[
    AnthropicResponseTextBlock, AnthropicResponseToolUseBlock
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
