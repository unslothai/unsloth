# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Inference API
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Dict, Literal, Optional, List, Union

from pydantic import BaseModel, Discriminator, Field, Tag


class LoadRequest(BaseModel):
    """Request to load a model for inference"""

    model_path: str = Field(..., description = "Model identifier or local path")
    hf_token: Optional[str] = Field(
        None, description = "HuggingFace token for gated models"
    )
    max_seq_length: int = Field(
        4096, ge = 128, le = 32768, description = "Maximum sequence length"
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
    cache_type_kv: Optional[str] = Field(
        None,
        description = "KV cache data type for both K and V (e.g. 'f16', 'bf16', 'q8_0', 'q4_1', 'q5_1')",
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
    context_length: Optional[int] = Field(
        None, description = "Model's native context length (from GGUF metadata)"
    )
    supports_reasoning: bool = Field(
        False,
        description = "Whether model supports thinking/reasoning mode (enable_thinking)",
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


class UnloadResponse(BaseModel):
    """Response after unloading a model"""

    status: str = Field(..., description = "Unload status")
    model: str = Field(..., description = "Model identifier that was unloaded")


class InferenceStatusResponse(BaseModel):
    """Current inference backend status"""

    active_model: Optional[str] = Field(
        None, description = "Currently active model identifier"
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
    supports_reasoning: bool = Field(
        False, description = "Whether the active model supports reasoning/thinking mode"
    )
    supports_tools: bool = Field(
        False, description = "Whether the active model supports tool calling"
    )
    context_length: Optional[int] = Field(
        None, description = "Context length of the active model"
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


def _content_part_discriminator(v):
    if isinstance(v, dict):
        return v.get("type")
    return getattr(v, "type", None)


ContentPart = Annotated[
    Union[
        Annotated[TextContentPart, Tag("text")],
        Annotated[ImageContentPart, Tag("image_url")],
    ],
    Discriminator(_content_part_discriminator),
]
"""Union type for multimodal content parts, discriminated by the 'type' field."""


# ── Messages ─────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """
    A single message in the conversation.

    ``content`` may be a plain string (text-only) or a list of
    content parts for multimodal messages (OpenAI vision format).
    """

    role: Literal["system", "user", "assistant"] = Field(
        ..., description = "Message role"
    )
    content: Union[str, list[ContentPart]] = Field(
        ..., description = "Message content (string or multimodal parts)"
    )


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Extensions (non-OpenAI fields) are marked with 'x-unsloth'.
    """

    model: str = Field(
        "default",
        description = "Model identifier (informational; the active model is used)",
    )
    messages: list[ChatMessage] = Field(..., description = "Conversation messages")
    stream: bool = Field(True, description = "Whether to stream the response via SSE")
    temperature: float = Field(0.6, ge = 0.0, le = 2.0)
    top_p: float = Field(0.95, ge = 0.0, le = 1.0)
    max_tokens: Optional[int] = Field(
        None, ge = 1, description = "Maximum tokens to generate (None = until EOS)"
    )
    presence_penalty: float = Field(0.0, ge = 0.0, le = 2.0, description = "Presence penalty")

    # ── Unsloth extensions (ignored by standard OpenAI clients) ──
    top_k: int = Field(20, ge = -1, le = 100, description = "[x-unsloth] Top-k sampling")
    min_p: float = Field(
        0.01, ge = 0.0, le = 1.0, description = "[x-unsloth] Min-p sampling threshold"
    )
    repetition_penalty: float = Field(
        1.1, ge = 1.0, le = 2.0, description = "[x-unsloth] Repetition penalty"
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
    enable_tools: Optional[bool] = Field(
        None,
        description = "[x-unsloth] Enable tool calling for supported models",
    )
    enabled_tools: Optional[list[str]] = Field(
        None,
        description = "[x-unsloth] List of enabled tool names (e.g. ['web_search', 'python', 'terminal']). If None, all tools are enabled.",
    )
    auto_heal_tool_calls: Optional[bool] = Field(
        True,
        description = "[x-unsloth] Auto-detect and fix malformed tool calls from model output.",
    )
    max_tool_calls_per_message: Optional[int] = Field(
        10,
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
