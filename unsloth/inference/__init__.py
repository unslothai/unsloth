# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Flex-attention inference engines.

``UNSLOTH_FAST_INFERENCE=1`` routes ``FastLanguageModel.from_pretrained``
through :func:`load_flex`, which wraps the selected HF model with a
:class:`FlexEngine` that presents the vLLM ``LLM`` surface used by
Unsloth / TRL GRPO (``.generate``, ``.chat``, ``.sleep``, ``.wake_up``,
``.llm_engine``, plus ``save_lora`` / ``load_lora`` via the module
shim).

Three architectures are supported today: Qwen3, Llama-3, Gemma-4-E2B-it.
Anything else raises :class:`NotImplementedError` — there is no silent
fallback; unset the env var or use vLLM instead."""

from .flex_engine import FlexEngine, load_flex
from .vllm_shim import (
    CompletionOutput,
    LoRARequest,
    RequestOutput,
    load_lora,
    save_lora,
)

__all__ = [
    "FlexEngine",
    "load_flex",
    "LoRARequest",
    "RequestOutput",
    "CompletionOutput",
    "save_lora",
    "load_lora",
]
