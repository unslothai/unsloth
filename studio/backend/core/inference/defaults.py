# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default model lists for inference, split by platform."""

import utils.hardware.hardware as hw

DEFAULT_MODELS_GGUF = [
    "unsloth/gemma-4-E2B-it-GGUF",
    "unsloth/gemma-4-E4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/Qwen3.5-4B-GGUF",
    "unsloth/Qwen3.5-9B-GGUF",
    "unsloth/Qwen3.5-35B-A3B-GGUF",
    "unsloth/Qwen3.5-0.8B-GGUF",
    "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "unsloth/Llama-3.2-3B-Instruct-GGUF",
    "unsloth/Llama-3.1-8B-Instruct-GGUF",
    "unsloth/gemma-3-1b-it-GGUF",
    "unsloth/gemma-3-4b-it-GGUF",
    "unsloth/Qwen3-4B-GGUF",
]

DEFAULT_MODELS_STANDARD = [
    "unsloth/gemma-4-E2B-it-GGUF",
    "unsloth/gemma-4-E4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/Qwen3.5-4B-GGUF",
    "unsloth/Qwen3.5-9B-GGUF",
    "unsloth/Qwen3.5-35B-A3B-GGUF",
    "unsloth/Qwen3.5-0.8B-GGUF",
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/Qwen3-4B-Instruct-2507",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Gemma-3-4B-it",
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
]


def get_default_models() -> list[str]:
    hw.get_device()  # ensure detect_hardware() has run
    if hw.CHAT_ONLY:
        return list(DEFAULT_MODELS_GGUF)
    return list(DEFAULT_MODELS_STANDARD)
