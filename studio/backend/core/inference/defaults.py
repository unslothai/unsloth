# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default model lists for inference, split by platform."""

from typing import Iterable

import utils.hardware.hardware as hw
from core.inference.mlx_bnb import mlx_bnb_base_repo

DEFAULT_MODELS_GGUF = [
    "unsloth/Qwen3.6-27B-MTP-GGUF",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    "unsloth/DeepSeek-V4-Flash-GGUF",
    "unsloth/gemma-4-E2B-it-GGUF",
    "unsloth/gemma-4-E4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.5-4B-MTP-GGUF",
    "unsloth/Qwen3.5-9B-MTP-GGUF",
    "unsloth/Qwen3.5-35B-A3B-MTP-GGUF",
    "unsloth/Qwen3.5-0.8B-MTP-GGUF",
    "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "unsloth/Llama-3.2-3B-Instruct-GGUF",
    "unsloth/Llama-3.1-8B-Instruct-GGUF",
    "unsloth/gemma-3-1b-it-GGUF",
    "unsloth/gemma-3-4b-it-GGUF",
    "unsloth/Qwen3-4B-GGUF",
]

DEFAULT_MODELS_STANDARD = [
    "unsloth/Qwen3.6-27B-MTP-GGUF",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    "unsloth/DeepSeek-V4-Flash-GGUF",
    "unsloth/gemma-4-E2B-it-GGUF",
    "unsloth/gemma-4-E4B-it-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
    "unsloth/Qwen3.5-4B-MTP-GGUF",
    "unsloth/Qwen3.5-9B-MTP-GGUF",
    "unsloth/Qwen3.5-35B-A3B-MTP-GGUF",
    "unsloth/Qwen3.5-0.8B-MTP-GGUF",
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


def suggestions_for_host(models: Iterable[str], device) -> list[str]:
    """*models* named as *device* really loads them; order kept, duplicates dropped.

    On MLX a bnb repo is a download the loader discards for the base, so suggesting one
    costs a wasted fetch. Takes the device rather than reading it so the caller decides
    whether it is cheap to ask (``hw.DEVICE`` off the event loop, ``get_device()`` when
    detection may still be owed).

    Diffusion is the exception, and it is the same one ``mlx_host_bnb_base_repo`` makes:
    those bnb repos run on the diffusers/MPS path, which reads bitsandbytes weights, so
    they are loaded exactly as named. Mapping one would advertise the full-precision repo
    in place of the smaller one the loader actually uses -- this fix inverted.
    """
    if device != hw.DeviceType.MLX:
        return list(models)
    from core.inference.diffusion_families import detect_family

    def _named(model: str) -> str:
        if detect_family(model) is not None:
            return model
        return mlx_bnb_base_repo(model) or model

    return list(dict.fromkeys(_named(model) for model in models))


def get_default_models() -> list[str]:
    device = hw.get_device()  # ensures detect_hardware() has run
    if hw.CHAT_ONLY:
        return list(DEFAULT_MODELS_GGUF)
    return suggestions_for_host(DEFAULT_MODELS_STANDARD, device)
