# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


def collect_generation(chunks: Iterable[str]) -> str:
    """generate_chat_response yields CUMULATIVE text; the last chunk is the full output."""
    final = ""
    for chunk in chunks:
        final = chunk
    if isinstance(final, str) and final.startswith("Error: "):
        raise RuntimeError(final)
    return final


def make_generate(backend, *, max_new_tokens: int, temperature: float) -> Callable[..., str]:
    def generate(
        messages: list, system_prompt: str, image: Any = None, **_: Any,
    ) -> str:
        chunks = backend.generate_chat_response(
            messages=messages, system_prompt=system_prompt,
            image=image,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        return collect_generation(chunks)
    return generate


def ensure_model_loaded(backend, model_identifier: str, *, hf_token: Optional[str] = None) -> None:
    """Load the model into the shared backend if it isn't already active.

    VERIFY-LIVE confirmed against routes/inference.py (~lines 706-710, 979-987):
    - ModelConfig.from_identifier(model_id=..., hf_token=..., gguf_variant=...)
    - backend.load_model(config=..., max_seq_length=..., load_in_4bit=...,
                         hf_token=..., trust_remote_code=..., gpu_ids=...)
    The `dtype` kwarg exists on the real load_model signature (default None);
    the production route omits it (takes the default). We omit it here too.
    generate_chat_response(messages=..., system_prompt=..., max_new_tokens=...,
                           temperature=...) confirmed in core/inference/inference.py:910.
    """
    if getattr(backend, "active_model_name", None) == model_identifier:
        return
    from utils.models import ModelConfig
    config = ModelConfig.from_identifier(model_id=model_identifier, hf_token=hf_token,
                                         gguf_variant=None)
    if getattr(config, "is_gguf", False):
        raise ValueError("GGUF eval not yet supported; choose a transformers/unsloth model.")
    ok = backend.load_model(config, max_seq_length=2048, load_in_4bit=True,
                            hf_token=hf_token, trust_remote_code=False, gpu_ids=None)
    if not ok:
        raise RuntimeError(f"failed to load model {model_identifier!r}")
