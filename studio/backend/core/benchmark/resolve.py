# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)


def build_gguf_kwargs(
    identifier: str,
    base_url: str,
    task: str,
    batch_size: str,
    max_tokens: Optional[int] = None,
    num_fewshot: Optional[int] = None,
) -> dict:
    """lm_eval kwargs for ``gguf_client.StudioGGUFLM``, which tokenizes and scores through
    this server's llama-server, so no HF tokenizer is needed."""
    kwargs = {
        # Registered by core.benchmark.gguf_client, imported in the worker.
        "model": "unsloth-studio-gguf",
        "model_args": {
            # Server root, no /v1.
            "base_url": base_url,
            "model": identifier,
            "max_gen_toks": max_tokens if max_tokens is not None else 32768,
            # None: one request per llama-server slot.
            "parallel": int(batch_size) if batch_size != "auto" else None,
            "timeout": 1200,
        },
        "tasks": [task],
        "batch_size": 1,
        # The history views read samples from the DB; the route writes results itself.
        "log_samples": True,
    }
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot
    return kwargs


def resolve_model_details(
    checkpoint_path: str,
    task: str = "mmlu",
    server_url: Optional[str] = None,
    batch_size: str = "auto",
    num_fewshot: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    """lm_eval kwargs for a run against the model llama-server already holds."""
    identifier = checkpoint_path.strip()
    # 8888 only when the bound address is unknown (called outside a running server).
    base_url = (server_url or os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")).rstrip(
        "/"
    )

    return build_gguf_kwargs(
        identifier,
        base_url,
        task,
        batch_size,
        max_tokens = max_tokens,
        num_fewshot = num_fewshot,
    )
