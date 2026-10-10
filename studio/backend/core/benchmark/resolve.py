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
    """lm_eval kwargs for the ``gguf`` backend pointed at this Studio server
    (see ``gguf_client.StudioGGUFLM``).

    Unlike ``local-completions`` this needs no HF tokenizer: the backend queries
    the server's own ``/tokenize`` endpoint and scores continuations with
    ``logit_bias`` teacher forcing, so it matches the server exactly.
    """
    kwargs = {
        # Registered by core.benchmark.gguf_client, imported in the worker.
        "model": "unsloth-studio-gguf",
        "model_args": {
            # Server root (no /v1): the backend derives /v1/completions,
            # /v1/tokenize and /props from it.
            "base_url": base_url,
            "model": identifier,
            "max_gen_toks": max_tokens if max_tokens is not None else 32768,
            # None lets the backend auto-detect the server's slot count from
            # /props (total_slots) and request concurrently up to it.
            "parallel": int(batch_size) if batch_size != "auto" else None,
            "timeout": 1200,
        },
        "tasks": [task],
        "batch_size": 1,
        # Samples are always logged: the Evals history/score views read them
        # back from the DB. Note ``simple_evaluate`` has no ``output_path``
        # parameter (that is a CLI-only arg backed by ``EvaluationTracker``),
        # so results are persisted by the route after the run instead.
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
) -> tuple[list[str], dict]:
    """Resolve model details and return (empty_log_lines, lm_eval_kwargs).

    Benchmarking never loads a model in-process: it always talks to the local
    inference server (llama.cpp behind the hood) over HTTP via the ``gguf``
    backend. The server-side model load, GGUF variant selection and
    tokenization are all the server's job, so there is nothing to resolve
    here beyond the request parameters.
    """
    identifier = checkpoint_path.strip()
    # The route passes the address Studio actually bound; 8888 is only the
    # last resort when it is unknown (e.g. called outside a running server).
    base_url = (server_url or os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")).rstrip(
        "/"
    )

    lm_eval_kwargs = build_gguf_kwargs(
        identifier,
        base_url,
        task,
        batch_size,
        max_tokens = max_tokens,
        num_fewshot = num_fewshot,
    )

    return [], lm_eval_kwargs
