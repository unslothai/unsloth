# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A sentence-transformers model's own sentence_bert_config.json feeds model_args / config_args /
tokenizer_args into from_pretrained. The keys refused here run code from the repo (an unrestricted
torch.load of its pickle, a Hub kernel import) or reach outside it; the caller's kwargs still can set
them. Same rule as unsloth/models/sentence_transformer.py. install() is idempotent."""

from __future__ import annotations

import importlib
import logging
import re

logger = logging.getLogger(__name__)

_REFUSED_ARGS = frozenset(
    {
        "weights_only",
        "allow_all_kernels",
        "use_kernels",
        "kernel_config",
        "offload_folder",
        "code_revision",
    }
)
_PLAIN_ATTN = re.compile(r"[A-Za-z0-9_|]+")
_MARKER = "_unsloth_scrubs_load_args"


def _path_escapes(value) -> bool:
    if not isinstance(value, str):
        return False
    norm = value.replace("\\", "/")
    return (
        norm.startswith("/") or re.match(r"[A-Za-z]:", norm) is not None or ".." in norm.split("/")
    )


def scrub_load_args(config):
    if not isinstance(config, dict):
        return config
    dropped = []
    for section in ("model_args", "config_args", "tokenizer_args"):
        args = config.get(section)
        if not isinstance(args, dict):
            continue
        for key in list(args):
            value = args[key]
            if (
                key in _REFUSED_ARGS
                or (key == "use_safetensors" and value is not True)
                or (
                    key in ("attn_implementation", "_attn_implementation")
                    and not (isinstance(value, str) and _PLAIN_ATTN.fullmatch(value))
                )
            ):
                args.pop(key)
                dropped.append(f"{section}.{key}")
    if _path_escapes(config.get("tokenizer_name_or_path")):
        config.pop("tokenizer_name_or_path")
        dropped.append("tokenizer_name_or_path")
    if dropped:
        logger.warning(
            "ignoring loader arguments set by the model's own sentence-transformers config: %s",
            ", ".join(dropped),
        )
    return config


def install() -> None:
    for module_name, class_name in (
        ("sentence_transformers.models", "Transformer"),
        ("sentence_transformers.sparse_encoder.models", "MLMTransformer"),
    ):
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except Exception:  # noqa: BLE001 - absent module / older sentence-transformers
            continue
        original = vars(cls).get("load_config")
        if not isinstance(original, classmethod) or getattr(original.__func__, _MARKER, False):
            continue
        func = original.__func__

        def load_config(
            klass,
            *args,
            _original = func,
            **kwargs,
        ):
            return scrub_load_args(_original(klass, *args, **kwargs))

        setattr(load_config, _MARKER, True)
        load_config.__wrapped__ = func
        cls.load_config = classmethod(load_config)
