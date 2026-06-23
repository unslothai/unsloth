# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from utils.paths.path_utils import (
    get_cache_path,
    is_local_path,
    resolve_cached_repo_id_case,
)

from .schemas import ValidateChatTemplateResponse

logger = logging.getLogger(__name__)

_TOKENIZER_CONFIG_PATHS = ("tokenizer_config.json", "LLM/tokenizer_config.json")


def validate_chat_template(template: str) -> ValidateChatTemplateResponse:
    text = (template or "").strip()
    if not text:
        return ValidateChatTemplateResponse(valid = True, error = None)
    try:
        env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
        env.parse(text)
        return ValidateChatTemplateResponse(valid = True, error = None)
    except TemplateError as exc:
        message = getattr(exc, "message", None) or str(exc)
        lineno = getattr(exc, "lineno", None)
        if lineno:
            message = f"Line {lineno}: {message}"
        return ValidateChatTemplateResponse(valid = False, error = message)
    except Exception as exc:
        return ValidateChatTemplateResponse(valid = False, error = str(exc))


def _chat_template_from_tokenizer_config(config: dict) -> Optional[str]:
    raw = config.get("chat_template")
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, list):
        fallback: Optional[str] = None
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            template = entry.get("template")
            if not isinstance(template, str):
                continue
            if entry.get("name") == "default":
                return template
            if fallback is None:
                fallback = template
        return fallback
    return None


def _chat_template_from_dir(dir_path: Path) -> Optional[str]:
    for rel in _TOKENIZER_CONFIG_PATHS:
        config_file = dir_path / rel
        if not config_file.exists():
            continue
        try:
            config = json.loads(config_file.read_text(encoding = "utf-8"))
        except Exception:
            continue
        template = _chat_template_from_tokenizer_config(config)
        if template:
            return template
    return None


def read_default_chat_template(
    model_name: str, hf_token: Optional[str] = None
) -> Optional[str]:
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    name = model_name.strip()

    if is_local_path(name):
        try:
            return _chat_template_from_dir(Path(name))
        except Exception as exc:
            logger.debug("Could not read local chat template for %s: %s", name, exc)
            return None

    resolved = resolve_cached_repo_id_case(name)

    try:
        repo_dir = get_cache_path(resolved)
        if repo_dir is not None and repo_dir.exists():
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    template = _chat_template_from_dir(snapshot)
                    if template:
                        return template
    except Exception as exc:
        logger.debug("Could not read cached chat template for %s: %s", resolved, exc)

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            resolved, "tokenizer_config.json", token = hf_token
        )
        config = json.loads(Path(downloaded).read_text(encoding = "utf-8"))
        return _chat_template_from_tokenizer_config(config)
    except Exception as exc:
        logger.debug("Could not fetch chat template for %s: %s", resolved, exc)
        return None
