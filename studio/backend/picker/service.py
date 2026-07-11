# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from hub.services.models.folder_browser import (
    _build_browse_allowlist,
    _is_path_inside_allowlist,
)
from hub.utils.gguf import iter_hf_cache_snapshots
from utils.models.gguf_metadata import read_gguf_chat_template
from utils.models.model_config import (
    _extract_quant_label,
    _is_big_endian_gguf_path,
    _is_mmproj,
    _is_mtp_drafter,
)
from utils.paths.path_utils import (
    is_local_path,
    normalize_path,
    resolve_cached_repo_id_case,
)

from .schemas import ValidateChatTemplateResponse

logger = logging.getLogger(__name__)

_VALID_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _is_valid_repo_id(repo_id: str) -> bool:
    return bool(_VALID_REPO_ID.fullmatch(repo_id))


_TOKENIZER_CONFIG_PATHS = ("tokenizer_config.json", "LLM/tokenizer_config.json")
_JINJA_TEMPLATE_PATHS = ("chat_template.jinja", "LLM/chat_template.jinja")
_PROCESSOR_TEMPLATE_PATHS = ("chat_template.json", "LLM/chat_template.json")


def validate_chat_template(template: str) -> ValidateChatTemplateResponse:
    text = (template or "").strip()
    if not text:
        return ValidateChatTemplateResponse(valid = True, error = None)
    try:
        env = ImmutableSandboxedEnvironment(
            trim_blocks = True,
            lstrip_blocks = True,
            extensions = ["jinja2.ext.loopcontrols"],
        )
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


def _chat_template_from_jinja_file(dir_path: Path) -> Optional[str]:
    for rel in _JINJA_TEMPLATE_PATHS:
        template_file = dir_path / rel
        if not template_file.exists():
            continue
        try:
            template = template_file.read_text(encoding = "utf-8")
        except Exception:
            continue
        if template.strip():
            return template
    return None


def _chat_template_from_processor_json(dir_path: Path) -> Optional[str]:
    for rel in _PROCESSOR_TEMPLATE_PATHS:
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


def _chat_template_from_tokenizer_dir(dir_path: Path) -> Optional[str]:
    jinja = _chat_template_from_jinja_file(dir_path)
    if jinja:
        return jinja
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
    return _chat_template_from_processor_json(dir_path)


_GGUF_SCAN_MAX_DEPTH = 2


def _iter_ggufs(dir_path: Path) -> list[Path]:
    if dir_path == dir_path.parent:
        return []
    root = str(dir_path)
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks = False):
        rel = os.path.relpath(current, root)
        depth = 0 if rel == os.curdir else rel.count(os.sep) + 1
        if depth >= _GGUF_SCAN_MAX_DEPTH:
            dirs[:] = []
        for name in files:
            if not name.lower().endswith(".gguf") or _is_mmproj(name):
                continue
            path = Path(current) / name
            try:
                rel = path.relative_to(dir_path).as_posix()
            except ValueError:
                rel = name
            quant = _extract_quant_label(rel)
            if _is_mtp_drafter(rel) or _is_big_endian_gguf_path(rel, quant):
                continue
            found.append(path)
    return found


def _variant_matches(relative_path: str, needle: str) -> bool:
    quant = _extract_quant_label(relative_path).lower()
    if quant == needle:
        return True
    prefix = f"{needle}-"
    if not quant.startswith(prefix):
        return False
    suffix = quant[len(prefix) :]
    if not suffix.endswith("bpw"):
        return False
    value = suffix[:-3]
    return bool(value) and value.replace(".", "", 1).isdigit()


def _find_gguf_in_dir(dir_path: Path, gguf_variant: Optional[str]) -> Optional[Path]:
    try:
        ggufs = sorted(_iter_ggufs(dir_path))
    except OSError:
        return None
    if not ggufs:
        return None
    needle = (gguf_variant or "").strip().lower()
    if needle:
        for path in ggufs:
            try:
                relative = path.relative_to(dir_path).as_posix()
            except ValueError:
                relative = path.name
            if _variant_matches(relative, needle):
                return path
        return None
    try:
        return max(ggufs, key = lambda path: path.stat().st_size)
    except OSError:
        return ggufs[0]


def _chat_template_from_dir(dir_path: Path, gguf_variant: Optional[str] = None) -> Optional[str]:
    def from_gguf() -> Optional[str]:
        gguf = _find_gguf_in_dir(dir_path, gguf_variant)
        return read_gguf_chat_template(str(gguf)) if gguf is not None else None

    if gguf_variant:
        return from_gguf() or _chat_template_from_tokenizer_dir(dir_path)
    return _chat_template_from_tokenizer_dir(dir_path) or from_gguf()


def read_default_chat_template(
    model_name: str,
    hf_token: Optional[str] = None,
    gguf_variant: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    name = model_name.strip()

    if is_local_path(name):
        try:
            target = Path(normalize_path(name)).expanduser()
            if not _is_path_inside_allowlist(target, _build_browse_allowlist()):
                logger.debug("Refused chat template read outside allowed folders: %s", name)
                return None
            if name.lower().endswith(".gguf"):
                return read_gguf_chat_template(str(target))
            return _chat_template_from_dir(target, gguf_variant)
        except Exception as exc:
            logger.debug("Could not read local chat template for %s: %s", name, exc)
            return None

    if not _is_valid_repo_id(name):
        return None

    resolved = resolve_cached_repo_id_case(name)

    try:
        for snapshot in iter_hf_cache_snapshots(resolved):
            template = _chat_template_from_dir(snapshot, gguf_variant)
            if template:
                return template
    except Exception as exc:
        logger.debug("Could not read cached chat template for %s: %s", resolved, exc)

    try:
        from huggingface_hub import hf_hub_download

        def _download_text(rel: str) -> Optional[str]:
            try:
                path = hf_hub_download(resolved, rel, token = hf_token)
                return Path(path).read_text(encoding = "utf-8")
            except Exception:
                return None

        for rel in _JINJA_TEMPLATE_PATHS:
            template = _download_text(rel)
            if template and template.strip():
                return template

        for rel in (*_TOKENIZER_CONFIG_PATHS, *_PROCESSOR_TEMPLATE_PATHS):
            raw = _download_text(rel)
            if not raw:
                continue
            try:
                config = json.loads(raw)
            except Exception:
                continue
            template = _chat_template_from_tokenizer_config(config)
            if template:
                return template

        return None
    except Exception as exc:
        logger.debug("Could not fetch chat template for %s: %s", resolved, exc)
        return None
