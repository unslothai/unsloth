# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

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


def _leaf_inside_allowlist(path: Path, allow_roots: Optional[list[Path]]) -> bool:
    # Block symlinked children from escaping the validated directory
    # (realpath-checked). None = trusted caller (HF cache / remote download).
    return allow_roots is None or _is_path_inside_allowlist(path, allow_roots)


def validate_chat_template(template: str) -> ValidateChatTemplateResponse:
    text = (template or "").strip()
    if not text:
        return ValidateChatTemplateResponse(valid = True, error = None)
    # Import Jinja lazily: it is optional at runtime (e.g. GGUF-only installs),
    # so a missing dependency must not crash API startup through this module.
    try:
        from jinja2 import TemplateError
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        return ValidateChatTemplateResponse(valid = True, error = None)

    class _GenerationTag(Extension):
        # Accept Transformers' {% generation %}...{% endgeneration %} assistant
        # mask tag so a pasted HF chat template validates (we only parse it).
        tags = {"generation"}

        def parse(self, parser):
            next(parser.stream)
            return parser.parse_statements(["name:endgeneration"], drop_needle = True)

    try:
        env = ImmutableSandboxedEnvironment(
            trim_blocks = True,
            lstrip_blocks = True,
            extensions = ["jinja2.ext.loopcontrols", _GenerationTag],
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
    if not isinstance(config, dict):
        return None
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


def _chat_template_from_jinja_file(
    dir_path: Path, allow_roots: Optional[list[Path]] = None
) -> Optional[str]:
    for rel in _JINJA_TEMPLATE_PATHS:
        template_file = dir_path / rel
        if not template_file.exists() or not _leaf_inside_allowlist(template_file, allow_roots):
            continue
        try:
            template = template_file.read_text(encoding = "utf-8")
        except Exception:
            continue
        if template.strip():
            return template
    return None


def _chat_template_from_processor_payload(payload: object) -> Optional[str]:
    # processor chat_template.json may be the template string itself or a
    # {name: template} map, not only a tokenizer_config-shaped object.
    if isinstance(payload, str):
        return payload if payload.strip() else None
    template = _chat_template_from_tokenizer_config(payload)  # type: ignore[arg-type]
    if template:
        return template
    if isinstance(payload, dict):
        # Named-template map: prefer "default", else the first non-empty entry
        # (mirrors the tokenizer-config list fallback).
        default = payload.get("default")
        if isinstance(default, str) and default.strip():
            return default
        for value in payload.values():
            if isinstance(value, str) and value.strip():
                return value
    return None


def _chat_template_from_processor_json(
    dir_path: Path, allow_roots: Optional[list[Path]] = None
) -> Optional[str]:
    for rel in _PROCESSOR_TEMPLATE_PATHS:
        config_file = dir_path / rel
        if not config_file.exists() or not _leaf_inside_allowlist(config_file, allow_roots):
            continue
        try:
            payload = json.loads(config_file.read_text(encoding = "utf-8"))
        except Exception:
            continue
        template = _chat_template_from_processor_payload(payload)
        if template:
            return template
    return None


def _chat_template_from_tokenizer_dir(
    dir_path: Path, allow_roots: Optional[list[Path]] = None
) -> Optional[str]:
    jinja = _chat_template_from_jinja_file(dir_path, allow_roots)
    if jinja:
        return jinja
    for rel in _TOKENIZER_CONFIG_PATHS:
        config_file = dir_path / rel
        if not config_file.exists() or not _leaf_inside_allowlist(config_file, allow_roots):
            continue
        try:
            config = json.loads(config_file.read_text(encoding = "utf-8"))
        except Exception:
            continue
        template = _chat_template_from_tokenizer_config(config)
        if template:
            return template
    return _chat_template_from_processor_json(dir_path, allow_roots)


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


def _chat_template_from_dir(
    dir_path: Path,
    gguf_variant: Optional[str] = None,
    allow_roots: Optional[list[Path]] = None,
) -> Optional[str]:
    def from_gguf() -> Optional[str]:
        gguf = _find_gguf_in_dir(dir_path, gguf_variant)
        if gguf is None or not _leaf_inside_allowlist(gguf, allow_roots):
            return None
        return read_gguf_chat_template(str(gguf))

    # Sidecar tokenizer files (chat_template.jinja / tokenizer_config.json) are
    # the model author's maintained template and supersede the GGUF's embedded
    # copy, which can be stale. The variant only selects which GGUF to fall back
    # to, so keep tokenizer-first precedence whether or not a variant is given.
    return _chat_template_from_tokenizer_dir(dir_path, allow_roots) or from_gguf()


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
            allow_roots = _build_browse_allowlist()
            if not _is_path_inside_allowlist(target, allow_roots):
                logger.debug("Refused chat template read outside allowed folders: %s", name)
                return None
            if name.lower().endswith(".gguf"):
                # Prefer a maintained sidecar template (chat_template.jinja /
                # tokenizer_config.json) next to the file over the GGUF's embedded
                # copy, matching the tokenizer-first precedence used for directory
                # and variant selections.
                sidecar = _chat_template_from_tokenizer_dir(target.parent, allow_roots)
                if sidecar:
                    return sidecar
                return read_gguf_chat_template(str(target))
            return _chat_template_from_dir(target, gguf_variant, allow_roots)
        except Exception as exc:
            logger.debug("Could not read local chat template for %s: %s", name, exc)
            return None

    if not _is_valid_repo_id(name):
        return None

    resolved = resolve_cached_repo_id_case(name)

    try:
        # Resolve within each cached revision, newest first. A revision's
        # maintained sidecar (chat_template.jinja / tokenizer_config.json)
        # supersedes its own embedded GGUF copy, but a newer revision must not be
        # overridden by an older revision's sidecar, so precedence stays
        # per-snapshot rather than searching all sidecars globally first.
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

        for rel in _TOKENIZER_CONFIG_PATHS:
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

        for rel in _PROCESSOR_TEMPLATE_PATHS:
            raw = _download_text(rel)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            template = _chat_template_from_processor_payload(payload)
            if template:
                return template

        return None
    except Exception as exc:
        logger.debug("Could not fetch chat template for %s: %s", resolved, exc)
        return None
