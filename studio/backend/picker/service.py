# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from hub.utils.hf_tokens import (
    cache_reads_authorized,
    cached_read_refused,
    recording_a_request_token_fetch,
)
from hub.services.models.folder_browser import (
    _build_browse_allowlist,
    _is_path_inside_allowlist,
)
from hub.utils.gguf import (
    extract_quant_label,
    iter_snapshots_preferring_whole,
)
from utils.models.gguf_metadata import read_gguf_chat_template
from utils.models.model_config import (
    _extract_quant_label,
    _is_big_endian_gguf_path,
    _is_mmproj,
    _is_mtp_drafter,
)
from utils.hf_cache_settings import active_hf_hub_cache
from utils.utils import hf_env_offline
from utils.paths.path_utils import (
    is_local_path,
    normalize_path,
    resolve_cached_repo_id_case,
)

from .schemas import MAX_CHAT_TEMPLATE_BYTES, ValidateChatTemplateResponse
from utils.paths.path_utils import is_appledouble_metadata

logger = logging.getLogger(__name__)

_VALID_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _is_valid_repo_id(repo_id: str) -> bool:
    return bool(_VALID_REPO_ID.fullmatch(repo_id))


_TOKENIZER_CONFIG_PATHS = ("tokenizer_config.json", "LLM/tokenizer_config.json")
_JINJA_TEMPLATE_PATHS = ("chat_template.jinja", "LLM/chat_template.jinja")
_PROCESSOR_TEMPLATE_PATHS = ("chat_template.json", "LLM/chat_template.json")

# Cap sidecar reads so a hostile metadata file cannot exhaust memory; the extracted
# template is still bounded by MAX_CHAT_TEMPLATE_BYTES downstream.
MAX_TEMPLATE_METADATA_BYTES = 4 * 1024 * 1024


def _read_bounded_text(path: Path, limit: int) -> Optional[str]:
    """Read at most `limit` bytes of UTF-8 text; None if larger or unreadable."""
    try:
        with path.open("rb") as f:
            data = f.read(limit + 1)
    except OSError:
        return None
    if len(data) > limit:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeError:
        return None


def _leaf_inside_allowlist(path: Path, allow_roots: Optional[list[Path]]) -> bool:
    # Blocks symlinked children escaping the validated directory (realpath-checked).
    # None = trusted caller (HF cache / remote download).
    return allow_roots is None or _is_path_inside_allowlist(path, allow_roots)


def validate_chat_template(template: str) -> ValidateChatTemplateResponse:
    text = (template or "").strip()
    if not text:
        return ValidateChatTemplateResponse(valid = True, error = None)
    # Jinja is optional (GGUF-only installs); must not crash startup.
    try:
        from jinja2 import TemplateError
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        return ValidateChatTemplateResponse(valid = True, error = None)

    class _GenerationTag(Extension):
        # Accept Transformers' {% generation %} tag so pasted HF templates validate.
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
            if template_file.stat().st_size > MAX_CHAT_TEMPLATE_BYTES:
                continue
            template = template_file.read_text(encoding = "utf-8")
        except Exception:
            continue
        if template.strip():
            return template
    return None


def _chat_template_from_processor_payload(payload: object) -> Optional[str]:
    if isinstance(payload, str):
        return payload if payload.strip() else None
    template = _chat_template_from_tokenizer_config(payload)  # type: ignore[arg-type]
    if template:
        return template
    if isinstance(payload, dict):
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
        raw = _read_bounded_text(config_file, MAX_TEMPLATE_METADATA_BYTES)
        if raw is None:
            continue
        try:
            payload = json.loads(raw)
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
        raw = _read_bounded_text(config_file, MAX_TEMPLATE_METADATA_BYTES)
        if raw is None:
            continue
        try:
            config = json.loads(raw)
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
            if is_appledouble_metadata(path):
                continue
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
    from hub.utils.gguf import gguf_variant_key

    # Variant key first: the bare label can name several checkpoints at one quant.
    if gguf_variant_key(relative_path).lower() == needle:
        return True
    quant = _extract_quant_label(relative_path).lower()
    if quant == needle:
        return True
    if extract_quant_label(relative_path).lower() == needle:
        return True
    prefix = f"{needle}-"
    if not quant.startswith(prefix):
        return False
    suffix = quant[len(prefix) :]
    if not suffix.endswith("bpw"):
        return False
    value = suffix[:-3]
    return bool(value) and value.replace(".", "", 1).isdigit()


_GGUF_SPLIT_INDEX_RE = re.compile(r"-(\d{3,})-of-\d{3,}$", re.IGNORECASE)


def _is_nonfirst_gguf_split(path: Path) -> bool:
    match = _GGUF_SPLIT_INDEX_RE.search(path.stem)
    return match is not None and int(match.group(1)) != 1


def _find_gguf_in_dir(dir_path: Path, gguf_variant: Optional[str]) -> Optional[Path]:
    try:
        ggufs = sorted(_iter_ggufs(dir_path))
    except OSError:
        return None
    if not ggufs:
        return None
    needle = (gguf_variant or "").strip().lower()
    if needle:
        from hub.utils.gguf import gguf_variant_key

        def _relative(path: Path) -> str:
            try:
                return path.relative_to(dir_path).as_posix()
            except ValueError:
                return path.name

        for owned in (True, False):
            for path in ggufs:
                relative = _relative(path)
                if owned != (gguf_variant_key(relative).lower() == needle):
                    continue
                if _variant_matches(relative, needle):
                    return path
        return None
    candidates = [path for path in ggufs if not _is_nonfirst_gguf_split(path)] or ggufs
    try:
        return max(candidates, key = lambda path: path.stat().st_size)
    except OSError:
        return candidates[0]


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

    # Sidecar tokenizer files supersede the GGUF's possibly-stale embedded template.
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

    # Walk locally first, then authorize: a denied caller is sent to the Hub.
    cached_template = None
    try:
        # Precedence is per-snapshot: a sidecar must not override a newer revision.
        for snapshot in iter_snapshots_preferring_whole(resolved, gguf_variant):
            cached_template = _chat_template_from_dir(snapshot, gguf_variant)
            if cached_template:
                break
    except Exception as exc:
        logger.debug("Could not read cached chat template for %s: %s", resolved, exc)
        cached_template = None
    if cached_template and cache_reads_authorized(hf_token, repo_id = resolved):
        return cached_template

    if hf_env_offline() and not cache_reads_authorized(hf_token, repo_id = resolved):
        # Offline hf_hub_download serves the cache without checking the credential.
        return None

    try:
        from huggingface_hub import HfApi, hf_hub_download

        _api = HfApi(token = hf_token)

        def _this_file_is_cached(rel: str) -> bool:
            """THIS file at this revision, not merely a directory for the repo.

            What the download could serve from disk is the one candidate template, so a
            snapshot holding only weights can answer nothing and refusing it costs an
            authorized caller a template the Hub would have given it. Fails closed.
            """
            try:
                from huggingface_hub import try_to_load_from_cache

                # Same cache root as the download below, or a miss opens the gate.
                return isinstance(
                    try_to_load_from_cache(
                        repo_id = resolved, filename = rel, cache_dir = active_hf_hub_cache()
                    ),
                    str,
                )
            except Exception:
                return True

        def _remote_worth_downloading(rel: str) -> bool:
            # hf_hub_download returns the cached pointer for any failed head call, a 403 included.
            if cached_read_refused(
                hf_token,
                repo_id = resolved,
                is_cached = lambda: _this_file_is_cached(rel),
            ):
                return False
            try:
                infos = _api.get_paths_info(resolved, [rel], repo_type = "model", token = hf_token)
            except Exception:
                # Nothing cached to serve: let the Hub enforce its own access.
                return True
            matched = [info for info in infos if getattr(info, "path", None) == rel]
            if not matched:
                return False
            size = getattr(matched[0], "size", None)
            return not (isinstance(size, int) and size > MAX_TEMPLATE_METADATA_BYTES)

        def _this_file_was_already_here(rel: str) -> bool:
            """``_this_file_is_cached`` asks the same question and answers True when it cannot
            tell, which is right for a gate and wrong here: not knowing must RECORD, since an
            unrecorded credentialed fetch is what hands a private repo to a tokenless caller."""
            try:
                from huggingface_hub import try_to_load_from_cache
                return isinstance(
                    try_to_load_from_cache(
                        repo_id = resolved, filename = rel, cache_dir = active_hf_hub_cache()
                    ),
                    str,
                )
            except Exception:  # noqa: BLE001 -- cannot tell, so record
                return False

        def _download_text(rel: str) -> Optional[str]:
            if not _remote_worth_downloading(rel):
                return None
            try:
                # Record only real fetches: a cached return does not prove the token's access.
                recording = (
                    recording_a_request_token_fetch(hf_token, resolved, "model")
                    if not _this_file_was_already_here(rel)
                    else nullcontext()
                )
                with recording:
                    path = hf_hub_download(
                        resolved,
                        rel,
                        token = hf_token,
                        cache_dir = active_hf_hub_cache(),
                    )
                return _read_bounded_text(Path(path), MAX_TEMPLATE_METADATA_BYTES)
            except Exception:
                return None

        for rel in _JINJA_TEMPLATE_PATHS:
            template = _download_text(rel)
            if not template or not template.strip():
                continue
            # An over-cap Jinja is dropped so the search falls through to tokenizer templates.
            if len(template.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES:
                continue
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
