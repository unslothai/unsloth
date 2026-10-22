# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Family evidence for a Hub pick whose id and filename name no family.

Fine-tunes ship under their own names (``Laxhar/noobai-XL-Vpred-1.0``,
``OnomaAIResearch/Illustrious-xl-early-release-v0``), so the name matcher answers None and the pick
is refused as an unknown family although the repo states what it is: its ``model_index.json``
names the pipeline class, and the picked checkpoint's tensor names give the denoiser layout.

``cached_family`` reads only what is already on disk (the HF cache, plus header verdicts fetched
earlier in this process), so ``detect_family_for_pick`` stays network-free.
``fetch_family_evidence`` is the route-side step that pulls the one small file or header first.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

_INDEX = "model_index.json"
_INDEX_MAX_BYTES = 1_000_000
_SINGLE_FILE_SUFFIXES = (".safetensors", ".sft")

# (repo id lowercased, filename) -> family name, or None for a header that was read and names no family.
_HEADER_VERDICTS: dict[tuple[str, str], Optional[str]] = {}
_LOCK = threading.Lock()
_MAX_VERDICTS = 1024


def is_hub_id(repo_id: Optional[str]) -> bool:
    """True for a bare ``owner/name`` that is not a path on this machine."""
    rid = (repo_id or "").strip()
    if rid.count("/") != 1 or "\\" in rid or rid.startswith((".", "/", "~")):
        return False
    try:
        return not Path(rid).expanduser().exists()
    except (OSError, RuntimeError, ValueError):
        return True


def _is_single_file(filename: Optional[str]) -> bool:
    return bool(filename) and str(filename).lower().endswith(_SINGLE_FILE_SUFFIXES)


def _applies(filename: Optional[str]) -> bool:
    """Only a pipeline pick or a single ``.safetensors`` pick; a GGUF has its own header path."""
    return filename is None or _is_single_file(filename)


def _index_speaks_for(filename: Optional[str]) -> bool:
    """The repo's index vouches for the pipeline itself and for a checkpoint at the repo ROOT (the
    packaged single file a diffusers repo ships beside its folders), never for a file in a subfolder."""
    return filename is None or "/" not in str(filename).strip("/")


def _key(repo_id: str, filename: str) -> tuple[str, str]:
    return repo_id.strip().lower(), str(filename).strip()


def _cached_file(repo_id: str, filename: str) -> Optional[str]:
    from .diffusion_compat import _local_gguf_path
    return _local_gguf_path(repo_id, filename)


def _dit_family(info: Any) -> Optional[str]:
    from .diffusion_content import ROLE_DIT
    return info.family if info.role == ROLE_DIT and info.family else None


def _family_from_index(path: str) -> Optional[str]:
    from .diffusion_families import detect_family_by_pipeline_class

    try:
        index = Path(path)
        if index.stat().st_size > _INDEX_MAX_BYTES:
            return None
        payload = json.loads(index.read_text(encoding = "utf-8-sig"))
    except (OSError, ValueError, RecursionError):
        return None
    value = payload.get("_class_name") if isinstance(payload, dict) else None
    fam = detect_family_by_pipeline_class(value if isinstance(value, str) else None)
    return fam.name if fam is not None else None


def cached_family(repo_id: Optional[str], filename: Optional[str] = None) -> Optional[str]:
    """The family a Hub pick's own files declare, from disk only; None when nothing is known.

    The picked checkpoint outranks the repo index: a header that was read and names no family
    (an SD 1.5 UNet, a LoRA) answers None even beside an SDXL index."""
    if not is_hub_id(repo_id) or not _applies(filename):
        return None
    repo_id = str(repo_id).strip()
    if filename:
        local = _cached_file(repo_id, filename)
        if local:
            from .diffusion_content import inspect_checkpoint
            return _dit_family(inspect_checkpoint(local))
        key = _key(repo_id, filename)
        with _LOCK:
            if key in _HEADER_VERDICTS:
                return _HEADER_VERDICTS[key]
    if _index_speaks_for(filename):
        index = _cached_file(repo_id, _INDEX)
        if index:
            return _family_from_index(index)
    return None


def _remember(key: tuple[str, str], family: Optional[str]) -> None:
    with _LOCK:
        if len(_HEADER_VERDICTS) >= _MAX_VERDICTS:
            _HEADER_VERDICTS.clear()
        _HEADER_VERDICTS[key] = family


def fetch_family_evidence(
    repo_id: Optional[str],
    filename: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Pull the evidence ``cached_family`` reads: the picked single file's safetensors HEADER (a range
    read, no weights) and the repo's ``model_index.json`` (a few KB, into the HF cache). Run by the
    routes before ``validate_load_request`` and only for a pick whose name matched no family. Fails
    open: offline, gated or odd leaves the pick as unknown as it was."""
    if not is_hub_id(repo_id) or not _applies(filename):
        return None
    repo_id = str(repo_id).strip()
    token = (hf_token.strip() if isinstance(hf_token, str) else hf_token) or None
    known = cached_family(repo_id, filename)
    if known is not None:
        return known
    if filename and _cached_file(repo_id, filename) is None:
        key = _key(repo_id, filename)
        with _LOCK:
            seen = key in _HEADER_VERDICTS
        if not seen:
            try:
                from huggingface_hub import HfApi

                from .diffusion_content import classify_tensors

                meta = HfApi().parse_safetensors_file_metadata(repo_id, filename, token = token)
                shapes = {name: list(info.shape) for name, info in meta.tensors.items()}
                _remember(key, _dit_family(classify_tensors(shapes, meta.metadata)))
            except Exception:  # noqa: BLE001 - offline / gated / not a safetensors: the index may still answer
                pass
        with _LOCK:
            if key in _HEADER_VERDICTS:
                return _HEADER_VERDICTS[key]
    if _index_speaks_for(filename) and _cached_file(repo_id, _INDEX) is None:
        try:
            from huggingface_hub import hf_hub_download
            from utils.hf_cache_settings import active_hf_hub_cache
            hf_hub_download(repo_id, _INDEX, token = token, cache_dir = active_hf_hub_cache())
        except Exception:  # noqa: BLE001 - no index (a single-file repo), offline or gated
            pass
    return cached_family(repo_id, filename)


def prefetch_for_pick(
    repo_id: Optional[str],
    filename: Optional[str],
    family_override: Optional[str],
    hf_token: Optional[str],
) -> None:
    """The route hook: fetch evidence only when the network-free detector has no answer yet."""
    if family_override or not is_hub_id(repo_id) or not _applies(filename):
        return
    try:
        from .diffusion_families import detect_family_for_pick
        if detect_family_for_pick(str(repo_id), filename, None) is not None:
            return
        fetch_family_evidence(repo_id, filename, hf_token)
    except Exception:  # noqa: BLE001 - evidence is best effort; validation still decides
        return
