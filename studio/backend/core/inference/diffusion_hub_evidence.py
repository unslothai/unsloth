# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Family evidence from a Hub single-file header (SDXL only); advisory, the load check stays final;
remote pipeline picks need no evidence (trusted repos); detect_family_for_pick is network-free;
prefetch_for_pick only for user requests."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

_SINGLE_FILE_SUFFIXES = (".safetensors", ".sft", ".gguf")
_GGUF_TABLE_BYTES = 1024 * 1024

# (repo id lowercased, filename) -> (monotonic time read, family name or None for a header that names no family).
# Only bridges a request's prefetch to the validation and load that follow it; a refresh REPLACES an entry (never
# clears it first, so an overlapping request keeps seeing the last good read), and an entry expires on its own.
_HEADER_VERDICTS: dict[tuple[str, str], tuple[float, Optional[str]]] = {}
_LOCK = threading.Lock()
_MAX_VERDICTS = 1024
_VERDICT_TTL_S = 600.0


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


def _key(repo_id: str, filename: str) -> tuple[str, str]:
    return repo_id.strip().lower(), str(filename).strip()


def _dit_family(info: Any) -> Optional[str]:
    """The whole-pipeline family (SDXL) a header classifies as, or None. A denoiser-only file (a bare
    UNet with no text encoders or VAE) is not evidence: SDXL loads its single file through the WHOLE
    pipeline's ``from_single_file``."""
    from .diffusion_content import ROLE_DIT
    from .diffusion_families import detect_family

    if info.role != ROLE_DIT or not info.whole_pipeline or not info.family:
        return None
    fam = detect_family("", override = info.family)
    return info.family if fam is not None and fam.single_file_is_pipeline else None


def _verdict(key: tuple[str, str]) -> tuple[bool, Optional[str]]:
    """``(known, family)`` for a remembered header read that has not expired."""
    with _LOCK:
        hit = _HEADER_VERDICTS.get(key)
        if hit is None:
            return False, None
        if time.monotonic() - hit[0] > _VERDICT_TTL_S:
            del _HEADER_VERDICTS[key]
            return False, None
        return True, hit[1]


def cached_family(repo_id: Optional[str], filename: Optional[str] = None) -> Optional[str]:
    """The family a Hub single-file header declares, from memory or disk only (None if unknown); a
    prefetch read off the Hub outranks a cached copy, which may be an older revision."""
    if not is_hub_id(repo_id) or not _is_single_file(filename):
        return None
    repo_id = str(repo_id).strip()
    known, family = _verdict(_key(repo_id, str(filename)))
    if known:
        return family
    from .diffusion_compat import _local_gguf_path

    local = _local_gguf_path(repo_id, str(filename))
    if not local:
        return None
    from .diffusion_content import inspect_checkpoint

    return _dit_family(inspect_checkpoint(local))


def read_hub_checkpoint_family(
    repo_id: str,
    filename: str,
    hf_token: Any,
    revision: Optional[str] = None,
) -> Optional[str]:
    """Identify a bundled pipeline by its tensor header, independent of the repo and file names."""
    from huggingface_hub import HfApi

    from .diffusion_content import classify_tensors

    from hub.utils.hf_tokens import normalize_token

    token = normalize_token(hf_token)
    meta = HfApi().parse_safetensors_file_metadata(
        repo_id, filename, token = token, revision = revision
    )
    shapes = {name: list(info.shape) for name, info in meta.tensors.items()}
    return _dit_family(classify_tensors(shapes, meta.metadata))


def read_hub_gguf_family(repo_id: str, filename: str, hf_token: Any) -> Optional[str]:
    from .diffusion_compat import _read_gguf_header
    from .diffusion_content import classify_tensors
    from .diffusion_families import _gguf_header_tensors

    from hub.utils.hf_tokens import normalize_token

    tensors = _gguf_header_tensors(
        _read_gguf_header(repo_id, filename, normalize_token(hf_token), max_bytes = _GGUF_TABLE_BYTES)
    )
    if tensors is None:
        raise ValueError("no complete GGUF tensor table in the header read")
    shapes = {str(t.name): [int(d) for d in reversed(t.shape)] for t in tensors}
    return _dit_family(classify_tensors(shapes))


def prefetch_for_pick(
    repo_id: Optional[str],
    filename: Optional[str],
    family_override: Optional[str],
    hf_token: Any,
    allow_network: bool = True,
) -> None:
    """Reads a picked single file's header off the Hub (range read, nothing cached) when name, path
    and content gave no verdict, or an uncached GGUF of a whole-pipeline family; allow_network=False does nothing; hf_token passes through, False
    stays anonymous; fails open."""
    if not allow_network or not is_hub_id(repo_id) or not _is_single_file(filename):
        return
    repo_id = str(repo_id).strip()
    filename = str(filename)
    try:
        from .diffusion_compat import _local_gguf_path
        from .diffusion_families import (
            _family_from_hub_evidence,
            detect_family,
            detect_family_for_pick,
        )

        is_gguf = filename.lower().endswith(".gguf")
        fam = (
            detect_family("", override = family_override)
            if family_override
            else detect_family_for_pick(repo_id, filename, None)
        )
        # An uncached whole-pipeline-family GGUF loads only if its header is whole, even under an override.
        unread_whole_gguf = (
            is_gguf
            and fam is not None
            and fam.single_file_is_pipeline
            and not _local_gguf_path(repo_id, filename)
        )
        if not unread_whole_gguf and (
            family_override
            or (fam is not None and _family_from_hub_evidence(repo_id, filename) is None)
        ):
            return
        if is_gguf:
            family = read_hub_gguf_family(repo_id, filename, hf_token)
        else:
            family = read_hub_checkpoint_family(repo_id, filename, hf_token)
    except Exception:  # noqa: BLE001 - evidence is best effort; validation and the load still decide
        return
    _remember(repo_id, filename, family)


def _remember(repo_id: str, filename: str, family: Optional[str]) -> None:
    with _LOCK:
        if len(_HEADER_VERDICTS) >= _MAX_VERDICTS:
            _HEADER_VERDICTS.clear()
        _HEADER_VERDICTS[_key(repo_id, filename)] = (time.monotonic(), family)


def hub_checkpoint_loads_as_sdxl(
    repo_id: str,
    filename: str,
    hf_token: Any,
    revision: Optional[str] = None,
) -> bool:
    """A whole-SDXL header AND the loader's own verdict, so name-vetoed variants (inpaint, edit) are not offered."""
    from .diffusion_families import detect_family_for_pick

    family = read_hub_checkpoint_family(repo_id, filename, hf_token, revision)
    if family != "sdxl":
        return False
    _remember(repo_id, filename, family)
    fam = detect_family_for_pick(repo_id, filename, None)
    return fam is not None and fam.name == "sdxl"
