# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Family evidence for a Hub single-file pick whose id and filename name no family.

Fine-tunes ship under their own names (``Laxhar/noobai-XL-Vpred-1.0``,
``OnomaAIResearch/Illustrious-xl-early-release-v0``), so the name matcher answers None and the pick
is refused as an unknown family although the checkpoint states what it is: its safetensors header
holds a whole SDXL checkpoint (UNet, both text encoders, VAE). Only that is read: the picked file's
header, and only for whole-pipeline families (SDXL), whose checkpoint is self-contained; a
transformer-only family resolves its base and defaults from the name. A pipeline pick needs no
evidence here: a remote one loads only from trusted (named) repos, and a local folder is already
read through its ``model_index.json``.

Evidence is advisory. ``prefetch_for_pick`` (the routes, user-initiated requests only) re-reads the
header off the Hub on every request; ``cached_family`` hands that verdict, or a cached copy's header,
to the network-free ``detect_family_for_pick``. The load opens the downloaded file itself, and that
stays the authoritative check.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

_SINGLE_FILE_SUFFIXES = (".safetensors", ".sft")

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
    """The family a Hub single-file pick's header declares, from memory or disk only; None when
    nothing is known. A header read off the Hub by this request's prefetch outranks a cached copy,
    which may be an older revision of the file the load will fetch."""
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


def prefetch_for_pick(
    repo_id: Optional[str],
    filename: Optional[str],
    family_override: Optional[str],
    hf_token: Any,
    allow_network: bool = True,
) -> None:
    """The route hook: read the picked single file's header off the Hub (a range read, no weights,
    nothing cached) when neither the name, a local path nor the content answered. ``allow_network=
    False`` (an automatic, cache-only load) does nothing. ``hf_token`` passes through as given apart
    from trimming a string: ``False`` (a managed account with no token of its own) stays anonymous.
    Fails open: offline, gated or odd keeps the previous verdict, if any."""
    if (
        not allow_network
        or family_override
        or not is_hub_id(repo_id)
        or not _is_single_file(filename)
    ):
        return
    repo_id = str(repo_id).strip()
    filename = str(filename)
    try:
        from .diffusion_families import _family_from_hub_evidence, detect_family_for_pick

        # Skip only when the name, a local path or the content answered; an answer from Hub evidence is re-read.
        if (
            detect_family_for_pick(repo_id, filename, None) is not None
            and _family_from_hub_evidence(repo_id, filename) is None
        ):
            return
        from huggingface_hub import HfApi

        from .diffusion_content import classify_tensors

        from hub.utils.hf_tokens import normalize_token

        token = normalize_token(hf_token)
        meta = HfApi().parse_safetensors_file_metadata(repo_id, filename, token = token)
        shapes = {name: list(info.shape) for name, info in meta.tensors.items()}
        family = _dit_family(classify_tensors(shapes, meta.metadata))
    except Exception:  # noqa: BLE001 - evidence is best effort; validation and the load still decide
        return
    with _LOCK:
        if len(_HEADER_VERDICTS) >= _MAX_VERDICTS:
            _HEADER_VERDICTS.clear()
        _HEADER_VERDICTS[_key(repo_id, filename)] = (time.monotonic(), family)
