# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether two model names mean the same model: Hub ids fold case, local paths and ref: handles keep it, and repo:variant names one GGUF quant of a repo. Names only; callers probe what is resident."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

REF_PREFIX = "ref:"
# audio.cpp folder ids name a GGUF repo plus a model folder: three segments, still a Hub id.
AUDIO_CPP_PREFIX = "audio-cpp/audio.cpp-gguf/"


def _text(name: Optional[str]) -> str:
    return str(name or "").strip()


def looks_like_path(name: Optional[str], *, strict: bool = False) -> bool:
    """A host path by its spelling: rooted, home, dot-relative, a backslash, or a drive letter. ``strict`` takes only rooted, home and backslash names, as the OpenAI route has always read them."""
    if strict:
        text = str(name)
        return text.startswith(("/", "~")) or "\\" in text
    text = _text(name)
    if text.startswith(("/", "\\", "~", ".")) or "\\" in text:
        return True
    return len(text) >= 2 and text[1] == ":" and text[0].isalpha()


def on_disk(name: Optional[str]) -> bool:
    """A relative path (models/Foo.gguf) reads like a Hub id; one that exists is a path."""
    text = _text(name)
    if not text:
        return False
    try:
        return Path(text).expanduser().exists()
    except (OSError, ValueError):
        return False


def is_local(name: Optional[str], *, probe: bool = False) -> bool:
    """A host path or the ref:<digest> handle of a local model: whole, never repo:variant. ``probe`` also counts a relative name that exists on disk."""
    text = _text(name)
    if text.startswith(REF_PREFIX) or looks_like_path(text):
        return True
    return probe and on_disk(text)


def is_audio_cpp(name: Optional[str]) -> bool:
    return _text(name).lower().startswith(AUDIO_CPP_PREFIX)


def path_key(name: Optional[str]) -> str:
    """normcase, not lower: on a case-sensitive filesystem /srv/models/Foo and /srv/models/foo differ."""
    # Replace after normcase: on Windows normcase rewrites '/' to a backslash.
    return os.path.normcase(_text(name)).replace("\\", "/").rstrip("/")


def identity_key(name: Optional[str], *, probe: bool = False) -> str:
    """A key that groups names of one model; compare two names with same_model, which keeps a path's case when either side is one."""
    text = _text(name)
    return path_key(text) if is_local(text, probe = probe) else text.lower()


def split_variant(name: Optional[str], *, probe: bool = False) -> tuple[str, Optional[str]]:
    """``repo:Q4_K_M`` into the repo and the variant. A path's drive letter or colon and a ref: handle are not a variant."""
    text = _text(name)
    if is_local(text, probe = probe):
        return text, None
    head, sep, tail = text.partition(":")
    if sep and head and "/" not in tail:
        return head, tail or None
    return text, None


def same_model(
    left: Optional[str],
    right: Optional[str],
    *,
    probe: bool = False,
) -> bool:
    """Hub ids match in any case; when either name is local both compare as paths."""
    left, right = _text(left), _text(right)
    if not left or not right:
        return False
    if is_local(left, probe = probe) or is_local(right, probe = probe):
        return path_key(left) == path_key(right)
    return left.lower() == right.lower()


def same_repo(
    left: Optional[str],
    right: Optional[str],
    *,
    probe: bool = False,
) -> bool:
    """The same model, or two variants of one GGUF repo (repo:variant beside the bare repo)."""
    if same_model(left, right, probe = probe):
        return True
    left_repo, _ = split_variant(left, probe = probe)
    right_repo, _ = split_variant(right, probe = probe)
    return same_model(left_repo, right_repo, probe = probe)


def matches_any(
    requested: Optional[str],
    candidates: Iterable[Optional[str]],
    *,
    probe: bool = False,
) -> bool:
    return any(same_model(requested, candidate, probe = probe) for candidate in candidates)


def same_variant(left: Optional[str], right: Optional[str]) -> bool:
    """GGUF quants match in any case; no variant on both sides is a match."""
    return _text(left).lower() == _text(right).lower()


def resident_quant_matches(
    resident_variant: Optional[str],
    wanted_filename: str,
    *,
    settled: bool = True,
) -> Optional[bool]:
    """Whether a status that names only a quant holds the GGUF ``wanted_filename``; None when the file names no quant to compare. Status drops a -3.97bpw style suffix, so until the load settles a bpw build is not taken as resident: a sibling build matches its quant too."""
    from hub.utils.gguf import extract_quant_token, quant_token_with_bpw

    quant = (
        extract_quant_token(wanted_filename) if wanted_filename.lower().endswith(".gguf") else None
    )
    if not quant:
        return None
    if not same_variant(resident_variant, quant):
        return False
    return settled or quant_token_with_bpw(wanted_filename) == quant
