# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Public model identifiers for the OpenAI-compatible API.

The exposed API must report a stable, clean model id rather than the absolute
on-disk path of a local GGUF. The internal identifier for a direct local load is
the absolute ``.gguf`` path, which leaks the host filesystem layout and is
awkward for clients to round-trip. ``public_model_id`` maps such an internal
identifier to a clean name while leaving Hugging Face repo ids (``org/model``)
and already-clean names untouched.
"""

from __future__ import annotations

import os
from typing import Optional

_GGUF_SUFFIX = ".gguf"


def _looks_like_path(identifier: str) -> bool:
    """True when *identifier* is a local filesystem path, not a HF repo id.

    A repo id is ``org/model`` (a single forward slash, no leading separator, no
    drive, no ``.gguf``). Anything ending in ``.gguf``, starting with a path
    separator or a relative/home prefix (``./``, ``../``, ``~``), carrying a
    Windows drive, or with three or more ``/`` segments is treated as a local
    path.
    """
    if identifier.lower().endswith(_GGUF_SUFFIX):
        return True
    if identifier.startswith(("/", "\\", "./", "../", ".\\", "..\\", "~")):
        return True
    if len(identifier) >= 2 and identifier[1] == ":":  # Windows drive, e.g. C:\
        return True
    if identifier.count("/") >= 2 or "\\" in identifier:
        return True
    return False


def hf_cache_repo_id(path: Optional[str]) -> Optional[str]:
    """``.../models--org--name/snapshots/<sha>`` -> ``org/name``, else None.

    A model loaded from the HF cache is identified by its snapshot dir, whose
    basename is a commit hash; recover the repo id so callers don't show that.
    """
    if not path:
        return None
    parts = str(path).replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        # Only inside the real cache layout: a "models--" name alone is not a repo id.
        if part.startswith("models--") and parts[index + 1 : index + 2] == ["snapshots"]:
            return part[len("models--") :].replace("--", "/")
    return None


def public_model_id(identifier: Optional[str]) -> Optional[str]:
    """Return a clean, path-free public id for *identifier*.

    - HF cache path -> the repo id it came from, e.g.
      ``~/.cache/huggingface/hub/models--unsloth--X-GGUF/snapshots/<sha>`` ->
      ``unsloth/X-GGUF``.
    - Other local GGUF path -> the file stem with ``.gguf`` stripped, e.g.
      ``/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf`` -> ``Qwen3-30B-A3B-Q4_K_M``.
    - HF repo id (``org/model``) and already-clean names -> returned unchanged.
    - ``None`` / empty -> returned unchanged.
    """
    if not identifier:
        return identifier
    if not _looks_like_path(identifier):
        return identifier
    repo_id = hf_cache_repo_id(identifier)
    if repo_id:
        return repo_id
    name = os.path.basename(identifier.replace("\\", "/").rstrip("/"))
    if name.lower().endswith(_GGUF_SUFFIX):
        name = name[: -len(_GGUF_SUFFIX)]
    return name or identifier


def display_model_name(identifier: Optional[str]) -> Optional[str]:
    """The short label a UI should show for *identifier*.

    ``public_model_id`` then its trailing segment, so a HF cache snapshot reads as
    ``X-GGUF`` and not the commit sha its directory is named after. Splitting the raw
    identifier instead leaks the host layout on Windows, where ``C:\\Users\\...`` has
    no ``/`` to split on and the whole path becomes the label.
    """
    clean = public_model_id(identifier)
    if not clean:
        return clean
    return clean.rsplit("/", 1)[-1] or clean


def model_id_matches(requested: Optional[str], internal: Optional[str]) -> bool:
    """Whether a client-supplied *requested* id refers to *internal*.

    Accepts the clean public id (preferred) and, for backward compatibility, the
    raw internal identifier (e.g. a legacy absolute path a client cached from an
    older ``/v1/models`` response).
    """
    if requested is None or internal is None:
        return False
    if requested == internal:
        return True
    return public_model_id(internal) == requested
