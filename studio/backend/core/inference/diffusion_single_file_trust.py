# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-file trust for single-file diffusion checkpoints from any Hugging Face repo.

The repo allowlists in ``diffusion.py`` / ``video.py`` exist because ``from_pretrained`` on an
arbitrary repo can unpickle ``.bin`` / ``.pt`` weights and, with ``trust_remote_code``, import
``.py`` files from it. A single ``.safetensors`` checkpoint has neither risk: Studio downloads that
one exact file, the format is a JSON header plus raw tensor bytes that the safetensors parser reads
without executing anything, and every config, scheduler, tokenizer, VAE and text encoder still comes
from the family's own trusted base repo. So a ``single_file`` load is judged by its FILE, not its
repo: a ``.safetensors`` name is admitted from any repo, every other suffix still needs a trusted
repo, and the downloaded bytes are checked to really be a safetensors container before any loader
touches them. Pipeline loads and ``base_repo`` stay gated by repo exactly as before.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Optional, Union

SAFETENSORS_SUFFIX = ".safetensors"

# The safetensors reference parser refuses headers above 100 MB; matching it keeps a crafted length
# prefix from turning a header read into an unbounded allocation.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def is_hub_safetensors_single_file(filename: Optional[str]) -> bool:
    """Whether ``filename`` names a ``.safetensors`` file INSIDE a repo, so it can be loaded from
    any repo. Repo-relative only: no absolute path, no ``..`` segment, no backslash or NUL, and the
    final suffix must be ``.safetensors`` (so ``x.safetensors.pt`` or ``x.bin`` never qualify)."""
    if not isinstance(filename, str):
        return False
    name = filename.strip()
    if not name or name != filename or "\\" in name or "\x00" in name:
        return False
    if name.startswith("/") or ":" in name:
        return False
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    leaf = parts[-1]
    return leaf.lower().endswith(SAFETENSORS_SUFFIX) and len(leaf) > len(SAFETENSORS_SUFFIX)


def single_file_load_allowed(repo_trusted: bool, kind: str, filename: Optional[str]) -> bool:
    """The non-GGUF trust decision shared by the image and video validators. A trusted repo keeps
    every kind it had; an untrusted one gains exactly a ``single_file`` load of a ``.safetensors``
    name. GGUF is not decided here (it was never repo-gated)."""
    if repo_trusted:
        return True
    return kind == "single_file" and is_hub_safetensors_single_file(filename)


def assert_safetensors_file(path: Union[str, os.PathLike]) -> None:
    """Raise ValueError unless ``path`` is a well-formed safetensors container: an 8-byte
    little-endian header length, a JSON object header within the file, and every tensor's
    ``data_offsets`` inside the data section. Header only, no weight bytes are read and nothing is
    deserialised beyond JSON, so it is safe to run on a file from any repo. It runs before the
    loaders so a ``.safetensors`` name over pickle or garbage bytes fails here with a clear message
    instead of somewhere inside a loader."""
    p = Path(path)
    label = p.name
    try:
        size = p.stat().st_size
        with open(p, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("file is shorter than a safetensors header")
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len < 2 or header_len > _MAX_HEADER_BYTES or 8 + header_len > size:
                raise ValueError("header length is out of range")
            raw = handle.read(header_len)
    except OSError as exc:
        raise ValueError(f"'{label}' could not be read as a safetensors checkpoint: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"'{label}' is not a valid safetensors checkpoint: {exc}") from exc
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            f"'{label}' is not a valid safetensors checkpoint: the header is not JSON"
        ) from exc
    if not isinstance(header, dict):
        raise ValueError(
            f"'{label}' is not a valid safetensors checkpoint: the header is not an object"
        )
    data_len = size - 8 - header_len
    tensors = 0
    for name, entry in header.items():
        if name == "__metadata__":
            if entry is not None and not isinstance(entry, dict):
                raise ValueError(
                    f"'{label}' is not a valid safetensors checkpoint: bad __metadata__"
                )
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("dtype"), str)
            or not isinstance(entry.get("shape"), list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_len
        ):
            raise ValueError(
                f"'{label}' is not a valid safetensors checkpoint: tensor '{name}' has a malformed entry"
            )
        tensors += 1
    if tensors == 0:
        raise ValueError(f"'{label}' is not a valid safetensors checkpoint: it holds no tensors")
