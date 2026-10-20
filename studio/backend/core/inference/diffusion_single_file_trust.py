# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-file trust for single-file diffusion checkpoints from any Hugging Face repo.

The repo allowlists guard ``from_pretrained`` (pickled weights, ``trust_remote_code``). A lone
``.safetensors`` file has neither risk: only that file is downloaded, it is parsed without
unpickling, and every config and companion still comes from the family's trusted base repo.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Optional, Union

SAFETENSORS_SUFFIX = ".safetensors"

# Same cap as the safetensors reference parser: a crafted length prefix cannot force a huge read.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def is_hub_safetensors_single_file(filename: Optional[str]) -> bool:
    """A repo-relative ``.safetensors`` name: no absolute path, ``..``, backslash or NUL, and the final
    suffix is ``.safetensors`` (``x.safetensors.pt`` never qualifies)."""
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
    # Case-sensitive like diffusers' load_state_dict: x.SafeTensors would fall through to torch.load.
    return leaf.endswith(SAFETENSORS_SUFFIX) and len(leaf) > len(SAFETENSORS_SUFFIX)


def single_file_load_allowed(repo_trusted: bool, kind: str, filename: Optional[str]) -> bool:
    """Non-GGUF trust decision: a trusted repo keeps every kind; an untrusted one gains only a
    ``single_file`` load of a ``.safetensors`` name."""
    if repo_trusted:
        return True
    return kind == "single_file" and is_hub_safetensors_single_file(filename)


def assert_safetensors_file(path: Union[str, os.PathLike]) -> None:
    """Raise ValueError unless ``path`` is a well-formed safetensors container (header only: length,
    JSON object, every ``data_offsets`` inside the data section). Runs before any loader opens it."""
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
