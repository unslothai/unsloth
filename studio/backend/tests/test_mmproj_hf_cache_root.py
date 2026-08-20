# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF-cache companion search root (#9286).

A model downloaded by Studio lives under
``models--<repo>/snapshots/<sha>/`` (optionally a quant subdir deeper). A user
adding a projector by hand drops it wherever their file manager showed the
model — most often the ``models--<repo>`` root — and the companion walk used to
stop at the snapshot dir, so vision never engaged even though the same flat
folder works from LM Studio's layout. The search root now stops at the
``models--<repo>`` dir; the cache root itself stays out, so a sibling repo's
projector is invisible.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models.model_config import (  # noqa: E402
    _local_gguf_companion_search_root,
    detect_mmproj_file,
)

_GGUF_MAGIC = 0x46554747


def _gguf_with_general(path: Path, fields: dict) -> Path:
    body = b""
    for k, v in fields.items():
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        body += struct.pack("<Q", len(kb)) + kb
        body += struct.pack("<I", 8)
        body += struct.pack("<Q", len(vb)) + vb
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, len(fields))
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(header + body)
    return path


def _hf_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "models--org--Model-GGUF"
    snapshot = repo / "snapshots" / "deadbeef"
    snapshot.mkdir(parents = True)
    weight = _gguf_with_general(
        snapshot / "model-Q4_K_M.gguf", {"general.name": "Model", "general.architecture": "qwen3vl"}
    )
    return repo, weight


def test_search_root_stops_at_the_hf_repo_dir(tmp_path):
    repo, weight = _hf_repo(tmp_path)

    root = _local_gguf_companion_search_root(str(weight.parent), str(weight))
    assert root == str(repo)


def test_search_root_from_a_quant_subdir_also_reaches_the_repo_dir(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    quant_dir = weight.parent / "Q4_K_M"
    quant_dir.mkdir()
    moved = weight.parent / "old.gguf"
    weight = weight.rename(quant_dir / weight.name)
    moved.write_bytes(b"")

    root = _local_gguf_companion_search_root(str(quant_dir), str(weight))
    assert root == str(repo)


def test_plain_layout_keeps_its_existing_root(tmp_path):
    model_dir = tmp_path / "MyModel"
    model_dir.mkdir()
    weight = _gguf_with_general(
        model_dir / "model.gguf", {"general.name": "Model", "general.architecture": "qwen3vl"}
    )

    root = _local_gguf_companion_search_root(str(model_dir), str(weight))
    assert root == str(model_dir)


def test_projector_at_the_hf_repo_root_is_found(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    projector = _gguf_with_general(
        repo / "mmproj-kquant.gguf", {"general.type": "mmproj", "general.architecture": "qwen3vl"}
    )

    found = detect_mmproj_file(
        str(weight), search_root = _local_gguf_companion_search_root(str(weight.parent), str(weight))
    )
    assert found is not None and Path(found).resolve() == projector.resolve()


def test_a_sibling_repo_s_projector_stays_invisible(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    other = tmp_path / "models--other--Vision-GGUF"
    other.mkdir()
    _gguf_with_general(
        other / "mmproj-F16.gguf", {"general.type": "mmproj", "general.architecture": "otherarch"}
    )

    found = detect_mmproj_file(
        str(weight), search_root = _local_gguf_companion_search_root(str(weight.parent), str(weight))
    )
    assert found is None
