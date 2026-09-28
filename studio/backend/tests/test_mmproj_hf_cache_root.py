# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The projector lookup a remote (-hf) load runs over its own HF cache checkout.

#9286: a repo that publishes no projector took the Hub listing's word for it, so a
hand-added file was invisible wherever the user put it. Only this lookup widens past
the weight's own directory; a local selection keeps the root it always had.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models.model_config import (  # noqa: E402
    _hf_cache_repo_dir,
    _hf_cached_local_mmproj,
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


def _hf_repo(tmp_path: Path, repo_name: str = "models--org--Model-GGUF") -> tuple[Path, Path]:
    repo = tmp_path / repo_name
    snapshot = repo / "snapshots" / "deadbeef"
    snapshot.mkdir(parents = True)
    weight = _gguf_with_general(
        snapshot / "model-Q4_K_M.gguf",
        {"general.name": "Model", "general.basename": "Model", "general.architecture": "qwen3vl"},
    )
    return repo, weight


def _projector(path: Path, basename: str | None = None) -> Path:
    fields = {"general.type": "mmproj", "general.architecture": "qwen3vl"}
    if basename is not None:
        fields["general.basename"] = basename
    return _gguf_with_general(path, fields)


def test_the_repo_dir_is_named_from_a_cached_weight(tmp_path):
    repo, weight = _hf_repo(tmp_path)

    assert _hf_cache_repo_dir(str(weight)) == str(repo)


def test_a_quant_subdir_still_names_the_repo_dir(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    quant_dir = weight.parent / "Q4_K_M"
    quant_dir.mkdir()
    weight = weight.rename(quant_dir / weight.name)

    assert _hf_cache_repo_dir(str(weight)) == str(repo)


def test_a_plain_layout_names_no_repo_dir(tmp_path):
    model_dir = tmp_path / "MyModel"
    model_dir.mkdir()
    weight = _gguf_with_general(
        model_dir / "model.gguf", {"general.name": "Model", "general.architecture": "qwen3vl"}
    )

    assert _hf_cache_repo_dir(str(weight)) is None
    # And the lookup degrades to the weight's own directory rather than refusing.
    projector = _projector(model_dir / "mmproj-F16.gguf")
    assert _hf_cached_local_mmproj(str(weight)) == str(projector.resolve())


def test_a_projector_beside_the_weight_is_found(tmp_path):
    """Where Studio's own Reveal in Folder lands, so the likeliest placement."""
    _repo, weight = _hf_repo(tmp_path)
    projector = _projector(weight.parent / "mmproj-kquant.gguf")

    assert _hf_cached_local_mmproj(str(weight)) == str(projector.resolve())


def test_a_projector_at_the_repo_root_is_found(tmp_path):
    """The snapshot dir is a hex sha, so browsing to "the model folder" often stops
    one or two levels up."""
    repo, weight = _hf_repo(tmp_path)
    projector = _projector(repo / "mmproj-kquant.gguf")

    assert _hf_cached_local_mmproj(str(weight)) == str(projector.resolve())


def test_a_projector_in_the_snapshots_container_is_found(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    projector = _projector(repo / "snapshots" / "mmproj-kquant.gguf")

    assert _hf_cached_local_mmproj(str(weight)) == str(projector.resolve())


def test_a_sibling_repo_and_the_cache_root_stay_out_of_reach(tmp_path):
    """The walk stops at the repo the weight came out of. Reaching the cache root would
    let any repo borrow any other repo's projector."""
    _repo, weight = _hf_repo(tmp_path)
    sibling, _sibling_weight = _hf_repo(tmp_path, "models--org--Other-GGUF")
    _projector(sibling / "mmproj-kquant.gguf")
    _projector(tmp_path / "mmproj-kquant.gguf")

    assert _hf_cached_local_mmproj(str(weight)) is None


def test_a_projector_for_another_model_is_refused(tmp_path):
    """Widening the walk widens what can be mispaired, so metadata still decides."""
    repo, weight = _hf_repo(tmp_path)
    _projector(repo / "mmproj-kquant.gguf", basename = "SomeOtherModel")

    assert _hf_cached_local_mmproj(str(weight)) is None


def test_the_snapshots_own_projector_wins_when_the_far_one_cannot_pair(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    _projector(repo / "mmproj-kquant.gguf", basename = "SomeOtherModel")
    near = _projector(weight.parent / "mmproj-F16.gguf", basename = "Model")

    assert _hf_cached_local_mmproj(str(weight)) == str(near.resolve())


def test_an_empty_projector_is_not_named(tmp_path):
    """An interrupted copy is a file llama-server cannot open."""
    repo, weight = _hf_repo(tmp_path)
    (repo / "mmproj-kquant.gguf").write_bytes(b"")

    assert _hf_cached_local_mmproj(str(weight)) is None


def test_a_local_selection_keeps_the_root_it_always_had(tmp_path):
    """The boundary this change deliberately leaves alone. A directory the user picked
    by hand is scanned exactly as before, so a native grant, the drafter walk and the
    cached rows all keep answering what they answered on main."""
    repo, weight = _hf_repo(tmp_path)
    _projector(repo / "mmproj-kquant.gguf")

    root = _local_gguf_companion_search_root(str(weight.parent), str(weight))
    assert root == str(weight.parent)
    assert detect_mmproj_file(str(weight), search_root = root) is None
