# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test HF cache companion search roots."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models.model_config import (  # noqa: E402
    ModelConfig,
    _detect_local_mmproj,
    _local_gguf_companion_search_root,
    detect_mmproj_file,
)
from hub.utils.inventory_scan import snapshot_has_gguf_projector  # noqa: E402
from routes.inference import (  # noqa: E402
    _native_drafter_accept,
    _validate_native_gguf_companion,
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


def _clip_projector(path: Path, *flags: str) -> Path:
    """A projector GGUF declaring the given ``clip.has_*_encoder`` bools."""
    body = b"".join(
        struct.pack("<Q", len(flag))
        + flag.encode()
        + struct.pack("<I", 7)
        + struct.pack("<?", True)
        for flag in flags
    )
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, len(flags)) + body)
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


def test_inventory_rejects_a_repo_root_projector_with_mismatched_metadata(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    _gguf_with_general(
        weight,
        {
            "general.name": "Model",
            "general.architecture": "qwen3vl",
            "general.base_model.0.repo_url": "https://huggingface.co/org/Model",
        },
    )
    projector = repo / "mmproj-F16.gguf"
    _gguf_with_general(
        projector,
        {
            "general.type": "mmproj",
            "general.architecture": "qwen3vl",
            "general.base_model.0.repo_url": "https://huggingface.co/other/Model",
        },
    )

    assert snapshot_has_gguf_projector(weight.parent) is False

    _gguf_with_general(
        projector,
        {
            "general.type": "mmproj",
            "general.architecture": "qwen3vl",
            "general.base_model.0.repo_url": "https://huggingface.co/org/Model",
        },
    )
    assert snapshot_has_gguf_projector(weight.parent) is True


def test_inventory_does_not_rescan_each_variant_without_a_projector(tmp_path, monkeypatch):
    repo, weight = _hf_repo(tmp_path)
    for quant in ("Q5_K_M", "Q8_0"):
        _gguf_with_general(
            weight.with_name(f"model-{quant}.gguf"),
            {"general.name": "Model", "general.architecture": "qwen3vl"},
        )

    calls = []

    def traced_detect(path, *args, **kwargs):
        calls.append(path)
        return detect_mmproj_file(path, *args, **kwargs)

    monkeypatch.setattr("utils.models.model_config.detect_mmproj_file", traced_detect)

    assert snapshot_has_gguf_projector(weight.parent) is False
    assert calls == [str(repo)]


def test_the_snapshots_own_projector_is_not_shadowed_by_the_repo_root(tmp_path):
    """The repo root is a fallback, not a wider pool: ranking both together lets a
    hand-added file win the shared-prefix tie-break over the shipped projector."""
    repo, weight = _hf_repo(tmp_path)
    shipped = _gguf_with_general(
        weight.parent / "mmproj-F16.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )
    hand_added = _gguf_with_general(
        repo / "model-Q4_K_M-mmproj.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )

    # One widened pass over both directories would take the hand-added file.
    assert detect_mmproj_file(str(weight), search_root = str(repo)) == str(hand_added.resolve())

    assert _detect_local_mmproj(str(weight.parent), str(weight)) == str(shipped.resolve())


def test_the_repo_root_still_answers_when_the_snapshot_has_none(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    hand_added = _gguf_with_general(
        repo / "mmproj-kquant.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )

    assert _detect_local_mmproj(str(weight.parent), str(weight)) == str(hand_added.resolve())


def test_an_audio_only_snapshot_projector_keeps_the_row_off_vision(tmp_path):
    """The row has to answer what the load will open. Snapshot first means the audio
    encoder wins, so advertising the repo root's image tower would be a button the
    launch never honours."""
    repo, weight = _hf_repo(tmp_path)
    _clip_projector(weight.parent / "mmproj-audio.gguf", "clip.has_audio_encoder")
    _clip_projector(repo / "mmproj-F16.gguf", "clip.has_vision_encoder")

    assert snapshot_has_gguf_projector(weight.parent) is False


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


def test_hf_snapshot_drafters_stay_in_scope_when_mmproj_walk_reaches_repo_root(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    quant_dir = weight.parent / "Q4_K_M"
    quant_dir.mkdir()
    weight = weight.rename(quant_dir / "Model-Q4_K_M.gguf")
    mtp = weight.parent.parent / "mtp-Model-Q8_0.gguf"
    mtp.write_bytes(b"draft")
    dspark = weight.parent.parent / "dspark-Model-Q8_0.gguf"
    dspark.write_bytes(b"draft")
    dflash = _gguf_with_general(
        weight.parent.parent / "dflash-kquant.gguf", {"general.architecture": "dflash"}
    )

    projector_root = _local_gguf_companion_search_root(str(quant_dir), str(weight))
    drafter_root = _local_gguf_companion_search_root(
        str(quant_dir), str(weight), include_hf_repo_root = False
    )
    assert projector_root == str(repo)
    assert drafter_root == str(weight.parent.parent)
    config = ModelConfig.from_identifier(str(quant_dir))
    assert config is not None
    assert config.gguf_mtp_file == str(mtp.resolve())
    assert config.gguf_dspark_file == str(dspark.resolve())
    assert config.gguf_dflash_file == str(dflash.resolve())


def test_models_prefix_without_a_snapshot_layout_does_not_widen_the_walk(tmp_path):
    outer = tmp_path / "models--ordinary-folder"
    model_dir = outer / "nested" / "Model"
    weight = _gguf_with_general(
        model_dir / "Model-Q4_K_M.gguf",
        {"general.name": "Model", "general.architecture": "qwen3vl"},
    )
    _gguf_with_general(
        outer / "mmproj-F16.gguf", {"general.type": "mmproj", "general.architecture": "qwen3vl"}
    )

    search_root = _local_gguf_companion_search_root(str(model_dir), str(weight))
    assert search_root == str(model_dir)
    assert detect_mmproj_file(str(weight), search_root = search_root) is None


def test_native_load_drops_a_repo_root_projector_outside_the_file_grant(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    _gguf_with_general(
        repo / "mmproj-kquant.gguf", {"general.type": "mmproj", "general.architecture": "qwen3vl"}
    )

    config = ModelConfig.from_identifier(str(weight), drafter_accept = _native_drafter_accept)
    assert config is not None
    assert config.gguf_mmproj_file is None
    assert config.is_vision is False


def test_native_load_still_refuses_a_projector_above_the_selected_quant(tmp_path):
    """The snapshot pass answers exactly as it did before the widening. A projector one
    level up is still discovered and still refused at the intent, rather than being
    quietly dropped by the boundary filter the repo-root pass adds."""
    _, weight = _hf_repo(tmp_path)
    quant_dir = weight.parent / "Q4_K_M"
    quant_dir.mkdir()
    weight = weight.rename(quant_dir / weight.name)
    projector = _gguf_with_general(
        weight.parent.parent / "mmproj-kquant.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )

    config = ModelConfig.from_identifier(str(weight), drafter_accept = _native_drafter_accept)
    assert config is not None
    assert config.gguf_mmproj_file == str(projector.resolve())

    with pytest.raises(HTTPException) as excinfo:
        _validate_native_gguf_companion(
            config.gguf_mmproj_file, config.gguf_file, "vision companion"
        )
    assert excinfo.value.status_code == 400


def test_native_load_keeps_a_projector_beside_the_selected_file(tmp_path):
    _, weight = _hf_repo(tmp_path)
    projector = _gguf_with_general(
        weight.parent / "mmproj-kquant.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )

    config = ModelConfig.from_identifier(str(weight), drafter_accept = _native_drafter_accept)
    assert config is not None
    assert config.gguf_mmproj_file == str(projector.resolve())
    assert config.is_vision is True
