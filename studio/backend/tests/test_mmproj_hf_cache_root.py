# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test HF cache companion search roots."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models.model_config import (  # noqa: E402
    ModelConfig,
    _local_gguf_companion_search_root,
    detect_mmproj_file,
)
from hub.utils.inventory_scan import snapshot_has_gguf_projector  # noqa: E402
from routes.inference import _native_drafter_accept  # noqa: E402

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


def test_precision_preference_cannot_override_stronger_metadata_pairing(tmp_path):
    repo, weight = _hf_repo(tmp_path)
    _gguf_with_general(
        weight,
        {"general.base_model.0.repo_url": "https://huggingface.co/org/Model"},
    )
    generic_f16 = _gguf_with_general(repo / "mmproj-F16.gguf", {"general.type": "mmproj"})
    matching_bf16 = _gguf_with_general(
        repo / "model-mmproj-BF16.gguf",
        {
            "general.type": "mmproj",
            "general.base_model.0.repo_url": "https://huggingface.co/org/Model",
        },
    )

    found = detect_mmproj_file(
        str(weight),
        search_root = str(repo),
        prefer = lambda candidates: next(
            (candidate for candidate in candidates if candidate == str(generic_f16.resolve())),
            candidates[0],
        ),
    )

    assert found == str(matching_bf16.resolve())


def test_symlinked_split_projector_requires_the_whole_snapshot_set(tmp_path):
    _, weight = _hf_repo(tmp_path)
    blobs = weight.parent.parent.parent / "blobs"
    blobs.mkdir()
    second_blob = blobs / "projector-2"
    second_blob.write_bytes(b"")
    (weight.parent / "mmproj-F16-00002-of-00002.gguf").symlink_to(second_blob)

    assert detect_mmproj_file(str(weight), search_root = str(weight.parent)) is None

    blob = blobs / "projector"
    _gguf_with_general(blob, {"general.type": "mmproj"})
    first = weight.parent / "mmproj-F16-00001-of-00002.gguf"
    first.symlink_to(blob)

    assert detect_mmproj_file(str(weight), search_root = str(weight.parent)) is None

    _gguf_with_general(second_blob, {"general.type": "mmproj"})

    assert detect_mmproj_file(str(weight), search_root = str(weight.parent)) == str(first.absolute())


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


def test_native_load_rejects_a_split_projector_with_an_ungranted_shard(tmp_path):
    _, weight = _hf_repo(tmp_path)
    _gguf_with_general(
        weight.parent / "mmproj-kquant-00001-of-00002.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )
    outside = _gguf_with_general(
        tmp_path / "outside.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )
    (weight.parent / "mmproj-kquant-00002-of-00002.gguf").symlink_to(outside)

    config = ModelConfig.from_identifier(str(weight), drafter_accept = _native_drafter_accept)

    assert config is not None
    assert config.gguf_mmproj_file is None
    assert config.is_vision is False
