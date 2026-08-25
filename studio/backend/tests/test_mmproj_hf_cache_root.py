# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test HF cache companion search roots."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

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
from routes.models import _repo_root_has_mmproj  # noqa: E402
from routes.inference import (  # noqa: E402
    _native_drafter_accept,
    _validate_native_gguf_companion,
    _validate_native_gguf_projector,
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

    # The two directories the widening adds, once for the snapshot, not once per variant.
    assert snapshot_has_gguf_projector(weight.parent) is False
    assert calls == [str(repo), str(weight.parent.parent)]


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


def test_a_nested_checkpoint_dir_keeps_its_snapshots_projector_ahead(tmp_path):
    """``snapshots/<sha>/distilled/`` is not a quant name, so the selected root stops
    inside it. The snapshot is still its own boundary, ahead of the repo dir."""
    repo, weight = _hf_repo(tmp_path)
    nested = weight.parent / "distilled"
    nested.mkdir()
    weight = weight.rename(nested / "model-Q6_K.gguf")
    shipped = _gguf_with_general(
        nested.parent / "mmproj-F16.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )
    hand_added = _gguf_with_general(
        repo / "model-Q6_K-mmproj.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )

    # One pass over both would take the hand-added file on the shared-prefix tie-break.
    assert detect_mmproj_file(str(weight), search_root = str(repo)) == str(hand_added.resolve())

    assert _detect_local_mmproj(str(nested), str(weight)) == str(shipped.resolve())


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


def test_a_split_projector_keeps_its_shard_names_and_starts_at_shard_one(tmp_path):
    """llama-server resolves the siblings from the path it is handed, so a symlinked
    set must come back on its own names rather than on the blob it points at, and on
    shard 1 whichever shard ranked first."""
    repo, weight = _hf_repo(tmp_path)
    blobs = repo / "blobs"
    blobs.mkdir()
    shards = []
    for index in (1, 2):
        blob = _gguf_with_general(
            blobs / f"projector-{index}",
            {"general.type": "mmproj", "general.architecture": "qwen3vl"},
        )
        shard = repo / f"mmproj-F16-0000{index}-of-00002.gguf"
        try:
            shard.symlink_to(blob)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        shards.append(shard)

    found = _detect_local_mmproj(str(weight.parent), str(weight))

    assert found == str(shards[0].absolute())


def test_a_projector_in_the_snapshots_container_reaches_the_row(tmp_path):
    """The widened walk passes through models--<repo>/snapshots/ on its way up, so the
    row's presence gate has to look there too or it answers no vision for a load that
    opens one."""
    _, weight = _hf_repo(tmp_path)
    _clip_projector(weight.parent.parent / "mmproj-F16.gguf", "clip.has_vision_encoder")

    assert _detect_local_mmproj(str(weight.parent), str(weight)) is not None
    assert snapshot_has_gguf_projector(weight.parent) is True
    # The route-level early return has to name the same directories as the walk.
    assert _repo_root_has_mmproj(SimpleNamespace(repo_path = weight.parent.parent.parent)) is True


def test_a_half_split_snapshot_projector_does_not_claim_vision_on_the_row(tmp_path):
    """The variant lister counts each file on its own, so one shard of a split set reads
    as vision support the load then refuses."""
    _, weight = _hf_repo(tmp_path)
    _clip_projector(weight.parent / "mmproj-F16-00001-of-00002.gguf", "clip.has_vision_encoder")

    assert _detect_local_mmproj(str(weight.parent), str(weight)) is None
    assert snapshot_has_gguf_projector(weight.parent) is False

    _clip_projector(weight.parent / "mmproj-F16-00002-of-00002.gguf", "clip.has_vision_encoder")

    assert snapshot_has_gguf_projector(weight.parent) is True


def test_an_upper_cased_cache_dir_is_still_an_hf_layout(tmp_path):
    """_iter_hf_cache_snapshots and the pre-download bound match a cache dir
    case-insensitively, so the walk has to widen inside the same directories."""
    repo = tmp_path / "MODELS--ORG--MODEL-GGUF"
    snapshot = repo / "SNAPSHOTS" / "deadbeef"
    snapshot.mkdir(parents = True)
    weight = _gguf_with_general(
        snapshot / "model-Q4_K_M.gguf",
        {"general.name": "Model", "general.architecture": "qwen3vl"},
    )
    projector = _gguf_with_general(
        repo / "mmproj-kquant.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )

    assert _local_gguf_companion_search_root(str(snapshot), str(weight)) == str(repo)
    assert _detect_local_mmproj(str(snapshot), str(weight)) == str(projector.resolve())


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


def test_native_load_rejects_a_split_projector_with_an_ungranted_shard(tmp_path):
    """llama-server opens the siblings implicitly, so validating only the launch path
    would let shard 2 be a symlink out of the granted directory."""
    _, weight = _hf_repo(tmp_path)
    first = _gguf_with_general(
        weight.parent / "mmproj-kquant-00001-of-00002.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )
    outside = _gguf_with_general(
        tmp_path / "outside.gguf",
        {"general.type": "mmproj", "general.architecture": "qwen3vl"},
    )
    try:
        (weight.parent / "mmproj-kquant-00002-of-00002.gguf").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    # Discovery still offers shard 1, which is a regular file inside the grant, and the
    # launch path on its own passes every native rule.
    assert _detect_local_mmproj(str(weight.parent), str(weight)) == str(first.absolute())
    _validate_native_gguf_companion(str(first.absolute()), str(weight), "vision companion")

    with pytest.raises(HTTPException) as excinfo:
        _validate_native_gguf_projector(str(first.absolute()), str(weight))
    assert excinfo.value.status_code == 400
    # And the pre-read filter the widened pass uses refuses the set for the same reason.
    assert _native_drafter_accept(str(first.absolute()), str(weight), "mmproj", "") is False


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
