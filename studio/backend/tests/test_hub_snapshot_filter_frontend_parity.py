# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeps dataset-size.ts in step with snapshot_filters.py, which it hand-ports; drift is silent."""

from __future__ import annotations

from pathlib import Path

from hub.utils.snapshot_filters import DUPLICATE_WEIGHT_FORMAT_PATTERNS

_DATASET_SIZE_TS = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "features"
    / "hub"
    / "lib"
    / "dataset-size.ts"
)

_TS_COUNTERPARTS = {
    "original/*": "original",
    "metal/*": "metal",
    "coreml/*": "coreml",
    "tf_model*.h5": r"tf_model.*\.h5",
    "tf_model.h5.index.json": r"tf_model\.h5",
    "flax_model*.msgpack": r"flax_model.*\.msgpack",
    "flax_model.msgpack.index.json": r"flax_model\.msgpack",
    "rust_model.ot": r"rust_model\.ot",
}


def _ts_source() -> str:
    return _DATASET_SIZE_TS.read_text(encoding = "utf-8")


def test_every_backend_duplicate_pattern_has_a_frontend_counterpart():
    assert set(DUPLICATE_WEIGHT_FORMAT_PATTERNS) == set(_TS_COUNTERPARTS), (
        "DUPLICATE_WEIGHT_FORMAT_PATTERNS changed; mirror it in dataset-size.ts's "
        "DUPLICATE_WEIGHT_FORMAT_RE and record the counterpart here"
    )
    src = _ts_source()
    for pattern in DUPLICATE_WEIGHT_FORMAT_PATTERNS:
        assert _TS_COUNTERPARTS[pattern] in src, f"{pattern} has no counterpart in dataset-size.ts"
    assert r"\.index\.json$" in src


def test_both_gates_spell_the_shard_number_with_ascii_digits():
    # Python's \d matches non-ASCII digits but JS's does not, so both use [0-9].
    assert r"model[-_][0-9]+-of-[0-9]+\.safetensors" in (
        Path(__file__).resolve().parents[1] / "hub" / "utils" / "snapshot_filters.py"
    ).read_text(encoding = "utf-8")
    assert r"^model[-_][0-9]+-of-[0-9]+\.safetensors$" in _ts_source()


def test_both_gates_require_the_index_before_trusting_numbered_shards():
    assert "model.safetensors.index.json" in _ts_source()
    assert "SAFETENSORS_INDEX" in _ts_source()


def test_the_torch_bin_family_is_resolved_per_repo_on_both_sides():
    # Globbing pytorch_model*.bin dropped dtype variants; resolve against the repo file list.
    src = _ts_source()
    assert "redundantTorchBinFiles(" in src
    assert "variantShipsAsSafetensors(" in src
    assert not any(p.startswith("pytorch_model") for p in DUPLICATE_WEIGHT_FORMAT_PATTERNS)
    assert r"pytorch_model.*\.bin" not in src


def test_the_frontend_still_gates_the_duplicate_rules_on_root_safetensors():
    src = _ts_source()
    assert "function shipsRootSafetensors(" in src
    assert "skipDuplicateFormats &&" in src
    assert "DUPLICATE_WEIGHT_FORMAT_RE.test(filename) || redundantBins.has(filename)" in src
