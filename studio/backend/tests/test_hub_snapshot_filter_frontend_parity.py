# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards that dataset-size.ts still mirrors snapshot_filters.py.

The backend decides what snapshot_download fetches, the frontend is a hand-ported copy
deciding the size shown and the progress denominator, and nothing is shared between them,
so drift is silent: an advertised size that never matches disk, or a bar stuck below 100%.
"""

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

# Each backend fnmatch pattern and the fragment its frontend regex counterpart must contain.
_TS_COUNTERPARTS = {
    "original/*": "original",
    "metal/*": "metal",
    "coreml/*": "coreml",
    "pytorch_model*.bin": r"pytorch_model.*\.bin",
    "pytorch_model.bin.index.json": r"pytorch_model\.bin",
    "tf_model*.h5": r"tf_model.*\.h5",
    "tf_model.h5.index.json": r"tf_model\.h5",
    "flax_model*.msgpack": r"flax_model.*\.msgpack",
    "flax_model.msgpack.index.json": r"flax_model\.msgpack",
    "rust_model.ot": r"rust_model\.ot",
}


def _ts_source() -> str:
    return _DATASET_SIZE_TS.read_text(encoding = "utf-8")


def test_every_backend_duplicate_pattern_has_a_frontend_counterpart():
    # Keyed off the live tuple, so a pattern added to the backend alone fails here.
    assert set(DUPLICATE_WEIGHT_FORMAT_PATTERNS) == set(_TS_COUNTERPARTS), (
        "DUPLICATE_WEIGHT_FORMAT_PATTERNS changed; mirror it in dataset-size.ts's "
        "DUPLICATE_WEIGHT_FORMAT_RE and record the counterpart here"
    )
    src = _ts_source()
    for pattern in DUPLICATE_WEIGHT_FORMAT_PATTERNS:
        assert _TS_COUNTERPARTS[pattern] in src, f"{pattern} has no counterpart in dataset-size.ts"
    assert r"\.index\.json$" in src


def test_both_gates_spell_the_shard_number_with_ascii_digits():
    # Python's \d matches non-ASCII digits and JavaScript's does not, so \d here would have
    # the two gates disagree about whether a repo ships a root checkpoint at all.
    assert r"model([-_][0-9]+-of-[0-9]+)?\.safetensors" in (
        Path(__file__).resolve().parents[1] / "hub" / "utils" / "snapshot_filters.py"
    ).read_text(encoding = "utf-8")
    assert r"^model([-_][0-9]+-of-[0-9]+)?\.safetensors$" in _ts_source()


def test_the_frontend_still_gates_the_duplicate_rules_on_root_safetensors():
    # A repo without a root checkpoint must download exactly as before on both sides.
    src = _ts_source()
    assert "function shipsRootSafetensors(" in src
    assert "skipDuplicateFormats && DUPLICATE_WEIGHT_FORMAT_RE.test(filename)" in src
