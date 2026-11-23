# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth/Inkling-2-Mini-GGUF publishes its original DSpark drafter as dspark-Inkling-2-Mini-Q8_0_Old.gguf so the
root name can carry the mainline-layout drafter: Studio picks the _Old copy, and nothing else changes."""

from __future__ import annotations

from core.inference.llama_cpp import _pick_dspark
from utils.models.drafters.common import _drafter_pairing_stem
from utils.models.model_config import detect_dspark_file, dspark_preference_key

ROOT = "dspark-Inkling-2-Mini-Q8_0.gguf"
OLD = "dspark-Inkling-2-Mini-Q8_0_Old.gguf"


def test_the_listing_pick_prefers_the_old_inkling2_mini_drafter():
    listing = [
        "Inkling-2-Mini-UD-Q4_K_XL.gguf",
        ROOT,
        OLD,
        "dspark/dspark-Inkling-2-Mini-BF16.gguf",
        "mmproj-F16.gguf",
    ]
    assert _pick_dspark(listing) == OLD
    assert _pick_dspark(list(reversed(listing))) == OLD


def test_without_the_old_copy_the_pick_is_unchanged():
    assert _pick_dspark([ROOT, "dspark/dspark-Inkling-2-Mini-BF16.gguf"]) == ROOT


def test_other_models_rank_as_before():
    names = ["dspark-Foo-BF16.gguf", "dspark-Foo-Q8_0.gguf", "dspark-Foo-Q8_0_Old.gguf"]
    # Only the Inkling-2-Mini file name is pinned; any other _Old file sorts by precision and name as before.
    assert _pick_dspark(names) == "dspark-Foo-Q8_0.gguf"
    assert dspark_preference_key(OLD) < dspark_preference_key(ROOT)


def test_the_old_name_pairs_with_the_inkling2_mini_family():
    assert _drafter_pairing_stem(OLD, kind = "dspark") == _drafter_pairing_stem(ROOT, kind = "dspark")
    assert _drafter_pairing_stem(OLD, kind = "dspark") == "inkling-2-mini"


def test_a_local_folder_with_both_drafters_loads_the_old_one(tmp_path):
    weight = tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
    weight.write_bytes(b"GGUF" + b"\0" * 60)
    for name in (ROOT, OLD):
        (tmp_path / name).write_bytes(b"GGUF" + b"\0" * 60)
    assert detect_dspark_file(str(weight)) == str(tmp_path / OLD)


def test_a_local_folder_with_only_the_root_drafter_still_finds_it(tmp_path):
    weight = tmp_path / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
    weight.write_bytes(b"GGUF" + b"\0" * 60)
    (tmp_path / ROOT).write_bytes(b"GGUF" + b"\0" * 60)
    assert detect_dspark_file(str(weight)) == str(tmp_path / ROOT)
