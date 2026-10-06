# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an already-installed Unsloth Studio does on its first load after this update.

The auto tensor-split fallback changes what a load DOES with a ``tensor_split``
it is handed. The question an existing install asks is narrower and more
important: can anything already on my disk start feeding it one?

Saved per-model settings live in one row of ``app_settings``
(``openai_api_auto_switch_overrides``), and the only things that ever reach a
load from there are what ``normalize_model_override`` allow-lists on the way in
and what ``model_override_load_kwargs`` emits on the way out. So the upgrade
question is answerable exactly, without a database: put a saved override
through both and see whether a ratio can come out the other side.

Forwards compatibility is the same argument run backwards. The change persists
nothing new, so a settings row written by this version is byte-identical to one
written before it, and an older install reading it finds nothing it does not
already understand.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest  # noqa: E402

from utils.openai_auto_switch_settings import (  # noqa: E402
    model_override_load_kwargs,
    normalize_model_override,
)

# A saved override from an existing install, using every placement key the
# allow-list does carry. This is the shape that replays on the first load after
# an update.
LEGACY_OVERRIDE = {
    "gpu_memory_mode": "manual",
    "gpu_layers": 99,
    "n_cpu_moe": 4,
    "tensor_parallel": True,
    "gpu_ids": [0, 1],
    "gpu_index_kind": "cuda",
    "max_seq_length": 8192,
    "kv_cache_dtype": "q8_0",
    "n_parallel": 2,
    "llama_extra_args": ["--flash-attn", "on"],
}


def test_a_saved_setting_can_never_carry_a_tensor_split():
    """The allow-list is the whole attack surface for this change. If a ratio
    cannot be stored, no existing install can begin sending one at upgrade."""
    stored = normalize_model_override(dict(LEGACY_OVERRIDE))
    assert "tensor_split" not in stored


@pytest.mark.parametrize(
    "smuggled",
    [
        {"tensor_split": [3, 1]},
        {"tensor_split": None},
        {"tensorSplit": [3, 1]},
        {"tensor_split": "3,1"},
    ],
)
def test_a_ratio_written_into_the_settings_row_by_hand_is_dropped(smuggled):
    """Not hypothetical: the row is JSON in a sqlite table a user can edit, and
    a future version could try to add the key. Either way it must not reach a
    load through this door without the allow-list being changed deliberately."""
    stored = normalize_model_override({**LEGACY_OVERRIDE, **smuggled})
    assert "tensor_split" not in stored
    assert "tensorSplit" not in stored


@pytest.mark.parametrize("is_gguf", [True, False])
def test_the_replayed_load_asks_for_no_ratio(is_gguf):
    """The other end of the same path: what the replay actually sends."""
    stored = normalize_model_override(dict(LEGACY_OVERRIDE))
    kwargs = model_override_load_kwargs(stored, is_gguf = is_gguf)
    assert "tensor_split" not in kwargs


def test_the_replayed_load_is_unchanged_in_every_other_respect():
    """A pin against silent drift. If a later edit to this area changes what an
    existing install replays, this says so in terms a reader can check against
    the saved row above rather than against a golden blob."""
    stored = normalize_model_override(dict(LEGACY_OVERRIDE))
    kwargs = model_override_load_kwargs(stored, is_gguf = True)

    assert kwargs["gpu_memory_mode"] == "manual"
    assert kwargs["gpu_layers"] == 99
    assert kwargs["n_cpu_moe"] == 4
    assert kwargs["tensor_parallel"] is True
    assert kwargs["gpu_ids"] == [0, 1]
    assert kwargs["max_seq_length"] == 8192
    assert kwargs["cache_type_kv"] == "q8_0"
    assert kwargs["n_parallel"] == 2
    assert kwargs["llama_extra_args"] == ["--flash-attn", "on"]


def test_an_override_from_a_newer_version_loses_nothing_an_older_one_kept():
    """Forwards compatibility. Round-tripping a stored row through the
    normalizer must be stable, or an install that upgrades and then rolls back
    finds its settings rewritten."""
    once = normalize_model_override(dict(LEGACY_OVERRIDE))
    twice = normalize_model_override(dict(once))
    assert once == twice
