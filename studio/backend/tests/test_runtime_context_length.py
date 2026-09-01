# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the shared runtime context-length resolver."""

import sys
from pathlib import Path
from types import SimpleNamespace

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.runtime_context import runtime_context_length


def test_attached_window_wins():
    model = SimpleNamespace(
        max_seq_length = 8192,
        args = SimpleNamespace(max_position_embeddings = 40960),
    )
    assert runtime_context_length(model, 4096) == 8192


def test_requested_length_beats_the_declared_window():
    model = SimpleNamespace(args = SimpleNamespace(max_position_embeddings = 40960))
    assert runtime_context_length(model, 4096) == 4096


def test_mlx_model_falls_back_to_its_declared_window():
    model = SimpleNamespace(args = SimpleNamespace(max_position_embeddings = 40960))
    assert runtime_context_length(model, 0) == 40960


def test_declared_window_read_from_config():
    model = SimpleNamespace(config = SimpleNamespace(max_position_embeddings = 32768))
    assert runtime_context_length(model, None) == 32768


def test_declared_window_read_from_a_mapping_config():
    model = SimpleNamespace(config = {"max_position_embeddings": 16384})
    assert runtime_context_length(model, 0) == 16384


def test_declared_window_read_from_a_multimodal_text_config():
    model = SimpleNamespace(
        config = SimpleNamespace(text_config = SimpleNamespace(max_position_embeddings = 131072)),
    )
    assert runtime_context_length(model, 0) == 131072


def test_a_module_that_is_itself_a_dict_reads_its_attributes():
    class _MlxLikeModule(dict):
        def __init__(self, args):
            super().__init__({"layers": []})
            self.args = args

    model = _MlxLikeModule(SimpleNamespace(max_position_embeddings = 40960))
    assert runtime_context_length(model, 0) == 40960


def test_declared_window_read_from_an_underscore_config():
    """mlx exposes a checkpoint's config under either name, so both are scanned."""
    model = SimpleNamespace(_config = {"max_position_embeddings": 40960})
    assert runtime_context_length(model, 0) == 40960


def test_a_placeholder_outer_window_does_not_shadow_the_real_one():
    """A wrapper carrying 0 / "n/a" must not hide the window on the config below it."""
    model = SimpleNamespace(
        max_position_embeddings = 0,
        args = SimpleNamespace(max_position_embeddings = 40960),
    )
    assert runtime_context_length(model, 0) == 40960

    wrapper = SimpleNamespace(
        config = SimpleNamespace(max_position_embeddings = "n/a"),
        args = SimpleNamespace(max_position_embeddings = 32768),
    )
    assert runtime_context_length(wrapper, 0) == 32768


def test_a_config_that_refuses_to_be_read_leaves_the_window_unknown():
    """A resolver that cannot answer must not be able to fail the load.

    Wrapper objects nobody controls (PEFT, accelerate, a config behind a property
    that wants a file) can raise on attribute access, and this runs on every load.
    """

    class _RefusesToLoad:
        @property
        def config(self):
            raise RuntimeError("config unavailable")

    assert runtime_context_length(_RefusesToLoad(), 0) is None


def test_an_attached_window_wins_without_touching_the_config():
    """The declared window is consulted last and lazily.

    A transformers load always has a requested length, so it must never reach the
    model's config -- which is what makes a raising config harmless there.
    """
    touched = []

    class _RecordsConfigReads:
        max_seq_length = 4096

        @property
        def config(self):
            touched.append("config")
            raise RuntimeError("config unavailable")

    assert runtime_context_length(_RecordsConfigReads(), 2048) == 4096
    assert touched == []


def test_no_window_anywhere_stays_unknown():
    assert runtime_context_length(SimpleNamespace(), 0) is None
    assert runtime_context_length(None, None) is None


def test_non_positive_and_non_numeric_values_are_skipped():
    model = SimpleNamespace(
        max_seq_length = 0,
        config = SimpleNamespace(max_position_embeddings = "n/a"),
    )
    assert runtime_context_length(model, -1) is None
    assert runtime_context_length(SimpleNamespace(max_seq_length = True), 2048) == 2048
