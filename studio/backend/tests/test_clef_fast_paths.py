# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clef (a Qwen3.5 backbone) installs the prebuilt causal-conv1d and its availability hook, whose
setup also applies the FLA TileLang guard."""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.systemone import clef_runtime  # noqa: E402
from core.training import worker  # noqa: E402


def test_clef_installs_causal_conv1d_and_its_hooks(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        worker,
        "_ensure_causal_conv1d_fast_path",
        lambda *args, **kwargs: calls.append(("install", kwargs.get("required"))),
    )
    monkeypatch.setattr(
        worker,
        "_install_fast_path_hooks",
        lambda *args, **kwargs: calls.append(("hooks", kwargs.get("install_causal_conv1d"))),
    )
    clef_runtime._fast_paths("/models/clef")
    assert calls == [("install", True), ("hooks", True)]


def test_the_fast_path_hooks_apply_the_tilelang_guard(monkeypatch):
    guarded = []
    monkeypatch.setattr(worker, "_guard_fla_tilelang", lambda: guarded.append(True))
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")
    worker._install_fast_path_hooks(event_queue = [], model_name = "/models/clef")
    assert guarded == [True]
