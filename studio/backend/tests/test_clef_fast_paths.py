# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clef's fast-path setup: nothing is installed where zoo's Triton conv runs, but the FLA TileLang
guard still applies; elsewhere the prebuilt causal-conv1d is installed as before."""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.systemone import clef_runtime  # noqa: E402
from core.training import worker  # noqa: E402
from utils import ssm_runtime  # noqa: E402


def _record(monkeypatch, usable: bool) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(ssm_runtime, "zoo_triton_causal_conv1d_usable", lambda: usable)
    monkeypatch.setattr(worker, "_guard_fla_tilelang", lambda: calls.append("guard"))
    monkeypatch.setattr(
        worker,
        "_ensure_causal_conv1d_fast_path",
        lambda *args, **kwargs: calls.append("install"),
    )
    monkeypatch.setattr(
        worker,
        "_install_fast_path_hooks",
        lambda *args, **kwargs: calls.append("hooks"),
    )
    return calls


def test_clef_installs_nothing_but_keeps_the_tilelang_guard(monkeypatch):
    calls = _record(monkeypatch, usable = True)
    clef_runtime._fast_paths("/models/clef")
    assert calls == ["guard"]


def test_clef_installs_causal_conv1d_where_zoo_conv_cannot_run(monkeypatch):
    calls = _record(monkeypatch, usable = False)
    clef_runtime._fast_paths("/models/clef")
    assert calls == ["install", "hooks"]
