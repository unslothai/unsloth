# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Twenty-second-round cells: the Metal Auto sidecar drop and the standard-load handoff.

The Metal Auto drop priced the 8192 floor even for an explicit context, which launches
verbatim, so Auto kept a sidecar that then made the explicit context refuse while the same
load with speculation off went through. It also ran with no KV estimate, where the zero KV
read as a target that fits alone (the discrete drop requires the estimate). And the
standard-load handoff killed a starting or recovering llama-server without waiting for the
driver to hand its memory back.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_metal_explicit_context_guard as _metal  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r19 as _r19  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

harness = _r19.harness  # the respawn fixture, reused as-is
_GIB = 1024**3


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _metal_load(
    tmp_path,
    monkeypatch,
    mode,
    *,
    budget_gib,
    wired_gib = None,
    **kw,
):
    """_r2._metal_cmd with the context, KV rate and estimate exposed. Returns (cmd, backend,
    error); a refusal is the error, with no launch."""
    sub = tmp_path / mode
    sub.mkdir()
    sidecar = sub / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    real_init = GgufLoadIntent.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("dspark_draft_path", str(sidecar))
        kwargs.setdefault("speculative_type", mode)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(GgufLoadIntent, "__init__", init)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(_r2._CAPS)),
    )
    backend = LlamaCppBackend()
    wired_gib = budget_gib if wired_gib is None else wired_gib
    error = None
    try:
        cmd = _metal._launch(
            sub,
            monkeypatch,
            backend = backend,
            real_fit = True,
            budget_bytes = int(budget_gib * _GIB),
            wired_bytes = int(wired_gib * _GIB),
            weights_bytes = 4 * _GIB,
            **kw,
        ).get("cmd")
    except RuntimeError as exc:
        cmd, error = None, str(exc)
    monkeypatch.undo()
    return cmd, backend, error


# Finding 1: an explicit context is priced at the context it launches with.


@pytest.mark.parametrize("budget_gib, wired_gib", [(20, 20), (15, 20), (20, 18), (20, 0), (30, 18)])
def test_metal_auto_drops_a_sidecar_an_explicit_context_cannot_hold(
    tmp_path, monkeypatch, budget_gib, wired_gib
):
    """65,536 tokens at 128 KiB each hold the 4 GiB target (Off loads), not target + the
    4 GiB sidecar: Auto must drop the sidecar and load like Off, not keep it and refuse."""
    kw = dict(n_ctx = 65_536, kv_per_token = 128 * 1024, budget_gib = budget_gib, wired_gib = wired_gib)
    off, _, off_error = _metal_load(tmp_path, monkeypatch, "off", **kw)
    assert off_error is None and _r2._last(off, "-c") == "65536"
    auto, backend, error = _metal_load(tmp_path, monkeypatch, "auto", **kw)
    assert error is None
    assert "--model-draft" not in auto
    assert backend.spec_fallback_reason == "drafter_no_vram"
    assert _r2._last(auto, "-c") == "65536"


def test_metal_auto_keeps_a_sidecar_an_explicit_context_holds(tmp_path, monkeypatch):
    """Control: a small explicit context holds both, so the sidecar stays."""
    cmd, backend, error = _metal_load(
        tmp_path, monkeypatch, "auto", n_ctx = 4096, kv_per_token = 1024, budget_gib = 20
    )
    assert error is None
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


def test_metal_auto_still_drops_at_the_floor_without_an_explicit_context(tmp_path, monkeypatch):
    """Control: the Auto context keeps the floor pricing (test_spec_sidecar_fit_reserve_r6)."""
    cmd, backend, error = _metal_load(tmp_path, monkeypatch, "auto", n_ctx = 0, budget_gib = 12)
    assert error is None
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"


# Finding 2: no KV estimate, no drop (the discrete drop's own guard).


def test_metal_auto_keeps_the_sidecar_when_kv_cannot_be_estimated(tmp_path, monkeypatch):
    """10 GiB is under target + sidecar priced with zero KV, but with no estimate the target
    alone is not known to fit either, so Auto must not decide drafter_no_vram."""
    cmd, backend, error = _metal_load(
        tmp_path, monkeypatch, "auto", n_ctx = 0, budget_gib = 10, can_estimate_kv = False
    )
    assert error is None
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


# Finding 3: a killed starting / recovering server is waited for like a loaded one.


def test_the_standard_handoff_waits_after_killing_an_unhealthy_server(harness, monkeypatch):
    import routes.inference as inf

    backend, _first, _loads = harness(_r19._OFF, "none")
    backend._healthy = False  # starting or recovering: a process, not loaded
    assert not backend.is_loaded and backend._process is not None
    waits = []
    monkeypatch.setattr(backend, "_wait_for_vram_settle", lambda **kw: waits.append(kw))
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert backend._process is None
    assert len(waits) == 1 and waits[0]["since_kill"] > 0


def test_the_standard_handoff_skips_the_wait_with_nothing_running(monkeypatch):
    """Control: no process, nothing killed, nothing to wait for."""
    import routes.inference as inf

    backend = _lc.LlamaCppBackend()
    waits = []
    monkeypatch.setattr(backend, "_wait_for_vram_settle", lambda **kw: waits.append(kw))
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert waits == []
