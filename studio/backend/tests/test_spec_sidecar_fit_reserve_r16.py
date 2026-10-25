# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sixteenth-round cells for the Metal fit probe and the Auto sidecar drop.

A llama-fit-params failure that is not cached across loads (timeout, crash) is still
remembered for the rest of the load, its message is logged once, the draft-free embedding
probe runs only when the drafter does not fit, and a discount the drop unlocks is logged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_metal_explicit_context_guard as _metal  # noqa: E402
import test_metal_fit_params_probe_reasons as _probe  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_GIB = 1024**3


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _messages(monkeypatch):
    seen = []
    real_info = _lc.logger.info

    def info(msg, *args, **kwargs):
        seen.append(str(msg))
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(_lc.logger, "info", info)
    return seen


@_probe.posix_only
def test_a_failed_probe_runs_once_per_load(tmp_path):
    calls, script = _probe._counting(tmp_path, "exit 3")
    binary, model, _ = _probe._install(tmp_path, script)
    backend = LlamaCppBackend()
    with _lc._load_probe_memo_scope():
        assert backend._metal_measured_model_mib(binary, model) is None
        assert backend._metal_measured_model_mib(binary, model) is None
        assert "exited with code 3" in backend._metal_probe_failure
    assert _probe._ncalls(calls) == 1
    # The next load probes again: a cold shader compile may have finished by then.
    with _lc._load_probe_memo_scope():
        assert backend._metal_measured_model_mib(binary, model) is None
    assert _probe._ncalls(calls) == 2


def test_an_unmeasured_model_is_reported_once_per_load(monkeypatch):
    messages = _messages(monkeypatch)
    case = _metal.TestWhichLoadsLeaveTheEmbeddingsInTheMapping()
    with _lc._load_probe_memo_scope():
        assert case._bytes(monkeypatch, measured = None) == 0
        assert case._bytes(monkeypatch, measured = None) == 0
    assert sum("could not measure how llama.cpp maps" in m for m in messages) == 1


def _counting_embd(monkeypatch, value):
    calls = []

    def embd(
        self,
        *a,
        mtp_may_engage = False,
        **k,
    ):
        calls.append(mtp_may_engage)
        return value(mtp_may_engage)

    monkeypatch.setattr(LlamaCppBackend, "_metal_demand_paged_embedding_bytes", embd)
    return calls


def test_the_draft_free_probe_waits_for_a_drafter_that_does_not_fit(tmp_path, monkeypatch):
    """16 GiB holds the target and the sidecar (test_spec_sidecar_fit_reserve.TestMetalLaunch),
    so the draft-free discount, a cold llama-fit-params run, is never needed."""
    calls = _counting_embd(monkeypatch, lambda mtp: 0)
    cmd, _backend = _r2._metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 16)
    assert "--model-draft" in cmd
    assert False not in calls


def test_a_discount_the_drop_unlocks_is_logged(tmp_path, monkeypatch):
    """Excluded NextN blocks forfeit the discount with the drafter engaged; once Auto drops
    it the discount applies and is reported like any other."""
    _counting_embd(monkeypatch, lambda mtp: 0 if mtp else 3 * _GIB)
    messages = _messages(monkeypatch)
    cmd, _backend = _r2._metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 9.1)
    assert "--model-draft" not in cmd
    assert any("of input embeddings stay in the CPU file mapping" in m for m in messages)
