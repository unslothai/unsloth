# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fifth-round cells for the DSpark/DFlash sidecar reserve and the Metal OOM episode.

The reserve is priced at the context the child is launched with, keeps the launched
device's own margin, stays off a drafter split across the host, leaves the target's
rollback copies to the fitter that measures them, and an explicit reload after a terminal
Metal OOM starts a fresh episode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent  # noqa: E402

_GIB = 1024 * 1024 * 1024
_FLOOR = 1024 + 8 * 1024  # base margin + the 8 GiB sidecar


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


# Finding 1: an explicit Manual context is emitted as -c, so it is the one to price.


def test_explicit_manual_context_is_priced_at_that_context(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._context_length = 131072
    per_token = 64 * 1024  # 8 GiB of draft KV at the native 131072

    def estimate(n_ctx, **kw):
        return 8 * _GIB + max(0, int(n_ctx)) * per_token

    backend._estimate_mtp_overhead_bytes = estimate
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        n_ctx = 4096,
    )["cmd"]
    if "--fit-target" not in cmd:
        pytest.skip("this host plans no fitter for Manual + Auto")
    assert _r2._last(cmd, "-c") == "4096"
    assert _r4._margin(cmd) < 512 + 12 * 1024


# Finding 2: one launched device keeps its own (first) entry of a per-device list.


@pytest.mark.parametrize("value", ["1024,65536", "1024/65536"], ids = ["comma", "slash"])
def test_single_device_reserve_uses_the_first_list_entry(value):
    flags = _lc._sidecar_reserve_margin_flags(
        ["--fit", "on", "-md", "/d.gguf", "--fit-target", value],
        ("--fit-target", value),
        {},
        1024,
        8192,
    )
    assert flags == ["--fit-target", str(1024 + 8192)]


# Finding 3: a positive -ngld below the drafter's depth splits it with the host.


@pytest.mark.parametrize(
    "extras", [("-ngld", "1"), ("--spec-draft-ngl", "1")], ids = ["short", "long"]
)
def test_partial_draft_offload_gets_no_whole_file_reserve(tmp_path, extras):
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        extra_args = extras,
    )["cmd"]
    assert _r4._margin(cmd) < _FLOOR


def test_full_draft_offload_still_gets_the_reserve(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = _r4._sidecar(tmp_path)
    )["cmd"]
    assert _r4._margin(cmd) >= _FLOOR


# Findings 4 and 5: target rollback copies are in the target context the fitter measures.

_ROLLBACK = 4 * _GIB


def _rollback_backend(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._rollback_state_bytes = lambda n_parallel = 1: _ROLLBACK

    def estimate(
        n_ctx,
        target_rollback = True,
        spec_draft_n_max = 0,
        **kw,
    ):
        rollback = _ROLLBACK * spec_draft_n_max if target_rollback else 0
        return 8 * _GIB + rollback

    backend._estimate_mtp_overhead_bytes = estimate
    return backend, gguf


def test_initial_reserve_leaves_target_rollback_to_the_fitter(tmp_path):
    backend, gguf = _rollback_backend(tmp_path)
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = _r4._sidecar(tmp_path)
    )["cmd"]
    assert _FLOOR <= _r4._margin(cmd) < _FLOOR + _ROLLBACK // (1024 * 1024)


def test_noflash_reprice_leaves_target_rollback_to_the_fitter(tmp_path):
    backend, gguf = _rollback_backend(tmp_path)
    cmds = _r4._run(
        backend,
        gguf,
        crash = lambda cmd: -11 if "on" == _r2._last(cmd, "--flash-attn") else None,
        healthy = lambda cmd: _r2._last(cmd, "--flash-attn") == "off",
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
    )
    noflash = [c for c in cmds if _r2._last(c, "--flash-attn") == "off" and "--model-draft" in c]
    if not noflash:
        pytest.skip("no --flash-attn off retry with the drafter in this harness")
    assert _r4._margin(noflash[0]) < _FLOOR + _ROLLBACK // (1024 * 1024)


# Findings 6 and 7: an explicit reload after a terminal Metal OOM starts a fresh episode.


def test_explicit_reload_clears_a_terminal_metal_oom(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 4096)
    key = backend._metal_oom_key(intent)
    backend._metal_oom_strikes = (key, 2)
    backend._metal_oom_terminal_key = key
    _r4._run(backend, gguf, crash = lambda cmd: None, healthy = lambda cmd: True, n_ctx = 4096)
    if backend._last_load_intent is None:
        pytest.skip("load did not reach the commit point in this harness")
    assert backend._metal_oom_terminal_key is None
    assert backend._metal_oom_strikes is None
    assert not backend._metal_oom_is_terminal(backend._last_load_intent)
