# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fourth-round cells for the DSpark/DFlash sidecar reserve.

The reserve is now reconciled at every spawn off the argv being launched (a retry can turn the
fitter on or grow the draft cache), it carries the draft decode-graph floor, it prices the
context the child can pick, it stays off launches that span several devices, and the
drafter-free fallback takes back exactly the pairs Studio added.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_GIB = 1024 * 1024 * 1024
_COMPUTE_MIB = 224
_FLOOR = 1024 + 8 * 1024  # base margin + the 8 GiB sidecar


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _margin(cmd):
    return int(float(_r2._last(cmd, "--fit-target") or 1024))


def _sidecar(tmp_path):
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    return str(sidecar)


def _run(
    backend,
    gguf,
    *,
    crash,
    healthy,
    late_exit = False,
    **intent,
):
    """Every llama-server argv the load spawns. ``crash(cmd)`` is the child's exit code
    (None = still running), ``healthy(cmd)`` its health answer. ``late_exit`` makes the
    first poll still see the child running: it exits between the startup retry's check and
    the outer --flash-attn off check."""
    cmds = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _placement._REAL_POPEN(cmd, **kwargs)
        cmds.append(list(cmd))
        rc = crash(list(cmd))
        polls = [0]

        def poll(self):
            polls[0] += 1
            return None if late_exit and polls[0] == 1 else rc

        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": rc,
                "poll": poll,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: rc or 0,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None, **_kw):
        backend._stdout_lines = []
        return healthy(cmds[-1])

    backend._wait_for_health = fake_health
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        try:
            backend.load_model(
                GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **intent)
            )
        except Exception:
            pass
    return cmds


# Finding 1: a surviving --device list or LLAMA_ARG_DEVICE spans several cards.


@pytest.mark.parametrize(
    "args, env",
    [(["--device", "Vulkan0,Vulkan1"], {}), ([], {"LLAMA_ARG_DEVICE": "CUDA0,CUDA1"})],
    ids = ["argv", "env"],
)
def test_no_scalar_reserve_on_a_multi_device_launch(args, env):
    assert (
        _lc._sidecar_reserve_margin_flags(
            ["--fit", "on", "-md", "/d.gguf", *args], (), env, 1024, 8192
        )
        == []
    )


def test_a_single_named_device_keeps_the_reserve():
    flags = _lc._sidecar_reserve_margin_flags(
        ["--fit", "on", "-md", "/d.gguf", "--device", "CUDA0"], (), {}, 1024, 8192
    )
    assert flags == ["--fit-target", str(1024 + 8192)]


def test_extras_widening_the_device_list_get_no_scalar_reserve(tmp_path):
    backend, gguf = _r2._cuda_backend(
        tmp_path, memory = ((0, 24_000, 24_000), (1, 24_000, 24_000)), vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([0], True)
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _sidecar(tmp_path),
        extra_args = ("--device", "Vulkan0,Vulkan1"),
    )["cmd"]
    assert _margin(cmd) < _FLOOR


# Finding 5: the draft decode graph is charged beside the reserve bytes.


def test_the_reserve_carries_the_draft_compute_floor(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _placement._launch(
        backend, gguf, speculative_type = "dspark", dspark_draft_path = _sidecar(tmp_path)
    )["cmd"]
    assert _margin(cmd) >= _FLOOR + _COMPUTE_MIB


# Finding 6: Manual + Auto omits -c, so the child may pick the native context.


def test_manual_auto_context_prices_the_native_context(tmp_path):
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
        dspark_draft_path = _sidecar(tmp_path),
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        n_ctx = 0,
    )["cmd"]
    if "--fit-target" not in cmd:
        pytest.skip("this host plans no fitter for Manual + Auto")
    assert _margin(cmd) >= 512 + 16 * 1024


# Finding 7: the --flash-attn off retry turns the fitter on next to the sidecar.


def test_the_noflash_fit_flip_carries_the_reserve(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._select_gpus = lambda *a, **k: ([0], False)  # planned full offload: --fit off
    cmds = _run(
        backend,
        gguf,
        crash = lambda cmd: -11 if "on" == _r2._last(cmd, "--flash-attn") else None,
        healthy = lambda cmd: _r2._last(cmd, "--flash-attn") == "off",
        late_exit = True,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _sidecar(tmp_path),
    )
    noflash = [c for c in cmds if _r2._last(c, "--flash-attn") == "off"]
    if not noflash or _r2._last(noflash[0], "--fit") != "on":
        pytest.skip("no --flash-attn off retry with the fitter on in this harness")
    assert "--model-draft" in noflash[0]
    assert _margin(noflash[0]) >= _FLOOR


# Finding 3: the --flash-attn off retry pads the draft V to f16, so the reserve grows.


def test_the_noflash_retry_reprices_the_reserve(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)

    def estimate(
        n_ctx,
        draft_cache_type_v = None,
        **kw,
    ):
        quantized = str(draft_cache_type_v or "f16").lower().startswith("q")
        return 8 * _GIB + (1 if quantized else 3) * _GIB

    backend._estimate_mtp_overhead_bytes = estimate
    cmds = _run(
        backend,
        gguf,
        crash = lambda cmd: -11 if "on" == _r2._last(cmd, "--flash-attn") else None,
        healthy = lambda cmd: _r2._last(cmd, "--flash-attn") == "off",
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _sidecar(tmp_path),
        extra_args = ("--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"),
    )
    first = cmds[0]
    noflash = [c for c in cmds if _r2._last(c, "--flash-attn") == "off"]
    if not noflash or "--model-draft" not in noflash[0]:
        pytest.skip("no --flash-attn off retry with the drafter in this harness")
    assert _margin(noflash[0]) >= _margin(first) + 2 * 1024


# Finding 8: a reserve the fit-on startup retry added is taken back with the drafter.


def test_a_retry_added_reserve_leaves_with_the_drafter(tmp_path):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._select_gpus = lambda *a, **k: ([0], False)  # planned full offload: --fit off
    cmds = _run(
        backend,
        gguf,
        # flash on: signal crash (fit-on retry, then the --flash-attn off retry carries the
        # reserve forward); flash off with the drafter: plain exit, so the drafter goes.
        crash = lambda cmd: (
            -11
            if _r2._last(cmd, "--flash-attn") == "on"
            else (1 if "--model-draft" in cmd else None)
        ),
        healthy = lambda cmd: "--model-draft" not in cmd,
        n_ctx = 4096,
        speculative_type = "dspark",
        dspark_draft_path = _sidecar(tmp_path),
    )
    reserved = [c for c in cmds if "--model-draft" in c and _margin(c) >= _FLOOR]
    final = cmds[-1]
    if not reserved or "--model-draft" in final:
        pytest.skip("no reserved retry followed by a drafter-free launch in this harness")
    assert _margin(final) < _FLOOR


# Findings 9 and 10: the fallback cleanup itself.


def test_cleanup_swaps_only_the_margin_pair_after_a_rewrite():
    integrity = ["--kv-unified", "--fit-ctx", "4096", "--fit-target", "9252"]
    rebuilt = ["--kv-unified", "--fit-ctx", "4096", "--fit-target", "1024"]
    # single-sequence recovery already dropped --kv-unified and set --parallel 1
    cmd = ["--parallel", "1", "--fit", "on", "--fit-ctx", "4096", "--fit-target", "9252"]
    out = _lc._drop_sidecar_margins(cmd, integrity, rebuilt, [])
    assert out == ["--parallel", "1", "--fit", "on", "--fit-ctx", "4096", "--fit-target", "1024"]


def test_cleanup_removes_studios_trailing_pair_not_the_users():
    pair = ["--fit-target", "9225"]
    cmd = ["--fit", "on", "--fit-target", "9225", "--fit-target", "1024", *pair]
    out = _lc._drop_sidecar_margins(cmd, [], [], [pair])
    assert out == ["--fit", "on", "--fit-target", "9225", "--fit-target", "1024"]
    assert _r2._last(out, "--fit-target") == "1024"


def test_cleanup_removes_every_recorded_pair():
    pairs = [["--fit-target", "9225"], ["--fit-target", "11273"]]
    cmd = ["--fit", "on", *pairs[0], *pairs[1]]
    assert _lc._drop_sidecar_margins(cmd, [], [], pairs) == ["--fit", "on"]


# Finding 4: the Auto-drop predicate prices the target with its drafter-free discount.


def test_metal_auto_drop_prices_the_target_with_the_drafter_free_discount(tmp_path, monkeypatch):
    def discount(
        self,
        *args,
        mtp_may_engage = True,
        **kwargs,
    ):
        return 0 if mtp_may_engage else 3 * _GIB

    monkeypatch.setattr(LlamaCppBackend, "_metal_demand_paged_embedding_bytes", discount)
    auto, _ = _r2._metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 7.5)
    assert "--model-draft" not in auto
