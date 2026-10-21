# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seventh-round cells for the DSpark/DFlash sidecar reserve.

The reserve follows the child as spawned: a launch the arch gate narrows to one card gets it,
a draft pin is compared in the spawned argv's device frame (and against an inherited
LLAMA_ARG_DEVICE), a trailing -c 0 prices the native context, a repeated cleanup leaves the
user's identical margin alone, and the underscore spellings llama.cpp accepts are read.
"""

from __future__ import annotations

import math
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

_GIB = 1024 * 1024 * 1024
_FLOOR = 1024 + 8 * 1024  # base margin + the 8 GiB sidecar
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)


@pytest.fixture
def _discrete(monkeypatch):
    # A discrete CUDA host on every runner: macOS would otherwise take the Metal path.
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)


def _dspark(backend, gguf, tmp_path, **kw):
    return _placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        **kw,
    )["cmd"]


def test_a_launch_the_arch_gate_narrows_to_one_card_gets_the_reserve(tmp_path, _discrete):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: [1]
    assert _r4._margin(_dspark(backend, gguf, tmp_path)) >= _FLOOR


def test_a_launch_left_on_two_cards_gets_no_reserve(tmp_path, _discrete):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    assert _r4._margin(_dspark(backend, gguf, tmp_path)) < _FLOOR


@pytest.mark.parametrize("pin, reserved", [("CUDA1", True), ("CUDA0", False)])
def test_an_inherited_target_device_decides_the_draft_pin(
    tmp_path, monkeypatch, _discrete, pin, reserved
):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA1")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    cmd = _dspark(backend, gguf, tmp_path, extra_args = ["--spec-draft-device", pin])
    assert (_r4._margin(cmd) >= _FLOOR) is reserved


@pytest.mark.parametrize(
    "argv, devices, on_target",
    [
        # The masked child's device 0 is the target.
        (["--spec-draft-device", "CUDA0"], None, True),
        (["--spec-draft-device", "CUDA1"], None, False),
        # A companion widening: the target is --device CUDA0, the drafter its CUDA1.
        (["--device", "CUDA0", "--spec-draft-device", "CUDA1"], "CUDA0", False),
        (["--device", "CUDA1", "--spec-draft-device", "CUDA1"], "CUDA1", True),
        # Settled at planning, where the backend is known.
        (["--spec-draft-device", "RPC0"], None, True),
        (["--spec-draft-device", "none"], None, True),
    ],
)
def test_the_spawned_argv_decides_a_cuda_draft_pin(argv, devices, on_target):
    assert _lc._spawn_draft_on_target(argv, devices) is on_target


def test_a_trailing_zero_context_is_priced_at_the_native_length(tmp_path, _discrete):
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend._context_length = 131072
    per_token = 64 * 1024  # 8 GiB of draft KV at the native 131072

    def estimate(n_ctx, **kw):
        return 8 * _GIB + max(0, int(n_ctx)) * per_token

    backend._estimate_mtp_overhead_bytes = estimate
    cmd = _dspark(
        backend,
        gguf,
        tmp_path,
        gpu_memory_mode = "auto",
        n_ctx = 4096,
        extra_args = ["-c", "0"],
    )
    assert _r2._last(cmd, "-c") == "0", cmd
    assert _r4._margin(cmd) >= 16 * 1024, cmd


def test_a_repeated_cleanup_keeps_the_users_identical_margin():
    integrity = ["--fit-target", "9449"]
    rebuilt = ["--fit-target", "1024"]
    user = ["--fit-target", "9449"]
    # Studio's copy is already gone; only the user's remains.
    cmd = ["llama-server", "-m", "m.gguf", "--fit-target", "1024", *user]
    assert _lc._drop_sidecar_margins(cmd, integrity, rebuilt, [], user_args = user) == cmd
    # With Studio's copy still there, it is the one swapped.
    cmd = ["llama-server", "--fit-target", "9449", *user]
    assert _lc._drop_sidecar_margins(cmd, integrity, rebuilt, [], user_args = user) == [
        "llama-server",
        "--fit-target",
        "1024",
        *user,
    ]
    # A trailing pair a launch appended goes; the user's identical earlier one stays.
    pair = ["--fit-target", "17874"]
    cmd = ["llama-server", *pair, *pair]
    assert _lc._drop_sidecar_margins(cmd, [], [], [pair], user_args = pair) == [
        "llama-server",
        *pair,
    ]
    assert _lc._drop_sidecar_margins(["llama-server", *pair], [], [], [pair], user_args = pair) == [
        "llama-server",
        *pair,
    ]


def test_underscore_spellings_are_read_like_llama_cpp():
    assert _lc._draft_card_share_unknown(["--spec_draft_n_cpu_moe", "8"], {})
    assert _lc._draft_n_ubatch(["--spec_draft_n_ubatch", "2048"], {}) == 2048


def test_metal_auto_keeps_a_drafter_pinned_to_another_device(tmp_path, monkeypatch):
    """Its weights allocate on the RPC host, so Metal's budget is no reason to drop it."""
    extras = ["--rpc", "192.0.2.1:50052", "--device", "MTL0", "--device-draft", "RPC0"]
    cmd, backend = _r2._metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 12, extra_args = extras)
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"
    # The same budget with the drafter on Metal still drops it.
    cmd, backend = _r2._metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 9.1, sub = "local")
    assert "--model-draft" not in cmd


def test_a_repeated_device_is_one_card_for_the_draft_pin():
    argv = ["--device", "CUDA0,CUDA0", "--device-draft", "CUDA0"]
    assert _lc._spawn_draft_on_target(argv, "CUDA0,CUDA0")
    assert not _lc._spawn_draft_on_target(["--device-draft", "CUDA1"], "CUDA0,CUDA0")


def test_an_unread_native_length_prices_the_fitters_minimum_context():
    assert _lc._sidecar_child_ctx([], {}, None, 1, True) == _lc._LLAMA_FIT_MIN_CTX
    assert _lc._sidecar_child_ctx([], {}, None, 4, False) == 4 * _lc._LLAMA_FIT_MIN_CTX
    assert _lc._sidecar_child_ctx(["-c", "8192"], {}, None, 4, False) == 8192


@pytest.mark.parametrize("extras", [("-ngld", "1"), ("-otd", "blk\\.0\\.=CPU")])
def test_a_partly_placed_drafter_keeps_its_compute_floor(tmp_path, _discrete, extras):
    backend, gguf = _r2._cuda_backend(tmp_path)
    margin = _r4._margin(_dspark(backend, gguf, tmp_path, extra_args = list(extras)))
    floor = math.ceil(_lc.LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES / (1024 * 1024))
    # Not the whole 8 GiB file, but not nothing either.
    assert 1024 + floor <= margin < _FLOOR
