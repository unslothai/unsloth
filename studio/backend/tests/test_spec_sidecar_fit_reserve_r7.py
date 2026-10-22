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
    # A discrete CUDA host on every runner: macOS would otherwise take the Metal path
    # (class-level budget reads too, such as the pass-through zero-context drop).
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(
        _lc.LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )


def _margins(cmd):
    """Per-device ``--fit-target`` entries (a scalar is one entry)."""
    raw = _r2._last(cmd, "--fit-target") or "1024"
    return [int(float(e)) for e in str(raw).split(",")]


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


def test_a_launch_left_on_two_cards_reserves_each_cards_share(tmp_path, _discrete):
    """The unpinned drafter layer-splits by free memory too, so each card's --fit-target
    entry carries that card's share of it: half on two equal cards."""
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    margins = _margins(_dspark(backend, gguf, tmp_path))
    assert len(margins) == 2
    assert all(1024 + 4 * 1024 <= m < 1024 + 5 * 1024 for m in margins)


def test_unequal_cards_carry_their_own_shares(tmp_path, _discrete):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = ((0, 30_000, 30_000), (1, 10_000, 10_000)))
    backend._arch_gate_survivors = lambda binary = None: []
    big, small = _margins(_dspark(backend, gguf, tmp_path))
    # 30,000 of 40,000 MiB free: three quarters of the ~8.3 GiB reserve, a quarter on the other.
    assert 1024 + 6 * 1024 <= big < 1024 + 7 * 1024
    assert 1024 + 2 * 1024 <= small < 1024 + 3 * 1024


def test_a_small_card_beside_a_large_one_is_not_over_reserved(tmp_path, _discrete):
    """A broadcast of the large card's share (~7.6 GiB) would exceed the 8,000 MiB card."""
    backend, gguf = _r2._cuda_backend(tmp_path, memory = ((0, 80_000, 80_000), (1, 8_000, 8_000)))
    backend._arch_gate_survivors = lambda binary = None: []
    big, small = _margins(_dspark(backend, gguf, tmp_path))
    assert big >= 1024 + 7 * 1024
    assert small < 1024 + 1024


@pytest.mark.parametrize(
    "extras, bands",
    [
        # Equal cards: blocks 0-1 below 2/4, block 2 at it, so two on card 0 and one on card 1.
        ([], [(5, 6), (2, 3)]),
        # 3:2 keeps every block under 0.6 of n_layer + 1, so all three on card 0.
        (["--tensor-split", "3,2"], [(8, 9), (0, 1)]),
    ],
)
def test_draft_shares_follow_llama_cpps_whole_layer_split(tmp_path, _discrete, extras, bands):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    read = backend._draft_backend_for

    def three_blocks(path):
        db = read(path)
        db._n_layers = 3
        return db

    backend._draft_backend_for = three_blocks
    margins = _margins(_dspark(backend, gguf, tmp_path, extra_args = extras))
    for m, (lo, hi) in zip(margins, bands):
        assert 1024 + lo * 1024 <= m < 1024 + hi * 1024, margins


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


def test_a_users_per_device_margins_stay_per_device_across_cards():
    flags = _lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        ["--fit-target", "1024,20000"],
        {},
        1024,
        4213,
        per_device = [4213, 4213],
    )
    assert flags == ["--fit-target", "5237,24213"]


def test_a_scalar_user_margin_is_broadcast_under_per_device_shares():
    flags = _lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        ["--fit-target", "2048"],
        {},
        1024,
        5000,
        per_device = [5000, 500],
    )
    assert flags == ["--fit-target", "7048,2548"]


@pytest.mark.parametrize(
    "argv, devices, on_target",
    [
        # --split-mode none puts the target on --main-gpu, not device 0.
        (["--split-mode", "none", "--main-gpu", "1", "--spec-draft-device", "CUDA1"], None, True),
        (["--split-mode", "none", "--main-gpu", "1", "--spec-draft-device", "CUDA0"], None, False),
        (
            [
                "--device",
                "CUDA0,CUDA1",
                "--split-mode",
                "none",
                "--main-gpu",
                "1",
                "--spec-draft-device",
                "CUDA1",
            ],
            "CUDA0,CUDA1",
            True,
        ),
        (
            [
                "--device",
                "CUDA0,CUDA1",
                "--split-mode",
                "none",
                "--main-gpu",
                "1",
                "--spec-draft-device",
                "CUDA0",
            ],
            "CUDA0,CUDA1",
            False,
        ),
    ],
)
def test_split_mode_none_compares_the_pin_with_the_main_gpu(argv, devices, on_target):
    assert _lc._spawn_draft_on_target(argv, devices, {}) is on_target


def test_split_mode_none_keeps_the_whole_reserve_on_the_main_card(tmp_path, _discrete):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    margin = _r4._margin(_dspark(backend, gguf, tmp_path, extra_args = ["--split-mode", "none"]))
    assert margin >= _FLOOR


def test_a_repeated_draft_pin_is_one_card():
    argv = ["--device", "CUDA0", "--device-draft", "CUDA0,CUDA0"]
    assert _lc._spawn_draft_on_target(argv, "CUDA0")


@pytest.mark.parametrize("pin, on_target", [("CUDA0", True), ("CUDA1", False)])
def test_split_mode_none_defaults_to_the_lists_first_device(pin, on_target):
    argv = ["--device", "CUDA0,CUDA1", "--split-mode", "none", "--spec-draft-device", pin]
    assert _lc._spawn_draft_on_target(argv, "CUDA0,CUDA1", {}) is on_target


@pytest.mark.parametrize(
    "user, env, want",
    [
        # A short list sets only the entries it names; the rest keep the earlier value.
        (["--fit-target", "4096", "--fit-target", "2048,3072"], {}, "3048,4072,5096"),
        # The env applies first, the argv after it.
        ([], {"LLAMA_ARG_FIT_TARGET": "4096"}, "5096,5096,5096"),
        (["--fit-target", "2048"], {"LLAMA_ARG_FIT_TARGET": "4096,4096,8192"}, "3048,3048,3048"),
        (["--fit-target=2048,3072"], {"LLAMA_ARG_FIT_TARGET": "8192"}, "3048,4072,9192"),
        # The underscore spelling, spaced and inline.
        (["--fit_target", "4096"], {}, "5096,5096,5096"),
        (["--fit_target=2048,3072", "-fitt", "4096"], {}, "5096,5096,5096"),
    ],
)
def test_per_device_margins_follow_llama_cpps_update_order(user, env, want):
    flags = _lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        user,
        env,
        1024,
        1000,
        per_device = [1000, 1000, 1000],
    )
    assert flags == ["--fit-target", want]


def test_an_explicit_tensor_split_decides_the_draft_shares(tmp_path, _discrete):
    """The drafter inherits the target's --tensor-split, so 3:1 puts 3/4 on card 0 even
    though both cards are equally free."""
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    big, small = _margins(_dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,1"]))
    assert 1024 + 6 * 1024 <= big < 1024 + 7 * 1024
    assert 1024 + 2 * 1024 <= small < 1024 + 3 * 1024


@pytest.mark.parametrize("pin, reserved", [("CUDA0", True), ("CUDA1", False)])
def test_split_mode_none_narrows_an_inherited_device_list_at_planning(
    tmp_path, monkeypatch, _discrete, pin, reserved
):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0,CUDA1")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    extras = ["--split-mode", "none", "--spec-draft-device", pin]
    cmd = _dspark(backend, gguf, tmp_path, extra_args = extras)
    assert (_r4._margin(cmd) >= _FLOOR) is reserved


def test_per_device_margins_follow_an_inherited_mask_order(tmp_path, monkeypatch, _discrete):
    """CUDA_VISIBLE_DEVICES=1,0 makes physical 1 the child's device 0, so the list's first
    entry is the smaller card's share."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = ((0, 30_000, 30_000), (1, 10_000, 10_000)))
    backend._arch_gate_survivors = lambda binary = None: []
    first, second = _margins(_dspark(backend, gguf, tmp_path))
    assert first < second


def test_the_output_card_keeps_the_draft_compute_buffers_without_a_block(tmp_path, _discrete):
    """3:2 puts all three blocks on card 0, but the draft still runs (and writes its
    logits) on the last card, so its compute floor stays reserved there."""
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    read = backend._draft_backend_for

    def three_blocks(path):
        db = read(path)
        db._n_layers = 3
        return db

    backend._draft_backend_for = three_blocks
    floor = math.ceil(_lc.LlamaCppBackend._MTP_DRAFT_COMPUTE_BYTES / (1024 * 1024))
    _big, small = _margins(_dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))
    assert small >= 1024 + floor
