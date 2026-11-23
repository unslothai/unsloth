# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused packed W4A8 / W6A8 GEMM: card gate, kill switch, spill rule, and bit-exactness under the interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_grouped_int8_gemm as fg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(fg.GROUPED_FUSED_ENV, raising = False)
    fg._DEVICE.clear()
    fg._SPILLS.clear()
    yield
    fg._DEVICE.clear()
    fg._SPILLS.clear()


@pytest.mark.parametrize(
    "raw, mode", [("0", "off"), ("off", "off"), ("1", "force"), ("", "auto"), ("x", "auto")]
)
def test_kill_switch_modes(monkeypatch, raw, mode):
    monkeypatch.setenv(fg.GROUPED_FUSED_ENV, raw)
    assert fg.fused_mode() == mode


def test_arch_tiles_listed_unlisted_and_off():
    assert fg.arch_tiles((12, 0), "auto")
    assert fg.arch_tiles((8, 9), "auto")
    assert fg.arch_tiles((12, 0), "off") == ()
    assert fg.arch_tiles((10, 0), "auto") == ()  # unmeasured: stays off
    assert fg.arch_tiles((10, 0), "force") == fg._FALLBACK_TILES
    assert fg.arch_tiles((7, 5), "force") == ()  # no int8 MMA tile below sm80 here
    assert fg.arch_tiles((8, 0), "auto") == ()  # every measured A100 tile spills under Triton 3.8
    assert fg.arch_tiles(None, "force") == ()


@pytest.mark.parametrize(
    "cap, version, need",
    [
        ((8, 9), (3, 8), True),
        ((8, 0), (3, 9), True),
        ((8, 9), (3, 7), False),
        ((12, 0), (3, 8), False),
        ((8, 6), None, False),
    ],
)
def test_spill_rule(cap, version, need, monkeypatch):
    monkeypatch.setattr(fg, "_triton_version", lambda: None)  # None = no Triton installed
    assert fg.spill_check_needed(cap, version) is need


def test_n_spills_reads_handle_or_inner_kernel():
    class H:
        n_spills = 3

    class W:
        kernel = H()

    assert fg.n_spills(H()) == 3
    assert fg.n_spills(W()) == 3
    assert fg.n_spills(object()) is None


def _fake_card(monkeypatch, cap, version, results):
    """``results``: tile -> (ok, spills) from check_tile on the fake card (records the strict flag it was called with)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
    monkeypatch.setattr(fg, "_kernel", lambda: (None, None))
    monkeypatch.setattr(fg, "_triton_version", lambda: version)
    import core.inference.rocm_bf16 as rb

    monkeypatch.setattr(rb, "is_rocm_torch", lambda t: False)

    def check(device, tile, strict):
        calls.append(strict)
        ok, spills = results.get(tile, (True, 0))
        if strict and (spills is None or spills > 0):
            return False, spills
        return ok, spills

    monkeypatch.setattr(fg, "check_tile", check)
    return calls


def test_fused_config_sm120_takes_both_tiles(monkeypatch):
    calls = _fake_card(monkeypatch, (12, 0), (3, 8), {})
    rules = fg.fused_config(0)
    assert rules == fg._ARCH_TILES[(12, 0)]
    assert calls and not any(calls)  # no spill rule off sm8x
    assert fg.tile_for(rules, 64, 6) == fg._PHASE
    assert fg.tile_for(rules, 64, 4) is None  # 4-bit codebook lookups lose on sm120
    assert fg.tile_for(rules, 65, 6) is None  # larger batches keep decode + int8 GEMM


def test_fused_config_drops_a_failing_rule_only(monkeypatch):
    monkeypatch.setitem(fg._ARCH_TILES, (12, 0), ((64, fg._PHASE, (6,)), (128, fg._JOIN, (4, 6))))
    _fake_card(monkeypatch, (12, 0), (3, 8), {fg._PHASE: (False, 0)})
    rules = fg.fused_config(0)
    assert rules == ((128, fg._JOIN, (4, 6)),)
    assert fg.tile_for(rules, 64, 6) == fg._JOIN


def test_fused_config_sm89_triton38_rejects_spilling_or_unreported_tiles(monkeypatch):
    calls = _fake_card(monkeypatch, (8, 9), (3, 8), {fg._JOIN: (True, 0)})
    rules = fg.fused_config(0)
    assert rules == fg._ARCH_TILES[(8, 9)]
    assert fg.tile_for(rules, 128, 4) == fg._JOIN and fg.tile_for(rules, 129, 6) is None
    assert all(calls)
    for spills in (12, None):
        fg._DEVICE.clear()
        _fake_card(monkeypatch, (8, 9), (3, 8), {fg._JOIN: (True, spills)})
        assert fg.fused_config(0) is None


def test_fused_config_sm89_triton37_no_spill_rule(monkeypatch):
    calls = _fake_card(monkeypatch, (8, 9), (3, 7), {fg._JOIN: (True, 12)})
    assert fg.fused_config(0) == fg._ARCH_TILES[(8, 9)]
    assert calls and not any(calls)


def test_fused_config_probe_failure_and_kill_switch(monkeypatch):
    _fake_card(monkeypatch, (12, 0), (3, 8), {r[1]: (False, 0) for r in fg._ARCH_TILES[(12, 0)]})
    assert fg.fused_config(0) is None
    fg._DEVICE.clear()
    monkeypatch.setenv(fg.GROUPED_FUSED_ENV, "0")
    calls = _fake_card(monkeypatch, (12, 0), (3, 8), {})
    assert fg.fused_config(0) is None
    assert calls == []


def test_unlisted_arch_only_when_forced(monkeypatch):
    calls = _fake_card(monkeypatch, (10, 0), (3, 8), {})
    assert fg.fused_config(0) is None and calls == []
    fg._DEVICE.clear()
    monkeypatch.setenv(fg.GROUPED_FUSED_ENV, "1")
    assert fg.fused_config(0) == fg._FALLBACK_TILES


def test_wrapper_declines_cpu_off_and_small_batches(monkeypatch):
    x = torch.zeros(64, 256, dtype = torch.int8)
    p = torch.zeros(32, 192, dtype = torch.int8)
    s = torch.zeros(32, 16, dtype = torch.uint8)
    one = torch.ones(64)
    assert fg.grouped_mm_dequant(x, p, s, None, one, torch.ones(32), None, 6, 16) is None  # CPU
    monkeypatch.setenv(fg.GROUPED_FUSED_ENV, "0")
    assert fg.grouped_mm_dequant(x, p, s, None, one, torch.ones(32), None, 6, 16) is None


def test_split_k_rule(monkeypatch):
    monkeypatch.setattr(fg, "_sm_count", lambda device: 100)
    tile = (32, 64, 128, 8, 4, 3, 2)
    # 2 x 48 output tiles: double to 4 splits (384 >= 200), each keeping >= 2 of the 24 K blocks
    assert fg.split_k(64, 3072, 3072, tile, None) == 4
    assert fg.split_k(16384, 3072, 3072, tile, None) == 1  # enough tiles already
    assert fg.split_k(64, 64, 512, tile, None) == 2  # K = 4 blocks: at most 2 splits
    assert fg.split_k(64, 3072, 3072, tile + (8,), None) == 8  # pinned


def test_covered_gate():
    tile = (32, 64, 128, 8, 4, 4)
    x = torch.zeros(40, 256, dtype = torch.int8)
    p6 = torch.zeros(32, 192, dtype = torch.int8)
    s = torch.zeros(32, 16, dtype = torch.uint8)
    assert fg.covered(x, p6, s, None, 6, 16, tile)
    assert not fg.covered(x, p6, s, None, 6, 16, (32, 64, 96, 8, 4, 4))  # K % BLOCK_K
    assert not fg.covered(x, p6, s, None, 4, 16, tile)  # width does not match 4 bits
    assert not fg.covered(x.float(), p6, s, None, 6, 16, tile)
    assert not fg.covered(x, p6, s, torch.zeros(16), 6, 16, tile)  # codebook only at 4 bits
    assert not fg.covered(x, p6, s[:, :8], None, 6, 16, tile)
    assert not fg.covered(x[:, ::2], p6, s, None, 6, 16, tile)
    assert not fg.covered(
        x, p6, s, None, 6, 16, (32, 64, 32, 8, 4, 4, 2)
    )  # phase loop needs BLOCK_K >= 64
    assert fg.covered(x, p6, s, None, 6, 16, (32, 64, 128, 8, 4, 4, 2))


_INTERPRETED = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    import torch
    import core.inference.diffusion_grouped_int8_gemm as fg
    from core.inference.diffusion_grouped_int8 import decode_int8_torch

    n = 0
    for tile in ((16, 32, 64, 1, 4, 1, 2), (16, 32, 64, 1, 4, 1, 2, 3), (32, 64, 128, 2, 4, 1, 1), (32, 64, 128, 2, 4, 1, 0)):
        for c in fg.probe_cases("cpu"):
            if not fg.covered(c["xq"], c["packed"], c["s_rel"], c["codebook"], c["bits"], c["group"], tile):
                continue
            out, _ = fg.launch(c["xq"], c["packed"], c["s_rel"], c["codebook"], c["xs"], c["ws"], c["bias"],
                               c["bits"], c["group"], tile)
            q8 = decode_int8_torch(c["packed"], c["s_rel"], c["codebook"], c["group"], c["bits"])
            ref = fg.reference(c["xq"], q8, c["xs"], c["ws"], c["bias"])
            assert torch.equal(out, ref), (tile, c["bits"], c["group"])
            n += 1
    print("OK", n)
    """
)


def test_fused_gemm_is_bit_identical_under_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _INTERPRETED.format(root = ROOT)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK 16" in proc.stdout
