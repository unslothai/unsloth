# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""gfx1151 block V^T (``vblk``, comfy-kitchen #368) on the Triton INT8 attention: only long calls with 56 heads
(MiniMax-H3) take it, only where the long check found it bit-identical to the channel-major kernel, and every other
call (24 / 32 heads, short calls, the switch off, a failed launch) gets the config main gives it. CPU: a faked
gfx1151 card whose Triton entry records the config it is handed and returns fp32 attention; the kernel itself runs
under TRITON_INTERPRET in a subprocess."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from core.inference import diffusion_int8_attention as I

HD = 128
GFX = "gfx1151"
BACKEND = Path(__file__).resolve().parents[1]


def _table():
    return I.rocm_kernel_config(GFX, HD)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in (I.INT8_ATTENTION_ENV, I.IMPL_ENV, I.VBLK_ENV):
        monkeypatch.delenv(env, raising = False)
    for name, value in (
        ("_VERDICTS", {}),
        ("_VERIFIED", set()),
        ("_BROKEN", set()),
        ("_LONG_VERIFIED", set()),
        ("_LONG_CHECKS", {}),
        ("_LONG_BROKEN", set()),
        ("_VBLK_VERIFIED", set()),
        ("_VBLK_BROKEN", set()),
    ):
        monkeypatch.setattr(I, name, value)
    stats = dict(I.STATS)
    yield
    I.STATS.clear()
    I.STATS.update(stats)


def _fake_gfx1151(monkeypatch, fail = None):
    """Returns the list of (heads, keys, config) handed to the Triton entry; ``fail(config)`` True raises there."""
    from core.inference import diffusion_int8_attention_triton as T

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(I, "unsupported_reason", lambda q, k, v: None)
    monkeypatch.setattr(I, "implementation", lambda index, head_dim: "triton")
    monkeypatch.setattr(I, "_fallback_card", lambda *a, **k: False)
    monkeypatch.setattr(I, "device_config", lambda index, head_dim, impl = None: _table())
    monkeypatch.setattr(I, "_call_route", lambda q, k, v: (None, 0, "triton", _table()))
    probe = I.probe_inputs
    monkeypatch.setattr(I, "probe_inputs", lambda device, *a, **k: probe("cpu", *a, **k))
    calls = []

    def triton_call(
        q,
        k,
        v,
        config,
        scale = None,
    ):
        calls.append((int(q.shape[2]), int(k.shape[1]), dict(config)))
        if fail is not None and fail(config):
            raise RuntimeError("hipModuleLaunchKernel failed")
        return I.reference_attention(q, k, v).to(v.dtype)

    monkeypatch.setattr(T, "_int8_attention", triton_call)
    return calls


def _qkv(
    heads,
    keys,
    queries = 16,
):
    q, k, v = I.probe_inputs("cpu", torch.float32, keys, heads, HD, seed = heads + keys)
    return q[:, :queries], k, v


def _verify(monkeypatch):
    # the short kernel's own check is out of scope here: record it as passed, then run the real long check
    I._VERIFIED.add((0, HD, "torch.float32", "triton"))
    found = I._long_check("cuda", torch.float32, HD, "triton")
    assert found is not None and found["ok"]
    return found


def test_gfx1151_long_config_names_56_heads_only():
    """Fails on main: the long V^T layout is keyed to 56 heads, nothing else in the table changed."""
    table = _table()
    assert table["long_vblk_heads"] == (56,)
    assert {k: v for k, v in table.items() if k != "long_vblk_heads"} == {
        "bm": 64,
        "bn": 32,
        "warps": 4,
        "stages": 1,
        "pv": "f16_t",
        "kscale": "tile",
        "long_keys": 8000,
        "long": {"bm": 128, "warps": 8},
    }
    assert "long_vblk_heads" not in I.rocm_kernel_config(GFX, 64)
    # the forced-short config drops the long-only key, as it drops long / long_keys
    assert "long_vblk_heads" not in I.select_config(table, 0)
    assert "long_vblk_heads" not in I.select_config(table, 100000)


def test_long_check_records_the_bit_identical_layout(monkeypatch):
    calls = _fake_gfx1151(monkeypatch)
    found = _verify(monkeypatch)
    assert found["vblk_equal"] is True
    assert (0, HD, "torch.float32", "triton") in I._VBLK_VERIFIED
    # the probe's own calls: plain long, the same with vblk (the equality check), then the sharp probe
    assert [c[2].get("vblk") for c in calls] == [None, 1, None]


def test_long_check_rejects_a_layout_that_differs(monkeypatch):
    calls = _fake_gfx1151(monkeypatch)
    from core.inference import diffusion_int8_attention_triton as T

    real = T._int8_attention

    def off_by_one_ulp(
        q,
        k,
        v,
        config,
        scale = None,
    ):
        out = real(q, k, v, config, scale)
        return torch.nextafter(out, out + 1) if config.get("vblk") else out

    monkeypatch.setattr(T, "_int8_attention", off_by_one_ulp)
    found = _verify(monkeypatch)
    assert found["vblk_equal"] is False and not I._VBLK_VERIFIED
    calls.clear()
    I.int8_attention_impl(*_qkv(56, 9000))
    assert [c[2].get("vblk") for c in calls] == [None]


@pytest.mark.parametrize(
    "heads, keys, expect",
    [
        (56, 9000, 1),
        (56, 37720, 1),
        (24, 16800, None),
        (24, 37720, None),
        (32, 24576, None),
        (56, 4096, None),
    ],
)
def test_only_long_56_head_calls_take_the_layout(monkeypatch, heads, keys, expect):
    calls = _fake_gfx1151(monkeypatch)
    _verify(monkeypatch)
    calls.clear()
    assert I.int8_attention_impl(*_qkv(heads, keys)) is not None
    ((h, n, config),) = calls
    assert (h, n) == (heads, keys)
    assert config.get("vblk") == expect
    # everything else is the config main hands the Triton entry (the card's table: its long variant is verified)
    assert {k: v for k, v in config.items() if k != "vblk"} == _table()


def test_switch_off_unverified_and_short_keep_channel_major(monkeypatch):
    calls = _fake_gfx1151(monkeypatch)
    # not verified yet: no layout even for 56 heads
    I._VERIFIED.add((0, HD, "torch.float32", "triton"))
    I._LONG_VERIFIED.add((0, HD, "torch.float32", "triton"))
    I.int8_attention_impl(*_qkv(56, 9000))
    _verify(monkeypatch)
    monkeypatch.setenv(I.VBLK_ENV, "0")
    I.int8_attention_impl(*_qkv(56, 9000))
    monkeypatch.delenv(I.VBLK_ENV)
    I.int8_attention_impl(*_qkv(56, 9000))
    real = [c for c in calls if c[0] == 56]
    assert [c[2].get("vblk") for c in real] == [None, None, 1]


def test_a_failed_layout_launch_drops_only_the_layout(monkeypatch):
    calls = _fake_gfx1151(
        monkeypatch, fail = lambda config: bool(config.get("vblk")) and len(calls) > 3
    )
    _verify(monkeypatch)
    native = I.STATS["native"]
    I.int8_attention_impl(*_qkv(56, 9000))  # raises inside: that call runs native attention
    assert I.STATS["native"] == native + 1
    assert I._VBLK_BROKEN == {(0, HD, "triton")} and not I._LONG_BROKEN and not I._BROKEN
    calls.clear()
    I.int8_attention_impl(*_qkv(56, 9000))
    ((_, _, config),) = calls
    assert (
        config.get("vblk") is None and config["long_keys"] == 8000
    )  # the long kernel still serves


def test_persisted_verdict_restores_the_layout(monkeypatch):
    class Store:
        def get(self, card, key):
            return {
                "ok": True,
                "impl": "triton",
                "ms": {},
                "long": {"ok": True, "vblk_equal": True},
            }

    monkeypatch.setattr(I, "persisted", lambda: Store())
    monkeypatch.setattr(I, "card_name", lambda index: "fake")
    monkeypatch.setattr(I, "_warm", lambda *a, **k: None)
    found = I._adopt_verdict(0, torch.float32, HD, (0, HD, "torch.float32", "triton"))
    assert found and (0, HD, "torch.float32", "triton") in I._VBLK_VERIFIED


def test_the_switch_is_part_of_the_persisted_key(monkeypatch):
    monkeypatch.setattr(I, "_library_versions", I._library_versions)
    from core.inference import diffusion_int8_attention_cuda as C

    monkeypatch.setattr(C, "nvrtc_version", lambda: (12, 8))
    on = I._library_versions()
    monkeypatch.setenv(I.VBLK_ENV, "0")
    assert I._library_versions() != on and on["vblk"] is True


_KERNEL = textwrap.dedent("""
    import os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path.insert(0, sys.argv[1])
    import torch
    from core.inference import diffusion_int8_attention_triton as T
    flip = sys.argv[2] == "flip"
    if flip:
        # a load that ignores the block layout (channel-major addressing on block-stored V^T) must not match
        real = T._make_plan
        def plan(*a, **k):
            p = real(*a, **k)
            if p["prep_meta"]["VB"]:
                p["attn_meta"] = dict(p["attn_meta"], V_BLK = 0)
            return p
        T._make_plan = plan
    torch.manual_seed(0)
    bad = []
    for sk, bn, limit in ((200, 32, None), (130, 32, None), (256, 64, None), (200, 32, 1)):
        if limit:
            T.GRID_LIMIT = limit  # one launch per quantiser
        q, k, v = (torch.randn(1, sk, 3, 128, dtype = torch.bfloat16) for _ in range(3))
        cfg = {"bm": 64, "bn": bn, "warps": 4, "stages": 1, "pv": "f16_t", "kscale": "tile"}
        a = T.int8_attention(q, k, v, cfg)
        b = T.int8_attention(q, k, v, dict(cfg, vblk = 1))
        T._PLANS.clear()
        if not torch.equal(a, b):
            bad.append((sk, bn, limit))
    print("BAD", bad)
""")


@pytest.mark.parametrize("mode", ["plain", "flip"])
def test_block_layout_kernel_is_bit_identical_under_interpret(mode, tmp_path):
    """The kernels themselves (TRITON_INTERPRET, CPU): ragged and even key counts, fused and per-quantiser launches,
    torch.equal to channel-major V^T. Flip: the same check with the block load reverted must fail."""
    pytest.importorskip("triton")
    script = tmp_path / "vblk_interp.py"
    script.write_text(_KERNEL, encoding = "utf-8")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    r = subprocess.run(
        [sys.executable, str(script), str(BACKEND), mode],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    line = [x for x in r.stdout.splitlines() if x.startswith("BAD")][-1]
    if mode == "plain":
        assert line == "BAD []"
    else:
        assert line != "BAD []"
