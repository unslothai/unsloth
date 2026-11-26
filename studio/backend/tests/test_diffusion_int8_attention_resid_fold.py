# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The INT8 attention sink path adds its clipped keys back inside the attention kernel (RFOLD): one launch and one
pass over the output less per call, ``UNSLOTH_DIFFUSION_INT8_ATTENTION_RESID_FOLD=0`` keeps unsloth_i8resid."""

from __future__ import annotations

import ctypes
import re

import pytest

from core.inference import diffusion_int8_attention as I
from core.inference import diffusion_int8_attention_cuda as K


def _nvrtc_archs() -> set:
    try:
        lib = K.nvrtc()
    except Exception:  # noqa: BLE001
        return set()
    count = ctypes.c_int()
    if lib.nvrtcGetNumSupportedArchs(ctypes.byref(count)) != 0:
        return set()
    archs = (ctypes.c_int * count.value)()
    if lib.nvrtcGetSupportedArchs(archs) != 0:
        return set()
    return set(archs)


def _params(ptx: bytes) -> int:
    return len(set(re.findall(r"unsloth_i8attn_param_\d+", ptx.decode(errors = "replace"))))


@pytest.mark.skipif(not _nvrtc_archs(), reason = "NVRTC ships with the CUDA builds of torch")
def test_every_card_config_compiles_the_folded_sink_kernel():
    """Each measured card config and head dim builds the folded kernel: the sink kernel's inputs plus the clipped
    keys' indices, residuals and row stride."""
    archs = _nvrtc_archs()
    built = 0
    for cap, by_dim in I.KERNEL_CONFIGS.items():
        arch = cap[0] * 10 + cap[1]
        if arch not in archs:
            continue
        for head_dim, config in by_dim.items():
            for keys in (0, 1 << 20):
                defines = I.kernel_defines(head_dim, I.select_config(config, keys))
                sink = K.compile_ptx(K.KERNEL_SOURCE, f"sm_{arch}", dict(defines, SINKS = 1))
                fold = K.compile_ptx(K.KERNEL_SOURCE, f"sm_{arch}", dict(defines, SINKS = 1, RFOLD = 1))
                assert _params(fold) == _params(sink) + 3, (cap, head_dim)
                built += 1
    assert built


def test_the_fold_stages_its_keys_inside_the_tile_buffers():
    """RF clipped keys (int8 K row, fp32 residual row, K scale) fit the free tile buffers of every config."""
    for by_dim in I.KERNEL_CONFIGS.values():
        for head_dim, config in by_dim.items():
            stages = int(config["stages"]) * (
                int(config["bn"]) * (head_dim + 16) + head_dim * (int(config["bn"]) + 16)
            )
            rf = 32 if stages >= 32 * (5 * head_dim + 4) else 16
            assert rf * (5 * head_dim + 4) <= stages


class _Fn:
    def __init__(self, name, seen):
        self.name, self.seen = name, seen

    def launch(self, grid, threads, args, shared, stream):
        self.seen.append((self.name, len(args), args))


def _plan(fold: bool, seen: list) -> dict:
    names = ("qi", "ki", "vt", "qs", "ks", "vs", "vg", "vc", "rcnt", "ridx", "rres", "rml")
    return {
        "off": {name: 1000 * (i + 1) for i, name in enumerate(names)},
        "attn_tail": (7, 16, 9, 128, 2, 1, 2, 3, 1),
        "attn_grid": (1, 2),
        "attn_threads": 384,
        "shared": 4096,
        "resid_tail": (7, 16, 128, 2, 5, 1, 2, 3, 1),
        "resid_grid": (2, 1),
        "rstride": 5,
        "fold": fold,
        "kernels": (None, None, _Fn("attn", seen), _Fn("resid", seen)),
        "kernels0": (None, None, _Fn("attn0", seen)),
    }


def test_a_folded_sink_call_is_one_launch():
    seen: list = []
    I._launch_attention(_plan(True, seen), 0, 77, 0, True)
    assert [s[0] for s in seen] == ["attn"]
    args = seen[0][2]
    # the sink kernel's pointers, then the clipped keys' indices and residuals and their row stride
    assert args[-5:] == (9000, 12000, 10000, 11000, 5)


def test_the_kill_switch_keeps_the_residual_launch(monkeypatch):
    seen: list = []
    I._launch_attention(_plan(False, seen), 0, 77, 0, True)
    assert [s[0] for s in seen] == ["attn", "resid"]
    assert seen[0][2][-2:] == (9000, 12000)
    monkeypatch.setenv(I.RESID_FOLD_ENV, "0")
    assert not I.resid_fold_enabled()
    monkeypatch.delenv(I.RESID_FOLD_ENV)
    assert I.resid_fold_enabled()


def test_without_sinks_nothing_changes():
    seen: list = []
    I._launch_attention(_plan(True, seen), 0, 77, 0, False)
    assert [s[0] for s in seen] == ["attn0"]


def test_a_warm_plan_follows_the_kill_switch(monkeypatch):
    """The switch flipped in a running process reaches the next call on a shape that already has a plan: the plan
    lookup is keyed on it (the kernel cache always was), so a warm shape does not keep the folded kernel."""
    torch = pytest.importorskip("torch")
    built: list = []

    def functions(
        index,
        head_dim,
        config,
        sinks = False,
        fold = None,
    ):
        built.append((sinks, fold))
        kernels = {
            name: name for name in ("unsloth_i8stats_q", "unsloth_i8quant_kv", "unsloth_i8resid")
        }
        return ("attn", kernels, 4096)

    monkeypatch.setattr(I, "_functions", functions)
    monkeypatch.delenv(I.RESID_FOLD_ENV, raising = False)
    I.forget_plans()
    config = next(iter(I.KERNEL_CONFIGS.values()))[128]
    q, k, v = (torch.zeros(1, 256, 2, 128, dtype = torch.bfloat16) for _ in range(3))
    try:
        on = I._find_plan(q, k, v, config, None, 0, I.MODE_KERNELS)[0]
        assert on["fold"] and I._find_plan(q, k, v, config, None, 0, I.MODE_KERNELS)[0] is on
        monkeypatch.setenv(I.RESID_FOLD_ENV, "0")
        off = I._find_plan(q, k, v, config, None, 0, I.MODE_KERNELS)[0]
        assert off is not on and not off["fold"]
        monkeypatch.delenv(I.RESID_FOLD_ENV)
        assert I._find_plan(q, k, v, config, None, 0, I.MODE_KERNELS)[0] is on
        assert [f for s, f in built if s] == [True, False]
    finally:
        I.forget_plans()


def test_a_flipped_switch_races_its_own_kernel(monkeypatch):
    """The plan's sink-path verdict (accuracy + timing) belongs to its kernel: after the fold switch flips, the new
    plan races again instead of inheriting the folded kernel's verdict."""
    import contextlib

    torch = pytest.importorskip("torch")

    def functions(
        index,
        head_dim,
        config,
        sinks = False,
        fold = None,
    ):
        kernels = {
            name: name for name in ("unsloth_i8stats_q", "unsloth_i8quant_kv", "unsloth_i8resid")
        }
        return ("attn", kernels, 4096)

    class _Done:
        def synchronize(self):
            pass

    def decide(plan, verdict, races):
        plan["calls"] = I.SINK_DECIDE_AT
        plan["checks"] = [(_Done(), torch.ones(2, dtype = torch.int32), False)]
        return I._choose_path(plan, race = lambda: races.append(verdict) or verdict)[0]

    monkeypatch.setattr(I, "_functions", functions)
    monkeypatch.setattr(I, "_outside_capture", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(I, "_drop_routes", lambda plan: None)
    monkeypatch.setattr(I, "_GRAPHED", [False])
    monkeypatch.delenv(I.RESID_FOLD_ENV, raising = False)
    I.forget_plans()
    config = next(iter(I.KERNEL_CONFIGS.values()))[128]
    q, k, v = (torch.zeros(1, 256, 2, 128, dtype = torch.bfloat16) for _ in range(3))
    races: list = []
    try:
        on = I._find_plan(q, k, v, config, None, 0, I.MODE_KERNELS)[0]
        assert on["fold"] and decide(on, True, races) is True
        monkeypatch.setenv(I.RESID_FOLD_ENV, "0")
        off = I._find_plan(q, k, v, config, None, 0, I.MODE_KERNELS)[0]
        assert not off["fold"] and off["decision"] != on["decision"]
        assert decide(off, False, races) is False
        assert races == [True, False]
    finally:
        I.forget_plans()
