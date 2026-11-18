# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RDNA W8A8 Triton kernels: gating on CPU / CUDA hosts, tile planning, and the kernels' maths in the Triton
interpreter (fp16: the interpreter has no bf16 dot)."""

from __future__ import annotations

import os
import types
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq
import core.inference.diffusion_native_w8a8_rdna as rdna

BACKEND = Path(__file__).resolve().parents[1]


def test_tile_plan_whole_rows_per_program():
    assert rdna._rotq_tile(3072, 256) == (256, 12, 64, 5)
    assert rdna._rotq_tile(12288, 256) == (256, 48, 64, 1)
    assert rdna._rotq_tile(256 * 100, 256) == (256, 100, 128, 1)
    assert rdna._rotq_tile(512, 0) == (256, 2, 64, 32)
    assert rdna._rotq_tile(256 * 129, 256) is None
    assert rdna._rotq_tile(3000, 256) is None


def test_shape_supported_needs_aligned_features():
    assert rdna.shape_supported(3072, 3072, 256)
    assert not rdna.shape_supported(3072, 3000, 256)
    assert not rdna.shape_supported(3000, 3072, 0)


@pytest.mark.parametrize("group", [8, 16, 32, 64, 128, 512])
def test_rotation_groups_the_probe_does_not_check_keep_the_stock_path(group):
    # Group 16 with BLOCK_K 32 ran the rotation loop zero times and quantised every activation to zero.
    assert rdna._rotq_tile(3072, group) is None
    assert not rdna.shape_supported(3072, 512, group)


def test_prepare_off_rocm_and_kill_switch(monkeypatch):
    monkeypatch.setattr(
        rdna, "_STATE", {"devices": {}, "on": {}, "broken": None, "rotq": None, "mm": None}
    )
    if not getattr(torch.version, "hip", None):
        assert rdna.prepare() is False
        assert rdna._STATE["devices"] == {}
    monkeypatch.setenv(rdna.RDNA_TRITON_ENV, "0")
    assert rdna.prepare() is False
    assert rdna.enabled_for(0) is False


def test_cpu_layer_keeps_stock_path():
    lin = torch.nn.Linear(512, 256, dtype = torch.float32)
    layer = nq.native_linear_class()(lin, nq.NATIVE_INT8, act_int8 = True, rot_group = 256)
    # ``_rdna`` follows the card's probe (a CPU-built layer may move to the GPU); a CPU input never engages it.
    x = torch.randn(4, 512)
    assert layer._rdna_on(x) is False
    assert layer(x).shape == (4, 256)


def test_enabled_for_reads_only_prepared_cards(monkeypatch):
    monkeypatch.setattr(
        rdna, "_STATE", {"devices": {}, "on": {1: True}, "broken": None, "rotq": None, "mm": None}
    )
    assert rdna.enabled_for(1) is True
    assert rdna.enabled_for(0) is False


_INTERP = textwrap.dedent(
    """
    import contextlib, os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path.insert(0, sys.argv[1])
    import torch
    from core.inference import diffusion_native_w8a8_rdna as r
    torch.cuda.device = lambda d: contextlib.nullcontext()
    torch.manual_seed(0)
    for group, k, m in ((256, 768, 70), (256, 3072, 33), (0, 512, 40)):
        x = (torch.randn(m, k) * 3).half()
        x[3] = 0
        q, s = r._rotq_launch(x, group)
        rq, rs = r.reference_rotq(x, group)
        assert torch.equal(q, rq), (group, k, m)
        assert torch.equal(s, rs), (group, k, m)
        wq = torch.randint(-127, 128, (48, k), dtype = torch.int8)
        ws = torch.rand(48) * 1e-2 + 1e-4
        for bias in (torch.randn(48).half(), None):
            out = r._mm_launch(rq, rs, wq, ws, bias, torch.float16)
            ref = r.reference_mm(rq, rs, wq, ws, bias, torch.float16)
            assert torch.equal(out, ref), (group, k, m, bias is None)
    print("INTERP_OK")
    """
)


def test_kernels_match_reference_in_interpreter(tmp_path):
    pytest.importorskip("triton")
    script = tmp_path / "interp.py"
    script.write_text(_INTERP)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, str(script), str(BACKEND)],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
    )
    if (
        "INTERP_OK" not in proc.stdout
        and "interpreter" in proc.stderr.lower()
        and "not supported" in proc.stderr.lower()
    ):
        pytest.skip("Triton interpreter unavailable")
    assert "INTERP_OK" in proc.stdout, proc.stderr[-3000:]


def test_failed_launch_falls_back_to_stock_maths(monkeypatch):
    monkeypatch.setattr(
        rdna, "_STATE", {"devices": {}, "on": {}, "broken": None, "rotq": None, "mm": None}
    )
    monkeypatch.setattr(rdna, "_kernels", lambda: types.SimpleNamespace(triton = None))

    def boom(*a, **k):
        raise RuntimeError("PassManager::run failed")

    monkeypatch.setattr(rdna, "_launch_rotq", boom)
    monkeypatch.setattr(rdna, "_launch_mm", boom)
    x = torch.randn(40, 512, dtype = torch.float32)
    q, s = rdna._rotq_launch(x, 256)
    rq, rs = rdna.reference_rotq(x, 256)
    assert torch.equal(q, rq) and torch.equal(s, rs)
    assert "rotate+quant" in rdna._STATE["broken"]
    wq = torch.randint(-127, 128, (32, 512), dtype = torch.int8)
    ws = torch.rand(32) * 1e-2
    b = torch.randn(32)
    out = rdna._mm_launch(rq, rs, wq, ws, b, torch.float32)
    torch.testing.assert_close(
        out, rdna.reference_mm(rq, rs, wq, ws, b, torch.float32), rtol = 1e-5, atol = 1e-4
    )
    assert set(rdna._STATE) == {"devices", "on", "broken", "rotq", "mm"}


def test_reduction_tile_meets_the_wmma_minimum():
    assert rdna._MIN_ROWS_P2 >= 16


def test_tile_overrides_parse_or_ignore():
    assert rdna._tile_overrides('{"rotq": [32, 64, 8, 1]}') == {"rotq": (32, 64, 8, 1)}
    assert rdna._tile_overrides('{"mm_large": [128, 256, 64, 8, 8, 2]}') == {
        "mm_large": (128, 256, 64, 8, 8, 2)
    }
    assert rdna._tile_overrides('{"rotq": [32, 64]}') == {}
    assert rdna._tile_overrides("not json") == {}
    assert rdna._tile_overrides(None) == {}


def test_gemm_tile_follows_rows_and_k():
    assert rdna._mm_cfg(512, 3072) == rdna._CFG["mm_small"]
    assert rdna._mm_cfg(4096, 3072) == rdna._CFG["mm_large_short_k"]
    assert rdna._mm_cfg(4096, 12288) == rdna._CFG["mm_large"]
    assert rdna._tile_overrides('{"mm_large_short_k": [64, 64, 128, 8, 4, 2]}') == {
        "mm_large_short_k": (64, 64, 128, 8, 4, 2)
    }
