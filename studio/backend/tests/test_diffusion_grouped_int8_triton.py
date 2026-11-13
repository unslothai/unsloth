# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton decode of packed W4A8 / W6A8 rows: the same int8 grid as the torch decode, bit for bit."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_grouped_int8 as gi  # noqa: E402
import core.inference.diffusion_grouped_int8_triton as gt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CASES = textwrap.dedent(
    """
    import torch


    def _cases(device = "cpu"):
        # (bits, group, codebook, rows, k): odd row counts, both groups, codebook and uniform codes
        g = torch.Generator().manual_seed(0)
        out = []
        for bits in (4, 6):
            for group in (16, 32):
                for with_book in ((False, True) if bits == 4 else (False,)):
                    rows, k = 37, 768
                    codes = torch.randint(0, 1 << bits, (rows, k), generator = g)
                    raw = torch.randint(0, 256, (rows, k // group), generator = g)
                    raw[(raw & 0x7F) == 0x7F] = 0x38  # no NaN bytes
                    raw[0, :6] = torch.tensor([0x00, 0x01, 0x07, 0x80, 0x81, 0x7E])  # 0, subnormals, -0, max
                    raw[1, :] = 0x30  # 0.5: every odd level lands exactly on a .5 tie
                    book = None
                    if with_book:
                        book = torch.sort(torch.randn(16, generator = g))[0].float()
                        book[3], book[4], book[5] = 0.5, 1.5, -2.5
                    out.append((bits, group, codes, raw.to(torch.uint8), book))
        return out
    """
)
exec(_CASES)  # noqa: S102 -- the same cases run in the interpreter subprocess below


def test_the_module_imports_without_torch():
    code = (
        "import sys; sys.path.insert(0, %r); import core.inference.diffusion_grouped_int8_triton as m; "
        "print('torch' in sys.modules, 'triton' in sys.modules)" % ROOT
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd = ROOT, capture_output = True, text = True, timeout = 120
    )
    assert proc.stdout.strip() == "False False", proc.stdout + proc.stderr[-2000:]


def test_cpu_tensors_and_bad_inputs_are_declined(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising = False)
    bits, group, codes, raw, book = _cases()[0]
    q = gi.pack_codes(codes, bits)
    assert gt.decode_int8_triton(q, raw, book, group, bits) is None
    assert gt.decode_int8_triton(q, raw, book, group, 5) is None
    assert gt.decode_int8_triton(q[:, :-2], raw, book, group, bits) is None


def test_the_op_falls_back_to_the_torch_decode_on_cpu_and_has_a_shape_only_fake(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising = False)
    op = gt.decode_int8_op()
    assert gt.decode_int8_op() is op
    for bits, group, codes, raw, book in _cases():
        q = gi.pack_codes(codes, bits)
        assert torch.equal(
            op(q, raw, book, group, bits), gi.decode_int8_torch(q, raw, book, group, bits)
        )
    from torch._subclasses.fake_tensor import FakeTensorMode

    bits, group, codes, raw, book = _cases()[-1]
    q = gi.pack_codes(codes, bits)
    with FakeTensorMode() as mode:
        fake = op(mode.from_tensor(q), mode.from_tensor(raw), None, group, bits)
    assert tuple(fake.shape) == tuple(codes.shape) and fake.dtype == torch.int8


_INTERPRETED = textwrap.dedent(
    """
    import sys, torch
    sys.path.insert(0, {root!r})
    import core.inference.diffusion_grouped_int8 as gi
    import core.inference.diffusion_grouped_int8_triton as gt
    exec({cases!r})
    bad = []
    for bits, group, codes, raw, book in _cases():
        q = gi.pack_codes(codes, bits)
        got = gt.decode_int8_triton(q, raw, book, group, bits)
        assert got is not None, gt._STATE.get("error")
        if not torch.equal(got, gi.decode_int8_torch(q, raw, book, group, bits)):
            bad.append((bits, group, book is not None))
    print("BAD", bad)
    """
)


def test_the_triton_decode_is_bit_identical_under_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", _INTERPRETED.format(root = ROOT, cases = _CASES)],
        cwd = ROOT,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert "BAD []" in proc.stdout, proc.stdout + proc.stderr[-2000:]


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "needs a CUDA / ROCm GPU to run the Triton decode"
)
def test_the_triton_decode_on_the_gpu_matches_the_torch_decode():
    pytest.importorskip("triton")
    for bits, group, codes, raw, book in _cases():
        q = gi.pack_codes(codes, bits)
        want = gi.decode_int8_torch(q, raw, book, group, bits)
        got = gt.decode_int8_triton(
            q.cuda(), raw.cuda(), None if book is None else book.cuda(), group, bits
        )
        assert got is not None, gt._STATE.get("error")
        assert torch.equal(got.cpu(), want), (bits, group, book is not None)
        dev = gi.decode_int8_torch(
            q.cuda(), raw.cuda(), None if book is None else book.cuda(), group, bits
        )
        assert torch.equal(dev.cpu(), want)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "needs a CUDA / ROCm GPU to run the Triton decode"
)
def test_the_op_is_one_graph_under_compile():
    from torch._dynamo.testing import CompileCounterWithBackend

    op = gt.decode_int8_op()
    bits, group, codes, raw, book = _cases()[-1]
    q, s = gi.pack_codes(codes, bits).cuda(), raw.cuda()
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("inductor")
    fn = torch.compile(
        lambda a, b: op(a, b, None, group, bits).to(torch.int32) * 2,
        backend = counter,
        fullgraph = True,
    )
    assert torch.equal(fn(q, s), gi.decode_int8_torch(q, s, None, group, bits).to(torch.int32) * 2)
    assert counter.frame_count == 1
