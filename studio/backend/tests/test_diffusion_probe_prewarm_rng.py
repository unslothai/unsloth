# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cold-card probe warm-up draws its operands from private seeded generators: it runs before the user's renders,
so it must leave the default host and CUDA RNG exactly where they were. CPU only (launches and the card faked)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_convrot_quant as cq


def _no_launches(monkeypatch):
    monkeypatch.setattr(cq, "_rotq_launch", lambda x, group, kind, cfg: None)
    seen = []
    monkeypatch.setattr(cq, "_h16_launch", lambda x, kind, v = None, tf32x3 = None: seen.append((x, v)))
    monkeypatch.setattr(cq, "_h16x_launch", lambda prefix, gate, value, kind, tf32x3 = None: None)
    return seen


def test_warm_operands_leave_the_default_cpu_rng_unchanged_and_repeat(monkeypatch):
    seen = _no_launches(monkeypatch)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    a = cq._warm_input(65, 512, "cpu")
    cq._warm_rotq("cpu", 65, 512, "v2", (64, 4, 2))
    cq._warm_h16("cpu", 65, 512, "v1", False, True)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(a, cq._warm_input(65, 512, "cpu"))  # seeded: the same operand every time
    x, v = seen[-1]
    assert not torch.equal(x, v)  # the SwiGLU gate and value differ
    # what the user's next seeded draw sees is what it saw without a warm-up
    torch.manual_seed(123)
    expected = torch.randn(4)
    torch.manual_seed(123)
    cq._warm_input(65, 512, "cpu")
    assert torch.equal(torch.randn(4), expected)


def test_every_warm_job_on_a_cuda_card_draws_from_a_private_generator(monkeypatch):
    # Fake card: every CUDA draw is redirected to the host, and each must carry its own generator on the job's device.
    _no_launches(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(cq, "is_rocm_torch", lambda t: False)
    monkeypatch.setattr(cq, "_triton_ok", lambda: True)
    monkeypatch.setattr(cq, "_kernels", lambda: object())
    for name in ("_rotq_op", "_swiglu_op", "_producer_op"):
        monkeypatch.setattr(cq, name, lambda: "op")
    real_gen, real_randn = torch.Generator, torch.randn
    gens, draws = [], []

    def generator(device = "cpu"):
        gens.append(str(device))
        return real_gen(device = "cpu")

    def randn(
        *size,
        device = None,
        generator = None,
        **kw,
    ):
        draws.append((str(device), generator is not None))
        return real_randn(*size, generator = generator, **kw)

    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(torch, "randn", randn)
    # the producer job builds its inputs on the host, then moves them: keep them on the host here
    real_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **k: real_to(
            self,
            *[x for x in a if not (isinstance(x, torch.device) and x.type == "cuda")],
            **{kk: vv for kk, vv in k.items() if kk != "device"},
        ),
    )
    jobs = cq.probe_warm_jobs(0, rotq = True, producer = True)
    assert jobs
    torch.manual_seed(7)
    before = torch.get_rng_state()
    for job in jobs:
        try:
            job()
        except Exception:  # noqa: BLE001 - a faked launch path may not run whole; only the draws are under test
            pass
    assert torch.equal(torch.get_rng_state(), before)
    cuda_draws = [d for d in draws if d[0].startswith("cuda")]
    assert cuda_draws and all(private for _, private in cuda_draws)
    assert "cuda:0" in gens
