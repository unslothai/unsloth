# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-GPU DiT LoRA: the ranks agree on a base precision up front, but the fp8 / W8A8 conversion fallbacks run per
rank. After converting, every rank compares the base it actually trains (one int64 all_gather) and all of them refuse
when they differ. Mocked process group: each rank sees the codes the other ranks would send."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch
import torch.distributed as dist

from core.training import diffusion_ddp
from core.training import diffusion_dit_trainer as dit


def _with_peers(monkeypatch, peers):
    """all_gather fills rank r's slot with peers[r] (the caller's own slot comes from its tensor)."""
    names = diffusion_ddp.DistContext._EFFECTIVE_PRECISIONS
    calls = []

    def fake_all_gather(out, tensor):
        calls.append(tensor.clone())
        for r, t in enumerate(out):
            t.copy_(tensor if peers[r] is None else torch.tensor([names.index(peers[r])]))

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    return calls


@pytest.mark.parametrize("rank", [0, 1])
def test_ranks_on_the_same_base_go_on(monkeypatch, rank):
    calls = _with_peers(monkeypatch, [None, "fp8"] if rank == 0 else ["fp8", None])
    ctx = diffusion_ddp.DistContext(rank, 2, backend = "gloo")
    ctx.check_effective_precision("fp8")
    assert len(calls) == 1 and calls[0].dtype == torch.int64 and calls[0].numel() == 1


@pytest.mark.parametrize(
    "own, other",
    [("fp8", "bf16"), ("fp8", "nf4"), ("int8_w8a8", "int8"), ("bf16", "mxfp8")],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_every_rank_refuses_a_mixed_base(monkeypatch, rank, own, other):
    _with_peers(monkeypatch, [None, other] if rank == 0 else [other, None])
    ctx = diffusion_ddp.DistContext(rank, 2, backend = "gloo")
    with pytest.raises(RuntimeError, match = "different base precisions") as exc:
        ctx.check_effective_precision(own)
    assert own in str(exc.value) and other in str(exc.value)


def test_trainer_checks_after_the_conversions_and_before_the_broadcast():
    src = inspect.getsource(dit)
    tree = ast.parse(src)
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            lines.setdefault(name, []).append(node.lineno)
    check = lines.get("check_effective_precision")
    assert check, "the trainer never compares the converted base across ranks"
    train_fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "check_effective_precision"
    )
    # Every conversion with a rank-local fallback sits before the check, in the same trainer function.
    for conv in (
        "_apply_fp8_training",
        "_apply_int8_w8a8",
        "_apply_mxfp8_training",
        "_int8_quantize_base",
    ):
        before = [n for n in lines.get(conv, []) if n < train_fn.lineno]
        assert before, conv
    broadcasts = [n for n in lines.get("broadcast_params", []) if n > min(check)]
    assert broadcasts, "the check must run before the LoRA broadcast"
    seg = src.splitlines()[min(check) - 1 : min(check) + 3]
    assert any("int8_w8a8" in line and "w8a8_on" in line for line in seg)


_CHILD = """
import sys
from datetime import timedelta
import torch.distributed as dist
from core.training import diffusion_ddp
rank, world, store_file, own = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")[int(sys.argv[1])]
store = dist.FileStore(store_file, world)
store.set_timeout(timedelta(seconds = 120))
dist.init_process_group("gloo", store = store, rank = rank, world_size = world)
ctx = diffusion_ddp.activate_for_tests(rank, world, store)
try:
    ctx.check_effective_precision(own)
    print("SAME")
except RuntimeError as exc:
    print("REFUSED", exc)
"""


@pytest.mark.parametrize("picks, verdict", [("fp8,fp8", "SAME"), ("fp8,bf16", "REFUSED")])
def test_real_gloo_two_ranks(tmp_path, picks, verdict):
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(backend), os.environ.get("PYTHONPATH", "")]),
    }
    store = str(tmp_path / "store")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(r), "2", store, picks],
            cwd = str(backend),
            env = env,
            stdout = subprocess.PIPE,
            text = True,
        )
        for r in range(2)
    ]
    outs = [p.communicate(timeout = 180)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0]
    # Gloo may print its own connection banner on stdout first; the verdict is the child's last line.
    assert all(o.strip().splitlines()[-1].startswith(verdict) for o in outs), outs
