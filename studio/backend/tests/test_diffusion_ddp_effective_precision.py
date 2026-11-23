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
import os
import sys
from datetime import timedelta
import torch.distributed as dist
from core.training import diffusion_ddp
rank, world, store_file, own = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")[int(sys.argv[1])]
store = dist.FileStore(store_file, world)
store.set_timeout(timedelta(seconds = 120))
# Bounded, so a rank stuck in a collective raises here instead of waiting out Gloo's 30 minute default.
dist.init_process_group(
    "gloo", store = store, rank = rank, world_size = world, timeout = timedelta(seconds = 60)
)
ctx = diffusion_ddp.activate_for_tests(rank, world, store)
try:
    ctx.check_effective_precision(own)
    print("SAME")
except RuntimeError as exc:
    print("REFUSED", exc)
sys.stdout.flush()
# Leave together, then skip interpreter teardown: a rank whose peer had already exited could hang in the Gloo
# group's destructor (CI: both verdicts printed, rank 0 exited, rank 1 never did). DistContext.close() bounds the
# same teardown in the trainer.
dist.barrier()
dist.destroy_process_group()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


@pytest.mark.parametrize("picks, verdict", [("fp8,fp8", "SAME"), ("fp8,bf16", "REFUSED")])
def test_real_gloo_two_ranks(tmp_path, picks, verdict):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(backend), os.environ.get("PYTHONPATH", "")]),
    }
    store = str(tmp_path / "store")
    # Each rank writes to files, not pipes: a rank cannot block on a full pipe while the other is waited on,
    # and its stderr is kept for the failure message.
    logs = [(tmp_path / f"rank{r}.out", tmp_path / f"rank{r}.err") for r in range(2)]
    procs = []
    for r, (out_path, err_path) in enumerate(logs):
        with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CHILD, str(r), "2", store, picks],
                    cwd = str(backend),
                    env = env,
                    stdout = out_f,
                    stderr = err_f,
                    text = True,
                )
            )

    def report():
        return "\n".join(
            f"rank {r} rc={p.returncode}\nstdout:\n{o.read_text()[-2000:]}\nstderr:\n{e.read_text()[-4000:]}"
            for r, (p, (o, e)) in enumerate(zip(procs, logs))
        )

    deadline = time.monotonic() + 180
    try:
        for p in procs:
            p.wait(timeout = max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
        pytest.fail(f"the two Gloo ranks did not finish within 180 s:\n{report()}")
    outs = [o.read_text() for o, _ in logs]
    assert [p.returncode for p in procs] == [0, 0], report()
    # Gloo may print its own connection banner on stdout first; the verdict is the child's last line.
    assert all(o.strip().splitlines()[-1].startswith(verdict) for o in outs), outs


def test_same_label_with_different_converted_layers_is_refused(monkeypatch):
    """Both ranks report fp8, but a per-layer fallback left a different layer bf16 on each."""
    a = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
    b = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Linear(16, 16))
    a[0]._studio_lowp = "fp8"
    b[1]._studio_lowp = "fp8"
    mine, other = diffusion_ddp.lowp_layers_digest(a), diffusion_ddp.lowp_layers_digest(b)
    assert mine[1] == other[1] == 1 and mine[0] != other[0]
    peer = torch.tensor([diffusion_ddp.DistContext._EFFECTIVE_PRECISIONS.index("fp8"), *other])

    def fake_all_gather(out, tensor):
        out[0].copy_(tensor)
        out[1].copy_(peer)

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    ctx = diffusion_ddp.DistContext(0, 2, backend = "gloo")
    with pytest.raises(RuntimeError, match = "different layers"):
        ctx.check_effective_precision("fp8", layers = mine)
    # Same converted set: goes on.
    peer = torch.tensor([diffusion_ddp.DistContext._EFFECTIVE_PRECISIONS.index("fp8"), *mine])
    ctx.check_effective_precision("fp8", layers = mine)


def test_trainer_passes_the_converted_layer_digest():
    src = inspect.getsource(dit)
    i = src.index("ddp.check_effective_precision(")
    assert "layers = lowp_layers_digest(transformer)" in src[i : i + 300]


_CHILD_LAYERS = """
import sys
from datetime import timedelta
from unittest.mock import patch
import torch
import torch.distributed as dist
from core.training import diffusion_ddp
from core.training import diffusion_frozen_fp8 as f8
rank, world, store_file, fail = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
store = dist.FileStore(store_file, world)
store.set_timeout(timedelta(seconds = 120))
dist.init_process_group("gloo", store = store, rank = rank, world_size = world)
ctx = diffusion_ddp.activate_for_tests(rank, world, store)
torch.manual_seed(0)
model = torch.nn.Sequential(*[torch.nn.Linear(32, 32, bias = False).bfloat16() for _ in range(2)]).requires_grad_(False)
quant = f8._quantize_weight
def inject(w):
    if fail and rank == 1 and w.data_ptr() == model[1].weight.data_ptr():
        raise RuntimeError("injected per-layer conversion failure")
    return quant(w)
with patch.object(f8, "_quantize_weight", side_effect = inject):
    done, failed = f8.convert_frozen_fp8(model, ["0", "1"])
try:
    ctx.check_effective_precision("fp8", layers = diffusion_ddp.lowp_layers_digest(model))
    print("SAME", done, failed)
except RuntimeError as exc:
    print("REFUSED", done, failed, exc)
"""


@pytest.mark.parametrize("fail, verdict", [("0", "SAME"), ("1", "REFUSED")])
def test_real_gloo_rank_local_layer_failure(tmp_path, fail, verdict):
    """Real two-rank Gloo run through the production fp8 converter: rank 1 keeps one layer bf16."""
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
            [sys.executable, "-c", _CHILD_LAYERS, str(r), "2", store, fail],
            cwd = str(backend),
            env = env,
            stdout = subprocess.PIPE,
            text = True,
        )
        for r in range(2)
    ]
    outs = [p.communicate(timeout = 180)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outs
    last = [o.strip().splitlines()[-1] for o in outs]
    assert all(line.startswith(verdict) for line in last), outs
    if fail == "1":
        assert last[0].split()[1:3] == ["2", "0"] and last[1].split()[1:3] == ["1", "1"], outs
