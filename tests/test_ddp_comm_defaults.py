# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""DDP comm defaults on a real 2-rank gloo accelerate DDP must match one process on the joined batch."""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest
import torch

pytest.importorskip("accelerate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # spawned ranks inherit sys.path
import _ddp_comm_defaults_worker as worker  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "two gloo ranks over TCP localhost")


def _bf16_hook_supported():
    # Mirrors DistributedDataParallel._check_comm_hook: a torch build check, not a GPU or
    # backend check. CPU wheels refuse bf16_compress_hook with TypeError even on gloo (#14065 CI).
    try:
        import torch.distributed as dist
        built = torch.version.cuda is not None or torch.version.hip is not None
        return built and dist.is_nccl_available() and torch.cuda.nccl.version() >= (2, 10)
    except Exception:
        return False


CASES = [
    ("full", "full", {}, {}),
    ("full_off", "full", {"UNSLOTH_DDP_GRADIENT_AS_BUCKET_VIEW": "0"}, {}),
    ("peft", "peft", {}, {}),
    ("fp16_hook", "peft", {"UNSLOTH_DDP_COMM_HOOK": "fp16"}, {}),
]
_HAS_LAZY = (
    "lazy_bucket_allocation"
    in __import__("inspect")
    .signature(torch.nn.parallel.DistributedDataParallel.__init__)
    .parameters
)
if _HAS_LAZY:
    CASES.append(("full_lazy", "full", {"UNSLOTH_DDP_LAZY_BUCKETS": "1"}, {}))
    CASES.append(("peft_lazy", "peft", {"UNSLOTH_DDP_LAZY_BUCKETS": "1"}, {}))
if _bf16_hook_supported():
    CASES.append(("bf16_hook", "full", {"UNSLOTH_DDP_COMM_HOOK": "bf16"}, {}))
_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


@pytest.fixture(scope = "module")
def ranks():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = worker.run_rank, args = (rank, port, queue, CASES)) for rank in range(2)
    ]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout = 600) for _ in procs)
    for rank, out in results.items():
        assert "error" not in out, out["error"]
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    return results


def _reference(kind):
    """Single process, both ranks' samples: the mean of two equal-size rank means is the global mean."""
    model = worker.build(kind)
    opt = torch.optim.SGD(model.parameters(), lr = 0.1)
    xs, ys = zip(*(worker.batch(rank) for rank in range(2)))
    loss = worker.loss_fn(model, torch.cat(xs), torch.cat(ys))
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    opt.step()
    return grads, [p.detach().clone() for p in model.parameters()]


def _assert_parity(
    case,
    kind,
    exact = True,
):
    ref_grads, ref_params = _reference(kind)
    tol = {} if exact else dict(rtol = 2e-2, atol = 2e-3)
    for g, r in zip(case["grads"], ref_grads):
        torch.testing.assert_close(
            torch.tensor(g), r, **({"rtol": 1e-6, "atol": 1e-7} if exact else tol)
        )
    for p, r in zip(case["params"], ref_params):
        torch.testing.assert_close(
            torch.tensor(p), r, **({"rtol": 1e-6, "atol": 1e-7} if exact else tol)
        )


def test_full_finetune_gets_bucket_view(ranks):
    for rank in (0, 1):
        case = ranks[rank]["full"]
        assert case["installed"] and case["is_ddp"]
        assert case["view"] is True
        assert case["n_grad_storages"] == 1
        assert case["hooks"] == 0
        _assert_parity(case, "full")


def test_kill_switch_restores_torch_default_bitwise(ranks):
    for rank in (0, 1):
        on, off = ranks[rank]["full"], ranks[rank]["full_off"]
        assert off["view"] is False
        assert off["n_grad_storages"] > 1
        assert on["losses"] == off["losses"]
        assert on["grads"] == off["grads"]
        assert on["params"] == off["params"]


def test_lora_keeps_gradients_unaliased(ranks):
    case = ranks[0]["peft"]
    assert case["view"] is False
    _assert_parity(case, "peft")


def test_comm_hook_opt_in(ranks):
    hook_cases = [(name, kind) for name, kind, *_ in CASES if name.endswith("_hook")]
    for name, kind in hook_cases:
        case = ranks[0][name]
        assert case["hooks"] >= 1, name
        _assert_parity(case, kind, exact = False)
    assert ranks[0]["full"]["hooks"] == 0


@pytest.mark.skipif(not _HAS_LAZY, reason = "torch < 2.15: DDP has no lazy_bucket_allocation")
def test_lazy_buckets_opt_in_is_bitwise(ranks):
    for rank in (0, 1):
        on, off = ranks[rank]["full_lazy"], ranks[rank]["full"]
        assert on["lazy"] is True and off["lazy"] is False
        assert on["view"] is True  # still one flat gradient storage per bucket
        assert on["losses"] == off["losses"]
        assert on["grads"] == off["grads"]
        assert on["params"] == off["params"]
        peft_on, peft_off = ranks[rank]["peft_lazy"], ranks[rank]["peft"]
        assert peft_on["lazy"] is True and peft_off["lazy"] is False
        assert peft_on["grads"] == peft_off["grads"] and peft_on["params"] == peft_off["params"]


def test_lazy_buckets_kwarg_composes_with_other_handler_subclasses(monkeypatch):
    from accelerate.utils import DistributedDataParallelKwargs

    defaults = worker.load_defaults()
    monkeypatch.setenv("UNSLOTH_DDP_LAZY_BUCKETS", "1")

    class Other(DistributedDataParallelKwargs):  # e.g. _utils.py's batched_grad_copy subclass
        def to_kwargs(self):
            return {**super().to_kwargs(), "batched_grad_copy": True}

    handler = defaults._lazy_buckets_handler(Other())
    assert handler.to_kwargs() == {"batched_grad_copy": True, "lazy_bucket_allocation": True}
    assert defaults._lazy_buckets_handler(handler) is handler  # idempotent
    plain = defaults._lazy_buckets_handler(
        DistributedDataParallelKwargs(gradient_as_bucket_view = True)
    )
    assert plain.to_kwargs() == {"gradient_as_bucket_view": True, "lazy_bucket_allocation": True}


def test_lazy_buckets_reason(monkeypatch):
    defaults = worker.load_defaults()
    monkeypatch.delenv("UNSLOTH_DDP_LAZY_BUCKETS", raising = False)
    assert defaults._ddp_lazy_buckets_reason() == "off"
    monkeypatch.setenv("UNSLOTH_DDP_LAZY_BUCKETS", "1")
    if _HAS_LAZY:
        assert defaults._ddp_lazy_buckets_reason() is None
    else:
        assert "lazy_bucket_allocation" in defaults._ddp_lazy_buckets_reason()


def test_trainer_init_installs_the_defaults():
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    init = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_trainer_init"
    ]
    assert len(init) == 1
    calls = [ast.unparse(n.func) for n in ast.walk(init[0]) if isinstance(n, ast.Call)]
    assert "install_unsloth_ddp_comm_defaults" in calls
    assert calls.index("_original_trainer_init") < calls.index("install_unsloth_ddp_comm_defaults")
