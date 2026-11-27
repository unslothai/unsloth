# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Forced float32 GradScaler (made after tp / pp Trainer setup) must unscale DTensor + local grads and agree on
found_inf over ranks. gloo CPU world of 2; the CUDA scaler is swapped for a CPU one."""

import ast
import builtins
import math
import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import pathlib  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.xdist_group("test_forced_float32_parallel_scaler_gloo")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
_FORCED_INIT = 2.0**8


def _free_globals(source):
    """Names a function (nested functions included) reads from module globals."""
    import symtable

    found, stack = set(), [symtable.symtable(source, "<extracted>", "exec")]
    while stack:
        table = stack.pop()
        if table.get_type() != "module":
            found.update(
                s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()
            )
        stack.extend(table.get_children())
    return found


def _forced_scaler_fn():
    """Load the forced scaler functions without importing unsloth."""
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    names = (
        "_unsloth_forced_float32_init_scale",
        "_unsloth_forced_float32_grad_scaler",
        "_parallel_forced_scaler",
        # #19442: the scaler asks the model being trained, not UNSLOTH_FORCE_FLOAT32.
        "_model_forced_float32",
    )
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {
        "torch": torch,
        "os": os,
        "math": math,
        "__name__": "unsloth.models._utils",
        "__package__": "unsloth.models",
    }
    # A helper these functions start calling must be added above, not surface as a NameError on a rank.
    missing = (
        set().union(*(_free_globals(ast.unparse(n)) for n in nodes))
        - set(ns)
        - set(names)
        - set(dir(builtins))
    )
    assert not missing, f"the extracted scaler functions read globals this harness does not provide: {sorted(missing)}"
    exec(compile(ast.Module(nodes, []), str(_UTILS), "exec"), ns)
    return ns["_unsloth_forced_float32_grad_scaler"]


class _CpuScaler(torch.amp.GradScaler):
    def __init__(
        self,
        device = "cpu",
        **kwargs,
    ):
        super().__init__("cpu", **kwargs)


def _forced_model(trainables):
    model = torch.nn.Module()
    model.frozen = torch.nn.Parameter(torch.ones(2, dtype = torch.float16), requires_grad = False)
    for i, p in enumerate(trainables):
        model.register_parameter(f"t{i}", p)
    model._unsloth_forced_float32 = True
    return model


def _trainer(model):
    from accelerate.utils import DistributedType
    accelerator = types.SimpleNamespace(
        scaler = None,
        mixed_precision = "no",
        distributed_type = DistributedType.MULTI_GPU,
        device = types.SimpleNamespace(type = "cuda"),
        scaler_handler = None,
        _optimizers = [],
    )
    return types.SimpleNamespace(accelerator = accelerator, model = model)


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _tp_case(inf_rank, dtensor = True):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    from test_tp_native_oracle import tpn

    rank = torch.distributed.get_rank()
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("tp",))
    torch.manual_seed(0)
    sharded = torch.nn.Parameter(distribute_tensor(torch.randn(8, 4), mesh, [Shard(0)]))
    if not dtensor:
        sharded = torch.nn.Parameter(sharded.detach().to_local().clone())
    plain = torch.nn.Parameter(torch.randn(4, 4))
    trainer = _trainer(_forced_model([sharded, plain]))
    tpn._tp_scaler(trainer, 1)  # tp setup runs before the forced scaler exists
    scaler = _forced_scaler_fn()(trainer, trainer.model)
    assert scaler is not None and scaler is trainer.accelerator.scaler
    opt = torch.optim.AdamW([sharded, plain], lr = 1e-2, foreach = False)
    out = {"scaler": type(scaler).__name__, "init": float(scaler._init_scale)}
    for step in range(3):
        loss = _local(sharded).pow(2).sum() + plain.pow(2).sum()
        trainer.accelerator.scaler.scale(loss).backward()
        if step == 1 and inf_rank in (rank, "all"):
            plain.grad[0, 0] = float("inf")
        trainer.accelerator.unscale_gradients(opt)
        if step == 0:
            out["unscaled"] = (
                torch.allclose(_local(sharded.grad), 2 * _local(sharded.detach())),
                torch.allclose(plain.grad, 2 * plain.detach()),
            )
        trainer.accelerator.scaler.step(opt)
        trainer.accelerator.scaler.update()
        opt.zero_grad(set_to_none = True)
    out["steps"] = (float(opt.state[sharded]["step"]), float(opt.state[plain]["step"]))
    out["scale"] = float(trainer.accelerator.scaler.get_scale())
    out["plain"] = plain.detach().clone()
    return out


def _pp_case(inf_rank):
    from torch.distributed.device_mesh import init_device_mesh
    from test_pp_native_gloo import LLAMA, batch, build, ppn

    rank = torch.distributed.get_rank()
    model = build(LLAMA)
    ids, labels = batch(model)
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    trainer = _trainer(_forced_model([torch.nn.Parameter(torch.zeros(2))]))
    trainer.accelerator.__dict__["_unsloth_scaler_dp"] = 1
    scaler = _forced_scaler_fn()(trainer, trainer.model)
    opt = torch.optim.SGD(pp.parameters(), lr = 1.0)
    out = {"scaler": type(scaler).__name__, "init": float(scaler._init_scale)}
    pp.step(ids, labels, scaler = scaler)
    if inf_rank in (rank, "all"):
        pp.parameters()[0].grad.view(-1)[0] = float("inf")
    before = [p.detach().clone() for p in pp.parameters()]
    pp.unscale_(scaler, opt)
    pp.clip_grad_norm_(1.0)
    scaler.step(opt)
    scaler.update()
    out["unchanged"] = all(torch.equal(a, b) for a, b in zip(before, pp.parameters()))
    out["scale"] = float(scaler.get_scale())
    return out


def _worker(rank, world, port, out_dir):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    torch.distributed.init_process_group("gloo", rank = rank, world_size = world)
    torch.amp.GradScaler = _CpuScaler
    _load_native()
    try:
        res = {}
        for kind, case in (
            ("tp", _tp_case),
            ("tp_local", lambda i: _tp_case(i, dtensor = False)),
            ("pp", _pp_case),
        ):
            for inf_rank in (None, 1, "all"):
                try:
                    res[kind, inf_rank] = case(inf_rank)
                except Exception as e:
                    res[kind, inf_rank] = f"{type(e).__name__}: {str(e)[:300]}"
        torch.save(res, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope = "module")
def ranks(tmp_path_factory):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("requires CPU gloo")
    from tests._rendezvous_port import free_port

    out = str(tmp_path_factory.mktemp("forced_scaler"))
    torch.multiprocessing.spawn(_worker, args = (2, free_port(), out), nprocs = 2, join = True)
    return [torch.load(os.path.join(out, f"rank{r}.pt"), weights_only = False) for r in range(2)]


_INF = [None, 1, "all"]
_IDS = ["no_inf", "one_rank_inf", "every_rank_inf"]


@pytest.mark.parametrize("inf_rank", _INF, ids = _IDS)
def test_tp_forced_scaler_unscales_dtensor_and_agrees_on_skip(ranks, inf_rank):
    for res in ranks:
        got = res["tp", inf_rank]
        assert not isinstance(got, str), got  # main: "got mixed torch.Tensor and DTensor"
        assert got["init"] == _FORCED_INIT, got["init"]  # divided by dp, never tp
        assert got["unscaled"] == (True, True), got["unscaled"]
        expected = 3.0 if inf_rank is None else 2.0
        assert got["steps"] == (expected, expected), got["steps"]
        assert got["scale"] == (_FORCED_INIT if inf_rank is None else _FORCED_INIT / 2), got[
            "scale"
        ]
        assert torch.isfinite(got["plain"]).all()
    torch.testing.assert_close(ranks[0]["tp", inf_rank]["plain"], ranks[1]["tp", inf_rank]["plain"])


@pytest.mark.parametrize("inf_rank", _INF, ids = _IDS)
def test_tp_forced_scaler_local_grads_agree_on_skip(ranks, inf_rank):
    for res in ranks:
        got = res["tp_local", inf_rank]
        assert not isinstance(got, str), got
        expected = 3.0 if inf_rank is None else 2.0
        assert got["steps"] == (expected, expected), got["steps"]
        assert got["scale"] == (_FORCED_INIT if inf_rank is None else _FORCED_INIT / 2), got[
            "scale"
        ]


@pytest.mark.parametrize("inf_rank", _INF, ids = _IDS)
def test_pp_forced_scaler_agrees_on_skip(ranks, inf_rank):
    for res in ranks:
        got = res["pp", inf_rank]
        assert not isinstance(got, str), got
        assert got["init"] == _FORCED_INIT, got["init"]
        assert got["unchanged"] is (inf_rank is not None), got
        assert got["scale"] == (_FORCED_INIT if inf_rank is None else _FORCED_INIT / 2), got[
            "scale"
        ]
