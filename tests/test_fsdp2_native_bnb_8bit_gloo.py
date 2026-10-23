# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes 8-bit AdamW under native FSDP2, stepped on the local shards (gloo world of 2, CPU,
bitsandbytes' CPU kernels):

* the local-shard optimizer equals bitsandbytes stepping each rank's plain shard (bitwise), keeps 8-bit
  state where one GPU would (min_8bit_size on the full size), handles an empty shard, a replicated
  parameter, two groups and an LR change;
* the full optimizer state (get / set_optimizer_state_dict, full_state_dict) round-trips the 8-bit
  state exactly, and a step after the reload equals a step without it;
* a Trainer with optim = 'adamw_8bit' keeps bitsandbytes (it was switched to torch's AdamW before), with
  8-bit state on lm_head, 32-bit on the embedding (transformers' override), and a checkpoint resume
  that ends on the same weights as the uninterrupted run."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _model, _stub_unsloth_save  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _install_wrapped_trainer_init  # noqa: E402

_SHAPES = {"w": (96, 64), "b": (96,), "row": (1, 5000), "odd": (3, 1000), "mis": (5, 1000)}
_REPLICATED = {"rep": (4096,)}


class _Toy(torch.nn.Module):
    def __init__(self, mesh, seed):
        device = mesh.device_type
        super().__init__()
        from torch.distributed.tensor import Shard, distribute_tensor

        g = torch.Generator().manual_seed(seed)
        for name, shape in _SHAPES.items():
            full = torch.randn(shape, generator = g).to(device)
            self.register_parameter(
                name, torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
            )
        for name, shape in _REPLICATED.items():
            self.register_parameter(
                name, torch.nn.Parameter(torch.randn(shape, generator = g).to(device))
            )


def _groups(model):
    return [
        {"params": [model.w, model.b, model.row]},
        {"params": [model.odd, model.mis, model.rep], "lr": 5e-3},
    ]


def _grads(step, device = "cpu"):
    g = torch.Generator().manual_seed(100 + step)
    shapes = {**_SHAPES, **_REPLICATED}
    return {
        name: (torch.randn(shape, generator = g) * (0.1 + step)).to(device)
        for name, shape in shapes.items()
    }


def _set_grads(model, grads):
    from torch.distributed.tensor import DTensor, distribute_tensor
    for name, p in model.named_parameters():
        full = grads[name]
        p.grad = (
            distribute_tensor(full, p.device_mesh, p.placements) if isinstance(p, DTensor) else full
        )


def _locals(model):
    from torch.distributed.tensor import DTensor
    return {
        n: (p._local_tensor if isinstance(p, DTensor) else p).detach().clone()
        for n, p in model.named_parameters()
    }


def _raw(optimizer, model):
    out = {}
    for name, p in model.named_parameters():
        state = optimizer.state.get(p, {})
        out[name] = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
    return out


def _case_unit(
    rank,
    tmp,
    device = "cpu",
):
    import bitsandbytes as bnb
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models.fsdp2_bnb_optim import (
        is_local_shard_bnb_optimizer,
        localize_bnb_optimizer,
        rename_optimizer_state_for,
    )

    mesh = init_device_mesh(device, (2,))
    model = _Toy(mesh, 0)
    optimizer = bnb.optim.AdamW8bit(_groups(model), lr = 1e-2, weight_decay = 0.01)
    assert localize_bnb_optimizer(optimizer) and is_local_shard_bnb_optimizer(optimizer)

    # Reference: bitsandbytes over this rank's plain shards, with one GPU's 8-bit choice per parameter
    # (full numel >= min_8bit_size) forced through bitsandbytes' own per-parameter override.
    manager = bnb.optim.GlobalOptimManager.get_instance()
    local0 = _locals(model)
    ref = {n: torch.nn.Parameter(t.clone()) for n, t in local0.items() if t.numel() > 0}
    for name, p in ref.items():
        full = getattr(model, name).numel()
        manager.override_config(p, "min_8bit_size", 0 if full >= 4096 else 10**9)
    ref_groups = [
        {"params": [ref[n] for n in ("w", "b", "row") if n in ref]},
        {"params": [ref[n] for n in ("odd", "mis", "rep") if n in ref], "lr": 5e-3},
    ]
    ref_opt = bnb.optim.AdamW8bit(ref_groups, lr = 1e-2, weight_decay = 0.01)

    # One GPU: bitsandbytes over the full tensors (different quantization blocks).
    single = {
        n: torch.nn.Parameter(getattr(model, n).full_tensor().clone())
        if n in _SHAPES
        else torch.nn.Parameter(getattr(model, n).detach().clone())
        for n in local0
    }
    single_opt = bnb.optim.AdamW8bit(
        [
            {"params": [single[n] for n in ("w", "b", "row")]},
            {"params": [single[n] for n in ("odd", "mis", "rep")], "lr": 5e-3},
        ],
        lr = 1e-2,
        weight_decay = 0.01,
    )

    def _step(step):
        grads = _grads(step, device)
        _set_grads(model, grads)
        local_grads = {
            n: (p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad)
            for n, p in model.named_parameters()
        }
        for n, p in ref.items():
            p.grad = local_grads[n].clone()
        for n, p in single.items():
            p.grad = grads[n].clone()
        optimizer.step()
        ref_opt.step()
        single_opt.step()
        optimizer.zero_grad(set_to_none = True)

    res = {}
    for step in range(3):
        if step == 2:  # an LR scheduler writes the groups' lr
            for opt in (optimizer, ref_opt, single_opt):
                opt.param_groups[0]["lr"] = 2e-3
        _step(step)
    now = _locals(model)
    res["vs_shard_bnb"] = max((now[n] - ref[n].detach()).abs().max().item() for n in ref)
    res["moved"] = min((now[n] - local0[n]).abs().max().item() for n in ref)
    res["dtypes"] = {
        n: str(optimizer.state[getattr(model, n)].get("state1", torch.empty(0)).dtype)
        if "state1" in optimizer.state[getattr(model, n)]
        else "none"
        for n in local0
    }
    res["steps"] = {n: optimizer.state[getattr(model, n)]["step"] for n in local0}
    for n in local0:  # one GPU vs FSDP2 (blocks differ): reported, not asserted
        got = getattr(model, n)
        got = got.full_tensor() if n in _SHAPES else got.detach()
        res.setdefault("vs_single_rel", {})[n] = (
            (got - single[n].detach()).norm() / (single[n].detach().norm() + 1e-12)
        ).item()

    # Full optimizer state round trip and a step after the reload.
    native = sys.modules["unsloth.models.fsdp2_native"]
    path = os.path.join(tmp, "optimizer.pt")
    native.native_save_optimizer(model, optimizer, path)  # collective, written by rank 0
    full = torch.load(path, map_location = "cpu", weights_only = True)
    res["full_keys"] = sorted(full["state"]["w"])
    res["full_shapes"] = {n: list(full["state"][n]["state1"].shape) for n in local0}
    res["full_dtype"] = str(full["state"]["w"]["state1"].dtype)
    model2 = _Toy(mesh, 0)
    with torch.no_grad():
        for (n, p), (_, q) in zip(model.named_parameters(), model2.named_parameters()):
            target = q._local_tensor if hasattr(q, "_local_tensor") else q
            target.copy_(p._local_tensor if hasattr(p, "_local_tensor") else p)
    optimizer2 = bnb.optim.AdamW8bit(_groups(model2), lr = 1e-2, weight_decay = 0.01)
    localize_bnb_optimizer(optimizer2)
    native.native_load_optimizer(model2, optimizer2, path)
    raw1, raw2 = _raw(optimizer, model), _raw(optimizer2, model2)
    mismatch = []
    for n in raw1:
        if sorted(raw1[n]) != sorted(raw2[n]):
            mismatch.append((n, sorted(raw1[n]), sorted(raw2[n])))
            continue
        for k, v in raw1[n].items():
            w = raw2[n][k]
            same = torch.equal(v, w) if torch.is_tensor(v) else v == w
            if not same:
                mismatch.append((n, k))
    res["reload_mismatch"] = mismatch
    res["lr_after_reload"] = [g["lr"] for g in optimizer2.param_groups]
    grads = _grads(7, device)
    _set_grads(model, grads)
    _set_grads(model2, grads)
    optimizer.step()
    optimizer2.step()
    a, b = _locals(model), _locals(model2)
    res["after_reload_step_diff"] = max((a[n] - b[n]).abs().max().item() for n in a if a[n].numel())

    # A torch AdamW checkpoint (exp_avg / exp_avg_sq, tensor step) maps onto the bitsandbytes keys.
    as_torch = {
        "state": {
            n: {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2), "step": torch.tensor(3.0)}
            for n in ("w",)
        },
        "param_groups": [],
    }
    rename_optimizer_state_for(optimizer, as_torch)
    res["renamed"] = sorted(as_torch["state"]["w"]) + [
        type(as_torch["state"]["w"]["step"]).__name__
    ]
    return res


def _cpu_percentile_clipping(functional):
    """bitsandbytes' percentile_clipping (CUDA only) in torch for the CPU ranks: the squared gradient
    norm goes to gnorm_vec[step % 100], and the clip value is the given percentile of the history."""
    original = functional.percentile_clipping

    def percentile_clipping(
        grad,
        gnorm_vec,
        step,
        percentile = 5,
    ):
        if grad.is_cuda:
            return original(grad, gnorm_vec, step, percentile)
        gnorm_vec[step % 100] = grad.float().pow(2).sum()
        current = torch.sqrt(gnorm_vec[step % 100])
        vals, _ = torch.sort(gnorm_vec)
        clip = torch.sqrt(vals[percentile])
        return current, clip, (clip / current if current > clip else 1.0)

    functional.percentile_clipping = percentile_clipping


def _case_pclip(
    rank,
    tmp,
    device = "cpu",
):
    """percentile_clipping = 5: bitsandbytes keeps the last 100 squared gradient norms per parameter
    (gnorm_vec) and clips at their 5th percentile, so a resume must restore that history."""
    import bitsandbytes as bnb
    import bitsandbytes.optim.optimizer as bnb_optimizer
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer

    _cpu_percentile_clipping(bnb_optimizer.F)
    native = sys.modules["unsloth.models.fsdp2_native"]
    mesh = init_device_mesh(device, (2,))

    def grads(step):
        g = torch.Generator().manual_seed(1000 + step)
        scale = 1.0 + (step % 7)
        shapes = {**_SHAPES, **_REPLICATED}
        return {n: (torch.randn(s, generator = g) * scale).to(device) for n, s in shapes.items()}

    def build(model):
        opt = bnb.optim.AdamW8bit(_groups(model), lr = 1e-3, percentile_clipping = 5)
        localize_bnb_optimizer(opt)
        return opt

    def run(model, opt, steps):
        for step in steps:
            _set_grads(model, grads(step))
            opt.step()
            opt.zero_grad(set_to_none = True)

    model = _Toy(mesh, 0)
    optimizer = build(model)
    run(model, optimizer, range(100))
    path = os.path.join(tmp, "optimizer_pclip.pt")
    native.native_save_optimizer(model, optimizer, path)
    at_save = _locals(model)
    run(model, optimizer, range(100, 104))

    model2 = _Toy(mesh, 0)
    with torch.no_grad():
        for n, q in model2.named_parameters():
            (q._local_tensor if hasattr(q, "_local_tensor") else q).copy_(at_save[n])
    optimizer2 = build(model2)
    native.native_load_optimizer(model2, optimizer2, path)
    run(model2, optimizer2, range(100, 104))
    a, b = _locals(model), _locals(model2)
    res = {
        "resume_diff": max((a[n] - b[n]).abs().max().item() for n in a if a[n].numel()),
        "has_gnorm": sorted(
            n for n, p in model.named_parameters() if "gnorm_vec" in optimizer.state[p]
        ),
        "gnorm_diff": max(
            (optimizer.state[p]["gnorm_vec"] - optimizer2.state[q]["gnorm_vec"]).abs().max().item()
            for (n, p), (_, q) in zip(model.named_parameters(), model2.named_parameters())
            if "gnorm_vec" in optimizer.state[p]
        ),
    }
    return res


def _case_refuse(rank, tmp):
    """A bitsandbytes optimizer that cannot step local shards, built by Trainer.create_optimizer (here
    through get_optimizer_cls_and_kwargs, as a non-blockwise Adam) or by a subclass create_optimizer
    (LAMB): refused when it is built, not inside the first step."""
    import bitsandbytes as bnb
    from transformers import Trainer, TrainingArguments

    from test_fsdp2_native_trainer_ckpt_gloo import _non_blockwise_adamw8bit

    _load_native()
    _stub_unsloth_save()
    _install_wrapped_trainer_init()
    res = {}

    def args():
        return TrainingArguments(
            output_dir = os.path.join(tmp, "out"),
            report_to = [],
            use_cpu = True,
            optim = "adamw_8bit",
            max_steps = 2,
        )

    trainer = Trainer(model = _model(False, 1, tmp), args = args())
    non_blockwise = _non_blockwise_adamw8bit()
    trainer.get_optimizer_cls_and_kwargs = lambda *a, **k: (non_blockwise, {"lr": 1e-3})
    try:
        trainer.create_optimizer()
        res["base_hook"] = "built " + type(trainer.optimizer).__name__
    except ValueError as exc:
        res["base_hook"] = "refused" if "cannot" in str(exc) else f"other: {exc}"

    class LambTrainer(Trainer):
        def create_optimizer(self, *a, **k):
            self.optimizer = bnb.optim.LAMB8bit(self.model.parameters(), lr = 1e-3)
            return self.optimizer

    trainer = LambTrainer(model = _model(False, 1, tmp), args = args())
    try:
        trainer.create_optimizer_and_scheduler(num_training_steps = 2)
        res["subclass_hook"] = "built " + type(trainer.optimizer).__name__
    except ValueError as exc:
        res["subclass_hook"] = "refused" if "cannot" in str(exc) else f"other: {exc}"
    return res


def _trainer_case(
    native,
    tmp,
    resume_from = None,
    max_steps = 4,
):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    _install_wrapped_trainer_init()
    model = _model(False, 1, tmp)
    rows = [{"input_ids": list(range(5, 37)), "labels": list(range(5, 37))}] * 8
    args = TrainingArguments(
        output_dir = os.path.join(tmp, "out"),
        report_to = [],
        use_cpu = True,
        optim = "adamw_8bit",
        max_steps = max_steps,
        learning_rate = 1e-3,
        per_device_train_batch_size = 1,
        lr_scheduler_type = "linear",
        save_strategy = "steps",
        save_steps = 2,
        logging_steps = 1,
        seed = 0,
        data_seed = 0,
    )
    trainer = Trainer(model = model, args = args, train_dataset = Dataset.from_list(rows))
    trainer.train(resume_from_checkpoint = resume_from)
    return trainer


def _case_trainer(rank, tmp):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    native = _load_native()
    _stub_unsloth_save()
    trainer = _trainer_case(native, tmp)
    inner = native._inner_optimizer(trainer.optimizer)
    res = {
        "optim": str(getattr(trainer.args.optim, "value", trainer.args.optim)),
        "cls": f"{type(inner).__module__}.{type(inner).__name__}",
        "local_shards": getattr(type(inner), "_unsloth_fsdp2_local_shards", False),
        "native": native.is_native_fsdp2(trainer.model),
        "losses": [h["loss"] for h in trainer.state.log_history if "loss" in h],
    }
    model = native._native_root(trainer.model)
    dtypes = {}
    for name, p in model.named_parameters():
        state = inner.state.get(p, {})
        if "state1" in state and ("lm_head" in name or "embed_tokens" in name):
            dtypes[name] = str(state["state1"].dtype)
    res["dtypes"] = dtypes
    final = {
        n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
        for n, p in model.named_parameters()
    }
    ckpt = os.path.join(tmp, "out", "checkpoint-2")
    res["ckpt"] = os.path.isfile(os.path.join(ckpt, "optimizer.pt"))
    if rank == 0 and res["ckpt"]:
        saved = torch.load(
            os.path.join(ckpt, "optimizer.pt"), map_location = "cpu", weights_only = True
        )
        any_state = next(iter(saved["state"].values()))
        res["saved_keys"] = sorted(any_state)
        step = any_state["step"]
        res["saved_step"] = float(step) if torch.is_tensor(step) else step  # plain, for the queue
    dist.barrier()
    resumed = _trainer_case(native, tmp, resume_from = ckpt)
    rmodel = native._native_root(resumed.model)
    again = {
        n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
        for n, p in rmodel.named_parameters()
    }
    res["resume_diff"] = max((again[n] - final[n]).abs().max().item() for n in final)
    res["resume_steps"] = resumed.state.global_step
    return res


def _worker(rank, world, port, out, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        _load_native()
        cases = {
            "unit": _case_unit,
            "trainer": _case_trainer,
            "pclip": _case_pclip,
            "refuse": _case_refuse,
        }
        res = cases[case](rank, tmp)
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp_path,
    timeout = 300,
):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, case, str(tmp_path)),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(
                    f"{case}: the ranks did not finish within {timeout} s (hung collective)"
                )
    except mp.ProcessExitedException as exc:
        pytest.fail(f"{case}: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("BNB8_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_local_shard_bnb_matches_shard_bnb_and_reloads_exactly(tmp_path):
    pytest.importorskip("bitsandbytes")
    results = _run("unit", tmp_path)
    for rank, r in results.items():
        assert r["vs_shard_bnb"] == 0.0, (rank, r)
        assert r["moved"] > 0, (rank, r)
        # 8-bit where one GPU picks it (full numel >= 4096), even for a 3072-element shard.
        assert r["dtypes"]["w"] == "torch.uint8" and r["dtypes"]["rep"] == "torch.uint8", r
        assert r["dtypes"]["mis"] == "torch.uint8", r
        assert r["dtypes"]["b"] == "torch.float32" and r["dtypes"]["odd"] == "torch.float32", r
        assert r["steps"] == {n: 3 for n in r["steps"]}, r
        assert r["full_keys"] == ["state1", "state2", "step"], r
        assert r["full_dtype"] == "torch.float32", r
        assert r["full_shapes"] == {
            "w": [96, 64],
            "b": [96],
            "row": [1, 5000],
            "odd": [3, 1000],
            "mis": [5, 1000],
            "rep": [4096],
        }, r
        assert r["reload_mismatch"] == [], (rank, r["reload_mismatch"])
        assert r["lr_after_reload"] == [2e-3, 5e-3], r
        assert r["after_reload_step_diff"] == 0.0, (rank, r)
        assert r["renamed"] == ["state1", "state2", "step", "int"], r
    # Shards that are whole 256-element blocks of the full tensor quantize exactly as one GPU does;
    # (5, 1000) splits into 3000 / 2000 elements, so its blocks differ from one GPU's.
    for rank, r in results.items():
        assert r["vs_single_rel"]["w"] == 0.0 and r["vs_single_rel"]["row"] == 0.0, r
        assert r["vs_single_rel"]["odd"] == 0.0 and r["vs_single_rel"]["b"] == 0.0, r
        assert 0.0 < r["vs_single_rel"]["mis"] < 1e-2, r
    # The (1, 5000) parameter: rank 0 holds the whole row in 8-bit, rank 1 an empty shard.
    assert results[0]["dtypes"]["row"] == "torch.uint8", results[0]
    assert results[1]["dtypes"]["row"] == "none", results[1]


def test_trainer_adamw_8bit_keeps_bnb_state_and_resumes(tmp_path):
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("datasets")
    results = _run("trainer", tmp_path, timeout = 600)
    for rank, r in results.items():
        assert r["native"] and r["optim"] == "adamw_8bit", (rank, r)
        assert r["local_shards"] and r["cls"].startswith("bitsandbytes."), (rank, r)
        assert any("lm_head" in n and d == "torch.uint8" for n, d in r["dtypes"].items()), r
        assert any("embed_tokens" in n and d == "torch.float32" for n, d in r["dtypes"].items()), r
        assert all(x == x and abs(x) < 1e4 for x in r["losses"]) and len(r["losses"]) == 4, r
        assert r["ckpt"] and r["resume_steps"] == 4, r
        assert r["resume_diff"] == 0.0, (rank, r)
    assert results[0]["saved_keys"] == ["state1", "state2", "step"], results[0]


def test_percentile_clipping_history_survives_resume(tmp_path):
    bnb = pytest.importorskip("bitsandbytes")
    import inspect

    if "percentile_clipping" not in inspect.signature(bnb.optim.AdamW8bit.__init__).parameters:
        pytest.skip("this bitsandbytes has no percentile_clipping")
    probe = torch.nn.Parameter(torch.zeros(8))
    probe.grad = torch.ones(8)
    try:
        bnb.optim.AdamW8bit([probe], percentile_clipping = 5, min_8bit_size = 0).step()
    except RuntimeError as exc:  # bitsandbytes <= 0.48: 8-bit kernels are CUDA only
        pytest.skip(f"this bitsandbytes cannot step 8-bit state on the CPU ({str(exc)[:60]})")
    for rank, r in _run("pclip", tmp_path).items():
        assert r["has_gnorm"], (rank, r)
        assert r["gnorm_diff"] == 0.0, (rank, r)
        assert r["resume_diff"] == 0.0, (rank, r)


def test_bnb_optimizer_that_cannot_step_shards_is_refused_when_built(tmp_path):
    pytest.importorskip("bitsandbytes")
    for rank, r in _run("refuse", tmp_path).items():
        assert r["base_hook"] == "refused", (rank, r)
        assert r["subclass_hook"] == "refused", (rank, r)


def test_chunked_requantization_is_bitwise_equal_to_one_pass():
    """The checkpoint (de)quantization runs a bounded number of blocks per pass; a state spanning
    several chunks with a ragged tail block gives the same codes, absmax and float32 values as one
    pass, and inverts bitsandbytes' own state exactly."""
    bnb = pytest.importorskip("bitsandbytes")
    _load_native()
    from unsloth.models.fsdp2_bnb_optim import _dequantize, _quantize_into

    torch.manual_seed(0)
    n = 256 * 10 + 77  # 11 blocks, the last one 77 elements
    p = torch.nn.Parameter(torch.randn(n))
    opt = bnb.optim.AdamW8bit([p], lr = 1e-2, min_8bit_size = 0)
    for step in range(3):
        p.grad = torch.randn(n) * (10.0 ** (step - 1))
        opt.step()
    raw = opt.state[p]
    for key in ("state1", "state2"):
        i = key[-1]
        whole = _dequantize(raw, key, raw[key].device, chunk_blocks = 10**6)
        for chunk_blocks in (1, 3, 4, 10**6):
            values = _dequantize(raw, key, raw[key].device, chunk_blocks = chunk_blocks)
            assert torch.equal(values, whole), (key, chunk_blocks)
            codes = torch.empty_like(raw[key])
            absmax = torch.empty_like(raw[f"absmax{i}"])
            _quantize_into(values, raw[f"qmap{i}"], codes, absmax, chunk_blocks = chunk_blocks)
            assert torch.equal(codes, raw[key]), (key, chunk_blocks)
            assert torch.equal(absmax, raw[f"absmax{i}"]), (key, chunk_blocks)
