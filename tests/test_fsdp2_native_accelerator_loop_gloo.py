# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 in a custom Accelerator loop (shard_model + accelerate, no Trainer), gloo world of 2:

* accelerator.prepare(optimizer, model) (optimizer first) still applies the native optimizer checks:
  a bitsandbytes optimizer is refused, a mixed DTensor / replicated group is split;
* accelerate's plain float16 GradScaler already skips the step on every rank when one rank's shard
  overflows (DTensor reduces the inf check: torch/distributed/tensor/_dispatch.py
  found_inf_reduce_handler), so a custom loop needs no ShardedGradScaler swap;
* get_state_dict / save_model return and write the full weights; save_state / load_state refuse."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _full, _stub_unsloth_save  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _lora  # noqa: E402
from test_fsdp2_native_optim_reentry_gloo import AdamW8bitLike  # noqa: E402


def _mixed_lora(native):
    model = _lora(1)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n)
    cfg = native.FSDP2Config(param_dtype = torch.float32, ignored_params = {first})
    return native.shard_model(model, cfg)


def _case_order(native):
    from accelerate import Accelerator

    res = {}
    model = _mixed_lora(native)
    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        Accelerator().prepare(AdamW8bitLike(trainable, lr = 1e-3), model)
        res["bnb"] = ""
    except Exception as exc:
        res["bnb"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    optimizer = torch.optim.AdamW(trainable, lr = 1e-3)
    Accelerator().prepare(optimizer, model)
    res["mixed_groups_left"] = native._mixed_param_groups(optimizer)
    res["groups"] = len(optimizer.param_groups)
    return res


def _case_scaler(native, rank):
    from accelerate import Accelerator
    from torch.distributed.tensor import DTensor

    model = native.shard_model(_lora(1), native.FSDP2Config(param_dtype = torch.float32))
    accelerator = Accelerator()
    accelerator.scaler = torch.amp.GradScaler(
        "cpu"
    )  # what mixed_precision = "fp16" builds on a GPU
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, model = accelerator.prepare(torch.optim.SGD(params, lr = 1.0), model)
    res = {"scaler": type(accelerator.scaler).__name__}
    ids = torch.randint(1, 100, (1, 8), generator = torch.Generator().manual_seed(0))
    accelerator.backward(model(input_ids = ids, labels = ids).loss)  # scaled by accelerator.scaler
    target = next(p for p in params if isinstance(p, DTensor) and p.to_local().numel())
    if rank == 0:
        target.grad.to_local().view(-1)[0] = float("inf")  # overflow in rank 0's shard only
    before = target.to_local().detach().clone()
    optimizer.step()
    res["stepped"] = not torch.equal(before, target.to_local().detach())
    return res


def _case_state(native, rank, tmp):
    from accelerate import Accelerator
    from safetensors.torch import load_file
    import torch.distributed as dist

    model = native.shard_model(_lora(1), native.FSDP2Config(param_dtype = torch.float32))
    want = _full(model)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    res = {}
    state = accelerator.get_state_dict(model)
    if rank == 0:
        res["n"] = len(state)
        res["plain"] = all(type(v) is torch.Tensor for v in state.values())
        res["diff"] = max(
            (state[k].float() - want[k].float()).abs().max().item() for k in want if k in state
        )
    else:
        res["n"] = len(state)
    out = os.path.join(tmp, "saved")
    accelerator.save_model(model, out)
    dist.barrier()
    if rank == 0:
        saved = {}
        for name in sorted(os.listdir(out)):
            if name.endswith(".safetensors"):
                saved.update(load_file(os.path.join(out, name)))
        res["saved_diff"] = max(
            (saved[k].float() - want[k].float()).abs().max().item() for k in want if k in saved
        )
        res["saved_n"] = sum(k in saved for k in want)
        res["want_n"] = len(want)
    for name in ("save_state", "load_state"):
        try:
            getattr(accelerator, name)(os.path.join(tmp, f"state_{rank}"))
            res[name] = ""
        except Exception as exc:
            res[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        if case == "order":
            res = _case_order(native)
        elif case == "scaler":
            res = _case_scaler(native, rank)
        elif case == "state":
            res = _case_state(native, rank, tmp)
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
    timeout = 150,
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
        print("ROUND10_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_optimizer_prepared_before_its_model_gets_the_native_checks(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("peft")
    for rank, r in _run("order", tmp_path).items():
        assert r["bnb"].startswith("ValueError") and "bitsandbytes" in r["bnb"], (rank, r)
        assert r["mixed_groups_left"] == 0 and r["groups"] == 2, (rank, r)


def test_fp16_plain_scaler_overflow_skips_on_every_rank(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("peft")
    for rank, r in _run("scaler", tmp_path).items():
        assert r["scaler"] == "GradScaler", (rank, r)
        assert r["stepped"] is False, (rank, r)  # rank 0's overflow skips rank 1's step too


def test_accelerator_state_dict_is_full_and_save_state_refuses(tmp_path):
    pytest.importorskip("accelerate")
    pytest.importorskip("peft")
    results = _run("state", tmp_path)
    r0 = results[0]
    assert r0["plain"] and r0["n"] > 0 and r0["diff"] == 0.0, r0
    assert r0["saved_n"] == r0["want_n"] and r0["saved_diff"] == 0.0, r0
    assert results[1]["n"] == 0, results[1]
    for rank, r in results.items():
        for name in ("save_state", "load_state"):
            assert r[name].startswith("RuntimeError") and "native" in r[name], (rank, name, r)
