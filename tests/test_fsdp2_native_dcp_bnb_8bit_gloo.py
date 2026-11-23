# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The sharded optimizer checkpoint (optimizer_dcp/, UNSLOTH_FSDP2_DCP_OPTIM=1) with the local-shard
bitsandbytes AdamW8bit under native FSDP2 (gloo CPU world, bitsandbytes' CPU kernels):

* save + resume at the same world size restores the 8-bit state (uint8 codes, absmax, qmap) bitwise
  and the next steps equal the uninterrupted run, bitwise; optimizer.pt does the same;
* a Trainer with optim = 'adamw_8bit' writes optimizer_dcp/ and resumes to the uninterrupted weights;
* a torch AdamW optimizer_dcp/ resumed into AdamW8bit (exp_avg -> state1) equals the optimizer.pt
  path of the same checkpoint, bitwise;
* save at one world size, resume at another: the restored (dequantized) state matches the saver's
  within the 8-bit quantization error, and steps."""

import os
import sys

import pytest
import torch

if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_dcp_resume_gloo import (  # noqa: E402
    _batch,
    _diff_models,
    _free_port,
    _full_model,
    _full_optim,
    _init,
    _model,
    _step,
)

_MIN_8BIT = 256  # the 16 x 16 attention projections and the MLP get 8-bit state too


def _bnb(model):
    import bitsandbytes as bnb

    from unsloth.models.fsdp2_bnb_optim import localize_bnb_optimizer

    opt = bnb.optim.AdamW8bit(
        model.parameters(), lr = 1e-2, weight_decay = 0.01, min_8bit_size = _MIN_8BIT
    )
    assert localize_bnb_optimizer(opt)
    return opt


def _torch(model):
    return torch.optim.AdamW(model.parameters(), lr = 1e-2, foreach = False)


def _new(native, seed, make):
    model = native.shard_model(_model(seed), native.FSDP2Config(param_dtype = torch.float32))
    return model, make(model)


def _raw(model, opt):
    """This rank's raw optimizer state per parameter (uint8 codes, absmax, qmap, step)."""
    out = {}
    for name, p in model.named_parameters():
        state = opt.state.get(p, {})
        out[name] = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state.items()}
    return out


def _diff_raw(a, b):
    out = sorted(set(a) ^ set(b))
    for n in a:
        if n not in b:
            continue
        if sorted(a[n]) != sorted(b[n]):
            out.append((n, sorted(a[n]), sorted(b[n])))
            continue
        for k, v in a[n].items():
            w = b[n][k]
            same = (
                torch.is_tensor(w) and v.dtype == w.dtype and torch.equal(v, w)
                if torch.is_tensor(v)
                else v == w
            )
            if not same:
                out.append((n, k))
    return out


_PAIRS = (("state1", "exp_avg"), ("state2", "exp_avg_sq"))


def _max_rel(a, b):
    """Largest relative difference of the Adam moments of two full (FQN-keyed) optimizer states,
    either optimizer's key names."""
    rel = 0.0
    for fqn, entry in a.items():
        other = b[fqn]
        for keys in _PAIRS:
            x = next(entry[k] for k in keys if k in entry).float()
            y = next(other[k] for k in keys if k in other).float()
            rel = max(rel, ((x - y).norm() / (x.norm() + 1e-30)).item())
    return rel


def _uint8_names(model, opt):
    return sorted(
        n
        for n, p in model.named_parameters()
        if opt.state.get(p, {}).get("state1") is not None
        and opt.state[p]["state1"].dtype == torch.uint8
    )


def _save(native, model, opt, ckpt, rank, dist):
    os.makedirs(ckpt, exist_ok = True)
    weights = native.native_full_state_dict(model)
    if rank == 0:
        model.save_pretrained(ckpt, state_dict = weights, safe_serialization = True)
    native.native_save_optimizer(
        model, opt, os.path.join(ckpt, "optimizer.pt"), should_save = rank == 0
    )
    native.native_save_optimizer_sharded(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    dist.barrier()


def _case_same_world(native, dist, rank, tmp):
    model, opt = _new(native, 1, _bnb)
    for step in (1, 2):
        _step(model, opt, step, rank)
    ckpt = os.path.join(tmp, "ckpt")
    _save(native, model, opt, ckpt, rank, dist)
    saved_raw = _raw(model, opt)
    res = {"uint8": _uint8_names(model, opt), "n_params": len(saved_raw)}
    want = [_step(model, opt, step, rank).item() for step in (3, 4)]
    want_after = _full_model(model)
    for arm, src in (("dcp", "optimizer_dcp"), ("pt", "optimizer.pt")):
        m, o = _new(native, 2, _bnb)
        native.native_load_model_checkpoint(m, ckpt)
        try:
            native.native_load_optimizer(m, o, os.path.join(ckpt, src))
            got = {"raw_diff": [str(d)[:200] for d in _diff_raw(saved_raw, _raw(m, o))]}
            got["losses"] = [_step(m, o, step, rank).item() for step in (3, 4)]
            got["want"] = want
            got["after_diff"] = _diff_models(want_after, _full_model(m))
        except BaseException as exc:  # CheckpointException is a BaseException
            got = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        res[arm] = got
    return res


def _case_cross(native, dist, rank, tmp, saver, loader):
    """One optimizer saved, the other resumed (torch AdamW <-> AdamW8bit): the sharded directory gives
    what optimizer.pt gives, bitwise."""
    model, opt = _new(native, 1, saver)
    for step in (1, 2):
        _step(model, opt, step, rank)
    ckpt = os.path.join(tmp, "ckpt")
    _save(native, model, opt, ckpt, rank, dist)
    saved = _full_optim(model, opt)["state"]
    arms = {}
    res = {}
    for arm, src in (("pt", "optimizer.pt"), ("dcp", "optimizer_dcp")):
        m, o = _new(native, 2, loader)
        native.native_load_model_checkpoint(m, ckpt)
        try:
            native.native_load_optimizer(m, o, os.path.join(ckpt, src))
            arms[arm] = {"raw": _raw(m, o), "uint8": _uint8_names(m, o)}
            arms[arm]["keys"] = sorted({k for st in o.state.values() for k in st})
            arms[arm]["group_keys"] = sorted(o.param_groups[0])
            arms[arm]["max_rel"] = _max_rel(saved, _full_optim(m, o)["state"])
            arms[arm]["losses"] = [_step(m, o, step, rank).item() for step in (3, 4)]
            arms[arm]["after"] = _full_model(m)
        except BaseException as exc:  # CheckpointException is a BaseException
            res[f"{arm}_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    if len(arms) == 2:
        res["uint8"] = arms["dcp"]["uint8"]
        res["keys"] = arms["dcp"]["keys"]
        res["max_rel"] = arms["dcp"]["max_rel"]
        res["group_keys"] = [arms["pt"]["group_keys"], arms["dcp"]["group_keys"]]
        res["steps"] = sorted(
            {float(v["step"]) for v in arms["dcp"]["raw"].values() if "step" in v}
        )
        res["raw_diff"] = [str(d)[:200] for d in _diff_raw(arms["pt"]["raw"], arms["dcp"]["raw"])]
        res["losses"] = [arms["pt"]["losses"], arms["dcp"]["losses"]]
        res["after_diff"] = _diff_models(arms["pt"]["after"], arms["dcp"]["after"])
    return res


def _case_reshard(native, dist, rank, tmp, phase):
    """AdamW8bit optimizer_dcp/ saved at one world size and resumed at another. The 8-bit blocks follow
    each rank's shard, so the restored state is re-quantized on other blocks: compared dequantized."""
    ckpt = os.path.join(tmp, "ckpt")
    ref = os.path.join(tmp, "ref.pt")
    if phase == "save":
        model, opt = _new(native, 1, _bnb)
        for step in (1, 2):
            _step(model, opt, step, rank)
        _save(native, model, opt, ckpt, rank, dist)
        full = {"model": _full_model(model), "optim": _full_optim(model, opt)}
        if rank == 0:
            torch.save(full, ref)
        dist.barrier()
        return {"world": dist.get_world_size()}
    model, opt = _new(native, 2, _bnb)
    native.native_load_model_checkpoint(model, ckpt)
    res = {"world": dist.get_world_size()}
    try:
        native.native_load_optimizer(model, opt, os.path.join(ckpt, "optimizer_dcp"))
    except BaseException as exc:  # CheckpointException is a BaseException
        res["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        return res
    want = torch.load(ref, weights_only = False)
    got = _full_optim(model, opt)
    res["model_diff"] = _diff_models(want["model"], _full_model(model))
    res["missing"] = sorted(set(want["optim"]["state"]) ^ set(got["state"]))
    res["max_rel"] = _max_rel(want["optim"]["state"], got["state"])
    res["steps"] = sorted(
        {
            (entry.get("step"), got["state"][fqn].get("step"))
            for fqn, entry in want["optim"]["state"].items()
        }
    )
    res["loss"] = _step(model, opt, 3, rank).item()  # the restored state steps
    return res


def _case_trainer(native, dist, rank, tmp):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    from test_fsdp2_native_trainer_args_gloo import _install_wrapped_trainer_init

    _install_wrapped_trainer_init()
    cfg = native.FSDP2Config(param_dtype = torch.float32)
    rows = [
        {"input_ids": list(range(5 + i, 21 + i)), "labels": list(range(5 + i, 21 + i))}
        for i in range(4)
    ]
    ds = Dataset.from_list(rows * 4)

    def _args(out):
        return TrainingArguments(
            output_dir = out,
            report_to = [],
            use_cpu = True,
            optim = "adamw_8bit",
            max_steps = 4,
            save_steps = 2,
            save_strategy = "steps",
            learning_rate = 1e-2,
            per_device_train_batch_size = 2,
            lr_scheduler_type = "linear",
            logging_steps = 1,
            seed = 0,
            data_seed = 0,
            dataloader_num_workers = 0,
        )

    os.environ["UNSLOTH_FSDP2_DCP_OPTIM"] = "1"
    out = os.path.join(tmp, "out")
    try:
        model = native.shard_model(_model(1), cfg)
        trainer = Trainer(model = model, args = _args(out), train_dataset = ds)
        trainer.train()
        inner = native._inner_optimizer(trainer.optimizer)
        want = _full_model(model)
        ck2 = os.path.join(out, "checkpoint-2")
        files = sorted(os.listdir(ck2))
        dist.barrier()
        resumed = native.shard_model(_model(2), cfg)
        Trainer(model = resumed, args = _args(out), train_dataset = ds).train(resume_from_checkpoint = ck2)
        got = _full_model(resumed)
        dist.barrier()
    finally:
        os.environ.pop("UNSLOTH_FSDP2_DCP_OPTIM", None)
    return {
        "local_shards": getattr(type(inner), "_unsloth_fsdp2_local_shards", False),
        "uint8": _uint8_names(native._native_root(model), inner),
        "files": files,
        "diff": _diff_models(want, got),
    }


def _worker(rank, world, port, out, case, tmp):
    dist = _init(rank, world, port, timeout = 120)
    try:
        native = _load_native()
        if case.startswith("reshard_"):
            res = _case_reshard(native, dist, rank, tmp, case[len("reshard_") :])
        elif case == "torch_to_bnb":
            res = _case_cross(native, dist, rank, tmp, _torch, _bnb)
        elif case == "bnb_to_torch":
            res = _case_cross(native, dist, rank, tmp, _bnb, _torch)
        else:
            res = {"same_world": _case_same_world, "trainer": _case_trainer}[case](
                native, dist, rank, tmp
            )
        out.put((rank, res))
    except BaseException as exc:  # CheckpointException is a BaseException
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    case,
    tmp,
    timeout = 600,
    world = 2,
):
    import queue

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world, port, out, case, tmp)) for r in range(world)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, r = out.get(timeout = timeout)
            results[rank] = r
    except queue.Empty:
        pass
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.kill()
    assert sorted(results) == list(
        range(world)
    ), f"{case}: ranks did not report within {timeout} s: {results}"
    for rank, r in results.items():
        print("DCP_BNB8", case, rank, {k: v for k, v in r.items() if k != "crash"})
        assert "crash" not in r, r["crash"]
    return results


def _skip_without_cpu_8bit():
    bnb = pytest.importorskip("bitsandbytes")
    pytest.importorskip("transformers")
    probe = torch.nn.Parameter(torch.zeros(8))
    probe.grad = torch.ones(8)
    try:
        bnb.optim.AdamW8bit([probe], min_8bit_size = 0).step()
    except RuntimeError as exc:  # bitsandbytes <= 0.48: 8-bit kernels are CUDA only
        pytest.skip(f"this bitsandbytes cannot step 8-bit state on the CPU ({str(exc)[:60]})")


def test_sharded_bnb_8bit_resume_is_bitwise(tmp_path):
    _skip_without_cpu_8bit()
    for rank, r in _run("same_world", str(tmp_path)).items():
        assert len(r["uint8"]) > 2, r  # embedding, lm_head and the projections hold 8-bit state
        for arm in ("dcp", "pt"):
            assert "error" not in r[arm], (arm, r[arm])
            assert r[arm]["raw_diff"] == [], (arm, r[arm]["raw_diff"])
            assert r[arm]["losses"] == r[arm]["want"], (arm, r[arm])
            assert r[arm]["after_diff"] == [], (arm, r[arm])


@pytest.mark.parametrize("case", ["torch_to_bnb", "bnb_to_torch"])
def test_sharded_checkpoint_resumes_across_torch_adamw_and_bnb_8bit(case, tmp_path):
    _skip_without_cpu_8bit()
    for rank, r in _run(case, str(tmp_path)).items():
        # Each optimizer's own hyperparameter keys (amsgrad / foreach vs alpha / t_alpha) used to
        # make both loaders refuse the other optimizer's checkpoint.
        assert "pt_error" not in r and "dcp_error" not in r, r
        assert r["group_keys"][0] == r["group_keys"][1], r
        if case == "torch_to_bnb":
            assert len(r["uint8"]) > 2 and "state1" in r["keys"], r
            assert "amsgrad" not in r["group_keys"][1] and "alpha" in r["group_keys"][1], r
        else:
            assert "exp_avg" in r["keys"] and "state1" not in r["keys"], r
            assert "amsgrad" in r["group_keys"][1] and "alpha" not in r["group_keys"][1], r
        assert r["steps"] == [2.0], r
        # The moments carried over: exactly into torch's float32 state, within 8-bit error into bnb.
        assert r["max_rel"] == 0.0 if case == "bnb_to_torch" else r["max_rel"] < 2e-2, r
        assert r["raw_diff"] == [], r["raw_diff"]
        assert r["losses"][0] == r["losses"][1], r
        assert r["after_diff"] == [], r


def test_trainer_adamw_8bit_resumes_from_sharded_optimizer(tmp_path):
    _skip_without_cpu_8bit()
    pytest.importorskip("datasets")
    for rank, r in _run("trainer", str(tmp_path), timeout = 900).items():
        assert r["local_shards"] and r["uint8"], r
        assert "optimizer_dcp" in r["files"] and "optimizer.pt" not in r["files"], r
        assert r["diff"] == [], r


@pytest.mark.parametrize("save_world, load_world", [(2, 1), (1, 2)])
def test_sharded_bnb_8bit_resume_on_another_world_size(save_world, load_world, tmp_path):
    _skip_without_cpu_8bit()
    _run("reshard_save", str(tmp_path), world = save_world)
    for rank, r in _run("reshard_load", str(tmp_path), world = load_world).items():
        assert r["world"] == load_world and "error" not in r, r
        assert r["model_diff"] == [] and r["missing"] == [], r
        assert r["steps"] == [(2, 2)], r
        # Re-quantized on the new world's blocks: within 8-bit blockwise error, not bitwise.
        assert r["max_rel"] < 2e-2, r
        assert r["loss"] == r["loss"], r
