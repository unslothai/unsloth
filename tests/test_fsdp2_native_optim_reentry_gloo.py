# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2 where a process group is needed):

* a prebuilt optimizer over mixed DTensor / replicated groups that already holds state is refused
  (its groups cannot be split), also without an LR scheduler;
* Trainer(optimizer_cls_and_kwargs = (bitsandbytes class, ...)), by keyword or position, is refused
  before sharding; a torch optimizer class passes;
* native_fsdp2_unsharded is re-entrant: generate() inside an outer block leaves the weights gathered
  until the outer block exits, also when an inner block raises;
* accelerate's FP8 mixed precision is refused before any model change."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import (  # noqa: E402
    _install_wrapped_trainer_init,
    _lora,
    _targs,
)
from test_fsdp2_native_guards_gloo import _Stop, _tiny_llama  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _stub_unsloth_save  # noqa: E402


class AdamW8bitLike(torch.optim.AdamW):
    """Stands for a bitsandbytes optimizer class (only its module is read)."""


AdamW8bitLike.__module__ = "bitsandbytes.optim.adamw"


def _sharded_lora(native):
    model = _lora(1)
    first = next(p for n, p in model.named_parameters() if "lora_A" in n)
    cfg = native.FSDP2Config(param_dtype = torch.float32, ignored_params = {first})
    return native.shard_model(model, cfg)


def _case_stateful(native, tmp):
    _install_wrapped_trainer_init()
    from transformers import Trainer

    model = _sharded_lora(native)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr = 1e-3, foreach = False)
    ids = torch.randint(1, 100, (1, 8))
    model(input_ids = ids, labels = ids).loss.backward()
    optimizer.step()  # the optimizer now holds state (as after load_state_dict)
    res = {"state": len(optimizer.state)}
    try:
        trainer = Trainer(
            model = model, args = _targs(os.path.join(tmp, "out")), optimizers = (optimizer, None)
        )
        trainer.create_optimizer()
        res["mixed_groups_left"] = native._mixed_param_groups(optimizer)
    except RuntimeError as exc:
        res["error"] = str(exc)
    return res


def _case_optimizer_cls(native, tmp):
    _install_wrapped_trainer_init()
    from transformers import Trainer

    classes = [("like", AdamW8bitLike)]
    try:
        import bitsandbytes as bnb
        classes.append(("bnb", bnb.optim.AdamW8bit))
    except Exception:
        pass
    res = {}
    for name, cls in classes + [("torch", torch.optim.AdamW)]:
        for how in ("keyword", "positional"):
            model = _lora(1)
            targs = _targs(os.path.join(tmp, "out"))
            try:
                if how == "keyword":
                    Trainer(model = model, args = targs, optimizer_cls_and_kwargs = (cls, {"lr": 1e-3}))
                else:
                    # model, args, data_collator, train_dataset, eval_dataset, processing_class,
                    # model_init, compute_loss_func, compute_metrics, callbacks, optimizers,
                    # optimizer_cls_and_kwargs.
                    Trainer(
                        model,
                        targs,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        (None, None),
                        (cls, {"lr": 1e-3}),
                    )
                err = ""
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
            res[f"{name}.{how}"] = {"error": err, "sharded": native.is_native_fsdp2(model)}
    # #15008: a real bitsandbytes Adam / AdamW steps local shards when this bitsandbytes can.
    res["_bnb_local"] = bool(native._bnb_local_shards_available())
    return res


def _case_reentry(native):
    model = native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))
    return _case_reentry_on(native, model)


def _case_reentry_on(native, model):
    from torch.distributed.tensor import DTensor

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = None

    def n_sharded():
        return sum(isinstance(p, DTensor) for p in model.parameters())

    res = {"before": n_sharded()}
    ids = torch.randint(1, 100, (1, 4), generator = torch.Generator().manual_seed(0))
    with native.native_fsdp2_unsharded(model):
        res["outer"] = n_sharded()
        model.generate(input_ids = ids, max_new_tokens = 2, do_sample = False)
        res["after_generate"] = n_sharded()
        try:
            with native.native_fsdp2_unsharded(model):
                raise KeyError("inner")
        except KeyError:
            pass
        res["after_inner_raise"] = n_sharded()
    res["after_outer"] = n_sharded()
    res["depth"] = model.__dict__.get("_unsloth_unsharded_depth", 0)
    return res


def _case_reentry_no_setter(native):
    """torch 2.6 / 2.7: FSDPModule has no set_reshard_after_forward (added in 2.8)."""
    FSDPModule = native._fsdp_module_class()
    removed = {}
    for cls in FSDPModule.__mro__:
        if "set_reshard_after_forward" in cls.__dict__:
            removed[cls] = cls.__dict__["set_reshard_after_forward"]
            delattr(cls, "set_reshard_after_forward")
    try:
        model = native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))
        units = [m for m in model.modules() if isinstance(m, FSDPModule)]
        meshes = [u._get_fsdp_state()._fsdp_param_group for u in units]
        before = [g.post_forward_mesh_info if g is not None else None for g in meshes]
        res = {"has_setter": hasattr(units[0], "set_reshard_after_forward")}
        res.update(_case_reentry_on(native, model))
        after = [g.post_forward_mesh_info if g is not None else None for g in meshes]
        res["mesh_restored"] = all(a is b for a, b in zip(before, after))
        res["units"] = len(units)
        return res
    finally:
        for cls, fn in removed.items():
            setattr(cls, "set_reshard_after_forward", fn)


def _worker_case(case, native, rank, tmp):
    if case == "stateful":
        return _case_stateful(native, tmp)
    if case == "optimizer_cls":
        return _case_optimizer_cls(native, tmp)
    if case == "reentry":
        return _case_reentry(native)
    if case == "reentry_no_setter":
        return _case_reentry_no_setter(native)
    raise KeyError(case)


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
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        out.put((rank, _worker_case(case, native, rank, tmp)))
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
    timeout = 180,
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
        print("ROUND5_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_stateful_mixed_optimizer_is_refused(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("stateful", tmp_path).items():
        assert r["state"] > 0, (rank, r)
        assert "split_param_groups" in r.get("error", ""), (rank, r)


def test_bnb_optimizer_cls_and_kwargs_is_refused(tmp_path):
    """A torch class is sharded. A bitsandbytes Adam / AdamW is kept and sharded where this
    bitsandbytes steps local shards (#15008), else refused. A class that only claims the
    bitsandbytes module (AdamW8bitLike) cannot step local shards, so it is always refused."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("optimizer_cls", tmp_path).items():
        bnb_local = r.pop("_bnb_local")
        for key, v in r.items():
            if key.startswith("torch.") or (key.startswith("bnb.") and bnb_local):
                assert v["error"] == "" and v["sharded"], (rank, key, v, bnb_local)
            else:
                assert v["error"].startswith("ValueError") and "bitsandbytes" in v["error"], (
                    rank,
                    key,
                    v,
                    bnb_local,
                )
                assert not v["sharded"], (rank, key, v)


def test_unsharded_context_without_reshard_setter(tmp_path):
    """torch 2.6 / 2.7 FSDPModule has no set_reshard_after_forward: the gather / reshard still runs
    and every unit's post-forward mesh comes back unchanged."""
    pytest.importorskip("transformers")
    for rank, r in _run("reentry_no_setter", tmp_path).items():
        assert r["has_setter"] is False and r["units"] > 1, (rank, r)
        assert r["before"] > 0 and r["outer"] == 0, (rank, r)
        assert r["after_generate"] == 0 and r["after_inner_raise"] == 0, (rank, r)
        assert r["after_outer"] == r["before"] and r["depth"] == 0, (rank, r)
        assert r["mesh_restored"], (rank, r)


def test_unsharded_context_is_reentrant(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("reentry", tmp_path).items():
        assert r["before"] > 0 and r["outer"] == 0, (rank, r)
        assert r["after_generate"] == 0 and r["after_inner_raise"] == 0, (rank, r)
        assert r["after_outer"] == r["before"] and r["depth"] == 0, (rank, r)


# ---- CPU unit checks (no process group) -------------------------------------------------------


def test_accelerate_fp8_is_refused_before_sharding(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    # transformers 5: TrainingArguments reads ACCELERATE_MIXED_PRECISION into args.mixed_precision.
    # Built before the distributed env below (a WORLD_SIZE without MASTER_ADDR fails its setup).
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp8")
    from transformers import TrainingArguments

    real = TrainingArguments(output_dir = str(tmp_path / "out"), report_to = [], use_cpu = True)
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising = False)
    native = _load_native()

    def _stop(device, *_):
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _stop)
    base = dict(optim = "adamw_torch", save_on_each_node = False, fsdp = None)
    model = torch.nn.Linear(2, 2)
    for args in (
        real,
        types.SimpleNamespace(mixed_precision = "fp8", **base),
        types.SimpleNamespace(fp16 = False, bf16 = False, **base),  # 4.x: only the env
    ):
        with pytest.raises(RuntimeError, match = "FP8"):
            native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
    # bf16 overrides the env (both transformers lines): not FP8, sharding proceeds.
    args = types.SimpleNamespace(fp16 = False, bf16 = True, **base)
    with pytest.raises(_Stop):
        native.native_fsdp2_trainer_setup(model = _tiny_llama(), phase = "pre", training_args = args)
