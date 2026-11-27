# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2:

* fsdp2_qlora's FSDP2 checks find FSDP2's private modules on torch 2.4 / 2.5's `_composable.fsdp`
  layout too (the new layout hidden in sys.modules);
* a Clef model (DecisionTrainer) is refused at Trainer setup: its `_save` is not collective, and a
  rank-0-only save that enters a collective hangs (gloo world of 2);
* Trainer.__init__ does not `.to(device)` a natively sharded model (CPU offload keeps shards on the
  CPU; everything else is already on the mesh device)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _stub_unsloth_save  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import (  # noqa: E402
    _install_wrapped_trainer_init,
    _lora,
    _targs,
)
from test_fsdp2_native_guards_gloo import _tiny_llama  # noqa: E402


def _load_qlora():
    path = os.path.join(REPO, "unsloth", "models", "fsdp2_qlora.py")
    spec = importlib.util.spec_from_file_location("_fsdp2_qlora_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qlora_checks_find_fsdp2_on_the_composable_layout(monkeypatch):
    qlora = _load_qlora()
    assert qlora._missing_fsdp2_anchors() == []  # this torch: the new layout
    real = {
        name: importlib.import_module("torch.distributed.fsdp._fully_shard." + name)
        for name in ("_fsdp_param", "_fsdp_collectives", "_fsdp_param_group", "_fsdp_init")
    }
    # torch 2.4 / 2.5: no torch.distributed.fsdp._fully_shard; the same modules under _composable.fsdp.
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp._fully_shard", None)
    for name, module in real.items():
        monkeypatch.setitem(sys.modules, "torch.distributed.fsdp._fully_shard." + name, None)
        monkeypatch.setitem(sys.modules, "torch.distributed._composable.fsdp." + name, module)
    assert qlora._missing_fsdp2_anchors() == []
    assert qlora._fsdp_private("_fsdp_init") is real["_fsdp_init"]


class _CollectiveSave:
    """Stands for DecisionTrainer: `_save` runs a collective (the encoder's gathered save_pretrained)."""

    def _save(
        self,
        output_dir = None,
        state_dict = None,
    ):
        import torch.distributed as dist
        dist.all_reduce(torch.ones(1))


def _case_clef(native, rank, tmp):
    _install_wrapped_trainer_init()
    from transformers import Trainer

    class DecisionLike(_CollectiveSave, Trainer):
        pass

    model = _tiny_llama()
    model.is_clef = True
    res = {}
    try:
        trainer = DecisionLike(model = model, args = _targs(os.path.join(tmp, "out")))
        res["init"] = "ok"
    except Exception as exc:
        res["init"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        res["sharded"] = native.is_native_fsdp2(model)
        return res
    try:
        trainer.save_model(os.path.join(tmp, "ckpt"))
        res["save"] = "ok"
    except Exception as exc:
        res["save"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return res


def _case_placement(native, tmp):
    from transformers import Trainer

    model = native.shard_model(_lora(1), native.FSDP2Config(param_dtype = torch.float32))
    calls = []
    original = torch.nn.Module.to

    def spy(self, *args, **kwargs):
        if self is model:
            calls.append(str(args or kwargs))
        return original(self, *args, **kwargs)

    torch.nn.Module.to = spy
    try:
        Trainer(model = model, args = _targs(os.path.join(tmp, "out")))
    finally:
        torch.nn.Module.to = original
    return {"to_calls": len(calls)}


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 20)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        if case == "clef":
            res = _case_clef(native, rank, tmp)
        elif case == "placement":
            res = _case_placement(native, tmp)
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
        print("ROUND9_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_clef_models_are_refused_before_a_non_collective_save(tmp_path):
    pytest.importorskip("transformers")
    for rank, r in _run("clef", tmp_path).items():
        assert r["init"].startswith("RuntimeError") and "Clef" in r["init"], (rank, r)
        assert r["sharded"] is False, (rank, r)


def test_trainer_does_not_move_a_natively_sharded_model(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("placement", tmp_path).items():
        assert r["to_calls"] == 0, (rank, r)
