# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 guards, on a gloo CPU world of 2 where a process group is needed:

* a resume restores every adapter PEFT saved (inactive ones under <ckpt>/<name>/ too);
* TrainingArguments(deepspeed = ...) / a DeepSpeed plugin is refused before the model is sharded;
* a model still on the CPU is sharded onto the local rank's GPU when CUDA is used (a deliberate
  CPU / disk device_map and CPU 4-bit weights are refused);
* replicated trainable parameters average their gradients once per backward on every rank, also
  when one rank did not use a parameter (a per-parameter hook all_reduce hangs or mismatches)."""

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
from test_fsdp2_native_trainer_ckpt_gloo import (  # noqa: E402
    REPO_ID,
    _free_port,
    _full,
    _stub_unsloth_save,
)
from test_fsdp2_native_trainer_args_gloo import (  # noqa: E402
    _install_wrapped_trainer_init,
    _lora,
    _targs,
)


class _Replicated(torch.nn.Module):
    """A sharded body plus three replicated (ignored_params) parameters: `rep` used by every rank,
    `extra` used only where the forward says so, `never` used by no rank."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.body = torch.nn.Linear(8, 8)
        self.rep = torch.nn.Parameter(torch.zeros(3))
        self.extra = torch.nn.Parameter(torch.zeros(5))
        self.never = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x, k, c):
        out = self.body(x).sum() + k * self.rep.sum()
        if c:
            out = out + c * self.extra.sum()
        return out


def _case_replicated_unused(rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    _load_native()
    save = sys.modules.get("unsloth.models.fsdp2_save")
    if save is None:
        import importlib
        save = importlib.import_module("unsloth.models.fsdp2_save")
    model = _Replicated()
    mesh = init_device_mesh("cpu", (2,))
    fully_shard(model.body, mesh = mesh)
    fully_shard(model, mesh = mesh, ignored_params = {model.rep, model.extra, model.never})
    names = save.reduce_unsharded_trainable_grads(model)
    x = torch.randn(4, 8, generator = torch.Generator().manual_seed(rank))
    # Two accumulated micro-steps: `extra` is used by rank 0 in the first, by rank 1 in the second.
    model(x, k = float(rank + 1), c = 3.0 if rank == 0 else 0.0).backward()
    model(x, k = float(10 * (rank + 1)), c = 5.0 if rank == 1 else 0.0).backward()
    return {
        "names": sorted(names),
        "rep": model.rep.grad.tolist(),
        "extra": model.extra.grad.tolist() if model.extra.grad is not None else None,
        "never": None if model.never.grad is None else model.never.grad.tolist(),
    }


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
        res = {}
        if case == "all_adapters":
            from transformers import Trainer

            cfg = native.FSDP2Config(param_dtype = torch.float32)
            ckpt = os.path.join(tmp, "ckpt")
            src = native.shard_model(_lora(1, same_config_adapter = True), cfg)
            Trainer(model = src, args = _targs(os.path.join(tmp, "out"))).save_model(ckpt)
            dist.barrier()
            want = _full(src)
            dst = native.shard_model(_lora(2, same_config_adapter = True), cfg)
            try:
                native.native_load_model_checkpoint(dst, ckpt)
                got = _full(dst)
                for adapter in ("default", "other"):
                    keys = [n for n in want if f".{adapter}." in n]
                    res[f"n_{adapter}"] = len(keys)
                    res[f"{adapter}_diff"] = max(
                        (got[n] - want[n]).abs().max().item() for n in keys
                    )
            except Exception as exc:
                res["load_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        elif case == "deepspeed":
            _install_wrapped_trainer_init()
            from transformers import Trainer

            model = _lora(1)
            targs = _targs(os.path.join(tmp, "out"))
            targs.deepspeed = {"train_micro_batch_size_per_gpu": 1}  # deepspeed is not installed
            try:
                Trainer(model = model, args = targs)
                res["error"] = ""
            except Exception as exc:
                res["error"] = f"{type(exc).__name__}: {exc}"
            res["sharded"] = native.is_native_fsdp2(model)
        elif case == "replicated_unused":
            res = _case_replicated_unused(rank)
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
    timeout = 240,
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
    while not procs.join(timeout = 5):
        if time.time() > deadline:
            for p in procs.processes:
                p.kill()
            pytest.fail(f"{case}: the ranks did not finish within {timeout} s (hung collective)")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("GUARD_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_resume_restores_inactive_adapters(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("all_adapters", tmp_path).items():
        assert "load_error" not in r, (rank, r)
        assert r["n_default"] > 0 and r["default_diff"] == 0.0, (rank, r)
        assert r["n_other"] > 0 and r["other_diff"] == 0.0, (rank, r)


def test_deepspeed_training_args_are_refused_before_sharding(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("deepspeed", tmp_path).items():
        assert r["error"].startswith("RuntimeError") and "DeepSpeed" in r["error"], (rank, r)
        assert r["sharded"] is False, (rank, r)


def test_replicated_grads_sync_when_one_rank_skips_a_parameter(tmp_path):
    results = _run("replicated_unused", tmp_path, timeout = 120)
    for rank, r in results.items():
        assert r["names"] == ["extra", "never", "rep"], (rank, r)
        # (1 + 2) / 2 + (10 + 20) / 2, and (3 + 5) / 2 (each rank used `extra` in one micro-step).
        assert r["rep"] == [16.5] * 3, (rank, r)
        assert r["extra"] == [4.0] * 5, (rank, r)
        assert r["never"] is None, (rank, r)


# ---- CPU unit checks (no process group) -------------------------------------------------------


@pytest.fixture
def native_env(monkeypatch):
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising = False)
    return _load_native()


def test_deepspeed_plugin_and_env_are_refused(native_env, monkeypatch):
    native = native_env
    model = torch.nn.Linear(2, 2)
    args = types.SimpleNamespace(deepspeed = None, deepspeed_plugin = object(), save_on_each_node = False)
    with pytest.raises(RuntimeError, match = "DeepSpeed"):
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = args)
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    with pytest.raises(RuntimeError, match = "DeepSpeed"):
        native.native_fsdp2_trainer_setup(model = model, phase = "pre", training_args = None)


class _Stop(Exception):
    pass


def _tiny_llama():
    import transformers
    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    return transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)


def _shard_device_seen(native, monkeypatch, model):
    seen = {}

    def _record(device, *_):
        seen["device"] = device
        raise _Stop

    monkeypatch.setattr(native, "_ensure_process_group", _record)
    with pytest.raises(_Stop):
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
    return seen["device"]


def test_cpu_model_is_sharded_onto_the_local_gpu(native_env, monkeypatch):
    pytest.importorskip("transformers")
    native = native_env
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert _shard_device_seen(native, monkeypatch, _tiny_llama()) == torch.device("cuda", 1)


def test_cpu_offloaded_or_4bit_model_is_refused(native_env, monkeypatch):
    pytest.importorskip("transformers")
    native = native_env
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("LOCAL_RANK", "1")
    model = _tiny_llama()
    model.hf_device_map = {"model.embed_tokens": 0, "model.layers": "cpu", "lm_head": "disk"}
    with pytest.raises(RuntimeError, match = "offloads weights"):
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))

    class Params4bit(torch.nn.Parameter):
        pass

    model = _tiny_llama()
    proj = model.model.layers[0].self_attn.q_proj
    proj.weight = Params4bit(proj.weight.detach().clone(), requires_grad = False)
    with pytest.raises(RuntimeError, match = "4-bit model on the CPU"):
        native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
    # No GPU in use (CUDA unavailable): the CPU run keeps its CPU mesh.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _shard_device_seen(native, monkeypatch, _tiny_llama()).type == "cpu"
