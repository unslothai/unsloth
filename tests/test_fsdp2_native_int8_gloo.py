# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 with bitsandbytes 8-bit (LLM.int8, Linear8bitLt) weights, gloo world of 2. Int8Params
keeps its quantization state in CB / SCB attributes beside the int8 data
(bitsandbytes/nn/modules.py Int8Params.__new__ / _quantize); a DTensor shard and FSDP2's gathered
plain tensor carry neither. So 8-bit weights stay replicated (ignored_params) in every layout, also
when frozen weights are sharded (shard_frozen = True), and the sharded model's forward matches the
unsharded one; a torch without fully_shard(ignored_params) refuses."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import copy  # noqa: E402
import datetime  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_guards_gloo import _tiny_llama  # noqa: E402


def _int8_llama():
    import bitsandbytes as bnb

    torch.manual_seed(0)
    model = _tiny_llama()
    for layer in model.model.layers:
        old = layer.mlp.down_proj
        new = bnb.nn.Linear8bitLt(
            old.in_features, old.out_features, bias = False, has_fp16_weights = False
        )
        new.weight = bnb.nn.Int8Params(
            old.weight.detach().clone(), requires_grad = False, has_fp16_weights = False
        )
        layer.mlp.down_proj = new.to("cpu")  # quantizes: int8 data + CB / SCB
    for name, p in model.named_parameters():
        p.requires_grad_("norm" in name)  # a trainable float set next to the frozen int8 weights
    return model


def _case(native, rank, refuse):
    from torch.distributed.tensor import DTensor

    model = _int8_llama()
    reference = copy.deepcopy(model)
    if refuse:
        native._supports_ignored_params = lambda fully_shard: False
    cfg = native.FSDP2Config(param_dtype = torch.float32, shard_frozen = True)
    try:
        model = native.shard_model(model, cfg)
    except Exception as exc:
        return {"shard": f"{type(exc).__name__}: {str(exc)[:300]}"}
    weights = [layer.mlp.down_proj.weight for layer in model.model.layers]
    res = {
        "shard": "ok",
        "int8_dtensor": sum(isinstance(w, DTensor) for w in weights),
        "int8_scb": sum(getattr(w, "SCB", None) is not None for w in weights),
        "int8": len(weights),
        "other_dtensor": sum(isinstance(p, DTensor) for p in model.parameters()),
    }
    ids = torch.randint(1, 100, (1, 8), generator = torch.Generator().manual_seed(rank))
    try:
        got = model(input_ids = ids, labels = ids)
        got.loss.backward()
        want = reference(input_ids = ids, labels = ids)
        res["logits_diff"] = (got.logits - want.logits).abs().max().item()
    except Exception as exc:
        res["forward"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


def _worker(rank, world, port, out, refuse):
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
        out.put((rank, _case(_load_native(), rank, refuse)))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(refuse, timeout = 150):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, refuse),
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
                pytest.fail("int8: the ranks did not finish (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"int8: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ROUND11_RESULT int8", refuse, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_int8_weights_stay_replicated_when_frozen_weights_are_sharded():
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("transformers")
    for rank, r in _run(False).items():
        assert r["shard"] == "ok" and "forward" not in r, (rank, r)
        assert r["int8_dtensor"] == 0 and r["int8_scb"] == r["int8"] > 0, (rank, r)
        assert r["other_dtensor"] > 0, (rank, r)  # the rest is still sharded
        assert r["logits_diff"] < 1e-4, (rank, r)


def test_int8_weights_refused_without_ignored_params():
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("transformers")
    for rank, r in _run(True).items():
        assert r["shard"].startswith("RuntimeError") and "8-bit" in r["shard"], (rank, r)


def test_cpu_int8_model_with_a_gpu_is_refused(monkeypatch):
    """Moving the int8 data to the GPU would leave CB / SCB on the CPU."""
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("transformers")
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0", LOCAL_RANK = "1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ACCELERATE_USE_CPU", raising = False)
    native = _load_native()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match = "8-bit model on the CPU"):
        native._shard_device(_int8_llama())
