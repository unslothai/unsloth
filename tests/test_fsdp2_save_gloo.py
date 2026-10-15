# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Saving under FSDP2 (unsloth/models/fsdp2_save.py), two gloo ranks on CPU.

- `fsdp2_gathered`: rank 0 sees whole parameters (on CPU) equal to an unsharded copy, also after an eval
  forward left a unit unsharded; the other rank writes nothing; the sharded parameters come back and
  training continues; a nested call does not gather again.
- bitsandbytes 4-bit weights come back as Params4bit with their QuantState and the exact packed bytes.
- trainable parameters FSDP2 leaves unsharded (`ignored_params`) drift apart across ranks unless their
  gradients are averaged (`reduce_unsharded_trainable_grads`).
- torch.distributed.checkpoint cannot build a state dict of a bitsandbytes 4-bit model (the Trainer's
  FSDP2 checkpoint) until `install_dcp_bnb_state_dict`.
The modules are loaded from their files (importing unsloth needs a GPU).
"""

import importlib.util
import os
import socket
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = _ROOT / "unsloth" / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_unsloth_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fsdp_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
if sys.platform == "win32" or not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)

N_LAYERS = 3


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Block(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        # Odd sizes so the shards are uneven.
        self.base = torch.nn.Linear(37, 37, bias = False).requires_grad_(False)
        self.lora_A = torch.nn.Linear(37, 4, bias = False)
        self.lora_B = torch.nn.Linear(4, 37, bias = False)
        torch.nn.init.normal_(self.lora_B.weight, std = 0.1)

    def forward(self, x):
        return x + torch.tanh(self.base(x) + self.lora_B(self.lora_A(x)))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(10 * i) for i in range(N_LAYERS)])
        torch.manual_seed(99)
        self.head = torch.nn.Linear(37, 5, bias = True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _shard(model, world_size, ignored = None):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (world_size,))
    kwargs = {"ignored_params": ignored} if ignored else {}
    for layer in model.layers:
        fully_shard(layer, mesh = mesh, **kwargs)
    fully_shard(model, mesh = mesh, **kwargs)
    # What Unsloth does to the last block: it stays unsharded after a forward.
    model.layers[-1].set_reshard_after_forward(False)
    return model


def _is_dtensor(t):
    from torch.distributed.tensor import DTensor
    return isinstance(t, DTensor)


def _case_gather(rank, world_size, m):
    model = _shard(_Model(), world_size)
    reference = _Model()
    x = torch.randn(6, 37, generator = torch.Generator().manual_seed(rank))
    with torch.no_grad():
        model(x)
    # The eval forward left the last block unsharded: a plain (non DTensor) parameter.
    unsharded_after_eval = sum(not _is_dtensor(p) for p in model.parameters())
    out = {"unsharded_after_eval": unsharded_after_eval}

    with m.fsdp2_gathered(model, trainable_only = True) as writer:
        out["trainable_writer"] = writer
        out["trainable_only"] = sorted(
            {("plain" if not _is_dtensor(p) else "dtensor", p.requires_grad) for p in model.parameters()}
        )

    with m.fsdp2_gathered(model) as writer:
        out["writer"] = writer
        with m.fsdp2_gathered(model) as nested:
            out["nested"] = nested
        if writer:
            ref = dict(reference.named_parameters())
            got = dict(model.named_parameters())
            out["names_match"] = sorted(ref) == sorted(got)
            out["plain_cpu"] = all(not _is_dtensor(p) and p.device.type == "cpu" for p in got.values())
            out["exact"] = all(torch.equal(got[k].detach(), ref[k].detach()) for k in ref)
            out["state_dict_keys"] = sorted(model.state_dict()) == sorted(reference.state_dict())
            out["requires_grad_kept"] = all(got[k].requires_grad == ref[k].requires_grad for k in ref)
        else:
            out["all_dtensor_on_other_rank"] = all(_is_dtensor(p) for p in model.parameters())
    out["all_dtensor_after"] = all(_is_dtensor(p) for p in model.parameters())

    # Training continues on the restored shards and matches the unsharded model step for step.
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.1)
    ref_opt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr = 0.1)
    import torch.distributed as dist
    xs = [torch.randn(6, 37, generator = torch.Generator().manual_seed(100 + r)) for r in range(world_size)]
    model(xs[rank]).square().mean().backward()
    opt.step()
    sum(reference(xr).square().mean() for xr in xs).div(world_size).backward()
    ref_opt.step()
    diffs = []
    for (n, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        full = p.full_tensor() if _is_dtensor(p) else p
        diffs.append(float((full.detach() - q.detach()).abs().max()))
    out["step_max_diff"] = max(diffs)
    return out


def _case_gather_4bit(rank, world_size, m):
    bnb = pytest.importorskip("bitsandbytes")
    q = _load("fsdp2_qlora")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    def lin(seed):
        torch.manual_seed(seed)
        layer = bnb.nn.Linear4bit(128, 96, bias = False, compute_dtype = torch.bfloat16,
                                  quant_type = "nf4", quant_storage = torch.bfloat16)
        layer.weight = bnb.nn.Params4bit(torch.randn(96, 128) * 0.05, requires_grad = False,
                                         quant_type = "nf4", quant_storage = torch.bfloat16,
                                         blocksize = 64, module = layer)
        return layer.to("cpu")

    model = torch.nn.Sequential(lin(0), lin(1))
    before = [(l.weight.data.view(torch.uint8).clone(),
               bnb.functional.dequantize_4bit(l.weight.data, l.weight.quant_state)) for l in model]
    info = q.prepare_fsdp2_qlora(model)
    mesh = init_device_mesh("cpu", (world_size,))
    for layer in model:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    out = {"patched": info["patched"]}
    with m.fsdp2_gathered(model) as writer:
        if writer:
            checks = []
            for layer, (raw, deq) in zip(model, before):
                w = layer.weight
                checks.append(dict(
                    params4bit = type(w).__name__ == "Params4bit",
                    quant_state = w.quant_state is not None,
                    bytes_equal = torch.equal(w.data.view(torch.uint8), raw),
                    dequant_equal = torch.equal(bnb.functional.dequantize_4bit(w.data, w.quant_state), deq),
                    module = w.module is layer,
                ))
            out["checks"] = checks
            # bitsandbytes' own state dict (what save_pretrained writes) has the QuantState entries.
            out["absmax_keys"] = sum(k.endswith("weight.absmax") for k in model.state_dict())
    out["dtensor_after"] = all(_is_dtensor(p) for p in model.parameters())
    return out


def _case_replicated(rank, world_size, m, hooked):
    import torch.distributed as dist

    model = _Model()
    reference = _Model()
    head = [model.head.weight, model.head.bias]
    _shard(model, world_size, ignored = set(head))
    names = m.reduce_unsharded_trainable_grads(model) if hooked else []
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.5)
    ref_opt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr = 0.5)
    for step in range(3):
        xs = [torch.randn(6, 37, generator = torch.Generator().manual_seed(10 * step + r)) for r in range(world_size)]
        opt.zero_grad()
        model(xs[rank]).square().mean().backward()
        opt.step()
        ref_opt.zero_grad()
        sum(reference(xr).square().mean() for xr in xs).div(world_size).backward()
        ref_opt.step()
    w = model.head.weight.detach().clone()
    gathered = [torch.empty_like(w) for _ in range(world_size)]
    dist.all_gather(gathered, w)
    return dict(
        names = names,
        head_is_dtensor = _is_dtensor(model.head.weight),
        rank_spread = max(float((g - gathered[0]).abs().max()) for g in gathered),
        vs_reference = float((w - reference.head.weight.detach()).abs().max()),
    )


def _case_dcp_bnb(rank, world_size, m):
    bnb = pytest.importorskip("bitsandbytes")
    q = _load("fsdp2_qlora")
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    import torch.distributed.checkpoint.state_dict as dcp

    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(128, 64, bias = False, compute_dtype = torch.bfloat16,
                              quant_type = "nf4", quant_storage = torch.bfloat16)
    layer.weight = bnb.nn.Params4bit(torch.randn(64, 128) * 0.05, requires_grad = False,
                                     quant_type = "nf4", quant_storage = torch.bfloat16,
                                     blocksize = 64, module = layer)
    model = torch.nn.Sequential(layer.to("cpu"), torch.nn.Linear(64, 8))
    q.prepare_fsdp2_qlora(model)
    fully_shard(model, mesh = init_device_mesh("cpu", (world_size,)))
    # What accelerate's save_fsdp_model / load_fsdp_model ask for on an adapter checkpoint.
    options = StateDictOptions(full_state_dict = True, cpu_offload = True, ignore_frozen_params = True)
    out = {}
    try:
        dcp.get_model_state_dict(model, options = options)
        out["before"] = "ok"
    except AttributeError as error:
        out["before"] = f"AttributeError: {error}"
    out["installed"] = m.install_dcp_bnb_state_dict()
    sd = dcp.get_model_state_dict(model, options = options)
    out["after_keys"] = sorted(sd)
    expected = {k: v.clone() for k, v in sd.items()}
    with torch.no_grad():
        model[1].weight.zero_()
    dcp.set_model_state_dict(model, sd if rank == 0 else {},
                             options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True, strict = False))
    restored = model[1].weight.full_tensor()
    out["roundtrip"] = rank != 0 or torch.equal(restored, expected["1.weight"])
    return out


def _worker(rank, world_size, port, case, queue):
    try:
        os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port),
                          RANK = str(rank), WORLD_SIZE = str(world_size))
        import torch.distributed as dist

        dist.init_process_group("gloo", rank = rank, world_size = world_size)
        m = _load("fsdp2_save")
        if case == "gather":
            result = _case_gather(rank, world_size, m)
        elif case == "gather_4bit":
            result = _case_gather_4bit(rank, world_size, m)
        elif case.startswith("replicated"):
            result = _case_replicated(rank, world_size, m, hooked = case.endswith("hooked"))
        else:
            result = _case_dcp_bnb(rank, world_size, m)
        queue.put((rank, result))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _run(case, world_size = 2):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target = _worker, args = (r, world_size, port, case, queue)) for r in range(world_size)]
    for p in procs:
        p.start()
    results = {}
    for _ in procs:
        rank, result = queue.get(timeout = 240)
        results[rank] = result
    for p in procs:
        p.join(timeout = 60)
    for r in results.values():
        assert "error" not in r, r.get("error")
    return results


def test_gather_gives_rank0_the_full_model_and_restores_shards():
    results = _run("gather")
    r0, r1 = results[0], results[1]
    assert r0["unsharded_after_eval"] > 0, r0
    assert r0["trainable_writer"] is True and r1["trainable_writer"] is False
    # trainable_only: adapters plain on rank 0, frozen weights stay sharded.
    assert r0["trainable_only"] == [("dtensor", False), ("plain", True)], r0
    assert r1["trainable_only"] == [("dtensor", False), ("dtensor", True)], r1
    assert r0["writer"] is True and r1["writer"] is False
    assert r0["nested"] is True and r1["nested"] is False
    assert r0["names_match"] and r0["plain_cpu"] and r0["exact"], r0
    assert r0["state_dict_keys"] and r0["requires_grad_kept"], r0
    assert r1["all_dtensor_on_other_rank"], r1
    for r in (r0, r1):
        assert r["all_dtensor_after"], r
        assert r["step_max_diff"] < 1e-6, r


def test_gather_rebuilds_params4bit_with_exact_bytes():
    pytest.importorskip("bitsandbytes")
    results = _run("gather_4bit")
    r0 = results[0]
    assert r0["patched"], r0
    assert len(r0["checks"]) == 2 and all(all(c.values()) for c in r0["checks"]), r0["checks"]
    assert r0["absmax_keys"] == 2, r0
    assert all(r["dtensor_after"] for r in results.values()), results


@pytest.mark.skipif(not hasattr(torch.distributed.fsdp, "fully_shard") or
                    "ignored_params" not in __import__("inspect").signature(
                        torch.distributed.fsdp.fully_shard).parameters,
                    reason = "fully_shard has no ignored_params")
def test_unsharded_trainable_params_need_their_grads_averaged():
    base = _run("replicated")
    assert not base[0]["head_is_dtensor"], base
    # Control: no gradient reduction for the ignored head, so the replicas drift apart.
    assert base[0]["rank_spread"] > 1e-3, base
    hooked = _run("replicated_hooked")
    assert hooked[0]["names"] == ["head.weight", "head.bias"], hooked
    assert hooked[0]["rank_spread"] == 0.0, hooked
    for r in hooked.values():
        assert r["vs_reference"] < 1e-6, r


def test_dcp_state_dict_of_bitsandbytes_4bit():
    pytest.importorskip("bitsandbytes")
    results = _run("dcp_bnb")
    r = results[0]
    # A sharded 4-bit weight is a DTensor, which has no `absmax` attribute for `weight.absmax`.
    assert r["before"].startswith("AttributeError") and "absmax" in r["before"], r
    assert r["installed"], r
    assert r["after_keys"] == ["1.bias", "1.weight"], r
    assert all(x["roundtrip"] for x in results.values()), results
