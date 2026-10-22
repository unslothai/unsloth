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
if (
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available()
):
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)

N_LAYERS = 3


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


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


def _shard(
    model,
    world_size,
    ignored = None,
):
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
            {
                ("plain" if not _is_dtensor(p) else "dtensor", p.requires_grad)
                for p in model.parameters()
            }
        )

    # A frozen parameter the save still writes (an inactive adapter, an embedding).
    with m.fsdp2_gathered(model, trainable_only = True, include = lambda name, p: True) as writer:
        out["included"] = sorted(
            {
                ("plain" if not _is_dtensor(p) else "dtensor", p.requires_grad)
                for p in model.parameters()
            }
        )

    with m.fsdp2_gathered(model) as writer:
        out["writer"] = writer
        with m.fsdp2_gathered(model) as nested:
            out["nested"] = nested
        if writer:
            ref = dict(reference.named_parameters())
            got = dict(model.named_parameters())
            out["names_match"] = sorted(ref) == sorted(got)
            out["plain_cpu"] = all(
                not _is_dtensor(p) and p.device.type == "cpu" for p in got.values()
            )
            out["exact"] = all(torch.equal(got[k].detach(), ref[k].detach()) for k in ref)
            out["state_dict_keys"] = sorted(model.state_dict()) == sorted(reference.state_dict())
            out["requires_grad_kept"] = all(
                got[k].requires_grad == ref[k].requires_grad for k in ref
            )
        else:
            out["all_dtensor_on_other_rank"] = all(_is_dtensor(p) for p in model.parameters())
    out["all_dtensor_after"] = all(_is_dtensor(p) for p in model.parameters())

    # Training continues on the restored shards and matches the unsharded model step for step.
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.1)
    ref_opt = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr = 0.1)
    import torch.distributed as dist

    xs = [
        torch.randn(6, 37, generator = torch.Generator().manual_seed(100 + r))
        for r in range(world_size)
    ]
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
        layer = bnb.nn.Linear4bit(
            128,
            96,
            bias = False,
            compute_dtype = torch.bfloat16,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
        )
        layer.weight = bnb.nn.Params4bit(
            torch.randn(96, 128) * 0.05,
            requires_grad = False,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = 64,
            module = layer,
        )
        return layer.to("cpu")

    model = torch.nn.Sequential(lin(0), lin(1))
    before = [
        (
            l.weight.data.view(torch.uint8).clone(),
            bnb.functional.dequantize_4bit(l.weight.data, l.weight.quant_state),
        )
        for l in model
    ]
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
                checks.append(
                    dict(
                        params4bit = type(w).__name__ == "Params4bit",
                        quant_state = w.quant_state is not None,
                        bytes_equal = torch.equal(
                            w.data.reshape(-1).view(torch.uint8), raw.reshape(-1)
                        ),
                        # Saved as the uint8 bitsandbytes writes without FSDP2 (merged_4bit requantizes
                        # with the weight's own quant_storage).
                        uint8_storage = w.dtype == torch.uint8 and w.quant_storage == torch.uint8,
                        dequant_equal = torch.equal(
                            bnb.functional.dequantize_4bit(w.data, w.quant_state), deq
                        ),
                        module = w.module is layer,
                    )
                )
            out["checks"] = checks
            # bitsandbytes' own state dict (what save_pretrained writes) has the QuantState entries.
            out["absmax_keys"] = sum(k.endswith("weight.absmax") for k in model.state_dict())
    out["dtensor_after"] = all(_is_dtensor(p) for p in model.parameters())
    return out


class Linear4bit(torch.nn.Linear):
    """Stands in for bitsandbytes' Linear4bit (matched by class name)."""


class _LoraLayer(torch.nn.Module):
    """PEFT's LoraLayer shape: `base_layer` + `lora_A` / `lora_B` ModuleDicts."""

    def __init__(self, seed, quantized):
        super().__init__()
        torch.manual_seed(seed)
        base = Linear4bit if quantized else torch.nn.Linear
        self.base_layer = base(37, 37, bias = False).requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(37, 4, bias = False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 37, bias = False)})

    def forward(self, x):
        return x + torch.tanh(
            self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))
        )


class _LoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_LoraLayer(10 * i, quantized = i != 1) for i in range(N_LAYERS)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _case_merge_offload(rank, world_size, m):
    """merged_4bit after an eval forward: the filter is built while the last unit is unsharded, the
    gather reshards it first. Every adapter of a 4-bit layer must stay put (False), everything else
    moves to the CPU (True)."""
    model = _shard(_LoraModel(), world_size)
    with torch.no_grad():
        model(torch.randn(6, 37))
    out = {"unsharded_after_eval": sum(not _is_dtensor(p) for p in model.parameters())}
    for layer in model.layers:
        if isinstance(layer.base_layer, Linear4bit):
            layer.base_layer.weight.quant_state = object()  # what the filter reads today
    offload = m.offload_all_but_4bit_merge_operands(model)
    decisions = {}

    def record(name, param):
        decisions[name] = offload(name, param)
        return decisions[name]

    with m.fsdp2_gathered(model, offload_to_cpu = record):
        pass
    out["decisions"] = decisions
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
        xs = [
            torch.randn(6, 37, generator = torch.Generator().manual_seed(10 * step + r))
            for r in range(world_size)
        ]
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
    layer = bnb.nn.Linear4bit(
        128,
        64,
        bias = False,
        compute_dtype = torch.bfloat16,
        quant_type = "nf4",
        quant_storage = torch.bfloat16,
    )
    layer.weight = bnb.nn.Params4bit(
        torch.randn(64, 128) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        quant_storage = torch.bfloat16,
        blocksize = 64,
        module = layer,
    )
    model = torch.nn.Sequential(layer.to("cpu"), torch.nn.Linear(64, 8))
    reference_dequant = bnb.functional.dequantize_4bit(
        model[0].weight.data, model[0].weight.quant_state
    )
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
    # What Trainer.save_model asks for through accelerate's get_state_dict: a full state dict, frozen
    # weights included. bitsandbytes' extra keys cannot be resolved on a sharded weight either way.
    try:
        full = dcp.get_model_state_dict(
            model, options = StateDictOptions(full_state_dict = True, cpu_offload = True)
        )
        out["full_sharded_keys"] = sorted(full)
    except AttributeError as error:
        out["full_sharded_keys"] = f"AttributeError: {error}"
        full = None
    if rank == 0 and full is not None:
        # The full checkpoint rebuilds the 4-bit weight on its own (what transformers does with a
        # pre-quantized checkpoint) and dequantizes to the same values. The packed bytes keep the
        # sharded storage dtype; they are the same bytes as uint8.
        try:
            stats = {k[len("0.weight.") :]: v for k, v in full.items() if k.startswith("0.weight.")}
            packed = full["0.weight"].reshape(-1).view(torch.uint8).reshape(-1, 1)
            fresh = bnb.nn.Params4bit.from_prequantized(
                data = packed, quantized_stats = stats, requires_grad = False, device = "cpu"
            )
            out["full_reload"] = torch.equal(
                bnb.functional.dequantize_4bit(fresh.data, fresh.quant_state), reference_dequant
            )
        except Exception as error:
            out["full_reload"] = f"{type(error).__name__}: {error}"
    if full is not None:
        # ... and loads back into the sharded model (the entries it added are dropped again).
        try:
            dcp.set_model_state_dict(
                model,
                full if rank == 0 else {},
                options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True),
            )
            out["full_set"] = "ok"
        except Exception as error:
            out["full_set"] = f"{type(error).__name__}: {error}"
    sd = dcp.get_model_state_dict(model, options = options)
    out["after_keys"] = sorted(sd)
    expected = {k: v.clone() for k, v in sd.items()}
    with torch.no_grad():
        model[1].weight.zero_()
    dcp.set_model_state_dict(
        model,
        sd if rank == 0 else {},
        options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True, strict = False),
    )
    restored = model[1].weight.full_tensor()
    out["roundtrip"] = rank != 0 or torch.equal(restored, expected["1.weight"])
    # A full DCP state dict of an unsharded 4-bit model keeps what dequantizes it.
    plain = torch.nn.Sequential(
        bnb.nn.Linear4bit(128, 64, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4")
    )
    plain[0].weight = bnb.nn.Params4bit(
        torch.randn(64, 128) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        blocksize = 64,
        module = plain[0],
    )
    plain = plain.to("cpu")
    out["full_keys"] = sorted(
        dcp.get_model_state_dict(plain, options = StateDictOptions(cpu_offload = True))
    )
    # accelerate's adapter-only path drops it (and no longer fails).
    import accelerate.utils.fsdp_utils as fsdp_utils

    from peft import LoraConfig, get_peft_model

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = bnb.nn.Linear4bit(
                128,
                64,
                bias = False,
                compute_dtype = torch.bfloat16,
                quant_type = "nf4",
                quant_storage = torch.bfloat16,
            )
            self.q.weight = bnb.nn.Params4bit(
                torch.randn(64, 128) * 0.05,
                requires_grad = False,
                quant_type = "nf4",
                quant_storage = torch.bfloat16,
                blocksize = 64,
                module = self.q,
            )

        def forward(self, x):
            return self.q(x)

    peft_model = get_peft_model(_Net().to("cpu"), LoraConfig(r = 4, target_modules = ["q"]))
    q.prepare_fsdp2_qlora(peft_model)
    fully_shard(peft_model, mesh = init_device_mesh("cpu", (world_size,)))
    sd = fsdp_utils._get_model_state_dict(
        peft_model,
        adapter_only = True,
        sd_options = StateDictOptions(full_state_dict = True, cpu_offload = True),
    )
    out["adapter_keys"] = sorted(k.split("q.", 1)[-1] for k in sd)
    return out


def _nf4_model(nested = False):
    bnb = pytest.importorskip("bitsandbytes")

    def lin(seed):
        torch.manual_seed(seed)
        layer = bnb.nn.Linear4bit(
            128,
            96,
            bias = False,
            compute_dtype = torch.bfloat16,
            compress_statistics = nested,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
        )
        layer.weight = bnb.nn.Params4bit(
            torch.randn(96, 128) * 0.05,
            requires_grad = False,
            compress_statistics = nested,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = 64,
            module = layer,
        )
        return layer.to("cpu")

    return torch.nn.Sequential(lin(0), lin(1))


def _case_gather_4bit_offload(rank, world_size, m):
    """A save that only writes the packed 4-bit bytes (`offload_4bit`) gets them and a copy of their
    QuantState on the CPU; the merge path (default) keeps them where they are. On CPU ranks `.cpu()`
    is a no-op, so the calls are recorded."""
    q = _load("fsdp2_qlora")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    model = _nf4_model()
    reference = {k: v.clone() for k, v in model.state_dict().items()}
    q.prepare_fsdp2_qlora(model)
    mesh = init_device_mesh("cpu", (world_size,))
    for layer in model:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    live = [layer.weight.quant_state for layer in model]
    live_absmax = [s.absmax for s in live]
    original_cpu = torch.Tensor.cpu
    moved = []

    def recording_cpu(self, *args, **kwargs):
        moved.append((tuple(self.shape), self.dtype))
        return original_cpu(self, *args, **kwargs)

    out = {}
    for offload_4bit in (False, True):
        moved.clear()
        torch.Tensor.cpu = recording_cpu
        try:
            kwargs = {"offload_4bit": True} if offload_4bit else {}
            with m.fsdp2_gathered(model, **kwargs) as writer:
                torch.Tensor.cpu = original_cpu
                if writer:
                    states = [layer.weight.quant_state for layer in model]
                    sd = model.state_dict()
                    out[offload_4bit] = dict(
                        # The packed bytes are the only (n, 1) tensors gathered here.
                        packed_moved = sum(len(shape) == 2 and shape[-1] == 1 for shape, _ in moved),
                        own_state = [s is not l for s, l in zip(states, live)],
                        state_cpu = [s.absmax.device.type == "cpu" for s in states],
                        same_keys = sorted(sd) == sorted(reference),
                        same_bytes = all(
                            torch.equal(
                                sd[k].reshape(-1).view(torch.uint8),
                                reference[k].reshape(-1).view(torch.uint8),
                            )
                            if isinstance(sd[k], torch.Tensor)
                            else sd[k] == reference[k]
                            for k in reference
                        ),
                    )
        finally:
            torch.Tensor.cpu = original_cpu
    # The live QuantState is untouched (same object, same tensors).
    out["live_kept"] = all(
        layer.weight.quant_state is s and s.absmax is a
        for layer, s, a in zip(model, live, live_absmax)
    )
    return out


def _case_dcp_quant_state(rank, world_size, m):
    """A full DCP load restores the QuantState it saved, not just the packed bytes, and refuses one
    that does not fit the model."""
    q = _load("fsdp2_qlora")
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    import torch.distributed.checkpoint.state_dict as dcp

    out = {}
    for nested in (False, True):
        model = _nf4_model(nested)
        q.prepare_fsdp2_qlora(model)
        mesh = init_device_mesh("cpu", (world_size,))
        for layer in model:
            fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)
        m.install_dcp_bnb_state_dict()
        save_opts = StateDictOptions(full_state_dict = True)  # every rank gets the full dict
        saved = {k: v.clone() for k, v in dcp.get_model_state_dict(model, options = save_opts).items()}
        states = [layer.weight.quant_state for layer in model]
        before = [s.absmax.clone() for s in states]
        results = {}
        for broadcast in (False, True):
            for s in states:
                s.absmax.mul_(2)  # a requantized / differently initialized base
            options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = broadcast)
            payload = saved if (rank == 0 or not broadcast) else {}
            dcp.set_model_state_dict(model, payload, options = options)
            results[broadcast] = all(torch.equal(s.absmax, b) for s, b in zip(states, before))
        out[f"restored_nested_{nested}"] = results
        bad = dict(saved)
        key = next(k for k in bad if k.endswith("weight.absmax"))
        bad[key] = bad[key][:-1].clone()
        try:
            dcp.set_model_state_dict(
                model, bad, options = StateDictOptions(full_state_dict = True)
            )
            out[f"mismatch_nested_{nested}"] = "loaded"
        except ValueError as error:
            out[f"mismatch_nested_{nested}"] = "ValueError" if "4-bit" in str(error) else str(error)
    return out


def _case_fp16_reduce(rank, world_size, m):
    """An fp16 trainable parameter FSDP2 does not manage: its gradient mean must not overflow."""

    class _Half(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16))

    model = _Half()
    m.reduce_unsharded_trainable_grads(model)
    (model.bias * 40000).sum().backward()
    return {"grad": model.bias.grad.float().tolist(), "dtype": str(model.bias.grad.dtype)}


def _case_slow_writer(rank, world_size, m):
    """rank 0's write (a merged or quantized export, a Hub upload) outlasts the collective timeout."""
    import time

    import torch.distributed as dist

    model = _shard(_Model(), world_size)
    with m.fsdp2_gathered(model) as writer:
        if writer:
            time.sleep(10)
    dist.barrier()
    return {"ok": True}


def _worker(rank, world_size, port, case, queue):
    try:
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(world_size),
        )
        import torch.distributed as dist

        import datetime

        timeout = datetime.timedelta(seconds = 4) if case == "slow_writer" else None
        dist.init_process_group(
            "gloo", rank = rank, world_size = world_size, **({"timeout": timeout} if timeout else {})
        )
        m = _load("fsdp2_save")
        if case == "slow_writer":
            result = _case_slow_writer(rank, world_size, m)
        elif case == "gather":
            result = _case_gather(rank, world_size, m)
        elif case == "fp16_reduce":
            result = _case_fp16_reduce(rank, world_size, m)
        elif case == "merge_offload":
            result = _case_merge_offload(rank, world_size, m)
        elif case == "gather_4bit":
            result = _case_gather_4bit(rank, world_size, m)
        elif case == "gather_4bit_offload":
            result = _case_gather_4bit_offload(rank, world_size, m)
        elif case == "dcp_quant_state":
            result = _case_dcp_quant_state(rank, world_size, m)
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
    procs = [
        ctx.Process(target = _worker, args = (r, world_size, port, case, queue))
        for r in range(world_size)
    ]
    # CPU only: a spawned child inherits this environment before it imports torch.
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        for p in procs:
            p.start()
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
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
    assert r0["included"] == [("plain", False), ("plain", True)], r0
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


@pytest.mark.skipif(
    not hasattr(torch.distributed.fsdp, "fully_shard")
    or "ignored_params"
    not in __import__("inspect").signature(torch.distributed.fsdp.fully_shard).parameters,
    reason = "fully_shard has no ignored_params",
)
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
    # A full checkpoint of the sharded 4-bit model keeps what dequantizes it, reloads, and loads back.
    assert "0.weight" in r["full_sharded_keys"], r["full_sharded_keys"]
    assert "0.weight.absmax" in r["full_sharded_keys"], r["full_sharded_keys"]
    assert "0.weight.quant_state.bitsandbytes__nf4" in r["full_sharded_keys"], r[
        "full_sharded_keys"
    ]
    assert r["full_reload"] is True, r["full_reload"]
    assert all(x["full_set"] == "ok" for x in results.values()), results
    assert all(x["roundtrip"] for x in results.values()), results
    assert "0.weight.absmax" in r["full_keys"], r["full_keys"]
    assert r["adapter_keys"] == ["lora_A.default.weight", "lora_B.default.weight"], r[
        "adapter_keys"
    ]


def test_a_slow_writer_does_not_time_out_the_other_ranks():
    results = _run("slow_writer")
    assert all(r["ok"] for r in results.values()), results


def test_unsharded_fp16_grads_are_averaged_without_overflow():
    # Each rank's fp16 gradient is 40000; their fp16 sum (80000) overflows, their mean does not.
    results = _run("fp16_reduce")
    for r in results.values():
        assert r["grad"] == [40000.0] * 4, r
        assert r["dtype"] == "torch.float16", r


def test_merge_offload_keeps_4bit_adapters_after_an_eval_forward():
    """The merged_4bit offload filter is built by name: after an eval forward the last unit's
    parameters are unsharded objects the gather replaces with DTensors before asking the filter, so an
    identity based filter moved that layer's adapters to the CPU while its 4-bit base stayed put."""
    r0 = _run("merge_offload")[0]
    assert r0["unsharded_after_eval"] > 0, r0
    decisions = r0["decisions"]
    expected = {}
    for i in range(N_LAYERS):
        for part in ("lora_A", "lora_B"):
            expected[f"layers.{i}.{part}.default.weight"] = i == 1  # layer 1 is not 4-bit
        expected[f"layers.{i}.base_layer.weight"] = True
    assert decisions == expected, decisions


def test_packed_4bit_save_stages_bytes_and_quant_state_on_the_cpu():
    """A full save of a sharded NF4 model kept every gathered packed weight on rank 0's accelerator,
    so the whole quantized base had to fit on one GPU. Saves that only write the bytes now get them
    on the CPU, with a CPU copy of the QuantState; the merge path is unchanged and the written state
    dict is byte for byte the same."""
    pytest.importorskip("bitsandbytes")
    r0 = _run("gather_4bit_offload")[0]
    kept, staged = r0[False], r0[True]
    assert kept["packed_moved"] == 0 and kept["own_state"] == [False, False], kept
    assert staged["packed_moved"] == 2, staged
    assert staged["own_state"] == [True, True] and all(staged["state_cpu"]), staged
    for r in (kept, staged):
        assert r["same_keys"] and r["same_bytes"], r
    assert r0["live_kept"], r0


def test_full_dcp_load_restores_the_saved_quant_state():
    """A full DCP load dropped the saved QuantState entries, so the packed bytes were silently paired
    with whatever scales the live model had."""
    pytest.importorskip("bitsandbytes")
    results = _run("dcp_quant_state")
    for r in results.values():
        for nested in (False, True):
            assert r[f"restored_nested_{nested}"] == {False: True, True: True}, r
            assert r[f"mismatch_nested_{nested}"] == "ValueError", r
