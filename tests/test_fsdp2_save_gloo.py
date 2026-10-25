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
from real_accelerator import has_real_cuda

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


class _QLoraBlock(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        bnb = pytest.importorskip("bitsandbytes")
        torch.manual_seed(seed)
        self.q = bnb.nn.Linear4bit(
            128,
            128,
            bias = False,
            compute_dtype = torch.bfloat16,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
        )
        self.q.weight = bnb.nn.Params4bit(
            torch.randn(128, 128) * 0.05,
            requires_grad = False,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = 64,
            module = self.q,
        )
        self.o = torch.nn.Linear(128, 128, bias = False).requires_grad_(False)

    def forward(self, x):
        return x + self.o(self.q(x))


class _QLoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_QLoraBlock(7 * i + 1) for i in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _qlora_peft(seed = 5):
    from peft import LoraConfig, get_peft_model

    model = _QLoraModel().to("cpu")  # quantizes the Params4bit
    model.is_loaded_in_4bit = True  # what PEFT reads to give Linear4bit its 4-bit LoRA layer
    torch.manual_seed(seed)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q", "o"], init_lora_weights = False)
    return get_peft_model(model, config)


def _packed_bytes(sd):
    return {
        k: v.detach().reshape(-1).view(torch.uint8).clone() if isinstance(v, torch.Tensor) else v
        for k, v in sd.items()
    }


def _case_merged_4bit_streamed(rank, world_size, m):
    """merged_4bit of a sharded NF4 QLoRA, as save_pretrained_merged runs it: the gather with the
    merged save's options, the streamed layer by layer merge, then PEFT's merge_and_unload and the
    state dict the export writes. `head` is #14739's placement (every packed weight stays on its
    device, PEFT merges them all there); `streamed` stages them on the CPU and brings one back at a
    time. Both must write the bytes an unsharded merge writes."""
    import warnings

    q = _load("fsdp2_qlora")
    from peft.tuners.lora import bnb as peft_bnb
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    warnings.filterwarnings("ignore")
    reference = _packed_bytes(_qlora_peft().base_model.merge_and_unload().state_dict())
    out = {}
    original_cpu = torch.Tensor.cpu
    original_merge = peft_bnb.Linear4bit.merge
    original_restage = getattr(m, "_restage_4bit", None)
    for mode in ("head", "streamed"):
        model = _qlora_peft()
        q.prepare_fsdp2_qlora(model)
        mesh = init_device_mesh("cpu", (world_size,))
        for layer in model.base_model.model.layers:
            fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)
        if mode == "head":
            options = dict(offload_to_cpu = m.offload_all_but_4bit_merge_operands(model))
        else:
            options = m.merged_save_gather_options(model, "merged_4bit")
        events = []
        moved = []

        def recording_cpu(self, *args, **kwargs):
            moved.append(tuple(self.shape))
            return original_cpu(self, *args, **kwargs)

        def recording_merge(self, *args, **kwargs):
            events.append("merge")
            return original_merge(self, *args, **kwargs)

        def recording_restage(weight, device):
            events.append("restage")
            return original_restage(weight, device)

        torch.Tensor.cpu = recording_cpu
        peft_bnb.Linear4bit.merge = recording_merge
        if original_restage is not None:
            m._restage_4bit = recording_restage
        try:
            with m.fsdp2_gathered(model, **options) as writer:
                torch.Tensor.cpu = original_cpu
                packed_moved = sum(len(s) == 2 and s[-1] == 1 for s in moved)
                if writer:
                    staged = m.merge_4bit_lora_layer_by_layer(model) if mode == "streamed" else []
                    on_cpu_before_unload = [
                        layer.q.base_layer.weight.quant_state.absmax.device.type
                        for layer in model.base_model.model.layers
                    ]
                    events.append("unload")
                    merged = model.base_model.merge_and_unload()
                    sd = _packed_bytes(merged.state_dict())
                    out[mode] = dict(
                        packed_moved = packed_moved,
                        streamed_layers = len(staged),
                        events = events,
                        states_cpu = on_cpu_before_unload,
                        same_keys = sorted(sd) == sorted(reference),
                        same_bytes = all(
                            torch.equal(sd[k], reference[k])
                            if isinstance(sd[k], torch.Tensor)
                            else sd[k] == reference[k]
                            for k in reference
                        ),
                    )
        finally:
            torch.Tensor.cpu = original_cpu
            peft_bnb.Linear4bit.merge = original_merge
            if original_restage is not None:
                m._restage_4bit = original_restage
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


def _case_dcp_restore(rank, world_size, m):
    """A full DCP checkpoint loaded into a 4-bit model whose QuantState differs from the one it was
    saved from: the packed bytes come from the checkpoint, so their scales must too."""
    bnb = pytest.importorskip("bitsandbytes")
    q = _load("fsdp2_qlora")
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    import torch.distributed.checkpoint.state_dict as dcp

    def model(
        scale,
        nested,
        blocksize = 64,
    ):
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
            torch.randn(64, 128) * scale,
            requires_grad = False,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = blocksize,
            compress_statistics = nested,
            module = layer,
        )
        net = torch.nn.Sequential(layer.to("cpu"), torch.nn.Linear(64, 8))
        reference = bnb.functional.dequantize_4bit(net[0].weight.data, net[0].weight.quant_state)
        q.prepare_fsdp2_qlora(net)
        fully_shard(net, mesh = init_device_mesh("cpu", (world_size,)))
        return net, reference

    def scales(quant_state):
        tensors = [quant_state.absmax, quant_state.code]
        if quant_state.state2 is not None:
            tensors += [quant_state.state2.absmax, quant_state.state2.code, quant_state.offset]
        return [t.detach().cpu().clone() for t in tensors]

    def restored(net, reference):
        out = {}
        with m.fsdp2_gathered(net) as writer:
            if writer:
                weight = net[0].weight
                actual = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
                out["dequant_equal"] = torch.equal(actual, reference)
        return out

    m.install_dcp_bnb_state_dict()
    out = {}
    full = StateDictOptions(full_state_dict = True, cpu_offload = True)
    for nested in (False, True):
        src, reference = model(0.1, nested)
        expected = scales(src[0].weight.quant_state)
        sd = dcp.get_model_state_dict(src, options = full)
        dst, _ = model(0.05, nested)
        dcp.set_model_state_dict(
            dst,
            sd if rank == 0 else {},
            options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True),
        )
        live = scales(dst[0].weight.quant_state)
        out[f"broadcast_{nested}"] = dict(
            restored(dst, reference),
            scales = all(torch.equal(a, b) for a, b in zip(live, expected)),
        )
        sharded = dcp.get_model_state_dict(src)
        dst, _ = model(0.05, nested)
        dcp.set_model_state_dict(dst, sharded)
        live = scales(dst[0].weight.quant_state)
        out[f"sharded_{nested}"] = dict(
            restored(dst, reference),
            scales = all(torch.equal(a, b) for a, b in zip(live, expected)),
        )
    src, _ = model(0.1, False, blocksize = 128)
    sd = dcp.get_model_state_dict(src, options = full)
    dst, _ = model(0.05, False)
    try:
        dcp.set_model_state_dict(
            dst,
            sd if rank == 0 else {},
            options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True, strict = False),
        )
        out["mismatch"] = "loaded"
    except ValueError as error:
        out["mismatch"] = "ValueError" if "4-bit quantization state" in str(error) else str(error)
    except Exception as error:
        out["mismatch"] = f"{type(error).__name__}: {error}"
    return out


def _case_dcp_restore_guard(rank, world_size, m):
    """Checkpoint quantization metadata that cannot be applied: entries without their packed weight
    (a partial state dict under strict=False) and a nested checkpoint loaded into a non-nested model."""
    bnb = pytest.importorskip("bitsandbytes")
    q = _load("fsdp2_qlora")
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    import torch.distributed.checkpoint.state_dict as dcp

    def model(scale, nested):
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
            torch.randn(64, 128) * scale,
            requires_grad = False,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = 64,
            compress_statistics = nested,
            module = layer,
        )
        net = torch.nn.Sequential(layer.to("cpu"), torch.nn.Linear(64, 8))
        reference = bnb.functional.dequantize_4bit(net[0].weight.data, net[0].weight.quant_state)
        q.prepare_fsdp2_qlora(net)
        fully_shard(net, mesh = init_device_mesh("cpu", (world_size,)))
        return net, reference

    def scales(quant_state):
        tensors = [quant_state.absmax, quant_state.code]
        if quant_state.state2 is not None:
            tensors += [quant_state.state2.absmax, quant_state.state2.code, quant_state.offset]
        return [t.detach().cpu().clone() for t in tensors]

    def dequant_equal(net, reference):
        out = None
        with m.fsdp2_gathered(net) as writer:
            if writer:
                weight = net[0].weight
                actual = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
                out = torch.equal(actual, reference)
        return out

    m.install_dcp_bnb_state_dict()
    out = {}
    full = StateDictOptions(full_state_dict = True, cpu_offload = True)
    loose = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True, strict = False)
    src, _ = model(0.1, False)
    for mode in ("broadcast", "sharded"):
        if mode == "broadcast":
            sd = dcp.get_model_state_dict(src, options = full)
        else:
            sd = dcp.get_model_state_dict(src)
        sd = {k: v for k, v in sd.items() if k != "0.weight"}
        dst, reference = model(0.05, False)
        expected = scales(dst[0].weight.quant_state)
        if mode == "broadcast":
            dcp.set_model_state_dict(dst, sd if rank == 0 else {}, options = loose)
        else:
            dcp.set_model_state_dict(dst, sd, options = StateDictOptions(strict = False))
        live = scales(dst[0].weight.quant_state)
        out[f"partial_{mode}"] = dict(
            scales = all(torch.equal(a, b) for a, b in zip(live, expected)),
            dequant_equal = dequant_equal(dst, reference),
        )
    src, _ = model(0.1, True)
    for mode in ("broadcast", "sharded"):
        dst, _ = model(0.05, False)
        try:
            if mode == "broadcast":
                sd = dcp.get_model_state_dict(src, options = full)
                dcp.set_model_state_dict(dst, sd if rank == 0 else {}, options = loose)
            else:
                sd = dcp.get_model_state_dict(src)
                dcp.set_model_state_dict(dst, sd, options = StateDictOptions(strict = False))
            out[f"nested_{mode}"] = "loaded"
        except ValueError as error:
            out[f"nested_{mode}"] = str(error)
        except Exception as error:
            out[f"nested_{mode}"] = f"{type(error).__name__}: {error}"
    return out


def _nf4_model(nested = False, device = "cpu"):
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
        return layer.to(device)

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
        saved = {
            k: v.clone() for k, v in dcp.get_model_state_dict(model, options = save_opts).items()
        }
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
            dcp.set_model_state_dict(model, bad, options = StateDictOptions(full_state_dict = True))
            out[f"mismatch_nested_{nested}"] = "loaded"
        except ValueError as error:
            out[f"mismatch_nested_{nested}"] = "ValueError" if "4-bit" in str(error) else str(error)
    return out


def _sharded_nf4(
    world_size,
    m,
    device = "cpu",
):
    q = _load("fsdp2_qlora")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    model = _nf4_model(device = device)
    q.prepare_fsdp2_qlora(model)
    mesh = init_device_mesh(device.split(":")[0], (world_size,))
    for layer in model:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    m.install_dcp_bnb_state_dict()
    return model


def _legacy_and_mismatched_loads(rank, model, saved, out):
    """Layer 0 saved without its QuantState entries (a checkpoint from before they were saved): with
    the live packed bytes it loads, keeps the live QuantState and dequantizes to the live weights;
    with other packed bytes it raises. Both with and without broadcast_from_rank0."""
    from bitsandbytes.functional import QuantState, dequantize_4bit
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    import torch.distributed.checkpoint.state_dict as dcp

    def dequantized(state_dict):
        stats = {
            k[len("0.weight.") :]: v for k, v in state_dict.items() if k.startswith("0.weight.")
        }
        state = QuantState.from_dict(stats, device = "cpu")
        packed = state_dict["0.weight"].cpu().reshape(-1, 1)
        state.absmax, state.code = state.absmax.cpu(), state.code.cpu()
        return dequantize_4bit(packed, state).float()

    reference = dequantized({k: v.cpu() for k, v in saved.items()})
    live_absmax = model[0].weight.quant_state.absmax.clone()
    legacy = {k: v for k, v in saved.items() if not k.startswith("0.weight.")}
    mismatched = dict(legacy)
    mismatched["0.weight"] = saved["1.weight"].clone()  # same shape, other bytes
    full = StateDictOptions(full_state_dict = True)  # every rank gets the full dict
    for broadcast in (False, True):
        options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = broadcast)
        try:
            dcp.set_model_state_dict(
                model, legacy if (rank == 0 or not broadcast) else {}, options = options
            )
            after = dcp.get_model_state_dict(model, options = full)
            out[f"legacy_{broadcast}"] = torch.equal(
                model[0].weight.quant_state.absmax, live_absmax
            ) and torch.equal(dequantized(after), reference)
        except Exception as error:
            out[f"legacy_{broadcast}"] = f"{type(error).__name__}: {error}"
        try:
            dcp.set_model_state_dict(
                model, mismatched if (rank == 0 or not broadcast) else {}, options = options
            )
            out[f"mismatch_{broadcast}"] = "loaded"
        except ValueError as error:
            out[f"mismatch_{broadcast}"] = (
                "ValueError" if "`0.weight`" in str(error) else str(error)
            )


def _case_dcp_quant_state_missing(rank, world_size, m):
    """A full DCP state dict holding a packed 4-bit weight but none of its QuantState entries loads
    only when the bytes are the live ones (older checkpoints of a frozen base); other bytes would be
    paired with the live scales and must not load. One without the packed weight still loads.
    The broadcast_from_rank0 presence check must not send a CPU tensor through the process group
    (NCCL rejects those): with tensor broadcasts of CPU tensors refused, the restore still works."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    import torch.distributed as dist
    import torch.distributed.checkpoint.state_dict as dcp

    model = _sharded_nf4(world_size, m)
    full = StateDictOptions(full_state_dict = True)
    saved = {k: v.clone() for k, v in dcp.get_model_state_dict(model, options = full).items()}
    out = {}
    _legacy_and_mismatched_loads(rank, model, saved, out)
    stripped = {k: v for k, v in saved.items() if not k.startswith("0.weight.")}
    # No packed weight and no QuantState for layer 0: nothing of it is loaded, so nothing to refuse.
    no_weight = {k: v for k, v in stripped.items() if k != "0.weight"}
    try:
        dcp.set_model_state_dict(
            model, no_weight, options = StateDictOptions(full_state_dict = True, strict = False)
        )
        out["no_weight"] = "loaded"
    except Exception as error:
        out["no_weight"] = f"{type(error).__name__}: {error}"

    # NCCL-like group: a tensor collective on a CPU tensor fails. Only the restore runs under it.
    states = [layer.weight.quant_state for layer in model]
    before = [s.absmax.clone() for s in states]
    for s in states:
        s.absmax.mul_(2)
    original = dist.broadcast

    def nccl_like_broadcast(tensor, *args, **kwargs):
        if tensor.device.type == "cpu":
            raise RuntimeError("No backend type associated with device type cpu")
        return original(tensor, *args, **kwargs)

    dist.broadcast = nccl_like_broadcast
    try:
        options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True)
        m._restore_quant_state(model, saved if rank == 0 else {}, options)
        out["nccl_like_flag"] = all(torch.equal(s.absmax, b) for s, b in zip(states, before))
    except RuntimeError as error:
        out["nccl_like_flag"] = str(error)
    finally:
        dist.broadcast = original
    return out


def _case_nccl_quant_state(rank, world_size, m):
    """Real NCCL group on two GPUs: a full DCP load with broadcast_from_rank0 restores the saved
    QuantState on every rank and refuses a packed weight saved without it."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    import torch.distributed.checkpoint.state_dict as dcp

    model = _sharded_nf4(world_size, m, device = f"cuda:{rank}")
    full = StateDictOptions(full_state_dict = True)  # every rank gets the full dict
    saved = {k: v.clone() for k, v in dcp.get_model_state_dict(model, options = full).items()}
    states = [layer.weight.quant_state for layer in model]
    before = [s.absmax.clone() for s in states]
    for s in states:
        s.absmax.mul_(2)
    options = StateDictOptions(full_state_dict = True, broadcast_from_rank0 = True)
    out = {"backend": torch.distributed.get_backend()}
    try:
        dcp.set_model_state_dict(model, saved if rank == 0 else {}, options = options)
        out["restored"] = all(torch.equal(s.absmax, b) for s, b in zip(states, before))
    except Exception as error:
        out["restored"] = f"{type(error).__name__}: {error}"
    _legacy_and_mismatched_loads(rank, model, saved, out)
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
        nccl = case.startswith("nccl")
        if nccl:
            torch.cuda.set_device(rank)
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(world_size),
        )
        import torch.distributed as dist

        import datetime

        # dcp_restore_guard: a rank left waiting on a collective fails in 60 s instead of hanging.
        timeout = {"slow_writer": 4, "dcp_restore_guard": 60}.get(case)
        timeout = datetime.timedelta(seconds = timeout) if timeout else None
        dist.init_process_group(
            "nccl" if nccl else "gloo",
            rank = rank,
            world_size = world_size,
            **({"timeout": timeout} if timeout else {}),
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
        elif case == "dcp_restore":
            result = _case_dcp_restore(rank, world_size, m)
        elif case == "dcp_restore_guard":
            result = _case_dcp_restore_guard(rank, world_size, m)
        elif case == "merged_4bit_streamed":
            result = _case_merged_4bit_streamed(rank, world_size, m)
        elif case == "gather_4bit_offload":
            result = _case_gather_4bit_offload(rank, world_size, m)
        elif case == "dcp_quant_state":
            result = _case_dcp_quant_state(rank, world_size, m)
        elif case == "dcp_quant_state_missing":
            result = _case_dcp_quant_state_missing(rank, world_size, m)
        elif case == "nccl_quant_state":
            result = _case_nccl_quant_state(rank, world_size, m)
        elif case.startswith("replicated"):
            result = _case_replicated(rank, world_size, m, hooked = case.endswith("hooked"))
        else:
            result = _case_dcp_bnb(rank, world_size, m)
        queue.put((rank, result))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _run(
    case,
    world_size = 2,
    cpu_only = True,
):
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
    if cpu_only:
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
    try:
        for _ in procs:
            rank, result = queue.get(timeout = 240)
            results[rank] = result
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.terminate()
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


def test_full_dcp_load_restores_the_checkpoint_quant_state():
    """Full and sharded DCP loads take the checkpoint's QuantState with its packed bytes."""
    pytest.importorskip("bitsandbytes")
    results = _run("dcp_restore")
    for rank, r in results.items():
        for nested in (False, True):
            for mode in ("broadcast", "sharded"):
                got = r[f"{mode}_{nested}"]
                assert got["scales"], (rank, mode, nested, r)
                if rank == 0:
                    assert got["dequant_equal"], (mode, nested, r)
        assert r["mismatch"] == "ValueError", (rank, r)


def test_full_dcp_load_skips_or_rejects_quant_state_it_cannot_apply():
    """QuantState without its packed weight keeps the live scales (it used to apply the checkpoint's),
    and a nested checkpoint into a non-nested model raises on every rank."""
    pytest.importorskip("bitsandbytes")
    results = _run("dcp_restore_guard")
    for rank, r in results.items():
        for mode in ("broadcast", "sharded"):
            assert r[f"partial_{mode}"]["scales"], (rank, mode, r)
            if rank == 0:
                assert r[f"partial_{mode}"]["dequant_equal"], (mode, r)
            message = r[f"nested_{mode}"]
            assert "nested" in message and "does not match" in message, (rank, mode, r)
        assert r["nested_broadcast"] == results[0]["nested_broadcast"], results


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


def test_full_dcp_load_refuses_a_packed_4bit_weight_without_its_quant_state():
    """A packed 4-bit weight whose QuantState entries the checkpoint lacks was loaded silently and
    dequantized with the live model's scales, also when its bytes were not the live ones; an older
    checkpoint of the same frozen base (same bytes) must keep loading; the broadcast_from_rank0 presence check sent a CPU
    tensor through the process group, which an NCCL group cannot carry."""
    pytest.importorskip("bitsandbytes")
    for r in _run("dcp_quant_state_missing").values():
        for broadcast in (False, True):
            assert r[f"legacy_{broadcast}"] is True, r
            assert r[f"mismatch_{broadcast}"] == "ValueError", r
        assert r["no_weight"] == "loaded", r
        assert r["nccl_like_flag"] is True, r


@pytest.mark.skipif(
    not has_real_cuda()
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_nccl_available(),
    reason = "needs two CUDA GPUs and NCCL",
)
def test_full_dcp_broadcast_load_restores_quant_state_over_nccl():
    pytest.importorskip("bitsandbytes")
    for r in _run("nccl_quant_state", cpu_only = False).values():
        assert r["backend"] == "nccl", r
        assert r["restored"] is True, r
        for broadcast in (False, True):
            assert r[f"legacy_{broadcast}"] is True, r
            assert r[f"mismatch_{broadcast}"] == "ValueError", r


def test_merged_4bit_save_streams_the_packed_base_one_layer_at_a_time():
    """merged_4bit of a sharded NF4 QLoRA kept every gathered packed weight on rank 0's accelerator
    until PEFT merged them all, so the whole quantized base had to fit on one GPU next to its shards.
    The gather now stages each packed weight on the CPU right after gathering it, and the merge
    brings one layer back at a time (restage, merge, restage), writing the same bytes as an unsharded
    merge and as the previous placement."""
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    r0 = _run("merged_4bit_streamed")[0]
    head, streamed = r0["head"], r0["streamed"]
    for r in (head, streamed):
        assert r["same_keys"] and r["same_bytes"], r
    assert head["packed_moved"] == 0, head
    # Every packed weight left the gathering device as soon as it was gathered.
    assert streamed["packed_moved"] == N_LAYERS, streamed
    assert streamed["streamed_layers"] == N_LAYERS, streamed
    assert all(device == "cpu" for device in streamed["states_cpu"]), streamed
    # One layer on its device at a time: each merge is bracketed by its own two restages, and PEFT's
    # unload merges nothing more.
    assert (
        streamed["events"]
        == ["restage", "merge", "restage"] * N_LAYERS + ["unload"] + ["merge"] * N_LAYERS
    ), streamed["events"]
