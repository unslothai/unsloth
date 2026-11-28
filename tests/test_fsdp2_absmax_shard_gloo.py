# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 QLoRA shards each bitsandbytes 4-bit weight's absmax with its packed bytes
(unsloth/models/fsdp2_qlora.py `ShardedAbsmax`), 2 and 4 gloo ranks on CPU, with the real
bitsandbytes and with Unsloth's stand-in (unsloth/bnb_shim).

- every rank keeps 1/world of the absmax of weights that split into self-contained shards (whole
  nested-scale groups), the whole one otherwise; the replicated original is freed after the wrap;
- dequantized weights, losses and LoRA grads are bitwise those of the replicated-absmax path, and
  match a single process;
- a gathered save writes bitsandbytes' exact state dict (no extra key), which loads back through
  `Params4bit.from_prequantized` of either bitsandbytes and dequantizes bitwise the same;
- a full DCP state dict carries the absmax, and loading it restores scales a run rewrote.
The modules are loaded from their files (importing unsloth needs a GPU).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SHIM_INIT = _ROOT / "unsloth" / "bnb_shim" / "__init__.py"

fsdp_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
if (
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available()
):
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)

# (in, out, nested): 256 x 256 splits into whole 64 x 256 element groups on 2 and 4 ranks; the last
# (24576 elements, nested) does not, so its absmax stays whole on every rank.
_LAYERS = ((256, 256, True), (256, 256, False), (256, 96, True))
_STEPS = 3


def _load(name):
    path = _ROOT / "unsloth" / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_unsloth_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bnb(shim):
    if shim:
        spec = importlib.util.spec_from_file_location("_unsloth_bnb_shim", _SHIM_INIT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.install() is True
    import bitsandbytes

    assert bool(getattr(bitsandbytes, "__unsloth_shim__", False)) == shim
    return bitsandbytes


class _Lora4bit(torch.nn.Module):
    def __init__(self, bnb, seed, in_f, out_f, nested):
        super().__init__()
        torch.manual_seed(seed)
        base = bnb.nn.Linear4bit(
            in_f,
            out_f,
            bias = False,
            compute_dtype = torch.bfloat16,
            compress_statistics = nested,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
        )
        base.weight = bnb.nn.Params4bit(
            torch.randn(out_f, in_f) * 0.05,
            requires_grad = False,
            compress_statistics = nested,
            quant_type = "nf4",
            quant_storage = torch.bfloat16,
            blocksize = 64,
            module = base,
        )
        self.base = base.to("cpu")
        self.lora_A = torch.nn.Linear(in_f, 8, bias = False)
        self.lora_B = torch.nn.Linear(8, out_f, bias = False)
        torch.nn.init.normal_(self.lora_B.weight, std = 0.05)

    def forward(self, x):
        out = self.base(x.to(torch.bfloat16)).to(x.dtype)
        return torch.tanh(out + self.lora_B(self.lora_A(x)))


def _model(bnb):
    return torch.nn.Sequential(*[_Lora4bit(bnb, 10 + i, *shape) for i, shape in enumerate(_LAYERS)])


def _inputs(rank):
    return torch.randn(4, _LAYERS[0][0], generator = torch.Generator().manual_seed(100 + rank))


def _full(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _shard(model, world_size, q, shard_absmax, mixed):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cpu", (world_size,))
    kwargs = {"mesh": mesh}
    if mixed:
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype = torch.bfloat16, reduce_dtype = torch.float32
        )
    with q.fsdp2_qlora_wrap(model, native = True, shard_absmax = shard_absmax) as info:
        for layer in model:
            fully_shard(layer, **kwargs)
        fully_shard(model, **kwargs)
    return info


def _absmax_bytes(model):
    """Bytes of absmax this rank holds per 4-bit weight (sharded holder or the QuantState's own)."""
    out = []
    for layer in model:
        holder = layer.base._parameters.get("weight_unsloth_absmax")
        if holder is not None:
            local = holder.to_local() if hasattr(holder, "to_local") else holder
            out.append(local.numel() * local.element_size())
        else:
            absmax = layer.base.weight.quant_state.absmax
            out.append(absmax.untyped_storage().nbytes())
    return out


def _train(
    model,
    bnb,
    rank,
    steps = _STEPS,
):
    dequant = []

    def capture(module, args):
        if len(dequant) < len(_LAYERS):
            w = module.weight
            dequant.append(bnb.functional.dequantize_4bit(w.data, w.quant_state).clone())

    hooks = [layer.base.register_forward_pre_hook(capture) for layer in model]
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr = 0.5)
    losses, grads = [], []
    x = _inputs(rank)
    for _ in range(steps):
        loss = model(x).square().mean()
        loss.backward()
        losses.append(loss.detach().clone())
        grads.append([_full(p.grad).detach().clone() for p in trainable])
        opt.step()
        opt.zero_grad(set_to_none = True)
    for h in hooks:
        h.remove()
    return losses, grads, dequant


def _single_process(bnb, world_size):
    """No FSDP: per-rank step-0 losses, dequantized weights and the rank-averaged LoRA grads."""
    model = _model(bnb)
    trainable = [p for p in model.parameters() if p.requires_grad]
    losses, local = [], []
    for r in range(world_size):
        loss = model(_inputs(r)).square().mean()
        loss.backward()
        losses.append(loss.detach().clone())
        local.append([p.grad.detach().clone() for p in trainable])
        model.zero_grad(set_to_none = True)
    mean = [sum(g[i] for g in local) / world_size for i in range(len(trainable))]
    dequant = [
        bnb.functional.dequantize_4bit(l.base.weight.data, l.base.weight.quant_state) for l in model
    ]
    return losses, mean, dequant, model


def _case(rank, world_size, shim):
    import torch.distributed as dist

    bnb = _bnb(shim)
    q = _load("fsdp2_qlora")
    s = _load("fsdp2_save")
    out = {}

    # Replicated absmax (today's path) vs sharded, same mixed precision, same seeds.
    base = _model(bnb)
    base_info = _shard(base, world_size, q, shard_absmax = False, mixed = True)
    head = _model(bnb)
    from torch.multiprocessing.reductions import StorageWeakRef

    # The replicated scales' memory, not just the tensor object.
    originals = [StorageWeakRef(l.base.weight.quant_state.absmax.untyped_storage()) for l in head]
    scales = [l.base.weight.quant_state.absmax.clone() for l in head]
    whole = _absmax_bytes(head)
    head_info = _shard(head, world_size, q, shard_absmax = True, mixed = True)
    out["absmax_sharded"] = (base_info["absmax_sharded"], head_info["absmax_sharded"])
    # torch < 2.8 (one dtype per unit, frozen weights included): no float storage, absmax replicated.
    out["float_storage"] = q.float_storage_survives_wrap(native = True)
    out["bytes_whole"] = whole
    out["bytes_base"] = _absmax_bytes(base)
    out["bytes_head"] = _absmax_bytes(head)
    out["originals_freed"] = [r.expired() for r in originals]
    b_losses, b_grads, b_deq = _train(base, bnb, rank)
    h_losses, h_grads, h_deq = _train(head, bnb, rank)
    out["losses_bitwise"] = all(torch.equal(a, b) for a, b in zip(b_losses, h_losses))
    out["grads_bitwise"] = all(
        torch.equal(a, b) for ga, gb in zip(b_grads, h_grads) for a, b in zip(ga, gb)
    )
    out["dequant_bitwise"] = [torch.equal(a, b) for a, b in zip(b_deq, h_deq)]
    # Between steps: the gathered scales are freed; the sharded weight's QuantState holds this
    # rank's slice (the absmax of its own packed bytes), the replicated one the whole absmax.
    between = []
    for layer, whole_scales in zip(head, scales):
        entry = (layer.base.__dict__.get("_unsloth_fsdp2_absmax") or {}).get(
            "weight_unsloth_absmax"
        )
        live = layer.base.weight.quant_state.absmax
        if entry is None:
            between.append(torch.equal(live, whole_scales))
            continue
        n = whole_scales.numel() // world_size
        between.append(
            entry.quant_state.absmax.untyped_storage().nbytes() == 0
            and layer.base.weight.quant_state is not entry.quant_state
            and torch.equal(live, whole_scales[rank * n : (rank + 1) * n])
        )
    out["absmax_between_steps"] = between
    out["steps_changed_loss"] = not torch.equal(h_losses[0], h_losses[-1])

    # Sharded absmax without mixed precision against one process holding the whole model.
    plain = _model(bnb)
    _shard(plain, world_size, q, shard_absmax = True, mixed = False)
    p_losses, p_grads, p_deq = _train(plain, bnb, rank, steps = 1)
    ref_losses, ref_grads, ref_deq, reference = _single_process(bnb, world_size)
    out["single_loss_bitwise"] = torch.equal(p_losses[0], ref_losses[rank])
    out["single_dequant_bitwise"] = [torch.equal(a, b) for a, b in zip(p_deq, ref_deq)]
    out["single_grads_close"] = all(
        torch.allclose(a, b, rtol = 1e-5, atol = 1e-7) for a, b in zip(p_grads[0], ref_grads)
    )

    # A gathered save: bitsandbytes' own state dict, byte for byte, and nothing else.
    fresh = _model(bnb)
    _shard(fresh, world_size, q, shard_absmax = True, mixed = True)
    expected = {k: v.detach().clone() for k, v in _model(bnb).state_dict().items()}
    with s.fsdp2_gathered(fresh) as writer:
        if writer:
            saved = {k: v.detach().clone() for k, v in fresh.state_dict().items()}
            out["save_keys_equal"] = sorted(saved) == sorted(expected)
            out["save_bytes_equal"] = all(
                torch.equal(_bytes(saved[k]), _bytes(expected[k])) for k in expected
            )
            reloaded = []
            for i, layer in enumerate(_model(bnb)):
                head_key = f"{i}.base.weight"
                stats = {
                    k[len(head_key) + 1 :]: v
                    for k, v in saved.items()
                    if k.startswith(head_key + ".")
                }
                w = bnb.nn.Params4bit.from_prequantized(
                    data = saved[head_key], quantized_stats = stats, device = "cpu"
                )
                reloaded.append(
                    torch.equal(bnb.functional.dequantize_4bit(w.data, w.quant_state), ref_deq[i])
                )
            out["reload_dequant_bitwise"] = reloaded
    out["save_left_shards"] = all(hasattr(l.base._parameters["weight"], "to_local") for l in fresh)
    # Still trains after the save put the shards back.
    f_losses, _, _ = _train(fresh, bnb, rank, steps = 1)
    out["after_save_loss_bitwise"] = torch.equal(f_losses[0], h_losses[0])

    # Full DCP state dict: carries the absmax; loading restores scales a run rewrote.
    import torch.distributed.checkpoint.state_dict as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    s.install_dcp_bnb_state_dict()
    opts = StateDictOptions(full_state_dict = True)
    dcp_model = _model(bnb)
    _shard(dcp_model, world_size, q, shard_absmax = True, mixed = True)
    full = {
        k: v.detach().clone() for k, v in dcp.get_model_state_dict(dcp_model, options = opts).items()
    }
    out["dcp_keys_equal"] = sorted(full) == sorted(expected)
    out["dcp_absmax_equal"] = all(
        torch.equal(_bytes(full[k]), _bytes(expected[k])) for k in expected if k.endswith("absmax")
    )
    x = _inputs(rank)
    with torch.no_grad():
        before = dcp_model(x).clone()
        with q.full_absmax(dcp_model):
            for layer in dcp_model:
                layer.base.weight.quant_state.absmax.mul_(2)
        changed = dcp_model(x).clone()
    dcp.set_model_state_dict(dcp_model, full, options = opts)
    with torch.no_grad():
        after = dcp_model(x).clone()
    # Optimizer-only DCP calls never gather the scales.
    gathers = []
    gather = q.ShardedAbsmax.gather
    q.ShardedAbsmax.gather = lambda self: gathers.append(1) or gather(self)
    try:
        sgd = torch.optim.SGD([p for p in dcp_model.parameters() if p.requires_grad], lr = 0.1)
        dcp.get_optimizer_state_dict(dcp_model, sgd)
        out["optimizer_dcp_gathers"] = len(gathers)
        dcp.get_model_state_dict(dcp_model, options = opts)
        out["model_dcp_gathers"] = len(gathers)
    finally:
        q.ShardedAbsmax.gather = gather
    out["dcp_rewrite_changed"] = not torch.equal(before, changed)
    out["dcp_restore_bitwise"] = torch.equal(before, after)
    dist.barrier()
    return out


def _reload_case(save_dir, shim):
    """Loads the checkpoint the other bitsandbytes wrote (the bytes a gathered save writes, see
    `save_bytes_equal`) and dequantizes it the same; then writes its own."""
    bnb = _bnb(shim)
    flavor, other = ("shim", "bnb") if shim else ("bnb", "shim")
    path = os.path.join(save_dir, f"{other}.pt")
    out = None
    if os.path.exists(path):
        blob = torch.load(path, weights_only = False)
        saved, dequant = blob["state_dict"], blob["dequant"]
        out = []
        for i in range(len(_LAYERS)):
            head_key = f"{i}.base.weight"
            stats = {
                k[len(head_key) + 1 :]: v for k, v in saved.items() if k.startswith(head_key + ".")
            }
            w = bnb.nn.Params4bit.from_prequantized(
                data = saved[head_key], quantized_stats = stats, device = "cpu"
            )
            out.append(
                torch.equal(bnb.functional.dequantize_4bit(w.data, w.quant_state), dequant[i])
            )
    model = _model(bnb)
    torch.save(
        {
            "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "dequant": [
                bnb.functional.dequantize_4bit(l.base.weight.data, l.base.weight.quant_state)
                for l in model
            ],
        },
        os.path.join(save_dir, f"{flavor}.pt"),
    )
    return out


def _worker(rank, world_size, port, shim, queue):
    try:
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(world_size),
        )
        import torch.distributed as dist

        dist.init_process_group("gloo", rank = rank, world_size = world_size)
        result = _case(rank, world_size, shim)
        queue.put((rank, result))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _reload_worker(save_dir, shim, queue):
    try:
        queue.put((0, {"reload": _reload_case(save_dir, shim)}))
    except BaseException as error:
        import traceback
        queue.put((0, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _spawn(target, args_per_rank, shim):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target = target, args = (*args, queue)) for args in args_per_rank]
    saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "UNSLOTH_BNB_SHIM")}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["UNSLOTH_BNB_SHIM"] = "1" if shim else "0"
    try:
        for p in procs:
            p.start()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    results = {}
    try:
        for _ in procs:
            rank, result = queue.get(timeout = 300)
            results[rank] = result
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.terminate()
    for r in results.values():
        assert "error" not in r, r.get("error")
    return results


def _run(world_size, shim):
    from tests._rendezvous_port import free_port
    port = free_port()
    return _spawn(_worker, [(r, world_size, port, shim) for r in range(world_size)], shim)


def _check(results, world_size):
    for rank, r in results.items():
        whole = r["bytes_whole"]
        assert r["bytes_base"] == whole, r
        if r["float_storage"]:
            # Two weights split evenly into self-contained shards; the third stays whole.
            assert r["absmax_sharded"] == (0, 2), r
            assert r["bytes_head"] == [whole[0] // world_size, whole[1] // world_size, whole[2]], r
            assert r["originals_freed"] == [True, True, False], r
        else:
            assert r["absmax_sharded"] == (0, 0), r
            assert r["bytes_head"] == whole, r
        assert r["absmax_between_steps"] == [True, True, True], r
        assert r["steps_changed_loss"], r
        assert r["losses_bitwise"] and r["grads_bitwise"], r
        assert r["dequant_bitwise"] == [True] * len(_LAYERS), r
        assert r["single_loss_bitwise"], r
        assert r["single_dequant_bitwise"] == [True] * len(_LAYERS), r
        assert r["single_grads_close"], r
        assert r["save_left_shards"] and r["after_save_loss_bitwise"], r
        assert r["dcp_keys_equal"] and r["dcp_absmax_equal"], r
        assert r["dcp_rewrite_changed"] and r["dcp_restore_bitwise"], r
        assert r["optimizer_dcp_gathers"] == 0, r
        assert r["model_dcp_gathers"] == r["absmax_sharded"][1], r
    writer = results[0]
    assert writer["save_keys_equal"] and writer["save_bytes_equal"], writer
    assert writer["reload_dequant_bitwise"] == [True] * len(_LAYERS), writer


@pytest.mark.parametrize("world_size", [2, 4])
def test_absmax_sharded_with_bitsandbytes(world_size):
    pytest.importorskip("bitsandbytes")
    _check(_run(world_size, False), world_size)


def test_absmax_sharded_with_the_stand_in():
    _check(_run(2, True), 2)


def test_checkpoints_load_across_bitsandbytes_and_the_stand_in(tmp_path):
    pytest.importorskip("bitsandbytes")
    save_dir = str(tmp_path)
    results = [
        _spawn(_reload_worker, [(save_dir, shim)], shim)[0]["reload"]
        for shim in (False, True, False)
    ]
    # The first run has nothing to load; then each loads the other's checkpoint.
    assert results == [None, [True] * len(_LAYERS), [True] * len(_LAYERS)], results


def test_absmax_eligibility():
    """Shards only whole nested-scale groups (or whole blocks without double quantization) per rank."""
    bnb = pytest.importorskip("bitsandbytes")
    q = _load("fsdp2_qlora")
    nested = _Lora4bit(bnb, 0, 256, 256, True).base.weight  # 65536 elements
    plain = _Lora4bit(bnb, 0, 256, 96, False).base.weight  # 24576 elements, 384 blocks
    assert q.absmax_shard_span(nested, 2) == 64 * 256
    assert q.absmax_shard_span(nested, 4) == 64 * 256
    assert q.absmax_shard_span(nested, 8) is None
    assert q.absmax_shard_span(nested, 1) is None
    assert q.absmax_shard_span(plain, 4) == 64
    assert q.absmax_shard_span(plain, 7) is None
    model = torch.nn.Sequential(_Lora4bit(bnb, 0, 256, 256, True))
    absmax = model[0].base.weight.quant_state.absmax.clone()
    entries = q.register_absmax_shards(model, 2)
    assert len(entries) == 1
    assert "0.base.weight_unsloth_absmax" not in model.state_dict()
    # Not taken by FSDP2: back to the QuantState's own absmax, same bytes.
    assert q.finish_absmax_shards(entries) == 0
    assert "weight_unsloth_absmax" not in model[0].base._parameters
    assert torch.equal(model[0].base.weight.quant_state.absmax, absmax)
