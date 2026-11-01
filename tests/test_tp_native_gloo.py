# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_tensor_parallel on a gloo CPU world (2 and 4 ranks) vs the same model in one process, fp32."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import (  # noqa: E402
    ARCHS,
    LLAMA,
    PHI3_PACKED_PLAN,
    QWEN3,
    _batch,
    build,
    skip_unsupported,
    tpn,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Small next to a typical trained-weight gradient (~6e-6 on the tiny llama lm_head), so Adam still moves it.
_ADAM_EPS = 1e-6


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _grads(model):
    return {
        n: tpn._full(p.grad, p).detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _train(model, ids, steps, clip):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(steps):
        out = model(input_ids = ids, labels = ids, use_cache = False)
        out.loss.backward()
        if step == 0:
            grads = _grads(model)
            logits = out.logits.detach().clone()
            # Replicated factors' Partial grads are summed in backward: grads land in their param's layout.
            mismatched = [
                n
                for n, p in model.named_parameters()
                if tpn._is_dtensor(p.grad) and tuple(p.grad.placements) != tuple(p.placements)
            ]
            assert not mismatched, mismatched[:4]
        norms.append(float(clip(params, 1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return logits, grads, losses, norms


def _clip_fp64(params, max_norm):
    """clip_grad_norm_ in float64: torch's CPU fp32 vector_norm is ~2e-4 off, above the parity tolerance."""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt().float()
    coef = torch.clamp(max_norm / (total + 1e-6), max = 1.0)
    for g in grads:
        g.mul_(coef)
    return total


def _tie_q_norms(model):
    # Two regions sharing one trainable per-head norm weight: its tp sum must run once, not once per region.
    layers = tpn._unwrap(model).model.layers
    layers[1].self_attn.q_norm.weight = layers[0].self_attn.q_norm.weight


def _case_parity(
    rank,
    world,
    tmp,
    arch,
    overrides = None,
    lora_kwargs = None,
    save_kwargs = None,
    plan = None,
    freeze_before_save = False,
    tie_norms = False,
    fp64_ref_clip = False,
):
    from torch.distributed.device_mesh import init_device_mesh
    from safetensors.torch import load_file

    repo, base_overrides = ARCHS[arch]
    overrides = {**base_overrides, **(overrides or {})}
    train_norms = arch in ("qwen3", "gemma3")
    ref = build(repo, overrides, train_norms = train_norms, **(lora_kwargs or {}))
    model = build(repo, overrides, train_norms = train_norms, **(lora_kwargs or {}))
    if tie_norms:
        _tie_q_norms(ref), _tie_q_norms(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"], plan = plan)
    assert tpn.is_tensor_parallel(model)
    assert tpn.apply_tensor_parallel(model, mesh["tp"]) is model  # idempotent
    # The training forward below releases the generate replica, so parity also covers that release.
    generated = model.generate(
        input_ids = torch.zeros(1, 2, dtype = torch.long), max_new_tokens = 2, pad_token_id = 0
    )
    assert generated.shape == (1, 4), generated.shape
    from test_fsdp2_native_units import _load_native

    try:
        _load_native()._refuse_tensor_parallel_model(model)
        fsdp2_refused = None
    except RuntimeError as e:
        fsdp2_refused = str(e)
    ids = _batch(ref)
    ref_out = _train(ref, ids, 3, _clip_fp64 if fp64_ref_clip else torch.nn.utils.clip_grad_norm_)
    tp_out = _train(model, ids, 3, tpn.tp_clip_grad_norm_)
    if freeze_before_save:
        # Frozen adapter (e.g. after training): PEFT still saves bias = "all" biases, trainable or not.
        for m in (ref, model):
            for p in m.parameters():
                p.requires_grad_(False)
    ref_dir, tp_dir = os.path.join(tmp, f"ref{rank}"), os.path.join(tmp, "tp")
    ref.save_pretrained(ref_dir, **(save_kwargs or {}))
    tpn.save_tp_adapter(model, tp_dir, **(save_kwargs or {}))
    q = tpn._weight(model._unsloth_tp_plan.regions[0].projs[0].module)[0]
    return {
        "ref": ref_out,
        "tp": tp_out,
        "saved_ref": load_file(os.path.join(ref_dir, "adapter_model.safetensors")),
        "saved_tp": load_file(os.path.join(tp_dir, "adapter_model.safetensors"))
        if rank == 0
        else None,
        "local_q": (tuple(q.to_local().shape), tuple(q.shape), q.placements[0].dim),
        "fsdp2_refused": fsdp2_refused,
    }


def _import_checkout_unsloth():
    """`import unsloth` from this checkout (not the stub tp_native is loaded under); None, or why it failed."""
    import pathlib

    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        del sys.modules[name]
    sys.path.insert(0, REPO)
    try:
        # conftest's device spoof (spawned ranks do not run it).
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_tp_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth
    except Exception as e:  # a CPU host that refuses `import unsloth`
        return repr(e)
    # By path: compile setup may put other directories in front of sys.path.
    root = pathlib.Path(REPO).resolve()
    assert root in pathlib.Path(unsloth.__file__).resolve().parents, unsloth.__file__
    return None


def _raised(fn):
    try:
        fn()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def _case_guards(rank, world, tmp):
    """Entry points outside the tp regions are refused on every wrapper level."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth import FastLanguageModel, FastModel
    from unsloth.models import fsdp2_native, tp_native, tp_replica
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    args = transformers.TrainingArguments(output_dir = tmp, report_to = [], use_cpu = True)
    out = {}
    for applied in ("peft", "causal"):
        model = build(LLAMA)
        levels = {
            "peft": model,
            "lora": model.base_model,
            "causal": model.base_model.model,
            "decoder": model.base_model.model.model,
        }
        tp_native.apply_tensor_parallel(levels[applied], mesh["tp"])
        ids = _batch(model)
        res = {}
        cached = model(input_ids = ids, use_cache = True)
        res["forward_without_cache"] = cached.logits is not None
        past = cached.past_key_values
        step = ids[:, -1:]
        for name, level in levels.items():
            res[f"is_tp:{name}"] = tp_native.is_tensor_parallel(level)
            res[f"fsdp2:{name}"] = _raised(
                lambda: fsdp2_native._refuse_tensor_parallel_model(level)
            )
            res[f"for_inference:{name}"] = _raised(lambda: FastLanguageModel.for_inference(level))
            res[f"replica_released:{name}"] = tp_replica.release_replica(level)
            res[f"FastModel.for_inference:{name}"] = _raised(lambda: FastModel.for_inference(level))
            tp_replica.release_replica(level)
            if name != "decoder":
                res[f"generate:{name}"] = _raised(
                    lambda: level.generate(input_ids = ids, max_new_tokens = 2, pad_token_id = 0)
                )
                tp_replica.release_replica(level)
                res[f"trainer:{name}"] = _raised(
                    lambda: transformers.Trainer(model = level, args = args)
                )
                res[f"model_init:{name}"] = _raised(
                    lambda: transformers.Trainer(model_init = lambda: level, args = args)
                )
            res[f"cached:{name}"] = _raised(
                lambda: level(input_ids = step, past_key_values = past, use_cache = True)
            )
        res["generate:get_base_model"] = _raised(
            lambda: model.get_base_model().generate(input_ids = ids)
        )
        params = [p for p in model.parameters() if p.requires_grad]
        model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        norm = tp_native.tp_clip_grad_norm_(params, 1e9)
        res["clip_mesh_equal"] = float(norm) == float(
            tp_native.tp_clip_grad_norm_(params, 1e9, mesh = mesh["tp"])
        )
        out[applied] = res
    return out


def _case_fused(rank, world, tmp):
    # The real package from this checkout, not the stub the oracle module loads tp_native under.
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from unsloth.kernels import fast_lora
    from unsloth.models import tp_native
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    calls = []
    apply = fast_lora._apply
    fast_lora._apply = lambda function, *args: calls.append(function.__name__) or apply(
        function, *args
    )
    out = {}
    for fused in (False, True):
        calls.clear()
        model = build(QWEN3, {"num_attention_heads": 8, "num_key_value_heads": 4})
        tp_native.apply_tensor_parallel(model, mesh["tp"])
        attn = tp_native._unwrap(model).model.layers[0].self_attn
        g = torch.Generator().manual_seed(5)
        X = torch.randn(
            2, 5, attn.q_proj.base_layer.weight.shape[1], generator = g, requires_grad = True
        )
        Xr = tp_native.copy_to_tp(X, attn._unsloth_tp_group)
        if fused:
            Q, K, V = fast_lora.apply_lora_qkv(attn, Xr, inplace = False)
        else:
            Q, K, V = attn.q_proj(Xr), attn.k_proj(Xr), attn.v_proj(Xr)
        Y = torch.cat([Q, K, V], -1)
        O_in = Y[..., : Q.shape[-1]].contiguous()
        O = fast_lora.apply_lora_o(attn, O_in) if fused else attn.o_proj(O_in)
        O = tp_native.reduce_from_tp(O, attn._unsloth_tp_group)
        (O.square().sum() + Y.square().sum()).backward()
        out[fused] = {
            "O": O.detach(),
            "dX": X.grad.detach(),
            "grads": _grads(attn),
            "calls": list(calls),
        }
    return out


def _quantize_layers(
    model,
    nested,
    blocksize = 64,
):
    """Every Linear in the decoder layers -> a bitsandbytes NF4 Linear4bit (the same bytes in every process)."""
    import bitsandbytes as bnb

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child) is not torch.nn.Linear or ".layers." not in f".{name}.":
                continue
            q = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias = child.bias is not None,
                compute_dtype = torch.float32,
                quant_type = "nf4",
                compress_statistics = nested,
            )
            packed, state = bnb.functional.quantize_4bit(
                child.weight.data, blocksize = blocksize, quant_type = "nf4", compress_statistics = nested
            )
            q.weight = bnb.nn.Params4bit(
                packed,
                requires_grad = False,
                quant_state = state,
                blocksize = blocksize,
                compress_statistics = nested,
                quant_type = "nf4",
                module = q,
                bnb_quantized = True,
            )
            if child.bias is not None:
                q.bias = child.bias
            setattr(module, child_name, q)
    return model


QLORA_SIZES = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
}


def _qlora_model(
    nested,
    seed = 0,
    blocksize = 64,
):
    from peft import LoraConfig, get_peft_model

    model = _quantize_layers(build(LLAMA, QLORA_SIZES, lora = False, seed = seed), nested, blocksize)
    torch.manual_seed(seed + 1)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    # train(): eval-mode CPU Linear4bit repacks for a CPU gemm that rounds differently.
    return get_peft_model(model, config).train()


def _case_qlora(rank, world, tmp, nested):
    import bitsandbytes as bnb
    from torch.distributed.device_mesh import init_device_mesh

    ref, model = _qlora_model(nested), _qlora_model(nested)
    full = {
        n: bnb.functional.dequantize_4bit(m.weight.data, m.weight.quant_state)
        for n, m in model.named_modules()
        if type(m).__name__ == "Linear4bit"
    }
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    shard_err = 0.0
    for region in model._unsloth_tp_plan.regions:
        for proj in region.projs:
            base = proj.module.base_layer
            name = next(n for n, m in model.named_modules() if m is base)
            local = bnb.functional.dequantize_4bit(base.weight.data, base.weight.quant_state)
            col = proj.style == tpn.COLWISE
            want = tpn.shard_slice(
                full[name],
                0 if col else 1,
                proj.parts if col else (full[name].shape[1],),
                rank,
                world,
            )
            assert local.shape == want.shape, (name, local.shape, want.shape)
            shard_err = max(shard_err, float((local - want).abs().max()))
    ids = _batch(ref)
    return {
        "ref": _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_),
        "tp": _train(model, ids, 3, tpn.tp_clip_grad_norm_),
        "shard_err": shard_err,
    }


class _DropoutMLP(torch.nn.Module):
    """LlamaMLP with dropout between the column (gate / up) and row (down) projections."""

    def forward(self, x):
        return self.down_proj(self.drop(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


def _add_mlp_dropout(model, p = 0.25):
    for module in list(model.modules()):
        if type(module).__name__ == "LlamaMLP":
            module.add_module("drop", torch.nn.Dropout(p))
            module.__class__ = type("LlamaMLP", (_DropoutMLP, type(module)), {})
    return model


def _case_mlp_dropout(rank, world, tmp, quantized):
    from torch.distributed.device_mesh import init_device_mesh

    def make():
        if quantized:
            return _add_mlp_dropout(_qlora_model(False))
        return _add_mlp_dropout(build(LLAMA, QLORA_SIZES).train())

    ref, model = make(), make()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    widths = [
        r.module._unsloth_tp_local_width for r in model._unsloth_tp_plan.regions if r.kind == "mlp"
    ]
    ids = _batch(ref)
    out = {}
    for name, m, clip in (
        ("ref", ref, torch.nn.utils.clip_grad_norm_),
        ("tp", model, tpn.tp_clip_grad_norm_),
    ):
        torch.manual_seed(1234)  # the same dropout RNG state on every rank and in the reference
        out[name] = _train(m, ids, 2, clip)
    out["widths"] = widths
    return out


def _case_qlora_misaligned(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    # blocksize 128: o_proj / down_proj row shards (in_features / 2 = 64) would split blocks.
    model = _qlora_model(False, blocksize = 128)
    before = {n: (type(p), p.data_ptr()) for n, p in model.named_parameters()}
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    with pytest.raises(ValueError, match = "row sharding would split blocks"):
        tpn.apply_tensor_parallel(model, mesh["tp"])
    after = {n: (type(p), p.data_ptr()) for n, p in model.named_parameters()}
    return {"unchanged": before == after, "flag": tpn.is_tensor_parallel(model)}


def _case_save_subgroup(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    # tp = 1 inside a 2-rank world (data parallel 2): the world-scoped writer / barrier would not match.
    mesh = init_device_mesh("cpu", (world, 1), mesh_dim_names = ("dp", "tp"))
    model = tpn.apply_tensor_parallel(build(LLAMA), mesh["tp"])
    return {"error": _raised(lambda: tpn.save_tp_adapter(model, os.path.join(tmp, "out")))}


def _save_checkpoint(
    arch,
    directory,
    overrides = None,
):
    """A tiny random model saved as several safetensors files (index + shards), like a Hub checkpoint."""
    repo, base_overrides = ARCHS[arch]
    model = build(repo, {**base_overrides, **(overrides or {})}, lora = False)
    model.generation_config.pad_token_id = None  # tiny repos ship -1, which save_pretrained rejects
    model.save_pretrained(directory, max_shard_size = "1MB", safe_serialization = True)
    return directory


def _case_sharded_load(
    rank,
    world,
    tmp,
    arch,
    ckpt,
    lora_kwargs = None,
):
    """sharded_loading + get_peft_model + apply_tensor_parallel vs a full load (1 process, and v1 TP)."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    kw = dict(torch_dtype = torch.float32, attn_implementation = "eager")

    def peft(model):
        torch.manual_seed(1)
        config = dict(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
        config.update(lora_kwargs or {})
        return get_peft_model(model, LoraConfig(**config)).eval()

    ref = peft(AutoModelForCausalLM.from_pretrained(ckpt, **kw))
    full_tp = peft(AutoModelForCausalLM.from_pretrained(ckpt, **kw))
    tpn.apply_tensor_parallel(full_tp, mesh["tp"])
    model = tpn.load_tensor_parallel(
        ckpt, mesh["tp"], loader = AutoModelForCausalLM.from_pretrained, **kw
    )
    stats = dict(model._unsloth_tp_load_stats)
    ids = _batch(ref)
    unapplied = _raised(lambda: model(input_ids = ids))
    model = peft(model)
    tpn.apply_tensor_parallel(model, mesh["tp"])
    # Every local tensor equals the v1 shard bitwise (weights, biases, LoRA, buffers).
    a, b = full_tp.state_dict(), model.state_dict()
    local = lambda t: t.to_local() if tpn._is_dtensor(t) else t  # noqa: E731
    diffs = sorted(set(a) ^ set(b)) + [
        k for k in a if k in b and not torch.equal(local(a[k]), local(b[k]))
    ]
    full_local = sum(local(p).numel() for p in full_tp.parameters())
    sharded_local = sum(local(p).numel() for p in model.parameters())
    from test_fsdp2_native_units import _load_native

    try:
        _load_native()._refuse_tensor_parallel_model(model)
        fsdp2_refused = None
    except RuntimeError as e:
        fsdp2_refused = str(e)
    ref_out = _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_)
    v1_out = _train(full_tp, ids, 3, tpn.tp_clip_grad_norm_)
    tp_out = _train(model, ids, 3, tpn.tp_clip_grad_norm_)
    ref_dir, tp_dir = os.path.join(tmp, f"ref{rank}"), os.path.join(tmp, "tp")
    ref.save_pretrained(ref_dir)
    tpn.save_tp_adapter(model, tp_dir)
    q = tpn._weight(model._unsloth_tp_plan.regions[0].projs[0].module)[0]
    return {
        "ref": ref_out,
        "tp": tp_out,
        "v1": v1_out,
        "diffs": diffs,
        "stats": stats,
        "unapplied": unapplied,
        "local_numel": (full_local, sharded_local),
        "saved_ref": load_file(os.path.join(ref_dir, "adapter_model.safetensors")),
        "saved_tp": load_file(os.path.join(tp_dir, "adapter_model.safetensors"))
        if rank == 0
        else None,
        "local_q": (tuple(q.to_local().shape), tuple(q.shape), q.placements[0].dim),
        "fsdp2_refused": fsdp2_refused,
    }


def _case_tp_fsdp2(
    rank,
    world,
    tmp,
    arch,
    fit_layout = True,
    lora_kwargs = None,
):
    """dp 2 x tp 2: apply_tensor_parallel over mesh['tp'], native FSDP2 shard_model over mesh['dp'], each dp
    replica on its own half of the batch, vs one process on the whole batch."""
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    repo, overrides = ARCHS[arch]
    mesh = init_device_mesh("cpu", (2, world // 2), mesh_dim_names = ("dp", "tp"))
    dp = mesh["dp"].get_local_rank()
    ref = build(repo, overrides, **(lora_kwargs or {}))
    model = build(repo, overrides, **(lora_kwargs or {}))
    ids = _batch(ref, bsz = 4)
    mine = ids[2 * dp : 2 * dp + 2]
    tpn.apply_tensor_parallel(model, mesh["tp"])
    if not native._supports_ignored_params(native._fsdp_api()[2]):
        # torch < 2.7: no fully_shard(ignored_params) to keep the frozen tp shards out of FSDP2.
        config = native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
        return {"unsupported": _raised(lambda: native.shard_model(model, config))}
    refusals = {
        "no_mesh": _raised(
            lambda: native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32))
        ),
        "tp_mesh": _raised(
            lambda: native.shard_model(
                model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["tp"])
            )
        ),
        "shard_frozen": _raised(
            lambda: native.shard_model(
                model,
                native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"], shard_frozen = True),
            )
        ),
    }
    native.shard_model(
        model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"], fit_layout = fit_layout)
    )
    layouts = tpn._param_layouts(model)
    params = [p for p in model.parameters() if p.requires_grad]
    ref_params = [p for p in ref.parameters() if p.requires_grad]
    # Adam's g / (sqrt(v) + eps) has slope lr / eps at g = 0. A full lm_head copy (modules_to_save) holds
    # hundreds of entries with |g| ~ 1e-9, where the default eps = 1e-8 turns ~1e-12 of reduction-order
    # noise into ~5e-6 of weight, half the saved-weight tolerance, and over it on some CI CPUs.
    opt = torch.optim.AdamW(params, lr = 1e-2, eps = _ADAM_EPS, foreach = False)
    ref_opt = torch.optim.AdamW(ref_params, lr = 1e-2, eps = _ADAM_EPS, foreach = False)
    losses, ref_losses, norms, native_norms, ref_norms = [], [], [], [], []
    for step in range(3):
        out = model(input_ids = mine, labels = mine, use_cache = False)
        out.loss.backward()
        r = ref(input_ids = ids, labels = ids, use_cache = False)
        r.loss.backward()
        if step == 0:
            grads = {
                n: tpn._full(p.grad, layout = layouts.get(id(p))).detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            ref_grads = {
                n: p.grad.detach().clone() for n, p in ref.named_parameters() if p.grad is not None
            }
            mismatched = [
                n
                for n, p in model.named_parameters()
                if p.grad is not None and tuple(p.grad.placements) != tuple(p.placements)
            ]
        native_norms.append(float(native.native_clip_grad_norm_(params, 1e9)))
        norms.append(float(tpn.tp_clip_grad_norm_(params, 1.0)))
        ref_norms.append(float(_clip_fp64(ref_params, 1.0)))
        opt.step(), opt.zero_grad(set_to_none = True)
        ref_opt.step(), ref_opt.zero_grad(set_to_none = True)
        loss = out.loss.detach().clone()
        dist.all_reduce(loss)  # mean over the 4 ranks = mean over the 2 dp halves (tp ranks agree)
        losses.append(float(loss) / world)
        ref_losses.append(float(r.loss))
    ref_dir, tp_dir = os.path.join(tmp, f"ref{rank}"), os.path.join(tmp, "tp")
    ref.save_pretrained(ref_dir)
    tpn.save_tp_adapter(model, tp_dir)
    dist.barrier()
    return {
        "grads": grads,
        "ref_grads": ref_grads,
        "mismatched": mismatched,
        "losses": losses,
        "ref_losses": ref_losses,
        "norms": norms,
        "native_norms": native_norms,
        "ref_norms": ref_norms,
        "refusals": refusals,
        "placements": {
            n: tuple(p.placements) for n, p in model.named_parameters() if p.requires_grad
        },
        "saved_ref": load_file(os.path.join(ref_dir, "adapter_model.safetensors")),
        "saved_tp": load_file(os.path.join(tp_dir, "adapter_model.safetensors"))
        if rank == 0
        else None,
    }


def _case_tp_fsdp2_trainable_norms(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names = ("dp", "tp"))
    model = build(QWEN3, train_norms = True)
    tpn.apply_tensor_parallel(model, mesh["tp"])
    return {
        "error": _raised(
            lambda: native.shard_model(
                model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
            )
        )
    }


class _Rows(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


def _trainer_args(tmp, bs, **overrides):
    import transformers

    kwargs = dict(
        output_dir = tmp,
        per_device_train_batch_size = bs,
        gradient_accumulation_steps = 2,
        max_steps = 3,
        learning_rate = 1e-2,
        optim = "adamw_torch",
        lr_scheduler_type = "constant",
        warmup_steps = 0,
        weight_decay = 0.0,
        max_grad_norm = 1.0,
        logging_steps = 1,
        save_strategy = "no",
        report_to = [],
        use_cpu = True,
        seed = 0,
        dataloader_drop_last = True,
        remove_unused_columns = False,
        disable_tqdm = True,
    )
    kwargs.update(overrides)
    return transformers.TrainingArguments(**kwargs)


def _case_trainer(
    rank,
    world,
    tmp,
    dp,
    tp,
    lora_kwargs = None,
):
    """The HF Trainer on a tp (dp = 1) or dp x tp (shard_model over mesh['dp']) model; tp = 1: the 1-process
    reference with a dp times larger batch. Sequential batches, so both read the same rows per step."""
    import torch.distributed as dist
    import transformers
    from safetensors.torch import load_file

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import fsdp2_native, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    seen = []

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, *args, **kwargs):
            seen.append(inputs["input_ids"].clone())
            return super().compute_loss(model, inputs, *args, **kwargs)

    model = build(LLAMA, **(lora_kwargs or {}))
    data = _Rows(list(_batch(model, bsz = 12 * dp, seq = 9)))
    refusals = {}
    if tp > 1:
        mesh = init_device_mesh("cpu", (dp, tp), mesh_dim_names = ("dp", "tp"))
        tp_native.apply_tensor_parallel(model, mesh["tp"])
        if dp > 1:
            refusals["no_shard_model"] = _raised(
                lambda: SequentialTrainer(
                    model = model, args = _trainer_args(tmp, 2), train_dataset = data
                )
            )
            fsdp2_native.shard_model(
                model, fsdp2_native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
            )
        for key, overrides in {
            "checkpoints": {"save_strategy": "steps", "save_steps": 1},
            "eval": {
                "eval_strategy": "steps",
                "eval_steps": 1,
            },  # evaluates over dp (test_tp_eval_gloo.py)
            "optim_8bit": {"optim": "adamw_bnb_8bit"},
        }.items():
            refusals[key] = _raised(
                lambda: SequentialTrainer(
                    model = model,
                    args = _trainer_args(tmp, 2, **overrides),
                    train_dataset = data,
                    eval_dataset = data,
                )
            )
    if tp > 1 and lora_kwargs:
        # A prebuilt scheduler already recorded the mixed groups: refused, not split under it.
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        refusals["prebuilt_scheduler"] = _raised(
            lambda: SequentialTrainer(
                model = model,
                args = _trainer_args(tmp, 2),
                train_dataset = data,
                optimizers = (opt, sched),
            ).train()
        )
    trainer = SequentialTrainer(
        model = model, args = _trainer_args(tmp, 2 * dp if tp == 1 else 2), train_dataset = data
    )
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    out_dir = os.path.join(tmp, f"adapter_{world}_{rank}")
    trainer.save_model(out_dir)
    if dist.is_initialized():
        dist.barrier()
    state = (
        tp_native.tp_full_state_dict(trainer.model, adapters_only = True)
        if tp > 1
        else {k: v.detach().clone() for k, v in trainer.model.state_dict().items() if "lora_" in k}
    )
    wrapped = type(trainer.model_wrapped).__name__
    return {
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h.get("grad_norm") for h in logs],
        "seen": [x.sum().item() for x in seen],
        "state": {k: v for k, v in state.items() if "lora_" in k},
        "saved": load_file(os.path.join(out_dir, "adapter_model.safetensors"))
        if rank == 0 or tp == 1
        else None,
        "wrapped": wrapped,
        "refusals": refusals,
        "mixed_groups": fsdp2_native._mixed_param_groups(trainer.optimizer),
    }


def _case_sharded_lora_init(rank, world, tmp, ckpt):
    """Weight-dependent LoRA init (PiSSA) on sharded_loading slices is refused before any mutation."""
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    out = {}
    for init in ("pissa", "gaussian"):
        model = tpn.load_tensor_parallel(
            ckpt, mesh["tp"], loader = AutoModelForCausalLM.from_pretrained, torch_dtype = torch.float32
        )
        config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = init)
        model = get_peft_model(model, config)
        out[init] = _raised(lambda: tpn.apply_tensor_parallel(model, mesh["tp"]))
        out[init + "_tp"] = tpn.is_tensor_parallel(model)
    return out


def _case_sampler(rank, world, tmp, dp, strategy):
    """Rows each rank's dataloader yields under a length-aware sampler, after a rank-local RNG draw."""
    import transformers

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    fields = transformers.TrainingArguments.__dataclass_fields__
    if strategy == "batch_rebalance" and "train_sampling_strategy" not in fields:
        return {"skip": "this transformers has no train_sampling_strategy = 'batch_rebalance'"}
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import fsdp2_native, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    tp = world // dp
    model = build(LLAMA)
    mesh = init_device_mesh("cpu", (dp, tp), mesh_dim_names = ("dp", "tp"))
    tp_native.apply_tensor_parallel(model, mesh["tp"])
    if dp > 1:
        try:
            fsdp2_native.shard_model(
                model, fsdp2_native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
            )
        except Exception as e:
            if "torch < 2.7" in str(e) or "torch 2.7+" in str(e):
                return {"skip": "torch < 2.7: tensor parallel x FSDP2 is refused"}
            raise
    g = torch.Generator().manual_seed(1)
    rows = []
    for i in range(48):
        ids = torch.randint(3, 100, (4 + int(torch.randint(0, 9, (1,), generator = g)),), generator = g)
        ids[0] = 3 + i
        rows.append(ids)

    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        ids = torch.zeros(len(features), n, dtype = torch.long)
        for i, f in enumerate(features):
            ids[i, : len(f["input_ids"])] = f["input_ids"]
        return {"input_ids": ids, "labels": ids.clone()}

    if strategy == "group_by_length" and "train_sampling_strategy" not in fields:
        overrides = {"group_by_length": True}
    else:
        overrides = {"train_sampling_strategy": strategy}
    trainer = transformers.Trainer(
        model = model,
        args = _trainer_args(tmp, 2, dataloader_drop_last = False, **overrides),
        train_dataset = _Rows(rows),
        data_collator = collate,
    )
    torch.rand(1 + 5 * rank)
    seen = []
    for step, batch in enumerate(trainer.get_train_dataloader()):
        if step == 6:
            break
        seen.append([int(r[0]) - 3 for r in batch["input_ids"]])
    return {"seen": seen}


def _case_sampler_resume(rank, world, tmp, dp, strategy):
    """Rows per rank for an uninterrupted run vs resumes at epoch boundaries and mid epoch."""
    import transformers

    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    fields = transformers.TrainingArguments.__dataclass_fields__
    if strategy == "batch_rebalance" and "train_sampling_strategy" not in fields:
        return {"skip": "this transformers has no train_sampling_strategy = 'batch_rebalance'"}
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import fsdp2_native, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    tp = world // dp
    mesh = init_device_mesh("cpu", (dp, tp), mesh_dim_names = ("dp", "tp"))
    g = torch.Generator().manual_seed(1)
    rows = []
    for i in range(12 * dp):  # 3 optimizer steps per epoch on every layout
        ids = torch.randint(3, 100, (4 + int(torch.randint(0, 9, (1,), generator = g)),), generator = g)
        ids[0] = 3 + i
        rows.append(ids)

    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        ids = torch.zeros(len(features), n, dtype = torch.long)
        for i, f in enumerate(features):
            ids[i, : len(f["input_ids"])] = f["input_ids"]
        return {"input_ids": ids, "labels": ids.clone()}

    if strategy == "group_by_length" and "train_sampling_strategy" not in fields:
        overrides = {"group_by_length": True}
    else:
        overrides = {"train_sampling_strategy": strategy}
    ckpt = os.path.join(tmp, "ckpt")

    def run(resume = None):
        model = build(LLAMA)
        tp_native.apply_tensor_parallel(model, mesh["tp"])
        if dp > 1:
            try:
                fsdp2_native.shard_model(
                    model, fsdp2_native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"])
                )
            except Exception as e:
                if "torch < 2.7" in str(e) or "torch 2.7+" in str(e):
                    return None
                raise
        args = _trainer_args(
            ckpt,
            2,
            max_steps = 8,
            dataloader_drop_last = False,
            save_strategy = "steps",
            save_steps = 1,
            logging_steps = 100,
            **overrides,
        )
        trainer = transformers.Trainer(
            model = model, args = args, train_dataset = _Rows(rows), data_collator = collate
        )
        seen = []
        training_step = trainer.training_step

        def record(model, inputs, *args, **kwargs):
            seen.append([int(r[0]) - 3 for r in inputs["input_ids"]])
            return training_step(model, inputs, *args, **kwargs)

        trainer.training_step = record
        torch.rand(1 + 5 * rank)  # a rank-local draw on the global RNG
        trainer.train(resume_from_checkpoint = resume)
        return seen

    full = run()
    if full is None:
        return {"skip": "torch < 2.7: tensor parallel x FSDP2 is refused"}
    # 3 steps per epoch, 2 micro-batches per step: 3 / 6 are epoch boundaries, 4 / 7 mid epoch 1 / 2.
    resumed = {k: run(os.path.join(ckpt, f"checkpoint-{k}")) for k in (3, 4, 6, 7)}
    return {"full": full, "resumed": resumed}


CASES = {
    "parity": _case_parity,
    "save_subgroup": _case_save_subgroup,
    "fused": _case_fused,
    "guards": _case_guards,
    "qlora": _case_qlora,
    "qlora_misaligned": _case_qlora_misaligned,
    "mlp_dropout": _case_mlp_dropout,
    "sharded_load": _case_sharded_load,
    "sharded_lora_init": _case_sharded_lora_init,
    "tp_fsdp2": _case_tp_fsdp2,
    "tp_fsdp2_trainable_norms": _case_tp_fsdp2_trainable_norms,
    "trainer": _case_trainer,
    "sampler": _case_sampler,
    "sampler_resume": _case_sampler_resume,
}


def _worker(rank, world, port, case, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def _check_parity(results, world):
    for rank, res in enumerate(results):
        ref_logits, ref_grads, ref_losses, ref_norms = res["ref"]
        logits, grads, losses, norms = res["tp"]
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads), sorted(set(grads) ^ set(ref_grads))[:6]
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6), (rank, losses, ref_losses)
        assert norms == pytest.approx(ref_norms, rel = 1e-5), (rank, norms, ref_norms)
        assert "does not compose with Unsloth tensor parallelism" in (res["fsdp2_refused"] or "")
        local, full, dim = res["local_q"]
        assert local[dim] * world == full[dim] and local[1 - dim] == full[1 - dim]
    saved_tp, saved_ref = results[0]["saved_tp"], results[0]["saved_ref"]
    assert set(saved_tp) == set(saved_ref)
    for key in saved_ref:
        assert saved_tp[key].shape == saved_ref[key].shape, key
        torch.testing.assert_close(saved_tp[key], saved_ref[key], rtol = 1e-4, atol = 1e-5, msg = key)


@pytest.mark.parametrize("arch", ["llama", "llama_bias", "qwen3", "phi3", "stablelm", "gemma2"])
def test_tp2_matches_one_process(arch):
    skip_unsupported(arch)
    _check_parity(_spawn("parity", 2, arch = arch), 2)


def test_tp2_refused_4bit_model_is_left_untouched():
    for res in _spawn("qlora_misaligned", 2):
        assert res["unchanged"] and not res["flag"]


def test_tp2_trainable_biases_match_one_process():
    # bias = "all": the row bias is added after the all-reduce, so its gradient is complete on every rank.
    results = _spawn("parity", 2, arch = "llama_bias", lora_kwargs = {"bias": "all"})
    _check_parity(results, 2)
    assert any("bias" in key for key in results[0]["saved_tp"])


def test_tp2_adapter_save_keeps_requested_embeddings():
    results = _spawn("parity", 2, arch = "llama", save_kwargs = {"save_embedding_layers": True})
    _check_parity(results, 2)
    assert any("embed_tokens" in key for key in results[0]["saved_tp"])


@pytest.mark.parametrize("nested", [False, True])
def test_tp2_qlora_matches_one_process(nested):
    results = _spawn("qlora", 2, nested = nested)
    for res in results:
        ref_logits, ref_grads, ref_losses, ref_norms = res["ref"]
        logits, grads, losses, norms = res["tp"]
        assert res["shard_err"] == 0.0
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads) and grads
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6)
        assert norms == pytest.approx(ref_norms, rel = 1e-5)


@pytest.mark.parametrize("quantized", [False, True], ids = ["bf16_weights", "4bit_weights"])
def test_tp2_mlp_dropout_between_column_and_row_matches_one_process(quantized):
    # The row projection's local in_features is intermediate_size / tp, also once a 4-bit weight is sharded.
    results = _spawn("mlp_dropout", 2, quantized = quantized)
    for res in results:
        assert res["widths"] and set(res["widths"]) == {QLORA_SIZES["intermediate_size"] // 2}, res[
            "widths"
        ]
        ref_logits, ref_grads, ref_losses, ref_norms = res["ref"]
        logits, grads, losses, norms = res["tp"]
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads) and grads
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6)
        assert norms == pytest.approx(ref_norms, rel = 1e-5)


def test_tp4_gqa_matches_one_process():
    overrides = {"num_attention_heads": 8, "num_key_value_heads": 4, "intermediate_size": 64}
    _check_parity(_spawn("parity", 4, arch = "qwen3", overrides = overrides), 4)


def test_fused_lora_functions_read_local_shards():
    results = _spawn("fused", 2)
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    for res in results:
        plain, fused = res[False], res[True]
        assert plain["calls"] == [] and fused["calls"] == ["LoRA_QKV", "LoRA_W"], (
            plain["calls"],
            fused["calls"],
        )
        for key in ("O", "dX"):
            # fp32 sums err by an ulp of their largest term, so a small element that cancels out of large ones
            # (dX here peaks near 1e3, where one ulp is 1.2e-4) needs an atol at the tensor's scale.
            atol = max(1e-5, 1e-6 * plain[key].abs().max().item())
            torch.testing.assert_close(
                fused[key], plain[key], rtol = 1e-5, atol = atol, msg = lambda m, key = key: f"{key}: {m}"
            )
        assert set(fused["grads"]) == set(plain["grads"]) and plain["grads"]
        for name in plain["grads"]:
            # The fused order sums in a different order than PEFT, so as for O / dX an element that cancels out of
            # large terms needs an atol at the tensor's scale; a wrong shard would be off by O(1).
            ref = plain["grads"][name]
            torch.testing.assert_close(
                fused["grads"][name],
                ref,
                rtol = 1e-4,
                atol = max(1e-6, 1e-5 * ref.abs().max().item()),
                msg = lambda m, name = name: f"{name}: {m}",
            )


def test_apply_refuses_a_2d_mesh():
    model = build(LLAMA)

    class _Mesh:
        ndim = 2

    with pytest.raises(ValueError, match = "1-D tp mesh"):
        tpn.apply_tensor_parallel(model, _Mesh())


def test_tp2_phi3_fused_qkv_under_a_packed_plan_matches_one_process():
    # The tiny Phi3 is GQA (4 heads, 2 kv heads): qkv_proj parts are (q, kv, kv), not two packed halves.
    _check_parity(_spawn("parity", 2, arch = "phi3", plan = PHI3_PACKED_PLAN), 2)


@pytest.mark.parametrize(
    "lora_kwargs,save_kwargs",
    [
        ({"bias": "all"}, None),  # frozen bias-bearing adapter: every bias, trainable or not
        ({"bias": "lora_only"}, None),
        ({"modules_to_save": ["lm_head"]}, None),
        ({"modules_to_save": ["lm_head"]}, {"save_embedding_layers": True}),
    ],
)
def test_tp2_adapter_save_matches_peft_selection(lora_kwargs, save_kwargs):
    results = _spawn(
        "parity",
        2,
        arch = "llama_bias",
        lora_kwargs = lora_kwargs,
        save_kwargs = save_kwargs,
        freeze_before_save = True,
        fp64_ref_clip = "modules_to_save" in lora_kwargs,
    )
    _check_parity(results, 2)
    saved = results[0]["saved_tp"]
    if (
        lora_kwargs.get("bias") == "all"
    ):  # "lora_only" saves no base_layer bias on PEFT 0.18 (1 process too)
        assert any("o_proj" in k and k.endswith("bias") for k in saved), sorted(saved)[:8]
    if "modules_to_save" in lora_kwargs:
        assert any("modules_to_save" in k or "lm_head" in k for k in saved), sorted(saved)[:8]


def test_tp2_tied_norm_weight_is_summed_once():
    _check_parity(_spawn("parity", 2, arch = "qwen3", tie_norms = True), 2)


def test_tp2_entry_points_outside_the_regions_are_refused_or_gather():
    results = _spawn("guards", 2)
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    for res in results:
        for applied, got in res.items():
            assert got["forward_without_cache"] and got["clip_mesh_equal"], (applied, got)
            for key, value in got.items():
                if key.startswith("is_tp:"):
                    assert value is True, (applied, key)
                elif key.startswith("fsdp2:"):
                    assert value and "does not compose with Unsloth tensor parallelism" in value, (
                        applied,
                        key,
                        value,
                    )
                elif "for_inference" in key or key.startswith("generate:"):
                    assert value is None, (
                        applied,
                        key,
                        value,
                    )  # gathered a replica (tp_replica.py)
                elif key.startswith("replica_released:"):
                    assert value is True, (applied, key, value)
                elif key.startswith("model_init:"):
                    assert value and "cannot be trained by the Trainer" in value, (
                        applied,
                        key,
                        value,
                    )
                elif key.startswith("trainer:"):
                    # The Trainer takes a tensor parallel model with its default args (save_strategy "steps":
                    # per-rank checkpoints).
                    assert value is None, (applied, key, value)
                elif key.startswith("cached:"):
                    assert value and "cached decoding" in value, (applied, key, value)


def test_save_refuses_a_tp_mesh_smaller_than_the_world():
    for res in _spawn("save_subgroup", 2):
        assert res["error"] and "data parallel > 1 is outside v1" in res["error"], res


@pytest.mark.parametrize(
    "arch,world,overrides",
    [
        ("llama", 2, None),
        ("llama_bias", 2, None),  # colwise biases sliced on read, row biases whole
        ("phi3", 2, None),  # fused qkv [q, kv, kv] and packed gate_up read as per-part slices
        ("gemma2", 2, None),  # tied embeddings, softcapping
        ("qwen3", 4, {"num_attention_heads": 8, "num_key_value_heads": 4, "intermediate_size": 64}),
    ],
)
def test_sharded_loading_matches_full_load(arch, world, overrides):
    skip_unsupported(arch)
    with tempfile.TemporaryDirectory() as ckpt:
        _save_checkpoint(arch, ckpt, overrides)
        results = _spawn("sharded_load", world, arch = arch, ckpt = ckpt)
    _check_parity(results, world)
    for res in results:
        assert res["diffs"] == [], res["diffs"][:6]
        # The sharded load trains exactly like v1's full load + apply_tensor_parallel.
        assert res["tp"][2] == res["v1"][2] and res["tp"][3] == res["v1"][3]
        torch.testing.assert_close(res["tp"][0], res["v1"][0], rtol = 0, atol = 0)
        assert res["unapplied"] and "call apply_tensor_parallel" in res["unapplied"], res[
            "unapplied"
        ]
        stats = res["stats"]
        # Only this rank's slices of the sharded projections were read; replicated tensors were read whole.
        assert stats["read_bytes_sliced"] * world == stats["sharded_numel"] * 4, stats
        assert stats["read_bytes_full"] == stats["replicated_numel"] * 4, stats
        assert stats["sharded_numel"] > 0 and stats["regions"] > 0, stats
        full_local, sharded_local = res["local_numel"]
        assert full_local == sharded_local


def test_sharded_loading_refuses_weight_dependent_lora_init():
    with tempfile.TemporaryDirectory() as ckpt:
        _save_checkpoint("llama", ckpt, None)
        results = _spawn("sharded_lora_init", 2, ckpt = ckpt)
    for res in results:
        assert "depends on the base weight" in (res["pissa"] or ""), res
        assert not res["pissa_tp"]
        assert res["gaussian"] is None and res["gaussian_tp"], res


def test_sharded_loading_refuses_quantized_loads():
    from transformers import AutoModelForCausalLM
    for key in ("load_in_4bit", "quantization_config"):
        with pytest.raises(ValueError, match = "16-bit safetensors checkpoints only"):
            tpn.load_tensor_parallel(
                LLAMA, None, loader = AutoModelForCausalLM.from_pretrained, **{key: True}
            )


# Per-step grad norms over 3 AdamW steps. phi3's fused qkv / gate_up shards add reduction-order noise that the
# optimizer compounds: on the CI runners its step-3 norm lands 6.9e-5 / 7.7e-5 relative off one process
# (1.5e-6 at step 1), while this host stays inside 1e-5. The step-1 per-gradient check in the test
# (rtol 1e-4, atol 1e-6) is what catches a real tp error such as a missed all-reduce, and it is unchanged.
_TP_FSDP2_NORM_REL = {"phi3": 2e-4}


@pytest.mark.parametrize(
    "arch,fit_layout,lora_kwargs",
    [
        ("llama", True, None),
        ("llama", None, None),  # automatic layout (frozen tp shards still replicated over dp)
        ("phi3", True, None),  # packed / fused parts: saved in the 1-GPU order through FSDP2
        (
            "llama_bias",
            True,
            {"bias": "all"},
        ),  # trainable column (Shard) and row (Replicate) biases
        (
            "llama",
            True,
            {"modules_to_save": ["lm_head"]},
        ),  # a dp-only (1-D) trainable outside the tp regions
    ],
)
def test_tp2_x_fsdp2_dp2_matches_one_process(arch, fit_layout, lora_kwargs):
    skip_unsupported(arch)
    results = _spawn("tp_fsdp2", 4, arch = arch, fit_layout = fit_layout, lora_kwargs = lora_kwargs)
    if "unsupported" in results[0]:
        assert all("torch 2.7+" in (r["unsupported"] or "") for r in results), results
        pytest.skip(
            "torch < 2.7: tensor parallel x FSDP2 is refused (no fully_shard(ignored_params))"
        )
    for rank, res in enumerate(results):
        assert set(res["grads"]) == set(res["ref_grads"]) and res["grads"]
        for name, ref in res["ref_grads"].items():
            torch.testing.assert_close(res["grads"][name], ref, rtol = 1e-4, atol = 1e-6, msg = name)
        assert not res["mismatched"], res["mismatched"][:4]
        assert res["losses"] == pytest.approx(res["ref_losses"], rel = 1e-5), (
            res["losses"],
            res["ref_losses"],
        )
        assert res["norms"] == pytest.approx(
            res["ref_norms"], rel = _TP_FSDP2_NORM_REL.get(arch, 1e-5)
        ), (
            res["norms"],
            res["ref_norms"],
        )
        # CPU vector_norm sums fp32 squares sequentially: about 3e-5 relative on the 2M-element lm_head copy.
        native_rel = 1e-4 if "modules_to_save" in (lora_kwargs or {}) else 1e-5
        assert res["native_norms"][0] == pytest.approx(res["ref_norms"][0], rel = native_rel)
        # FSDP2 shards the LoRA DTensors over dp on top of their tp placement.
        assert all(
            len(pl) == (1 if "lm_head" in n else 2) for n, pl in res["placements"].items()
        ), res["placements"]
        assert "the data parallel dimension" not in (res["refusals"]["no_mesh"] or "")
        assert "does not compose with Unsloth tensor parallelism" in (
            res["refusals"]["no_mesh"] or ""
        )
        assert "must be the data parallel dimension" in (res["refusals"]["tp_mesh"] or ""), res[
            "refusals"
        ]
        assert "shard_frozen = True" in (res["refusals"]["shard_frozen"] or ""), res["refusals"]
    saved_tp, saved_ref = results[0]["saved_tp"], results[0]["saved_ref"]
    assert set(saved_tp) == set(saved_ref)
    for key in saved_ref:
        torch.testing.assert_close(
            saved_tp[key],
            saved_ref[key],
            rtol = 1e-4,
            atol = 1e-5,
            msg = lambda m, key = key: f"{key}: {m}",
        )


def test_tp_x_fsdp2_refuses_trainable_head_norms():
    for res in _spawn("tp_fsdp2_trainable_norms", 4):
        assert res["error"] and "trainable per-head norms" in res["error"], res


def test_trainer_splits_mixed_optimizer_groups_for_tp_only():
    # modules_to_save lm_head: a plain trainable next to the LoRA DTensors (foreach / fused take one kind).
    kw = {"modules_to_save": ["lm_head"]}
    ref = _spawn("trainer", 1, dp = 1, tp = 1, lora_kwargs = kw)[0]
    if "skip" in ref:
        pytest.skip(f"import unsloth failed on this host: {ref['skip']}")
    for rank, res in enumerate(_spawn("trainer", 2, dp = 1, tp = 2, lora_kwargs = kw)):
        assert res["mixed_groups"] == 0, rank
        assert "already built the scheduler" in (res["refusals"]["prebuilt_scheduler"] or ""), res[
            "refusals"
        ]
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (
            rank,
            res["losses"],
            ref["losses"],
        )
        # transformers 5.x would clip mixed DTensor / plain gradients itself (_mixed_mesh_grad_norm). The
        # reference norm is torch's CPU fp32 vector_norm over the plain lm_head gradient: ~2e-4 off.
        assert res["grad_norms"] == pytest.approx(ref["grad_norms"], rel = 1e-3), (
            rank,
            res["grad_norms"],
            ref["grad_norms"],
        )


@pytest.mark.parametrize("dp,tp", [(1, 2), (2, 2)])
def test_trainer_trains_a_tensor_parallel_model_like_one_process(dp, tp):
    ref = _spawn("trainer", 1, dp = dp, tp = 1)[0]
    if "skip" in ref:
        pytest.skip(f"import unsloth failed on this host: {ref['skip']}")
    results = _spawn("trainer", dp * tp, dp = dp, tp = tp)
    if "torch < 2.7" in str(results[0]["refusals"].get("no_shard_model", "")):
        pytest.skip("torch < 2.7: tensor parallel x FSDP2 is refused")
    assert len(ref["losses"]) == 3
    for rank, res in enumerate(results):
        assert "DistributedDataParallel" not in res["wrapped"], res["wrapped"]
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (
            rank,
            res["losses"],
            ref["losses"],
        )
        assert res["grad_norms"] == pytest.approx(ref["grad_norms"], rel = 1e-5), (
            res["grad_norms"],
            ref["grad_norms"],
        )
        assert set(res["state"]) == set(ref["state"]) and ref["state"]
        for key, value in ref["state"].items():
            torch.testing.assert_close(res["state"][key], value, rtol = 1e-4, atol = 1e-6, msg = key)
        refusals = res["refusals"]
        # Checkpoints are per rank, bitsandbytes AdamW steps local shards (or switches to torch's AdamW).
        assert refusals["checkpoints"] is None, refusals
        assert refusals["eval"] is None, refusals
        assert refusals["optim_8bit"] is None, refusals
        if dp > 1:
            assert "covers 2 of 4 ranks" in (refusals["no_shard_model"] or ""), refusals
    # The tp ranks of a replica read the same batches; dp replicas read different ones.
    for rank, res in enumerate(results):
        peer = results[rank ^ 1]  # same dp replica (tp is the inner mesh dimension)
        assert res["seen"] == peer["seen"]
        if dp > 1:
            assert res["seen"] != results[(rank + tp) % (dp * tp)]["seen"]
    saved, saved_ref = results[0]["saved"], ref["saved"]
    assert set(saved) == set(saved_ref)
    for key in saved_ref:
        torch.testing.assert_close(saved[key], saved_ref[key], rtol = 1e-4, atol = 1e-6, msg = key)


@pytest.mark.parametrize("strategy", ["batch_rebalance", "group_by_length"])
@pytest.mark.parametrize("dp", [1, 2])
def test_trainer_length_samplers_give_tp_peers_the_same_rows(dp, strategy):
    results = _spawn("sampler", 2 * dp, dp = dp, strategy = strategy)
    if "skip" in results[0]:
        pytest.skip(results[0]["skip"])
    for rank, res in enumerate(results):
        assert res["seen"] and res["seen"] == results[rank ^ 1]["seen"], (
            rank,
            res,
            results[rank ^ 1],
        )
    if dp > 1:
        replicas = [{i for batch in results[r]["seen"] for i in batch} for r in (0, 2)]
        assert not replicas[0] & replicas[1], replicas


@pytest.mark.parametrize("strategy", ["batch_rebalance", "group_by_length"])
@pytest.mark.parametrize("dp", [1, 2])
def test_trainer_length_samplers_resume_like_an_uninterrupted_run(dp, strategy):
    results = _spawn("sampler_resume", 2 * dp, dp = dp, strategy = strategy)
    if "skip" in results[0]:
        pytest.skip(results[0]["skip"])
    for rank, res in enumerate(results):
        peer = results[rank ^ 1]
        assert res["full"] == peer["full"], (rank, res["full"], peer["full"])
        for k, seen in res["resumed"].items():
            assert seen == peer["resumed"][k], (rank, k, seen, peer["resumed"][k])
            assert seen == res["full"][2 * k :], (rank, k, seen, res["full"][2 * k :])
