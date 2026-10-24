# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_tensor_parallel on a gloo CPU world (2 and 4 ranks) vs the same model in one process, fp32."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import socket  # noqa: E402
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


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    with pytest.raises(RuntimeError, match = "generation is not supported"):
        model.generate(input_ids = torch.zeros(1, 2, dtype = torch.long))
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
    from unsloth.models import fsdp2_native, tp_native
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
            res[f"FastModel.for_inference:{name}"] = _raised(lambda: FastModel.for_inference(level))
            if name != "decoder":
                res[f"generate:{name}"] = _raised(lambda: level.generate(input_ids = ids))
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


CASES = {
    "parity": _case_parity,
    "save_subgroup": _case_save_subgroup,
    "fused": _case_fused,
    "guards": _case_guards,
    "qlora": _case_qlora,
    "qlora_misaligned": _case_qlora_misaligned,
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
        torch.testing.assert_close(fused["O"], plain["O"], rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(fused["dX"], plain["dX"], rtol = 1e-5, atol = 1e-5)
        assert set(fused["grads"]) == set(plain["grads"]) and plain["grads"]
        for name in plain["grads"]:
            torch.testing.assert_close(
                fused["grads"][name], plain["grads"][name], rtol = 1e-4, atol = 1e-6, msg = name
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


def test_tp2_entry_points_outside_the_regions_are_refused():
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
                elif "for_inference" in key:
                    assert value and "for_inference is not supported" in value, (
                        applied,
                        key,
                        value,
                    )
                elif key.startswith("generate:"):
                    assert value and "generation is not supported" in value, (applied, key, value)
                elif key.startswith(("trainer:", "model_init:")):
                    assert value and "cannot be trained by the Trainer" in value, (
                        applied,
                        key,
                        value,
                    )
                elif key.startswith("cached:"):
                    assert value and "cached decoding" in value, (applied, key, value)


def test_save_refuses_a_tp_mesh_smaller_than_the_world():
    for res in _spawn("save_subgroup", 2):
        assert res["error"] and "data parallel > 1 is outside v1" in res["error"], res
