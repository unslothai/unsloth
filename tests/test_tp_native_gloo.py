# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_tensor_parallel on a gloo CPU world (2 and 4 ranks) against the same model in one process, fp32:

* forward logits and loss equal on every rank;
* step-0 LoRA gradients (DTensor grads gathered with full_tensor, packed layouts un-permuted) equal 1 process,
  including the Partial sums of replicated factors and of per-head q_norm / k_norm weights;
* tp_clip_grad_norm_ equals clip_grad_norm_ and 3 AdamW steps follow the same losses;
* save_tp_adapter writes the 1-process PEFT adapter (same keys, same values);
* the fused LoRA_QKV / LoRA_W Functions, fed by get_lora_parameters, match the non-fused TP forward (needs
  `import unsloth`, which some CPU hosts refuse: then skipped). The fused MLP is Triton: GPU only."""

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
from test_tp_native_oracle import ARCHS, QWEN3, LLAMA, _batch, build, skip_unsupported, tpn  # noqa: E402


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


def _case_parity(
    rank,
    world,
    tmp,
    arch,
    overrides = None,
    lora_kwargs = None,
    save_kwargs = None,
):
    from torch.distributed.device_mesh import init_device_mesh
    from safetensors.torch import load_file

    repo, base_overrides = ARCHS[arch]
    overrides = {**base_overrides, **(overrides or {})}
    train_norms = arch in ("qwen3", "gemma3")
    ref = build(repo, overrides, train_norms = train_norms, **(lora_kwargs or {}))
    model = build(repo, overrides, train_norms = train_norms, **(lora_kwargs or {}))
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
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
    ref_out = _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_)
    tp_out = _train(model, ids, 3, tpn.tp_clip_grad_norm_)
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


def _case_fused(rank, world, tmp):
    # The real package from this checkout, not the stub the oracle module loads tp_native under.
    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        del sys.modules[name]
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        # The same device spoof pytest's conftest applies in the parent (spawned ranks do not run it).
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "_tp_conftest", os.path.join(here, "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth
        from unsloth.kernels import fast_lora
        from unsloth.models import tp_native
    except Exception as e:  # a CPU host that refuses `import unsloth`
        return {"skip": repr(e)}
    from torch.distributed.device_mesh import init_device_mesh

    assert unsloth.__file__.startswith(sys.path[0]), unsloth.__file__

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


def _quantize_layers(model, nested, blocksize = 64):
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


def _qlora_model(nested, seed = 0, blocksize = 64):
    from peft import LoraConfig, get_peft_model

    model = _quantize_layers(build(LLAMA, QLORA_SIZES, lora = False, seed = seed), nested, blocksize)
    torch.manual_seed(seed + 1)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    # train(): bitsandbytes repacks eval-mode CPU Linear4bit weights for its CPU gemm, which rounds
    # differently from dequantize + matmul (no dropout here, so train mode changes nothing else).
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
    # Every local 4-bit shard dequantizes to its slice of the full dequantized weight.
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


CASES = {
    "parity": _case_parity,
    "fused": _case_fused,
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
        # The shard's codes and absmax (dequantized when double quantized) are sliced, not re-quantized:
        # every local 4-bit weight dequantizes bitwise to its slice of the 1-process weight.
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
