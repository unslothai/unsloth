# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel finds routed expert stacks by shape, not name: Nemotron-H up/down and block-FP8 codes + scales."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

pytest.importorskip("transformers.core_model_loading")

from test_ep_sharded_load_gloo import _step  # noqa: E402

NEMOTRON_H = "trl-internal-testing/tiny-NemotronHForCausalLM-3.5-lightning"
NEMOTRON_STACKS = ("mixer.experts.up_proj", "mixer.experts.down_proj")
# PEFT refuses LoRA on Mamba's out_proj / conv1d; the shared expert's up_proj / down_proj are Linear modules.
NEMOTRON_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "up_proj", "down_proj"]
QWEN3_MOE = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
FP8_BLOCK = (4, 4)
# Attention only: PEFT's all-linear also wraps the expert stacks, and its CPU merge cannot add to float8 codes.
FP8_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _modules():
    from test_tp_native_oracle import build, tpn
    from unsloth_models_loader import load_models_module
    return build, tpn, load_models_module("ep_native")


def _worker(rank, world, tmp, case):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 300),
    )
    try:
        res = globals()[case](rank, world, tmp)
    except BaseException as exc:  # a rank must report, not vanish
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-4000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(case, world = 2):
    import tempfile
    import time

    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, tmp, case)) for r in range(world)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out


def _lora(
    model,
    target_parameters = (),
    target_modules = "all-linear",
):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = target_modules,
        target_parameters = list(target_parameters) or None,
        init_lora_weights = False,
    )
    return get_peft_model(model, config).train()


def _load(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype = torch.float32,
        attn_implementation = "eager",
        experts_implementation = "eager",
        device_map = {"": "cpu"},
    )


def _bits(t):
    return t.detach().view(torch.uint8) if t.element_size() == 1 else t.detach()


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _stacks(tpn, model):
    names = {id(tpn._experts_chain(b)[-1]) for b in tpn.moe_blocks_with_experts(model)}
    out = {}
    for path, module in model.named_modules():
        if id(module) in names:
            for n, p in module._parameters.items():
                if p is not None:
                    out[f"{path}.{n}"] = p
    return out


def _sliced(full, local, rank):
    """Every local stack is bitwise rows [rank * per, (rank + 1) * per) of the full one (matched by suffix)."""

    def plain(k):
        return k.removeprefix("base_model.model.").replace("base_layer.", "")

    full = {plain(k): v for k, v in full.items()}
    ok, per, seen = True, None, 0
    for k, p in local.items():
        whole = full[plain(k)]
        per = p.shape[0]
        ok &= per * 2 == whole.shape[0] and _same(p, whole[rank * per : (rank + 1) * per])
        seen += 1
    return bool(ok and seen), per, seen


def _grads_match(
    r,
    a = "sharded",
    b = "ref",
    exact = False,
):
    got, want = r[a][1], r[b][1]
    assert set(got) == set(want) and got, sorted(set(got) ^ set(want))[:6]
    for k, w in want.items():
        if exact:
            assert torch.equal(got[k], w), k
        else:
            torch.testing.assert_close(
                got[k], w, rtol = 1e-5, atol = 1e-6, msg = lambda m, k = k: f"{k}: {m}"
            )


def _case_nemotron_h(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        source = build(NEMOTRON_H, lora = False)
        source.generation_config.pad_token_id = None
        source.save_pretrained(ckpt, safe_serialization = True)
    dist.barrier()
    out = {"ep_native": epn.__file__}
    full = _load(ckpt)
    vocab = tpn._text_config(full).vocab_size
    leaf = tpn._experts_chain(tpn.moe_blocks_with_experts(full)[0])[-1]
    out["stacks"] = tpn.expert_stacks(leaf) if hasattr(tpn, "expert_stacks") else None
    out["problem"] = tpn.expert_shard_problem(tpn.moe_blocks_with_experts(full)[0], world)
    whole = {k: p.detach().clone() for k, p in _stacks(tpn, full).items()}
    with epn.expert_parallel_loading(mesh):
        model = _load(ckpt)
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    out["sliced"] = _sliced(whole, _stacks(tpn, model), rank)
    ref, model, full = (
        _lora(m, NEMOTRON_STACKS, NEMOTRON_MODULES) for m in (_load(ckpt), model, full)
    )
    out["ref"] = _step(epn, ref, rank, world, None, vocab)
    out["full"] = _step(epn, full, rank, world, mesh, vocab)
    out["sharded"] = _step(epn, model, rank, world, mesh, vocab)
    out["local"] = _sliced(whole, _stacks(tpn, full), rank)
    return out


def test_ep_nemotron_h_non_gated_experts_shard_by_shape_and_match_one_process():
    results = _spawn("_case_nemotron_h")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["problem"] is None, r["problem"]
        assert set(r["stacks"]) == {"up_proj", "down_proj"}, r["stacks"]
        assert r["local"] == (True, 2, 2), r["local"]
        assert r["sliced"] == (True, 2, 2) and r["stats"]["sharded_expert_stacks"] == 2, (
            r["sliced"],
            r["stats"],
        )
        assert r["sharded"][0] == r["full"][0]
        _grads_match(r, "sharded", "full", exact = True)
        assert r["full"][0] == pytest.approx(r["ref"][0], rel = 1e-6)
        _grads_match(r, "full", "ref")
        assert any("experts.lora_" in k for k in r["ref"][1])


def _fp8_on_cpu():
    """FP8Experts on CPU: transformers needs a GPU for FP8 and has no CPU kernel."""
    from transformers.integrations import finegrained_fp8 as F8
    from transformers.quantizers import quantizer_finegrained_fp8 as Q

    Q.FineGrainedFP8HfQuantizer.validate_environment = lambda self, *a, **k: None

    def linear(
        self,
        input,
        weight,
        weight_scale_inv,
        activation_scale = None,
    ):
        rows, cols = weight.shape[-2:]
        b0, b1 = self.block_size
        scale = weight_scale_inv.to(input.dtype).repeat_interleave(b0, -2)[..., :rows, :]
        scale = scale.repeat_interleave(b1, -1)[..., :cols]
        return torch.nn.functional.linear(input, weight.to(input.dtype) * scale)

    F8.FP8Experts.linear = linear
    return F8


def _fp8_checkpoint(build, path, tmp):
    """A tiny Qwen3-MoE whose routed experts are random block-FP8 codes + scales, saved as an FP8 checkpoint."""
    from transformers import FineGrainedFP8Config
    from transformers import AutoModelForCausalLM

    F8 = _fp8_on_cpu()
    plain = os.path.join(tmp, "plain")
    source = build(QWEN3_MOE, lora = False)
    source.generation_config.pad_token_id = None
    source.save_pretrained(plain, safe_serialization = True)
    config = FineGrainedFP8Config(
        weight_block_size = FP8_BLOCK,
        modules_to_not_convert = ["model.layers.*.self_attn", "model.layers.*.mlp.gate", "lm_head"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        plain, quantization_config = config, dtype = torch.float32, device_map = {"": "cpu"}
    )
    g = torch.Generator().manual_seed(5)
    for module in model.modules():
        if isinstance(module, F8.FP8Experts):
            for n, p in list(module.named_parameters(recurse = False)):
                if n.endswith("_scale_inv"):
                    new = torch.rand(p.shape, generator = g) * 0.02 + 0.005
                else:
                    new = (torch.randn(p.shape, generator = g) * 50).clamp(-448, 448)
                    new = new.to(torch.float8_e4m3fn)
                setattr(module, n, torch.nn.Parameter(new, requires_grad = False))
    model.save_pretrained(path, safe_serialization = True)


def _case_fp8(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    F8 = _fp8_on_cpu()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ckpt = os.path.join(tmp, "fp8")
    if rank == 0:
        _fp8_checkpoint(build, ckpt, tmp)
    dist.barrier()
    out = {"ep_native": epn.__file__}
    full = _load(ckpt)
    vocab = tpn._text_config(full).vocab_size
    out["fp8_experts"] = sum(isinstance(m, F8.FP8Experts) for m in full.modules())
    whole = {k: p.detach().clone() for k, p in _stacks(tpn, full).items()}
    out["dtypes"] = sorted({str(p.dtype) for p in whole.values()})
    out["problem"] = tpn.expert_shard_problem(tpn.moe_blocks_with_experts(full)[0], world)
    with epn.expert_parallel_loading(mesh):
        model = _load(ckpt)
    out["stats"] = dict(model.__dict__.get("_unsloth_tp_load_stats") or {})
    out["sliced"] = _sliced(whole, _stacks(tpn, model), rank)
    out["held"] = sum(p.numel() for p in _stacks(tpn, model).values())
    out["whole"] = sum(p.numel() for p in whole.values())
    ref, model, full = (_lora(m, (), FP8_MODULES) for m in (_load(ckpt), model, full))
    out["ref"] = _step(epn, ref, rank, world, None, vocab)
    out["full"] = _step(epn, full, rank, world, mesh, vocab)
    out["local"] = _sliced(whole, _stacks(tpn, full), rank)
    leaf = tpn._experts_chain(tpn.moe_blocks_with_experts(full)[0])[-1]
    out["num_experts"] = leaf.num_experts
    out["sharded"] = _step(epn, model, rank, world, mesh, vocab)
    return out


def test_ep_block_fp8_experts_keep_bitwise_local_codes_and_scales_and_match_one_process():
    results = _spawn("_case_fp8")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["fp8_experts"] == 2 and r["dtypes"] == ["torch.float32", "torch.float8_e4m3fn"], r
        assert r["problem"] is None, r["problem"]
        assert r["local"] == (True, 2, 8), r["local"]
        assert r["sliced"] == (True, 2, 8), r["sliced"]
        assert r["held"] * 2 == r["whole"]
        assert r["stats"]["sharded_expert_stacks"] == 8 and r["stats"].get("prequantized"), r[
            "stats"
        ]
        assert r["num_experts"] == 2  # FP8Experts.forward one-hots over its local experts
        assert r["sharded"][0] == r["full"][0]
        _grads_match(r, "sharded", "full", exact = True)
        assert r["full"][0] == pytest.approx(r["ref"][0], rel = 1e-6)
        _grads_match(r, "full", "ref")
