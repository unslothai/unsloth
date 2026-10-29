# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched expert LoRA on a fused-experts MoE: each adapter's rows match a lone PEFT run of that adapter."""

from __future__ import annotations

import copy

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import multi_lora as ml

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
Qwen3_5Moe = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
pytestmark = pytest.mark.skipif(Qwen3_5Moe is None, reason = "transformers has no Qwen3.5 MoE")

REPO = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"
DENSE = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXPERTS = ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
RANKS, ALPHAS = [2, 4, 3], [4.0, 8.0, 3.0]
T = 6


# Zoo's MoE forward launches Triton whenever CUDA is up, so on a GPU host everything runs on the GPU.
DEVICE = "cuda" if has_real_cuda() else "cpu"
TOL = 1e-4 if DEVICE == "cuda" else 1e-5


@pytest.fixture(scope = "module")
def base():
    from huggingface_hub import snapshot_download
    path = snapshot_download(REPO)
    return Qwen3_5Moe.from_pretrained(path, torch_dtype = torch.float32).eval().to(DEVICE)


def _batched(base):
    torch.manual_seed(0)
    model = copy.deepcopy(base)
    lm = model.model.language_model
    ml.inject_multi_lora(lm, DENSE, 3, RANKS, ALPHAS)
    wrapped = ml.inject_multi_lora_experts(lm, EXPERTS, 3, RANKS, ALPHAS)
    with torch.no_grad():
        # Non-zero B: with PEFT's zero init a wrong adapter would still give the base output.
        for _, layer in ml._layers(model):
            for B in layer.lora_B:
                B.normal_(0, 0.5)
    return model, wrapped


def _lone(base, model, u):
    """Adapter ``u`` exported, loaded into a plain PEFT model of the base, as a lone job would run it."""
    cfg = peft.LoraConfig(
        r = RANKS[u],
        lora_alpha = ALPHAS[u],
        target_modules = DENSE,
        target_parameters = EXPERTS,
        lora_dropout = 0.0,
    )
    lone = peft.get_peft_model(copy.deepcopy(base), cfg)
    state = ml.adapter_state_dict(model, u)
    result = peft.set_peft_model_state_dict(lone, state)
    assert not result.unexpected_keys, result.unexpected_keys[:3]
    assert not [k for k in result.missing_keys if "lora_" in k]
    return lone


def _ids():
    return torch.randint(0, 100, (3, T), generator = torch.Generator().manual_seed(1)).to(DEVICE)


@pytest.mark.parametrize("mode", ["stash", "fold"])
def test_each_adapter_matches_its_lone_peft_run(base, monkeypatch, mode):
    if mode == "fold":
        monkeypatch.setattr(ml, "_zoo_stash_read_store", lambda experts: None)
    model, wrapped = _batched(base)
    assert len(wrapped) == 2 * len(model.model.language_model.layers)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    out = model(input_ids = ids, use_cache = False).logits
    modes = {m.mode for m in model.modules() if isinstance(m, ml.MultiLoraExperts)}
    assert modes == {mode}
    for u in range(3):
        lone = _lone(base, model, u)
        ref = lone(input_ids = ids[u : u + 1], use_cache = False).logits[0]
        moved = (ref - base(input_ids = ids[u : u + 1]).logits[0]).abs().max()
        assert moved > 1e-3  # the adapter really changes the output
        torch.testing.assert_close(out[u], ref, atol = TOL, rtol = 1e-4)


def test_expert_grads_match_the_lone_run(base):
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    u = 1
    model(input_ids = ids, use_cache = False).logits[u].pow(2).mean().backward()
    lone = _lone(base, model, u)
    lone(input_ids = ids[u : u + 1], use_cache = False).logits[0].pow(2).mean().backward()
    layout = ml._peft_moe_lora_b_layout()
    for name, experts in model.named_modules():
        if not isinstance(experts, ml.MultiLoraExperts):
            continue
        depth = len(experts.parameter_names)
        for i, param in enumerate(experts.parameter_names):
            holder = lone.base_model.model.get_submodule(name)
            for _ in range(depth - 1 - i):
                holder = holder.base_layer
            mine = experts.adapters[param]
            gA, gB = mine.lora_A[u].grad, mine.lora_B[u].grad
            flat = ml.MultiLoraExpertParam.peft_tensors
            packed = type("G", (), {"lora_A": {u: gA}, "lora_B": {u: gB}})
            exp_A, exp_B = flat(packed, u, layout)
            torch.testing.assert_close(
                exp_A, holder.lora_A["default"].weight.grad, atol = 1e-6, rtol = 1e-4
            )
            torch.testing.assert_close(
                exp_B, holder.lora_B["default"].weight.grad, atol = 1e-6, rtol = 1e-4
            )
            other = experts.adapters[param].lora_A[0].grad
            assert other is None or other.abs().max() == 0
    dense = [(n, m) for n, m in model.named_modules() if isinstance(m, ml.MultiLoraLinear)]
    assert dense
    for name, layer in dense:
        holder = lone.base_model.model.get_submodule(name)
        assert layer.lora_A[u].requires_grad and layer.lora_B[u].requires_grad
        torch.testing.assert_close(
            layer.lora_A[u].grad, holder.lora_A["default"].weight.grad, atol = 1e-6, rtol = 1e-4
        )
        torch.testing.assert_close(
            layer.lora_B[u].grad, holder.lora_B["default"].weight.grad, atol = 1e-6, rtol = 1e-4
        )


def test_another_adapter_cannot_move_a_row(base):
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    before = model(input_ids = ids, use_cache = False).logits.detach()
    with torch.no_grad():
        for _, layer in ml._layers(model):
            layer.lora_A[0].mul_(3.0)
            layer.lora_B[0].add_(1.0)
    after = model(input_ids = ids, use_cache = False).logits.detach()
    assert (after[0] - before[0]).abs().max() > 1e-3
    torch.testing.assert_close(after[1:], before[1:], atol = 0, rtol = 0)


def test_fold_mode_survives_non_reentrant_checkpointing(base, monkeypatch):
    # A Zoo stash store that this forward never fills: the experts fall back to folding.
    monkeypatch.setattr(ml, "_zoo_stash_read_store", lambda experts: {})
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    experts = next(m for m in model.modules() if isinstance(m, ml.MultiLoraExperts))
    block = next(m for m in model.modules() if getattr(m, "experts", None) is experts)
    hidden = torch.randn(
        3 * T, model.config.text_config.hidden_size, device = DEVICE, requires_grad = True
    )
    router = block.gate
    with torch.no_grad():
        _, weights, indices = router(hidden)
    out = torch.utils.checkpoint.checkpoint(experts, hidden, indices, weights, use_reentrant = False)
    out.sum().backward()
    assert experts.mode == "fold" and hidden.grad is not None


def test_absent_adapters_stay_in_the_graph(base):
    model, _ = _batched(base)
    ids = _ids()
    clean = None
    with torch.no_grad():
        ml.set_multi_lora_batch(model, torch.tensor([0, 1, 1], device = DEVICE), seq_len = T)
        clean = model(input_ids = ids, use_cache = False).logits
        # A diverged absent adapter must not reach the others' outputs.
        for m in model.modules():
            if isinstance(m, ml.MultiLoraExperts):
                for adapter in m.adapters.values():
                    adapter.lora_B[2].fill_(float("nan"))
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 1], device = DEVICE), seq_len = T)
    logits = model(input_ids = ids, use_cache = False).logits
    torch.testing.assert_close(logits, clean, atol = 0, rtol = 0)
    logits.sum().backward()
    for m in model.modules():
        if isinstance(m, ml.MultiLoraExperts):
            for adapter in m.adapters.values():
                assert (
                    adapter.lora_A[2].grad is not None and adapter.lora_A[2].grad.abs().max() == 0
                )
                assert adapter.lora_A[1].grad is not None


@pytest.fixture(scope = "module")
def input_major(base):
    """The same experts stored input-major and flagged as Zoo's grouped-mm format (Qwen3-VL-MoE, GPT-OSS)."""
    model = copy.deepcopy(base)
    flagged = 0
    for module in model.modules():
        if (
            hasattr(module, "gate_up_proj")
            and hasattr(module, "down_proj")
            and module.gate_up_proj.dim() == 3
        ):
            with torch.no_grad():
                for name in ("gate_up_proj", "down_proj"):
                    param = getattr(module, name)
                    param.data = param.data.transpose(1, 2).contiguous()
            module._unsloth_grouped_mm_format = True
            flagged += 1
    assert flagged
    return model


@pytest.mark.parametrize("mode", ["stash", "fold"])
def test_input_major_grouped_mm_experts_match_their_lone_peft_run(
    base, input_major, monkeypatch, mode
):
    if mode == "fold":
        monkeypatch.setattr(ml, "_zoo_stash_read_store", lambda experts: None)
    ids = _ids()
    # Zoo's forward reads the flag: the relaid experts compute what the original ones do.
    torch.testing.assert_close(
        input_major(input_ids = ids).logits, base(input_ids = ids).logits, atol = TOL, rtol = 1e-4
    )
    model, _ = _batched(input_major)
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    out = model(input_ids = ids, use_cache = False).logits
    assert {m.mode for m in model.modules() if isinstance(m, ml.MultiLoraExperts)} == {mode}
    for u in range(3):
        ref = _lone(input_major, model, u)(input_ids = ids[u : u + 1], use_cache = False).logits[0]
        torch.testing.assert_close(out[u], ref, atol = TOL, rtol = 1e-4)


def test_fold_mode_under_autocast_matches_the_lone_run(base, monkeypatch):
    monkeypatch.setattr(ml, "_zoo_stash_read_store", lambda experts: None)
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    with torch.autocast(device_type = DEVICE, dtype = torch.bfloat16):
        out = model(input_ids = ids, use_cache = False).logits
        refs = [
            _lone(base, model, u)(input_ids = ids[u : u + 1], use_cache = False).logits[0]
            for u in range(3)
        ]
    for u in range(3):
        # One bf16 ulp of the logits; the bf16 fold was off by about 0.1.
        torch.testing.assert_close(out[u].float(), refs[u].float(), atol = 3e-2, rtol = 0)


def test_experts_read_directly_by_their_parent_are_refused():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(2, 4, 8))
            self.down_proj = torch.nn.Parameter(torch.zeros(2, 4, 4))

    class FastBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

        def forward(self, x):  # a grouped-GEMM fast path that never calls self.experts
            return x @ self.experts.gate_up_proj[0]

    with pytest.raises(ValueError, match = "reads experts.gate_up_proj itself"):
        ml.inject_multi_lora_experts(FastBlock(), ["experts.gate_up_proj"], 2, 2, 2)
