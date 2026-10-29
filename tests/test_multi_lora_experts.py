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


MODES = ["grouped", "stash", "fold"]


def _need_grouped(dtype = torch.float32):
    """Skip where this torch / device has no native grouped GEMM in ``dtype`` (the module then runs per adapter)."""
    if not ml._grouped_supported(torch.device(DEVICE), dtype):
        pytest.skip(f"no native grouped GEMM in {dtype} on {DEVICE} with torch {torch.__version__}")


def _use_mode(monkeypatch, mode):
    """grouped: every adapter in one pass; stash / fold: one adapter at a time, through Zoo's stash or folded."""
    if mode == "grouped":
        _need_grouped()
    if mode == "grouped":
        monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", "0")
    if mode == "fold":
        monkeypatch.setattr(ml, "_zoo_stash_read_store", lambda experts: None)


def _modes(model):
    return {m.mode for m in model.modules() if isinstance(m, ml.MultiLoraExperts)}


@pytest.mark.parametrize("mode", MODES)
def test_each_adapter_matches_its_lone_peft_run(base, monkeypatch, mode):
    _use_mode(monkeypatch, mode)
    model, wrapped = _batched(base)
    assert len(wrapped) == 2 * len(model.model.language_model.layers)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    out = model(input_ids = ids, use_cache = False).logits
    assert _modes(model) == {mode}
    for u in range(3):
        lone = _lone(base, model, u)
        ref = lone(input_ids = ids[u : u + 1], use_cache = False).logits[0]
        moved = (ref - base(input_ids = ids[u : u + 1]).logits[0]).abs().max()
        assert moved > 1e-3  # the adapter really changes the output
        torch.testing.assert_close(out[u], ref, atol = TOL, rtol = 1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_expert_grads_match_the_lone_run(base, monkeypatch, mode):
    _use_mode(monkeypatch, mode)
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    u = 1
    model(input_ids = ids, use_cache = False).logits[u].pow(2).mean().backward()
    assert _modes(model) == {mode}
    lone = _lone(base, model, u)
    lone(input_ids = ids[u : u + 1], use_cache = False).logits[0].pow(2).mean().backward()
    for name, experts in model.named_modules():
        if not isinstance(experts, ml.MultiLoraExperts):
            continue
        layout = "rank_major" if experts.peft_mode == "fold" else ml._peft_moe_lora_b_layout()
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


@pytest.mark.parametrize("mode", MODES)
def test_another_adapter_cannot_move_a_row(base, monkeypatch, mode):
    _use_mode(monkeypatch, mode)
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    before = model(input_ids = ids, use_cache = False).logits.detach()
    with torch.no_grad():
        for _, layer in ml._layers(model):
            layer.lora_A[0].mul_(3.0)
            layer.lora_B[0].add_(1.0)
    after = model(input_ids = ids, use_cache = False).logits.detach()
    assert _modes(model) == {mode}
    assert (after[0] - before[0]).abs().max() > 1e-3
    # Adapter 0's attention LoRA reroutes its tokens, so in grouped mode the other adapters' rows sit at other
    # positions of each expert's base GEMM. GPU kernels are row-position independent (bit-exact); AVX2 / NEON CPU
    # BLAS tails round a row by an ulp or two depending on where it falls, so allow that much there.
    atol = 1e-6 if mode == "grouped" and DEVICE == "cpu" else 0
    torch.testing.assert_close(after[1:], before[1:], atol = atol, rtol = 0)


@pytest.mark.parametrize("mode", ["grouped", "fold"])
def test_non_reentrant_checkpointing(base, monkeypatch, mode):
    _use_mode(monkeypatch, mode)
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
    assert experts.mode == mode and experts.peft_mode == "fold" and hidden.grad is not None
    eager = hidden.grad.clone()
    hidden.grad = None
    experts(hidden, indices, weights).sum().backward()
    torch.testing.assert_close(hidden.grad, eager)


@pytest.mark.parametrize("mode", MODES)
def test_absent_adapters_stay_in_the_graph(base, monkeypatch, mode):
    _use_mode(monkeypatch, mode)
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
    assert _modes(model) == {mode}
    for m in model.modules():
        if isinstance(m, ml.MultiLoraExperts):
            for adapter in m.adapters.values():
                assert (
                    adapter.lora_B[2].grad is not None and adapter.lora_B[2].grad.abs().max() == 0
                )
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


@pytest.mark.parametrize("mode", MODES)
def test_input_major_grouped_mm_experts_match_their_lone_peft_run(
    base, input_major, monkeypatch, mode
):
    _use_mode(monkeypatch, mode)
    ids = _ids()
    # Zoo's forward reads the flag: the relaid experts compute what the original ones do.
    torch.testing.assert_close(
        input_major(input_ids = ids).logits, base(input_ids = ids).logits, atol = TOL, rtol = 1e-4
    )
    model, _ = _batched(input_major)
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    out = model(input_ids = ids, use_cache = False).logits
    assert _modes(model) == {mode}
    for u in range(3):
        ref = _lone(input_major, model, u)(input_ids = ids[u : u + 1], use_cache = False).logits[0]
        torch.testing.assert_close(out[u], ref, atol = TOL, rtol = 1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_autocast_matches_the_lone_run(base, monkeypatch, mode):
    if DEVICE == "cuda" and not torch.cuda.is_bf16_supported(including_emulation = False):
        pytest.skip("no bfloat16 autocast on this GPU")
    _use_mode(monkeypatch, mode)
    model, _ = _batched(base)
    ids = _ids()
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
    with torch.autocast(device_type = DEVICE, dtype = torch.bfloat16):
        out = model(input_ids = ids, use_cache = False).logits
        refs = [
            _lone(base, model, u)(input_ids = ids[u : u + 1], use_cache = False).logits[0]
            for u in range(3)
        ]
    assert _modes(model) == {mode}
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


class _Experts(torch.nn.Module):
    """Fused experts in transformers' layout and eager forward: gate_up_proj (E, 2I, H), down_proj (E, H, I)."""

    def __init__(
        self,
        E = 5,
        H = 20,
        I = 12,
        bias = False,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * I, H, generator = g) * 0.3)
        self.down_proj = torch.nn.Parameter(torch.randn(E, H, I, generator = g) * 0.3)
        if bias:
            self.gate_up_proj_bias = torch.nn.Parameter(torch.randn(E, 2 * I, generator = g))
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for t in range(hidden_states.shape[0]):
            for j, e in enumerate(top_k_index[t].tolist()):
                if e >= self.gate_up_proj.shape[0]:
                    continue
                h = hidden_states[t] @ self.gate_up_proj[e].t()
                if hasattr(self, "gate_up_proj_bias"):
                    h = h + self.gate_up_proj_bias[e]
                gate, up = h.chunk(2, dim = -1)
                y = (self.act_fn(gate) * up) @ self.down_proj[e].t()
                out[t] = out[t] + y * top_k_weights[t, j]
        return out


def _routes(
    T,
    k,
    E,
    seed = 0,
):
    """Skewed routing: most routes on expert 0, expert E-2 never hit, one dropped route (id E)."""
    g = torch.Generator().manual_seed(seed)
    index = torch.randint(0, E, (T, k), generator = g)
    index[torch.rand(T, k, generator = g) < 0.5] = 0
    index[index == E - 2] = E - 1
    index[0, -1] = E
    weights = torch.rand(T, k, generator = g)
    return index, weights


def _wrapped(module, ranks, alphas, rslora, ids):
    experts = ml.MultiLoraExperts(
        module, ["gate_up_proj", "down_proj"], len(ranks), ranks, alphas, rslora
    )
    torch.manual_seed(0)
    with torch.no_grad():
        for adapter in experts.adapters.values():
            adapter.reset_parameters()
            for B in adapter.lora_B:
                B.normal_(0, 0.5, generator = torch.Generator().manual_seed(B.numel()))
    ml.set_multi_lora_batch(experts, ids)
    return experts


def _oracle(experts, hidden, index, weights, ids, rslora):
    """float64, route by route: base plus the token's own adapter delta (scale alpha / r or alpha / sqrt(r))."""
    base = experts.base_layer
    E = base.gate_up_proj.shape[0]
    factors = {
        name: [(A.double(), B.double()) for A, B in zip(adapter.lora_A, adapter.lora_B)]
        for name, adapter in experts.adapters.items()
    }
    out = []
    for t in range(hidden.shape[0]):
        u = int(ids[t])
        acc = 0
        for j in range(index.shape[1]):
            e = int(index[t, j])
            if e >= E:
                continue

            def proj(name, x):
                W = getattr(base, name)[e].double()
                if u >= 0:
                    A, B = factors[name][u]
                    r = A.shape[1]
                    scale = ALPHAS[u] / (r**0.5 if rslora else r)
                    W = W + scale * B[e] @ A[e]
                return x @ W.t()

            gate, up = proj("gate_up_proj", hidden[t]).chunk(2, dim = -1)
            acc = acc + proj("down_proj", torch.nn.functional.silu(gate) * up) * weights[t, j]
        out.append(acc)
    return torch.stack(out), factors


@pytest.mark.parametrize("rslora", [False, True])
@pytest.mark.parametrize("present", ["all", "no_adapter_1"])
def test_grouped_matches_a_float64_oracle(monkeypatch, rslora, present):
    _need_grouped()
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    T, k, E = 24, 2, 5
    ids = torch.tensor([0, 0, 2, -1, 0, 1, 2, 0] * 3)
    if present == "no_adapter_1":
        ids[ids == 1] = 2
    experts = _wrapped(_Experts(E), RANKS, ALPHAS, rslora, ids)
    assert not experts.transposed and not experts.reversed_io
    index, weights = _routes(T, k, E)
    hidden = torch.randn(T, 20, generator = torch.Generator().manual_seed(2))
    hidden64, weights64 = hidden.double().requires_grad_(), weights.double().requires_grad_()
    hidden, weights = hidden.requires_grad_(), weights.requires_grad_()
    calls = ml.MULTI_LORA_MOE_STATS["grouped_calls"]
    out = experts(hidden, index, weights)
    assert experts.mode == "grouped" and ml.MULTI_LORA_MOE_STATS["grouped_calls"] == calls + 1
    ref, factors = _oracle(experts, hidden64, index, weights64, ids, rslora)
    torch.testing.assert_close(out.double(), ref, atol = 1e-5, rtol = 1e-4)
    grad = torch.randn(out.shape, generator = torch.Generator().manual_seed(3))
    (out * grad).sum().backward()
    leaves = [hidden64, weights64] + [
        t for pairs in factors.values() for pair in pairs for t in pair
    ]
    ref_grads = torch.autograd.grad((ref * grad.double()).sum(), leaves, allow_unused = True)
    mine = [hidden.grad, weights.grad] + [
        p.grad
        for adapter in experts.adapters.values()
        for pair in zip(adapter.lora_A, adapter.lora_B)
        for p in pair
    ]
    for got, want in zip(mine, ref_grads):
        want = torch.zeros_like(got, dtype = torch.float64) if want is None else want
        assert got is not None
        torch.testing.assert_close(got.double(), want, atol = 1e-5, rtol = 1e-4)
    if present == "no_adapter_1":
        for adapter in experts.adapters.values():
            assert (
                adapter.lora_A[1].grad.abs().max() == 0 and adapter.lora_B[1].grad.abs().max() == 0
            )


@pytest.mark.parametrize("U", [2, 3])
def test_grouped_is_one_pass_on_zoo_gemms(monkeypatch, U):
    """Every adapter in one pass: 2 base GEMMs through Zoo's base path and 4 LoRA grouped GEMMs, whatever U is."""
    _need_grouped()
    from unsloth_zoo.temporary_patches import moe_utils

    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    T, k, E = 24, 2, 5
    ids = torch.tensor([u % U for u in range(T)])
    experts = _wrapped(_Experts(E), RANKS[:U], ALPHAS[:U], False, ids)
    index, weights = _routes(T, k, E)
    hidden = torch.randn(T, 20, generator = torch.Generator().manual_seed(2))
    experts(hidden, index, weights)
    assert experts.mode == "grouped"
    calls = {"base": 0, "lora": 0}
    base_mm, lora_mm = moe_utils._base_grouped_mm, moe_utils._grouped_mm_with_backward_fix

    def count(kind, fn):
        def wrapper(*args, **kwargs):
            # LoRA GEMMs are the ones on the stacked (E * (U + 1), ...) factors; Zoo's base path may call the same op.
            if kind == "base" or args[1].shape[0] == E * (U + 1):
                calls[kind] += 1
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(moe_utils, "_base_grouped_mm", count("base", base_mm))
    monkeypatch.setattr(moe_utils, "_grouped_mm_with_backward_fix", count("lora", lora_mm))
    experts(hidden, index, weights)
    assert calls == {"base": 2, "lora": 4}


def test_one_present_adapter_runs_its_own_pass(monkeypatch):
    """A batch with one adapter's tokens has nothing to fuse: that call runs per adapter, with the same result."""
    _need_grouped()
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    T, k, E = 24, 2, 5
    index, weights = _routes(T, k, E)
    hidden = torch.randn(T, 20, generator = torch.Generator().manual_seed(2))
    experts = _wrapped(_Experts(E), RANKS, ALPHAS, False, torch.tensor([0, 1, 2, -1] * 6))
    experts(hidden, index, weights)
    assert experts.mode == "grouped"
    ml.set_multi_lora_batch(experts, torch.tensor([1, -1] * 12))
    before = dict(ml.MULTI_LORA_MOE_STATS)
    out = experts(hidden, index, weights)
    assert ml.MULTI_LORA_MOE_STATS["per_adapter_calls"] == before["per_adapter_calls"] + 1
    assert ml.MULTI_LORA_MOE_STATS["grouped_calls"] == before["grouped_calls"]
    torch.testing.assert_close(
        out, experts._grouped(hidden, index, weights, torch.tensor([1, -1] * 12))
    )


def test_grouped_engages_and_the_switch_turns_it_off(base, monkeypatch):
    _need_grouped()
    ids = _ids()
    counts = []
    for value in ("1", "0"):
        monkeypatch.setenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", value)
        model, wrapped = _batched(base)
        ml.set_multi_lora_batch(model, torch.tensor([0, 1, 2], device = DEVICE), seq_len = T)
        before = dict(ml.MULTI_LORA_MOE_STATS)
        model(input_ids = ids, use_cache = False)
        counts.append({key: ml.MULTI_LORA_MOE_STATS[key] - before[key] for key in before})
    layers = len(wrapped) // 2
    assert counts[0] == {"grouped_calls": layers, "per_adapter_calls": 0, "grouped_fallbacks": 0}
    assert counts[1] == {"grouped_calls": 0, "per_adapter_calls": layers, "grouped_fallbacks": 0}


@pytest.mark.parametrize("why", ["bias", "parity"])
def test_unsupported_experts_fall_back_to_one_adapter_at_a_time(monkeypatch, why):
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    if why == "parity":
        grouped = ml.MultiLoraExperts._grouped
        monkeypatch.setattr(ml.MultiLoraExperts, "_grouped", lambda *a: grouped(*a) * 1.01)
    T, k, E = 24, 2, 5
    ids = torch.tensor([0, 1, 2, -1] * 6)
    index, weights = _routes(T, k, E)
    hidden = torch.randn(T, 20, generator = torch.Generator().manual_seed(2))
    experts = _wrapped(_Experts(E, bias = why == "bias"), RANKS, ALPHAS, False, ids)
    before = dict(ml.MULTI_LORA_MOE_STATS)
    out = experts(hidden, index, weights)
    assert experts.mode == "fold"
    assert ml.MULTI_LORA_MOE_STATS["grouped_fallbacks"] == before["grouped_fallbacks"] + 1
    assert ml.MULTI_LORA_MOE_STATS["grouped_calls"] == before["grouped_calls"]
    monkeypatch.setenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", "0")
    experts.mode = experts.peft_mode = None
    torch.testing.assert_close(out, experts(hidden, index, weights), atol = 0, rtol = 0)


def test_grouped_reads_bnb_4bit_stacks_through_zoo(monkeypatch):
    _need_grouped()
    bnb = pytest.importorskip("bitsandbytes")
    from unsloth_zoo.temporary_patches import moe_utils

    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    # The synthetic forward cannot run a 4-bit stack, so there is no per-adapter path to probe.
    monkeypatch.setattr(ml.MultiLoraExperts, "_run_stash", lambda *a: None)
    monkeypatch.setattr(ml.MultiLoraExperts, "_grouped_parity", lambda *a: None)
    T, k, E = 24, 2, 5
    module = _Experts(E, H = 32, I = 16)
    for name in ("gate_up_proj", "down_proj"):
        W = getattr(module, name).data.to(torch.bfloat16)
        packed, state = bnb.functional.quantize_4bit(W.reshape(-1, W.shape[-1]), quant_type = "nf4")
        param = torch.Tensor._make_subclass(bnb.nn.Params4bit, packed, False)
        param.quant_state, param._original_shape = state, W.shape
        module._parameters[name] = param
    ids = torch.tensor([0, 1, 2, -1] * 6)
    experts = _wrapped(module, RANKS, ALPHAS, False, ids)
    index, weights = _routes(T, k, E)
    hidden = torch.randn(T, 32, generator = torch.Generator().manual_seed(2), requires_grad = True)
    seen = []
    get_base = moe_utils._get_base_weight
    monkeypatch.setattr(
        moe_utils, "_get_base_weight", lambda *a, **kw: seen.append(1) or get_base(*a, **kw)
    )
    out = experts(hidden, index, weights)
    assert experts.mode == "grouped" and out.dtype == torch.float32
    # Zoo's provider dequantizes both stacks in forward and, where Zoo recomputes, again in backward.
    forward_reads = len(seen)
    assert forward_reads >= 2
    out.sum().backward()
    if moe_utils._moe_recompute_enabled(experts.base_layer.gate_up_proj, dtype = torch.bfloat16):
        assert len(seen) >= 2 * forward_reads
    float_module = _Experts(E, H = 32, I = 16)
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            # The oracle reads the stack as Zoo's provider does (CPU bitsandbytes slices differ from full reads).
            param = getattr(experts.base_layer, name)
            getattr(float_module, name).copy_(get_base(param, torch.bfloat16).float())
    reference = _wrapped(float_module, RANKS, ALPHAS, False, ids)
    ref, _ = _oracle(reference, hidden.detach().double(), index, weights.double(), ids, False)
    # bfloat16 compute, as Unsloth Zoo runs 4-bit experts.
    torch.testing.assert_close(out.double(), ref, atol = 2e-2 * ref.abs().max().item(), rtol = 0)


def test_grouped_dropped_route_never_poisons_gradients(monkeypatch):
    _need_grouped()
    # A dropped route (expert id E) with a NaN weight must add exactly nothing, forward and backward.
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    T, k, E = 8, 2, 5
    ids = torch.tensor([0, 1, 2, -1, 0, 1, 2, 0])
    experts = _wrapped(_Experts(E), RANKS, ALPHAS, False, ids)
    index, weights = _routes(T, k, E)
    hidden = torch.randn(T, 20, generator = torch.Generator().manual_seed(2))
    experts(hidden, index, weights)
    assert experts.mode == "grouped"
    weights[0, -1] = float("nan")
    kept, clean = index.clone(), weights.clone()
    kept[0, -1], clean[0, -1] = 0, 0.0

    def run(index, weights):
        h, w = hidden.clone().requires_grad_(), weights.clone().requires_grad_()
        experts.zero_grad(set_to_none = True)
        out = experts(h, index, w)
        out.backward(torch.ones_like(out))
        grads = [p.grad.clone() for p in experts.parameters() if p.grad is not None]
        return out, h.grad, w.grad, grads

    out, d_hidden, d_weights, grads = run(index, weights)
    ref, ref_hidden, ref_weights, ref_grads = run(kept, clean)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(d_hidden, ref_hidden)
    assert d_weights[0, -1] == 0
    torch.testing.assert_close(d_weights[:, :1], ref_weights[:, :1])
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r)


def test_grouped_empty_top_k_gives_zeros(monkeypatch):
    _need_grouped()
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    ids = torch.tensor([0, 1, 2, -1])
    experts = _wrapped(_Experts(5), RANKS, ALPHAS, False, ids)
    index, weights = _routes(4, 2, 5)
    experts(torch.randn(4, 20), index, weights)
    assert experts.mode == "grouped"
    hidden = torch.randn(4, 20, requires_grad = True)
    out = experts(hidden, index[:, :0], weights[:, :0])
    assert out.shape == (4, 20) and not out.any()
    out.sum().backward()
    assert not hidden.grad.any()
    for adapter in experts.adapters.values():
        for p in (*adapter.lora_A, *adapter.lora_B):
            assert p.grad is not None and not p.grad.any()


class _StaticExperts(_Experts):
    """The same experts with a data-independent forward (every expert, masked), so it traces with fullgraph."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for e in range(self.gate_up_proj.shape[0]):
            w = (top_k_weights * (top_k_index == e)).sum(-1, keepdim = True)
            gate, up = (hidden_states @ self.gate_up_proj[e].t()).chunk(2, dim = -1)
            out = out + ((self.act_fn(gate) * up) @ self.down_proj[e].t()) * w
        return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_under_torch_compile(monkeypatch, dtype):
    # Grouped GEMM only traces in bfloat16; other dtypes take the per-adapter path inside a compiled graph.
    _need_grouped(dtype)
    monkeypatch.delenv("UNSLOTH_MULTI_LORA_MOE_GROUPED", raising = False)
    T, k, E = 16, 2, 5
    ids = torch.tensor([0, 1, 2, -1] * 4)
    experts = _wrapped(_StaticExperts(E, H = 24, I = 16).to(dtype), RANKS, ALPHAS, False, ids).to(
        DEVICE
    )
    for _, layer in ml._layers(experts):
        layer.seen = None  # a plain attribute, so .to() left it on the CPU
    ml.set_multi_lora_batch(experts, ids.to(DEVICE))
    index, weights = _routes(T, k, E)
    index[0, -1] = 0
    index, weights = index.to(DEVICE), weights.to(DEVICE)
    hidden = (
        torch.randn(T, 24, generator = torch.Generator().manual_seed(2))
        .to(DEVICE, dtype)
        .requires_grad_()
    )
    weights = weights.to(dtype)
    eager = experts(hidden, index, weights)
    assert experts.mode == "grouped"
    torch._dynamo.reset()
    # fullgraph on CUDA, where Zoo's grouped GEMM traces as a custom op; on CPU it is a compile-disabled loop.
    compiled = torch.compile(experts, backend = "eager", fullgraph = DEVICE == "cuda")(
        hidden, index, weights
    )
    torch.testing.assert_close(compiled.float(), eager.float(), atol = 2e-2, rtol = 2e-2)
    compiled.sum().backward()
    assert torch.isfinite(hidden.grad).all()
