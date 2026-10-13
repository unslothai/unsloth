# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cached block diffusion (SDAR, LLaDA2) matches the recompute sampler, the Dream port matches Dream's own loop,
and the grouped MoE eval dispatch matches the remote per-expert loop."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unsloth.models import diffusion_samplers as S
from unsloth.models.diffusion_infer_block import (
    _Frames,
    _llada2_graph,
    _PrefixCache,
    _select,
    _StaticPrefixCache,
    cached_block_sample,
)
from unsloth.models.diffusion_infer_dream import dream_fast_sample, dream_fast_supported

VOCAB, MASK_ID, EOS_ID, HIDDEN = 53, 52, 2, 16


def _prepare_4d_causal_attention_mask_for_sdpa(mask, shape, hidden, past):
    # The LLaDA2 runner looks this up in the model's module, as the remote code imports it.
    return mask


class _Layer(torch.nn.Module):
    """One attention layer over a block-causal mask, with either cache protocol: SDAR's `store_kv` or a
    LLaDA2-style `past_key_value.update`."""

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(HIDDEN, 3 * HIDDEN)
        self.out = torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(
        self,
        hidden,
        attention_mask = None,
        past_key_value = None,
        position_embeddings = None,
        store_kv = True,
        **unused,
    ):
        hidden = hidden + position_embeddings
        q, k, v = self.qkv(hidden)[:, None].chunk(3, dim = -1)
        if past_key_value is not None:
            if store_kv:
                k, v = past_key_value.update(k, v, 0)
            elif past_key_value.get_seq_length() > 0:
                layer = past_key_value.layers[0]
                k, v = torch.cat([layer.keys, k], dim = -2), torch.cat([layer.values, v], dim = -2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask)
        return (hidden + self.out(out[:, 0]),)


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(VOCAB, HIDDEN)
        self.positions = torch.nn.Embedding(256, HIDDEN)
        self.layers = torch.nn.ModuleList([_Layer()])
        self.norm = torch.nn.LayerNorm(HIDDEN)

    def rotary_emb(self, hidden, position_ids):
        return self.positions(position_ids)


class _ToyBlockModel(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Inner()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB)
        with torch.no_grad():
            self.lm_head.bias[MASK_ID] = -1e4
        self.config = SimpleNamespace(mask_token_id = MASK_ID, eos_token_id = EOS_ID)

    def forward(
        self,
        ids,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        store_kv = True,
        **unused,
    ):
        hidden = self.model.layers[0](
            self.model.word_embeddings(ids),
            attention_mask = attention_mask,
            past_key_value = past_key_values,
            position_embeddings = self.model.rotary_emb(None, position_ids),
            store_kv = store_kv,
        )[0]
        return SimpleNamespace(logits = self.lm_head(self.model.norm(hidden)))


@pytest.mark.parametrize(
    "family, block_length, steps", [("sdar", 4, 4), ("sdar", 8, 3), ("llada2", 8, 8)]
)
@pytest.mark.parametrize("prompt_length", [3, 8, 11, 16, 17])
def test_cached_block_sample_matches_recompute(family, block_length, steps, prompt_length):
    base = _ToyBlockModel(seed = prompt_length).eval()
    torch.manual_seed(1)
    ids = torch.randint(3, MASK_ID, (2, prompt_length))
    kwargs = dict(
        max_new_tokens = 13,
        steps = steps,
        block_length = block_length,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        confidence_threshold = 0.3,
    )
    with torch.no_grad():
        reference = S._block_sample(
            base,
            None,
            ids,
            None,
            mask_format = "additive" if family == "llada2" else "bool",
            **kwargs,
        )
    cached = cached_block_sample(base, ids, family = family, **kwargs)
    assert torch.equal(reference, cached)
    assert not bool((cached == MASK_ID).any())


def test_cached_block_sample_stops_after_eos_block():
    base = _ToyBlockModel(seed = 0).eval()
    with torch.no_grad():
        base.lm_head.bias[EOS_ID] = 50.0
    ids = torch.randint(3, MASK_ID, (1, 5))
    kwargs = dict(
        max_new_tokens = 30,
        steps = 4,
        block_length = 4,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [EOS_ID],
        confidence_threshold = 0.9,
    )
    with torch.no_grad():
        reference = S._block_sample(base, None, ids, None, mask_format = "bool", **kwargs)
    cached = cached_block_sample(base, ids, family = "sdar", **kwargs)
    assert torch.equal(reference, cached) and cached.shape[1] == 3


def test_cached_block_sample_frames_and_cancel():
    base = _ToyBlockModel(seed = 0).eval()
    ids = torch.randint(3, MASK_ID, (1, 5))
    frames = []
    out = cached_block_sample(
        base,
        ids,
        family = "sdar",
        max_new_tokens = 11,
        steps = 4,
        block_length = 4,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        confidence_threshold = 0.9,
        on_frame = frames.append,
    )
    ends = [f for f in frames if f["step"] == f["total"]]
    # Exactly one block-end frame per decoded block; in-block frames stay below total.
    assert [f["block"] for f in ends] == [1, 2, 3]
    assert all(0 <= f["step"] < f["total"] for f in frames if f not in ends)
    assert len(ends) == 3 and ends[-1]["tokens"] == out[0].tolist()
    assert all(set(f) == {"block", "step", "total", "tokens", "mask_id"} for f in frames)

    def cancel():
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        cached_block_sample(
            base,
            ids,
            family = "sdar",
            max_new_tokens = 11,
            steps = 4,
            block_length = 4,
            temperature = 0.0,
            mask_id = MASK_ID,
            eos_ids = [],
            confidence_threshold = 0.9,
            check_stop = cancel,
        )


def _reference_select(logits, mask_index, count, threshold):
    """The recompute sampler's per-row selection (greedy)."""
    probs = F.softmax(logits, dim = -1)
    x0 = probs.argmax(dim = -1)
    confidence = torch.where(mask_index, probs.gather(-1, x0[..., None])[..., 0], -math.inf)
    transfer = torch.zeros_like(mask_index)
    for row in range(logits.shape[0]):
        high = confidence[row] > threshold
        if int(high.sum()) >= count:
            transfer[row] = high
        else:
            k = min(count, int(mask_index[row].sum()))
            transfer[row, torch.topk(confidence[row], k).indices] = True
    return x0, transfer


@pytest.mark.parametrize("count", [1, 2, 5])
@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.99])
def test_select_matches_per_row_reference(count, threshold):
    torch.manual_seed(count)
    logits = torch.randn(3, 6, 11) * 3
    mask_index = torch.rand(3, 6) < 0.5
    mask_index[0] = True
    mask_index[1, :] = False
    mask_index[1, 2] = True
    x0, transfer = _select(logits, mask_index, 0.0, 0, 1.0, torch.full((3, 1), count), threshold)
    ref_x0, ref_transfer = _reference_select(logits, mask_index, count, threshold)
    assert torch.equal(x0, ref_x0)
    assert torch.equal(transfer, ref_transfer)


def test_prefix_cache_reads_without_writing():
    cache = _PrefixCache()
    k = torch.ones(1, 1, 2, 4)
    cache.commit = True
    cache.update(k, k, 0)
    cache.commit = False
    key, _ = cache.update(2 * k, 2 * k, 0)
    assert key.shape[-2] == 4 and cache.get_seq_length() == 2


def test_static_prefix_cache_writes_rows():
    cache = _StaticPrefixCache(8)
    cache.rows = torch.tensor([2, 3])
    key, value = cache.update(torch.ones(1, 1, 2, 4), 2 * torch.ones(1, 1, 2, 4), 0)
    assert key.shape == (1, 1, 8, 4)
    assert key[0, 0, :, 0].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert value[0, 0, 3, 0] == 2


def test_llada2_graph_only_on_cuda_without_adapters(monkeypatch):
    base = _ToyBlockModel(seed = 0)
    assert _llada2_graph(base, 1, 4, 16, torch.float32, torch.device("cpu")) is None
    monkeypatch.setenv("UNSLOTH_DIFFUSION_CUDA_GRAPHS", "0")
    assert _llada2_graph(base, 1, 4, 16, torch.float32, torch.device("cuda")) is None


@pytest.mark.parametrize(
    "compile_mode, expected", [("graphs", True), (None, False), ("default", False)]
)
def test_llada2_graph_follows_generate_text_compile(monkeypatch, compile_mode, expected):
    from unsloth.models import diffusion_infer_block, diffusion_samplers, remote_moe_shims

    seen = {}
    monkeypatch.setattr(remote_moe_shims, "prepare_remote_moe_for_inference", lambda base: None)
    monkeypatch.setattr(
        diffusion_infer_block, "cached_block_sample", lambda base, ids, **kw: seen.update(kw)
    )
    diffusion_samplers._llada2_sample(
        None,
        None,
        torch.zeros(1, 3, dtype = torch.long),
        None,
        steps = None,
        block_length = None,
        max_new_tokens = 4,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        compile_mode = compile_mode,
    )
    assert seen["cuda_graphs"] is expected and "compile_mode" not in seen


def test_frames_throttle_and_stop():
    seen, stops = [], []
    frames = _Frames(seen.append, lambda: stops.append(1), interval = 3600)
    x = torch.arange(10)[None]
    for step in range(5):
        frames.step(x, 4, 8, 0, step, 5, MASK_ID)
    frames.step(x, 4, 8, 0, 5, 5, MASK_ID, force = True)
    assert len(stops) == 6
    assert [frame["step"] for frame in seen] == [0, 5]
    assert seen[-1]["tokens"] == [4, 5, 6, 7]

    def cancel():
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        _Frames(None, cancel).step(x, 4, 8, 0, 0, 5, MASK_ID)


def test_cached_block_sample_frames_through_generate_text_progress():
    # generate_text hands samplers its Progress (not on_frame): frames must still reach the caller.
    from unsloth.models.diffusion_infer import make_progress

    base = _ToyBlockModel(seed = 0).eval()
    ids = torch.randint(3, MASK_ID, (1, 5))
    frames = []
    progress = make_progress(frames.append, None, 0.0, MASK_ID, 11)
    out = cached_block_sample(
        base,
        ids,
        family = "sdar",
        max_new_tokens = 11,
        steps = 4,
        block_length = 4,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        confidence_threshold = 0.9,
        progress = progress,
    )
    progress.close()
    ends = [f for f in frames if f["step"] == f["total"]]
    assert len(ends) == 3 and ends[-1]["tokens"] == out[0].tolist()[:11]
    assert all(0 <= f["step"] < f["total"] for f in frames if f not in ends)
    assert all(set(f) == {"block", "step", "total", "tokens", "mask_id"} for f in frames)


class _ToyDream(torch.nn.Module):
    """Bidirectional toy with Dream's call shape: model(input_ids, attention_mask) -> (hidden,), lm_head."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.mix = torch.nn.Linear(HIDDEN, HIDDEN)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB)
        with torch.no_grad():
            self.lm_head.bias[MASK_ID] = -1e4
        toy = self

        class _Model(torch.nn.Module):
            def forward(
                self,
                input_ids = None,
                attention_mask = None,
            ):
                h = toy.embed(input_ids)
                return (h + torch.tanh(toy.mix(h.mean(1, keepdim = True))),)

        self.model = _Model()

    def forward(
        self,
        x,
        attention_mask = None,
        tok_idx = None,
    ):
        return SimpleNamespace(logits = self.lm_head(self.model(x, attention_mask)[0]))


def _dream_reference(
    model,
    input_ids,
    max_new_tokens,
    steps,
    alg,
    eps = 1e-3,
):
    """Dream generation_utils `_sample` at temperature 0, alg_temp 0 (official loop, full logits)."""
    x = F.pad(input_ids, (0, max_new_tokens), value = MASK_ID)
    timesteps = torch.linspace(1, eps, steps + 1)
    for i in range(steps):
        mask_index = x == MASK_ID
        logits = model(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim = 1)
        mask_logits = logits[mask_index]
        t, s = timesteps[i], timesteps[i + 1]
        probs = torch.softmax(mask_logits, dim = -1)
        confidence, x0 = probs.max(dim = -1)
        if alg == "topk_margin":
            top2 = torch.topk(probs, 2, dim = -1).values
            confidence = top2[..., 0] - top2[..., 1]
        elif alg == "entropy":
            confidence = torch.sum(probs * torch.log(probs + 1e-10), dim = -1)
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        number_transfer_tokens = (
            int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        )
        full_confidence = torch.full_like(x, -torch.inf, dtype = logits.dtype)
        full_confidence[mask_index] = confidence
        if number_transfer_tokens > 0:
            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
            x_ = torch.zeros_like(x) + MASK_ID
            x_[mask_index] = x0.clone()
            row_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(transfer_index)
            x[row_indices, transfer_index] = x_[row_indices, transfer_index]
    return x


@pytest.mark.parametrize("alg", ["entropy", "maskgit_plus", "topk_margin"])
@pytest.mark.parametrize("max_new_tokens, steps", [(12, 12), (12, 5), (7, 3)])
def test_dream_fast_sample_matches_reference(alg, max_new_tokens, steps):
    model = _ToyDream().eval()
    ids = torch.randint(3, MASK_ID, (2, 5))
    with torch.no_grad():
        reference = _dream_reference(model, ids.clone(), max_new_tokens, steps, alg)
    fast = dream_fast_sample(
        model, ids, max_new_tokens = max_new_tokens, steps = steps, mask_id = MASK_ID, alg = alg
    )
    assert torch.equal(reference, fast)


def test_dream_fast_sample_frames_contract():
    model = _ToyDream().eval()
    frames = []
    dream_fast_sample(
        model,
        torch.randint(3, MASK_ID, (1, 5)),
        max_new_tokens = 6,
        steps = 3,
        mask_id = MASK_ID,
        on_frame = frames.append,
    )
    ends = [f for f in frames if f["step"] == f["total"]]
    assert len(ends) == 1 and frames[-1] is ends[0]
    assert all(0 <= f["step"] < f["total"] for f in frames[:-1])


def test_dream_fast_sample_defers_when_the_mask_is_proposed():
    model = _ToyDream().eval()
    with torch.no_grad():
        model.lm_head.bias[MASK_ID] = 1e4
    ids = torch.randint(3, MASK_ID, (1, 5))
    assert dream_fast_sample(model, ids, max_new_tokens = 6, steps = 3, mask_id = MASK_ID) is None


def test_dream_fast_path_only_for_greedy_defaults():
    ids = torch.tensor([[5, 6]])
    assert dream_fast_supported(ids, None, MASK_ID, 0.0, None, None, "entropy", 0.0)
    assert not dream_fast_supported(ids, None, MASK_ID, 0.7, 0.95, None, "entropy", 0.0)
    assert not dream_fast_supported(ids, None, MASK_ID, 0.0, None, None, "origin", 0.0)
    assert not dream_fast_supported(ids, None, MASK_ID, 0.0, None, None, "entropy", 0.5)
    assert not dream_fast_supported(
        ids, torch.tensor([[0, 1]]), MASK_ID, 0.0, None, None, "entropy", 0.0
    )
    assert not dream_fast_supported(
        torch.tensor([[MASK_ID, 6]]), None, MASK_ID, 0.0, None, None, "entropy", 0.0
    )


def test_dream_fast_path_rejects_an_empty_prompt():
    empty = torch.zeros((1, 0), dtype = torch.long)
    assert not dream_fast_supported(empty, None, MASK_ID, 0.0, None, None, "entropy", 0.0)


def test_dream_fallback_frames_never_step_back():
    # The fast pass proposes the mask token and defers to diffusion_generate after streaming its own frames.
    from unsloth.models.diffusion_infer import make_progress
    from unsloth.models.diffusion_samplers import _dream_sample

    model = _ToyDream().eval()
    with torch.no_grad():
        model.lm_head.bias[MASK_ID] = 1e4

    def diffusion_generate(
        input_ids, *, max_new_tokens, steps, generation_tokens_hook_func, **kwargs
    ):
        x = generation_tokens_hook_func(
            None, F.pad(input_ids, (0, max_new_tokens), value = MASK_ID), None
        )
        for i in range(steps):
            x[:, input_ids.shape[1] + i] = 7
            x = generation_tokens_hook_func(i, x, None)
        return SimpleNamespace(sequences = x)

    model.diffusion_generate = diffusion_generate
    frames = []
    progress = make_progress(frames.append, None, 0.0, MASK_ID, 4)
    out = _dream_sample(
        model,
        None,
        torch.randint(3, MASK_ID, (1, 5)),
        None,
        max_new_tokens = 4,
        steps = 4,
        block_length = None,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        progress = progress,
        check_stop = progress.check,
    )
    progress.close()
    steps = [f["step"] for f in frames]
    assert steps == sorted(steps) and steps.count(4) == 1 and steps[-1] == 4
    assert frames[-1]["tokens"] == out[0].tolist() == [7, 7, 7, 7]


def test_llada2_graph_recaptures_after_weights_move(monkeypatch):
    from unsloth.models import diffusion_infer_block

    built = []
    monkeypatch.setattr(
        diffusion_infer_block, "_Llada2Graph", lambda *args: built.append(args) or object()
    )
    base = _ToyBlockModel(seed = 0)
    device = torch.device("cuda")
    first = _llada2_graph(base, 1, 4, 16, torch.float32, device)
    assert _llada2_graph(base, 1, 4, 16, torch.float32, device) is first and len(built) == 1
    # model.to(...) and back re-points every parameter: the old graph would replay freed memory.
    with torch.no_grad():
        base.lm_head.weight.data = base.lm_head.weight.data.clone()
    assert _llada2_graph(base, 1, 4, 16, torch.float32, device) is not first and len(built) == 2


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(8, 4, bias = False, dtype = torch.bfloat16)
        self.up_proj = torch.nn.Linear(8, 4, bias = False, dtype = torch.bfloat16)
        self.down_proj = torch.nn.Linear(4, 8, bias = False, dtype = torch.bfloat16)
        self.act_fn = torch.nn.SiLU()


class _RemoteMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([_Expert() for _ in range(3)])

    def moe_infer(self, x, topk_ids, topk_weight):
        counts = topk_ids.bincount().cpu()
        return self.experts[0](x) * counts[0]


_RemoteMoE.__module__ = "transformers_modules.toy.modeling"


def _stack(monkeypatch, block):
    from unsloth.models import remote_moe_shims
    monkeypatch.setattr(remote_moe_shims, "_grouped_mm_for", lambda weights: "grouped_mm")
    return remote_moe_shims._grouped_experts(block)


def test_grouped_experts_keep_parameters_and_alias_the_stack(monkeypatch):
    block = _RemoteMoE()
    params = [p for p in block.parameters()]
    values = [p.detach().clone() for p in params]
    gate, up, down, _ = _stack(monkeypatch, block)
    # Same Parameter objects (an optimizer made before generate keeps training them), now views of the stack.
    assert [p for p in block.parameters()] == params
    assert all(torch.equal(p, v) for p, v in zip(params, values))
    assert block.experts[2].down_proj.weight.data_ptr() == down[2].data_ptr()


def test_prepare_drops_stale_grouped_experts(monkeypatch):
    from unsloth.models import remote_moe_shims

    block = _RemoteMoE()
    model = torch.nn.ModuleDict({"mlp": block})
    remote_moe_shims.prepare_remote_moe_for_inference(model)
    assert _stack(monkeypatch, block) is not None
    remote_moe_shims.prepare_remote_moe_for_inference(model)
    assert block.__dict__["_unsloth_grouped_experts"] is not None

    # model.to(...) and back: the cached stack no longer backs the weights, so it is not reused or kept alive.
    with torch.no_grad():
        for expert in block.experts:
            expert.up_proj.weight.data = expert.up_proj.weight.data.clone()
    remote_moe_shims.prepare_remote_moe_for_inference(model)
    assert "_unsloth_grouped_experts" not in block.__dict__
    assert _stack(monkeypatch, block) is not None

    # A LoRA wrap added after a generate: the grouped path would skip the adapters.
    class _Wrapped(torch.nn.Linear):
        pass

    for expert in block.experts:
        wrapped = _Wrapped(8, 4, bias = False, dtype = torch.bfloat16)
        wrapped.weight = expert.gate_proj.weight
        expert.gate_proj = wrapped
    assert remote_moe_shims._grouped_experts(block) is None


def test_grouped_mm_for_refuses_cpu_weights():
    from unsloth.models import remote_moe_shims
    assert remote_moe_shims._grouped_mm_for((torch.zeros(4, 8, dtype = torch.bfloat16),)) is None


@pytest.mark.parametrize(
    "capability, hip, choice, expected",
    [
        ((9, 0), None, "auto", ["torch", "triton"]),
        ((10, 0), None, "auto", ["torch", "triton"]),
        # A100 / L4 / RTX: torch._grouped_mm loops over host-copied offsets there (or raises on 2.8).
        ((8, 0), None, "auto", ["triton"]),
        ((8, 9), None, "auto", ["triton"]),
        ((12, 0), None, "auto", ["triton"]),
        ((7, 5), None, "auto", []),
        ((8, 0), None, "torch", ["torch"]),
        ((9, 0), None, "0", []),
        ((9, 4), "6.4", "auto", ["torch"]),
    ],
)
def test_grouped_mm_candidates_by_gpu(monkeypatch, capability, hip, choice, expected):
    from unsloth.models import remote_moe_shims

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: capability)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setenv("UNSLOTH_MOE_INFER_GROUPED", choice)
    names = {
        remote_moe_shims._torch_grouped_mm: "torch",
        remote_moe_shims._triton_grouped_mm: "triton",
    }
    found = remote_moe_shims._grouped_mm_candidates(torch.zeros(1))
    assert [names[f] for f in found] == expected


def test_grouped_mm_for_logs_once_and_falls_back(monkeypatch, capsys):
    from unsloth.models import remote_moe_shims

    def broken(inputs, weight, offsets):
        raise RuntimeError("grouped gemm is only supported on SM90")

    weight = torch.ones(4, 8, dtype = torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(remote_moe_shims, "_grouped_mm_candidates", lambda weight: [broken])
    monkeypatch.setattr(remote_moe_shims, "_FALLBACK_LOGGED", set())
    assert remote_moe_shims._grouped_mm_for((weight,)) is None
    assert remote_moe_shims._grouped_mm_for((weight,)) is None
    assert capsys.readouterr().out.count("keeps the per-expert loop") == 1

    def matmul(inputs, weight, offsets):
        return torch.cat([inputs[:1] @ weight[0], inputs[1:] @ weight[1]])

    monkeypatch.setattr(remote_moe_shims, "_grouped_mm_candidates", lambda weight: [broken, matmul])
    assert remote_moe_shims._grouped_mm_for((weight,)) is matmul


def test_progress_frames_carry_the_whole_generated_region():
    # Later blocks are still masks in every frame, so the canvas length never changes mid-reply.
    from unsloth.models.diffusion_infer import make_progress

    base = _ToyBlockModel(seed = 0).eval()
    frames = []
    progress = make_progress(frames.append, None, 0.0, MASK_ID, 11)
    cached_block_sample(
        base,
        torch.randint(3, MASK_ID, (1, 5)),
        family = "sdar",
        max_new_tokens = 11,
        steps = 4,
        block_length = 4,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        confidence_threshold = 0.9,
        progress = progress,
    )
    progress.close()
    assert {len(f["tokens"]) for f in frames} == {11}
    assert frames[0]["tokens"][-1] == MASK_ID


def test_llada2_graph_cache_is_freed_once_adapters_appear():
    from unsloth.models import diffusion_infer_block

    base = _ToyBlockModel(seed = 0)
    base.__dict__["_unsloth_llada2_graphs"] = {"weights": (), (1, 4, 1024): object()}
    base.model.lora_A = torch.nn.Linear(1, 1)
    assert (
        diffusion_infer_block._llada2_graph(base, 1, 4, 16, torch.float32, torch.device("cuda"))
        is None
    )
    assert "_unsloth_llada2_graphs" not in base.__dict__


@pytest.mark.parametrize("error, latched", [(RuntimeError("host sync"), True), ("oom", False)])
def test_llada2_graph_latches_only_real_capture_failures(monkeypatch, error, latched):
    from unsloth.models import diffusion_infer_block

    if error == "oom":
        error = torch.cuda.OutOfMemoryError("CUDA out of memory")

    def capture(*args, **kwargs):
        raise error

    monkeypatch.setattr(diffusion_infer_block, "_Llada2Graph", capture)
    base = _ToyBlockModel(seed = 0)
    assert (
        diffusion_infer_block._llada2_graph(base, 1, 4, 16, torch.float32, torch.device("cuda"))
        is None
    )
    assert bool(base.__dict__["_unsloth_llada2_graphs"].get("failed")) is latched


def test_llada2_graph_capture_lets_a_stop_through(monkeypatch):
    from unsloth.models import diffusion_infer_block
    from unsloth.models.diffusion_infer import DiffusionGenerationCancelled

    def capture(*args, **kwargs):
        raise DiffusionGenerationCancelled("stop")

    monkeypatch.setattr(diffusion_infer_block, "_Llada2Graph", capture)
    base = _ToyBlockModel(seed = 0)
    with pytest.raises(DiffusionGenerationCancelled):
        diffusion_infer_block._llada2_graph(base, 1, 4, 16, torch.float32, torch.device("cuda"))
    assert not base.__dict__["_unsloth_llada2_graphs"].get("failed")


def test_llada2_graph_is_not_attempted_over_a_host_synced_expert_loop(monkeypatch):
    from unsloth.models import diffusion_infer_block

    monkeypatch.setattr(
        diffusion_infer_block, "_Llada2Graph", lambda *a, **k: pytest.fail("capture attempted")
    )
    base = _ToyBlockModel(seed = 0)
    base.model.moe = torch.nn.Module()
    base.model.moe.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    assert (
        diffusion_infer_block._llada2_graph(base, 1, 4, 16, torch.float32, torch.device("cuda"))
        is None
    )


@pytest.mark.parametrize(
    "config, fast, eps",
    [
        (SimpleNamespace(eps = 0.05, top_k = None), True, 0.05),
        (SimpleNamespace(eps = 1e-3, top_k = 5), False, None),
        (None, True, 1e-3),
    ],
)
def test_dream_fast_path_follows_the_generation_config(monkeypatch, config, fast, eps):
    from unsloth.models import diffusion_infer_dream

    seen = {}

    def fast_sample(base, input_ids, **kwargs):
        seen.update(kwargs)
        return torch.cat([input_ids, torch.full((1, kwargs["max_new_tokens"]), 7)], dim = 1)

    monkeypatch.setattr(diffusion_infer_dream, "dream_fast_sample", fast_sample)
    model = _ToyDream().eval()
    if config is not None:
        model.generation_config = config
    model.diffusion_generate = lambda input_ids, **kw: SimpleNamespace(
        sequences = torch.cat([input_ids, torch.full((1, kw["max_new_tokens"]), 8)], dim = 1)
    )
    out = S._dream_sample(
        model,
        None,
        torch.randint(3, MASK_ID, (1, 5)),
        None,
        max_new_tokens = 4,
        steps = 4,
        block_length = None,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
    )
    assert out[0].tolist() == ([7] * 4 if fast else [8] * 4)
    assert seen.get("eps") == eps
