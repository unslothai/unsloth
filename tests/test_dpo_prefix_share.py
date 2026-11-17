# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO prefix share: the shared-prefix stream (one prompt, both completions behind the
PrefixGrouper mask, hidden states gathered back to [2B, L]) equals the unshared padded forward.

CPU, fp64: a small causal transformer run on the padded rows and on the shared stream, with the
mask taken from the kernel's own mask_mod. Per-sequence logps and parameter grads must agree to
1e-10; each injected fault (leaky mask, shifted positions, wrong gather) must break that."""

import importlib.util
import math
import random
import sys
import types
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch.nn.attention.flex_attention")  # PrefixGrouper's kernel (torch >= 2.5)

_ROOT = Path(__file__).resolve().parents[1] / "unsloth"


def _load():
    pkg = "_dps_test_pkg"
    if pkg not in sys.modules:
        root = types.ModuleType(pkg)
        root.__path__ = [str(_ROOT)]
        sys.modules[pkg] = root
        for sub in ("utils", "models"):
            m = types.ModuleType(f"{pkg}.{sub}")
            m.__path__ = [str(_ROOT / sub)]
            sys.modules[f"{pkg}.{sub}"] = m
    out = []
    for sub, name in (
        ("utils", "prefix_grouper_kernel"),
        ("models", "dpo_prefix_share"),
    ):
        full = f"{pkg}.{sub}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, _ROOT / sub / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
        out.append(sys.modules[full])
    return out


pgk, dps = _load()

V, D, H, LAYERS = 64, 16, 2, 2
HD = D // H
PAD = 0


def _params(seed = 0):
    g = torch.Generator().manual_seed(seed)

    def p(*shape, scale = 0.3):
        return (torch.randn(*shape, generator = g, dtype = torch.float64) * scale).requires_grad_(True)

    params = {"emb": p(V, D, scale = 1.0), "out": p(D, V)}
    for i in range(LAYERS):
        for name in ("q", "k", "v", "o"):
            params[f"{name}{i}"] = p(D, D)
        params[f"up{i}"] = p(D, 2 * D)
        params[f"down{i}"] = p(2 * D, D)
    return params


def _rope(x, pos):
    # x [T, H, HD], pos [T]
    half = HD // 2
    freq = 1.0 / (10000 ** (torch.arange(half, dtype = torch.float64) / half))
    ang = pos.to(torch.float64)[:, None] * freq[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim = -1)


def _stack(params, ids, pos, mask):
    """ids [T], pos [T], mask [T, T] bool (q, kv) -> final hidden [T, D]."""
    h = params["emb"][ids]
    for i in range(LAYERS):
        T = h.shape[0]
        q = _rope((h @ params[f"q{i}"]).view(T, H, HD), pos)
        k = _rope((h @ params[f"k{i}"]).view(T, H, HD), pos)
        v = (h @ params[f"v{i}"]).view(T, H, HD)
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(HD)
        s = s.masked_fill(~mask[None], float("-inf"))
        a = torch.softmax(s, dim = -1).nan_to_num(0.0)
        h = h + torch.einsum("hqk,khd->qhd", a, v).reshape(T, D) @ params[f"o{i}"]
        h = h + torch.tanh(h @ params[f"up{i}"]) @ params[f"down{i}"]
    return h


def _unshared_hidden(params, input_ids, attention_mask):
    rows, L = input_ids.shape
    out = []
    for r in range(rows):
        keep = attention_mask[r].bool()
        causal = torch.ones(L, L, dtype = torch.bool).tril()
        mask = causal & keep[None, :] & keep[:, None]
        mask = mask | torch.eye(L, dtype = torch.bool)  # pad rows see themselves (ignored)
        out.append(_stack(params, input_ids[r], torch.arange(L), mask))
    return torch.stack(out)


def _dense_mask(seg, mask_mod = None):
    T = seg.T
    q = torch.arange(T)[:, None].expand(T, T)
    kv = torch.arange(T)[None, :].expand(T, T)
    mod = (mask_mod or pgk._make_mask_mod)(seg.group_of_kv, seg.is_prefix, seg.suffix_of_kv)
    return mod(0, 0, q, kv)


def _shared_hidden(
    params,
    layout,
    mask_mod = None,
    gather = None,
    positions = None,
):
    mask = _dense_mask(layout.prefix_seg_info, mask_mod)
    pos = layout.position_ids[0] if positions is None else positions
    h = _stack(params, layout.flat_ids[0], pos, mask)
    g = layout.gather if gather is None else gather
    return h.index_select(0, g).view(layout.rows, layout.L, D)


def _seq_logps(params, hidden, input_ids, completion_mask):
    logits = hidden[:, :-1] @ params["out"]
    logps = torch.log_softmax(logits, dim = -1)
    tgt = input_ids[:, 1:]
    tok = logps.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (tok * completion_mask[:, 1:]).sum(-1)


def _batch(
    prompts,
    chosens,
    rejecteds,
    extra_pad = 0,
):
    """TRL 1.x DataCollatorForPreference layout: [prompt + chosen ; prompt + rejected], right padded."""
    rows = [p + c for p, c in zip(prompts, chosens)] + [p + r for p, r in zip(prompts, rejecteds)]
    cmask = [[0] * len(p) + [1] * len(c) for p, c in zip(prompts, chosens)] + [
        [0] * len(p) + [1] * len(r) for p, r in zip(prompts, rejecteds)
    ]
    L = max(len(r) for r in rows) + extra_pad
    ids = torch.full((len(rows), L), PAD, dtype = torch.long)
    am = torch.zeros((len(rows), L), dtype = torch.long)
    cm = torch.zeros((len(rows), L), dtype = torch.long)
    for i, (r, m) in enumerate(zip(rows, cmask)):
        ids[i, : len(r)] = torch.tensor(r)
        am[i, : len(r)] = 1
        cm[i, : len(m)] = torch.tensor(m)
    return ids, am, cm


def _rand_tokens(rng, n):
    return [rng.randrange(1, V) for _ in range(n)]


def _cases():
    rng = random.Random(0)
    cases = []
    # Random batches, B = 1..4.
    for B in (1, 2, 3, 4):
        for _ in range(3):
            P = [rng.randint(1, 40) for _ in range(B)]
            prompts = [_rand_tokens(rng, p) for p in P]
            ch = [_rand_tokens(rng, rng.randint(1, 30)) for _ in range(B)]
            rj = [_rand_tokens(rng, rng.randint(1, 30)) for _ in range(B)]
            cases.append((f"rand_B{B}", prompts, ch, rj, rng.choice([0, 0, 5])))
    # Boundary cases.
    p = _rand_tokens(rng, 12)
    c = _rand_tokens(rng, 7)
    cases.append(("identical_rows", [p], [c], [list(c)], 0))
    cases.append(("one_token_prompt", [[5]], [_rand_tokens(rng, 6)], [_rand_tokens(rng, 3)], 0))
    cases.append(("one_token_completions", [p], [[7]], [[9]], 0))
    shared_head = _rand_tokens(rng, 4)
    cases.append(("completions_share_head", [p], [shared_head + [11, 12]], [shared_head + [13]], 0))
    cases.append(("chosen_prefix_of_rejected", [p], [[3, 4]], [[3, 4, 5, 6]], 2))
    cases.append(("rejected_prefix_of_chosen", [p], [[3, 4, 5, 6]], [[3, 4]], 0))
    cases.append(("same_first_token", [p], [[8, 1, 2]], [[8, 3]], 0))
    cases.append(("long_prompt_short_comp", [_rand_tokens(rng, 150)], [[1]], [[2, 3]], 0))
    cases.append(
        (
            "t_crosses_128",
            [_rand_tokens(rng, 100)],
            [_rand_tokens(rng, 20)],
            [_rand_tokens(rng, 9)],
            0,
        )
    )
    return cases


_CASES = _cases()


def _compare(ids, am, cm, layout, **fault):
    params = _params()
    ref = _seq_logps(params, _unshared_hidden(params, ids, am), ids, cm)
    w = torch.linspace(0.5, 1.5, ref.numel(), dtype = torch.float64)
    g_ref = torch.autograd.grad((ref * w).sum(), list(params.values()))
    got = _seq_logps(params, _shared_hidden(params, layout, **fault), ids, cm)
    g_got = torch.autograd.grad((got * w).sum(), list(params.values()))
    logp_err = float((ref - got).abs().max())
    grad_err = max(float((a - b).abs().max()) for a, b in zip(g_ref, g_got))
    return logp_err, grad_err


@pytest.mark.parametrize(
    "name,prompts,chosens,rejecteds,extra_pad", _CASES, ids = [c[0] for c in _CASES]
)
def test_shared_forward_matches_unshared(name, prompts, chosens, rejecteds, extra_pad):
    ids, am, cm = _batch(prompts, chosens, rejecteds, extra_pad)
    layout = dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 0)
    assert layout is not None, name
    B = len(prompts)
    # Structure: every real token of every row is read from a flat slot holding that token, at its column.
    flat = layout.flat_ids[0]
    pos = layout.position_ids[0]
    g = layout.gather.view(2 * B, -1)
    keep = am.bool()
    assert torch.equal(flat[g][keep], ids[keep])
    cols = torch.arange(ids.shape[1]).expand_as(ids)
    assert torch.equal(pos[g][keep], cols[keep])
    # T = sum(real) - sum(shared): each pair's prompt is stored once.
    assert layout.flat_ids.shape[1] == int(am.sum()) - sum(layout.shared)
    for i in range(B):
        assert (
            layout.shared[i] >= len(prompts[i])
            or layout.shared[i]
            == min(len(prompts[i]) + len(chosens[i]), len(prompts[i]) + len(rejecteds[i])) - 1
        )
    logp_err, grad_err = _compare(ids, am, cm, layout)
    assert logp_err < 1e-10, (name, logp_err)
    assert grad_err < 1e-10, (name, grad_err)


def _leaky_mask(group_of_kv, is_prefix, suffix_of_kv):
    # Completions see their sibling's tokens.
    def mask_mod(b, h, q_idx, kv_idx):
        return (group_of_kv[q_idx] == group_of_kv[kv_idx]) & (kv_idx <= q_idx)

    return mask_mod


def _noncausal_prefix_mask(group_of_kv, is_prefix, suffix_of_kv):
    good = pgk._make_mask_mod(group_of_kv, is_prefix, suffix_of_kv)

    def mask_mod(b, h, q_idx, kv_idx):
        # Prompt tokens see the whole prompt (future included).
        both_prefix = (
            is_prefix[q_idx] & is_prefix[kv_idx] & (group_of_kv[q_idx] == group_of_kv[kv_idx])
        )
        return good(b, h, q_idx, kv_idx) | both_prefix

    return mask_mod


def _shifted_positions(layout):
    pos = layout.position_ids[0].clone()
    pos[~layout.prefix_seg_info.is_prefix[: layout.prefix_seg_info.T]] += 1
    return pos


def _swapped_gather(layout):
    rows = layout.rows
    g = layout.gather.view(rows, -1).clone()
    B = rows // 2
    g[[0, B]] = g[[B, 0]]
    return g.view(-1)


@pytest.mark.parametrize(
    "fault",
    ["leaky_mask", "noncausal_prefix", "shifted_positions", "swapped_gather"],
)
def test_fault_injection_is_caught(fault):
    rng = random.Random(1)
    prompts = [_rand_tokens(rng, 9), _rand_tokens(rng, 5)]
    chosens = [_rand_tokens(rng, 6), _rand_tokens(rng, 4)]
    rejecteds = [_rand_tokens(rng, 5), _rand_tokens(rng, 7)]
    ids, am, cm = _batch(prompts, chosens, rejecteds)
    layout = dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 0)
    kwargs = {
        "leaky_mask": dict(mask_mod = _leaky_mask),
        "noncausal_prefix": dict(mask_mod = _noncausal_prefix_mask),
        "shifted_positions": dict(positions = _shifted_positions(layout)),
        "swapped_gather": dict(gather = _swapped_gather(layout)),
    }[fault]
    logp_err, _ = _compare(ids, am, cm, layout, **kwargs)
    assert logp_err > 1e-3, (fault, logp_err)


def test_declines():
    rng = random.Random(2)
    p, c, r = _rand_tokens(rng, 10), _rand_tokens(rng, 4), _rand_tokens(rng, 5)
    ids, am, _ = _batch([p], [c], [r])
    assert dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 0) is not None
    # Odd number of rows.
    assert dps.build_pair_layout(ids[:1], am[:1], min_tok_r = 1.0, min_saved = 0) is None
    # Left padding: real tokens do not start at column 0.
    left_ids = torch.cat([torch.zeros(2, 3, dtype = torch.long), ids], dim = 1)
    left_am = torch.cat([torch.zeros(2, 3, dtype = torch.long), am], dim = 1)
    assert dps.build_pair_layout(left_ids, left_am, min_tok_r = 1.0, min_saved = 0) is None
    # A hole in the attention mask.
    holed = am.clone()
    holed[0, 2] = 0
    assert dps.build_pair_layout(ids, holed, min_tok_r = 1.0, min_saved = 0) is None
    # keep_end truncation cut the rows differently: no common first token.
    diff = ids.clone()
    diff[1, 0] = (diff[0, 0] % (V - 1)) + 1
    assert dps.build_pair_layout(diff, am, min_tok_r = 1.0, min_saved = 0) is None
    # Not worth it: shared prompt of 1 token, long completions; or too few tokens saved.
    ids2, am2, _ = _batch([[3]], [_rand_tokens(rng, 40)], [_rand_tokens(rng, 40)])
    assert dps.build_pair_layout(ids2, am2, min_tok_r = 1.1, min_saved = 0) is None
    assert dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 11) is None
    assert dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 10) is not None
    # Sliding window shorter than a row.
    assert dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 0, max_segment_cap = 12) is None
    assert (
        dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 0, max_segment_cap = 15) is not None
    )
    # Float ids / mismatched shapes.
    assert dps.build_pair_layout(ids.float(), am, min_tok_r = 1.0, min_saved = 0) is None
    assert dps.build_pair_layout(ids, am[:, :-1], min_tok_r = 1.0, min_saved = 0) is None


def test_token_ratio():
    # P = 200, C = 300 -> 2(P + C) / (P + 2C) = 1000 / 800.
    rng = random.Random(3)
    ids, am, _ = _batch(
        [_rand_tokens(rng, 200)], [_rand_tokens(rng, 300)], [_rand_tokens(rng, 300)]
    )
    layout = dps.build_pair_layout(ids, am, min_tok_r = 1.0, min_saved = 0)
    # The completions may share leading tokens by chance; at most a few.
    assert abs(layout.tok_r - 1000 / 800) < 0.02
    assert layout.flat_ids.shape[1] <= 800


def test_default_gate(monkeypatch):
    # B = 4, C = 300: P = 200 saves 800 tokens (measured slower on A100), P = 512 saves 2048.
    rng = random.Random(4)
    for P, engaged in ((200, False), (512, True)):
        ids, am, _ = _batch(
            [_rand_tokens(rng, P) for _ in range(4)],
            [_rand_tokens(rng, 300) for _ in range(4)],
            [_rand_tokens(rng, 300) for _ in range(4)],
        )
        monkeypatch.delenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", raising = False)
        monkeypatch.delenv("UNSLOTH_DPO_PREFIX_SHARE_TOKR", raising = False)
        assert (dps.build_pair_layout(ids, am) is not None) == engaged, P


def test_mode_parsing(monkeypatch):
    for value, mode in (("0", "off"), ("off", "off"), ("1", "on"), ("auto", "auto"), ("", "auto")):
        monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE", value)
        assert dps.prefix_share_mode() == mode
    monkeypatch.delenv("UNSLOTH_DPO_PREFIX_SHARE")
    assert dps.prefix_share_mode() == "auto"


def test_wrap_preference_trainer_scopes_model_methods():
    calls = []

    class Base:
        def _compute_loss(self, model, inputs, return_outputs):
            calls.append(("loss", model))
            return 1

        def compute_ref_log_probs(self, model, inputs):
            return 2

        def concatenated_forward(
            self,
            model,
            batch,
            is_ref_model = False,
        ):
            return 3

    class OldDPO:
        def compute_ref_log_probs(self, batch):  # TRL < 1.0: no model argument
            return 4

    assert sorted(dps.wrap_preference_trainer(Base)) == [
        "_compute_loss",
        "compute_ref_log_probs",
        "concatenated_forward",
    ]
    assert dps.wrap_preference_trainer(OldDPO) == []
    # Wrapping twice is a no-op.
    first = Base._compute_loss
    dps.wrap_preference_trainer(Base)
    assert Base._compute_loss is first
    assert Base()._compute_loss("m", {}, False) == 1 and calls == [("loss", "m")]


def test_flex_compile_failure_reruns_the_forward_unshared():
    # The first shared call raises PrefixGrouperCompileError from inside the forward (flex failed its
    # first-call check, compile_check_failed() is now set); the wrapper must rerun it, not crash.
    calls = []

    class Trainer:
        def _compute_loss(
            self,
            model,
            inputs,
            return_outputs = False,
        ):
            calls.append(model)
            if len(calls) == 1:
                raise pgk.PrefixGrouperCompileError("flex does not compile")
            return 7

    class Other:
        def _compute_loss(self, model, inputs):
            raise ValueError("unrelated")

    dps.wrap_preference_trainer(Trainer)
    dps.wrap_preference_trainer(Other)
    assert Trainer()._compute_loss("m", {}) == 7 and calls == ["m", "m"]
    with pytest.raises(ValueError):
        Other()._compute_loss("m", {})


def test_neftune_declines_in_train_mode_only():
    # NEFTune noise is scaled by 1 / sqrt(seq_len * hidden): the flat stream would get less of it than
    # the padded rows, so a training forward with it on stays unshared; eval forwards still share.
    rng = random.Random(6)
    ids, am, _ = _batch([_rand_tokens(rng, 30)], [_rand_tokens(rng, 5)], [_rand_tokens(rng, 6)])

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(V, D)

    class CausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

        def get_input_embeddings(self):
            return self.model.embed_tokens

    lm = CausalLM()
    backbone = lm.model
    setattr(backbone, dps._STATE, dps._HookState(lm))
    kwargs = {"input_ids": ids, "attention_mask": am}

    def shared():
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("UNSLOTH_DPO_PREFIX_SHARE_TOKR", "1.0")
            mp.setenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "0")
            out = dps._pre_hook(backbone, (), dict(kwargs))
        getattr(backbone, dps._STATE).pending.clear()
        return out is not None

    lm.train()
    assert shared()
    lm.model.embed_tokens.neftune_noise_alpha = 5.0
    assert not shared()
    lm.eval()
    assert shared()
    lm.train()
    del lm.model.embed_tokens.neftune_noise_alpha
    assert shared()


def test_training_dropout_is_detected():
    """Any dropout module in train mode with p > 0 keeps the forward unshared; eval mode or p == 0 shares."""
    import torch
    from unsloth.models import dpo_prefix_share as dps

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.1))
    assert dps._dropout_active(model) is True
    model.eval()
    assert dps._dropout_active(model) is False
    off = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.0))
    assert dps._dropout_active(off) is False


def test_length_dependent_rope_declines():
    from types import SimpleNamespace as NS

    rope = dps._length_dependent_rope
    assert rope(NS(rope_scaling = {"type": "longrope", "short_factor": [1.0]}))
    assert rope(NS(rope_scaling = {"rope_type": "dynamic", "factor": 2.0}))
    assert rope(NS(rope_parameters = {"full_attention": {"rope_type": "longrope"}}))
    assert not rope(NS(rope_scaling = {"rope_type": "llama3", "factor": 8.0}))
    assert not rope(NS(rope_parameters = {"rope_type": "default", "rope_theta": 1e6}))
    assert not rope(NS(rope_scaling = None))
