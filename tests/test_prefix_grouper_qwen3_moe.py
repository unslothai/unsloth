# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO PrefixGrouper on transformers-native Qwen3-MoE: per-completion log-probs and grads match
unshared rows. CPU flex has no backward, so the grad test uses a dense-mask SDPA reference."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "unsloth" / "utils"
PKG = "_pg_qwen3_moe_test_pkg"


def _load(pkg_name, utils_dir, names):
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(utils_dir)]
        sys.modules[pkg_name] = pkg
    out = []
    for name in names:
        full = f"{pkg_name}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, Path(utils_dir) / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
        out.append(sys.modules[full])
    return out


pgk, pg, pghf = _load(PKG, UTILS, ("prefix_grouper_kernel", "prefix_grouper", "prefix_grouper_hf"))
for _m in (pgk, pg, pghf):
    assert Path(_m.__file__).resolve().parent == UTILS.resolve(), _m.__file__

transformers = pytest.importorskip("transformers")
try:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
except Exception:  # pragma: no cover
    pytest.skip("transformers without Qwen3-MoE", allow_module_level = True)

if not pgk.TORCH_SUPPORTS_PREFIX_GROUPER:
    pytest.skip("flex attention PrefixGrouper needs torch >= 2.7", allow_module_level = True)

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
DTYPE = torch.float32
if not CUDA:
    pgk._COMPILE_CHECK_ON = False
TOL = 1e-4
GRAD_TOL = 1e-4
VOCAB = 128


_FLEX = pgk.flex_shared_prefix_attention


def _eager_flex(*args, **kwargs):
    kwargs["compiled"] = False
    return _FLEX(*args, **kwargs)


@pytest.fixture(autouse = True)
def _moe_env(monkeypatch):
    monkeypatch.setenv(pghf.MOE_ENV, "1")
    monkeypatch.delenv("UNSLOTH_GRPO_ROUTING_REPLAY", raising = False)
    if not CUDA:
        # Same mask and kernel, plus the zoo-patched router, run eager: CPU Inductor builds cost ~20 s.
        monkeypatch.setattr(pgk, "flex_shared_prefix_attention", _eager_flex)
        with torch.compiler.set_stance("force_eager"):
            yield
        return
    yield


def _config(**overrides):
    kwargs = dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_experts = 4,
        num_experts_per_tok = 2,
        norm_topk_prob = True,
        decoder_sparse_step = 1,
        mlp_only_layers = [],
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,  # GQA
        head_dim = 16,
        max_position_embeddings = 64,
        attn_implementation = "sdpa",
    )
    kwargs.update(overrides)
    config = Qwen3MoeConfig(**kwargs)
    config._attn_implementation = kwargs["attn_implementation"]
    return config


def _model(config, seed = 0):
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config).eval().to(DEVICE, DTYPE)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std = 0.3)
    return model


def _groups(seed = 1):
    g = torch.Generator().manual_seed(seed)
    spec = [(10, [7, 12, 5]), (14, [9, 3, 11])]
    out = []
    for P, Rs in spec:
        prompt = torch.randint(1, VOCAB, (P,), generator = g)
        comps = [torch.randint(1, VOCAB, (r,), generator = g) for r in Rs]
        out.append((prompt, comps))
    return out


def _row_logps(model, prompt, comp):
    ids = torch.cat([prompt, comp]).unsqueeze(0).to(DEVICE)
    logits = model(input_ids = ids, use_cache = False).logits.float()
    P = prompt.numel()
    lp = torch.log_softmax(logits[0, P - 1 : -1], dim = -1)
    return lp.gather(-1, comp.to(DEVICE).unsqueeze(-1)).squeeze(-1)


def _stream(groups):
    specs = [(p.numel(), [c.numel() for c in cs]) for p, cs in groups]
    seg, meta = pgk.build_seg_info_multigroup(specs, DEVICE)
    flat, pos = [], []
    for p, cs in groups:
        P = p.numel()
        flat.append(p)
        pos.append(torch.arange(P))
        for c in cs:
            flat.append(c)
            pos.append(torch.arange(P, P + c.numel()))
    ids = torch.cat(flat).unsqueeze(0).to(DEVICE)
    position_ids = torch.cat(pos).unsqueeze(0).to(DEVICE)
    return seg, meta, ids, position_ids


def _pg_logps(model, groups):
    """Flat [P_g, c_g0, c_g1, ...] stream; per-completion logps read back from it."""
    seg, meta, ids, position_ids = _stream(groups)
    logits = model(
        input_ids = ids, position_ids = position_ids, prefix_seg_info = seg, use_cache = False
    ).logits.float()
    lp_all = torch.log_softmax(logits[0], dim = -1)
    out = []
    for (p, cs), gm in zip(groups, meta):
        rows = []
        for c, (s, e) in zip(cs, gm["suffix_slices"]):
            pred = torch.tensor([gm["prefix_last_index"]] + list(range(s, e - 1)), device = DEVICE)
            rows.append(lp_all[pred].gather(-1, c.to(DEVICE).unsqueeze(-1)).squeeze(-1))
        out.append(rows)
    return out


def _max_diff(model, groups):
    with torch.no_grad():
        pgl = _pg_logps(model, groups)
        worst = 0.0
        for (p, cs), rows in zip(groups, pgl):
            for c, r in zip(cs, rows):
                worst = max(worst, float((r - _row_logps(model, p, c)).abs().max()))
    return worst


def _dense_reference_kernel(groups):
    """Dense-mask SDPA stand-in for flex_shared_prefix_attention, mask from the specs only."""
    labels = []  # (group, suffix or -1 for prefix)
    for gi, (p, cs) in enumerate(groups):
        labels += [(gi, -1)] * p.numel()
        for si, c in enumerate(cs):
            labels += [(gi, si)] * c.numel()
    T = len(labels)
    allow = torch.zeros(T, T, dtype = torch.bool)
    for qi, (gq, sq) in enumerate(labels):
        for ki in range(qi + 1):
            gk, sk = labels[ki]
            allow[qi, ki] = gq == gk and (sk == -1 or sk == sq)
    calls = []

    def kernel(
        Q,
        K,
        V,
        seg,
        scale = None,
        block_mask = None,
        compiled = True,
    ):
        calls.append(1)
        q, k, v = (x.transpose(1, 2) for x in (Q, K, V))
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask = allow.to(q.device),
            scale = scale,
            enable_gqa = q.shape[1] != k.shape[1],
        )
        return out.transpose(1, 2).contiguous()

    return kernel, calls


def test_qwen3_moe_is_covered_marked_and_matches_per_row():
    model = _model(_config())
    assert pg.prefix_seg_info_applied(model) is True
    assert getattr(model, pghf.MOE_OK_ATTR, False) is True
    worst = _max_diff(model, _groups())
    print(f"[pg-moe] qwen3_moe {DEVICE.type} fp32 max |PG - per-row| = {worst:.3e} nats")
    assert worst <= TOL


def test_eager_attention_matches_too():
    model = _model(_config(attn_implementation = "eager"))
    assert pg.prefix_seg_info_applied(model) is True
    worst = _max_diff(model, _groups(seed = 3))
    print(f"[pg-moe] eager max |PG - per-row| = {worst:.3e} nats")
    assert worst <= TOL


def test_no_leak_across_completions():
    """Changing completion 1 must not move completion 0 or 2 (same group) or the other group."""
    model = _model(_config())
    assert pg.prefix_seg_info_applied(model) is True
    groups = _groups(seed = 5)
    with torch.no_grad():
        base = _pg_logps(model, groups)
        p, cs = groups[0]
        cs2 = list(cs)
        cs2[1] = (cs[1] + 17) % (VOCAB - 1) + 1
        moved = _pg_logps(model, [(p, cs2)] + groups[1:])
    # Rounding noise only (CUDA expert GEMM shapes, CPU MoE batch variance ~1e-6); a leak is whole nats.
    atol = 1e-4 if CUDA else 1e-5
    for idx in (0, 2):
        assert torch.allclose(base[0][idx], moved[0][idx], rtol = 0, atol = atol)
    for a, b in zip(base[1], moved[1]):
        assert torch.allclose(a, b, rtol = 0, atol = atol)
    assert not torch.allclose(base[0][1], moved[0][1], rtol = 0, atol = 1e-2)


def _grads(model, fn):
    model.zero_grad(set_to_none = True)
    fn().backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_grads_match_per_row(monkeypatch):
    """Weighted sum of completion logps: param grads (router + experts too) equal the unshared rows."""
    model = _model(_config())
    model.train()
    assert pg.prefix_seg_info_applied(model) is True
    groups = _groups(seed = 7)
    calls = None
    if not CUDA:
        kernel, calls = _dense_reference_kernel(groups)
        monkeypatch.setattr(pgk, "flex_shared_prefix_attention", kernel)
    g = torch.Generator().manual_seed(9)
    weights = [[torch.randn(c.numel(), generator = g).to(DEVICE) for c in cs] for _, cs in groups]

    def pg_loss():
        rows = _pg_logps(model, groups)
        return sum((r * w).sum() for rs, ws in zip(rows, weights) for r, w in zip(rs, ws))

    def row_loss():
        return sum(
            (_row_logps(model, p, c) * w).sum()
            for (p, cs), ws in zip(groups, weights)
            for c, w in zip(cs, ws)
        )

    g_pg = _grads(model, pg_loss)
    g_row = _grads(model, row_loss)
    assert set(g_pg) == set(g_row)
    assert any("mlp.gate" in n for n in g_pg) and any("experts" in n for n in g_pg)
    worst = max(float((g_pg[n] - g_row[n]).norm() / (g_row[n].norm() + 1e-12)) for n in g_row)
    if calls is not None:
        assert len(calls) == model.config.num_hidden_layers
    print(f"[pg-moe] {DEVICE.type} grad max rel-L2 PG vs per-row = {worst:.3e}")
    assert worst <= GRAD_TOL


def test_moe_needs_opt_in(monkeypatch):
    monkeypatch.setenv(pghf.MOE_ENV, "0")
    model = _model(_config())
    assert pg.prefix_seg_info_applied(model) is False
    assert not getattr(model, pghf.MOE_OK_ATTR, False)


def test_routing_replay_keeps_it_off(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_ROUTING_REPLAY", "1")
    model = _model(_config())
    assert pg.prefix_seg_info_applied(model) is False
    assert not getattr(model, pghf.MOE_OK_ATTR, False)


def test_router_logits_in_loss_keeps_it_off():
    model = _model(_config(output_router_logits = True))
    assert pg.prefix_seg_info_applied(model) is False


def test_disabled_after_wrap_raises_instead_of_leaking(monkeypatch):
    """A wrapped class used by a model that no longer qualifies raises (caller falls back)."""
    model = _model(_config())
    assert pg.prefix_seg_info_applied(model) is True
    monkeypatch.setenv(pghf.MOE_ENV, "0")
    with pytest.raises(RuntimeError, match = "does not support prefix_seg_info"):
        _max_diff(model, _groups())


def test_plain_forward_unchanged_by_wrap():
    """Without a seg table the wrapped forward is the original one, bit for bit."""
    config = _config()
    fresh = _model(config)
    ids = torch.randint(1, VOCAB, (2, 11), generator = torch.Generator().manual_seed(4)).to(DEVICE)
    with torch.no_grad():
        before = fresh(input_ids = ids, use_cache = False).logits
    assert pg.prefix_seg_info_applied(fresh) is True
    with torch.no_grad():
        after = fresh(input_ids = ids, use_cache = False).logits
    assert torch.equal(before, after)
