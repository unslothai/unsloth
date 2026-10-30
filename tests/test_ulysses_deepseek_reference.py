# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DeepSeek Ulysses context parallelism against independent fp64 single-process references, on CPU gloo
ranks: the explicit softmax scale through the attention dispatcher, V3.2's global top-k indexer, V4
sliding / CSA / HCA sink attention at release compression ratios, right-padded batches, a rank with no
labels, a label across the shard seam, CP = 2 / 4 and DP 2 x CP 2, LoRA, one AdamW step, and negative
controls that must fail."""

import math
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import unsloth
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_deepseek as ud
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
# Each arm must import the unsloth tree that holds this test file.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assert os.path.abspath(unsloth.__file__).startswith(_ROOT + os.sep), unsloth.__file__
if not all(hasattr(transformers, n) for n in ("DeepseekV3Config", "DeepseekV32Config", "DeepseekV4Config")):
    pytest.skip("transformers without DeepSeek V3 / V3.2 / V4", allow_module_level = True)

VOCAB = 96
# DeepSeek-V3's released YaRN parameters (factor 40, mscale 1.0): softmax scale = mscale^2 / sqrt(d).
YARN = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 40.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 64,
}


def _yarn_scale(head_dim):
    mscale = 0.1 * 1.0 * math.log(40.0) + 1.0
    return head_dim**-0.5 * mscale * mscale


def _named(name):
    return lambda detail: f"{name}: {detail}"


def _spawn(fn, world, *args):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _entry,
            args = (world, os.path.join(d, "init"), fn, args),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _cp_groups(world, cp_size):
    """Contiguous CP groups (rank // cp_size = DP replica); every rank creates every group."""
    mine = None
    for start in range(0, world, cp_size):
        ranks = list(range(start, start + cp_size))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            mine = group
    return mine


def _manager(cp_size, group):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = cp_size
    manager.backend = "ulysses"
    local = dist.get_rank(group)
    manager.mesh = types.SimpleNamespace(get_group = lambda: group, get_local_rank = lambda: local)
    return manager


# ---------------------------------------------------------------------------------------------
# 1. Explicit softmax scale through _ulysses_hf_attention -> ulysses_attention -> _run_attention.
# ---------------------------------------------------------------------------------------------


def _scale_worker(rank, world, scaling, packed, control):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    if control:
        # Broken variant: the dispatcher drops the explicit scale (SDPA falls back to 1/sqrt(d)).
        real = ad._run_attention

        def dropped(*, config, **kw):
            config.sdpa_kwargs = None
            return real(config = config, **kw)

        ad._run_attention = dropped
    g = torch.Generator().manual_seed(3)
    bsz, heads, seq, qk_dim, v_dim = (1 if packed else 2), 4, 16, 24, 16
    q = torch.randn(bsz, heads, seq, qk_dim, generator = g, dtype = torch.float64)
    k = torch.randn(bsz, heads, seq, qk_dim, generator = g, dtype = torch.float64)
    v = torch.randn(bsz, heads, seq, v_dim, generator = g, dtype = torch.float64)
    grad_out = torch.randn(bsz, seq, heads, v_dim, generator = g, dtype = torch.float64)
    lengths = torch.tensor([5, 7, 4]) if packed else None

    # Independent reference: softmax(scale * QK^T + mask) V, block-causal for packed rows.
    pos = torch.arange(seq)
    keep = pos[:, None] >= pos[None, :]
    if packed:
        doc = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        keep = keep & (doc[:, None] == doc[None, :])
    leaves = [x.clone().requires_grad_() for x in (q, k, v)]
    logits = torch.matmul(leaves[0], leaves[1].transpose(-1, -2)) * scaling
    logits = logits.masked_fill(~keep, float("-inf"))
    ref = torch.matmul(torch.softmax(logits, -1), leaves[2]).transpose(1, 2)
    (ref * grad_out).sum().backward()

    local = seq // world
    sl = slice(rank * local, (rank + 1) * local)
    parts = [x[:, :, sl].clone().requires_grad_() for x in (q, k, v)]
    kwargs = {"scaling": scaling}
    if packed:
        kwargs["packed_seq_lengths"] = lengths
    module = types.SimpleNamespace(training = True)
    out, _ = uh._ulysses_hf_attention(module, *parts, None, group = dist.group.WORLD, **kwargs)
    (out * grad_out[:, sl]).sum().backward()
    torch.testing.assert_close(out, ref[:, sl], rtol = 1e-10, atol = 1e-10, msg = _named("output"))
    for name, part, leaf in zip("qkv", parts, leaves):
        torch.testing.assert_close(part.grad, leaf.grad[:, :, sl], rtol = 1e-10, atol = 1e-10, msg = _named(name))


SCALES = {"yarn_v3_release": _yarn_scale(24), "far_from_default": 0.03, "default": 24**-0.5}


@pytest.mark.parametrize("packed", [False, True], ids = ["dense", "packed"])
@pytest.mark.parametrize("scale", list(SCALES))
def test_explicit_scale_sdpa_numerics(scale, packed):
    _spawn(_scale_worker, 2, SCALES[scale], packed, False)


@pytest.mark.parametrize("packed", [False, True], ids = ["dense", "packed"])
def test_dropped_scale_is_wrong(packed):
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_scale_worker, 2, SCALES["yarn_v3_release"], packed, True)


# ---------------------------------------------------------------------------------------------
# 2. Model-level parity: V3 (registry path), V3.2 (indexer), V4 (sliding / CSA / HCA).
# ---------------------------------------------------------------------------------------------


def _mla(**extra):
    return dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        intermediate_size = 96,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        kv_lora_rank = 16,
        q_lora_rank = 24,
        qk_rope_head_dim = 8,
        v_head_dim = 16,
        qk_nope_head_dim = 16,
        n_routed_experts = 4,
        n_shared_experts = 1,
        num_experts_per_tok = 2,
        first_k_dense_replace = 1,
        n_group = 1,
        topk_group = 1,
        rope_parameters = YARN,
        **extra,
    )


def _build(kind, v4_topk):
    t = transformers
    if kind == "v3":
        return t.DeepseekV3ForCausalLM(t.DeepseekV3Config(**_mla()))
    if kind == "v32":
        # 8 indexer heads keep exact zero-score ties (all ReLUs zero) improbable; top-6 of up to 24 keys.
        config = t.DeepseekV32Config(
            **_mla(index_topk = 6, index_head_dim = 16, index_n_heads = 8)
        )
        return t.DeepseekV32ForCausalLM(config)
    # Release ratios: CSA 4, HCA 128, sliding window 128 (DeepSeek-V4 config defaults).
    config = t.DeepseekV4Config(
        vocab_size = VOCAB,
        hidden_size = 64,
        intermediate_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
        rope_head_dim = 16,
        q_lora_rank = 32,
        o_lora_rank = 16,
        o_groups = 2,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 128,
        # 32 indexer heads: an all-zero ReLU score (a tie whose top-k pick depends on the row
        # length, also without CP) is improbable; the release uses 64.
        index_n_heads = 32,
        index_head_dim = 16,
        index_topk = v4_topk,
        layer_types = [
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types = ["hash_moe", "moe", "moe"],
        compress_rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
    )
    model = t.DeepseekV4ForCausalLM(config)
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        for module in model.modules():
            if hasattr(module, "sinks"):
                module.sinks.copy_(0.5 * torch.randn(module.sinks.shape, generator = g))
            if hasattr(module, "position_bias"):
                module.position_bias.copy_(0.3 * torch.randn(module.position_bias.shape, generator = g))
            if hasattr(module, "tid2eid"):
                ids = torch.arange(VOCAB)
                module.tid2eid[:, 0] = ids % 4
                module.tid2eid[:, 1] = (ids + 1) % 4
    return model


def _model(kind, v4_topk = 16, lora = False):
    torch.manual_seed(0)
    model = _build(kind, v4_topk).double().train()
    model.config._attn_implementation = "eager" if kind in ("v32", "v4") else "sdpa"
    model.config.use_cache = False
    for module in model.modules():
        if hasattr(module, "set_experts_implementation"):
            module.set_experts_implementation("eager")
    if lora:
        from peft import LoraConfig, get_peft_model

        targets = (
            ["q_b_proj", "kv_proj", "o_b_proj"]
            if kind == "v4"
            else ["q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
        )
        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = targets))
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:  # nonzero B: the adapter changes the forward
                    p.copy_(0.05 * torch.randn(p.shape, generator = g, dtype = p.dtype))
    return model


def _batch(seq, lengths, labels_scheme, cp_size):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, VOCAB, (len(lengths), seq), generator = g)
    mask = torch.zeros_like(ids)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
    ids = ids * mask  # right padding with pad id 0
    labels = ids.masked_fill(mask == 0, -100)
    if labels_scheme == "late_rank_empty":
        # The last CP rank of the padded row gets no labels; the label predicting the token right
        # after the first seam is kept so a label crosses the seam.
        padded = seq + (-seq) % cp_size
        start = padded - padded // cp_size
        labels[:, start:] = -100
        labels[:, padded // cp_size] = ids[:, padded // cp_size].masked_fill(
            mask[:, padded // cp_size] == 0, -100
        )
    return ids, mask, labels


def _sum_ce(logits, shift_labels):
    return F.cross_entropy(
        logits.flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
    )


def _controls(model, kind, control, monkey):
    if control is None:
        return
    if control == "shard_local":
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
    elif control == "default_scale":
        for module in model.modules():
            if hasattr(module, "qk_head_dim") and hasattr(module, "scaling"):
                module.scaling = module.qk_head_dim**-0.5
    elif control == "dropped_window":
        real = ud._sink_attention
        monkey(ud, "_sink_attention", lambda q, kv, bb, s, w, *a: real(q, kv, bb, s, None, *a))
    elif control == "wrong_offset":
        real = ud.ulysses_deepseek_v4_forward

        def shard_positions(module, *args, position_ids = None, group = None, **kw):
            return real(module, *args, position_ids = position_ids - position_ids[:, :1], group = group, **kw)

        monkey(ud, "ulysses_deepseek_v4_forward", shard_positions)
    elif control == "local_topk":
        real = ud._v32_global_topk

        def local_topk(indexer, hidden, q_resid, pe, group, module):
            # Shard-local selection: each rank scores only its own keys, then indices are offset.
            local = real(indexer, hidden, q_resid, pe, _SOLO[0], module)
            local = local + dist.get_rank(group) * hidden.shape[1]
            parts = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
            dist.all_gather(parts, local.contiguous(), group = group)
            return torch.cat(parts, dim = 1)

        monkey(ud, "_v32_global_topk", local_topk)
    else:
        raise ValueError(control)


_SOLO = [None]


def _parity_worker(rank, world, case):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    if case.get("chunk"):
        os.environ["UNSLOTH_CP_ATTN_CHUNK"] = str(case["chunk"])
    kind, cp_size = case["kind"], case["cp"]
    dp = world // cp_size
    group = _cp_groups(world, cp_size)
    solos = [dist.new_group([r]) for r in range(world)]
    _SOLO[0] = solos[rank]
    patched = []

    def monkey(obj, name, value):
        patched.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    try:
        _parity_body(rank, world, case, kind, cp_size, dp, group, monkey)
    finally:
        for obj, name, value in reversed(patched):
            setattr(obj, name, value)


def _parity_body(rank, world, case, kind, cp_size, dp, group, monkey):
    import copy

    model = _model(kind, case.get("v4_topk", 16), case.get("lora", False))
    reference = copy.deepcopy(model)
    seq, lengths = case["seq"], case["lengths"]
    ids, mask, labels = _batch(seq, lengths, case.get("labels", "all"), cp_size)
    shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    assert count > 0

    # transformers runs the MoE router, the V3.2 / V4 indexer scores and V4's compressor gate softmax
    # in fp32, so even the reference moves at ~1e-8 with the batch shape.
    tol = dict(rtol = 1e-6, atol = 1e-6)
    # Single-process fp64 reference on the same right-padded batch (mask applied).
    ref_logits = reference(input_ids = ids, attention_mask = mask, use_cache = False).logits
    (_sum_ce(ref_logits, shift) / count).backward()
    # Valid tokens must not depend on the padding: each row alone, unpadded.
    with torch.no_grad():
        for row, n in enumerate(lengths):
            alone = reference(input_ids = ids[row : row + 1, :n], use_cache = False).logits
            torch.testing.assert_close(
                ref_logits[row, :n].detach(), alone[0], msg = _named("ref pad"), **tol
            )

    # CP arm: this DP replica's rows, CP over its group.
    rows = list(range(rank // cp_size, len(lengths), dp))
    manager = _manager(cp_size, group)
    manager.attach_attention_hooks(model)
    _controls(model, kind, case.get("control"), monkey)
    inputs = {
        "input_ids": ids[rows].clone(),
        "attention_mask": mask[rows].clone(),
        "labels": labels[rows].clone(),
    }
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            position_ids = inputs["position_ids"],
            use_cache = False,
        )
        local_ce = _sum_ce(out.logits, inputs["shift_labels"])
        (local_ce / count).backward()
    if case.get("labels") == "late_rank_empty" and dist.get_rank(group) == cp_size - 1:
        assert inputs["shift_labels"].ne(-100).sum() == 0
    logits = torch.cat(dist_all_gather(out.logits.detach(), group), dim = 1)
    for i, row in enumerate(rows):
        n = lengths[row]
        torch.testing.assert_close(
            logits[i, :n], ref_logits[row, :n].detach(), msg = _named("logits"), **tol
        )
        with torch.no_grad():
            alone = reference(input_ids = ids[row : row + 1, :n], use_cache = False).logits
        torch.testing.assert_close(logits[i, :n], alone[0], msg = _named("unpadded"), **tol)
    total = local_ce.detach().clone()
    dist.all_reduce(total)
    torch.testing.assert_close(
        total / count, _sum_ce(ref_logits.detach(), shift) / count, **tol
    )
    # Same fp32 ops; chunked attention also accumulates full-sequence gradients in fp32.
    gtol = dict(rtol = 1e-5, atol = 1e-8)

    # Gradients: None stays None, zero stays zero, values match; then one AdamW step.
    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        has = torch.tensor([0.0 if param.grad is None else 1.0])
        dist.all_reduce(has)
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        ref_grad = ref_params[name].grad
        if ref_grad is None:
            assert has.item() == 0, f"{name}: reference grad is None, CP produced one"
            param.grad = None
            continue
        assert has.item() > 0, f"{name}: CP grad is None, reference has one"
        torch.testing.assert_close(grad, ref_grad, msg = _named(name), **gtol)
        param.grad = grad
    # Adam's first step is lr * g / (|g| + eps): with eps = 1e-8 the fp32 noise of tiny gradients
    # (V4's Sinkhorn hyper-connections run in fp32) is amplified to ~lr. eps = 1e-3 bounds that at
    # ~1e-8 while weight decay on a zero-instead-of-None gradient (lr * wd * |p| ~ 1e-4) still fails.
    lr = dict(lr = 1e-2, weight_decay = 0.1, eps = 1e-3)
    atol = dict(rtol = 1e-6, atol = 1e-7)
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **lr).step()
    torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad], **lr).step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, ref_params[name], msg = _named(f"adamw {name}"), **atol)


def dist_all_gather(t, group):
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, t.contiguous(), group = group)
    return parts


# V3.2: 23 tokens pads to 24 under CP; row 1 holds 17 valid tokens, so CP ranks see unequal counts.
V32 = dict(kind = "v32", seq = 23, lengths = [23, 17])
# V4 at release ratios: 397 tokens -> 3 full HCA blocks (128) + partial, ~99 CSA blocks, window 128;
# the CP = 2 seam (199) sits inside HCA block 1, the CP = 4 seams (100, 200, 300) inside blocks 0-2.
V4 = dict(kind = "v4", seq = 397, lengths = [397, 290])
V3 = dict(kind = "v3", seq = 23, lengths = [23, 17])

CASES = {
    "v3_cp2": dict(V3, cp = 2, world = 2),
    "v3_dp2cp2": dict(V3, cp = 2, world = 4),
    "v32_cp2": dict(V32, cp = 2, world = 2),
    "v32_cp2_chunk1": dict(V32, cp = 2, world = 2, chunk = 1),
    "v32_cp2_chunk5": dict(V32, cp = 2, world = 2, chunk = 5),
    "v32_cp4_late_rank_empty": dict(V32, cp = 4, world = 4, labels = "late_rank_empty"),
    "v32_dp2cp2": dict(V32, cp = 2, world = 4),
    "v32_cp2_lora": dict(V32, cp = 2, world = 2, lora = True),
    "v4_cp2": dict(V4, cp = 2, world = 2),
    "v4_cp2_topk512": dict(V4, cp = 2, world = 2, v4_topk = 512),
    "v4_cp2_chunk1": dict(V4, cp = 2, world = 2, chunk = 1),
    "v4_cp2_chunk37": dict(V4, cp = 2, world = 2, chunk = 37),
    "v4_cp4_late_rank_empty": dict(V4, cp = 4, world = 4, labels = "late_rank_empty", chunk = 37),
    "v4_dp2cp2": dict(V4, cp = 2, world = 4, chunk = 64),
    "v4_cp2_lora": dict(V4, cp = 2, world = 2, lora = True, chunk = 50),
}

CONTROLS = {
    "v32_local_topk": dict(V32, cp = 2, world = 2, control = "local_topk"),
    "v32_default_scale": dict(V32, cp = 2, world = 2, control = "default_scale"),
    "v3_default_scale": dict(V3, cp = 2, world = 2, control = "default_scale"),
    "v4_shard_local": dict(V4, cp = 2, world = 2, control = "shard_local"),
    "v4_dropped_window": dict(V4, cp = 2, world = 2, control = "dropped_window"),
    "v4_wrong_offset": dict(V4, cp = 2, world = 2, control = "wrong_offset"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_deepseek_cp_matches_fp64_reference(name):
    case = CASES[name]
    _spawn(_parity_worker, case["world"], case)


@pytest.mark.parametrize("name", list(CONTROLS))
def test_deepseek_cp_negative_control_fails(name):
    case = CONTROLS[name]
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, case["world"], case)


# ---------------------------------------------------------------------------------------------
# 3. V3.2 indexer: global causal top-k vs transformers' full-sequence indexer.
# ---------------------------------------------------------------------------------------------


def _topk_worker(rank, world, seq):
    model = _model("v32")
    attn = model.model.layers[0].self_attn
    modeling = __import__(type(attn).__module__, fromlist = ["x"])
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, seq, 64, generator = g, dtype = torch.float64)
    pos = torch.arange(seq)[None].expand(2, -1)
    pe = model.model.rotary_emb(hidden, pos)
    q_resid = attn.q_a_layernorm(attn.q_a_proj(hidden)).detach()
    causal = torch.full((2, seq, seq), torch.finfo(torch.float64).min, dtype = torch.float64).triu(1)
    ref = attn.indexer(hidden, q_resid, pe, causal, pos)

    # Reference scores, to prove the compared rows are not tied at the top-k boundary.
    idx = attn.indexer
    q = idx.wq_b(q_resid).view(2, seq, idx.n_heads, idx.head_dim)
    k = idx.k_norm(idx.wk(hidden)).unsqueeze(2)
    r = idx.qk_rope_head_dim
    qr, kr = modeling.apply_rotary_pos_emb(q[..., :r], k[..., :r], *pe, unsqueeze_dim = 2)
    q = torch.cat([qr, q[..., r:]], -1).float()
    k = torch.cat([kr, k[..., r:]], -1).squeeze(2).float()
    scores = F.relu(torch.matmul(q, k.transpose(-1, -2).unsqueeze(1)) * idx.softmax_scale)
    w = idx.weights_proj(hidden).float() * idx.n_heads**-0.5
    s = torch.matmul(w.unsqueeze(-2), scores).squeeze(-2)

    local = seq // world
    sl = slice(rank * local, (rank + 1) * local)
    pe_local = tuple(t[:, sl] for t in pe)
    got = ud._v32_global_topk(idx, hidden[:, sl], q_resid[:, sl], pe_local, dist.group.WORLD, modeling)
    assert got.shape == ref.shape
    topk = idx.index_topk
    short = 0
    for b in range(2):
        for t in range(seq):
            want = {int(i) for i in ref[b, t] if int(i) <= t}
            have = {int(i) for i in got[b, t] if int(i) <= t}
            if t + 1 <= topk:
                short += 1
                assert want == have == set(range(t + 1)), (b, t)
                continue
            row = s[b, t, : t + 1].sort(descending = True).values
            assert row[topk - 1] - row[topk] > 1e-6, f"tie at ({b}, {t}): pick another seed"
            assert want == have, (b, t, want, have)
    assert short == 2 * topk  # early queries with fewer causal keys than top-k were exercised


@pytest.mark.parametrize("world", [2, 4])
def test_v32_global_topk_matches_full_sequence_indexer(world):
    _spawn(_topk_worker, world, 24)


# ---------------------------------------------------------------------------------------------
# 4. Chunked sink attention custom backward vs an independent fp64 reference.
# ---------------------------------------------------------------------------------------------


def _reference_sink(q, kv, block_bias, sinks, window, scaling):
    rows, seq = q.shape[2], q.shape[2]
    pos = torch.arange(seq)
    diff = pos[:, None] - pos[None, :]
    keep = (diff >= 0) & (diff < window)
    logits = torch.matmul(q, kv.transpose(-1, -2)) * scaling
    dense = logits[..., :seq].masked_fill(~keep, float("-inf"))
    full = torch.cat([dense, logits[..., seq:] + block_bias], -1)
    full = torch.cat([full, sinks.view(1, -1, 1, 1).expand(q.shape[0], -1, rows, 1)], -1)
    probs = torch.softmax(full, -1)[..., :-1]
    return torch.matmul(probs, kv).transpose(1, 2)


def _sink_case(chunk, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CP_ATTN_CHUNK", chunk)
    g = torch.Generator().manual_seed(0)
    seq, blocks, window, scaling = 40, 5, 9, 0.21
    q = torch.randn(2, 3, seq, 8, generator = g, dtype = torch.float64)
    kv = torch.randn(2, 1, seq + blocks, 8, generator = g, dtype = torch.float64)
    sinks = torch.randn(3, generator = g, dtype = torch.float64)
    # Block w is visible to query t once (t + 1) // 8 > w (HCA rule), plus one indexer-dropped block.
    t = torch.arange(seq)[:, None]
    bias = torch.where(torch.arange(blocks)[None] < (t + 1) // 8, 0.0, float("-inf"))
    bias = bias.to(torch.float64).expand(2, 1, seq, blocks).clone()
    bias[1, :, 30:, 1] = float("-inf")
    grad_out = torch.randn(2, seq, 3, 8, generator = g, dtype = torch.float64)
    a = [x.clone().requires_grad_() for x in (q, kv, sinks)]
    b = [x.clone().requires_grad_() for x in (q, kv, sinks)]
    out = ud._sink_attention(a[0], a[1], bias, a[2], window, scaling, 0.0, True)
    ref = _reference_sink(b[0], b[1], bias, b[2], window, scaling)
    torch.testing.assert_close(out, ref, rtol = 1e-12, atol = 1e-12)
    (out * grad_out).sum().backward()
    (ref * grad_out).sum().backward()
    # q's rows are written once; kv and sinks accumulate over chunks in fp32 by design.
    exact = dict(rtol = 1e-12, atol = 1e-12)
    tol = exact if int(chunk) >= seq else dict(rtol = 1e-6, atol = 1e-7)
    for name, x, y in zip(("q", "kv", "sinks"), a, b):
        torch.testing.assert_close(x.grad, y.grad, msg = _named(name), **(exact if name == "q" else tol))


# 1 and 7 run the chunked autograd op (7 leaves a partial last chunk); 40 = the sequence and 4096
# take the plain autograd path.
@pytest.mark.parametrize("chunk", ["1", "7", "40", "4096"])
def test_chunked_sink_attention_backward_matches_reference(chunk, monkeypatch):
    _sink_case(chunk, monkeypatch)


@pytest.mark.parametrize("chunk", ["1", "7"])
def test_chunked_sink_attention_dropping_a_chunk_is_wrong(chunk, monkeypatch):
    real = ud._recompute_rows

    def drop_first(chunk_size, full_seq, run, grad_out, *args):
        grad_out = grad_out.clone()
        grad_out[:, :chunk_size] = 0
        return real(chunk_size, full_seq, run, grad_out, *args)

    monkeypatch.setattr(ud, "_recompute_rows", drop_first)
    with pytest.raises(AssertionError):
        _sink_case(chunk, monkeypatch)


# ---------------------------------------------------------------------------------------------
# 5. The own-mask exemption covers only layer types and the window.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["v32", "v4"])
def test_own_mask_exemption_is_narrow(kind):
    config = _model(kind).config
    assert cp._unsupported_attention(config, "ulysses") is None
    assert cp._unsupported_attention(config, "ring") is not None
    config.attn_logit_softcapping = 30.0
    assert cp._unsupported_attention(config, "ulysses") == "attention logit softcapping"
    config.attn_logit_softcapping = None
    config.attention_chunk_size = 64
    assert cp._unsupported_attention(config, "ulysses") == "chunked attention"
    config.attention_chunk_size = None
    config.layer_types = ["chunked_attention"] * config.num_hidden_layers
    assert cp._unsupported_attention(config, "ulysses") is not None
