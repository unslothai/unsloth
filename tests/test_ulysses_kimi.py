# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Kimi-Linear / Kimi K3 (KDA + MLA), on CPU gloo ranks.

Kimi ships remote modeling code, so the layers here are stand-ins with the remote class names and
attribute layout, and a plain full-sequence torch reference forward.
"""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

HIDDEN, HEADS, HEAD_DIM, VOCAB = 32, 4, 8, 64


def _acc(x):
    # fp32 math for low precision, fp64 kept for the fp64 reference.
    return torch.promote_types(x.dtype, torch.float32)


class _ShortConv(nn.Conv1d):
    # fla.modules.ShortConvolution: depthwise causal conv with a SiLU.
    def __init__(
        self,
        channels,
        kernel = 4,
    ):
        super().__init__(
            channels, channels, kernel, groups = channels, bias = False, padding = kernel - 1
        )
        self.activation = "silu"

    def forward(self, x):
        return F.silu(super().forward(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))


class _GatedNorm(nn.Module):
    # fla FusedRMSNormGated(activation = "sigmoid").
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) + 0.1 * torch.randn(dim))

    def forward(self, x, g):
        x = x.to(_acc(x))
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6)
        return (self.weight * x * torch.sigmoid(g.to(_acc(g)))).to(g.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, k3):
        super().__init__()
        proj = HEADS * HEAD_DIM
        self.num_heads, self.head_dim = HEADS, HEAD_DIM
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(HIDDEN, proj, bias = False) for _ in range(3)
        )
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (_ShortConv(proj) for _ in range(3))
        self.A_log = nn.Parameter(0.5 * torch.randn(HEADS))
        self.dt_bias = nn.Parameter(0.5 * torch.randn(proj))
        self.f_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias = False)
        self.f_b_proj = nn.Linear(HEAD_DIM, proj, bias = False)
        self.b_proj = nn.Linear(HIDDEN, HEADS, bias = False)
        self.gate_lower_bound = -5.0 if k3 else None
        if k3:  # use_full_rank_gate
            self.g_proj = nn.Linear(HIDDEN, proj, bias = False)
        else:
            self.g_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias = False)
            self.g_b_proj = nn.Linear(HEAD_DIM, proj, bias = False)
        self.o_norm = _GatedNorm(HEAD_DIM)
        self.o_proj = nn.Linear(proj, HIDDEN, bias = False)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        cache_params = None,
        **kwargs,
    ):
        bsz, seq, _ = hidden_states.shape
        shape = (bsz, seq, HEADS, HEAD_DIM)
        q = self.q_conv1d(self.q_proj(hidden_states)).view(shape)
        k = self.k_conv1d(self.k_proj(hidden_states)).view(shape)
        v = self.v_conv1d(self.v_proj(hidden_states)).view(shape)
        acc = _acc(hidden_states)
        f = self.f_b_proj(self.f_a_proj(hidden_states)).view(shape).to(acc) + self.dt_bias.view(
            HEADS, HEAD_DIM
        )
        a = self.A_log.to(acc).exp().view(HEADS, 1)
        if self.gate_lower_bound is not None:
            g = self.gate_lower_bound * torch.sigmoid(a * f)
        else:
            g = -a * F.softplus(f)
        beta = self.b_proj(hidden_states).to(acc).sigmoid()
        q = F.normalize(q.to(acc), dim = -1) * HEAD_DIM**-0.5
        k = F.normalize(k.to(acc), dim = -1)
        state = q.new_zeros(bsz, HEADS, HEAD_DIM, HEAD_DIM)
        outs = []
        for t in range(seq):
            state = state * g[:, t].exp().unsqueeze(-1)
            pred = (k[:, t].unsqueeze(-1) * state).sum(-2)
            state = state + k[:, t].unsqueeze(-1) * (
                beta[:, t, :, None] * (v[:, t].to(acc) - pred)
            ).unsqueeze(-2)
            outs.append((q[:, t].unsqueeze(-1) * state).sum(-2))
        o = torch.stack(outs, dim = 1).to(hidden_states.dtype)
        if hasattr(self, "g_proj"):
            gate = self.g_proj(hidden_states)
        else:
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        o = self.o_norm(o, gate.view(shape))
        return self.o_proj(o.reshape(bsz, seq, -1))


class KimiMLAAttention(nn.Module):
    # No RoPE (mla_use_nope); query / key heads 12 wide, value heads 8.
    def __init__(self, config, k3):
        super().__init__()
        self.config = config
        self.num_heads, self.num_key_value_groups, self.is_causal = HEADS, 1, True
        self.qk_nope, self.qk_rope, self.v_dim, self.kv_lora = 8, 4, 8, 16
        self.q_dim = self.qk_nope + self.qk_rope
        self.scaling = self.q_dim**-0.5
        self.q_proj = nn.Linear(HIDDEN, HEADS * self.q_dim, bias = False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, self.kv_lora + self.qk_rope, bias = False)
        self.kv_b_proj = nn.Linear(self.kv_lora, HEADS * (self.qk_nope + self.v_dim), bias = False)
        self.o_proj = nn.Linear(HEADS * self.v_dim, HIDDEN, bias = False)
        self.use_output_gate = k3  # mla_use_output_gate
        if k3:
            self.g_proj = nn.Linear(HIDDEN, HEADS * self.v_dim, bias = False)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        **kwargs,
    ):
        bsz, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq, HEADS, self.q_dim).transpose(1, 2)
        kv, k_rot = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora, self.qk_rope], dim = -1
        )
        kv = self.kv_b_proj(kv).view(bsz, seq, HEADS, -1).transpose(1, 2)
        k_nope, value = kv.split([self.qk_nope, self.v_dim], dim = -1)
        k_rot = k_rot.view(bsz, 1, seq, self.qk_rope).expand(-1, HEADS, -1, -1)
        key = torch.cat([k_nope, k_rot], dim = -1)
        attention = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        out, _ = attention(self, q, key, value, attention_mask, dropout = 0.0, scaling = self.scaling)
        out = out.reshape(bsz, seq, -1)
        if self.use_output_gate:
            out = out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(out)


class _Layer(nn.Module):
    def __init__(self, config, linear, k3):
        super().__init__()
        self.self_attn = KimiDeltaAttention(k3) if linear else KimiMLAAttention(config, k3)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 2 * HIDDEN), nn.SiLU(), nn.Linear(2 * HIDDEN, HIDDEN)
        )
        self.norm1, self.norm2 = nn.LayerNorm(HIDDEN), nn.LayerNorm(HIDDEN)

    def forward(
        self,
        h,
        attention_mask = None,
    ):
        # Kimi hands the delta layers the 2D padding mask and MLA the causal mask (None here).
        mask = attention_mask if isinstance(self.self_attn, KimiDeltaAttention) else None
        h = h + self.self_attn(hidden_states = self.norm1(h), attention_mask = mask)
        return h + self.mlp(self.norm2(h))


class _KimiForCausalLM(nn.Module):
    def __init__(self, k3):
        super().__init__()
        self.config = types.SimpleNamespace(
            _attn_implementation = "sdpa",
            num_attention_heads = HEADS,
            num_key_value_heads = HEADS,
            output_router_logits = False,
            vision_config = None,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            _Layer(self.config, linear, k3) for linear in (True, False, True)
        )
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias = False)

    def forward(
        self,
        input_ids,
        position_ids = None,
        use_cache = False,
        attention_mask = None,
        **kwargs,
    ):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h, attention_mask)
        return types.SimpleNamespace(logits = self.lm_head(h))


def _model(k3):
    torch.manual_seed(0)
    return _KimiForCausalLM(k3).train()


def _manager(
    size,
    rank,
    group = None,
):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_group = lambda: group, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.to(_acc(logits)).flatten(0, 1),
            shift_labels.flatten(),
            ignore_index = -100,
            reduction = "sum",
        )
        / count
    )


def _parity_worker(rank, world, k3, control):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(k3)
    ids = torch.randint(0, VOCAB, (2, 13), generator = torch.Generator().manual_seed(1))
    shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(model(input_ids = ids).logits, shift, count).backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none = True)

    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control:
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(input_ids = inputs["input_ids"], position_ids = inputs["position_ids"])
        _loss(out.logits, inputs["shift_labels"], count).backward()
    for name, param in model.named_parameters():
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, reference[name], rtol = 1e-4, atol = 1e-6, msg = name)


def _spawn(fn, world, *args):
    # Fork: a spawned child would re-import unsloth without the conftest CUDA spoof.
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


@pytest.mark.parametrize("k3", [False, True], ids = ["kimi_linear", "kimi_k3"])
def test_ulysses_kimi_gradients_match_one_rank(k3):
    _spawn(_parity_worker, 2, k3, False)


def test_shard_local_kimi_delta_is_wrong():
    # Negative control: the stock forward on a sequence shard restarts the scan at the shard seam.
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, False, True)


def _shard_local(fn, size, *xs):
    # Negative-control helper: run fn on each sequence shard separately, as a shard-local layer would.
    n = xs[0].shape[1] // size
    return torch.cat([fn(*(x[:, i * n : (i + 1) * n] for x in xs)) for i in range(size)], dim = 1)


# Deliberately broken copies of ulysses_kimi_delta_forward: each must fail the parity check.
_BROKEN = {
    "shard_local_scan": (
        (
            "out = _kda_scan(q, k, v, g, b.float().sigmoid())",
            "out = _shard_local(_kda_scan, size, q, k, v, g, b.float().sigmoid())",
        ),
    ),
    "shard_local_conv": tuple(
        (
            f"{x} = _kda_short_conv(module.{x}_conv1d, {x}, rows)",
            f"{x} = _shard_local(lambda t: _kda_short_conv(module.{x}_conv1d, t, rows), size, {x})",
        )
        for x in "qkv"
    ),
    "a_log_of_rank0": (
        (
            "module.A_log.reshape(-1)[rank * heads : (rank + 1) * heads]",
            "module.A_log.reshape(-1)[:heads]",
        ),
    ),
    "dt_bias_of_rank0": (
        (
            "module.dt_bias.reshape(-1).index_select(0, rows)",
            "module.dt_bias.reshape(-1)[: heads * dim]",
        ),
    ),
    "lower_bound_dropped": (('getattr(module, "gate_lower_bound", None)', "None"),),
    # MLA: keep the zero padded tail of the value heads instead of the real ones.
    "mla_value_tail": (("out[..., :v_dim]", "out[..., head_dim - v_dim :]"),),
}


def _install_broken(name):
    import inspect

    target = "_ulysses_hf_attention" if name.startswith("mla") else "ulysses_kimi_delta_forward"
    src = inspect.getsource(getattr(uh, target))
    for before, after in _BROKEN[name]:
        assert src.count(before) == 1, before
        src = src.replace(before, after)
    namespace = dict(vars(uh), _shard_local = _shard_local)
    exec(src, namespace)
    setattr(uh, target, namespace[target])


def _audit_batch(
    kind,
    seed,
    seq = 13,
):
    """Right padded rows of unequal length. "full": every valid token is a label. "seam": labels
    stop one token after the CP=2 seam (position 7), so the label at 6 crosses it and every shard
    from the second on has none."""
    ids = torch.randint(0, VOCAB, (2, seq), generator = torch.Generator().manual_seed(seed))
    mask = torch.ones_like(ids)
    mask[0, 11:] = 0
    mask[1, 9 - seed % 2 :] = 0
    labels = ids.masked_fill(mask == 0, -100)
    if kind == "seam":
        labels[:, 8:] = -100
    return ids, mask, labels


def _inject_lora(model):
    from peft import LoraConfig, inject_adapter_in_model

    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "f_b_proj",
        "b_proj",
        "g_b_proj",
        "g_proj",
        "o_proj",
        "kv_b_proj",
    ]
    model = inject_adapter_in_model(LoraConfig(r = 4, lora_alpha = 8, target_modules = targets), model)
    torch.manual_seed(2)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
        if "lora_B" in name:
            with torch.no_grad():
                param.normal_(0, 0.2)  # nonzero B: the adapters change the forward
    return model


def _audit_worker(rank, world, k3, cp_size, kind, lora, broken):
    import copy

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    dp_size = world // cp_size
    dp_rank, cp_rank = divmod(rank, cp_size)
    groups = [dist.new_group(list(range(d * cp_size, (d + 1) * cp_size))) for d in range(dp_size)]
    model = _model(k3).double()
    if lora:
        model = _inject_lora(model)
    batches = [_audit_batch(kind, seed = d) for d in range(dp_size)]
    shifts = [F.pad(labels, (0, 1), value = -100)[:, 1:] for _, _, labels in batches]
    count = sum(shift.ne(-100).sum() for shift in shifts)

    # Independent single-process fp64 reference over every DP replica's batch.
    for (ids, _, _), shift in zip(batches, shifts):
        _loss(model(input_ids = ids).logits, shift, count).backward()
    reference = copy.deepcopy(model)  # deepcopy drops .grad
    for param, ref in zip(model.parameters(), reference.parameters()):
        ref.grad = None if param.grad is None else param.grad.clone()
    model.zero_grad(set_to_none = True)

    manager = _manager(cp_size, cp_rank, groups[dp_rank] if dp_size > 1 else None)
    manager.attach_attention_hooks(model)
    if broken is not None:
        _install_broken(broken)
    ids, mask, labels = batches[dp_rank]
    inputs = {"input_ids": ids.clone(), "attention_mask": mask.clone(), "labels": labels.clone()}
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
            attention_mask = inputs["attention_mask"],
        )
        local_labels = inputs["shift_labels"].ne(-100).sum()
        _loss(out.logits, inputs["shift_labels"], count).backward()
    if kind == "seam" and cp_rank * inputs["input_ids"].shape[1] >= 7:
        assert local_labels == 0  # this shard trains on nothing of its own
    for (name, param), ref in zip(model.named_parameters(), reference.parameters()):
        # None vs zero vs value, per rank: DDP needs every rank to agree with the reference.
        assert (param.grad is None) == (ref.grad is None), name
        if param.grad is None:
            continue
        dist.all_reduce(param.grad)
        # The CP scan runs in fp32 (as fla does), the reference in fp64.
        torch.testing.assert_close(
            param.grad, ref.grad, rtol = 1e-4, atol = 1e-6, msg = lambda m: f"{name}: {m}"
        )
    if not lora:
        assert all(p.grad is not None for p in model.parameters())
    optimizers = [
        torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr = 1e-2, eps = 1e-4, weight_decay = 0.1
        )
        for m in (model, reference)
    ]
    for optimizer in optimizers:
        optimizer.step()
    for (name, param), ref in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(
            param, ref, rtol = 1e-5, atol = 1e-5, msg = lambda m: f"{name} after AdamW: {m}"
        )


@pytest.mark.parametrize(
    "k3, world, cp_size, kind, lora",
    [
        (False, 2, 2, "full", False),
        (True, 2, 2, "full", False),
        (False, 2, 2, "seam", False),
        (True, 2, 2, "seam", True),
        (False, 2, 2, "full", True),
        (True, 4, 4, "full", False),
        (False, 4, 4, "seam", True),
        (True, 4, 2, "full", True),
    ],
    ids = [
        "linear-cp2",
        "k3-cp2",
        "linear-cp2-seam",
        "k3-cp2-seam-lora",
        "linear-cp2-lora",
        "k3-cp4",
        "linear-cp4-seam-lora",
        "k3-dp2xcp2-lora",
    ],
)
def test_ulysses_kimi_matches_fp64_reference(k3, world, cp_size, kind, lora):
    """Padded rows of unequal length, a shard without labels, a label across the seam, LoRA with
    nonzero B, one AdamW step: CP=2, CP=4 and DP2 x CP2 against one fp64 process."""
    _spawn(_audit_worker, world, k3, cp_size, kind, lora, None)


@pytest.mark.parametrize(
    "broken, k3",
    [
        ("shard_local_scan", False),
        ("shard_local_conv", False),
        ("a_log_of_rank0", False),
        ("dt_bias_of_rank0", True),
        ("lower_bound_dropped", True),
        ("mla_value_tail", False),
    ],
)
def test_broken_kimi_delta_shards_fail_the_reference(broken, k3):
    # Negative controls: a scan or conv restarted at the seam, another rank's per-head decay
    # parameters, or K3's lower-bounded gate read as the standard one.
    with pytest.raises(Exception, match = "not close"):
        _spawn(_audit_worker, 2, k3, 2, "full", False, broken)


def test_kimi_gates():
    model = _model(False)
    kda = model.model.layers[0].self_attn
    assert uh.is_ulysses_kimi_delta(kda) and uh.is_ulysses_mixer(kda)
    assert not uh.is_ulysses_gated_delta(kda)
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    cp._check_stateful_mixers(model, "ulysses", 2)
    with pytest.raises(ValueError, match = "num_heads"):
        cp._check_stateful_mixers(model, "ulysses", 8)
    # An unknown attention class next to the delta layers still blocks CP.
    model.model.layers[1].self_attn.__class__ = type("OtherAttention", (KimiMLAAttention,), {})
    assert not cp._supports_context_parallel(model, "ulysses")


def test_kimi_delta_refuses_packing_and_cache():
    kda = _model(False).model.layers[0].self_attn
    h = torch.zeros(1, 4, HIDDEN)
    with pytest.raises(NotImplementedError, match = "packing"):
        uh.ulysses_kimi_delta_forward(kda, h, packed_seq_lengths = torch.tensor([4]), group = None)
    with pytest.raises(RuntimeError, match = "cache"):
        uh.ulysses_kimi_delta_forward(kda, h, cache_params = object(), group = None)


def test_kimi_delta_gate_matches_fla_reference():
    f, A_log, dt = torch.randn(1, 3, 2, 4), torch.randn(2), torch.randn(8)
    x = f + dt.view(2, 4)
    expected = -A_log.exp().view(2, 1) * F.softplus(x)
    torch.testing.assert_close(uh._kda_gate(f, A_log, dt, None), expected)
    expected = -5.0 * torch.sigmoid(A_log.exp().view(2, 1) * x)
    torch.testing.assert_close(uh._kda_gate(f, A_log, dt, -5.0), expected)


def test_moe_unrouted_experts_get_zero_grads_under_context_parallelism(monkeypatch):
    from unsloth.models.remote_moe_shims import _moe_train_dispatch

    block = nn.Module()
    block.experts = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
    x = torch.randn(2, 4)
    topk_idx = torch.tensor([[0], [0]])  # experts 1 and 2 get no tokens
    topk_weight = torch.ones(2, 1)

    _moe_train_dispatch(block, x, topk_idx, topk_weight).sum().backward()
    assert block.experts[1].weight.grad is None

    block.zero_grad(set_to_none = True)
    monkeypatch.setattr(cp, "_ACTIVE_MANAGER", object())
    out = _moe_train_dispatch(block, x, topk_idx, topk_weight)
    out.sum().backward()
    for expert in block.experts[1:]:
        assert expert.weight.grad is not None and not expert.weight.grad.any()
    torch.testing.assert_close(out, block.experts[0](x))


def test_shimmed_remote_moe_is_not_blocked():
    from unsloth.models.remote_moe_shims import _moe_forward_with_training_path

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(2, 2)])

        def forward(self, x):
            return x

    model = _model(False)
    model.moe = Block()
    assert "sparse MoE" in cp._hybrid_config_blocker(model)
    Block.forward = _moe_forward_with_training_path(Block.forward)
    assert cp._hybrid_config_blocker(model) is None


def test_mla_value_heads_are_padded_for_ulysses(monkeypatch):
    seen = {}

    def fake_ulysses(run, config, context, q, k, v, group):
        seen["v"] = v.shape[-1]
        return torch.zeros(q.shape[0], q.shape[2], q.shape[1], q.shape[3])

    monkeypatch.setattr(uh, "ulysses_attention", fake_ulysses)
    module = types.SimpleNamespace(training = False)
    q = torch.zeros(1, HEADS, 3, 12)
    out, _ = uh._ulysses_hf_attention(module, q, q, torch.zeros(1, HEADS, 3, 8), None, group = None)
    assert seen["v"] == 12 and out.shape[-1] == 8
