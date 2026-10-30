# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MiniMax Ulysses parity against the upstream full-sequence forward in float64, on CPU gloo ranks.

Lightning attention is checked block by block (lengths below / at / above the block size, seams
inside a block and exactly between blocks, a partial final block), then whole M1 hybrid and M2
models with padding, a rank without labels, a label across the seam, global position offsets,
GQA / MQA, LoRA, one AdamW step, CP = 2 / 4 and DP2 x CP2. Each mutant of the Lightning forward
and each negative control must break parity."""

import copy
import inspect
import os
import tempfile
import textwrap
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "MiniMaxConfig"):
    pytest.skip("transformers without MiniMax", allow_module_level = True)
HAS_M2 = hasattr(transformers, "MiniMaxM2Config")

TOL = dict(rtol = 1e-8, atol = 1e-10)
# A broken variant must fail on numbers (assert_close with these names), not crash on a shape.
PARITY_FAILURE = r"AssertionError: (output|input grad|logits)"
SEQ = 22


def _config(kind, block_size = 4, heads = 4, kv_heads = 2, layer_types = None):
    common = dict(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 32,
        num_attention_heads = heads,
        num_key_value_heads = kv_heads,
        head_dim = 8,
        num_local_experts = 2,
        num_experts_per_tok = 1,
        attention_dropout = 0.0,
        tie_word_embeddings = False,
    )
    if kind == "m1":
        layer_types = layer_types or [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
        ]
        common.update(
            num_hidden_layers = len(layer_types), layer_types = layer_types, block_size = block_size
        )
        return transformers.MiniMaxConfig(**common), transformers.MiniMaxForCausalLM
    common.update(
        num_hidden_layers = 2,
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    return transformers.MiniMaxM2Config(**common), transformers.MiniMaxM2ForCausalLM


def _model(kind, lora = False, checkpoint = None, **config_kwargs):
    torch.manual_seed(0)
    config, cls = _config(kind, **config_kwargs)
    model = cls(config).train()
    if checkpoint is not None:
        # The recompute repeats every all-to-all in backward, in the same order on each rank.
        model.gradient_checkpointing_enable({"use_reentrant": checkpoint == "reentrant"})
    model.config._attn_implementation = "sdpa"
    for module in model.modules():
        if hasattr(module, "set_experts_implementation"):
            module.set_experts_implementation("eager")
    if lora:
        from peft import LoraConfig, get_peft_model

        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj", "output_gate"]
        # init_lora_weights = False: B is random, so the adapters change the forward.
        model = get_peft_model(
            model,
            LoraConfig(r = 4, lora_alpha = 8, target_modules = targets, init_lora_weights = False),
        )
    return model.double()


def _mutate(old, new):
    """Replace ulysses_lightning_forward in this (forked) worker by a source-edited copy."""
    src = textwrap.dedent(inspect.getsource(uh.ulysses_lightning_forward))
    assert old in src, old
    namespace = dict(vars(uh))
    exec(compile(src.replace(old, new), uh.__file__, "exec"), namespace)
    uh.ulysses_lightning_forward = namespace["ulysses_lightning_forward"]


MUTANTS = {
    # Every rank uses rank 0's per-head slopes / decays.
    "slope_rate_slice": ("module.slope_rate[head]", "module.slope_rate[:heads]"),
    "diagonal_decay_slice": ("module.diagonal_decay[:, head]", "module.diagonal_decay[:, :heads]"),
    "query_decay_slice": ("module.query_decay[head]", "module.query_decay[:heads]"),
    # A partial block read with the tail of the query decay instead of its head.
    "partial_block_query_decay": ("query_decay[:, :n]", "query_decay[:, -n:]"),
    # The carried state is not decayed between blocks.
    "undecayed_state": ("state * torch.exp(-slope * n)", "state"),
    # q | k | v as three contiguous thirds instead of interleaved per head.
    "qkv_layout": (
        "qkv.reshape(bsz, local_len, module.num_attention_heads, 3 * dim)",
        "qkv.reshape(bsz, local_len, 3, module.num_attention_heads, dim).transpose(2, 3)"
        ".reshape(bsz, local_len, module.num_attention_heads, 3 * dim)",
    ),
    # Output gate applied before the full-width norm.
    "gate_before_norm": (
        "out = module.norm(out.reshape(bsz, local_len, -1))\n"
        "    out = torch.sigmoid(module.output_gate(hidden_states)) * out",
        "out = torch.sigmoid(module.output_gate(hidden_states)) * out.reshape(bsz, local_len, -1)\n"
        "    out = module.norm(out)",
    ),
    # Blocks restart at the shard start instead of the global sequence start.
    "shard_aligned_blocks": ("block, total = module.block_size, q.shape[2]",
                             "block, total = local_len, q.shape[2]"),
}


def _lightning_worker(rank, world, seq_len, block_size, mutant):
    if mutant is not None and mutant != "shard_local":
        _mutate(*MUTANTS[mutant])
    config, _ = _config("m1", block_size = block_size)
    from transformers.models.minimax.modeling_minimax import MiniMaxLightningAttention

    torch.manual_seed(0)
    # layer_idx 2 of 4: the slope factor is neither 1 nor ~0.
    module = MiniMaxLightningAttention(config, layer_idx = 2).double()
    for p in module.parameters():
        p.data.normal_(0, 0.3)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, seq_len, config.hidden_size, generator = g, dtype = torch.float64)
    cot = torch.randn(2, seq_len, config.hidden_size, generator = g, dtype = torch.float64)

    x_ref = x.clone().requires_grad_()
    ref = module(x_ref, None, None)[0]
    (ref * cot).sum().backward()
    ref_grads = {n: p.grad.clone() for n, p in module.named_parameters()}
    module.zero_grad(set_to_none = True)

    local = seq_len // world
    shard = slice(rank * local, (rank + 1) * local)
    x_local = x[:, shard].clone().requires_grad_()
    if mutant == "shard_local":
        out = module(x_local, None, None)[0]
    else:
        out = uh.ulysses_lightning_forward(module, x_local, group = None)[0]
    (out * cot[:, shard]).sum().backward()
    torch.testing.assert_close(out, ref[:, shard], **TOL, msg = "output")
    torch.testing.assert_close(x_local.grad, x_ref.grad[:, shard], **TOL, msg = "input grad")
    for name, param in module.named_parameters():
        dist.all_reduce(param.grad)
        torch.testing.assert_close(param.grad, ref_grads[name], **TOL, msg = name)


# (seq_len, block_size, world): what the seam and the final block look like.
LIGHTNING_CASES = {
    "below_block": (4, 8, 2),  # one partial block, seam inside it
    "at_block": (8, 8, 2),  # one full block, seam inside it
    "above_block_partial": (12, 8, 2),  # seam inside block 0, partial final block
    "seam_between_blocks": (16, 4, 2),  # seam exactly at a block boundary
    "many_blocks_partial": (22, 4, 2),  # 5 full blocks + 2, seam inside block 2
    "cp4_seams_inside": (20, 3, 4),  # seams at 5 / 10 / 15, final block of 2
    "cp4_seams_between": (16, 4, 4),  # every seam on a block boundary
}


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


@pytest.mark.parametrize("case", list(LIGHTNING_CASES))
def test_lightning_matches_upstream_full_sequence(case):
    seq_len, block_size, world = LIGHTNING_CASES[case]
    _spawn(_lightning_worker, world, seq_len, block_size, None)


# Each mutant on a case where it changes the math (a full-size final block hides partial-block slips).
MUTANT_CASES = {
    "shard_local": "many_blocks_partial",
    "slope_rate_slice": "many_blocks_partial",
    "diagonal_decay_slice": "many_blocks_partial",
    "query_decay_slice": "many_blocks_partial",
    "partial_block_query_decay": "above_block_partial",
    "undecayed_state": "seam_between_blocks",
    "qkv_layout": "at_block",
    "gate_before_norm": "below_block",
    "shard_aligned_blocks": "above_block_partial",
}


@pytest.mark.parametrize("mutant", list(MUTANT_CASES))
def test_lightning_mutants_break_parity(mutant):
    seq_len, block_size, world = LIGHTNING_CASES[MUTANT_CASES[mutant]]
    with pytest.raises(Exception, match = PARITY_FAILURE):
        _spawn(_lightning_worker, world, seq_len, block_size, mutant)


def _batch(dp_rank, vocab = 64):
    """Right padded rows of unequal length; labels stop before position 12, so the CP=2 rank 1
    shard has none and position 10's target (token 11) crosses the CP=2 seam."""
    g = torch.Generator().manual_seed(10 + dp_rank)
    ids = torch.randint(1, vocab, (2, SEQ), generator = g)
    lengths = [(SEQ, 9), (17, SEQ)][dp_rank]
    mask = torch.zeros(2, SEQ, dtype = torch.long)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
    ids = ids * mask
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, 12:] = -100
    # Nonzero global offsets: RoPE must see these positions on every shard.
    offsets = torch.tensor([[5], [17]]) + 3 * dp_rank
    positions = torch.arange(SEQ).expand(2, -1) + offsets
    return {"input_ids": ids, "attention_mask": mask, "labels": labels, "position_ids": positions}


def _ce_sum(logits, shift_labels):
    return F.cross_entropy(
        logits.flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
    )


def _model_worker(rank, world, kind, cp_size, lora, control, model_kwargs):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    cp.DEVICE_TYPE_TORCH = "cpu"
    model = _model(kind, lora = lora, **model_kwargs)
    reference = copy.deepcopy(model)
    dp_size, dp_rank = world // cp_size, rank // cp_size
    batches = [_batch(d) for d in range(dp_size)]
    shift = [F.pad(b["labels"], (0, 1), value = -100)[:, 1:] for b in batches]
    count = sum(int(s.ne(-100).sum()) for s in shift)

    # Upstream full-sequence forward, one process, float64.
    ref_logits = []
    ref_loss = 0
    for b, s in zip(batches, shift):
        # No labels: a loss-computing forward may hand back placeholder logits.
        inputs = {k: v for k, v in b.items() if k != "labels"}
        logits = reference(**inputs, use_cache = False).logits
        ref_logits.append(logits.detach())
        ref_loss = ref_loss + _ce_sum(logits, s)
    ref_loss = ref_loss / count
    ref_loss.backward()

    manager = cp.ContextParallelManager(cp_size, "ulysses")
    cp_rank = manager.mesh.get_local_rank()
    manager.attach_attention_hooks(model)
    if control == "shard_local":
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
            module._unsloth_cp_ulysses_attention = False
    if control == "local_positions":
        shard = manager._ulysses_shard

        def local_positions(inputs):
            shard(inputs)
            p = inputs["position_ids"]
            inputs["position_ids"] = p - p[:, :1] + batches[dp_rank]["position_ids"][:, :1]

        manager._ulysses_shard = local_positions
    if control == "qk_norm_after_exchange":
        # What normalising q / k on the received head group (after the exchange) would compute.
        for module in model.modules():
            for norm in (getattr(module, "q_norm", None), getattr(module, "k_norm", None)):
                if norm is None:
                    continue

                def grouped(self, x, size = cp_size):
                    parts = x.chunk(size, dim = -1)
                    weights = self.weight.chunk(size)
                    out = []
                    for part, weight in zip(parts, weights):
                        var = part.pow(2).mean(-1, keepdim = True)
                        out.append(weight * part * torch.rsqrt(var + self.variance_epsilon))
                    return torch.cat(out, dim = -1)

                norm.forward = types.MethodType(grouped, norm)

    calls = {"lightning": 0, "attention": 0}
    for key, fn_name in (("lightning", "ulysses_lightning_forward"), ("attention", "_ulysses_hf_attention")):

        def counted(*args, _fn = getattr(uh, fn_name), _key = key, **kwargs):
            calls[_key] += 1
            return _fn(*args, **kwargs)

        setattr(uh, fn_name, counted)

    inputs = {k: v.clone() for k, v in batches[dp_rank].items()}
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            position_ids = inputs["position_ids"],
            use_cache = False,
        )
        local_loss = _ce_sum(out.logits, inputs["shift_labels"]) / count
        local_loss.backward()

    if control is None:
        # The Ulysses paths ran once per layer of their kind: no silent fallback.
        layer_types = getattr(model.config, "layer_types", None) or (
            ["full_attention"] * model.config.num_hidden_layers
        )
        # Checkpointing recomputes each layer once in backward.
        runs = 2 if model_kwargs.get("checkpoint") else 1
        assert calls["lightning"] == runs * layer_types.count("linear_attention"), calls
        assert calls["attention"] == runs * layer_types.count("full_attention"), calls

    # Outputs: valid positions of the gathered CP shards against the reference.
    gathered = [torch.empty_like(out.logits) for _ in range(cp_size)]
    dist.all_gather(gathered, out.logits.detach().contiguous(), group = manager.group)
    logits = torch.cat(gathered, dim = 1)[:, :SEQ]
    valid = batches[dp_rank]["attention_mask"].bool()
    torch.testing.assert_close(logits[valid], ref_logits[dp_rank][valid], **TOL, msg = "logits")
    assert cp_rank == rank % cp_size

    loss = local_loss.detach().clone()
    dist.all_reduce(loss)
    torch.testing.assert_close(loss, ref_loss.detach(), **TOL, msg = "loss")

    # Gradients: None on every rank iff None in the reference; zero stays zero; else the value.
    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        ref_grad = ref_params[name].grad
        present = torch.tensor(float(param.grad is not None))
        dist.all_reduce(present)
        if ref_grad is None:
            assert present.item() == 0, f"{name}: None in the reference, set under CP"
            continue
        assert present.item() > 0, f"{name}: set in the reference, None on every CP rank"
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        param.grad = grad
        if not ref_grad.any():
            assert not grad.any(), f"{name}: zero in the reference, nonzero under CP"
        torch.testing.assert_close(grad, ref_grad, **TOL, msg = name)

    # One AdamW step with weight decay from the reduced gradients.
    trainable = lambda m: [p for p in m.parameters() if p.requires_grad]
    for m in (model, reference):
        torch.optim.AdamW(trainable(m), lr = 1e-2, weight_decay = 0.1).step()
    for (name, p), q in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, **TOL, msg = f"after AdamW: {name}")


# (kind, cp_size, world, lora, model kwargs)
MODEL_CASES = {
    "m1_cp2": ("m1", 2, 2, False, {}),
    "m1_cp2_lora": ("m1", 2, 2, True, {}),
    "m1_cp4": ("m1", 4, 4, False, {}),
    "m1_dp2_cp2": ("m1", 2, 4, False, {}),
    "m1_cp2_block_8": ("m1", 2, 2, False, {"block_size": 8}),
    "m1_cp2_lora_checkpoint": ("m1", 2, 2, True, {"checkpoint": "non_reentrant"}),
    "m2_cp2_lora_reentrant": ("m2", 2, 2, True, {"checkpoint": "reentrant"}),
    "m2_cp2": ("m2", 2, 2, False, {}),
    "m2_cp2_lora": ("m2", 2, 2, True, {}),
    "m2_cp4_gqa": ("m2", 4, 4, False, {}),
    "m2_cp2_mqa": ("m2", 2, 2, False, {"kv_heads": 1}),
    "m2_dp2_cp2": ("m2", 2, 4, False, {}),
}


@pytest.mark.parametrize("case", list(MODEL_CASES))
def test_minimax_model_matches_upstream(case):
    kind, cp_size, world, lora, kwargs = MODEL_CASES[case]
    if kind == "m2" and not HAS_M2:
        pytest.skip("transformers without MiniMax-M2")
    _spawn(_model_worker, world, kind, cp_size, lora, None, kwargs)


CONTROLS = [
    ("m1", "shard_local"),
    ("m2", "shard_local"),
    ("m1", "local_positions"),
    ("m2", "local_positions"),
    ("m2", "qk_norm_after_exchange"),
]


@pytest.mark.parametrize("kind, control", CONTROLS)
def test_minimax_negative_controls_break_parity(kind, control):
    if kind == "m2" and not HAS_M2:
        pytest.skip("transformers without MiniMax-M2")
    with pytest.raises(Exception, match = PARITY_FAILURE):
        _spawn(_model_worker, 2, kind, 2, False, control, {})


@pytest.mark.skipif(not HAS_M2, reason = "transformers without MiniMax-M2")
@pytest.mark.parametrize(
    "heads, kv_heads, size, match",
    [
        (4, 2, 8, "number of attention heads"),
        (6, 3, 2, "divide one another"),
        (4, 3, 2, "multiple of key/value heads"),
    ],
)
def test_minimax_m2_head_counts_are_validated(heads, kv_heads, size, match):
    config, cls = _config("m2", heads = heads, kv_heads = kv_heads)
    model = cls(config)
    with pytest.raises(ValueError, match = match):
        cp._check_cp_model(model, size, "ulysses")


def test_minimax_m1_head_counts_are_validated():
    config, cls = _config("m1", heads = 4, kv_heads = 4)
    model = cls(config)
    if any(isinstance(getattr(m, "experts", None), torch.nn.ModuleList) for m in model.modules()):
        # transformers 4.x MiniMax keeps per-expert modules, which CP refuses after the head check.
        with pytest.raises(NotImplementedError, match = "per-expert modules"):
            cp._check_cp_model(model, 4, "ulysses")
    else:
        cp._check_cp_model(model, 4, "ulysses")
    with pytest.raises(ValueError, match = "number of attention heads"):
        cp._check_cp_model(model, 8, "ulysses")
