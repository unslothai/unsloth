# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses CP for gpt-oss on CPU gloo: fp64 parity vs one process, plus controls that must fail."""

import copy
import inspect
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
gpt_oss = pytest.importorskip("transformers.models.gpt_oss.modeling_gpt_oss")
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
SEQ = 13


def _config(window = 4, **overrides):
    common = dict(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 16,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = window,
        layer_types = LAYER_TYPES,
        router_aux_loss_coef = 0.0,
        output_router_logits = False,
        pad_token_id = 0,
    )
    return gpt_oss.GptOssConfig(**{**common, **overrides})


def _model(
    *,
    window = 4,
    dtype = torch.float64,
    checkpoint = False,
    lora = None,
):
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "native_torch")
    torch.manual_seed(0)
    model = gpt_oss.GptOssForCausalLM(_config(window))
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for module in model.modules():
            if isinstance(module, gpt_oss.GptOssAttention):
                # The default init is N(0, 0.02): near-zero sinks would hide a wrong head slice.
                module.sinks.copy_(1.5 * torch.randn(module.sinks.shape, generator = g))
    model = model.to(dtype).train()
    # gpt-oss has no SDPA path (sdpa ignores the sinks): eager is the attention it trains with.
    model.config._attn_implementation = "eager"
    model.config.use_cache = False
    for module in model.modules():
        if hasattr(module, "set_experts_implementation"):
            module.set_experts_implementation("eager")
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    if lora:
        from peft import LoraConfig, get_peft_model

        extra = {}
        if lora == "experts":
            extra["target_parameters"] = ["experts.gate_up_proj", "experts.down_proj"]
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        model = get_peft_model(
            model, LoraConfig(r = 4, lora_alpha = 8, target_modules = targets, **extra)
        )
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:  # nonzero B: the adapter changes the forward
                    p.copy_(0.05 * torch.randn(p.shape, generator = g, dtype = p.dtype))
    return model


def _manager(size, group):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    local = dist.get_rank(group)
    manager.mesh = types.SimpleNamespace(get_group = lambda: group, get_local_rank = lambda: local)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.to(torch.promote_types(logits.dtype, torch.float32)).flatten(0, 1),
            shift_labels.flatten(),
            ignore_index = -100,
            reduction = "sum",
        )
        / count
    )


def _batch(labels_scheme):
    ids = torch.randint(1, 64, (2, SEQ), generator = torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0  # row 1: 6 valid tokens, right padded (unequal lengths, odd total)
    labels = ids.masked_fill(mask == 0, -100)
    if labels_scheme == "prompt_masked":
        # Completion-only labels: the first shard(s) carry no labels at all.
        labels[:, :9] = -100
    return ids, mask, labels


def _grad_status(grad):
    return 0 if grad is None else (1 if not grad.any() else 2)


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


def _cp_groups(world, size):
    mine = None
    for start in range(0, world, size):
        ranks = list(range(start, start + size))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            mine = group
    return mine


class _FirstHeadsSinks(uh._HeadShard):
    """Control: every rank gets the first heads' sinks (rank 0's), as a slice that ignores the rank would."""

    def __init__(self, module, **overrides):
        if "sinks" in overrides:
            overrides["sinks"] = module.sinks[: overrides["sinks"].shape[0]]
        super().__init__(module, **overrides)


class _UnslicedSinks(uh._HeadShard):
    """Control: the shard keeps all heads' sinks (no slicing at all)."""

    def __init__(self, module, **overrides):
        overrides.pop("sinks", None)
        super().__init__(module, **overrides)


def _install_zoo_attention_forward():
    """Install zoo's GptOssAttention forward (worker only); under Ulysses the stashed HF forward must run.
    On CPU zoo installs nothing, so a must-not-run forward goes in via the same patch_function_past_key_values."""
    from unsloth_zoo.temporary_patches import gpt_oss as zoo_gpt_oss
    from unsloth_zoo.temporary_patches.utils import patch_function_past_key_values

    os.environ["UNSLOTH_MODEL_NAME"] = "unsloth/gpt-oss-20b"
    hf_forward = gpt_oss.GptOssAttention.forward
    zoo_gpt_oss.patch_GptOssAttention()
    if gpt_oss.GptOssAttention.forward is hf_forward:

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values = None,
            **kwargs,
        ):
            raise AssertionError("the zoo forward ran under Ulysses")

        forward.__signature__ = inspect.signature(hf_forward)  # zoo's strict signature match
        assert patch_function_past_key_values(gpt_oss.GptOssAttention, "forward", [forward])
    assert gpt_oss.GptOssAttention.forward is not hf_forward
    assert (
        vars(gpt_oss.GptOssAttention)["_original_modeling_gpt_oss_GptOssAttention_forward"]
        is hf_forward
    )
    assert not uh._calls_attention_registry(gpt_oss.GptOssAttention.forward)


def _model_case(
    rank,
    world,
    case,
    control = None,
):
    size = case["cp"]
    dp = world // size
    group = _cp_groups(world, size)
    window = case.get("window", 4)
    model = _model(window = window, checkpoint = case.get("checkpoint", False), lora = case.get("lora"))
    reference = copy.deepcopy(model)
    ids, mask, labels = _batch(case.get("labels", "pad"))
    shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(
        reference(input_ids = ids, attention_mask = mask, use_cache = False).logits, shift, count
    ).backward()

    if case.get("zoo_forward"):
        _install_zoo_attention_forward()
    rows = list(range(rank // size, ids.shape[0], dp))
    manager = _manager(size, group)
    cp._check_cp_model(model, size, "ulysses")
    real_route = uh._route_hf_forward
    if control == "zoo_unrouted":
        uh._route_hf_forward = lambda module, get_cp_manager: None
    try:
        manager.attach_attention_hooks(model)
    finally:
        uh._route_hf_forward = real_route
    attention = [m for n, m in model.named_modules() if n.endswith("self_attn")]
    assert len(attention) == 4 and all(isinstance(m, gpt_oss.GptOssAttention) for m in attention)
    if control == "shard_local":
        for module in attention:
            module._unsloth_cp_ulysses_attention = False
    elif control == "dropped_window":
        for module in attention:
            module.sliding_window = None
    calls = []
    real = uh._ulysses_hf_attention
    real_shard = uh._HeadShard

    def counted(*args, **kwargs):
        calls.append((args[1].shape[1], args[2].shape[1], args[1].shape[-1]))
        return real(*args, **kwargs)

    uh._ulysses_hf_attention = counted
    if control == "first_heads_sinks":
        uh._HeadShard = _FirstHeadsSinks
    elif control == "unsliced_sinks":
        uh._HeadShard = _UnslicedSinks
    try:
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
            _loss(out.logits, inputs["shift_labels"], count).backward()
    finally:
        uh._ulysses_hf_attention = real
        uh._HeadShard = real_shard
    text = _config()
    if control is None:
        runs = 2 if case.get("checkpoint") else 1
        assert len(calls) == runs * text.num_hidden_layers, (len(calls), runs)
        assert set(calls) == {
            (text.num_attention_heads, text.num_key_value_heads, text.head_dim)
        }, calls
    elif control == "shard_local":
        assert not calls, calls

    ref_params = dict(reference.named_parameters())
    sinks_checked = 0
    for name, param in model.named_parameters():
        statuses = [None] * world
        dist.all_gather_object(statuses, _grad_status(param.grad))
        expected = ref_params[name].grad
        if expected is None:
            assert statuses == [0] * world, (name, statuses)
            continue
        # Every rank holds a gradient (a rank without labels too), or the reduction desyncs.
        assert 0 not in statuses, (name, statuses)
        grad = param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(
            grad, expected, rtol = 1e-5, atol = 1e-6, msg = lambda m: f"{name}: {m}"
        )
        assert _grad_status(grad) == _grad_status(expected), name
        sinks_checked += name.endswith("sinks")
    if case.get("lora") is None:
        assert sinks_checked == text.num_hidden_layers, sinks_checked
    # Large eps: GptOssRMSNorm runs in float32 even under float64, so gradients carry float32 rounding.
    for net in (reference, model):
        trainable = [p for p in net.parameters() if p.requires_grad]
        torch.optim.AdamW(trainable, lr = 1e-3, eps = 1e-3, weight_decay = 0.1).step()
    for (name, mine), theirs in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(
            mine, theirs, rtol = 1e-5, atol = 1e-6, msg = lambda m: f"after AdamW {name}: {m}"
        )


def _case_worker(rank, world, case):
    _model_case(rank, world, case)


CASES = {
    "cp2": dict(cp = 2, world = 2, window = 3),
    "cp2_window9_prompt_masked": dict(cp = 2, world = 2, window = 9, labels = "prompt_masked"),
    "cp4": dict(cp = 4, world = 4, window = 3),
    "dp2cp2": dict(cp = 2, world = 4, window = 5, labels = "prompt_masked"),
    "cp2_checkpoint": dict(cp = 2, world = 2, window = 3, checkpoint = True),
    "cp2_lora": dict(cp = 2, world = 2, window = 3, lora = "dense"),
    "cp4_lora": dict(cp = 4, world = 4, window = 5, lora = "dense"),
    "cp2_lora_experts": dict(cp = 2, world = 2, window = 3, lora = "experts"),
    "cp2_lora_experts_checkpoint": dict(cp = 2, world = 2, window = 3, lora = "experts", checkpoint = True),
    "cp4_lora_experts": dict(cp = 4, world = 4, window = 5, lora = "experts"),
    "cp2_zoo_forward": dict(cp = 2, world = 2, window = 3, zoo_forward = True),
    "cp2_lora_experts_zoo_forward": dict(cp = 2, world = 2, window = 3, lora = "experts", zoo_forward = True),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gpt_oss_cp_matches_fp64_reference(name):
    case = CASES[name]
    _spawn(_case_worker, case["world"], case)


# Expected failure per control (unsliced sinks: shape error).
CONTROLS = {
    "shard_local": AssertionError,
    "dropped_window": AssertionError,
    "first_heads_sinks": AssertionError,
    "unsliced_sinks": RuntimeError,
    "zoo_unrouted": AssertionError,
}


def _controls_worker(rank, world):
    wrong = []
    for control, expected in CONTROLS.items():
        try:
            case = dict(cp = 2, window = 3, zoo_forward = control.startswith("zoo"))
            _model_case(rank, world, case, control = control)
            wrong.append((control, "matched the reference"))
        except Exception as error:
            if type(error) is not expected:
                wrong.append((control, repr(error)))
    assert not wrong, f"negative controls that did not fail as expected: {wrong}"


def test_gpt_oss_negative_controls_fail():
    _spawn(_controls_worker, 2)


def test_gpt_oss_attention_is_generic_hf_route():
    model = _model(dtype = torch.float32)
    assert cp._generic_hf_attention(model)
    assert cp._supports_context_parallel(model, "ulysses")
    cp._check_cp_model(model, 2, "ulysses")
    cp._check_cp_model(model, 4, "ulysses")
    assert cp.resolve_cp_backend("auto", 2, model) == "ulysses"
