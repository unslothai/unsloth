# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""transformers' every_n_layers / offload gradient checkpointing options under Unsloth's checkpoint shims."""

import ast, functools, inspect, os
import pytest
import torch

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as hf_modeling_utils
from transformers import PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
FUNCS = (
    "_skip_aware_flag",
    "_apply_every_n_layers",
    "_gc_kwargs_note",
    "patch_gradient_checkpointing_enable_kwargs",
)
ASSIGNS = ("_OFFLOADING_CHECKPOINT_SHIMS", "_GC_KWARGS_WARNED")
NATIVE = "offload" in inspect.signature(PreTrainedModel.gradient_checkpointing_enable).parameters


def _load():
    src = open(UTILS, encoding = "utf-8").read()
    ns = {"inspect": inspect, "functools": functools}
    for n in ast.parse(src).body:
        named = isinstance(n, ast.FunctionDef) and n.name in FUNCS
        assigned = isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ASSIGNS for t in n.targets
        )
        if named or assigned:
            exec(ast.get_source_segment(src, n), ns)
    return ns


@pytest.fixture
def patched():
    """Install the _utils patch (absent on main: then the stock method runs) and restore everything after."""
    original = PreTrainedModel.__dict__["gradient_checkpointing_enable"]
    previous_checkpoint = hf_modeling_utils.checkpoint
    patch = _load().get("patch_gradient_checkpointing_enable_kwargs")
    if patch is not None:
        patch()
    yield
    PreTrainedModel.gradient_checkpointing_enable = original
    hf_modeling_utils.checkpoint = previous_checkpoint


@pytest.fixture
def smart_gc(monkeypatch):
    """patch_unsloth_smart_gradient_checkpointing without buffers; MINIMUM_SIZE keeps offload off."""
    gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    for name, value in (
        ("FIRST_PASS", False), ("LAST_GC_INDEX", 0), ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 1 << 40), ("CPU_INDEX", 0), ("BACKWARD_PASS", False),
    ):
        monkeypatch.setattr(gc, name, value, raising = False)
    monkeypatch.setattr(hf_modeling_utils, "checkpoint", gc.unsloth_checkpoint)
    return gc


def _model(layers = 4):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 96, hidden_size = 32, intermediate_size = 64, num_hidden_layers = layers,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
    )
    return Qwen3ForCausalLM(config).float().train()


def _spy_layer_kwargs(model):
    seen = []
    for layer in model.model.layers:
        forward = layer.forward

        def wrapped(*args, _forward = forward, **kwargs):
            seen.extend(k for k in kwargs if k in ("offload", "every_n_layers"))
            return _forward(*args, **kwargs)

        layer.forward = wrapped
    return seen


def _flags(model):
    return [bool(layer.gradient_checkpointing) for layer in model.model.layers]


def _grads(model):
    ids = torch.randint(0, 96, (2, 12), generator = torch.Generator().manual_seed(1))
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_dict_every_n_layers_and_offload_are_routed_not_bound_into_layers(patched, smart_gc):
    reference = _model()
    ref_loss, ref_grads = _grads(reference)

    model = _model()
    seen = _spy_layer_kwargs(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": True, "every_n_layers": 2, "offload": True}
    )
    # main: every layer stays checkpointed and both keys reach every decoder layer's forward.
    assert _flags(model) == [True, False, True, False]
    loss, grads = _grads(model)
    assert seen == []
    assert torch.equal(loss, ref_loss)
    assert grads.keys() == ref_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol = 0, atol = 1e-6, msg = name)


def test_every_n_layers_survives_for_training_flag_rewrite(patched):
    model = _model()
    if NATIVE:
        model.gradient_checkpointing_enable(every_n_layers = 2)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"every_n_layers": 2})
    assert _flags(model) == [True, False, True, False]
    # for_training / for_inference (GRPO, every generation step) write the flag on every module.
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
    assert _flags(model) == [True, False, True, False]
    model.gradient_checkpointing_enable()
    assert _flags(model) == [True, True, True, True]


@pytest.mark.skipif(not NATIVE, reason = "offload is a gradient_checkpointing_enable argument from transformers 5.17")
def test_native_offload_dropped_only_under_offloading_shim(patched, smart_gc, monkeypatch):
    entered = []
    real = hf_modeling_utils.save_on_cpu

    class Counting(real):
        def __enter__(self):
            entered.append(1)
            return super().__enter__()

    monkeypatch.setattr(hf_modeling_utils, "save_on_cpu", Counting)
    reference = _model()
    ref_loss, ref_grads = _grads(reference)
    model = _model()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": True}, offload = True
    )
    loss, grads = _grads(model)
    # main: save_on_cpu wraps every checkpointed layer on top of Unsloth's own offload.
    assert entered == []
    assert torch.equal(loss, ref_loss)
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol = 0, atol = 1e-6, msg = name)


@pytest.mark.skipif(not NATIVE, reason = "offload is a gradient_checkpointing_enable argument from transformers 5.17")
def test_native_offload_kept_with_stock_checkpoint(patched, monkeypatch):
    entered = []
    real = hf_modeling_utils.save_on_cpu

    class Counting(real):
        def __enter__(self):
            entered.append(1)
            return super().__enter__()

    monkeypatch.setattr(hf_modeling_utils, "save_on_cpu", Counting)
    model = _model()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": False}, offload = True
    )
    _grads(model)
    assert len(entered) == 4


def test_plain_enable_unchanged(patched):
    model = _model()
    model.gradient_checkpointing_enable()
    assert _flags(model) == [True, True, True, True]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    assert _flags(model) == [True, True, True, True]
