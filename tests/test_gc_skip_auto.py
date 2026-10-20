# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""checkpoint_skip_layers = "auto" / UNSLOTH_GC_KEEP_LAYERS, extracted from _utils.py with ast."""

import ast, logging, os
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = (
    "_skip_aware_flag",
    "skip_checkpointing",
    "_gc_keep_layers_env",
    "_AutoCheckpointSkip",
    "usable_cuda_bytes",
)


def _load():
    src = open(UTILS, encoding = "utf-8").read()
    ns = {
        "torch": torch,
        "os": os,
        "logger": logging.getLogger("test_gc_skip_auto"),
        "find_decoder_layers": lambda m: list(_base(m).model.layers),
    }
    found = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES:
            exec(ast.get_source_segment(src, n), ns)
            found.add(n.name)
    return ns, found


def _base(model):
    return getattr(model, "base", model)


def _model(layers = 6, device = "cpu"):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 256,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
    )
    model = Qwen3ForCausalLM(config).float().to(device).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.max_seq_length = 64
    return model


def _checkpointed(model):
    return [bool(layer.gradient_checkpointing) for layer in model.model.layers]


def test_zero_and_unset_env_change_nothing(monkeypatch):
    ns, found = _load()
    monkeypatch.delenv("UNSLOTH_GC_KEEP_LAYERS", raising = False)
    model = _model()
    hooks = dict(model._forward_pre_hooks)
    assert ns["skip_checkpointing"](model, 0) == []
    assert _checkpointed(model) == [True] * 6
    assert dict(model._forward_pre_hooks) == hooks
    assert not hasattr(model, "_unsloth_auto_skip")


@pytest.mark.parametrize(
    "value, expected", [("3", 3), ("max", "max"), ("auto", "auto"), ("0", 0), ("x", 0)]
)
def test_env_parsing(monkeypatch, value, expected):
    ns, found = _load()
    assert "_gc_keep_layers_env" in found, "UNSLOTH_GC_KEEP_LAYERS is not implemented"
    monkeypatch.setenv("UNSLOTH_GC_KEEP_LAYERS", value)
    assert ns["_gc_keep_layers_env"]() == expected


def test_env_count_applies_when_argument_is_zero(monkeypatch):
    ns, found = _load()
    monkeypatch.setenv("UNSLOTH_GC_KEEP_LAYERS", "2")
    model = _model()
    assert ns["skip_checkpointing"](model, 0) == [2, 5]
    assert _checkpointed(model) == [True, True, False, True, True, False]


def _step(model, ids):
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    return loss.detach(), {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
def test_auto_plans_after_first_step_and_keeps_gradients():
    ns, found = _load()
    assert "_AutoCheckpointSkip" in found, 'checkpoint_skip_layers = "auto" is not implemented'
    ids = torch.randint(0, 128, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()

    reference = _model(device = "cuda")
    ref = [_step(reference, ids) for _ in range(3)]

    model = _model(device = "cuda")
    hooks = dict(model._forward_pre_hooks)
    assert ns["skip_checkpointing"](model, "auto") == []
    planner = model._unsloth_auto_skip
    first = _step(model, ids)
    assert _checkpointed(model) == [True] * 6
    assert planner.layer_bytes and planner.layer_bytes > 0
    second = _step(model, ids)
    third = _step(model, ids)
    # A tiny model on any card: every layer fits, and the planner is gone.
    assert planner.kept == [0, 1, 2, 3, 4, 5]
    assert _checkpointed(model) == [False] * 6
    assert not hasattr(model, "_unsloth_auto_skip")
    assert dict(model._forward_pre_hooks) == hooks and not model.model.layers[-1]._forward_hooks
    for (loss, grads), (ref_loss, ref_grads) in zip((first, second, third), ref):
        assert torch.equal(loss, ref_loss)
        assert grads.keys() == ref_grads.keys()
        for name in grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 0, atol = 0, msg = name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
def test_auto_keeps_everything_checkpointed_without_headroom(monkeypatch):
    ns, found = _load()
    ids = torch.randint(0, 128, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()
    model = _model(device = "cuda")
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    # A card other processes have filled: 1 MiB free, nothing cached.
    total = torch.cuda.get_device_properties(0).total_memory
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1 << 20, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 0)
    _step(model, ids)
    _step(model, ids)
    assert planner.kept == []
    assert _checkpointed(model) == [True] * 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
def test_auto_backs_off_without_reentrant_checkpointing():
    ns, found = _load()
    ids = torch.randint(0, 128, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()
    model = _model(device = "cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    ns["skip_checkpointing"](model, "auto")
    _step(model, ids)
    _step(model, ids)
    assert not hasattr(model, "_unsloth_auto_skip")
    assert _checkpointed(model) == [True] * 6


def test_auto_without_cuda_says_so_once(monkeypatch, capsys):
    ns, found = _load()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _model()
    assert ns["skip_checkpointing"](model, "auto") == []
    assert ns["skip_checkpointing"](model, "auto") == []
    out = capsys.readouterr().out
    assert out.count("needs a CUDA GPU") == 1, out
    assert _checkpointed(model) == [True] * 6
    assert not hasattr(model, "_unsloth_auto_skip")


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
def test_auto_ignores_allocation_spike_before_first_step():
    ns, found = _load()
    ids = torch.randint(0, 128, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()
    model = _model(device = "cuda")
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    # 3 GiB usable; a 2 GiB loading spike before training must not be read as activations.
    ns["usable_cuda_bytes"] = lambda device: 3 << 30
    spike = torch.empty(2 << 30, dtype = torch.uint8, device = "cuda")
    del spike
    _step(model, ids)
    _step(model, ids)
    assert planner.kept == [0, 1, 2, 3, 4, 5]
    assert _checkpointed(model) == [False] * 6


def _big_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 32000,
        hidden_size = 1024,
        intermediate_size = 4096,
        num_hidden_layers = 6,
        num_attention_heads = 8,
        num_key_value_heads = 4,
        head_dim = 128,
    )
    model = Qwen3ForCausalLM(config).float().cuda().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.max_seq_length = 64
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
@pytest.mark.parametrize("optimizer_stepped", [True, False])
def test_auto_reserves_optimizer_state_only_when_not_allocated(optimizer_stepped):
    ns, found = _load()
    ids = torch.randint(0, 32000, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()
    model = _big_model()
    numel = sum(p.numel() for p in model.parameters())
    assert numel * 8 > (1 << 30)  # AdamW state well above the planner's slack
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    # Usable memory after step 1 (state already excluded when it exists): grads + 1.6 GiB.
    ns["usable_cuda_bytes"] = lambda device: numel * 4 + int(1.6 * (1 << 30))
    model(input_ids = ids, labels = ids).loss.backward()
    if optimizer_stepped:
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
    model(input_ids = ids, labels = ids).loss.backward()  # plans at this forward
    if optimizer_stepped:
        # Full finetuning: the state is in memory_allocated already and must not be reserved again.
        assert planner.kept == [0, 1, 2, 3, 4, 5]
    else:
        # Gradient accumulation: the state does not exist yet, so it is reserved and nothing fits.
        assert planner.kept == []
    del optimizer, model
    torch.cuda.empty_cache()


def test_auto_with_model_off_cuda_on_a_cuda_host_installs_nothing(monkeypatch, capsys):
    from torch.optim import optimizer as torch_optimizer

    ns, found = _load()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    global_hooks = dict(getattr(torch_optimizer, "_global_optimizer_pre_hooks", {}))
    model = _model()  # CPU (or XPU) weights: no CUDA decoder device
    hooks = dict(model._forward_pre_hooks)
    assert ns["skip_checkpointing"](model, "auto") == []
    assert ns["skip_checkpointing"](model, "auto") == []
    out = capsys.readouterr().out
    assert out.count("needs a CUDA GPU") == 1, out
    assert not hasattr(model, "_unsloth_auto_skip")
    assert dict(model._forward_pre_hooks) == hooks
    assert not model.model.layers[-1]._forward_hooks
    assert dict(getattr(torch_optimizer, "_global_optimizer_pre_hooks", {})) == global_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
def test_auto_reserves_state_for_params_the_optimizer_has_not_seen():
    ns, found = _load()
    ids = torch.randint(0, 32000, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()
    model = _big_model()
    numel = sum(p.numel() for p in model.parameters())
    # A trainable param the first batch never reaches (an unrouted MoE expert, an unused tower).
    model.unused = torch.nn.Parameter(torch.zeros(1 << 27, device = "cuda"))
    assert model.unused.numel() * 8 >= (1 << 30)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    ns["usable_cuda_bytes"] = lambda device: numel * 4 + int(1.6 * (1 << 30))
    model(input_ids = ids, labels = ids).loss.backward()
    assert model.unused.grad is None
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)
    assert model.unused not in optimizer.state
    model(input_ids = ids, labels = ids).loss.backward()  # plans at this forward
    # main: one step happened, so nothing is reserved for `unused` and every layer is kept.
    assert planner.kept == []
    del optimizer, model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the auto planner sizes CUDA memory")
def test_auto_does_not_scale_persistent_grads_by_sequence_length():
    ns, found = _load()
    model = _big_model()
    numel = sum(p.numel() for p in model.parameters())
    model.max_seq_length = 1024
    short = torch.randint(0, 32000, (1, 64), generator = torch.Generator().manual_seed(1)).cuda()
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    # Room for 16x the short batch's activations, but not for 16x the fp32 grads it allocated.
    usable = numel * 4 * 6
    ns["usable_cuda_bytes"] = lambda device: usable
    model(input_ids = short, labels = short).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)
    model(input_ids = short, labels = short).loss.backward()  # plans at this forward
    assert planner.layer_rows == 64
    # main: (activations + grads) x 16 leaves no headroom, so nothing is kept.
    assert planner.kept == [0, 1, 2, 3, 4, 5]
    del optimizer, model
    torch.cuda.empty_cache()
