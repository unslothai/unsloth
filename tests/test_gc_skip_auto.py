# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""checkpoint_skip_layers = "auto" / UNSLOTH_GC_KEEP_LAYERS, extracted from _utils.py with ast."""

import ast, logging, os
import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

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
    "_size_auto_checkpoint_skip_for_trainer",
)


def _load():
    src = open(UTILS, encoding = "utf-8").read()
    ns = {
        "torch": torch,
        "os": os,
        "math": __import__("math"),
        "_PAIRED_FORWARD_TRAINERS": ("DPOTrainer", "ORPOTrainer", "CPOTrainer"),
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


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
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
    assert planner.kept == [0, 1, 2, 3, 4, 5]
    assert _checkpointed(model) == [False] * 6
    assert not hasattr(model, "_unsloth_auto_skip")
    assert dict(model._forward_pre_hooks) == hooks and not model.model.layers[-1]._forward_hooks
    for (loss, grads), (ref_loss, ref_grads) in zip((first, second, third), ref):
        assert torch.equal(loss, ref_loss)
        assert grads.keys() == ref_grads.keys()
        for name in grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 0, atol = 0, msg = name)


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
def test_auto_keeps_everything_checkpointed_without_headroom(monkeypatch):
    ns, found = _load()
    ids = torch.randint(0, 128, (2, 64), generator = torch.Generator().manual_seed(1)).cuda()
    model = _model(device = "cuda")
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    total = torch.cuda.get_device_properties(0).total_memory
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1 << 20, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: 0)
    _step(model, ids)
    _step(model, ids)
    assert planner.kept == []
    assert _checkpointed(model) == [True] * 6


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
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
    # Usable after step 1: grads + foreach AdamW workspace + 1.6 GiB.
    ns["usable_cuda_bytes"] = lambda device: numel * 8 + int(1.6 * (1 << 30))
    model(input_ids = ids, labels = ids).loss.backward()
    if optimizer_stepped:
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
    model(input_ids = ids, labels = ids).loss.backward()
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
    model = _model()
    hooks = dict(model._forward_pre_hooks)
    assert ns["skip_checkpointing"](model, "auto") == []
    assert ns["skip_checkpointing"](model, "auto") == []
    out = capsys.readouterr().out
    assert out.count("needs a CUDA GPU") == 1, out
    assert not hasattr(model, "_unsloth_auto_skip")
    assert dict(model._forward_pre_hooks) == hooks
    assert not model.model.layers[-1]._forward_hooks
    assert dict(getattr(torch_optimizer, "_global_optimizer_pre_hooks", {})) == global_hooks


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
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
    model(input_ids = ids, labels = ids).loss.backward()
    # main: one step happened, so nothing is reserved for `unused` and every layer is kept.
    assert planner.kept == []
    del optimizer, model
    torch.cuda.empty_cache()


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
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
    model(input_ids = short, labels = short).loss.backward()
    assert planner.layer_rows == 64
    # main: (activations + grads) x 16 leaves no headroom, so nothing is kept.
    assert planner.kept == [0, 1, 2, 3, 4, 5]
    del optimizer, model
    torch.cuda.empty_cache()


class _FakeLayer(torch.nn.Module):
    """A decoder layer whose recompute leaves `cost` fake CUDA bytes behind."""

    def __init__(self, cost, memory):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad = False)
        self.gradient_checkpointing = True
        self.cost, self.memory = cost, memory

    def forward(self, hidden):
        if torch.is_grad_enabled():
            self.memory["allocated"] += self.cost
        return hidden


class _FakeModel(torch.nn.Module):
    def __init__(
        self,
        costs,
        memory,
        attn = "sdpa",
    ):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_FakeLayer(c, memory) for c in costs)
        self.config = type("Config", (), {"_attn_implementation": attn})()
        self.max_seq_length = 16

    def forward(self, hidden):
        return hidden


def _fake_cuda(monkeypatch, ns, usable):
    """CPU stand-ins for the CUDA calls the planner makes; memory_allocated is a counter."""
    memory = {"allocated": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device = None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: memory["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device = None: memory["allocated"])
    ns["usable_cuda_bytes"] = lambda device: usable
    return memory


def _fake_step(model, rows):
    """One reentrant-checkpointed step: no_grad forward of every layer, recompute in reverse."""
    hidden = torch.zeros(1, rows, 4)
    model(hidden)
    with torch.no_grad():
        for layer in model.model.layers:
            layer(hidden)
    for layer in reversed(model.model.layers):
        layer(hidden)


def test_auto_sizes_by_the_most_expensive_layer_type(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    costs = [gib, gib // 8] * 4
    memory = _fake_cuda(monkeypatch, ns, usable = 10 * gib)
    model = _FakeModel(costs, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    _fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    assert planner.layer_bytes == gib
    # (10 GiB - 4.5 GiB step) * 0.75 - 1 GiB = 3.125 GiB: 3 one-GiB layers, not 25 eighth-GiB ones.
    assert len(planner.kept) == 3
    monkeypatch.undo()


@pytest.mark.parametrize("attn, exponent", [("sdpa", 1), ("flash_attention_2", 1), ("eager", 2)])
def test_auto_scales_eager_attention_quadratically(monkeypatch, attn, exponent):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 12 * gib)
    model = _FakeModel([gib // 16] * 8, memory, attn = attn)
    model.max_seq_length = 64  # 4x the probed 16 rows
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    _fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    scale = 4**exponent
    step = 8 * (gib // 16)
    budget = (12 * gib - step * scale) * 0.75 - gib
    assert len(planner.kept) == min(8, int(budget // (gib // 16 * scale)))
    # Linear: 8 of 8 fit. Quadratic: a 4x longer batch costs 16x, so only 2 do.
    assert len(planner.kept) == (8 if exponent == 1 else 2)
    monkeypatch.undo()


def test_auto_planner_does_not_keep_an_unplanned_model_alive(monkeypatch):
    import gc, weakref
    from torch.optim import optimizer as torch_optimizer

    ns, found = _load()
    global_hooks = dict(getattr(torch_optimizer, "_global_optimizer_pre_hooks", {}))
    memory = _fake_cuda(monkeypatch, ns, usable = 1 << 30)
    model = _FakeModel([1] * 4, memory)
    ns["skip_checkpointing"](model, "auto")
    assert model._unsloth_auto_skip is not None
    alive = weakref.ref(model)
    del model
    gc.collect()
    assert alive() is None, "the global optimizer hook kept the model alive"
    assert dict(getattr(torch_optimizer, "_global_optimizer_pre_hooks", {})) == global_hooks
    monkeypatch.undo()


@pytest.mark.parametrize("path", ["llama.py", "vision.py"])
def test_full_finetuning_early_return_applies_checkpoint_skip(path):
    src = open(os.path.join(HERE, "unsloth", "models", path), encoding = "utf-8").read()
    mod = ast.parse(src)
    fn = next(
        f
        for cls in mod.body
        if isinstance(cls, ast.ClassDef)
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name == "get_peft_model"
    )
    branches = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and "UNSLOTH_ENABLE_FULL_FINETUNING" in ast.get_source_segment(src, node.test)
        and isinstance(node.body[-1], ast.Return)
    ]
    assert branches, f"full finetuning early return not found in {path}"
    for branch in branches:
        calls = [
            c
            for stmt in branch.body
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "skip_checkpointing"
        ]
        assert calls and any(
            getattr(a, "id", None) == "checkpoint_skip_layers" for c in calls for a in c.args
        ), f"{path}: full finetuning returns before checkpoint_skip_layers is applied"


def test_auto_sizes_for_the_trainer_batch_not_the_first_one(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 4 * gib)
    model = _FakeModel([gib // 16] * 8, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    trainer = type("SFTTrainer", (), {})()
    trainer.model = model
    trainer.args = type("Args", (), {"per_device_train_batch_size": 4, "max_length": None})()
    ns["_size_auto_checkpoint_skip_for_trainer"](trainer)
    _fake_step(model, rows = 16)  # first batch: 1 row at max_seq_length
    model(torch.zeros(1, 16, 4))
    # Sized for 4 rows: (4 GiB - 2 GiB) * 0.75 - 1 GiB = 0.5 GiB, two 0.25 GiB layers (not all 8).
    assert planner.target_batch == 4
    assert len(planner.kept) == 2
    monkeypatch.undo()


def test_auto_ignores_gradients_on_another_gpu(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 5 * gib)
    model = _FakeModel([gib // 16] * 8, memory)
    model.max_seq_length = 64  # 4x the probed rows
    # A trainable vision tower on another card ("meta" stands in for cuda:1).
    model.tower = torch.nn.Parameter(torch.empty(gib // 8, device = "meta"))
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    _fake_step(model, rows = 16)
    model.tower.grad = torch.empty(gib // 8, device = "meta")  # 0.5 GiB, not on the planned card
    model(torch.zeros(1, 16, 4))
    # 0.5 GiB step x 4 = 2 GiB activations, tower AdamW state on its own card: (5 - 2) * 0.75 - 1 =
    # 1.25 GiB, five 0.25 GiB layers.
    assert len(planner.kept) == 5
    monkeypatch.undo()


def test_auto_still_plans_when_the_probe_layer_moves_to_host(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 10 * gib)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: self.device.type != "meta"))
    model = _FakeModel([gib // 16] * 8, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    # offload_layers = "auto" re-plans for the trainer and swaps the last layer (the probe) to host.
    model.model.layers[-1].to("meta")
    _fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    assert planner.kept is not None and len(planner.kept) > 0
    monkeypatch.undo()


def test_auto_never_scales_below_the_measured_length(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 4 * gib)
    model = _FakeModel([gib // 16] * 8, memory)  # max_seq_length 16
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    planner.target_batch = 4
    _fake_step(model, rows = 64)  # a collator expanded the batch past max_seq_length
    model(torch.zeros(1, 64, 4))
    # 4 rows x 64 measured tokens: scale 4, not 4 x 16 / 64 = 1.
    # (4 GiB - 2 GiB) * 0.75 - 1 GiB = 0.5 GiB, two 0.25 GiB layers (before: all 8).
    assert len(planner.kept) == 2
    monkeypatch.undo()


def _trainable_head(model, numel):
    # Expanded: numel elements of bookkeeping without allocating them.
    model.head = torch.nn.Parameter(torch.zeros(1).expand(numel))
    return model.head


@pytest.mark.parametrize("accumulating", [True, False])
def test_auto_counts_resident_accumulated_grads_once(monkeypatch, accumulating):
    ns, found = _load()
    gib = 1 << 30
    numel = gib // 8  # fp32 grad 0.5 GiB, AdamW state + foreach workspace 1.5 GiB
    # Resident grads are allocated, so the card has that much less usable memory.
    memory = _fake_cuda(monkeypatch, ns, usable = 4 * gib - (gib // 2 if accumulating else 0))
    model = _FakeModel([gib // 16] * 8, memory)
    head = _trainable_head(model, numel)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    _fake_step(model, rows = 16)
    head.grad = torch.zeros(1).expand(numel)
    memory["allocated"] += gib // 2
    if not accumulating:
        planner._optimizer_pre_hook(torch.optim.SGD([head], lr = 0.1), (), {})
        head.grad = None
        memory["allocated"] -= gib // 2
    model(torch.zeros(1, 16, 4))
    # Both: (4 GiB - 0.5 GiB activations - 0.5 GiB grads - 1.5 GiB) * 0.75 - 1 GiB = 0.125 GiB.
    assert len(planner.kept) == 2
    monkeypatch.undo()


def test_auto_reserves_the_foreach_optimizer_workspace(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    numel = gib // 8
    memory = _fake_cuda(monkeypatch, ns, usable = 3 * gib)
    model = _FakeModel([gib // 16] * 8, memory)
    head = _trainable_head(model, numel)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    _fake_step(model, rows = 16)
    head.grad = torch.zeros(1).expand(numel)
    memory["allocated"] += gib // 2
    optimizer = torch.optim.AdamW([head])
    planner._optimizer_pre_hook(optimizer, (), {})
    optimizer.state[head] = {"step": torch.tensor(1.0)}  # state allocated by the first step
    head.grad = None
    memory["allocated"] -= gib // 2
    model(torch.zeros(1, 16, 4))
    # 0.5 GiB foreach workspace still reserved: (3 - 1 - 0.5) * 0.75 - 1 = 0.125 GiB, two layers.
    assert len(planner.kept) == 2
    monkeypatch.undo()


def test_explicit_value_removes_an_earlier_auto_planner(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 10 * gib)
    model = _FakeModel([gib // 16] * 8, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    monkeypatch.delenv("UNSLOTH_GC_KEEP_LAYERS", raising = False)
    ns["skip_checkpointing"](model, 0)
    assert not hasattr(model, "_unsloth_auto_skip") and planner.handles == []
    _fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    assert planner.kept is None
    monkeypatch.undo()


def test_auto_reserves_pending_state_only_on_the_planned_card(monkeypatch):
    ns, found = _load()
    gib = 1 << 30
    memory = _fake_cuda(monkeypatch, ns, usable = 3 * gib)
    model = _FakeModel([gib // 16] * 8, memory)
    # A large trainable tower on another card ("meta" stands in for cuda:1): its AdamW state lives there.
    model.tower = torch.nn.Parameter(torch.empty(gib, device = "meta"))
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    _fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    # (3 - 0.5) * 0.75 - 1 = 0.875 GiB: all 8 layers (before: 8 GiB tower state, nothing kept).
    assert len(planner.kept) == 8
    monkeypatch.undo()


def test_counts_apply_only_to_checkpointable_layers():
    ns, found = _load()
    model = _FakeModel([1] * 6, {"allocated": 0})
    for i in (1, 3, 5):
        model.model.layers[i] = torch.nn.Linear(4, 4)
    assert ns["skip_checkpointing"](model, 3) == [0, 2, 4]


@pytest.mark.skipif(not has_real_cuda(), reason = "the auto planner sizes CUDA memory")
@pytest.mark.parametrize("mode", ["fullgraph", "dynamic"])
def test_auto_planner_stays_out_of_compiled_regions(mode):
    # TrainingArguments(torch_compile = True) compiles the whole model; the planner must stay out of it.
    from torch._dynamo.utils import counters

    class Wrapped(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.config = base.config
            self.max_seq_length = base.max_seq_length

        def forward(self, input_ids):
            hidden = self.base.model(input_ids = input_ids, use_cache = False).last_hidden_state
            return self.base.lm_head(hidden).float().logsumexp(-1).mean()

    ns, _ = _load()
    base = _model(layers = 4, device = "cuda")
    model = Wrapped(base)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(
        model,
        backend = "aot_eager",
        fullgraph = mode == "fullgraph",
        dynamic = True if mode == "dynamic" else None,
    )
    try:
        for seq in (8, 12, 16) if mode == "dynamic" else (8, 8, 8):
            ids = torch.randint(0, 128, (2, seq), device = "cuda")
            if mode == "dynamic":
                torch._dynamo.mark_dynamic(ids, 1)
            compiled(ids).backward()
            model.zero_grad(set_to_none = True)
        assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
        assert planner.started is None and planner.kept is None
        assert _checkpointed(base) == [True] * 4
    finally:
        planner.remove()
        torch._dynamo.reset()
