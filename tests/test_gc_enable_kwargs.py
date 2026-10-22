# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""transformers' every_n_layers / offload gradient checkpointing options under Unsloth's checkpoint shims."""

import ast, contextlib, functools, inspect, os
import pytest
import torch

transformers = pytest.importorskip("transformers")
import transformers.modeling_utils as hf_modeling_utils
from transformers import PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")
LOADER_UTILS = os.path.join(HERE, "unsloth", "models", "loader_utils.py")
FUNCS = (
    "_skip_aware_flag",
    "_apply_every_n_layers",
    "_gc_kwargs_note",
    "_checkpoint_offloads",
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
    gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    for name, value in (
        ("FIRST_PASS", False),
        ("LAST_GC_INDEX", 0),
        ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 1 << 40),
        ("CPU_INDEX", 0),
        ("BACKWARD_PASS", False),
    ):
        monkeypatch.setattr(gc, name, value, raising = False)
    monkeypatch.setattr(hf_modeling_utils, "checkpoint", gc.unsloth_checkpoint)
    return gc


def _model(layers = 4):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 96,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
    )
    return Qwen3ForCausalLM(config).float().train()


def _spy_layer_kwargs(model):
    seen = []
    for layer in model.model.layers:
        forward = layer.forward

        def wrapped(
            *args,
            _forward = forward,
            **kwargs,
        ):
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
    return loss.detach(), {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }


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
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
    assert _flags(model) == [True, False, True, False]
    model.gradient_checkpointing_enable()
    assert _flags(model) == [True, True, True, True]


@pytest.mark.skipif(
    not NATIVE, reason = "offload is a gradient_checkpointing_enable argument from transformers 5.17"
)
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


@pytest.mark.skipif(
    not NATIVE, reason = "offload is a gradient_checkpointing_enable argument from transformers 5.17"
)
def test_native_offload_kept_with_stock_checkpoint(patched, monkeypatch):
    entered = []
    real = hf_modeling_utils.save_on_cpu

    class Counting(real):
        def __init__(
            self,
            pin_memory = False,
            device_type = "cuda",
        ):
            # transformers always asks for pinned memory, which torch refuses with no usable accelerator.
            pin_memory = pin_memory and torch.accelerator.is_available()
            super().__init__(pin_memory = pin_memory, device_type = device_type)

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


@pytest.mark.parametrize("via_dict", [False, True])
def test_every_n_layers_zero_is_rejected(patched, via_dict):
    model = _model()
    if via_dict:
        call = lambda: model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"every_n_layers": 0}
        )
    else:
        if not NATIVE:
            pytest.skip(
                "every_n_layers is a gradient_checkpointing_enable argument from transformers 5.17"
            )
        call = lambda: model.gradient_checkpointing_enable(every_n_layers = 0)
    with pytest.raises(ValueError, match = "every_n_layers"):
        call()


def _vision_nonre_checkpoint(offload_hooks):
    """vision.py's _nonre_checkpoint (DDP vision models, DeepSeek-V4.1) with use_gradient_checkpointing = "unsloth"."""
    src = open(VISION, encoding = "utf-8").read()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        if not any(isinstance(n, ast.FunctionDef) and n.name == "_nonre_checkpoint" for n in body):
            continue
        ns = {
            "_orig_checkpoint": torch.utils.checkpoint.checkpoint,
            "_offload": True,
            "_offload_activation_hooks": offload_hooks,
        }
        for n in body:
            if isinstance(n, ast.FunctionDef) and n.name == "_nonre_checkpoint":
                exec(ast.get_source_segment(src, n), ns)
            elif isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "_nonre_checkpoint"
                for t in n.targets
            ):
                exec(ast.get_source_segment(src, n), ns)
        return ns["_nonre_checkpoint"]
    pytest.fail("_nonre_checkpoint not found in vision.py")


@pytest.mark.skipif(
    not NATIVE, reason = "offload is a gradient_checkpointing_enable argument from transformers 5.17"
)
def test_native_offload_dropped_under_vision_non_reentrant_offload(patched, monkeypatch):
    entered, hooked = [], []
    real = hf_modeling_utils.save_on_cpu

    class Counting(real):
        def __enter__(self):
            entered.append(1)
            return super().__enter__()

    @contextlib.contextmanager
    def offload_hooks():
        hooked.append(1)
        yield

    monkeypatch.setattr(hf_modeling_utils, "save_on_cpu", Counting)
    monkeypatch.setattr(hf_modeling_utils, "checkpoint", _vision_nonre_checkpoint(offload_hooks))
    reference = _model()
    ref_loss, ref_grads = _grads(reference)
    model = _model()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": False}, offload = True
    )
    loss, grads = _grads(model)
    assert len(hooked) == 4
    # main: save_on_cpu nests inside the offload _nonre_checkpoint already does.
    assert entered == []
    assert torch.equal(loss, ref_loss)
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol = 0, atol = 1e-6, msg = name)


def _load_remote_gc():
    src = open(LOADER_UTILS, encoding = "utf-8").read()
    ns = {"torch": torch, "functools": functools}
    names = (
        "_forward_calls_checkpointing",
        "_checkpointed_layer_forward",
        "install_remote_gradient_checkpointing",
    )
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(ast.get_source_segment(src, n), ns)
    return ns["install_remote_gradient_checkpointing"]


class _RemoteLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = torch.nn.Linear(width, width)

    def forward(self, hidden_states):
        return hidden_states + torch.tanh(self.proj(hidden_states))


class _RemoteBackbone(torch.nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = torch.nn.ModuleList(_RemoteLayer(width) for _ in range(layers))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


_RemoteBackbone.__module__ = "transformers_modules.tiny_remote.modeling_tiny_remote"
_RemoteLayer.__module__ = "transformers_modules.tiny_remote.modeling_tiny_remote"


class _RemoteModel(PreTrainedModel):
    config_class = transformers.PretrainedConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.model = _RemoteBackbone(16, 4)

    def forward(self, hidden_states):
        return self.model(hidden_states).square().mean()


def _remote_model():
    torch.manual_seed(0)
    return _RemoteModel(transformers.PretrainedConfig()).float().train()


def _remote_grads(model):
    x = torch.randn(3, 5, 16, generator = torch.Generator().manual_seed(1), requires_grad = True)
    model.zero_grad(set_to_none = True)
    loss = model(x)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}


def test_every_n_layers_reaches_manually_wrapped_remote_layers(patched, monkeypatch):
    calls = []

    def counting_checkpoint(function, *args, **kwargs):
        calls.append(1)
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(hf_modeling_utils, "checkpoint", counting_checkpoint)
    install = _load_remote_gc()
    reference = _remote_model()
    ref_loss, ref_grads = _remote_grads(reference)

    model = _remote_model()
    assert install(model, verbose = False) == ["_RemoteLayer"]
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": False, "every_n_layers": 2}
    )
    loss, grads = _remote_grads(model)
    # main: the holder-wide flag checkpoints all 4 remote layers.
    assert len(calls) == 2
    assert torch.equal(loss, ref_loss)
    for name in grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol = 0, atol = 1e-6, msg = name)

    model.model.gradient_checkpointing = False
    model.model.gradient_checkpointing = True
    calls.clear()
    _remote_grads(model)
    assert len(calls) == 2
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    calls.clear()
    _remote_grads(model)
    assert len(calls) == 4
