# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vision.py's non-reentrant offloaded checkpoint gives each model its own zoo offloader."""

import ast, contextlib, gc, os, weakref
import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")
MODULE_FUNCS = (
    "_offload_activation_pack",
    "_offload_activation_unpack",
    "_offload_activation_hooks",
    "_gc_offload_owner",
)


@pytest.fixture
def G(monkeypatch):
    G = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    if not hasattr(G, "activation_offloader_for"):
        pytest.skip("unsloth_zoo without per-model offloaders")
    for name in ("UNSLOTH_GC_MAX_INFLIGHT_COPIES", "UNSLOTH_GC_PINNED_MAX_GB"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_MIN_BYTES", "0")
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", "0")
    monkeypatch.setattr(G, "_reserve_pinned", lambda nbytes: True)
    monkeypatch.setattr(G, "_release_pinned", lambda nbytes: None)
    monkeypatch.setattr(G, "PINNED_MEMORY_AVAILABLE", True)
    monkeypatch.setattr(G, "NONREENTRANT_OFFLOADER", None)
    return G


class _Stream:
    def wait_event(self, event):
        pass

    def wait_stream(self, other):
        pass


class _Event:
    def record(self, stream = None):
        pass

    def query(self):
        return True


def _labelling_offloader(G, phase, labels, made):
    """unsloth_stream offloader on CPU tensors (CPU plays the device); labels each record with the model packing it."""

    class Offloader(G.NonReentrantActivationOffloader):
        def __init__(self):
            super().__init__(stream_mode = True)
            made.append(self)

        def _on_host(self, x):
            return x.numel() == 0

        def _streamable(self, x):
            return True

        def _new_stream(self, device_index):
            return _Stream()

        def _current_stream(self, device_index):
            return _Stream()

        def _stream_ctx(self, stream):
            return contextlib.nullcontext()

        def _new_event(self):
            return _Event()

        def _record_stream(self, x, stream):
            pass

        def _host_copy(
            self,
            x,
            non_blocking = True,
        ):
            return x.detach().clone()

        def _device_copy(self, host, device):
            return host.clone()

        def _empty_on_device(self, host, device):
            return torch.empty_like(host)

        def pack(self, x):
            packed = super().pack(x)
            if type(packed) is not tuple:
                labels[packed] = phase[0]
            return packed

    return Offloader


def _vision_ns(G, **overrides):
    """vision.py's module-level offload helpers, bound to the real zoo."""
    src = open(VISION, encoding = "utf-8").read()
    ns = {
        "torch": torch,
        "non_reentrant_offload_hooks": G.non_reentrant_offload_hooks,
        "activation_offloader_for": G.activation_offloader_for,
    }
    ns.update(overrides)
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name in MODULE_FUNCS:
            exec(ast.get_source_segment(src, n), ns)
    return ns, src


def _vision_nonre_checkpoint(G, model, **overrides):
    """from_pretrained's _nonre_checkpoint (DDP vision models, DeepSeek-V4.1) for `model`, offloading."""
    ns, src = _vision_ns(G, **overrides)
    ns.update(
        model = model,
        weakref = weakref,
        _orig_checkpoint = torch_checkpoint.checkpoint,
        _offload = True,
    )
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        if not any(isinstance(n, ast.FunctionDef) and n.name == "_nonre_checkpoint" for n in body):
            continue
        for n in body:
            if isinstance(n, ast.FunctionDef) and n.name == "_nonre_checkpoint":
                exec(ast.get_source_segment(src, n), ns)
            elif isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_gc_owner" for t in n.targets
            ):
                exec(ast.get_source_segment(src, n), ns)
        del ns["model"]  # from_pretrained's frame is gone once loading returns
        return ns["_nonre_checkpoint"]
    pytest.fail("_nonre_checkpoint not found in vision.py")


class _Backbone(torch.nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(width, width) for _ in range(layers))
        self._gradient_checkpointing_func = None

    def forward(self, h):
        for layer in self.layers:
            h = self._gradient_checkpointing_func(lambda t, layer = layer: torch.tanh(layer(t)), h)
        return h


class _VLM(torch.nn.Module):
    """A ForConditionalGeneration-shaped model: the checkpointed layers live in `.model`."""

    def __init__(self, width, layers, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Backbone(width, layers)

    def forward(self, h):
        return self.model(h)


def _grads(model, x):
    out = [p.grad.clone() for p in model.parameters()] + [x.grad.clone()]
    model.zero_grad(set_to_none = True)
    x.grad = None
    return out


def _input(seed, *shape):
    torch.manual_seed(seed)
    return torch.randn(*shape, requires_grad = True)


@pytest.mark.parametrize("keep_last", ["0", "1"])
def test_interleaved_models_use_their_own_offloaders(G, monkeypatch, keep_last):
    # Two models trained in one process: forward A, forward B, then B's backward before A's.
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", keep_last)
    phase, labels, made = ["A"], weakref.WeakKeyDictionary(), []
    monkeypatch.setattr(
        G, "_new_activation_offloader", _labelling_offloader(G, phase, labels, made)
    )
    a, b = _VLM(16, 4, seed = 0), _VLM(16, 6, seed = 1)
    xa, xb = _input(2, 2, 5, 16), _input(3, 3, 4, 16)

    def plain(function, *args, **kwargs):
        return torch_checkpoint.checkpoint(function, *args, use_reentrant = False)

    a.model._gradient_checkpointing_func = b.model._gradient_checkpointing_func = plain
    a(xa).square().mean().backward()
    b(xb).square().mean().backward()
    ref = _grads(a, xa) + _grads(b, xb)

    a.model._gradient_checkpointing_func = _vision_nonre_checkpoint(G, a)
    b.model._gradient_checkpointing_func = _vision_nonre_checkpoint(G, b)
    la = a(xa).square().mean()
    phase[0] = "B"
    lb = b(xb).square().mean()
    lb.backward()
    touched = [r for r, who in labels.items() if who == "A" and (r.ready is not None or r.unpacked)]
    # One shared offloader: B's first layer prefetches A's last layer input.
    assert touched == [], f"B's backward reloaded {len(touched)} of model A's saved inputs"
    assert len(made) == 2
    la.backward()
    got = _grads(a, xa) + _grads(b, xb)
    for x, y in zip(got, ref):
        assert torch.equal(x, y)


def test_owner_is_the_backbone_and_is_held_weakly(G, monkeypatch):
    phase, labels, made = ["A"], weakref.WeakKeyDictionary(), []
    monkeypatch.setattr(
        G, "_new_activation_offloader", _labelling_offloader(G, phase, labels, made)
    )
    model = _VLM(8, 2, seed = 0)
    model.model._gradient_checkpointing_func = _vision_nonre_checkpoint(G, model)
    x = _input(1, 2, 8)
    model(x).sum().backward()
    # PEFT 0.18 calls the top model's .forward() directly; the backbone's pre-hook still fires.
    assert G._OWNED_OFFLOADERS.get(model.model) is made[0]
    assert made[0].stats["forwards"] == 1
    checkpoint = model.model._gradient_checkpointing_func
    ref = weakref.ref(model)
    model.model._gradient_checkpointing_func = None
    del model
    gc.collect()
    assert ref() is None
    assert checkpoint is not None


def test_older_zoo_without_owner_uses_the_shared_offloader(G, monkeypatch):
    calls = []

    @contextlib.contextmanager
    def old_hooks():  # unsloth_zoo before per-model offloaders: no owner argument
        calls.append(1)
        yield

    model = _VLM(8, 2, seed = 0)
    model.model._gradient_checkpointing_func = _vision_nonre_checkpoint(
        G, model, non_reentrant_offload_hooks = old_hooks, activation_offloader_for = None
    )
    x = _input(1, 2, 8)
    model(x).sum().backward()
    assert calls == [1, 1]
    assert x.grad is not None


def test_no_zoo_hooks_falls_back_to_cpu_offload(G):
    ns, _ = _vision_ns(G, non_reentrant_offload_hooks = None, activation_offloader_for = None)
    x = torch.randn(4, requires_grad = True)
    with ns["_offload_activation_hooks"](torch.nn.Linear(1, 1)):
        y = (x * x).sum()
    y.backward()
    assert torch.equal(x.grad, 2 * x)
