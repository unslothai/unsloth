# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block CUDA graphs (opt-in) must copy block outputs and input gradients out of the graph pool.

Under Unsloth checkpointing every block call starts a new cudagraph tree generation, so a
tensor the checkpoint keeps (the saved block input, the gradient handed to the previous
block) would be overwritten by the next replay.
"""

import importlib.util
import os
import sys

import types

import pytest
import torch

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "_ublock_compile_cg", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    # The CPU rest shard runs with UNSLOTH_COMPILE_DISABLE=1, which makes every wrapped block take
    # the eager path, so the graph-cap test would see no compiled call at all.
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)
    # Graphs go off process wide; every test starts with them on.
    monkeypatch.setattr(
        bc,
        "_CUDAGRAPH_STATE",
        getattr(bc, "_new_cudagraph_state", dict)(),
        raising = False,
    )


def test_clone_grad_is_identity_with_copied_gradient():
    x = torch.randn(3, 4, requires_grad = True)
    seen = {}

    def keep(grad):
        seen["ptr"] = grad.data_ptr()
        return grad

    y = bc._CloneGrad.apply(x)
    y.register_hook(keep)
    assert torch.equal(y, x)
    (y * 2).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2.0))
    assert x.grad.data_ptr() != seen["ptr"]


def test_clone_outputs_copies_every_tensor():
    a, b = torch.randn(2), torch.randn(3)
    out = bc._clone_outputs((a, None, b))
    assert out[1] is None
    assert torch.equal(out[0], a) and out[0].data_ptr() != a.data_ptr()
    assert torch.equal(out[2], b) and out[2].data_ptr() != b.data_ptr()
    single = bc._clone_outputs(a)
    assert torch.equal(single, a) and single.data_ptr() != a.data_ptr()


def test_cudagraphs_opt_in_flag(monkeypatch):
    try:
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "1")
        assert _load()._COMPILE_OPTIONS["triton.cudagraphs"] is True
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")
        assert _load()._COMPILE_OPTIONS["triton.cudagraphs"] is False
    finally:
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS", raising = False)
        _load()


def _cap_layer_step(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def run(*args, **kw):
            calls.append(tag)
            return fn(*args, **kw)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(bc, "_CUDAGRAPH_MAX_SHAPES", 2)

    class CapDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    bc._wrap(CapDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = CapDecoderLayer().train()

    def step(L):
        calls.clear()
        x = torch.randn(2, L, 8, requires_grad = True)
        layer(x).sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape
        return set(calls)

    return step


def test_new_shapes_past_the_cap_skip_graphs(monkeypatch):
    """Variable lengths recorded a graph set per new length: 3.5x slower steps and a pool that
    grew every step on an L4. One shape past the cap runs every later call without graphs."""
    step = _cap_layer_step(monkeypatch)
    assert step(16) == {"graphs"}
    assert step(24) == {"graphs"}
    assert step(16) == {"graphs"}
    assert step(32) == {"plain"}  # third shape: over the cap
    assert step(16) == {"plain"}  # graphs stay off, also for admitted shapes
    assert step(40) == {"plain"}


def test_cap_reads_the_environment(monkeypatch):
    try:
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", "3")
        assert _load()._CUDAGRAPH_MAX_SHAPES == 3
        monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", "junk")
        assert _load()._CUDAGRAPH_MAX_SHAPES == 2
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES")
        # One fixed length plus a short last batch; T4 variable lengths paid 15-65 s for 8.
        assert _load()._CUDAGRAPH_MAX_SHAPES == 2
    finally:
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", raising = False)
        _load()


def test_compile_disable_still_bypasses_the_block(monkeypatch):
    # The kill switch is read per call, so it also turns off a block wrapped before it was set.
    step = _cap_layer_step(monkeypatch)
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    assert step(16) == set()


# Replayed-backward stand-in: one reused weight grad buffer per shape, like the cudagraph pool.
_POOL = {}


class _PoolMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        g_w = (grad.reshape(-1, grad.shape[-1]).t() @ x.reshape(-1, x.shape[-1])).contiguous()
        buf = _POOL.setdefault(tuple(g_w.shape), torch.empty_like(g_w))
        buf.copy_(g_w)
        return (
            grad @ weight,
            buf.detach(),
        )


def _pool_layers(monkeypatch, graphs):
    _POOL.clear()
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)

    class PoolDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8) / 8)

        def forward(self, hidden_states, **kwargs):
            return torch.sin(_PoolMatmul.apply(hidden_states, self.weight))

    bc._wrap(PoolDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": graphs})
    torch.manual_seed(0)
    layers = [PoolDecoderLayer().train() for _ in range(3)]
    refs = [torch.nn.Parameter(l.weight.detach().clone()) for l in layers]
    return layers, refs


def _reference_grads(refs, xs):
    for x in xs:
        y = x
        for w in refs:
            y = torch.sin(y @ w.t())
        y.square().mean().backward()
    return [w.grad for w in refs]


@pytest.mark.parametrize("micro_steps", [1, 2])
def test_param_grads_are_copied_out_of_the_graph_pool(monkeypatch, micro_steps):
    """Reentrant checkpointing: a .grad left in the replay buffer is overwritten by the next replay."""
    from torch.utils.checkpoint import checkpoint

    layers, refs = _pool_layers(monkeypatch, graphs = True)
    xs = [torch.randn(2, 4, 8) for _ in range(micro_steps)]
    for x in xs:
        y = x.clone().requires_grad_()
        for layer in layers:
            y = checkpoint(layer, y, use_reentrant = True)
        y.square().mean().backward()
    want = _reference_grads(refs, xs)
    ptrs = {buf.data_ptr() for buf in _POOL.values()}
    for layer, grad in zip(layers, want):
        assert layer.weight.grad.data_ptr() not in ptrs
        torch.testing.assert_close(layer.weight.grad, grad)


def test_preallocated_grads_are_accumulated_in_place(monkeypatch):
    # An existing .grad (zero_grad(set_to_none=False), DDP bucket views) is not replaced.
    layers, refs = _pool_layers(monkeypatch, graphs = True)
    for layer in layers:
        layer.weight.grad = torch.zeros_like(layer.weight)
    owned = [layer.weight.grad.data_ptr() for layer in layers]
    x = torch.randn(2, 4, 8)
    y = x
    for layer in layers:
        y = layer(y)
    y.square().mean().backward()
    want = _reference_grads(refs, [x])
    for layer, ptr, grad in zip(layers, owned, want):
        assert layer.weight.grad.data_ptr() == ptr
        torch.testing.assert_close(layer.weight.grad, grad)


def test_param_grads_hooked_only_with_graphs(monkeypatch):
    layers, _ = _pool_layers(monkeypatch, graphs = False)
    x = torch.randn(2, 4, 8)
    y = x
    for layer in layers:
        y = layer(y)
    y.sum().backward()
    assert not any(l.__dict__.get("_unsloth_block_grad_hooks") for l in layers)


def test_nonreentrant_checkpoint_runs_without_graphs(monkeypatch):
    """Non-reentrant checkpoint forward and recompute skip graphs; reentrant keeps them."""
    if not hasattr(torch._C._autograd, "_top_saved_tensors_default_hooks"):
        pytest.skip("torch < 2.8 cannot read the saved tensor hooks stack")
    from torch.utils.checkpoint import checkpoint

    seen = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def inner(*args, **kw):
            seen.append(tag)
            return fn(*args, **kw)

        return inner

    monkeypatch.setattr(torch, "compile", fake_compile)

    class CkptDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    bc._wrap(CkptDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = CkptDecoderLayer().train()
    for reentrant, want in ((False, {"plain"}), (True, {"graphs"})):
        seen.clear()
        x = torch.randn(2, 16, 8, requires_grad = True)
        checkpoint(layer, x, use_reentrant = reentrant).sum().backward()
        assert len(seen) == 2 and set(seen) == want
        assert x.grad is not None
    assert not bc._in_nonreentrant_checkpoint()


def _tagging_compile(monkeypatch, calls):
    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def run(*args, **kw):
            calls.append(tag)
            return fn(*args, **kw)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)


def _linear_layer_class(name, module = None):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.proj = torch.nn.Linear(8, 8)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)

    if module is not None:
        forward.__module__ = module
    return type(name, (torch.nn.Module,), {"__init__": __init__, "forward": forward})


def test_past_the_cap_every_block_class_goes_off_and_the_pool_is_released(monkeypatch):
    """One cudagraph tree (one pool) per device serves every block class: graphs go off for all of
    them together, so the release can free the whole pool."""
    calls, released = [], []
    _tagging_compile(monkeypatch, calls)
    monkeypatch.setattr(bc, "_CUDAGRAPH_MAX_SHAPES", 1)
    monkeypatch.setattr(
        bc,
        "_release_graph_pool",
        lambda: released.append(1) or bc._CUDAGRAPH_STATE.update(release = "done"),
    )
    A, B = _linear_layer_class("ADecoderLayer"), _linear_layer_class("BDecoderLayer")
    for cls in (A, B):
        bc._wrap(cls, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    a, b = A().train(), B().train()

    def step(L):
        calls.clear()
        x = torch.randn(2, L, 8, requires_grad = True)
        b(a(x)).sum().backward()
        return list(calls)

    assert step(16) == ["graphs", "graphs"]
    assert step(16) == ["graphs", "graphs"]
    assert not released
    assert step(24) == ["plain", "plain"]  # A saw the new length first: B is off too
    assert bc._CUDAGRAPH_STATE["off"] == "shapes"
    assert released == [1]
    assert step(16) == ["plain", "plain"]
    assert released == [1]


def test_cap_notice_is_printed_once(monkeypatch, capsys):
    calls = []
    _tagging_compile(monkeypatch, calls)
    monkeypatch.setattr(bc, "_CUDAGRAPH_MAX_SHAPES", 1)
    A = _linear_layer_class("NoteDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = A().train()
    for L in (8, 16, 24, 32):
        layer(torch.randn(2, L, 8, requires_grad = True)).sum().backward()
    out = capsys.readouterr().out
    assert out.count("block CUDA graphs turned off") == 1
    assert "UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES" in out


@pytest.mark.parametrize("key", ["packed_seq_lengths", "cu_seq_lens_q"])
def test_packed_rows_turn_graphs_off_before_any_recording(monkeypatch, capsys, key):
    """Packed / padding-free rows change total length every step: decide before recording."""
    calls = []
    _tagging_compile(monkeypatch, calls)
    A = _linear_layer_class("PackedDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = A().train()
    lengths = torch.tensor([5, 11], dtype = torch.int32)
    layer(torch.randn(1, 16, 8, requires_grad = True), **{key: lengths}).sum().backward()
    assert calls == ["plain"]
    assert bc._CUDAGRAPH_STATE["off"] == "packed"
    calls.clear()
    layer(torch.randn(2, 16, 8, requires_grad = True)).sum().backward()  # later padded rows too
    assert calls == ["plain"]
    assert capsys.readouterr().out.count("packed / padding-free rows") == 1


def test_packed_rows_on_unsloth_forward_say_graphs_do_not_apply(monkeypatch, capsys):
    # Unsloth's own forwards run packed rows stock (no compiled call at all): say so once.
    calls = []
    _tagging_compile(monkeypatch, calls)
    A = _linear_layer_class("UPackedDecoderLayer", module = "unsloth.models.fake")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = A().train()
    lengths = torch.tensor([5, 11], dtype = torch.int32)
    layer(torch.randn(1, 16, 8, requires_grad = True), packed_seq_lengths = lengths).sum().backward()
    assert calls == []
    assert bc._CUDAGRAPH_STATE["off"] == "packed"
    assert "packed / padding-free rows" in capsys.readouterr().out


def test_flag_unset_ignores_packed_rows(monkeypatch, capsys):
    calls = []
    _tagging_compile(monkeypatch, calls)
    A = _linear_layer_class("NoFlagDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": False})
    layer = A().train()
    lengths = torch.tensor([5, 11], dtype = torch.int32)
    layer(torch.randn(1, 16, 8, requires_grad = True), packed_seq_lengths = lengths).sum().backward()
    assert calls == ["plain"]
    assert bc._CUDAGRAPH_STATE["off"] is None
    assert capsys.readouterr().out == ""


def _counting_clones(monkeypatch):
    seen = {"outputs": 0, "grad": 0}
    clone_outputs = bc._clone_outputs

    def counted(out):
        seen["outputs"] += 1
        return clone_outputs(out)

    class Counted(bc._CloneGrad):
        @staticmethod
        def forward(ctx, x):
            seen["grad"] += 1
            return x.view_as(x)

    monkeypatch.setattr(bc, "_clone_outputs", counted)
    monkeypatch.setattr(bc, "_CloneGrad", Counted)
    return seen


@pytest.mark.parametrize("trainable_input", [True, False])
def test_no_pool_copies_outside_checkpointing(monkeypatch, trainable_input):
    """A grad-mode block call outside backward leaves a forward pending, so cudagraph trees keep
    the next block and the backward in the same generation: its outputs need no copy."""
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("FreeDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layers = [A().train() for _ in range(3)]
    x = torch.randn(2, 4, 8, requires_grad = trainable_input)
    y = x
    for layer in layers:
        y = layer(y)
    y.sum().backward()
    # Only a leaf input's gradient is copied: its .grad outlives the generation.
    assert seen == {"outputs": 0, "grad": 1 if trainable_input else 0}
    assert all(layer.proj.weight.grad is not None for layer in layers)
    assert (x.grad is not None) is trainable_input


@pytest.mark.parametrize(
    "through", ["view", "view_add", "cat", "addmm_input", "addmm_bias", "embedding"]
)
def test_input_viewing_a_trainable_param_copies_its_gradient(monkeypatch, through):
    """inputs_embeds = param.view(...): the view's backward passes the pool gradient straight to
    AccumulateGrad, so the parameter's first .grad would alias the pool. An embedding lookup writes
    a new gradient, so it needs no copy."""
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("ViewInputDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = A().train()
    param = torch.nn.Parameter(torch.randn(8, 8))
    if through == "embedding":
        x = torch.nn.functional.embedding(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), param)
    else:
        x = param.view(2, 4, 8)
        if through == "view_add":
            x = x + torch.zeros(2, 4, 8)
        elif through in ("addmm_input", "addmm_bias"):
            # addmm's backward passes `grad` itself to its additive input; a broadcast bias is summed
            a, b = torch.randn(8, 8), torch.randn(8, 8)
            if through == "addmm_bias":
                param = torch.nn.Parameter(torch.randn(8))
            x = torch.addmm(param, a, b).view(2, 4, 8)
        elif through == "cat":
            # cat's backward hands each input a narrowed view of the incoming gradient
            param = torch.nn.Parameter(torch.randn(4, 8))
            x = torch.cat([param.view(1, 4, 8), torch.zeros(1, 4, 8)], dim = 0)
    layer(x).sum().backward()
    assert seen["grad"] == (0 if through in ("embedding", "addmm_bias") else 1), seen
    assert param.grad is not None


@pytest.mark.parametrize("source", ["block", "foreign"])
def test_only_graphed_block_regions_stop_the_input_gradient_walk(monkeypatch, source):
    """A previous graphed block decided its own input gradient, so the walk stops at its compiled
    node. Any other compiled function (a compiled prompt preprocessor) can hand the incoming
    gradient straight to a trainable leaf, so its first .grad would alias the pool: copy it."""
    real_compile = torch.compile
    torch._dynamo.reset()
    # aot_eager gives the real CompiledFunctionBackward nodes on CPU; graphs themselves are faked.
    monkeypatch.setattr(
        torch, "compile", lambda fn, **kwargs: real_compile(fn, backend = "aot_eager")
    )
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("CompiledRegionDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layers = [A().train() for _ in range(2)]
    if source == "foreign":
        prompt = torch.nn.Parameter(torch.randn(2, 4, 8))
        x = real_compile(lambda p: p.clone(), backend = "aot_eager")(prompt)
        assert type(x.grad_fn).__name__.startswith("CompiledFunction")
        layers[0](x).sum().backward()
        assert seen["grad"] == 1, seen
        assert prompt.grad is not None
    else:
        # block-to-block hand-off: no copy, the memory win stays
        y = layers[0](torch.randn(2, 4, 8))
        assert type(y.grad_fn).__name__.startswith("CompiledFunction")
        layers[1](y).sum().backward()
        assert seen["grad"] == 0, seen
        assert layers[0].proj.weight.grad is not None
    torch._dynamo.reset()


@pytest.mark.parametrize("exposed", ["kwarg", "config", None])
def test_returned_hidden_states_keep_their_pool_copies(monkeypatch, exposed):
    """output_hidden_states hands every block output to the caller, who may keep it past the
    generation: those calls copy out of the pool as before; other calls still skip the copies."""
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setitem(bc._COMPILE_OPTIONS, "triton.cudagraphs", True)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("ExposedDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(output_hidden_states = exposed == "config")
            self.layers = torch.nn.ModuleList([A().train() for _ in range(2)])

        def forward(
            self,
            x,
            output_hidden_states = None,
        ):
            hidden = []
            for layer in self.layers:
                x = layer(x)
                hidden.append(x)
            return x, hidden

    inner = Inner()
    bc._watch_hidden_states(inner)
    x = torch.randn(2, 4, 8)
    kwargs = {"output_hidden_states": True} if exposed == "kwarg" else {}
    y, _ = inner(x, **kwargs)
    y.sum().backward()
    assert seen["outputs"] == (0 if exposed is None else 2), seen
    bc._HIDDEN_STATES_EXPOSED[0] = False


@pytest.mark.parametrize("use_reentrant", [True])
def test_block_after_a_checkpointed_block_copies_its_input_gradient(monkeypatch, use_reentrant):
    """Checkpointed blocks then an uncheckpointed one: the checkpoint's backward recomputes (a new
    torch.compile invocation, a new generation) before it reads the gradient handed to it."""
    from torch.utils.checkpoint import checkpoint

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("MixedDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layers = [A().train() for _ in range(3)]
    x = torch.randn(2, 4, 8, requires_grad = True)
    y = x
    for layer in layers[:2]:
        y = checkpoint(layer, y, use_reentrant = use_reentrant)
    before = dict(seen)
    out = layers[2](y)
    # the last block (grad mode, outside backward) still routes its input gradient out of the pool
    assert seen == {"outputs": before["outputs"], "grad": before["grad"] + 1}
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("reentrant_kind", ["checkpoint", "no_grad"])
def test_pool_copies_kept_where_a_block_call_starts_a_generation(monkeypatch, reentrant_kind):
    from torch.utils.checkpoint import checkpoint

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("CopyDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layers = [A().train() for _ in range(3)]
    x = torch.randn(2, 4, 8, requires_grad = True)
    if reentrant_kind == "no_grad":
        with torch.no_grad():
            for layer in layers:
                x = layer(x)
        assert seen == {"outputs": 6, "grad": 3}  # each first call runs a grad-mode trial too
        return
    y = x
    for layer in layers:
        y = checkpoint(layer, y, use_reentrant = True)
    y.sum().backward()
    # Trial + no-grad forward + recompute inside backward: every call copies (main did the same).
    assert seen == {"outputs": 9, "grad": 6}


def test_frozen_block_without_grad_input_still_copies(monkeypatch):
    # Nothing requires grad: an inference graph, which may start a new generation per call.
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("FrozenDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = A().train().requires_grad_(False)
    layer(torch.randn(2, 4, 8))
    assert seen == {"outputs": 1, "grad": 0}


class _FakeManager:
    def __init__(
        self,
        compile_ids,
        pending = False,
        live = (),
    ):
        self.id_to_compile_id = dict(enumerate(compile_ids))
        self.ids_to_funcs = dict.fromkeys(self.id_to_compile_id)
        self.running_forwards_with_pending_backwards = pending
        self._live = list(live)

    def live_cudagraph_pool_storages_in_curr_execution(self):
        return self._live


def test_pool_release_verdict():
    ours = {"a", "b"}
    assert bc._pool_release_verdict([_FakeManager(["a", "b", "a"])], ours) == "ok"
    assert bc._pool_release_verdict([_FakeManager(["a", "c"])], ours) == "foreign"
    assert bc._pool_release_verdict([_FakeManager(["a", None])], ours) == "foreign"
    assert bc._pool_release_verdict([_FakeManager(["a"], pending = True)], ours) == "pending"
    assert bc._pool_release_verdict([_FakeManager(["a"], live = [object()])], ours) == "pending"


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "x"]])
def test_release_resets_the_trees_only_when_every_function_is_ours(monkeypatch, capsys, ids):
    from types import SimpleNamespace

    from torch._inductor import cudagraph_trees

    manager = _FakeManager(ids)
    resets = []
    containers = {0: SimpleNamespace(tree_manager = manager)}
    monkeypatch.setattr(cudagraph_trees, "get_obj", lambda local, name: containers)
    monkeypatch.setattr(cudagraph_trees, "reset_cudagraph_trees", lambda: resets.append(1))
    reserved = [6 * 2**30]
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d = None: reserved[0])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: reserved.__setitem__(0, 2 * 2**30))
    bc._CUDAGRAPH_STATE["ours"].add("a")
    bc._CUDAGRAPH_STATE["managers"][0] = manager
    bc._release_graph_pool()
    out = capsys.readouterr().out
    if ids == ["a", "a"]:
        assert resets == [1] and bc._CUDAGRAPH_STATE["release"] == "done"
        assert "4.00 GiB reserved freed" in out
    else:
        assert resets == [] and bc._CUDAGRAPH_STATE["release"] == "kept"
        assert "stays allocated" in out
    bc._release_graph_pool()  # decided: never again
    assert len(resets) == (1 if ids == ["a", "a"] else 0)


def test_release_waits_for_a_pending_backward(monkeypatch):
    from types import SimpleNamespace

    from torch._inductor import cudagraph_trees

    manager = _FakeManager(["a"], pending = True)
    resets = []
    monkeypatch.setattr(
        cudagraph_trees, "get_obj", lambda local, name: {0: SimpleNamespace(tree_manager = manager)}
    )
    monkeypatch.setattr(cudagraph_trees, "reset_cudagraph_trees", lambda: resets.append(1))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d = None: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    bc._CUDAGRAPH_STATE["ours"].add("a")
    bc._release_graph_pool()
    assert resets == [] and bc._CUDAGRAPH_STATE["release"] is None
    manager.running_forwards_with_pending_backwards = False
    bc._release_graph_pool()
    assert resets == [1] and bc._CUDAGRAPH_STATE["release"] == "done"


def test_block_marked_by_fsdp2_runs_without_graphs(monkeypatch):
    calls = []
    _tagging_compile(monkeypatch, calls)
    A = _linear_layer_class("MarkedDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    kept, marked = A().train(), A().train()
    marked.__dict__["_unsloth_block_graphs_off"] = True
    marked(kept(torch.randn(2, 4, 8, requires_grad = True))).sum().backward()
    assert calls == ["graphs", "plain"]
    assert bc._CUDAGRAPH_STATE["off"] is None


def test_graphed_call_records_the_compile_ids_it_created(monkeypatch):
    # The first call creates the tree (and its manager); later calls read the cached manager only.
    manager = _FakeManager([])
    lookups = []

    def tree_manager(device):
        lookups.append(device)
        return manager

    def compiled(n, cid):
        for _ in range(n):
            i = len(manager.ids_to_funcs)
            manager.ids_to_funcs[i] = None
            manager.id_to_compile_id[i] = cid
        return n

    monkeypatch.setattr(bc, "_tree_manager", tree_manager)
    manager.ids_to_funcs[0], manager.id_to_compile_id[0] = None, "foreign"  # made before us
    assert bc._graphed_call(compiled, 0, 0, "x") == 0
    bc._graphed_call(compiled, 0, 2, "a")
    bc._graphed_call(compiled, 0, 0, "y")
    bc._graphed_call(compiled, 0, 1, "b")
    assert lookups == [0]
    assert bc._CUDAGRAPH_STATE["ours"] == {"a", "b"}
    assert bc._pool_release_verdict([manager], bc._CUDAGRAPH_STATE["ours"]) == "foreign"


def test_grad_trial_of_a_checkpointed_block_takes_the_recompute_path(monkeypatch):
    # The first no-grad call (checkpoint forward) runs a grad-mode trial first; it stands in for the
    # recompute, so it copies like the recompute does.
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    seen = _counting_clones(monkeypatch)
    A = _linear_layer_class("TrialDecoderLayer")
    bc._wrap(A, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = A().train()
    with torch.no_grad():
        layer(torch.randn(2, 4, 8, requires_grad = True))
    assert seen == {"outputs": 2, "grad": 1}  # trial (outputs + input grad) and the no-grad call
    assert bc._IN_TRIAL == [False]


class FakeMoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 8, 8) / 8)

    def forward(self, x):
        return torch.einsum("btd,ed->bte", x, self.weight.sum(0))


class FakeSparseMoeBlock(torch.nn.Module):
    # transformers 4.57: ModuleList experts + num_experts.
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.experts = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])

    def forward(self, x):
        return sum(e(x) for e in self.experts)


class FakeRoutedMoe(torch.nn.Module):
    # transformers 4.57 DeepSeek-V2 / V3, GLM4-MoE: n_routed_experts, num_experts only on the gate.
    def __init__(self):
        super().__init__()
        self.n_routed_experts = 2
        self.gate = torch.nn.Linear(8, 2)
        self.gate.num_experts = 2
        self.experts = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])
        self.shared_experts = torch.nn.Linear(8, 8)

    def forward(self, x):
        return sum(e(x) for e in self.experts) + self.shared_experts(x)


@pytest.mark.parametrize("moe", [None, FakeMoeExperts, FakeSparseMoeBlock, FakeRoutedMoe])
def test_moe_layers_run_without_graphs(monkeypatch, moe):
    """Qwen3-MoE broke under block CUDA graphs (capture host copy / replay-overwritten routing)."""
    seen = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def inner(*args, **kw):
            seen.append(tag)
            return fn(*args, **kw)

        return inner

    monkeypatch.setattr(torch, "compile", fake_compile)

    class MoeDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.mlp = moe() if moe is not None else torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.mlp(self.proj(hidden_states))

    bc._wrap(MoeDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = MoeDecoderLayer().train()
    for _ in range(2):
        x = torch.randn(2, 16, 8, requires_grad = True)
        layer(x).sum().backward()
        assert x.grad is not None
    assert set(seen) == ({"graphs"} if moe is None else {"plain"})
    assert not layer.__dict__.get("_unsloth_block_grad_hooks") or moe is None


class BlockMask:
    """Stand-in for torch.nn.attention.flex_attention.BlockMask (matched by class name)."""

    def __init__(self):
        self.kv_num_blocks = torch.ones(1, 1, 1, dtype = torch.int32)


@pytest.mark.parametrize("mask", [None, "tensor", "block_mask"])
def test_flex_block_mask_calls_run_without_graphs(monkeypatch, mask):
    """Flex BlockMask calls re-record block graphs every step."""
    seen = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def inner(*args, **kw):
            seen.append(tag)
            return fn(*args, **kw)

        return inner

    monkeypatch.setattr(torch, "compile", fake_compile)

    class FlexDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
            **kwargs,
        ):
            return self.proj(hidden_states)

    bc._wrap(FlexDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    layer = FlexDecoderLayer().train()
    value = {None: None, "tensor": torch.ones(2, 1, 16, 16), "block_mask": BlockMask()}[mask]
    for _ in range(2):
        x = torch.randn(2, 16, 8, requires_grad = True)
        layer(x, attention_mask = value).sum().backward()
        assert x.grad is not None
    assert set(seen) == ({"plain"} if mask == "block_mask" else {"graphs"})
