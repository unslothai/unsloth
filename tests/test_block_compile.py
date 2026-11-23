# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One compiled region per decoder layer (unsloth/models/_block_compile.py).

CPU: sequence dims (not widths) are marked dynamic, rebound globals resolve, and a block that
fails to compile retries with flex_attention outside, then keeps the stock forward.
GPU: compiled Qwen3 and Qwen3.5 layers match the stock forward and gradients with no graph break
(Qwen3.5's fla kernel runs as the opaque op inside the block).
"""

import functools
import importlib.util
import os
import sys
import types

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    # Other test modules set these process-wide (e.g. UNSLOTH_COMPILE_DISABLE=1 at import).
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(key, raising = False)


def _load(name = "_block_compile"):
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(f"_u{name}", os.path.join(_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_mark_seq_dynamic_skips_widths(monkeypatch):
    M = _load()
    marked = []
    monkeypatch.setattr(
        torch._dynamo, "maybe_mark_dynamic", lambda t, d: marked.append((t.dim(), d))
    )
    L = 16
    hidden = torch.zeros(2, L, L)  # hidden size equal to the sequence length
    mask4d = torch.zeros(2, 1, L, L)
    mask2d = torch.zeros(2, L)
    mrope_ids = torch.zeros(3, 2, L, dtype = torch.long)
    M._mark_seq_dynamic(L, (hidden, {"m": mask4d}, [mask2d, "x", None], mrope_ids))
    assert sorted(marked) == [(2, 1), (3, 1), (3, 2), (4, 2), (4, 3)]


def test_mark_batch_dynamic_marks_dim0_only(monkeypatch):
    M = _load()
    marked = []
    monkeypatch.setattr(
        torch._dynamo, "maybe_mark_dynamic", lambda t, d: marked.append((t.dim(), d))
    )
    B, L = 4, 16
    hidden = torch.zeros(B, L, 8)
    mask4d = torch.zeros(B, 1, L, L)
    pos = torch.zeros(1, L, dtype = torch.long)  # broadcast over the batch: not marked
    scale = torch.zeros(B)  # 1D: not marked
    M._mark_batch_dynamic(B, (hidden, {"m": mask4d}, [pos, scale, "x", None]))
    assert sorted(marked) == [(3, 0), (4, 0)]


@pytest.mark.parametrize("dynamic_batch", ["1", "0"])
def test_batch_change_does_not_recompile(monkeypatch, dynamic_batch):
    """An uneven last batch recompiled every compiled block (+8 graphs, 40 s on a 4-bit Qwen3-0.6B, B200)."""
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE_DYNAMIC_BATCH", dynamic_batch)
    M = _load()
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend = counter, fullgraph = False, dynamic = None),
    )
    torch._dynamo.reset()

    class BatchDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
            position_ids = None,
            **kwargs,
        ):
            out = self.proj(hidden_states) + position_ids[..., None].to(hidden_states.dtype)
            return out * attention_mask[:, 0, :, :1]

    M._wrap(BatchDecoderLayer, {})
    layer = BatchDecoderLayer().train()

    def step(B, L):
        x = torch.randn(B, L, 8, requires_grad = True)
        mask = torch.ones(B, 1, L, L)
        pos = torch.arange(L)[None].expand(B, -1)
        layer(x, attention_mask = mask, position_ids = pos).sum().backward()

    step(4, 16)
    step(4, 24)  # sequence dim is dynamic: no recompile
    before = counter.frame_count
    step(3, 20)  # uneven last batch
    step(6, 12)
    extra = counter.frame_count - before
    torch._dynamo.reset()
    if dynamic_batch == "1":
        assert extra == 0
    else:
        assert extra >= 1  # the control: without the batch mark the batch change recompiles


class _MaskDecoderLayer(torch.nn.Module):
    """SDPA over a mask that transformers drops (None) for an unpadded micro batch."""

    def __init__(self, causal = True):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.self_attn = types.SimpleNamespace(is_causal = causal)
        self.causal = causal

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        **kwargs,
    ):
        q = self.proj(hidden_states)[:, None]
        if attention_mask is None:
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal = self.causal)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, q, q, attn_mask = attention_mask
            )
        return out[:, 0]


def _compile_counter(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend = counter, fullgraph = False, dynamic = None),
    )
    torch._dynamo.reset()
    return counter


def _padded_mask(
    lengths,
    L,
    causal = True,
):
    keep = torch.arange(L)[None] < torch.tensor(lengths)[:, None]
    base = torch.ones(L, L, dtype = torch.bool)
    if causal:
        base = base.tril()
    return base[None, None] & keep[:, None, None, :]


@pytest.mark.parametrize("causal", [True, False], ids = ["causal", "bidirectional"])
def test_unpadded_batch_reuses_the_masked_graph(monkeypatch, causal):
    """Padded vs unpadded micro batches must not flip mask/None and recompile."""
    M = _load()
    counter = _compile_counter(monkeypatch)
    M._wrap(_MaskDecoderLayer, {})
    layer = _MaskDecoderLayer(causal).train()
    reference = _MaskDecoderLayer(causal).train()
    reference.load_state_dict(layer.state_dict())

    def step(lengths, L, padded):
        x = torch.randn(len(lengths), L, 8)
        mask = _padded_mask(lengths, L, causal) if padded else None
        xs = x.clone().requires_grad_()
        out = layer(xs, attention_mask = mask)
        out.square().sum().backward()
        xr = x.clone().requires_grad_()
        ref = _MaskDecoderLayer.forward.__wrapped__(reference, xr, attention_mask = mask)
        ref.square().sum().backward()
        torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)
        torch.testing.assert_close(xs.grad, xr.grad, rtol = 1e-5, atol = 1e-5)

    step([16, 12], 16, True)
    step([20, 17], 20, True)
    before = counter.frame_count
    step([24, 24], 24, False)
    step([18, 9], 18, True)
    step([14, 14], 14, False)
    extra = counter.frame_count - before
    torch._dynamo.reset()
    assert extra == 0


def test_mask_rebuilt_only_after_a_mask_was_seen(monkeypatch):
    M = _load()
    layer = _MaskDecoderLayer().train()
    hidden = torch.zeros(2, 6, 8)
    assert M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"] is None
    layer.self_attn = types.SimpleNamespace()
    M._normalise_mask(layer, hidden, {"attention_mask": _padded_mask([6, 4], 6)})
    assert M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"] is None
    M._normalise_mask(layer, hidden, {"attention_mask": torch.tensor([[1] * 6, [1] * 4 + [0] * 2])})
    rebuilt = M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"]
    assert rebuilt.dtype == torch.long and rebuilt.shape == (2, 6) and bool((rebuilt == 1).all())
    # transformers expands its 4D mask (a view) and dynamo guards the base: the rebuilt mask is a view too.
    layer.self_attn = types.SimpleNamespace(is_causal = True)
    M._normalise_mask(
        layer, hidden, {"attention_mask": _padded_mask([6, 4], 6).expand(2, -1, 6, 6)}
    )
    rebuilt = M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"]
    assert rebuilt._base is not None and rebuilt._base.shape == rebuilt.shape
    assert torch.equal(rebuilt, torch.ones(6, 6, dtype = torch.bool).tril().expand(2, 1, 6, 6))
    M._normalise_mask(layer, hidden, {"attention_mask": _padded_mask([6, 4], 6).contiguous()})
    assert (
        M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"]._base is None
    )
    padding = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])[:, None, None, :].clone()
    for causal, want in ((False, torch.ones(2, 1, 6, 6, dtype = torch.bool)), (True, None)):
        layer = _MaskDecoderLayer().train()
        layer.self_attn = types.SimpleNamespace(is_causal = causal)
        M._MATERIALISED[0] = None
        M._normalise_mask(layer, hidden, {"attention_mask": padding.expand(2, 1, 6, 6)})
        rebuilt = M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"]
        assert rebuilt is None if want is None else torch.equal(rebuilt, want)
    # Batch size 1: the rebuilt causal mask owns its storage (no 2D base dynamo would guard as a view).
    layer = _MaskDecoderLayer().train()
    layer.self_attn = types.SimpleNamespace(is_causal = True)
    M._MATERIALISED[0] = None
    one = torch.zeros(1, 6, 8)
    M._normalise_mask(layer, one, {"attention_mask": _padded_mask([6], 6).contiguous()})
    rebuilt = M._normalise_mask(layer, one, {"attention_mask": None})["attention_mask"]
    assert rebuilt._base is None and torch.equal(
        rebuilt, torch.ones(6, 6, dtype = torch.bool).tril()[None, None]
    )
    assert (
        M._normalise_mask(layer, one, {"attention_mask": None, "is_causal": False})[
            "attention_mask"
        ]
        is None
    )
    layer = _MaskDecoderLayer().train()
    M._MATERIALISED[0] = None
    M._normalise_mask(layer, hidden, {"attention_mask": torch.tensor([[1], [0]]).expand(2, 6)})
    rebuilt = M._normalise_mask(layer, hidden, {"attention_mask": None})["attention_mask"]
    assert torch.equal(rebuilt, torch.ones(2, 6, dtype = torch.long))


def test_rebind_globals_resolves():
    M = _load()
    ns = {"__name__": "not_a_registered_module", "OFFSET": 3}
    exec("def f(x):\n    return x + OFFSET\n", ns)
    f = M._rebind_globals(ns["f"])
    assert f is not ns["f"] and f(1) == 4
    assert sys.modules[f.__module__].__dict__ is f.__globals__
    g = M._rebind_globals(f)
    assert g is f  # already owned by a registered module
    exec("def h():\n    global OFFSET\n    OFFSET += 1\n", ns)
    assert M._rebind_globals(ns["h"]) is ns["h"]  # a copied namespace would miss the write


class _Boom(torch._dynamo.exc.TorchDynamoException):
    pass


def _toy_class():
    class ToyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            return self.proj(hidden_states) + hidden_states

    return ToyDecoderLayer


@pytest.mark.parametrize("fails_in", [("full",), ("full", "split")])
def test_compile_failure_falls_back(monkeypatch, fails_in):
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            mode = {M._FULL: "full", M._SPLIT: "split"}[M._IN_BLOCK[0]]
            calls.append(mode)
            if mode in fails_in:
                raise _Boom("lowering failed")
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    original = cls.forward
    M._wrap(cls, {})
    layer = cls()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        want = original(layer, x)
        got = layer(x)
        torch.testing.assert_close(got, want)
        layer(x)
    if fails_in == ("full",):
        assert layer._unsloth_block_mode == M._SPLIT
        # Grad-mode trial (full fails, split), then the no-grad call twice.
        assert calls == ["full", "split", "split", "split"]
    else:
        assert layer._unsloth_block_mode == M._OFF
        assert calls == ["full", "split"]  # then the stock forward, never compiled again
    # A sibling of the same kind starts where this layer ended, without retrying what failed.
    calls.clear()
    with torch.no_grad():
        cls()(x)
    assert calls == (["split", "split"] if fails_in == ("full",) else [])
    # Eval keeps the stock forward (logits equal transformers bit for bit).
    calls.clear()
    sibling = cls().eval()
    sibling(x)
    assert calls == []


@pytest.mark.parametrize(
    "fullgraph, suppress",
    [(True, False), (False, False), (False, True)],
    ids = ["fullgraph", "default", "suppress_errors"],
)
def test_outer_compile_owns_the_block(monkeypatch, fullgraph, suppress):
    """Enclosing compile: no maybe_mark_dynamic or nested compile, one graph, stock output and grads."""
    from torch._dynamo.utils import counters

    M = _load()
    real_compile = torch.compile
    inner = []

    def eager_compile(fn, **kwargs):
        compiled = real_compile(fn, backend = "eager", fullgraph = False, dynamic = None)

        def call(*args, **kw):
            inner.append(1)
            return compiled(*args, **kw)

        return call

    monkeypatch.setattr(torch, "compile", eager_compile)
    cls = _toy_class()
    original = cls.forward
    M._wrap(cls, {})
    monkeypatch.setattr(torch, "compile", real_compile)
    torch.manual_seed(0)
    layer = cls().train()
    x = torch.randn(2, 4, 8)

    xr = x.clone().requires_grad_()
    original(layer, xr).sum().backward()
    want = [xr.grad] + [p.grad.clone() for p in layer.parameters()]
    layer.zero_grad(set_to_none = True)

    torch._dynamo.reset()
    counters.clear()
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", suppress)
    outer = torch.compile(layer, backend = "aot_eager", fullgraph = fullgraph)
    xc = x.clone().requires_grad_()
    outer(xc).sum().backward()
    got = [xc.grad] + [p.grad for p in layer.parameters()]
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w)
    assert inner == []
    assert dict(counters["graph_break"]) == {}
    assert counters["stats"]["unique_graphs"] == 1
    torch._dynamo.reset()

    layer(x.clone().requires_grad_()).sum().backward()
    assert inner


@pytest.mark.parametrize("fullgraph", [True, False], ids = ["fullgraph", "default"])
def test_compiled_region_flag_only_in_own_block(monkeypatch, fullgraph):
    """_IN_COMPILED_REGION marks Unsloth's own block compile (a graph break is legal there), not the block
    traced inside a user's compile, which also sets _IN_BLOCK (unsloth_zoo's sparse flex reads it)."""
    M = _load()
    real_compile = torch.compile

    class ToyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states):
            # Flags read where the zoo reads them: inside the traced layer.
            in_block = float(M._IN_BLOCK[0] != M._OFF)
            region = float(M._IN_COMPILED_REGION[0])
            return self.proj(hidden_states) + in_block + 10 * region

    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend = "eager", fullgraph = False, dynamic = None),
    )
    original = ToyDecoderLayer.forward
    M._wrap(ToyDecoderLayer, {})
    monkeypatch.setattr(torch, "compile", real_compile)
    torch.manual_seed(0)
    layer = ToyDecoderLayer().train()
    x = torch.randn(2, 4, 8)
    stock = original(layer, x).detach()

    torch._dynamo.reset()
    torch.testing.assert_close(layer(x).detach(), stock + 11)  # Unsloth's block: both flags
    torch._dynamo.reset()
    outer = torch.compile(layer, backend = "eager", fullgraph = fullgraph)
    torch.testing.assert_close(outer(x).detach(), stock + 1)  # user's compile: _IN_BLOCK only
    torch._dynamo.reset()
    assert M._IN_BLOCK[0] == M._OFF and M._IN_COMPILED_REGION[0] is False


def _install_blocks(M, model, monkeypatch):
    """patch_block_compile on CPU with an eager inner compile, restored at teardown."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for cls in {type(m) for m in model.modules()}:
        for key, value in list(cls.__dict__.items()):
            if callable(value):
                monkeypatch.setattr(cls, key, value)
    for key in ("sdpa", "flex_attention"):
        if key in ALL_ATTENTION_FUNCTIONS:
            monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, key, ALL_ATTENTION_FUNCTIONS[key])
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    monkeypatch.setattr(config, name, getattr(config, name))
    real_compile = torch.compile
    inner = []

    def eager_compile(fn, **kwargs):
        compiled = real_compile(fn, backend = "eager", fullgraph = False, dynamic = None)

        def call(*args, **kw):
            inner.append(1)
            return compiled(*args, **kw)

        return call

    real_available = torch.cuda.is_available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # the install gate only
    monkeypatch.setattr(torch, "compile", eager_compile)
    assert M.patch_block_compile(model)
    monkeypatch.setattr(torch, "compile", real_compile)
    monkeypatch.setattr(torch.cuda, "is_available", real_available)
    return inner


@pytest.mark.parametrize("fp8_first", [False, True])
def test_fp8_precision_options_are_per_instance_and_reused(monkeypatch, fp8_first):
    import copy

    M = _load()

    class PrecisionDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x)

    plain, fp8 = PrecisionDecoderLayer(), PrecisionDecoderLayer()
    fp8.proj._unsloth_fp8_lora = object()
    options = []

    def compile(fn, **kwargs):
        precision = kwargs["options"].get("emulate_precision_casts", False)
        options.append(precision)

        def run(*args, **kw):
            return fn(*args, **kw) + int(precision)

        return run

    monkeypatch.setattr(torch, "compile", compile)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_layers", lambda model: [(model, True)])
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    monkeypatch.setattr(M, "_zero3_partitioned", lambda *args: False)
    original = PrecisionDecoderLayer.forward
    for layer in (fp8, plain) if fp8_first else (plain, fp8):
        M.patch_block_compile(layer)
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(plain(x), original(plain, x))
    torch.testing.assert_close(fp8(x), original(fp8, x) + 1)
    assert "forward" not in plain.__dict__
    assert not M._COMPILE_OPTIONS.get("emulate_precision_casts", False)
    assert options == [False, True]
    forward = fp8.forward
    M.patch_block_compile(fp8)
    assert fp8.forward is forward and options == [False, True]
    twin = copy.deepcopy(fp8)
    assert twin.forward.__self__ is twin
    torch.testing.assert_close(twin(x), original(twin, x) + 1)
    fp8.eval()
    torch.testing.assert_close(fp8(x), original(fp8, x))


def _outer_loss(
    model,
    ids,
    fullgraph,
    backend = "aot_eager",
):
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    model.zero_grad(set_to_none = True)
    loss = torch.compile(model, backend = backend, fullgraph = fullgraph)(
        input_ids = ids, labels = ids, use_cache = False
    ).loss
    loss.backward()
    return loss.detach(), dict(counters["graph_break"]), counters["stats"]["unique_graphs"]


def _stock_modeling(monkeypatch, name):
    """Reload stock modeling classes: zoo patches swap Qwen MoE experts for a compiler-disabled loop."""
    import importlib

    module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
    saved = dict(module.__dict__)
    importlib.reload(module)

    def restore():
        module.__dict__.clear()
        module.__dict__.update(saved)

    return module, restore


def _tiny_qwen3_moe(request, monkeypatch):
    from transformers import AutoModelForCausalLM

    _, restore = _stock_modeling(monkeypatch, "qwen3_moe")
    request.addfinalizer(restore)
    try:
        # batched_mm: the stock grouped_mm / eager expert loops break on their own (data dependent).
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            dtype = torch.float32,
            experts_implementation = "batched_mm",
        )
    except Exception as e:  # offline without the cached repo
        pytest.skip(f"tiny Qwen3-MoE unavailable: {e}")
    return model.train()


@pytest.mark.parametrize("fullgraph", [True, False], ids = ["fullgraph", "default"])
def test_outer_compile_traces_decode_aware_moe(request, monkeypatch, fullgraph):
    """Qwen3-MoE sparse block is disable_unless_decode: an enclosing compile needs the same flag."""
    import unsloth_zoo.temporary_patches.utils as zoo_utils

    M = _load()
    torch.manual_seed(0)
    model = _tiny_qwen3_moe(request, monkeypatch)
    ids = torch.randint(0, 100, (2, 8))
    want = model(input_ids = ids, labels = ids, use_cache = False).loss.detach()
    block = type(model.model.layers[0].mlp)
    monkeypatch.setattr(
        block,
        "forward",
        zoo_utils.torch_compiler_disable_unless_decode(recursive = True)(block.forward),
    )
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    inner = _install_blocks(M, model, monkeypatch)
    assert not any(l.__dict__.get("_unsloth_block_moe_may_move") for l in model.model.layers)
    loss, breaks, graphs = _outer_loss(model, ids, fullgraph)
    torch.testing.assert_close(loss, want)
    assert breaks == {} and graphs == 1 and inner == []
    assert zoo_utils.UNSLOTH_DECODE_COMPILE[0] is False and M._IN_BLOCK[0] == M._OFF
    torch._dynamo.reset()


def test_outer_compile_keeps_selective_experts_eager(request, monkeypatch):
    """Selective dequant under an enclosing compile still moves 4-bit experts out (one break)."""
    M = _load()
    torch.manual_seed(0)
    model = _tiny_qwen3_moe(request, monkeypatch)
    ids = torch.randint(0, 100, (2, 8))
    want = model(input_ids = ids, labels = ids, use_cache = False).loss.detach()

    class Params4bit(torch.nn.Parameter):
        pass

    experts_cls = type(model.model.layers[0].mlp.experts)
    for layer in model.model.layers:
        experts = layer.mlp.experts
        experts._parameters["gate_up_proj"] = Params4bit(experts.gate_up_proj.data)
    seen = []
    original = experts_cls.forward

    def spy(self, *args, **kwargs):
        seen.append(torch.compiler.is_compiling())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(experts_cls, "forward", spy)
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1")
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    inner = _install_blocks(M, model, monkeypatch)
    loss, breaks, _ = _outer_loss(model, ids, fullgraph = False)
    torch.testing.assert_close(loss, want)
    assert seen and not any(seen) and inner == []
    assert all(l.__dict__["_unsloth_block_moe_done"] for l in model.model.layers)
    assert all(getattr(l.mlp.experts, "_unsloth_block_outside", False) for l in model.model.layers)
    assert all(l.__dict__.get("_unsloth_block_moe_may_move") for l in model.model.layers)
    torch._dynamo.reset()


def _fla_present():
    import importlib.util
    import sys

    if sys.modules.get("fla") is not None:
        return True
    try:
        return importlib.util.find_spec("fla") is not None
    except (ImportError, ValueError):
        return False


def test_outer_compile_prepares_gdn_eagerly(request, monkeypatch):
    """GDN first reached under an enclosing compile runs _prepare_gdn eagerly; later traces stay whole."""
    from transformers import AutoModelForCausalLM

    M = _load()
    _, restore = _stock_modeling(monkeypatch, "qwen3_next")
    request.addfinalizer(restore)
    torch.manual_seed(0)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-Qwen3NextForCausalLM",
            dtype = torch.float32,
            experts_implementation = "batched_mm",
        ).train()
    except Exception as e:
        pytest.skip(f"tiny Qwen3-Next unavailable: {e}")
    if _fla_present():
        # The stock GDN op then resolves to fla (installed or vendored), which is compiler-disabled.
        pytest.skip("stock GDN op resolves to fla")
    ids = torch.randint(0, 100, (2, 8))
    want = model(input_ids = ids, labels = ids, use_cache = False).loss.detach()
    calls = []
    prepare = M._prepare_gdn

    def spy(layer, hidden_states):
        calls.append(torch.compiler.is_compiling())
        return prepare(layer, hidden_states)

    monkeypatch.setattr(M, "_prepare_gdn", spy)
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    inner = _install_blocks(M, model, monkeypatch)
    assert any(hasattr(l, "linear_attn") for l in model.model.layers)
    assert M._GDN_OP[0] is None
    # backend "eager": stock Qwen3-Next fails aot_eager on CPU with or without blocks (a GDN view).
    loss, _, _ = _outer_loss(model, ids, fullgraph = False, backend = "eager")
    torch.testing.assert_close(loss, want)
    assert calls == [False] and M._GDN_OP[0] is not None and inner == []
    loss, breaks, graphs = _outer_loss(model, ids, fullgraph = True, backend = "eager")
    torch.testing.assert_close(loss, want)
    assert breaks == {} and graphs == 1 and calls == [False]
    torch._dynamo.reset()


def test_stop_recomputation_does_not_demote(monkeypatch):
    """#14063: checkpoint early-stop is control flow; demoting on it raised CheckpointError."""
    from torch.utils.checkpoint import _StopRecomputationError

    M = _load()

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            raise _StopRecomputationError()

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()
    with pytest.raises(_StopRecomputationError):
        layer(torch.randn(2, 4, 8, requires_grad = True))
    assert layer._unsloth_block_mode == M._FIRST_MODE
    assert layer.__dict__.get("_unsloth_block_checked", 0) == 0


def test_cache_calls_keep_the_stock_forward(monkeypatch):
    from transformers.cache_utils import DynamicCache

    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    class CacheLayerDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            past_key_value = None,
            **kwargs,
        ):
            return self.proj(hidden_states)

    M._wrap(CacheLayerDecoderLayer, {})
    layer = CacheLayerDecoderLayer()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        layer(x, past_key_value = DynamicCache())
        layer(x, DynamicCache())
        layer(x, past_key_values = DynamicCache())
    assert calls == []
    with torch.no_grad():
        layer(x)
    assert calls  # no cache: compiled


@pytest.mark.parametrize("key", ["packed_seq_lengths", "prefix_seg_info"])
def test_unsloth_host_metadata_calls_keep_the_stock_forward(monkeypatch, key):
    """Host-read packed lengths / PrefixGrouper tables recompiled a block at every new length."""
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    class UnslothDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    UnslothDecoderLayer.forward.__module__ = "unsloth.models.llama"
    M._wrap(UnslothDecoderLayer, {})
    layer = UnslothDecoderLayer()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        layer(x, **{key: object()})
    assert calls == []
    with torch.no_grad():
        layer(x, **{key: None})
    assert calls  # no host metadata: compiled


def test_prefix_grouper_table_keeps_the_stock_forward_on_transformers_layers(monkeypatch):
    """prefix_grouper_hf passes the table through transformers-native layers too (Qwen3-MoE)."""
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    class Qwen3MoeDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    Qwen3MoeDecoderLayer.forward.__module__ = "transformers.models.qwen3_moe.modeling_qwen3_moe"
    M._wrap(Qwen3MoeDecoderLayer, {})
    layer = Qwen3MoeDecoderLayer().train()
    x = torch.randn(1, 4, 8)
    layer(x, prefix_seg_info = object())
    assert calls == []
    layer(x, prefix_seg_info = None)
    assert calls  # no table: compiled


def test_packed_stock_forward_says_blocks_run_uncompiled(monkeypatch, capsys):
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)

    class PackedDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    PackedDecoderLayer.forward.__module__ = "unsloth.models.llama"
    M._wrap(PackedDecoderLayer, {})
    layer = PackedDecoderLayer().train()
    x = torch.randn(1, 4, 8)
    layer(x)
    layer(x, packed_seq_lengths = None)
    assert "uncompiled" not in capsys.readouterr().out
    layer(x, packed_seq_lengths = torch.tensor([2, 2]))
    layer(x, packed_seq_lengths = torch.tensor([1, 3]))
    out = capsys.readouterr().out
    assert out.count("packed / padding-free rows run each transformer block uncompiled") == 1


def test_compiled_block_flags_packed_attention(monkeypatch):
    """hf_packed_attention runs packed rows outside the graph only while a block's compiled call
    runs: set there, restored after, never on the stock forward."""
    M = _load()
    packed_mod = types.ModuleType(M._PACKED_MODULE)
    packed_mod.IN_COMPILED_BLOCK = [False]
    monkeypatch.setitem(sys.modules, M._PACKED_MODULE, packed_mod)
    seen = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            seen.append(packed_mod.IN_COMPILED_BLOCK[0])
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        layer(x)
    assert seen and all(seen)
    assert packed_mod.IN_COMPILED_BLOCK == [False]
    seen.clear()
    layer.eval()(x)  # stock forward
    assert seen == [] and packed_mod.IN_COMPILED_BLOCK == [False]
    # Without the module (never imported) the block still runs.
    monkeypatch.delitem(sys.modules, M._PACKED_MODULE)
    layer.train()
    with torch.no_grad():
        layer(x)
    assert M._NO_PACKED == [False]


def test_gemma4_band_fast_path_in_blocks(monkeypatch):
    M = _load()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    scans, routed, plain_calls = [], [], []
    band = types.ModuleType("fake_banded_attention")
    band._mask_is_plain_band = lambda mask, S, w: scans.append((S, w)) or True
    zoo = types.ModuleType("fake_flash_sliding")

    def plain(module, q, k, v, mask, **kw):
        plain_calls.append(mask)
        return "plain", None

    def router(module, q, k, v, mask, **kw):
        routed.append(mask)
        return "router", None

    router._unsloth_gemma4_flash = True
    router.__module__ = zoo.__name__
    zoo._ORIG_SDPA = [plain]
    monkeypatch.setitem(sys.modules, band.__name__, band)
    monkeypatch.setitem(sys.modules, zoo.__name__, zoo)
    monkeypatch.setattr(M, "_BAND_MODULE", band.__name__)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", router)
    M._gemma4_router_hook()
    wrapped = ALL_ATTENTION_FUNCTIONS["sdpa"]
    assert wrapped is not router
    # Carries zoo's marker, so zoo's patch on a later model load does not wrap it again
    # with it as the fallback (that loop recursed forever).
    assert wrapped._unsloth_gemma4_flash
    M._gemma4_router_hook()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is wrapped

    class Gemma4TextAttention(torch.nn.Module):
        is_sliding, sliding_window, is_causal = True, 64, True

    layer = torch.nn.Module()
    layer.self_attn = attn = Gemma4TextAttention()
    short, long = torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 96, 96)
    M._precheck_band(layer, 32, (), {"attention_mask": short})
    M._precheck_band(layer, 96, (long,), {})
    assert scans == [(32, 64)]  # longer than the window: never scanned
    assert (short._unsloth_block_band, long._unsloth_block_band) == (True, False)

    q = torch.randn(1, 2, 32, 16)
    assert wrapped(attn, q, q, q, short)[0] == "router"  # eager: the stock router
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    M._IN_BLOCK[0] = M._FULL
    try:
        out, _ = wrapped(attn, q, q, q, short, scaling = 0.25)
        want = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal = True, scale = 0.25)
        torch.testing.assert_close(out, want.transpose(1, 2))
        ql = torch.randn(1, 2, 96, 16)
        assert wrapped(attn, ql, ql, ql, long)[0] == "plain"
    finally:
        M._IN_BLOCK[0] = M._OFF
    assert routed == [short] and plain_calls == [long]


def _fake_gemma4_router_stack(M, monkeypatch, scans):
    """zoo's gemma-4 router wrapped by another router from a module without _ORIG_SDPA (unsloth's
    head_dim 512 kernel is installed on top at load, before the block hook runs)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    band = types.ModuleType("fake_banded_attention_2")
    band._mask_is_plain_band = lambda mask, S, w: scans.append((S, w)) or bool(
        getattr(mask, "_ok", True)
    )
    zoo = types.ModuleType("fake_flash_sliding_2")
    top_mod = types.ModuleType("fake_large_head_2")
    calls = {"plain": [], "top": []}

    def plain(module, q, k, v, mask, **kw):
        calls["plain"].append(mask)
        return "plain", None

    def top(module, q, k, v, mask, **kw):
        calls["top"].append(mask)
        return "top", None

    top._unsloth_gemma4_flash = True  # forwarded sentinels, like the real wrappers do
    top.__module__ = top_mod.__name__
    zoo._ORIG_SDPA = [plain]
    for mod in (band, zoo, top_mod):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(M, "_BAND_MODULE", band.__name__)
    monkeypatch.setattr(M, "_FLASH_MODULE", zoo.__name__)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", top)
    return calls, top


def test_gemma4_hook_installs_beneath_a_wrapping_router(monkeypatch):
    M = _load()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    scans = []
    calls, top = _fake_gemma4_router_stack(M, monkeypatch, scans)
    M._gemma4_router_hook()
    wrapped = ALL_ATTENTION_FUNCTIONS["sdpa"]
    assert wrapped is not top and wrapped._unsloth_block_band and M._BAND_CHECK[0] is not None


def test_gemma4_global_layer_mask_gets_a_causal_verdict(monkeypatch):
    M = _load()
    scans = []
    _fake_gemma4_router_stack(M, monkeypatch, scans)
    M._gemma4_router_hook()

    class Gemma4TextAttention(torch.nn.Module):
        is_sliding, sliding_window, head_dim = False, None, 512

    layer = torch.nn.Module()
    layer.self_attn = Gemma4TextAttention()
    plain_causal, padded = torch.zeros(1, 1, 96, 96), torch.zeros(1, 1, 96, 96)
    padded._ok = False
    M._precheck_band(layer, 96, (plain_causal,), {})
    M._precheck_band(layer, 96, (), {"attention_mask": padded})
    assert scans == [(96, 96), (96, 96)]
    assert plain_causal._unsloth_plain_causal is True and padded._unsloth_plain_causal is False
    assert plain_causal._unsloth_block_long_band is False


def _sliding_layer(window = 64, head_dim = 256):
    class Gemma4TextAttention(torch.nn.Module):
        is_sliding, sliding_window, is_causal, training = True, window, True, True

    attn = Gemma4TextAttention()
    attn.head_dim = head_dim
    layer = torch.nn.Module()
    layer.self_attn = attn
    return layer, attn


def test_gemma4_long_band_verdict_needs_the_kernel(monkeypatch):
    M = _load()
    scans = []
    _fake_gemma4_router_stack(M, monkeypatch, scans)
    M._gemma4_router_hook()
    layer, _ = _sliding_layer()
    hidden = torch.zeros(1, 96, 8, dtype = torch.bfloat16)
    ready = []
    monkeypatch.setattr(M, "_band_kernel_ready", lambda layer, h: ready.append(h) or True)
    long = torch.zeros(1, 1, 96, 96)
    M._precheck_band(layer, 96, (long,), {}, hidden)
    assert scans == [(96, 64)] and long._unsloth_block_long_band is True
    assert long._unsloth_block_band is False and long._unsloth_plain_causal is False
    monkeypatch.setattr(M, "_band_kernel_ready", lambda layer, h: False)
    other = torch.zeros(1, 1, 96, 96)
    M._precheck_band(layer, 96, (other,), {}, hidden)
    assert other._unsloth_block_long_band is False and scans == [(96, 64)]
    # Within the window the existing is_causal route decides; no long-band scan.
    short = torch.zeros(1, 1, 32, 32)
    M._precheck_band(layer, 32, (short,), {}, hidden)
    assert short._unsloth_block_long_band is False and short._unsloth_block_band is True


def test_gemma4_band_kernel_kill_switch(monkeypatch):
    M = _load()
    layer, _ = _sliding_layer()
    monkeypatch.setenv("UNSLOTH_BLOCK_SLIDING_KERNEL", "0")
    assert M._band_kernel_ready(layer, torch.zeros(1, 4, 8)) is False
    monkeypatch.delenv("UNSLOTH_BLOCK_SLIDING_KERNEL")
    assert M._band_kernel_ready(layer, torch.zeros(1, 4, 8)) is False  # CPU tensor


def test_gemma4_long_band_routes_to_the_interval_kernel(monkeypatch):
    M = _load()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    scans = []
    calls, _ = _fake_gemma4_router_stack(M, monkeypatch, scans)
    M._gemma4_router_hook()
    wrapped = ALL_ATTENTION_FUNCTIONS["sdpa"]
    _, attn = _sliding_layer(window = 16, head_dim = 32)
    kernel = []
    monkeypatch.setattr(
        M,
        "_band_kernel_attention",
        lambda q, k, v, w, s: kernel.append((q.shape, w, s)) or "kernel",
    )
    monkeypatch.setattr(M, "_band_kernel_checked", lambda q: True)
    q = torch.randn(1, 4, 48, 32, dtype = torch.bfloat16)
    kv = torch.randn(1, 2, 48, 32, dtype = torch.bfloat16)
    mask = torch.zeros(1, 1, 48, 48)
    mask._unsloth_block_long_band = True
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    M._IN_BLOCK[0] = M._FULL
    try:
        assert wrapped(attn, q, kv, kv, mask, scaling = 1.0)[0] == "kernel"
        assert kernel == [(q.shape, 16, 1.0)]
        mask._unsloth_block_long_band = False
        assert wrapped(attn, q, kv, kv, mask, scaling = 1.0)[0] == "plain"
        mask._unsloth_block_long_band = True
        assert (
            wrapped(attn, q, kv, kv, mask, dropout = 0.1, scaling = 1.0)[0] == "plain"
        )  # training dropout
        assert wrapped(attn, q.float(), kv.float(), kv.float(), mask, scaling = 1.0)[0] == "plain"
    finally:
        M._IN_BLOCK[0] = M._OFF
    assert len(kernel) == 1 and len(calls["plain"]) == 3


def test_gemma4_long_band_past_int32_keeps_the_plain_route(monkeypatch):
    M = _load()
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    scans = []
    calls, _ = _fake_gemma4_router_stack(M, monkeypatch, scans)
    M._gemma4_router_hook()
    wrapped = ALL_ATTENTION_FUNCTIONS["sdpa"]
    _, attn = _sliding_layer(window = 16, head_dim = 32)
    monkeypatch.setattr(M, "_band_kernel_attention", lambda q, k, v, w, s: "kernel")
    monkeypatch.setattr(M, "_band_kernel_checked", lambda q: True)
    import importlib

    lh = importlib.import_module(M._LH_MODULE)
    q = torch.randn(1, 4, 48, 32, dtype = torch.bfloat16)
    kv = torch.randn(1, 2, 48, 32, dtype = torch.bfloat16)
    mask = torch.zeros(1, 1, 48, 48)
    mask._unsloth_block_long_band = True
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    M._IN_BLOCK[0] = M._FULL
    try:
        assert wrapped(attn, q, kv, kv, mask, scaling = 1.0)[0] == "kernel"
        monkeypatch.setattr(lh, "_INT32_MAX", q.numel() - 1)
        assert wrapped(attn, q, kv, kv, mask, scaling = 1.0)[0] == "plain"
    finally:
        M._IN_BLOCK[0] = M._OFF


def test_kill_switch(monkeypatch):
    M = _load()
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE", "0")
    assert M.patch_block_compile(torch.nn.Module()) == []
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE", "1")
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    assert not M._enabled()
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    assert M._enabled()
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    assert not M._torch_supported()
    monkeypatch.setattr(torch, "__version__", "2.7.1+cu126")
    assert M._torch_supported()


@pytest.mark.parametrize("version", ["2.6.0+cu124", "2.5.1"])
def test_old_torch_installs_nothing(monkeypatch, version):
    """#10744: torch < 2.7 keeps every layer on the stock forward (no class wrapped, nothing compiled)."""
    M = _load()
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(1) or fn)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "__version__", version)
    cls = _toy_class()
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([cls(), cls()])
    original = cls.forward
    assert M.patch_block_compile(model) == []
    assert cls.forward is original
    model.layers[0](torch.randn(2, 4, 8, requires_grad = True)).sum().backward()
    assert calls == []
    monkeypatch.setattr(torch, "__version__", "2.7.1+cu126")
    assert M.patch_block_compile(model) == [cls.__name__]


def test_forced_float32_is_decided_per_model(monkeypatch):
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)
    cls = _toy_class()

    def model_of(forced):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([cls(), cls()])
        if forced is not None:
            model._unsloth_forced_float32 = forced
        return model

    # Forced by this load's env (the stamp lands after the first post_patch_model).
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    forced = model_of(None)
    assert M.patch_block_compile(forced) == []
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    plain = model_of(False)
    assert M.patch_block_compile(plain) == [cls.__name__]
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        forced.layers[0](x)
    assert calls == []
    with torch.no_grad():
        plain.layers[0](x)
    assert len(calls) == 2  # grad-mode trial + the call
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    with torch.no_grad():
        plain.layers[1](x)
    assert len(calls) == 4
    # get_peft_model re-patches a wrapper after another load reset the env: the stamp holds.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")

    class Delegating(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    forced._unsloth_forced_float32 = True
    wrapper = torch.nn.Module()
    wrapper.base = forced
    M.patch_block_compile(wrapper)
    M.patch_block_compile(Delegating(forced))
    calls.clear()
    with torch.no_grad():
        forced.layers[0](x)
    assert calls == []


def test_checkpoint_recompute_keeps_the_no_grad_mode(monkeypatch):
    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            mode = {M._FULL: "full", M._SPLIT: "split"}[M._IN_BLOCK[0]]
            grad = torch.is_grad_enabled()
            calls.append((mode, grad))
            if grad:
                torch.rand(3)
                if mode == "full":
                    raise _Boom("grad-mode lowering failed")
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()
    x = torch.randn(2, 4, 8)
    rng = torch.get_rng_state()
    with torch.no_grad():
        layer(x)
    # The grad-mode trial demoted first, so the no-grad half already ran in SPLIT.
    assert calls == [("full", True), ("split", True), ("split", False)]
    assert torch.equal(torch.get_rng_state(), rng)
    calls.clear()
    layer(x.requires_grad_())
    assert calls == [("split", True)]


def test_failed_attempt_stays_out_of_a_non_reentrant_checkpoint(monkeypatch):
    """A SPLIT attempt that ran part of the block, then failed, packed its tensors into the checkpoint:
    CheckpointError "saved during forward: 94, during recomputation: 70" (tiny Gemma3, head_dim 8)."""
    from torch.utils.checkpoint import checkpoint

    M = _load()

    def fake_compile(fn, **kwargs):
        def compiled(self, hidden_states, *args, **kw):
            if M._IN_BLOCK[0] == M._SPLIT:
                self.proj(hidden_states * 2).sum()  # ran compiled, then the next graph fails
            raise _Boom("lowering failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    original = cls.forward
    M._wrap(cls, {})
    torch.manual_seed(0)
    layer, ref = cls(), cls()
    ref.load_state_dict(layer.state_dict())
    x = torch.randn(2, 4, 8, requires_grad = True)
    checkpoint(layer, x, use_reentrant = False).square().sum().backward()
    assert layer._unsloth_block_mode == M._OFF
    x_ref = x.detach().requires_grad_()
    original(ref, x_ref).square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(layer.proj.weight.grad, ref.proj.weight.grad)


def _record_proj(layer, style, recorded):
    if style == "hook":
        from transformers.utils.output_capturing import _active_collector

        def hook(module, args, output):
            collected = _active_collector.get()
            if collected is not None:
                collected["router_logits"].append(output)

        layer.proj.register_forward_hook(hook)
        return _active_collector.set({"router_logits": recorded})

    def make_capture_wrapper(orig):
        @functools.wraps(orig)
        def wrapped_forward(*args, **kwargs):
            out = orig(*args, **kwargs)
            recorded.append(out)
            return out

        return wrapped_forward

    layer.proj.forward = make_capture_wrapper(layer.proj.forward)
    return None


@pytest.mark.parametrize("style", ["hook", "forward"])
def test_trial_forward_is_not_recorded_as_a_model_output(monkeypatch, style):
    # Qwen3-MoE tiny: trial forward added an extra router_logits entry (grad norm 1.04 vs 1.21).
    from torch.utils.checkpoint import checkpoint

    if style == "hook":
        pytest.importorskip("transformers.utils.output_capturing")
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()
    recorded = []
    token = _record_proj(layer, style, recorded)
    try:
        checkpoint(layer, torch.randn(2, 4, 8, requires_grad = True), use_reentrant = False)
    finally:
        if token is not None:
            from transformers.utils.output_capturing import _active_collector
            _active_collector.reset(token)
    assert len(recorded) == 1, f"{len(recorded)} recorded outputs for one forward"


class _TraceableCollector:
    """unsloth_zoo's compiled-code mirror of transformers' collector: dynamo traces get() and guards on it."""

    def __init__(self):
        self.value = None

    def get(self):
        return self.value

    def set(self, value):
        old, self.value = self.value, value
        return old

    def reset(self, token):
        self.value = token


def test_trial_forward_graph_is_reused_by_the_real_call(monkeypatch):
    # tiny Gemma-4 vision: a trial hidden from the collector (None) compiled graphs no real call matched.
    from torch._dynamo.utils import counters

    output_capturing = pytest.importorskip("transformers.utils.output_capturing")
    M = _load()
    real_compile = torch.compile

    def eager_compile(fn, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in ("options", "backend")}
        return real_compile(fn, backend = "eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager_compile)
    collector = _TraceableCollector()
    monkeypatch.setattr(output_capturing, "_active_collector", collector)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()

    def hook(module, args, output):
        collected = output_capturing._active_collector.get()
        if collected is not None and "attentions" in collected:
            collected["attentions"].append(output)

    layer.proj.register_forward_hook(hook)
    collector.set({})  # capture_outputs with nothing requested
    torch._dynamo.reset()
    counters.clear()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():  # reentrant checkpoint forward: grad-mode trial, then the real call
        layer(x)
    layer(x.requires_grad_()).sum().backward()  # the recompute
    assert counters["stats"]["unique_graphs"] == 2, dict(counters["stats"])


def test_gdn_self_check_leaves_default_rng_alone():
    # Checkpoint recompute replays the default generator for LoRA dropout masks.
    pytest.importorskip("fla.ops.gated_delta_rule.chunk")
    if not has_real_cuda():
        pytest.skip("needs a real CUDA device")
    G = _load("_gdn_op")
    gen = torch.Generator(device = "cuda").manual_seed(0)
    B, T, H, D = 1, 70, 2, 128
    mk = lambda *s, dt = torch.bfloat16: torch.randn(*s, device = "cuda", dtype = dt, generator = gen)
    q, k, v = mk(B, T, H, D), mk(B, T, H, D), mk(B, T, H, D)
    g = -mk(B, T, H, dt = torch.float32).abs() * 0.1
    beta = mk(B, T, H).sigmoid()
    state = torch.cuda.get_rng_state()
    assert G._fla_ops(lambda: (q, k, v, g, beta))
    assert torch.equal(torch.cuda.get_rng_state(), state)


def test_gdn_self_check_stays_out_of_a_non_reentrant_checkpoint(monkeypatch):
    # #14063: the check's autograd.grad must not recompute the checkpointed layer mid-forward.
    from torch.utils.checkpoint import checkpoint

    G = _load("_gdn_op")
    chunk = types.ModuleType("fla.ops.gated_delta_rule.chunk")
    for name in (
        "chunk_gated_delta_rule_fwd",
        "chunk_gated_delta_rule_bwd",
        "l2norm_fwd",
        "l2norm_bwd",
    ):
        setattr(chunk, name, lambda *a, **k: None)
    for name in ("fla", "fla.ops", "fla.ops.gated_delta_rule"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule.chunk", chunk)

    def self_check(q, k, v, g, beta):
        ins = [t.requires_grad_() for t in (q, k, v)]
        out = (ins[0] * ins[1]).sigmoid() * ins[2]
        return all(gr is not None for gr in torch.autograd.grad(out.sum(), ins))

    monkeypatch.setattr(G, "_self_check", self_check)
    calls = []
    w = torch.randn(8, 8, requires_grad = True)

    def layer(x):
        calls.append(1)
        h = (x @ w).sigmoid()
        G._fla_ops(lambda: (h, h, h, h, h))
        return (h @ w).sigmoid()

    checkpoint(layer, torch.randn(4, 8), use_reentrant = False).sum().backward()
    assert G._FLA
    assert len(calls) == 2  # forward, then one recompute in backward
    assert w.grad is not None


def test_sm_count_is_read_outside_the_trace(monkeypatch):
    # Stands in for Triton's hip_utils.get_device_properties, which Dynamo cannot trace.
    M = _load()
    fla = types.ModuleType("fake.fla.modules.fused_norm_gate")
    calls = []

    @torch._dynamo.disable
    def get_multiprocessor_count(tensor_idx = 0):
        calls.append(tensor_idx)
        return 7

    fla.get_multiprocessor_count = get_multiprocessor_count
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch._C, "_cuda_init", lambda: pytest.fail("CPU test initialized CUDA"), raising = False
    )
    M._route_sm_count({fla.__name__: fla})
    out = torch.compile(
        lambda x: x * fla.get_multiprocessor_count(0), fullgraph = True, backend = "eager"
    )(torch.ones(2))
    assert out.tolist() == [7.0, 7.0] and calls == [0]


def _grads_after_step(layers, run, x):
    for layer in layers:
        layer.zero_grad(set_to_none = True)
    out = x
    for layer in layers:
        out = run(layer, out)
    loss = out.float().pow(2).mean()
    loss.backward()
    return loss.detach(), {
        n: p.grad.clone() for layer in layers for n, p in layer.named_parameters()
    }


def _check_block(layers, run, x, cls):
    """Stock forward + backward vs the compiled block, two sequence lengths, no graph break."""
    from torch._dynamo.utils import counters

    import unsloth.models._block_compile as M

    want = [_grads_after_step(layers, run, xi) for xi in x]
    original = cls.forward
    try:
        M._wrap(M._rebind_class(cls) or cls, M._COMPILE_OPTIONS)
        counters.clear()
        got = [_grads_after_step(layers, run, xi) for xi in x]
        assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    finally:
        cls.forward = original
        torch._dynamo.reset()
    for (lw, gw), (lg, gg) in zip(want, got):
        torch.testing.assert_close(lg, lw, rtol = 1e-4, atol = 1e-5)
        for k in gw:
            torch.testing.assert_close(gg[k], gw[k], rtol = 1e-3, atol = 1e-5)


@pytest.mark.skipif(not has_real_cuda(), reason = "compiled block needs CUDA")
def test_qwen3_block_matches_stock():
    pytest.importorskip("unsloth")
    from transformers.models.qwen3 import modeling_qwen3 as modeling
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    cfg = Qwen3Config(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layers = [modeling.Qwen3DecoderLayer(cfg, layer_idx = i).cuda() for i in range(2)]
    rotary = modeling.Qwen3RotaryEmbedding(cfg).cuda()

    def run(layer, h):
        pos = torch.arange(h.shape[1], device = h.device)[None].expand(h.shape[0], -1)
        out = layer(h, position_embeddings = rotary(h, pos), position_ids = pos)
        return out[0] if isinstance(out, tuple) else out

    x = [torch.randn(2, L, 64, device = "cuda", requires_grad = True) for L in (24, 40)]
    _check_block(layers, run, x, modeling.Qwen3DecoderLayer)


def test_block_flex_forces_the_main_kernel(monkeypatch):
    """Inside a block every flex call asks for the main kernel; a caller's choice is kept."""
    M = _load()
    seen = []
    funcs = {"flex_attention": lambda *a, **kw: seen.append(kw.get("kernel_options"))}
    tmu = types.ModuleType("transformers.modeling_utils")
    tmu.ALL_ATTENTION_FUNCTIONS = funcs
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", tmu)
    M._flex_split_hook()
    flex = funcs["flex_attention"]
    flex(None, None, None, None)
    for mode in (M._FULL, M._SPLIT):
        monkeypatch.setattr(M, "_IN_BLOCK", [mode])
        flex(None, None, None, None)
        flex(None, None, None, None, kernel_options = {"BLOCK_M": 32})
        flex(None, None, None, None, kernel_options = {"FORCE_USE_FLEX_ATTENTION": False})
    forced = {"FORCE_USE_FLEX_ATTENTION": True}
    assert seen == [None] + 2 * [
        forced,
        {"BLOCK_M": 32, **forced},
        {"FORCE_USE_FLEX_ATTENTION": False},
    ]


@pytest.mark.skipif(not has_real_cuda(), reason = "compiled block needs CUDA")
@pytest.mark.parametrize("guard", ["0", "1"], ids = ["no_align_guard", "torch_guard"])
def test_block_does_not_recompile_per_length_mod_8(monkeypatch, guard):
    """torch 2.14 SDPA lowering guards `Mod(s, 8)` on strides: a block recompiled per alignment class."""
    import inspect

    import torch._inductor.lowering as lowering
    from transformers.models.qwen3 import modeling_qwen3 as modeling
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    M = _load()
    stock = lowering.sdpa_constraint
    stock = (
        getattr(stock, "__wrapped__", stock)
        if getattr(stock, "_unsloth_no_align_guard", False)
        else stock
    )
    if "is_aligned_realized_tensor" not in inspect.getsource(stock):
        pytest.skip("this torch has no alignment check in the SDPA lowering")
    monkeypatch.setattr(lowering, "sdpa_constraint", stock)
    for op, fn in list(lowering._maybe_layout_constraints.items()):
        if getattr(fn, "_unsloth_no_align_guard", False):
            monkeypatch.setitem(lowering._maybe_layout_constraints, op, stock)
    monkeypatch.setenv("UNSLOTH_BLOCK_SDPA_ALIGN_GUARD", guard)
    assert M._sdpa_alignment_without_guard() is (guard == "0")
    # A cached graph from the other arm would carry its guards.
    monkeypatch.setattr(torch._inductor.config, "fx_graph_cache", False)
    monkeypatch.setattr(torch._functorch.config, "enable_autograd_cache", False)
    import torch._dynamo.convert_frame as convert_frame

    reasons = []
    stock_reasons = convert_frame.get_and_maybe_log_recompilation_reasons

    def record(*a, **k):
        out = stock_reasons(*a, **k)
        reasons.extend(str(r) for r in out)
        return out

    monkeypatch.setattr(convert_frame, "get_and_maybe_log_recompilation_reasons", record)

    cfg = Qwen3Config(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    cls = modeling.Qwen3DecoderLayer
    model = modeling.Qwen3Model(cfg).cuda().to(torch.bfloat16).train()
    layer = model.layers[0]

    def run(L):
        # Causal flash SDPA: backward reads a (B, H, L) logsumexp, strides (H*L, L, 1).
        ids = torch.randint(0, 64, (2, L), device = "cuda")
        model(input_ids = ids, use_cache = False).last_hidden_state.float().sum().backward()

    original = cls.forward
    try:
        M._wrap(cls, M._COMPILE_OPTIONS)
        torch._dynamo.reset()
        for L in (517, 520, 523, 528):  # unaligned, aligned, unaligned, aligned
            run(L)
        assert layer.__dict__.get("_unsloth_block_checked", 0) & 2
    finally:
        cls.forward = original
        torch._dynamo.reset()
    mod_8 = [r for r in reasons if "% 8" in r]
    assert bool(mod_8) is (guard == "1"), reasons


@pytest.mark.skipif(not has_real_cuda(), reason = "flex_attention kernels need CUDA")
def test_block_flex_options_lower_short_gqa_training_queries():
    """Qwen3.5 GRPO shapes: with the block options, compiled flex fwd + bwd lower and match eager."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    M = _load()
    gen = torch.Generator(device = "cuda").manual_seed(0)
    mk = lambda *s: torch.randn(*s, device = "cuda", dtype = torch.bfloat16, generator = gen)
    q, k, v = mk(4, 8, 84, 256), mk(4, 2, 84, 256), mk(4, 2, 84, 256)
    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, 4, None, 84, 84, device = "cuda")
    ins = [t.clone().requires_grad_() for t in (q, k, v)]
    torch._dynamo.reset()
    try:
        fn = torch.compile(flex_attention, dynamic = False)
        got = fn(
            *ins,
            block_mask = bm,
            enable_gqa = True,
            scale = 1 / 16,
            kernel_options = M._training_flex_options(None),
        )
        got.float().sum().backward()
    finally:
        torch._dynamo.reset()
    ref_ins = [t.detach().float().requires_grad_() for t in (q, k, v)]
    ref = flex_attention(*ref_ins, block_mask = bm, enable_gqa = True, scale = 1 / 16)
    ref.sum().backward()
    torch.testing.assert_close(got.float(), ref, atol = 2e-2, rtol = 2e-2)
    for a, b in zip(ins, ref_ins):
        torch.testing.assert_close(a.grad.float(), b.grad, atol = 5e-2, rtol = 5e-2)


@pytest.mark.skipif(not has_real_cuda(), reason = "compiled block needs CUDA")
def test_short_gqa_flex_block_stays_compiled():
    """Short GQA head_dim 256 batches used to fail flex decoding lowering and demote every block."""
    pytest.importorskip("unsloth")
    from transformers.models.qwen3 import modeling_qwen3 as modeling
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    import unsloth.models._block_compile as M

    cfg = Qwen3Config(
        vocab_size = 64,
        hidden_size = 256,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        head_dim = 256,
    )
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    layers = [
        modeling.Qwen3DecoderLayer(cfg, layer_idx = i).cuda().to(torch.bfloat16) for i in range(2)
    ]
    rotary = modeling.Qwen3RotaryEmbedding(cfg).cuda()
    M._flex_split_hook()
    from torch.nn.attention.flex_attention import create_block_mask

    def run(layer, h):
        B, L = h.shape[0], h.shape[1]
        pos = torch.arange(L, device = h.device)[None].expand(B, -1)
        mask = create_block_mask(lambda b, hd, q, k: q >= k, B, None, L, L, device = h.device)
        out = layer(h, attention_mask = mask, position_embeddings = rotary(h, pos), position_ids = pos)
        return out[0] if isinstance(out, tuple) else out

    from torch._dynamo.utils import counters

    x = [torch.randn(4, L, 256, device = "cuda", dtype = torch.bfloat16) for L in (84, 100)]
    cls = modeling.Qwen3DecoderLayer
    original = cls.forward
    want = [_grads_after_step(layers, run, xi.clone().requires_grad_()) for xi in x]
    try:
        M._wrap(M._rebind_class(cls) or cls, M._COMPILE_OPTIONS)
        counters.clear()
        got = [_grads_after_step(layers, run, xi.clone().requires_grad_()) for xi in x]
        modes = [layer.__dict__.get("_unsloth_block_mode") for layer in layers]
        assert modes == [M._FULL, M._FULL], modes
        assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    finally:
        cls.forward = original
        torch._dynamo.reset()
    for (lw, gw), (lg, gg) in zip(want, got):
        torch.testing.assert_close(lg, lw, rtol = 2e-2, atol = 1e-3)
        for k in gw:
            num = (gg[k].float() - gw[k].float()).norm()
            assert num <= 0.05 * gw[k].float().norm() + 1e-4, (k, float(num))


@pytest.mark.skipif(not has_real_cuda(), reason = "fla kernels need CUDA")
def test_qwen3_5_gdn_block_has_no_break():
    pytest.importorskip("unsloth")
    pytest.importorskip("transformers.models.qwen3_5")  # transformers >= 5.2
    pytest.importorskip("fla.ops.gated_delta_rule")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    import unsloth.models._block_compile as M

    cfg = Qwen3_5TextConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        linear_num_key_heads = 4,
        linear_num_value_heads = 16,
        linear_key_head_dim = 32,
        linear_value_head_dim = 32,
        layer_types = ["linear_attention", "full_attention"],
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    layers = [
        modeling.Qwen3_5DecoderLayer(cfg, layer_idx = i).cuda().to(torch.bfloat16) for i in range(2)
    ]
    rotary = modeling.Qwen3_5TextRotaryEmbedding(cfg).cuda()
    x = [torch.randn(2, L, 64, device = "cuda", dtype = torch.bfloat16) for L in (64, 128)]
    M._prepare_gdn(layers[0], x[0])
    if not M._GDN_OP[0]:
        pytest.skip("fla's opaque op failed its self-check on this GPU")

    def run(layer, h):
        pos = torch.arange(h.shape[1], device = h.device)[None, None].expand(3, h.shape[0], -1)
        out = layer(h, position_embeddings = rotary(h, pos), position_ids = pos)
        return out[0] if isinstance(out, tuple) else out

    from torch._dynamo.utils import counters

    cls = modeling.Qwen3_5DecoderLayer
    original = cls.forward
    want = [_grads_after_step(layers, run, xi.clone().requires_grad_()) for xi in x]
    try:
        M._wrap(cls, M._COMPILE_OPTIONS)
        counters.clear()
        got = [_grads_after_step(layers, run, xi.clone().requires_grad_()) for xi in x]
        assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
        # A stock-forward fallback also has no break: assert the block ran (grouped heads demoted it on 5.23).
        for layer in layers:
            assert layer.__dict__.get("_unsloth_block_mode") == M._FULL, layer.__dict__.get(
                "_unsloth_block_mode"
            )
            assert layer.__dict__.get("_unsloth_block_checked", 0) & 2
    finally:
        cls.forward = original
        torch._dynamo.reset()
    for (lw, gw), (lg, gg) in zip(want, got):
        torch.testing.assert_close(lg, lw, rtol = 2e-2, atol = 1e-3)
        for k in gw:
            num = (gg[k].float() - gw[k].float()).norm()
            assert num <= 0.05 * gw[k].float().norm() + 1e-4, (
                k,
                float(num),
                float(gw[k].float().norm()),
            )


def test_gdn_op_name_pins_its_fake_meta():
    """Inductor's disk cache keys on the op name, not its fake meta: changing the fake requires a new op name."""
    M = _load("_gdn_op")
    if M._gdn_chunk is None:
        pytest.skip("torch without torch.library.custom_op")
    q = torch.empty(2, 64, 4, 32, device = "meta", dtype = torch.bfloat16)
    v = torch.empty(2, 64, 16, 32, device = "meta", dtype = torch.bfloat16)
    g = torch.empty(2, 64, 16, device = "meta")
    A = [tuple(shape) for shape, _ in M._chunk_meta(q, q, v, g)][-1]
    published = {
        "unsloth::qwen3_5_gdn_chunk": (2, 64, 4, 64),
        "unsloth::qwen3_5_gdn_chunk_v2": (2, 64, 16, 64),
    }
    assert published[M._gdn_chunk._qualname] == A, (M._gdn_chunk._qualname, A)


@pytest.mark.skipif(not has_real_cuda(), reason = "fla kernels need a GPU")
@pytest.mark.parametrize("qk_heads", [4, 2], ids = ["equal_heads", "grouped_value_heads"])
def test_opaque_fla_op_matches_fla_function(qk_heads):
    """grouped_value_heads: transformers 5.23 passes q / k unrepeated (huggingface/transformers#49141); the fake
    meta of A had the q / k head count, so Inductor's output check failed and every GVA layer ran stock."""
    pytest.importorskip("fla.ops.gated_delta_rule.chunk")
    M = _load("_gdn_op")
    gen = torch.Generator(device = "cuda").manual_seed(0)
    B, T, H, D = 2, 150, 4, 128
    mk = lambda *s, dt = torch.bfloat16: torch.randn(*s, device = "cuda", dtype = dt, generator = gen)
    q, k, v = mk(B, T, qk_heads, D), mk(B, T, qk_heads, D), mk(B, T, H, D)
    g = -mk(B, T, H, dt = torch.float32).abs() * 0.1
    beta = mk(B, T, H).sigmoid()
    assert M._fla_ops(lambda: (q, k, v, g, beta))
    assert M._VARLEN[0], "the packed (cu_seqlens) layout failed its self-check"
    # A wrong fake meta or a numeric mismatch must keep the op out of the block.
    M._FLA = None
    orig = M._chunk_meta
    M._chunk_meta = lambda *a: orig(*a)[:-1] + (((1,), torch.bfloat16),)
    try:
        assert M._fla_ops(lambda: (q, k, v, g, beta)) is False
    finally:
        M._chunk_meta = orig


def test_gdn_adapter_routes_only_the_self_checked_call(monkeypatch):
    M = _load()
    seen = []
    monkeypatch.setattr(M, "_GDN_OP", [True])
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch.ops,
        "unsloth",
        types.SimpleNamespace(qwen3_5_gdn_chunk_v2 = lambda *a: seen.append("op") or ("out",)),
        raising = False,
    )
    adapter = M._gdn_chunk_adapter(lambda *a, **kw: seen.append("fla") or ("fla", None))
    t = torch.zeros(1, 4, 2, 8)
    gb = torch.zeros(1, 4, 2)
    stock = dict(
        use_qk_l2norm_in_kernel = True, initial_state = None, output_final_state = False, cu_seqlens = None
    )
    adapter(t, t, t, g = gb, beta = gb, **stock)
    for extra in (
        {"head_first": True},
        {"use_beta_sigmoid_in_kernel": True},
        {"scale": 0.1},
        {"output_final_state": True},
        {"cu_seqlens": torch.tensor([0, 4])},
        {"use_qk_l2norm_in_kernel": False},
    ):
        adapter(t, t, t, g = gb, beta = gb, **dict(stock, **extra))
    assert seen == ["op"] + ["fla"] * 6


def test_gdn_adapter_routes_packed_batches_once_self_checked(monkeypatch):
    """Packed (cu_seqlens) fla calls stay in the block once the op's packed self-check passed."""
    M = _load()
    seen = []
    monkeypatch.setattr(M, "_GDN_OP", [True])
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def op(*a):
        seen.append(("op", a[6].tolist() if len(a) > 6 else None))
        return ("out",)

    monkeypatch.setattr(
        torch.ops, "unsloth", types.SimpleNamespace(qwen3_5_gdn_chunk_v2 = op), raising = False
    )
    adapter = M._gdn_chunk_adapter(lambda *a, **kw: seen.append(("fla", None)) or ("fla", None))
    stock = dict(use_qk_l2norm_in_kernel = True, initial_state = None, output_final_state = False)
    cu = torch.tensor([0, 3, 4], dtype = torch.int32)
    one, two = torch.zeros(1, 4, 2, 8), torch.zeros(2, 4, 2, 8)
    g1, g2 = torch.zeros(1, 4, 2), torch.zeros(2, 4, 2)
    for varlen in (False, True):
        monkeypatch.setattr(M, "_GDN_VARLEN", [varlen])
        adapter(one, one, one, g = g1, beta = g1, cu_seqlens = cu, **stock)
        adapter(two, two, two, g = g2, beta = g2, cu_seqlens = cu, **stock)  # fla wants B = 1
        adapter(one, one, one, g = g1, beta = g1, cu_seqlens = None, **stock)
    assert seen == [("fla", None), ("fla", None), ("op", None)] + [
        ("op", [0, 3, 4]),
        ("fla", None),
        ("op", None),
    ]


def _hub_wrapper(implementation, applicable_params):
    # Shape of transformers' use_kernel_func_from_hub_with_fallback wrapper.
    def torch_chunk_gated_delta_rule(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}
        return implementation(*args, **kwargs)

    return torch_chunk_gated_delta_rule


def test_gdn_adapter_ignores_kwargs_the_hub_wrapper_drops(monkeypatch):
    # The decoder's TransformersKwargs (num_items_in_batch, use_cache, ...) reach the hub wrapper,
    # which drops them; the adapter declined on them, so the op never ran in training.
    M = _load()
    seen = []
    monkeypatch.setattr(M, "_GDN_OP", [True])
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch.ops,
        "unsloth",
        types.SimpleNamespace(qwen3_5_gdn_chunk_v2 = lambda *a: seen.append("op") or ("out",)),
        raising = False,
    )
    fla = lambda *a, **kw: seen.append("fla") or ("fla", None)
    params = ("q", "k", "v", "g", "beta", "scale", "initial_state", "output_final_state")
    params += ("use_qk_l2norm_in_kernel", "cu_seqlens")
    adapter = M._gdn_chunk_adapter(_hub_wrapper(fla, params))
    t, gb = torch.zeros(1, 4, 2, 8), torch.zeros(1, 4, 2)
    stock = dict(use_qk_l2norm_in_kernel = True, initial_state = None, output_final_state = False)
    adapter(t, t, t, g = gb, beta = gb, num_items_in_batch = torch.tensor(8), use_cache = False, **stock)
    adapter(t, t, t, g = gb, beta = gb, cu_seqlens = torch.tensor([0, 4]), **stock)
    assert seen == ["op", "fla"]


def test_route_fla_kernel_sees_a_frozen_heap(monkeypatch):
    # Trainer.train() gc.freeze()s the heap (_gc_train_scope) before the first GatedDeltaNet
    # forward routes the kernel, and gc.get_referrers skips frozen objects: nothing was routed.
    import gc

    if gc.get_freeze_count():
        pytest.skip("the heap is already frozen")
    M = _load()
    chunk = types.ModuleType("fake_frozen.fla.ops.gated_delta_rule.chunk")
    chunk.chunk_gated_delta_rule = lambda *a, **kw: ("fla", None)
    monkeypatch.setitem(sys.modules, chunk.__name__, chunk)
    modeling = types.ModuleType("fake_frozen_modeling")
    modeling.torch_chunk_gated_delta_rule = _hub_wrapper(chunk.chunk_gated_delta_rule, ("q",))
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    gc.freeze()
    try:
        assert M._route_fla_kernel() >= 1
        assert gc.get_freeze_count() > 0  # the caller's freeze is kept
    finally:
        gc.unfreeze()
    assert getattr(modeling.torch_chunk_gated_delta_rule, "_unsloth_block_adapter", False)


@pytest.mark.parametrize("linear_env", ["auto", "0"])
def test_nf4_block_route_matches_both_checkpoint_halves(monkeypatch, linear_env):
    # Package context so `from ..kernels import bnb_override` resolves to the fakes below.
    pkg = types.ModuleType("_ufake")
    pkg.__path__ = []
    models = types.ModuleType("_ufake.models")
    models.__path__ = []
    kernels = types.ModuleType("_ufake.kernels")
    kernels.__path__ = []
    O = types.ModuleType("_ufake.kernels.bnb_override")
    O._TRACE, O._TRACE_BF16, O._LINEAR = True, True, linear_env
    O._eligible = lambda self, x, weight, quant_state: True
    O._linear = lambda x, weight, quant_state, bias: torch.full_like(x, 7.0)
    O._float_packed_as_uint8 = lambda weight: weight
    kernels.bnb_override = O

    class Linear4bit(torch.nn.Module):
        def __init__(self, dtype, bias):
            super().__init__()
            self.weight = types.SimpleNamespace(
                dtype = torch.uint8, quant_state = types.SimpleNamespace(dtype = dtype)
            )
            self.bias = torch.nn.Parameter(torch.zeros(8)) if bias else None

        def forward(self, x):
            return torch.zeros_like(x)

    bnb = types.ModuleType("bitsandbytes")
    bnb.nn = types.SimpleNamespace(Linear4bit = Linear4bit)
    for name, mod in (
        ("_ufake", pkg),
        ("_ufake.models", models),
        ("_ufake.kernels", kernels),
        ("_ufake.kernels.bnb_override", O),
        ("bitsandbytes", bnb),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location(
        "_ufake.models._block_compile", os.path.join(_DIR, "_block_compile.py")
    )
    M = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, M)
    spec.loader.exec_module(M)
    M._nf4_in_blocks()
    if linear_env == "0":
        assert not getattr(Linear4bit.forward, "_unsloth_block_nf4", False)
        return
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(M, "_IN_BLOCK", [M._FULL])

    def routed(
        dtype,
        rows,
        bias = False,
    ):
        return bool(Linear4bit(dtype, bias)(torch.ones(rows, 8, dtype = dtype)).eq(7).all())

    assert routed(torch.bfloat16, 16)
    assert routed(torch.float32, 8)
    assert not routed(torch.bfloat16, 1)  # one row: fast_gemv only without grad
    assert not routed(torch.float32, 4)  # bitsandbytes' fp32 small-row kernel
    assert not routed(torch.bfloat16, 16, bias = True)  # trainable bias


def test_causal_depthwise_conv_matches_conv1d():
    M = _load("_gdn_op")
    torch.manual_seed(0)
    x, w = torch.randn(2, 6, 37), torch.randn(6, 4)
    want = torch.nn.functional.conv1d(x, w[:, None], groups = 6, padding = 3)[..., :37]
    torch.testing.assert_close(M._causal_depthwise_conv(x, w), want, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("call", ["positional", "x_keyword"])
def test_causal_conv_adapter_takes_both_call_conventions(call):
    """transformers < 5.15 GatedDeltaNet calls self.causal_conv1d_fn(x = ..., weight = ..., ...): the adapter
    required `hidden_states`, so a Qwen3-Next model built after the first block forward raised TypeError."""
    M = _load()
    G = _load("_gdn_op")
    seen = []

    def legacy(
        x,
        weight,
        bias = None,
        activation = None,
        seq_idx = None,
        **kwargs,
    ):
        seen.append(sorted(kwargs) + (["seq_idx"] if seq_idx is not None else []))
        out = torch.nn.functional.conv1d(
            x, weight[:, None], bias, groups = x.shape[1], padding = weight.shape[-1] - 1
        )
        out = out[..., : x.shape[-1]]
        return torch.nn.functional.silu(out) if activation else out

    fn = M._causal_conv1d_adapter(legacy, G._causal_depthwise_conv)
    torch.manual_seed(0)
    x, w, b = torch.randn(2, 6, 17), torch.randn(6, 4), torch.randn(6)

    def call_it(f):
        if call == "positional":
            return f(x, w, b, activation = "silu")
        return f(x = x, weight = w, bias = b, activation = "silu", seq_idx = None)

    want = call_it(legacy)
    torch.testing.assert_close(call_it(fn), want)  # eager: passes through unchanged
    M._IN_BLOCK[0] = M._FULL
    try:
        seen.clear()
        got = torch.compile(lambda: call_it(fn), backend = "eager", fullgraph = True)()
    finally:
        M._IN_BLOCK[0] = M._OFF
        torch._dynamo.reset()
    assert seen == []  # in-block: the traceable depthwise conv, not the legacy kernel
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-5)


def test_disable_unless_decode_keeps_original_traceable_on_old_torch(monkeypatch):
    # torch < 2.8: compiler.disable(fn, recursive = False) marks fn itself and returns it.
    import unsloth.import_fixes as F
    import unsloth_zoo.temporary_patches.utils as zoo_utils

    marked = []

    def old_disable(
        fn = None,
        recursive = True,
        **kwargs,
    ):
        fn._old_disable_marked = True
        marked.append(fn)
        return fn

    monkeypatch.setattr(torch.compiler, "disable", old_disable)
    monkeypatch.setattr(
        zoo_utils,
        "torch_compiler_disable_unless_decode",
        zoo_utils.torch_compiler_disable_unless_decode,
    )
    F.fix_torch_compiler_disable_unless_decode()
    fixed = zoo_utils.torch_compiler_disable_unless_decode
    assert fixed._unsloth_disable_copy

    def attention(x):
        return x + 1

    wrapped = fixed(attention)
    assert wrapped._unsloth_undisabled is attention
    assert not getattr(attention, "_old_disable_marked", False)
    disabled_copy = marked[-1]
    assert disabled_copy is not attention and disabled_copy.__code__ is not attention.__code__
    assert disabled_copy(1) == attention(1) == 2
    assert torch.compiler.disable is old_disable


def test_moe_experts_run_outside_the_block(monkeypatch):
    M = _load()

    class ToyExperts(torch.nn.Module):
        def forward(self, x):
            return x * 2

    class LoraWrapper(torch.nn.Module):  # PEFT-style wrapper around the experts
        def __init__(self):
            super().__init__()
            self.base_layer = ToyExperts()

        def forward(self, x):
            return self.base_layer(x) + 1

    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = LoraWrapper()
    layer.self_attn = torch.nn.Linear(4, 4)
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: True)
    M._experts_outside(layer)
    outer = layer.mlp.experts
    assert (
        outer.forward.__func__ is not LoraWrapper.forward
        if hasattr(outer.forward, "__func__")
        else True
    )
    assert outer._unsloth_block_outside and outer.base_layer._unsloth_block_outside
    assert not getattr(layer.self_attn, "_unsloth_block_outside", False)
    from torch._dynamo.utils import counters

    counters.clear()
    out = torch.compile(lambda x: layer.mlp.experts(x) * 3, backend = "eager")(torch.ones(2))
    assert out.tolist() == [9.0, 9.0] and sum(counters["graph_break"].values()) >= 1
    # A stash-reading experts forward (no fold) stays inside; so does everything under "inside".
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    other = torch.nn.Module()
    other.experts = LoraWrapper()
    M._experts_outside(other)
    assert not getattr(other.experts, "_unsloth_block_outside", False)
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: True)
    monkeypatch.setenv("UNSLOTH_BLOCK_MOE", "inside")
    third = torch.nn.Module()
    third.experts = LoraWrapper()
    M._experts_outside(third)
    assert not getattr(third.experts, "_unsloth_block_outside", False)


def test_selective_dequant_keeps_bnb4bit_experts_outside_the_block(monkeypatch):
    # The zoo plan reads token counts on the host and declines in a trace; opted-in experts stay eager.
    M = _load()

    class Params4bit(torch.nn.Parameter):
        pass

    class ToyExperts(torch.nn.Module):
        def __init__(self, quantized):
            super().__init__()
            weight = torch.zeros(2, 4, 4)
            self.gate_up_proj = (
                Params4bit(weight, requires_grad = False) if quantized else torch.nn.Parameter(weight)
            )

        def forward(self, x):
            return x * 2

    def layer(quantized):
        out = torch.nn.Module()
        out.mlp = torch.nn.Module()
        out.mlp.experts = ToyExperts(quantized)
        return out

    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    for key in (
        "UNSLOTH_BLOCK_MOE",
        "UNSLOTH_MOE_SELECTIVE_DEQUANT",
        "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS",
    ):
        monkeypatch.delenv(key, raising = False)
    default = layer(True)
    M._experts_outside(default)
    assert not getattr(default.mlp.experts, "_unsloth_block_outside", False)
    for key, value in (
        ("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1"),
        ("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "32"),
    ):
        monkeypatch.setenv(key, value)
        quantized, plain = layer(True), layer(False)
        M._experts_outside(quantized)
        M._experts_outside(plain)
        assert quantized.mlp.experts._unsloth_block_outside
        assert not getattr(plain.mlp.experts, "_unsloth_block_outside", False)
        monkeypatch.delenv(key)
    monkeypatch.setenv("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "abc")
    bad = layer(True)
    M._experts_outside(bad)
    assert not getattr(bad.mlp.experts, "_unsloth_block_outside", False)


def test_selective_dequant_off_never_reads_a_packed_expert_stack(monkeypatch):
    # MXFP4 experts expose gate_up_proj as a property that dequantizes the whole stack.
    M = _load()
    reads = []

    class PackedExperts(torch.nn.Module):
        @property
        def gate_up_proj(self):
            reads.append(1)
            return torch.zeros(1)

        def forward(self, x):
            return x

    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    for key in (
        "UNSLOTH_BLOCK_MOE",
        "UNSLOTH_MOE_SELECTIVE_DEQUANT",
        "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS",
    ):
        monkeypatch.delenv(key, raising = False)
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = PackedExperts()
    M._experts_outside(layer)
    assert reads == []
    # Opted in, a packed stack is still not a bnb 4-bit one: never read, never moved out.
    monkeypatch.setenv("UNSLOTH_MOE_SELECTIVE_DEQUANT", "1")
    layer.__dict__.pop("_unsloth_block_moe_done", None)
    M._experts_outside(layer)
    assert reads == []
    assert not getattr(layer.mlp.experts, "_unsloth_block_outside", False)


def test_zero3_partitioned_training_keeps_the_stock_forward(monkeypatch):
    import weakref

    M = _load()
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M, "_torch_supported", lambda: True)

    class FakeDsConfig:
        def is_zero3(self):
            return True

    ds = types.ModuleType("transformers.integrations.deepspeed")
    config = FakeDsConfig()
    ds._hf_deepspeed_config_weak_ref = weakref.ref(config)
    monkeypatch.setitem(sys.modules, ds.__name__, ds)

    cls = _toy_class()
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([cls(), cls()])
    assert M.patch_block_compile(model) == []

    # ZeRO-3 configured after the load (TrainingArguments built later): the forward checks.
    ds._hf_deepspeed_config_weak_ref = None
    assert M.patch_block_compile(model) == [cls.__name__]
    x = torch.randn(2, 4, 8, requires_grad = True)
    ds._hf_deepspeed_config_weak_ref = weakref.ref(config)
    model.layers[0](x).sum().backward()
    assert calls == []
    # Parameters partitioned by deepspeed.zero.Init (ds_id) without a live HF config; the first
    # parameter can be an unpartitioned adapter.
    ds._hf_deepspeed_config_weak_ref = None
    monkeypatch.setitem(sys.modules, "deepspeed", types.ModuleType("deepspeed"))
    params = list(model.layers[1].parameters())
    assert len(params) > 1
    params[-1].ds_id = 0
    model.layers[1](x).sum().backward()
    assert calls == []
    model.layers[0](x).sum().backward()
    assert calls


def test_router_logits_after_a_forward_without_them(request, monkeypatch):
    """transformers installs router_logits hooks on first forward asking; Dynamo does not guard
    hooks (skip_nnmodule_hook_guards), so a block compiled earlier lost its gate hook (IndexError
    in load_balancing_loss_func); hooks exist before compile; hook body (guarded on collector
    keys) decides per call."""
    misc = pytest.importorskip("unsloth_zoo.temporary_patches.misc")
    if not hasattr(misc, "output_capture_scratch"):
        pytest.skip("needs an unsloth_zoo with output_capture_scratch (unsloth-zoo#2041)")
    misc.patch_output_collector_for_compiled_submodules()
    M = _load()
    torch.manual_seed(0)
    model = _tiny_qwen3_moe(request, monkeypatch)
    ids = torch.randint(0, 100, (2, 8))
    want = model(input_ids = ids, labels = ids, use_cache = False, output_router_logits = True)
    model = (
        type(model)
        .from_pretrained(
            "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
            dtype = torch.float32,
            experts_implementation = "batched_mm",
        )
        .train()
    )
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    inner = _install_blocks(M, model, monkeypatch)
    torch._dynamo.reset()
    model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
    out = model(input_ids = ids, labels = ids, use_cache = False, output_router_logits = True)
    out.loss.backward()
    assert inner, "the blocks did not run compiled"
    assert len(out.router_logits) == len(want.router_logits) == model.config.num_hidden_layers
    torch.testing.assert_close(out.aux_loss.detach(), want.aux_loss.detach())
    torch.testing.assert_close(out.loss.detach(), want.loss.detach())
    torch._dynamo.reset()


@pytest.mark.parametrize("registry_kind", ["dict", "model_keyed"])
def test_router_hooks_install_with_either_registry_api(monkeypatch, registry_kind):
    """transformers 5.23 (huggingface/transformers#49594) made _CAN_RECORD_REGISTRY.get take the
    model, so a class-string get returned {} and no router hook was installed."""
    M = _load()

    class Registry(dict):
        def get(self, model):
            return super().get(str(model.__class__)) or {}

    class Collector:
        _unsloth_eager_mirror = True

    class MoeModel(torch.nn.Module):
        pass

    registry = (dict if registry_kind == "dict" else Registry)()
    dict.__setitem__(registry, str(MoeModel), {"router_logits": object()})
    installed = []
    fake = types.ModuleType("transformers.utils.output_capturing")
    fake._CAN_RECORD_REGISTRY = registry
    fake.maybe_install_capturing_hooks = installed.append
    fake.CompileableContextVar = Collector
    monkeypatch.setitem(sys.modules, "transformers.utils.output_capturing", fake)
    import transformers.utils

    monkeypatch.setattr(transformers.utils, "output_capturing", fake, raising = False)
    model = MoeModel()
    model.inner = torch.nn.Linear(2, 2)
    assert M._install_router_capture_hooks(model) == 1
    assert installed == [model]


def test_router_hooks_are_not_installed_twice_after_an_eager_capture(request, monkeypatch):
    """Failing before: an eager capturing forward hooks the inner model; installing on the unflagged outer
    CausalLM recursed into it again and every captured tuple held each layer twice."""
    misc = pytest.importorskip("unsloth_zoo.temporary_patches.misc")
    if not hasattr(misc, "output_capture_scratch"):
        pytest.skip("needs an unsloth_zoo with output_capture_scratch (unsloth-zoo#2041)")
    misc.patch_output_collector_for_compiled_submodules()
    M = _load()
    torch.manual_seed(0)
    model = _tiny_qwen3_moe(request, monkeypatch).train()
    ids = torch.randint(0, 100, (2, 8))
    kwargs = dict(use_cache = False, output_router_logits = True, output_hidden_states = True)
    model(input_ids = ids, **kwargs)
    monkeypatch.setattr(M, "_lora_would_fold", lambda experts: False)
    _install_blocks(M, model, monkeypatch)
    torch._dynamo.reset()
    out = model(input_ids = ids, **kwargs)
    layers = model.config.num_hidden_layers
    assert len(out.router_logits) == layers, len(out.router_logits)
    assert len(out.hidden_states) == layers + 1, len(out.hidden_states)
    torch._dynamo.reset()


def test_router_logits_capture_compiles_one_graph_for_every_layer(monkeypatch):
    """A capture hook appends to an empty scratch list per call, so Dynamo does not guard on the
    collector's length, which compiled one graph per layer; now at most 2 graphs across 10 layers
    and 10 lengths, with logits matching the stock forward."""
    output_capturing = pytest.importorskip("transformers.utils.output_capturing")
    misc = pytest.importorskip("unsloth_zoo.temporary_patches.misc")
    if not hasattr(misc, "output_capture_scratch"):
        pytest.skip("needs an unsloth_zoo with output_capture_scratch (unsloth-zoo#2041)")
    misc.patch_output_collector_for_compiled_submodules()
    from torch._dynamo.testing import CompileCounter

    M = _load()
    cc = CompileCounter()
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: real_compile(fn, backend = cc, fullgraph = False, dynamic = None),
    )
    var = output_capturing._active_collector

    class RouterDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.router = torch.nn.Linear(8, 4)

            def hook(
                module, args, output
            ):  # transformers' capture hook for a non hidden_states key
                collected = var.get()
                if collected is None or "router_logits" not in collected.keys():
                    return
                collected["router_logits"].append(output)

            self.router.register_forward_hook(hook)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            return self.proj(hidden_states) + self.router(hidden_states).sum(-1, keepdim = True)

    M._wrap(RouterDecoderLayer, {})
    lengths = [5, 9, 12, 17, 23, 31, 40, 47, 58, 64]
    layers = [RouterDecoderLayer() for _ in lengths]
    xs = [torch.randn(2, n, 8) for n in lengths]
    with torch.no_grad():
        want = [layer.router(x) for layer, x in zip(layers, xs)]
    torch._dynamo.reset()
    collected = {"router_logits": []}
    token = var.set(collected)
    try:
        for layer, x in zip(layers, xs):
            layer(x.requires_grad_())
    finally:
        var.reset(token)
    assert cc.frame_count <= 2, f"{cc.frame_count} graphs for {len(lengths)} layers"
    got = collected["router_logits"]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        torch.testing.assert_close(g.detach(), w)


@pytest.mark.parametrize("backend", ["AUTO", "TRITON"])
def test_training_flex_options_keep_a_callers_backend(backend):
    """BACKEND supersedes FORCE_USE_FLEX_ATTENTION, and torch 2.10+ raises when both are set."""
    M = _load()
    options = {"BACKEND": backend}
    assert M._training_flex_options(options) == options
    assert M._training_flex_options({"ROWS_GUARANTEED_SAFE": True}) == {
        "ROWS_GUARANTEED_SAFE": True,
        "FORCE_USE_FLEX_ATTENTION": True,
    }
    flex = pytest.importorskip("torch.nn.attention.flex_attention")
    if (
        not hasattr(flex, "_apply_kernel_options")
        or "BACKEND" not in getattr(flex, "FlexKernelOptions", {}).__annotations__
    ):
        return
    q = torch.randn(1, 1, 16, 16)
    flex._apply_kernel_options(q, q, q, False, M._training_flex_options(options))


@pytest.mark.skipif(
    "isolate_recompiles" not in __import__("inspect").signature(torch.compile).parameters,
    reason = "torch < 2.13: no isolate_recompiles",
)
def test_recompile_storm_stops_at_the_region_limit(monkeypatch):
    """A recompile storm stops at _region_recompile_limit (not the global 1024) and runs eager."""
    M = _load()
    from torch._dynamo.testing import CompileCounter

    config = torch._dynamo.config
    monkeypatch.setattr(config, "recompile_limit", 1024)
    monkeypatch.setattr(config, "accumulated_recompile_limit", 1024)
    counter, seen = CompileCounter(), []
    real_compile = torch.compile

    def counting_compile(fn, **kwargs):
        seen.append(kwargs)
        return real_compile(
            fn,
            backend = counter,
            fullgraph = False,
            dynamic = None,
            isolate_recompiles = kwargs["isolate_recompiles"],
            recompile_limit = kwargs["recompile_limit"],
        )

    monkeypatch.setattr(torch, "compile", counting_compile)
    torch._dynamo.reset()

    class StormDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.tag = "a"

        def forward(self, hidden_states, **kwargs):
            # A str attribute is guarded by value: a new value every call recompiles.
            return self.proj(hidden_states) + len(self.tag)

    M._wrap(StormDecoderLayer, {}, 1)
    layer = StormDecoderLayer().train()
    for k in range(60):
        layer.tag = "a" * (k + 1)
        x = torch.randn(2, 5, 8, requires_grad = True)
        out = layer(x)
        torch.testing.assert_close(out, layer.proj(x) + (k + 1))
        out.sum().backward()
    torch._dynamo.reset()
    assert seen[0]["isolate_recompiles"] is True
    assert counter.frame_count == seen[0]["recompile_limit"] == M._region_recompile_limit(1) == 32


def test_graphs_off_compile_gets_the_same_region_limit(monkeypatch):
    # The graphs-off compile from UNSLOTH_BLOCK_CUDAGRAPHS needs the same isolation options.
    M = _load()
    if not M._ISOLATE_RECOMPILES:
        pytest.skip("torch < 2.13: no per-region recompile limit")
    seen = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: seen.append(kwargs) or fn)

    class GraphsDecoderLayer(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    M._wrap(GraphsDecoderLayer, {"triton.cudagraphs": True}, 2)
    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs["isolate_recompiles"] is True
        assert kwargs["recompile_limit"] == M._region_recompile_limit(2) == 48


def test_shared_kv_writer_recompute_reuses_the_forward_graph(monkeypatch):
    """Gemma-4 E2B: recompute saw `shared_kv_states` filled and compiled a second graph (CheckpointError)."""
    from collections import UserDict
    from torch._dynamo.testing import CompileCounter
    from torch.utils.checkpoint import checkpoint

    M = _load()
    counter = CompileCounter()
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend = counter, fullgraph = False, dynamic = None),
    )
    torch._dynamo.reset()

    class Attn(torch.nn.Module):
        def __init__(self, store, shared):
            super().__init__()
            self.store_full_length_kv, self.is_kv_shared_layer = store, shared
            self.q = torch.nn.Linear(8, 8)
            self.k = torch.nn.Linear(8, 8)

        def forward(self, x, shared_kv_states):
            if self.is_kv_shared_layer:
                k = shared_kv_states["sliding"]
            else:
                k = self.k(x)
            if self.store_full_length_kv:
                shared_kv_states["sliding"] = k
            return self.q(x) * k

    class KvDecoderLayer(torch.nn.Module):
        def __init__(self, store, shared):
            super().__init__()
            self.self_attn = Attn(store, shared)

        def forward(
            self,
            hidden_states,
            shared_kv_states = None,
            **kwargs,
        ):
            return hidden_states + self.self_attn(hidden_states, shared_kv_states)

    torch.manual_seed(0)
    layers = [
        KvDecoderLayer(False, False),
        KvDecoderLayer(True, False),
        KvDecoderLayer(False, True),
    ]
    stock = [KvDecoderLayer(False, False), KvDecoderLayer(True, False), KvDecoderLayer(False, True)]
    for a, b in zip(stock, layers):
        b.load_state_dict(a.state_dict())
    M._wrap(KvDecoderLayer, {})
    for layer in layers + stock:
        layer.train()
    for layer in stock:
        layer.__dict__["_unsloth_block_mode"] = M._OFF

    def step(stack, x):
        shared = UserDict()
        h = x
        for layer in stack:
            h = checkpoint(layer, h, shared_kv_states = shared, use_reentrant = False)
        return h, shared

    x = torch.randn(2, 5, 8)
    ref, ref_shared = step(stock, x)
    ref.sum().backward()
    out, shared = step(layers, x)
    forward_frames = counter.frame_count
    out.sum().backward()
    recompute_frames = counter.frame_count - forward_frames
    torch._dynamo.reset()
    assert recompute_frames == 0
    assert list(shared.keys()) == ["sliding"]
    torch.testing.assert_close(out, ref)
    for a, b in zip(stock, layers):
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad)


def test_parametrized_class_is_reused_per_module(monkeypatch):
    """PEFT's ParamWrapper registers a parametrization per forward: each module keeps one
    Parametrized<Class> (a stable type id for Dynamo), and two modules never share one."""
    from torch.nn.utils import parametrize

    monkeypatch.setattr(parametrize, "_inject_new_class", parametrize._inject_new_class)
    bc = _load()
    bc._stable_parametrized_classes()
    bc._stable_parametrized_classes()  # idempotent

    class Double(torch.nn.Module):
        def forward(self, w):
            return 2 * w

    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    seen = []
    for _ in range(3):
        parametrize.register_parametrization(a, "weight", Double())
        seen.append(type(a))
        expected = 2 * a.parametrizations["weight"].original
        assert torch.equal(a.weight, expected)
        parametrize.remove_parametrizations(a, "weight", leave_parametrized = False)
        assert type(a) is torch.nn.Linear and "weight" in a._parameters
    assert seen[0] is seen[1] is seen[2] and seen[0] is not torch.nn.Linear
    parametrize.register_parametrization(b, "weight", Double())
    assert type(b) is not seen[0] and torch.equal(
        b.weight, 2 * b.parametrizations["weight"].original
    )


def test_relax_strict_dynamic_aliases_only_marked_tensors():
    M = _load()
    ids = torch.zeros(2, 16, dtype = torch.long)
    torch._dynamo.mark_dynamic(ids, 1)
    other = torch.zeros(2, 16)
    pos = (torch.zeros(1, 16), torch.zeros(1, 16))
    out = M._relax_strict_dynamic((other, pos, {"ple_input_ids": ids, "x": None}))
    assert out[0] is other and out[1] is pos
    alias = out[2]["ple_input_ids"]
    assert alias is not ids and alias.data_ptr() == ids.data_ptr() and alias.shape == ids.shape
    assert not getattr(alias, "_dynamo_dynamic_indices", None)
    assert ids._dynamo_dynamic_indices == {1}  # the caller's tensor keeps its own mark
    unmarked = (other, {"a": other})
    assert M._relax_strict_dynamic(unmarked) is unmarked


def test_strict_marked_kwarg_does_not_demote_the_block(monkeypatch, capsys):
    """Qwen4Exp passes input_ids (strictly marked by unsloth_zoo's loss bookkeeping) as
    ple_input_ids; a specialized length raised ConstraintViolationError, demoting the layer."""
    M = _load()
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend = "eager", fullgraph = False, dynamic = None),
    )
    torch._dynamo.reset()

    class PleDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            ple_input_ids = None,
            **kwargs,
        ):
            out = self.proj(hidden_states)
            if ple_input_ids.shape[1] == 16:  # a block that specializes the length
                out = out + 1
            return out

    M._wrap(PleDecoderLayer, {})
    layer = PleDecoderLayer().train()
    x = torch.randn(2, 16, 8, requires_grad = True)
    ids = torch.zeros(2, 16, dtype = torch.long)
    torch._dynamo.mark_dynamic(ids, 1)
    layer(x, ple_input_ids = ids).sum().backward()
    torch._dynamo.reset()
    assert "block compile failed" not in capsys.readouterr().out
    assert layer._unsloth_block_mode != M._OFF


def _like_transformers_5_23(forward):
    """GradientCheckpointingLayer.__init_subclass__ (transformers >= 5.23): checkpoint in `forward`."""

    @functools.wraps(forward)
    def forward_with_checkpointing(self, *args, **kwargs):
        if (
            self.gradient_checkpointing
            and self.training
            and type(self).forward is forward_with_checkpointing
        ):
            return self._gradient_checkpointing_func(
                functools.partial(forward, self, **kwargs), *args
            )
        return forward(self, *args, **kwargs)

    return forward_with_checkpointing


def _checkpointed_layer(cls, reentrant = False):
    layer = cls()
    layer.gradient_checkpointing = True
    calls = []

    def checkpoint(fn, *args):
        calls.append(len(args))
        if reentrant:  # the forward of a reentrant checkpoint runs without grad
            with torch.no_grad():
                return fn(*args)
        return fn(*args)

    layer._gradient_checkpointing_func = checkpoint
    return layer, calls


def test_block_keeps_transformers_layer_checkpointing(monkeypatch):
    # The block replaced the class forward, hiding transformers' wrapper, so nothing was checkpointed.
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    cls = _toy_class()
    stock = cls.forward
    cls.forward = _like_transformers_5_23(stock)
    M._wrap(cls, {})
    assert getattr(cls.forward, M._MARK) and cls.forward._unsloth_block_original is stock
    layer, calls = _checkpointed_layer(cls)
    x = torch.randn(1, 4, 8, requires_grad = True)
    out = layer(x, attention_mask = None)
    torch.testing.assert_close(out, layer.proj(x) + x)
    assert calls == [1]
    layer.eval()(x)
    assert calls == [1]
    # Wrapping again (a second model load) keeps one checkpoint.
    M._wrap(cls, {})
    layer.train()(x)
    assert calls == [1, 1]
    # The grad-mode trial must call the block, not the checkpoint shim (that recursed until RecursionError).
    fresh, fresh_calls = _checkpointed_layer(cls, reentrant = True)
    out = fresh(x)
    torch.testing.assert_close(out, fresh.proj(x) + x)
    assert fresh_calls == [1]


def test_block_leaves_call_level_checkpointing_alone(monkeypatch):
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    cls = _toy_class()
    M._wrap(cls, {})
    assert not getattr(cls.forward, "_unsloth_block_checkpoint_outside", False)
    layer, calls = _checkpointed_layer(cls)
    layer(torch.randn(1, 4, 8))
    assert calls == []  # transformers <= 5.22 checkpoints in __call__, outside the block already


def test_block_keeps_real_transformers_checkpointing(monkeypatch):
    modeling_layers = pytest.importorskip("transformers.modeling_layers")
    M = _load()

    class ToyGCDecoderLayer(modeling_layers.GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
        ):
            return self.proj(hidden_states) + hidden_states

    if ToyGCDecoderLayer.forward.__code__.co_name != "forward_with_checkpointing":
        pytest.skip("this transformers checkpoints in __call__")
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    M._wrap(ToyGCDecoderLayer, {})
    layer, calls = _checkpointed_layer(ToyGCDecoderLayer)
    layer(torch.randn(1, 4, 8, requires_grad = True))
    assert calls == [1]


@pytest.mark.parametrize("extra_fwd", [0, 1], ids = ["fla_0.5_six_outputs", "fla_main_seven_outputs"])
def test_opaque_gdn_op_takes_any_fla_return_length(monkeypatch, extra_fwd):
    # fla main (#1731) returns an extra h; a fixed 6-way unpack silently disabled the op.
    import contextlib

    M = _load("_gdn_op")
    if M._gdn_chunk is None:
        pytest.skip("torch.library.custom_op unavailable")
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())

    def fwd(q, k, v, g, beta, scale, initial_state, output_final_state, **kw):
        A = torch.zeros(*k.shape[:-1], 64, dtype = k.dtype)
        return (g.float().cumsum(1), v * scale, A, None, None, None) + (None,) * extra_fwd

    def bwd(q, k, v, g, beta, A, scale, initial_state, do, dht, **kw):
        return (do * 0 + 1, do * 0 + 2, do * scale, beta * 0 + 3, g * 0 + 4, None, None, None)

    def l2_fwd(x):
        return x.clone(), x.new_ones(x.shape[:-1], dtype = torch.float32)

    def l2_bwd(x, rstd, dx):
        return dx.clone()

    monkeypatch.setattr(M, "_FLA", (fwd, bwd, l2_fwd, l2_bwd))
    B, T, H, D = 1, 8, 2, 4
    q, k, v = (torch.randn(B, T, H, D, requires_grad = True) for _ in range(3))
    g, beta = (torch.randn(B, T, H, requires_grad = True) for _ in range(2))
    out = M._gdn_chunk(q, k, v, g, beta, 0.5)
    assert len(out) == 7
    torch.testing.assert_close(out[0], v.detach() * 0.5)
    out[0].sum().backward()
    assert torch.all(v.grad == 0.5) and torch.all(q.grad == 1) and torch.all(g.grad == 4)


def test_block_keeps_checkpointing_under_accelerate_device_map_hooks(monkeypatch):
    # A model split over GPUs: accelerate's hook calls `_old_forward`, bound at load to transformers' wrapper. After
    # the block replaced the class forward that wrapper stopped checkpointing (and the block never ran), so every
    # layer kept its activations: Inkling-2-Mini on 2x T4 grew ~0.45 GB per layer and ran out of memory.
    hooks = pytest.importorskip("accelerate.hooks")
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    cls = _toy_class()
    stock = cls.forward
    cls.forward = _like_transformers_5_23(stock)
    layer, calls = _checkpointed_layer(cls)
    hooks.add_hook_to_module(layer, hooks.ModelHook())
    before = cls.forward
    M._wrap(cls, {})
    x = torch.randn(1, 4, 8, requires_grad = True)
    layer(x)
    assert calls == []  # the stale bound wrapper: no checkpoint
    holder = torch.nn.Module()
    holder.layer = layer
    assert M._rebind_hooked_instances(holder, cls, before) == 1
    out = layer(x)
    torch.testing.assert_close(out, layer.proj(x) + x)
    assert calls == [1]
    # Unhooked instances and other classes are left alone.
    plain, plain_calls = _checkpointed_layer(cls)
    holder.plain = plain
    assert M._rebind_hooked_instances(holder, cls, before) == 0
    plain(x)
    assert plain_calls == [1]


def _like_transformers_5_24(forward):
    """transformers >= 5.24: checkpoint only under grad, and clear the renamed `cache` keyword first."""

    @functools.wraps(forward)
    def forward_with_checkpointing(self, *args, **kwargs):
        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and type(self).forward is forward_with_checkpointing
        ):
            kwargs["use_cache"] = False
            kwargs["cache"] = None
            return self._gradient_checkpointing_func(
                functools.partial(forward, self, **kwargs), *args
            )
        return forward(self, *args, **kwargs)

    return forward_with_checkpointing


def _cache_writing_class():
    class CacheDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(
            self,
            hidden_states,
            cache = None,
            use_cache = False,
        ):
            if cache is not None:  # stands in for DynamicCache.update
                cache.append(hidden_states.shape[1])
            return self.proj(hidden_states) + hidden_states

    return CacheDecoderLayer


def test_block_checkpoint_shim_keeps_the_5_24_cache_contract(monkeypatch):
    # transformers 5.24 renamed `past_key_values` to `cache` and runs a training-mode layer as is without grad.
    # The shim checkpointed under no_grad and passed `cache` into the block, which wrote it on each run.
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    cls = _cache_writing_class()
    cls.forward = _like_transformers_5_24(cls.forward)
    M._wrap(cls, {})
    layer, calls = _checkpointed_layer(cls)
    layer.train()
    x = torch.randn(1, 4, 8, requires_grad = True)
    cache = []
    with torch.no_grad():  # generate() from a model left in training mode
        out = layer(x, cache = cache, use_cache = True)
    torch.testing.assert_close(out, layer.proj(x) + x)
    assert calls == [] and cache == [4]
    cache = []
    out = layer(x, cache = cache, use_cache = True)  # a training forward handed a cache
    out.sum().backward()
    assert calls == [1] and cache == []
    assert M._has_cache(
        (), {"cache": []}
    )  # cache calls keep the stock forward, as `past_key_values` does


def test_block_rebinds_hooked_layers_of_a_second_model(monkeypatch):
    # Two models of one layer class, both dispatched (accelerate hooks bound) before either was patched: the
    # class was wrapped for the first, so the second's rebind was skipped and it ran with no checkpointing.
    hooks = pytest.importorskip("accelerate.hooks")
    M = _load()
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "__version__", "2.7.1+cu126")
    cls = _toy_class()
    cls.forward = _like_transformers_5_23(cls.forward)
    models, counts = [], []
    for _ in range(2):
        layer, calls = _checkpointed_layer(cls)
        hooks.add_hook_to_module(layer, hooks.ModelHook())
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([layer])
        models.append(model)
        counts.append(calls)
    assert M.patch_block_compile(models[0]) == [cls.__name__]
    assert M.patch_block_compile(models[1]) == []  # already wrapped
    x = torch.randn(1, 4, 8, requires_grad = True)
    for model, calls in zip(models, counts):
        layer = model.layers[0].train()
        torch.testing.assert_close(layer(x), layer.proj(x) + x)
        assert calls == [1]
