"""A second load in the same process must route its GatedDeltaNet kernels like the first one.

transformers < 5.23 copies `causal_conv1d_fn` and `chunk_gated_delta_rule` into each GatedDeltaNet
instance at __init__. The block compiler's GDN routing ran once per process, at the first block call,
so the first model kept the raw conv (built before the module global was routed) and a later model kept
the raw chunk kernel (its loader re-bound the global). Each load then trained a different graph, and the
first load's LoRA grads differed from a second identical load (qwen3next, 0.0227 vs a 0.02 tolerance).
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models")


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "_ugdn_reload._block_compile", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _depthwise(x, w):
    return x


def _install_fakes(monkeypatch):
    """A fake fla chunk module, a fake `_gdn_op` sibling and a fake modeling module with a GDN class."""
    pkg = types.ModuleType("_ugdn_reload")
    pkg.__path__ = []
    gdn_op = types.ModuleType("_ugdn_reload._gdn_op")
    gdn_op._causal_depthwise_conv = _depthwise
    gdn_op._fla_ops = lambda make_inputs: True
    gdn_op.varlen_ok = lambda: False
    fla = types.ModuleType("_ugdn_fake.fla.ops.gated_delta_rule.chunk")

    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g = None,
        beta = None,
        **kwargs,
    ):
        return q, None

    chunk_gated_delta_rule.__module__ = fla.__name__
    fla.chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling = types.ModuleType("_ugdn_fake_modeling")
    source = (
        "import torch\n"
        "def legacy_causal_conv1d_fn(x, weight, bias = None, activation = None, seq_idx = None):\n"
        "    return x\n"
        "causal_conv1d_fn = legacy_causal_conv1d_fn\n"
        "class GatedDeltaNet(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.num_v_heads, self.head_k_dim, self.head_v_dim = 2, 4, 4\n"
        "        self.causal_conv1d_fn = causal_conv1d_fn\n"
        "        self.chunk_gated_delta_rule = chunk_gated_delta_rule\n"
        "    def forward(self, x):\n"
        "        x = self.causal_conv1d_fn(x, None)\n"
        "        return self.chunk_gated_delta_rule(x, x, x)[0]\n"
        "class Layer(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.linear_attn = GatedDeltaNet()\n"
    )
    modeling.chunk_gated_delta_rule = fla.chunk_gated_delta_rule
    exec(compile(source, "<fake_modeling>", "exec"), modeling.__dict__)
    for name, module in (
        ("_ugdn_reload", pkg),
        ("_ugdn_reload._gdn_op", gdn_op),
        (fla.__name__, fla),
        (modeling.__name__, modeling),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return fla, modeling


def _routed(fn):
    return getattr(fn, "_unsloth_block_adapter", False)


def test_second_load_routes_gdn_like_the_first(monkeypatch):
    M = _load()
    M.__package__ = "_ugdn_reload"
    monkeypatch.setattr(M, "_GDN_OP", [None])
    monkeypatch.setattr(M, "_route_sm_count", lambda: None)
    fla, modeling = _install_fakes(monkeypatch)
    raw = fla.chunk_gated_delta_rule
    hidden = torch.zeros(1, 4, 8)

    # Load 1: the instance is built before the process-wide self-check routes anything.
    first = modeling.Layer()
    M._prepare_gdn(first, hidden)
    assert M._GDN_OP[0] is True
    gdn1 = first.linear_attn
    assert _routed(gdn1.__dict__["chunk_gated_delta_rule"]), "first load: chunk kernel not routed"
    assert _routed(gdn1.__dict__["causal_conv1d_fn"]), "first load: conv kept the raw kernel"

    # Load 2: its loader re-binds the module global to the raw fla kernel before building the model.
    modeling.chunk_gated_delta_rule = raw
    second = modeling.Layer()
    pending = (
        M._gdn_setup_pending(second) if hasattr(M, "_gdn_setup_pending") else M._GDN_OP[0] is None
    )
    if pending:
        M._prepare_gdn(second, hidden)
    gdn2 = second.linear_attn
    assert _routed(gdn2.__dict__["chunk_gated_delta_rule"]), "second load: chunk kernel not routed"
    assert _routed(gdn2.__dict__["causal_conv1d_fn"]), "second load: conv not routed"
    # Both loads now trace the same code: Dynamo guards the instance kernels' __code__.
    for name in ("chunk_gated_delta_rule", "causal_conv1d_fn"):
        assert gdn1.__dict__[name].__code__ is gdn2.__dict__[name].__code__, name
    assert not M._gdn_setup_pending(second) and not M._gdn_setup_pending(first)


def test_packing_wrapped_conv_stays_outermost(monkeypatch):
    M = _load()
    M.__package__ = "_ugdn_reload"
    monkeypatch.setattr(M, "_GDN_OP", [True])
    fla, modeling = _install_fakes(monkeypatch)
    layer = modeling.Layer()
    gdn = layer.linear_attn
    wrapper = lambda *a, **k: None  # noqa: E731
    gdn.causal_conv1d_fn = wrapper
    gdn._unsloth_varlen_wrapped = True
    M._prepare_gdn(layer, torch.zeros(1, 4, 8))
    assert gdn.__dict__["causal_conv1d_fn"] is wrapper


def _packing_scan_wrapper(scan_orig, module, calls):
    # The shape of utils/packing.py's per-instance scan wrapper: @wraps copies fla's __module__.
    from functools import wraps

    @wraps(scan_orig)
    def scan_fn(
        *args,
        _orig = scan_orig,
        _module = module,
        **kwargs,
    ):
        calls.append(("packing", dict(kwargs)))
        if kwargs.get("cu_seqlens") is None:
            kwargs["cu_seqlens"] = "packed"
        return _orig(*args, **kwargs)

    return scan_fn


@pytest.mark.parametrize("first_load", [True, False])
def test_packing_wrapped_scan_stays_outside_the_chunk_adapter(monkeypatch, first_load):
    # Hybrid packing wraps an instance's kernels before the first compiled forward; the adapter must go under it.
    M = _load()
    M.__package__ = "_ugdn_reload"
    monkeypatch.setattr(M, "_GDN_OP", [None if first_load else True])
    monkeypatch.setattr(M, "_route_sm_count", lambda: None)
    fla, modeling = _install_fakes(monkeypatch)
    layer = modeling.Layer()
    gdn = layer.linear_attn
    calls = []
    raw = gdn.chunk_gated_delta_rule
    wrapper = _packing_scan_wrapper(raw, gdn, calls)
    gdn._unsloth_varlen_orig_scan = raw
    gdn.chunk_gated_delta_rule = wrapper
    gdn._unsloth_varlen_wrapped = True

    M._prepare_gdn(layer, torch.zeros(1, 4, 8))

    outer = gdn.__dict__["chunk_gated_delta_rule"]
    assert outer is wrapper, "the chunk adapter was put outside packing's scan wrapper"
    assert _routed(
        outer.__kwdefaults__["_orig"]
    ), "the kernel under packing's wrapper was not routed"
    q = torch.zeros(1, 4, 2, 4)
    outer(q, q, q, g = q[..., 0], beta = q[..., 0], use_qk_l2norm_in_kernel = True)
    assert calls and calls[0][0] == "packing" and calls[0][1].get("cu_seqlens") is None
