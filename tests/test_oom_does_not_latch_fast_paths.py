# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One OOM near the memory ceiling must not turn a fast path off for the rest of the process, and checkpoint control
flow must not read as a kernel failure. Each test injects the error once, then the next call takes the fast path.
CPU only: every module is loaded by file path, so no accelerator is needed."""

import ast
import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.join(os.path.dirname(__file__), "..", "unsloth")


def _load(relative, name):
    path = os.path.join(_ROOT, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _oom():
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _compile_oom():
    # What Dynamo raises when the OOM happens while Inductor compiles or autotunes the graph.
    from torch._dynamo.exc import BackendCompilerFailed
    try:
        return BackendCompilerFailed(lambda *a: None, _oom(), None)
    except TypeError:  # older signature without first_useful_frame
        return BackendCompilerFailed(lambda *a: None, _oom())


# Block compile: BackendCompilerFailed wrapping an OOM must not demote the block.


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


def test_block_compile_oom_while_compiling_does_not_demote(monkeypatch):
    M = _load("models/_block_compile.py", "_u_block_compile_oom")
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(M._IN_BLOCK[0])
            if len(calls) == 1:
                raise _compile_oom()
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cls = _toy_class()
    M._wrap(cls, {})
    layer = cls()
    x = torch.randn(2, 4, 8, requires_grad = True)
    with pytest.raises(Exception) as error:
        layer(x)
    assert isinstance(getattr(error.value, "inner_exception", None), torch.OutOfMemoryError)
    assert layer._unsloth_block_mode == M._FIRST_MODE
    # The next call compiles the block in its first mode again.
    out = layer(x)
    assert calls[-1] == M._FIRST_MODE
    torch.testing.assert_close(out, layer.proj(x) + x)
    assert layer._unsloth_block_mode == M._FIRST_MODE


# Block compile: an OOM in the band kernel's self check must not cache "not ready" on the layer.


def test_band_kernel_self_check_oom_is_not_cached(monkeypatch):
    M = _load("models/_block_compile.py", "_u_block_compile_band")

    class Attention(torch.nn.Module):
        is_sliding, sliding_window, head_dim = True, 64, 256

    layer = torch.nn.Module()
    layer.self_attn = Attention()
    checks = []
    lh = types.ModuleType("unsloth.kernels.large_head_attention")
    lh._SELF_CHECK = {}
    lh._enabled = lambda: True
    lh._measured_faster_on = lambda device: True

    def self_check(device, dtype, head_dim):
        checks.append(dtype)
        if len(checks) == 1:
            return False  # what the real self check returns after an OOM, caching nothing
        lh._SELF_CHECK[(str(device), dtype, head_dim)] = True
        return True

    lh._self_check = self_check
    kernels = types.ModuleType("unsloth.kernels")
    kernels.large_head_attention = lh
    package = types.ModuleType("unsloth")
    package.kernels = kernels
    monkeypatch.setitem(sys.modules, "unsloth", package)
    monkeypatch.setitem(sys.modules, "unsloth.kernels", kernels)
    monkeypatch.setitem(sys.modules, "unsloth.kernels.large_head_attention", lh)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    monkeypatch.setattr(torch.version, "hip", None)
    hidden = types.SimpleNamespace(is_cuda = True, device = torch.device("cuda", 0))
    assert M._band_kernel_ready(layer, hidden) is False
    assert "_unsloth_band_kernel_ready" not in layer.__dict__
    assert M._band_kernel_ready(layer, hidden) is True
    assert layer.__dict__["_unsloth_band_kernel_ready"] == ("cuda:0", True)


# GRPO no-grad PrefixGrouper: an OOM skips PG for this batch only.


def _pg_handler():
    path = os.path.join(_ROOT, "models", "rl_replacements.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.name == "_pg_err"
    ]
    assert len(handlers) == 1, "expected one `except Exception as _pg_err` in rl_replacements.py"
    body = ast.Module(body = handlers[0].body, type_ignores = [])
    return compile(body, path, "exec")


@pytest.mark.parametrize(
    "make_error, latched",
    [(_oom, False), (_compile_oom, False), (lambda: RuntimeError("flex compile failed"), True)],
)
def test_prefix_grouper_nograd_oom_does_not_disable_it(monkeypatch, make_error, latched):
    code = _pg_handler()
    model = types.SimpleNamespace()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")
    namespace = {
        "torch": torch,
        "os": os,
        "unwrapped_model": model,
        "_unsloth_head_hooked": False,
        "UNSLOTH_ENABLE_LOGGING": False,
        "_pg_err": make_error(),
    }
    exec(code, namespace)
    assert namespace["_pg_use"] is False and namespace["_pg_result"] is None
    # The gate the next batch reads: still off only after a real failure.
    assert getattr(model, "_unsloth_prefix_grouper_nograd_disabled", False) is latched


# large_head_attention: checkpoint control flow must propagate and keep the kernel on.


def _large_head(monkeypatch):
    pytest.importorskip("triton")
    LH = _load("kernels/large_head_attention.py", "_u_large_head_attention_oom")
    monkeypatch.setattr(LH, "_enabled", lambda: True)
    monkeypatch.setattr(LH, "_shapes_qualify", lambda q, k, v: True)
    monkeypatch.setattr(LH, "_is_capturing", lambda device: False)
    monkeypatch.setattr(LH, "_dense_intervals", lambda B, S, causal, device: (None, None))
    monkeypatch.setattr(LH, "_self_check", lambda device, dtype, head_dim = 512: True)
    monkeypatch.setattr(torch.version, "hip", None)
    return LH


@pytest.mark.parametrize("error_name", ["_StopRecomputationError", "CheckpointError"])
def test_large_head_checkpoint_control_flow_propagates(monkeypatch, error_name):
    import torch.utils.checkpoint as checkpoint

    LH = _large_head(monkeypatch)
    error_type = getattr(checkpoint, error_name)
    raised = []

    def kernel(*args):
        if not raised:
            raised.append(1)
            raise error_type()
        return args[0]

    monkeypatch.setattr(LH, "large_head_attention", kernel)
    seen = []

    def original(module, query, *args, **kwargs):
        seen.append(1)
        return "sdpa", None

    router = LH._make_router(original)
    module = types.SimpleNamespace(is_causal = True)
    q = torch.zeros(1, 4, 8, 512, dtype = torch.bfloat16)
    kv = torch.zeros(1, 2, 8, 512, dtype = torch.bfloat16)
    key = (str(q.device), q.dtype)
    monkeypatch.setitem(LH._SELF_CHECK, key, True)
    with pytest.raises(error_type):
        router(module, q, kv, kv, None)
    assert LH._SELF_CHECK[key] is True and seen == []
    out, _ = router(module, q, kv, kv, None)
    assert isinstance(out, torch.Tensor) and seen == []


# Gemma2 softcapping attention: control flow and an OOM while compiling must not latch eager.


@pytest.mark.parametrize("make_error", ["stop", "checkpoint", "compile_oom"])
def test_softcap_attention_does_not_latch_eager(monkeypatch, make_error):
    import torch.utils.checkpoint as checkpoint

    fk = _load("kernels/flex_attention.py", "_u_flex_attention_oom")
    if not hasattr(fk, "_softcapping_attention"):
        pytest.skip("flex attention build without the softcapping fallback")
    error = {
        "stop": checkpoint._StopRecomputationError,
        "checkpoint": checkpoint.CheckpointError,
        "compile_oom": _compile_oom,
    }[make_error]()
    calls = []

    def compiled(*args):
        calls.append("compiled")
        if len(calls) == 1:
            raise error
        return "compiled"

    compiled._torchdynamo_orig_callable = lambda *args: "eager"
    monkeypatch.setattr(fk, "_compiled_slow_attention_softcapping", compiled)
    monkeypatch.setattr(fk, "_SOFTCAP_EAGER", {})
    Q = torch.zeros(1, 1, 4, 2)
    with pytest.raises(type(error)):
        fk._softcapping_attention(Q, Q, Q, None, None, 1, 4)
    assert fk._SOFTCAP_EAGER == {}
    assert fk._softcapping_attention(Q, Q, Q, None, None, 1, 4) == "compiled"
