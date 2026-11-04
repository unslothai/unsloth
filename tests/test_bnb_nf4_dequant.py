# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch
from real_accelerator import has_real_cuda

REPO = Path(__file__).resolve().parents[1]
_SETUP = textwrap.dedent("""
import importlib, inspect, os, sys, types
import torch
root = sys.argv[1]
package = types.ModuleType("bdq"); package.__path__ = [os.path.join(root, "unsloth", "kernels")]
sys.modules["bdq"] = package
M = importlib.import_module("bdq.bnb_dequant")
""")
_CPU = _SETUP + textwrap.dedent(r"""
import builtins
from unittest.mock import patch
functional = types.ModuleType("bitsandbytes.functional")
bnb = types.ModuleType("bitsandbytes"); bnb.__path__ = []; bnb.functional = functional
utils = types.ModuleType("bdq.utils"); utils._USE_NF4_KERNELS = True
nf4 = types.ModuleType("bdq.nf4")
def forbidden(*args, **kwargs): raise AssertionError("eager fast call")
nf4.dequantize_nf4 = forbidden
functional.QuantState = type("QuantState", (), {})
sys.modules.update({"bitsandbytes": bnb, "bitsandbytes.functional": functional,
    "bdq.utils": utils, "bdq.nf4": nf4}); package.utils = utils
imports = builtins.__import__
def no_probe(name, *args, **kwargs):
    if name in ("", "utils", "bdq.utils", "bitsandbytes.functional"):
        raise AssertionError("DEFAULT_OFF_PROBE_IMPORT")
    return imports(name, *args, **kwargs)
calls = []
def original(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
    calls.append((A, quant_state, absmax, out, blocksize, quant_type)); return A + 7
functional.dequantize_4bit = original
os.environ["UNSLOTH_BNB_TRITON"] = "1"; os.environ["UNSLOTH_BNB_NF4_LINEAR"] = "auto"
with patch.object(torch.version, "hip", None):
    for flag in (None, "0", "yes"):
        if flag is None: os.environ.pop("UNSLOTH_DECODE_FUSED_NF4", None)
        else: os.environ["UNSLOTH_DECODE_FUSED_NF4"] = flag
        with patch.object(builtins, "__import__", no_probe):
            assert not M.install_bnb_nf4_dequant(), "DEFAULT_OFF"
        assert functional.dequantize_4bit is original
    os.environ["UNSLOTH_DECODE_FUSED_NF4"] = "1"
    for name in ("UNSLOTH_BNB_TRITON", "UNSLOTH_BNB_NF4_LINEAR"):
        with patch.dict(os.environ, {name: "0"}): assert not M.install_bnb_nf4_dequant()
    with patch.object(torch.version, "hip", "6.4"): assert not M.install_bnb_nf4_dequant()
    utils._USE_NF4_KERNELS = False; assert not M.install_bnb_nf4_dequant()
    utils._USE_NF4_KERNELS = True
    state_class = functional.QuantState
    del functional.QuantState
    assert not M.install_bnb_nf4_dequant() and functional.dequantize_4bit is original
    functional.QuantState = state_class
    functional.dequantize_4bit = lambda *args, **kwargs: None
    assert not M.install_bnb_nf4_dequant(); functional.dequantize_4bit = original
    assert M.install_bnb_nf4_dequant()
    wrapped = functional.dequantize_4bit
    assert wrapped.__wrapped__ is original and inspect.signature(wrapped) == inspect.signature(original)
    assert M.install_bnb_nf4_dequant() and functional.dequantize_4bit is wrapped
    value = torch.ones(2, 4); state = functional.QuantState()
    assert torch.equal(wrapped(value, state, quant_type="nf4"), value + 7)
    assert calls[-1][0] is value and calls[-1][1:] == (state, None, None, None, "nf4")
    os.environ["UNSLOTH_DECODE_FUSED_NF4"] = "force"; assert M.install_bnb_nf4_dequant()
    assert torch.equal(wrapped(value), value + 7)
    os.environ["UNSLOTH_DECODE_FUSED_NF4"] = "0"; assert torch.equal(wrapped(value), value + 7)
    M.uninstall_bnb_nf4_dequant(); assert functional.dequantize_4bit is original
    import functools
    def legacy(A, quant_state=None, absmax=None, out=None, blocksize=64, quant_type="fp4"):
        if blocksize not in (64, 128, 256, 512, 1024, 2048, 4096):
            raise ValueError(blocksize)
        return A + blocksize
    @functools.wraps(legacy)
    def decorated(*args, **kwargs):
        return legacy(*args, **kwargs)
    assert decorated.__defaults__ is None
    os.environ["UNSLOTH_DECODE_FUSED_NF4"] = "1"
    for current in (legacy, decorated):
        functional.dequantize_4bit = current
        assert M.install_bnb_nf4_dequant()
        installed = functional.dequantize_4bit
        assert torch.equal(installed(value, state), current(value, state)), "LEGACY_DEFAULT_FORWARD"
        assert torch.equal(installed(value, state, blocksize=128), current(value, state, blocksize=128))
        assert installed.__defaults__ == (None, None, None, 64, "fp4")
        M.uninstall_bnb_nf4_dequant()
        assert functional.dequantize_4bit is current
    def required(A, quant_state, absmax=None, out=None, blocksize=64, quant_type="fp4"):
        return A
    functional.dequantize_4bit = required
    assert not M.install_bnb_nf4_dequant()
    assert functional.dequantize_4bit is required
assert not torch.cuda.is_initialized()
print("PUBLIC DEQUANT CPU OK")
""")
_GPU = _SETUP + textwrap.dedent(r"""
os.environ["UNSLOTH_DECODE_FUSED_NF4"] = "1"
os.environ["UNSLOTH_BNB_TRITON"] = "1"; os.environ["UNSLOTH_BNB_NF4_LINEAR"] = "auto"
sys.path.insert(0, root)
import unsloth
import bitsandbytes as bnb
import bitsandbytes.functional as BF
from unittest.mock import patch
from unsloth.kernels import nf4 as K, utils as U
assert U._USE_NF4_KERNELS
sys.modules.update({"bdq.nf4": K, "bdq.utils": U}); package.utils = U
original = BF.dequantize_4bit
assert M.install_bnb_nf4_dequant() and BF.dequantize_4bit.__wrapped__ is original
def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
def quant(dtype, nested=True, storage=torch.uint8, kind="nf4"):
    torch.manual_seed(3407)
    return BF.quantize_4bit(torch.randn(64, 256, device="cuda", dtype=dtype),
        quant_type=kind, compress_statistics=nested, quant_storage=storage)
def has_nf4(fn, q):
    report = torch._dynamo.explain(fn)(q)
    assert report.graph_break_count == 0, report.break_reasons
    return any("unsloth.dequantize_nf4" in str(n.target) for g in report.graphs for n in g.graph.nodes)
def check(q, s, eligible):
    fn = lambda value: BF.dequantize_4bit(value, s)
    reference_eager = original(q, s).clone()
    reference_compiled = torch.compile(lambda value: original(value, s), fullgraph=True)(q).clone()
    with patch.object(M, "_FAST", side_effect=AssertionError("eager fast call")):
        assert bits(fn(q), reference_eager), ("DEQUANT_EAGER_EXACT", s.code.dtype)
    engaged = has_nf4(fn, q)
    run = torch.compile(fn, fullgraph=True)
    actual = run(q)
    assert bits(actual, reference_compiled), ("DEQUANT_WEIGHT_EXACT", "compiled_stock", s.code.dtype)
    if eligible:
        assert bits(actual, reference_eager), ("DEQUANT_WEIGHT_EXACT", "eager_golden", s.code.dtype)
    assert engaged == eligible, ("DEQUANT_ENGAGEMENT", s.code.dtype, eligible)
    graphs = dict(torch._dynamo.utils.counters["stats"])
    assert bits(run(q), reference_compiled)
    assert dict(torch._dynamo.utils.counters["stats"]) == graphs
dtype2 = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.inference_mode():
    for dtype in (torch.float16, dtype2, torch.float32):
        q, s = quant(dtype)
        check(q, s, True); check(q.t(), s, True)
        with patch.object(M, "_FAST", side_effect=AssertionError("eager fast call")):
            assert bits(BF.dequantize_4bit(q, s), original(q, s))
    q, s = quant(torch.float32)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        check(q, s, True)
        low_q, low_s = quant(torch.float32)
    finally: torch.set_default_dtype(previous)
    assert low_s.code.dtype == torch.float16
    print("LOW_PRECISION_CODE_BEGIN", flush=True)
    check(low_q, low_s, False)
    print("LOW_PRECISION_CODE_STOCK_OK", flush=True)
    for nested, storage, kind in ((False, torch.uint8, "nf4"),
            (True, torch.float32, "nf4"), (True, torch.uint8, "fp4")):
        q, s = quant(torch.float16, nested, storage, kind); check(q, s, False)
    q, flat = quant(torch.float16, nested=False)
    offset = flat.absmax.mean()
    secondary, state2 = BF.quantize_blockwise((flat.absmax - offset).half(), blocksize=256)
    assert state2.dtype == torch.float16
    half_secondary = BF.QuantState(absmax=secondary, shape=flat.shape, dtype=flat.dtype,
        blocksize=flat.blocksize, code=flat.code, quant_type="nf4", offset=offset, state2=state2)
    check(q, half_secondary, False)
    for nested in (True, False):
        q, s = quant(torch.float16, nested=nested)
        out = torch.empty(s.shape, device="cuda", dtype=s.dtype)
        fn = lambda value: BF.dequantize_4bit(value, s, out=out)
        reference_eager = original(q, s, out=out).clone()
        reference_compiled = None
        try:
            reference_compiled = torch.compile(lambda value: original(value, s, out=out), fullgraph=True)(q).clone()
        except torch._dynamo.exc.TorchRuntimeError as error:
            assert "bitsandbytes.dequantize_4bit.out" in str(error) and "Expected out.shape" in str(error), str(error)
            print("STOCK_COMPILED_OUT_SHAPE_ERROR", nested, flush=True)
        with patch.object(M, "_FAST", side_effect=AssertionError("eager out fast call")):
            eager = fn(q)
            assert eager.data_ptr() == out.data_ptr() and bits(eager, reference_eager)
        print("COMPILED_OUT_BEGIN", nested, flush=True)
        assert has_nf4(fn, q), ("DEQUANT_OUT_ENGAGEMENT", nested)
        run = torch.compile(fn, fullgraph=True)
        actual = run(q)
        assert actual.data_ptr() == out.data_ptr(), ("DEQUANT_OUT_ALIAS", nested)
        assert bits(actual, reference_eager), ("DEQUANT_OUT_EXACT", nested)
        if reference_compiled is not None:
            assert bits(actual, reference_compiled), ("DEQUANT_OUT_COMPILED_EXACT", nested)
        graphs = dict(torch._dynamo.utils.counters["stats"])
        assert bits(run(q), reference_eager)
        assert dict(torch._dynamo.utils.counters["stats"]) == graphs
import peft
q, s = quant(torch.float16)
base = bnb.nn.Linear4bit(256, 64, bias=True, compute_dtype=torch.float16,
    quant_type="nf4", compress_statistics=True).cuda()
base.weight = bnb.nn.Params4bit(q, quant_state=s, bnb_quantized=True, quant_type="nf4", requires_grad=False)
def no_fast(*args, **kwargs): raise AssertionError("eager/native forward entered NF4")
with torch.inference_mode(), patch.object(M, "_FAST", no_fast):
    for rows in (1, 4):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.float16)
        with patch.object(BF, "dequantize_4bit", original): reference = base(x)
        assert bits(base(x), reference)
        if hasattr(torch.ops.bitsandbytes, "gemm_4bit"):
            assert not has_nf4(lambda value: base(value), x)
            assert bits(torch.compile(base, fullgraph=True)(x), reference)
owner = torch.nn.ModuleDict({"proj": base}); owner.is_loaded_in_4bit = True
model = peft.get_peft_model(owner, peft.LoraConfig(r=8, lora_alpha=8, lora_dropout=0, target_modules=["proj"]))
module = owner.proj
with torch.no_grad():
    module.lora_A["default"].weight.normal_(std=0.1); module.lora_B["default"].weight.normal_(std=0.1)
value = torch.randn(17, 256, device="cuda", dtype=torch.float16)
dy = torch.randn(17, 64, device="cuda", dtype=torch.float32)
def step(run):
    model.zero_grad(set_to_none=True); x = value.detach().clone().requires_grad_()
    output = run(x); (output.float() * dy).sum().backward()
    result = [output.detach().clone(), x.grad.clone(), module.lora_A["default"].weight.grad.clone(), module.lora_B["default"].weight.grad.clone()]
    assert all(torch.isfinite(t).all() for t in result) and all(t.abs().max() > 0 for t in result[1:])
    return result
M.uninstall_bnb_nf4_dequant(); torch._dynamo.reset()
stock = step(torch.compile(module, fullgraph=True))
assert M.install_bnb_nf4_dequant(); torch._dynamo.reset()
run = torch.compile(module, fullgraph=True); ours = step(run)
assert all(bits(a, b) for a, b in zip(ours, stock))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
    replay = step(run)
assert all(bits(a, b) for a, b in zip(replay, stock))
names = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
assert any("nf4_dequant" in n or "dequantize_nf4" in n for n in names), names
M.uninstall_bnb_nf4_dequant(); assert BF.dequantize_4bit is original
print("PUBLIC DEQUANT GPU EXACT OK")
""")


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code, str(REPO)],
        text=True,
        capture_output=True,
        env=dict(os.environ),
        timeout=420,
    )


def test_bnb_dequant_installer_cpu():
    proc = _run(_CPU)
    assert proc.returncode == 0 and "PUBLIC DEQUANT CPU OK" in proc.stdout, (
        proc.stdout + proc.stderr
    )


@pytest.mark.skipif(
    not has_real_cuda() or torch.version.hip is not None, reason="needs NVIDIA CUDA"
)
def test_bnb_dequant_actual_backend_on_gpu():
    proc = _run(_GPU)
    assert proc.returncode == 0 and "PUBLIC DEQUANT GPU EXACT OK" in proc.stdout, (
        proc.stdout + proc.stderr
    )
