# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Unsloth's bitsandbytes stand-in on a GPU (CUDA or ROCm): Triton quantize / dequantize against
the torch reference and, where a working bitsandbytes is installed, against its kernels byte for
byte; Linear4bit forward and backward. Fresh interpreters, installer loaded by path."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

# Seconds, no downloads: not marked `gpu`, so it runs wherever a GPU is visible.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA or ROCm GPU")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM_INIT = os.path.join(REPO, "unsloth", "bnb_shim", "__init__.py")
SHAPES = [(64,), (65,), (17, 33), (11, 4099), (1024, 1000)]


def run(
    body,
    shim = True,
    argv = (),
):
    env = dict(os.environ)
    env["UNSLOTH_BNB_SHIM"] = "1" if shim else "0"
    env["UNSLOTH_BNB_SHIM_QUIET"] = "1"
    env.pop("PYTHONPATH", None)
    prelude = ""
    if shim:
        prelude = textwrap.dedent(
            f"""
            import importlib.util
            spec = importlib.util.spec_from_file_location("_unsloth_bnb_shim", {SHIM_INIT!r})
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            assert shim.install()
            """
        )
    code = prelude + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code, *argv], capture_output = True, text = True, env = env, timeout = 600
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    lines = [l for l in proc.stdout.splitlines() if l.startswith("OUT ")]
    assert lines, proc.stdout + proc.stderr
    return json.loads(lines[-1][4:])


QUANTIZE = f"""
import json, sys, torch
import bitsandbytes as bnb
from bitsandbytes import functional as F
torch.manual_seed(0)
out = {{}}
for shape in {SHAPES!r}:
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x = torch.randn(shape, device = "cuda").to(dtype)
        for qt in ("nf4", "fp4"):
            q, st = F.quantize_4bit(x, blocksize = 64, quant_type = qt, compress_statistics = True)
            out[f"{{shape}}-{{dtype}}-{{qt}}"] = dict(
                q = q.cpu(), absmax = st.absmax.cpu(), absmax2 = st.state2.absmax.cpu(),
                offset = st.offset.cpu(), dq = F.dequantize_4bit(q, st).float().cpu(),
            )
torch.save(out, sys.argv[1])
print("OUT " + json.dumps({{"shim": bool(getattr(bnb, "__unsloth_shim__", False))}}))
"""


def _bnb_usable():
    probe = (
        "import bitsandbytes, torch; from bitsandbytes import functional as F;"
        "F.quantize_4bit(torch.ones(64, device='cuda'), quant_type='nf4'); print('OK')"
    )
    env = dict(os.environ, UNSLOTH_BNB_SHIM = "0")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output = True, text = True, env = env)
    return proc.returncode == 0 and "OK" in proc.stdout


def test_triton_matches_torch_reference():
    out = run(
        f"""
        import json, torch
        from bitsandbytes import _quantize as Q
        from bitsandbytes import functional as F
        assert Q.triton_quantize_available("cuda")
        torch.manual_seed(1)
        diffs, total, bad_cases, bad_elems, elems, worst = 0, 0, [], 0, 0, 0.0
        for shape in {SHAPES!r}:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                x = torch.randn(shape, device = "cuda").to(dtype)
                for qt in ("nf4", "fp4"):
                    for bs in (64, 128, 4096):
                        q, a = Q.quantize_4bit_blockwise(x, bs, qt)
                        tq, ta = Q.quantize_4bit_torch(x, bs, qt)
                        assert torch.equal(a, ta)
                        diffs += int((q != tq).sum()); total += q.numel()
                        ref = Q.dequantize_4bit_blockwise(q, a, bs, qt, x.shape, dtype)
                        st = F.QuantState(absmax = a, shape = x.shape, code = F.get_4bit_type(qt, "cuda"),
                                          blocksize = bs, quant_type = qt, dtype = dtype)
                        got = F.dequantize_4bit(q.view(-1, 1), st)
                        miss = got != ref
                        elems += ref.numel()
                        if bool(miss.any()):
                            bad_cases.append(f"{{shape}} {{dtype}} {{qt}} {{bs}}")
                            bad_elems += int(miss.sum())
                            rel = ((got.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-30))[miss]
                            worst = max(worst, float(rel.max()))
        print("OUT " + json.dumps({{"diffs": diffs, "total": total, "bad_cases": bad_cases,
                                    "bad_elems": bad_elems, "elems": elems, "worst_rel": worst,
                                    "hip": torch.version.hip is not None}}))
        """
    )
    if not out["hip"]:
        # NVIDIA: Unsloth's Triton NF4 dequantize is bit-exact with the reference.
        assert not out["bad_cases"], out
    else:
        # ROCm: the kernel runs without mul.rn / RNE guarantees; allow last-bit rounding only.
        assert out["bad_elems"] <= max(1, out["elems"] // 1000), out
        assert out["worst_rel"] <= 2.0**-7, out
    # Same algorithm; the PTX path's approximate reciprocal may flip a code on a threshold tie.
    assert out["diffs"] <= max(1, out["total"] // 10000), out


def test_bytes_match_installed_bitsandbytes(tmp_path):
    if not _bnb_usable():
        pytest.skip("no working bitsandbytes to compare against")
    paths = {arm: str(tmp_path / f"{arm}.pt") for arm in ("shim", "real")}
    assert run(QUANTIZE, shim = True, argv = [paths["shim"]])["shim"]
    assert not run(QUANTIZE, shim = False, argv = [paths["real"]])["shim"]
    shim, real = torch.load(paths["shim"]), torch.load(paths["real"])
    mismatched = []
    for case, ref in real.items():
        got = shim[case]
        if not all(torch.equal(ref[k], got[k]) for k in ("q", "absmax", "absmax2", "offset", "dq")):
            mismatched.append(case)
            # ROCm: IEEE reciprocal against the ROCm build, dequantized values must still agree.
            bad = int((ref["dq"] != got["dq"]).sum())
            assert bad <= max(1, ref["dq"].numel() // 1000), (case, bad)
    if torch.version.hip is None:
        # NVIDIA: the Triton kernels issue the same PTX as bitsandbytes' fast math build.
        assert not mismatched, mismatched[:5]


def test_linear4bit_forward_backward_gpu():
    out = run(
        """
        import json, torch
        import bitsandbytes as bnb
        from bitsandbytes import functional as F
        torch.manual_seed(0)
        layer = bnb.nn.Linear4bit(256, 128, bias = True, compute_dtype = torch.bfloat16,
                                  compress_statistics = True, quant_type = "nf4").to(torch.bfloat16)
        layer = layer.to("cuda")
        x = torch.randn(3, 7, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        y = layer(x)
        y.float().pow(2).sum().backward()
        W = F.dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.bfloat16)
        x2 = x.detach().clone().requires_grad_(True)
        ref = torch.nn.functional.linear(x2, W, layer.bias.to(torch.bfloat16))
        ref.float().pow(2).sum().backward()
        print("OUT " + json.dumps({
            "fwd": bool(torch.equal(y, ref)), "bwd": float((x.grad - x2.grad).abs().max()),
            "wdtype": str(layer.weight.dtype), "qs_dtype": str(layer.weight.quant_state.dtype),
        }))
        """
    )
    assert out["fwd"] and out["bwd"] == 0.0, out
    assert out["wdtype"] == "torch.uint8"


def test_quantize_compiles_one_variant_across_shapes():
    # Weight shapes vary per layer: the element / byte / block counts are do_not_specialize, so a
    # model load compiles each kernel once per (dtype, blocksize, quant type).
    out = run(
        f"""
        import json, torch, triton
        from bitsandbytes import _quantize as Q
        from bitsandbytes import functional as F
        assert Q.triton_quantize_available("cuda")
        from unsloth.kernels import nf4_quantize as K

        def variants(kernel):
            while not isinstance(kernel, triton.runtime.jit.JITFunction):
                kernel = kernel.fn
            return sum(len(v[0]) for v in kernel.device_caches.values())

        for shape in {SHAPES!r} + [(4096, 4096), (151936, 64), (1, 1000001)]:
            x = torch.randn(shape, device = "cuda", dtype = torch.bfloat16)
            F.quantize_4bit(x, blocksize = 64, quant_type = "nf4", compress_statistics = True)
        print("OUT " + json.dumps({{"q4": variants(K._quant4_kernel), "q8": variants(K._quant8_kernel)}}))
        """
    )
    assert out["q4"] <= 2 and out["q8"] <= 2, out
