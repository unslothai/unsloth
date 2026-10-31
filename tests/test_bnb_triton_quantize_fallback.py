# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""#10844: bitsandbytes' Triton quantize ops with no host C compiler. One subprocess per case:
kernel registrations are process-wide."""

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")
torch = pytest.importorskip("torch")

_FIXES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "import_fixes.py"
)

# Stand-ins raise from a triton/runtime/build.py frame, as Triton's launcher build does.
_CPU_RUNNER = textwrap.dedent(
    """
    import importlib.util, json, sys
    import torch, bitsandbytes as bnb, bitsandbytes.functional as F
    if torch.accelerator.current_accelerator() is None:
        # triton/ops.py reads the accelerator's type at import; the stand-ins below never use it.
        torch.accelerator.current_accelerator = lambda *a, **k: torch.device("cpu")
    from bitsandbytes.backends.triton import ops as triton_ops
    behaviour, patch = sys.argv[1], sys.argv[2] == "1"

    build = {}
    exec(compile(
        "def _build():\\n    raise RuntimeError('Failed to find C compiler. Please specify via CC environment variable')\\n",
        "/site-packages/triton/runtime/build.py", "exec"), build)
    calls = []
    default_keys = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd)

    def stand_in(name):
        op = getattr(torch.ops.bitsandbytes, name).default
        def kernel(*args):
            calls.append(name)
            if behaviour == "no_compiler":
                build["_build"]()
            if behaviour == "bad_input":
                raise ValueError("not a build failure")
            return op.redispatch(default_keys, *args)
        return kernel

    lib = torch.library.Library("bitsandbytes", "FRAGMENT")
    for name in ("quantize_4bit", "quantize_blockwise"):
        setattr(triton_ops, name, stand_in(name))
        lib.impl(name, getattr(triton_ops, name), "CPU", allow_override = True)
    if patch:
        spec = importlib.util.spec_from_file_location("fixes_under_test", sys.argv[3])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        fixes.patch_bitsandbytes_triton_quantize_fallback("cpu")
        state = triton_ops._unsloth_quantize_fallback
        fixes.patch_bitsandbytes_triton_quantize_fallback("cpu")
        assert triton_ops._unsloth_quantize_fallback is state

    torch.manual_seed(3407)
    W = torch.randn(256, 192, dtype = torch.bfloat16)
    out = {"calls": calls}
    try:
        q, qs = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
        F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
        dq = F.dequantize_4bit(q, qs)
        out.update(ok = True, rel_err = float((dq.float() - W.float()).norm() / W.float().norm()))
    except Exception as e:
        out.update(ok = False, error = f"{type(e).__name__}: {e}")
    state = getattr(triton_ops, "_unsloth_quantize_fallback", None)
    out["fallback_error"] = state and state["error"]
    print("RESULT " + json.dumps(out))
    """
)


def _run(argv, env = None):
    proc = subprocess.run(
        [sys.executable, "-c", *argv], capture_output = True, text = True, timeout = 600, env = env
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :]), proc.stdout


def _cpu(behaviour, patch):
    return _run([_CPU_RUNNER, behaviour, "1" if patch else "0", _FIXES])


def test_no_compiler_fails_without_the_fallback():
    result, _ = _cpu("no_compiler", patch = False)
    assert not result["ok"]
    assert "Failed to find C compiler" in result["error"]


def test_no_compiler_falls_back_to_the_pytorch_kernels_once():
    result, stdout = _cpu("no_compiler", patch = True)
    assert result["ok"], result
    assert result["rel_err"] < 0.15
    assert "Failed to find C compiler" in result["fallback_error"]
    # Triton is tried once, then never again: the second quantize goes straight to PyTorch.
    assert result["calls"] == ["quantize_4bit"]
    assert stdout.count("Triton could not build its kernel launcher") == 1


def test_working_compiler_keeps_triton():
    result, stdout = _cpu("works", patch = True)
    assert result["ok"], result
    assert result["fallback_error"] is None
    assert result["calls"] == ["quantize_4bit", "quantize_blockwise"] * 2
    assert "Triton could not build" not in stdout


def test_other_errors_are_not_swallowed():
    result, _ = _cpu("bad_input", patch = True)
    assert not result["ok"]
    assert result["error"] == "ValueError: not a build failure"
    assert result["fallback_error"] is None


_GPU_RUNNER = textwrap.dedent(
    """
    import importlib.util, json, sys
    import torch, bitsandbytes.functional as F
    from bitsandbytes.backends.triton import ops as triton_ops
    torch.manual_seed(3407)
    W = torch.randn(1024, 768, device = "cuda", dtype = torch.bfloat16)
    ref_q, ref_s = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
    lib = torch.library.Library("bitsandbytes", "FRAGMENT")
    for name in ("quantize_blockwise", "quantize_4bit"):
        lib.impl(name, getattr(triton_ops, name), "CUDA", allow_override = True)
    if sys.argv[1] == "1":
        spec = importlib.util.spec_from_file_location("fixes_under_test", sys.argv[2])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        fixes.patch_bitsandbytes_triton_quantize_fallback("cuda")
    out = {}
    try:
        q, s = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
        out.update(ok = True, packed_equal = bool(torch.equal(q, ref_q)),
                   absmax_equal = bool(torch.equal(s.absmax, ref_s.absmax)))
    except Exception as e:
        out.update(ok = False, error = f"{type(e).__name__}: {e}")
    print("RESULT " + json.dumps(out))
    """
)


@pytest.mark.skipif(not has_real_cuda(), reason = "real Triton launches need a GPU")
@pytest.mark.parametrize("patch", [False, True])
def test_real_triton_without_a_compiler(tmp_path, patch):
    """Real Triton, with gcc / clang hidden and a cold cache: its launcher build raises."""
    env = dict(os.environ, TRITON_CACHE_DIR = str(tmp_path), PATH = os.path.dirname(sys.executable))
    env.pop("CC", None)
    env.pop("CXX", None)
    if shutil.which("gcc", path = env["PATH"]) or shutil.which("clang", path = env["PATH"]):
        pytest.skip("a C compiler sits beside the interpreter")
    result, _ = _run([_GPU_RUNNER, "1" if patch else "0", _FIXES], env = env)
    if not patch:
        assert not result["ok"] and "Failed to find C compiler" in result["error"], result
    else:
        # bitsandbytes' PyTorch kernels match its native CUDA ones bit for bit.
        assert result == {"ok": True, "packed_equal": True, "absmax_equal": True}, result
