# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""pytorch/pytorch#181735: torch < 2.13 Inductor asserts decomposing a mutating Triton kernel under dynamic shapes."""

import inspect
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fix():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_uif_probe", os.path.join(ROOT, "unsloth", "import_fixes.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_only_where_torch_propagates_triton_eager_vals():
    from torch._inductor import pattern_matcher as pm

    module = _load_fix()
    before = pm.Match.replace_by_example
    original = getattr(before, "__wrapped__", before)
    source = inspect.getsource(inspect.unwrap(original))
    affected = "graph_with_eager_vals" in source and "triton_kernel_wrapper_functional" in source
    try:
        module.fix_inductor_triton_wrapper_eager_vals()
        module.fix_inductor_triton_wrapper_eager_vals()  # idempotent
        patched = getattr(pm.Match.replace_by_example, "_unsloth_triton_eager_vals", False)
        assert patched == (affected or getattr(before, "_unsloth_triton_eager_vals", False))
        if patched:
            assert pm.Match.replace_by_example.__wrapped__ is original
    finally:
        pm.Match.replace_by_example = before


_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import unsloth  # noqa: F401
    import torch
    import triton
    import triton.language as tl
    from unsloth import import_fixes

    @triton.jit
    def _fill(buf_ptr, n, stride_t, stride_i, INNER: tl.constexpr):
        pid = tl.program_id(0)
        if pid >= n:
            return
        for i in range(INNER):
            tl.store(buf_ptr + pid * stride_t + i * stride_i, (pid * INNER + i).to(tl.float32))

    def fn(x):
        n = x.shape[0]
        aligned = ((n + 3) // 4) * 4
        # A size-1 leading dim whose stride is symbolic: the two traces disagree on `0 * stride`.
        buf = torch.empty(8 * aligned, dtype = torch.float32, device = x.device).as_strided((1, n, 8), (8 * aligned, 1, aligned))
        _fill[(n,)](buf, n, stride_t = buf.stride(1), stride_i = buf.stride(2), INNER = 8)
        return buf * 2 + x[:, :1].unsqueeze(0)

    compiled = torch.compile(fn, dynamic = True, fullgraph = True)
    for n in (64, 128, 256):
        x = torch.randn(n, 8, device = "cuda")
        torch.testing.assert_close(compiled(x), fn(x), rtol = 0, atol = 0)
    print("RESULT", import_fixes.STATS_TRITON_EAGER_VALS["fallbacks"])
""")


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU with Triton")
def test_dynamic_size1_strided_triton_mutation_compiles(tmp_path):
    env = dict(os.environ, UNSLOTH_DISABLE_AUTO_UPDATES = "1")
    child = tmp_path / "child.py"  # triton.jit reads the kernel's source file
    child.write_text(_CHILD, encoding = "utf-8")
    out = subprocess.run(
        [sys.executable, str(child), ROOT],
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
        cwd = str(tmp_path),
    )
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT")]
    assert out.returncode == 0 and line, (out.stdout + out.stderr)[-4000:]
    from torch._inductor import pattern_matcher as pm

    source = inspect.getsource(
        getattr(pm.Match.replace_by_example, "__wrapped__", pm.Match.replace_by_example)
    )
    if "graph_with_eager_vals" in source and "triton_kernel_wrapper_functional" in source:
        assert int(line[-1].split()[1]) >= 1  # the fallback ran (torch < 2.13)
