# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Compiled NF4 Linear4bit hits the AOT autograd cache warm, matches eager _NF4Linear, and misses
after a kernel file or version change."""

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("bitsandbytes")
if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("needs a CUDA GPU", allow_module_level = True)

_CHILD = textwrap.dedent(
    r"""
    import json, os, sys
    import unsloth  # noqa: F401  (tags the compile cache, installs the kernels)
    import torch, bitsandbytes as bnb
    from torch._dynamo.utils import counters
    from unsloth.kernels import bnb_override as O
    from unsloth.kernels import utils as U

    if not (O._TRACE and U._USE_NF4_KERNELS):
        print("RESULT " + json.dumps({"skip": "NF4 kernels are not traced on this torch / GPU"}))
        sys.exit(0)
    config = torch.compiler.config
    swap = os.environ.get("NF4_TEST_SWAP_TAG")
    if swap:
        old, new = swap.split("=", 1)
        tags = [t for t in config.cache_key_tag.split(",") if t]
        assert any(t.startswith(old) for t in tags), tags
        config.cache_key_tag = ",".join(new if t.startswith(old) else t for t in tags)

    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(512, 384, bias = False, compute_dtype = torch.bfloat16,
                              compress_statistics = True, quant_type = "nf4")
    layer.weight = bnb.nn.Params4bit(torch.randn(384, 512, dtype = torch.bfloat16) / 16, requires_grad = False,
                                     compress_statistics = True, quant_type = "nf4")
    layer = layer.cuda()
    assert layer.weight.quant_state.dtype is torch.bfloat16
    weight = layer.weight
    state = weight.quant_state
    x0 = torch.randn(2, 48, 512, device = "cuda", dtype = torch.bfloat16)
    dy = torch.randn(2, 48, 384, device = "cuda", dtype = torch.bfloat16)

    def step(x):
        return O._linear(x * 2, weight, state, None)

    def run(fn):
        x = x0.clone().requires_grad_(True)
        out = fn(x)
        out.backward(dy)
        return out.detach(), x.grad

    if os.environ.get("NF4_TEST_GRAPH") == "1":
        # The partitioned graphs Inductor would get: what crosses to backward, what backward runs.
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch.partitioners import min_cut_rematerialization_partition

        graphs = {}

        def grab(name):
            def compiler(gm, example_inputs):
                graphs[name] = gm
                return make_boxed_func(gm.forward)
            return compiler

        backend = aot_autograd(fw_compiler = grab("fw"), bw_compiler = grab("bw"),
                               partition_fn = min_cut_rematerialization_partition)
        # gelu's backward needs the linear output: it must be saved, not recomputed.
        run(torch.compile(lambda x: torch.nn.functional.gelu(step(x)), backend = backend, fullgraph = True))
        fw_out = list(graphs["fw"].graph.find_nodes(op = "output"))[0].args[0]
        saved = [tuple(n.meta["val"].shape) for n in fw_out[1:] if hasattr(n, "meta") and hasattr(n.meta.get("val"), "shape")]
        bw = graphs["bw"].graph
        aten = torch.ops.aten
        print("RESULT " + json.dumps({
            "bw_mm": sum(1 for n in bw.nodes if n.target in (aten.mm.default, aten.addmm.default, aten.bmm.default)),
            "bw_dequant": sum(1 for n in bw.nodes if n.op == "call_function" and ("dequantize" in str(n.target) or "triton_kernel_wrapper" in str(n.target))),
            "saved": saved,
        }))
        sys.exit(0)

    eager = run(lambda x: O._apply(O._NF4Linear, x * 2, weight, state, None))
    compiled = run(torch.compile(step, fullgraph = True))
    print("RESULT " + json.dumps({
        "out_equal": bool(torch.equal(eager[0], compiled[0])),
        "grad_equal": bool(torch.equal(eager[1], compiled[1])),
        "aot": dict(counters["aot_autograd"]),
        "tag": config.cache_key_tag,
    }))
    """
)


def _child(cache_dir, **env):
    full = dict(os.environ)
    full.update(
        TORCHINDUCTOR_CACHE_DIR = str(cache_dir / "inductor"),
        TRITON_CACHE_DIR = str(cache_dir / "triton"),
        UNSLOTH_COMPILE_LOCATION = str(cache_dir / "unsloth_compiled_cache"),
        UNSLOTH_MEGA_CACHE_DIR = str(cache_dir / "mega"),
        UNSLOTH_DISABLE_AUTO_UPDATES = "1",
        PYTHONPATH = os.pathsep.join([str(ROOT), full.get("PYTHONPATH", "")]),
    )
    full.update(env)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env = full,
        capture_output = True,
        text = True,
        cwd = str(cache_dir),
        timeout = 900,
    )
    lines = [line for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    assert out.returncode == 0 and lines, out.stdout[-3000:] + out.stderr[-6000:]
    result = json.loads(lines[-1][len("RESULT ") :])
    if "skip" in result:
        pytest.skip(result["skip"])
    return result


def _tag(tags, prefix):
    found = [t for t in tags.split(",") if t.startswith(prefix)]
    assert len(found) == 1, tags
    return found[0]


def test_warm_start_hits_and_invalidates(tmp_path):
    cold = _child(tmp_path)
    assert cold["out_equal"] and cold["grad_equal"]
    assert cold["aot"].get("autograd_cache_bypass", 0) == 0, cold["aot"]
    assert cold["aot"].get("autograd_cache_saved", 0) >= 1, cold["aot"]

    warm = _child(tmp_path)
    assert warm["out_equal"] and warm["grad_equal"]
    assert warm["aot"].get("autograd_cache_hit", 0) >= 1, warm["aot"]
    assert warm["aot"].get("autograd_cache_miss", 0) == 0, warm["aot"]

    nf4 = _tag(cold["tag"], "unsloth/nf4.py:")
    version = _tag(cold["tag"], "unsloth/version:")
    for old, new in ((nf4, "unsloth/nf4.py:0000000000000000"), (version, "unsloth/version:0.0.0")):
        changed = _child(tmp_path, NF4_TEST_SWAP_TAG = f"{old}={new}")
        assert changed["out_equal"] and changed["grad_equal"]
        assert changed["aot"].get("autograd_cache_hit", 0) == 0, (new, changed["aot"])
        assert changed["aot"].get("autograd_cache_miss", 0) >= 1, (new, changed["aot"])


def test_kill_switch_keeps_the_autograd_function(tmp_path):
    old = _child(tmp_path, UNSLOTH_BNB_NF4_CACHEABLE = "0")
    assert old["out_equal"] and old["grad_equal"]
    assert old["aot"].get("autograd_cache_bypass", 0) >= 1, old["aot"]


def test_backward_recomputes_the_weight_not_the_matmul(tmp_path):
    # Dequantized weight recomputed, matmul output saved.
    for env in ({}, {"UNSLOTH_BNB_NF4_CACHEABLE": "0"}):
        graph = _child(tmp_path, NF4_TEST_GRAPH = "1", **env)
        assert graph["bw_mm"] == 1, (env, graph)
        assert graph["bw_dequant"] >= 1, (env, graph)
        assert [384, 512] not in [list(shape) for shape in graph["saved"]], (env, graph)
        assert any(
            shape and shape[-1] == 384 and math.prod(shape) == 2 * 48 * 384
            for shape in graph["saved"]
        ), (env, graph)
