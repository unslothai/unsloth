# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""ROCm: AOTriton SDPA kernels that fail Unsloth's self-check on a GPU stay out of its attention (#18403)."""

import ast
import json
import importlib
import pathlib
import signal
import sys
import types

import pytest
import torch
import torch.nn.functional as F

ad = pytest.importorskip("unsloth.utils.attention_dispatch")
_CHECK = "unsloth.utils.rocm_sdpa_check"
try:
    check = importlib.import_module(_CHECK)
except ImportError:  # a tree without the check: the routing tests must fail there, not skip
    check = None
needs_check = pytest.mark.skipif(check is None, reason = "tree has no rocm_sdpa_check")


class _CudaLooking(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _flags():
    cuda = torch.backends.cuda
    return (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled())


@pytest.fixture
def rocm(monkeypatch):
    """A ROCm build whose device 0 verdict the test sets; every SDPA call records the enabled kernels."""
    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(ad, "_ROCM_SDPA_GUARD", True, raising = False)
    # fp32 packed rows go to xformers where it is installed; these tests are about the SDPA route.
    monkeypatch.setattr(ad, "_packed_fp32_prefers_xformers", lambda *a, **k: False, raising = False)
    verdict = {"flash": True, "efficient": True, "arch": "gfx1200", "source": "check", "detail": ""}
    module = check if check is not None else types.ModuleType(_CHECK)
    monkeypatch.setitem(sys.modules, _CHECK, module)
    monkeypatch.setattr(module, "rocm_sdpa_verdict", lambda index = 0: verdict, raising = False)
    seen = []

    def fake_sdpa(q, k, v, *args, **kwargs):
        seen.append(_flags())
        return torch.zeros_like(q)

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    ad._sdpa_gqa_choices.cache_clear()
    yield verdict, seen
    ad._sdpa_gqa_choices.cache_clear()


def _context(
    seq_info = None,
    attention_mask = None,
    q_len = 12,
):
    return ad.AttentionContext(
        bsz = 1,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = 4,
        head_dim = 8,
        requires_grad = True,
        seq_info = seq_info,
        attention_mask = attention_mask,
        causal_mask = None,
        training = True,
    )


def _qkv(q_len = 12):
    Q = torch.randn(1, 4, q_len, 8).as_subclass(_CudaLooking)
    K = torch.randn(1, 2, q_len, 8).as_subclass(_CudaLooking)
    return Q, K, K.clone()


def _packed(seen):
    Q, K, V = _qkv()
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = 2, n_groups = 2)
    seq_info = (
        torch.tensor([7, 5], dtype = torch.int32),
        torch.tensor([0, 7, 12], dtype = torch.int32),
        7,
    )
    ad.run_attention(config = config, context = _context(seq_info = seq_info), Q = Q, K = K, V = V)
    return seen


@pytest.mark.parametrize(
    "flash, efficient, expect",
    [
        (False, False, (False, False, True)),
        (False, True, (False, True, True)),
        (True, False, (True, False, True)),
    ],
)
def test_packed_segments_skip_refused_kernels(rocm, monkeypatch, flash, efficient, expect):
    monkeypatch.setattr(ad, "_SDPA_PACKED_SEGMENTS", True)
    verdict, seen = rocm
    verdict.update(flash = flash, efficient = efficient)
    before = _flags()
    _packed(seen)
    assert seen and all(flags == expect for flags in seen), seen
    assert _flags() == before


def test_dense_masked_call_skips_refused_kernels(rocm):
    verdict, seen = rocm
    verdict.update(flash = False, efficient = False)
    Q, K, V = _qkv()
    mask = torch.ones(1, 12, dtype = torch.bool)
    config = ad.AttentionConfig(backend = ad.SDPA, n_kv_heads = 2, n_groups = 2)
    ad.run_attention(config = config, context = _context(attention_mask = mask), Q = Q, K = K, V = V)
    assert seen == [(False, False, True)]


def test_decode_call_skips_refused_kernels(rocm):
    verdict, seen = rocm
    verdict.update(efficient = False)
    Q, K, V = _qkv(1)
    ad.decode_scaled_dot_product_attention(Q, K, V, enable_gqa = True)
    assert seen == [(True, False, True)]


def test_gqa_probe_sees_the_narrowed_kernels(rocm, monkeypatch):
    """enable_gqa vs expanded K/V is decided under the kernels the call will run with."""
    verdict, _ = rocm
    verdict.update(flash = False, efficient = False)
    probed = []

    def choice(*args, **kwargs):
        probed.append(_flags())
        return 0  # MATH

    monkeypatch.setattr(torch, "_fused_sdp_choice", choice)
    Q, K, V = _qkv()
    assert ad._sdpa_takes_gqa(Q, K, True, False, V)
    assert probed and all(flags == (False, False, True) for flags in probed)


def test_passing_device_and_cuda_builds_are_untouched(rocm, monkeypatch):
    verdict, seen = rocm
    before = _flags()
    Q, K, V = _qkv()
    ad.scaled_dot_product_attention(Q, K, V, is_causal = True)
    assert seen == [before]
    # CUDA build: the verdict is never asked.
    monkeypatch.setattr(ad, "_ROCM_SDPA_GUARD", False)
    monkeypatch.setattr(
        sys.modules[_CHECK], "rocm_sdpa_verdict", lambda index = 0: pytest.fail("asked on CUDA")
    )
    ad.scaled_dot_product_attention(Q, K, V, is_causal = True)
    assert seen[-1] == before


def test_rerouted_xformers_call_skips_refused_kernels(rocm):
    """Compiled plain-causal attention rerouted off xformers on ROCm calls _rerouted_sdpa."""
    verdict, seen = rocm
    verdict.update(flash = False)
    Q, K, V = _qkv()
    ad._rerouted_sdpa(Q, K, V, is_causal = True, enable_gqa = True)
    assert seen == [(False, True, True)]


def test_dispatch_has_no_unguarded_sdpa_call():
    """Every SDPA call in the dispatch goes through its guarded scaled_dot_product_attention, so a
    new direct torch call (or an alias of it) cannot run kernels the ROCm self-check refused."""
    path = pathlib.Path(ad.__file__)
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def function_of(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return None

    direct = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            a.name == "scaled_dot_product_attention" for a in node.names
        ):
            direct.append((node.lineno, "import"))
        if not (isinstance(node, ast.Attribute) and node.attr == "scaled_dot_product_attention"):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.attr == "__doc__":
            continue  # reads the docstring, runs nothing
        if function_of(node) == "scaled_dot_product_attention":
            continue  # the guarded wrapper itself
        direct.append((node.lineno, function_of(node)))
    assert not direct, f"{path.name}: SDPA reached without the ROCm guard at {direct}"


def test_caller_policy_of_only_refused_kernels_is_kept(rocm):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    verdict, seen = rocm
    verdict.update(flash = False)
    Q, K, V = _qkv()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        ad.scaled_dot_product_attention(Q, K, V, is_causal = True)
    assert seen == [(True, False, False)]


@pytest.fixture
def fresh_verdicts(monkeypatch):
    if check is None:
        pytest.skip("tree has no rocm_sdpa_check")
    monkeypatch.setattr(check, "_VERDICTS", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv(check.ROCM_SDPA_CHECK_ENV, raising = False)


def _no_children(index):
    pytest.fail("the check ran")


@needs_check
def test_verdict_cuda_build(fresh_verdicts, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(check, "_run_children", _no_children)
    v = check.rocm_sdpa_verdict(0)
    assert (v["flash"], v["efficient"], v["source"]) == (True, True, "not_rocm")


@needs_check
@pytest.mark.parametrize("arch, env, source", [("gfx942", None, "cdna"), ("gfx1200", "0", "env")])
def test_verdict_trusted_without_a_check(fresh_verdicts, monkeypatch, arch, env, source):
    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(check, "_arch", lambda index: arch)
    monkeypatch.setattr(check, "_run_children", _no_children)
    if env is not None:
        monkeypatch.setenv(check.ROCM_SDPA_CHECK_ENV, env)
    v = check.rocm_sdpa_verdict(0)
    assert (v["flash"], v["efficient"], v["source"]) == (True, True, source)


@needs_check
def test_verdict_refuses_failed_backends_once(fresh_verdicts, monkeypatch, capsys):
    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(check, "_arch", lambda index: "gfx1200")
    calls = []

    def children(index):
        calls.append(index)
        return {
            "flash": (False, "segments float16 hd128: hipErrorInvalidValue"),
            "efficient": (None, "no start"),
        }

    monkeypatch.setattr(check, "_run_children", children)
    v = check.rocm_sdpa_verdict(0)
    # A check that could not start says nothing about the kernel.
    assert (v["flash"], v["efficient"], v["source"]) == (False, True, "check")
    assert "flash attention failed a self-check" in capsys.readouterr().out
    assert check.rocm_sdpa_verdict(0) is v and calls == [0]


class _FakeChild:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self._out = returncode, (stdout, stderr)

    def communicate(self, timeout = None):
        return self._out

    def kill(self):
        pass


def _report(**fields):
    report = {"backend": "flash", "ok": True, "ran": 8, "detail": [], **fields}
    return "UNSLOTH_ROCM_SDPA_CHECK " + json.dumps(report) + "\n"


# Windows has no signal.SIGKILL.
_SIGKILL = getattr(signal, "SIGKILL", 9)
_OOM_TRACE = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 140.00 MiB. GPU 0 has a total capacity of 15.92 GiB"


@needs_check
@pytest.mark.parametrize(
    "child, usable",
    [
        # Resource exhaustion says nothing about the kernel: kept (inconclusive).
        ((1, "", "Traceback ...\n" + _OOM_TRACE), True),
        pytest.param(
            (-_SIGKILL, "", ""),
            True,
            marks = pytest.mark.skipif(
                sys.platform == "win32", reason = "no out-of-memory killer signal on Windows"
            ),
        ),  # the kernel's out-of-memory killer
        ((0, _report(ran = 3, out_of_memory = ["masked float16 hd128"]), ""), True),
        # Kernel faults and wrong numbers stay refusals.
        ((-signal.SIGABRT, "", "HIP error: an illegal memory access was encountered"), False),
        ((1, "", "RuntimeError: HIP error: invalid argument"), False),
        (
            (0, _report(ok = False, detail = ["segments float16 hd128: worst error / bound 3.1"]), ""),
            False,
        ),
        ((0, _report(ok = False, detail = ["x: bad"], out_of_memory = ["y"]), ""), False),
        ((0, _report(), ""), True),
    ],
)
def test_out_of_memory_check_is_inconclusive(fresh_verdicts, monkeypatch, capsys, child, usable):
    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(check, "_arch", lambda index: "gfx1200")
    monkeypatch.setattr(check.subprocess, "Popen", lambda *a, **k: _FakeChild(*child))
    v = check.rocm_sdpa_verdict(0)
    assert (v["flash"], v["efficient"]) == (usable, usable), v
    out = capsys.readouterr().out
    assert ("failed a self-check" in out) == (not usable), out
    if usable and child != (0, _report(), ""):
        assert "could not finish" in out, out


@needs_check
def test_timeout_stays_a_refusal(fresh_verdicts, monkeypatch):
    class Hung(_FakeChild):
        def communicate(self, timeout = None):
            if timeout is not None:
                raise check.subprocess.TimeoutExpired("child", timeout)
            return "", ""

    monkeypatch.setattr(torch.version, "hip", "7.0.0")
    monkeypatch.setattr(check, "_arch", lambda index: "gfx1200")
    monkeypatch.setattr(check.subprocess, "Popen", lambda *a, **k: Hung(-_SIGKILL, "", ""))
    v = check.rocm_sdpa_verdict(0)
    assert (v["flash"], v["efficient"]) == (False, False), v


@needs_check
def test_child_script_compiles():
    compile(check._CHILD, "rocm_sdpa_check_child", "exec")


@needs_check
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_child_passes_on_a_working_gpu():
    """The child's shapes, reference and tolerance accept kernels that are correct."""
    results = check._run_children(0)
    assert all(ok for ok, _ in results.values()), results


@needs_check
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")
def test_child_out_of_memory_is_inconclusive(monkeypatch):
    """A child that cannot allocate its tensors (the model is resident) keeps both backends."""
    starved = "import torch\ntorch.cuda.set_per_process_memory_fraction(1e-6, int(__import__('sys').argv[1]))\n"
    monkeypatch.setattr(check, "_CHILD", starved + check._CHILD)
    results = check._run_children(0)
    assert all(ok is None for ok, _ in results.values()), results
