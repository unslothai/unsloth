# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows ROCm: SDPA with a head wider than 256 (Gemma 4's global layers) must run on math there,
since that torch's flash / efficient kernels take it and compute it wrong. Heads up to 256, other
platforms and compiled graphs keep their stock dispatch; Studio's own guard and this one never nest."""

import functools
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
attention = pytest.importorskip("torch.nn.attention")

import unsloth.import_fixes as fixes  # noqa: E402

F = torch.nn.functional
FLAG = "_unsloth_windows_rocm_wide_head_guard"  # Studio's #17960 module checks this same name


@pytest.fixture
def stock_sdpa(monkeypatch):
    """Restore whatever F.scaled_dot_product_attention was, whatever a test installs."""
    stock = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention", stock)
    monkeypatch.delenv("UNSLOTH_ALLOW_ROCM_WIDE_HEAD_FUSED_SDPA", raising = False)
    return stock


@pytest.fixture
def windows_rocm(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)


@pytest.fixture
def cpu_counts_as_gpu(monkeypatch):
    """Let real CPU tensors take the GPU branch, so the numerics and compile checks run anywhere."""
    monkeypatch.setattr(fixes, "_ROCM_WINDOWS_WIDE_HEAD_DEVICE_TYPES", ("cuda", "cpu"))


@pytest.fixture
def recorder(monkeypatch):
    entered, calls = [], []

    def fake_set(backends):
        entered.append(list(backends))

    monkeypatch.setattr(attention, "_sdpa_kernel", fake_set)
    monkeypatch.setattr(attention, "_cur_sdpa_kernel_backends", lambda: ["saved"])

    def sdpa(q, k, v, *args, **kwargs):
        calls.append((args, kwargs))
        return "out"

    return entered, calls, sdpa


def _meta(head_dim, v_head_dim = None):
    # Meta tensors stand in for GPU ones: the tests add "meta" to the guarded device types.
    q = torch.empty(1, 8, 128, head_dim, device = "meta")
    v = torch.empty(1, 1, 128, v_head_dim or head_dim, device = "meta")
    return q, v


@pytest.mark.parametrize("head_dim, v_head_dim", [(512, None), (384, None), (256, 512)])
def test_wide_head_runs_on_math(recorder, monkeypatch, head_dim, v_head_dim):
    entered, calls, sdpa = recorder
    monkeypatch.setattr(fixes, "_ROCM_WINDOWS_WIDE_HEAD_DEVICE_TYPES", ("meta",))
    q, v = _meta(head_dim, v_head_dim)
    guarded = fixes.rocm_windows_wide_head_sdpa(sdpa)
    assert guarded(q, v, v, None, 0.0, True, scale = 0.1, enable_gqa = True) == "out"
    assert entered == [[attention.SDPBackend.MATH], ["saved"]]
    assert calls == [((None, 0.0, True), {"scale": 0.1, "enable_gqa": True})]


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_heads_up_to_256_keep_fused_kernels(recorder, monkeypatch, head_dim):
    entered, calls, sdpa = recorder
    monkeypatch.setattr(fixes, "_ROCM_WINDOWS_WIDE_HEAD_DEVICE_TYPES", ("meta",))
    q, v = _meta(head_dim)
    assert fixes.rocm_windows_wide_head_sdpa(sdpa)(q, v, v) == "out"
    assert entered == [] and len(calls) == 1


def test_non_gpu_tensors_keep_stock_dispatch(recorder):
    entered, calls, sdpa = recorder
    q = torch.empty(1, 2, 4, 512)  # CPU: not a guarded device type
    assert fixes.rocm_windows_wide_head_sdpa(sdpa)(q, q, q) == "out"
    assert entered == [] and len(calls) == 1


def test_install_on_windows_rocm_only(stock_sdpa, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)
    monkeypatch.setattr(sys, "platform", "linux")  # Linux ROCm refuses the kernels: math already
    assert fixes.fix_rocm_windows_wide_head_sdpa() is False
    assert F.scaled_dot_product_attention is stock_sdpa

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")  # Windows NVIDIA
    assert fixes.fix_rocm_windows_wide_head_sdpa() is False
    assert F.scaled_dot_product_attention is stock_sdpa


@pytest.mark.parametrize(
    "hip, version", [("7.14.0", "2.11.0+rocm7.14.0"), (None, "2.11.0+rocm7.14.0")]
)
def test_install_wraps_once(stock_sdpa, monkeypatch, hip, version):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "__version__", version)  # some AMD wheels tag only the version
    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    wrapped = F.scaled_dot_product_attention
    assert wrapped is not stock_sdpa and getattr(wrapped, FLAG) is True
    assert wrapped.__wrapped__ is stock_sdpa
    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    assert F.scaled_dot_product_attention is wrapped  # idempotent: no double wrap


def test_opt_out(stock_sdpa, windows_rocm, monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_ROCM_WIDE_HEAD_FUSED_SDPA", "1")
    assert fixes.fix_rocm_windows_wide_head_sdpa() is False
    assert F.scaled_dot_product_attention is stock_sdpa


def _studio_style_guard(sdpa):
    """Shape of Studio's core/_rocm_windows_sdpa.guard_sdpa: functools.wraps + the shared flag."""

    @functools.wraps(sdpa)
    def guarded(q, k, v, *a, **kw):
        return sdpa(q, k, v, *a, **kw)

    setattr(guarded, FLAG, True)
    return guarded


def test_studio_guard_first_makes_core_a_noop(stock_sdpa, windows_rocm):
    studio = _studio_style_guard(stock_sdpa)
    F.scaled_dot_product_attention = studio
    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    assert F.scaled_dot_product_attention is studio

    # Still a no-op when another wrapper (functools.wraps) sits on top of Studio's guard.
    on_top = functools.wraps(studio)(lambda *a, **kw: studio(*a, **kw))
    F.scaled_dot_product_attention = on_top
    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    assert F.scaled_dot_product_attention is on_top


def test_core_guard_first_makes_studio_a_noop(stock_sdpa, windows_rocm):
    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    core = F.scaled_dot_product_attention
    # Studio's guard_sdpa returns its input unchanged when the input carries the flag.
    assert getattr(core, FLAG, False) is True


def test_wide_head_matches_math_and_small_head_is_untouched(stock_sdpa, cpu_counts_as_gpu):
    """Real CPU tensors: the guard's output is bitwise math's; under 256 it is bitwise stock."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    guarded = fixes.rocm_windows_wide_head_sdpa(stock_sdpa)
    g = torch.Generator().manual_seed(0)
    for head_dim in (64, 512):
        q = torch.randn(1, 4, 64, head_dim, generator = g, requires_grad = True)
        k = torch.randn(1, 2, 64, head_dim, generator = g, requires_grad = True)
        v = torch.randn(1, 2, 64, head_dim, generator = g, requires_grad = True)
        out = guarded(q, k, v, is_causal = True, enable_gqa = True)
        grads = torch.autograd.grad(out.sum(), (q, k, v))
        if head_dim > 256:
            with sdpa_kernel([SDPBackend.MATH]):
                ref = stock_sdpa(q, k, v, is_causal = True, enable_gqa = True)
        else:
            ref = stock_sdpa(q, k, v, is_causal = True, enable_gqa = True)
        ref_grads = torch.autograd.grad(ref.sum(), (q, k, v))
        assert torch.equal(out, ref)
        assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads))


def _attend(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal = True) * 2


def _frames(seq_lens, head_dim, dynamic):
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    fn = torch.compile(_attend, backend = counter, fullgraph = True, dynamic = dynamic)
    for n in seq_lens:
        q = torch.randn(1, 2, n, head_dim)
        fn(q, q, q)
    return counter.frame_count


def _aten_targets(head_dim):
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    torch._dynamo.reset()
    seen = []

    def fw(gm, _):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return make_boxed_func(gm.forward)

    fn = torch.compile(_attend, backend = aot_autograd(fw_compiler = fw), fullgraph = True)
    q = torch.randn(1, 2, 16, head_dim, requires_grad = True)
    fn(q, q, q).sum().backward()
    return seen


@pytest.mark.parametrize("dynamic", [None, True])
def test_compiled_graph_has_no_break_no_extra_recompile_and_runs_math(
    stock_sdpa, windows_rocm, cpu_counts_as_gpu, dynamic
):
    seq_lens = (8, 8, 9, 16, 33)
    baseline = {d: _frames(seq_lens, d, dynamic) for d in (64, 512)}
    assert all(
        n > 0 for n in baseline.values()
    ), baseline  # something compiled, so equality means something
    stock_512 = _aten_targets(512)
    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    # fullgraph=True raises on any graph break; frame counts equal stock = no recompile added.
    assert {d: _frames(seq_lens, d, dynamic) for d in (64, 512)} == baseline
    guarded_512 = _aten_targets(512)
    guarded_64 = _aten_targets(64)
    fused = "_scaled_dot_product_flash_attention_for_cpu"
    if not any(fused in t for t in stock_512):
        pytest.skip(reason = "this torch's CPU SDPA never picks its fused kernel at head 512")
    assert not any(fused in t for t in guarded_512), guarded_512  # traced under the math pin
    assert any(fused in t for t in guarded_64), guarded_64  # small heads keep the fused kernel
    torch._dynamo.reset()


def test_import_time_install_is_wired():
    src = (Path(fixes.__file__).resolve().parent / "_gpu_init.py").read_text(encoding = "utf-8")
    call = "\nfix_rocm_windows_wide_head_sdpa()\n"
    assert call in src
    # Before anything later in _gpu_init can bind SDPA by value (model modules load after it).
    assert src.index(call) < src.index("\nfix_triton_compiled_kernel_missing_attrs()\n")


def test_torch_function_modes_still_match_f_sdpa(stock_sdpa, windows_rocm, cpu_counts_as_gpu):
    """Matchers written `func is F.scaled_dot_product_attention` (diffusers layer_skip, Studio's VAE
    chunking, tp_native) see the wrapper itself, and their fallthrough still runs on math."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.overrides import TorchFunctionMode

    class Seen(TorchFunctionMode):
        hits = 0

        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            if func is F.scaled_dot_product_attention:
                Seen.hits += 1
            return func(*args, **(kwargs or {}))

    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    q = torch.randn(1, 2, 16, 512)
    with Seen():
        out = F.scaled_dot_product_attention(q, q, q, is_causal = True)
        F.scaled_dot_product_attention(query = q, key = q, value = q)
    assert Seen.hits == 2
    with sdpa_kernel([SDPBackend.MATH]):
        ref = stock_sdpa(q, q, q, is_causal = True)
    assert torch.equal(out, ref)


def test_compiled_call_restores_the_callers_backends(stock_sdpa, windows_rocm, cpu_counts_as_gpu):
    """The math switch is one graph node: its exit restores the backends live at run time, not the
    ones Dynamo saw while tracing (an inlined sdpa_kernel bakes those in as constants)."""
    from torch._dynamo.testing import EagerAndRecordGraphs

    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    torch._dynamo.reset()
    recorder = EagerAndRecordGraphs()
    fn = torch.compile(_attend, backend = recorder, fullgraph = True)
    q = torch.randn(1, 2, 16, 512)
    fn(q, q, q)  # traced with every backend enabled
    cuda = torch.backends.cuda
    saved = cuda.flash_sdp_enabled()
    try:
        cuda.enable_flash_sdp(False)
        fn(q, q, q)
        assert cuda.flash_sdp_enabled() is False
    finally:
        cuda.enable_flash_sdp(saved)
        torch._dynamo.reset()
    targets = [
        str(n.target) for g in recorder.graphs for n in g.graph.nodes if n.op == "call_function"
    ]
    assert not any("sdpa_kernel" in t or "_backend_from_string" in t for t in targets), targets


def test_overlapping_wide_head_calls_restore_the_callers_backends_once(
    windows_rocm, cpu_counts_as_gpu
):
    """Two wide-head calls in flight (two threads), the first returning first: the second must not see the fused
    kernels back, and the caller's backends come back only after the last one returns."""
    from torch.nn.attention import _cur_sdpa_kernel_backends

    seen = {}
    inside = {"a": threading.Event(), "b": threading.Event()}
    release = {"a": threading.Event(), "b": threading.Event()}

    def sdpa(
        q,
        k,
        v,
        *args,
        name = None,
        **kwargs,
    ):
        inside[name].set()
        release[name].wait(5)
        seen[name] = set(_cur_sdpa_kernel_backends())
        return q

    guarded = fixes.rocm_windows_wide_head_sdpa(sdpa)
    before = set(_cur_sdpa_kernel_backends())
    q = torch.zeros(1, 1, 4, 512)
    threads = {
        n: threading.Thread(target = guarded, args = (q, q, q), kwargs = {"name": n}) for n in "ab"
    }
    threads["a"].start()
    assert inside["a"].wait(5)
    threads["b"].start()
    assert inside["b"].wait(5)
    release["a"].set()
    threads["a"].join(5)
    release["b"].set()
    threads["b"].join(5)
    assert seen == {"a": {attention.SDPBackend.MATH}, "b": {attention.SDPBackend.MATH}}
    assert set(_cur_sdpa_kernel_backends()) == before


def _jagged(head_dim):
    g = torch.Generator().manual_seed(0)
    seqs = [torch.randn(2, n, head_dim, generator = g) for n in (5, 7)]  # (heads, seq, head_dim)
    nt = torch.nested.nested_tensor([s.transpose(0, 1) for s in seqs], layout = torch.jagged)
    return seqs, nt.transpose(1, 2)  # (batch, heads, jagged seq, head_dim)


@pytest.mark.parametrize("head_dim", [64, 512])
def test_nested_jagged_tensors_keep_working(stock_sdpa, windows_rocm, cpu_counts_as_gpu, head_dim):
    """Nested jagged tensors route SDPA only when their __torch_function__ sees torch's builtin: re-dispatching
    with func = the wrapper falls through to the dense kernel, which refuses them."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    seqs, nt = _jagged(head_dim)
    out = F.scaled_dot_product_attention(nt, nt, nt)
    for got, s in zip(out.transpose(1, 2).unbind(), seqs):
        with sdpa_kernel([SDPBackend.MATH]):
            ref = stock_sdpa(s[None], s[None], s[None])[0]
        torch.testing.assert_close(got.transpose(0, 1), ref, atol = 1e-5, rtol = 1e-5)


def test_nested_jagged_tensors_under_a_mode(stock_sdpa, windows_rocm, cpu_counts_as_gpu):
    from torch.overrides import TorchFunctionMode

    class Seen(TorchFunctionMode):
        hits = 0

        def __torch_function__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            if func is F.scaled_dot_product_attention:
                Seen.hits += 1
            return func(*args, **(kwargs or {}))

    assert fixes.fix_rocm_windows_wide_head_sdpa() is True
    _, nt = _jagged(512)
    with Seen():
        out = F.scaled_dot_product_attention(nt, nt, nt)
    assert Seen.hits == 1 and out.is_nested


def test_a_wide_call_entering_after_another_thread_moved_the_flags_still_runs_math(
    windows_rocm, cpu_counts_as_gpu
):
    """Flags are process-wide: if another thread's sdpa_kernel changes them while a wide call holds the pin, a wide call
    starting after that is pinned to math again rather than running under the other thread's choice."""
    from torch.nn.attention import SDPBackend, _cur_sdpa_kernel_backends, _sdpa_kernel

    seen = {}
    inside, release = threading.Event(), threading.Event()

    def sdpa(
        q,
        k,
        v,
        *args,
        name = None,
        **kwargs,
    ):
        if name == "held":
            inside.set()
            release.wait(5)
        seen[name] = set(_cur_sdpa_kernel_backends())
        return q

    guarded = fixes.rocm_windows_wide_head_sdpa(sdpa)
    before = _cur_sdpa_kernel_backends()
    q = torch.zeros(1, 1, 4, 512)
    held = threading.Thread(target = guarded, args = (q, q, q), kwargs = {"name": "held"})
    held.start()
    assert inside.wait(5)
    _sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])  # another thread's choice
    guarded(q, q, q, name = "late")
    release.set()
    held.join(5)
    assert seen["late"] == {SDPBackend.MATH}
    assert set(_cur_sdpa_kernel_backends()) == set(before)
