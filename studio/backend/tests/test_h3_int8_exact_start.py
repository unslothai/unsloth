# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 video: wherever Auto engages INT8 attention (sm89 again), the first denoise steps of each render run the
16-bit kernel (exact start); explicit On keeps INT8 on every step, Off is unchanged. The gate is driven directly; the runtime routing runs on a tiny real H3 transformer on CPU, where the
INT8 entry point is a spy (the CUDA kernel cannot run there)."""

from __future__ import annotations

import ast
import contextlib
import copy
from pathlib import Path

import pytest
import torch

from core.inference import diffusion_int8_attention as I8
from core.inference import video_minimax_h3_attn as A

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _gate(**kw):
    """The evidenced Auto config on an L4 unless overridden."""
    args = dict(
        mode = "auto",
        attention_backend = None,
        auto_attention = True,
        speed_on = True,
        hosted_int8 = True,
        comfy_checkpoint = False,
        adapters = False,
        capability = (8, 9),
    )
    args.update(kw)
    return A.h3_int8_load_gate(**args)


@pytest.fixture(autouse = True)
def _no_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_H3_INT8_EXACT_STEPS", raising = False)
    monkeypatch.delenv(I8.INT8_ATTENTION_ENV, raising = False)


# ── the load gate ──


@pytest.mark.parametrize(
    "capability", [(8, 9), (8, 0), (8, 6), (12, 0), (9, 0), (10, 0), (7, 5), None]
)
def test_auto_engages_with_an_exact_start_on_every_card(capability):
    # None: ROCm / an unknown card; the decision is "Auto engaged INT8", not the arch
    gate = _gate(capability = capability)
    assert (gate.forced, gate.auto, gate.reason) == (False, True, None)
    assert gate.exact_steps == A.H3_INT8_EXACT_START_STEPS > 0


def test_sm89_is_off_the_auto_off_list():
    assert (8, 9) not in A.H3_INT8_AUTO_OFF_ARCHS
    assert A.h3_int8_auto_off_reason((8, 9)) is None


def test_rocm_auto_never_engages_int8_for_h3_so_no_exact_start_installs(monkeypatch):
    # Auto's H3 race is CUDA-kernel only: on ROCm it installs nothing, so the load never reaches the exact start
    monkeypatch.setattr(I8, "_is_rocm", lambda: True)
    model = _tiny_model()
    A.install_strided_attention(model)
    assert (
        I8.install_family_auto(
            model, device = "cpu", dtype = torch.bfloat16, policy = I8.resolve_policy(None)
        )
        == 0
    )
    assert A.install_int8_exact_start(model, _Pipe(), A.h3_int8_exact_steps()) == 0
    assert A.int8_exact_start_count(model) == 0


@pytest.mark.parametrize("pick", ["on", None])
def test_forced_runs_int8_on_every_step(monkeypatch, pick):
    if pick is None:
        monkeypatch.setenv(I8.INT8_ATTENTION_ENV, "1")
    gate = _gate(mode = I8.resolve_policy(pick).mode)
    assert (gate.forced, gate.auto, gate.reason, gate.exact_steps) == (True, False, None, 0)


@pytest.mark.parametrize("env, pick", [("0", None), (None, "off"), ("1", "off")])
def test_explicit_off_still_wins(monkeypatch, env, pick):
    if env is not None:
        monkeypatch.setenv(I8.INT8_ATTENTION_ENV, env)
    monkeypatch.setenv(A.H3_INT8_EXACT_STEPS_ENV, "4")
    gate = _gate(mode = I8.resolve_policy(pick).mode)
    assert gate == A.H3Int8Gate(forced = False, auto = False, reason = None, exact_steps = 0)


@pytest.mark.parametrize(
    "kw, reason",
    [
        (
            dict(speed_on = False),
            "speed_mode=off keeps the output bit-identical, so the kernel is not installed",
        ),
        (
            dict(attention_backend = "flash", auto_attention = False),
            "attention_backend='flash' was picked explicitly; Auto keeps it",
        ),
        (
            dict(hosted_int8 = False),
            "Auto engages it on MiniMax-H3 video only over Studio's hosted INT8 denoiser without adapters",
        ),
        (
            dict(adapters = True),
            "Auto engages it on MiniMax-H3 video only over Studio's hosted INT8 denoiser without adapters",
        ),
    ],
)
def test_auto_off_reasons_on_sm89(kw, reason):
    gate = _gate(**kw)
    assert (gate.auto, gate.reason, gate.exact_steps) == (False, reason, 0)


@pytest.mark.parametrize(
    "raw, capability, want",
    [("0", (8, 9), 0), ("4", (8, 9), 4), ("3", (8, 0), 3), ("0", (12, 0), 0), ("2", None, 2)],
)
def test_exact_steps_env_override(monkeypatch, raw, capability, want):
    monkeypatch.setenv(A.H3_INT8_EXACT_STEPS_ENV, raw)
    gate = _gate(capability = capability)
    assert gate.auto and gate.exact_steps == want


@pytest.mark.parametrize("raw", ["x", "-2", "2.5"])
def test_a_bad_env_value_keeps_the_default_and_warns(monkeypatch, caplog, raw):
    monkeypatch.setenv(A.H3_INT8_EXACT_STEPS_ENV, raw)
    with caplog.at_level("WARNING"):
        assert _gate().exact_steps == A.H3_INT8_EXACT_START_STEPS
    assert A.H3_INT8_EXACT_STEPS_ENV in caplog.text


# ── runtime: the step-gated op on a tiny H3 transformer ──

h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = h3.MiniMaxH3Transformer3DModel(
        num_attention_heads = 2,
        attention_head_dim = 32,
        hidden_size = 64,
        num_layers = 2,
        num_refiner_layers = 1,
        ffn_dim = 128,
        in_channels = 4,
        audio_in_channels = 4,
        patch_size = (1, 2, 2),
        text_dim = 32,
        freq_dim = 32,
        time_embed_hidden_dim = 64,
        time_embed_dim = 32,
        rope_freq_dim = 4,
    )
    return model.eval()


def _inputs(
    seed: int = 1,
    n_text: int = 5,
    n_video: int = 8,
    n_audio: int = 3,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    seq = n_text + n_video + n_audio
    text_idx = torch.arange(0, n_text)
    video_idx = torch.arange(n_text, n_text + n_video)
    audio_idx = torch.arange(n_text + n_video, seq)
    tags = torch.zeros(seq, dtype = torch.long)
    tags[text_idx] = 1
    tags[audio_idx] = 2
    return dict(
        hidden_states = torch.randn(1, n_video, 16, generator = g),
        audio_hidden_states = torch.randn(1, n_audio, 4, generator = g),
        encoder_hidden_states = torch.randn(1, n_text, 32, generator = g),
        timestep = torch.tensor([0.7, 0.2]),
        timestep_indices = torch.randint(0, 2, (seq,), generator = g),
        token_tags = tags,
        position_ids = torch.randint(0, 6, (seq, 3), generator = g),
        video_indices = video_idx,
        audio_indices = audio_idx,
        text_indices = text_idx,
        return_dict = False,
    )


def _run(model):
    with torch.no_grad():
        return model(**_inputs())


def _equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


class _Scheduler:
    _step_index = None


class _Pipe:
    def __init__(self):
        self.scheduler = _Scheduler()


def _int8_model(monkeypatch):
    """Strided processors marked INT8 (as install_family_auto leaves them), the INT8 entry point a spy that moves the
    output, and an off twin (same weights, 16-bit attention)."""
    model = _tiny_model()
    model.set_attention_backend("_native_math")
    assert A.install_strided_attention(model) == 3
    off = copy.deepcopy(model)
    for m in A._h3_attention_modules(model):
        m.processor._unsloth_int8 = True
        m.processor._unsloth_int8_min_tokens = 0
    calls = []
    native = I8.int8_attention_impl

    def spy(
        q,
        k,
        v,
        mode = 0,
    ):
        calls.append(tuple(q.shape))
        return native(q, k, v) * 1.01

    monkeypatch.setattr(I8, "int8_attention_impl", spy)
    return model, off, calls


@pytest.fixture(autouse = True)
def _reset_state():
    yield
    state = getattr(A, "_EXACT_STATE", None)
    if state is not None:
        state["active"] = False


def test_first_steps_run_the_16bit_kernel_then_int8(monkeypatch):
    model, off, calls = _int8_model(monkeypatch)
    pipe = _Pipe()
    assert A.install_int8_exact_start(model, pipe, 2) == 3
    assert A.int8_exact_start_count(model) == 3
    ref = _run(off)
    # the step index is None before the first scheduler step, then counts the steps taken
    for index, int8 in ((None, False), (1, False), (2, True), (19, True)):
        pipe.scheduler._step_index = index
        calls.clear()
        out = _run(model)
        assert bool(calls) is int8
        assert _equal(out, ref) is (not int8)
    # a new render resets the scheduler's index: the exact start comes back
    pipe.scheduler._step_index = None
    calls.clear()
    assert _equal(_run(model), ref) and not calls


def test_a_render_that_begins_later_still_runs_n_exact_steps(monkeypatch):
    # i2v / v2v strength: the scheduler starts at a later grid point (begin index 10)
    model, off, calls = _int8_model(monkeypatch)
    pipe = _Pipe()
    A.install_int8_exact_start(model, pipe, 2)
    seen = []
    for index in (None, 11, 12, 13):
        pipe.scheduler._step_index = index
        calls.clear()
        _run(model)
        seen.append(bool(calls))
    assert seen == [False, False, True, True]
    # the next render starts over
    pipe.scheduler._step_index = None
    calls.clear()
    _run(model)
    assert not calls


def test_the_op_refuses_cuda_graph_capture(monkeypatch):
    seen = []

    def capturing(device):
        seen.append(device)
        return True

    monkeypatch.setattr(A, "_capturing_on", capturing)
    q = torch.randn(1, 8, 2, 32)
    for flag in (1, 0):
        with pytest.raises(RuntimeError, match = "not CUDA-graph safe"):
            A.int8_attention_exact_start_impl(q, q, q, torch.tensor([flag], dtype = torch.int8), 0)
    # judged on the operand's device
    assert seen == [q.device, q.device]


def test_the_capture_check_runs_on_the_operands_card(monkeypatch):
    import torch.cuda

    current = {"index": 0}

    @contextlib.contextmanager
    def device(d):
        saved = current["index"]
        current["index"] = torch.device(d).index
        try:
            yield
        finally:
            current["index"] = saved

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", device)
    # a capture runs on cuda:1 while cuda:0 is current
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: current["index"] == 1)
    assert A._capturing_on(torch.device("cuda", 1)) is True
    assert A._capturing_on(torch.device("cuda", 0)) is False
    assert current["index"] == 0
    assert A._capturing_on(torch.device("cpu")) is False


def test_the_fake_dtype_matches_the_real_output_under_autocast():
    from torch._subclasses.fake_tensor import FakeTensorMode

    A.int8_attention_exact_start_op()
    q = torch.randn(1, 16, 2, 32)
    plain = A.int8_attention_exact_start_impl(q, q, q, torch.ones(1, dtype = torch.int8), 4)
    for flag in (1, 0):
        f = torch.tensor([flag], dtype = torch.int8)
        with torch.autocast("cpu", dtype = torch.bfloat16):
            real = torch.ops.unsloth_h3.int8_attention_exact_start(q, q, q, f, 4)
            with FakeTensorMode() as mode:
                fq = mode.from_tensor(q)
                fake = torch.ops.unsloth_h3.int8_attention_exact_start(
                    fq, fq, fq, mode.from_tensor(f), 4
                )
        assert real.dtype == fake.dtype == q.dtype
        assert real.shape == fake.shape and real.is_contiguous()
        if flag:
            # the exact branch ignores autocast: the same bits as without it
            assert torch.equal(real, plain)


def test_graphs_are_refused_for_an_exact_start_denoiser(monkeypatch):
    import importlib.util
    import types

    from core.inference import diffusion_cuda_graph as cg

    # the tiny H3 model is refused for its own reason first: a plain DiT shows this check on its own
    spec = importlib.util.spec_from_file_location(
        "_tbg", Path(__file__).with_name("test_diffusion_block_graph.py")
    )
    tbg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbg)
    model, off, calls = _int8_model(monkeypatch)
    assert A.install_int8_exact_start(model, _Pipe(), 2) == 3
    assert model.__dict__[A.GRAPH_UNSAFE_ATTR] == A.EXACT_GRAPH_REASON
    net = tbg.Net(blocks = 2)
    net.__dict__[A.GRAPH_UNSAFE_ATTR] = A.EXACT_GRAPH_REASON
    # per-block graphs (a streamed / pinned denoiser)
    pipe = tbg._pipe_with(net)
    handles, applied = tbg._arm(pipe, monkeypatch, hooked = True, pinned = True)
    assert handles == () and not applied["cuda_graph"]
    assert cg.status_reason(pipe, False) == A.EXACT_GRAPH_REASON
    # (the same Net without the marker records: test_diffusion_block_graph.py)
    # the whole-forward graph (a resident denoiser)
    monkeypatch.delenv(cg.CUDA_GRAPH_DISABLE_ENV, raising = False)
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", dtype = "bfloat16")
    kw = dict(
        family = types.SimpleNamespace(),
        offload_active = False,
        cache_active = False,
        speed_mode = "default",
    )
    assert cg.graph_eligible(target, pipe = types.SimpleNamespace(transformer = net), **kw) == (
        False,
        A.EXACT_GRAPH_REASON,
    )
    assert cg.graph_eligible(
        target, pipe = types.SimpleNamespace(transformer = tbg.Net(blocks = 2)), **kw
    )[1] != (A.EXACT_GRAPH_REASON)


def test_without_the_exact_start_every_step_is_int8(monkeypatch):
    model, off, calls = _int8_model(monkeypatch)
    pipe = _Pipe()
    assert A.install_int8_exact_start(model, pipe, 0) == 0
    assert A.int8_exact_start_count(model) == 0
    for index in (None, 1, 5):
        pipe.scheduler._step_index = index
        calls.clear()
        _run(model)
        assert calls


def test_a_scheduler_without_a_step_index_runs_int8(monkeypatch):
    model, off, calls = _int8_model(monkeypatch)
    pipe = _Pipe()
    pipe.scheduler = object()
    assert A.install_int8_exact_start(model, pipe, 4) == 3
    _run(model)
    assert calls


def test_a_processor_the_race_left_on_16bit_is_not_touched(monkeypatch):
    model, off, calls = _int8_model(monkeypatch)
    first = A._h3_attention_modules(model)[0].processor
    first._unsloth_int8 = False
    assert A.install_int8_exact_start(model, _Pipe(), 2) == 2
    assert first._unsloth_int8_exact is False


def test_the_exact_kernel_is_the_processors_own():
    procs = []
    for name in ("_native_flash", "_native_cudnn", "_native_math", None):
        model = _tiny_model()
        if name:
            model.set_attention_backend(name)
        A.install_strided_attention(model)
        for m in A._h3_attention_modules(model):
            m.processor._unsloth_int8 = True
        A.install_int8_exact_start(model, _Pipe(), 2)
        proc = A._h3_attention_modules(model)[0].processor
        procs.append(A._EXACT_KERNELS[proc._unsloth_int8_exact_kernel])
    assert procs[:3] == ["_native_flash", "_native_cudnn", "_native_math"]


def test_the_exact_start_op_adds_no_graph_break(monkeypatch):
    import torch._dynamo

    model, off, calls = _int8_model(monkeypatch)

    def breaks(m):
        torch._dynamo.reset()
        with torch.no_grad():
            n = torch._dynamo.explain(m)(**_inputs()).graph_break_count
        torch._dynamo.reset()
        return n

    # the 16-bit strided processor is the bar (a break splits every compiled block in two)
    A.int8_attention_exact_start_op()
    flag = torch.ones(1, dtype = torch.int8)
    for m in A._h3_attention_modules(model):
        m.processor._unsloth_int8_exact_flag = flag
        m.processor._unsloth_int8_exact = True
    assert breaks(model) == breaks(off)


def test_an_enclosing_fullgraph_compile_traces_the_route_without_recompiling(monkeypatch):
    # a caller compiling a function around the whole denoiser (hook included): no graph break, no Unsupported, and
    # the step decides the route at run time
    import torch._dynamo
    from torch._dynamo.utils import counters

    model, off, calls = _int8_model(monkeypatch)
    pipe = _Pipe()
    assert A.install_int8_exact_start(model, pipe, 2) == 3
    torch._dynamo.reset()
    counters.clear()

    @torch.compile(fullgraph = True, backend = "aot_eager")
    def step(**kw):
        return model(**kw)[0].sum()

    ref = _run(off)[0].sum()
    try:
        with torch.no_grad():
            for index, int8 in ((None, False), (1, False), (2, True), (7, True), (None, False)):
                pipe.scheduler._step_index = index
                calls.clear()
                out = step(**_inputs())
                assert bool(calls) is int8
                assert torch.equal(out, ref) is (not int8) or int8
        assert counters["stats"]["unique_graphs"] == 1
    finally:
        torch._dynamo.reset()


def test_overlapping_denoisers_keep_their_own_route(monkeypatch):
    model_a, _, calls = _int8_model(monkeypatch)
    model_b = copy.deepcopy(model_a)
    pipe_a, pipe_b = _Pipe(), _Pipe()
    assert A.install_int8_exact_start(model_a, pipe_a, 2) == 3
    assert A.install_int8_exact_start(model_b, pipe_b, 2) == 3
    # A is inside its exact start (step 0); its hook has run, then B runs a whole forward past its own
    pipe_a.scheduler._step_index = None
    A._exact_start_pre_hook(model_a, (), {})
    for index in (None, 1, 2):
        pipe_b.scheduler._step_index = index
        calls.clear()
        _run(model_b)
    assert calls  # B is past its exact start and took INT8
    flag_a = model_a.__dict__[A._EXACT_HOOK_ATTR]["flag"]
    flag_b = model_b.__dict__[A._EXACT_HOOK_ATTR]["flag"]
    assert flag_a is not flag_b and int(flag_a[0]) == 1 and int(flag_b[0]) == 0
    # A's attention still takes the exact route
    proc = A._h3_attention_modules(model_a)[1].processor
    assert proc._unsloth_int8_exact_flag is flag_a
    calls.clear()
    q = torch.randn(1, 8, 2, 32)
    A.int8_attention_exact_start_impl(q, q, q, proc._unsloth_int8_exact_flag, 0)
    assert not calls


# ── the H3 video load ──


def _load_fn():
    tree = ast.parse((BACKEND_DIR / "core" / "inference" / "video.py").read_text(encoding = "utf-8"))
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_load_h3_modular_pipeline"
    )


def _finish(
    model,
    pipe,
    policy = None,
):
    return A.finish_h3_int8_auto(
        model, pipe, 4, modules = 3, policy = I8.resolve_policy(policy), status_pipe = None
    )


def test_the_loader_step_names_the_exact_start(monkeypatch):
    model, off, calls = _int8_model(monkeypatch)
    monkeypatch.setattr(
        I8,
        "resolved_entry",
        lambda policy, pipe: (None, "on", "unsloth_int8 (cuda) on 3 attention modules"),
    )
    modules, reason = _finish(model, _Pipe())
    assert modules == 3 and A.int8_exact_start_count(model) == 3
    assert (
        reason
        == "unsloth_int8 (cuda) on 3 attention modules, the first 4 denoise steps on 16-bit attention"
    )


@pytest.mark.parametrize("failure", ["raises", "zero"])
def test_the_loader_step_takes_int8_off_when_the_exact_start_cannot_install(monkeypatch, failure):
    model, off, calls = _int8_model(monkeypatch)
    real = A.install_int8_exact_start

    def broken(
        transformer,
        pipe,
        steps,
        logger = None,
    ):
        if failure == "raises":
            real(transformer, pipe, steps)  # half installed, then fails
            raise RuntimeError("boom")
        return 0

    monkeypatch.setattr(A, "install_int8_exact_start", broken)
    modules, reason = _finish(model, _Pipe())
    assert modules == 0 and reason.startswith("Auto turned it off for this load")
    assert I8.int8_attention_count(model) == 0 and A.int8_exact_start_count(model) == 0
    assert A._EXACT_HOOK_ATTR not in model.__dict__ and A.GRAPH_UNSAFE_ATTR not in model.__dict__
    calls.clear()
    assert _equal(_run(model), _run(off)) and not calls


def test_the_h3_load_uses_the_loader_step():
    load = _load_fn()
    calls = [
        n
        for n in ast.walk(load)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "finish_h3_int8_auto"
    ]
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == [
        "h3_denoiser",
        "pipe",
        "h3_int8_gate.exact_steps",
    ]
    assert {k.arg: ast.unparse(k.value) for k in calls[0].keywords}["modules"] == "h3_int8_modules"
    guard = next(
        n
        for n in ast.walk(load)
        if isinstance(n, ast.If)
        and any(c is calls[0] for c in ast.walk(n))
        and isinstance(n.test, ast.BoolOp)
    )
    assert ast.unparse(guard.test) == "int8_auto and h3_int8_gate.exact_steps > 0"
