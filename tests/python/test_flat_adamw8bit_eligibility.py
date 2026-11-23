# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth/optimizers/flat_adamw8bit.py on the CPU: eligibility, install and the fused step itself.

bitsandbytes >= 0.50 steps blockwise 8-bit Adam on CPU tensors, so the fused path runs here bitwise
against the stock step with the module's device type switched to "cpu" (the default-stream check is
CUDA-only and is stubbed; its own test flips it)."""

import copy
import importlib.util
import io
import os
import pathlib
import pickle
import tempfile
import types

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flat = _load("_unsloth_flat_adamw8bit_eligibility", "unsloth/optimizers/flat_adamw8bit.py")
fsdp2_bnb = _load("_unsloth_fsdp2_bnb_optim_eligibility", "unsloth/models/fsdp2_bnb_optim.py")
offload = _load("_unsloth_state_offload_eligibility", "unsloth/optimizers/state_offload.py")


def _bnb_steps_cpu():
    try:
        p = torch.nn.Parameter(torch.zeros(4096))
        p.grad = torch.ones(4096)
        bnb.optim.AdamW8bit([p], lr = 1e-3).step()
        return True
    except Exception:
        return False


needs_cpu_kernels = pytest.mark.skipif(
    not _bnb_steps_cpu(), reason = "bitsandbytes < 0.50 has no CPU 8-bit kernels"
)

# Multiples of 256 elements, >= min_8bit_size (4096); a 12288-element chunk fuses 1-3 of them.
_SHAPES = [(16, 256), (256, 16), (16, 512), (32, 256), (16, 256), (512, 16)] * 2


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(flat, "_DEVICE_TYPE", "cpu")
    monkeypatch.setattr(flat, "_on_default_stream", lambda device: True)
    monkeypatch.setenv("UNSLOTH_FLAT_ADAMW8BIT_CHUNK", "12288")
    for env in ("UNSLOTH_FLAT_ADAMW8BIT", "UNSLOTH_OFFLOAD_OPTIMIZER_STATE"):
        monkeypatch.delenv(env, raising = False)
    for env in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
        monkeypatch.delenv(env, raising = False)


def _params(shapes = _SHAPES, seed = 0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator = g) * 0.02) for s in shapes]


def _grads(params, step):
    g = torch.Generator().manual_seed(1000 + step)
    for p in params:
        p.grad = torch.randn(p.shape, generator = g) * 1e-3


def _make(
    params,
    cls = None,
    **kw,
):
    cls = cls or bnb.optim.AdamW8bit
    half = len(params) // 2
    groups = [
        {"params": params[:half], "weight_decay": 0.01},
        {"params": params[half:], "weight_decay": 0.0},
    ]
    if cls in (bnb.optim.Adam, bnb.optim.AdamW):
        kw.setdefault("optim_bits", 8)
    return cls(groups, lr = 2e-4, **kw)


def _run(
    opt,
    params,
    steps,
    start = 0,
    before = None,
):
    for s in range(start, start + steps):
        _grads(params, s)
        if before is not None:
            before(s, opt, params)
        opt.step()
        opt.zero_grad(set_to_none = True)


def _same(pa, pb, oa, ob):
    for a, b in zip(pa, pb):
        assert torch.equal(a, b)
    for a, b in zip(pa, pb):
        sa, sb = oa.state[a], ob.state[b]
        assert sa["step"] == sb["step"]
        assert sa.keys() == sb.keys()
        for key in ("state1", "state2", "absmax1", "absmax2"):
            assert torch.equal(sa[key], sb[key]), key


def _fused(opt):
    return opt.__dict__.get("_unsloth_flat_fused_steps", 0)


def _pair(
    cls = None,
    shapes = _SHAPES,
    install = True,
):
    pa, pb = _params(shapes), _params(shapes)
    oa, ob = _make(pa, cls), _make(pb, cls)
    if install:
        assert flat.install_flat_adamw8bit(ob) is ob
    return pa, pb, oa, ob


# ---- the fused step: parity, chunking, engagement --------------------------------------------------


@needs_cpu_kernels
@pytest.mark.parametrize("cls_name", ["Adam", "AdamW", "Adam8bit", "AdamW8bit"])
@pytest.mark.parametrize("chunk", ["4096", "12288", "1e9"])
def test_bitwise_equal_to_stock(cpu, monkeypatch, cls_name, chunk):
    monkeypatch.setenv("UNSLOTH_FLAT_ADAMW8BIT_CHUNK", chunk)
    pa, pb, oa, ob = _pair(getattr(bnb.optim, cls_name))
    _run(oa, pa, 5)
    _run(ob, pb, 5)
    assert _fused(ob) == 5
    _same(pa, pb, oa, ob)
    segments = [plan["segments"] for plan in ob.__dict__["_unsloth_flat_plans"].values()]
    multi = any(end - start > 1 for segs in segments for start, end in segs)
    assert multi == (chunk != "4096")


@needs_cpu_kernels
def test_checkpoint_resume_and_move_to_device(cpu):
    # Row 3: both forms of move_to_device reach bitsandbytes; the restored state is repacked.
    pr = _params()
    orf = _make(pr)
    _run(orf, pr, 6)
    p1 = _params()
    o1 = _make(p1)
    flat.install_flat_adamw8bit(o1)
    _run(o1, p1, 3)
    buf = io.BytesIO()
    torch.save(o1.state_dict(), buf)
    for form in ("positional", "keyword"):
        buf.seek(0)
        state = torch.load(buf, weights_only = False)
        p2 = [torch.nn.Parameter(p.detach().clone()) for p in p1]
        o2 = _make(p2)
        flat.install_flat_adamw8bit(o2)
        if form == "positional":
            o2.load_state_dict(copy.deepcopy(state), True)
        else:
            o2.load_state_dict(copy.deepcopy(state), move_to_device = True)
        _run(o2, p2, 3, start = 3)
        assert _fused(o2) == 3
        _same(pr, p2, orf, o2)


@needs_cpu_kernels
def test_empty_group_and_hooks_once(cpu):
    p = _params()
    opt = bnb.optim.AdamW8bit([{"params": p}, {"params": [], "weight_decay": 0.0}], lr = 1e-4)
    flat.install_flat_adamw8bit(opt)
    calls = []
    opt.register_step_pre_hook(lambda *a: calls.append("pre"))
    opt.register_step_post_hook(lambda *a: calls.append("post"))
    _grads(p, 0)
    assert opt.step(closure = lambda: torch.tensor(3.0)).item() == 3.0
    assert calls == ["pre", "post"] and _fused(opt) == 1


# ---- optimizer-wide eligibility: every row falls back to the stock step, bitwise ------------------


def _flip_stream(s, opt, params):
    flat._on_default_stream = lambda device: s % 2 == 0


def _late_override(s, opt, params):
    if s == 2:
        bnb.optim.GlobalOptimManager.get_instance().pid2config[id(params[7])] = {"lr": 1e-2}


def _replace_interior_scale(s, opt, params):
    if s == 3:
        state = opt.state[params[1]]
        state["absmax2"] = state["absmax2"].clone()
        state["state1"] = state["state1"].clone()


def _tensor_step(s, opt, params):
    if s == 2:
        for p in params:
            opt.state[p]["step"] = torch.tensor(opt.state[p]["step"])


_FALLBACKS = {
    # name: (before-hook, fused steps expected out of 5)
    "row1_side_stream": (_flip_stream, 3),
    "row4_late_override": (_late_override, 2),
    "row13_replaced_views_repacked": (_replace_interior_scale, 5),
    "row25_tensor_step": (_tensor_step, 2),
}


@needs_cpu_kernels
@pytest.mark.parametrize("name", list(_FALLBACKS))
def test_live_fallbacks_match_stock(cpu, monkeypatch, name):
    before, fused = _FALLBACKS[name]
    monkeypatch.setattr(flat, "_on_default_stream", flat._on_default_stream)
    mng = bnb.optim.GlobalOptimManager.get_instance()
    pa, pb, oa, ob = _pair()
    try:
        _run(oa, pa, 5, before = before)
        _run(ob, pb, 5, before = before)
    finally:
        for p in pa + pb:
            mng.pid2config.pop(id(p), None)
            mng.index2config.clear()
    assert _fused(ob) == fused
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_row7_replaced_member_rebuilds_plan(cpu):
    shapes = _SHAPES + [(16, 256)]
    pa, pb = _params(shapes), _params(shapes)
    oa = _make(pa[:-1])
    ob = _make(pb[:-1])
    flat.install_flat_adamw8bit(ob)
    _run(oa, pa[:-1], 2)
    _run(ob, pb[:-1], 2)
    removed_b = ob.param_groups[1]["params"][2]
    removed_before = removed_b.detach().clone()
    for o, ps in ((oa, pa), (ob, pb)):
        o.param_groups[1]["params"][2] = ps[-1]
    live_a = pa[:8] + [pa[-1]] + pa[9:12]
    live_b = pb[:8] + [pb[-1]] + pb[9:12]
    _run(oa, live_a, 2, start = 2)
    _run(ob, live_b, 2, start = 2)
    # The new member has no state (then a younger step counter): stock from here, as it must be,
    # and the cached plan never touches the removed parameter.
    assert _fused(ob) == 2
    assert torch.equal(removed_b, removed_before)
    _same(live_a, live_b, oa, ob)


@needs_cpu_kernels
def test_row9_ineligible_later_group_allocates_nothing(cpu):
    shapes = _SHAPES[:6] + [(17, 300)] + _SHAPES[7:]  # 5100 elements: not a multiple of 256
    pa, pb, oa, ob = _pair(shapes = shapes)
    _run(oa, pa, 3)
    _run(ob, pb, 3)
    assert _fused(ob) == 0
    plans = ob.__dict__.get("_unsloth_flat_plans", {})
    assert not any(plan["packed"] for plan in plans.values())
    for p in pb:
        assert ob.state[p]["state1"]._base is None  # never rebound into a packed buffer
    _same(pa, pb, oa, ob)


def _multi_segments(opt, gindex = 0):
    plan = opt.__dict__["_unsloth_flat_plans"][gindex]
    return [(s, e) for s, e in plan["segments"] if e - s > 1]


@needs_cpu_kernels
def test_row5_pack_oom_falls_back_untouched(cpu, monkeypatch):
    pa, pb, oa, ob = _pair()
    _run(oa, pa, 1)
    real_empty = torch.empty
    calls = []

    def empty(*args, **kwargs):
        calls.append(1)
        if len(calls) == 6:  # the second fused chunk's second buffer
            raise flat._OOM("simulated")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(flat.torch, "empty", empty)
    _run(ob, pb, 1)
    monkeypatch.setattr(flat.torch, "empty", real_empty)
    assert _fused(ob) == 0
    # The failed chunk kept its own tensors; the step ran stock.
    (s0, e0), (s1, e1) = _multi_segments(ob)[:2]
    assert all(ob.state[p]["state1"]._base is not None for p in pb[s0:e0])
    assert all(ob.state[p]["state1"]._base is None for p in pb[s1:e1])
    _same(pa, pb, oa, ob)
    _run(oa, pa, 2, start = 1)
    _run(ob, pb, 2, start = 1)
    assert _fused(ob) == 2
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_row21_chunk_failure_keeps_counters_consistent(cpu, monkeypatch):
    _, pb, _, ob = _pair()
    _run(ob, pb, 1)
    real_cat, calls = torch.cat, []

    def cat(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:  # the second chunk's gather
            raise RuntimeError("simulated")
        return real_cat(*args, **kwargs)

    _grads(pb, 1)
    monkeypatch.setattr(flat.torch, "cat", cat)
    with pytest.raises(RuntimeError, match = "simulated"):
        ob.step()
    monkeypatch.setattr(flat.torch, "cat", real_cat)
    # Chunks before the failing one advanced and say so; the failing one and later did not.
    failing = _multi_segments(ob)[1][0]
    assert all(ob.state[p]["step"] == 2 for p in pb[:failing])
    assert all(ob.state[p]["step"] == 1 for p in pb[failing:])


def _raise_oom_in(monkeypatch, when):
    """torch.empty / torch.cat called from the module raise OOM when they would allocate and
    when(frame) holds (torch.cat with out= writes into an existing buffer and is let through)."""
    import sys

    real_empty, real_cat, raised = torch.empty, torch.cat, []

    def guard(real, allocates):
        def fn(*args, **kwargs):
            frame = sys._getframe(1)
            if frame.f_globals is vars(flat) and allocates(kwargs) and when(frame):
                raised.append(frame.f_code.co_name)
                raise flat._OOM("simulated")
            return real(*args, **kwargs)

        return fn

    monkeypatch.setattr(flat.torch, "empty", guard(real_empty, lambda kw: True))
    monkeypatch.setattr(flat.torch, "cat", guard(real_cat, lambda kw: kw.get("out") is None))
    return raised


@needs_cpu_kernels
def test_oom_after_first_update_cannot_happen(cpu, monkeypatch):
    # Any allocation the fused step makes once a parameter has been updated fails: the step must
    # still complete, bitwise the stock one, with every counter advanced once.
    pa, pb, oa, ob = _pair()
    _run(oa, pa, 2)
    _run(ob, pb, 1)
    updated = []
    real_update = bnb.optim.optimizer.Optimizer2State.update_step

    def update_step(self, *args, **kwargs):
        updated.append(1)
        return real_update(self, *args, **kwargs)

    monkeypatch.setattr(bnb.optim.optimizer.Optimizer2State, "update_step", update_step)
    raised = _raise_oom_in(monkeypatch, lambda frame: bool(updated))
    _run(ob, pb, 1, start = 1)
    assert raised == []
    assert _fused(ob) == 2
    assert all(ob.state[p]["step"] == 2 for p in pb)
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_scratch_oom_runs_the_step_stock(cpu):
    pa, pb, oa, ob = _pair()
    _run(oa, pa, 2)
    _run(ob, pb, 1)
    with pytest.MonkeyPatch.context() as mp:
        raised = _raise_oom_in(mp, lambda frame: frame.f_code.co_name == "_scratch")
        _run(ob, pb, 1, start = 1)
    assert raised == ["_scratch"]
    assert _fused(ob) == 1
    assert all(ob.state[p]["step"] == 2 for p in pb)
    _same(pa, pb, oa, ob)
    _run(oa, pa, 2, start = 2)
    _run(ob, pb, 2, start = 2)
    assert _fused(ob) == 3
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_gradient_sharing_a_parameter_runs_stock(cpu):
    # pb[1] and pb[0] share a chunk; stock updates pb[0] before reading pb[1].grad.
    def alias(s, opt, params):
        if s == 1:
            params[1].grad = params[0].detach().view(params[1].shape)

    pa, pb, oa, ob = _pair()
    _run(oa, pa, 3, before = alias)
    _run(ob, pb, 3, before = alias)
    assert (0, 2) in _multi_segments(ob)
    assert _fused(ob) == 2
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_noncontiguous_gradient_runs_stock(cpu):
    pa, pb, oa, ob = _pair()
    for params, opt in ((pa, oa), (pb, ob)):
        for s in range(3):
            _grads(params, s)
            if s == 1:
                params[1].grad = params[1].grad.t().contiguous().t()
            opt.step()
            if s == 1:
                assert params[1].grad.is_contiguous()  # stock replaces it with a contiguous copy
            opt.zero_grad(set_to_none = True)
    assert _fused(ob) == 2
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_row6_instance_update_step_sees_real_params(cpu):
    pa, pb, oa, ob = _pair()
    seen = []
    original = ob.update_step

    def update_step(group, p, gindex, pindex):
        seen.append(p)
        return original(group, p, gindex, pindex)

    ob.update_step = update_step  # as bf16_stochastic_rounding._wrap_bitsandbytes does
    _run(oa, pa, 2)
    _run(ob, pb, 2)
    assert _fused(ob) == 0
    assert {id(p) for p in seen} == {id(p) for p in pb}
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
@pytest.mark.parametrize("shape", ["duplicate", "alias", "overlap"])
def test_rows11_12_shared_storage_runs_stock(cpu, shape):
    def build():
        ps = _params()
        if shape == "duplicate":
            group = ps[:6] + [ps[2]]
        elif shape == "alias":
            alias = torch.nn.Parameter(ps[2].data)
            group = ps[:6] + [alias]
        else:
            base = torch.zeros(8192)
            group = ps[:6] + [
                torch.nn.Parameter(base[:4096].view(16, 256)),
                torch.nn.Parameter(base[4096:].view(16, 256)),
            ]
        return group, bnb.optim.AdamW8bit([{"params": group}], lr = 2e-4)

    ga, oa = build()
    gb, ob = build()
    flat.install_flat_adamw8bit(ob)
    _run(oa, ga, 2)
    _run(ob, gb, 2)
    assert _fused(ob) == 0
    for a, b in zip(ga, gb):
        assert torch.equal(a, b)


@needs_cpu_kernels
def test_row14_changed_layout_runs_stock(cpu):
    def noncontiguous(s, opt, params):
        if s == 2:
            params[3].data = params[3].data.t().contiguous().t()

    pa, pb, oa, ob = _pair()
    _run(oa, pa, 4, before = noncontiguous)
    _run(ob, pb, 4, before = noncontiguous)
    assert _fused(ob) == 3  # stock's update_step makes it contiguous again
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
@pytest.mark.parametrize("damage", ["row15_off_device", "row18_missing_key", "row18_wrong_dtype"])
def test_rows15_18_bad_state_is_read_only_refused(cpu, damage):
    _, pb, _, ob = _pair()
    _run(ob, pb, 1)
    state = ob.state[pb[4]]
    if damage == "row15_off_device":
        state["absmax1"] = torch.empty(state["absmax1"].shape, device = "meta")
    elif damage == "row18_missing_key":
        del state["absmax1"]
    else:
        state["state2"] = state["state2"].float()
    before = {k: v for k, v in state.items()}
    _grads(pb, 1)
    assert flat._live_plans(ob) is None
    assert state == before


@needs_cpu_kernels
def test_row19_prebuilt_scheduler_reaches_fused_step(cpu):
    p = _params()
    opt = _make(p)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    assert flat.install_flat_adamw8bit(opt) is opt
    _grads(p, 0)
    opt.step()
    sched.step()
    assert _fused(opt) == 1


@needs_cpu_kernels
def test_row22_row24_offload_and_parking_run_stock(cpu):
    pa, pb, oa, ob = _pair()
    _run(oa, pa, 1)
    _run(ob, pb, 1)
    assert _fused(ob) == 1
    ob.__dict__["_unsloth_flat_disabled"] = "rollout parking"
    _run(oa, pa, 1, start = 1)
    _run(ob, pb, 1, start = 1)
    assert _fused(ob) == 1
    _same(pa, pb, oa, ob)
    # Programmatic offload_optimizer_state drops the packed buffers and the step runs stock.
    del ob.__dict__["_unsloth_flat_disabled"]
    assert ob.__dict__["_unsloth_flat_plans"]
    assert offload.offload_optimizer_state(ob, pin_memory = False) is None  # CPU params: no-op
    ob._unsloth_state_offloader = object()
    assert flat._live_plans(ob) is None
    fresh = _make(_params())
    fresh._unsloth_state_offloader = object()
    assert flat.install_flat_adamw8bit(fresh) is None


def test_offload_drops_flat_plans_before_offloading():
    src = (_ROOT / "unsloth/optimizers/state_offload.py").read_text()
    pop = src.index('optimizer.__dict__.pop("_unsloth_flat_plans", None)')
    assert pop < src.index("offloader = _StateOffloader(")
    rl = (_ROOT / "unsloth/models/rl.py").read_text()
    park = rl.index('optimizer.__dict__["_unsloth_flat_disabled"] = "rollout parking"')
    assert park < rl.index("    for state, name, host in states:\n        state[name] = host")


@needs_cpu_kernels
def test_compiling_runs_stock(cpu, monkeypatch):
    pa, pb, oa, ob = _pair()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _run(oa, pa, 1)
    _run(ob, pb, 1)
    assert _fused(ob) == 0
    _same(pa, pb, oa, ob)


@needs_cpu_kernels
def test_pickle_roundtrip_is_stock_class(cpu):
    _, pb, _, ob = _pair()
    _run(ob, pb, 1)
    clone = pickle.loads(pickle.dumps(ob))
    assert type(clone) is bnb.optim.AdamW8bit


# ---- static eligibility: options, wrappers, install refusals ------------------------------------


class _FakeParam:
    def __init__(
        self,
        n,
        device = "cpu",
    ):
        self.data = torch.empty(0)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._n = n

    def numel(self):
        return self._n


def _fake_opt(**args):
    base = dict(optim_bits = 8, min_8bit_size = 4096, max_unorm = 0.0)
    base.update(args)
    return types.SimpleNamespace(args = types.SimpleNamespace(**base), is_paged = False)


# bnb 0.50 dropped percentile_clipping and block_wise from args: absent means the defaults.
_REJECTED = [
    ({"percentile_clipping": 5}, "percentile_clipping"),
    ({"percentile_clipping": 99}, "percentile_clipping"),
    ({"max_unorm": 1.0}, "max_unorm"),
    ({"block_wise": False}, "block_wise"),
    ({"optim_bits": 32}, "optim_bits"),
    ({"skip_zeros": True}, "skip_zeros"),
]


@pytest.mark.parametrize("args, reason", _REJECTED, ids = [r for _, r in _REJECTED])
def test_per_tensor_options_rejected(cpu, args, reason):
    assert reason in flat._unsupported_args(_fake_opt(**args))
    opt = _make(_params())
    for key, value in args.items():
        setattr(opt.args, key, value)
    assert flat._live_plans(opt) is None
    assert flat.install_flat_adamw8bit(opt) is None


@pytest.mark.parametrize(
    "args", [{}, {"percentile_clipping": 100, "block_wise": True, "skip_zeros": False}]
)
def test_stock_defaults_accepted(args):
    assert flat._unsupported_args(_fake_opt(**args)) is None


def test_paged_rejected():
    opt = _fake_opt()
    opt.is_paged = True
    assert flat._unsupported_args(opt) == "paged"


def test_install_refusals(cpu):
    assert flat.install_flat_adamw8bit(_make(_params())) is not None
    opt = _make(_params())
    opt.update_step = opt.update_step
    assert flat.install_flat_adamw8bit(opt) is None
    opt = _make(_params())
    opt.step = lambda *a, **k: None  # some other instance step wrapper
    assert flat.install_flat_adamw8bit(opt) is None


def test_cuda_only_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FLAT_ADAMW8BIT", raising = False)
    assert flat._DEVICE_TYPE == "cuda"
    assert flat._install_refusal(_make(_params())) == "parameters not on one CUDA device"


def test_kill_switch(cpu, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FLAT_ADAMW8BIT", "0")
    assert flat.install_flat_adamw8bit(_make(_params())) is None


@pytest.mark.parametrize(
    "value, expected",
    [("inf", 1 << 20), ("-inf", 1 << 20), ("1e400", 1 << 20), ("nan", 1 << 20), ("x", 1 << 20)]
    + [("4294967296", 1 << 30), ("100", 256), ("1000", 768)],
)
def test_chunk_setting(monkeypatch, value, expected):
    monkeypatch.setenv("UNSLOTH_FLAT_ADAMW8BIT_CHUNK", value)
    assert flat._chunk_elements() == expected


def test_nested_profile_wrapper_declined():
    import functools

    class A(bnb.optim.AdamW8bit):
        pass

    hooked = torch.optim.Optimizer.profile_hook_step(bnb.optim.AdamW8bit.step)
    A.step = hooked
    assert flat._stock_step_for(A) is hooked.__wrapped__

    @functools.wraps(hooked)
    def outer(self, closure = None):
        return hooked(self, closure)

    A.step = outer
    assert flat._stock_step_for(A) is None


# ---- Trainer / distributed install points --------------------------------------------------------


def _trainer(**attrs):
    calls = []

    def create_optimizer(*args, **kwargs):
        calls.append(1)
        return trainer.optimizer

    trainer = types.SimpleNamespace(create_optimizer = create_optimizer, optimizer = None, **attrs)
    return trainer, create_optimizer, calls


def _state(**kw):
    return {"accelerator": types.SimpleNamespace(state = types.SimpleNamespace(**kw))}


_DISTRIBUTED = [
    ("is_fsdp_enabled", {"is_fsdp_enabled": True}),
    ("is_fsdp_xla_enabled", {"is_fsdp_xla_enabled": True}),
    ("is_deepspeed_enabled", {"is_deepspeed_enabled": True}),
    ("native_fsdp2", {"model": types.SimpleNamespace(_unsloth_fsdp2_native = True)}),
    ("args_fsdp", {"args": types.SimpleNamespace(fsdp = ["full_shard"], deepspeed = None)}),
    ("args_deepspeed", {"args": types.SimpleNamespace(fsdp = [], deepspeed = "ds.json")}),
    ("fsdp_plugin", _state(fsdp_plugin = object())),
    ("deepspeed_plugin", _state(deepspeed_plugin = object())),
    ("distributed_type_fsdp", _state(distributed_type = "DistributedType.FSDP")),
    ("distributed_type_deepspeed", _state(distributed_type = "DistributedType.DEEPSPEED")),
]


@pytest.mark.parametrize("attrs", [a for _, a in _DISTRIBUTED], ids = [i for i, _ in _DISTRIBUTED])
def test_hook_not_installed_under_fsdp_or_deepspeed(cpu, attrs):
    trainer, original, _ = _trainer(**attrs)
    assert flat.install_flat_adamw8bit_hook(trainer) is False
    assert trainer.create_optimizer is original


@pytest.mark.parametrize(
    "env, value, installs",
    [
        ("ACCELERATE_USE_FSDP", "true", False),
        ("ACCELERATE_USE_DEEPSPEED", "1", False),
        ("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "1", False),
        # Row 16: values the offloader reads as off leave the fused step on.
        ("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "false", True),
        ("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "off", True),
        ("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "", True),
    ],
)
def test_hook_env(cpu, monkeypatch, env, value, installs):
    monkeypatch.setenv(env, value)
    trainer, _, _ = _trainer()
    assert flat.install_flat_adamw8bit_hook(trainer) is installs


def test_trainer_offload_branch_uses_offloader_predicate():
    src = (_ROOT / "unsloth/models/_utils.py").read_text()
    assert 'os.environ.get("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "0") != "0"' not in src
    assert 'if offload.strip().lower() in ("1", "true", "yes", "on"):' in src
    off = (_ROOT / "unsloth/optimizers/state_offload.py").read_text()
    assert 'os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")' in off


def test_hook_installs_at_create_optimizer_and_rechecks(cpu):
    trainer, _, calls = _trainer(
        args = types.SimpleNamespace(fsdp = [], deepspeed = None),
        **_state(distributed_type = "DistributedType.MULTI_GPU"),
    )
    assert flat.install_flat_adamw8bit_hook(trainer) is True
    trainer.optimizer = _make(_params())
    assert trainer.create_optimizer() is trainer.optimizer
    assert calls == [1] and type(trainer.optimizer)._unsloth_flat_adamw8bit
    # FSDP switched on after Trainer.__init__ (before create_optimizer): the class stays stock.
    trainer.is_fsdp_enabled = True
    trainer.optimizer = _make(_params())
    trainer.create_optimizer()
    assert type(trainer.optimizer) is bnb.optim.AdamW8bit


def test_localize_restores_flat_class_for_fsdp2(cpu):
    # accelerate FSDP2 prepare -> _localize_or_refuse_bnb -> localize_bnb_optimizer: the flat class
    # overrides step, so the plain-class check refused it and the Trainer raised ValueError.
    opt = _make(_params())
    assert flat.install_flat_adamw8bit(opt) is opt
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    assert fsdp2_bnb.localize_bnb_optimizer(opt) is True
    assert fsdp2_bnb.is_local_shard_bnb_optimizer(opt)
    assert not getattr(type(opt), "_unsloth_flat_adamw8bit", False)
    assert "_unsloth_flat_plans" not in vars(opt)
    assert opt.step.__wrapped__ is type(opt).step
    del sched


@pytest.fixture
def _single_rank_gloo():
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    if dist.is_initialized():
        yield
        return
    with tempfile.TemporaryDirectory() as d:
        dist.init_process_group(
            "gloo", init_method = f"file://{os.path.join(d, 'store')}", rank = 0, world_size = 1
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def test_install_refuses_dtensor_params(cpu, _single_rank_gloo):
    from torch.distributed.device_mesh import init_device_mesh

    try:
        from torch.distributed.tensor import Shard, distribute_tensor
    except ImportError:  # torch < 2.5
        from torch.distributed._tensor import Shard, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    p = torch.nn.Parameter(distribute_tensor(torch.zeros(4096), mesh, [Shard(0)]))
    opt = bnb.optim.AdamW8bit([p], lr = 1e-3)
    assert flat.install_flat_adamw8bit(opt) is None
    assert type(opt) is bnb.optim.AdamW8bit


@needs_cpu_kernels
def test_real_trainer_installs_before_prepare(cpu, monkeypatch, tmp_path):
    # Row 10: transformers 5.x prepares the optimizer before create_scheduler; Accelerate (fp16
    # GradScaler) captures optimizer.step at prepare, so the fused class must be in place by then.
    transformers = pytest.importorskip("transformers")
    from accelerate import Accelerator

    seen = []
    prepare = Accelerator.prepare

    def recording_prepare(self, *objs, **kw):
        seen.extend(type(o) for o in objs if isinstance(o, torch.optim.Optimizer))
        return prepare(self, *objs, **kw)

    monkeypatch.setattr(Accelerator, "prepare", recording_prepare)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 64, bias = False)
            self.b = torch.nn.Linear(64, 128, bias = False)

        def forward(
            self,
            x,
            labels = None,
        ):
            y = self.b(self.a(x))
            return {"loss": (y.float() ** 2).mean(), "logits": y}

    data = [{"x": torch.randn(64), "labels": torch.zeros(1)} for _ in range(8)]
    args = transformers.TrainingArguments(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 3,
        optim = "adamw_8bit",
        learning_rate = 1e-3,
        report_to = [],
        use_cpu = True,
        save_strategy = "no",
        logging_strategy = "no",
    )
    trainer = transformers.Trainer(model = Tiny(), args = args, train_dataset = data)
    assert flat.install_flat_adamw8bit_hook(trainer)
    trainer.train()
    inner = flat._innermost(trainer.optimizer)
    assert getattr(type(inner), "_unsloth_flat_adamw8bit", False)
    assert seen and all(getattr(t, "_unsloth_flat_adamw8bit", False) for t in seen)
    assert _fused(inner) == 3
