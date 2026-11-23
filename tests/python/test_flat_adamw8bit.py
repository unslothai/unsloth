# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth/optimizers/flat_adamw8bit.py: the per-group 8-bit AdamW step is bitwise the stock one."""

import copy
import importlib.util
import io
import pathlib

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device"),
]

_PATH = pathlib.Path(__file__).resolve().parents[2] / "unsloth" / "optimizers" / "flat_adamw8bit.py"
_spec = importlib.util.spec_from_file_location("_unsloth_flat_adamw8bit_under_test", _PATH)
flat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flat)

# LoRA r16 shapes (A: r x in, B: out x r), all multiples of 256 elements.
_SHAPES = [(16, 1024), (2048, 16), (16, 1024), (1024, 16), (16, 3072), (1024, 16)] * 3


def _params(shapes, seed = 0):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, device = "cuda", generator = g) * 0.02) for s in shapes]


def _grads(params, step):
    g = torch.Generator(device = "cuda").manual_seed(1000 + step)
    for p in params:
        p.grad = torch.randn(p.shape, device = "cuda", generator = g) * 1e-3


def _make(
    params,
    cls = None,
    **kw,
):
    cls = cls or bnb.optim.AdamW
    half = len(params) // 2
    groups = [
        {"params": params[:half], "weight_decay": 0.01},
        {"params": params[half:], "weight_decay": 0.0},
    ]
    return cls(groups, lr = 2e-4, optim_bits = 8, **kw)


def _run(
    opt,
    params,
    steps,
    start = 0,
    none_at = None,
):
    for s in range(start, start + steps):
        _grads(params, s)
        if none_at is not None and s == none_at:
            params[3].grad = None
        for group in opt.param_groups:
            group["lr"] = 2e-4 * (1.0 - 0.05 * s)  # a scheduler between steps
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
            if key not in sa:  # 32-bit state (an override) has no absmax
                continue
            assert sa[key].shape == sb[key].shape
            assert torch.equal(sa[key], sb[key]), key


def _fused(opt):
    return opt.__dict__.get("_unsloth_flat_fused_steps", 0)


def _engaged(opt):
    return _fused(opt) > 0


def _multi_chunks(opt):
    plans = opt.__dict__.get("_unsloth_flat_plans", {})
    return any(e - s > 1 for plan in plans.values() for s, e in plan["segments"])


@pytest.mark.parametrize("cls_name", ["Adam", "AdamW", "AdamW8bit", "Adam8bit"])
@pytest.mark.parametrize("chunk", [None, "4096", "65536"])
def test_bitwise_equal_to_stock(monkeypatch, cls_name, chunk):
    if chunk is not None:
        monkeypatch.setenv("UNSLOTH_FLAT_ADAMW8BIT_CHUNK", chunk)
    cls = getattr(bnb.optim, cls_name)
    pa, pb = _params(_SHAPES), _params(_SHAPES)
    if cls_name.endswith("8bit"):
        oa = cls([{"params": pa[:9], "weight_decay": 0.01}, {"params": pa[9:]}], lr = 2e-4)
        ob = cls([{"params": pb[:9], "weight_decay": 0.01}, {"params": pb[9:]}], lr = 2e-4)
    else:
        oa, ob = _make(pa, cls), _make(pb, cls)
    assert flat.install_flat_adamw8bit(ob) is ob
    assert type(ob).__name__ == cls.__name__ and "bitsandbytes" in str(type(ob))
    _run(oa, pa, 6)
    _run(ob, pb, 6)
    assert _fused(ob) == 6
    assert _multi_chunks(ob) == (chunk != "4096")  # 4096: every LoRA tensor is its own chunk
    _same(pa, pb, oa, ob)


def test_fallbacks_match_stock(monkeypatch):
    # A tensor whose size is not a multiple of 256 sends its group to the stock step.
    shapes = _SHAPES[:5] + [(17, 300)] + _SHAPES[5:]
    pa, pb = _params(shapes), _params(shapes)
    oa, ob = _make(pa), _make(pb)
    flat.install_flat_adamw8bit(ob)
    _run(oa, pa, 4)
    _run(ob, pb, 4)
    assert not _engaged(ob)
    _same(pa, pb, oa, ob)

    # A missing gradient runs that step stock; the skipped tensor's step counter then lags, so
    # every later step stays stock too.
    pa, pb = _params(_SHAPES), _params(_SHAPES)
    oa, ob = _make(pa), _make(pb)
    flat.install_flat_adamw8bit(ob)
    _run(oa, pa, 5, none_at = 2)
    _run(ob, pb, 5, none_at = 2)
    assert _fused(ob) == 2
    _same(pa, pb, oa, ob)

    # A per-parameter override (Trainer's 32-bit embedding state) runs stock. Made after
    # construction it reaches pid2config only; recent bitsandbytes honours that in get_config,
    # older ones (0.45.5) read index2config alone and keep 8-bit state, so compare with stock.
    pa, pb = _params(_SHAPES), _params(_SHAPES)
    mng = bnb.optim.GlobalOptimManager.get_instance()
    oa, ob = _make(pa), _make(pb)
    mng.override_config(pa[0], "optim_bits", 32)
    mng.override_config(pb[0], "optim_bits", 32)
    try:
        flat.install_flat_adamw8bit(ob)
        _run(oa, pa, 3)
        _run(ob, pb, 3)
        assert not _engaged(ob)
        assert ob.state[pb[0]]["state1"].dtype == oa.state[pa[0]]["state1"].dtype
        _same(pa, pb, oa, ob)
    finally:
        mng.pid2config.pop(id(pa[0]), None)
        mng.pid2config.pop(id(pb[0]), None)

    # The same override registered by index (index2config, read by every release) is fp32.
    pa, pb = _params(_SHAPES), _params(_SHAPES)
    mng.index2config[(0, 0)] = {"optim_bits": 32}
    try:
        oa, ob = _make(pa), _make(pb)
        flat.install_flat_adamw8bit(ob)
        _run(oa, pa, 3)
        _run(ob, pb, 3)
        assert not _engaged(ob)
        assert ob.state[pb[0]]["state1"].dtype == torch.float32
        _same(pa, pb, oa, ob)
    finally:
        mng.index2config.pop((0, 0), None)

    # Kill switch.
    monkeypatch.setenv("UNSLOTH_FLAT_ADAMW8BIT", "0")
    pb = _params(_SHAPES)
    assert flat.install_flat_adamw8bit(_make(pb)) is None

    # Paged and 32-bit optimizers are left alone.
    monkeypatch.delenv("UNSLOTH_FLAT_ADAMW8BIT")
    assert flat.install_flat_adamw8bit(bnb.optim.PagedAdamW8bit(_params(_SHAPES), lr = 1e-4)) is None
    assert (
        flat.install_flat_adamw8bit(bnb.optim.AdamW(_params(_SHAPES), lr = 1e-4, optim_bits = 32))
        is None
    )


def _roundtrip(src_opt):
    buf = io.BytesIO()
    torch.save(src_opt.state_dict(), buf)
    buf.seek(0)
    return torch.load(buf, weights_only = False)


@pytest.mark.parametrize("direction", ["flat_to_stock", "stock_to_flat"])
def test_checkpoint_resume_either_way(direction):
    # Reference: 8 uninterrupted stock steps.
    pr = _params(_SHAPES)
    orf = _make(pr)
    _run(orf, pr, 8)

    p1 = _params(_SHAPES)
    o1 = _make(p1)
    if direction == "flat_to_stock":
        flat.install_flat_adamw8bit(o1)
    _run(o1, p1, 4)
    state = _roundtrip(o1)
    p2 = [torch.nn.Parameter(p.detach().clone()) for p in p1]
    o2 = _make(p2)
    if direction == "stock_to_flat":
        flat.install_flat_adamw8bit(o2)
    o2.load_state_dict(copy.deepcopy(state))
    _run(o2, p2, 4, start = 4)
    if direction == "stock_to_flat":
        assert _engaged(o2)
    _same(pr, p2, orf, o2)


def test_empty_group_does_not_block_the_fused_path():
    # transformers.Trainer.create_optimizer keeps an empty no-decay group for a LoRA model.
    pa, pb = _params(_SHAPES), _params(_SHAPES)
    oa = bnb.optim.AdamW(
        [{"params": pa, "weight_decay": 0.01}, {"params": [], "weight_decay": 0.0}],
        lr = 2e-4,
        optim_bits = 8,
    )
    ob = bnb.optim.AdamW(
        [{"params": pb, "weight_decay": 0.01}, {"params": [], "weight_decay": 0.0}],
        lr = 2e-4,
        optim_bits = 8,
    )
    flat.install_flat_adamw8bit(ob)
    _run(oa, pa, 3)
    _run(ob, pb, 3)
    assert _engaged(ob)
    _same(pa, pb, oa, ob)


def test_step_hooks_fire_once_and_closure():
    p = _params(_SHAPES)
    opt = _make(p)
    flat.install_flat_adamw8bit(opt)
    calls = []
    opt.register_step_pre_hook(lambda *a: calls.append("pre"))
    opt.register_step_post_hook(lambda *a: calls.append("post"))
    _grads(p, 0)
    assert opt.step(closure = lambda: torch.tensor(3.0)).item() == 3.0
    assert calls == ["pre", "post"]
    assert _engaged(opt)


def test_side_stream_runs_stock():
    # bitsandbytes launches on the legacy default stream: off it the fused gather / copy-back would
    # race the update, so every step taken under a side stream is the stock one.
    pa, pb = _params(_SHAPES), _params(_SHAPES)
    oa, ob = _make(pa), _make(pb)
    flat.install_flat_adamw8bit(ob)
    side = torch.cuda.Stream()
    for opt, params in ((oa, pa), (ob, pb)):
        for s in range(4):
            _grads(params, s)
            torch.cuda.synchronize()
            if s % 2:
                with torch.cuda.stream(side):
                    opt.step()
                side.synchronize()
            else:
                opt.step()
            opt.zero_grad(set_to_none = True)
    torch.cuda.synchronize()
    assert _fused(ob) == 2
    _same(pa, pb, oa, ob)
