# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HunyuanVideo-1.5's capture-safe forward (``diffusion_capture_safe``).

CPU: a tiny random ``HunyuanVideo15Transformer3DModel`` run through the stock and the rewritten
forward must agree bit for bit over t2v (all-zero and emptied image stream) and i2v, batch 1 and 2,
partially padded masks, and the inputs Studio's eager trim pre-hook hands the forward. The rewritten
forward must also dispatch no op that reads a device value on the host.

CUDA: the rewritten forward records into a ``torch.cuda.CUDAGraph`` and replays bit-identical to
eager; the stock forward does not record.
"""

from __future__ import annotations

import pytest

from core.inference import diffusion_capture_safe as cs
from core.inference import diffusion_cuda_graph as cg

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

_CLS = getattr(diffusers, "HunyuanVideo15Transformer3DModel", None)
pytestmark = pytest.mark.skipif(
    _CLS is None, reason = "diffusers has no HunyuanVideo15Transformer3DModel"
)

_TEXT_DIM, _BYT5_DIM, _IMAGE_DIM = 32, 24, 16


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(cs, "_CACHE", {})
    monkeypatch.delenv(cs.CAPTURE_SAFE_ENV, raising = False)


def _tiny_model(device = "cpu"):
    torch.manual_seed(0)
    model = _CLS(
        in_channels = 4,
        out_channels = 4,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 1,
        num_refiner_layers = 1,
        text_embed_dim = _TEXT_DIM,
        text_embed_2_dim = _BYT5_DIM,
        image_embed_dim = _IMAGE_DIM,
        rope_axes_dim = (4, 6, 6),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    return model.to(device).eval()


_MASKS = {
    "b1_right_padded": ([[1, 1, 1, 0, 0, 0, 0]], [[1, 1, 0, 0, 0]]),
    "b2_interleaved": (
        [[1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0]],
        [[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]],
    ),
    "b2_valid_and_empty_byt5": ([[1] * 7, [1, 1, 1, 1, 0, 0, 0]], [[1] * 5, [0] * 5]),
}
_IMAGES = ("t2v_zero", "t2v_empty", "i2v")


def _inputs(
    masks,
    image,
    device = "cpu",
    seed = 1,
):
    gen = torch.Generator().manual_seed(seed)
    m1 = torch.tensor(masks[0])
    m2 = torch.tensor(masks[1])
    batch = m1.shape[0]
    if image == "t2v_zero":
        img = torch.zeros(batch, 3, _IMAGE_DIM)
    elif image == "t2v_empty":
        img = torch.zeros(batch, 0, _IMAGE_DIM)
    else:
        img = torch.randn(batch, 3, _IMAGE_DIM, generator = gen)
    kw = dict(
        hidden_states = torch.randn(batch, 4, 2, 4, 4, generator = gen),
        timestep = torch.full((batch,), 500.0),
        encoder_hidden_states = torch.randn(batch, m1.shape[1], _TEXT_DIM, generator = gen),
        encoder_attention_mask = m1,
        encoder_hidden_states_2 = torch.randn(batch, m2.shape[1], _BYT5_DIM, generator = gen),
        encoder_attention_mask_2 = m2,
        image_embeds = img,
        return_dict = False,
    )
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}


def _trimmed(model, kw):
    """The kwargs Studio's eager trim pre-hook hands the forward (empty t2v image, trimmed text)."""
    from core.inference import diffusion_attention as da

    _, out = da._hunyuan_trim_pre_hook(model, (), dict(kw))
    da._set_hunyuan_null_mask(model, False)
    return out


def test_hv15_rewrite_is_registered_and_resolves(fresh_cache):
    safe, why = cs.resolve(_CLS)
    assert why is None, why
    assert getattr(safe, "__unsloth_capture_safe__", False) is True
    import inspect

    assert inspect.signature(safe) == inspect.signature(_CLS.forward)
    src = inspect.getsource(safe)
    assert cs._HV15_MERGE_HELPER_NAME in src and cs._HV15_T2V_HELPER_NAME in src
    assert "if is_t2v:" not in src and "image[image_mask]" not in src


@pytest.mark.parametrize("trim", [False, True], ids = ["untrimmed", "trimmed"])
@pytest.mark.parametrize("image", _IMAGES)
@pytest.mark.parametrize("masks", list(_MASKS.values()), ids = list(_MASKS))
def test_hv15_rewritten_forward_is_bit_identical(fresh_cache, masks, image, trim):
    safe, why = cs.resolve(_CLS)
    assert why is None, why
    model = _tiny_model()
    kw = _inputs(masks, image)
    if trim:
        kw = _trimmed(model, kw)
    with torch.no_grad():
        want = _CLS.forward(model, **kw)[0]
        got = safe(model, **kw)[0]
    assert torch.isfinite(want).all()
    assert torch.equal(want, got)


def _stock_image_stream(image_embeds, states, primary_mask, batch_size):
    is_t2v = torch.all(image_embeds == 0)
    if is_t2v:
        states = states * 0.0
        mask = torch.zeros((batch_size, states.shape[1]), dtype = primary_mask.dtype)
    else:
        mask = torch.ones((batch_size, states.shape[1]), dtype = primary_mask.dtype)
    return states, mask


def _stock_merge(e1, m1, e2, m2, e3, m3):
    states, masks = [], []
    for t, tm, t2, tm2, im, imm in zip(e1, m1, e2, m2, e3, m3):
        states.append(
            torch.cat(
                [
                    im[imm],
                    t2[tm2],
                    t[tm],
                    im[~imm],
                    torch.zeros_like(t2[~tm2]),
                    torch.zeros_like(t[~tm]),
                ],
                dim = 0,
            )
        )
        masks.append(torch.cat([imm[imm], tm2[tm2], tm[tm], imm[~imm], tm2[~tm2], tm[~tm]], dim = 0))
    return torch.stack(states), torch.stack(masks)


@pytest.mark.parametrize("t2v", [True, False])
@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.bool, torch.float32])
def test_hv15_image_stream_matches_both_stock_arms_bitwise(t2v, mask_dtype):
    # inf and NaN make ``states * 0.0`` differ from zeros: the t2v arm must keep stock's NaNs and -0.0.
    states = torch.tensor([[[1.0, -2.0, float("inf")], [float("nan"), -0.0, 3.0]]]).repeat(2, 1, 1)
    image = torch.zeros(2, 2, 5) if t2v else torch.randn(2, 2, 5)
    primary = torch.ones(2, 4, dtype = mask_dtype)
    want = _stock_image_stream(image, states, primary, 2)
    got = cs.hv15_image_stream(image, states, primary, 2)
    assert torch.equal(want[0].view(torch.int32), got[0].view(torch.int32))
    assert got[1].dtype == want[1].dtype and torch.equal(got[1], want[1])


@pytest.mark.parametrize("image_len", [0, 3])
@pytest.mark.parametrize("image_valid", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hv15_merge_is_the_stock_permutation_bitwise(image_len, image_valid, dtype):
    gen = torch.Generator().manual_seed(0)
    m1 = torch.tensor([[1, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]).bool()
    m2 = torch.tensor([[0, 1, 0, 1, 1], [0, 0, 0, 0, 0]]).bool()
    m3 = torch.full((2, image_len), image_valid)
    e1 = torch.randn(2, 7, 6, generator = gen).to(dtype)
    e2 = torch.randn(2, 5, 6, generator = gen).to(dtype)
    e3 = torch.randn(2, image_len, 6, generator = gen).to(dtype)
    e2[0, 0, 0] = -0.0
    e1[0, 1, 1] = float("nan")
    want = _stock_merge(e1, m1, e2, m2, e3, m3)
    got = cs.hv15_merge_streams(e1, m1, e2, m2, e3, m3)
    assert got[0].dtype == want[0].dtype and got[1].dtype == torch.bool
    assert torch.equal(
        want[0].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
        got[0].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
    )
    assert torch.equal(want[1], got[1])


_HOST_READS = (
    "aten.index.Tensor",
    "aten.nonzero",
    "aten.masked_select",
    "aten.unique",
    "aten.item",
    "aten._local_scalar_dense",
    "aten.is_nonzero",
)


def _aten_ops(fn):
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: set = set()

        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            self.ops.add(str(func))
            return func(*args, **(kwargs or {}))

    with _Record() as rec:
        fn()
    return rec.ops


@pytest.mark.parametrize("image", ["t2v_zero", "i2v"])
def test_hv15_rewritten_forward_reads_nothing_on_the_host(fresh_cache, image):
    safe, why = cs.resolve(_CLS)
    assert why is None, why
    model = _tiny_model()
    kw = _inputs(_MASKS["b2_interleaved"], image)
    with torch.no_grad():
        stock = _aten_ops(lambda: _CLS.forward(model, **kw))
        ours = _aten_ops(lambda: safe(model, **kw))
    assert any(op.startswith(_HOST_READS) for op in stock), sorted(stock)
    assert not any(op.startswith(_HOST_READS) for op in ours), sorted(
        o for o in ours if o.startswith(_HOST_READS)
    )


def test_hv15_drifted_block_declines_with_the_reason(fresh_cache, monkeypatch):
    drifted = tuple(
        "image[~image_mask],  # padded image"
        if line == "image[~image_mask],  # invalid image"
        else line
        for line in cs._HV15_MERGE_BLOCK
    )
    monkeypatch.setattr(cs, "_HV15_MERGE_BLOCK", drifted)
    safe, why = cs.resolve(_CLS)
    assert safe is None
    assert "HunyuanVideo15Transformer3DModel forward is not capture-safe" in why
    assert "three-stream merge block changed" in why


def test_hv15_kill_switch_declines(fresh_cache, monkeypatch):
    monkeypatch.setenv(cs.CAPTURE_SAFE_ENV, "0")
    safe, why = cs.resolve(_CLS)
    assert safe is None and cs.CAPTURE_SAFE_ENV in why


_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _capture(
    fn,
    kw,
    pool = None,
):
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(2):
            fn(**static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, pool = pool):
        out = fn(**static)[0]
    return graph, static, out


@_cuda
@pytest.mark.parametrize("image", ["t2v_zero", "t2v_empty", "i2v"])
def test_hv15_rewritten_forward_captures_and_replays_bit_identical(fresh_cache, image):
    safe, why = cs.resolve(_CLS)
    assert why is None, why
    model = _tiny_model("cuda")
    kw = _inputs(_MASKS["b2_interleaved"], image, device = "cuda")
    graph, static, out = _capture(lambda **k: safe(model, **k), kw)
    for seed in (2, 3):
        fresh = _inputs(_MASKS["b2_interleaved"], image, device = "cuda", seed = seed)
        for k, v in fresh.items():
            if torch.is_tensor(v):
                static[k].copy_(v)
        graph.replay()
        torch.cuda.synchronize()
        with torch.no_grad():
            eager = safe(model, **fresh)[0]
            stock = _CLS.forward(model, **fresh)[0]
        assert torch.equal(out, eager)
        assert torch.equal(eager, stock)
    del graph


@_cuda
def test_hv15_stock_forward_does_not_capture(fresh_cache):
    # Last in the file: a failed capture can leave the CUDA context unusable for later captures.
    model = _tiny_model("cuda")
    kw = _inputs(_MASKS["b2_interleaved"], "t2v_zero", device = "cuda")
    stream = torch.cuda.current_stream()
    pool = torch.cuda.graph_pool_handle()
    with pytest.raises(RuntimeError):
        _capture(lambda **k: _CLS.forward(model, **k), kw, pool = pool)
    # capture_end raises before leaving the capture stream, so restore the stream for later tests.
    torch.cuda.set_stream(stream)
    # torch leaves the CUDA generators in capture mode after an invalidated capture; later tests draw from them
    cg._heal_generators()
    # Also stop allocating into the dead capture's pool, or later empty_cache frees nothing.
    cg._abandon_capture_pool(pool)
