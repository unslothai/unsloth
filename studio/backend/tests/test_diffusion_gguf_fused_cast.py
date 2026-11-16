# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUFLinear's fp16-dequantised types under a bf16 compute dtype: the compiled dequant's weight is cast in its own
buffer, bit-identical to the stock ``.to``, and every other case keeps the stock forward. Runs on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_compile as gc  # noqa: E402

T = gguf.GGMLQuantizationType
needs_fused = pytest.mark.skipif(
    not gc._torch_supports_fused_q8_0(torch.__version__),
    reason = "the GGUF forward patch needs torch >= 2.7",
)
# byte offsets of each block's fp16 scales (diffusers' split_block_dims layouts): finite, realistic scales
_F16_FIELDS = {
    T.Q4_0: [0],
    T.Q4_1: [0, 2],
    T.Q5_0: [0],
    T.Q5_1: [0, 2],
    T.Q2_K: [80, 82],
    T.Q3_K: [108],
    T.Q4_K: [0, 2],
    T.Q5_K: [0, 2],
    T.Q6_K: [208],
    T.IQ4_XS: [0],
}
FP16_TYPES = [T.Q4_K, T.Q5_K, T.Q6_K, T.Q3_K, T.Q2_K, T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1]


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    from core.inference import diffusion_compile_config as cc

    from core.inference import diffusion_gguf_exact as ex

    monkeypatch.delenv("UNSLOTH_DIFFUSION_GGUF_FUSED_CAST", raising = False)
    cc._reset_for_tests()
    gc.uninstall_all()
    # diffusers' fp16 functions: the path UNSLOTH_DIFFUSION_GGUF_EXACT=0 keeps
    ex.uninstall_exact_dequant()
    torch._dynamo.reset()
    with torch._inductor.config.patch(emulate_precision_casts = False):
        yield
    gc.uninstall_all()
    ex.uninstall_exact_dequant()
    cc._reset_for_tests()


def _raw(
    qtype,
    rows,
    cols,
    seed = 0,
    adversarial = False,
):
    rng = np.random.default_rng(seed)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    if qtype == T.BF16:
        w = (rng.standard_normal((rows, cols)) * 0.05).astype(np.float32).view(np.uint32)
        return torch.from_numpy((w >> 16).astype(np.uint16).view(np.uint8).reshape(rows, -1).copy())
    raw = rng.integers(0, 256, size = (rows, cols // block, size), dtype = np.uint8)
    if not adversarial:
        for off in _F16_FIELDS[qtype]:
            scale = (np.abs(rng.standard_normal((rows, cols // block))) * 0.01 + 1e-4).astype(
                np.float16
            )
            raw[:, :, off : off + 2] = scale.view(np.uint8).reshape(rows, cols // block, 2)
    return torch.from_numpy(raw.reshape(rows, -1).copy())


def _linear(
    qtype,
    rows,
    cols,
    dtype,
    seed = 0,
    adversarial = False,
):
    lin = gguf_utils.GGUFLinear(cols, rows, bias = True, compute_dtype = dtype)
    lin.weight = gguf_utils.GGUFParameter(
        _raw(qtype, rows, cols, seed, adversarial), quant_type = qtype
    )
    lin.bias = torch.nn.Parameter(torch.randn(rows, generator = torch.Generator().manual_seed(1)))
    return lin


def _bits(t):
    return t.view(
        {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[
            t.dtype
        ]
    )


def _watch(monkeypatch):
    """Record the dequant's output buffer and the weight F.linear receives, per forward."""
    import weakref

    seen = {"dequant": [], "linear": [], "dequant_alive_in_linear": []}
    dequant = gguf_utils.dequantize_gguf_tensor
    linear = torch.nn.functional.linear
    refs = []

    def recorded_dequant(t):
        out = dequant(t)
        seen["dequant"].append(out.data_ptr())
        refs.append(weakref.ref(out))
        return out

    def recorded_linear(
        x,
        w,
        b = None,
    ):
        seen["linear"].append(w.data_ptr())
        seen["dequant_alive_in_linear"].append([r() is not None for r in refs])
        return linear(x, w, b)

    monkeypatch.setattr(gguf_utils, "dequantize_gguf_tensor", recorded_dequant)
    monkeypatch.setattr(torch.nn.functional, "linear", recorded_linear)
    return seen


@needs_fused
@pytest.mark.parametrize("adversarial", [False, True], ids = ["scaled", "raw_bytes"])
@pytest.mark.parametrize("qtype", FP16_TYPES, ids = lambda q: q.name)
def test_an_fp16_dequant_is_cast_in_its_own_buffer_bit_for_bit(qtype, adversarial, monkeypatch):
    # raw bytes: fp16 overflow, subnormals and NaN scales; 1023 rows: an odd count through the same dynamic graph
    stock_forward = gguf_utils.GGUFLinear.forward_native
    assert gc.install_compiled_dequant() is True
    for rows in (256, 1023):
        lin = _linear(qtype, rows, 512, torch.bfloat16, seed = rows, adversarial = adversarial)
        x = torch.randn(3, 512, generator = torch.Generator().manual_seed(2)).to(torch.bfloat16)
        expected = stock_forward(
            lin, x
        )  # today: the compiled dequant, then .to(bf16) into a second buffer
        seen = _watch(monkeypatch)
        got = lin(x)
        assert torch.equal(_bits(got), _bits(expected))
        # one dense weight: the linear reads the dequant's own buffer, not a copy of it
        assert len(seen["dequant"]) == 1 and seen["linear"] == seen["dequant"]
        monkeypatch.undo()
        monkeypatch.delenv("UNSLOTH_DIFFUSION_GGUF_FUSED_CAST", raising = False)


@needs_fused
@pytest.mark.parametrize("qtype", [T.IQ4_XS, T.BF16], ids = lambda q: q.name)
def test_an_fp32_dequant_keeps_the_stock_cast(qtype, monkeypatch):
    stock_forward = gguf_utils.GGUFLinear.forward_native
    assert gc.install_compiled_dequant() is True
    lin = _linear(qtype, 64, 256, torch.bfloat16)
    x = torch.randn(2, 256).to(torch.bfloat16)
    expected = stock_forward(lin, x)
    seen = _watch(monkeypatch)
    assert torch.equal(_bits(lin(x)), _bits(expected))
    assert seen["linear"] != seen["dequant"]  # a 4-byte weight cannot become a 2-byte one in place
    # and, as in the stock forward, the fp32 dequant is gone before the matmul runs
    assert seen["dequant_alive_in_linear"] == [[False]]
    monkeypatch.undo()


@needs_fused
def test_fp16_compute_and_the_kill_switch_take_the_stock_forward(monkeypatch):
    assert gc.install_compiled_dequant() is True
    lin = _linear(T.Q4_K, 64, 256, torch.float16)
    seen = _watch(monkeypatch)
    lin(torch.randn(2, 256).to(torch.float16))
    assert (
        seen["linear"] == seen["dequant"]
    )  # the stock forward: an fp16 weight's .to(fp16) is itself
    monkeypatch.undo()

    gc.uninstall_all()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_FUSED_CAST", "0")
    assert gc.install_compiled_dequant() is True and not gc.is_fused_cast_installed()
    lin = _linear(T.Q4_K, 64, 256, torch.bfloat16)
    seen = _watch(monkeypatch)
    lin(torch.randn(2, 256).to(torch.bfloat16))
    assert seen["linear"] != seen["dequant"]  # the stock cast into a second buffer
    monkeypatch.undo()


def test_cast_in_place_matches_to_bit_for_bit_including_nan():
    bits = np.random.default_rng(3).integers(0, 1 << 16, size = 4096, dtype = np.uint16)
    half = torch.from_numpy(bits.view(np.float16).copy())
    want = half.to(torch.bfloat16)
    got = gc._cast_in_place(half.clone(), torch.bfloat16)
    assert torch.equal(_bits(got), _bits(want))
