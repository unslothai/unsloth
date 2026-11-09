# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp-exact GGUF dequant: fp32 bit-identical to ggml-quants.c (vendored digests of its output, no C compiler
needed) and to gguf-py, NaN / inf / subnormal / -0 scale bits included, one rounding to the compute dtype, eager and
compiled (no graph breaks, one dynamic graph per type), install / uninstall / kill switch, and the GGUFLinear
forward. Runs on CPU."""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_compile as gc  # noqa: E402
from core.inference import diffusion_gguf_exact as ex  # noqa: E402

T = gguf.GGMLQuantizationType
TYPES = list(ex._FUNCTIONS)

# byte offsets of each block's fp16 fields (ggml-common.h block structs)
F16_OFFSETS = {
    "Q4_0": (0,), "Q4_1": (0, 2), "Q5_0": (0,), "Q5_1": (0, 2), "Q8_0": (0,),
    "Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "IQ4_XS": (0,),
}  # fmt: skip
# +inf, -inf, qNaN, sNaN, -qNaN, -0, +0, smallest subnormal, largest subnormal, max, -max, smallest normal
SPECIAL = (
    0x7C00,
    0xFC00,
    0x7E00,
    0x7C01,
    0xFE00,
    0x8000,
    0x0000,
    0x0001,
    0x03FF,
    0x7BFF,
    0xFBFF,
    0x0400,
)
ORACLE_BLOCKS = {name: 37 for name in F16_OFFSETS}  # odd: no even-count assumption
# ggml-org/llama.cpp ggml-quants.c dequantize_row_* (f477f79, -O0, no FMA) on oracle_blocks(name, 37): sha256 of the
# fp32 output, NaN canonicalised. Regenerate with scripts/ggml_parity/make_oracle.py against a compiled ggml-quants.c.
GGML_DIGESTS = {
    "Q4_0": "11f2f408ad3ecc40298fc34e7d36e010e5a7f3338b2e49c3d67d1c1ea801c95f",
    "Q4_1": "ea2dc4832791b6bb8c0a1800d75a153840f29b89064a6909109786ec0a6538b6",
    "Q5_0": "84de0e848a5d8f7e62b9eadc0d9ee4f254e5f426643723da9ade8ed8a1e3ede9",
    "Q5_1": "9cf75e735c74127787ffe0c1d5b02189c9aa5ac9339d3714693c7ed57e9e2e4b",
    "Q8_0": "f46e3b0ed5965df250f13728201a942c5f06de257a2a7ae96d20c7fc1545d729",
    "Q2_K": "d5437758841860ef8b258b2c7d4fc223bc8421b371b0e0c22f0da3b6a10e466f",
    "Q3_K": "7a2776ee811eaa09ed093f1ebc856869783bffc16c4999ee95cf9735194edbe4",
    "Q4_K": "5cae66ba58e1344b4c7a8ff706a5f8ce7e7d9bf346f6a3f20dc341e1e1f9d567",
    "Q5_K": "28591e1ba7c6aa0f852e8107ad503b0e6b4c2e3a7b9e4718c800505301a9c03c",
    "Q6_K": "94229b879c82509f6cdefa56bdd8852fefd787fe9946aa36f03997f1c57bb680",
    "IQ4_XS": "510c1a39bb8c42cba718de1238d99df8ef6abaa6d8a84a0a6ed89f32f1fa221d",
}


def oracle_blocks(name: str, n: int):
    """Deterministic blocks: sha256 counter bytes; the fp16 fields cycle through SPECIAL for the first 24 blocks,
    then take normal values of either sign with exponents 2^-6 .. 2^-1."""
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    stream = b"".join(
        hashlib.sha256(f"{name}:{i}".encode()).digest() for i in range((n * sz) // 32 + 1)
    )
    raw = np.frombuffer(stream[: n * sz], dtype = np.uint8).reshape(n, sz).copy()
    for k, off in enumerate(F16_OFFSETS[name]):
        u = raw[:, off].astype(np.uint16) | (raw[:, off + 1].astype(np.uint16) << 8)
        normal = (u & 0x83FF) | ((9 + (u >> 10) % 6) << 10).astype(np.uint16)
        for b in range(n):
            bits = SPECIAL[(b + 5 * k) % len(SPECIAL)] if b < 2 * len(SPECIAL) else int(normal[b])
            raw[b, off], raw[b, off + 1] = bits & 0xFF, bits >> 8
    return raw


def canonical_digest(values) -> str:
    bits = np.ascontiguousarray(values, dtype = np.float32).reshape(-1).view(np.uint32).copy()
    bits[np.isnan(np.asarray(values, dtype = np.float32).reshape(-1))] = 0x7FC00000
    return hashlib.sha256(bits.tobytes()).hexdigest()


def _same(got, ref) -> bool:
    """Equal bits wherever not NaN, NaN at the same places (fp32 / bf16 / fp16)."""
    got = got.reshape(-1)
    ref = ref.reshape(-1)
    int_dtype = torch.int32 if got.dtype == torch.float32 else torch.int16
    nan = torch.isnan(ref)
    return bool(
        got.dtype == ref.dtype
        and torch.equal(torch.isnan(got), nan)
        and torch.equal(got[~nan].view(int_dtype), ref[~nan].view(int_dtype))
    )


def _random_blocks(
    name: str,
    n: int,
    seed: int = 0,
    special: bool = True,
):
    rng = np.random.default_rng(seed)
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    raw = rng.integers(0, 256, size = (n, sz), dtype = np.uint8)
    for off in F16_OFFSETS[name]:
        v = (rng.standard_normal(n) * 0.02).astype(np.float16).view(np.uint16)
        if special:
            k = min(n // 4, 48)
            v[rng.choice(n, size = k, replace = False)] = rng.choice(SPECIAL, size = k)
        raw[:, off : off + 2] = v.view(np.uint8).reshape(-1, 2)
    return raw


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    from core.inference import diffusion_compile_config as cc

    monkeypatch.delenv(ex.EXACT_ENV, raising = False)
    cc._reset_for_tests()
    gc.uninstall_all()
    ex.uninstall_exact_dequant()
    torch._dynamo.reset()
    yield
    gc.uninstall_all()
    ex.uninstall_exact_dequant()
    cc._reset_for_tests()


def test_digest_table_covers_every_type():
    assert set(GGML_DIGESTS) == set(TYPES) == set(F16_OFFSETS)


@pytest.mark.parametrize("name", TYPES)
def test_fp32_matches_the_vendored_ggml_output(name):
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    raw = oracle_blocks(name, ORACLE_BLOCKS[name])
    out = ex._FUNCTIONS[name](torch.from_numpy(raw), bs, sz)
    assert out.dtype == torch.float32 and out.shape == (raw.shape[0], bs)
    assert canonical_digest(out.numpy()) == GGML_DIGESTS[name]
    assert np.isnan(out.numpy()).any() and np.isinf(out.numpy()).any()


@pytest.mark.parametrize("name", TYPES)
@pytest.mark.parametrize("n", [1, 7, 513])
def test_fp32_is_bit_identical_to_gguf_py(name, n):
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    raw = _random_blocks(name, n, seed = n)
    with np.errstate(all = "ignore"):
        ref = torch.from_numpy(
            np.ascontiguousarray(gguf.quants.dequantize(raw, getattr(T, name)), dtype = np.float32)
        )
    assert _same(ex._FUNCTIONS[name](torch.from_numpy(raw), bs, sz), ref.reshape(n, bs))


@pytest.mark.parametrize("name", TYPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_compute_dtype_is_one_rounding_of_the_fp32_result(name, dtype):
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    raw = torch.from_numpy(_random_blocks(name, 301, seed = 3))
    fp32 = ex._FUNCTIONS[name](raw, bs, sz)
    assert _same(ex._FUNCTIONS[name](raw, bs, sz, dtype), fp32.to(dtype))


def test_diffusers_fp16_functions_are_not_exact():
    # if diffusers' own function were already exact, this module would be redundant
    bs, sz = gguf.GGML_QUANT_SIZES[T.Q4_K]
    raw = torch.from_numpy(_random_blocks("Q4_K", 512, seed = 5))
    stock = gguf_utils.dequantize_functions[T.Q4_K](raw, bs, sz)
    ours = ex.dequantize_blocks_Q4_K(raw, bs, sz)
    fin = torch.isfinite(ours) & torch.isfinite(stock.float())
    assert not torch.equal(stock.to(torch.bfloat16)[fin], ours.to(torch.bfloat16)[fin])


def _gguf_py(raw, qtype):
    return torch.from_numpy(
        np.ascontiguousarray(gguf.quants.dequantize(raw.numpy(), qtype), np.float32)
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_q8_0_bf16_is_not_rounded_through_fp16(compiled):
    # d = 1371 / 4096, q = 3: fp16 d * q = 1 + 2**-8 (a bf16 tie, to 1); ggml's fp32 1 + 2**-8 + 2**-12 -> 1 + 2**-7
    raw = torch.zeros(1, 34, dtype = torch.uint8)
    raw[0, :2] = torch.tensor([np.float16(1371 / 4096)]).view(torch.uint8)
    raw[0, 2:] = 3
    fn = (
        torch.compile(ex.dequantize_blocks_Q8_0, fullgraph = True)
        if compiled
        else ex.dequantize_blocks_Q8_0
    )
    got = fn(raw, 32, 34, torch.bfloat16)
    assert torch.equal(got, torch.full_like(got, 1 + 2**-7))
    assert torch.equal(
        got.float(), _gguf_py(raw, T.Q8_0).to(torch.bfloat16).float().reshape(got.shape)
    )
    stock = gguf_utils.dequantize_functions[T.Q8_0]
    if stock is not ex.dequantize_blocks_Q8_0:
        assert float(stock(raw, 32, 34).to(torch.bfloat16)[0, 0]) == 1.0


@pytest.mark.parametrize("compiled", [False, True])
def test_iq4_xs_large_scale_does_not_overflow_in_fp16(compiled):
    # d = 4096, ls - 32 = -32: ggml's fp32 dl = -131072 (fp16 would be -inf); * kvalue -127 is finite in fp32
    bs, sz = gguf.GGML_QUANT_SIZES[T.IQ4_XS]
    raw = torch.zeros(1, sz, dtype = torch.uint8)
    raw[0, :2] = torch.tensor([np.float16(4096)]).view(torch.uint8)
    fn = (
        torch.compile(ex.dequantize_blocks_IQ4_XS, fullgraph = True)
        if compiled
        else ex.dequantize_blocks_IQ4_XS
    )
    got = fn(raw, bs, sz).reshape(-1)
    assert torch.equal(got, _gguf_py(raw, T.IQ4_XS).reshape(-1))
    assert torch.equal(got, torch.full_like(got, 4096.0 * 32 * 127))


@pytest.mark.parametrize("name", TYPES)
@pytest.mark.parametrize("rows", [1, 3, 5])
def test_dequantize_cast_keeps_rows_of_whole_blocks(name, rows):
    qtype = getattr(T, name)
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    raw = torch.from_numpy(_random_blocks(name, rows * 3, seed = rows)).reshape(rows, 3 * sz)
    ex.install_exact_dequant()
    dense = ex.dequantize_cast(raw, qtype, torch.bfloat16)
    assert dense.shape == (rows, 3 * bs) and dense.dtype == torch.bfloat16
    ref = gguf_utils.dequantize_gguf_tensor(gguf_utils.GGUFParameter(raw, quant_type = qtype)).to(
        torch.bfloat16
    )
    assert _same(dense, ref)
    assert ex.row_aligned(raw, qtype) and not ex.row_aligned(raw[:, 1:], qtype)


def test_install_replaces_only_the_fp16_types_and_uninstall_restores_them():
    table = gguf_utils.dequantize_functions
    before = dict(table)
    supported = list(gguf_utils.SUPPORTED_GGUF_QUANT_TYPES)
    assert ex.install_exact_dequant() == TYPES
    assert ex.install_exact_dequant() == []
    for name in TYPES:
        assert table[getattr(T, name)] is ex._FUNCTIONS[name]
        assert ex.exact_function(getattr(T, name)) is ex._FUNCTIONS[name]
    for qtype in (T.IQ4_NL, T.BF16):
        assert table[qtype] is before[qtype] and ex.exact_function(qtype) is None
    assert gguf_utils.SUPPORTED_GGUF_QUANT_TYPES == supported
    assert gguf_utils.GGUFLinear.forward_native is ex._exact_forward
    ex.uninstall_exact_dequant()
    ex.uninstall_exact_dequant()
    assert table == before
    assert ex.exact_function(T.Q4_K) is None
    assert gguf_utils.GGUFLinear.forward_native is ex._STATE["forward"]
    assert ex._STATE["forward"].__module__ == gguf_utils.__name__


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_kill_switch_keeps_diffusers_functions(monkeypatch, value):
    monkeypatch.setenv(ex.EXACT_ENV, value)
    before = dict(gguf_utils.dequantize_functions)
    forward = gguf_utils.GGUFLinear.forward_native
    assert ex.install_exact_dequant() == []
    assert gguf_utils.dequantize_functions == before
    assert gguf_utils.GGUFLinear.forward_native is forward


def test_uninstall_leaves_a_later_replacement_alone():
    ex.install_exact_dequant()
    other = lambda *a, **k: None  # noqa: E731
    gguf_utils.dequantize_functions[T.Q8_0] = other
    try:
        ex.uninstall_exact_dequant()
        assert gguf_utils.dequantize_functions[T.Q8_0] is other
    finally:
        gguf_utils.dequantize_functions[T.Q8_0] = gguf_utils.dequantize_blocks_Q8_0


@pytest.mark.parametrize("name", TYPES)
def test_compiled_has_no_graph_breaks_and_one_graph_for_every_weight(name):
    # as the patched GGUFLinear forward runs it: one compile per type, only the block count dynamic
    from torch._dynamo.testing import CompileCounter

    qtype = getattr(T, name)
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    counter = CompileCounter()
    fn = torch.compile(ex._FUNCTIONS[name], backend = counter, fullgraph = True)
    for rows, n in ((2, 8), (13, 13), (4, 64), (7, 301)):
        raw = torch.from_numpy(_random_blocks(name, rows * n, seed = n)).reshape(rows, n * sz)
        got = ex.dequantize_cast(raw, qtype, torch.bfloat16, fn)
        assert got.shape == (rows, n * bs)
        assert _same(
            got,
            ex._FUNCTIONS[name](raw.reshape(-1, sz), bs, sz).to(torch.bfloat16).reshape(rows, -1),
        )
    assert counter.frame_count == 1


@pytest.mark.parametrize("name", ["Q8_0", "Q4_K", "Q6_K", "Q5_1", "Q3_K", "Q2_K"])
@pytest.mark.parametrize(
    "emulate", [False, True], ids = ["inductor_default", "emulate_precision_casts"]
)
def test_inductor_dequant_cast_is_bit_identical(name, emulate):
    bs, sz = gguf.GGML_QUANT_SIZES[getattr(T, name)]
    with torch._inductor.config.patch(emulate_precision_casts = emulate):
        fn = torch.compile(ex._FUNCTIONS[name])
        for n in (16, 37):
            raw = torch.from_numpy(_random_blocks(name, n, seed = n))
            fp32 = ex._FUNCTIONS[name](raw, bs, sz)
            assert _same(ex.dequantize_cast(raw, getattr(T, name), torch.float32, fn), fp32)
            assert _same(
                ex.dequantize_cast(raw, getattr(T, name), torch.bfloat16, fn),
                fp32.to(torch.bfloat16),
            )


@pytest.mark.parametrize("name", ["Q2_K", "Q4_K", "Q5_K"])
def test_a_zero_code_under_a_negative_scale_and_zero_min_is_negative_zero(name):
    # ggml: d1 * 0 - (+0) = -0 when d1 < 0. A lowering of ``-m + d1 * q`` as ``(0 - m) + ...`` gives +0.
    qtype = getattr(T, name)
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.zeros((4, sz), dtype = np.uint8)
    d_off, dmin_off = F16_OFFSETS[name]
    raw[:, d_off : d_off + 2] = np.array([0xBC00], np.uint16).view(np.uint8)  # d = -1
    raw[:, dmin_off : dmin_off + 2] = np.array([0x3C00], np.uint16).view(np.uint8)  # dmin = +1
    if name == "Q2_K":
        raw[:, :16] = 0x01  # scale 1, min 0
    else:
        raw[:, 4:8] = 0x01  # sc = 1, m = 0 for the first four groups
    with np.errstate(all = "ignore"):
        ref = torch.from_numpy(
            np.ascontiguousarray(gguf.quants.dequantize(raw, qtype), np.float32)
        ).reshape(4, bs)
    assert (torch.signbit(ref) & (ref == 0)).any()
    t = torch.from_numpy(raw)
    assert _same(ex._FUNCTIONS[name](t, bs, sz), ref)
    assert _same(
        ex.dequantize_cast(t, qtype, torch.float32, torch.compile(ex._FUNCTIONS[name])), ref
    )


needs_fused = pytest.mark.skipif(
    not gc._torch_supports_fused_q8_0(torch.__version__),
    reason = "the patched forward needs torch >= 2.7",
)


def _linear(qtype, rows, cols, dtype):
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    raw = torch.from_numpy(_random_blocks(qtype.name, rows * cols // bs, seed = 11, special = False))
    lin = gguf_utils.GGUFLinear(cols, rows, bias = False, compute_dtype = dtype)
    lin.weight = gguf_utils.GGUFParameter(raw.reshape(rows, cols // bs * sz), quant_type = qtype)
    return lin


@needs_fused
@pytest.mark.parametrize("name", ["Q8_0", "Q4_K", "Q6_K"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_patched_forward_uses_the_exact_fused_dequant(name, dtype, monkeypatch):
    qtype = getattr(T, name)
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    ex.install_exact_dequant()
    assert gc.install_compiled_dequant()
    calls = []
    real = ex.dequantize_cast
    monkeypatch.setattr(ex, "dequantize_cast", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    lin = _linear(qtype, 6, 512, dtype)
    raw = lin.weight.as_tensor()
    dense = ex._FUNCTIONS[name](raw.reshape(-1, sz), bs, sz).reshape(6, 512).to(dtype)
    x = torch.randn(3, 512).to(dtype)
    out = lin(x)
    assert calls == [qtype]
    assert torch.equal(out, torch.nn.functional.linear(x, dense))


@needs_fused
def test_kill_switch_restores_the_stock_q8_0_forward(monkeypatch):
    monkeypatch.setenv(ex.EXACT_ENV, "0")
    assert ex.install_exact_dequant() == []
    assert gc.install_compiled_dequant()
    calls = []
    monkeypatch.setattr(ex, "dequantize_cast", lambda *a, **k: calls.append(1))
    lin = _linear(T.Q8_0, 4, 64, torch.bfloat16)
    lin(torch.ones(1, 64, dtype = torch.bfloat16))
    assert calls == []


@pytest.mark.parametrize("name", ["Q8_0", "Q4_K", "Q6_K", "Q2_K"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eager_forward_dequantises_straight_to_the_compute_dtype(name, dtype, monkeypatch):
    qtype = getattr(T, name)
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    ex.install_exact_dequant()
    calls = []
    real = ex.dequantize_cast
    monkeypatch.setattr(ex, "dequantize_cast", lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    lin = _linear(qtype, 6, 512, dtype)
    raw = lin.weight.as_tensor()
    x = torch.randn(3, 512).to(dtype)
    out = lin(x)
    assert calls == [dtype]
    dense = ex._FUNCTIONS[name](raw.reshape(-1, sz), bs, sz).reshape(6, 512).to(dtype)
    assert torch.equal(out, torch.nn.functional.linear(x, dense))


def test_eager_forward_leaves_other_types_to_diffusers(monkeypatch):
    ex.install_exact_dequant()
    calls = []
    monkeypatch.setattr(ex, "dequantize_cast", lambda *a, **k: calls.append(1))
    raw = torch.randn(4, 64).to(torch.bfloat16).view(torch.uint8)
    lin = gguf_utils.GGUFLinear(64, 4, bias = False, compute_dtype = torch.bfloat16)
    lin.weight = gguf_utils.GGUFParameter(raw, quant_type = T.BF16)
    x = torch.randn(2, 64).to(torch.bfloat16)
    assert torch.equal(lin(x), torch.nn.functional.linear(x, raw.view(torch.bfloat16)))
    assert calls == []


@pytest.mark.parametrize("name", ["Q8_0", "Q4_K", "Q6_K"])
def test_a_compiled_block_traces_the_eager_forward_without_graph_breaks(name):
    # the max tier compiles whole blocks around GGUFLinear: the forward and dequant must trace in one graph
    from torch._dynamo.testing import CompileCounter

    ex.install_exact_dequant()
    lin = _linear(getattr(T, name), 6, 512, torch.bfloat16)
    x = torch.randn(3, 512).to(torch.bfloat16)
    counter = CompileCounter()
    out = torch.compile(lin, backend = counter, fullgraph = True)(x)
    assert counter.frame_count == 1
    assert torch.equal(out, lin(x))


def test_install_order_with_the_compiled_forward_does_not_matter():
    stock = gguf_utils.GGUFLinear.forward_native
    assert ex._stock_forward(gguf_utils.GGUFLinear) is stock
    # compiled forward first: its wrapper stays, diffusers' own forward is what the exact path falls back to
    assert gc.install_compiled_dequant()
    wrapper = gguf_utils.GGUFLinear.forward_native
    assert ex.install_exact_dequant() == TYPES
    assert gguf_utils.GGUFLinear.forward_native is wrapper
    assert ex.stock_forward() is stock
    gc.uninstall_all()
    assert gguf_utils.GGUFLinear.forward_native is stock
    # exact first, compiled on top, then out-of-order teardown: still back to diffusers' forward
    ex.install_exact_dequant()
    assert gguf_utils.GGUFLinear.forward_native is ex._exact_forward
    assert gc.install_compiled_dequant()
    ex.uninstall_exact_dequant()
    gc.uninstall_all()
    ex.uninstall_exact_dequant()
    assert gguf_utils.GGUFLinear.forward_native is stock


@needs_fused
def test_fused_cast_switch_off_runs_the_compiled_stock_dequant_once_rounded(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_FUSED_CAST", "0")
    ex.install_exact_dequant()
    assert gc.install_compiled_dequant()
    calls = []
    monkeypatch.setattr(ex, "dequantize_cast", lambda *a, **k: calls.append(1))
    qtype = T.Q4_K
    bs, sz = gguf.GGML_QUANT_SIZES[qtype]
    lin = _linear(qtype, 6, 512, torch.bfloat16)
    raw = lin.weight.as_tensor()
    x = torch.randn(3, 512).to(torch.bfloat16)
    dense = ex._FUNCTIONS["Q4_K"](raw.reshape(-1, sz), bs, sz).reshape(6, 512).to(torch.bfloat16)
    assert torch.equal(lin(x), torch.nn.functional.linear(x, dense))
    assert calls == []
