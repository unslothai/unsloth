# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The low-bit IQ types diffusers' GGUF table lacks: bit-identical to gguf-py's dequant, loadable, reversible."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_lowbit as lb  # noqa: E402

T = gguf.GGMLQuantizationType
TYPES = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"]


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    from core.inference import diffusion_gguf_exact as ex

    monkeypatch.delenv(lb.LOWBIT_IQ_ENV, raising = False)
    lb.uninstall_lowbit_iq()
    ex.uninstall_exact_dequant()
    yield
    lb.uninstall_lowbit_iq()
    # the loaders under test also install the exact dequant
    ex.uninstall_exact_dequant()


def _blocks(
    name,
    rows,
    per_row = 2,
    seed = 0,
):
    """Random blocks with a finite fp16 super-scale (IQ1_M keeps its scale in the top nibbles of its scale words)."""
    qtype = getattr(T, name)
    _, size = gguf.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size = (rows * per_row, size), dtype = np.uint8)
    scale = (
        (np.abs(rng.standard_normal(rows * per_row)) * 0.01 + 1e-3)
        .astype(np.float16)
        .view(np.uint16)
    )
    if name == "IQ1_M":
        words = (
            raw[:, -8:].copy().view(np.uint16)
        )  # 4 words; nibble k of the scale in the top nibble of word k
        for k in range(4):
            words[:, k] = (words[:, k] & 0x0FFF) | (((scale >> (4 * k)) & 0xF) << 12)
        raw[:, -8:] = words.view(np.uint8)
    else:
        raw[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    return raw.reshape(rows, -1)


@pytest.mark.parametrize("rows", [1023, 1024])
@pytest.mark.parametrize("name", TYPES)
def test_dequant_is_bit_identical_to_gguf_py(name, rows):
    qtype = getattr(T, name)
    raw = _blocks(name, rows, seed = rows)
    want = gguf.quants.dequantize(raw, qtype).astype(np.float32)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    blocks = torch.from_numpy(raw).reshape(-1, size)
    got = lb._FUNCTIONS[name](blocks, block, size).reshape(rows, -1)
    assert got.dtype == torch.float32 and tuple(got.shape) == want.shape
    assert np.array_equal(got.numpy().view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("name", TYPES)
def test_raw_random_blocks_match_gguf_py_too(name):
    # any scale bits, NaN and inf included: same values, same NaN positions
    qtype = getattr(T, name)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = np.random.default_rng(9).integers(0, 256, size = (64, size), dtype = np.uint8)
    with np.errstate(all = "ignore"):
        want = gguf.quants.dequantize(raw, qtype).astype(np.float32)
    got = lb._FUNCTIONS[name](torch.from_numpy(raw), block, size).numpy()
    nan = np.isnan(want)
    assert np.array_equal(nan, np.isnan(got))
    assert np.array_equal(got[~nan].view(np.uint32), want[~nan].view(np.uint32))


def _write_gguf(path, tensors):
    writer = gguf.GGUFWriter(str(path), "flux")
    for name, (qtype, raw, shape) in tensors.items():
        writer.add_tensor(name, raw, raw_dtype = qtype)  # raw: the byte shape
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_a_gguf_with_low_bit_iq_tensors_loads_and_runs(tmp_path):
    from diffusers.models.model_loading_utils import load_gguf_checkpoint

    rows, cols = 64, 512
    tensors = {}
    for i, name in enumerate(TYPES):
        raw = _blocks(name, rows, per_row = cols // 256, seed = i)
        tensors[f"blocks.{i}.weight"] = (getattr(T, name), raw, (rows, cols))
    path = tmp_path / "lowbit.gguf"
    _write_gguf(path, tensors)
    with pytest.raises(ValueError, match = "unsupported"):
        load_gguf_checkpoint(str(path))  # the stock table refuses the file

    assert lb.install_lowbit_iq() == TYPES
    state = load_gguf_checkpoint(str(path))
    x = torch.randn(3, cols, generator = torch.Generator().manual_seed(1))
    for i, name in enumerate(TYPES):
        weight = state[f"blocks.{i}.weight"]
        assert isinstance(weight, gguf_utils.GGUFParameter) and weight.quant_type == getattr(
            T, name
        )
        lin = gguf_utils.GGUFLinear(cols, rows, bias = False, compute_dtype = torch.float32)
        lin.weight = weight
        dense = torch.from_numpy(
            gguf.quants.dequantize(tensors[f"blocks.{i}.weight"][1], getattr(T, name))
        )
        assert torch.equal(lin.forward_native(x), torch.nn.functional.linear(x, dense.float()))


def test_kill_switch_install_idempotence_and_uninstall(monkeypatch):
    before = dict(gguf_utils.dequantize_functions), list(gguf_utils.SUPPORTED_GGUF_QUANT_TYPES)
    monkeypatch.setenv(lb.LOWBIT_IQ_ENV, "0")
    assert lb.install_lowbit_iq() == [] and T.IQ2_XXS not in gguf_utils.dequantize_functions
    monkeypatch.delenv(lb.LOWBIT_IQ_ENV)
    assert lb.install_lowbit_iq() == TYPES
    assert lb.install_lowbit_iq() == []
    assert all(getattr(T, n) in gguf_utils.SUPPORTED_GGUF_QUANT_TYPES for n in TYPES)
    assert gguf_utils.dequantize_functions[T.IQ4_XS] is before[0][T.IQ4_XS]
    lb.uninstall_lowbit_iq()
    assert dict(gguf_utils.dequantize_functions) == before[0]
    assert list(gguf_utils.SUPPORTED_GGUF_QUANT_TYPES) == before[1]


def test_compiled_dequant_matches_eager():
    # Studio compiles dequantize_gguf_tensor on the default tier; the new types must trace and keep their bits
    lb.install_lowbit_iq()
    compiled = torch.compile(gguf_utils.dequantize_gguf_tensor, dynamic = True)
    for name in ("IQ2_XXS", "IQ3_S", "IQ1_M"):
        raw = torch.from_numpy(_blocks(name, 32, seed = 4))
        weight = gguf_utils.GGUFParameter(raw, quant_type = getattr(T, name))
        eager = gguf_utils.dequantize_gguf_tensor(weight)
        got = compiled(weight)
        assert torch.equal(got.view(torch.int32), eager.view(torch.int32))


def _oracle(name, raw):
    qtype = getattr(T, name)
    return torch.from_numpy(gguf.quants.dequantize(raw.numpy(), qtype).astype(np.float32))


@pytest.mark.parametrize("name", TYPES)
def test_cold_fullgraph_dequant_has_no_graph_break(name):
    # Studio's resident max tier compiles blocks with fullgraph=True before any eager forward: no device table yet
    from torch._dynamo.testing import CompileCounter

    lb.install_lowbit_iq()
    lb._CONSTS.clear()
    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(
        gguf_utils.dequantize_gguf_tensor, backend = counter, fullgraph = True, dynamic = True
    )
    raw = torch.from_numpy(_blocks(name, 8, seed = 5))
    for rows in (2, 3, 8):
        part = raw[:rows].contiguous()
        got = compiled(gguf_utils.GGUFParameter(part, quant_type = getattr(T, name)))
        assert torch.equal(got.float().view(torch.int32), _oracle(name, part).view(torch.int32))
    assert counter.frame_count == 1  # no break, no recompile
    torch._dynamo.reset()


def test_prepared_tables_are_per_type_and_recompile_at_most_once():
    from torch._dynamo.testing import CompileCounter

    lb.install_lowbit_iq()
    lb._CONSTS.clear()
    module = torch.nn.Module()
    raw = torch.from_numpy(_blocks("IQ2_XS", 4, seed = 6))
    module.weight = gguf_utils.GGUFParameter(raw, quant_type = T.IQ2_XS, requires_grad = False)
    assert lb.prepare_lowbit_iq_tables("cpu", torch.nn.Linear(2, 2)) == []
    assert lb._CONSTS == {}
    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(
        gguf_utils.dequantize_gguf_tensor, backend = counter, fullgraph = True, dynamic = True
    )
    cold = compiled(module.weight)
    assert lb.prepare_lowbit_iq_tables("cpu", module) == ["IQ2_XS", "ksigns"]
    assert set(lb._CONSTS) == {("IQ2_XS", "cpu"), ("ksigns", "cpu")}
    warm = compiled(module.weight)
    assert torch.equal(cold, warm) and torch.equal(warm.float(), _oracle("IQ2_XS", raw))
    assert counter.frame_count <= 2  # a prepared table recompiles at most once
    torch._dynamo.reset()


class _ReadStop(Exception):
    pass


def _read_sites():
    from core.inference import diffusion_gguf_pipeline as pipe_mod, video_ltx2, video_ltx25
    return {
        "whole_pipeline": lambda p: pipe_mod.load_whole_pipeline_gguf(
            object, object, p, {}, dtype = torch.bfloat16
        ),
        "ltx23_assembly": lambda p: video_ltx2._assemble_ltx23_pipeline(
            p, base_repo = "x", torch_dtype = torch.bfloat16, is_gguf = True
        ),
        "ltx25_transformer": lambda p: video_ltx25.load_ltx25_transformer(
            p, base_repo = "x", torch_dtype = torch.bfloat16, is_gguf = True, hf_token = None
        ),
        "ltx25_read_tensors": lambda p: video_ltx25._read_tensors(p, lambda key: True),
    }


@pytest.mark.parametrize(
    "site", ["whole_pipeline", "ltx23_assembly", "ltx25_transformer", "ltx25_read_tensors"]
)
def test_every_gguf_read_sees_the_low_bit_types_on_a_first_load(site, monkeypatch):
    # diffusers' GGUF reader refuses unknown types, so the types must be in its table before the file is read
    import diffusers.loaders.single_file_utils as sfu

    seen = []

    def read(path, *args, **kwargs):
        seen.append(T.IQ2_XXS in gguf_utils.dequantize_functions)
        raise _ReadStop

    monkeypatch.setattr(sfu, "load_single_file_checkpoint", read)
    assert T.IQ2_XXS not in gguf_utils.dequantize_functions
    with pytest.raises(_ReadStop):
        _read_sites()[site]("model.gguf")
    assert seen == [True]
