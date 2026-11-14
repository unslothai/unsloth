# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A model converter over a tensor a pre-quantized checkpoint keeps in 16 bit must still load it.

transformers 5.21+ rewrites every model converter ending in `.weight` to collect only the quantizer's serialized
names. Qwen4Exp concatenates its n-gram table from `ngram_embedding.shard_*.weight`, which an NVFP4 / MXFP4
compressed-tensors checkpoint stores unquantized, so the table was left randomly initialized."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressed_tensors")

try:
    from transformers.core_model_loading import Concatenate, WeightConverter
    from transformers.quantizers.base import HfQuantizer
    from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
    from transformers.utils.quantization_config import CompressedTensorsConfig
except Exception:  # pragma: no cover
    pytest.skip("this transformers has no quantizer weight conversions", allow_module_level = True)
if "update_weight_conversions" not in HfQuantizer.__dict__:  # pragma: no cover
    pytest.skip("this transformers has no update_weight_conversions", allow_module_level = True)

from unsloth.import_fixes import fix_transformers_quantizer_unquantized_converters  # noqa: E402

SHARDS = ["ngram_embedding.shard_*.weight"]
TARGET = ["ngram_embedding.weight"]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 4)


def _quantizer(preset, fmt):
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme(preset, ["re:.*proj$"])
    config = CompressedTensorsConfig(
        config_groups = {"group_0": scheme}, format = fmt, quantization_status = "compressed"
    )
    quantizer = CompressedTensorsHfQuantizer(config)
    quantizer.pre_quantized = True
    # Set during validate_environment in a real load; the packed formats here never take the FP8 kernel path.
    if not hasattr(quantizer, "use_fp8_kernel"):
        quantizer.use_fp8_kernel = False
    return quantizer


def _table_converter():
    return WeightConverter(SHARDS, TARGET, [Concatenate(dim = 0)])


def _sources(converters):
    return [list(c._original_source_patterns) for c in converters if isinstance(c, WeightConverter)]


def _loads_plain_shards(converters):
    return any(SHARDS[0] in sources for sources in _sources(converters))


@pytest.fixture
def stock_method():
    method = HfQuantizer.__dict__["update_weight_conversions"]
    stock = getattr(method, "__wrapped__", method)
    yield stock
    HfQuantizer.update_weight_conversions = method


@pytest.mark.parametrize(
    "preset, fmt",
    [("NVFP4A16", "nvfp4-pack-quantized"), ("MXFP4A16", "mxfp4-pack-quantized")],
)
def test_the_unquantized_table_converter_survives_a_packed_checkpoint(preset, fmt, stock_method):
    quantizer = _quantizer(preset, fmt)
    HfQuantizer.update_weight_conversions = stock_method
    stock = stock_method(quantizer, [_table_converter()], _Model())
    if _loads_plain_shards(
        stock
    ):  # pragma: no cover - this transformers already collects the plain shards
        pytest.skip("this transformers keeps loading the unquantized shards")

    fix_transformers_quantizer_unquantized_converters()
    fixed = HfQuantizer.update_weight_conversions(quantizer, [_table_converter()], _Model())
    sources = _sources(fixed)
    assert _loads_plain_shards(fixed), sources
    # The rewritten copy stays, so a checkpoint that does quantize the shards still loads.
    assert any(s and s[0].startswith("ngram_embedding.shard_*.weight_") for s in sources), sources
    assert len(fixed) == len(stock) + 1


def test_no_duplicate_when_the_serialized_names_keep_the_bare_name(stock_method):
    # An FP8-style scheme serializes `weight` itself: the rewritten converter already loads plain shards.
    quantizer = _quantizer("MXFP8A16", "mxfp8-quantized")
    HfQuantizer.update_weight_conversions = stock_method
    stock = stock_method(quantizer, [_table_converter()], _Model())
    fix_transformers_quantizer_unquantized_converters()
    fixed = HfQuantizer.update_weight_conversions(quantizer, [_table_converter()], _Model())
    assert len(fixed) == len(stock)


def test_the_fix_is_idempotent(stock_method):
    HfQuantizer.update_weight_conversions = stock_method
    fix_transformers_quantizer_unquantized_converters()
    first = HfQuantizer.__dict__["update_weight_conversions"]
    fix_transformers_quantizer_unquantized_converters()
    assert HfQuantizer.__dict__["update_weight_conversions"] is first
    assert first.__wrapped__ is stock_method


def test_the_dequantize_op_passes_unquantized_sources_through():
    # transformers 5.23 hands the plain shards to the dequantize op, whose packed branch dropped them.
    try:
        from transformers.integrations.compressed_tensors import CompressedTensorsDequantize
    except Exception:  # pragma: no cover
        pytest.skip("no CompressedTensorsDequantize in this transformers")
    fix_transformers_quantizer_unquantized_converters()
    shards = [torch.randn(4, 4, dtype = torch.bfloat16) for _ in range(2)]
    op = CompressedTensorsDequantize(_quantizer("NVFP4A16", "nvfp4-pack-quantized"))
    out = op.convert(
        {SHARDS[0]: list(shards)},
        source_patterns = SHARDS,
        target_patterns = TARGET,
        full_layer_name = TARGET[0],
        model = _Model(),
    )
    assert list(out) == [SHARDS[0]]
    assert all(torch.equal(a, b) for a, b in zip(out[SHARDS[0]], shards))


def test_the_fix_keeps_the_stock_arity(stock_method):
    # Development builds of transformers 5.18 call update_weight_conversions(weight_conversions) without the model.
    calls = []

    def one_argument(self, weight_conversions):
        calls.append(len(weight_conversions))
        return list(weight_conversions)

    HfQuantizer.update_weight_conversions = one_argument
    fix_transformers_quantizer_unquantized_converters()
    converter = _table_converter()
    assert HfQuantizer.update_weight_conversions(
        _quantizer("NVFP4A16", "nvfp4-pack-quantized"), [converter]
    ) == [converter]
    assert calls == [1]


@pytest.fixture
def ct_override():
    had = "update_weight_conversions" in CompressedTensorsHfQuantizer.__dict__
    saved = CompressedTensorsHfQuantizer.__dict__.get("update_weight_conversions")
    yield
    if had:
        CompressedTensorsHfQuantizer.update_weight_conversions = saved
    elif "update_weight_conversions" in CompressedTensorsHfQuantizer.__dict__:
        del CompressedTensorsHfQuantizer.update_weight_conversions


EXPERTS = ["mlp.experts.*.w1.weight"]


def _expert_converter():
    return WeightConverter(EXPERTS, ["mlp.experts.gate_up_proj"], [Concatenate(dim = 0)])


@pytest.mark.parametrize(
    "suffixes, restored", [(("_packed$", "_scale$"), True), (("$", "_scale$"), False)]
)
def test_a_compressed_tensors_override_gets_the_replaced_converter_back_last(
    suffixes, restored, ct_override
):
    # Earlier 5.x builds override update_weight_conversions on the compressed-tensors quantizer and rewrite every
    # expert converter: packed formats to `weight_packed` only (bf16 experts were lost), FP8 keeping `weight$`.
    def port_style(self, weight_conversions):
        return [
            WeightConverter(
                [p + s for p in c._original_source_patterns for s in suffixes],
                c._original_target_patterns,
                list(c.operations),
            )
            for c in weight_conversions
        ]

    CompressedTensorsHfQuantizer.update_weight_conversions = port_style
    fix_transformers_quantizer_unquantized_converters()
    original = _expert_converter()
    out = CompressedTensorsHfQuantizer.update_weight_conversions(
        _quantizer("NVFP4A16", "nvfp4-pack-quantized"), [original]
    )
    assert (out[-1] is original) is restored
    assert len(out) == 1 + restored
