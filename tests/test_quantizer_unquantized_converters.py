# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A model converter over a tensor a pre-quantized checkpoint keeps in 16 bit must still load it."""

import inspect

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


def _call_update(method, quantizer, converters):
    # transformers 5.21+ passes the model; 5.18 - 5.20 call update_weight_conversions(weight_conversions).
    if "model" in inspect.signature(method).parameters:
        return method(quantizer, converters, _Model())
    return method(quantizer, converters)


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
    stock = _call_update(stock_method, quantizer, [_table_converter()])
    if _loads_plain_shards(
        stock
    ):  # pragma: no cover - this transformers already collects the plain shards
        pytest.skip("this transformers keeps loading the unquantized shards")

    fix_transformers_quantizer_unquantized_converters()
    fixed = _call_update(HfQuantizer.update_weight_conversions, quantizer, [_table_converter()])
    sources = _sources(fixed)
    assert _loads_plain_shards(fixed), sources
    # The rewritten copy stays, so a checkpoint that does quantize the shards still loads.
    assert any(s and s[0].startswith("ngram_embedding.shard_*.weight_") for s in sources), sources
    assert len(fixed) == len(stock) + 1


def test_no_duplicate_when_the_serialized_names_keep_the_bare_name(stock_method):
    # An FP8-style scheme serializes `weight` itself: the rewritten converter already loads plain shards.
    quantizer = _quantizer("MXFP8A16", "mxfp8-quantized")
    HfQuantizer.update_weight_conversions = stock_method
    stock = _call_update(stock_method, quantizer, [_table_converter()])
    fix_transformers_quantizer_unquantized_converters()
    fixed = _call_update(HfQuantizer.update_weight_conversions, quantizer, [_table_converter()])
    assert len(fixed) == len(stock)


def test_the_fix_is_idempotent(stock_method):
    HfQuantizer.update_weight_conversions = stock_method
    fix_transformers_quantizer_unquantized_converters()
    first = HfQuantizer.__dict__["update_weight_conversions"]
    fix_transformers_quantizer_unquantized_converters()
    assert HfQuantizer.__dict__["update_weight_conversions"] is first
    assert first.__wrapped__ is stock_method


def _dequantize_op():
    try:
        from transformers.integrations.compressed_tensors import CompressedTensorsDequantize
    except Exception:  # pragma: no cover
        pytest.skip("no CompressedTensorsDequantize in this transformers")
    fix_transformers_quantizer_unquantized_converters()
    return CompressedTensorsDequantize(_quantizer("NVFP4A16", "nvfp4-pack-quantized"))


@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_dequantize_op_passes_unquantized_sources_through(model_dtype):
    # transformers 5.23 hands the plain shards to the dequantize op, whose packed branch dropped them; they land in
    # the parameter's dtype, as the rest of the load casts them.
    shards = [torch.randn(4, 4, dtype = torch.bfloat16) for _ in range(2)]
    out = _dequantize_op().convert(
        {SHARDS[0]: list(shards)},
        source_patterns = SHARDS,
        target_patterns = TARGET,
        full_layer_name = TARGET[0],
        model = _Model().to(model_dtype),
    )
    assert list(out) == [SHARDS[0]]
    assert all(a.dtype == model_dtype for a in out[SHARDS[0]])
    assert all(torch.equal(a, b.to(model_dtype)) for a, b in zip(out[SHARDS[0]], shards))


def test_one_pattern_mixing_unquantized_and_packed_shards_is_refused():
    shard = torch.randn(4, 4, dtype = torch.bfloat16)
    packed = {
        "ngram_embedding.shard_*.weight_packed": [torch.zeros(4, 2, dtype = torch.uint8)],
        "ngram_embedding.shard_*.weight_scale": [torch.ones(4, 1)],
    }
    with pytest.raises(ValueError, match = "unquantized and others packed"):
        _dequantize_op().convert(
            {SHARDS[0]: [shard], **packed},
            source_patterns = SHARDS,
            target_patterns = TARGET,
            full_layer_name = TARGET[0],
            model = _Model(),
        )


def test_an_unanchored_matcher_gets_an_anchored_converter_back(stock_method):
    # transformers <= 5.20 searches source patterns unanchored: a restored `.weight` would also take a quantized
    # tensor's `.weight_global_scale`. The restored copy is anchored instead.
    import re

    class _SearchConverter(WeightConverter):
        def rename_source_key(self, source_key):
            for pattern in self._original_source_patterns:
                if re.search(re.escape(pattern.rstrip("$")).replace(r"\*", ".*"), source_key):
                    return source_key, pattern
            return source_key, None

    quantizer = _quantizer("NVFP4A16", "nvfp4-pack-quantized")
    HfQuantizer.update_weight_conversions = stock_method
    stock = _call_update(
        stock_method, quantizer, [_SearchConverter(SHARDS, TARGET, [Concatenate(dim = 0)])]
    )
    if _loads_plain_shards(stock):  # pragma: no cover
        pytest.skip("this transformers keeps loading the unquantized shards")
    fix_transformers_quantizer_unquantized_converters()
    fixed = _call_update(
        HfQuantizer.update_weight_conversions,
        quantizer,
        [_SearchConverter(SHARDS, TARGET, [Concatenate(dim = 0)])],
    )
    restored = fixed[-1]
    assert type(restored) is WeightConverter
    assert list(restored._original_source_patterns) == [SHARDS[0] + "$"]
    assert restored.rename_source_key("ngram_embedding.shard_0.weight")[1] is not None
    for suffix in ("_global_scale", "_zero_point", "_scale", "_packed"):
        assert restored.rename_source_key("ngram_embedding.shard_0.weight" + suffix)[1] is None


def test_the_cpu_route_installs_the_fix():
    import os
    import subprocess
    import sys

    code = (
        "import unsloth\n"
        "from transformers.quantizers.base import HfQuantizer\n"
        "print('INSTALLED', getattr(HfQuantizer.update_weight_conversions, '_unsloth_keeps_unquantized', False))\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_IS_PRESENT = "1")
    env.pop("UNSLOTH_DISABLE_CPU_FALLBACK", None)
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600
    )
    assert "INSTALLED True" in out.stdout, (out.stdout[-2000:], out.stderr[-2000:])


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
    assert len(out) == 1 + restored
    if restored:
        # The original, or on transformers <= 5.20 (unanchored matching) a copy anchored at the bare name.
        back = out[-1]
        assert [p.rstrip("$") for p in back._original_source_patterns] == list(
            original._original_source_patterns
        )
        assert list(back._original_target_patterns) == list(original._original_target_patterns)
        assert back is original or all(p.endswith("$") for p in back._original_source_patterns)
