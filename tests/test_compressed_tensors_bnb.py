# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading a compressed-tensors packed INT4 / INT8 checkpoint straight into bitsandbytes 4-bit.

Offline tests cover the checkpoint classifier, the storage-dtype plan for the packed tensors,
the config stripping (root and sub-configs) and the loader hand-off. The GPU tests build a
tiny packed Llama with compressed-tensors' own compressor, load it through
`FastLanguageModel.from_pretrained(load_in_4bit = True)` and compare every Linear4bit bit for
bit against the same checkpoint decompressed to bf16 on disk first.
"""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch

# Import unsloth first to set UNSLOTH_IS_PRESENT env var.
import unsloth  # noqa: F401
from unsloth.models.compressed_tensors_bnb import (
    UNSLOTH_COMPRESSED_TENSORS_ATTR,
    _config_and_subconfigs,
    _generalize,
    arm_compressed_tensors_bnb_loading,
    compressed_tensors_bnb_plan,
    install_compressed_tensors_bnb_quantizer,
    packed_weight_dtype_plan,
)
from unsloth.models.loader_utils import check_and_disable_bitsandbytes_loading

try:
    import compressed_tensors  # noqa: F401
    HAS_CT = True
except Exception:
    HAS_CT = False

# The re-quantization hooks the quantizer's `update_weight_conversions` (transformers 5.8+);
# 5.0 to 5.7 have the converter loader without that hook, and the module stays inert there.
from unsloth.models.compressed_tensors_bnb import _transformers_supports_weight_converters

HAS_CONVERTERS = _transformers_supports_weight_converters()


def _w4a16(**overrides):
    weights = {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 32,
        "block_structure": None,
        "dynamic": False,
        "actorder": None,
        "observer": "minmax",
        "observer_kwargs": {},
    }
    weights.update(overrides.pop("weights", {}))
    quant = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": None,
                "output_activations": None,
            }
        },
    }
    quant.update(overrides)
    return quant


class _Config(SimpleNamespace):
    """Enough of PretrainedConfig for the classifier: attribute access plus to_dict."""

    def to_dict(self):
        return dict(self.__dict__)


# ----------------------------------------------------------------------------- classifier


def test_plan_accepts_w4a16_pack_quantized():
    assert compressed_tensors_bnb_plan(_Config(quantization_config = _w4a16())) is not None


def _mxfp4(**weights):
    """moonshotai/Kimi-K3's layout: MXFP4 routed experts, everything else left in bf16."""
    quant = _w4a16(
        format = "mxfp4-pack-quantized",
        weights = {"type": "float", "num_bits": 4, "group_size": 32, "scale_dtype": "torch.uint8"},
    )
    quant["config_groups"]["group_0"]["format"] = "mxfp4-pack-quantized"
    quant["config_groups"]["group_0"]["weights"].update(weights)
    return quant


def test_plan_accepts_mxfp4_pack_quantized():
    plan = compressed_tensors_bnb_plan(_Config(quantization_config = _mxfp4()))
    assert plan is not None and plan["format"] == "mxfp4-pack-quantized"


def test_plan_declines_mxfp4_with_a_layout_it_cannot_decode():
    for weights in ({"group_size": 16}, {"type": "int"}, {"num_bits": 8}):
        assert (
            compressed_tensors_bnb_plan(_Config(quantization_config = _mxfp4(**weights))) is None
        ), weights


def test_plan_accepts_int8_and_asymmetric_and_actorder():
    for weights in (
        {"num_bits": 8},
        {"symmetric": False},
        {"actorder": "group"},
        {"strategy": "channel", "group_size": None},
    ):
        assert (
            compressed_tensors_bnb_plan(_Config(quantization_config = _w4a16(weights = weights)))
            is not None
        ), weights


def test_plan_accepts_legacy_nested_sparseml_spelling():
    inner = _w4a16()
    inner.pop("quant_method")
    inner["quant_method"] = "sparseml"
    inner["quantization_status"] = "frozen"
    outer = {
        "quantization_config": inner,
        "sparsity_config": {"format": "dense"},
        "quant_method": "compressed-tensors",
    }
    plan = compressed_tensors_bnb_plan(_Config(quantization_config = outer))
    assert plan is not None and "config_groups" in plan


def test_plan_declines_other_formats_and_schemes():
    cases = {
        "no config": _Config(),
        "bitsandbytes": _Config(
            quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True}
        ),
        "fp8": _Config(
            quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        ),
        "float weights": _Config(
            quantization_config = _w4a16(weights = {"type": "float", "num_bits": 8})
        ),
        "nvfp4 format": _Config(quantization_config = _w4a16(format = "nvfp4-pack-quantized")),
        "activations quantized": _Config(quantization_config = _w4a16()),
        "sparse": _Config(
            quantization_config = _w4a16(sparsity_config = {"format": "sparse-24-bitmask"})
        ),
        "gptq": _Config(quantization_config = {"quant_method": "gptq", "bits": 4}),
        "no groups": _Config(quantization_config = _w4a16(config_groups = {})),
    }
    cases["activations quantized"].quantization_config["config_groups"]["group_0"][
        "input_activations"
    ] = {"num_bits": 8, "type": "int"}
    for name, config in cases.items():
        assert compressed_tensors_bnb_plan(config) is None, name


def test_plan_declines_mixed_groups_with_a_float_scheme():
    quant = _w4a16()
    quant["config_groups"]["group_1"] = {
        "targets": ["re:.*self_attn.*"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True},
        "input_activations": None,
        "output_activations": None,
    }
    assert compressed_tensors_bnb_plan(_Config(quantization_config = quant)) is None


# ----------------------------------------------------------------------------- dtype plan


def test_generalize_widens_numeric_path_components_only():
    rx = _generalize("model.layers.0.mlp.experts.12.gate_proj")
    assert rx == r"model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj"
    assert _generalize("model.embed_tokens") == r"model\.embed_tokens"
    assert _generalize("layer0.block") == r"layer0\.block"


def test_packed_dtype_plan_uses_families_and_exact_names_on_collision():
    import re

    keys = [
        "model.layers.0.mlp.experts.12.gate_proj.weight_packed",
        "model.layers.0.mlp.experts.12.gate_proj.weight_scale",
        "model.layers.1.mlp.experts.3.gate_proj.weight_packed",
        "model.layers.0.self_attn.q_proj.weight_packed",
        "model.layers.1.self_attn.q_proj.weight",  # this layer's q_proj is stored unpacked
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    plan = packed_weight_dtype_plan(keys)
    assert all(v is None for v in plan.values())
    rx = re.compile("|".join(plan))
    kept = [
        k
        for k in [
            "model.layers.0.mlp.experts.12.gate_proj.weight",
            "model.layers.7.mlp.experts.0.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        if rx.search(k)
    ]
    assert kept == [
        "model.layers.0.mlp.experts.12.gate_proj.weight",
        "model.layers.7.mlp.experts.0.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert packed_weight_dtype_plan(["model.embed_tokens.weight"]) == {}


# ----------------------------------------------------------------------------- config stripping


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_arm_strips_root_and_subconfigs_and_parks_the_plan():
    from transformers import LlamaConfig

    root = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    text = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    root.quantization_config = _w4a16()
    text.quantization_config = _w4a16()
    root.text_config = text
    assert text in _config_and_subconfigs(root)
    plan = arm_compressed_tensors_bnb_loading(root, verbose = False)
    assert plan is not None
    assert not hasattr(root, "quantization_config") and not hasattr(text, "quantization_config")
    assert getattr(root, UNSLOTH_COMPRESSED_TENSORS_ATTR) is plan
    assert "quantization_config" not in root.to_dict()


def test_prepared_config_is_the_armed_config_and_nothing_else():
    from transformers import LlamaConfig
    from unsloth.models.loader_utils import compressed_tensors_prepared_config

    plain = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    assert compressed_tensors_prepared_config(plain) is None
    assert compressed_tensors_prepared_config(None) is None
    armed = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    armed.quantization_config = _w4a16()
    if arm_compressed_tensors_bnb_loading(armed, verbose = False) is None:
        pytest.skip("this transformers cannot arm the re-quantization")
    assert compressed_tensors_prepared_config(armed) is armed


def test_planner_gets_the_prepared_config_instead_of_rebuilding_it(monkeypatch):
    """The planner rebuilds the repo's config from the model name. For a re-quantized packed
    checkpoint that config still carries compressed-tensors, and `merge_quantization_configs`
    then refuses the bitsandbytes flags, so Kimi-K2.7-Code lost its plan and fell back to
    `sequential`, which spilled to CPU and bitsandbytes refused the load. The armed config
    object is what the planner has to size."""
    import unsloth_zoo.device_map_planner as planner
    from unsloth.models import loader_utils

    if not loader_utils.planner_accepts_prepared_config():
        pytest.skip("this unsloth_zoo planner does not take a prepared config")
    seen = {}

    def fake_plan(
        model_name,
        *,
        max_memory = None,
        **kwargs,
    ):
        seen.update(kwargs)
        seen["max_memory"] = max_memory
        return None

    monkeypatch.setattr(planner, "plan_device_map_for_pretrained", fake_plan)
    monkeypatch.setattr(loader_utils, "DEVICE_TYPE_TORCH", "cuda")
    monkeypatch.setattr(loader_utils, "is_distributed", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (10 * 1024**3, 16 * 1024**3))
    prepared = object()
    device_map = loader_utils.resolve_unsloth_device_map(
        loader_utils.UNSLOTH_DEVICE_MAP, "some/repo", prepared_config = prepared, load_in_4bit = True
    )
    assert seen["config"] is prepared
    assert seen["load_in_4bit"] is True
    assert device_map == loader_utils._PLANNED_DEVICE_MAPS[loader_utils.UNSLOTH_DEVICE_MAP]

    # An unsloth_zoo whose planner cannot take the object declines the plan instead of handing it a config it would size wrong.
    seen.clear()
    monkeypatch.setattr(loader_utils, "planner_accepts_prepared_config", lambda: False)
    device_map = loader_utils.resolve_unsloth_device_map(
        loader_utils.UNSLOTH_DEVICE_MAP, "some/repo", prepared_config = prepared, load_in_4bit = True
    )
    assert seen == {}
    assert device_map == loader_utils._PLANNED_DEVICE_MAPS[loader_utils.UNSLOTH_DEVICE_MAP]


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_check_and_disable_keeps_4bit_for_packed_int4():
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    config.quantization_config = _w4a16()
    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False
    )
    assert (load_in_4bit, load_in_8bit, method) == (True, False, None)
    assert not hasattr(config, "quantization_config")


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
@pytest.mark.parametrize("spelling", ["compressed_tensors", "sparseml"])
def test_check_and_disable_accepts_the_method_aliases(spelling):
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    quantization_config = dict(_w4a16())
    quantization_config["quant_method"] = spelling
    config.quantization_config = quantization_config
    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False
    )
    assert (load_in_4bit, load_in_8bit, method) == (True, False, None)


def test_check_and_disable_still_disables_8bit_and_other_methods():
    for quant, flags in (
        (_w4a16(), dict(load_in_4bit = False, load_in_8bit = True)),
        ({"quant_method": "gptq", "bits": 4}, dict(load_in_4bit = True, load_in_8bit = False)),
    ):
        config = _Config(quantization_config = quant)
        load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
            config, verbose = False, **flags
        )
        assert (load_in_4bit, load_in_8bit) == (False, False)
        assert method in ("compressed-tensors", "gptq")
        assert hasattr(config, "quantization_config")


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_check_and_disable_does_not_arm_when_the_4bit_load_is_not_happening():
    """`fast_inference` hands the packed checkpoint to vLLM and `full_finetuning` turns 4-bit off
    right after this call: arming there stripped the checkpoint's own quantization config with
    nothing left to consume the plan, so the packed tensors loaded as unmatched keys."""
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    config.quantization_config = _w4a16()
    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False, requantize_packed = False
    )
    assert (load_in_4bit, load_in_8bit, method) == (False, False, "compressed-tensors")
    assert hasattr(config, "quantization_config")
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_arm_leaves_the_config_alone_when_the_quantizer_cannot_be_installed(monkeypatch):
    """A foreign class in the bitsandbytes quantizer slot would load the packed tensors without
    the converters; the config must then keep its own quantization config."""
    from transformers import LlamaConfig
    from unsloth.models import compressed_tensors_bnb

    monkeypatch.setattr(
        compressed_tensors_bnb, "install_compressed_tensors_bnb_quantizer", lambda: False
    )
    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    config.quantization_config = _w4a16()
    assert compressed_tensors_bnb.arm_compressed_tensors_bnb_loading(config, verbose = False) is None
    assert hasattr(config, "quantization_config")
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_arm_declines_a_plan_the_installed_compressed_tensors_cannot_parse(monkeypatch):
    """Fields get retired between compressed-tensors releases (`actorder = "group"` left in
    0.19.0) and the parse used to happen inside `from_pretrained`, after the checkpoint's own
    quantization config had already been stripped, so the load died with nothing to fall back
    to. The plan is parsed before anything is touched."""
    from transformers import LlamaConfig
    from unsloth.models import compressed_tensors_bnb

    def cannot_parse(plan):
        raise ValueError("actorder='group' has been removed")

    monkeypatch.setattr(compressed_tensors_bnb, "_build_quantization_config", cannot_parse)
    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    config.quantization_config = _w4a16(weights = {"actorder": "group"})
    assert compressed_tensors_bnb.arm_compressed_tensors_bnb_loading(config, verbose = False) is None
    assert config.quantization_config["config_groups"]["group_0"]["weights"]["actorder"] == "group"
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_the_planner_pass_does_not_consume_the_plan():
    """The device-map planner preprocesses a meta model built from the very config object the
    load uses. The plan used to be taken off the config by the first pass, so on a multi-GPU
    load the real quantizer found none, added no converters, and every packed expert was
    reported missing and quantized from its random initialisation."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.quantizers import AutoHfQuantizer
    from transformers.utils.quantization_config import BitsAndBytesConfig
    from accelerate import init_empty_weights

    assert install_compressed_tensors_bnb_quantizer()
    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    config.quantization_config = _w4a16()
    plan = arm_compressed_tensors_bnb_loading(config, verbose = False)
    assert plan is not None
    bnb = BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16, bnb_4bit_quant_type = "nf4"
    )

    def preprocess():
        quantizer = AutoHfQuantizer.from_config(bnb, pre_quantized = False)
        with init_empty_weights():
            model = LlamaForCausalLM(config)
        quantizer._process_model_before_weight_loading(
            model, dtype = torch.bfloat16, device_map = None, checkpoint_files = []
        )
        return quantizer, model

    first, _ = preprocess()  # the planner's meta pass
    assert getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None) is plan
    second, model = preprocess()  # the real load
    assert second._unsloth_ct_config is not None
    conversions = second.update_weight_conversions([])
    assert any(
        getattr(c, "source_patterns", None) and "weight_packed$" in c.source_patterns
        for c in conversions
    ), [getattr(c, "source_patterns", None) for c in conversions]
    second._process_model_after_weight_loading(model)
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_armed_conversions_still_run_a_composed_subclass_hook(monkeypatch):
    """When another bitsandbytes 4-bit subclass is already registered, ours is layered on top of
    it. An armed load used to build its converter list and return it directly, so the other
    subclass's ``update_weight_conversions`` never ran."""
    from accelerate import init_empty_weights
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.quantizers import AutoHfQuantizer
    from transformers.quantizers import auto as quantizers_auto

    from unsloth.models import compressed_tensors_bnb
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from transformers.utils.quantization_config import BitsAndBytesConfig

    marker = object()

    class MarkerBnb4BitHfQuantizer(Bnb4BitHfQuantizer):
        def update_weight_conversions(self, weight_conversions):
            return super().update_weight_conversions(weight_conversions) + [marker]

    monkeypatch.setitem(
        quantizers_auto.AUTO_QUANTIZER_MAPPING, "bitsandbytes_4bit", MarkerBnb4BitHfQuantizer
    )
    monkeypatch.setattr(compressed_tensors_bnb, "_installed", False)
    assert install_compressed_tensors_bnb_quantizer()
    composed = quantizers_auto.AUTO_QUANTIZER_MAPPING["bitsandbytes_4bit"]
    assert issubclass(composed, MarkerBnb4BitHfQuantizer)

    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    config.quantization_config = _w4a16()
    plan = arm_compressed_tensors_bnb_loading(config, verbose = False)
    assert plan is not None
    bnb = BitsAndBytesConfig(
        load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16, bnb_4bit_quant_type = "nf4"
    )
    quantizer = AutoHfQuantizer.from_config(bnb, pre_quantized = False)
    assert isinstance(quantizer, composed)
    with init_empty_weights():
        model = LlamaForCausalLM(config)
    quantizer._process_model_before_weight_loading(
        model, dtype = torch.bfloat16, device_map = None, checkpoint_files = []
    )
    assert quantizer._unsloth_ct_config is not None
    conversions = quantizer.update_weight_conversions([])
    assert marker in conversions
    assert any(
        getattr(c, "source_patterns", None) and "weight_packed$" in c.source_patterns
        for c in conversions
    ), [getattr(c, "source_patterns", None) for c in conversions]


@pytest.mark.skipif(not HAS_CONVERTERS, reason = "needs the transformers 5 loader")
def test_quantizer_registration_is_idempotent_and_a_subclass():
    from transformers.quantizers import auto as quantizers_auto
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

    assert install_compressed_tensors_bnb_quantizer()
    first = quantizers_auto.AUTO_QUANTIZER_MAPPING["bitsandbytes_4bit"]
    assert install_compressed_tensors_bnb_quantizer()
    assert quantizers_auto.AUTO_QUANTIZER_MAPPING["bitsandbytes_4bit"] is first
    assert issubclass(first, Bnb4BitHfQuantizer)


# ----------------------------------------------------------------------------- GPU: real loads


def _write_tiny_packed_llama(
    root,
    asymmetric = False,
    actorder = False,
    num_bits = 4,
    group_size = 32,
    mxfp4 = False,
):
    """A 2-layer Llama with every Linear but lm_head packed by compressed-tensors' compressor.
    Returns (packed_dir, bf16_dir) where bf16_dir holds the exact decompressed weights."""
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig
    from compressed_tensors.quantization.utils import calculate_qparams

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        intermediate_size = 128,
        vocab_size = 256,
        max_position_embeddings = 128,
        tie_word_embeddings = False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    # `actorder = "group"` was deprecated in compressed-tensors 0.18.0 and removed in 0.19.0;
    # "weight" is the spelling its validator still accepts and exercises the same `weight_g_idx`
    # path in the compressor.
    quant = _w4a16(
        weights = {
            "num_bits": num_bits,
            "group_size": group_size,
            "symmetric": not asymmetric,
            "actorder": "weight" if actorder else None,
        }
    )
    if mxfp4:
        quant = _mxfp4()
    ctc = QuantizationConfig.model_validate(quant)
    scheme = list(ctc.config_groups.values())[0]
    comp = BaseCompressor.get_value_from_registry(quant["format"])
    linears = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    packed, plain = {}, {}
    for k, w in sd.items():
        mod = k[: -len(".weight")] if k.endswith(".weight") else None
        if mod is None or mod not in linears or mod == "lm_head":
            packed[k] = w
            plain[k] = w
            continue
        out_f, in_f = w.shape
        wf = w.float()
        g_idx = None
        if actorder:
            # Activation ordering permutes input columns into groups; use a fixed permutation.
            g_idx = torch.randperm(in_f) // group_size
            g_idx = g_idx.to(torch.int32)
            grouped = wf[:, torch.argsort(g_idx)].reshape(out_f, in_f // group_size, group_size)
        else:
            grouped = wf.reshape(out_f, in_f // group_size, group_size)
        if mxfp4:
            # One power-of-two scale per 32 so the group maximum lands in E2M1's top binade.
            amax = grouped.abs().amax(-1).clamp_min(2**-126)
            scale, zp = 2.0 ** (torch.floor(torch.log2(amax)) - 2), None
        else:
            scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), scheme.weights)
        state = {"weight": wf, "weight_scale": scale.to(torch.bfloat16)}
        if asymmetric:
            state["weight_zero_point"] = zp.to(torch.int8)
        if g_idx is not None:
            state["weight_g_idx"] = g_idx
        out = comp.compress(state, scheme)
        for pk, pv in out.items():
            packed[mod + "." + pk] = pv.contiguous()
        dec = comp.decompress({k2: v2 for k2, v2 in out.items()}, scheme)["weight"].to(
            torch.bfloat16
        )
        plain[k] = dec.contiguous()
    packed_dir, bf16_dir = os.path.join(root, "packed"), os.path.join(root, "bf16")
    os.makedirs(packed_dir)
    os.makedirs(bf16_dir)
    save_file(packed, os.path.join(packed_dir, "model.safetensors"), metadata = {"format": "pt"})
    save_file(plain, os.path.join(bf16_dir, "model.safetensors"), metadata = {"format": "pt"})
    cfg = config.to_dict()
    json.dump(cfg, open(os.path.join(bf16_dir, "config.json"), "w"))
    cfg["quantization_config"] = quant
    json.dump(cfg, open(os.path.join(packed_dir, "config.json"), "w"))
    return packed_dir, bf16_dir


def _same_linear4bit(a, b):
    import bitsandbytes as bnb

    mods_b = dict(b.named_modules())
    n = 0
    for name, m in a.named_modules():
        if not isinstance(m, bnb.nn.Linear4bit):
            continue
        o = mods_b[name]
        assert isinstance(o, bnb.nn.Linear4bit), name
        qa, qb = m.weight.quant_state, o.weight.quant_state
        assert torch.equal(m.weight.data.view(-1), o.weight.data.view(-1)), name
        assert (
            torch.equal(qa.absmax, qb.absmax) and qa.shape == qb.shape and qa.dtype == qb.dtype
        ), name
        n += 1
    return n


def _tokenizer_free_load(path, root):
    """Unsloth's loader wants a tokenizer next to the weights; a tiny one is enough."""
    from transformers import AutoTokenizer

    tok_dir = os.path.join(root, "tok")
    if not os.path.isdir(tok_dir):
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(
            tok_dir
        )
    for f in os.listdir(tok_dir):
        if not os.path.exists(os.path.join(path, f)):
            shutil.copy(os.path.join(tok_dir, f), os.path.join(path, f))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
@pytest.mark.parametrize("variant", ["symmetric", "asymmetric", "actorder", "int8", "mxfp4"])
def test_packed_checkpoint_loads_as_linear4bit_bit_identical_to_disk_route(variant, tmp_path):
    from unsloth import FastLanguageModel

    packed_dir, bf16_dir = _write_tiny_packed_llama(
        str(tmp_path),
        asymmetric = variant == "asymmetric",
        actorder = variant == "actorder",
        num_bits = 8 if variant == "int8" else 4,
        mxfp4 = variant == "mxfp4",
    )
    for d in (packed_dir, bf16_dir):
        _tokenizer_free_load(d, str(tmp_path))
    kw = dict(max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True)
    model_a, _ = FastLanguageModel.from_pretrained(packed_dir, **kw)
    model_b, _ = FastLanguageModel.from_pretrained(bf16_dir, **kw)
    n = _same_linear4bit(model_a, model_b)
    assert n == 2 * 7, n  # q k v o gate up down per layer
    assert not any(k.endswith("weight_packed") for k, _ in model_a.named_parameters())
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        assert torch.equal(model_a(input_ids = ids).logits, model_b(input_ids = ids).logits)
    # The stripped checkpoint config never carries the plan or the old quantization config.
    assert not hasattr(model_a.config, UNSLOTH_COMPRESSED_TENSORS_ATTR)
    assert getattr(model_a.config, "quantization_config", None) is not None  # the bitsandbytes one
    assert "compressed" not in str(model_a.config.quantization_config).lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_packed_checkpoint_trains_with_lora(tmp_path):
    from unsloth import FastLanguageModel

    packed_dir, _ = _write_tiny_packed_llama(str(tmp_path))
    _tokenizer_free_load(packed_dir, str(tmp_path))
    model, _ = FastLanguageModel.from_pretrained(
        packed_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True
    )
    model = FastLanguageModel.get_peft_model(
        model, r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"]
    )
    model.train()
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
def test_many_to_many_expert_op_keeps_its_source_contract():
    """transformers' ErnieFuseAndSplitTextVisionExperts iterates the converter's
    source_patterns and requires every one of them in the collected dict. After
    decompression the packed metadata patterns are gone, so the op must see the
    patterns it was declared with and the decompressed buckets under them."""
    from transformers.core_model_loading import ErnieFuseAndSplitTextVisionExperts, WeightConverter
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig
    from compressed_tensors.quantization.utils import calculate_qparams
    from unsloth.models.compressed_tensors_bnb import _WithOriginalSources, _DecompressPackedWeights

    torch.manual_seed(0)
    quant = _w4a16(weights = {"num_bits": 4, "group_size": 32, "symmetric": True})
    ctc = QuantizationConfig.model_validate(quant)
    scheme = list(ctc.config_groups.values())[0]
    comp = BaseCompressor.get_value_from_registry("pack-quantized")
    sources = ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"]
    targets = ["mlp.text_experts.gate_up_proj", "mlp.vision_experts.gate_up_proj"]

    plain, packed = {}, {}
    for p in sources:
        plain[p], packed[p + "_packed$"], packed[p + "_scale$"], packed[p + "_shape$"] = (
            [],
            [],
            [],
            [],
        )
        for _ in range(4):  # two text experts then two vision experts
            w = torch.randn(16, 64)
            grouped = w.reshape(16, 2, 32)
            scale, _zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), scheme.weights)
            out = comp.compress({"weight": w, "weight_scale": scale.to(torch.bfloat16)}, scheme)
            plain[p].append(comp.decompress(dict(out), scheme)["weight"].to(torch.bfloat16))
            packed[p + "_packed$"].append(out["weight_packed"])
            packed[p + "_scale$"].append(out["weight_scale"])
            packed[p + "_shape$"].append(out["weight_shape"])

    ernie = ErnieFuseAndSplitTextVisionExperts()
    want = ernie.convert(dict(plain), source_patterns = sources, target_patterns = targets, config = None)

    # The rebuilt converter: decompression first, then the converter's own op under its contract.
    rebuilt_sources = (
        [p + "_packed$" for p in sources]
        + [p + "_scale$" for p in sources]
        + [p + "_shape$" for p in sources]
        + [p + "$" for p in sources]
    )
    ops = [
        _DecompressPackedWeights(ctc, torch.bfloat16, stacked = True, scheme = scheme),
        _WithOriginalSources(ernie, sources, sources),
    ]
    got = dict(packed)
    for op in ops:
        got = op.convert(
            got,
            source_patterns = rebuilt_sources,
            target_patterns = targets,
            full_layer_name = "model.layers.0",
            model = None,
            config = None,
        )
    assert set(got) == set(targets)
    for k in targets:
        assert torch.equal(got[k], want[k]), k

    # Without the adapter the many-to-many op fails on the consumed metadata patterns.
    got = dict(packed)
    got = ops[0].convert(
        got,
        source_patterns = rebuilt_sources,
        target_patterns = targets,
        full_layer_name = "model.layers.0",
        model = None,
        config = None,
    )
    with pytest.raises((ValueError, TypeError)):
        ernie.convert(got, source_patterns = rebuilt_sources, target_patterns = targets, config = None)


def test_classification_load_under_fast_inference_still_requantizes():
    """fast_inference with num_labels loads through transformers (vLLM has no classification
    head), so the packed re-quantization must stay armed there."""
    import inspect
    from unsloth.models import llama

    assert llama._vllm_will_load_weights(True, num_labels = 2) is False
    assert llama._vllm_will_load_weights(False, None) is False
    source = inspect.getsource(llama.FastLlamaModel.from_pretrained)
    assert "requantize_packed = not _vllm_will_load_weights(fast_inference, num_labels)" in source


@pytest.mark.skipif(not HAS_CONVERTERS, reason = "needs the transformers 5 loader")
def test_wrapped_many_to_many_op_still_passes_the_converter_type_check():
    """transformers' WeightConverter allows a many-to-many mapping only when `operations`
    holds an instance of its internal Ernie ops. The adapter that feeds such an op its
    original source contract must remain an instance of the op's class, or building the
    replacement converter raises before any weight loads."""
    from transformers.core_model_loading import ErnieFuseAndSplitTextVisionExperts, WeightConverter
    from unsloth.models.compressed_tensors_bnb import _WithOriginalSources, _with_original_sources

    op = ErnieFuseAndSplitTextVisionExperts(stack_dim = 0, concat_dim = 1)
    sources = ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"]
    targets = ["mlp.text_experts.gate_up_proj", "mlp.vision_experts.gate_up_proj"]
    wrapped = _with_original_sources(op, sources, sources)
    assert isinstance(wrapped, _WithOriginalSources) and isinstance(
        wrapped, ErnieFuseAndSplitTextVisionExperts
    )
    WeightConverter(
        source_patterns = [s + "_packed$" for s in sources] + [s + "$" for s in sources],
        target_patterns = targets,
        operations = [wrapped],
    )


def test_a_pickled_shard_is_refused_rather_than_read_in_the_model_dtype():
    """A .bin shard is materialised in the model dtype before any converter runs, so the packed
    words would be cast to bf16 garbage; the plan builder refuses it with an instruction."""
    from unsloth.models.compressed_tensors_bnb import _checkpoint_keys

    with pytest.raises(RuntimeError, match = "safetensors"):
        _checkpoint_keys(["/x/pytorch_model-00001-of-00002.bin"])
    assert _checkpoint_keys([]) == []


def test_load_only_converters_are_dropped_before_save():
    """transformers reverses every converter on model._weight_conversions in save_pretrained;
    the decompression converters must not be there, or a bitsandbytes model is saved under
    the packed names."""
    from unsloth.models.compressed_tensors_bnb import (
        _DecompressPackedWeights,
        drop_load_only_conversions,
    )

    class Conv:
        def __init__(self, ops):
            self.operations = ops

    model = SimpleNamespace(
        _weight_conversions = [
            Conv([object()]),
            Conv([_DecompressPackedWeights.__new__(_DecompressPackedWeights)]),
        ]
    )
    assert drop_load_only_conversions(model) == 1
    assert len(model._weight_conversions) == 1
    assert drop_load_only_conversions(SimpleNamespace()) == 0


def test_partially_packed_expert_bucket_is_refused():
    from unsloth.models.compressed_tensors_bnb import _WithOriginalSources

    class Op:
        def convert(self, d, **kw):
            return d

    adapter = _WithOriginalSources(Op(), ["experts.*.up_proj.weight"], ["experts.*.up_proj.weight"])
    with pytest.raises(RuntimeError, match = "partially packed"):
        adapter.convert(
            {
                "experts.*.up_proj.weight_packed$": [torch.zeros(2, 2)],
                "experts.*.up_proj.weight$": [torch.zeros(2, 2)],
            }
        )


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_scheme_is_resolved_per_converter_source_and_mixed_buckets_are_refused():
    from compressed_tensors.quantization import QuantizationConfig
    from unsloth.models.compressed_tensors_bnb import _scheme_for_sources

    quant = _w4a16()
    quant["config_groups"] = {
        "group_0": {
            "targets": ["re:.*gate_proj.*"],
            "weights": dict(quant["config_groups"]["group_0"]["weights"], num_bits = 4),
            "input_activations": None,
            "output_activations": None,
        },
        "group_1": {
            "targets": ["re:.*up_proj.*"],
            "weights": dict(quant["config_groups"]["group_0"]["weights"], num_bits = 8),
            "input_activations": None,
            "output_activations": None,
        },
    }
    ctc = QuantizationConfig.model_validate(quant)
    assert _scheme_for_sources(ctc, ["mlp.experts.*.gate_proj.weight"]).weights.num_bits == 4
    assert _scheme_for_sources(ctc, ["mlp.experts.*.up_proj.weight"]).weights.num_bits == 8
    with pytest.raises(RuntimeError, match = "different config groups"):
        _scheme_for_sources(ctc, ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"])


def test_an_explicit_quantizer_other_than_bnb_4bit_keeps_the_checkpoint_config():
    from transformers import BitsAndBytesConfig
    from unsloth.models.loader_utils import quantization_config_selects_bnb_4bit

    assert quantization_config_selects_bnb_4bit(None)
    assert quantization_config_selects_bnb_4bit(BitsAndBytesConfig(load_in_4bit = True))
    assert quantization_config_selects_bnb_4bit(
        {"quant_method": "bitsandbytes", "load_in_4bit": True}
    )
    assert not quantization_config_selects_bnb_4bit(BitsAndBytesConfig(load_in_8bit = True))
    assert not quantization_config_selects_bnb_4bit({"quant_method": "gptq", "bits": 4})
    # With the 8-bit quantizer the plan is never armed, so the packed config stays on the model.
    config = _Config(quantization_config = _w4a16())
    load_in_4bit, _, method = check_and_disable_bitsandbytes_loading(
        config,
        load_in_4bit = True,
        load_in_8bit = False,
        verbose = False,
        requantize_packed = quantization_config_selects_bnb_4bit(
            BitsAndBytesConfig(load_in_8bit = True)
        ),
    )
    assert method == "compressed-tensors" and not load_in_4bit
    assert getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None) is None


def test_both_loaders_gate_packed_requantization_on_the_callers_quantizer():
    import ast, inspect
    from unsloth.models import llama, vision
    for module in (llama, vision):
        calls = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "check_and_disable_bitsandbytes_loading"
        ]
        assert calls, module.__name__
        for call in calls:
            flag = next(k.value for k in call.keywords if k.arg == "requantize_packed")
            assert "quantization_config_selects_bnb_4bit" in ast.unparse(flag), module.__name__


def test_only_an_explicit_bnb_4bit_config_keeps_the_callers_flags():
    # A GPTQ / AWQ / FP8 config with the default load_in_4bit = True must still take the checker's
    # answer, or the 4-bit branch below swaps the caller's quantizer for a BitsAndBytesConfig.
    import ast, inspect
    from unsloth.models import llama, vision
    for module in (llama, vision):
        guards = [
            ast.unparse(node.test)
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.If)
            and any("_checked_4bit" in ast.unparse(stmt) for stmt in node.body)
        ]
        assert guards == ["not _explicit_bnb_4bit"], (module.__name__, guards)


def test_an_explicit_bnb_4bit_config_is_not_replaced_by_the_default_nf4_one():
    import ast, inspect
    from unsloth.models import llama

    guards = [
        ast.unparse(node.test)
        for node in ast.walk(ast.parse(inspect.getsource(llama)))
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Assign)
            and "quantization_config" in ast.unparse(stmt.targets[0])
            and ast.unparse(stmt.value) == "bnb_config"
            for stmt in node.body
        )
    ]
    assert guards and all("not _explicit_bnb_4bit" in guard for guard in guards), guards


@pytest.mark.skipif(not HAS_CONVERTERS, reason = "needs the transformers 5 loader")
def test_both_loaders_treat_an_explicit_bnb_4bit_config_as_the_4bit_request():
    # The public loader forwards load_in_4bit = False when a quantization_config is passed, so a
    # BitsAndBytesConfig(load_in_4bit = True) must still arm the plan at both call sites.
    import ast, inspect
    from transformers import BitsAndBytesConfig
    from unsloth.models import llama, vision
    from unsloth.models.loader_utils import quantization_config_selects_bnb_4bit

    for module in (llama, vision):
        calls = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "check_and_disable_bitsandbytes_loading"
        ]
        for call in calls:
            flag = next(k.value for k in call.keywords if k.arg == "load_in_4bit")
            assert "_explicit_bnb_4bit" in ast.unparse(flag), module.__name__

    explicit = BitsAndBytesConfig(load_in_4bit = True)
    config = _Config(quantization_config = _w4a16())
    check_and_disable_bitsandbytes_loading(
        config,
        load_in_4bit = False or quantization_config_selects_bnb_4bit(explicit),
        load_in_8bit = False,
        verbose = False,
        requantize_packed = quantization_config_selects_bnb_4bit(explicit),
    )
    assert getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None) is not None


def test_exact_module_targets_win_over_a_class_target():
    # compressed-tensors lists some groups by exact module path; the expert converters resolve
    # the scheme without a module object, so the path has to be compared first.
    from types import SimpleNamespace
    from unsloth.models.compressed_tensors_bnb import _layer_expert_scheme, _scheme_for_module

    dense = SimpleNamespace(targets = ["Linear"])
    exact = SimpleNamespace(targets = [f"model.layers.1.mlp.experts.{e}.gate_proj" for e in range(4)])
    ct = SimpleNamespace(config_groups = {"dense": dense, "experts": exact})
    assert _scheme_for_module(ct, "model.layers.1.mlp.experts.2.gate_proj", None) is exact
    assert _scheme_for_module(ct, "model.layers.0.self_attn.q_proj", None) is dense
    key = "mlp.experts.*.gate_proj.weight_packed$"
    assert (
        _layer_expert_scheme(ct, "model.layers.1.mlp.experts.gate_up_proj", key, 4, None) is exact
    )


def test_exact_and_regex_targets_win_when_the_module_is_known():
    # A broad class group listed first must not win over an exact path or a regex.
    pytest.importorskip("compressed_tensors")
    from types import SimpleNamespace
    from unsloth.models.compressed_tensors_bnb import _scheme_for_module

    dense = SimpleNamespace(targets = ["Linear"])
    exact = SimpleNamespace(targets = [f"model.layers.1.mlp.experts.{e}.gate_proj" for e in range(4)])
    ct = SimpleNamespace(config_groups = {"dense": dense, "experts": exact})
    linear = torch.nn.Linear(4, 4)
    assert _scheme_for_module(ct, "model.layers.1.mlp.experts.2.gate_proj", linear) is exact
    assert _scheme_for_module(ct, "model.layers.0.self_attn.q_proj", linear) is dense
    regex = SimpleNamespace(targets = [r"re:.*self_attn\.q_proj$"])
    ct = SimpleNamespace(config_groups = {"dense": dense, "q": regex})
    assert _scheme_for_module(ct, "model.layers.0.self_attn.q_proj", linear) is regex


def test_expert_scheme_is_resolved_per_layer():
    # One converter serves every layer, so a checkpoint that quantizes layer 1's experts and
    # layer 5's experts under different groups must get each layer's own scheme.
    from types import SimpleNamespace
    from unsloth.models.compressed_tensors_bnb import _layer_expert_scheme

    early = SimpleNamespace(targets = [r"re:.*layers\.[0-3]\.mlp\.experts\..*"])
    late = SimpleNamespace(targets = [r"re:.*layers\.([4-9]|\d\d+)\.mlp\.experts\..*"])
    ct = SimpleNamespace(config_groups = {"group_0": early, "group_1": late})
    key = "mlp.experts.*.gate_proj.weight_packed$"
    assert (
        _layer_expert_scheme(ct, "model.layers.1.mlp.experts.gate_up_proj", key, 4, None) is early
    )
    assert _layer_expert_scheme(ct, "model.layers.5.mlp.experts.gate_up_proj", key, 4, None) is late
    assert (
        _layer_expert_scheme(ct, "model.layers.12.mlp.experts.gate_up_proj", key, 4, None) is late
    )

    split = SimpleNamespace(targets = [r"re:.*experts\.[0-1]\..*"])
    rest = SimpleNamespace(targets = [r"re:.*experts\.[2-9]\..*"])
    mixed = SimpleNamespace(config_groups = {"a": split, "b": rest})
    with pytest.raises(RuntimeError, match = "different config groups"):
        _layer_expert_scheme(mixed, "model.layers.0.mlp.experts.gate_up_proj", key, 4, None)

    # Mixtral and PhiMoE source patterns begin with the separator.
    for dotted in (".experts.*.w1.weight_packed$", "^.experts.*.w1.weight_packed"):
        assert (
            _layer_expert_scheme(ct, "model.layers.1.mlp.experts.gate_up_proj", dotted, 4, None)
            is early
        )
        assert (
            _layer_expert_scheme(ct, "model.layers.5.mlp.experts.gate_up_proj", dotted, 4, None)
            is late
        )

    single = SimpleNamespace(config_groups = {"only": early})
    assert (
        _layer_expert_scheme(single, "model.layers.9.mlp.experts.gate_up_proj", key, 4, "d") == "d"
    )


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
@pytest.mark.parametrize("stack_only_merge", [False, True])
def test_chained_expert_ops_all_see_the_original_sources(stack_only_merge):
    """Qwen2/Qwen3-MoE gate/up is MergeModulelist then Concatenate. Every op of the rebuilt chain
    must run under the original source names, and a MergeModulelist that only stacks lists
    (transformers 5.0 to 5.6) must be handed the per-expert list, not the stacked bucket."""
    from transformers.core_model_loading import Concatenate, MergeModulelist
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig
    from compressed_tensors.quantization.utils import calculate_qparams
    from unsloth.models.compressed_tensors_bnb import (
        _DecompressPackedWeights,
        _with_original_sources,
    )

    class StackOnlyMerge(MergeModulelist):
        def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
            size = len(input_dict)
            return {
                self.get_target_pattern(size, key, target_patterns): torch.stack(
                    input_dict.pop(key), dim = self.dim
                )
                for key in list(input_dict)
            }

    torch.manual_seed(0)
    quant = _w4a16(weights = {"num_bits": 4, "group_size": 32, "symmetric": True})
    ctc = QuantizationConfig.model_validate(quant)
    scheme = list(ctc.config_groups.values())[0]
    comp = BaseCompressor.get_value_from_registry("pack-quantized")
    sources = ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"]
    targets = ["mlp.experts.gate_up_proj"]
    plain, packed = {}, {}
    for p in sources:
        plain[p] = []
        packed[p + "_packed$"], packed[p + "_scale$"], packed[p + "_shape$"] = [], [], []
        for _ in range(4):
            w = torch.randn(16, 64)
            scale, _zp = calculate_qparams(
                w.reshape(16, 2, 32).amin(-1), w.reshape(16, 2, 32).amax(-1), scheme.weights
            )
            out = comp.compress({"weight": w, "weight_scale": scale.to(torch.bfloat16)}, scheme)
            plain[p].append(comp.decompress(dict(out), scheme)["weight"].to(torch.bfloat16))
            packed[p + "_packed$"].append(out["weight_packed"])
            packed[p + "_scale$"].append(out["weight_scale"])
            packed[p + "_shape$"].append(out["weight_shape"])

    merge = StackOnlyMerge(dim = 0) if stack_only_merge else MergeModulelist(dim = 0)
    want = dict(plain)
    for op in (MergeModulelist(dim = 0), Concatenate(dim = 1)):
        want = op.convert(want, source_patterns = sources, target_patterns = targets)

    rebuilt_sources = (
        [p + "_packed$" for p in sources]
        + [p + "_scale$" for p in sources]
        + [p + "_shape$" for p in sources]
        + [p + "$" for p in sources]
    )
    ops = [_DecompressPackedWeights(ctc, torch.bfloat16, stacked = True, scheme = scheme)] + [
        _with_original_sources(op, sources, sources, receives_buckets = index == 0)
        for index, op in enumerate([merge, Concatenate(dim = 1)])
    ]
    got = dict(packed)
    for op in ops:
        got = op.convert(
            got,
            source_patterns = rebuilt_sources,
            target_patterns = targets,
            full_layer_name = "model.layers.0",
            model = None,
            config = None,
        )
    assert set(got) == set(targets)
    torch.testing.assert_close(got[targets[0]], want[targets[0]])
