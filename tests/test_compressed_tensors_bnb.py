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

try:
    from transformers.core_model_loading import WeightConverter  # noqa: F401

    HAS_CONVERTERS = True
except Exception:
    HAS_CONVERTERS = False


def _w4a16(**overrides):
    weights = {
        "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32,
        "block_structure": None, "dynamic": False, "actorder": None, "observer": "minmax", "observer_kwargs": {},
    }
    weights.update(overrides.pop("weights", {}))
    quant = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights, "input_activations": None, "output_activations": None}},
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


def test_plan_accepts_int8_and_asymmetric_and_actorder():
    for weights in ({"num_bits": 8}, {"symmetric": False}, {"actorder": "group"}, {"strategy": "channel", "group_size": None}):
        assert compressed_tensors_bnb_plan(_Config(quantization_config = _w4a16(weights = weights))) is not None, weights


def test_plan_accepts_legacy_nested_sparseml_spelling():
    inner = _w4a16()
    inner.pop("quant_method")
    inner["quant_method"] = "sparseml"
    inner["quantization_status"] = "frozen"
    outer = {"quantization_config": inner, "sparsity_config": {"format": "dense"}, "quant_method": "compressed-tensors"}
    plan = compressed_tensors_bnb_plan(_Config(quantization_config = outer))
    assert plan is not None and "config_groups" in plan


def test_plan_declines_other_formats_and_schemes():
    cases = {
        "no config": _Config(),
        "bitsandbytes": _Config(quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True}),
        "fp8": _Config(quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}),
        "float weights": _Config(quantization_config = _w4a16(weights = {"type": "float", "num_bits": 8})),
        "nvfp4 format": _Config(quantization_config = _w4a16(format = "nvfp4-pack-quantized")),
        "activations quantized": _Config(quantization_config = _w4a16()),
        "sparse": _Config(quantization_config = _w4a16(sparsity_config = {"format": "sparse-24-bitmask"})),
        "gptq": _Config(quantization_config = {"quant_method": "gptq", "bits": 4}),
        "no groups": _Config(quantization_config = _w4a16(config_groups = {})),
    }
    cases["activations quantized"].quantization_config["config_groups"]["group_0"]["input_activations"] = {"num_bits": 8, "type": "int"}
    for name, config in cases.items():
        assert compressed_tensors_bnb_plan(config) is None, name


def test_plan_declines_mixed_groups_with_a_float_scheme():
    quant = _w4a16()
    quant["config_groups"]["group_1"] = {
        "targets": ["re:.*self_attn.*"],
        "weights": {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True},
        "input_activations": None, "output_activations": None,
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
        "model.layers.1.self_attn.q_proj.weight",     # this layer's q_proj is stored unpacked
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    plan = packed_weight_dtype_plan(keys)
    assert all(v is None for v in plan.values())
    rx = re.compile("|".join(plan))
    kept = [k for k in [
        "model.layers.0.mlp.experts.12.gate_proj.weight", "model.layers.7.mlp.experts.0.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.1.self_attn.q_proj.weight",
        "model.embed_tokens.weight", "lm_head.weight",
    ] if rx.search(k)]
    assert kept == ["model.layers.0.mlp.experts.12.gate_proj.weight", "model.layers.7.mlp.experts.0.gate_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight"]
    assert packed_weight_dtype_plan(["model.embed_tokens.weight"]) == {}


# ----------------------------------------------------------------------------- config stripping


@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
def test_arm_strips_root_and_subconfigs_and_parks_the_plan():
    from transformers import LlamaConfig

    root = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
    text = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
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

    plain = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
    assert compressed_tensors_prepared_config(plain) is None
    assert compressed_tensors_prepared_config(None) is None
    armed = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
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

    def fake_plan(model_name, *, max_memory = None, **kwargs):
        seen.update(kwargs)
        seen["max_memory"] = max_memory
        return None

    monkeypatch.setattr(planner, "plan_device_map_for_pretrained", fake_plan)
    monkeypatch.setattr(loader_utils, "DEVICE_TYPE_TORCH", "cuda")
    monkeypatch.setattr(loader_utils, "is_distributed", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (10 * 1024 ** 3, 16 * 1024 ** 3))
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

@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
def test_check_and_disable_keeps_4bit_for_packed_int4():
    from transformers import LlamaConfig

    config = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
    config.quantization_config = _w4a16()
    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(config, load_in_4bit = True, load_in_8bit = False, verbose = False)
    assert (load_in_4bit, load_in_8bit, method) == (True, False, None)
    assert not hasattr(config, "quantization_config")


def test_check_and_disable_still_disables_8bit_and_other_methods():
    for quant, flags in (
        (_w4a16(), dict(load_in_4bit = False, load_in_8bit = True)),
        ({"quant_method": "gptq", "bits": 4}, dict(load_in_4bit = True, load_in_8bit = False)),
    ):
        config = _Config(quantization_config = quant)
        load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(config, verbose = False, **flags)
        assert (load_in_4bit, load_in_8bit) == (False, False)
        assert method in ("compressed-tensors", "gptq")
        assert hasattr(config, "quantization_config")


@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
def test_check_and_disable_does_not_arm_when_the_4bit_load_is_not_happening():
    """`fast_inference` hands the packed checkpoint to vLLM and `full_finetuning` turns 4-bit off
    right after this call: arming there stripped the checkpoint's own quantization config with
    nothing left to consume the plan, so the packed tensors loaded as unmatched keys."""
    from transformers import LlamaConfig

    config = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
    config.quantization_config = _w4a16()
    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, load_in_8bit = False, verbose = False, requantize_packed = False
    )
    assert (load_in_4bit, load_in_8bit, method) == (False, False, "compressed-tensors")
    assert hasattr(config, "quantization_config")
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)


@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
def test_arm_leaves_the_config_alone_when_the_quantizer_cannot_be_installed(monkeypatch):
    """A foreign class in the bitsandbytes quantizer slot would load the packed tensors without
    the converters; the config must then keep its own quantization config."""
    from transformers import LlamaConfig
    from unsloth.models import compressed_tensors_bnb

    monkeypatch.setattr(compressed_tensors_bnb, "install_compressed_tensors_bnb_quantizer", lambda: False)
    config = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
    config.quantization_config = _w4a16()
    assert compressed_tensors_bnb.arm_compressed_tensors_bnb_loading(config, verbose = False) is None
    assert hasattr(config, "quantization_config")
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)



@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
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
    config = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16)
    config.quantization_config = _w4a16()
    plan = arm_compressed_tensors_bnb_loading(config, verbose = False)
    assert plan is not None
    bnb = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16, bnb_4bit_quant_type = "nf4")

    def preprocess():
        quantizer = AutoHfQuantizer.from_config(bnb, pre_quantized = False)
        with init_empty_weights():
            model = LlamaForCausalLM(config)
        quantizer._process_model_before_weight_loading(model, dtype = torch.bfloat16, device_map = None, checkpoint_files = [])
        return quantizer, model

    first, _ = preprocess()          # the planner's meta pass
    assert getattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR, None) is plan
    second, model = preprocess()     # the real load
    assert second._unsloth_ct_config is not None
    conversions = second.update_weight_conversions([])
    assert any(
        getattr(c, "source_patterns", None) and "weight_packed$" in c.source_patterns for c in conversions
    ), [getattr(c, "source_patterns", None) for c in conversions]
    second._process_model_after_weight_loading(model)
    assert not hasattr(config, UNSLOTH_COMPRESSED_TENSORS_ATTR)

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


def _write_tiny_packed_llama(root, asymmetric = False, actorder = False, num_bits = 4, group_size = 32):
    """A 2-layer Llama with every Linear but lm_head packed by compressed-tensors' compressor.
    Returns (packed_dir, bf16_dir) where bf16_dir holds the exact decompressed weights."""
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig
    from compressed_tensors.quantization.utils import calculate_qparams

    torch.manual_seed(0)
    config = LlamaConfig(hidden_size = 64, num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
                         intermediate_size = 128, vocab_size = 256, max_position_embeddings = 128, tie_word_embeddings = False)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    quant = _w4a16(weights = {"num_bits": num_bits, "group_size": group_size, "symmetric": not asymmetric, "actorder": "group" if actorder else None})
    ctc = QuantizationConfig.model_validate(quant)
    scheme = list(ctc.config_groups.values())[0]
    comp = BaseCompressor.get_value_from_registry("pack-quantized")
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
        scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), scheme.weights)
        state = {"weight": wf, "weight_scale": scale.to(torch.bfloat16)}
        if asymmetric:
            state["weight_zero_point"] = zp.to(torch.int8)
        if g_idx is not None:
            state["weight_g_idx"] = g_idx
        out = comp.compress(state, scheme)
        for pk, pv in out.items():
            packed[mod + "." + pk] = pv.contiguous()
        dec = comp.decompress({k2: v2 for k2, v2 in out.items()}, scheme)["weight"].to(torch.bfloat16)
        plain[k] = dec.contiguous()
    packed_dir, bf16_dir = os.path.join(root, "packed"), os.path.join(root, "bf16")
    os.makedirs(packed_dir); os.makedirs(bf16_dir)
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
        assert torch.equal(qa.absmax, qb.absmax) and qa.shape == qb.shape and qa.dtype == qb.dtype, name
        n += 1
    return n


def _tokenizer_free_load(path, root):
    """Unsloth's loader wants a tokenizer next to the weights; a tiny one is enough."""
    from transformers import AutoTokenizer

    tok_dir = os.path.join(root, "tok")
    if not os.path.isdir(tok_dir):
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(tok_dir)
    for f in os.listdir(tok_dir):
        if not os.path.exists(os.path.join(path, f)):
            shutil.copy(os.path.join(tok_dir, f), os.path.join(path, f))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
@pytest.mark.parametrize("variant", ["symmetric", "asymmetric", "actorder", "int8"])
def test_packed_checkpoint_loads_as_linear4bit_bit_identical_to_disk_route(variant, tmp_path):
    from unsloth import FastLanguageModel

    packed_dir, bf16_dir = _write_tiny_packed_llama(
        str(tmp_path), asymmetric = variant == "asymmetric", actorder = variant == "actorder",
        num_bits = 8 if variant == "int8" else 4,
    )
    for d in (packed_dir, bf16_dir):
        _tokenizer_free_load(d, str(tmp_path))
    kw = dict(max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True)
    model_a, _ = FastLanguageModel.from_pretrained(packed_dir, **kw)
    model_b, _ = FastLanguageModel.from_pretrained(bf16_dir, **kw)
    n = _same_linear4bit(model_a, model_b)
    assert n == 2 * 7, n   # q k v o gate up down per layer
    assert not any(k.endswith("weight_packed") for k, _ in model_a.named_parameters())
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        assert torch.equal(model_a(input_ids = ids).logits, model_b(input_ids = ids).logits)
    # The stripped checkpoint config never carries the plan or the old quantization config.
    assert not hasattr(model_a.config, UNSLOTH_COMPRESSED_TENSORS_ATTR)
    assert getattr(model_a.config, "quantization_config", None) is not None   # the bitsandbytes one
    assert "compressed" not in str(model_a.config.quantization_config).lower()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
def test_packed_checkpoint_trains_with_lora(tmp_path):
    from unsloth import FastLanguageModel

    packed_dir, _ = _write_tiny_packed_llama(str(tmp_path))
    _tokenizer_free_load(packed_dir, str(tmp_path))
    model, _ = FastLanguageModel.from_pretrained(packed_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True)
    model = FastLanguageModel.get_peft_model(model, r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"])
    model.train()
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
