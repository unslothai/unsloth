# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A Linear an FP8 checkpoint stores in bf16 must stay a plain Linear. No GPU needed.

stepfun-ai/Step-3.7-Flash-FP8 stores its vision projector `vit_large_projector.weight` in bf16
with no `weight_scale_inv` and lists it in `modules_to_not_convert` under that checkpoint name.
transformers renames skip-list entries with the checkpoint key renames, but those are written for
parameter keys (`^vit_large_projector\\.`, trailing dot), so the bare module name is never mapped
to `model.multi_modal_projector`. The projector became an FP8Linear holding a bf16 weight and an
uninitialised scale ("weight_scale_inv MISSING"), and the first forward died in Triton with
"Unsupported lhs dtype fp8e4nv". The fix reads the checkpoint's own dtypes (shard headers; the
index can omit tensors) and keeps every stored-bf16 Linear unconverted.

Every test drives the real FineGrainedFP8HfQuantizer against real models built on the meta
device and real (tiny) safetensors files carrying the checkpoint's tensor names.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

import transformers  # noqa: E402

try:
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    from transformers.utils.quantization_config import FineGrainedFP8Config
except Exception:  # pragma: no cover
    pytest.skip("no FineGrainedFP8 quantizer in this transformers", allow_module_level = True)

if not hasattr(torch, "float8_e4m3fn"):  # pragma: no cover
    pytest.skip("torch without float8", allow_module_level = True)

try:
    from transformers.quantizers.quantizers_utils import should_convert_module  # noqa: F401
    def exact(name):
        import re
        return re.escape(name) + "$"
except Exception:  # transformers 4.x matches skip entries as substrings

    def exact(name):
        return name


from unsloth.import_fixes import (  # noqa: E402
    _fp8_checkpoint_tensor_dtypes,
    _fp8_unscaled_linear_patterns,
    fix_transformers_fp8_unscaled_checkpoint_linears,
)

F8 = torch.float8_e4m3fn


def _write(path, tensors):
    safetensors_torch.save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    return str(path)


def _fp8(*shape):
    return torch.zeros(*shape).to(F8)


def _quantizer(skip):
    config = FineGrainedFP8Config(weight_block_size = [128, 128], modules_to_not_convert = skip)
    return FineGrainedFP8HfQuantizer(config, pre_quantized = True), config


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 512,
    )
    with torch.device("meta"):
        return LlamaForCausalLM(config)


def _llama_checkpoint(tmp_path, bf16_module):
    tensors = {}
    for name, module in _llama().named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            if name == bf16_module:
                tensors[name + ".weight"] = torch.zeros(2, 2, dtype = torch.bfloat16)
            else:
                tensors[name + ".weight"] = _fp8(2, 2)
                tensors[name + ".weight_scale_inv"] = torch.ones(1, 1)
    tensors["lm_head.weight"] = torch.zeros(2, 2, dtype = torch.bfloat16)
    return _write(tmp_path / "model.safetensors", tensors)


def _preprocess(model, skip, files):
    quantizer, config = _quantizer(list(skip))
    quantizer.preprocess_model(model = model, dtype = torch.bfloat16, checkpoint_files = files)
    return {n: type(m).__name__ for n, m in model.named_modules()}, config


BF16 = "model.layers.1.self_attn.o_proj"


def test_header_dtypes_mark_the_bf16_linear(tmp_path):
    files = [_llama_checkpoint(tmp_path, BF16)]
    dtypes = _fp8_checkpoint_tensor_dtypes(files, None)
    assert dtypes[BF16 + ".weight"] == "BF16"
    assert set(_fp8_unscaled_linear_patterns(_llama(), dtypes)) == {exact("lm_head"), exact(BF16)}


def test_bf16_linear_stays_linear_and_skip_list_is_restored(tmp_path):
    files = [_llama_checkpoint(tmp_path, BF16)]
    fix_transformers_fp8_unscaled_checkpoint_linears()
    types, config = _preprocess(_llama(), ["lm_head"], files)
    assert types[BF16] == "Linear"
    assert types["model.layers.0.self_attn.o_proj"] == "FP8Linear"
    assert types["model.layers.1.self_attn.q_proj"] == "FP8Linear"
    # The saved config keeps the checkpoint's own list, not the derived patterns.
    assert config.modules_to_not_convert == ["lm_head"]


def test_not_an_fp8_checkpoint_changes_nothing(tmp_path):
    files = [
        _write(
            tmp_path / "model.safetensors", {"a.weight": torch.zeros(2, 2, dtype = torch.bfloat16)}
        )
    ]
    assert _fp8_unscaled_linear_patterns(_llama(), _fp8_checkpoint_tensor_dtypes(files, None)) == []


def test_index_only_falls_back_to_missing_scale(tmp_path):
    import json

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "x.safetensors",
                    "model.layers.0.self_attn.q_proj.weight_scale_inv": "x.safetensors",
                    "model.layers.0.self_attn.k_proj.weight": "x.safetensors",
                }
            }
        )
    )
    dtypes = _fp8_checkpoint_tensor_dtypes([str(tmp_path / "x.safetensors")], None)
    assert dtypes and all(v is None for v in dtypes.values())
    assert _fp8_unscaled_linear_patterns(_llama(), dtypes) == [
        exact("model.layers.0.self_attn.k_proj")
    ]


Step3p7 = getattr(transformers, "Step3p7ForConditionalGeneration", None)


def _step():
    config = transformers.Step3p7Config()
    config.text_config.num_hidden_layers = 1
    with torch.device("meta"):
        return Step3p7(config)


def _step_checkpoint(tmp_path):
    # The real checkpoint's names: projector bf16 without a scale, experts FP8 with one.
    return [
        _write(
            tmp_path / "model.safetensors",
            {
                "vit_large_projector.weight": torch.zeros(2, 2, dtype = torch.bfloat16),
                "model.layers.0.self_attn.q_proj.weight": _fp8(2, 2),
                "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(1, 1),
            },
        )
    ]


@pytest.mark.skipif(Step3p7 is None, reason = "no native step3p7 in this transformers")
def test_step37_config_skip_list_alone_converts_the_projector(tmp_path):
    # The defect on the installed transformers: its own path, with the fix bypassed.
    fix_transformers_fp8_unscaled_checkpoint_linears()
    patched = FineGrainedFP8HfQuantizer._process_model_before_weight_loading
    quantizer, _ = _quantizer(["lm_head", "vit_large_projector"])
    model = _step()
    patched.__wrapped__(quantizer, model, checkpoint_files = _step_checkpoint(tmp_path))
    assert type(model.model.multi_modal_projector).__name__ == "FP8Linear"


@pytest.mark.skipif(Step3p7 is None, reason = "no native step3p7 in this transformers")
def test_step37_projector_under_its_checkpoint_name(tmp_path):
    files = _step_checkpoint(tmp_path)
    fix_transformers_fp8_unscaled_checkpoint_linears()
    types, config = _preprocess(_step(), ["lm_head", "vit_large_projector"], files)
    assert types["model.multi_modal_projector"] == "Linear"
    assert types["model.language_model.layers.0.self_attn.q_proj"] == "FP8Linear"
    assert "vit_large_projector" in config.modules_to_not_convert


def test_fp8_forward_falls_back_for_a_bf16_weight():
    try:
        import unsloth.kernels.fp8 as fp8
    except ImportError as e:  # environment (e.g. vLLM refusing transformers 4.x), not this fix
        pytest.skip(f"unsloth.kernels.fp8 not importable: {e}")
    forward = fp8.module_forward_patch(
        lambda *a: (_ for _ in ()).throw(AssertionError("fp8 kernel")), "weight_scale_inv"
    )
    layer = torch.nn.Linear(4, 3, bias = True)
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1))
    x = torch.randn(2, 4)
    torch.testing.assert_close(forward(layer, x), layer(x))
