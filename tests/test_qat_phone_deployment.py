# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QAT scheme construction on tiny CPU models."""

from __future__ import annotations

import importlib.util
import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

from unsloth.models._utils import (
    TorchAOConfig,
    _convert_torchao_model,
    _prepare_model_for_qat,
)


def _tiny_gemma3():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size = 256,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        tie_word_embeddings = True,
    )
    return Gemma3ForCausalLM(config).eval()


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
    )
    return Qwen3ForCausalLM(config).eval()


def _is_fake_quantized(module):
    return hasattr(module, "weight_fake_quantizer")


def test_phone_deployment_alias_in_prepare():
    model = _prepare_model_for_qat(_tiny_qwen3(), "phone-deployment")
    assert model._torchao_config.qat_scheme == "int8-int4"
    assert _is_fake_quantized(model.model.layers[0].self_attn.q_proj)
    assert _is_fake_quantized(model.model.embed_tokens)


def test_gemma3_int8_int4_keeps_phone_scheme():
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    assert model._torchao_config.qat_scheme == "int8-int4"
    configs = [c for c, _ in model._torchao_config.base_config_and_filter_fns]
    assert any(isinstance(c, Int8DynamicActivationIntxWeightConfig) for c in configs)
    assert _is_fake_quantized(model.model.layers[0].self_attn.q_proj)


def _int8_embedding_config():
    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis
    return TorchAOConfig(
        qat_scheme = None,
        base_config_and_filter_fns = [
            (
                IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)),
                lambda m, _: isinstance(m, torch.nn.Embedding),
            )
        ],
    )


@pytest.mark.parametrize("scheme", ["int8-int4", "custom"])
@pytest.mark.parametrize("convert", ["none", "unsloth", "manual"])
def test_gemma3_fake_quantized_embedding_keeps_embed_scale(scheme, convert):
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    model = _tiny_gemma3()
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    with torch.no_grad():
        ref = model.get_input_embeddings()(ids)
    assert float(model.get_input_embeddings().embed_scale) == 8.0

    model = _prepare_model_for_qat(
        model, "int8-int4" if scheme == "int8-int4" else _int8_embedding_config()
    )
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb)
    if convert == "unsloth":
        _convert_torchao_model(model)
    elif convert == "manual":
        quantize_(model, QATConfig(step = "convert"))
    emb = model.get_input_embeddings()
    assert isinstance(emb, Gemma3TextScaledWordEmbedding)
    with torch.no_grad():
        out = emb(ids)
    # int8 per-row error is ~0.4% of the row max; a lost sqrt(hidden_size) = 8x scale is not.
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel < 0.02, float(rel)


def test_int4_skips_linears_not_divisible_by_group_size():
    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ok = torch.nn.Linear(256, 64, bias = False)
            self.odd = torch.nn.Linear(192, 64, bias = False)

        def forward(self, x):
            return self.ok(torch.nn.functional.pad(x, (0, 64))) + self.odd(x)

    model = _M().to(torch.bfloat16)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "int4")
    assert _is_fake_quantized(model.ok)
    assert not _is_fake_quantized(model.odd)
    assert any("odd" in str(w.message) for w in caught)
    model(torch.randn(2, 192, dtype = torch.bfloat16))


@pytest.mark.parametrize("qat_scheme", ["int8-int4", "int4", "int8", "fp8-fp8"])
def test_lora_adapters_are_not_fake_quantized(qat_scheme, monkeypatch):
    from peft import LoraConfig, get_peft_model

    monkeypatch.delenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", raising = False)
    model = get_peft_model(
        _tiny_qwen3(),
        LoraConfig(r = 32, lora_alpha = 32, target_modules = ["q_proj", "o_proj"]),
    )
    model = _prepare_model_for_qat(model, qat_scheme)
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj
    assert _is_fake_quantized(q_proj.base_layer)
    assert not _is_fake_quantized(q_proj.lora_A["default"])
    assert not _is_fake_quantized(q_proj.lora_B["default"])


def test_lora_fake_quant_opt_out(monkeypatch):
    from peft import LoraConfig, get_peft_model

    monkeypatch.setenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", "1")
    model = get_peft_model(
        _tiny_qwen3(), LoraConfig(r = 32, lora_alpha = 32, target_modules = ["q_proj"])
    )
    model = _prepare_model_for_qat(model, "int8-int4")
    q_proj = model.base_model.model.model.layers[0].self_attn.q_proj
    assert _is_fake_quantized(q_proj.lora_A["default"])


class _EoiEmbedding(torch.nn.Embedding):
    # T5Gemma2TextScaledWordEmbedding pattern: an extra registered parameter read in forward.
    def __init__(self, num, dim):
        super().__init__(num, dim)
        self.eoi_embedding = torch.nn.Parameter(torch.ones(dim))

    def forward(self, input_ids):
        out = super().forward(input_ids)
        out[input_ids == 0] = self.eoi_embedding.to(out.dtype)
        return out


@pytest.mark.parametrize("convert", [False, True])
def test_embedding_subclass_keeps_extra_parameters(convert):
    model = _tiny_gemma3()
    model.model.embed_tokens = _EoiEmbedding(model.config.vocab_size, model.config.hidden_size)
    eoi = model.model.embed_tokens.eoi_embedding
    model = _prepare_model_for_qat(model, "int8-int4")
    if convert:
        _convert_torchao_model(model)
    emb = model.get_input_embeddings()
    assert isinstance(emb, _EoiEmbedding)
    assert emb.eoi_embedding is eoi
    with torch.no_grad():
        out = emb(torch.tensor([[0, 1, 2]]))
    assert torch.equal(out[0, 0], eoi)


def test_compiled_fake_quantized_embedding_pickles():
    import pickle

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    emb.compile()
    restored = pickle.loads(pickle.dumps(emb))
    assert type(restored) is type(emb)
    assert "_compiled_call_impl" not in vars(restored) or restored._compiled_call_impl is None


def test_fake_quantized_embedding_subclass_pickles():
    import io
    import pickle

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    assert _is_fake_quantized(emb)
    restored = pickle.loads(pickle.dumps(emb))
    assert type(restored) is type(emb)
    buf = io.BytesIO()
    torch.save(emb, buf)
    buf.seek(0)
    assert type(torch.load(buf, weights_only = False)) is type(emb)


def test_fake_quantized_embedding_unpickles_in_fresh_process(tmp_path):
    import os
    import pickle
    import subprocess
    import sys

    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    emb = model.get_input_embeddings()
    path = tmp_path / "emb.pkl"
    path.write_bytes(pickle.dumps(emb))
    ids = torch.tensor([[1, 2, 3]])
    expected = emb(ids)
    code = (
        "import pickle, sys, torch\n"
        f"emb = pickle.loads(open({str(path)!r}, 'rb').read())\n"
        "print(type(emb).__name__, float(emb.embed_scale))\n"
        "torch.save(emb(torch.tensor([[1, 2, 3]])), sys.argv[1])\n"
    )
    out_path = tmp_path / "out.pt"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.getcwd(), env.get("PYTHONPATH", "")])
    res = subprocess.run(
        [sys.executable, "-c", code, str(out_path)],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    assert res.stdout.split()[-2] == type(emb).__name__
    assert float(res.stdout.split()[-1]) == float(emb.embed_scale)
    assert torch.equal(torch.load(out_path), expected)


def test_builtin_recipe_flag_only_for_scheme_names():
    model = _prepare_model_for_qat(_tiny_gemma3(), "int8-int4")
    assert model._torchao_config._unsloth_builtin_recipe is True
    custom = _int8_embedding_config()
    model = _prepare_model_for_qat(_tiny_gemma3(), custom)
    assert not getattr(model._torchao_config, "_unsloth_builtin_recipe", False)
