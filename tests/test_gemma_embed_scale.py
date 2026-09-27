# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma / Gemma2 inputs are scaled by sqrt(hidden_size) exactly once, on every transformers version."""

import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from unsloth.models._utils import embedding_applies_scale
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")


class _Scaled(torch.nn.Embedding):
    def __init__(self):
        super().__init__(4, 2)
        self.register_buffer("embed_scale", torch.tensor(2.0), persistent = False)


def test_embedding_applies_scale_detection():
    assert embedding_applies_scale(torch.nn.Embedding(4, 2)) is False
    assert embedding_applies_scale(_Scaled()) is True
    assert embedding_applies_scale(None) is False
    assert embedding_applies_scale(types.SimpleNamespace(base_layer = _Scaled())) is True
    assert embedding_applies_scale(types.SimpleNamespace(original_module = _Scaled())) is True
    assert (
        embedding_applies_scale(types.SimpleNamespace(base_layer = torch.nn.Embedding(4, 2))) is False
    )


@pytest.mark.parametrize("arch", ["gemma", "gemma2"])
def test_detection_tracks_installed_transformers(arch):
    module = pytest.importorskip(f"transformers.models.{arch}.modeling_{arch}")
    config_cls = getattr(module, "GemmaConfig" if arch == "gemma" else "Gemma2Config", None)
    if config_cls is None:
        from transformers import GemmaConfig, Gemma2Config
        config_cls = GemmaConfig if arch == "gemma" else Gemma2Config
    config = config_cls(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 8,
    )
    model = getattr(module, "GemmaModel" if arch == "gemma" else "Gemma2Model")(config)
    ids = torch.tensor([[1, 2, 3]])
    raw = torch.nn.functional.embedding(ids, model.embed_tokens.weight)
    scales_itself = not torch.equal(model.embed_tokens(ids), raw)
    assert embedding_applies_scale(model.embed_tokens) is scales_itself


def _tiny_checkpoint(path, repo):
    # The hub tiny Gemma uses head_dim 2, which no xformers kernel takes; keep its config and tokenizer, widen heads.
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(repo)
    config.update(
        dict(
            hidden_size = 64,
            intermediate_size = 128,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 32,
        )
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    AutoTokenizer.from_pretrained(repo).save_pretrained(path)
    return str(path)


@pytest.mark.skipif(
    not has_real_cuda(), reason = "loads tiny Gemma checkpoints through FastLanguageModel on CUDA"
)
@pytest.mark.parametrize(
    "repo, module_name",
    [
        ("trl-internal-testing/tiny-GemmaForCausalLM", "unsloth.models.gemma"),
        ("trl-internal-testing/tiny-Gemma2ForCausalLM", "unsloth.models.gemma2"),
    ],
)
def test_prefill_and_decode_scale_once(monkeypatch, tmp_path, repo, module_name):
    import importlib
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        _tiny_checkpoint(tmp_path, repo), max_seq_length = 64, load_in_4bit = False, dtype = torch.float32
    )
    FastLanguageModel.for_inference(model)
    hidden = model.config.hidden_size
    weight = model.get_input_embeddings().weight.detach()
    expected = lambda ids: torch.nn.functional.embedding(ids, weight) * torch.tensor(
        hidden**0.5, dtype = weight.dtype
    )

    prefill = {}

    def capture(mod, args, kwargs):
        prefill.setdefault("x", (args[0] if args else kwargs["hidden_states"]).detach())

    model.model.layers[0].register_forward_pre_hook(capture, with_kwargs = True)
    decode = []
    mod = importlib.import_module(module_name)
    original = mod.fast_rms_layernorm_inference_gemma
    # The first norm of each decode step sees the scaled embeddings (residual stream input).
    monkeypatch.setattr(
        mod,
        "fast_rms_layernorm_inference_gemma",
        lambda ln, x, *a, **k: (decode.append(x.detach().clone()), original(ln, x, *a, **k))[1],
    )
    ids = torch.tensor([[2, 10, 11, 12, 13]], device = "cuda")
    out = model.generate(
        input_ids = ids, attention_mask = torch.ones_like(ids), max_new_tokens = 2, do_sample = False
    )
    torch.testing.assert_close(prefill["x"], expected(ids), rtol = 1e-5, atol = 1e-5)
    assert decode, "decode path did not run"
    torch.testing.assert_close(
        decode[0], expected(out[:, ids.shape[1] : ids.shape[1] + 1]), rtol = 1e-5, atol = 1e-5
    )
