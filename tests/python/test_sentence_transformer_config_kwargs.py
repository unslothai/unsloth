"""FastSentenceTransformer: config_kwargs reach the config on the training path, and the
FORCE_FLOAT32 family check used by for_inference on GPUs without bfloat16, and float16 pooling."""

import json

import pytest
from transformers import AutoConfig, BertConfig, LlamaConfig, Qwen3MoeConfig

import unsloth  # noqa: F401
from unsloth.models import sentence_transformer as st_mod
from unsloth.models.sentence_transformer import FastSentenceTransformer, _is_force_float32_config


class _Captured(Exception):
    pass


def _tiny_decoder(tmp_path):
    config = LlamaConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path)


def test_config_kwargs_go_to_the_config_not_the_model(tmp_path, monkeypatch):
    path = _tiny_decoder(tmp_path)
    captured = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(st_mod.FastModel, "from_pretrained", fake_from_pretrained)
    with pytest.raises(_Captured):
        FastSentenceTransformer.from_pretrained(
            path, config_kwargs = {"num_hidden_layers": 1}, local_files_only = True
        )
    assert "config_kwargs" not in captured
    assert captured["config"].num_hidden_layers == 1


def test_explicit_config_wins_over_config_kwargs(tmp_path, monkeypatch):
    path = _tiny_decoder(tmp_path)
    explicit = AutoConfig.from_pretrained(path)
    captured = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(st_mod.FastModel, "from_pretrained", fake_from_pretrained)
    with pytest.raises(_Captured):
        FastSentenceTransformer.from_pretrained(
            path,
            config = explicit,
            config_kwargs = {"num_hidden_layers": 1},
            local_files_only = True,
        )
    assert captured["config"] is explicit


def test_without_config_kwargs_nothing_is_injected(tmp_path, monkeypatch):
    path = _tiny_decoder(tmp_path)
    captured = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(st_mod.FastModel, "from_pretrained", fake_from_pretrained)
    with pytest.raises(_Captured):
        FastSentenceTransformer.from_pretrained(path, local_files_only = True)
    assert "config" not in captured
    assert "config_kwargs" not in captured


def test_hub_keys_inside_config_kwargs_do_not_collide(tmp_path, monkeypatch):
    path = _tiny_decoder(tmp_path)
    captured = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        raise _Captured

    monkeypatch.setattr(st_mod.FastModel, "from_pretrained", fake_from_pretrained)
    with pytest.raises(_Captured):
        FastSentenceTransformer.from_pretrained(
            path,
            config_kwargs = {"trust_remote_code": False, "token": None, "num_hidden_layers": 2},
            local_files_only = True,
        )
    assert captured["config"].num_hidden_layers == 2


def test_force_float32_family_entries_are_normalised(monkeypatch):
    # An entry spelled with "-" or "_" must still match the normalised model type.
    import unsloth.models.loader as loader_mod
    from transformers import PretrainedConfig

    config = PretrainedConfig()
    config.model_type = "foobar"
    monkeypatch.setattr(loader_mod, "FORCE_FLOAT32", ["foo-bar"])
    assert _is_force_float32_config(config)


def test_force_float32_family_check():
    assert _is_force_float32_config(Qwen3MoeConfig())
    assert not _is_force_float32_config(BertConfig())
    assert not _is_force_float32_config(None)


def test_float16_mean_pooling_does_not_overflow():
    import torch
    from sentence_transformers.models import Pooling

    pooling = Pooling(word_embedding_dimension = 8, pooling_mode = "mean")
    # 2000 tokens at 4000: the float16 sum is 8e6, far past float16's 65504.
    tokens = torch.full((1, 2000, 8), 4000.0, dtype = torch.float16)
    features = {"token_embeddings": tokens, "attention_mask": torch.ones(1, 2000, dtype = torch.long)}
    out = pooling(features)
    assert torch.isfinite(out["sentence_embedding"]).all()
    assert torch.allclose(out["sentence_embedding"].float(), torch.full((1, 8), 4000.0))
    # The caller's token embeddings are handed back untouched.
    assert out["token_embeddings"] is tokens


def test_float16_pooling_with_batch_encoding_features():
    # SentenceTransformer.encode passes a BatchEncoding (UserDict), not a dict.
    import torch
    from transformers import BatchEncoding
    from sentence_transformers.models import Pooling

    pooling = Pooling(word_embedding_dimension = 8, pooling_mode = "mean")
    tokens = torch.full((1, 2000, 8), 4000.0, dtype = torch.float16)
    features = BatchEncoding(
        {"token_embeddings": tokens, "attention_mask": torch.ones(1, 2000, dtype = torch.long)}
    )
    out = pooling(features)
    assert torch.isfinite(out["sentence_embedding"]).all()


def test_float16_dense_head_after_float32_pooling():
    # EmbeddingGemma on T4: float16 Dense head after the float32 pooled vector, autocast off.
    import torch
    from sentence_transformers.models import Dense, Normalize, Pooling

    pooling = Pooling(word_embedding_dimension = 8, pooling_mode = "mean")
    torch.manual_seed(0)
    dense = Dense(
        in_features = 8, out_features = 4, bias = False, activation_function = torch.nn.Identity()
    ).half()
    tokens = torch.randn(2, 5, 8, dtype = torch.float16)
    features = {"token_embeddings": tokens, "attention_mask": torch.ones(2, 5, dtype = torch.long)}
    out = Normalize()(dense(pooling(features)))
    assert out["sentence_embedding"].shape == (2, 4)
    assert torch.isfinite(out["sentence_embedding"]).all()
    want = dense.linear(tokens.float().mean(1).half())
    torch.testing.assert_close(
        out["sentence_embedding"].float(),
        torch.nn.functional.normalize(want.float(), dim = -1),
        atol = 2e-3,
        rtol = 2e-3,
    )


def test_bfloat16_pooling_is_unchanged():
    import torch
    from sentence_transformers.models import Pooling

    pooling = Pooling(word_embedding_dimension = 8, pooling_mode = "mean")
    tokens = torch.randn(2, 5, 8, dtype = torch.bfloat16)
    out = pooling(
        {"token_embeddings": tokens, "attention_mask": torch.ones(2, 5, dtype = torch.long)}
    )
    assert out["sentence_embedding"].dtype == torch.bfloat16


def test_embedding_gemma2_text_only_is_force_float32():
    try:
        from transformers import EmbeddingGemma2Config
    except ImportError:
        pytest.skip("transformers has no EmbeddingGemma2")
    # Dropping the towers removes the gemma4 sub-configs that used to match by accident.
    config = EmbeddingGemma2Config(vision_config = None, audio_config = None)
    assert _is_force_float32_config(config)
