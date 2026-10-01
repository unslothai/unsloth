# SPDX-License-Identifier: Apache-2.0
"""Gemma's patched SDPA attention must receive masks before ST can flatten inputs."""

import inspect

import pytest
import torch

from real_accelerator import has_real_accelerator


@pytest.mark.parametrize("loader_backend", ["sdpa", "flash_attention_2"])
def test_patched_gemma_loader_preserves_sequence_isolation(tmp_path, loader_backend):
    if not has_real_accelerator() or not torch.cuda.is_available():
        pytest.skip("real Gemma loader requires CUDA")
    pytest.importorskip("sentence_transformers")
    from transformers.utils import is_flash_attn_2_available

    if loader_backend == "flash_attention_2" and not is_flash_attn_2_available():
        pytest.skip("regression exercises an installed FA2 loader before the SDPA safety fallback")
    from unsloth import FastSentenceTransformer
    from transformers import Gemma3TextConfig, Gemma3TextModel, PreTrainedTokenizerFast
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    config = Gemma3TextConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 48,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        max_position_embeddings = 32,
        sliding_window = 4,
        layer_types = ["sliding_attention", "full_attention"],
        use_bidirectional_attention = True,
        use_cache = False,
        pad_token_id = 0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(4460)
    checkpoint = tmp_path / "base"
    Gemma3TextModel(config).save_pretrained(checkpoint)
    vocab = {f"word{i}": i for i in range(64)}
    del vocab["word0"], vocab["word1"]
    vocab.update({"[PAD]": 0, "[UNK]": 1})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        pad_token = "[PAD]",
        unk_token = "[UNK]",
        model_max_length = 32,
    ).save_pretrained(checkpoint)
    options = (
        "model_kwargs"
        if "model_kwargs" in inspect.signature(Transformer).parameters
        else "model_args"
    )
    transformer = Transformer(str(checkpoint), **{options: {"attn_implementation": "sdpa"}})
    SentenceTransformer(modules = [transformer, Pooling(32)], device = "cpu").save_pretrained(
        tmp_path / "sentence"
    )
    model = FastSentenceTransformer.from_pretrained(
        str(tmp_path / "sentence"),
        dtype = torch.bfloat16,
        load_in_4bit = False,
        full_finetuning = True,
        max_seq_length = 32,
        use_gradient_checkpointing = False,
        attn_implementation = loader_backend,
    )
    base = model[0].auto_model
    if not base.layers[0].self_attn.forward.__module__.startswith(
        "unsloth_zoo.temporary_patches.gemma"
    ):
        pytest.skip("installed Zoo does not use the affected Gemma attention implementation")
    if loader_backend == "sdpa":
        from unsloth.models.sentence_transformer import _ensure_sentence_attention_masks

        # Exercise the backend repair without requiring an FA extension to load.
        base.config._attn_implementation = "flash_attention_2"
        assert _ensure_sentence_attention_masks(base)
        if hasattr(model[0], "unpad_inputs"):
            model[0].unpad_inputs = model[0].unpad_inputs
    assert base.config._attn_implementation == "sdpa"
    assert base.config.use_bidirectional_attention is True
    assert not getattr(model[0], "can_flatten_inputs", False)
    base.config.use_cache = False
    masks = []
    handle = base.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args, kwargs: masks.append(kwargs["attention_mask"].detach()),
        with_kwargs = True,
    )
    features = {
        "input_ids": torch.tensor([[5, 6, 7, 0, 0, 0], [8, 9, 10, 11, 12, 13]], device = "cuda"),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], device = "cuda"),
    }
    for training in (False, True):
        model.train(training)
        altered = {key: value.clone() for key, value in features.items()}
        altered["input_ids"][1] = 14
        with torch.autocast("cuda", dtype = torch.bfloat16):
            expected = model({key: value.clone() for key, value in features.items()})[
                "sentence_embedding"
            ]
            actual = model(altered)["sentence_embedding"]
        torch.testing.assert_close(actual[0], expected[0], rtol = 0, atol = 0)
        assert not torch.equal(actual[1], expected[1])
        model.zero_grad(set_to_none = True)
        actual[0].float().square().sum().backward()
        # Token 14 appears only in the unrelated row.
        assert torch.count_nonzero(base.get_input_embeddings().weight.grad[14]) == 0
    assert all(mask.ndim == 4 for mask in masks)
    mask = masks[0]
    allowed = mask if mask.dtype == torch.bool else mask == 0
    assert allowed[0, 0, 0, 1]  # Future token is visible in bidirectional attention.
    assert not allowed[0, 0, 0, 3]  # Padding is never visible.
    assert not allowed[1, 0, 0, 5]  # Out of the sliding window.
    handle.remove()
