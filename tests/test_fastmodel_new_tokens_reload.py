# SPDX-License-Identifier: AGPL-3.0-only
"""FastModel trains and saves the rows add_new_tokens adds, so a reload keeps them (#1343)."""

import pytest

# Padded vocab (151936 rows, 151669 tokens): new tokens fit without a resize, so PEFT saves no embedding by itself.
TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
NEW_TOKENS = ["<|tok_a|>", "<|tok_b|>", "<|tok_c|>"]


def test_fastmodel_keeps_added_token_rows_across_reload(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from unsloth import FastModel, add_new_tokens

    model, tokenizer = FastModel.from_pretrained(TINY, max_seq_length = 128, load_in_4bit = False)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    old_len = len(tokenizer)
    add_new_tokens(model, tokenizer, NEW_TOKENS)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 8)

    assert {"embed_tokens", "lm_head"} <= {
        m.rsplit(".", 1)[-1] for m in (model.peft_config["default"].modules_to_save or ())
    }
    rows = slice(old_len, old_len + len(NEW_TOKENS))
    with torch.no_grad():
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            emb.weight[rows] += 0.5
    saved_in = model.get_input_embeddings().weight[rows].detach().float().cpu().clone()
    saved_out = model.get_output_embeddings().weight[rows].detach().float().cpu().clone()
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    del model
    torch.cuda.empty_cache()

    model2, _ = FastModel.from_pretrained(str(tmp_path), max_seq_length = 128, load_in_4bit = False)
    torch.testing.assert_close(
        model2.get_input_embeddings().weight[rows].detach().float().cpu(), saved_in
    )
    torch.testing.assert_close(
        model2.get_output_embeddings().weight[rows].detach().float().cpu(), saved_out
    )


def test_fastmodel_leaves_trainable_token_indices_alone():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from unsloth import FastModel, add_new_tokens

    model, tokenizer = FastModel.from_pretrained(TINY, max_seq_length = 128, load_in_4bit = False)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    old_len = len(tokenizer)
    add_new_tokens(model, tokenizer, NEW_TOKENS)
    indices = list(range(old_len, old_len + len(NEW_TOKENS)))
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 8, trainable_token_indices = indices)

    assert not model.peft_config["default"].modules_to_save
    assert not any(type(m).__name__ == "ModulesToSaveWrapper" for m in model.modules())
