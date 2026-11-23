# SPDX-License-Identifier: AGPL-3.0-only

import importlib.util
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")
pytest.importorskip("datasets")


@pytest.fixture(scope = "module")
def raw_text():
    path = Path(__file__).resolve().parents[1] / "unsloth" / "dataprep" / "raw_text.py"
    spec = importlib.util.spec_from_file_location("raw_text_special_tokens", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tokenizer():
    vocabulary = {
        "[UNK]": 0,
        "[EOS]": 1,
        "[BOS]": 2,
        "[USER]": 3,
        "[ASSISTANT]": 4,
        "alpha": 5,
        "beta": 6,
        "gamma": 7,
        "delta": 8,
        ".": 9,
        "?": 10,
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token = "[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object = backend,
        unk_token = "[UNK]",
        eos_token = "[EOS]",
        bos_token = "[BOS]",
        additional_special_tokens = ["[USER]", "[ASSISTANT]"],
        clean_up_tokenization_spaces = True,
    )


@pytest.mark.parametrize("chunk_size", [3, 4, 64])
@pytest.mark.parametrize("stride", [0, 1])
def test_text_chunks_preserve_the_same_tokens_as_tokenized_chunks(
    raw_text, tokenizer, chunk_size, stride
):
    text = "[BOS] alpha . [USER] beta ? [EOS] [ASSISTANT] gamma . delta"
    loader = raw_text.RawTextDataLoader(tokenizer, chunk_size = chunk_size, stride = stride)
    tokenized = loader.chunk_text(text, return_tokenized = True)
    texts = loader.chunk_text(text, return_tokenized = False)

    assert len(texts) == len(tokenized)
    assert all(chunk.removesuffix("[EOS]").strip() in text for chunk in texts)
    assert [tokenizer(chunk, add_special_tokens = False)["input_ids"] for chunk in texts] == [
        chunk["input_ids"] for chunk in tokenized
    ]


def test_raw_file_text_dataset_keeps_chat_role_tokens(raw_text, tokenizer, tmp_path):
    text = "[BOS] alpha [USER] beta [EOS] [ASSISTANT] gamma delta"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding = "utf-8")
    loader = raw_text.RawTextDataLoader(
        tokenizer,
        chunk_size = 4,
        stride = 0,
        return_tokenized = False,
    )

    dataset = loader.load_from_file(path)
    output = " ".join(dataset["text"])

    for token in ("[BOS]", "[USER]", "[ASSISTANT]"):
        assert token in output
    assert output.count("[EOS]") == 2


@pytest.fixture
def byte_tokenizer():
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {symbol: index for index, symbol in enumerate(alphabet)}
    vocabulary["[EOS]"] = len(vocabulary)
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocabulary, []))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(tokenizer_object = backend, eos_token = "[EOS]")


@pytest.mark.parametrize("stride", [0, 2])
def test_text_chunks_cut_between_characters_not_inside_one(raw_text, byte_tokenizer, stride):
    tokenizer = byte_tokenizer
    text = "日本語のテキスト 😀 naïve café " * 4
    loader = raw_text.RawTextDataLoader(tokenizer, chunk_size = 8, stride = stride)
    chunks = loader.chunk_text(text, return_tokenized = False)

    assert len(chunks) > 4
    assert not any("�" in chunk for chunk in chunks)
    assert chunks[-1].endswith("[EOS]")
    lengths = [len(tokenizer(chunk, add_special_tokens = False)["input_ids"]) for chunk in chunks]
    assert max(lengths) <= 8
    if stride == 0:
        assert "".join(chunks) == text + "[EOS]"

    # Lone-EOS window also starts on a character edge.
    loader = raw_text.RawTextDataLoader(tokenizer, chunk_size = 6, stride = 4)
    chunks = loader.chunk_text("aébb é語\né", return_tokenized = False)
    assert chunks[-1] == "\né[EOS]" and not any("�" in chunk for chunk in chunks)


def test_text_chunks_keep_literal_replacement_characters(raw_text, byte_tokenizer):
    loader = raw_text.RawTextDataLoader(byte_tokenizer, chunk_size = 4, stride = 0)
    assert loader.chunk_text("a\ufffd\ufffd", return_tokenized = False) == ["a\ufffd", "\ufffd[EOS]"]


def test_text_chunks_keep_overlap_when_a_cut_moves(raw_text, byte_tokenizer):
    loader = raw_text.RawTextDataLoader(byte_tokenizer, chunk_size = 12, stride = 10)
    chunks = loader.chunk_text("日本語日本語", return_tokenized = False)
    assert chunks == ["日本語日", "本語日本", "語日本語", "日本語[EOS]"]
