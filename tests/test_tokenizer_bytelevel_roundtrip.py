# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""transformers v5 rebuilds a tokenizer declared as LlamaTokenizerFast from tokenizer.json's vocab and
merges only, then installs Llama's SentencePiece Metaspace pre_tokenizer and decoder. For a byte-level BPE
vocab (Mistral-Large-3, Step-3.7-Flash) that drops every space. The Unsloth load path must rebuild the
backend from tokenizer.json, and must leave tokenizers that already round-trip untouched."""

import json

import pytest
from packaging.version import Version

import transformers
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer

import unsloth.tokenizer_utils as tu

TRANSFORMERS_V5 = Version(transformers.__version__).major >= 5
PROBE = "Hello world! def f(x): return x  # code"
CORPUS = [
    "Hello world! This is a tiny corpus for a test tokenizer.",
    "def f(x): return x  # code",
    "The quick brown fox jumps over the lazy dog.",
] * 20
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{{ eos_token }}"
    "{% endfor %}{% if add_generation_prompt %}[assistant] {% endif %}"
)


def _write_dir(path, tokenizer, tokenizer_class):
    path.mkdir(parents = True, exist_ok = True)
    tokenizer.save(str(path / "tokenizer.json"))
    config = {
        "tokenizer_class": tokenizer_class,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": False,
        "legacy": True,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 128,
        "chat_template": CHAT_TEMPLATE,
    }
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    return str(path)


def _byte_level_tokenizer(add_prefix_space = False):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = add_prefix_space)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size = 400,
        special_tokens = SPECIALS,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer = trainer)
    return tok


def _metaspace_tokenizer():
    tok = Tokenizer(models.BPE(unk_token = "<unk>", byte_fallback = True, fuse_unk = True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement = "▁", prepend_scheme = "first")
    tok.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content = " ", left = 1),
        ]
    )
    trainer = trainers.BpeTrainer(vocab_size = 400, special_tokens = SPECIALS)
    tok.train_from_iterator(CORPUS, trainer = trainer)
    return tok


def _ids(tokenizer, text = PROBE):
    return tokenizer(text, add_special_tokens = False).input_ids


def _reference_ids(path, text = PROBE):
    return Tokenizer.from_file(f"{path}/tokenizer.json").encode(text, add_special_tokens = False).ids


@pytest.fixture
def byte_level_llama_dir(tmp_path):
    return _write_dir(tmp_path / "bytelevel", _byte_level_tokenizer(), "LlamaTokenizerFast")


def test_premise_transformers_v5_mangles_byte_level_llama(byte_level_llama_dir):
    tok = AutoTokenizer.from_pretrained(byte_level_llama_dir)
    if not TRANSFORMERS_V5:
        assert tok.decode(_ids(tok)) == PROBE
        pytest.skip("transformers < 5 loads tokenizer.json as-is")
    assert tok.decode(_ids(tok)) != PROBE


def test_load_correct_tokenizer_round_trips_byte_level_llama(byte_level_llama_dir):
    tok = tu.load_correct_tokenizer(byte_level_llama_dir, padding_side = "right")
    ids = _ids(tok)
    assert tok.decode(ids) == PROBE
    assert ids == _reference_ids(byte_level_llama_dir)
    assert tok.decode(_ids(tok, "Hello world")) == "Hello world"
    # Special tokens, bos/eos/pad and padding side survive the rebuild.
    assert (tok.bos_token, tok.eos_token, tok.pad_token) == ("<s>", "</s>", "<pad>")
    assert tok.padding_side == "right"
    assert tok.convert_tokens_to_ids(["<s>", "</s>", "<pad>"]) == [1, 2, 3]
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Hello world"}], tokenize = False, add_generation_prompt = True
    )
    assert rendered == "<s>[user] Hello world</s>[assistant] "
    assert tok.decode(tok(rendered, add_special_tokens = False).input_ids) == rendered


def test_fast_model_post_load_fix_round_trips_byte_level_llama(byte_level_llama_dir):
    # FastModel loads with AutoTokenizer / AutoProcessor, then calls _apply_post_load_tokenizer_fixes.
    tok = AutoTokenizer.from_pretrained(byte_level_llama_dir, padding_side = "left")
    post_processor = json.loads(tok.backend_tokenizer.to_str())["post_processor"]
    tok = tu._apply_post_load_tokenizer_fixes(tok, fix_tokenizer = False)
    # The loaded post_processor (bos/eos insertion) is kept, only the text pipeline is restored.
    assert json.loads(tok.backend_tokenizer.to_str())["post_processor"] == post_processor
    assert tok.decode(_ids(tok)) == PROBE
    assert _ids(tok) == _reference_ids(byte_level_llama_dir)
    assert tok.padding_side == "left"


@pytest.mark.parametrize(
    "builder, tokenizer_class",
    [
        (_byte_level_tokenizer, "PreTrainedTokenizerFast"),
        (_metaspace_tokenizer, "LlamaTokenizerFast"),
    ],
    ids = ["bytelevel-generic", "metaspace-llama"],
)
def test_correct_tokenizer_is_untouched(tmp_path, builder, tokenizer_class):
    path = _write_dir(tmp_path / "ok", builder(), tokenizer_class)
    loaded = AutoTokenizer.from_pretrained(path)
    before_ids = _ids(loaded)
    backend = loaded.backend_tokenizer
    before = backend.to_str()
    fixed = tu._apply_post_load_tokenizer_fixes(loaded, fix_tokenizer = True)
    assert fixed.backend_tokenizer.to_str() == before
    assert _ids(fixed) == before_ids
    assert fixed.decode(before_ids) == PROBE
    # Full loader too: ids unchanged against a plain AutoTokenizer load.
    assert _ids(tu.load_correct_tokenizer(path)) == before_ids


def test_prefix_space_byte_level_llama_is_repaired(tmp_path):
    # ByteLevel(add_prefix_space = True) decodes the probe with one leading space by design; that
    # reference is valid, so the space-dropping rebuilt backend must still be replaced.
    path = _write_dir(
        tmp_path / "prefix", _byte_level_tokenizer(add_prefix_space = True), "LlamaTokenizerFast"
    )
    tok = tu._apply_post_load_tokenizer_fixes(
        AutoTokenizer.from_pretrained(path), fix_tokenizer = False
    )
    assert _ids(tok) == _reference_ids(path)
    assert tok.decode(_ids(tok)).lstrip(" ") == PROBE


def test_prefix_space_byte_level_generic_is_untouched(tmp_path):
    path = _write_dir(
        tmp_path / "prefix_ok",
        _byte_level_tokenizer(add_prefix_space = True),
        "PreTrainedTokenizerFast",
    )
    loaded = AutoTokenizer.from_pretrained(path)
    before = loaded.backend_tokenizer.to_str()
    fixed = tu._apply_post_load_tokenizer_fixes(loaded, fix_tokenizer = True)
    assert fixed.backend_tokenizer.to_str() == before


def test_tokenizer_json_that_does_not_round_trip_is_left_alone(tmp_path):
    # A lowercasing normalizer never round-trips "Hello"; with nothing better to restore, stay a no-op.
    tok = _byte_level_tokenizer()
    tok.normalizer = normalizers.Lowercase()
    path = _write_dir(tmp_path / "lower", tok, "PreTrainedTokenizerFast")
    loaded = AutoTokenizer.from_pretrained(path)
    before = loaded.backend_tokenizer.to_str()
    fixed = tu._apply_post_load_tokenizer_fixes(loaded, fix_tokenizer = True)
    assert fixed.backend_tokenizer.to_str() == before


def test_repair_can_be_disabled(byte_level_llama_dir, monkeypatch):
    if not TRANSFORMERS_V5:
        pytest.skip("nothing to repair on transformers < 5")
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    tok = AutoTokenizer.from_pretrained(byte_level_llama_dir)
    tok = tu._apply_post_load_tokenizer_fixes(tok, fix_tokenizer = True)
    assert tok.decode(_ids(tok)) != PROBE
