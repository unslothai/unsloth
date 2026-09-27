# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json

import pytest
from packaging.version import Version

import transformers
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
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
        pytest.skip(reason = "transformers < 5 loads tokenizer.json as-is")
    assert tok.decode(_ids(tok)) != PROBE


def test_load_correct_tokenizer_round_trips_byte_level_llama(byte_level_llama_dir):
    tok = tu.load_correct_tokenizer(byte_level_llama_dir, padding_side = "right")
    ids = _ids(tok)
    assert tok.decode(ids) == PROBE
    assert ids == _reference_ids(byte_level_llama_dir)
    assert tok.decode(_ids(tok, "Hello world")) == "Hello world"
    assert (tok.bos_token, tok.eos_token, tok.pad_token) == ("<s>", "</s>", "<pad>")
    assert tok.padding_side == "right"
    assert tok.convert_tokens_to_ids(["<s>", "</s>", "<pad>"]) == [1, 2, 3]
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Hello world"}], tokenize = False, add_generation_prompt = True
    )
    assert rendered == "<s>[user] Hello world</s>[assistant] "
    assert tok.decode(tok(rendered, add_special_tokens = False).input_ids) == rendered


def test_fast_model_post_load_fix_round_trips_byte_level_llama(byte_level_llama_dir):
    tok = AutoTokenizer.from_pretrained(byte_level_llama_dir, padding_side = "left")
    post_processor = json.loads(tok.backend_tokenizer.to_str())["post_processor"]
    tok = tu._apply_post_load_tokenizer_fixes(tok, fix_tokenizer = False)
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
    assert _ids(tu.load_correct_tokenizer(path)) == before_ids


def test_prefix_space_byte_level_llama_is_repaired(tmp_path):
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
    tok = _byte_level_tokenizer()
    tok.normalizer = normalizers.Lowercase()
    path = _write_dir(tmp_path / "lower", tok, "PreTrainedTokenizerFast")
    loaded = AutoTokenizer.from_pretrained(path)
    before = loaded.backend_tokenizer.to_str()
    fixed = tu._apply_post_load_tokenizer_fixes(loaded, fix_tokenizer = True)
    assert fixed.backend_tokenizer.to_str() == before


def test_repair_can_be_disabled(byte_level_llama_dir, monkeypatch):
    if not TRANSFORMERS_V5:
        pytest.skip(reason = "nothing to repair on transformers < 5")
    monkeypatch.setenv("UNSLOTH_DISABLE_TOKENIZER_JSON_REPAIR", "1")
    tok = AutoTokenizer.from_pretrained(byte_level_llama_dir)
    tok = tu._apply_post_load_tokenizer_fixes(tok, fix_tokenizer = True)
    assert tok.decode(_ids(tok)) != PROBE


def test_saved_repaired_tokenizer_reloads_with_plain_transformers(byte_level_llama_dir, tmp_path):
    from unsloth.save import patch_saving_functions

    tok = tu.load_correct_tokenizer(byte_level_llama_dir)
    patch_saving_functions(tok)
    tok.save_pretrained(str(tmp_path / "saved"))
    reloaded = AutoTokenizer.from_pretrained(str(tmp_path / "saved"))
    assert reloaded.decode(_ids(reloaded)) == PROBE
    assert _ids(reloaded) == _reference_ids(byte_level_llama_dir)
    assert reloaded(PROBE).input_ids == tok(PROBE).input_ids
    assert reloaded.chat_template == tok.chat_template


# tiny-aya ships a Split regex + ByteLevel(use_regex = False) pre-tokenizer under tokenizer_class
# CohereTokenizerFast. transformers v5 maps it to CohereTokenizer, whose __init__ installs
# Digits + ByteLevel instead: text still round-trips, but ids no longer match tokenizer.json.
COHERE_SPECIALS = ["<PAD>", "<UNK>", "<BOS_TOKEN>", "<|END_OF_TURN_TOKEN|>"]
COHERE_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
COHERE_CORPUS = ["with open(path) as f: read(path) 123 456 789", "नमस्ते दुनिया (path)"] * 30
COHERE_PROBE = "with open(path) as f: 123456 नमस्ते"


def _cohere_like_tokenizer(split_regex = True):
    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = not split_regex)
    if split_regex:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split(Regex(COHERE_REGEX), behavior = "isolated"), byte_level]
        )
    else:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits = True), byte_level]
        )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size = 500,
        special_tokens = COHERE_SPECIALS,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(COHERE_CORPUS, trainer = trainer)
    bos = tok.token_to_id("<BOS_TOKEN>")
    tok.post_processor = processors.TemplateProcessing(
        single = "<BOS_TOKEN> $A", pair = "<BOS_TOKEN> $A $B", special_tokens = [("<BOS_TOKEN>", bos)]
    )
    return tok


def _write_cohere_dir(path, tokenizer):
    path.mkdir(parents = True, exist_ok = True)
    tokenizer.save(str(path / "tokenizer.json"))
    config = {
        "tokenizer_class": "CohereTokenizerFast",
        "bos_token": "<BOS_TOKEN>",
        "eos_token": "<|END_OF_TURN_TOKEN|>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "add_bos_token": True,
        "add_eos_token": False,
        "chat_template": CHAT_TEMPLATE,
    }
    (path / "tokenizer_config.json").write_text(json.dumps(config))
    return str(path)


def test_cohere_split_regex_ids_match_tokenizer_json(tmp_path):
    path = _write_cohere_dir(tmp_path / "aya", _cohere_like_tokenizer(split_regex = True))
    loaded = AutoTokenizer.from_pretrained(path)
    reference = _reference_ids(path, COHERE_PROBE)
    if TRANSFORMERS_V5:
        # Premise: v5 keeps the round trip but changes the ids.
        assert loaded.decode(_ids(loaded, COHERE_PROBE)) == COHERE_PROBE
        assert _ids(loaded, COHERE_PROBE) != reference
    for tok in (
        tu._apply_post_load_tokenizer_fixes(loaded, fix_tokenizer = False),
        tu.load_correct_tokenizer(path),
    ):
        assert _ids(tok, COHERE_PROBE) == reference
        assert tok.decode(_ids(tok, COHERE_PROBE)) == COHERE_PROBE
        assert tok(COHERE_PROBE).input_ids[0] == tok.convert_tokens_to_ids("<BOS_TOKEN>")
        assert (tok.bos_token, tok.eos_token, tok.pad_token) == (
            "<BOS_TOKEN>",
            "<|END_OF_TURN_TOKEN|>",
            "<PAD>",
        )


def test_cohere_matching_tokenizer_json_is_untouched(tmp_path):
    path = _write_cohere_dir(tmp_path / "expanse", _cohere_like_tokenizer(split_regex = False))
    loaded = AutoTokenizer.from_pretrained(path)
    before = loaded.backend_tokenizer.to_str()
    fixed = tu._apply_post_load_tokenizer_fixes(loaded, fix_tokenizer = True)
    assert fixed.backend_tokenizer.to_str() == before
    assert not getattr(fixed, "_unsloth_tokenizer_json_repaired", False)
    assert _ids(fixed, COHERE_PROBE) == _reference_ids(path, COHERE_PROBE)
