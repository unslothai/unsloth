# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tokenizer Unsloth patched for saving must pickle: spawn DataLoader workers,
datasets.map(num_proc = N) and TRL's AsyncGRPO rollout worker all receive it pickled. The
wrapper is a bound method of a function no tokenizer has an attribute for, so pickle.loads
failed with `Qwen2Tokenizer has no attribute unsloth_tokenizer_save_pretrained`."""

import copy
import inspect
import os
import pickle
import subprocess
import sys

import pytest

TOKENIZER = os.environ.get(
    "UNSLOTH_TEST_TINY_TOKENIZER", "trl-internal-testing/tiny-Qwen3ForCausalLM"
)


@pytest.fixture
def patched():
    transformers = pytest.importorskip("transformers")
    save = pytest.importorskip("unsloth.save")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as error:
        pytest.skip(f"could not load {TOKENIZER}: {error}")
    return save, save.patch_saving_functions(tokenizer)


def test_patched_tokenizer_pickles_to_a_stock_tokenizer(patched):
    _, tokenizer = patched
    assert tokenizer.save_pretrained.__name__ == "unsloth_tokenizer_save_pretrained"
    loaded = pickle.loads(pickle.dumps(tokenizer))
    assert loaded("hello world")["input_ids"] == tokenizer("hello world")["input_ids"]
    assert loaded.save_pretrained.__func__ is type(loaded).save_pretrained
    assert loaded.push_to_hub.__func__ is type(loaded).push_to_hub
    assert loaded.original_save_pretrained.__func__ is type(loaded).save_pretrained


def test_live_wrapper_still_behaves_like_a_bound_method(patched, tmp_path):
    _, tokenizer = patched
    wrapper = tokenizer.save_pretrained
    assert wrapper.__self__ is tokenizer
    assert "save_directory" in inspect.signature(wrapper).parameters
    assert "self" not in inspect.signature(wrapper).parameters
    wrapper(str(tmp_path))
    assert (tmp_path / "tokenizer_config.json").is_file()
    clone = copy.deepcopy(tokenizer)
    assert clone.save_pretrained.__self__ is clone
    assert clone.save_pretrained.__name__ == "unsloth_tokenizer_save_pretrained"


def test_patching_twice_keeps_one_wrapper(patched):
    save, tokenizer = patched
    original = tokenizer.original_save_pretrained
    save.patch_saving_functions(tokenizer)
    assert tokenizer.original_save_pretrained is original
    assert tokenizer.push_to_hub.__name__ == "unsloth_push_to_hub"


def test_unsloth_only_methods_unpickle_as_none(patched):
    save, tokenizer = patched
    save.patch_saving_functions(tokenizer, vision = True)
    assert callable(tokenizer.save_pretrained_merged)
    loaded = pickle.loads(pickle.dumps(tokenizer))
    assert loaded.save_pretrained_merged is None
    assert loaded.push_to_hub_gguf is None


def test_reader_needs_no_unsloth(patched):
    _, tokenizer = patched
    code = (
        "import pickle, sys\n"
        "t = pickle.loads(sys.stdin.buffer.read())\n"
        "assert 'unsloth' not in sys.modules, 'unpickling imported unsloth'\n"
        "print(t('hello world')['input_ids'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        input = pickle.dumps(tokenizer),
        capture_output = True,
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    assert result.stdout.decode().strip() == str(tokenizer("hello world")["input_ids"])
