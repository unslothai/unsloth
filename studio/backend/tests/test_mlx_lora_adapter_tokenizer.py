# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import utils.models.model_config as model_config  # noqa: E402
from test_export_absolute_paths import _install_export_backend_stubs, _load_module  # noqa: E402
from test_mlx_inference_backend import (  # noqa: E402,F401
    _install_fake_text_stack,
    mlx_inference_patches,
    native_vlm_generation_context,
)

CHATML = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content']"
    " + '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def _tokenizer(specials, eos_token, chat_template = None):
    vocab = {"<unk>": 0, **{token: i + 1 for i, token in enumerate(specials)}}
    for word in ("user", "assistant", "Hi", "Hello!", "\n"):
        vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.add_special_tokens(specials)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, unk_token = "<unk>", eos_token = eos_token
    )
    tokenizer.chat_template = chat_template
    return tokenizer


@pytest.fixture
def base_tokenizer():
    return _tokenizer(SPECIAL, "<|endoftext|>")


@pytest.fixture
def adapter_dir(tmp_path):
    # Training moves <|im_end|> onto the base EOS id.
    trained = _tokenizer(["<|im_end|>", "<|im_start|>", "<|endoftext|>"], "<|im_end|>", CHATML)
    adapter = tmp_path / "unsloth_Qwen3-0.6B-Base_1"
    trained.save_pretrained(adapter)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-0.6B-Base", "fine_tune_type": "lora"})
    )
    return adapter


@pytest.fixture(autouse = True)
def mlx_lm_tokenizer_loader(monkeypatch):
    utils = types.ModuleType("mlx_lm.utils")
    utils.load_tokenizer = lambda path, eos_token_ids = None: AutoTokenizer.from_pretrained(path)
    package = types.ModuleType("mlx_lm")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)


def _assert_trained_tokenizer(tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], tokenize = False, add_generation_prompt = True
    )
    assert rendered == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    assert tokenizer.eos_token == "<|im_end|>"
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == 1


def test_mlx_chat_uses_the_tokenizer_a_base_model_lora_was_trained_with(
    monkeypatch, adapter_dir, base_tokenizer
):
    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    sys.modules["mlx.core"].clear_cache = lambda: None
    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = SimpleNamespace(
        from_pretrained = lambda *a, **k: (SimpleNamespace(config = {}), base_tokenizer)
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)

    config = SimpleNamespace(identifier = str(adapter_dir), is_vision = False, is_lora = True)
    assert backend.load_model(config) is True

    _assert_trained_tokenizer(backend._tokenizer)


def _export_backend(monkeypatch, base_tokenizer):
    monkeypatch.setitem(sys.modules, "utils.models.model_config", model_config)
    mod = _load_module("test_core_export_mlx_adapter_tokenizer", "core/export/export.py", monkeypatch)
    loader = SimpleNamespace(from_pretrained = lambda **k: (SimpleNamespace(), base_tokenizer))
    monkeypatch.setattr(mod, "FastLanguageModel", loader)
    monkeypatch.setattr(mod, "_IS_MLX", True)
    monkeypatch.setattr(mod, "get_base_model_from_lora", lambda *a, **k: "unsloth/Qwen3-0.6B-Base")
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    return backend


def test_mlx_export_saves_the_tokenizer_a_base_model_lora_was_trained_with(
    monkeypatch, tmp_path, adapter_dir, base_tokenizer
):
    _install_export_backend_stubs(monkeypatch)
    backend = _export_backend(monkeypatch, base_tokenizer)

    ok, message = backend.load_checkpoint(str(adapter_dir))
    assert ok, message

    _assert_trained_tokenizer(backend.current_tokenizer)
    backend.current_tokenizer.save_pretrained(tmp_path / "merged")
    _assert_trained_tokenizer(AutoTokenizer.from_pretrained(tmp_path / "merged"))


def test_mlx_lora_on_a_model_with_its_own_template_keeps_it(monkeypatch, adapter_dir):
    _install_export_backend_stubs(monkeypatch)
    own = _tokenizer(SPECIAL, "<|endoftext|>", "{{ messages[0]['content'] }}")
    backend = _export_backend(monkeypatch, own)

    ok, message = backend.load_checkpoint(str(adapter_dir))
    assert ok, message

    assert backend.current_tokenizer is own
