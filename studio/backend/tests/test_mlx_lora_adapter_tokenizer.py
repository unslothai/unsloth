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


def _tokenizer(
    specials,
    eos_token,
    chat_template = None,
):
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


def test_mlx_chat_fetches_the_tokenizer_of_a_lora_loaded_from_the_hub(
    monkeypatch, adapter_dir, base_tokenizer
):
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs["token"]))
        return str(adapter_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    tokenizer = model_config.load_mlx_adapter_tokenizer(
        base_tokenizer, "someone/qwen3-base-lora", "hf_secret"
    )

    assert calls == [("someone/qwen3-base-lora", "hf_secret")]
    _assert_trained_tokenizer(tokenizer)


def _export_backend(monkeypatch, base_tokenizer):
    monkeypatch.setitem(sys.modules, "utils.models.model_config", model_config)
    mod = _load_module(
        "test_core_export_mlx_adapter_tokenizer", "core/export/export.py", monkeypatch
    )
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


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def chat_template(self):
        return self._tokenizer.chat_template


def _base_processor(base_tokenizer):
    base_tokenizer._unsloth_modality_model_type = "qwen3_5"
    base_tokenizer.stopping_criteria = "mlx-vlm criteria"
    return SimpleNamespace(tokenizer = base_tokenizer, chat_template = None)


@pytest.mark.parametrize("wrapped", [False, True])
def test_mlx_processor_of_a_multimodal_base_gets_the_trained_tokenizer(
    monkeypatch, adapter_dir, base_tokenizer, wrapped
):
    if wrapped:
        load = sys.modules["mlx_lm.utils"].load_tokenizer
        monkeypatch.setattr(
            sys.modules["mlx_lm.utils"],
            "load_tokenizer",
            lambda path, eos_token_ids = None: TokenizerWrapper(load(path)),
        )
    processor = _base_processor(base_tokenizer)

    assert model_config.load_mlx_adapter_processor(processor, str(adapter_dir)) is processor

    _assert_trained_tokenizer(processor.tokenizer)
    assert processor.chat_template == CHATML
    assert processor.tokenizer._unsloth_modality_model_type == "qwen3_5"
    assert processor.tokenizer.stopping_criteria == "mlx-vlm criteria"


def test_mlx_processor_with_its_own_template_is_left_alone(adapter_dir, base_tokenizer):
    processor = _base_processor(base_tokenizer)
    processor.chat_template = "{{ messages[0]['content'] }}"

    model_config.load_mlx_adapter_processor(processor, str(adapter_dir))

    assert processor.tokenizer is base_tokenizer


def test_mlx_vision_export_saves_the_template_a_base_model_lora_was_trained_with(
    monkeypatch, adapter_dir, base_tokenizer
):
    _install_export_backend_stubs(monkeypatch)
    backend = _export_backend(monkeypatch, base_tokenizer)
    mod = sys.modules[type(backend).__module__]
    processor = _base_processor(base_tokenizer)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: True)
    monkeypatch.setattr(
        mod,
        "FastVisionModel",
        SimpleNamespace(from_pretrained = lambda **k: (SimpleNamespace(), processor)),
    )

    ok, message = backend.load_checkpoint(str(adapter_dir))
    assert ok, message

    assert backend.current_tokenizer is processor
    assert processor.chat_template == CHATML
    _assert_trained_tokenizer(processor.tokenizer)
