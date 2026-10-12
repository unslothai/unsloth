# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cached snapshot without tokenizer files must take the incomplete-cache Hub retry (#6958).

Transformers 5 loads such a snapshot as a one-token tokenizer instead of raising, so the retry
never ran and the run trained to NaN loss and reported success.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from unittest.mock import MagicMock

import pytest


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Same reason as test_audio_type_inconclusive.py: the backend pytest job has no unsloth."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

import core.training.trainer as trainer_mod  # noqa: E402
from core.training import worker  # noqa: E402
from core.training.trainer import UnslothTrainer  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

import transformers  # noqa: E402

transformers.AutoProcessor  # noqa: B018
transformers.AutoTokenizer  # noqa: B018
transformers = sys.modules["transformers"]

_QWEN2_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
}


@pytest.fixture
def text_model(monkeypatch):
    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", lambda *a, **kw: (None, True))
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)


def _fake_tokenizer(monkeypatch, size):
    class _Tok:
        def __len__(self):
            return size

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(2)) if size > 1 else []}

    class _Auto:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return _Tok()

    monkeypatch.setattr(transformers, "AutoTokenizer", _Auto, raising = False)


def test_tokenizerless_snapshot_is_a_cache_artifact_error(tmp_path, monkeypatch, text_model):
    snapshot = tmp_path / "snapshots" / "6834895"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text(json.dumps(_QWEN2_CONFIG))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(Exception) as caught:
        UnslothTrainer().pre_detect_and_load_tokenizer(
            "org/model", model_load_name = str(snapshot), local_files_only = True
        )

    # Transformers 4 raises its own TypeError here; either way the worker must retry from the Hub.
    assert worker._is_model_cache_artifact_error(caught.value)


def test_snapshot_with_only_tokenizer_config_is_a_cache_artifact_error(
    tmp_path, monkeypatch, text_model
):
    # An interrupted tokenizer download keeps tokenizer_config.json and its special tokens, not the vocab.
    snapshot = tmp_path / "snapshots" / "6834895"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text(json.dumps(_QWEN2_CONFIG))
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {
                    str(151643 + i): {"content": t, "special": True} for i, t in enumerate(special)
                },
            }
        )
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(Exception) as caught:
        UnslothTrainer().pre_detect_and_load_tokenizer(
            "org/model", model_load_name = str(snapshot), local_files_only = True
        )

    assert worker._is_model_cache_artifact_error(caught.value)


def test_pinned_revision_without_tokenizer_names_the_cause(monkeypatch, text_model):
    _fake_tokenizer(monkeypatch, 1)

    with pytest.raises(FileNotFoundError, match = "no tokenizer files at revision 6834895"):
        UnslothTrainer().pre_detect_and_load_tokenizer("org/model", model_revision = "6834895")


@pytest.mark.parametrize(
    "kwargs",
    [{"local_files_only": True, "model_load_name": "/cache/snap"}, {"model_revision": "abc"}, {}],
)
def test_a_real_tokenizer_still_loads(monkeypatch, text_model, kwargs):
    _fake_tokenizer(monkeypatch, 151665)
    trainer = UnslothTrainer()

    trainer.pre_detect_and_load_tokenizer("org/model", **kwargs)

    assert len(trainer.tokenizer) == 151665


def test_an_unpinned_hub_load_is_left_alone(monkeypatch, text_model):
    _fake_tokenizer(monkeypatch, 1)
    trainer = UnslothTrainer()

    trainer.pre_detect_and_load_tokenizer("org/model")

    assert len(trainer.tokenizer) == 1
