# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A GGUF export borrows the caller's chat_template for the save it writes: every exit path,
success or failure, must hand the tokenizer back with its own template."""

from __future__ import annotations

import pytest

import unsloth.save as save_mod
from unsloth.tokenizer_utils import _tokenizer_objects


_BOS_TEMPLATE = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
_STRIPPED_TEMPLATE = _BOS_TEMPLATE.replace("{{ bos_token }}", "")


class _FakeModel:
    config = type(
        "cfg",
        (),
        {
            "_name_or_path": "fake/model",
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        },
    )()


class _FakeTokenizer:
    """Prepends BOS on encode and emits it from its chat template, like the models the dedupe exists for."""

    bos_token_id = 2

    def __init__(self):
        self.chat_template = _BOS_TEMPLATE
        self.saved_templates = []

    def __call__(self, text, **kwargs):
        return {"input_ids": [[self.bos_token_id, 7]]}

    def save_pretrained(self, path):
        self.saved_templates.append(self.chat_template)


class _FakeProcessor:
    """A ProcessorMixin stand-in: its own chat_template plus an inner tokenizer's."""

    def __init__(self, inner):
        self.tokenizer = inner
        self.chat_template = _BOS_TEMPLATE
        self.saved_templates = []

    def save_pretrained(self, path):
        self.saved_templates.append(self.chat_template)


def _make_tokenizer(wrap_processor):
    inner = _FakeTokenizer()
    return (_FakeProcessor(inner) if wrap_processor else inner), inner


def _raise(exc):
    def _inner(*_args, **_kwargs):
        raise exc

    return _inner


def _strip_bos_templates(tokenizer):
    """fix_tokenizer_bos_token as the export uses it: mutates every wrapped tokenizer object."""
    for obj in _tokenizer_objects(tokenizer):
        if getattr(obj, "chat_template", None):
            obj.chat_template = _STRIPPED_TEMPLATE
    return True, None


def _stub_gguf_internals(monkeypatch):
    """Everything around the save/write section under test."""
    monkeypatch.setattr(save_mod, "_is_vlm", lambda _model: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _model: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", _strip_bos_templates)
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        save_mod,
        "_preflight_gguf_disk",
        lambda **kw: (kw["save_directory"], True),
    )
    monkeypatch.setattr(save_mod, "dtype_from_config", lambda _c: save_mod.torch.float16)
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_a, **_kw: None)


@pytest.mark.parametrize("wrap_processor", [False, True])
def test_imatrix_failure_restores_chat_template(monkeypatch, tmp_path, wrap_processor):
    _stub_gguf_internals(monkeypatch)
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", _raise(RuntimeError("imatrix missing")))
    tokenizer, inner = _make_tokenizer(wrap_processor)

    with pytest.raises(RuntimeError, match = "imatrix missing"):
        save_mod.unsloth_save_pretrained_gguf(
            _FakeModel(),
            str(tmp_path / "out"),
            tokenizer = tokenizer,
            quantization_method = "q8_0",
        )

    assert tokenizer.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


@pytest.mark.parametrize("wrap_processor", [False, True])
def test_peft_merge_failure_restores_chat_template(monkeypatch, tmp_path, wrap_processor):
    _stub_gguf_internals(monkeypatch)

    class _FakePeft(_FakeModel):
        pass

    monkeypatch.setattr(save_mod, "PeftModelForCausalLM", _FakePeft)
    monkeypatch.setattr(save_mod, "PeftModel", _FakePeft)
    monkeypatch.setattr(save_mod, "unsloth_generic_save", _raise(OSError("merge exploded")))
    tokenizer, inner = _make_tokenizer(wrap_processor)

    with pytest.raises(RuntimeError, match = "Failed to save/merge model"):
        save_mod.unsloth_save_pretrained_gguf(
            _FakePeft(),
            str(tmp_path / "out"),
            tokenizer = tokenizer,
            quantization_method = "q8_0",
        )

    assert tokenizer.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


@pytest.mark.parametrize("wrap_processor", [False, True])
def test_direct_save_failure_restores_chat_template(monkeypatch, tmp_path, wrap_processor):
    _stub_gguf_internals(monkeypatch)
    model = _FakeModel()
    model.save_pretrained = _raise(OSError("disk full"))
    tokenizer, inner = _make_tokenizer(wrap_processor)

    with pytest.raises(RuntimeError, match = "Failed to save model"):
        save_mod.unsloth_save_pretrained_gguf(
            model,
            str(tmp_path / "out"),
            tokenizer = tokenizer,
            quantization_method = "q8_0",
        )

    assert tokenizer.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


@pytest.mark.parametrize("wrap_processor", [False, True])
def test_base_exception_restores_chat_template(monkeypatch, tmp_path, wrap_processor):
    """KeyboardInterrupt is not caught by the save's except Exception, so only the finally restores."""
    _stub_gguf_internals(monkeypatch)
    model = _FakeModel()
    model.save_pretrained = _raise(KeyboardInterrupt())
    tokenizer, inner = _make_tokenizer(wrap_processor)

    with pytest.raises(KeyboardInterrupt):
        save_mod.unsloth_save_pretrained_gguf(
            model,
            str(tmp_path / "out"),
            tokenizer = tokenizer,
            quantization_method = "q8_0",
        )

    assert tokenizer.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


@pytest.mark.parametrize("wrap_processor", [False, True])
def test_successful_export_restores_chat_template(monkeypatch, tmp_path, wrap_processor):
    _stub_gguf_internals(monkeypatch)
    model = _FakeModel()
    model.save_pretrained = lambda *_a, **_kw: None

    def _save_to_gguf(**_kwargs):
        output = tmp_path / "out_gguf" / "model.Q8_0.gguf"
        output.parent.mkdir(exist_ok = True)
        output.write_bytes(b"GGUF")
        return [str(output)], True, False

    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)
    tokenizer, inner = _make_tokenizer(wrap_processor)

    result = save_mod.unsloth_save_pretrained_gguf(
        model,
        str(tmp_path / "out"),
        tokenizer = tokenizer,
        quantization_method = "q8_0",
    )

    assert result["fix_bos_token"] is True
    # The save wrote the BOS-stripped template; the caller's object got its own back.
    assert tokenizer.saved_templates == [_STRIPPED_TEMPLATE]
    assert tokenizer.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


def test_dedupe_dying_mid_strip_still_restores(monkeypatch, tmp_path):
    """fix_tokenizer_bos_token raising after a partial mutation must not leak either."""

    def _dying_fix(tokenizer):
        tokenizer.chat_template = _STRIPPED_TEMPLATE
        raise RuntimeError("dedupe blew up")

    _stub_gguf_internals(monkeypatch)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", _dying_fix)
    inner = _FakeTokenizer()
    processor = _FakeProcessor(inner)

    with pytest.raises(RuntimeError, match = "dedupe blew up"):
        save_mod.unsloth_save_pretrained_gguf(
            _FakeModel(),
            str(tmp_path / "out"),
            tokenizer = processor,
            quantization_method = "q8_0",
        )

    assert processor.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


def test_real_bos_dedupe_restores_processor_and_tokenizer(monkeypatch, tmp_path):
    """With the real fix_tokenizer_bos_token, the restore must reach every _tokenizer_objects member."""
    real_fix = save_mod.fix_tokenizer_bos_token
    _stub_gguf_internals(monkeypatch)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", real_fix)
    model = _FakeModel()
    model.save_pretrained = _raise(OSError("disk full"))
    inner = _FakeTokenizer()
    processor = _FakeProcessor(inner)

    with pytest.raises(RuntimeError, match = "Failed to save model"):
        save_mod.unsloth_save_pretrained_gguf(
            model,
            str(tmp_path / "out"),
            tokenizer = processor,
            quantization_method = "q8_0",
        )

    assert processor.chat_template == _BOS_TEMPLATE
    assert inner.chat_template == _BOS_TEMPLATE


def test_unchanged_readonly_chat_template_is_not_rewritten(monkeypatch, tmp_path):
    """When fix_tokenizer_bos_token does not mutate chat_template, a read-only property must not fail."""
    _stub_gguf_internals(monkeypatch)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda tok: (False, None))

    class _ReadOnlyTemplateTokenizer(_FakeTokenizer):
        def __init__(self):
            self.saved_templates = []

        @property
        def chat_template(self):
            return _BOS_TEMPLATE

    model = _FakeModel()
    model.save_pretrained = lambda *a, **kw: None

    def _save_to_gguf(**_kwargs):
        output = tmp_path / "out_gguf" / "model.Q8_0.gguf"
        output.parent.mkdir(exist_ok = True)
        output.write_bytes(b"GGUF")
        return [str(output)], True, False

    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)
    tok = _ReadOnlyTemplateTokenizer()
    save_mod.unsloth_save_pretrained_gguf(
        model,
        str(tmp_path / "out"),
        tokenizer = tok,
        quantization_method = "q8_0",
    )

    assert tok.chat_template == _BOS_TEMPLATE
