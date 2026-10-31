# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.inference import native_audio


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_disabled_sentencepiece_fails_before_runtime_import_and_preserves_guard(
    monkeypatch, platform
):
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    monkeypatch.setenv("UNSLOTH_DISABLE_SENTENCEPIECE", "1")
    monkeypatch.setattr(
        native_audio, "sys", SimpleNamespace(platform = platform, modules = sys.modules)
    )
    imported = []
    original_import = builtins.__import__

    def reject_runtime_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"sentencepiece", "transformers"}:
            imported.append(name)
            raise AssertionError(f"Runtime import attempted before dependency preflight: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_runtime_import)
    backend = native_audio.NativeAudioBackend.__new__(native_audio.NativeAudioBackend)
    entry = {}

    with pytest.raises(RuntimeError, match = "MOSS-TTS-Nano requires SentencePiece") as error:
        backend._load_moss_nano(entry, "OpenMOSS-Team/MOSS-TTS-Nano-100M", None, True)

    assert "disabled in this Unsloth process" in str(error.value)
    assert "Restart Unsloth with UNSLOTH_DISABLE_SENTENCEPIECE=0" in str(error.value)
    assert "Windows" not in str(error.value)
    assert imported == []
    assert entry == {}
    assert "sentencepiece" in sys.modules and sys.modules["sentencepiece"] is None


@pytest.mark.parametrize("already_imported", [False, True])
def test_unblocked_sentencepiece_reaches_normal_loader_without_importing_extension(
    monkeypatch, already_imported
):
    monkeypatch.setenv("UNSLOTH_DISABLE_SENTENCEPIECE", "0")
    sentencepiece = ModuleType("sentencepiece")
    if already_imported:
        monkeypatch.setitem(sys.modules, "sentencepiece", sentencepiece)
    else:
        monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(native_audio, "sys", SimpleNamespace(platform = "win32", modules = sys.modules))
    transformers = ModuleType("transformers")
    transformers.AutoModel = object()
    transformers.AutoModelForCausalLM = object()
    transformers.AutoTokenizer = object()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    calls = []

    class ReachedLoader(Exception):
        pass

    def reached_compat(codec_source, token_kwargs):
        calls.append((codec_source, token_kwargs))
        raise ReachedLoader

    monkeypatch.setattr(native_audio, "_moss_transformers5_config_compat", reached_compat)
    backend = native_audio.NativeAudioBackend.__new__(native_audio.NativeAudioBackend)

    with pytest.raises(ReachedLoader):
        backend._load_moss_nano({}, "OpenMOSS-Team/MOSS-TTS-Nano-100M", None, True)

    assert calls == [(native_audio.MOSS_NANO_CODEC_REPO, {})]
    if already_imported:
        assert sys.modules["sentencepiece"] is sentencepiece
    else:
        assert "sentencepiece" not in sys.modules
