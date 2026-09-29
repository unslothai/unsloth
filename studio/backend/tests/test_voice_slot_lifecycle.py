# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice/chat ownership regressions without starting servers or loading weights."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import torch

from core.inference.llama_cpp import LlamaCppBackend


@pytest.fixture
def codec_state(monkeypatch):
    manager = Mock()
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", manager)
    monkeypatch.setattr(LlamaCppBackend, "_codec_owners", 0)
    monkeypatch.setattr(LlamaCppBackend, "_codec_owner_lock", threading.RLock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return manager


def test_codec_survives_until_last_slot_releases_it(codec_state):
    chat = LlamaCppBackend.__new__(LlamaCppBackend)
    voice = LlamaCppBackend.__new__(LlamaCppBackend)
    chat._claim_audio_codec()
    voice._claim_audio_codec()
    voice._claim_audio_codec()
    assert LlamaCppBackend._codec_owners == 2
    chat._unload_audio_codec()
    chat._unload_audio_codec()
    assert LlamaCppBackend._codec_mgr is codec_state
    codec_state.unload.assert_not_called()
    voice._unload_audio_codec()
    codec_state.unload.assert_called_once()
    assert LlamaCppBackend._codec_mgr is None
    assert LlamaCppBackend._codec_owners == 0


def test_chat_unload_cannot_clear_a_voice_codec_being_initialized(codec_state):
    chat = LlamaCppBackend.__new__(LlamaCppBackend)
    voice = LlamaCppBackend.__new__(LlamaCppBackend)
    voice._gpu_offload_active = False
    chat._claim_audio_codec()
    loading = threading.Event()
    release = threading.Event()
    unloading = threading.Event()

    def load(*args, **kwargs):
        loading.set()
        assert release.wait(5)

    def unload():
        unloading.set()
        chat._unload_audio_codec()

    codec_state.load_codec.side_effect = load
    with ThreadPoolExecutor(max_workers = 2) as pool:
        started = pool.submit(voice.init_audio_codec, "snac")
        assert loading.wait(5)
        stopped = pool.submit(unload)
        assert unloading.wait(5)
        try:
            assert LlamaCppBackend._codec_mgr is codec_state
            codec_state.unload.assert_not_called()
        finally:
            release.set()
        started.result(timeout = 5)
        stopped.result(timeout = 5)
    assert LlamaCppBackend._codec_owners == 1
    assert LlamaCppBackend._codec_mgr is codec_state
    voice._unload_audio_codec()
    codec_state.unload.assert_called_once()


def test_slot_pidfiles_do_not_overwrite_or_remove_each_other(monkeypatch, tmp_path):
    from utils import process_lifetime

    base = tmp_path / "llama.pid"
    monkeypatch.setattr(LlamaCppBackend, "_server_pidfile_path", classmethod(lambda cls: base))
    monkeypatch.setattr(LlamaCppBackend, "_pid_start_identity", staticmethod(lambda pid: "test"))
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda *args, **kwargs: None)
    chat = LlamaCppBackend.__new__(LlamaCppBackend)
    voice = LlamaCppBackend.__new__(LlamaCppBackend)
    voice._pid_slot = "voice"
    chat._record_own_server_pid(101)
    voice._record_own_server_pid(102)
    assert base.read_text().startswith("101:")
    assert (tmp_path / "llama-voice.pid").read_text().startswith("102:")
    voice._clear_own_server_pid()
    assert base.exists()
    assert not (tmp_path / "llama-voice.pid").exists()
    visited = []
    monkeypatch.setattr(LlamaCppBackend, "_reap_pidfile", classmethod(lambda cls, path: visited.append(path) or 0))
    LlamaCppBackend._reap_recorded_pid()
    assert visited == [base]
