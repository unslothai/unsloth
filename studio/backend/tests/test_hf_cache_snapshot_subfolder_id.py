# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A ``<namespace>/<repo>/<folder>`` id (audio.cpp) is not a repo id: the cache lookup must say "not cached"."""

from utils import utils


def test_snapshot_lookup_of_a_subfolder_model_id_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_hf_cache_roots", lambda: [tmp_path])
    model = "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF"
    assert utils.hf_cache_snapshot_dir(model) is None
    assert utils.hf_cache_snapshot_dir_for_repo(model) is None
