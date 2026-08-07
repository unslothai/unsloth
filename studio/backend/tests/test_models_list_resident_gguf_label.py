# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/models/list must label the resident GGUF the way the chat model bar reads it.

The bar renders the ``name`` of the models[] entry whose ``id`` equals the client's
``params.checkpoint``, which for a GGUF out of the HF cache is the snapshot directory.
Naming that entry by splitting the raw identifier on "/" showed the commit sha on POSIX
and, on Windows (backslashes, nothing to split on), the entire home directory.
"""

import asyncio
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inf  # noqa: E402
import routes.models as models_route  # noqa: E402

_POSIX_SNAPSHOT = (
    "/home/u/.cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF"
    "/snapshots/57326b941c4603e24d1a5e71c22520c66e086eb8"
)
_WINDOWS_SNAPSHOT = (
    "C:\\Users\\An\\.cache\\huggingface\\hub"
    "\\models--unsloth--DeepSeek-V4-Flash-0731-GGUF"
    "\\snapshots\\57326b941c4603e24d1a5e71c22520c66e086eb8"
)


class _FakeLlama:
    is_vision = False

    def __init__(self, identifier, *, native_grant_backed = False, display_label = None):
        self.is_loaded = True
        self.model_identifier = identifier
        self._native_grant_backed = native_grant_backed
        self._native_display_label = display_label


class _FakeUnsloth:
    default_models: list = []
    models: dict = {}


def _list_models(monkeypatch, llama):
    monkeypatch.setattr(models_route, "get_inference_backend", lambda: _FakeUnsloth())
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)
    return asyncio.run(models_route.list_models(current_subject = "unsloth")).models


def test_hf_cache_snapshot_is_labelled_by_its_repo_leaf(monkeypatch):
    for snapshot in (_POSIX_SNAPSHOT, _WINDOWS_SNAPSHOT):
        entries = _list_models(monkeypatch, _FakeLlama(snapshot))

        assert len(entries) == 1
        entry = entries[0]
        # The id stays the loadable identifier the client holds as its checkpoint, so the
        # picker still finds this entry; only the label is cleaned up.
        assert entry.id == snapshot
        assert entry.name == "DeepSeek-V4-Flash-0731-GGUF"
        assert entry.is_gguf is True


def test_standalone_gguf_is_labelled_by_its_file_stem(monkeypatch):
    entries = _list_models(
        monkeypatch, _FakeLlama("/srv/models/Qwen3-30B-A3B-Q4_K_M.gguf")
    )

    assert entries[0].name == "Qwen3-30B-A3B-Q4_K_M"


def test_native_lease_keeps_the_path_shaped_id_agents_tab_filters_on(monkeypatch):
    # agents-tab's discoverGgufModels drops path-shaped ids so a native lease's label,
    # which cannot reload the file, never becomes a named model in a --model command.
    # Only the label is cleaned here.
    entries = _list_models(
        monkeypatch,
        _FakeLlama(
            "/private/var/leases/xyz/Qwen3-4B-Q4_K_M.gguf",
            native_grant_backed = True,
            display_label = "Qwen3-4B-Q4_K_M.gguf",
        ),
    )

    assert entries[0].id == "/private/var/leases/xyz/Qwen3-4B-Q4_K_M.gguf"
    assert entries[0].name == "Qwen3-4B-Q4_K_M"


def test_plain_repo_ids_are_unchanged(monkeypatch):
    class _Backend(_FakeUnsloth):
        default_models = ["unsloth/Qwen3-4B-Instruct-2507", "Qwen3-4B"]

    monkeypatch.setattr(models_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(None))

    entries = asyncio.run(models_route.list_models(current_subject = "unsloth")).models

    assert [(e.id, e.name) for e in entries] == [
        ("unsloth/Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct-2507"),
        ("Qwen3-4B", "Qwen3-4B"),
    ]


def test_already_resident_load_response_never_echoes_the_snapshot_path(monkeypatch):
    # The already-loaded fast path leaves display_name unset, so the response fell back
    # to the identifier the GGUF loaded from -- the client then labels the model with it.
    import routes.inference as inf_route

    monkeypatch.setattr(
        inf_route, "_llama_runtime_fields", lambda backend: {}
    )
    monkeypatch.setattr(inf_route, "load_inference_config", lambda identifier: {})

    resp = inf_route._gguf_load_response(
        _FakeLlama(_POSIX_SNAPSHOT),
        "already_loaded",
        _POSIX_SNAPSHOT,
        is_local_model = True,
    )

    assert resp.display_name == "DeepSeek-V4-Flash-0731-GGUF"
    # The loadable identifier itself is unchanged; only the label is cleaned up.
    assert resp.model == _POSIX_SNAPSHOT


def test_an_explicit_display_name_still_wins(monkeypatch):
    import routes.inference as inf_route

    monkeypatch.setattr(inf_route, "_llama_runtime_fields", lambda backend: {})
    monkeypatch.setattr(inf_route, "load_inference_config", lambda identifier: {})

    resp = inf_route._gguf_load_response(
        _FakeLlama(_POSIX_SNAPSHOT),
        "loaded",
        _POSIX_SNAPSHOT,
        display_name = "DeepSeek-V4-Flash-0731 (UD-IQ2_M)",
        is_local_model = True,
    )

    assert resp.display_name == "DeepSeek-V4-Flash-0731 (UD-IQ2_M)"


def test_a_hub_repo_id_ending_in_gguf_keeps_its_suffix(monkeypatch):
    # lex-au/Orpheus-3b-FT-Q8_0.gguf and friends are real repo ids, not file paths, so
    # the label is the repo leaf whole; only a >= 2-slash id names a file inside a repo.
    class _Backend(_FakeUnsloth):
        default_models = ["lex-au/Orpheus-3b-FT-Q8_0.gguf"]

    monkeypatch.setattr(models_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(None))

    entries = asyncio.run(models_route.list_models(current_subject = "unsloth")).models

    assert entries[0].name == "Orpheus-3b-FT-Q8_0.gguf"
