# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compatibility cached listings must probe the Hub off the event loop."""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Keep this test runnable without optional logging deps.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from auth import policy
from hub.services.models import account_access as access
from utils.account_context import OWNER, AccountContext, arun_as

ALICE = AccountContext("a" * 32, "alice")

PROBE_SECONDS = 0.5
# Well above scheduling jitter, well below one probe.
MAX_STALL = 0.2


def _repo(repo_id: str, repo_path: Path) -> SimpleNamespace:
    snapshot = repo_path / "snapshots" / "rev"
    snapshot.mkdir(parents = True, exist_ok = True)
    weight = snapshot / "model.safetensors"
    weight.write_bytes(b"0" * 16)
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                commit_hash = "rev",
                snapshot_path = snapshot,
                files = [
                    SimpleNamespace(file_name = weight.name, size_on_disk = 16, blob_path = str(weight))
                ],
            )
        ],
    )


@pytest.fixture
def slow_hub(monkeypatch, tmp_path):
    """Two ungranted cached repos, each answered by a slow anonymous Hub probe."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_public_repos", {})
    monkeypatch.setattr(access, "_public_flights", {})

    probes: list[str] = []

    def answer(repo_id, repo_type):
        probes.append(repo_id)
        time.sleep(PROBE_SECONDS)
        return True

    monkeypatch.setattr(access, "_hub_public_answer", answer)

    cache = tmp_path / "hub"
    repos = [
        _repo("Org/Ungranted-One", cache / "models--Org--Ungranted-One"),
        _repo("Org/Ungranted-Two", cache / "models--Org--Ungranted-Two"),
    ]
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = repos)])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: cache)
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    # GGUF listing helpers: keep the row builders trivial so only the filter costs anything.
    monkeypatch.setattr(models_route, "_repo_gguf_size_bytes", lambda repo_info: 16)
    monkeypatch.setattr(models_route, "_repo_gguf_last_modified", lambda repo_info: 1.0)
    monkeypatch.setattr(models_route, "_repo_gguf_load_id", lambda *a, **k: None)
    monkeypatch.setattr(models_route, "_gguf_copy_is_usable", lambda *a, **k: True)
    monkeypatch.setattr(models_route, "_cached_gguf_row_has_vision", lambda *a, **k: False)
    monkeypatch.setattr(models_route, "_repo_gguf_task", lambda *a, **k: "text-generation")
    # Non-GGUF listing helpers.
    monkeypatch.setattr(models_route, "_repo_has_gguf_files", lambda repo_info: False)
    monkeypatch.setattr(models_route, "_repo_model_selection", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        models_route, "_recovered_repo_is_unusable_by_repo_id", lambda *a, **k: False
    )
    monkeypatch.setattr(models_route, "_cached_repo_partial", lambda *a, **k: False)
    monkeypatch.setattr(models_route, "_repo_pipeline_missing_denoiser", lambda *a, **k: False)
    monkeypatch.setattr(models_route, "_cached_repo_task", lambda *a, **k: None)
    monkeypatch.setattr(models_route, "_repo_is_diffusers", lambda *a, **k: False)
    monkeypatch.setattr(models_route, "_repo_has_pipeline_index", lambda *a, **k: True)
    monkeypatch.setattr(models_route, "_repo_model_format", lambda *a, **k: None)
    monkeypatch.setattr(models_route, "_repo_model_can_chat", lambda *a, **k: True)
    monkeypatch.setattr(models_route, "_is_sd_cpp_companion_repo", lambda repo_id: False)
    monkeypatch.setattr(models_route, "_blob_mtime", lambda f: 1.0)
    return probes


async def _stall_during(coro):
    """Longest gap an unrelated event-loop task suffers while ``coro`` runs."""
    ticks = [time.perf_counter()]

    async def ticker():
        while True:
            await asyncio.sleep(0.01)
            ticks.append(time.perf_counter())

    watcher = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    result = await coro
    # Record the caller's own resume point: a task cancelled before it can tick again
    # would otherwise hide the gap the blocking call just created.
    ticks.append(time.perf_counter())
    watcher.cancel()
    gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:])]
    return result, max(gaps)


@pytest.mark.parametrize("route", ["cached-gguf", "cached-models"])
def test_cached_listings_probe_the_hub_off_the_event_loop(slow_hub, route):
    def call():
        if route == "cached-gguf":
            return models_route.list_cached_gguf(current_subject = "alice")
        return models_route.list_cached_models(current_subject = "alice", hf_token = None)

    async def scenario():
        return await _stall_during(arun_as(ALICE, call()))

    payload, stall = asyncio.run(scenario())

    assert len(payload["cached"]) == 2, payload
    assert slow_hub, "the managed listing must have probed the Hub"
    assert stall < MAX_STALL, f"event loop blocked for {stall:.2f}s during /{route}"


@pytest.mark.parametrize("route", ["cached-gguf", "cached-models"])
def test_single_owner_listings_skip_the_probe_entirely(slow_hub, monkeypatch, route):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)

    def call():
        if route == "cached-gguf":
            return models_route.list_cached_gguf(current_subject = "owner")
        return models_route.list_cached_models(current_subject = "owner", hf_token = None)

    async def scenario():
        return await _stall_during(arun_as(OWNER, call()))

    payload, stall = asyncio.run(scenario())

    assert len(payload["cached"]) == 2, payload
    assert slow_hub == [], "the owner must not probe the Hub"
    assert stall < MAX_STALL
