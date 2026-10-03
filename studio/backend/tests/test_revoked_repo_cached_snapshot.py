# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A repo revoked upstream must still resolve from the snapshot already on disk.

Reported as #10929: a public model downloaded earlier, then made private or deleted,
stops loading and surfaces "Unsloth cannot find any torch accelerator? You need a GPU."
The GPU text is the tail of the chain, not its cause. ``_current_cached_snapshot`` learns
the current commit from ``_hub_model_info``, so a 401 there returns None, the cached
snapshot is never consulted, and every capability probe falls back to guessing from the
repo NAME. For a GGUF repo named ``...Qwen3.6...`` that guess selects a transformers tier,
whose subprocess imports unsloth and raises at import on a CPU-only host.

``HF_HUB_OFFLINE=1`` worked around it precisely because it skips the commit lookup.

These tests pin the cache-first behaviour and the warning, and the authorization boundary
that must not widen with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import hf_cache_settings
from utils.models import model_config


REPO = "LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V13-GGUF"
SHA = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00"


def _repository_not_found():
    """The error huggingface_hub raises for a deleted/privated repo, across layouts."""
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # pragma: no cover - older hub layout
        from huggingface_hub.utils import RepositoryNotFoundError
    message = (
        "401 Client Error. Repository Not Found for url: "
        f"https://huggingface.co/{REPO}/resolve/main/config.json."
    )

    # hub >= 1.x requires the originating response; older releases take the message alone.
    class _Request:
        method = "HEAD"
        url = f"https://huggingface.co/{REPO}/resolve/main/config.json"
        headers: dict = {}

    class _Response:
        status_code = 401
        headers: dict = {}
        text = message
        request = _Request()

    try:
        return RepositoryNotFoundError(message, response = _Response())
    except TypeError:  # pragma: no cover - older hub layout
        return RepositoryNotFoundError(message)


@pytest.fixture
def cached_repo(monkeypatch, tmp_path):
    """A snapshot on disk for REPO, with a config.json, and a Hub that 401s."""
    cache = tmp_path / "hub-cache"
    repo_dir = cache / f"models--{REPO.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / SHA
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}),
        encoding = "utf-8",
    )

    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(SHA, encoding = "utf-8")

    # hf_cache_snapshot_dir resolves through the HF env, so point that at the fixture too:
    # patching get_cache_path alone leaves it reading the developer's real cache.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(model_config, "active_hf_hub_cache", lambda: str(cache), raising = False)
    monkeypatch.setattr(model_config, "get_cache_path", lambda name, *a, **k: str(repo_dir))

    # The repo is gone: every Hub read for it raises, as it does in the report.
    def _dead_hub(*args, **kwargs):
        raise _repository_not_found()

    monkeypatch.setattr(model_config, "_hub_model_info", _dead_hub)
    # Single-user by default: the cache holds only this person's downloads.
    monkeypatch.setattr(model_config, "_shared_cache_installation", lambda: False)
    # The warning is deliberately once per repo per PROCESS, so it must not carry between
    # tests; a fresh set also proves the dedupe rather than inheriting an earlier test's.
    monkeypatch.setattr(model_config, "_revoked_repo_warned", set())
    return snapshot


def test_revoked_repo_reads_cached_config(cached_repo, monkeypatch):
    """The config on disk answers the vision probe instead of a name guess.

    ``_raw_config_has_vision_config`` returning None is the signal that starts the whole failure
    chain: it means "I could not tell", which drops tier selection onto the substring
    matcher. With a snapshot present, the answer is knowable and must be False here.
    """

    # Nothing may go to the wire once the snapshot is found.
    def _no_download(*args, **kwargs):
        raise AssertionError("hf_hub_download called despite a cached snapshot")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _no_download, raising = False)

    result = model_config._raw_config_has_vision_config(REPO, hf_token = "hf_" + "a" * 34)

    assert result is False, (
        "a cached config.json must answer the probe; None sends tier selection to the "
        "name-substring guess that misroutes GGUF into a transformers subprocess"
    )


def test_revoked_repo_warns_once(cached_repo, monkeypatch):
    """Serving withdrawn weights is allowed, but never silently.

    Captured through the logger rather than caplog: this module's is a structlog bound
    logger, so it never reaches the stdlib handlers caplog installs.
    """
    warnings: list = []
    monkeypatch.setattr(
        model_config.logger,
        "warning",
        lambda fmt, *args, **kwargs: warnings.append(fmt % args if args else fmt),
    )

    # Three probes ask per load in the report; the operator must hear it once.
    for _ in range(3):
        model_config._raw_config_has_vision_config(REPO, hf_token = "hf_" + "a" * 34)

    revocation_warnings = [w for w in warnings if "no longer readable" in w and REPO in w]
    assert len(revocation_warnings) == 1, (
        f"expected exactly one revocation warning across three probes, got "
        f"{len(revocation_warnings)}: {warnings}"
    )
    assert "cached copy" in revocation_warnings[0]


def test_multi_user_install_does_not_serve_private_cached_copy(cached_repo, monkeypatch):
    """Prior possession is not this caller's possession when the cache is shared.

    On an installation with managed accounts the snapshot may be another account's
    download, so a repo that cannot be shown to have been public stays refused -- the leak
    ``test_account_offline_dataset_fallback.py`` guards for datasets.
    """
    monkeypatch.setattr(model_config, "_shared_cache_installation", lambda: True)
    monkeypatch.setattr(
        model_config, "public_cache_read_authorized", lambda **k: False, raising = False
    )

    result = model_config._raw_config_has_vision_config(REPO, hf_token = "hf_" + "b" * 34)

    assert result is None, (
        "a shared-cache installation must not serve a possibly-private cached repo "
        "across accounts just because the Hub now refuses it"
    )


def test_multi_user_install_serves_cached_copy_of_public_repo(cached_repo, monkeypatch):
    """A repo that was public leaks nothing, so the fallback still applies when shared."""
    monkeypatch.setattr(model_config, "_shared_cache_installation", lambda: True)
    monkeypatch.setattr(
        model_config, "public_cache_read_authorized", lambda **k: True, raising = False
    )

    result = model_config._raw_config_has_vision_config(REPO, hf_token = "hf_" + "b" * 34)

    assert result is False, "a public repo's cached copy stays readable after revocation"


def test_transient_hub_failure_still_raises(cached_repo, monkeypatch):
    """A timeout is not a revocation: it must not silently pin a stale local copy.

    Only a definitive refusal (404/403) may take the fallback. Anything else keeps the
    previous behaviour, so a flaky network never changes what a model is understood to be.
    """

    def _timeout(*args, **kwargs):
        raise TimeoutError("connection timed out")

    monkeypatch.setattr(model_config, "_hub_model_info", _timeout)

    # The outer probe still swallows it; what matters is that the revocation fallback
    # did not claim it, so no cached answer is invented from a transient failure.
    result = model_config._raw_config_has_vision_config(REPO, hf_token = "hf_" + "a" * 34)

    assert result is None, "a transient Hub failure must not resolve from cache"
