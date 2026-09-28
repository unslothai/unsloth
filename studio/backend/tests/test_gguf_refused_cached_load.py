# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF the owner already downloaded still runs when the Hub refuses its repo.

A repo that was deleted, made private or gated after the download answers 401/403/404 to every
listing, and #12117 made that a clear error instead of a Transformers misroute. The machine
owner's own session may still run the complete copy on its disk, with a warning; anyone the
cache rules would refuse (anonymous, an API key without a token, another account, a token the
Hub says cannot reach the repo) is refused exactly as before.
"""

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

import core.inference.llama_cpp as llama_cpp
import hub.utils.hf_tokens as hf_tokens
import utils.hf_cache_settings as hf_cache_settings
import utils.models.model_config as mc
from hub.utils.hf_tokens import (
    AmbientAuthorizedToken,
    collecting_hub_token_rejections,
    hub_refused_cached_copy_warning,
)
from utils.models.model_config import GgufRepoUnreadableError, ModelConfig

REPO = "unsloth/Qwen3-0.6B-GGUF"
VARIANT = "UD-Q4_K_XL"
GGUF = f"Qwen3-0.6B-{VARIANT}.gguf"
REV = "a" * 40


class RepositoryNotFoundError(Exception):
    """Matched by type name, as detect_gguf_model_remote does."""

    def __init__(
        self,
        message,
        status_code = 401,
    ):
        super().__init__(message)
        self.response = SimpleNamespace(status_code = status_code, headers = {})


class GatedRepoError(RepositoryNotFoundError):
    pass


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(
        hf_cache_settings, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path)
    )
    monkeypatch.setattr(mc.time, "sleep", lambda *_: None)
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda *, include_denied = False: "/fake/llama-server"),
    )
    monkeypatch.setattr(mc, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mc, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mc, "is_model_cached", lambda *a, **k: False)
    # Only this test's cache: no copies remembered from caches the host used before.
    import hub.utils.gguf_sources as gguf_sources

    monkeypatch.setattr(gguf_sources, "gguf_cache_snapshots", lambda repo: [])
    return tmp_path


def _snapshot(root: Path) -> Path:
    snapshot = root / f"models--{REPO.replace('/', '--')}" / "snapshots" / REV
    snapshot.mkdir(parents = True, exist_ok = True)
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok = True)
    (refs / "main").write_text(REV)
    return snapshot


def _download(root: Path, *names: str) -> Path:
    snapshot = _snapshot(root)
    for name in names:
        path = snapshot / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"GGUF" + b"\0" * 64)
    return snapshot


def _hub(behaviour):
    def model_info(repo_id, **kwargs):
        if isinstance(behaviour, BaseException):
            raise behaviour
        return SimpleNamespace(siblings = [SimpleNamespace(rfilename = f) for f in behaviour])

    return model_info


def _refuse(monkeypatch, error = None):
    monkeypatch.setattr(
        huggingface_hub,
        "model_info",
        _hub(error if error is not None else RepositoryNotFoundError("401 Client Error")),
    )


def _load(**kwargs):
    with collecting_hub_token_rejections() as rejections:
        config = ModelConfig.from_identifier(REPO, **kwargs)
    return config, rejections


@pytest.mark.parametrize(
    "error",
    [
        RepositoryNotFoundError("401 Client Error"),
        RepositoryNotFoundError("404 Client Error", status_code = 404),
        GatedRepoError("403 Client Error", status_code = 403),
    ],
    ids = ["401", "404", "gated"],
)
@pytest.mark.parametrize("variant", [VARIANT, None], ids = ["variant", "auto"])
def test_owner_runs_the_downloaded_copy_when_the_hub_refuses(
    monkeypatch, _isolated, error, variant
):
    snapshot = _download(_isolated, GGUF)
    _refuse(monkeypatch, error)
    config, rejections = _load(gguf_variant = variant, owner_session = True)
    assert config.is_gguf
    assert config.gguf_hf_repo is None, "must not go back to the Hub for the weights"
    assert Path(config.gguf_file) == snapshot / GGUF
    assert config.identifier == REPO
    assert config.gguf_variant == VARIANT
    assert rejections.served_from_cache == [REPO]


@pytest.mark.parametrize(
    "token",
    [None, AmbientAuthorizedToken("hf_" + "u" * 34)],
    ids = ["ambient", "ui_saved_token"],
)
def test_the_owner_session_is_authorized_with_or_without_a_saved_token(
    monkeypatch, _isolated, token
):
    _download(_isolated, GGUF)
    _refuse(monkeypatch)
    config, _ = _load(hf_token = token, gguf_variant = VARIANT, owner_session = True)
    assert config.is_gguf and config.gguf_file


def test_the_warning_names_the_repo():
    warning = hub_refused_cached_copy_warning(REPO)
    assert REPO in warning and "downloaded" in warning and "updates" in warning


def test_a_split_copy_missing_a_shard_is_refused(monkeypatch, _isolated):
    _download(_isolated, f"{VARIANT}/Qwen3-0.6B-{VARIANT}-00001-of-00002.gguf")
    _refuse(monkeypatch)
    with pytest.raises(GgufRepoUnreadableError):
        _load(gguf_variant = VARIANT, owner_session = True)


def test_a_copy_its_manifest_calls_incomplete_is_refused(monkeypatch, _isolated):
    # The download recorded a projector the variant needs, and it never arrived.
    _download(_isolated, GGUF)
    import hub.utils.gguf_sources as gguf_sources

    monkeypatch.setattr(
        gguf_sources, "cached_gguf_manifest_complete", lambda repo, quant, snapshot: False
    )
    _refuse(monkeypatch)
    with pytest.raises(GgufRepoUnreadableError):
        _load(gguf_variant = VARIANT, owner_session = True)


def test_only_a_projector_on_disk_is_refused(monkeypatch, _isolated):
    _download(_isolated, "mmproj-F16.gguf")
    _refuse(monkeypatch)
    with pytest.raises(GgufRepoUnreadableError):
        _load(owner_session = True)


@pytest.mark.parametrize(
    "token",
    [False, None, "hf_" + "k" * 34],
    ids = ["anonymous", "api_key_without_token", "api_key_with_token"],
)
def test_callers_the_cache_rules_refuse_stay_refused(monkeypatch, _isolated, token):
    # owner_session=False: an sk-unsloth API key or a managed account. No token there is
    # anonymous, not the installation's; an explicit token must reach the repo, which the
    # Hub just refused.
    _download(_isolated, GGUF)
    _refuse(monkeypatch)
    monkeypatch.setattr(
        hf_tokens,
        "_explicit_token_reaches_repo",
        lambda repo, token, repo_type, offline = False: False,
    )
    with pytest.raises(GgufRepoUnreadableError):
        _load(hf_token = token, gguf_variant = VARIANT)


@pytest.mark.parametrize(
    "verdict", [True, None], ids = ["cached_auth_check_yes", "probe_cannot_answer"]
)
def test_a_token_the_hub_just_refused_is_not_outranked_by_the_cache_rule(
    monkeypatch, _isolated, verdict
):
    # The listing refused this very token. A /auth-check "yes" still in its cache, or a probe
    # that times out on a host whose only credential downloaded the repo, must not let an API
    # key or another account read the downloaded copy.
    _download(_isolated, GGUF)
    _refuse(monkeypatch, GatedRepoError("403 Client Error", status_code = 403))
    monkeypatch.setattr(
        hf_tokens,
        "_explicit_token_reaches_repo",
        lambda repo, token, repo_type, offline = False: verdict,
    )
    # This token downloaded the copy, so an unanswerable probe would otherwise be settled by it.
    monkeypatch.setattr(hf_tokens, "_caller_populated_the_cache", lambda *a, **k: True)
    monkeypatch.setattr(hf_tokens, "_repo_present_on_disk", lambda *a, **k: True)
    for _ in range(2):
        with pytest.raises(GgufRepoUnreadableError):
            _load(hf_token = "hf_" + "k" * 34, gguf_variant = VARIANT, owner_session = False)


def test_a_revoked_grant_is_refused_even_for_a_managed_accounts_token(monkeypatch, _isolated):
    # A managed account's own token whose access the Hub has withdrawn: the cache is another
    # person's download, and /auth-check says this token no longer reaches the repo.
    _download(_isolated, GGUF)
    _refuse(monkeypatch)
    monkeypatch.setattr(
        hf_tokens,
        "_explicit_token_reaches_repo",
        lambda repo, token, repo_type, offline = False: False,
    )
    with pytest.raises(GgufRepoUnreadableError):
        _load(hf_token = "hf_" + "m" * 34, gguf_variant = VARIANT, owner_session = False)


def test_timeouts_keep_todays_cache_route(monkeypatch, _isolated):
    _download(_isolated, GGUF)
    monkeypatch.setattr(huggingface_hub, "model_info", _hub(TimeoutError("read timed out")))
    monkeypatch.setattr(
        mc,
        "list_gguf_variants",
        lambda identifier, hf_token = None: (
            [mc.GgufVariantInfo(filename = GGUF, quant = VARIANT, size_bytes = 1)],
            False,
        ),
    )
    config, rejections = _load(gguf_variant = VARIANT, owner_session = True)
    assert config.is_gguf and config.gguf_hf_repo == REPO
    assert rejections.served_from_cache == []


def test_a_healthy_hub_is_unchanged(monkeypatch, _isolated):
    _download(_isolated, GGUF)
    monkeypatch.setattr(huggingface_hub, "model_info", _hub([GGUF, "README.md"]))
    monkeypatch.setattr(
        mc,
        "list_gguf_variants",
        lambda identifier, hf_token = None: (
            [mc.GgufVariantInfo(filename = GGUF, quant = VARIANT, size_bytes = 1)],
            False,
        ),
    )
    config, rejections = _load(gguf_variant = VARIANT, owner_session = True)
    assert config.is_gguf and config.gguf_hf_repo == REPO
    assert rejections.served_from_cache == []


def test_an_uncached_refused_repo_still_raises_the_clear_error(monkeypatch, _isolated):
    _refuse(monkeypatch)
    with pytest.raises(GgufRepoUnreadableError, match = "HTTP 401"):
        _load(gguf_variant = VARIANT, owner_session = True)


def test_the_load_response_carries_the_cached_copy_warning():
    from models.inference import LoadResponse
    from routes.inference import _with_token_rejected_warning

    response = LoadResponse(status = "loaded", model = REPO, display_name = REPO, inference = {})
    with collecting_hub_token_rejections() as rejections:
        assert _with_token_rejected_warning(response, rejections) is response
        rejections.served_from_cache += [REPO, REPO]
        warned = _with_token_rejected_warning(response, rejections)
    assert warned.memory_warning == hub_refused_cached_copy_warning(REPO)


@pytest.mark.parametrize(
    "managed,api_key,expected",
    [(False, False, True), (False, True, False), (True, False, False)],
    ids = ["owner_ui", "api_key", "managed_account"],
)
def test_only_the_owners_own_session_counts_as_the_owner(monkeypatch, managed, api_key, expected):
    import routes.inference as inference

    monkeypatch.setattr(inference.account_access, "managed_account", lambda: managed)
    monkeypatch.setattr(inference, "_request_has_api_key", lambda request: api_key)
    assert inference._owner_session(object()) is expected
    assert inference._owner_session(None) is False


def test_an_internal_workflow_key_is_not_the_owner(monkeypatch):
    # A data-recipe subprocess holds an internal sk-unsloth key: never the owner's session.
    import routes.inference as inference

    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(inference, "_request_api_key_token", lambda request: "sk-unsloth-internal")
    monkeypatch.setattr(inference.auth_storage, "is_internal_api_key", lambda token: True)
    assert inference._request_used_api_key(object()) is False
    assert inference._owner_session(object()) is False


def test_a_cancelled_download_is_not_run_when_the_hub_refuses(monkeypatch, _isolated):
    # Its main file is on disk, but the download was cancelled: not a complete copy.
    _download(_isolated, GGUF)
    import hub.utils.gguf_sources as gguf_sources

    monkeypatch.setattr(
        gguf_sources, "cached_gguf_source_partial", lambda repo, quant, snapshot: True
    )
    _refuse(monkeypatch)
    with pytest.raises(GgufRepoUnreadableError):
        _load(gguf_variant = VARIANT, owner_session = True)


def test_auto_selection_falls_back_to_the_next_complete_variant(monkeypatch, _isolated):
    # The preferred UD-Q4_K_XL was interrupted after its first shard; Q8_0 is complete.
    snapshot = _download(
        _isolated,
        f"{VARIANT}/Qwen3-0.6B-{VARIANT}-00001-of-00002.gguf",
        "Qwen3-0.6B-Q8_0.gguf",
    )
    _refuse(monkeypatch)
    config, rejections = _load(owner_session = True)
    assert Path(config.gguf_file) == snapshot / "Qwen3-0.6B-Q8_0.gguf"
    assert config.gguf_variant == "Q8_0"
    assert rejections.served_from_cache == [REPO]


def test_the_refusal_fallback_passes_the_snapshot_root_as_the_companion_root(
    monkeypatch, _isolated
):
    # A main file in a non-quant subdirectory still finds a repo-root projector or drafter.
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = _isolated, source = "environment"),
    )
    snapshot = _download(_isolated, f"distilled/Qwen3-0.6B-{VARIANT}.gguf")
    _refuse(monkeypatch)
    seen = []
    real = ModelConfig.from_identifier.__func__

    def spy(cls, model_id, *args, **kwargs):
        if model_id != REPO:
            seen.append(kwargs.get("gguf_companion_roots"))
        return real(cls, model_id, *args, **kwargs)

    monkeypatch.setattr(ModelConfig, "from_identifier", classmethod(spy))
    config, _ = _load(owner_session = True)
    assert config.is_gguf
    assert seen == [(str(snapshot),)]


def test_a_repo_not_named_gguf_still_runs_its_downloaded_gguf(monkeypatch, _isolated):
    # _looks_like_gguf_repo says no for this name, yet a healthy Hub would have found the GGUF.
    monkeypatch.setattr(mc, "_looks_like_gguf_repo", lambda *a, **k: False)
    snapshot = _download(_isolated, GGUF)
    _refuse(monkeypatch)
    config, rejections = _load(gguf_variant = VARIANT, owner_session = True)
    assert Path(config.gguf_file) == snapshot / GGUF
    assert rejections.served_from_cache == [REPO]


def test_auto_selection_finds_a_complete_variant_in_an_older_snapshot(monkeypatch, _isolated):
    # The newest snapshot holds only an interrupted UD-Q4_K_XL; an older one a complete Q8_0.
    import os

    older = _download(_isolated, "Qwen3-0.6B-Q8_0.gguf")
    newer = older.parent / ("b" * 40)
    (newer / VARIANT).mkdir(parents = True)
    (newer / VARIANT / f"Qwen3-0.6B-{VARIANT}-00001-of-00002.gguf").write_bytes(
        b"GGUF" + b"\0" * 64
    )
    (older.parent.parent / "refs" / "main").write_text("b" * 40)
    os.utime(older, (1_000_000_000, 1_000_000_000))
    _refuse(monkeypatch)
    config, rejections = _load(owner_session = True)
    assert Path(config.gguf_file) == older / "Qwen3-0.6B-Q8_0.gguf"
    assert config.gguf_variant == "Q8_0"
    assert rejections.served_from_cache == [REPO]


def test_the_downloaded_copy_keeps_its_repo_for_the_download_interlock(monkeypatch, _isolated):
    # Not gguf_hf_repo (that would fetch from the Hub), but /load's marker and 409 need the repo.
    _download(_isolated, GGUF)
    _refuse(monkeypatch)
    config, _ = _load(gguf_variant = VARIANT, owner_session = True)
    assert config.gguf_hf_repo is None
    assert config.gguf_cache_repo == REPO


def test_a_missing_llama_server_fails_before_the_cached_copy_is_returned(monkeypatch, _isolated):
    # /load must not unload the resident model for a launch that cannot start.
    _download(_isolated, GGUF)
    _refuse(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda *, include_denied = False: None),
    )
    with pytest.raises(llama_cpp.LlamaServerNotFoundError):
        _load(gguf_variant = VARIANT, owner_session = True)


@pytest.mark.parametrize("listed", ["root_first", "subdir_first"])
def test_auto_selection_prefers_the_root_checkpoint_whatever_the_listing_order(
    monkeypatch, _isolated, listed
):
    # As the healthy-Hub pick: the bare repo id is the root checkpoint, not distilled/.
    root, sub = "Qwen3-0.6B-Q6_K.gguf", "distilled/Qwen3-0.6B-Q6_K.gguf"
    snapshot = _download(_isolated, root, sub)
    order = [root, sub] if listed == "root_first" else [sub, root]
    monkeypatch.setattr(mc, "_hf_cache_main_gguf_files", lambda repo, **k: list(order))
    _refuse(monkeypatch)
    config, _ = _load(owner_session = True)
    assert Path(config.gguf_file) == snapshot / root


def test_auto_selection_finds_a_copy_in_a_remembered_cache(
    monkeypatch, _isolated, tmp_path_factory
):
    # The active cache was switched to an empty one; the explicit-variant lookup still finds the
    # remembered copy, so automatic selection must too.
    import hub.utils.gguf_sources as gguf_sources

    remembered = _download(tmp_path_factory.mktemp("previous_cache"), GGUF)
    monkeypatch.setattr(
        gguf_sources,
        "gguf_cache_snapshots",
        lambda repo: [remembered] if repo == REPO else [],
    )
    _refuse(monkeypatch)
    config, _ = _load(owner_session = True)
    assert Path(config.gguf_file) == remembered / GGUF
    assert config.gguf_variant == VARIANT


def test_the_downloaded_copy_is_reported_as_a_hub_model(monkeypatch, _isolated):
    # Not the user's own file: /load reports is_local_model from this, and the chat settings
    # then give Hub recovery guidance rather than "put the drafter beside your file".
    _download(_isolated, GGUF)
    _refuse(monkeypatch)
    config, _ = _load(gguf_variant = VARIANT, owner_session = True)
    assert config.is_local is False
