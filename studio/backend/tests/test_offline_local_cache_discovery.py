# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub that cannot be ASKED is not a Hub that said no (#10264 put a live /auth-check in front
of every disk-cache read). Every test pairs the fix with the boundary it must not move: an
ANSWERED no still refuses the cached copy. Only the list of cache ROOTS is redirected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import picker.service as picker_service
from hub.utils import dataset_cache, hf_cache_state, hf_tokens
from hub.utils.hf_tokens import (
    AmbientAuthorizedToken,
    cache_reads_authorized,
    cached_read_refused,
    public_cache_read_authorized,
    reset_repo_access_cache,
)
from utils import transformers_version
from utils.models import model_config as model_config_module

# Captured before the autouse fixture replaces the attribute, so the one test ABOUT the reader
# can still reach the real one.
_REAL_AMBIENT_HF_TOKEN = hf_tokens._ambient_hf_token
_REAL_SAVED_STUDIO_HF_TOKEN = hf_tokens._saved_studio_hf_token

ON_DISK = "acme/downloaded-model"
ABSENT = "acme/never-downloaded"
# The credential this HOST downloads with. FOREIGN_TOKEN is the second principal that must NOT
# inherit what is on the disk.
OPERATOR_TOKEN = "hf_operator_ambient_token"
# The SAME operator, credential saved through Studio Settings instead: `get_token()` never
# returns it, and the UI replays it per request in X-Unsloth-HF-Token.
STUDIO_UI_TOKEN = "hf_saved_in_studio_settings"
FOREIGN_TOKEN = "hf_some_other_callers_token"
REVISION = "a" * 40


@pytest.fixture(autouse = True)
def _isolate_repo_access_cache():
    reset_repo_access_cache()
    yield
    reset_repo_access_cache()


@pytest.fixture(autouse = True)
def _host_credential(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OPERATOR_TOKEN))


@pytest.fixture(autouse = True)
def _saved_studio_credential(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, None))


def _no_host_credential(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))


def _saved_ui_credential(monkeypatch, token = STUDIO_UI_TOKEN):
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, token))


def _materialize_repo(
    root: Path,
    repo_id: str,
    repo_type: str = "model",
    *,
    files: tuple[str, ...] = ("config.json", "model.safetensors"),
) -> Path:
    repo_dir = root / f"{repo_type}s--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / REVISION
    snapshot.mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(REVISION, encoding = "utf-8")
    for name in files:
        (snapshot / name).write_text(
            '{"model_type": "llama"}' if name.endswith(".json") else "x",
            encoding = "utf-8",
        )
    return snapshot


def _cache_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    root.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [root])
    return root


def _counting_probe(
    monkeypatch,
    verdict,
    *,
    offline: bool = False,
) -> dict:
    calls = {"n": 0}

    def _probe(*_a, **_k):
        calls["n"] += 1
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict

    monkeypatch.setattr(hf_tokens, "_probe_repo_access", _probe)
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: offline)
    return calls


def _probe_against(monkeypatch, response_factory):
    class _Session:
        def get(
            self,
            url,
            *,
            headers = None,
            timeout = None,
            **_k,
        ):
            return response_factory(url)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)


def _response(
    status: int,
    *,
    error_code: str | None = None,
    url: str | None = None,
):
    headers = {"X-Error-Code": error_code} if error_code else {}
    return SimpleNamespace(
        status_code = status,
        headers = headers,
        url = url,
        raise_for_status = _raiser(status, url),
    )


def _raiser(status: int, url):
    def _raise():
        if status < 400:
            return
        import httpx

        request = httpx.Request("GET", url or "http://mirror.invalid/x")
        raise httpx.HTTPStatusError(
            f"{status}",
            request = request,
            response = httpx.Response(status, request = request),
        )

    return _raise


def test_an_offline_explicit_token_reads_a_repo_already_on_disk(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    assert probes["n"] == 0, "the offline answer went to the network"


def test_the_callers_own_offline_contract_reads_a_repo_on_disk(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK, offline = True) is True
    assert probes["n"] == 0


def test_an_offline_caller_is_still_denied_a_repo_that_is_not_on_disk(monkeypatch, tmp_path):
    _cache_root(monkeypatch, tmp_path)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ABSENT) is False
    assert public_cache_read_authorized(repo_id = ABSENT) is False


def test_an_interrupted_download_is_not_a_repo_on_disk(monkeypatch, tmp_path):
    """BOUNDARY. A cancelled download leaves an empty revision, which answers nothing."""
    root = _cache_root(monkeypatch, tmp_path)
    (root / f"models--{ON_DISK.replace('/', '--')}" / "snapshots" / REVISION).mkdir(parents = True)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


def test_an_anonymous_caller_keeps_its_own_cached_repo_offline(monkeypatch, tmp_path):
    _no_host_credential(monkeypatch)
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is True
    assert cached_read_refused(False, repo_id = ON_DISK, is_cached = lambda: True) is False


def test_the_operators_own_session_is_unchanged(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = True)

    assert cache_reads_authorized(AmbientAuthorizedToken("hf_ui"), repo_id = ON_DISK) is True
    assert cache_reads_authorized(None, repo_id = ON_DISK) is True


def test_an_unreachable_hub_does_not_deny_a_repo_on_disk(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True


def test_an_unreachable_probe_is_memoized_as_unknown_not_as_a_denial(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    probes = _counting_probe(
        monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False
    )

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert [v for _expiry, v in hf_tokens._repo_access_cache.values()] == [None]

    _materialize_repo(root, ON_DISK)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    assert probes["n"] == 1, "the memo did not spare the second caller the dead round trip"


def test_a_hub_that_answered_no_still_denies_a_repo_on_disk(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert cached_read_refused(OPERATOR_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_an_answered_no_is_not_overturned_by_a_later_outage(monkeypatch, tmp_path):
    """429 and 5xx are unaskable while reachable, so a denied caller is one outage away."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert (
        cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    ), "an outage handed out a repo the Hub had already refused this caller"
    assert cached_read_refused(OPERATOR_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_waiting_does_not_hand_back_a_repo_the_hub_refused(monkeypatch, tmp_path):
    """An expiry made the boundary defeatable with the one input the caller controls: the clock."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False

    later = hf_tokens.time.monotonic() + 86_400.0
    monkeypatch.setattr(hf_tokens, "time", SimpleNamespace(monotonic = lambda: later))
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert (
        cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    ), "the refusal was handed back to the caller by the passage of time alone"
    assert cached_read_refused(OPERATOR_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_a_remembered_denial_is_dropped_once_the_hub_says_yes(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, True, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True


def test_a_repo_fetched_with_a_one_off_token_is_not_served_to_a_tokenless_caller(
    monkeypatch, tmp_path
):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ()))

    recorded: dict = {}
    monkeypatch.setattr(
        hf_tokens,
        "_repo_was_fetched_with_a_request_token",
        lambda repo_id, repo_type: recorded.get(
            hf_tokens._request_token_repo_key(repo_id, repo_type), False
        ),
    )
    assert public_cache_read_authorized(repo_id = ON_DISK) is True

    recorded[hf_tokens._request_token_repo_key(ON_DISK, "model")] = True
    hf_tokens._repo_access_cache.clear()
    assert (
        public_cache_read_authorized(repo_id = ON_DISK) is False
    ), "a repo downloaded with a one-off token was served to a caller with no token"
    _materialize_repo(root, "acme/other")
    hf_tokens._repo_access_cache.clear()
    assert public_cache_read_authorized(repo_id = "acme/other") is True

    monkeypatch.setattr(
        hf_tokens, "_repo_was_fetched_with_a_request_token", lambda repo_id, repo_type: None
    )
    hf_tokens._repo_access_cache.clear()
    assert public_cache_read_authorized(repo_id = "acme/other") is False


def test_the_provenance_key_is_case_folded_like_every_other_repo_lookup():
    assert hf_tokens._request_token_repo_key("Org/Private", "model") == (
        hf_tokens._request_token_repo_key("org/private", "MODEL")
    )
    assert hf_tokens._request_token_repo_key("org/private", "dataset") != (
        hf_tokens._request_token_repo_key("org/private", "model")
    )


def test_every_route_that_can_fetch_with_a_one_off_token_records_it():
    import inspect

    from hub.services import download_lifecycle
    from routes import inference as inference_routes
    from routes import video as video_routes

    lifecycle = inspect.getsource(download_lifecycle)
    assert "note_repo_fetched_with_a_request_token(hf_token, repo_id, repo_type)" in lifecycle
    assert lifecycle.index("note_repo_fetched_with_a_request_token") < lifecycle.index(
        "proc = spawn()"
    )

    text_load = inspect.getsource(inference_routes._load_model_impl)
    assert "_note_load_fetched_with_a_request_token(request.model_path" in text_load
    assert text_load.rindex("finally:") < text_load.index("_note_load_fetched_with_a_request_token")
    assert "_load_fetched_bytes(" in text_load

    # The media loads decide at ENTRY: `begin_load` returns before the worker moves a byte, so a
    # before/after comparison in the route would record nothing.
    for media_load in (
        inspect.getsource(inference_routes.load_diffusion_model_gated),
        inspect.getsource(video_routes.load_video_model_gated),
    ):
        assert "_repo_is_in_the_hub_cache(_ref) is not True" in media_load
        assert "_note_load_fetched_with_a_request_token(_ref" in media_load
        assert (
            "_load_fetched_bytes" not in media_load
        ), "a media route cannot compare before and after: its fetch has not happened yet"

    assert "_note_load_fetched_with_a_request_token" not in inspect.getsource(
        inference_routes.load_model_gated
    ), "the route records again, before the load is admitted"

    from hub.services.models import downloads

    assert "note_repo_fetched_with_a_request_token" not in inspect.getsource(downloads)


def test_every_credentialed_fetch_is_recorded_not_only_a_foreign_one(monkeypatch):
    calls: list = []
    monkeypatch.setattr(hf_tokens, "_as_owner", lambda call, *a, **k: calls.append(a))
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ("hf_theoperators",)))

    def written():
        return [call for call in calls if len(call) == 3]

    hf_tokens.note_repo_fetched_with_a_request_token(False, "acme/public", "model")
    assert written() == [], "an anonymous fetch says nothing about anybody"

    hf_tokens.note_repo_fetched_with_a_request_token("hf_theoperators", "acme/saved", "model")
    hf_tokens.note_repo_fetched_with_a_request_token(None, "acme/ambient", "model")
    assert [call[1] for call in written()] == [
        hf_tokens._request_token_repo_key("acme/saved", "model"),
        hf_tokens._request_token_repo_key("acme/ambient", "model"),
    ]

    hf_tokens.note_repo_fetched_with_a_request_token("hf_someoneelses", "acme/private", "model")
    assert written()[-1][1] == hf_tokens._request_token_repo_key("acme/private", "model")

    calls.clear()
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ()))
    hf_tokens.note_repo_fetched_with_a_request_token(None, "acme/public2", "model")
    assert written() == []


def test_a_deleted_credential_does_not_reopen_the_cache(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, None))
    monkeypatch.setattr(hf_tokens, "_repo_present_on_disk", lambda repo, repo_type: True)
    monkeypatch.setattr(
        hf_tokens,
        "_recorded_request_token_repos",
        lambda: {hf_tokens._request_token_repo_key("acme/private", "model"): {"at": 0}},
    )
    hf_tokens.reset_repo_access_cache()

    assert not hf_tokens.public_cache_read_authorized(repo_id = "acme/private", offline = True)
    assert hf_tokens.public_cache_read_authorized(repo_id = "acme/public", offline = True)


def test_a_repo_that_was_never_refused_still_resolves_against_the_disk(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _materialize_repo(root, "acme/other")
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = "acme/other") is True
    assert (
        cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False
    ), "a second principal inherited the operator's cache through an outage"


def test_a_refusal_the_table_had_to_forget_closes_the_outage_path(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(hf_tokens, "_DENIAL_MEMORY_MAX", 4)

    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert not hf_tokens._denial_memory_lost_an_entry()

    # The filler has to be repos that EARN a slot, which now means on this disk and refused to
    # a credential this host holds: a denial the disk fallback could never overturn no longer
    # occupies one, precisely so a caller naming absent repositories cannot force this
    # eviction. The claim under test is unchanged -- once the table HAS forgotten, an outage
    # must not read a forgotten refusal as "never refused".
    for index in range(8):
        filler = f"acme/filler-{index}"
        _materialize_repo(root, filler)
        hf_tokens._repo_access_cache.clear()
        assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = filler) is False
    assert hf_tokens._denial_memory_lost_an_entry(), "the table dropped nothing"

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert (
        cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    ), "a forgotten refusal read as never refused, and the outage served the cached repo"


def test_forgetting_one_refusal_is_not_a_reason_to_refuse_an_answered_hub(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(hf_tokens, "_DENIAL_MEMORY_MAX", 1)
    hf_tokens._remember_denial(("acme/one", "model", "hash"), 1.0)
    hf_tokens._remember_denial(("acme/two", "model", "hash"), 2.0)
    assert hf_tokens._denial_memory_lost_an_entry()

    _counting_probe(monkeypatch, True, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True

    reset_repo_access_cache()
    assert (
        not hf_tokens._denial_memory_lost_an_entry()
    ), "the reset left the fail-closed marker set for every following test"


def test_a_refusal_still_being_re_asked_is_not_the_one_dropped(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_DENIAL_MEMORY_MAX", 3)
    keys = [(f"acme/{index}", "model", "hash") for index in range(3)]
    for at, key in enumerate(keys):
        hf_tokens._remember_denial(key, float(at))
    hf_tokens._remember_denial(keys[0], 9.0)
    hf_tokens._remember_denial(("acme/new", "model", "hash"), 10.0)
    assert hf_tokens._denial_is_remembered(keys[0])
    assert not hf_tokens._denial_is_remembered(keys[1])


def test_the_denial_memory_is_dropped_by_the_test_reset(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert hf_tokens._denied_repo_access

    reset_repo_access_cache()
    assert not hf_tokens._denied_repo_access


# "The repo is on this disk" is a fact about the OPERATOR, not about whoever is asking, and the
# remembered denial does not close the gap: it may never have been collected, and is lost on
# restart.


def test_a_foreign_credential_is_not_handed_the_operators_cached_repo_offline(
    monkeypatch, tmp_path
):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False
    assert cached_read_refused(FOREIGN_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True
    assert probes["n"] == 0
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True


@pytest.mark.parametrize(
    "failure",
    [
        requests.exceptions.ConnectionError("refused"),
        requests.exceptions.ReadTimeout("slow"),
    ],
    ids = ["unreachable", "timeout"],
)
def test_a_foreign_credential_is_refused_through_every_unaskable_shape(
    monkeypatch, tmp_path, failure
):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, failure, offline = False)

    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False


def test_a_credential_less_caller_is_refused_on_a_host_that_holds_a_token(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is False
    assert cached_read_refused(False, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_the_ownership_check_reads_the_hosts_token_every_time(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, "hf_rotated"))
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert cache_reads_authorized("hf_rotated", repo_id = ON_DISK) is True


def test_the_ambient_token_reader_prefers_the_hub_and_falls_back_to_the_env(monkeypatch):
    read = _REAL_AMBIENT_HF_TOKEN
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "  hf_from_hub  ")
    for key in hf_tokens._HF_TOKEN_ENV_KEYS:
        monkeypatch.delenv(key, raising = False)
    assert read() == (True, "hf_from_hub")

    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert read() == (True, None)

    # Asked and NOT answered is the third outcome; collapsing it into "no credential" is a
    # fail-open, since the token file could hold one this process cannot read.
    def _raises():
        raise OSError("token file unreadable")

    monkeypatch.setattr("huggingface_hub.get_token", _raises)
    assert read() == (False, None)
    assert hf_tokens._caller_populated_the_cache(None) is False
    assert hf_tokens._caller_populated_the_cache("hf_anything") is False


def test_the_ui_token_that_filled_the_cache_is_authorized_without_a_global_token(
    monkeypatch, tmp_path
):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _no_host_credential(monkeypatch)
    _saved_ui_credential(monkeypatch)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(STUDIO_UI_TOKEN, repo_id = ON_DISK) is True
    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False
    assert public_cache_read_authorized(repo_id = ON_DISK) is False


def test_either_store_answers_for_the_host(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _saved_ui_credential(monkeypatch, OPERATOR_TOKEN)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False

    _no_host_credential(monkeypatch)
    _saved_ui_credential(monkeypatch)
    assert cache_reads_authorized(STUDIO_UI_TOKEN, repo_id = ON_DISK) is True
    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False


def test_a_credential_less_caller_still_needs_both_stores_empty(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _no_host_credential(monkeypatch)
    _saved_ui_credential(monkeypatch)
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is False

    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, None))
    assert public_cache_read_authorized(repo_id = ON_DISK) is True


def test_an_unreadable_credential_store_authorizes_nobody(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (False, None))
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert public_cache_read_authorized(repo_id = ON_DISK) is False


def test_an_unreadable_saved_credential_is_unknown_rather_than_absent(monkeypatch):
    """`get_secret` answers None for an absent row AND an undecryptable one; the row EXISTING
    separates them, readable without decrypting anything."""
    from storage import credential_secrets

    read = _REAL_SAVED_STUDIO_HF_TOKEN
    monkeypatch.setattr(credential_secrets, "get_hf_token", lambda: None)

    monkeypatch.setattr(credential_secrets, "hf_token_row_exists", lambda: True)
    assert read() == (False, None), "an unreadable saved credential must not read as absent"

    monkeypatch.setattr(credential_secrets, "hf_token_row_exists", lambda: False)
    assert read() == (True, None), "a host with nothing saved is still a real answer"

    def _raises():
        raise RuntimeError("credential database is locked")

    monkeypatch.setattr(credential_secrets, "hf_token_row_exists", _raises)
    assert read() == (False, None)


def test_an_unreadable_saved_credential_refuses_the_anonymous_caller(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _no_host_credential(monkeypatch)
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (False, None))
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is False
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


def test_the_row_check_does_not_need_to_decrypt(monkeypatch):
    from pathlib import Path as _Path

    from storage import credential_secrets

    source = _Path(credential_secrets.__file__).read_text(encoding = "utf-8")
    body = source.split("def secret_row_exists(", 1)[1].split("\ndef ", 1)[0]
    # Comments out: the docstring's own prose would otherwise be read as code.
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    code = code.split('"""')[0] + "".join(code.split('"""')[2:])
    assert "get_secret" not in code, "the row check goes back through the decrypting reader"
    assert "SELECT 1" in code


def test_the_saved_token_reader_reads_the_studio_credential_store(monkeypatch):
    from storage import credential_secrets

    read = _REAL_SAVED_STUDIO_HF_TOKEN
    monkeypatch.setattr(credential_secrets, "get_hf_token", lambda: "  hf_from_store  ")
    assert read() == (True, "hf_from_store")

    monkeypatch.setattr(credential_secrets, "get_hf_token", lambda: None)
    assert read() == (True, None)

    def _raises():
        raise RuntimeError("credential database is locked")

    monkeypatch.setattr(credential_secrets, "get_hf_token", _raises)
    assert read() == (False, None)
    assert hf_tokens._caller_populated_the_cache(None) is False
    assert hf_tokens._caller_populated_the_cache("hf_anything") is False


@pytest.mark.parametrize(
    "status, error_code, expected",
    [
        (200, None, True),
        (401, None, False),
        (403, None, False),
        (404, "RepoNotFound", False),
        (410, None, False),
        # Could not be asked: none of these is a statement about this caller.
        (404, None, None),  # an HF_ENDPOINT mirror with no /auth-check route
        (405, None, None),  # a mirror that rejects the method
        (429, None, None),  # rate limited
        (500, None, None),  # outage
        (502, None, None),  # gateway in front of the Hub
    ],
)
def test_the_probe_separates_an_answer_from_a_failure_to_answer(
    monkeypatch, status, error_code, expected
):
    _probe_against(monkeypatch, lambda url: _response(status, error_code = error_code, url = url))

    assert hf_tokens._probe_repo_access(ON_DISK, OPERATOR_TOKEN, "model") is expected


def test_a_redirected_probe_answers_nothing_about_the_repo(monkeypatch):
    _probe_against(
        monkeypatch,
        lambda url: SimpleNamespace(
            status_code = 200,
            headers = {},
            raise_for_status = lambda: None,
            url = "https://huggingface.co/login",
        ),
    )

    assert hf_tokens._probe_repo_access(ON_DISK, OPERATOR_TOKEN, "model") is None


def test_a_probe_timeout_is_not_an_answer(monkeypatch):
    def _stall(*_a, **_k):
        raise requests.exceptions.Timeout("stalled")

    class _Session:
        get = staticmethod(_stall)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())

    with pytest.raises(hf_tokens._ProbeTimedOut):
        hf_tokens._probe_repo_access(ON_DISK, OPERATOR_TOKEN, "model")


def test_the_presence_check_never_reaches_the_network(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)

    def _forbidden(*_a, **_k):
        raise AssertionError("the local presence check opened a connection")

    monkeypatch.setattr("huggingface_hub.utils.get_session", _forbidden)

    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is True
    assert hf_tokens._repo_present_on_disk(ABSENT, "model") is False


def test_a_dataset_counts_when_only_the_prepared_cache_holds_it(monkeypatch, tmp_path):
    _cache_root(monkeypatch, tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    monkeypatch.setattr(
        dataset_cache,
        "latest_processed_dataset_cache_path",
        lambda repo_id: prepared if repo_id == "acme/ds" else None,
    )

    assert hf_tokens._repo_present_on_disk("acme/ds", "dataset") is True
    assert hf_tokens._repo_present_on_disk("acme/other", "dataset") is False


def test_an_unreadable_cache_does_not_authorize_anything(monkeypatch, tmp_path):
    _cache_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        hf_cache_state,
        "repo_cache_has_usable_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("EIO")),
    )
    _counting_probe(monkeypatch, True, offline = True)

    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is False
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


def test_tier_detection_reads_a_cached_config_offline(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(
        transformers_version,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: True)
    _counting_probe(monkeypatch, True, offline = True)
    transformers_version._config_json_cache.clear()

    assert transformers_version._load_config_json(ON_DISK, OPERATOR_TOKEN) == {
        "model_type": "llama"
    }


def test_the_capability_probes_cache_only_gate_allows_a_repo_on_disk(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert (
        model_config_module._offline_cache_read_refused(OPERATOR_TOKEN, ON_DISK, ON_DISK, True)
        is False
    )
    assert (
        model_config_module._offline_cache_read_refused(OPERATOR_TOKEN, ABSENT, ABSENT, True)
        is True
    )


def test_the_picker_serves_a_cached_chat_template_offline(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    snapshot = _materialize_repo(root, ON_DISK)
    (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding = "utf-8")
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: True)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter([snapshot])
    )
    _counting_probe(monkeypatch, True, offline = True)

    assert picker_service.read_default_chat_template(ON_DISK, OPERATOR_TOKEN) == "{{ messages }}"


def test_the_picker_still_withholds_a_template_the_hub_refused(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    snapshot = _materialize_repo(root, ON_DISK)
    (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding = "utf-8")
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: True)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter([snapshot])
    )
    _counting_probe(monkeypatch, False, offline = False)

    assert picker_service.read_default_chat_template(ON_DISK, OPERATOR_TOKEN) is None


def _stub_autoconfig(monkeypatch) -> dict:
    import transformers

    seen: dict = {}

    def _from_pretrained(_cls_or_name, *args, **kwargs):
        name = args[0] if args else _cls_or_name
        seen["name"] = name
        seen["local_files_only"] = kwargs.get("local_files_only")
        return SimpleNamespace(model_type = "llama")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", classmethod(_from_pretrained))
    return seen


def test_an_anonymous_cache_only_config_read_serves_a_repo_on_disk(monkeypatch, tmp_path):
    """The sentinel's branch refused EVERY cache-only read, on the one host where the cache is the
    only answer there is."""
    _no_host_credential(monkeypatch)
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    probes = _counting_probe(monkeypatch, True, offline = True)
    seen = _stub_autoconfig(monkeypatch)

    config = model_config_module.load_model_config(ON_DISK, token = False, local_files_only = True)

    assert config is not None
    assert seen["name"] == ON_DISK and seen["local_files_only"] is True
    assert probes["n"] == 0, "a cache-only read went to the network"


def test_an_anonymous_config_read_is_still_refused_an_answered_no(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    _counting_probe(monkeypatch, False, offline = False)
    _stub_autoconfig(monkeypatch)

    with pytest.raises(OSError):
        model_config_module.load_model_config(ON_DISK, token = False)

    assert cached_read_refused(False, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_one_unreadable_cache_root_does_not_make_every_repo_present(monkeypatch, tmp_path):
    """``repo_cache_has_usable_snapshot`` answers True for an unenumerable root, which here would
    declare every repo present."""
    root = tmp_path / "hub"
    root.mkdir()

    def _roots(scan_errors = None):
        if scan_errors is not None:
            scan_errors.append(OSError("EACCES"))
        return []

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", _roots)
    assert hf_cache_state.repo_cache_has_usable_snapshot("model", ON_DISK) is True
    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is False

    _counting_probe(monkeypatch, True, offline = True)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


def _offline_route_guard(monkeypatch, tmp_path, repo_id: str, *, on_disk: bool):
    import utils.utils as utils_module

    _no_host_credential(monkeypatch)
    root = _cache_root(monkeypatch, tmp_path)
    if on_disk:
        _materialize_repo(root, repo_id)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: True)
    return utils_module.anonymous_and_offline(False, repo_id = repo_id)


def test_the_route_entry_guard_serves_a_repo_that_is_on_disk(monkeypatch, tmp_path):
    assert _offline_route_guard(monkeypatch, tmp_path, ON_DISK, on_disk = True) is False


def test_the_route_entry_guard_still_refuses_a_repo_that_is_not_on_disk(monkeypatch, tmp_path):
    assert _offline_route_guard(monkeypatch, tmp_path, ABSENT, on_disk = False) is True


def test_the_route_entry_guard_keeps_refusing_on_a_host_that_holds_a_token(monkeypatch, tmp_path):
    import utils.utils as utils_module

    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: True)
    assert utils_module.anonymous_and_offline(False, repo_id = ON_DISK) is True


def test_the_route_entry_guard_keeps_its_blanket_answer_without_a_repo_id(monkeypatch, tmp_path):
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    assert utils_module.anonymous_and_offline(False) is True
    assert utils_module.anonymous_and_offline(None) is False


def test_the_route_entry_guard_leaves_online_behaviour_exactly_as_it_was(monkeypatch, tmp_path):
    import utils.utils as utils_module

    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: False)
    probes = _counting_probe(monkeypatch, False)
    assert utils_module.anonymous_and_offline(False, repo_id = ON_DISK) is False
    assert probes["n"] == 0


def test_two_different_host_credentials_authorize_neither_of_them(monkeypatch, tmp_path):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)
    _saved_ui_credential(monkeypatch)  # STUDIO_UI_TOKEN, alongside the ambient OPERATOR_TOKEN

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert cache_reads_authorized(STUDIO_UI_TOKEN, repo_id = ON_DISK) is False

    _saved_ui_credential(monkeypatch, OPERATOR_TOKEN)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True


def test_a_repo_already_recorded_is_not_written_again(monkeypatch):
    writes: list = []
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ("hf_theoperators",)))
    monkeypatch.setattr(
        hf_tokens,
        "_recorded_request_token_repos",
        lambda: {hf_tokens._request_token_repo_key("acme/private", "model"): {"at": 1.0}},
    )
    monkeypatch.setattr(hf_tokens, "_as_owner", lambda call, *a, **k: writes.append(a))

    hf_tokens.note_repo_fetched_with_a_request_token("hf_someoneelses", "acme/private", "model")
    assert writes == []
    hf_tokens.note_repo_fetched_with_a_request_token("hf_someoneelses", "acme/other", "model")
    assert len(writes) == 1


def test_the_provenance_map_is_bounded_and_says_so_when_it_is_full(monkeypatch):
    writes: list = []
    full = {
        f"model:acme/repo-{index}": {"at": 1.0}
        for index in range(hf_tokens._REQUEST_TOKEN_REPOS_MAX)
    }
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ("hf_theoperators",)))
    monkeypatch.setattr(hf_tokens, "_recorded_request_token_repos", lambda: full)
    monkeypatch.setattr(hf_tokens, "_as_owner", lambda call, *a, **k: writes.append(a))

    hf_tokens.note_repo_fetched_with_a_request_token("hf_someoneelses", "acme/new", "model")
    assert writes == [], "the map kept growing on a caller-supplied repo id"
    assert hf_tokens._repo_was_fetched_with_a_request_token("acme/new", "model") is None
    assert hf_tokens._repo_was_fetched_with_a_request_token("acme/repo-0", "model") is True
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ()))
    assert (
        hf_tokens._caller_populated_the_cache(None, repo_id = "acme/new", repo_type = "model") is False
    )


def test_a_map_below_the_cap_still_answers_no_for_a_repo_it_does_not_hold(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_recorded_request_token_repos", lambda: {})
    assert hf_tokens._repo_was_fetched_with_a_request_token("acme/other", "model") is False


def test_a_lora_loads_base_is_recorded_too(monkeypatch):
    from routes import inference as inference_routes
    from utils import transformers_version

    recorded: list = []
    monkeypatch.setattr(
        transformers_version, "_adapter_base_from_hf_cache", lambda repo: "acme/private-base"
    )
    monkeypatch.setattr(
        inference_routes,
        "_note_load_fetched_with_a_request_token",
        lambda ref, token: recorded.append((ref, token)),
    )
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)
    inference_routes._note_lora_base_fetched_with_a_request_token("acme/adapter", "hf_someoneelses")
    assert recorded == [("acme/private-base", "hf_someoneelses")]


def test_a_base_that_cannot_be_resolved_records_nothing(monkeypatch):
    from routes import inference as inference_routes
    from utils import transformers_version

    recorded: list = []
    monkeypatch.setattr(
        inference_routes,
        "_note_load_fetched_with_a_request_token",
        lambda ref, token: recorded.append(ref),
    )
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)
    monkeypatch.setattr(transformers_version, "_adapter_base_from_hf_cache", lambda repo: None)
    inference_routes._note_lora_base_fetched_with_a_request_token("acme/adapter", "hf_x")
    monkeypatch.setattr(
        transformers_version, "_adapter_base_from_hf_cache", lambda repo: "acme/never-pulled"
    )
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: False)
    inference_routes._note_lora_base_fetched_with_a_request_token("acme/adapter", "hf_x")
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)

    def _raises(_repo):
        raise OSError(13, "denied")

    monkeypatch.setattr(transformers_version, "_adapter_base_from_hf_cache", _raises)
    inference_routes._note_lora_base_fetched_with_a_request_token("acme/adapter", "hf_x")
    assert recorded == []


def test_the_load_reads_the_base_after_the_fetch_not_before_it():
    import inspect

    from routes import inference as inference_routes

    impl = inspect.getsource(inference_routes._load_model_impl)
    assert "_note_lora_base_fetched_with_a_request_token(" in impl
    assert impl.rindex("finally:") < impl.index("_note_lora_base_fetched_with_a_request_token(")
    assert "already_cached = _lora_base_before_this_load" in impl
    assert impl.index("_lora_base_before_this_load = ") < impl.rindex("finally:")


def test_only_a_repo_this_load_actually_pulled_is_recorded(monkeypatch):
    from routes import inference as inference_routes

    recorded: list = []
    monkeypatch.setattr(
        inference_routes,
        "_note_load_fetched_with_a_request_token",
        lambda ref, token: recorded.append(ref),
    )

    def _record(before, after):
        recorded.clear()
        if before is False and after:
            inference_routes._note_load_fetched_with_a_request_token("acme/private", "hf_x")
        return list(recorded)

    assert _record(False, True) == ["acme/private"], "a repo this load pulled was not recorded"
    assert _record(True, True) == [], "a repo that was already cached was recorded as fetched"
    assert _record(False, False) == [], "a load that fetched nothing recorded a fetch"
    assert _record(None, True) == [], "an unanswerable reading recorded a fetch"


def test_the_presence_probe_answers_none_for_anything_that_is_not_a_repo(monkeypatch):
    from routes import inference as inference_routes

    for local in ("/srv/models/model.gguf", "./model.gguf", "~/model.gguf", "C:/models", ""):
        assert inference_routes._repo_is_in_the_hub_cache(local) is None, local

    from hub.utils import hf_tokens as _hf_tokens

    monkeypatch.setattr(_hf_tokens, "_repo_present_on_disk", lambda repo, kind: True)
    assert inference_routes._repo_is_in_the_hub_cache("acme/model") is True

    def _raises(_repo, _kind):
        raise OSError(13, "denied")

    monkeypatch.setattr(_hf_tokens, "_repo_present_on_disk", _raises)
    assert inference_routes._repo_is_in_the_hub_cache("acme/model") is None


def test_a_load_that_added_blobs_to_an_existing_repo_is_recorded(monkeypatch, tmp_path):
    from routes import inference as inference_routes

    blobs = tmp_path / "models--acme--private" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "a").write_bytes(b"x" * 10)
    monkeypatch.setattr(
        hf_cache_state,
        "iter_repo_cache_dirs",
        lambda repo_type, repo_id, **kw: iter([tmp_path / "models--acme--private"]),
    )

    before = inference_routes._hub_cache_footprint("acme/private")
    assert before == (1, 10)
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)
    assert not inference_routes._load_fetched_bytes(
        "acme/private", True, before
    ), "a pure cache hit was recorded as a fetch"

    (blobs / "b").write_bytes(b"y" * 4096)  # the blob this load pulled
    assert inference_routes._load_fetched_bytes(
        "acme/private", True, before
    ), "a credentialed refetch of an existing repo was lost"
    assert not inference_routes._load_fetched_bytes(
        "acme/private", True, None
    ), "an unreadable footprint guessed instead of abstaining"

    assert inference_routes._load_fetched_bytes("acme/private", False, None)


def test_refs_and_no_exist_markers_are_not_read_as_a_fetch(monkeypatch, tmp_path):
    from routes import inference as inference_routes

    repo_dir = tmp_path / "models--acme--public"
    (repo_dir / "blobs").mkdir(parents = True)
    (repo_dir / "blobs" / "a").write_bytes(b"x" * 32)
    monkeypatch.setattr(
        hf_cache_state,
        "iter_repo_cache_dirs",
        lambda repo_type, repo_id, **kw: iter([repo_dir]),
    )
    before = inference_routes._hub_cache_footprint("acme/public")

    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef")
    (repo_dir / ".no_exist" / "deadbeef").mkdir(parents = True)
    (repo_dir / ".no_exist" / "deadbeef" / "adapter_config.json").write_text("")
    assert inference_routes._hub_cache_footprint("acme/public") == before


def test_a_base_that_was_already_cached_is_not_marked_as_fetched(monkeypatch):
    from routes import inference as inference_routes

    recorded: list = []
    monkeypatch.setattr(
        transformers_version, "_adapter_base_from_hf_cache", lambda repo: "acme/public-base"
    )
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)
    monkeypatch.setattr(
        inference_routes,
        "_note_load_fetched_with_a_request_token",
        lambda ref, token: recorded.append(ref),
    )

    before = inference_routes._lora_base_already_in_the_hub_cache("acme/adapter")
    assert before == "acme/public-base"
    inference_routes._note_lora_base_fetched_with_a_request_token(
        "acme/adapter", "hf_someoneelses", already_cached = before
    )
    assert recorded == [], "a base that was already cached was recorded as fetched"

    inference_routes._note_lora_base_fetched_with_a_request_token(
        "acme/adapter", "hf_someoneelses", already_cached = None
    )
    assert recorded == ["acme/public-base"]


def test_only_growth_counts_as_a_fetch(monkeypatch, tmp_path):
    from routes import inference as inference_routes

    blobs = tmp_path / "models--acme--public" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "a").write_bytes(b"x" * 4096)
    (blobs / "b").write_bytes(b"y" * 4096)
    monkeypatch.setattr(
        hf_cache_state,
        "iter_repo_cache_dirs",
        lambda repo_type, repo_id, **kw: iter([tmp_path / "models--acme--public"]),
    )
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)
    before = inference_routes._hub_cache_footprint("acme/public")
    assert before == (2, 8192)

    (blobs / "b").unlink()  # a prune, not a fetch
    assert not inference_routes._load_fetched_bytes("acme/public", True, before)

    monkeypatch.setattr(inference_routes, "_hub_cache_footprint", lambda ref: None)
    assert not inference_routes._load_fetched_bytes(
        "acme/public", True, before
    ), "an unreadable after-reading was counted as a fetch"

    # Growth in either component is a fetch: a new blob moves the count, an appended partial
    # moves only the bytes.
    monkeypatch.setattr(inference_routes, "_hub_cache_footprint", lambda ref: (3, 8192))
    assert inference_routes._load_fetched_bytes("acme/public", True, before)
    monkeypatch.setattr(inference_routes, "_hub_cache_footprint", lambda ref: (2, 9000))
    assert inference_routes._load_fetched_bytes("acme/public", True, before)


def test_a_media_load_records_at_entry_because_its_fetch_is_on_a_worker_thread():
    import inspect

    from core.inference import diffusion, video as video_core
    for begin in (diffusion.DiffusionBackend.begin_load, video_core.VideoBackend.begin_load):
        doc = inspect.getdoc(begin) or ""
        assert "Returns at once" in doc or "daemon thread" in doc, (
            f"{begin.__qualname__} no longer hands off; the media routes could compare "
            "before and after like the text load does"
        )


# ---------------------------------------------------------------------------------------
# The denial memory is a fixed number of slots keyed partly on caller-supplied input, so what
# is allowed to occupy one decides whether a caller can exhaust it. These pin that only a
# denial the disk fallback could actually overturn is stored.
# ---------------------------------------------------------------------------------------


def test_a_denial_for_a_repo_not_on_disk_does_not_occupy_a_slot(monkeypatch, tmp_path):
    """The flood shape: 401s for repositories that do not exist. Nothing on disk can overturn
    them, so remembering them protects nothing and only spends the memory."""
    _cache_root(monkeypatch, tmp_path)
    _counting_probe(monkeypatch, False)

    for i in range(64):
        assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = f"attacker/absent-{i}") is False

    assert len(hf_tokens._denied_repo_access) == 0
    assert hf_tokens._denial_memory_lost_an_entry() is False


def test_a_denial_for_a_credential_this_host_never_held_does_not_occupy_a_slot(
    monkeypatch, tmp_path
):
    """The other flood shape: one real on-disk repo, many caller-supplied tokens. A token the
    host does not hold can never be authorized by the fallback, whatever the disk says."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False)

    for i in range(64):
        assert cache_reads_authorized(f"hf_stranger_{i}", repo_id = ON_DISK) is False

    assert len(hf_tokens._denied_repo_access) == 0
    assert hf_tokens._denial_memory_lost_an_entry() is False


def test_a_flood_does_not_take_the_offline_fallback_away_from_everyone_else(monkeypatch, tmp_path):
    """The defect this closes: one evicted entry set `_denial_memory_is_complete` False for the
    life of the process, after which every unaskable probe read as a refusal, for every repo
    and every caller."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False)

    for i in range(hf_tokens._DENIAL_MEMORY_MAX + 16):
        cache_reads_authorized(f"hf_stranger_{i}", repo_id = f"attacker/absent-{i}")

    assert hf_tokens._denial_memory_lost_an_entry() is False
    reset_repo_access_cache()
    _counting_probe(monkeypatch, _ProbeUnreachable(), offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True


def test_a_denial_that_could_be_overturned_is_still_remembered(monkeypatch, tmp_path):
    """The boundary the three above must not move: the Hub refused the operator's OWN
    credential for a repo that IS on this disk, so an outage must not hand it over."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert len(hf_tokens._denied_repo_access) == 1

    _counting_probe(monkeypatch, _ProbeUnreachable(), offline = False)
    # The verdict cache is what expires; the denial is what does not.
    hf_tokens._repo_access_cache.clear()
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


class _ProbeUnreachable(Exception):
    """Stands in for a refused connection: not an answer, so the fallback applies."""


# ---------------------------------------------------------------------------------------
# `load_model_config`'s anonymous branch. The hole it closes is an ONLINE read the Hub
# answered no to. Silence is not that answer, and reading it as one refused a PUBLIC repo
# already on the disk on every host that holds a credential.
# ---------------------------------------------------------------------------------------


def _anonymous_config_read(monkeypatch, tmp_path, *, probe, env_offline: bool):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, probe, offline = False)
    monkeypatch.setattr(model_config_module, "_env_offline", lambda: env_offline)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    served = {"n": 0}

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*_a, **_k):
            served["n"] += 1
            return SimpleNamespace(model_type = "llama")

    # Imported inside the function under test, so the stand-in goes on the package.
    import transformers

    monkeypatch.setattr(transformers, "AutoConfig", _AutoConfig)
    return served


def test_an_unaskable_hub_still_serves_a_cached_config_to_the_anonymous_caller(
    monkeypatch, tmp_path
):
    """The host holds a credential, so the provenance rule cannot vouch for a tokenless
    caller; but the Hub never said no, and the merge base served this. A DNS blip must not
    take a downloaded PUBLIC model away."""
    served = _anonymous_config_read(
        monkeypatch, tmp_path, probe = _ProbeUnreachable(), env_offline = False
    )

    model_config_module.load_model_config(ON_DISK, use_auth = False, token = False)
    assert served["n"] == 1


def test_an_answered_no_still_refuses_the_cached_config(monkeypatch, tmp_path):
    """BOUNDARY. The reverse hole #10264 left open: online, the Hub refuses this caller, and
    the cached config.json was handed over anyway. That must stay closed."""
    served = _anonymous_config_read(monkeypatch, tmp_path, probe = False, env_offline = False)

    with pytest.raises(OSError, match = "not available to an unauthorized caller"):
        model_config_module.load_model_config(ON_DISK, use_auth = False, token = False)
    assert served["n"] == 0


def test_offline_the_narrow_provenance_rule_still_decides(monkeypatch, tmp_path):
    """BOUNDARY. Declared offline is the path the PR narrows on purpose: a tokenless caller on
    a host that HOLDS a credential is refused, because either could have filled the cache."""
    served = _anonymous_config_read(
        monkeypatch, tmp_path, probe = _ProbeUnreachable(), env_offline = True
    )

    with pytest.raises(OSError, match = "not available to an unauthorized caller"):
        model_config_module.load_model_config(ON_DISK, use_auth = False, token = False)
    assert served["n"] == 0


def test_offline_a_credential_less_host_still_reads_its_own_cache(monkeypatch, tmp_path):
    """BOUNDARY. The case the whole PR exists for keeps working through the added clause."""
    _no_host_credential(monkeypatch)
    served = _anonymous_config_read(
        monkeypatch, tmp_path, probe = _ProbeUnreachable(), env_offline = True
    )

    model_config_module.load_model_config(ON_DISK, use_auth = False, token = False)
    assert served["n"] == 1


def test_a_non_ascii_credential_is_compared_not_crashed_on(monkeypatch, tmp_path):
    """`hmac.compare_digest` refuses a str with any non-ASCII character. Both operands here are
    text somebody else chose: the caller's own X-Unsloth-HF-Token header, and the credential the
    host happens to hold. Either one carrying a single high byte turned an authorization
    question into a 500 that quoted the exception."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, _ProbeUnreachable(), offline = False)

    # A caller-supplied header byte. Starlette latin-1 decodes 0xE9 to U+00E9.
    assert cache_reads_authorized("hf_é_not_the_hosts", repo_id = ON_DISK) is False

    # And the mirror image: the HOST's credential is the non-ASCII one, and the caller presents
    # it correctly, so the answer is a real match rather than a crash.
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, "hf_opérateur"))
    assert cache_reads_authorized("hf_opérateur", repo_id = ON_DISK) is True
