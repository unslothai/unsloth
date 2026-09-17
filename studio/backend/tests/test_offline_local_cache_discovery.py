# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub that cannot be ASKED is not a Hub that said no.

The cache-read gate added in #10264 put a live /auth-check round trip in front of every
read of the host's own HF disk cache, and collapsed "could not ask" into "denied". That
made a purely LOCAL question -- is this repo on my disk, what can it do, load it -- depend
on reaching huggingface.co, so an operator offline with a fully downloaded model, an
air-gapped install, a Hub outage, a rate limit, or an HF_ENDPOINT mirror that never
implemented the undocumented /auth-check route was told the model was unavailable.

Every test here pairs the fix with the boundary it must not move: a Hub that ANSWERS no
still refuses the operator's cached copy, and a repo that is not on disk is still refused,
because then there is nothing local to decide with and saying yes would only authorize a
new network fetch. The caches are real directory trees; only the list of cache ROOTS is
redirected, so the presence check runs its real walk over real files.
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

# Captured before the autouse fixture below replaces the attribute, so the one test that
# is ABOUT the reader can still reach the real one.
_REAL_AMBIENT_HF_TOKEN = hf_tokens._ambient_hf_token
_REAL_SAVED_STUDIO_HF_TOKEN = hf_tokens._saved_studio_hf_token

ON_DISK = "acme/downloaded-model"
ABSENT = "acme/never-downloaded"
# The credential this HOST downloads with: the operator's own. The unaskable fallback
# resolves against the disk only for the caller that could have produced what is on it, so
# every "offline still reads my model" case below is the operator's credential by
# construction, and FOREIGN_TOKEN is the second principal that must NOT inherit it.
OPERATOR_TOKEN = "hf_operator_ambient_token"
# The SAME operator, credential saved the other way: through Studio Settings, which writes
# the encrypted credential store rather than the HF token file. `get_token()` never returns
# it, and the UI replays it per request in X-Unsloth-HF-Token. On the ordinary Studio
# install this is the only credential the host holds.
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
    """Give the host an ambient credential, the operator's.

    Autouse because `_ambient_hf_token` reads `huggingface_hub.get_token()` and the real
    environment this suite runs in, which would otherwise decide these tests. Overridden
    per test with `_no_host_credential` where the case is a host that has none.
    """
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OPERATOR_TOKEN))


@pytest.fixture(autouse = True)
def _saved_studio_credential(monkeypatch):
    """Nothing saved in Studio Settings unless the case says so.

    Autouse for the reason `_host_credential` is: this reader opens the real credential
    store, and the machine running the suite must not be what decides these tests.
    """
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, None))


def _no_host_credential(monkeypatch):
    """A host that never configured an HF token: the ordinary install."""
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))


def _saved_ui_credential(monkeypatch, token = STUDIO_UI_TOKEN):
    """A host whose operator saved a token in Studio Settings and configured none globally."""
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, token))


def _materialize_repo(
    root: Path,
    repo_id: str,
    repo_type: str = "model",
    *,
    files: tuple[str, ...] = ("config.json", "model.safetensors"),
) -> Path:
    """Write the repo the way huggingface_hub leaves a completed download."""
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
    """Point every cache-root enumeration at one throwaway tree.

    The presence walk itself is untouched: it iterates this directory for real.
    """
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
    """Drive the real probe against a fake session, so the classifier is what is measured."""

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


# --------------------------------------------------------------------------- declared offline
def test_an_offline_explicit_token_reads_a_repo_already_on_disk(monkeypatch, tmp_path):
    """The headline defect. HF_HUB_OFFLINE=1 plus a fully downloaded repo denied every
    explicit token, so Studio reported the operator's own model as unavailable."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    assert probes["n"] == 0, "the offline answer went to the network"


def test_the_callers_own_offline_contract_reads_a_repo_on_disk(monkeypatch, tmp_path):
    """``offline=True`` is a cache-only request. It must be answered from the cache, not
    refused because the cache-only branch was not allowed to consult the cache."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK, offline = True) is True
    assert probes["n"] == 0


def test_an_offline_caller_is_still_denied_a_repo_that_is_not_on_disk(monkeypatch, tmp_path):
    """BOUNDARY. With nothing local there is no local answer, and yes would only buy a new
    network fetch, so the gate stays shut."""
    _cache_root(monkeypatch, tmp_path)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ABSENT) is False
    assert public_cache_read_authorized(repo_id = ABSENT) is False


def test_an_interrupted_download_is_not_a_repo_on_disk(monkeypatch, tmp_path):
    """BOUNDARY. "On disk" means a snapshot a load could consume. A cancelled download
    leaves a repo directory with an empty revision under it, which answers nothing."""
    root = _cache_root(monkeypatch, tmp_path)
    (root / f"models--{ON_DISK.replace('/', '--')}" / "snapshots" / REVISION).mkdir(parents = True)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


def test_an_anonymous_caller_keeps_its_own_cached_repo_offline(monkeypatch, tmp_path):
    """The sentinel can never authorize itself and the public probe failed closed, so
    offline an API caller with no token could read nothing at all -- public included.

    On a host with NO ambient credential, which is the ordinary install: nothing in that
    cache can have been fetched under a credential this caller lacks, so everything in it
    was public when it was downloaded.
    """
    _no_host_credential(monkeypatch)
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is True
    assert cached_read_refused(False, repo_id = ON_DISK, is_cached = lambda: True) is False


def test_the_operators_own_session_is_unchanged(monkeypatch, tmp_path):
    """BOUNDARY. A UI session was already entitled to ambient and never behind the probe."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = True)

    assert cache_reads_authorized(AmbientAuthorizedToken("hf_ui"), repo_id = ON_DISK) is True
    assert cache_reads_authorized(None, repo_id = ON_DISK) is True


# --------------------------------------------------------------------------- unreachable Hub
def test_an_unreachable_hub_does_not_deny_a_repo_on_disk(monkeypatch, tmp_path):
    """No offline variable set, the Hub simply not there: a dead proxy, a severed link, an
    outage. Nothing about that is a statement on the credential."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True


def test_an_unreachable_probe_is_memoized_as_unknown_not_as_a_denial(monkeypatch, tmp_path):
    """The short TTL existed so a flapping Hub is not re-dialled per call, but it stored a
    denial the Hub never gave: for 30 s every caller was refused, and a download finishing
    inside the window changed nothing. Memoize the fact that nobody answered instead."""
    root = _cache_root(monkeypatch, tmp_path)
    probes = _counting_probe(
        monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False
    )

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert [v for _expiry, v in hf_tokens._repo_access_cache.values()] == [None]

    # Inside the same window, the repo lands on disk. No new probe, and the local fact wins.
    _materialize_repo(root, ON_DISK)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    assert probes["n"] == 1, "the memo did not spare the second caller the dead round trip"


def test_a_hub_that_answered_no_still_denies_a_repo_on_disk(monkeypatch, tmp_path):
    """BOUNDARY, and the whole point of #10264: online, an API key that cannot reach a
    private repo must not be handed the operator's cached copy of it."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert cached_read_refused(OPERATOR_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_an_answered_no_is_not_overturned_by_a_later_outage(monkeypatch, tmp_path):
    """The gate's one boundary has to survive the Hub going away.

    Unaskable resolves against the disk, and a 429 or a 5xx is unaskable while being
    reachable from outside: a burst of probes rate-limits this host's own endpoint. Without a
    memory of the refusal, a caller the Hub denied a minute ago would be one outage away from
    the operator's cached copy of the private repo. The denial is remembered for longer than
    the verdict is cached, so "could not ask" answers no while it stands.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False

    # The verdict cache expires; the Hub is now rate limiting rather than answering.
    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert (
        cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    ), "an outage handed out a repo the Hub had already refused this caller"
    assert cached_read_refused(OPERATOR_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_waiting_does_not_hand_back_a_repo_the_hub_refused(monkeypatch, tmp_path):
    """The memory had a 900 s expiry, which made the boundary defeatable with the one input the
    caller controls. Wait it out, ask again while the Hub is unaskable, and the local resolution
    authorizes on the strength of the operator's disk -- for the same credential the Hub
    refused, with nothing changed but the clock. Only an answer is evidence about access."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False

    # A day passes, and the Hub is rate limiting rather than answering when the caller returns.
    later = hf_tokens.time.monotonic() + 86_400.0
    monkeypatch.setattr(hf_tokens, "time", SimpleNamespace(monotonic = lambda: later))
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert (
        cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    ), "the refusal was handed back to the caller by the passage of time alone"
    assert cached_read_refused(OPERATOR_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_a_remembered_denial_is_dropped_once_the_hub_says_yes(monkeypatch, tmp_path):
    """Not a lockout. Access granted after a refusal takes effect on the next answer, and the
    outage behaviour goes back to reading the disk."""
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
    """The download route takes a ONE-OFF `X-Unsloth-HF-Token` and saves it nowhere.

    So a private repo can be sitting in the cache of a host whose credential set is empty,
    and the tokenless branch reads an empty credential set as "nothing here needed one". The
    provenance is recorded per repo at download time instead of inferred afterwards.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    # A host with no credential of its own, which is the case the fallback exists for.
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ()))

    recorded: dict = {}
    monkeypatch.setattr(
        hf_tokens,
        "_repo_was_fetched_with_a_request_token",
        lambda repo_id, repo_type: recorded.get(
            hf_tokens._request_token_repo_key(repo_id, repo_type), False
        ),
    )
    # The caller class this is about: no credential at all, which is the sentinel
    # `public_cache_read_authorized` answers for.
    assert public_cache_read_authorized(repo_id = ON_DISK) is True

    recorded[hf_tokens._request_token_repo_key(ON_DISK, "model")] = True
    hf_tokens._repo_access_cache.clear()
    assert (
        public_cache_read_authorized(repo_id = ON_DISK) is False
    ), "a repo downloaded with a one-off token was served to a caller with no token"
    # Another repo on the same host is unaffected: the record is per repo.
    _materialize_repo(root, "acme/other")
    hf_tokens._repo_access_cache.clear()
    assert public_cache_read_authorized(repo_id = "acme/other") is True

    # And a record that cannot be read answers nobody.
    monkeypatch.setattr(
        hf_tokens, "_repo_was_fetched_with_a_request_token", lambda repo_id, repo_type: None
    )
    hf_tokens._repo_access_cache.clear()
    assert public_cache_read_authorized(repo_id = "acme/other") is False


def test_the_provenance_key_is_case_folded_like_every_other_repo_lookup():
    """A download recorded as `Org/Private` is later asked for as `org/private`.

    The authorization cache and the cache-directory walk both compare repo ids
    case-insensitively, so a case-sensitive record missed while the disk lookup succeeded --
    which is the tokenless caller being authorized for the private repo.
    """
    assert hf_tokens._request_token_repo_key("Org/Private", "model") == (
        hf_tokens._request_token_repo_key("org/private", "MODEL")
    )
    assert hf_tokens._request_token_repo_key("org/private", "dataset") != (
        hf_tokens._request_token_repo_key("org/private", "model")
    )


def test_every_route_that_can_fetch_with_a_one_off_token_records_it():
    """The record is only sound if nothing writes the cache without leaving one.

    The download lifecycle covers every spawned downloader, and it records in the CLAIMED
    launch path, after the in-flight check, the validation and the registry claim, so a
    request rejected with a 400 or a 409 cannot mark a repo it never fetched. The three load
    routes fetch through their own loaders instead, so each records where the request is.
    """
    import inspect

    from hub.services import download_lifecycle
    from routes import inference as inference_routes
    from routes import video as video_routes

    lifecycle = inspect.getsource(download_lifecycle)
    assert "note_repo_fetched_with_a_request_token(hf_token, repo_id, repo_type)" in lifecycle
    # After the claim, before the spawn.
    assert lifecycle.index("note_repo_fetched_with_a_request_token") < lifecycle.index(
        "proc = spawn()"
    )

    text_load = inspect.getsource(inference_routes.load_model_gated)
    assert "_note_load_fetched_with_a_request_token(request.model_path" in text_load
    image_load = inspect.getsource(inference_routes.load_diffusion_model_gated)
    assert "_note_load_fetched_with_a_request_token(request.model_path" in image_load
    video_load = inspect.getsource(video_routes.load_video_model_gated)
    assert "_note_load_fetched_with_a_request_token(request.model_path" in video_load

    # And the download service no longer records before admission.
    from hub.services.models import downloads

    assert "note_repo_fetched_with_a_request_token" not in inspect.getsource(downloads)


def test_only_a_credential_the_host_does_not_hold_is_recorded(monkeypatch):
    """The note is about a credential that LEAVES no trace on the host.

    An anonymous download says nothing about anybody, and a download with the host's own
    credential is the operator either way -- the tokenless branch already refuses on a host
    that holds any credential at all.
    """
    written: list = []
    monkeypatch.setattr(hf_tokens, "_as_owner", lambda call, *a, **k: written.append(a))
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, ("hf_theoperators",)))

    hf_tokens.note_repo_fetched_with_a_request_token(None, "acme/private", "model")
    hf_tokens.note_repo_fetched_with_a_request_token(False, "acme/private", "model")
    hf_tokens.note_repo_fetched_with_a_request_token("hf_theoperators", "acme/private", "model")
    assert written == []

    hf_tokens.note_repo_fetched_with_a_request_token("hf_someoneelses", "acme/private", "model")
    assert len(written) == 1, written
    assert written[0][1] == hf_tokens._request_token_repo_key("acme/private", "model")


def test_a_repo_that_was_never_refused_still_resolves_against_the_disk(monkeypatch, tmp_path):
    """The memory is per repo and per credential, so an air-gapped host, a Hub outage and a
    mirror with no /auth-check route are all exactly as they were."""
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
    """Eviction is the same hole an expiry would be. A route that takes a repo id collects one
    refusal per id it is not granted, so a caller can push its own earlier refusal out of a
    bounded table and then ask again while the Hub is unaskable -- and an unaskable Hub with
    nothing remembered resolves against the operator's disk."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(hf_tokens, "_DENIAL_MEMORY_MAX", 4)

    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert not hf_tokens._denial_memory_lost_an_entry()

    # The refusals that push it out are for repos this caller was never granted.
    for index in range(8):
        hf_tokens._repo_access_cache.clear()
        assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = f"acme/filler-{index}") is False
    assert hf_tokens._denial_memory_lost_an_entry(), "the table dropped nothing"

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert (
        cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    ), "a forgotten refusal read as never refused, and the outage served the cached repo"


def test_forgetting_one_refusal_is_not_a_reason_to_refuse_an_answered_hub(monkeypatch, tmp_path):
    """Fail closed for the offline fallback only: an answer is still an answer, and a public
    repo is still public."""
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
    """Re-insertion order, so the table sheds the oldest refusal rather than the live one."""
    monkeypatch.setattr(hf_tokens, "_DENIAL_MEMORY_MAX", 3)
    keys = [(f"acme/{index}", "model", "hash") for index in range(3)]
    for at, key in enumerate(keys):
        hf_tokens._remember_denial(key, float(at))
    hf_tokens._remember_denial(keys[0], 9.0)
    hf_tokens._remember_denial(("acme/new", "model", "hash"), 10.0)
    assert hf_tokens._denial_is_remembered(keys[0])
    assert not hf_tokens._denial_is_remembered(keys[1])


def test_the_denial_memory_is_dropped_by_the_test_reset(monkeypatch, tmp_path):
    """It outlives the verdict cache, so the reset has to clear it or one test's refusal
    decides the next test's outage."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert hf_tokens._denied_repo_access

    reset_repo_access_cache()
    assert not hf_tokens._denied_repo_access


# ------------------------------------------------------------------ caller isolation offline
#
# "The repo is on this disk" is a fact about the OPERATOR, not about whoever is asking. A
# fallback that resolves an unanswerable probe against disk presence for ANY caller hands a
# second principal the operator's private downloads the moment the Hub is offline, times
# out, 429s, 5xx's or sits behind an HF_ENDPOINT mirror with no /auth-check route. The
# remembered denial does not close that: there may never have been an online denial to
# remember, a new credential is a new key, and it is lost on restart.
# So presence only answers for a caller that could have produced it.


def test_a_foreign_credential_is_not_handed_the_operators_cached_repo_offline(
    monkeypatch, tmp_path
):
    """THE regression this guards. A holder of a valid Studio API key, with no permission
    to the HF repo, must not receive the operator's private cached copy because the Hub
    happened to be unreachable. Nothing was ever refused online here, so the denial memory
    is empty and only the ownership rule stands between the two callers."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False
    assert cached_read_refused(FOREIGN_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True
    assert probes["n"] == 0
    # and the operator, on the same host and the same repo, still reads it
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
    """Offline is only one of the ways the probe stops answering. A dead link, a timeout, a
    429, a 5xx and a mirror with no route all arrive at the same fallback, and a slow real
    denial is deliberately converted to unaskable, so the ownership rule has to hold for
    all of them rather than for HF_HUB_OFFLINE alone."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, failure, offline = False)

    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False


def test_a_credential_less_caller_is_refused_on_a_host_that_holds_a_token(monkeypatch, tmp_path):
    """The anonymous half of the same hole. On a host whose operator HAS an HF token, the
    cache can hold repos that token fetched, so a caller with no credential at all cannot be
    told "it is on the disk, therefore you may read it"."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is False
    assert cached_read_refused(False, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_the_ownership_check_reads_the_hosts_token_every_time(monkeypatch, tmp_path):
    """Not memoized. An operator who revokes or rotates a token must stop authorizing the
    old one on the very next call, and one who sets a token must stop authorizing the
    credential-less caller from that call on."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, "hf_rotated"))
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert cache_reads_authorized("hf_rotated", repo_id = ON_DISK) is True


def test_the_ambient_token_reader_prefers_the_hub_and_falls_back_to_the_env(monkeypatch):
    """`get_token()` is what a download in this process actually resolves, so it is the
    authority on what filled the cache. The env aliases are the fallback for a hub too old
    to export it, and HF_OIDC_RESOURCE is skipped because it names a token rather than
    holding one, so it could never equal a caller's."""
    read = _REAL_AMBIENT_HF_TOKEN
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "  hf_from_hub  ")
    for key in hf_tokens._HF_TOKEN_ENV_KEYS:
        monkeypatch.delenv(key, raising = False)
    assert read() == (True, "hf_from_hub")

    # The Hub library ANSWERING "none" is an answer, and the only one that lets a
    # credential-less caller read the cache offline.
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert read() == (True, None)

    # Asked and NOT answered is the third outcome. Collapsing it into "this host has no
    # credential" is a fail-open: the token file could hold one this process cannot read.
    def _raises():
        raise OSError("token file unreadable")

    monkeypatch.setattr("huggingface_hub.get_token", _raises)
    assert read() == (False, None)
    assert hf_tokens._caller_populated_the_cache(None) is False
    assert hf_tokens._caller_populated_the_cache("hf_anything") is False


def test_the_ui_token_that_filled_the_cache_is_authorized_without_a_global_token(
    monkeypatch, tmp_path
):
    """The ordinary Studio install, and the one the ambient-only comparison got wrong.

    The operator saves an HF token in Settings, downloads a private model with it, and
    never touches the HF token file or HF_TOKEN. `get_token()` therefore answers "this host
    has no credential" while the host plainly has one, and the UI's own request -- carrying
    the very token the bytes were fetched with, in X-Unsloth-HF-Token -- was refused its own
    cached model the moment the Hub could not be asked. That is the exact flow this PR
    exists to keep working, so it has to be authorized while the second principal is not.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _no_host_credential(monkeypatch)
    _saved_ui_credential(monkeypatch)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(STUDIO_UI_TOKEN, repo_id = ON_DISK) is True
    # The boundary does not move: a different caller's token is still not the host's.
    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False
    # And the host DOES hold a credential, so the credential-less caller is refused even
    # though `get_token()` alone would have said there was nothing to protect.
    assert public_cache_read_authorized(repo_id = ON_DISK) is False


def test_either_store_answers_for_the_host(monkeypatch, tmp_path):
    """Either STORE answers, as long as the host holds one credential.

    A host can hold its credential in the token file, in Studio's credential store, or in
    both: the CLI writes one, Settings writes the other, and which store it came out of says
    nothing about who the caller is. What does say something is whether the two DIFFER, and
    that case is `test_two_different_host_credentials_authorize_neither_of_them`.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _saved_ui_credential(monkeypatch, OPERATOR_TOKEN)
    _counting_probe(monkeypatch, True, offline = True)

    # `_host_credential` leaves the ambient one set to OPERATOR_TOKEN.
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False

    # The saved store alone, with nothing ambient, is the ordinary UI case and still works.
    _no_host_credential(monkeypatch)
    _saved_ui_credential(monkeypatch)
    assert cache_reads_authorized(STUDIO_UI_TOKEN, repo_id = ON_DISK) is True
    assert cache_reads_authorized(FOREIGN_TOKEN, repo_id = ON_DISK) is False


def test_a_credential_less_caller_still_needs_both_stores_empty(monkeypatch, tmp_path):
    """The anonymous branch is presence-based, so it may only fire when the host holds
    nothing at all.

    Reading only the ambient store, a host whose sole credential lives in Studio Settings
    looks tokenless, and an API-key caller with no HF credential is handed a cache that may
    hold private bytes.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _no_host_credential(monkeypatch)
    _saved_ui_credential(monkeypatch)
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is False

    # With BOTH stores empty it is the tokenless offline install, which must still work.
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, None))
    assert public_cache_read_authorized(repo_id = ON_DISK) is True


def test_an_unreadable_credential_store_authorizes_nobody(monkeypatch, tmp_path):
    """The third outcome, on the second store too.

    A locked database or a missing encryption key leaves the host's credential set
    unestablished. Collapsing that into "the host has none" is the same fail-open the
    ambient reader refuses to make, one store over.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (False, None))
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert public_cache_read_authorized(repo_id = ON_DISK) is False


def test_an_unreadable_saved_credential_is_unknown_rather_than_absent(monkeypatch):
    """`get_secret` answers None for an absent row AND for a row it could not decrypt.

    A lost or rotated encryption key, corrupted ciphertext, or a format a newer build wrote
    all arrive as None, and reading that as "this host holds no credential" is a fail-open:
    the credential-less branch would then hand an API-key caller a cache that may hold
    whatever that unreadable token downloaded. The row existing is what separates the two,
    and it is readable without decrypting anything.
    """
    from storage import credential_secrets

    read = _REAL_SAVED_STUDIO_HF_TOKEN
    monkeypatch.setattr(credential_secrets, "get_hf_token", lambda: None)

    monkeypatch.setattr(credential_secrets, "hf_token_row_exists", lambda: True)
    assert read() == (False, None), "an unreadable saved credential must not read as absent"

    monkeypatch.setattr(credential_secrets, "hf_token_row_exists", lambda: False)
    assert read() == (True, None), "a host with nothing saved is still a real answer"

    # The row check failing is itself unanswered, not an answer.
    def _raises():
        raise RuntimeError("credential database is locked")

    monkeypatch.setattr(credential_secrets, "hf_token_row_exists", _raises)
    assert read() == (False, None)


def test_an_unreadable_saved_credential_refuses_the_anonymous_caller(monkeypatch, tmp_path):
    """End to end through the gate, which is where it matters: an API-key caller with no
    HF credential must not be authorized for the cache on a host whose saved token cannot
    be opened."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _no_host_credential(monkeypatch)
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (False, None))
    _counting_probe(monkeypatch, True, offline = True)

    assert public_cache_read_authorized(repo_id = ON_DISK) is False
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


def test_the_row_check_does_not_need_to_decrypt(monkeypatch):
    """`has_secret` is `get_secret(...) is not None`, so it cannot answer this question.
    The new reader has to query the table rather than route through the decrypting one."""
    from pathlib import Path as _Path

    from storage import credential_secrets

    source = _Path(credential_secrets.__file__).read_text(encoding = "utf-8")
    body = source.split("def secret_row_exists(", 1)[1].split("\ndef ", 1)[0]
    # Comments out: the docstring explains why it does NOT call the decrypting reader, and
    # a check that read them as code would fail on the sentence describing the fix.
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    code = code.split('"""')[0] + "".join(code.split('"""')[2:])
    assert "get_secret" not in code, "the row check goes back through the decrypting reader"
    assert "SELECT 1" in code


def test_the_saved_token_reader_reads_the_studio_credential_store(monkeypatch):
    """The reader itself, against the real store function.

    Three outcomes, the same as the ambient reader: a saved token, an answered "none", and
    an unanswered question. The store is the one Settings writes with `save_hf_token`.
    """
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


# --------------------------------------------------------------------------- probe classifier
@pytest.mark.parametrize(
    "status, error_code, expected",
    [
        (200, None, True),
        # Asked and answered: the credential was rejected or the repo withheld.
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
    """One blanket ``except: return False`` made a rate limit, an outage and a route that
    does not exist indistinguishable from a rejected token."""
    _probe_against(monkeypatch, lambda url: _response(status, error_code = error_code, url = url))

    assert hf_tokens._probe_repo_access(ON_DISK, OPERATOR_TOKEN, "model") is expected


def test_a_redirected_probe_answers_nothing_about_the_repo(monkeypatch):
    """A 200 from a login page, or a recorded hop, answered for somewhere else. That is not
    a denial, so it resolves locally like any other non-answer."""
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


# --------------------------------------------------------------------------- presence check
def test_the_presence_check_never_reaches_the_network(monkeypatch, tmp_path):
    """It is the fallback for "there is no network", so it has to be a disk walk."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)

    def _forbidden(*_a, **_k):
        raise AssertionError("the local presence check opened a connection")

    monkeypatch.setattr("huggingface_hub.utils.get_session", _forbidden)

    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is True
    assert hf_tokens._repo_present_on_disk(ABSENT, "model") is False


def test_a_dataset_counts_when_only_the_prepared_cache_holds_it(monkeypatch, tmp_path):
    """A preview is served by ``datasets`` out of its own prepared cache, a different tree
    from the hub snapshot, so presence has to look there too."""
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
    """The local fact is the ground the unaskable case stands on. If it cannot be
    established, there is no authorization to derive from it."""
    _cache_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        hf_cache_state,
        "repo_cache_has_usable_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("EIO")),
    )
    _counting_probe(monkeypatch, True, offline = True)

    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is False
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


# --------------------------------------------------------------------------- real consumers
def test_tier_detection_reads_a_cached_config_offline(monkeypatch, tmp_path):
    """``_load_config_json`` is how the tier/transformers-version decision is made. Denied,
    it returned None and the model was treated as unknown on a host that had it."""
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
    """``_offline_cache_read_refused`` gates the vision and audio probes on /loras and the
    model-config route. Offline it refused every explicit token, so a cached model reported
    no capabilities at all."""
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
    """Without a chat template the model cannot be chatted with, which is the "load it"
    half of the local question."""
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
    """BOUNDARY for the same site: an answered no still keeps the cached template back."""
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
    """Replace the loader, not the gate: these tests are about what reaches it."""
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
    """The sentinel's branch refused EVERY cache-only config read outright, so an API client
    with no Hub token lost its own downloaded public models on any offline host -- the one
    host where the cache is the only answer there is."""
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
    """BOUNDARY. Online, a repo the unauthenticated /auth-check cannot reach is private or
    gated, and the operator's cached copy of it stays out of reach."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    _counting_probe(monkeypatch, False, offline = False)
    _stub_autoconfig(monkeypatch)

    with pytest.raises(OSError):
        model_config_module.load_model_config(ON_DISK, token = False)

    assert cached_read_refused(False, repo_id = ON_DISK, is_cached = lambda: True) is True


def test_one_unreadable_cache_root_does_not_make_every_repo_present(monkeypatch, tmp_path):
    """``repo_cache_has_usable_snapshot`` answers True when a root could not be enumerated,
    which is right for its own caller ("do not delete this") and would here declare EVERY
    repo present on a host whose hub cache belongs to another user or sits on a stale mount.
    Not being able to look is not presence."""
    root = tmp_path / "hub"
    root.mkdir()

    def _roots(scan_errors = None):
        if scan_errors is not None:
            scan_errors.append(OSError("EACCES"))
        return []

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", _roots)
    # The helper on its own opens the gate; the wrapper must not.
    assert hf_cache_state.repo_cache_has_usable_snapshot("model", ON_DISK) is True
    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is False

    _counting_probe(monkeypatch, True, offline = True)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False


# ---------------------------------------------------------------------------------------
# The route-entry rule, which no change inside hf_tokens can reach
# ---------------------------------------------------------------------------------------


def _offline_route_guard(monkeypatch, tmp_path, repo_id: str, *, on_disk: bool):
    """``utils.utils.anonymous_and_offline`` under the conditions the routes call it in.

    On a host with no ambient credential: this guard is the ANONYMOUS precondition, and a
    credential-less caller only inherits the disk on a host whose operator has no token
    either. `test_the_route_entry_guard_keeps_refusing_on_a_host_that_holds_a_token` is the
    other half.
    """
    import utils.utils as utils_module

    _no_host_credential(monkeypatch)
    root = _cache_root(monkeypatch, tmp_path)
    if on_disk:
        _materialize_repo(root, repo_id)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: True)
    return utils_module.anonymous_and_offline(False, repo_id = repo_id)


def test_the_route_entry_guard_serves_a_repo_that_is_on_disk(monkeypatch, tmp_path):
    """The blanket version refused every cached model at the route entry, before any of the
    per-reader gates ran, so fixing the gates alone left the operator with the same 404.

    `GET /api/models/config` and the remote-code scan are the two routes; both reach this."""
    assert _offline_route_guard(monkeypatch, tmp_path, ON_DISK, on_disk = True) is False


def test_the_route_entry_guard_still_refuses_a_repo_that_is_not_on_disk(monkeypatch, tmp_path):
    """Nothing local to decide with, and saying yes would only authorize a network fetch."""
    assert _offline_route_guard(monkeypatch, tmp_path, ABSENT, on_disk = False) is True


def test_the_route_entry_guard_keeps_refusing_on_a_host_that_holds_a_token(monkeypatch, tmp_path):
    """The caller-isolation half at the route entry. On a host whose operator HAS an HF
    token the cache can hold private repos, so a credential-less caller is refused even
    though the repo is right there on the disk."""
    import utils.utils as utils_module

    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: True)
    assert utils_module.anonymous_and_offline(False, repo_id = ON_DISK) is True


def test_the_route_entry_guard_keeps_its_blanket_answer_without_a_repo_id(monkeypatch, tmp_path):
    """A caller that names no repo has nothing to resolve against, so the old answer stands."""
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    assert utils_module.anonymous_and_offline(False) is True
    assert utils_module.anonymous_and_offline(None) is False


def test_the_route_entry_guard_leaves_online_behaviour_exactly_as_it_was(monkeypatch, tmp_path):
    """This guard has always been the offline precondition. Online, the per-reader gates own
    the question, and this must not start putting a probe in front of every request."""
    import utils.utils as utils_module

    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: False)
    probes = _counting_probe(monkeypatch, False)
    assert utils_module.anonymous_and_offline(False, repo_id = ON_DISK) is False
    assert probes["n"] == 0


def test_two_different_host_credentials_authorize_neither_of_them(monkeypatch, tmp_path):
    """One host, two principals.

    An ambient token the operator uses from the CLI and a DIFFERENT token saved in Studio
    Settings by whoever is using the UI are not interchangeable: either could have filled
    the cache and nothing on disk records which. Matching one of them would hand its holder
    the repos the other downloaded, which on a managed install is another account's private
    model. Unknown authorizes nobody.
    """
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)
    _saved_ui_credential(monkeypatch)  # STUDIO_UI_TOKEN, alongside the ambient OPERATOR_TOKEN

    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is False
    assert cache_reads_authorized(STUDIO_UI_TOKEN, repo_id = ON_DISK) is False

    # The same credential in both stores is one credential, and is unaffected.
    _saved_ui_credential(monkeypatch, OPERATOR_TOKEN)
    assert cache_reads_authorized(OPERATOR_TOKEN, repo_id = ON_DISK) is True
