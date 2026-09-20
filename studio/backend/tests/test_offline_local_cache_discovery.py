# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub that cannot be ASKED is not a Hub that said no (#10264 put a live /auth-check in front
of every disk-cache read). Only the list of cache ROOTS is redirected, so presence walks are real.
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
from routes import inference as inference_routes
from utils import transformers_version
from utils.models import model_config as model_config_module

# Captured before the autouse fixture replaces it, so the test ABOUT the reader reaches it.
_REAL_AMBIENT_HF_TOKEN = hf_tokens._ambient_hf_token
_REAL_SAVED_STUDIO_HF_TOKEN = hf_tokens._saved_studio_hf_token

ON_DISK = "acme/downloaded-model"
ABSENT = "acme/never-downloaded"
OPERATOR_TOKEN = "hf_operator_ambient_token"
# Saved through Studio Settings: `get_token()` never returns it; the UI replays it per request.
STUDIO_UI_TOKEN = "hf_saved_in_studio_settings"
FOREIGN_TOKEN = "hf_some_other_callers_token"
HOST_CREDENTIAL = "hf_the_operators_own_credential"
ONE_OFF = "hf_a_one_off_someone_else_sent"
ROTATED_AWAY = "hf_the_credential_this_host_used_to_hold"
REVISION = "a" * 40

# Asks through a different entry point than a credentialed caller; `_reads`/`_refused` route it.
ANON = SimpleNamespace(caller = "no credential at all")


class _ProbeUnreachable(Exception):
    """Stands in for a refused connection: not an answer, so the fallback applies."""


@pytest.fixture(autouse = True)
def _isolate_repo_access_cache():
    reset_repo_access_cache()
    yield
    reset_repo_access_cache()


@pytest.fixture(autouse = True)
def _no_credential_ledger(monkeypatch):
    """An install whose ledger predates it, which is what most hosts are on upgrade."""
    monkeypatch.setattr(hf_tokens, "_host_credential_identities", lambda: {})


@pytest.fixture(autouse = True)
def _host_credential(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, OPERATOR_TOKEN))


@pytest.fixture(autouse = True)
def _saved_studio_credential(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_saved_studio_hf_token", lambda: (True, None))


_HF_STUBS = {
    "ambient": "_ambient_hf_token",
    "saved": "_saved_studio_hf_token",
    "credentials": "_host_hf_credentials",
    "identities": "_host_credential_identities",
    "recorded": "_recorded_request_token_repos",
    "present": "_repo_present_on_disk",
    "no_other_held": "_no_other_credential_ever_held",
    "filled_by_caller": "_caller_populated_the_cache",
    "fetched_with_request_token": "_repo_was_fetched_with_a_request_token",
    "unrecorded": "_unrecorded_fetches",
    "as_owner": "_as_owner",
    "denial_max": "_DENIAL_MEMORY_MAX",
    "noted_identities": "_noted_credential_identities",
}
_HF_PLAIN = ("unrecorded", "denial_max", "noted_identities")


def _hf_state(monkeypatch, **state) -> None:
    """A reader's argument is the VALUE it should answer with, or a callable installed as it is."""
    for key, value in state.items():
        plain = key in _HF_PLAIN or callable(value)
        stub = value if plain else (lambda *_a, _v = value, **_k: _v)
        monkeypatch.setattr(hf_tokens, _HF_STUBS[key], stub)


def _no_host_credential(monkeypatch):
    _hf_state(monkeypatch, ambient = (True, None))


def _reads(caller, repo, **kw) -> bool:
    """The same question through the two entry points a credentialed and an ANON caller use."""
    if caller is ANON:
        return public_cache_read_authorized(repo_id = repo, **kw)
    return cache_reads_authorized(caller, repo_id = repo, **kw)


def _refused(caller, repo, **kw) -> bool:
    """The mirror of `_reads` that the route guards actually call."""
    caller = False if caller is ANON else caller
    return cached_read_refused(caller, repo_id = repo, is_cached = lambda: True, **kw)


def _key(repo: str):
    return hf_tokens._request_token_repo_key(repo, "model")


def _filled_by(token, repo: str) -> bool:
    return hf_tokens._caller_populated_the_cache(token, repo_id = repo, repo_type = "model")


def _noted(token, repo: str) -> None:
    hf_tokens.note_repo_fetched_with_a_request_token(token, repo, "model")


def _fetched_with_a_token(repo: str):
    return hf_tokens._repo_was_fetched_with_a_request_token(repo, "model")


def _materialize_repo(root: Path, repo_id: str) -> Path:
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / REVISION
    snapshot.mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(REVISION, encoding = "utf-8")
    (snapshot / "config.json").write_text('{"model_type": "llama"}', encoding = "utf-8")
    (snapshot / "model.safetensors").write_text("x", encoding = "utf-8")
    return snapshot


def _cached_repo_dir(monkeypatch, tmp_path: Path, repo: str, **blobs: int) -> Path:
    """A real cache directory for `repo` holding `blobs` (name -> size); the reader walks it."""
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    (repo_dir / "blobs").mkdir(parents = True)
    for name, size in blobs.items():
        (repo_dir / "blobs" / name).write_bytes(b"x" * size)
    monkeypatch.setattr(
        hf_cache_state,
        "iter_repo_cache_dirs",
        lambda repo_type, repo_id, **kw: iter([repo_dir]),
    )
    return repo_dir


def _cache_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    root.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [root])
    return root


@pytest.fixture
def cache_root(monkeypatch, tmp_path) -> Path:
    """Only the ROOT LIST is redirected, so every presence walk under it runs for real."""
    return _cache_root(monkeypatch, tmp_path)


@pytest.fixture
def on_disk(cache_root) -> Path:
    """`cache_root` with ON_DISK fully materialised."""
    _materialize_repo(cache_root, ON_DISK)
    return cache_root


@pytest.fixture
def recorded_fetches(monkeypatch) -> list[tuple[str, str]]:
    """(repo, token) pairs: the token too, since the right repo against the wrong caller leaks."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        inference_routes,
        "_note_load_fetched_with_a_request_token",
        lambda ref, token: seen.append((ref, token)),
    )
    return seen


@pytest.fixture
def writes(monkeypatch) -> list:
    """Arguments captured whole: the right repo under the wrong key or arity is still bad."""
    seen: list = []
    _hf_state(
        monkeypatch,
        credentials = (True, ("hf_theoperators",)),
        as_owner = lambda call, *a, **k: seen.append(a),
    )
    return seen


class _LiveProvenance:
    """Read-only view of the provenance map as the writer actually left it."""

    def __getitem__(self, key):
        return (hf_tokens._recorded_request_token_repos() or {})[key]

    def get(self, key, default = None):
        return (hf_tokens._recorded_request_token_repos() or {}).get(key, default)

    def __contains__(self, key):
        return key in (hf_tokens._recorded_request_token_repos() or {})

    def __len__(self):
        return len(hf_tokens._recorded_request_token_repos() or {})


@pytest.fixture
def provenance(monkeypatch) -> _LiveProvenance:
    """The REAL setting store, under this test's own studio home (conftest isolates it).

    Not an in-memory stand-in any more: the first-writer rule and the collapse to an ambiguous
    `by` now happen inside the write's own transaction, because two requests holding different
    credentials for the same uncached repo otherwise both read "absent" and the second stores
    its own identity over the first. A fake `as_owner` that just assigns would be the test
    re-implementing the thing under test, and would pass however the race goes."""
    _hf_state(
        monkeypatch,
        credentials = (True, (HOST_CREDENTIAL,)),
        no_other_held = True,
        present = True,
    )
    return _LiveProvenance()


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
        def get(self, url, **_k):
            return response_factory(url)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)


def _response(status: int, error_code: str | None, url: str | None):
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


UNASKABLE = [
    ("env-offline", True, True, {}),
    ("caller-offline", True, False, {"offline": True}),
    ("unreachable", requests.exceptions.ConnectionError("refused"), False, {}),
    ("timeout", requests.exceptions.ReadTimeout("slow"), False, {}),
]


@pytest.mark.parametrize(
    ("verdict", "hub_offline", "call_kw"),
    [case[1:] for case in UNASKABLE],
    ids = [case[0] for case in UNASKABLE],
)
def test_an_unaskable_hub_resolves_against_the_disk_and_only_for_its_own_caller(
    monkeypatch, on_disk, verdict, hub_offline, call_kw
):
    probes = _counting_probe(monkeypatch, verdict, offline = hub_offline)

    assert _reads(OPERATOR_TOKEN, ON_DISK, **call_kw) is True
    if hub_offline or call_kw.get("offline"):
        assert probes["n"] == 0, "the offline answer went to the network"

    # BOUNDARY. On this disk is a fact about the OPERATOR, not about whoever is asking.
    assert _reads(FOREIGN_TOKEN, ON_DISK, **call_kw) is False
    assert _refused(FOREIGN_TOKEN, ON_DISK, **call_kw) is True

    # BOUNDARY. Not on this disk answers nothing, so there is nothing to fall back to.
    assert _reads(OPERATOR_TOKEN, ABSENT, **call_kw) is False
    assert _reads(ANON, ABSENT, **call_kw) is False


def test_an_interrupted_download_is_not_a_repo_on_disk(monkeypatch, tmp_path):
    """BOUNDARY. A cancelled download leaves an empty revision, which answers nothing."""
    root = _cache_root(monkeypatch, tmp_path)
    (root / f"models--{ON_DISK.replace('/', '--')}" / "snapshots" / REVISION).mkdir(parents = True)
    _counting_probe(monkeypatch, True, offline = True)

    assert _reads(OPERATOR_TOKEN, ON_DISK) is False


# The two credential stores, as the `(readable, token)` pair each reader returns.
HOST_STORES = {
    "ambient": ((True, OPERATOR_TOKEN), (True, None)),
    "saved-in-studio": ((True, None), (True, STUDIO_UI_TOKEN)),
    "both-the-same": ((True, OPERATOR_TOKEN), (True, OPERATOR_TOKEN)),
    "two-different": ((True, OPERATOR_TOKEN), (True, STUDIO_UI_TOKEN)),
    "none": ((True, None), (True, None)),
    "unreadable-saved": ((True, OPERATOR_TOKEN), (False, None)),
    "unreadable-saved-only": ((True, None), (False, None)),
}


@pytest.mark.parametrize(
    ("stores", "caller", "authorized"),
    [
        ("ambient", OPERATOR_TOKEN, True),
        ("ambient", FOREIGN_TOKEN, False),
        ("ambient", ANON, False),
        # The UI's ambient marker and a bare None are the host itself asking, never probe-gated.
        ("ambient", AmbientAuthorizedToken("hf_ui"), True),
        ("ambient", None, True),
        ("none", ANON, True),
        ("saved-in-studio", STUDIO_UI_TOKEN, True),
        ("saved-in-studio", FOREIGN_TOKEN, False),
        ("saved-in-studio", ANON, False),
        ("both-the-same", OPERATOR_TOKEN, True),
        ("both-the-same", FOREIGN_TOKEN, False),
        # BOUNDARY. Either of two DIFFERENT host credentials could have filled this cache.
        ("two-different", OPERATOR_TOKEN, False),
        ("two-different", STUDIO_UI_TOKEN, False),
        # BOUNDARY. "Cannot be read" is not "empty", so an unreadable store authorizes nobody.
        ("unreadable-saved", OPERATOR_TOKEN, False),
        ("unreadable-saved", ANON, False),
        ("unreadable-saved-only", ANON, False),
        ("unreadable-saved-only", OPERATOR_TOKEN, False),
    ],
)
def test_offline_only_the_caller_whose_credential_filled_the_cache_may_read_it(
    monkeypatch, on_disk, stores, caller, authorized
):
    ambient, saved = HOST_STORES[stores]
    _hf_state(monkeypatch, ambient = ambient, saved = saved)
    probes = _counting_probe(monkeypatch, True, offline = True)

    assert _reads(caller, ON_DISK) is authorized
    assert probes["n"] == 0, "the offline answer went to the network"
    if caller is ANON or isinstance(caller, str):
        assert _refused(caller, ON_DISK) is not authorized


def test_an_unreachable_probe_is_memoized_as_unknown_not_as_a_denial(monkeypatch, cache_root):
    probes = _counting_probe(
        monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False
    )

    assert _reads(OPERATOR_TOKEN, ON_DISK) is False
    assert [v for _expiry, v in hf_tokens._repo_access_cache.values()] == [None]

    # The repo arrives only now, so the second call proves the memo was reconsulted, not reprobed.
    _materialize_repo(cache_root, ON_DISK)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is True
    assert probes["n"] == 1, "the memo did not spare the second caller the dead round trip"


@pytest.mark.parametrize(
    ("overturn", "still_refused"),
    [
        ("nothing", True),
        # The verdict cache is what expires; the denial is what does not.
        ("an-outage", True),
        # An expiry made the boundary defeatable with the one input the caller controls.
        ("the-clock", True),
        ("the-hub-saying-yes", False),
    ],
)
def test_nothing_but_the_hub_overturns_a_hub_that_answered_no(
    monkeypatch, on_disk, overturn, still_refused
):
    _counting_probe(monkeypatch, False, offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False
    assert _refused(OPERATOR_TOKEN, ON_DISK) is True
    assert len(hf_tokens._denied_repo_access) == 1

    if overturn == "nothing":
        reset_repo_access_cache()
        assert not hf_tokens._denied_repo_access, "the reset left the denial behind"
        return

    if overturn == "the-hub-saying-yes":
        hf_tokens._repo_access_cache.clear()
        _counting_probe(monkeypatch, True, offline = False)
        assert _reads(OPERATOR_TOKEN, ON_DISK) is True

    if overturn == "the-clock":
        later = hf_tokens.time.monotonic() + 86_400.0
        monkeypatch.setattr(hf_tokens, "time", SimpleNamespace(monotonic = lambda: later))
    else:
        hf_tokens._repo_access_cache.clear()

    _counting_probe(monkeypatch, _ProbeUnreachable(), offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is not still_refused, "an outage overturned the Hub"
    if still_refused:
        assert _refused(OPERATOR_TOKEN, ON_DISK) is True


def test_a_repo_fetched_with_a_one_off_token_is_not_served_to_a_tokenless_caller(
    monkeypatch, on_disk
):
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    recorded: dict = {}
    _hf_state(
        monkeypatch,
        credentials = (True, ()),
        fetched_with_request_token = lambda repo, kind: recorded.get(_key(repo), False),
    )
    assert _reads(ANON, ON_DISK) is True

    recorded[_key(ON_DISK)] = True
    hf_tokens._repo_access_cache.clear()
    assert _reads(ANON, ON_DISK) is False, "a one-off token's repo went to a tokenless caller"
    _materialize_repo(on_disk, "acme/other")
    hf_tokens._repo_access_cache.clear()
    assert _reads(ANON, "acme/other") is True

    _hf_state(monkeypatch, fetched_with_request_token = None)
    hf_tokens._repo_access_cache.clear()
    assert _reads(ANON, "acme/other") is False


def test_the_provenance_key_is_case_folded_like_every_other_repo_lookup():
    assert _key("Org/Private") == hf_tokens._request_token_repo_key("org/private", "MODEL")
    assert hf_tokens._request_token_repo_key("org/private", "dataset") != _key("org/private")


def test_every_route_that_can_fetch_with_a_one_off_token_records_it():
    import inspect

    from hub.services import download_lifecycle
    from routes import video as video_routes

    text_load = inspect.getsource(inference_routes._load_model_impl)
    assert "_note_load_fetched_with_a_request_token(request.model_path" in text_load
    assert text_load.rindex("finally:") < text_load.index("_note_load_fetched_with_a_request_token")
    assert "_load_fetched_bytes(" in text_load

    # `begin_load` returns before the worker moves a byte, so a before/after comparison in a
    # media route would record nothing; they write at the launch, after the 400-able validation.
    for media_load in (
        inspect.getsource(inference_routes.load_diffusion_model_gated),
        inspect.getsource(video_routes.load_video_model_gated),
    ):
        assert "_repo_is_in_the_hub_cache(ref) is not True" in media_load
        assert "_note_load_fetched_with_a_request_token(_ref" in media_load
        assert (
            "_load_fetched_bytes" not in media_load
        ), "a media route cannot compare before and after: its fetch has not happened yet"
        tested_at = media_load.index("_repo_is_in_the_hub_cache(ref) is not True")
        validated_at = media_load.index("validate_load_request")
        written_at = media_load.index("_note_load_fetched_with_a_request_token(_ref")
        launched_at = media_load.index("status_dict = await asyncio.to_thread(")
        assert tested_at < validated_at, "the cache test has to precede anything that can fetch"
        assert validated_at < written_at, "a 400 from validation must leave no record behind"
        assert written_at < launched_at, "nothing may be fetched unrecorded"

    assert "_note_load_fetched_with_a_request_token" not in inspect.getsource(
        inference_routes.load_model_gated
    ), "the route records again, before the load is admitted"

    from hub.services.models import downloads

    assert "note_repo_fetched_with_a_request_token" not in inspect.getsource(downloads)

    # An unrecorded fetch reads back later as "nothing here needed a credential".
    from hub.services.datasets import formatting
    from picker import service as picker_module

    for source, call, fetches_at in (
        (inspect.getsource(download_lifecycle), "(hf_token, repo_id, repo_type)", "proc = spawn()"),
        (
            inspect.getsource(formatting.check_format_response),
            "(hf_token, request.dataset_name",
            "load_dataset(",
        ),
        (inspect.getsource(picker_module), "(hf_token, resolved", "path = hf_hub_download("),
    ):
        assert f"note_repo_fetched_with_a_request_token{call}" in source
        assert source.index("note_repo_fetched_with_a_request_token") < source.index(
            fetches_at
        ), "the fetch was recorded only after making it"

    config_read = inspect.getsource(model_config_module)
    assert "note_repo_fetched_with_a_request_token(token, model_name" in config_read
    # And only where the call can actually fetch: both of these resolve an already-cached file
    # without asking the Hub, so an unconditional record marks a repo the cache may have held
    # anonymously and withholds it from the tokenless offline caller.
    assert "not _config_json_already_cached(model_name, revision)" in config_read
    assert "if not _this_file_was_already_here(rel):" in inspect.getsource(picker_module)
    # Same rule on the preview: the record sits BELOW the prefer-local branches, which read
    # the cache or 404 without a round trip.
    preview = inspect.getsource(formatting.check_format_response)
    assert preview.index("_LOCAL_CACHE_MISS_ERROR_CODE") < preview.index(
        "note_repo_fetched_with_a_request_token"
    ), "a cache-only preview records a fetch it never made"


def test_every_credentialed_fetch_is_recorded_not_only_a_foreign_one(monkeypatch, writes):
    def written():
        return [call for call in writes if len(call) == 3]

    _noted(False, "acme/public")
    assert written() == [], "an anonymous fetch says nothing about anybody"

    _noted("hf_theoperators", "acme/saved")
    _noted(None, "acme/ambient")
    assert [call[1] for call in written()] == [_key("acme/saved"), _key("acme/ambient")]

    _noted("hf_someoneelses", "acme/private")
    assert written()[-1][1] == _key("acme/private")

    writes.clear()
    _hf_state(monkeypatch, credentials = (True, ()))
    _noted(None, "acme/public2")
    assert written() == []


def test_a_deleted_credential_does_not_reopen_the_cache(monkeypatch):
    _hf_state(
        monkeypatch,
        ambient = (True, None),
        saved = (True, None),
        present = True,
        recorded = {_key("acme/private"): {"at": 0}},
    )
    hf_tokens.reset_repo_access_cache()

    assert not _reads(ANON, "acme/private", offline = True)
    assert _reads(ANON, "acme/public", offline = True)


def test_a_repo_that_was_never_refused_still_resolves_against_the_disk(monkeypatch, on_disk):
    _materialize_repo(on_disk, "acme/other")
    _counting_probe(monkeypatch, False, offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert _reads(OPERATOR_TOKEN, "acme/other") is True
    assert _reads(FOREIGN_TOKEN, ON_DISK) is False, "a second principal inherited that cache"


def test_a_refusal_the_table_had_to_forget_closes_the_outage_path(monkeypatch, on_disk):
    _hf_state(monkeypatch, denial_max = 4)
    _counting_probe(monkeypatch, False, offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False
    assert not hf_tokens._denial_memory_lost_an_entry()

    # The filler must EARN a slot: on this disk and refused to a credential this host holds.
    for index in range(8):
        filler = f"acme/filler-{index}"
        _materialize_repo(on_disk, filler)
        hf_tokens._repo_access_cache.clear()
        assert _reads(OPERATOR_TOKEN, filler) is False
    assert hf_tokens._denial_memory_lost_an_entry(), "the table dropped nothing"

    hf_tokens._repo_access_cache.clear()
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False, "a forgotten refusal read as never refused"


def test_forgetting_one_refusal_is_not_a_reason_to_refuse_an_answered_hub(monkeypatch, on_disk):
    _hf_state(monkeypatch, denial_max = 1)
    hf_tokens._remember_denial(("acme/one", "model", "hash"), 1.0)
    hf_tokens._remember_denial(("acme/two", "model", "hash"), 2.0)
    assert hf_tokens._denial_memory_lost_an_entry()

    _counting_probe(monkeypatch, True, offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is True

    reset_repo_access_cache()
    assert (
        not hf_tokens._denial_memory_lost_an_entry()
    ), "the reset left the fail-closed marker set for every following test"


def test_a_refusal_still_being_re_asked_is_not_the_one_dropped(monkeypatch):
    _hf_state(monkeypatch, denial_max = 3)
    keys = [(f"acme/{index}", "model", "hash") for index in range(3)]
    for at, key in enumerate(keys):
        hf_tokens._remember_denial(key, float(at))
    hf_tokens._remember_denial(keys[0], 9.0)
    hf_tokens._remember_denial(("acme/new", "model", "hash"), 10.0)
    assert hf_tokens._denial_is_remembered(keys[0])
    assert not hf_tokens._denial_is_remembered(keys[1])


def test_the_ownership_check_reads_the_hosts_token_every_time(monkeypatch, on_disk):
    _counting_probe(monkeypatch, True, offline = True)

    assert _reads(OPERATOR_TOKEN, ON_DISK) is True
    _hf_state(monkeypatch, ambient = (True, "hf_rotated"))
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False
    assert _reads("hf_rotated", ON_DISK) is True


@pytest.mark.parametrize(
    ("reader", "target", "raw", "stripped", "failure"),
    [
        (
            "ambient",
            "huggingface_hub.get_token",
            "  hf_from_hub  ",
            "hf_from_hub",
            OSError("token file unreadable"),
        ),
        (
            "saved",
            "storage.credential_secrets.get_hf_token",
            "  hf_from_store  ",
            "hf_from_store",
            RuntimeError("credential database is locked"),
        ),
    ],
    ids = ("the-hosts-own-token", "the-studio-credential-store"),
)
def test_a_credential_reader_separates_an_absent_credential_from_an_unreadable_one(
    monkeypatch, reader, target, raw, stripped, failure
):
    read = _REAL_AMBIENT_HF_TOKEN if reader == "ambient" else _REAL_SAVED_STUDIO_HF_TOKEN
    if reader == "ambient":
        # The env fallback must not answer in place of the Hub's own store.
        for key in hf_tokens._HF_TOKEN_ENV_KEYS:
            monkeypatch.delenv(key, raising = False)

    monkeypatch.setattr(target, lambda: raw)
    assert read() == (True, stripped)

    monkeypatch.setattr(target, lambda: None)
    assert read() == (True, None)

    # Asked and NOT answered is the third outcome; collapsing it into "no credential" fails open.
    def _raises():
        raise failure

    monkeypatch.setattr(target, _raises)
    assert read() == (False, None)
    assert hf_tokens._caller_populated_the_cache(None) is False
    assert hf_tokens._caller_populated_the_cache("hf_anything") is False


@pytest.mark.parametrize(
    "alias",
    ["HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"],
)
def test_a_credential_held_only_under_a_legacy_alias_still_counts(monkeypatch, alias):
    """`get_token` reads HF_TOKEN, HUGGING_FACE_HUB_TOKEN, the OIDC exchange and the token file,
    and nothing else. The aliases this module elsewhere treats as credentials (it strips all of
    them from a download subprocess) therefore came back as "this host holds nothing", which
    authorizes a tokenless caller against a cache that credential may have filled."""
    for key in hf_tokens._HF_TOKEN_ENV_KEYS:
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)

    assert _REAL_AMBIENT_HF_TOKEN() == (True, None)

    monkeypatch.setenv(alias, f"  {HOST_CREDENTIAL}  ")
    assert _REAL_AMBIENT_HF_TOKEN() == (True, HOST_CREDENTIAL)

    # And when the reader itself could not answer, the alias is still knowledge.
    def _raises():
        raise OSError("token file unreadable")

    monkeypatch.setattr("huggingface_hub.get_token", _raises)
    assert _REAL_AMBIENT_HF_TOKEN() == (True, HOST_CREDENTIAL)


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


def test_the_row_check_does_not_need_to_decrypt(monkeypatch):
    from pathlib import Path as _Path

    from storage import credential_secrets

    source = _Path(credential_secrets.__file__).read_text(encoding = "utf-8")
    body = source.split("def secret_row_exists(", 1)[1].split("\ndef ", 1)[0]
    # Strip comments and the docstring, whose prose would otherwise be read as code.
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    code = code.split('"""')[0] + "".join(code.split('"""')[2:])
    assert "get_secret" not in code, "the row check goes back through the decrypting reader"
    assert "SELECT 1" in code


@pytest.mark.parametrize(
    ("status", "error_code", "landed_on", "expected"),
    [
        (200, None, None, True),
        (401, None, None, False),
        (403, None, None, False),
        (404, "RepoNotFound", None, False),
        (410, None, None, False),
        # Could not be asked: none of these is a statement about this caller.
        (404, None, None, None),  # an HF_ENDPOINT mirror with no /auth-check route
        (405, None, None, None),  # a mirror that rejects the method
        (429, None, None, None),  # rate limited
        (500, None, None, None),  # outage
        (502, None, None, None),  # gateway in front of the Hub
        (200, None, "https://huggingface.co/login", None),  # a redirect answers about elsewhere
    ],
)
def test_the_probe_separates_an_answer_from_a_failure_to_answer(
    monkeypatch, status, error_code, landed_on, expected
):
    _probe_against(monkeypatch, lambda url: _response(status, error_code, landed_on or url))

    assert hf_tokens._probe_repo_access(ON_DISK, OPERATOR_TOKEN, "model") is expected


def test_a_probe_timeout_is_not_an_answer(monkeypatch):
    def _stall(*_a, **_k):
        raise requests.exceptions.Timeout("stalled")

    class _Session:
        get = staticmethod(_stall)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())

    with pytest.raises(hf_tokens._ProbeTimedOut):
        hf_tokens._probe_repo_access(ON_DISK, OPERATOR_TOKEN, "model")


def test_the_presence_check_never_reaches_the_network(monkeypatch, on_disk):
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
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False


def test_one_unreadable_cache_root_does_not_make_every_repo_present(monkeypatch, tmp_path):
    """``repo_cache_has_usable_snapshot`` answers True for an unenumerable root."""

    def _roots(scan_errors = None):
        if scan_errors is not None:
            scan_errors.append(OSError("EACCES"))
        return []

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", _roots)
    assert hf_cache_state.repo_cache_has_usable_snapshot("model", ON_DISK) is True
    assert hf_tokens._repo_present_on_disk(ON_DISK, "model") is False

    _counting_probe(monkeypatch, True, offline = True)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is False


def test_tier_detection_reads_a_cached_config_offline(monkeypatch, on_disk):
    monkeypatch.setattr(
        transformers_version,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = on_disk),
    )
    monkeypatch.setattr(transformers_version, "_env_offline", lambda: True)
    _counting_probe(monkeypatch, True, offline = True)
    transformers_version._config_json_cache.clear()

    assert transformers_version._load_config_json(ON_DISK, OPERATOR_TOKEN) == {
        "model_type": "llama"
    }


def test_the_capability_probes_cache_only_gate_allows_a_repo_on_disk(monkeypatch, on_disk):
    _counting_probe(monkeypatch, True, offline = True)
    refused = model_config_module._offline_cache_read_refused

    assert refused(OPERATOR_TOKEN, ON_DISK, ON_DISK, True) is False
    assert refused(OPERATOR_TOKEN, ABSENT, ABSENT, True) is True


@pytest.mark.parametrize(
    ("hub_says", "offline", "template"),
    [(True, True, "{{ messages }}"), (False, False, None)],
    ids = ("serves-what-it-cannot-ask-about", "withholds-what-the-hub-refused"),
)
def test_the_picker_serves_a_cached_chat_template_offline(
    monkeypatch, cache_root, hub_says, offline, template
):
    snapshot = _materialize_repo(cache_root, ON_DISK)
    (snapshot / "chat_template.jinja").write_text("{{ messages }}", encoding = "utf-8")
    monkeypatch.setattr(picker_service, "hf_env_offline", lambda: True)
    monkeypatch.setattr(picker_service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        picker_service, "iter_snapshots_preferring_whole", lambda *_a, **_k: iter([snapshot])
    )
    _counting_probe(monkeypatch, hub_says, offline = offline)

    assert picker_service.read_default_chat_template(ON_DISK, OPERATOR_TOKEN) == template


def _stub_autoconfig(monkeypatch) -> dict:
    """Imported inside the function under test, so the stand-in goes on the package."""
    import transformers

    seen = {"n": 0}

    class _AutoConfig:
        @staticmethod
        def from_pretrained(name, *_a, **kwargs):
            seen["n"] += 1
            seen["name"] = name
            seen["local_files_only"] = kwargs.get("local_files_only")
            return SimpleNamespace(model_type = "llama")

    monkeypatch.setattr(transformers, "AutoConfig", _AutoConfig)
    return seen


def test_an_anonymous_cache_only_config_read_serves_a_repo_on_disk(monkeypatch, on_disk):
    """The sentinel's branch refused EVERY cache-only read, on the one host where the cache is
    the only answer there is."""
    _no_host_credential(monkeypatch)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    probes = _counting_probe(monkeypatch, True, offline = True)
    seen = _stub_autoconfig(monkeypatch)

    config = model_config_module.load_model_config(ON_DISK, token = False, local_files_only = True)

    assert config is not None
    assert seen["name"] == ON_DISK and seen["local_files_only"] is True
    assert probes["n"] == 0, "a cache-only read went to the network"


@pytest.mark.parametrize(
    "already_cached, expected",
    [(True, []), (False, [("hf_a_one_off", ON_DISK, "model")])],
    ids = ["a cache hit records nothing", "a real fetch is recorded"],
)
def test_a_credentialed_config_read_records_only_what_it_could_fetch(
    monkeypatch, already_cached, expected
):
    """AutoConfig resolves an already-cached config without asking the Hub. Recording that marks a
    repo the cache may have held anonymously, which then withholds it from the tokenless offline
    caller this path exists for."""
    recorded: list = []
    monkeypatch.setattr(
        model_config_module,
        "note_repo_fetched_with_a_request_token",
        lambda token, repo, kind: recorded.append((token, repo, kind)),
    )
    monkeypatch.setattr(
        model_config_module, "_config_json_already_cached", lambda *_a, **_k: already_cached
    )
    # The Hub says this caller may reach the repo, so the gate above is not what is under test.
    _counting_probe(monkeypatch, True)
    _stub_autoconfig(monkeypatch)

    model_config_module.load_model_config(ON_DISK, token = "hf_a_one_off")

    assert recorded == expected


@pytest.mark.parametrize(
    "prefer_local_cache, expected",
    [(True, []), (False, [("hf_a_one_off", "acme/ds", "dataset")])],
    ids = ["a cache-only preview records nothing", "a preview that may fetch is recorded"],
)
def test_a_dataset_preview_records_only_where_it_could_fetch(
    monkeypatch, prefer_local_cache, expected
):
    """`prefer_local_cache` reads the datasets cache or 404s, never the network. Recording it
    relabels a dataset the cache may have held anonymously, and the tokenless offline caller
    that path exists for is then refused."""
    import contextlib

    from hub.services.datasets import formatting
    from hub.schemas.datasets import CheckFormatRequest

    recorded: list = []
    monkeypatch.setattr(
        formatting,
        "note_repo_fetched_with_a_request_token",
        lambda token, repo, kind: recorded.append((token, repo, kind)),
    )
    monkeypatch.setattr(formatting, "refuse_unauthorized_dataset_preview", lambda *_a, **_k: None)

    class _NoHub:
        def list_repo_files(self, *_a, **_k):
            raise RuntimeError("no network in this test")

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *_a, **_k: _NoHub())
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no network in this test")),
    )

    # Whatever the branch does after recording is another test's subject: the network is
    # stubbed off, so both legs end in an error and only the ledger is read here.
    with contextlib.suppress(Exception):
        formatting.check_format_response(
            CheckFormatRequest(dataset_name = "acme/ds", prefer_local_cache = prefer_local_cache),
            "hf_a_one_off",
        )

    assert recorded == expected


def _offline_route_guard(monkeypatch, tmp_path, repo_id, present, host_token):
    import utils.utils as utils_module

    if host_token is None:
        _no_host_credential(monkeypatch)
    root = _cache_root(monkeypatch, tmp_path)
    if present:
        _materialize_repo(root, repo_id)
    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: True)
    return utils_module.anonymous_and_offline(False, repo_id = repo_id)


@pytest.mark.parametrize(
    ("repo_id", "present", "host_token", "refused"),
    [
        (ON_DISK, True, None, False),
        (ABSENT, False, None, True),
        # BOUNDARY. A host that HOLDS a credential could have filled this cache with it.
        (ON_DISK, True, OPERATOR_TOKEN, True),
    ],
    ids = ("on-disk", "not-on-disk", "host-holds-a-token"),
)
def test_the_route_entry_guard_serves_only_what_the_anonymous_caller_may_have(
    monkeypatch, tmp_path, repo_id, present, host_token, refused
):
    guard = _offline_route_guard(monkeypatch, tmp_path, repo_id, present, host_token)
    assert guard is refused


def test_the_route_entry_guard_keeps_its_blanket_answer_without_a_repo_id(monkeypatch):
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: True)
    assert utils_module.anonymous_and_offline(False) is True
    assert utils_module.anonymous_and_offline(None) is False


def test_the_route_entry_guard_leaves_online_behaviour_exactly_as_it_was(monkeypatch, on_disk):
    import utils.utils as utils_module

    monkeypatch.setattr(utils_module, "hf_env_offline", lambda: False)
    probes = _counting_probe(monkeypatch, False)
    assert utils_module.anonymous_and_offline(False, repo_id = ON_DISK) is False
    assert probes["n"] == 0


def test_a_repo_already_recorded_keeps_the_record_the_first_writer_left(provenance):
    """Asserted on the STORED value rather than on whether a write was attempted: the first-writer
    rule lives inside the transaction now, so a second call reaching the store is expected and it
    is what the store does with it that matters."""
    _noted(ONE_OFF, "acme/private")
    first = provenance[_key("acme/private")]
    assert first["by"] == hf_tokens._credential_identity(ONE_OFF)

    _noted(ONE_OFF, "acme/private")
    assert provenance[_key("acme/private")] == first, "the same credential rewrote its own record"

    # A DIFFERENT credential does not take it over; it makes the record say "cannot attribute".
    _noted("hf_someoneelses", "acme/private")
    again = provenance[_key("acme/private")]
    assert again["by"] is None
    assert again["at"] == first["at"], "the collapse rewrote the timestamp of the first fetch"

    _noted(ONE_OFF, "acme/other")
    assert provenance[_key("acme/other")]["by"] == hf_tokens._credential_identity(ONE_OFF)


def test_the_provenance_map_is_bounded_and_says_so_when_it_is_full(monkeypatch, writes):
    full = {
        f"model:acme/repo-{index}": {"at": 1.0}
        for index in range(hf_tokens._REQUEST_TOKEN_REPOS_MAX)
    }
    _hf_state(monkeypatch, recorded = full)

    _noted("hf_someoneelses", "acme/new")
    assert writes == [], "the map kept growing on a caller-supplied repo id"
    assert _fetched_with_a_token("acme/new") is None
    assert _fetched_with_a_token("acme/repo-0") is True
    _hf_state(monkeypatch, credentials = (True, ()))
    assert _filled_by(None, "acme/new") is False


def test_a_map_below_the_cap_still_answers_no_for_a_repo_it_does_not_hold(monkeypatch):
    _hf_state(monkeypatch, recorded = {})
    assert _fetched_with_a_token("acme/other") is False


def test_a_credential_does_not_inherit_a_repo_another_credential_fetched(provenance):
    """A one-off X-Unsloth-HF-Token reaches private repos the host credential cannot, and its
    bytes land in the same cache, so holding the host credential now is not having filled it."""
    _noted(ONE_OFF, "acme/theirs")
    _noted(None, "acme/ours")

    assert _filled_by(HOST_CREDENTIAL, "acme/theirs") is False, "the host inherited a foreign repo"
    assert hf_tokens._resolve_unaskable("acme/theirs", "model", token = HOST_CREDENTIAL) is False

    # A cache older than the record is the tokenless offline install on first upgrade.
    assert _filled_by(HOST_CREDENTIAL, "acme/ours") is True
    assert _filled_by(HOST_CREDENTIAL, "acme/never-seen") is True
    assert _filled_by(ONE_OFF, "acme/theirs") is False
    assert provenance[_key("acme/theirs")]["by"] == hf_tokens._credential_identity(ONE_OFF)


def test_a_repo_two_different_credentials_fetched_belongs_to_neither(provenance):
    _noted(None, "acme/shared")
    _noted(ONE_OFF, "acme/shared")

    assert provenance[_key("acme/shared")]["by"] is None
    assert (
        _filled_by(HOST_CREDENTIAL, "acme/shared") is False
    ), "a repo more than one credential fetched was claimed by one of them"
    assert _fetched_with_a_token("acme/shared") is True


def test_a_provenance_write_that_failed_is_not_read_as_nothing_to_record(monkeypatch):
    """A failed write leaves the same absence a repo nobody ever fetched leaves, and only one of
    the two authorizes."""

    def _raises(*_a, **_k):
        raise RuntimeError("database is locked")

    _hf_state(
        monkeypatch,
        unrecorded = set(),
        credentials = (True, ()),
        no_other_held = True,
        recorded = {},
        as_owner = _raises,
    )
    _noted("hf_a_one_off", "acme/private")

    assert _fetched_with_a_token("acme/private") is None
    assert (
        _filled_by(None, "acme/private") is False
    ), "a repo whose provenance write failed was read as one nothing ever fetched"
    # Only that repo: a locked database must not cost the whole host its own cache.
    assert _filled_by(None, "acme/other") is True


def test_a_denial_is_remembered_when_the_facts_that_could_overturn_it_cannot_be_read(monkeypatch):
    """The rule below the memory answers a plain no for a fact it could not establish, because
    its callees swallow their own failures; a denial dropped in that window let the next outage
    authorize the caller the Hub had refused."""
    overturnable = hf_tokens._denial_can_be_overturned
    _hf_state(monkeypatch, present = True, unrecorded = set())

    _hf_state(monkeypatch, credentials = (False, ()))
    assert overturnable("acme/private", "model", HOST_CREDENTIAL) is True

    _hf_state(monkeypatch, credentials = (True, (HOST_CREDENTIAL,)), recorded = None)
    assert overturnable("acme/private", "model", HOST_CREDENTIAL) is True

    # BOUNDARY. The fallback could never authorize this caller, so no slot is spent.
    _hf_state(monkeypatch, recorded = {}, filled_by_caller = False)
    assert overturnable("acme/private", "model", HOST_CREDENTIAL) is False


def test_a_ledger_write_that_failed_is_retried_rather_than_remembered(monkeypatch):
    """A shorter ledger reads as "this host never held anything else", which authorizes more."""
    identity = hf_tokens._credential_identity(HOST_CREDENTIAL)
    ledger: dict = {}
    attempts = {"n": 0}

    def _write(call, _key_unused, entry, value):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("database is locked")
        ledger[entry] = value

    _hf_state(
        monkeypatch,
        noted_identities = set(),
        identities = lambda: dict(ledger),
        as_owner = _write,
    )

    hf_tokens._note_host_credential_identities([HOST_CREDENTIAL])
    assert ledger == {}, "the write did not fail"
    hf_tokens._note_host_credential_identities([HOST_CREDENTIAL])
    assert attempts["n"] == 2, "a failed ledger write was never retried"
    assert identity in ledger
    hf_tokens._note_host_credential_identities([HOST_CREDENTIAL])
    assert attempts["n"] == 2, "a ledger entry already present was rewritten"


def test_an_unreadable_ledger_is_not_remembered_as_written(monkeypatch):
    _hf_state(
        monkeypatch,
        noted_identities = set(),
        identities = None,
        as_owner = lambda *a, **k: pytest.fail("wrote into a ledger it cannot read"),
    )
    hf_tokens._note_host_credential_identities([HOST_CREDENTIAL])
    assert hf_tokens._noted_credential_identities == set()


def test_the_ledger_records_an_identity_and_never_the_credential(monkeypatch):
    written: dict = {}
    _hf_state(
        monkeypatch,
        identities = lambda: dict(written),
        as_owner = lambda call, _key_unused, entry, value: written.setdefault(entry, value),
    )
    hf_tokens.reset_repo_access_cache()

    hf_tokens._note_host_credential_identities((OPERATOR_TOKEN,))
    assert list(written) == [hf_tokens._credential_identity(OPERATOR_TOKEN)]
    assert OPERATOR_TOKEN not in repr(written)
    assert hf_tokens._credential_identity(OPERATOR_TOKEN) != hf_tokens._credential_identity(
        ROTATED_AWAY
    )

    # Written once: this runs on the authorization path, not on a settings save.
    written.clear()
    hf_tokens._note_host_credential_identities((OPERATOR_TOKEN,))
    assert written == {}, "the ledger was rewritten on every check"


def _ledger(monkeypatch, *tokens: str) -> None:
    _hf_state(
        monkeypatch,
        identities = {hf_tokens._credential_identity(token): {"at": 1.0} for token in tokens},
    )


@pytest.mark.parametrize(
    ("held", "caller", "authorized"),
    [
        # After a rotation the new credential must not be handed the old one's downloads.
        ((ROTATED_AWAY, OPERATOR_TOKEN), OPERATOR_TOKEN, False),
        # BOUNDARY: a host that never rotated is unchanged, which is nearly every host.
        ((OPERATOR_TOKEN,), OPERATOR_TOKEN, True),
        # Deleting it leaves its private repos on disk, so holding none stops meaning none
        # was needed.
        ((ROTATED_AWAY,), ANON, False),
        ((), ANON, True),
        # Cannot be established is not "empty".
        (None, OPERATOR_TOKEN, False),
        (None, ANON, False),
    ],
)
def test_an_outage_serves_the_cache_only_to_a_credential_that_really_filled_it(
    monkeypatch, on_disk, held, caller, authorized
):
    _counting_probe(monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False)
    if caller is ANON:
        _hf_state(monkeypatch, ambient = (True, None), recorded = {})
    if held is None:
        _hf_state(monkeypatch, identities = None)
    else:
        _ledger(monkeypatch, *held)

    assert _reads(caller, ON_DISK) is authorized


def test_a_non_ascii_credential_is_compared_not_crashed_on(monkeypatch, on_disk):
    """`hmac.compare_digest` refuses a str with any non-ASCII character, and both operands are
    text somebody else chose, so one high byte turned authorization into a 500."""
    _counting_probe(monkeypatch, _ProbeUnreachable(), offline = False)

    # Starlette latin-1 decodes the caller's header byte 0xE9 to U+00E9.
    assert _reads("hf_é_not_the_hosts", ON_DISK) is False

    _hf_state(monkeypatch, ambient = (True, "hf_opérateur"))
    assert _reads("hf_opérateur", ON_DISK) is True


@pytest.mark.parametrize(
    ("base", "in_cache", "already_cached", "recorded"),
    [
        ("acme/private-base", True, None, [("acme/private-base", "hf_someoneelses")]),
        (None, True, None, []),
        ("acme/never-pulled", False, None, []),
        (OSError(13, "denied"), True, None, []),
        ("acme/public-base", True, "acme/public-base", []),
    ],
    ids = ("pulled", "unresolved", "not-in-the-cache", "cache-unreadable", "already-cached"),
)
def test_only_a_lora_base_this_load_really_pulled_is_recorded(
    monkeypatch, recorded_fetches, base, in_cache, already_cached, recorded
):
    def _resolve(_repo):
        if isinstance(base, BaseException):
            raise base
        return base

    monkeypatch.setattr(transformers_version, "_adapter_base_from_hf_cache", _resolve)
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: in_cache)

    inference_routes._note_lora_base_fetched_with_a_request_token(
        "acme/adapter", "hf_someoneelses", already_cached = already_cached
    )

    assert recorded_fetches == recorded


def test_the_base_a_load_started_with_is_the_one_read_before_it(monkeypatch):
    monkeypatch.setattr(
        transformers_version, "_adapter_base_from_hf_cache", lambda repo: "acme/public-base"
    )
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)

    assert (
        inference_routes._lora_base_already_in_the_hub_cache("acme/adapter") == "acme/public-base"
    )


def test_the_load_reads_the_base_after_the_fetch_not_before_it():
    import inspect

    impl = inspect.getsource(inference_routes._load_model_impl)
    assert "_note_lora_base_fetched_with_a_request_token(" in impl
    assert impl.rindex("finally:") < impl.index("_note_lora_base_fetched_with_a_request_token(")
    assert "already_cached = _lora_base_before_this_load" in impl
    assert impl.index("_lora_base_before_this_load = ") < impl.rindex("finally:")


def test_only_a_repo_this_load_actually_pulled_is_recorded(recorded_fetches):
    def _record(before, after):
        recorded_fetches.clear()
        if before is False and after:
            inference_routes._note_load_fetched_with_a_request_token("acme/private", "hf_x")
        return [ref for ref, _token in recorded_fetches]

    assert _record(False, True) == ["acme/private"], "a repo this load pulled was not recorded"
    assert _record(True, True) == [], "a repo that was already cached was recorded as fetched"
    assert _record(False, False) == [], "a load that fetched nothing recorded a fetch"
    assert _record(None, True) == [], "an unanswerable reading recorded a fetch"


@pytest.mark.parametrize("route", ("image", "video"))
def test_a_media_load_refused_by_validation_records_no_fetch(monkeypatch, recorded_fetches, route):
    """The record is permanent, so recording a fetch that never happened would withhold the repo
    from a later tokenless offline read of a copy downloaded anonymously."""
    import asyncio

    from fastapi import HTTPException

    from hub.services.models import account_access

    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    monkeypatch.setattr(account_access, "require_idle_other_accounts", lambda: None)
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: False)

    def _refuse(*_a, **_k):
        raise ValueError("unsupported model_kind")

    if route == "image":
        from core.inference import diffusion as _diffusion
        from models.inference import DiffusionLoadRequest

        monkeypatch.setattr(_diffusion, "resolve_model_kind", _refuse)
        request = DiffusionLoadRequest(model_path = "acme/private-image")
        load = inference_routes.load_diffusion_model_gated
    else:
        from core.inference import video as _video
        from models.inference import VideoLoadRequest
        from routes import video as video_routes

        monkeypatch.setattr(_video, "resolve_video_model_kind", _refuse)
        request = VideoLoadRequest(model_path = "acme/private-video")
        load = video_routes.load_video_model_gated

    request.hf_token = "hf_a_one_off_request_token"
    with pytest.raises(HTTPException) as refusal:
        asyncio.run(load(request, "owner", user_initiated = True))
    assert refusal.value.status_code == 400
    assert recorded_fetches == [], "a load refused before any worker started recorded a fetch"


def test_a_media_load_records_at_entry_because_its_fetch_is_on_a_worker_thread():
    import inspect

    from core.inference import diffusion, video as video_core
    for begin in (diffusion.DiffusionBackend.begin_load, video_core.VideoBackend.begin_load):
        doc = inspect.getdoc(begin) or ""
        assert "Returns at once" in doc or "daemon thread" in doc, (
            f"{begin.__qualname__} no longer hands off; the media routes could compare "
            "before and after like the text load does"
        )


def test_the_presence_probe_answers_none_for_anything_that_is_not_a_repo(monkeypatch):
    for local in ("/srv/models/model.gguf", "./model.gguf", "~/model.gguf", "C:/models", ""):
        assert inference_routes._repo_is_in_the_hub_cache(local) is None, local

    _hf_state(monkeypatch, present = True)
    assert inference_routes._repo_is_in_the_hub_cache("acme/model") is True

    def _raises(_repo, _kind):
        raise OSError(13, "denied")

    _hf_state(monkeypatch, present = _raises)
    assert inference_routes._repo_is_in_the_hub_cache("acme/model") is None


def test_only_real_growth_in_the_hub_cache_counts_as_a_fetch(monkeypatch, tmp_path):
    blobs = _cached_repo_dir(monkeypatch, tmp_path, "acme/private", a = 10) / "blobs"
    monkeypatch.setattr(inference_routes, "_repo_is_in_the_hub_cache", lambda ref: True)
    fetched = inference_routes._load_fetched_bytes

    before = inference_routes._hub_cache_footprint("acme/private")
    assert before == (1, 10)
    assert not fetched("acme/private", True, before), "a pure cache hit was recorded as a fetch"

    (blobs / "b").write_bytes(b"y" * 4096)
    assert fetched("acme/private", True, before), "a refetch of an existing repo was lost"
    assert not fetched("acme/private", True, None), "an unreadable before-reading guessed"
    assert fetched("acme/private", False, None)

    # A prune is not a fetch, and neither is an after-reading that could not be taken.
    after_the_pull = inference_routes._hub_cache_footprint("acme/private")
    (blobs / "b").unlink()
    assert not fetched("acme/private", True, after_the_pull)
    monkeypatch.setattr(inference_routes, "_hub_cache_footprint", lambda ref: None)
    assert not fetched("acme/private", True, before), "an unreadable after-reading counted"

    # Either component: a new blob moves the count, an appended partial moves only the bytes.
    monkeypatch.setattr(inference_routes, "_hub_cache_footprint", lambda ref: (2, 10))
    assert fetched("acme/private", True, before)
    monkeypatch.setattr(inference_routes, "_hub_cache_footprint", lambda ref: (1, 9000))
    assert fetched("acme/private", True, before)


def test_refs_and_no_exist_markers_are_not_read_as_a_fetch(monkeypatch, tmp_path):
    repo_dir = _cached_repo_dir(monkeypatch, tmp_path, "acme/public", a = 32)
    before = inference_routes._hub_cache_footprint("acme/public")

    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("deadbeef")
    (repo_dir / ".no_exist" / "deadbeef").mkdir(parents = True)
    (repo_dir / ".no_exist" / "deadbeef" / "adapter_config.json").write_text("")
    assert inference_routes._hub_cache_footprint("acme/public") == before


# The slots are keyed partly on caller-supplied input, so what may occupy one decides
# whether a caller can exhaust them.


@pytest.mark.parametrize(
    ("caller_for", "repo_for"),
    [
        (lambda i: OPERATOR_TOKEN, lambda i: f"attacker/absent-{i}"),
        (lambda i: f"hf_stranger_{i}", lambda i: ON_DISK),
    ],
    ids = ("a-repo-not-on-disk", "a-credential-this-host-never-held"),
)
def test_a_denial_the_disk_could_never_overturn_does_not_occupy_a_slot(
    monkeypatch, on_disk, caller_for, repo_for
):
    _counting_probe(monkeypatch, False)

    for i in range(64):
        assert _reads(caller_for(i), repo_for(i)) is False

    assert len(hf_tokens._denied_repo_access) == 0
    assert hf_tokens._denial_memory_lost_an_entry() is False


def test_a_flood_does_not_take_the_offline_fallback_away_from_everyone_else(monkeypatch, on_disk):
    """The defect this closes: one evicted entry set `_denial_memory_is_complete` False for the
    life of the process, after which every unaskable probe read as a refusal."""
    _counting_probe(monkeypatch, False)

    for i in range(hf_tokens._DENIAL_MEMORY_MAX + 16):
        _reads(f"hf_stranger_{i}", f"attacker/absent-{i}")

    assert hf_tokens._denial_memory_lost_an_entry() is False
    reset_repo_access_cache()
    _counting_probe(monkeypatch, _ProbeUnreachable(), offline = False)
    assert _reads(OPERATOR_TOKEN, ON_DISK) is True


# `load_model_config`'s anonymous branch closes an ONLINE read the Hub answered no to. Silence
# is not that answer: reading it as one refused a PUBLIC repo already on disk.


def _anonymous_config_read(monkeypatch, tmp_path, probe, env_offline):
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, probe, offline = False)
    monkeypatch.setattr(model_config_module, "_env_offline", lambda: env_offline)
    monkeypatch.setattr(model_config_module, "_config_json_already_cached", lambda *_a, **_k: True)
    return _stub_autoconfig(monkeypatch)


@pytest.mark.parametrize(
    ("host_holds_a_credential", "probe", "env_offline", "served"),
    [
        # A DNS blip must not take a downloaded PUBLIC model away.
        (True, _ProbeUnreachable(), False, True),
        # BOUNDARY. The reverse hole #10264 left open: an answered no must stay refused.
        (True, False, False, False),
        # BOUNDARY. Declared offline is narrowed on purpose: either could have filled the cache.
        (True, _ProbeUnreachable(), True, False),
        (False, _ProbeUnreachable(), True, True),
    ],
    ids = ("unaskable-online", "an-answered-no", "declared-offline", "offline-no-credential"),
)
def test_an_anonymous_cached_config_read_turns_only_on_an_answered_no(
    monkeypatch, tmp_path, host_holds_a_credential, probe, env_offline, served
):
    if not host_holds_a_credential:
        _no_host_credential(monkeypatch)
    reads = _anonymous_config_read(monkeypatch, tmp_path, probe, env_offline)

    if served:
        model_config_module.load_model_config(ON_DISK, use_auth = False, token = False)
        assert reads["n"] == 1
    else:
        with pytest.raises(OSError, match = "not available to an unauthorized caller"):
            model_config_module.load_model_config(ON_DISK, use_auth = False, token = False)
        assert reads["n"] == 0

    if probe is False:
        assert _refused(ANON, ON_DISK) is True
