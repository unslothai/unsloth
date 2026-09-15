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

ON_DISK = "acme/downloaded-model"
ABSENT = "acme/never-downloaded"
API_KEY_TOKEN = "hf_api_key_never_validated"
REVISION = "a" * 40


@pytest.fixture(autouse = True)
def _isolate_repo_access_cache():
    reset_repo_access_cache()
    yield
    reset_repo_access_cache()


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

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is True
    assert probes["n"] == 0, "the offline answer went to the network"


def test_the_callers_own_offline_contract_reads_a_repo_on_disk(monkeypatch, tmp_path):
    """``offline=True`` is a cache-only request. It must be answered from the cache, not
    refused because the cache-only branch was not allowed to consult the cache."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    probes = _counting_probe(monkeypatch, True, offline = False)

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK, offline = True) is True
    assert probes["n"] == 0


def test_an_offline_caller_is_still_denied_a_repo_that_is_not_on_disk(monkeypatch, tmp_path):
    """BOUNDARY. With nothing local there is no local answer, and yes would only buy a new
    network fetch, so the gate stays shut."""
    _cache_root(monkeypatch, tmp_path)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ABSENT) is False
    assert public_cache_read_authorized(repo_id = ABSENT) is False


def test_an_interrupted_download_is_not_a_repo_on_disk(monkeypatch, tmp_path):
    """BOUNDARY. "On disk" means a snapshot a load could consume. A cancelled download
    leaves a repo directory with an empty revision under it, which answers nothing."""
    root = _cache_root(monkeypatch, tmp_path)
    (root / f"models--{ON_DISK.replace('/', '--')}" / "snapshots" / REVISION).mkdir(parents = True)
    _counting_probe(monkeypatch, True, offline = True)

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is False


def test_an_anonymous_caller_keeps_its_own_cached_repo_offline(monkeypatch, tmp_path):
    """The sentinel can never authorize itself and the public probe failed closed, so
    offline an API caller with no token could read nothing at all -- public included."""
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

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is True


def test_an_unreachable_probe_is_memoized_as_unknown_not_as_a_denial(monkeypatch, tmp_path):
    """The short TTL existed so a flapping Hub is not re-dialled per call, but it stored a
    denial the Hub never gave: for 30 s every caller was refused, and a download finishing
    inside the window changed nothing. Memoize the fact that nobody answered instead."""
    root = _cache_root(monkeypatch, tmp_path)
    probes = _counting_probe(
        monkeypatch, requests.exceptions.ConnectionError("refused"), offline = False
    )

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is False
    assert [v for _expiry, v in hf_tokens._repo_access_cache.values()] == [None]

    # Inside the same window, the repo lands on disk. No new probe, and the local fact wins.
    _materialize_repo(root, ON_DISK)
    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is True
    assert probes["n"] == 1, "the memo did not spare the second caller the dead round trip"


def test_a_hub_that_answered_no_still_denies_a_repo_on_disk(monkeypatch, tmp_path):
    """BOUNDARY, and the whole point of #10264: online, an API key that cannot reach a
    private repo must not be handed the operator's cached copy of it."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, False, offline = False)

    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is False
    assert cached_read_refused(API_KEY_TOKEN, repo_id = ON_DISK, is_cached = lambda: True) is True


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

    assert hf_tokens._probe_repo_access(ON_DISK, API_KEY_TOKEN, "model") is expected


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

    assert hf_tokens._probe_repo_access(ON_DISK, API_KEY_TOKEN, "model") is None


def test_a_probe_timeout_is_not_an_answer(monkeypatch):
    def _stall(*_a, **_k):
        raise requests.exceptions.Timeout("stalled")

    class _Session:
        get = staticmethod(_stall)

    monkeypatch.setattr("huggingface_hub.utils.get_session", lambda: _Session())

    with pytest.raises(hf_tokens._ProbeTimedOut):
        hf_tokens._probe_repo_access(ON_DISK, API_KEY_TOKEN, "model")


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
    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is False


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

    assert transformers_version._load_config_json(ON_DISK, API_KEY_TOKEN) == {"model_type": "llama"}


def test_the_capability_probes_cache_only_gate_allows_a_repo_on_disk(monkeypatch, tmp_path):
    """``_offline_cache_read_refused`` gates the vision and audio probes on /loras and the
    model-config route. Offline it refused every explicit token, so a cached model reported
    no capabilities at all."""
    root = _cache_root(monkeypatch, tmp_path)
    _materialize_repo(root, ON_DISK)
    _counting_probe(monkeypatch, True, offline = True)

    assert (
        model_config_module._offline_cache_read_refused(API_KEY_TOKEN, ON_DISK, ON_DISK, True)
        is False
    )
    assert (
        model_config_module._offline_cache_read_refused(API_KEY_TOKEN, ABSENT, ABSENT, True) is True
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

    assert picker_service.read_default_chat_template(ON_DISK, API_KEY_TOKEN) == "{{ messages }}"


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

    assert picker_service.read_default_chat_template(ON_DISK, API_KEY_TOKEN) is None


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
    assert cache_reads_authorized(API_KEY_TOKEN, repo_id = ON_DISK) is False
