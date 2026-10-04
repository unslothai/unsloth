# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP size-limit transport selection and recovery tests."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import HTTPException

from hub.services import download_lifecycle as dl
from hub.utils import download_registry

# Sizes from the repository reported in #10840.
_OVERSIZED = 54_400_261_312
_UNDER_THE_CEILING = 49_859_583_136


class _Sibling:
    def __init__(self, rfilename: str, size: int):
        self.rfilename = rfilename
        self.size = size
        self.lfs = {"sha256": f"{abs(hash(rfilename)):064x}"[:64]}


def _flash_next_siblings() -> list[_Sibling]:
    """A two-variant GGUF repo: one variant past the ceiling, one under it."""
    return [
        _Sibling("UD-Q5_K_XL/M-UD-Q5_K_XL-00001-of-00002.gguf", _OVERSIZED),
        _Sibling("UD-Q5_K_XL/M-UD-Q5_K_XL-00002-of-00002.gguf", 43_500_000_000),
        _Sibling("UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf", _UNDER_THE_CEILING),
        _Sibling("UD-Q4_K_XL/M-UD-Q4_K_XL-00002-of-00002.gguf", 20_000_000_000),
        _Sibling("README.md", 1_024),
    ]


# --------------------------------------------------------------------------------------------
# The ceiling itself
# --------------------------------------------------------------------------------------------


def test_the_ceiling_is_read_from_the_installed_hub():
    from huggingface_hub import constants as hf_constants
    assert download_registry.http_max_file_bytes() == hf_constants.MAX_HTTP_DOWNLOAD_SIZE


def test_only_a_file_past_the_ceiling_refuses_http():
    ceiling = download_registry.http_max_file_bytes()
    assert download_registry.http_size_ceiling_reason(ceiling) is None
    assert download_registry.http_size_ceiling_reason(ceiling + 1) is not None
    # Unknown sizes preserve the previous transport choice.
    assert download_registry.http_size_ceiling_reason(None) is None
    assert download_registry.http_size_ceiling_reason(0) is None


def test_the_refusal_names_the_size_and_the_limit():
    reason = download_registry.http_size_ceiling_reason(_OVERSIZED)
    assert "54.4GB" in reason and "50GB" in reason and "Xet" in reason


def test_transport_unavailable_reason_is_size_aware():
    assert (
        download_registry.download_transport_unavailable_reason(download_registry.TRANSPORT_HTTP)
        is None
    )
    assert (
        download_registry.download_transport_unavailable_reason(
            download_registry.TRANSPORT_HTTP, largest_file_bytes = _OVERSIZED
        )
        is not None
    )


# --------------------------------------------------------------------------------------------
# Which file the ceiling is asked about
# --------------------------------------------------------------------------------------------


def test_the_size_is_measured_per_variant_not_per_repo(monkeypatch):
    """The 46GB variant in a repo that also holds a 54GB one keeps HTTP."""
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(_flash_next_siblings()))
    assert (
        dl.largest_download_file_bytes("model", "unsloth/M-GGUF", variant = "ud-q5_k_xl")
        == _OVERSIZED
    )
    assert (
        dl.largest_download_file_bytes("model", "unsloth/M-GGUF", variant = "ud-q4_k_xl")
        == _UNDER_THE_CEILING
    )


def test_a_scoped_file_list_is_measured_on_its_own_files(monkeypatch):
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(_flash_next_siblings()))
    assert (
        dl.largest_download_file_bytes(
            "model",
            "unsloth/M-GGUF",
            files = ["UD-Q4_K_XL/M-UD-Q4_K_XL-00002-of-00002.gguf"],
        )
        == 20_000_000_000
    )


def test_a_model_snapshot_is_measured_on_the_files_it_actually_fetches(monkeypatch):
    """Ignored exports must not affect snapshot transport selection."""
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(_flash_next_siblings()))
    assert dl.largest_download_file_bytes("model", "unsloth/M-GGUF") == 1_024


def test_an_ignored_oversized_export_does_not_decide_the_transport(monkeypatch):
    siblings = [
        _Sibling("model-00001-of-00002.safetensors", 30_000_000_000),
        _Sibling("model-00002-of-00002.safetensors", 20_000_000_000),
        _Sibling("onnx/model.onnx", _OVERSIZED),
        _Sibling("mlx/weights.safetensors", _OVERSIZED),
    ]
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(siblings))
    assert dl.largest_download_file_bytes("model", "org/Weights") == 30_000_000_000


def test_a_dataset_is_measured_unfiltered(monkeypatch):
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: (_Sibling("shard.onnx", _OVERSIZED),))
    assert dl.largest_download_file_bytes("dataset", "org/Data") == _OVERSIZED


def test_unreadable_metadata_measures_nothing(monkeypatch):
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: ())
    assert dl.largest_download_file_bytes("model", "unsloth/M-GGUF") is None


def test_an_unresolvable_variant_measures_nothing(monkeypatch):
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(_flash_next_siblings()))
    assert dl.largest_download_file_bytes("model", "unsloth/M-GGUF", variant = "nope") is None


def test_a_finalized_blob_is_not_a_file_this_download_has_to_fetch(monkeypatch, tmp_path):
    """A finalized oversized shard must not block HTTP for smaller remaining files."""
    siblings = _flash_next_siblings()
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(siblings))
    oversized = next(s for s in siblings if s.size == _OVERSIZED)

    blobs = tmp_path / "models--unsloth--M-GGUF" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / oversized.lfs["sha256"]).write_bytes(b"")
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *a, **k: iter([blobs.parent]),
    )

    assert (
        dl.largest_download_file_bytes("model", "unsloth/M-GGUF", variant = "ud-q5_k_xl")
        == 43_500_000_000
    )


def test_a_scoped_job_is_measured_on_its_own_file_list(monkeypatch):
    """Scoped jobs must use metadata files instead of their synthetic variant key."""
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(_flash_next_siblings()))
    metadata = _types.SimpleNamespace(
        variant = "@ab-big",
        scoped_files = ("UD-Q5_K_XL/M-UD-Q5_K_XL-00001-of-00002.gguf",),
        hub_cache = None,
    )
    assert dl._largest_file_bytes_for_job("model", "unsloth/M-GGUF", metadata) == _OVERSIZED
    assert (
        dl._largest_file_bytes_for_job(
            "model",
            "unsloth/M-GGUF",
            _types.SimpleNamespace(variant = "@ab-big", scoped_files = (), hub_cache = None),
        )
        is None
    )


def test_a_probe_failure_never_escapes_into_the_watcher(monkeypatch):
    monkeypatch.setattr(dl, "largest_download_file_bytes", _raise)
    assert dl._largest_file_bytes_for_job("model", "unsloth/M-GGUF") is None


def _raise(*a, **k):
    raise RuntimeError("hub is down")


# --------------------------------------------------------------------------------------------
# What the transport choice does with it
# --------------------------------------------------------------------------------------------


def test_auto_takes_xet_over_a_demoted_verdict_when_http_cannot_serve(monkeypatch):
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    fake = _types.ModuleType("utils.hf_xet_fallback")
    fake.xet_health = lambda **kw: _types.SimpleNamespace(
        use_xet = False, reason = "Xet stalled twice on this machine"
    )
    fake.free_ram_pressure_reason = lambda: "HTTP: only 1.0GB RAM free"
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)

    assert dl.resolve_auto_use_xet()[0] is False
    use_xet, reason = dl.resolve_auto_use_xet(largest_file_bytes = _OVERSIZED)
    assert use_xet is True
    assert "HTTPS cannot fetch" in reason


def test_auto_still_defers_to_the_verdict_under_the_ceiling(monkeypatch):
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    monkeypatch.setattr(dl, "resolve_auto_use_xet", lambda **kw: (False, "demoted"))
    assert dl.resolve_requested_use_xet("auto", True, largest_file_bytes = _UNDER_THE_CEILING) == (
        False,
        "demoted",
    )


def test_an_explicit_https_request_is_rejected_by_name(monkeypatch):
    """An impossible explicit HTTP request should fail instead of switching transport."""
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    use_xet, _ = dl.resolve_requested_use_xet("http", False, largest_file_bytes = _OVERSIZED)
    assert use_xet is False
    with pytest.raises(HTTPException) as excinfo:
        dl.resolve_transport(use_xet, largest_file_bytes = _OVERSIZED)
    assert excinfo.value.status_code == 400
    assert "HTTPS cannot fetch" in excinfo.value.detail
    assert (
        dl.resolve_transport(False, largest_file_bytes = _UNDER_THE_CEILING)
        == download_registry.TRANSPORT_HTTP
    )


def test_an_install_without_hf_xet_is_told_that_too(monkeypatch):
    """Report when the only suitable transport is also unavailable."""
    monkeypatch.setattr(
        download_registry.importlib.util,
        "find_spec",
        lambda name: None if name == "hf_xet" else object(),
    )
    use_xet, _ = dl.resolve_requested_use_xet("auto", True, largest_file_bytes = _OVERSIZED)
    assert use_xet is False
    with pytest.raises(HTTPException) as excinfo:
        dl.resolve_transport(use_xet, largest_file_bytes = _OVERSIZED)
    assert "HTTPS cannot fetch" in excinfo.value.detail
    assert "hf_xet is not installed" in excinfo.value.detail


def test_capabilities_mark_http_unavailable_for_an_oversized_download(monkeypatch):
    monkeypatch.setattr(download_registry.importlib.util, "find_spec", lambda name: object())
    caps = download_registry.get_download_transport_capabilities(largest_file_bytes = _OVERSIZED)
    assert caps.http.available is False
    assert "HTTPS cannot fetch" in caps.http.reason
    assert caps.auto_resolves_to == download_registry.TRANSPORT_XET
    assert (
        download_registry.get_download_transport_capabilities(
            largest_file_bytes = _UNDER_THE_CEILING
        ).http.available
        is True
    )


# --------------------------------------------------------------------------------------------
# What the user is told
# --------------------------------------------------------------------------------------------


_HUB_REFUSAL = (
    "ValueError: The file is too large to be downloaded using the regular download method. "
    " Install `hf_xet` with `pip install hf_xet` for xet-powered downloads."
)


def test_the_hub_refusal_is_rewritten_into_what_happened():
    rewritten = download_registry.humanize_worker_error(_HUB_REFUSAL)
    assert "hf_xet" not in rewritten
    assert "HTTPS cannot fetch" in rewritten


def test_a_known_size_makes_the_rewrite_specific():
    rewritten = download_registry.humanize_worker_error(_HUB_REFUSAL, largest_file_bytes = _OVERSIZED)
    assert "54.4GB" in rewritten


def test_every_other_failure_is_passed_through():
    assert download_registry.humanize_worker_error("404 Client Error") == "404 Client Error"
    assert download_registry.humanize_worker_error("") == ""


# --------------------------------------------------------------------------------------------
# The recovery ladder
# --------------------------------------------------------------------------------------------


class _Proc:
    pid = 4242

    def __init__(
        self,
        rc,
        stderr = b"",
    ):
        import io

        self.rc = rc
        self.stderr = io.BytesIO(stderr)
        self.waited = False

    def poll(self):
        return self.rc if self.waited else None

    def wait(self, timeout = None):
        self.waited = True
        return self.rc

    def kill(self):
        pass


class _ImmediateThread:
    def __init__(
        self,
        *,
        target,
        args = (),
        kwargs = None,
        **_kwargs,
    ):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


def _claimed_registry():
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("unsloth/M-GGUF::UD-Q5_K_XL")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "unsloth/M-GGUF",
        variant = "UD-Q5_K_XL",
        blob_hashes = frozenset({"blob"}),
    )[0]
    return key, registry


_WATCHDOG_STARTS: list = []


def _ladder_setup(
    monkeypatch,
    tmp_path,
    *,
    largest_file_bytes,
    stall_message = None,
    probe = True,
):
    """Set up the recovery ladder without a real worker."""
    from hub.utils import state_dir

    _WATCHDOG_STARTS.clear()
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(dl.threading, "Thread", _ImmediateThread)
    if probe:
        monkeypatch.setattr(dl, "_largest_file_bytes_for_job", lambda *a, **k: largest_file_bytes)

    def _start(*_a, on_stall, **_k):
        _WATCHDOG_STARTS.append(True)
        if stall_message is not None:
            on_stall(stall_message)
        return None

    monkeypatch.setattr(dl, "_start_stall_watchdog", _start)


def _run_worker(registry, key, proc):
    return dl.register_worker(
        registry,
        key,
        proc,
        hf_token = None,
        label = "unsloth/M-GGUF",
        log_prefix = "Download",
        logger = dl.logger,
        repo_type = "model",
        repo_id = "unsloth/M-GGUF",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )


def test_the_ladder_does_not_spend_the_job_on_a_doomed_http_retry(monkeypatch, tmp_path):
    """An oversized download must not fall back to HTTP."""
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = _OVERSIZED)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 2)
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"xet transport failed"))

    assert rungs == [], "an oversized download was retried over a transport that cannot serve it"
    state, error, _generation = dl.idle_status(
        registry, key, repo_type = "model", repo_id = "unsloth/M-GGUF", variant = "UD-Q5_K_XL"
    )
    assert state == "error"
    assert "xet transport failed" in error
    assert "HTTPS cannot fetch" in error


def test_an_oversized_download_still_gets_its_xet_retry(monkeypatch, tmp_path):
    """HTTP unavailability must not remove the Xet retry."""
    _ladder_setup(
        monkeypatch,
        tmp_path,
        largest_file_bytes = _OVERSIZED,
        stall_message = "Download appears stalled (xet transport) -- no progress for 30s",
    )
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 2)
    rungs: list = []
    monkeypatch.setattr(
        dl,
        "_try_transport_retry",
        lambda *a, **kw: rungs.append((kw.get("retry_transport"), kw.get("xet_attempt"))),
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"stalled"))

    assert rungs == [(download_registry.TRANSPORT_XET, 2)]


def test_a_download_under_the_ceiling_keeps_its_http_rung(monkeypatch, tmp_path):
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = _UNDER_THE_CEILING)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"xet transport failed"))

    assert rungs == [download_registry.TRANSPORT_HTTP]


def test_a_failed_xet_respawn_does_not_drop_to_https_either(monkeypatch, tmp_path):
    """A failed Xet respawn must not bypass HTTP eligibility checks."""
    from hub.utils import state_dir

    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(dl, "_largest_file_bytes_for_job", lambda *a, **k: _OVERSIZED)
    monkeypatch.setattr(dl, "spawn_worker", _boom_spawn)
    key, registry = _claimed_registry()
    rungs: list = []
    real_retry = dl._try_transport_retry

    def _record(registry_, key_, **kwargs):
        rungs.append(kwargs.get("retry_transport"))
        return real_retry(registry_, key_, **kwargs)

    monkeypatch.setattr(dl, "_try_transport_retry", _record)

    assert (
        real_retry(
            registry,
            key,
            hf_token = None,
            label = "unsloth/M-GGUF",
            log_prefix = "Download",
            logger = dl.logger,
            repo_type = "model",
            repo_id = "unsloth/M-GGUF",
            watch_name = "model-watch",
            retry_transport = download_registry.TRANSPORT_XET,
            xet_attempt = 2,
        )
        is False
    )

    assert download_registry.TRANSPORT_HTTP not in rungs
    state, error, _generation = dl.idle_status(
        registry, key, repo_type = "model", repo_id = "unsloth/M-GGUF", variant = "UD-Q5_K_XL"
    )
    assert state == "error"
    assert "HTTPS cannot fetch" in error


def _boom_spawn(*_args, **_kwargs):
    raise RuntimeError("could not fork")


def test_the_rung_is_judged_on_the_cache_the_worker_leaves_behind(monkeypatch, tmp_path):
    """Judge HTTP eligibility from the cache left by the failed Xet worker."""
    siblings = _flash_next_siblings()
    oversized = next(s for s in siblings if s.size == _OVERSIZED)
    blobs = tmp_path / "models--unsloth--M-GGUF" / "blobs"
    blobs.mkdir(parents = True)

    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(siblings))
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *a, **k: iter([blobs.parent]),
    )
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = None, probe = False)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert dl._largest_file_bytes_for_job(
        "model", "unsloth/M-GGUF", _metadata_of(registry, key)
    ) == (_OVERSIZED)

    class _LandsTheBigShard(_Proc):
        def wait(self, timeout = None):
            (blobs / oversized.lfs["sha256"]).write_bytes(b"")
            return super().wait(timeout)

    assert _run_worker(registry, key, _LandsTheBigShard(1, b"xet failed on the second shard"))

    assert rungs == [
        download_registry.TRANSPORT_HTTP
    ], "the HTTP rung stayed closed on a measurement the worker had already invalidated"


def test_a_final_xet_attempt_with_no_http_rung_is_still_watched(monkeypatch, tmp_path):
    """Watch the final Xet attempt even when no retry remains."""
    _ladder_setup(
        monkeypatch,
        tmp_path,
        largest_file_bytes = _OVERSIZED,
        stall_message = "Download appears stalled (xet transport) -- no progress for 30s",
    )
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"killed"))

    assert _WATCHDOG_STARTS, "a final Xet worker was left unwatched"
    assert rungs == []
    state, error, _generation = dl.idle_status(
        registry, key, repo_type = "model", repo_id = "unsloth/M-GGUF", variant = "UD-Q5_K_XL"
    )
    assert state == "error"
    assert "HTTPS cannot fetch" in error


def _metadata_of(registry, key):
    return registry.get_job_metadata(key)


# --------------------------------------------------------------------------------------------
# A metadata refresh that fails
# --------------------------------------------------------------------------------------------


class _OneShotHfApi:
    """Answer ``repo_info`` once, then fail the way a dropped connection does."""

    calls = 0

    def __init__(self, token = None):
        self.token = token

    def repo_info(self, *_args, **_kwargs):
        type(self).calls += 1
        if type(self).calls > 1:
            raise RuntimeError("connection reset by peer")
        return _types.SimpleNamespace(siblings = _flash_next_siblings())


def _hub_answers_once(monkeypatch):
    """Install a hub whose second listing fails, with the sibling TTL already elapsed."""
    import time as _time
    import huggingface_hub

    _OneShotHfApi.calls = 0
    dl._REPO_SIBLINGS.clear()
    monkeypatch.setattr(huggingface_hub, "HfApi", _OneShotHfApi)
    clock = iter([0.0])
    monkeypatch.setattr(
        dl,
        "time",
        _types.SimpleNamespace(
            monotonic = lambda: next(clock, dl._REPO_SIBLINGS_TTL_SECONDS * 100),
            sleep = _time.sleep,
        ),
    )


def test_a_failed_refresh_serves_the_listing_it_already_read(monkeypatch):
    """A dropped refresh must not turn a measured size into an unmeasured one."""
    _hub_answers_once(monkeypatch)

    first = dl.largest_download_file_bytes("model", "unsloth/M-GGUF", variant = "ud-q5_k_xl")
    second = dl.largest_download_file_bytes("model", "unsloth/M-GGUF", variant = "ud-q5_k_xl")

    assert _OneShotHfApi.calls == 2, "the second measurement must really have gone to the hub"
    assert first == _OVERSIZED
    assert second == _OVERSIZED


def test_filling_the_cache_evicts_one_entry_not_all_of_them(monkeypatch):
    """An in-flight download's listing must survive unrelated traffic."""
    import huggingface_hub

    dl._REPO_SIBLINGS.clear()
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda token = None: _types.SimpleNamespace(
            repo_info = lambda *a, **k: _types.SimpleNamespace(siblings = ())
        ),
    )
    for i in range(dl._REPO_SIBLINGS_MAX + 1):
        dl._repo_siblings("model", f"org/r{i}", None)

    assert len(dl._REPO_SIBLINGS) == dl._REPO_SIBLINGS_MAX
    assert ("model", "org/r0", dl.hf_cache_scan.token_fingerprint(None)) not in dl._REPO_SIBLINGS


def test_the_http_rung_stays_closed_when_the_refresh_fails(monkeypatch, tmp_path):
    """The post-exit recheck must not reopen HTTP on metadata it could not refresh."""
    _hub_answers_once(monkeypatch)
    blobs = tmp_path / "models--unsloth--M-GGUF" / "blobs"
    blobs.mkdir(parents = True)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *a, **k: iter([blobs.parent]),
    )
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = None, probe = False)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"xet transport failed"))

    assert rungs == [], "a doomed HTTP retry was spent on a size the refresh could not re-read"
    state, error, _generation = dl.idle_status(
        registry, key, repo_type = "model", repo_id = "unsloth/M-GGUF", variant = "UD-Q5_K_XL"
    )
    assert state == "error"
    assert "HTTPS cannot fetch" in error


def test_a_finalized_shard_still_reopens_http_on_a_failed_refresh(monkeypatch, tmp_path):
    """Whether the oversized shard landed is read from the local cache, not from the hub."""
    _hub_answers_once(monkeypatch)
    oversized = next(s for s in _flash_next_siblings() if s.size == _OVERSIZED)
    blobs = tmp_path / "models--unsloth--M-GGUF" / "blobs"
    blobs.mkdir(parents = True)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *a, **k: iter([blobs.parent]),
    )
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = None, probe = False)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    class _LandsTheBigShard(_Proc):
        def wait(self, timeout = None):
            (blobs / oversized.lfs["sha256"]).write_bytes(b"")
            return super().wait(timeout)

    assert _run_worker(registry, key, _LandsTheBigShard(1, b"xet failed on the second shard"))

    assert rungs == [download_registry.TRANSPORT_HTTP]


# --------------------------------------------------------------------------------------------
# Whose credential the probe measures with
# --------------------------------------------------------------------------------------------


def _record_probe_tokens(monkeypatch) -> list:
    """Capture the token every ``repo_info`` call is made under."""
    import huggingface_hub

    seen: list = []
    dl._REPO_SIBLINGS.clear()

    class _Recorder:
        def __init__(self, token = None):
            self.token = token

        def repo_info(self, *_args, **_kwargs):
            seen.append(self.token)
            return _types.SimpleNamespace(siblings = _flash_next_siblings())

    monkeypatch.setattr(huggingface_hub, "HfApi", _Recorder)
    return seen


def test_a_caller_denied_the_ambient_login_measures_anonymously(monkeypatch):
    """``token=None`` would borrow the backend's saved login; ``False`` is the anonymous sentinel."""
    seen = _record_probe_tokens(monkeypatch)

    dl.largest_download_file_bytes(
        "model", "unsloth/M-GGUF", variant = "ud-q5_k_xl", allow_ambient_token = False
    )

    assert seen == [False]


def test_a_ui_session_still_measures_under_the_ambient_login(monkeypatch):
    seen = _record_probe_tokens(monkeypatch)

    dl.largest_download_file_bytes(
        "model", "unsloth/M-GGUF", variant = "ud-q5_k_xl", allow_ambient_token = True
    )

    assert seen == [None]


def test_an_anonymous_probe_does_not_read_an_ambient_cache_entry(monkeypatch):
    """The two boundaries take separate cache identities, so neither serves the other."""
    seen = _record_probe_tokens(monkeypatch)

    dl.largest_download_file_bytes("model", "unsloth/M-GGUF", allow_ambient_token = True)
    dl.largest_download_file_bytes("model", "unsloth/M-GGUF", allow_ambient_token = False)

    assert seen == [None, False]
    assert len(dl._REPO_SIBLINGS) == 2


def test_the_watcher_probes_under_the_jobs_own_boundary(monkeypatch, tmp_path):
    """Every rung of the ladder measures with the token policy the job started under."""
    seen = _record_probe_tokens(monkeypatch)
    blobs = tmp_path / "models--unsloth--M-GGUF" / "blobs"
    blobs.mkdir(parents = True)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *a, **k: iter([blobs.parent]),
    )
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = None, probe = False)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    monkeypatch.setattr(dl, "_try_transport_retry", lambda *a, **kw: None)
    key, registry = _claimed_registry()

    assert dl.register_worker(
        registry,
        key,
        _Proc(1, b"xet transport failed"),
        hf_token = None,
        label = "unsloth/M-GGUF",
        log_prefix = "Download",
        logger = dl.logger,
        repo_type = "model",
        repo_id = "unsloth/M-GGUF",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
        allow_ambient_token = False,
    )

    assert seen, "the watcher never measured"
    assert set(seen) == {False}, "a watcher probe borrowed the backend's own login"


# --------------------------------------------------------------------------------------------
# The request path measures what the worker will actually fetch
# --------------------------------------------------------------------------------------------


def _request_path(monkeypatch):
    """Take a model download request as far as the transport decision, then stop."""
    from hub.services.models import downloads as model_downloads

    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(_flash_next_siblings()))
    monkeypatch.setattr(model_downloads, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(model_downloads, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(model_downloads, "scoped_file_blob_hashes", lambda *a, **k: frozenset())
    monkeypatch.setattr(dl, "launch_worker", lambda *a, **k: "running")
    return model_downloads


def _download(model_downloads, **over):
    import asyncio

    from hub.schemas.downloads import DownloadModelRequest

    body = {"repo_id": "unsloth/M-GGUF", "transport_mode": "http"}
    body.update(over)
    return asyncio.run(model_downloads.download_model_response(DownloadModelRequest(**body)))


_BIG_SHARD = "UD-Q5_K_XL/M-UD-Q5_K_XL-00001-of-00002.gguf"


def test_files_without_a_scope_id_do_not_decide_the_transport(monkeypatch):
    """``files`` is ignored without ``scope_id``, so it must not shrink the measured size."""
    model_downloads = _request_path(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        _download(
            model_downloads,
            gguf_variant = "UD-Q5_K_XL",
            files = ["README.md"],
        )

    assert excinfo.value.status_code == 400
    assert "HTTPS cannot fetch" in excinfo.value.detail


def test_an_ignored_file_list_does_not_refuse_a_download_that_fits(monkeypatch):
    """The other direction: an ignored oversized name must not reject a valid snapshot."""
    model_downloads = _request_path(monkeypatch)

    result = _download(model_downloads, files = [_BIG_SHARD])

    assert result["transport"] == download_registry.TRANSPORT_HTTP


def test_a_real_scope_is_still_measured_on_its_files(monkeypatch):
    model_downloads = _request_path(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        _download(model_downloads, scope_id = "diffusion", files = [_BIG_SHARD])

    assert excinfo.value.status_code == 400
    assert "HTTPS cannot fetch" in excinfo.value.detail


# --------------------------------------------------------------------------------------------
# A measurement the cache no longer holds
# --------------------------------------------------------------------------------------------


def test_nothing_left_to_fetch_is_zero_not_unknown(monkeypatch, tmp_path):
    """A carried measurement only stands over an unknown, so the two must not share a value."""
    siblings = [_Sibling("only.gguf", _OVERSIZED)]
    monkeypatch.setattr(dl, "_repo_siblings", lambda *a, **k: tuple(siblings))
    blobs = tmp_path / "models--unsloth--M-GGUF" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / siblings[0].lfs["sha256"]).write_bytes(b"")
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *a, **k: iter([blobs.parent]),
    )

    assert dl.largest_download_file_bytes("model", "unsloth/M-GGUF") == 0


def _evicting_probe(values):
    """A probe that measures once and is then unable to measure at all."""
    remaining = list(values)

    def _probe(*_a, **_k):
        return remaining.pop(0) if remaining else None

    return _probe


def test_an_evicted_listing_does_not_reopen_the_http_rung(monkeypatch, tmp_path):
    """Losing the cached listing must not turn a measured size back into an unknown one."""
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = None, probe = False)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    monkeypatch.setattr(dl, "_largest_file_bytes_for_job", _evicting_probe([_OVERSIZED]))
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"xet transport failed"))

    assert rungs == [], "an evicted listing reopened a rung that cannot serve this download"
    state, error, _generation = dl.idle_status(
        registry, key, repo_type = "model", repo_id = "unsloth/M-GGUF", variant = "UD-Q5_K_XL"
    )
    assert state == "error"
    assert "HTTPS cannot fetch" in error


def test_a_finalized_shard_still_overrides_the_carried_size(monkeypatch, tmp_path):
    """A fresh measurement always wins; only an unmeasurable one defers to the carried size."""
    _ladder_setup(monkeypatch, tmp_path, largest_file_bytes = None, probe = False)
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 1)
    monkeypatch.setattr(dl, "_largest_file_bytes_for_job", _evicting_probe([_OVERSIZED, 0]))
    rungs: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: rungs.append(kw.get("retry_transport"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"xet failed on the second shard"))

    assert rungs == [download_registry.TRANSPORT_HTTP]


def test_the_carried_size_survives_a_xet_retry(monkeypatch, tmp_path):
    """The next rung inherits it, so an hours-long job never measures from scratch alone."""
    _ladder_setup(
        monkeypatch,
        tmp_path,
        largest_file_bytes = None,
        probe = False,
        stall_message = "Download appears stalled (xet transport) -- no progress for 30s",
    )
    monkeypatch.setattr(dl, "_xet_attempt_budget", lambda: 2)
    monkeypatch.setattr(dl, "_largest_file_bytes_for_job", _evicting_probe([_OVERSIZED]))
    carried: list = []
    monkeypatch.setattr(
        dl, "_try_transport_retry", lambda *a, **kw: carried.append(kw.get("largest_file_bytes"))
    )
    key, registry = _claimed_registry()

    assert _run_worker(registry, key, _Proc(1, b"stalled"))

    assert carried == [_OVERSIZED]
