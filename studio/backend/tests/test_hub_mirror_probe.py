# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A custom Hub endpoint that hands traffic back to huggingface.co is named in the server log."""

from __future__ import annotations

from pathlib import Path
import logging
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import httpx
import pytest

import utils.hub_mirror_probe as probe
import utils.hub_settings as hub_settings

MIRROR = "https://mirror.example.com"
API = f"/api/models/{probe.PROBE_REPO}"
FILE = f"/{probe.PROBE_REPO}/resolve/main/{probe.PROBE_FILE}"


@pytest.fixture(autouse = True)
def _probe_on(monkeypatch):
    monkeypatch.delenv(probe.PROBE_ENV, raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(probe, "_probed", set())


def _serve(monkeypatch, routes):
    """Answer each path with ``routes[path]`` = (status, location); record every request."""
    seen = []

    def handler(request):
        seen.append(
            (
                request.method,
                request.url.host,
                request.url.path,
                request.headers.get("authorization"),
            )
        )
        status, location = routes.get(request.url.path, (404, None))
        return httpx.Response(status, headers = {"location": location} if location else {})

    monkeypatch.setattr(probe, "transport", httpx.MockTransport(handler))
    return seen


def test_whole_site_redirect_is_found_without_contacting_huggingface(monkeypatch):
    # hf-mirror.com outside mainland China: every path is a 308 to huggingface.co.
    seen = _serve(
        monkeypatch,
        {
            API: (308, f"https://huggingface.co{API}"),
            FILE: (308, f"https://huggingface.co{FILE}"),
        },
    )
    findings = probe.find_official_redirects(MIRROR)
    assert [what for what, _ in findings] == ["Hub API calls", "model downloads"]
    assert {host for _, host, _, _ in seen} == {"mirror.example.com"}
    assert all(auth is None for *_, auth in seen)
    assert [method for method, *_ in seen] == ["GET", "HEAD"]


def test_a_mirror_serving_files_from_its_own_cdn_is_quiet(monkeypatch):
    _serve(monkeypatch, {API: (200, None), FILE: (302, "https://cdn.mirror.example.com/blob")})
    assert probe.find_official_redirects(MIRROR) == []


def test_relative_redirects_stay_on_the_mirror(monkeypatch):
    _serve(monkeypatch, {API: (200, None), FILE: (307, "/blobs/abc")})
    assert probe.find_official_redirects(MIRROR) == []


def test_a_pass_through_mirror_sending_files_to_the_official_cdn_is_found(monkeypatch):
    _serve(monkeypatch, {API: (200, None), FILE: (302, "https://cas-bridge.xethub.hf.co/x")})
    findings = probe.find_official_redirects(MIRROR)
    assert findings == [("model downloads", "https://cas-bridge.xethub.hf.co/x")]


def test_a_lookalike_host_is_not_official():
    assert probe.is_official_host("huggingface.co") and probe.is_official_host("cdn-lfs.hf.co")
    assert not probe.is_official_host("huggingface.co.evil.example")
    assert not probe.is_official_host("nothf.co")


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append(message % args)

    def debug(self, *_):
        pass


def test_the_warning_names_the_endpoint_and_the_route(monkeypatch):
    log = _Log()
    monkeypatch.setattr(probe, "logger", log)
    _serve(
        monkeypatch,
        {API: (308, f"https://huggingface.co{API}"), FILE: (308, f"https://huggingface.co{FILE}")},
    )
    probe._probe(MIRROR)
    assert len(log.warnings) == 1
    assert MIRROR in log.warnings[0]
    assert "Hub API calls and model downloads to huggingface.co," in log.warnings[0]


def test_an_unreachable_mirror_logs_no_warning(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request = request)

    log = _Log()
    monkeypatch.setattr(probe, "logger", log)
    monkeypatch.setattr(probe, "transport", httpx.MockTransport(handler))
    probe._probe(MIRROR)
    assert log.warnings == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "https://huggingface.co",
        "https://hf.co",
        "http://127.0.0.1:9000",
        "http://localhost:1234",
    ],
)
def test_nothing_to_probe(endpoint):
    assert not probe.should_probe(endpoint)


def test_offline_and_opt_out_skip_the_probe(monkeypatch):
    assert probe.should_probe(MIRROR)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert not probe.should_probe(MIRROR)
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setenv(probe.PROBE_ENV, "0")
    assert not probe.should_probe(MIRROR)


def test_each_endpoint_is_probed_once_per_process():
    started = []
    for endpoint in (MIRROR, MIRROR + "/", MIRROR, "https://other.example.com"):
        probe.warn_if_mirror_redirects_to_official(endpoint, start = started.append)
    assert [t.daemon for t in started] == [True, True]
    assert [t._args for t in started] == [(MIRROR,), ("https://other.example.com",)]


def test_saving_a_mirror_probes_it_and_modelscope_does_not(monkeypatch):
    import storage.studio_db as studio_db
    import hub.modelscope.router as modelscope

    values: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_settings", lambda keys: {k: values[k] for k in keys if k in values}
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates, **_: values.update(updates) or values
    )
    monkeypatch.setattr(hub_settings, "_operator_env", None)
    monkeypatch.setattr(hub_settings, "_operator_endpoints", None, raising = False)
    monkeypatch.setattr(modelscope, "internal_endpoint", lambda: "http://127.0.0.1:1234")
    monkeypatch.delenv(hub_settings.SOURCE_ENV, raising = False)
    monkeypatch.delenv("HF_ENDPOINT", raising = False)
    asked = []
    monkeypatch.setattr(probe, "warn_if_mirror_redirects_to_official", asked.append)
    try:
        hub_settings.set_hub_settings(MIRROR, False)
        assert asked == [MIRROR]
        hub_settings.set_hub_source("modelscope")
        assert asked == [MIRROR]
    finally:
        monkeypatch.undo()
        hub_settings._refresh_imported_hub_libraries()


def test_the_default_hub_never_imports_the_probe(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising = False)
    monkeypatch.setattr(hub_settings, "_operator_env", None)
    monkeypatch.setattr(hub_settings, "_operator_endpoints", None, raising = False)
    monkeypatch.setattr(hub_settings, "_saved_only_endpoints", frozenset(), raising = False)
    monkeypatch.setattr(
        hub_settings, "get_hub_settings", lambda: hub_settings.HubSettings("", False, False)
    )
    monkeypatch.delitem(sys.modules, "utils.hub_mirror_probe")
    try:
        hub_settings.apply_hub_settings()
        assert "utils.hub_mirror_probe" not in sys.modules
    finally:
        sys.modules["utils.hub_mirror_probe"] = probe


def test_one_failing_route_does_not_hide_the_other(monkeypatch):
    def handler(request):
        if request.method == "HEAD":
            raise httpx.ConnectTimeout("timed out", request = request)
        return httpx.Response(308, headers = {"location": f"https://huggingface.co{API}"})

    monkeypatch.setattr(probe, "transport", httpx.MockTransport(handler))
    assert probe.find_official_redirects(MIRROR) == [
        ("Hub API calls", f"https://huggingface.co{API}")
    ]
