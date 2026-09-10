# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""url_exists must tell a missing wheel (404) from GitHub refusing us (403/429/5xx, dropped
connection). A refusal is retried once and then reported as None. Callers still take their
slow path either way, since the wheel may well exist and a source build needs no GitHub;
what changes is the message ("could not check", never "not published") and that a refusal
is never treated as proof that no prebuilt exists."""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import wheel_utils  # noqa: E402


class _Ok:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, responder):
    calls = []

    def fake_urlopen(req, timeout = 10):
        calls.append(req.full_url)
        return responder(len(calls))

    monkeypatch.setattr(wheel_utils.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(wheel_utils.time, "sleep", lambda s: None)
    return calls


def test_a_404_is_final_after_one_probe(monkeypatch):
    def gone(_n):
        raise urllib.error.HTTPError("u", 404, "not found", None, None)

    calls = _patch(monkeypatch, gone)
    assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is False
    assert len(calls) == 1


@pytest.mark.parametrize("status", [403, 429, 503])
def test_a_refusal_is_retried_once_then_reported(monkeypatch, caplog, status):
    def refused(_n):
        raise urllib.error.HTTPError("u", status, "unavailable", None, None)

    calls = _patch(monkeypatch, refused)
    with caplog.at_level("WARNING", logger = wheel_utils._logger.name):
        assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is None
    assert len(calls) == 2
    assert any(f"HTTP {status}" in rec.getMessage() for rec in caplog.records)


def test_a_transient_failure_recovers_on_the_retry(monkeypatch):
    def flaky(n):
        if n == 1:
            raise urllib.error.URLError("connection reset")
        return _Ok()

    calls = _patch(monkeypatch, flaky)
    assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is True
    assert len(calls) == 2


def test_a_timeout_is_not_retried(monkeypatch):
    def slow(_n):
        raise urllib.error.URLError(TimeoutError("timed out"))

    calls = _patch(monkeypatch, slow)
    assert wheel_utils.url_exists("https://github.com/x/releases/download/v1/w.whl") is None
    assert len(calls) == 1


@pytest.mark.parametrize("installer", ["training", "inference"])
@pytest.mark.parametrize("status", [403, 429, 503, 404])
def test_a_refused_probe_still_reaches_the_source_build(monkeypatch, installer, status):
    """Whatever the probe answered, the slow path still runs. A refusal skips only the
    prebuilt fast path: PyPI and the source build need no GitHub, and returning early
    would make a throttled release host cost the package -- for mamba-ssm, the model."""
    from core.training import worker
    from utils import ssm_runtime

    module = worker if installer == "training" else ssm_runtime
    url = "https://github.com/x/releases/download/v1/w.whl"

    def refused(_n):
        raise urllib.error.HTTPError(url, status, "refused", None, None)

    _patch(monkeypatch, refused)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(module, "probe_torch_wheel_env", lambda **kw: {})
    monkeypatch.setattr(module, "direct_wheel_url", lambda **kw: url)
    monkeypatch.setattr(module, "_is_importable", lambda name: False)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    wheel_install = Mock(side_effect = AssertionError("must not download a refused wheel"))
    monkeypatch.setattr(module, "install_wheel", wheel_install)
    source_build = Mock(return_value = SimpleNamespace(returncode = 1, stdout = "build failed"))
    statuses = []
    kwargs = dict(import_name = "mamba_ssm", display_name = "mamba-ssm", pypi_name = "mamba-ssm")
    if installer == "training":
        monkeypatch.setattr(worker._sp, "run", source_build)
        monkeypatch.setattr(worker, "_send_status", lambda queue, message: statuses.append(message))
        installed = worker._attempt_package_install(
            event_queue = None, pypi_version = "2.3.1", **kwargs
        )
    else:
        installed = ssm_runtime._install_kernel(
            package_version = "2.3.1",
            release_tag = "v2.3.1",
            release_base_url = "https://github.com/x/releases/download",
            status_cb = statuses.append,
            run = source_build,
            **kwargs,
        )
    assert installed is False
    wheel_install.assert_not_called()
    assert source_build.call_count == 1
    if status != 404:
        assert any("Could not check" in message for message in statuses)


def test_a_refused_probe_does_not_fail_a_mamba_model(monkeypatch):
    """ensure_ssm_runtime raises when mamba-ssm does not install, so a refused probe
    that returned early turned a throttled GitHub into an unloadable model."""
    from utils import ssm_runtime

    url = "https://github.com/x/releases/download/v1/w.whl"

    def refused(_n):
        raise urllib.error.HTTPError(url, 503, "refused", None, None)

    _patch(monkeypatch, refused)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setattr(ssm_runtime, "probe_torch_wheel_env", lambda **kw: {})
    monkeypatch.setattr(ssm_runtime, "direct_wheel_url", lambda **kw: url)
    # Absent before the build, present after it.
    built = []
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: bool(built))
    monkeypatch.setattr(ssm_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        ssm_runtime, "install_wheel", Mock(side_effect = AssertionError("refused wheel"))
    )

    def build(*args, **kwargs):
        built.append(True)
        return SimpleNamespace(returncode = 0, stdout = "")

    source_build = Mock(side_effect = build)
    installed = ssm_runtime._install_kernel(
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        package_version = "2.3.1",
        release_tag = "v2.3.1",
        release_base_url = "https://github.com/x/releases/download",
        status_cb = lambda message: None,
        run = source_build,
    )
    assert installed is True
    assert source_build.call_count == 1
