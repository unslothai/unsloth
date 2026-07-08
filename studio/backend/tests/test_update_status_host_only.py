# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""utils.update_status: the remote (non-host) Studio update response.

A remote client must never be prompted to update Studio, so the host-only
status suppresses the notification and skips the PyPI probe entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.update_status as us  # noqa: E402


def test_host_only_status_is_not_actionable():
    out = us.get_host_only_update_status("1.2.3")
    assert out["host_only"] is True
    assert out["can_show_web_notification"] is False
    assert out["update_available"] is False
    assert out["reason"] == "host_only"
    assert out["latest_version"] is None


def test_host_only_status_skips_probes(monkeypatch):
    # Neither the network probe (PyPI) nor the local filesystem probe
    # (detect_install_source) should run for a remote caller.
    def fail_fetch():
        raise AssertionError("host-only status must not check PyPI")

    def fail_detect():
        raise AssertionError("host-only status must not probe the install source")

    monkeypatch.setattr(us, "get_latest_pypi_version", fail_fetch)
    monkeypatch.setattr(us, "detect_install_source", fail_detect)
    out = us.get_host_only_update_status("1.2.3")
    assert out["host_only"] is True


def test_regular_status_carries_host_only_false(monkeypatch):
    # The host path always reports host_only=False so the shape is uniform.
    monkeypatch.setattr(us, "detect_install_source", lambda: "local_repo")
    out = us.get_studio_update_status("1.2.3")
    assert out["host_only"] is False
