# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in Windows checks against Studio's disposable WSL environment."""

import os
import sys
import uuid

import pytest

from core.inference import wsl_host


@pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("UNSLOTH_TEST_WSL_DISTRO"),
    reason = "requires Windows and an explicitly selected Studio test WSL distro",
)
def test_real_wsl_preserves_shell_arguments(monkeypatch):
    monkeypatch.setattr(wsl_host, "distro_name", lambda: os.environ["UNSLOTH_TEST_WSL_DISTRO"])
    output = wsl_host.guest(
        ["sh", "-c", 'printf "%s\n" "$0" "$1"', "literal $HOME", 'value "quotes"']
    )
    assert output.splitlines() == ["literal $HOME", 'value "quotes"']


@pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("UNSLOTH_TEST_WSL_DISTRO"),
    reason = "requires Windows and an explicitly selected Studio test WSL distro",
)
def test_real_wsl_put_writes_only_its_destination(monkeypatch):
    monkeypatch.setattr(wsl_host, "distro_name", lambda: os.environ["UNSLOTH_TEST_WSL_DISTRO"])
    path = "/tmp/unsloth put " + uuid.uuid4().hex
    try:
        wsl_host.put(path, "first line\nsecond line\n")
        assert wsl_host.guest(["cat", path]) == "first line\nsecond line\n"
    finally:
        wsl_host.guest(["rm", "-f", "--", path])
