# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from utils.host_policy import is_wildcard_host


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "::0",
        "0:0:0:0:0:0:0:0",
        "0",
        "00",
        "0.0",
        "0.0.0",
    ],
)
def test_unspecified_bind_aliases_are_wildcards(host):
    assert is_wildcard_host(host) is True


@pytest.mark.parametrize(
    "host", ["", "127.0.0.1", "localhost", "::1", "192.168.1.24", "fd00::5"]
)
def test_specific_bind_hosts_are_not_wildcards(host):
    assert is_wildcard_host(host) is False


def test_run_server_rejects_an_empty_bind_before_startup():
    from run import run_server

    with pytest.raises(SystemExit, match = "--host cannot be empty"):
        run_server(host = "")
