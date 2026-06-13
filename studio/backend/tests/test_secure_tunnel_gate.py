# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Cloudflare tunnel start gate, incl. --secure on loopback.

`--secure` lets the tunnel start on a 127.0.0.1 bind (cloudflared reaches the
server over http://localhost:port), while non-secure keeps the historical
0.0.0.0-only behaviour. Imports run.py directly, so run under the Studio venv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from run import _cloudflare_tunnel_should_start as should_start  # noqa: E402


@pytest.mark.parametrize(
    "cloudflare,host,secure,api_only,is_colab,expected",
    [
        # Non-secure: historical 0.0.0.0-only behaviour preserved.
        (True, "0.0.0.0", False, False, False, True),
        (True, "127.0.0.1", False, False, False, False),
        (True, "localhost", False, False, False, False),
        # --secure tunnels a loopback bind too.
        (True, "127.0.0.1", True, False, False, True),
        (True, "0.0.0.0", True, False, False, True),
        # --no-cloudflare always wins.
        (False, "0.0.0.0", False, False, False, False),
        (False, "127.0.0.1", True, False, False, False),
        # api-only and Colab never tunnel.
        (True, "0.0.0.0", False, True, False, False),
        (True, "127.0.0.1", True, True, False, False),
        (True, "0.0.0.0", False, False, True, False),
        (True, "127.0.0.1", True, False, True, False),
    ],
)
def test_cloudflare_gate(cloudflare, host, secure, api_only, is_colab, expected):
    assert (
        should_start(
            cloudflare = cloudflare,
            host = host,
            secure = secure,
            api_only = api_only,
            is_colab = is_colab,
        )
        is expected
    )


def test_run_server_accepts_secure_kwarg():
    import inspect

    import run

    assert "secure" in inspect.signature(run.run_server).parameters
    assert inspect.signature(run.run_server).parameters["secure"].default is False


def test_run_server_rejects_secure_without_cloudflare():
    # Direct backend callers (not just the CLI) must reject the contradictory
    # combo before binding anything.
    import run
    with pytest.raises(SystemExit) as exc:
        run.run_server(secure = True, cloudflare = False)
    assert "A secure Cloudflare link is not allowed" in str(exc.value)


def test_failclosed_message_present_in_source():
    # The exact, user-facing fail-closed message must not drift.
    src = (_BACKEND / "run.py").read_text(encoding = "utf-8")
    assert (
        "A secure Cloudflare link is not allowed, use --not-secure which provides a 0.0.0.0 link"
        in src
    )
