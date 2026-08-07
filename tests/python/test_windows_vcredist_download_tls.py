# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The direct VC++ runtime download must negotiate TLS 1.2 on legacy protocol defaults."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"

_START = '$url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"'
_END = "Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue\n        }"


def _download_block() -> str:
    """Slice the real download block out of setup.ps1 so the test cannot drift."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    start = source.index(_START)
    end = source.index(_END, start) + len(_END)
    return source[start:end]


def _script(starting_protocol: str) -> str:
    # Start from a non-zero set that lacks Tls12. Tls13 is the only such value modern .NET
    # accepts, and it stands in for the legacy Ssl3/Tls default of Windows PowerShell 5.1.
    return f"""
function substep {{ param($a, $b) }}
function Refresh-Environment {{ }}
function Invoke-WebRequest {{
    param($Uri, $OutFile, [switch]$UseBasicParsing, $TimeoutSec)
    Write-Output "DURING=$([System.Net.ServicePointManager]::SecurityProtocol)"
    throw "stop before Start-Process"
}}
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::{starting_protocol}
{_download_block()}
Write-Output "AFTER=$([System.Net.ServicePointManager]::SecurityProtocol)"
"""


def _run(starting_protocol: str) -> dict[str, str]:
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", _script(starting_protocol)],
        check = True,
        capture_output = True,
        text = True,
    )
    out = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


pwsh_only = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")


@pwsh_only
def test_tls12_is_added_for_the_download_and_restored_after():
    seen = _run("Tls13")
    during = {part.strip() for part in seen["DURING"].split(",")}
    assert "Tls12" in during, "the download must negotiate TLS 1.2 or aka.ms refuses it"
    assert "Tls13" in during, "adding TLS 1.2 must not drop protocols the host already allowed"
    assert seen["AFTER"] == "Tls13", "the process-wide protocol must be restored"


@pwsh_only
def test_system_default_is_left_alone():
    # SystemDefault means "let the OS choose" and already covers TLS 1.2+; pinning it to
    # Tls12 would strip TLS 1.3 from every later request in the process.
    seen = _run("SystemDefault")
    assert seen["DURING"] == "SystemDefault"
    assert seen["AFTER"] == "SystemDefault"
