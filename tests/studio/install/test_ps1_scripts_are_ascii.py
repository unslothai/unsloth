# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The PowerShell installers stay ASCII-only: Windows hashes a BOM-less signed script in the
machine's ANSI code page, so any non-ASCII byte means HashMismatch outside Windows-1252 (#19908)."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ("install.ps1", "studio/setup.ps1", "scripts/uninstall.ps1")


@pytest.mark.parametrize("relpath", SCRIPTS)
def test_script_is_ascii_only(relpath):
    data = (REPO_ROOT / relpath).read_bytes()
    offenders = []
    for lineno, line in enumerate(data.split(b"\n"), 1):
        if any(byte > 0x7F for byte in line):
            offenders.append(f"{relpath}:{lineno}: {line.decode('utf-8', 'replace').strip()[:80]}")
    assert not offenders, "non-ASCII text (use '-', '->' or plain words):\n" + "\n".join(
        offenders[:20]
    )
