# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accuracy guards for the sandbox-paths note appended to the python/terminal
tool descriptions (#7242).

The note must not misdirect the model with claims that are false for the real
sandbox: (1) the sandbox is not network-isolated -- the python tool can reach an
allowlist of public hosts (github.com / huggingface.co / pypi.org), so it must
not claim it "cannot reach ... remote hosts" at all; (2) a project sandbox is a
shared per-project directory, so a new thread can inherit files from prior
threads -- the note must not claim the workdir "starts empty" or "persists only
for this conversation".
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tools import PYTHON_TOOL, TERMINAL_TOOL, _SANDBOX_PATHS_NOTE


def test_note_does_not_claim_full_network_isolation():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # The old absolute claim is false for the python tool (allowlisted egress).
    assert "cannot reach other machines or remote hosts" not in lowered
    # It should instead scope the block to private/arbitrary hosts and name the
    # allowlist so the model still fetches valid public URLs.
    assert "allowlist" in lowered
    assert "github.com" in lowered and "huggingface.co" in lowered
    assert "arbitrary" in lowered or "private" in lowered


def test_note_does_not_claim_project_sandbox_starts_empty():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # Project sandboxes are shared across threads, so these are inaccurate.
    assert "starts empty" not in lowered
    assert "persists only for this conversation" not in lowered
    # It should acknowledge that project files may already be present.
    assert "project" in lowered


def test_note_is_appended_to_both_tool_descriptions():
    assert PYTHON_TOOL["function"]["description"].endswith(_SANDBOX_PATHS_NOTE)
    assert TERMINAL_TOOL["function"]["description"].endswith(_SANDBOX_PATHS_NOTE)
