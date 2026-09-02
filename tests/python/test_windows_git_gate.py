# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Git is optional on the consumer Windows path, but still required for source builds."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"

_START = "$gitNeeded = ($env:STUDIO_LOCAL_INSTALL -eq '1')"
_TAIL = "if (-not $_localLlamaBuilt) {"


def _git_gate_block() -> str:
    """Slice the real $gitNeeded computation out of setup.ps1 so the test cannot drift."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    start = source.index(_START)
    brace = source.index("{", source.index(_TAIL, start))
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError("Unclosed git gate block in setup.ps1")


def _function(name: str) -> str:
    """Inject the real helper the block calls; an undefined one is a silent no-op
    under Continue, which would let the layout scan always report 'nothing built'."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    start = source.index(f"function {name} {{")
    depth = 0
    for index in range(source.index("{", start), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Unclosed function {name} in setup.ps1")


def _script() -> str:
    return f"""
$DefaultLlamaPrForce = "0"
$DefaultLlamaSource = "https://github.com/ggml-org/llama.cpp"
$DefaultLlamaTag = "latest"
{_function("Test-AccessDeniedError")}
{_function("Get-PathState")}
function Exit-PathAccessDenied {{ param($Path, $Label, [switch]$UserSupplied) throw "denied: $Path" }}
{_git_gate_block()}
Write-Output $gitNeeded
"""


def _needs_git(env: dict[str, str]) -> bool:
    merged = {k: v for k, v in os.environ.items() if not k.startswith(("UNSLOTH_", "STUDIO_"))}
    merged.update(env)
    # run_pwsh, not subprocess.run: every case in this file goes through here, so a pwsh that died at startup would come
    # back as $gitNeeded computing the wrong answer for one environment.
    # See tests/_shared/unsloth_pwsh_runner.py.
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", _script()],
        check = True,
        capture_output = True,
        text = True,
        env = merged,
    )
    return result.stdout.strip() == "True"


pwsh_only = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")


@pwsh_only
@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # The consumer install: prebuilt wheels and a prebuilt llama.cpp, so no git.
        ({}, False),
        ({"STUDIO_LOCAL_INSTALL": "1"}, True),
        ({"UNSLOTH_LLAMA_FORCE_COMPILE": "1"}, True),
        ({"UNSLOTH_LLAMA_PR": "1234"}, True),
        # PR_FORCE only forces a build for a positive integer.
        ({"UNSLOTH_LLAMA_PR_FORCE": "0"}, False),
        ({"UNSLOTH_LLAMA_PR_FORCE": "not-a-number"}, False),
        ({"UNSLOTH_LLAMA_PR_FORCE": "1234"}, True),
        # "master" is a branch with no release, so Phase 4 always builds it from source.
        ({"UNSLOTH_LLAMA_TAG": "master"}, True),
        # A release tag resolves to a prebuilt bundle.
        ({"UNSLOTH_LLAMA_TAG": "latest"}, False),
        ({"UNSLOTH_LLAMA_TAG": "b8635"}, False),
    ],
)
def test_git_is_required_only_for_local_and_source_builds(env, expected):
    assert _needs_git(env) is expected


@pwsh_only
def test_a_built_local_llama_dir_drops_the_source_build_git_requirement(tmp_path):
    (tmp_path / "llama-server.exe").write_text("", encoding = "utf-8")
    env = {
        "UNSLOTH_LOCAL_LLAMA_CPP_DIR": str(tmp_path),
        "UNSLOTH_LLAMA_FORCE_COMPILE": "1",
    }
    # Reusing an existing binary skips both the prebuilt download and the source build.
    assert _needs_git(env) is False


@pwsh_only
@pytest.mark.parametrize("trigger", ["UNSLOTH_LLAMA_FORCE_COMPILE", "UNSLOTH_LLAMA_PR"])
def test_an_unbuilt_local_llama_dir_still_requires_git(tmp_path, trigger):
    # Nothing built at the canonical install location falls through to the normal install, so the source build still
    # runs and still needs git. Suppressing the requirement here let a no-git host silently degrade to a prebuilt
    # instead.
    env = {
        "UNSLOTH_LOCAL_LLAMA_CPP_DIR": str(tmp_path),
        trigger: "1",
    }
    assert _needs_git(env) is True


@pwsh_only
def test_an_unbuilt_local_llama_dir_alone_does_not_require_git(tmp_path):
    assert _needs_git({"UNSLOTH_LOCAL_LLAMA_CPP_DIR": str(tmp_path)}) is False
