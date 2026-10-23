# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the native Windows setup path honouring --no-torch."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest
from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


def _powershell_block(source: str, marker: str) -> str:
    assert marker in source, f"PowerShell marker not found: {marker!r}"
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Unclosed PowerShell block after {marker!r}")


def test_windows_direct_torch_installs_are_skipped_in_no_torch_mode():
    source = SETUP_PS1.read_text(encoding = "utf-8")
    guarded = _powershell_block(source, "if (-not $NoTorchMode) {")

    for install_path in (
        "installing PyTorch (AMD ROCm",
        "installing PyTorch (CPU-only)",
        "installing PyTorch with CUDA support",
        "installing Triton for Windows",
    ):
        assert install_path in guarded

    # The shared dependency pass installs the no-torch runtime, so it stays outside the guard.
    assert 'python "$PSScriptRoot\\install_python_stack.py"' not in guarded


def test_no_torch_value_is_normalized_before_shared_dependency_install():
    source = SETUP_PS1.read_text(encoding = "utf-8")
    parsed = source.index(
        "$NoTorchMode = $env:UNSLOTH_NO_TORCH -match '^\\s*(?i:true|1|yes|on)\\s*$'"
    )
    normalized = source.index(
        '$env:UNSLOTH_NO_TORCH = if ($NoTorchMode) { "true" } else { "false" }'
    )
    stack_install = source.index('python "$PSScriptRoot\\install_python_stack.py"')

    assert parsed < normalized < stack_install


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"setup.ps1 block not found: {pattern}"
    return match.group(0)


def _no_torch_resolution_script() -> str:
    """Extracted from setup.ps1, not reimplemented, so the test cannot drift from Get-PersistedNoTorch."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    getter = _extract(r"function Get-PersistedNoTorch \{.*?\n\}\n", source)
    setter = _extract(r"function Set-PersistedNoTorch \{.*?\n\}\n", source)
    marker = _extract(r'\$NoTorchMarker = "[^"]+"', source)
    resolution = _extract(
        r"\$NoTorchMode = \$env:UNSLOTH_NO_TORCH -match .*?"
        r'\$env:UNSLOTH_NO_TORCH = if \(\$NoTorchMode\) \{ "true" \} else \{ "false" \}',
        source,
    )
    return (
        "function substep { param($a, $b) }\n"
        f"{marker}\n{getter}\n{setter}\n{resolution}\n"
        'Write-Output "$NoTorchMode|$env:UNSLOTH_NO_TORCH"'
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("env_value", "manifest", "marker", "expected"),
    [
        # The manifest is dropped before each pass, so a killed install leaves only the marker.
        (None, None, True, "True|true"),
        (None, {}, True, "True|true"),
        (None, {"no_torch": False}, True, "False|false"),
        (None, {"no_torch": True}, False, "True|true"),
    ]
    + [
        (env_value, manifest, False, expected)
        for env_value, manifest, expected in [
            # `unsloth studio update` exports nothing, so the manifest decides.
            (None, {"no_torch": True}, "True|true"),
            (None, {"no_torch": False}, "False|false"),
            (None, {}, "False|false"),
            (None, None, "False|false"),
            (None, "{not json", "False|false"),
            ("false", {"no_torch": True}, "False|false"),
            ("1", {"no_torch": False}, "True|true"),
            ("true", None, "True|true"),
            ("yes", None, "True|true"),
            ("ON", None, "True|true"),
            (" true ", None, "True|true"),
            ("0", None, "False|false"),
            ("", {"no_torch": True}, "True|true"),
        ]
    ],
)
def test_no_torch_mode_survives_a_studio_update(tmp_path, env_value, manifest, marker, expected):
    venv_dir = tmp_path / "unsloth_studio"
    venv_dir.mkdir()
    if manifest is not None:
        payload = manifest if isinstance(manifest, str) else json.dumps(manifest)
        (venv_dir / "unsloth_install_manifest.json").write_text(payload, encoding = "utf-8")
    if marker:
        (venv_dir / ".unsloth-no-torch").write_text("", encoding = "utf-8")

    env = os.environ.copy()
    env.pop("UNSLOTH_NO_TORCH", None)
    if env_value is not None:
        env["UNSLOTH_NO_TORCH"] = env_value

    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f'$VenvDir = "{venv_dir.as_posix()}"\n{_no_torch_resolution_script()}',
        ],
        check = True,
        capture_output = True,
        text = True,
        env = env,
    )
    # install_python_stack.py drops the manifest first, so the env var is all it has.
    assert result.stdout.strip() == expected

    assert (venv_dir / ".unsloth-no-torch").exists() is expected.startswith("True")
