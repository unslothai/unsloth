"""setup.ps1 must report the AMD GPU on a host with exactly one AMD adapter.

`$wmiGpus = if (...) { $healthyGpus } else { $amdGpus }` unrolled a one-element branch into a bare
WMI object, and a bare WMI object has no .Count in PS 5.1, so the guard after it never fired: setup
printed "gpu none (chat-only / GGUF)" while install.ps1 had just resolved the same GPU. Setup then
expected cpu torch against the ROCm wheels the installer placed, called the venv stale and exited,
the installer rolled back, and the desktop app retried the same failure forever.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")

_RADEON = "AMD Radeon(TM) 8060S Graphics"
_RX = "AMD Radeon RX 9070 XT"


def _extract_amd_scan_block() -> str:
    """Extract setup.ps1's AMD adapter scan (the `if (-not $HasROCm)` WMI fallback block)."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    m = re.search(
        r"^    if \(-not \$HasROCm\) \{\n        try \{\n.*?^    \}\n",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "AMD adapter scan block not found in setup.ps1"
    return m.group(0)


def _build_scan_script(adapters: list[tuple[str, int]]) -> str:
    """Wrap the extracted block in a Get-CimInstance stub returning (name, error code) adapters."""
    items = ", ".join(
        f"[pscustomobject]@{{ Name = '{name}'; ConfigManagerErrorCode = {code} }}"
        for name, code in adapters
    )
    return (
        "$ErrorActionPreference = 'Stop'\n"
        "function Get-CimInstance { param([Parameter(ValueFromRemainingArguments = $true)]$Rest) "
        f"@({items}) }}\n"
        "$HasROCm = $false\n"
        "$ROCmGpuLabel = $null\n"
        "$script:ROCmGpuLabels = @()\n"
        + _extract_amd_scan_block()
        + 'Write-Output "LABEL=$ROCmGpuLabel"\n'
        'Write-Output "COUNT=$($script:ROCmGpuLabels.Count)"\n'
    )


def _run_scan(tmp_path: Path, adapters: list[tuple[str, int]]) -> str:
    script = tmp_path / "scan.ps1"
    script.write_text(_build_scan_script(adapters), encoding = "utf-8")
    proc = subprocess.run(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert proc.returncode == 0, f"scan block failed: {proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_scan_wraps_the_whole_if_in_an_array():
    """The unwrapped form is the bug, so keep it out of the source."""
    block = _extract_amd_scan_block()
    assert "$wmiGpus = @(if (" in block
    assert re.search(r"\$wmiGpus = if \(", block) is None


def test_installer_forwards_the_resolved_gfx_arch():
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    forward = src.index("if ($ROCmGfxArch) { $env:UNSLOTH_ROCM_GFX_ARCH = $ROCmGfxArch }")
    invoke = src.index("$studioArgs = @('studio', 'setup')")
    assert forward < invoke, "the arch must be exported before setup.ps1 is invoked"


@requires_pwsh
def test_single_amd_adapter_is_reported(tmp_path):
    out = _run_scan(tmp_path, [(_RADEON, 0)])
    assert f"LABEL={_RADEON}" in out
    assert "COUNT=1" in out


@requires_pwsh
def test_every_amd_adapter_is_kept_for_shadowing_inference(tmp_path):
    out = _run_scan(tmp_path, [(_RADEON, 0), (_RX, 0)])
    assert f"LABEL={_RADEON}" in out
    assert "COUNT=2" in out


@requires_pwsh
def test_a_parked_adapter_still_reports_when_it_is_the_only_one(tmp_path):
    """Error code 45 ("not connected") is routine on a muxless laptop, so do not drop the host."""
    out = _run_scan(tmp_path, [(_RX, 45)])
    assert f"LABEL={_RX}" in out
    assert "COUNT=1" in out


@requires_pwsh
def test_a_healthy_adapter_wins_over_a_parked_one(tmp_path):
    out = _run_scan(tmp_path, [(_RX, 45), (_RADEON, 0)])
    assert f"LABEL={_RADEON}" in out
    assert "COUNT=1" in out


@requires_pwsh
def test_intel_only_host_is_not_read_as_amd(tmp_path):
    out = _run_scan(tmp_path, [("Intel(R) Arc(TM) A770", 0)])
    assert "COUNT=0" in out
