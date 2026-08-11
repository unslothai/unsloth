# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""setup.ps1 must report the AMD GPU on a host with exactly one AMD adapter.

`$wmiGpus = if (...) { $healthyGpus } else { $amdGpus }` unrolled a one-element branch into a bare
WMI object, and a bare WMI object has no .Count in PS 5.1, so the guard after it never fired: setup
printed "gpu none (chat-only / GGUF)" while install.ps1 had just resolved the same GPU. Setup then
expected cpu torch against the ROCm wheels the installer placed, called the venv stale and exited,
the installer rolled back, and the desktop app retried the same failure forever.

Two notes on why these tests are shaped the way they are.

PowerShell has two member-binding paths and only the optimized one carries the PSv3 scalar
Count/Length fallback. Types with a custom adapter -- CimInstance, ManagementObject, COM,
PSCustomObject -- take the other one, which returned null in the 5.1-era engine; PowerShell/
PowerShell#5745 added the fallback there and shipped it in 6.1, which Windows PowerShell 5.1 never
received. So under the pwsh that runs these tests, `.Count` on a scalar answers 1 and the bug is
INVISIBLE. Asserting on `.Count` would pass against the unfixed source and guard nothing. Every
runtime case here therefore asserts the SHAPE of the value, which is identical on both engines, and
`ps51` re-runs the same shipped block against stubs carrying an explicit `Count = $null` to
reproduce 5.1's observable consequence rather than only its cause.

The second bug is the same expression one block down: `$gpuNames = if (...) { @(...) } else { @(...) }`
wraps each BRANCH but not the if, so a single adapter name unrolls to a bare String and
`$gpuNames[$nameIdx]` indexes the name, yielding "A". Nothing maps, and the `$nameArches[0]` rescue
is skipped under a visible-device mask, so a pinned single-GPU host inferred no arch at all. That
one does reproduce under pwsh, so it is asserted directly.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")

_RADEON = "AMD Radeon(TM) 8060S Graphics"  # Strix Halo iGPU  -> gfx1151
_RX9070 = "AMD Radeon RX 9070 XT"  # RDNA 4 discrete  -> gfx1201
_R780M = "AMD Radeon 780M Graphics"  # Phoenix iGPU     -> gfx1103, a shadowing arch
_ARC = "Intel(R) Arc(TM) A770 Graphics"

HANDOFF = "_UNSLOTH_ROCM_GFX_ARCH_HANDOFF"


# ── extracting the shipped source, so these tests exercise it rather than a copy ──────────────


def _balanced(src: str, start: int, opener: str, closer: str) -> str:
    """Slice from `start` through the delimiter that closes the first `opener` after it."""
    depth, i = 0, src.index(opener, start)
    while True:
        if src[i] == opener:
            depth += 1
        elif src[i] == closer:
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1


def _function(src: str, name: str) -> str:
    return _balanced(src, src.index(f"function {name} {{"), "{", "}")


def _setup_source() -> str:
    return SETUP_PS1.read_text(encoding = "utf-8")


# The two fixes, as (fixed, unfixed) pairs. Undoing them textually is what the red/green check
# below runs against, rather than an older commit: a revision that is "before the fix" is only
# before it until this merges, and reaching for it through git also fails in a shallow CI clone.
# Reverting the wraps in memory keeps the comparison immutable and isolates the one thing under
# test, since everything else about the two sources is identical by construction.
_ARRAY_WRAPS = (
    (
        "$wmiGpus = @(if ($healthyGpus.Count -gt 0) { $healthyGpus } else { $amdGpus })",
        "$wmiGpus = if ($healthyGpus.Count -gt 0) { $healthyGpus } else { $amdGpus }",
    ),
    (
        "$gpuNames = @(if ($script:ROCmGpuLabels) { @($script:ROCmGpuLabels) } else { @($ROCmGpuLabel) })",
        "$gpuNames = if ($script:ROCmGpuLabels) { @($script:ROCmGpuLabels) } else { @($ROCmGpuLabel) }",
    ),
)


def _without_the_array_wraps(src: str) -> str:
    """setup.ps1 with only the two `@()` wraps undone: the source as it behaved before the fix."""
    for fixed, unfixed in _ARRAY_WRAPS:
        assert src.count(fixed) == 1, f"expected exactly one occurrence of {fixed!r}"
        src = src.replace(fixed, unfixed)
    return src


def _amd_scan_block(src: str) -> str:
    """The `if (-not $HasROCm)` WMI fallback: the adapter list the label is read from."""
    m = re.search(
        r"^    if \(-not \$HasROCm\) \{\n        try \{\n.*?^    \}\n",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "AMD adapter scan block not found in setup.ps1"
    return m.group(0)


def _arch_resolution_block(src: str) -> str:
    """Everything from the arch-resolution header up to the ROCm version capture that follows."""
    start = src.index("    # ── Arch resolution:")
    end = src.index("    # Capture ROCm version early", start)
    return src[start:end]


def _prelude(src: str) -> str:
    """The declarations and helpers the two blocks close over."""
    shadowing = _balanced(src, src.index("$script:ShadowingIntegratedGfx = @("), "(", ")")
    arch_family = _balanced(src, src.index("$archFamilyMap = @{"), "{", "}")
    return "\n".join(
        [
            "$script:ShadowingIntegratedGfx = " + shadowing[shadowing.index("@(") :],
            "$archFamilyMap = " + arch_family[arch_family.index("@{") :],
            _function(src, "Test-VisibleDevicesPinned"),
            _function(src, "Resolve-VisibleGpuIndex"),
            _function(src, "Resolve-ShadowingGfxPick"),
        ]
    )


# ── the driver: run the shipped blocks against a stubbed adapter list ─────────────────────────


def _driver(
    src: str,
    adapters: list[tuple[str, int]],
    *,
    ps51: bool = False,
    strict: bool = False,
) -> str:
    """Wrap the shipped blocks in a Get-CimInstance stub and report the result as JSON.

    ps51 gives every stub adapter an explicit `Count = $null`, which is what a bare CimInstance
    answers on Windows PowerShell 5.1 and what pwsh would otherwise paper over with 1.
    """
    count_member = "; Count = $null" if ps51 else ""
    items = ", ".join(
        f"[pscustomobject]@{{ Name = '{name}'; ConfigManagerErrorCode = {code}{count_member} }}"
        for name, code in adapters
    )
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "Set-StrictMode -Version Latest" if strict else "Set-StrictMode -Off",
            f"function Get-CimInstance {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) @({items}) }}",
            "function substep { param($a, $b) }",
            "$HasROCm = $false",
            "$ROCmGpuLabel = $null",
            "$script:ROCmGpuLabels = @()",
            "$script:ROCmGfxArch = $null",
            "$script:GpuNamesProbe = $null",
            "$wmiGpus = $null",
            _prelude(src),
            _amd_scan_block(src),
            # Captured from inside the arch block's own scope: $gpuNames is the value the indexing
            # bug corrupts, and its first element is what Get-GfxArchFromGpuName is actually handed.
            _arch_resolution_block(src).replace(
                "$nameIdx = Resolve-VisibleGpuIndex $gpuNames.Count",
                "$script:GpuNamesProbe = $gpuNames\n            $nameIdx = Resolve-VisibleGpuIndex $gpuNames.Count",
            ),
            # ConvertTo-Json, not string interpolation: a null stays null instead of becoming "".
            "@{",
            "  wmi_type    = $(if ($null -ne $wmiGpus) { $wmiGpus.GetType().FullName } else { $null })",
            "  wmi_array   = [bool]($wmiGpus -is [array])",
            "  label       = $ROCmGpuLabel",
            "  labels      = @($script:ROCmGpuLabels)",
            "  arch        = $script:ROCmGfxArch",
            "  names_type  = $(if ($null -ne $script:GpuNamesProbe) { $script:GpuNamesProbe.GetType().FullName } else { $null })",
            "  names_first = $(if ($null -ne $script:GpuNamesProbe) { $script:GpuNamesProbe[0] } else { $null })",
            "} | ConvertTo-Json -Compress",
        ]
    )


def _run(
    tmp_path: Path,
    adapters: list[tuple[str, int]],
    *,
    env: dict[str, str] | None = None,
    source: str | None = None,
    ps51: bool = False,
    strict: bool = False,
) -> dict:
    script = tmp_path / "scan.ps1"
    script.write_text(
        _driver(source or _setup_source(), adapters, ps51 = ps51, strict = strict), encoding = "utf-8"
    )
    # Only what each case names may reach the child: a developer's own exported
    # UNSLOTH_ROCM_GFX_ARCH would otherwise silently win every inference assertion here.
    child_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    child_env.update(env or {})
    proc = subprocess.run(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
        env = child_env,
    )
    assert proc.returncode == 0, f"scan block failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout)


# ── source assertions ─────────────────────────────────────────────────────────────────────────


def test_scan_wraps_the_whole_if_in_an_array():
    """The unwrapped form is the bug, so keep it out of the source."""
    block = _amd_scan_block(_setup_source())
    assert "$wmiGpus = @(if (" in block
    assert re.search(r"\$wmiGpus = if \(", block) is None


def test_gpu_name_list_wraps_the_whole_if_in_an_array():
    """Same expression, one block down: wrapping each branch is not enough."""
    block = _arch_resolution_block(_setup_source())
    assert "$gpuNames = @(if (" in block
    assert re.search(r"\$gpuNames = if \(", block) is None


def test_installer_forwards_the_arch_through_a_private_handoff():
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    forward = src.index(f"$env:{HANDOFF} = $ROCmGfxArch")
    invoke = src.index("& $UnslothExe @studioArgs")
    assert forward < invoke, "the arch must be handed over before setup.ps1 is invoked"


def test_installer_never_exports_the_public_override():
    """UNSLOTH_ROCM_GFX_ARCH is the operator's to set.

    install_llama_prebuilt.py reads it back as _manual to decide whether a forwarded --rocm-gfx
    outranks its own probe, so publishing an auto-detected arch there disarms that safeguard on
    exactly the multi-GPU hosts it exists for.
    """
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert re.search(r"\$env:UNSLOTH_ROCM_GFX_ARCH\s*=", src) is None


def test_installer_restores_the_private_handoff_after_setup():
    """Matching the save/restore every adjacent handoff variable already does.

    install.ps1 is documented as `irm ... | iex`, so it runs in the caller's own process and a
    value left behind would be read as an override by the next install in that terminal.
    """
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert f"$previousRocmGfxHandoff = $env:{HANDOFF}" in src
    assert f"$env:{HANDOFF} = $previousRocmGfxHandoff" in src
    assert f"Remove-Item Env:{HANDOFF} -ErrorAction SilentlyContinue" in src
    # Saved after the last early return, so no path skips the restore.
    assert src.index("$previousRocmGfxHandoff") > src.index(
        "--with-llama-cpp-dir path does not exist"
    )


def test_setup_consumes_the_handoff_only_after_its_own_inference():
    """Order is the whole point: the installer takes the first AMD adapter with no mask and no
    shadowing repick, both of which setup applies, so the handoff must never pre-empt it."""
    block = _arch_resolution_block(_setup_source())
    assert block.index("gfx arch inferred from GPU name") < block.index(f"$env:{HANDOFF}")


# ── runtime: the adapter scan ─────────────────────────────────────────────────────────────────


@requires_pwsh
@pytest.mark.parametrize("ps51", [False, True], ids = ["pwsh", "ps51"])
@pytest.mark.parametrize("strict", [False, True], ids = ["lax", "strict"])
def test_single_amd_adapter_is_reported(tmp_path, ps51, strict):
    out = _run(tmp_path, [(_RADEON, 0)], ps51 = ps51, strict = strict)
    assert out["wmi_array"], f"one adapter must stay an array, got {out['wmi_type']}"
    assert out["labels"] == [_RADEON]
    # The name reached the inference, which is what "reported" has to mean here: a label alone
    # still lands on the "AMD ROCm" branch with no arch and installs cpu torch.
    assert out["arch"] == "gfx1151"
    assert out["label"] == "AMD ROCm (gfx1151)"


@requires_pwsh
def test_every_amd_adapter_is_kept_for_shadowing_inference(tmp_path):
    out = _run(tmp_path, [(_R780M, 0), (_RX9070, 0)])
    assert out["labels"] == [_R780M, _RX9070]


@requires_pwsh
def test_a_parked_adapter_still_reports_when_it_is_the_only_one(tmp_path):
    """Error code 45 ("not connected") is routine on a muxless laptop, so do not drop the host."""
    out = _run(tmp_path, [(_RX9070, 45)])
    assert out["wmi_array"]
    assert out["labels"] == [_RX9070]


@requires_pwsh
def test_a_healthy_adapter_wins_over_a_parked_one(tmp_path):
    out = _run(tmp_path, [(_RX9070, 45), (_RADEON, 0)])
    assert out["labels"] == [_RADEON]


@requires_pwsh
@pytest.mark.parametrize("adapters", [[], [(_ARC, 0)]], ids = ["no_adapters", "intel_only"])
def test_a_host_with_no_amd_adapter_is_not_read_as_amd(tmp_path, adapters):
    out = _run(tmp_path, adapters)
    assert out["labels"] == []
    assert out["label"] is None
    assert out["arch"] is None


# ── runtime: name inference, where the second unwrapped if bites ──────────────────────────────


@requires_pwsh
def test_the_adapter_name_reaches_inference_whole(tmp_path):
    """Unwrapped, $gpuNames is a String and $gpuNames[0] is the character "A"."""
    out = _run(tmp_path, [(_RADEON, 0)])
    assert out["names_first"] == _RADEON, "the name was indexed as a string, not as a list"
    assert "Object[]" in (out["names_type"] or "")


@requires_pwsh
@pytest.mark.parametrize(
    "name, expected",
    [
        (_RADEON, "gfx1151"),
        (_RX9070, "gfx1201"),
        ("AMD Radeon RX 9060 XT", "gfx1200"),
        ("AMD Radeon 890M Graphics", "gfx1150"),
        ("AMD Radeon 860M Graphics", "gfx1152"),
        ("AMD Radeon RX 7900 XTX", "gfx1100"),
        ("AMD Radeon RX 7600", "gfx1102"),
        (_R780M, "gfx1103"),
        ("AMD Radeon RX 6800 XT", "gfx1030"),
        ("AMD Radeon RX 6500 XT", "gfx1034"),
        ("AMD Radeon HD 8570", None),
    ],
)
def test_a_single_adapter_infers_its_arch(tmp_path, name, expected):
    assert _run(tmp_path, [(name, 0)])["arch"] == expected


@requires_pwsh
@pytest.mark.parametrize("mask", ["0", "1", "", "-1", "not-a-number", "9", " 0 ", "0,1"])
@pytest.mark.parametrize(
    "var", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
def test_a_pinned_single_gpu_host_still_infers_its_arch(tmp_path, mask, var):
    """The mask disables the $nameArches[0] rescue, so the string-indexing bug surfaced here as a
    host with a perfectly good Radeon reporting no arch and looping the installer."""
    assert _run(tmp_path, [(_RADEON, 0)], env = {var: mask})["arch"] == "gfx1151"


@requires_pwsh
def test_a_discrete_card_is_preferred_over_a_shadowing_igpu(tmp_path):
    out = _run(tmp_path, [(_R780M, 0), (_RX9070, 0)])
    assert out["arch"] == "gfx1201", "the gfx1103 iGPU shadowed the discrete card"


@requires_pwsh
def test_a_pinned_mask_is_honoured_over_the_shadowing_preference(tmp_path):
    out = _run(tmp_path, [(_R780M, 0), (_RX9070, 0)], env = {"HIP_VISIBLE_DEVICES": "0"})
    assert out["arch"] == "gfx1103", "an explicit selection must never be repicked"


# ── runtime: handoff precedence ───────────────────────────────────────────────────────────────


@requires_pwsh
def test_the_handoff_fills_the_gap_when_nothing_else_resolves(tmp_path):
    """The case the handoff exists for: setup's own scan came up empty where the installer's did
    not, and without this the two disagree and the install rolls back."""
    out = _run(tmp_path, [], env = {HANDOFF: "gfx1151"})
    assert out["arch"] == "gfx1151"
    assert out["label"] == "AMD ROCm (gfx1151)"


@requires_pwsh
def test_the_handoff_never_deposes_setups_own_inference(tmp_path):
    """install.ps1 would forward the iGPU here: it takes the first AMD adapter, with no shadowing
    repick. Setup's answer is the better one and has to win."""
    out = _run(tmp_path, [(_R780M, 0), (_RX9070, 0)], env = {HANDOFF: "gfx1103"})
    assert out["arch"] == "gfx1201"


@requires_pwsh
def test_a_user_override_still_wins_over_the_handoff(tmp_path):
    out = _run(
        tmp_path,
        [(_R780M, 0)],
        env = {"UNSLOTH_ROCM_GFX_ARCH": "gfx90a", HANDOFF: "gfx1103"},
    )
    assert out["arch"] == "gfx90a", "the documented operator override outranks an inferred value"


@requires_pwsh
@pytest.mark.parametrize("value", ["GFX1151", "  gfx1151  "])
def test_the_handoff_is_normalized_like_the_override(tmp_path, value):
    assert _run(tmp_path, [], env = {HANDOFF: value})["arch"] == "gfx1151"


@requires_pwsh
def test_an_empty_handoff_is_ignored(tmp_path):
    assert _run(tmp_path, [], env = {HANDOFF: ""})["arch"] is None


# ── runtime: install.ps1 leaves the caller's environment as it found it ───────────────────────


def _handoff_lifecycle_block() -> str:
    """install.ps1's save / set / try / finally around the setup call, as shipped."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    start = src.index("    $previousRocmGfxHandoff = $env:")
    end = src.index("    if ($setupExit -ne 0) {", start)
    return src[start:end]


def _run_handoff_lifecycle(
    tmp_path: Path, *, arch: str | None, inherited: str | None, fails: bool
) -> dict:
    body = _handoff_lifecycle_block().replace(
        "& $UnslothExe @studioArgs",
        "throw 'setup exploded'"
        if fails
        else "$script:SeenByChild = $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF",
    )
    script = tmp_path / "handoff.ps1"
    script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                # The neighbouring handoffs the shipped finally also restores; not under test here, but
                # the block does not compile without them.
                "$previousUnslothStudioHome = $null; $hadPreviousUnslothStudioHome = $false",
                "$previousTauriMode = $null; $hadPreviousTauriMode = $false",
                "$previousSetupRuntimeGateHandoff = $null; $hadPreviousSetupRuntimeGateHandoff = $false",
                "$previousProxyHandoff = $null; $hadPreviousProxyHandoff = $false",
                "$UnslothProxyHandoffJson = $null",
                "$UnslothExe = 'stub'; $studioArgs = @(); $setupExit = 0",
                "$script:SeenByChild = '<never ran>'",
                "$ROCmGfxArch = " + ("$null" if arch is None else f"'{arch}'"),
                "try {",
                body,
                "} catch { }",
                "@{",
                "  seen_by_child = $script:SeenByChild",
                "  after = $(if (Test-Path Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF) { $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF } else { $null })",
                "  after_set = [bool](Test-Path Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF)",
                "  public = $(if (Test-Path Env:UNSLOTH_ROCM_GFX_ARCH) { $env:UNSLOTH_ROCM_GFX_ARCH } else { $null })",
                "} | ConvertTo-Json -Compress",
            ]
        ),
        encoding = "utf-8",
    )
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "UNSLOTH_ROCM_GFX_ARCH": "gfx90a"}
    if inherited is not None:
        env[HANDOFF] = inherited
    proc = subprocess.run(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
        env = env,
    )
    assert proc.returncode == 0, f"handoff block failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout)


@requires_pwsh
@pytest.mark.parametrize("fails", [False, True], ids = ["setup_ok", "setup_throws"])
@pytest.mark.parametrize(
    "arch, inherited",
    [(None, None), ("gfx1151", None), (None, "gfx1030"), ("gfx1151", "gfx1030")],
    ids = ["nothing", "resolved", "inherited", "resolved_over_inherited"],
)
def test_the_caller_environment_survives_the_setup_call(tmp_path, arch, inherited, fails):
    """`irm ... | iex` runs install.ps1 in the caller's own shell, so anything set for the child
    has to be put back -- on the failure path too, which is the one that rolls back and retries."""
    out = _run_handoff_lifecycle(tmp_path, arch = arch, inherited = inherited, fails = fails)
    assert out["after_set"] is (inherited is not None), "the handoff outlived the setup call"
    assert out["after"] == inherited
    assert out["public"] == "gfx90a", "a user's own override must come back untouched"


@requires_pwsh
@pytest.mark.parametrize(
    "arch, inherited, expected",
    [("gfx1151", None, "gfx1151"), ("gfx1151", "gfx1030", "gfx1151"), (None, "gfx1030", None)],
    ids = ["resolved", "resolved_over_inherited", "stale_only"],
)
def test_only_this_runs_arch_is_handed_to_the_child(tmp_path, arch, inherited, expected):
    """A value inherited from an outer process is not this run's answer, so it is cleared rather
    than forwarded as though the scan had produced it."""
    out = _run_handoff_lifecycle(tmp_path, arch = arch, inherited = inherited, fails = False)
    assert out["seen_by_child"] == expected


# ── the guard on the guards ───────────────────────────────────────────────────────────────────


@requires_pwsh
def test_these_assertions_fail_without_the_array_wraps(tmp_path):
    """A regression test that passes on the unfixed source is not one.

    The first version of this file asserted on `.Count`, which pwsh answers as 1 for a scalar, so
    it was green either way. Undo just the two wraps and confirm the failures come back, so a
    later edit here cannot quietly re-lose them.
    """
    unfixed = _without_the_array_wraps(_setup_source())
    before = _run(tmp_path, [(_RADEON, 0)], source = unfixed, ps51 = True)
    pinned = _run(tmp_path, [(_RADEON, 0)], source = unfixed, env = {"HIP_VISIBLE_DEVICES": "0"})
    assert not before["wmi_array"], "the unwrapped scan should collapse to a scalar"
    assert before["label"] is None, "the unwrapped scan should report no GPU under 5.1 semantics"
    assert pinned["arch"] is None, "the unwrapped name list should infer nothing when pinned"
