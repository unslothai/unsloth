# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""setup.ps1 must report the AMD GPU on a host with exactly one AMD adapter.

`$wmiGpus = if (...) { $healthyGpus } else { $amdGpus }` unrolled a one-element branch into a bare
WMI object, which has no .Count in PS 5.1, so the guard after it never fired: setup printed "gpu
none (chat-only / GGUF)" while install.ps1 had just resolved the same GPU, then expected cpu torch
against the ROCm wheels the installer placed, called the venv stale and exited, the installer rolled
back, and the desktop app retried forever. Same expression one block down, `$gpuNames`, wraps each
BRANCH but not the if, so a lone adapter name unrolls to a String and `$gpuNames[$nameIdx]` yields
"A"; the `$nameArches[0]` rescue hides that unless a visible-device mask is set.

Why the assertions look the way they do: only PowerShell's optimized member-binding path carries the
PSv3 scalar Count fallback, and custom-adapter types (CimInstance, ManagementObject, COM,
PSCustomObject) take the other one, which returned null until PowerShell/PowerShell#5745 shipped in
6.1 -- never backported to 5.1. So under pwsh `.Count` answers 1 and the bug is INVISIBLE; asserting
on it would pass against the unfixed source. Every runtime case asserts the SHAPE of the value,
which is identical on both engines, and `ps51` re-runs the same block against stubs carrying an
explicit `Count = $null` to reproduce 5.1's consequence rather than only its cause.
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
_R9700 = "AMD Radeon AI PRO R9700"  # RDNA 4 workstation -> gfx1201 (#7624, #7307)
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


# The two fixes, as (fixed, unfixed) pairs.
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
    """The `if (-not $HasROCm)` WMI fallback: the adapter list the label is read from.
    The report-only peer scan added for #8529 is a SEPARATE block and is deliberately not
    covered here: it feeds no label and no arch."""
    m = re.search(
        r"^    if \(-not \$HasROCm\) \{\n        try \{\n.*?^    \}\n",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "AMD adapter scan block not found in setup.ps1"
    return m.group(0)


def _arch_resolution_block(src: str) -> str:
    """Everything from the arch-resolution guard up to the hipconfig probe that follows.

    Anchored on CODE at both ends for the same reason as _installer_scan_block: the two
    comments this used to key on are exactly the kind a comment pass rewrites, and the
    failure it produces is a ValueError rather than an assertion that says anything."""
    marker = src.index("$script:ROCmUnsupportedGfxArch = $null")
    start = src.index("    if (-not $script:ROCmGfxArch) {", marker)
    end = src.index("    if ($HasROCm -or $HipSdkInstalled) {", start)
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
            # Captured from inside the arch block's own scope: $gpuNames is the value the indexing bug corrupts, and
            # its first element is what Get-GfxArchFromGpuName is actually handed.
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
    # Only what each case names may reach the child: a developer's own exported UNSLOTH_ROCM_GFX_ARCH would otherwise
    # silently win every inference assertion here.
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
    invoke = src.index("Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs")
    assert forward < invoke, "the arch must be handed over before setup.ps1 is invoked"


def test_installer_never_exports_the_public_override():
    """install_llama_prebuilt.py reads UNSLOTH_ROCM_GFX_ARCH back as _manual to decide whether a
    forwarded --rocm-gfx outranks its own probe, so publishing an auto-detected arch there disarms
    that safeguard on exactly the multi-GPU hosts it exists for."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert re.search(r"\$env:UNSLOTH_ROCM_GFX_ARCH\s*=", src) is None


def test_installer_restores_the_private_handoff_after_setup():
    """install.ps1 is documented as `irm ... | iex`, so it runs in the caller's own process and a
    value left behind would be read as an override by the next install in that terminal."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert f"$previousRocmGfxHandoff = $env:{HANDOFF}" in src
    assert f"$env:{HANDOFF} = $previousRocmGfxHandoff" in src
    assert f"Remove-Item Env:{HANDOFF} -ErrorAction SilentlyContinue" in src
    # Cheap companion to test_the_bail_restores_the_caller_environment, which drives the
    # bail: no path out of the setup call may skip the restore, and the bail now lives
    # inside the restoring try, so assert the arrangement save < bail < restore.
    saved = src.index("$previousRocmGfxHandoff = $env:")
    bail = src.index("--with-llama-cpp-dir path does not exist")
    restored = src.index(f"$env:{HANDOFF} = $previousRocmGfxHandoff")
    assert saved < bail < restored, (saved, bail, restored)


def test_setup_consumes_the_handoff_only_after_its_own_inference():
    """Order is the whole point: the installer takes the first AMD adapter with no mask and no
    shadowing repick, both of which setup applies, so the handoff must never pre-empt it."""
    block = _arch_resolution_block(_setup_source())
    assert block.index("gfx arch inferred from GPU name") < block.index(f"$env:{HANDOFF}")


@requires_pwsh
@pytest.mark.parametrize("ps51", [False, True], ids = ["pwsh", "ps51"])
@pytest.mark.parametrize("strict", [False, True], ids = ["lax", "strict"])
def test_single_amd_adapter_is_reported(tmp_path, ps51, strict):
    out = _run(tmp_path, [(_RADEON, 0)], ps51 = ps51, strict = strict)
    assert out["wmi_array"], f"one adapter must stay an array, got {out['wmi_type']}"
    assert out["labels"] == [_RADEON]
    # A label alone still lands on the "AMD ROCm" branch with no arch and installs cpu torch, so "reported" has to mean
    # the name reached the inference.
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
        (_R9700, "gfx1201"),
        ("ATI Radeon 9700 PRO", None),
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


@requires_pwsh
@pytest.mark.parametrize(
    "var", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
def test_a_mask_suppresses_the_handoff(tmp_path, var):
    """The mask selects the unrecognized discrete card, so the inference above deliberately
    resolves nothing rather than borrowing the 780M's arch. The installer scans without the mask
    and forwards that very arch, and taking it would install for a GPU the mask hides from the
    runtime entirely (ROCR filters below HIP, so masked devices never reach enumeration)."""
    adapters = [(_R780M, 0), ("AMD Radeon RX 5700 XT", 0)]
    assert _run(tmp_path, adapters, env = {var: "1", HANDOFF: "gfx1103"})["arch"] is None


@requires_pwsh
def test_a_mask_suppresses_the_handoff_even_with_no_adapters_to_check_it_against(tmp_path):
    """Setup saw no names at all, so it cannot confirm the forwarded arch is the selected device.
    Refuse rather than guess."""
    assert _run(tmp_path, [], env = {"HIP_VISIBLE_DEVICES": "0", HANDOFF: "gfx1151"})["arch"] is None


@requires_pwsh
def test_a_user_override_is_the_escape_hatch_under_a_mask(tmp_path):
    out = _run(
        tmp_path,
        [(_R780M, 0), ("AMD Radeon RX 5700 XT", 0)],
        env = {"HIP_VISIBLE_DEVICES": "1", "UNSLOTH_ROCM_GFX_ARCH": "gfx1010", HANDOFF: "gfx1103"},
    )
    assert out["arch"] == "gfx1010"


# ── runtime: install.ps1 picks the adapter setup would keep ───────────────────────────────────


def _installer_scan_block() -> str:
    """install.ps1's own `if (-not $HasROCm)` WMI fallback plus the name table it feeds.
    The report-only peer scan added for #8529 is a SEPARATE block, deliberately outside
    this one: it feeds no label and no arch."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    # Anchored on CODE at both ends. The end anchor used to be a comment and a comment
    # pass deleted it, which turned four tests into ValueError instead of a failure
    # that said anything. The next statement after the block is the hipconfig probe.
    body = src.index("$amdAdapters = @(Get-CimInstance Win32_VideoController")
    start = src.rindex("        if (-not $HasROCm) {", 0, body)
    end = src.index("        if ($HasROCm -or $HipSdkInstalled) {", start)
    return src[start:end]


def _run_installer_scan(tmp_path: Path, adapters: list[tuple[str, int]]) -> dict:
    items = ", ".join(
        f"[pscustomobject]@{{ Name = '{name}'; ConfigManagerErrorCode = {code} }}"
        for name, code in adapters
    )
    script = tmp_path / "installer_scan.ps1"
    script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                # Both names:
                # Both names: the routing scan asks WMI (unchanged by #8529), the report-only peer scan asks CIM.
                f"function Get-CimInstance {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) @({items}) }}",
                f"function Get-WmiObject {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) @({items}) }}",
                "function substep { param($a, $b) }",
                "$HasROCm = $false",
                "$ROCmGpuLabel = $null",
                "$ROCmGfxArch = $null",
                _installer_scan_block(),
                "@{ label = $ROCmGpuLabel; arch = $ROCmGfxArch } | ConvertTo-Json -Compress",
            ]
        ),
        encoding = "utf-8",
    )
    proc = subprocess.run(
        [shutil.which("pwsh") or "pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, f"installer scan failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout)


@requires_pwsh
def test_the_installer_skips_a_disabled_adapter(tmp_path):
    """A disabled Radeon listed first used to win here, and a mapped arch installs ROCm wheels
    right there, so the dead card got the wheels and the live unsupported one went to nothing.
    Setup filters this adapter out, so forwarding its arch also made the two disagree."""
    out = _run_installer_scan(
        tmp_path, [("AMD Radeon RX 9070 XT", 22), ("AMD Radeon RX 5700 XT", 0)]
    )
    assert out["label"] == "AMD Radeon RX 5700 XT"
    assert out["arch"] is None, "the active card is unsupported, so this host belongs on CPU"


@requires_pwsh
def test_the_installer_keeps_a_parked_adapter_when_it_is_the_only_one(tmp_path):
    """Same fallback setup makes: code 45 is routine on a muxless laptop, and with no healthy
    peer there is nothing to prefer."""
    out = _run_installer_scan(tmp_path, [("AMD Radeon RX 9070 XT", 45)])
    assert out["arch"] == "gfx1201"


@requires_pwsh
def test_a_lone_r9700_is_detected_by_both_scans(tmp_path):
    """Reported on PR #8398: single R9700 on Windows 11, not detected. One healthy adapter
    and no HIP SDK, so the name is the only evidence left, and it holds neither "9070" nor
    "9080". Both scans must reach gfx1201 (#7624, #7307) or the install lands on CPU torch."""
    assert _run(tmp_path, [(_R9700, 0)])["arch"] == "gfx1201"
    assert _run_installer_scan(tmp_path, [(_R9700, 0)])["arch"] == "gfx1201"


@requires_pwsh
def test_the_installer_and_setup_agree_on_which_adapter_is_active(tmp_path):
    """The handoff is only sound because both scans start from the same healthy set."""
    adapters = [("AMD Radeon RX 9070 XT", 22), (_RADEON, 0)]
    setup = _run(tmp_path, adapters)
    assert setup["labels"] == [_RADEON], "setup discards the disabled card"
    assert _run_installer_scan(tmp_path, adapters)["arch"] == setup["arch"] == "gfx1151"


# ── runtime: install.ps1 leaves the caller's environment as it found it ───────────────────────


# The block's fifteen save/restore pairs bar the ROCm handoff, which the arch parametrisation drives.
_CALLER_ENV_NAMES = (
    "SKIP_STUDIO_BASE",
    "UNSLOTH_STUDIO_HOME",
    "UNSLOTH_TAURI_MODE",
    "_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF",
    "_UNSLOTH_PS_PROXY_DEFAULTS",
    "STUDIO_PACKAGE_NAME",
    "UNSLOTH_NO_TORCH",
    "UNSLOTH_INSTALLER_TORCH_TAG",
    "SKIP_STUDIO_FRONTEND",
    "STUDIO_LOCAL_INSTALL",
    "STUDIO_LOCAL_REPO",
    "UNSLOTH_LOCAL_LLAMA_CPP_DIR",
    "UNSLOTH_INSTALL_ROLLBACK_MANAGED",
    "UNSLOTH_SETUP_PYTHON",
)

# Distinct per variable, or a finally restoring everything from the wrong save reads as a pass.
_CALLER_ENV = tuple(
    (name, "outer-" + name.strip("_").lower().replace("_", "-")) for name in _CALLER_ENV_NAMES
)
assert len({sentinel for _, sentinel in _CALLER_ENV}) == len(_CALLER_ENV), "sentinels must differ"

# Which variables the caller already has. Presence has to vary PER variable: all-present and
# all-absent give every $hadPrevious* flag the same value, so a wrong-flag restore passes by luck.
_PRESENCE_MASK_BITS = max((len(_CALLER_ENV_NAMES) - 1).bit_length(), 1)
_PRESENCE_PATTERNS = {
    "all": lambda i: True,
    "none": lambda i: False,
    **{f"mask{k}": (lambda i, k = k: bool((i >> k) & 1)) for k in range(_PRESENCE_MASK_BITS)},
    **{f"cmask{k}": (lambda i, k = k: not ((i >> k) & 1)) for k in range(_PRESENCE_MASK_BITS)},
}


def _present_names(pattern: str) -> tuple[str, ...]:
    keep = _PRESENCE_PATTERNS[pattern]
    return tuple(name for i, name in enumerate(_CALLER_ENV_NAMES) if keep(i))


def test_the_presence_masks_separate_every_ordered_pair():
    """ORDERED, not unordered: a restore of A reading B's $hadPrevious flag only shows where A is
    present and B is absent, since the other direction assigns A's saved $null, which removes the
    variable exactly as the correct code does. Under the unordered form 43 of the 182 directed
    substitutions survived."""
    patterns = [_present_names(p) for p in _PRESENCE_PATTERNS]
    for a in _CALLER_ENV_NAMES:
        for b in _CALLER_ENV_NAMES:
            if a == b:
                continue
            assert any(a in p and b not in p for p in patterns), (
                f"no pattern has {a} present while {b} is absent, so a restore of {a} reading "
                f"{b}'s $hadPrevious flag is invisible"
            )


def _assert_caller_env_restored(out: dict, present: tuple[str, ...], what: str) -> None:
    """The caller's shell as it was: same value, or still no variable at all.

    Absence is `Test-Path Env:NAME` being false, not an empty value: 7.5+ keeps a variable present
    when assigned "", so a restore writing "" instead of removing would pass a value check.
    Present-but-EMPTY is unasserted: `$null -ne $previous` cannot tell "" from unset, so the answer
    is engine-dependent."""
    for name, sentinel in _CALLER_ENV:
        key = name.lower()
        if name in present:
            assert out[key + "_set"] is True, f"{what} removed {name}, which the caller had set"
            assert out[key] == sentinel, f"{what} left {name} as install.ps1 set it"
        else:
            assert (
                out[key + "_set"] is False
            ), f"{what} left {name} behind in a shell that never had it, as {out[key]!r}"


def _existing_llama_dir(tmp_path: Path) -> Path:
    path = tmp_path / "llama.cpp"
    path.mkdir(exist_ok = True)
    return path


def _studio_home_dir(tmp_path: Path) -> Path:
    path = tmp_path / "studio-home"
    path.mkdir(exist_ok = True)
    return path


def _studio_repo_dir(tmp_path: Path) -> Path:
    """DISTINCT from the studio home, so a restore that swapped the two shows rather than passing."""
    path = tmp_path / "studio-local-repo"
    path.mkdir(exist_ok = True)
    return path


# UNSLOTH_STUDIO_HOME and STUDIO_LOCAL_REPO are the only two the try does not always assign, so
# only `env_redirect_local` makes their else-arm restores load-bearing.
_STUDIO_MODES = ("default", "env_redirect_local")


def _studio_mode_lines(tmp_path: Path, studio_mode: str) -> list[str]:
    if studio_mode == "default":
        return [
            "$StudioLocalInstall = $false; $RepoRoot = $null",
            "$StudioRedirectMode = 'none'; $StudioHome = $null",
        ]
    return [
        f"$StudioLocalInstall = $true; $RepoRoot = '{_studio_repo_dir(tmp_path)}'",
        f"$StudioRedirectMode = 'env'; $StudioHome = '{_studio_home_dir(tmp_path)}'",
    ]


def _caller_env_report() -> str:
    return "\n".join(
        f"  {name.lower()} = $(if (Test-Path Env:{name}) {{ $env:{name} }} else {{ $null }})\n"
        f"  {name.lower()}_set = [bool](Test-Path Env:{name})"
        for name, _ in _CALLER_ENV
    )


def _handoff_lifecycle_block() -> str:
    """install.ps1's save / set / try / finally around the setup call, as shipped.

    Anchored on the FIRST save, not the ROCm one: slicing below the five pairs above it left the
    harness supplying their $previous*, so those restores were measured against harness constants."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    start = src.index("    $previousSkipStudioBase = $env:SKIP_STUDIO_BASE")
    end = src.index("    if ($setupExit -ne 0) {", start)
    return src[start:end]


def _run_handoff_lifecycle(
    tmp_path: Path,
    *,
    arch: str | None,
    inherited: str | None,
    fails: bool,
    bails: bool = False,
    with_llama_cpp_dir: bool = False,
    caller_env: str = "all",
    studio_mode: str = "default",
) -> dict:
    assert not (bails and with_llama_cpp_dir), "the bail and the success path are exclusive"
    assert studio_mode in _STUDIO_MODES, studio_mode
    present = _present_names(caller_env)
    call = "Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs"
    block = _handoff_lifecycle_block()
    # Loudly: a silent miss leaves the probe unrun and every assertion reading
    # "<never ran>" with nothing saying why.
    assert call in block, "install.ps1 no longer makes the setup call this harness replaces"
    # Read at the point of the call: what the child would inherit, not what the finally leaves.
    probe = (
        "$script:SeenByChild = $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF; "
        "$script:SeenLlamaCppDir = $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR; "
        # Test-Path, not the bare value: the default modes REMOVE these two.
        "$script:SeenStudioHome = $(if (Test-Path Env:UNSLOTH_STUDIO_HOME) "
        "{ $env:UNSLOTH_STUDIO_HOME } else { $null }); "
        "$script:SeenLocalRepo = $(if (Test-Path Env:STUDIO_LOCAL_REPO) "
        "{ $env:STUDIO_LOCAL_REPO } else { $null })"
    )
    body = block.replace(call, probe + ("; throw 'setup exploded'" if fails else ""))
    script = tmp_path / "handoff.ps1"
    script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                "$UnslothProxyHandoffJson = $null",
                "$UnslothExe = 'stub'; $studioArgs = @(); $setupExit = 0",
                # Installer inputs the block reads. Undefined, they throw under
                # ErrorActionPreference Stop, the catch swallows it, and the probe never runs.
                "$PackageName = 'unsloth'; $SkipTorch = $false; $TauriMode = $false",
                *_studio_mode_lines(tmp_path, studio_mode),
                (
                    f"$WithLlamaCppDir = '{tmp_path / 'no-such-llama.cpp'}'"
                    if bails
                    # A directory that EXISTS reaches the block's only write to that variable.
                    else f"$WithLlamaCppDir = '{_existing_llama_dir(tmp_path)}'"
                    if with_llama_cpp_dir
                    else "$WithLlamaCppDir = $null"
                )
                + "; $VenvPython = 'stub-python'; $VenvDir = 'stub-venv'",
                "$TorchIndexUrl = $null; $ROCmIndexUrl = $null",
                # Every installer function the block reaches, stubbed.
                "function Get-ExpectedTorchFlavorTag { param($TorchIndexUrl, $ROCmIndexUrl) 'cu128' }",
                "function Get-InstalledTorchVersionRaw { param($Python) '' }",
                "function ConvertTo-TorchNumericRelease { param($Raw) $null }",
                "function Write-StudioLine { param($Message, $ForegroundColor) }",
                "function Write-ApplicationControlBlocked { param($Message, $Detail) }",
                "function Exit-InstallFailure { param($Message) 1 }",
                "$script:SeenByChild = '<never ran>'; $script:SeenLlamaCppDir = '<never ran>'",
                "$script:SeenStudioHome = '<never ran>'; $script:SeenLocalRepo = '<never ran>'",
                "$script:BlockError = $null",
                # Read right after the setup call; null makes the block return early.
                "$script:ManagedUnslothCliExit = 0",
                "$script:PrevTorchPin = $null",
                "$ROCmGfxArch = " + ("$null" if arch is None else f"'{arch}'"),
                # In a function so the block's own return -- the --with-llama-cpp-dir bail --
                # leaves the block, not the script, letting us read the environment after it.
                "function Invoke-HandoffBlock {",
                "try {",
                body,
                "} catch { $script:BlockError = $_.ToString() }",
                "}",
                "Invoke-HandoffBlock | Out-Null",
                "@{",
                "  seen_by_child = $script:SeenByChild",
                "  seen_llama_cpp_dir = $script:SeenLlamaCppDir",
                "  seen_studio_home = $script:SeenStudioHome",
                "  seen_local_repo = $script:SeenLocalRepo",
                "  block_error = $script:BlockError",
                "  after = $(if (Test-Path Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF) { $env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF } else { $null })",
                "  after_set = [bool](Test-Path Env:_UNSLOTH_ROCM_GFX_ARCH_HANDOFF)",
                "  public = $(if (Test-Path Env:UNSLOTH_ROCM_GFX_ARCH) { $env:UNSLOTH_ROCM_GFX_ARCH } else { $null })",
                # Value AND presence: a variable assigned "" is still present on 7.5+, so only
                # Test-Path separates "put back as it was" from "recreated empty".
                _caller_env_report(),
                "} | ConvertTo-Json -Compress",
            ]
        ),
        encoding = "utf-8",
    )
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "UNSLOTH_ROCM_GFX_ARCH": "gfx90a"}
    # Built from scratch, so a name simply not added reads back $null: the finally's remove arm.
    env.update({name: sentinel for name, sentinel in _CALLER_ENV if name in present})
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
    # Last JSON object only: a stub may emit its return value into the pipeline first.
    reports = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert reports, f"the handoff block printed no report:\n{proc.stdout}\n{proc.stderr}"
    out = json.loads(reports[-1])
    # The block may throw only where the test asked; anything else is a missing stub.
    if fails:
        # The injected throw specifically: an earlier helper failure would still restore
        # the environment and pass every assertion without the failure path running.
        assert "setup exploded" in (
            out.get("block_error") or ""
        ), f"the block failed before the injected throw: {out.get('block_error')}"
    else:
        assert not out.get("block_error"), f"the handoff block threw: {out['block_error']}"
    return out


@requires_pwsh
@pytest.mark.parametrize(
    "caller_env", list(_PRESENCE_PATTERNS), ids = [f"caller_env_{p}" for p in _PRESENCE_PATTERNS]
)
@pytest.mark.parametrize("fails", [False, True], ids = ["setup_ok", "setup_throws"])
@pytest.mark.parametrize(
    "arch, inherited",
    [(None, None), ("gfx1151", None), (None, "gfx1030"), ("gfx1151", "gfx1030")],
    ids = ["nothing", "resolved", "inherited", "resolved_over_inherited"],
)
def test_the_caller_environment_survives_the_setup_call(
    tmp_path, arch, inherited, fails, caller_env
):
    """`irm ... | iex` runs install.ps1 in the caller's own shell, so anything set for the child
    has to be put back -- on the failure path too, which is the one that rolls back and retries,
    and whether or not the caller had the variable to begin with."""
    out = _run_handoff_lifecycle(
        tmp_path,
        arch = arch,
        inherited = inherited,
        fails = fails,
        caller_env = caller_env,
    )
    assert out["after_set"] is (inherited is not None), "the handoff outlived the setup call"
    assert out["after"] == inherited
    assert out["public"] == "gfx90a", "a user's own override must come back untouched"
    _assert_caller_env_restored(out, _present_names(caller_env), "the setup call")


@requires_pwsh
@pytest.mark.parametrize(
    "caller_env", list(_PRESENCE_PATTERNS), ids = [f"caller_env_{p}" for p in _PRESENCE_PATTERNS]
)
@pytest.mark.parametrize("fails", [False, True], ids = ["setup_ok", "setup_throws"])
def test_the_optional_handoffs_are_restored_when_this_run_sets_them(tmp_path, fails, caller_env):
    """UNSLOTH_STUDIO_HOME and STUDIO_LOCAL_REPO are the two the try does not always assign.

    The case above hard-codes redirect mode 'none' and no local install, so the try left these two
    ABSENT and deleting the finally's else arm was invisible; every other saved variable is
    assigned unconditionally, so the same deletion shows there at once."""
    out = _run_handoff_lifecycle(
        tmp_path,
        arch = "gfx1151",
        inherited = None,
        fails = fails,
        caller_env = caller_env,
        studio_mode = "env_redirect_local",
    )
    assert out["seen_studio_home"] == str(
        _studio_home_dir(tmp_path)
    ), "the env-redirect install did not hand its studio home to the child"
    assert out["seen_local_repo"] == str(
        _studio_repo_dir(tmp_path)
    ), "the --local install did not hand its repo to the child"
    _assert_caller_env_restored(out, _present_names(caller_env), "the env-redirect local install")


def test_every_save_sits_above_the_handoff_try():
    """Ordering, not just membership: a save that drifts INSIDE the try is a live hazard.

    A set comparison still matches, and no runtime case catches it either, since both injected
    failures sit BELOW where such a save would land. What bites is a throw ABOVE it, leaving
    $hadPrevious* unbound, hence $null, hence the finally removing a value the caller owned."""
    block = _handoff_lifecycle_block()
    assert block.count("\n    try {") == 1, "the block no longer has exactly one handoff try"
    try_at = block.index("\n    try {")
    saves = list(re.finditer(r"\$previous(\w+) = \$env:(\w+)", block))
    flags = list(re.finditer(r"\$hadPrevious(\w+) = \(\$null -ne \$previous\w+\)", block))
    assert {m.group(2) for m in saves} == {name for name, _ in _CALLER_ENV} | {HANDOFF}
    assert len(flags) == len(saves), (
        f"{len(saves)} saves but {len(flags)} $hadPrevious flags; a save without its flag "
        "restores through an unbound $null and takes the remove arm"
    )
    for m in saves:
        assert m.start() < try_at, (
            f"the save of {m.group(2)} sits inside the try, so anything that throws above it "
            "leaves $hadPrevious unbound and the finally removes a value the caller owned"
        )
    for m in flags:
        assert m.start() < try_at, (
            f"$hadPrevious{m.group(1)} is bound inside the try, so a throw above it leaves the "
            "flag $null and the finally takes the remove arm against a caller that had a value"
        )


def test_every_saved_variable_in_the_block_is_covered():
    """_CALLER_ENV checked against the source, so a save added to the block names the variable
    whose restore nothing exercises. UV_CACHE_DIR, TMP and TEMP are out of scope by construction:
    saved and restored hundreds of lines outside this block, so covering them means slicing most
    of install.ps1 and stubbing the venv build, the torch install and the llama.cpp fetch."""
    block = _handoff_lifecycle_block()
    covered = {name for name, _ in _CALLER_ENV} | {HANDOFF}
    saved = set(re.findall(r"\$previous\w+ = \$env:(\w+)", block))
    # The restore side closes the anchor hole: a save PREPENDED above the anchor is invisible to
    # any save-side check, but its restore cannot escape the finally.
    restored = set(re.findall(r"\$env:(\w+) = \$previous\w+", block))
    assert restored == covered, (
        "the block restores variables this file does not claim to cover: "
        f"{sorted(restored - covered)} (a save prepended above the slice anchor looks like this); "
        f"and claims ones it no longer restores: {sorted(covered - restored)}"
    )
    assert saved == restored, (
        "saves and restores in the block disagree: saved but never restored "
        f"{sorted(saved - restored)}; restored but not saved inside the slice "
        f"{sorted(restored - saved)}"
    )


@requires_pwsh
@pytest.mark.parametrize(
    "caller_env", list(_PRESENCE_PATTERNS), ids = [f"caller_env_{p}" for p in _PRESENCE_PATTERNS]
)
def test_the_bail_restores_the_caller_environment(tmp_path, caller_env):
    """The --with-llama-cpp-dir bail returns from inside the try, so the finally still runs.

    Textual ordering cannot show that: move the try below the bail and `saved < bail <
    restored` still holds while the return walks out past the restore. So this takes the
    bail, with a directory that does not exist, and reads the environment afterwards.

    STUDIO_PACKAGE_NAME and SKIP_STUDIO_BASE, asserted individually here before, are now two of the
    fifteen the helper checks, in both directions rather than only for removal."""
    out = _run_handoff_lifecycle(
        tmp_path,
        arch = "gfx1151",
        inherited = "gfx1030",
        fails = False,
        bails = True,
        caller_env = caller_env,
    )
    assert out["seen_by_child"] == "<never ran>", "the bail did not happen before the setup call"
    assert out["after"] == "gfx1030", "the caller's inherited handoff was not restored by the bail"
    assert out["after_set"] is True
    # Put back only by the finally, so install.ps1's value still showing means the bail escaped.
    _assert_caller_env_restored(out, _present_names(caller_env), "the bail")


@requires_pwsh
@pytest.mark.parametrize(
    "caller_env", list(_PRESENCE_PATTERNS), ids = [f"caller_env_{p}" for p in _PRESENCE_PATTERNS]
)
def test_a_real_llama_cpp_dir_is_handed_over_and_then_put_back(tmp_path, caller_env):
    """The other side of the bail: --with-llama-cpp-dir naming a directory that is there.

    Nothing reached the block's only write to UNSLOTH_LOCAL_LLAMA_CPP_DIR, so its restore was
    satisfied by never being dirtied, which is not the same as being correct."""
    out = _run_handoff_lifecycle(
        tmp_path,
        arch = "gfx1151",
        inherited = "gfx1030",
        fails = False,
        with_llama_cpp_dir = True,
        caller_env = caller_env,
    )
    assert out["seen_by_child"] == "gfx1151", "the block did not reach the setup call"
    # The RESOLVED path, since that is what the block writes and what setup.ps1 goes on to read.
    assert out["seen_llama_cpp_dir"] == str(
        _existing_llama_dir(tmp_path).resolve()
    ), "the child did not inherit the --with-llama-cpp-dir directory"
    assert out["after"] == "gfx1030", "the caller's inherited handoff was not restored"
    _assert_caller_env_restored(out, _present_names(caller_env), "the llama.cpp handoff")


@requires_pwsh
@pytest.mark.parametrize(
    "arch, inherited, expected",
    [("gfx1151", None, "gfx1151"), ("gfx1151", "gfx1030", "gfx1151"), (None, "gfx1030", None)],
    ids = ["resolved", "resolved_over_inherited", "stale_only"],
)
@pytest.mark.parametrize("caller_env", sorted(_PRESENCE_PATTERNS), ids = sorted(_PRESENCE_PATTERNS))
def test_only_this_runs_arch_is_handed_to_the_child(
    tmp_path, arch, inherited, expected, caller_env
):
    """A value inherited from an outer process is not this run's answer, so it is cleared rather
    than forwarded as though the scan had produced it.

    Parametrised over the caller patterns, not left on the helper's default. The default is
    `all`, which sets every caller variable, and a clear that wrongly keyed itself off some
    other variable's $hadPrevious flag would then find that flag true and clear anyway. The
    empty-caller case is the one that catches it, and running only the default lost it.
    """
    out = _run_handoff_lifecycle(
        tmp_path, arch = arch, inherited = inherited, fails = False, caller_env = caller_env
    )
    assert out["seen_by_child"] == expected


@requires_pwsh
def test_these_assertions_fail_without_the_array_wraps(tmp_path):
    """A regression test that passes on the unfixed source is not one, and the first version of
    this file was exactly that. Undo just the two wraps and confirm the failures come back."""
    unfixed = _without_the_array_wraps(_setup_source())
    before = _run(tmp_path, [(_RADEON, 0)], source = unfixed, ps51 = True)
    pinned = _run(tmp_path, [(_RADEON, 0)], source = unfixed, env = {"HIP_VISIBLE_DEVICES": "0"})
    assert not before["wmi_array"], "the unwrapped scan should collapse to a scalar"
    assert before["label"] is None, "the unwrapped scan should report no GPU under 5.1 semantics"
    assert pinned["arch"] is None, "the unwrapped name list should infer nothing when pinned"
