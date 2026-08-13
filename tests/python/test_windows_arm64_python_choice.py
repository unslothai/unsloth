# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM: install.ps1 must not settle for a native ARM64 interpreter.

pyarrow (via datasets) and hf-transfer publish no win_arm64 wheels, so an ARM64
Python source-builds both and dies minutes into the run. The resolver prefers an
x64 build of the requested minor and bootstraps one otherwise; the case pinned
here is the recovery path, where nothing can be downloaded but an x64 build of a
lower-priority supported minor is already installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _resolver_script(installed: list[tuple[str, str]], can_download: bool) -> str:
    """Both production functions verbatim, over a fake set of interpreters.

    Extracted rather than reimplemented so the test cannot drift away from the
    text install.ps1 actually runs. `installed` is (minor, arch) in py-launcher
    order, so the first entry for a minor is what a bare `py -3.13` resolves to.
    The fake interpreters are named `*.exe` and invoked through the call operator,
    which resolves a string to a function, so no real binary is needed.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    finder = _extract(r"    function Find-CompatiblePython \{.*?\n    \}\n", source)
    installer = _extract(r"    function Install-X64Python \{.*?\n    \}\n", source)

    names = [f"Py{minor.replace('.', '')}{arch}.exe" for minor, arch in installed]
    table = ", ".join(
        f'@{{ Minor = "{minor}"; Arch = "{arch}"; Name = "{name}" }}'
        for (minor, arch), name in zip(installed, names)
    )
    downloaded = (
        '@{ Version = "3.13"; Path = "Downloaded.exe"; Arch = "x86_64" }'
        if can_download
        else "$null"
    )
    version_stubs = "\n".join(
        f"function {name} {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest)\n"
        f'    if ($Rest -contains "--version") {{ return "Python {minor}.0" }}\n'
        f'    return "{name}" }}'
        for (minor, _arch), name in zip(installed, names)
    )
    return f"""
$ErrorActionPreference = "Stop"
$PythonVersion = "3.13"
$script:WingetAvailable = $false
$script:CondaSkipPattern = 'conda'
$Interpreters = @({table})
{version_stubs}
# `py -0p` lists every registration; `py -3.x` runs the launcher's preferred build
# for that minor, which on an ARM64 host is normally the native one.
function FakePy {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Rest -contains "-0p") {{
        return @($Interpreters | ForEach-Object {{ "  -V:$($_.Minor) *        $($_.Name)" }})
    }}
    $minor = ([string]$Rest[0]).TrimStart('-')
    $hit = @($Interpreters | Where-Object {{ $_.Minor -eq $minor }})
    if ($hit.Count -eq 0) {{ return "" }}
    if ($Rest -contains "--version") {{ return "Python $minor.0" }}
    return $hit[0].Name
}}
function substep {{ param($a, $b) }}
function Get-HostMachineArch {{ return "arm64" }}
function Get-Command {{
    param([Parameter(Position = 0)][string]$Name,
          [Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Name -eq "py") {{ return @([pscustomobject]@{{ Source = "FakePy" }}) }}
    return @()
}}
function Test-Path {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) return $true }}
function Test-IsCondaPython {{ param([string]$Exe) return $false }}
function Get-PythonPlatformTag {{
    param([string]$Exe)
    foreach ($i in $Interpreters) {{
        if ($i.Name -eq $Exe) {{
            if ($i.Arch -eq "x86_64") {{ return "win-amd64" }} else {{ return "win-arm64" }}
        }}
    }}
    return "win-amd64"
}}
function Refresh-SessionPath {{ }}
function Install-PythonFromPythonOrg {{ param([string]$Arch = "") return {downloaded} }}
{finder}
{installer}
# The caller's ARM64 swap, condensed to what decides the interpreter.
$found = Find-CompatiblePython
if ($found -and $found.Arch -ne "x86_64") {{
    $x64 = Install-X64Python
    if ($x64) {{ $found = $x64 }}
}}
if ($found) {{ Write-Output "$($found.Version)|$($found.Arch)" }} else {{ Write-Output "none" }}
"""


def _pwsh(script: str) -> str:
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = os.environ.copy(),
    )
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("installed", "can_download", "expected"),
    [
        # An x64 build of the requested minor wins outright, downloads irrelevant.
        ([("3.13", "arm64"), ("3.13", "x86_64")], False, "3.13|x86_64"),
        # Requested minor is ARM64-only: bootstrap x64 rather than take the native one.
        ([("3.13", "arm64")], True, "3.13|x86_64"),
        # Offline, but an x64 build of a lower-priority minor is here. Use it: the native
        # 3.13 cannot resolve pyarrow or hf-transfer, and this one can.
        ([("3.13", "arm64"), ("3.11", "x86_64")], False, "3.11|x86_64"),
        # ARM64 everywhere: still returned, and the caller warns.
        ([("3.13", "arm64"), ("3.11", "arm64")], False, "3.13|arm64"),
    ],
)
def test_arm64_host_prefers_an_x64_interpreter(installed, can_download, expected):
    assert _pwsh(_resolver_script(installed, can_download)) == expected


# ── setup.ps1: every path that REUSES an interpreter has to re-ask its arch ──
# install.ps1's swap above only runs over a freshly selected interpreter. setup.ps1
# is handed one (UNSLOTH_SETUP_PYTHON, or the venv python), and validated it by
# version and conda-ness alone, so an ARM64 environment sailed through and every
# update it ran ended in the same pyarrow source build (issue #8495).

SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


def _setup_arch_script(interpreter_tag: str, host_arch: str, no_datasets: bool) -> str:
    """Both production functions verbatim, over one fake interpreter.

    Extracted from setup.ps1 rather than reimplemented, for the same reason as the
    install.ps1 harness above: a copy here would keep passing after the original
    changed. The fake interpreter is a function invoked through the call operator,
    so `& $Exe -S -c ...` needs no real binary.
    """
    source = SETUP_PS1.read_text(encoding = "utf-8")
    tag_fn = _extract(r"function Get-PythonPlatformTag \{.*?\r?\n\}", source)
    arch_fn = _extract(r"function Test-CompatibleSetupPythonArch \{.*?\r?\n\}", source)
    return f"""
$ErrorActionPreference = "Stop"
$script:NoDatasetsMode = {'$true' if no_datasets else '$false'}
function Get-HostMachineArch {{ return "{host_arch}" }}
function FakePython {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    return "{interpreter_tag}"
}}
{tag_fn}
{arch_fn}
if (Test-CompatibleSetupPythonArch "FakePython") {{ Write-Output "accept" }} else {{ Write-Output "reject" }}
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("interpreter_tag", "host_arch", "no_datasets", "expected"),
    [
        # The reported failure: an ARM64 interpreter on an ARM64 host, full install.
        ("win-arm64", "arm64", False, "reject"),
        # The supported Windows-on-ARM configuration: x64 CPython under emulation.
        ("win-amd64", "arm64", False, "accept"),
        # The ARM64 inference-only tier runs on win-arm64 on purpose.
        ("win-arm64", "arm64", True, "accept"),
        # Every other host is unconstrained -- and pays no subprocess for the probe.
        ("win-amd64", "x86_64", False, "accept"),
        ("win-arm64", "x86_64", False, "accept"),
        # An unreadable interpreter is not evidence of an x64 build.
        ("", "arm64", False, "reject"),
    ],
)
def test_setup_arch_gate(interpreter_tag, host_arch, no_datasets, expected):
    assert _pwsh(_setup_arch_script(interpreter_tag, host_arch, no_datasets)) == expected


def test_every_setup_interpreter_path_consults_the_arch_gate():
    """One missed call site re-admits what the others reject, and the reused-venv
    path is precisely the one that reached a user (issue #8495)."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    # The 1g reuse gate, the py-launcher loop, the bare-python fallback, and both
    # phase-3 resolution paths.
    assert source.count("Test-CompatibleSetupPythonArch") >= 6


def test_setup_winget_fallback_asks_for_x64():
    """winget defaults to the ARM64 package on an ARM64 host, which is the build
    that cannot resolve the stack."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    assert '@("--architecture", "x64")' in source
    winget_call = source[source.index("Python.Python.3.12 --source winget") - 600 :]
    assert "_wingetArchArgs" in winget_call


def test_setup_refuses_an_arm64_environment_with_an_actionable_message():
    """Failing here beats spending minutes to fail inside a pyarrow build, but only
    if the message names the fix rather than the symptom."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index("Environment uses ARM64 Python")
    message = source[index - 1500 : index + 200]
    # The whole URL, not the bare host: a hostname substring test is what CodeQL's
    # incomplete-URL-sanitization query flags (alert 808), and the command the user
    # is told to paste is the thing worth pinning anyway.
    assert "irm https://unsloth.ai/install.ps1 | iex" in message
    assert "UNSLOTH_NO_DATASETS" in message


def test_install_ps1_rechecks_a_migrated_venv():
    """A migrated environment never met the x64 swap, so its interpreter is still
    whatever built it. Probe it, set it aside through the existing rollback, and
    clear $_Migrated so the fresh-install path (not the migrated-upgrade path) runs."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("migrated environment is")
    block = source[index - 2400 : index + 1200]
    assert "Get-PythonPlatformTag $VenvPython" in block
    assert "Start-StudioVenvRollback" in block
    assert "$_Migrated = $false" in block


def test_an_unreadable_arch_tag_never_rebuilds_a_migrated_venv():
    """The probe returns "" for a broken interpreter, a one-shot an antivirus
    blocked, or a base install that moved. Treating that as a wrong architecture
    sends a working environment -- and whatever the user keeps inside it -- through
    a rollback whose SUCCESS deletes the original tree. Only a tag we can read and
    that disagrees is evidence."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("migrated environment is")
    condition = source[source.index("$migratedTag = Get-PythonPlatformTag") : index]
    assert "if ($migratedTag -and $migratedTag -ne $wantedTag)" in condition


def test_a_migrated_x64_venv_is_kept_even_when_the_tier_is_on():
    """The tier turns on when no x64 interpreter can be FOUND. If the migrated venv
    then turns out to have one, rebuilding would delete a training-capable
    environment and replace it with the reduced one."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("$migratedTag = Get-PythonPlatformTag")
    block = source[index : source.index("migrated environment is", index)]
    assert "$script:ArmInferenceOnly -and" in block
    assert '$migratedTag -eq "win-amd64"' in block
    assert "$script:ArmInferenceOnly = $false" in block
    # And the handoff variable goes back with it, or setup.ps1 would still be told
    # to install the tier into the x64 environment we just decided to keep.
    assert "Remove-Item Env:UNSLOTH_NO_DATASETS" in block


def test_an_explicitly_requested_tier_survives_a_migrated_x64_venv():
    """The auto-withdrawal above is for a tier the installer FELL BACK into. Someone
    who set UNSLOTH_NO_DATASETS=1 asked for inference-only on purpose -- to keep a
    native environment, or to skip the training stack -- and finding an x64 venv is
    not a reason to overrule them."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("$migratedTag = Get-PythonPlatformTag")
    block = source[index : source.index("migrated environment is", index)]
    assert "-not $script:ArmInferenceOnlyRequested" in block
    # And the flag is set where the environment variable is read, not anywhere the
    # installer's own fallback could also reach.
    parsed = source[source.index("$script:ArmInferenceOnly = $false") :][:900]
    assert "$script:ArmInferenceOnlyRequested = $true" in parsed
    fallback = source[source.index("Could not install an x64 Python") - 500 :][:1700]
    assert "$script:ArmInferenceOnlyRequested" not in fallback


def test_the_tier_branch_honours_the_requested_package():
    """--package unsloth-fork must not silently install released unsloth. Every
    other branch passes $PackageName through; this one may not be the exception."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    block = source[source.index("install unsloth (arm64 inference-only)") - 700 :][:1200]
    assert "$armCoreSpec" in block
    assert '$PackageName -eq "unsloth"' in block
    # The version floor still applies to the default name: that is the release which
    # knows about this tier.
    assert "unsloth>=2026.8.15" in block


def test_every_arm_filtered_copy_is_cleaned_up():
    """Get-ArmFilteredRequirements writes a temp file per call. A branch that
    forgets the cleanup leaves it in the requirements directory of the install,
    where the next run's digest of that directory sees a file that is not shipped."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    calls = source.count("= Get-ArmFilteredRequirements ")
    cleanups = source.count("\n                Remove-ArmFilteredRequirements\n")
    assert (
        calls and calls == cleanups
    ), f"{calls} Get-ArmFilteredRequirements call sites but {cleanups} cleanups"


def test_a_failed_install_does_not_leave_the_tier_in_the_callers_shell():
    """`irm ... | iex` runs this in the caller's PowerShell. The restore beside the
    other handoff variables happens after the setup call, and every failure before
    that skips it -- including the one whose message tells the user to install x64
    Python and re-run in the same terminal."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    failure = source[source.index("function Exit-InstallFailure") :][:1400]
    assert "UNSLOTH_NO_DATASETS" in failure
    assert "$script:PreviousNoDatasetsEnv" in failure


def test_the_env_snapshot_is_taken_before_anything_can_fail():
    """$script: state outlives one `irm ... | iex`, and Exit-InstallFailure restores
    UNSLOTH_NO_DATASETS from it: snapshotted after the flag parsing, a second run that
    dies on its own arguments puts back the value the first run saw."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    snapshot = source.index("$script:PreviousNoDatasetsEnv = $env:UNSLOTH_NO_DATASETS")
    first_failure = source.index("return (Exit-InstallFailure")
    assert snapshot < first_failure, "the snapshot must precede every Exit-InstallFailure call site"


def test_install_ps1_falls_back_to_the_inference_only_tier():
    """When no x64 interpreter can be obtained, continuing into `uv pip install
    unsloth` only buys a CMake failure. The tier drops the wheel-less packages
    instead, and UNSLOTH_NO_DATASETS carries the choice into setup.ps1."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("Could not install an x64 Python")
    block = source[index - 500 : index + 1200]
    assert "$script:ArmInferenceOnly = $true" in block
    assert 'UNSLOTH_NO_DATASETS = "1"' in block
    assert "--no-deps" in source[source.index("arm64 inference-only") :][:800]


def test_setup_hands_the_resolved_tier_to_the_python_child():
    """setup.ps1 resolves manifest over marker, then the dependency pass deletes the
    manifest before the child re-infers the tier. Without the handoff a completed x64
    manifest saying false loses to a stale marker."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index("$_recorded = $null")
    block = source[index : index + 2200]
    assert '$env:UNSLOTH_NO_DATASETS = if ($_recorded) { "1" } else { "0" }' in block
    # The marker branch too: it is the record that survives a pass which died before
    # the manifest was written.
    marker = block[block.index(".unsloth-no-datasets") :][:400]
    assert 'UNSLOTH_NO_DATASETS = "1"' in marker
    # And "0" really does mean off downstream, rather than being a truthy string.
    stack = (REPO_ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    infer = stack[stack.index("def _infer_no_datasets") :][:1400]
    assert "NO_TORCH_TRUTHY" in infer


def test_setup_reads_the_tier_marker_not_just_the_env_var():
    """install.ps1 exports UNSLOTH_NO_DATASETS for its own run only. Without the
    marker, `unsloth studio update` on a tier install would judge it by the
    full-install rule and refuse the environment the installer had just built."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index("$NoDatasetsMode = (")
    block = source[index : index + 2600]
    assert ".unsloth-no-datasets" in block
    # And it has to come after the venv interpreter is resolved, or there is no
    # venv path to look in.
    assert source.index("$ReusedSetupPython = Resolve-ReusedSetupPython") < index


# ── Get-ArmFilteredRequirements: what the tier hands uv ──


def _filter_script(requirements: str, host_arch: str, tier: bool) -> str:
    """The real function and its two tables, over one requirements file."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    skip = _extract(r"\$script:ArmInferenceSkipPackages = .*?\)", source)
    lift = _extract(r"\$script:ArmInferenceLiftPackages = @\{.*?\n    \}", source)
    body = _extract(r"    function Get-ArmFilteredRequirements \{.*?\n    \}", source)
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.splitlines())
    path = Path(tempfile.mkdtemp()) / "reqs.txt"
    path.write_text(requirements, encoding = "utf-8")
    return f"""
$ErrorActionPreference = "Stop"
function substep {{ param($m, $c) }}
function Get-HostMachineArch {{ return "{host_arch}" }}
{skip}
{lift}
{body}
$script:ArmInferenceOnly = ${str(tier).lower()}
$out = Get-ArmFilteredRequirements -Path "{path}"
if ($out -eq "{path}") {{ Write-Output "UNCHANGED" }} else {{ Get-Content -LiteralPath $out -Raw }}
"""


_TIER_SAMPLE = """\
# keep me
datasets==4.3.0
hf_transfer==0.1.9
pymupdf==1.27.2.3
transformers>=4.57.6
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_filter_is_a_no_op_outside_the_tier():
    """Every non-ARM install goes through this function too, and must get its own
    file back untouched."""
    assert _pwsh(_filter_script(_TIER_SAMPLE, "arm64", tier = False)) == "UNCHANGED"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_filter_drops_and_lifts_on_an_arm64_host():
    out = _pwsh(_filter_script(_TIER_SAMPLE, "arm64", tier = True))
    assert "datasets==" not in out
    assert "hf_transfer" not in out  # spelled with an underscore here, hyphen in the list
    assert "pymupdf>=1.28.2" in out
    assert "transformers>=4.57.6" in out
    assert "# keep me" in out


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_lifts_do_not_apply_on_an_x64_host():
    """UNSLOTH_NO_DATASETS=1 turns the tier on for an x64 install as well. There the
    pinned versions are the tested ones and every one of them has a win_amd64 wheel,
    so only the skips apply -- install_python_stack.py gates its own lift on
    IS_WINDOWS_ARM64_PYTHON for the same reason."""
    out = _pwsh(_filter_script(_TIER_SAMPLE, "x64", tier = True))
    assert "datasets==" not in out
    assert "pymupdf==1.27.2.3" in out
    assert "pymupdf>=1.28.2" not in out


# ── The inference-only tier picks NATIVE ARM64, and picks it deliberately ──


def _tier_resolver_script(installed: list[tuple[str, str]]) -> str:
    """Find-CompatiblePython verbatim, with the tier on and the x64 swap skipped.

    install.ps1 skips the swap entirely under $script:ArmInferenceOnly, so whatever
    the resolver returns IS the interpreter the venv is built with.
    """
    script = _resolver_script(installed, can_download = False)
    script = script.replace(
        '$PythonVersion = "3.13"',
        '$PythonVersion = "3.13"\n$script:ArmInferenceOnly = $true',
        1,
    )
    swap = """$found = Find-CompatiblePython
if ($found -and $found.Arch -ne "x86_64") {
    $x64 = Install-X64Python
    if ($x64) { $found = $x64 }
}"""
    assert swap in script
    return script.replace(swap, "$found = Find-CompatiblePython")


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        # The case that made this a bug: the py launcher hands out an x64 build first.
        # Returning it immediately produced an EMULATED install with the training
        # packages stripped out -- the disadvantages of both tiers, when opting in is
        # precisely a choice to keep the native interpreter.
        ([("3.13", "x86_64"), ("3.13", "arm64")], "3.13|arm64"),
        # Native build of the requested minor wins outright.
        ([("3.13", "arm64"), ("3.13", "x86_64")], "3.13|arm64"),
        # Requested minor is x64-only: a native build of a lower-priority supported
        # minor is what the tier is for.
        ([("3.13", "x86_64"), ("3.11", "arm64")], "3.11|arm64"),
        # Nothing native anywhere: still returns something rather than failing.
        ([("3.13", "x86_64")], "3.13|x86_64"),
    ],
)
def test_inference_only_tier_prefers_the_native_interpreter(installed, expected):
    assert _pwsh(_tier_resolver_script(installed)) == expected


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_x64only_is_never_answered_with_arm64_in_the_tier():
    """-X64Only is Install-X64Python's own last resort. Answering it with the native
    build would make the bootstrap report success and change nothing."""
    script = _tier_resolver_script([("3.13", "arm64"), ("3.11", "x86_64")]).replace(
        "$found = Find-CompatiblePython", "$found = Find-CompatiblePython -X64Only"
    )
    assert _pwsh(script) == "3.11|x86_64"


def test_fresh_tier_branch_installs_the_cli_runtime_dependencies():
    """--no-deps leaves unsloth without typer, and unsloth_cli/__init__.py imports it
    at module scope: `& $UnslothExe studio setup` would exit ModuleNotFoundError
    before install_python_stack.py could install anything. The two other --no-deps
    branches already lay down no-torch-runtime.txt for exactly this reason."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index('"install unsloth (arm64 inference-only)"')
    branch = source[index : source.index("} elseif ($StudioLocalInstall)", index)]
    assert "Get-ArmFilteredRequirements (Find-NoTorchRuntimeFile)" in branch
    assert "--no-deps -r $NoTorchReq" in branch
    runtime = (
        REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
    ).read_text(encoding = "utf-8")
    assert "typer" in runtime


def test_no_datasets_env_override_is_restored():
    """`irm ... | iex` runs in the CALLER's process, so the fallback's
    $env:UNSLOTH_NO_DATASETS = "1" outlived the installer and pinned the session to
    the tier -- including the x64 retry the failure message itself asks for."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "$script:PreviousNoDatasetsEnv = $env:UNSLOTH_NO_DATASETS" in source
    assert source.index("$script:PreviousNoDatasetsEnv = ") < source.index(
        '$env:UNSLOTH_NO_DATASETS = "1"'
    )
    restore = source[source.index("if ($script:HadPreviousNoDatasetsEnv)") :][:400]
    assert "$env:UNSLOTH_NO_DATASETS = $script:PreviousNoDatasetsEnv" in restore
    assert "Remove-Item Env:UNSLOTH_NO_DATASETS" in restore


SETUP_PS1_FILE = REPO_ROOT / "studio" / "setup.ps1"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", "True"),
        ("true", "True"),
        ("TRUE", "True"),
        ("yes", "True"),
        ("on", "True"),
        (" on ", "True"),
        ("0", "False"),
        ("", "False"),
        ("maybe", "False"),
    ],
)
def test_setup_accepts_every_documented_truthy_no_datasets_value(value, expected):
    """install.ps1 accepts 1/true/yes/on and keeps the native ARM64 interpreter on
    any of them. An exact -eq "1" here read the rest as "off", and with no marker
    yet on a fresh install the arch gate then rejected that very interpreter and
    aborted setup, against an opt-in the user had stated explicitly."""
    source = SETUP_PS1_FILE.read_text(encoding = "utf-8")
    index = source.index("$NoDatasetsMode = (")
    expression = source[index : source.index("$HasPython = ", index)]
    script = (
        f'$env:UNSLOTH_NO_DATASETS = "{value}"\n'
        "$ReusedSetupPython = $null\n"
        f"{expression}\n"
        "Write-Output $NoDatasetsMode"
    )
    assert _pwsh(script) == expected


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_setup_treats_an_unset_no_datasets_variable_as_off():
    source = SETUP_PS1_FILE.read_text(encoding = "utf-8")
    index = source.index("$NoDatasetsMode = (")
    expression = source[index : source.index("$HasPython = ", index)]
    script = (
        "Remove-Item Env:UNSLOTH_NO_DATASETS -ErrorAction SilentlyContinue\n"
        "$ReusedSetupPython = $null\n"
        f"{expression}\n"
        "Write-Output $NoDatasetsMode"
    )
    assert _pwsh(script) == "False"


def test_setup_bootstraps_x64_when_only_a_native_python_is_on_path():
    """With a supported native ARM64 python on PATH, $PythonOk was false and
    $HasPython true, so the winget branch -- guarded by `-not $HasPython`, and the
    only thing that would install the x64 build -- was skipped and setup hard-exited
    with 'No supported Python (3.11-3.13)' on a machine that has a supported 3.12."""
    source = SETUP_PS1_FILE.read_text(encoding = "utf-8")
    index = source.index("if (-not $PythonOk -and $HasPython -and (Get-HostMachineArch) -eq")
    block = source[index : source.index("if ($PythonOk) {", index)]
    assert "Test-CompatibleSetupPythonArch" in block
    assert "$HasPython = $false" in block
    assert "-not $NoDatasetsMode" in block
    # And it has to run BEFORE the branch it unblocks.
    assert index < source.index("elseif (-not $HasPython) {")


def test_the_filtered_requirements_copy_is_unique_and_removed():
    """$PID is shared by every runspace in one PowerShell host, so two installs for
    different Studio homes could rewrite this file while uv was reading the other
    one's copy. And nothing deleted it, on success or failure."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("function Get-ArmFilteredRequirements")
    body = source[index : source.index("\n    }", index)]
    assert "[guid]::NewGuid()" in body
    # The comment above the line explains why $PID was wrong, so look at the code.
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
    assert "$PID" not in code
    assert "$script:ArmFilteredRequirementFiles += $filtered" in body
    # Cleanup removes only what this run wrote: outside the tier the function
    # returns the CALLER's requirements file, and deleting that would take a file
    # out of the installed wheel.
    cleanup = source[source.index("function Remove-ArmFilteredRequirements") :][:400]
    assert "foreach ($path in $script:ArmFilteredRequirementFiles)" in cleanup
    assert source.count("Remove-ArmFilteredRequirements") >= 3  # definition + both call sites


def test_setup_adopts_the_tier_for_an_arm64_venv_with_no_marker():
    """Otherwise `unsloth studio update` dead-ends forever on an environment built
    before the tier existed, or one whose install died before the marker landed:
    the arch gate refuses the venv, and setup.ps1 cannot go and fetch an x64
    interpreter the way install.ps1 can. The desktop app will not download a new
    build until its backend update succeeds, so that loop strands the app too."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index('if (-not $NoDatasetsMode -and (Get-HostMachineArch) -eq "arm64"')
    block = source[index : index + 1200]
    assert '(Get-PythonPlatformTag $ReusedSetupPython) -eq "win-arm64"' in block
    assert "$NoDatasetsMode = $true" in block
    assert '$env:UNSLOTH_NO_DATASETS = "1"' in block  # so the marker is written this time
    # And it says so, with the remedy: this is a reduced install, not a silent one.
    assert "inference-only" in block
    assert "python.org" in block
    # After the marker lookup, or a tier venv would take this path instead.
    assert source.index(".unsloth-no-datasets") < index


def test_setup_reads_the_tier_from_the_manifest_too():
    """The marker can be lost on its own: set_no_datasets_marker swallows OSError,
    and a copy or restore that skipped dotfiles leaves the manifest behind without
    it. Get-PersistedNoTorch already reads the manifest for no-torch; the tier has
    to be readable the same way or an update judges a tier venv by the full rule."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index("$_payload.no_datasets")
    block = source[max(0, index - 900) : index + 700]
    assert "unsloth_install_manifest.json" in block
    assert "$_recorded = (" in block
    assert "$NoDatasetsMode = $_recorded" in block
    # Env var, then manifest, then marker. The variable is this run's explicit
    # request; the manifest records a COMPLETED pass, so its `false` has to be able
    # to override a marker a failed removal left behind; the marker answers alone
    # when a pass died before the manifest was written.
    assert (
        source.index("$NoDatasetsMode = ($null -ne $env:UNSLOTH_NO_DATASETS")
        < index
        < source.index(".unsloth-no-datasets")
    )


def test_setup_prefers_the_manifest_over_a_stale_marker():
    """install_manifest.recorded_no_datasets() reads the manifest first, and the
    manifest is written when a pass COMPLETES: an explicit `no_datasets: false` is
    the record of a finished full install and must override a marker left behind by
    a failed removal or carried along by a copy. Reading the marker first sent an x64
    environment back into the tier on every update, and recorded it as the tier
    again."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index("$_recorded = $null")
    block = source[index : index + 2400]
    assert "unsloth_install_manifest.json" in block
    assert "if ($null -ne $_recorded) {" in block
    # The marker is the fallback, not the first answer: it still has to work alone,
    # since it survives a pass that died before the manifest was written.
    assert block.index("$_recorded = (") < block.index(".unsloth-no-datasets")
