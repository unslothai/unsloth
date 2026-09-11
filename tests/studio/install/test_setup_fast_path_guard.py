# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh / setup.ps1 must not skip the dependency pass on a half-built venv.

Both short-circuit all dependency work when the installed unsloth version equals
PyPI's latest, which is true on an interrupted install: unsloth goes in early and
studio.txt never finishes. So update, and the desktop Repair button behind it,
said "up to date" while the server kept dying on `import structlog`.

That branch only runs for a non-local update, which reinstalls from PyPI and
clobbers the tree under test, so assert the guard structurally instead.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_fast_path_consults_the_install_manifest(script: pathlib.Path):
    text = script.read_text(encoding = "utf-8")
    assert "install_manifest" in text, (
        f"{script.name} no longer consults studio/install_manifest.py. Without it "
        "the 'up to date' fast path skips the dependency pass on an interrupted "
        "install, and `unsloth studio update` becomes a silent no-op."
    )
    assert "verify_install" in text, (
        f"{script.name} must call install_manifest.verify_install() so the check "
        "matches what `unsloth studio verify-install` and the desktop preflight use."
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_guard_can_still_force_the_dependency_pass(script: pathlib.Path):
    """The guard has to clear the skip flag, not merely log a warning."""
    text = script.read_text(encoding = "utf-8")
    if script.name.endswith(".ps1"):
        pattern = r"studio install incomplete[\s\S]{0,200}?\$SkipPythonDeps\s*=\s*\$false"
    else:
        pattern = r"studio install incomplete[\s\S]{0,200}?_SKIP_PYTHON_DEPS=false"
    assert re.search(pattern, text), (
        f"{script.name} detects an incomplete install but does not clear the "
        "skip flag, so the dependency pass would still be skipped."
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_duplicate_core_metadata_cannot_take_the_version_fast_path(script: pathlib.Path):
    text = script.read_text(encoding = "utf-8")
    probe = text.find("install_manifest.installed_version_probe")
    zoo_probe = text.find("'unsloth-zoo'", probe)
    repair = text.find("duplicate metadata found", probe)
    if script.name.endswith(".ps1"):
        skip = text.find("$SkipPythonDeps = $true", repair)
    else:
        skip = text.find("_SKIP_PYTHON_DEPS=true", repair)

    assert probe != -1 and zoo_probe != -1 and repair != -1 and skip != -1
    assert probe <= zoo_probe < repair < skip, (
        f"{script.name} must detect duplicate metadata before an arbitrary "
        "version can select the up-to-date fast path"
    )


def test_ps1_drops_the_manifest_before_its_first_install():
    """Nothing may mutate the venv while the marker still says "install finished".

    install_python_stack.py drops it before its own dependency pass, which is
    enough for setup.sh: the stack is the first thing that pass runs. setup.ps1
    replaces pip, torch and triton first, so a run killed there would leave a
    manifest that still verifies and a venv with half a PyTorch.
    """
    text = SETUP_PS1.read_text(encoding = "utf-8")
    pass_start = text.index("if (-not $SkipPythonDeps) {")
    removal = text.find("remove_manifest", pass_start)
    first_install = text.index("Fast-Install", pass_start)
    stack = text.index(r'python "$PSScriptRoot\install_python_stack.py"', pass_start)

    assert removal != -1, (
        "setup.ps1 never drops the install manifest; install_python_stack.py "
        "only does so after setup.ps1 has already replaced pip and torch"
    )
    assert removal < first_install < stack, (
        "setup.ps1 must invalidate the install manifest before its first "
        "Fast-Install, not leave it to install_python_stack.py"
    )


def test_sh_dependency_pass_mutates_nothing_before_the_stack():
    """setup.sh relies on install_python_stack.py dropping the marker, which only
    holds while the stack is the first thing its dependency pass runs."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    pass_start = text.index('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    body = text[pass_start : text.index("install_python_stack", pass_start)]
    assert "fast_install" not in body and "pip install" not in body, (
        "setup.sh installs something before install_python_stack.py drops the "
        "manifest, so an interrupted run would keep a marker that verifies"
    )


def test_sh_guard_runs_before_the_skip_decision():
    text = SETUP_SH.read_text(encoding = "utf-8")
    guard = text.find("studio install incomplete")
    decision = text.find('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    assert guard != -1 and decision != -1
    assert guard < decision, (
        "the incomplete-install guard must run before setup.sh acts on "
        "_SKIP_PYTHON_DEPS, otherwise it can never change the outcome"
    )


INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"


@pytest.mark.parametrize("script", [INSTALL_SH, INSTALL_PS1], ids = ["install.sh", "install.ps1"])
def test_the_installer_reports_duplicate_metadata_on_every_platform(script: pathlib.Path):
    """Both installers print the version they just installed.

    importlib.metadata.version() answers from whichever record the finder
    yields first, so on a duplicated install it prints an arbitrary one and the
    run looks clean. Windows and POSIX have to agree here, or the same broken
    venv is reported differently depending on the host.
    """
    text = script.read_text(encoding = "utf-8")
    assert "installed_version_probe" in text, (
        f"{script.name} still reports the installed version through "
        "importlib.metadata.version(), which cannot see a duplicate record"
    )
    assert (
        "duplicate metadata found" in text
    ), f"{script.name} detects the conflict but never says so"


def test_the_sidecar_predicate_asks_the_shim_on_colab_too():
    """No venv interpreter on Colab; the shim is stdlib-only and the installer's own
    `python` asks it. The version grep alone read a sidecar interrupted after
    transformers landed as current on every later run."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_sidecar_current() {")
    body = text[start : text.index("\n}\n", start)]
    assert "command -v python" in body
    assert '"$_sc_python" "$SCRIPT_DIR/install_manifest.py" sidecar' in body
    # The grep is the last resort, for a tree with no interpreter to ask at all.
    assert body.index("command -v python") < body.index("_target_has_pkg_version")


def test_the_ps1_sidecar_predicate_runs_the_shim_as_a_bounded_process():
    """Two reasons, one mechanism. The shim answers "stale" with exit 1, which a native
    command turns into a terminating error under $PSNativeCommandUseErrorActionPreference,
    and the shim's scan budget cannot interrupt a stalled read on a wedged mount. A bounded
    process has neither problem; a timeout reads as stale."""
    text = SETUP_PS1.read_text(encoding = "utf-8")
    start = text.index("function Test-SidecarCurrent {")
    body = text[start : text.index("\nfunction ", start + 1)]
    assert "& python $shim" not in body
    assert "Invoke-BoundedPythonProbe -PythonExe $pythonExe -Code $code -TimeoutSec 60" in body
    # The argv travels base64-encoded: a path with quotes or backslashes cannot break -c.
    assert "[Convert]::ToBase64String" in body and "base64.b64decode" in body
    assert "runpy.run_path(sys.argv[0], run_name='__main__')" in body
    assert body.index("$probe.TimedOut") < body.index('$out = "sidecar: audit did not answer')
    # The shell mirror: the shim call is bounded where a timeout exists and a timeout is stale.
    sh = SETUP_SH.read_text(encoding = "utf-8")
    call = sh.index('install_manifest.py" sidecar "$_sc_dir"')
    window = sh[call - 400 : call + 900]
    assert "timeout -k 5 60" in window
    assert '[ "$_sc_rc" -eq 124 ] || [ "$_sc_rc" -eq 137 ]' in window
    assert "sidecar: audit did not answer" in window


def test_the_ps1_sidecar_installs_are_isolated_from_uv_override():
    """setup.sh routes every sidecar install through fast_install_sidecar, which unsets
    UV_OVERRIDE; an override naming huggingface_hub or hf_xet would otherwise install
    another version than the exact pin and the audit would rebuild the sidecar to the
    same wrong answer on every run. The PowerShell helper mirrors it."""
    text = SETUP_PS1.read_text(encoding = "utf-8")
    start = text.index("function Fast-Install-Sidecar {")
    body = text[start : text.index("\nfunction ", start + 1)]
    assert "Remove-Item Env:UV_OVERRIDE" in body and "Fast-Install @Args_" in body
    assert "finally" in body and "$env:UV_OVERRIDE = $savedOverride" in body
    for name in ("function Repair-SidecarTiktoken {", "function Install-T5Sidecar {"):
        start = text.index(name)
        body = text[start : text.index("\nfunction ", start + 1)]
        assert "Fast-Install --target" not in body, name
        assert "Fast-Install-Sidecar --target" in body, name


def test_the_tiktoken_top_up_checks_the_payload_not_the_dist_info_alone():
    """An interrupted install leaves tiktoken-*.dist-info with no package beside it; the
    sidecar predicate accepts that sidecar (tiktoken is optional), so the top-up is the
    only repair left, and a dist-info-only check would skip it forever."""
    sh = SETUP_SH.read_text(encoding = "utf-8")
    start = sh.index("_sidecar_top_up_tiktoken() {")
    assert '"$_stt_dir/tiktoken/__init__.py"' in sh[start : sh.index("\n}\n", start)]
    ps1 = SETUP_PS1.read_text(encoding = "utf-8")
    start = ps1.index("function Repair-SidecarTiktoken {")
    body = ps1[start : ps1.index("\nfunction ", start + 1)]
    assert 'Join-Path $payload "__init__.py"' in body
    # ...and the repair replaces what is there (--target without --upgrade keeps damaged files).
    assert "--no-deps --upgrade tiktoken" in body
    sh_body = sh[sh.index("_sidecar_top_up_tiktoken() {") :]
    assert '--no-deps --upgrade "tiktoken"' in sh_body[: sh_body.index("\n}\n")]


def test_the_ps1_sidecar_predicate_reads_no_version_gated_variable():
    """$PSNativeCommandUseErrorActionPreference exists from PowerShell 7.3 and reading an
    absent variable under Set-StrictMode is a terminating error; the predicate no longer
    touches it at all, and must not grow a version check in its place."""
    ps1 = SETUP_PS1.read_text(encoding = "utf-8")
    start = ps1.index("function Test-SidecarCurrent {")
    body = ps1[start : ps1.index("\nfunction ", start + 1)]
    assert "$PSNativeCommandUseErrorActionPreference =" not in body
    assert "PSVersion.Major -ge 7" not in body


# ── the offline rule ──
#
# "could not reach PyPI, updating to be safe" is the right default: an unreachable PyPI
# is usually a blip, and a pass over a warm cache is cheap. It is the wrong answer when
# the caller has SET UV_OFFLINE, because then every install command in that pass can only
# fail -- the update does the slow half of its work and exits non-zero on a venv that was
# already complete. The rule keeps such an install, and only on the evidence the
# incomplete-install guard already demands.


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_an_unreachable_pypi_still_updates_by_default(script: pathlib.Path):
    """Nothing above changes for a plain offline blip."""
    text = script.read_text(encoding = "utf-8")
    assert text.count('substep "could not reach PyPI, updating to be safe..."') == 1


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_the_offline_rule_needs_all_three_conditions(script: pathlib.Path):
    """An installed version, a declared offline mode, and a verified tree. Any two of
    them is a skip that ships a half-built venv or a venv that was never built."""
    text = script.read_text(encoding = "utf-8")
    if script.name.endswith(".ps1"):
        condition = (
            "if ($InstalledVer -and (Test-UvOfflineRequested) -and "
            "(Test-StudioInstallVerified)) {"
        )
        taken = "$SkipPythonDeps = $true"
    else:
        condition = (
            'if [ -n "$INSTALLED_VER" ] && _uv_offline_requested '
            "&& _setup_install_is_verified; then"
        )
        taken = "_SKIP_PYTHON_DEPS=true"
    assert condition in text, f"{script.name} no longer gates the offline skip on all three"
    start = text.index(condition)
    body = text[start : start + 400]
    assert taken in body
    assert "could not reach PyPI" in body, (
        f"{script.name} lost the else branch, so a host that fails any one of the three "
        "conditions now skips silently instead of updating to be safe"
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_the_two_callers_share_one_definition_of_complete(script: pathlib.Path):
    """The guard forces the pass when the tree is not verified and the offline rule keeps
    it when it is. Two copies of that check is how they come to disagree."""
    text = script.read_text(encoding = "utf-8")
    helper = (
        "function Test-StudioInstallVerified"
        if script.name.endswith(".ps1")
        else "_setup_install_is_verified() {"
    )
    assert helper in text
    assert text.count("install_manifest.verify_install(deep = True)") == 1, (
        f"{script.name} has more than one deep verify; the offline rule and the "
        "incomplete-install guard must ask the same question"
    )


def test_the_posix_offline_switch_reads_the_boolish_spellings(tmp_path):
    """Same spelling UV_NO_CACHE accepts, because a user who set one expects the other
    to be read the same way."""
    import subprocess

    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_uv_offline_requested() {")
    body = text[start : text.index("\n}\n", start) + 3]
    probe = tmp_path / "probe.sh"
    probe.write_text(body + "\nif _uv_offline_requested; then echo yes; else echo no; fi\n")
    for value, expected in (
        ("1", "yes"),
        ("true", "yes"),
        ("TRUE", "yes"),
        ("  yes  ", "yes"),
        ("on", "yes"),
        # uv's boolish parser also takes the single letters; checked against uv 0.10.7,
        # where UV_OFFLINE=t and =y both disable the network.
        ("t", "yes"),
        ("T", "yes"),
        ("y", "yes"),
        ("0", "no"),
        ("false", "no"),
        ("", "no"),
        ("maybe", "no"),
        ("tr", "no"),
    ):
        result = subprocess.run(
            ["sh", str(probe)],
            capture_output = True,
            text = True,
            env = {"PATH": "/usr/bin:/bin", "UV_OFFLINE": value},
        )
        assert result.stdout.strip() == expected, (value, result.stdout)


def test_the_offline_fast_path_never_wipes_a_sidecar():
    """The offline rule keeps the install because nothing can be fetched. A sidecar
    rebuild is a wipe followed by four fetches, so under that rule it would either reach
    for the network or destroy a usable sidecar and then fail. Both shells flag the
    offline keep and clear every rebuild flag behind it."""
    sh = SETUP_SH.read_text(encoding = "utf-8")
    ps1 = SETUP_PS1.read_text(encoding = "utf-8")
    keep_sh = sh.index("keeping the verified install")
    assert "_OFFLINE_FAST_PATH=true" in sh[keep_sh : keep_sh + 400]
    guard_sh = sh.index('if [ "${_OFFLINE_FAST_PATH:-false}" = true ]; then')
    assert (
        sh.index('_sidecar_current "$VENV_T5_510_DIR"')
        < guard_sh
        < sh.index('if [ "$_NEED_T5_530" = true ]; then')
    )
    assert 'eval "_NEED_T5_$1=false"' in sh[guard_sh : guard_sh + 900]
    keep_ps1 = ps1.index("keeping the verified install")
    assert "$script:OfflineFastPath = $true" in ps1[keep_ps1 : keep_ps1 + 400]
    guard_ps1 = ps1.index("if ($script:OfflineFastPath) {\n    foreach ($tier in")
    assert (
        ps1.index("Test-SidecarCurrent -TargetDir $VenvT5_510Dir")
        < guard_ps1
        < ps1.index("if ($_NeedT5_530 -or $_NeedT5_550 -or $_NeedT5_510) {")
    )
    assert "Set-Variable -Name $flag -Value $false" in ps1[guard_ps1 : guard_ps1 + 900]
    # The legacy migration is itself a wipe, and it sits above the guard; under the
    # offline keep it has to be skipped, not merely followed by cleared flags.
    # ...and under UV_OFFLINE without the fast path as well: the legacy tree is the only
    # sidecar the install has, and the three rebuilds would come from a cache that may be cold.
    assert sh.index(
        '[ "${_OFFLINE_FAST_PATH:-false}" = true ] || _uv_offline_requested; }; then\n    # The migration'
    ) < sh.index('rm -rf "$STUDIO_HOME/.venv_t5"')
    assert ps1.index(
        "(Test-Path -LiteralPath $VenvT5Legacy) -and ($script:OfflineFastPath -or (Test-UvOfflineRequested))"
    ) < ps1.index("Remove-Item -LiteralPath $VenvT5Legacy -Recurse -Force")
    # ...and the tiktoken top-up a current tier gets must not run either: its pip
    # fallback reaches the network, and a deferred tier would gain a tiktoken-only dir.
    top_up = sh[sh.index("_sidecar_top_up_tiktoken() {") :]
    assert '[ "${_OFFLINE_FAST_PATH:-false}" = true ] && return 0' in top_up[:600]
    repair = ps1[ps1.index("function Repair-SidecarTiktoken {") :]
    assert "if ($script:OfflineFastPath) { return }" in repair[:600]


def test_uv_offline_without_the_fast_path_still_keeps_an_existing_sidecar():
    """UV_OFFLINE with the core not verified (or PyPI still answering) takes the ordinary
    path, whose sidecar rebuild is a wipe followed by four fetches from a cache that may
    be cold, and an absent tier would go through the pip fallback that does not read
    UV_OFFLINE. Both shells defer every stale or missing tier under the offline request
    itself, ahead of the fast-path guard; the runtime self-heal covers a missing tier."""
    sh = SETUP_SH.read_text(encoding = "utf-8")
    ps1 = SETUP_PS1.read_text(encoding = "utf-8")
    offline_sh = sh.index(
        'if [ "${_OFFLINE_FAST_PATH:-false}" != true ] && _uv_offline_requested; then'
    )
    assert (
        sh.index('_sidecar_current "$VENV_T5_510_DIR"')
        < offline_sh
        < sh.index('if [ "${_OFFLINE_FAST_PATH:-false}" = true ]; then')
    )
    block_sh = sh[offline_sh : offline_sh + 1100]
    # Stale AND missing: an absent tier would reach fast_install's pip fallback, which
    # does not read UV_OFFLINE. No path in the loop's word list either (a Studio home
    # with a space in its path would split there).
    assert '[ -d "$' not in block_sh.split("for _ofp in", 1)[1].split("done", 1)[0]
    assert "$VENV_T5_530_DIR" not in block_sh.split("for _ofp in", 1)[1].split("\n", 1)[0]
    assert 'eval "_NEED_T5_$1=false"' in block_sh and 'eval "_DEFER_T5_$1=true"' in block_sh
    offline_ps1 = ps1.index("if (-not $script:OfflineFastPath -and (Test-UvOfflineRequested)) {")
    assert (
        ps1.index("Test-SidecarCurrent -TargetDir $VenvT5_510Dir")
        < offline_ps1
        < ps1.index("if ($script:OfflineFastPath) {\n    foreach ($tier in")
    )
    block_ps1 = ps1[offline_ps1 : offline_ps1 + 1100]
    assert "Test-Path -LiteralPath $tier[2]" not in block_ps1
    assert "Set-Variable -Name $flag -Value $false" in block_ps1
    # The tiktoken top-up stays home under the offline request too.
    top_up = sh[sh.index("_sidecar_top_up_tiktoken() {") :]
    assert "_uv_offline_requested && return 0" in top_up[:900]
    repair = ps1[ps1.index("function Repair-SidecarTiktoken {") :]
    assert "if (Test-UvOfflineRequested) { return }" in repair[:900]


def test_the_ps1_offline_flag_is_initialised_before_its_unconditional_reads():
    """Only the offline keep assigns the flag, and the sidecar block reads it on every
    update. Under a caller's Set-StrictMode an unassigned script variable is a
    terminating error, and a dot-sourced rerun would otherwise inherit an earlier
    offline run's $true."""
    text = SETUP_PS1.read_text(encoding = "utf-8")
    init = text.index("$script:OfflineFastPath = $false")
    assert init < text.index("$script:OfflineFastPath = $true")
    assert init < text.index("if ($script:OfflineFastPath)")
