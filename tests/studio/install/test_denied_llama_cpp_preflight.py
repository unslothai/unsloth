# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check managed llama.cpp access before Windows installer work.

The standalone installer copies setup.ps1's preflight helpers. These tests enforce
their parity and cover real denied trees through the PowerShell harness.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = (ROOT / "install.ps1").read_text(encoding = "utf-8")
SETUP_PS1 = (ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
SETUP_SH = (ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
ACL_PS1 = (ROOT / "tests" / "studio" / "test_path_probe_access_denied.ps1").read_text(
    encoding = "utf-8"
)

SHARED_BEGIN = "# ── BEGIN SHARED WITH studio/setup.ps1 ──"
SHARED_END = "# ── END SHARED WITH studio/setup.ps1 ──"

# Helpers copied byte-for-byte into install.ps1.
SHARED_FUNCTIONS = (
    "Test-AccessDeniedError",
    "Get-PathState",
    "Get-LlamaCppInstallReadState",
    "Get-PathDenialDetail",
    "Get-SecuritySoftwareNote",
    "Write-PathAccessDenied",
    "Get-CanonicalDir",
    "Test-StudioHomeIsCustom",
    "Get-ManagedLlamaCppDir",
    "Invoke-ManagedLlamaCppPreflight",
)


def _function_source(text: str, name: str) -> str:
    """Extract a PowerShell function by matching balanced braces."""
    match = re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", text)
    assert match, f"{name} is not defined"
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _normalized(source: str) -> str:
    """Remove only the common indentation before comparing helper copies."""
    lines = [line.rstrip() for line in source.splitlines()]
    body = [line for line in lines if line.strip()]
    indent = min(len(line) - len(line.lstrip(" ")) for line in body)
    return "\n".join(line[indent:] if line.strip() else "" for line in lines)


@pytest.mark.parametrize("name", SHARED_FUNCTIONS)
def test_installer_copy_matches_setup(name: str) -> None:
    """Edit one file, not the other, and this fails naming the function."""
    assert (
        _normalized(_function_source(INSTALL_PS1, name))
        == _normalized(_function_source(SETUP_PS1, name))
    ), f"{name} differs between install.ps1 and studio/setup.ps1; run python3 scripts/sync_shared_ps1_helpers.py"


def test_every_function_in_the_shared_block_is_compared() -> None:
    """The parity list cannot silently fall behind the block it guards."""
    assert SHARED_BEGIN in INSTALL_PS1
    assert SHARED_END in INSTALL_PS1
    block = INSTALL_PS1.split(SHARED_BEGIN, 1)[1].split(SHARED_END, 1)[0]
    declared = re.findall(r"(?m)^[ \t]*function[ \t]+([A-Za-z-]+)", block)
    assert sorted(declared) == sorted(SHARED_FUNCTIONS), declared


def test_the_shared_block_carries_the_drift_note() -> None:
    """A reader who does not know these are copies will paraphrase one of them."""
    head = INSTALL_PS1.split(SHARED_BEGIN, 1)[0].rsplit("\n\n", 3)[-1]
    assert "byte-identical copies" in head
    assert "test_denied_llama_cpp_preflight.py" in head
    assert "cannot dot-source" in head
    setup_note = SETUP_PS1.split("function Get-LlamaCppInstallReadState", 1)[0]
    assert "install.ps1 carries a verbatim copy" in setup_note


def test_setup_and_the_installer_use_the_same_probe() -> None:
    """Both Windows entrypoints must use the shared tri-state probe."""
    assert "$llamaDirState = Get-LlamaCppInstallReadState -Path $LlamaCppDir" in SETUP_PS1
    assert '$llamaDirState -eq "Denied"' in SETUP_PS1
    assert '$llamaDirState -eq "Readable"' in SETUP_PS1
    assert '(Get-LlamaCppInstallReadState -Path $dir) -ne "Denied"' in INSTALL_PS1
    assert (
        "$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight -StagingRoot $StageRoot"
        in SETUP_PS1
    )


def test_the_probe_keeps_all_three_answers() -> None:
    """Keep denied, absent, and readable states distinct."""
    probe = _function_source(SETUP_PS1, "Get-LlamaCppInstallReadState")
    for verdict in ('return "Denied"', 'return "Absent"', 'return "Readable"'):
        assert verdict in probe, verdict
    # Listing catches denied directories whose missing marker appears absent.
    assert "Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop" in probe
    assert "Test-AccessDeniedError" in probe
    # A readable marker is not enough: replacement also needs directory listing.
    # Regex, not a literal: whitespace alone must not reintroduce the early return.
    assert not re.search(r'"Present"\s*\{\s*return\s+"Readable"', probe)
    assert probe.index("Get-ChildItem -LiteralPath") < probe.rindex('return "Readable"')
    # It runs before anything is installed, so it must not terminate.
    assert probe.count("try {") == 1
    assert "catch" in probe


def test_the_preflight_runs_before_anything_expensive() -> None:
    """The whole fix is the ordering. Every step below costs network, disk or both."""
    call = "$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight"
    assert INSTALL_PS1.count(call) == 1
    position = INSTALL_PS1.index(call)
    for later in (
        'Write-TauriLog "STEP" "Checking system dependencies"',
        'Write-TauriLog "STEP" "Installing Python"',
        'Write-TauriLog "STEP" "Installing uv package manager"',
        'Write-TauriLog "STEP" "Creating virtual environment"',
        'Write-TauriLog "STEP" "Installing PyTorch"',
        'Write-TauriLog "STEP" "Installing unsloth"',
        'Write-TauriLog "STEP" "Running studio setup"',
    ):
        assert position < INSTALL_PS1.index(later), later
    # Relocation decides which user profile owns the managed cache.
    relocation = "# ── Leave Windows system directories before installing ──"
    assert INSTALL_PS1.index(relocation) < position


def test_direct_setup_and_update_preflight_before_phase_one() -> None:
    """Direct setup, update, and repair must preflight before phase one."""
    call = "$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight"
    assert SETUP_PS1.count(call) == 1
    position = SETUP_PS1.index(call)
    assert SETUP_PS1.index("$LlamaCppDir = Get-ManagedLlamaCppDir") < position
    for later in (
        "PHASE 1: System-level prerequisites",
        "PHASE 2: Frontend build",
        "PHASE 3: Python environment + dependencies",
        "PHASE 3.4: Prefer prebuilt llama.cpp",
    ):
        assert position < SETUP_PS1.index(later), later
    failure = SETUP_PS1[position : SETUP_PS1.index("# Back up User PATH", position)]
    assert "Exit-SetupFailure $llamaPreflightFailure" in failure


def test_acl_suite_runs_every_complete_windows_entrypoint() -> None:
    """Windows CI must run every entrypoint and trap expensive work."""
    assert '& (Join-Path $repoRoot "install.ps1") --tauri' in ACL_PS1
    assert '& (Join-Path $repoRoot "studio/setup.ps1")' in ACL_PS1
    assert 'foreach ($mode in @("install", "setup", "update", "repair"))' in ACL_PS1
    assert "icacls $entryLocked /deny" in ACL_PS1
    assert "else { chmod 000 $entryLocked }" in ACL_PS1
    for trap in (
        "Invoke-WebRequest",
        "Invoke-RestMethod",
        "Start-Process",
        "winget",
        "python",
        "uv",
        "git",
        "npm",
    ):
        assert f'function global:{trap} {{ Stop-EntrypointExpense "{trap}" }}' in ACL_PS1
    for marker in (
        "Checking system dependencies",
        "frontend",
        "Installing Python",
        "Installing uv package manager",
        "Creating virtual environment",
        "Installing PyTorch",
        "Installing unsloth",
        "Unsloth Studio Installed",
    ):
        assert marker in ACL_PS1
    assert ".unsloth-studio-owned" in ACL_PS1
    assert "unsloth_install_manifest.json" in ACL_PS1


def test_the_preflight_fails_the_install_with_the_shared_reason() -> None:
    """The shared reason must reach the desktop app."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    assert 'Write-PathAccessDenied -Path $dir -Label "llama.cpp install"' in body
    assert "Nothing was installed." in body
    call = INSTALL_PS1.split("$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight", 1)[1]
    call = call.split("# ── Check winget ──", 1)[0]
    assert "Exit-InstallFailure $llamaPreflightFailure" in call


def test_the_preflight_cannot_be_the_thing_that_breaks_the_run() -> None:
    """The early preflight must tolerate an unavailable profile or path."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    guard = "if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { return $null }"
    assert guard in body
    assert body.index(guard) < body.index("Get-ManagedLlamaCppDir")
    # Both probes swallow their own failures rather than terminating.
    probe = _function_source(INSTALL_PS1, "Get-LlamaCppInstallReadState")
    assert "-ErrorAction Stop" in probe and "catch" in probe
    assert "Get-PathState" in probe


def test_a_custom_studio_home_is_never_called_a_cache_we_own() -> None:
    """Do not call an unreadable custom Unsloth home a managed cache."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    # Use the same predicate for path selection and ownership wording.
    assert "$homeIsCustom = Test-StudioHomeIsCustom" in body
    assert "-OwnershipUnverified:$homeIsCustom" in body
    assert (
        'Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install"'
        " -OwnershipUnverified:$StudioHomeIsCustom" in SETUP_PS1
    )


def test_a_tree_the_user_pointed_at_is_never_called_a_cache_we_own() -> None:
    """Preserve user-supplied wording when an override names the managed path."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    assert "-UserSupplied:$userSupplied" in body
    assert (
        "$suppliedDir = if ($WithLlamaCppDir) { $WithLlamaCppDir }"
        " else { $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR }" in body
    )
    # Compare canonical paths, including denied paths whose spelling differs,
    # and every ancestor of the override: a build supplied from inside the
    # managed tree is the user's too.
    assert "$probe = [string](Get-CanonicalDir -Path $suppliedDir)" in body
    assert "if ([string](Get-CanonicalDir -Path $probe) -eq $canonicalDir) {" in body
    assert "$LocalIsCanonical = ($ResolvedLocal -eq $LlamaCppDir)" in SETUP_PS1
    assert (
        "Exit-PathAccessDenied -Path $ResolvedLocal"
        ' -Label "the UNSLOTH_LOCAL_LLAMA_CPP_DIR build" -UserSupplied' in SETUP_PS1
    )


def test_the_managed_path_rule_is_not_duplicated_in_the_installer() -> None:
    """Keep managed path selection in one resolver."""
    resolver = _function_source(INSTALL_PS1, "Get-ManagedLlamaCppDir")
    assert "param([AllowNull()][string]$StagingRoot = $null)" in resolver
    assert 'Join-Path $StagingRoot "llama.cpp"' in resolver
    assert "$StageRoot" not in resolver
    preflight = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    assert "param([AllowNull()][string]$StagingRoot = $null)" in preflight
    assert "Get-ManagedLlamaCppDir -StagingRoot $StagingRoot" in preflight
    assert "$StageRoot" not in preflight
    assert 'Join-Path $env:USERPROFILE ".unsloth\\llama.cpp"' in resolver
    assert 'Join-Path (Get-CanonicalDir -Path $StudioHome) "llama.cpp"' in resolver
    assert INSTALL_PS1.count('Join-Path $StudioHome "llama.cpp"') == 0
    assert INSTALL_PS1.count("$_llamaPath = Get-ManagedLlamaCppDir") == 1
    # The resolver uses the single default-versus-custom predicate.
    assert "if (-not (Test-StudioHomeIsCustom)) {" in resolver
    assert "$legacyStudio" not in resolver
    predicate = _function_source(INSTALL_PS1, "Test-StudioHomeIsCustom")
    assert predicate.count("Get-CanonicalDir -Path") == 2
    canonicalizer = _function_source(INSTALL_PS1, "Get-CanonicalDir")
    assert "Resolve-Path -LiteralPath $trimmedPath" in canonicalizer
    # A denied path cannot resolve, so compare lexical full paths instead.
    assert "GetUnresolvedProviderPathFromPSPath" in canonicalizer
    assert "[System.IO.Path]::GetFullPath(" in canonicalizer
    # One trim, after both branches: Resolve-Path keeps a trailing separator too.
    assert canonicalizer.count("TrimEnd('\\', '/')") == 1
    assert canonicalizer.index("Resolve-Path") < canonicalizer.index("TrimEnd")
    assert INSTALL_PS1.count("Resolve-Path -LiteralPath $trimmedPath") == 1


def test_both_entrypoints_resolve_and_reuse_the_same_managed_directory() -> None:
    """Both entrypoints must resolve and reuse one managed path."""
    assert '$LegacyStudioHome = Join-Path $env:USERPROFILE ".unsloth\\studio"' in SETUP_PS1
    assert "$StudioHomeIsCustom = Test-StudioHomeIsCustom" in SETUP_PS1
    assert SETUP_PS1.count("$LlamaCppDir = Get-ManagedLlamaCppDir -StagingRoot $StageRoot") == 1
    assert "$UnslothHome = Split-Path -Parent $LlamaCppDir" in SETUP_PS1
    for name in (
        "Get-CanonicalDir",
        "Test-StudioHomeIsCustom",
        "Get-ManagedLlamaCppDir",
    ):
        assert _normalized(_function_source(INSTALL_PS1, name)) == _normalized(
            _function_source(SETUP_PS1, name)
        )
    phase = SETUP_PS1.split("PHASE 3.4", 1)[1].split("$NeedLlamaSourceBuild", 1)[0]
    assert "resolved and preflighted before phase 1" in phase
    assert "$LlamaCppDir =" not in phase


def test_the_installer_never_repairs_permissions_by_itself() -> None:
    """Print ACL repair commands but never run them."""
    # Match direct, chained, captured, and delegated invocation forms.
    invocation = re.compile(
        r"(^|[&|;=]\s*|\(\s*|Start-Process\s+|Invoke-Expression\s+)(takeown|icacls)\b"
    )
    for text, label in ((INSTALL_PS1, "install.ps1"), (SETUP_PS1, "setup.ps1")):
        for line in text.splitlines():
            code = line.split("#", 1)[0].strip()
            if "takeown" not in code and "icacls" not in code:
                continue
            assert not invocation.search(code), f"{label}: {line.strip()}"


def test_setup_sh_reports_a_denied_default_home_cache() -> None:
    """The POSIX prebuilt path must report a denied default cache."""
    block = SETUP_SH.split('substep "installing prebuilt llama.cpp..."', 1)[1]
    block = block.split("_PREBUILT_CMD=(", 1)[0]
    # Listing, not just search: mode 111 passes cd and still breaks the installer.
    assert 'if _studio_dir_unreadable "$LLAMA_CPP_DIR"; then' in block
    assert '_path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"' in block
    # Preserve the custom-home ownership guard's more cautious wording.
    assert block.index("_assert_studio_owned_or_absent") < block.index("_studio_dir_unreadable")


def test_the_denial_detail_survives_a_directory_it_cannot_open() -> None:
    """Get-Item returns nothing for a directory whose ACL denies read, which is the
    one case this detail exists for, so the attributes have to come through an API
    that needs only FILE_READ_ATTRIBUTES."""
    for text in (INSTALL_PS1, SETUP_PS1):
        body = _function_source(text, "Get-PathDenialDetail")
        assert "[System.IO.File]::GetAttributes($Path)" in body
        # Get-Item may still run, but only after the attributes are in hand and
        # only to name a link target, which is the one thing it adds.
        assert body.index("GetAttributes($Path)") < body.index("Get-Item -LiteralPath")
        # The three causes elevation cannot fix, each named rather than folded
        # into the takeown and icacls advice that cannot clear them.
        assert "Encrypted" in body
        assert "cloud placeholder" in body
        assert "ReparsePoint" in body
        # Every caller passes a directory, where Encrypted only means new
        # descendants are encrypted by default. Listing never needs the key, so
        # claiming EFS there sends an ordinary ACL denial after a certificate.
        assert body.index("FileAttributes]::Directory") < body.index("FileAttributes]::Encrypted")
        assert "-not $isDirectory" in body
        # RECALL_ON_OPEN and RECALL_ON_DATA_ACCESS are missing from the
        # FileAttributes enum on Windows PowerShell 5.1.
        assert "0x00040000" in body and "0x00400000" in body


def test_the_security_software_holding_the_folder_is_named() -> None:
    """takeown and icacls cannot clear a filter-driver block, and neither can
    elevation, so the Defender mode that blocks file access, and any third-party
    antivirus registered instead, are called out by name. A user told only that
    "antivirus can deny this" cannot tell which product to open."""
    for text in (INSTALL_PS1, SETUP_PS1):
        body = _function_source(text, "Get-SecuritySoftwareNote")
        # Neither branch correlates the block with this path, so neither may say
        # the ACL repair printed above it is pointless. Controlled folder access
        # gates writes, and a registered antivirus need not be involved at all.
        assert "will not help" not in body
        # A product that is not running cannot be holding the folder, and naming
        # it sends the user to the wrong console: 0xF000 carries the run state.
        assert "productState" in body and "0xF000" in body
        # Name the log that can actually attribute a Defender block.
        assert body.count("1123 and 1124") == 2
        # Absent Defender must read the same as a Defender that says no.
        assert "Get-Command Get-MpPreference -ErrorAction SilentlyContinue" in body
        # Every failure to tell must answer "" rather than guess: no Defender
        # module, no SecurityCenter registration, and a query that throws all
        # read the same as a machine that says no.
        assert "catch { $mode = $null }" in body
        assert "catch { $others = @() }" in body
        assert _normalized(body).rstrip().endswith('return ""\n}')
        # 1 Enabled blocks file access. 3 and 4 are direct disk-sector writes, so
        # they cannot explain a denied folder and must not be reported as if they
        # could. 2 AuditMode logs instead of blocking, so it rules itself out.
        assert "$mode -eq 1" in body
        assert "$mode -eq 2" in body
        assert "$mode -eq 3" not in body and "$mode -eq 4" not in body
        # Third-party suites ship the same feature under their own names, and
        # SecurityCenter2 is the registration all of them make.
        assert "root/SecurityCenter2" in body
        assert "AntiVirusProduct" in body
        # Defender registers there too, and it is covered by the mode check.
        assert "Windows Defender" in body and "Microsoft Defender" in body
        reporter = _function_source(text, "Write-PathAccessDenied")
        assert "Get-SecuritySoftwareNote" in reporter
        # After the generic line, so an empty answer leaves it standing.
        assert reporter.index("Antivirus or Controlled folder access") < reporter.index(
            "$securitySoftware = Get-SecuritySoftwareNote"
        )


def test_a_denied_cache_is_moved_aside_only_when_it_is_ours_to_move() -> None:
    """Renaming needs DELETE on the folder plus write on its parent, neither of
    which is read access, so it recovers denials takeown and icacls do not. It is
    only allowed for the tree the guidance already tells the user to delete."""
    for text in (INSTALL_PS1, SETUP_PS1):
        body = _function_source(text, "Invoke-ManagedLlamaCppPreflight")
        assert "Move-Item -LiteralPath $dir -Destination $asideDir" in body
        guard = "if (-not $userSupplied -and -not $homeIsCustom -and -not $isLink) {"
        assert guard in body
        # Setup never makes this a link, so a link is the user's own arrangement
        # and moving it would change which tree they run.
        assert body.index("ReparsePoint") < body.index(guard)
        # Unreadable attributes prove nothing, which must not mean movable.
        assert "$isLink = $true" in body
        # A failed move falls through to the guidance rather than stopping.
        assert body.index(guard) < body.index("Write-PathAccessDenied -Path $dir")
        # A build supplied from inside this tree goes with it, and the later
        # --with-llama-cpp-dir check then aborts on a path we made disappear, so
        # containment counts as user-supplied and not just an exact match.
        assert "Split-Path -Parent $probe" in body
        assert body.index("$probe = $parent") < body.index(guard)
        # This runs before the install lock, so a second run can move the folder
        # first; reporting a denial for a path that is gone stops an install
        # that could have carried on.
        reprobe = '(Get-LlamaCppInstallReadState -Path $dir) -ne "Denied"'
        assert body.count(reprobe) == 2
        assert body.rindex(reprobe) < body.index("Write-PathAccessDenied -Path $dir")


def test_the_shared_helpers_have_a_sync_script() -> None:
    """Hand-copying ten helpers between two files is how they drifted before."""
    script = (ROOT / "scripts" / "sync_shared_ps1_helpers.py").read_text(encoding = "utf-8")
    assert SHARED_BEGIN in script
    assert "--check" in script


def test_a_denied_node_cache_gets_the_same_guidance_as_the_llama_cache() -> None:
    """The same denial reaches the Node cache, where it used to read "unexpected
    error" and then "install Node yourself, or check your network". Neither is
    the fix, and a user whose antivirus holds the folder can spend a long time on
    the second one."""
    node = (ROOT / "studio" / "install_node_prebuilt.py").read_text(encoding = "utf-8")
    assert "EXIT_DENIED = 4" in node
    assert "except PermissionError as exc:" in node
    # Windows does not always deliver winerror 5 as PermissionError.
    assert 'getattr(exc, "winerror", None) == 5' in node
    assert "errno.EACCES" in node
    assert "import errno" in node
    # The catch-all must stay last, or the classification never runs.
    tail = node[node.rindex("def main(") :]
    assert tail.index("except PermissionError") < tail.index("except Exception as exc:")
    # setup.ps1 turns the new code into the shared guidance rather than the
    # nodejs.org and network advice that follows every other nonzero exit.
    caller = SETUP_PS1.split("install_node_prebuilt.py", 1)[1].split("$env:PATH = ", 1)[0]
    assert "$nodeExit -eq 4" in caller
    assert 'Exit-PathAccessDenied -Path $NodeDir -Label "Node install"' in caller
    assert caller.index("$nodeExit -eq 4") < caller.index("https://nodejs.org/")
    # The install lock and the .staging root live in the parent, so half of the
    # denials that reach exit 4 are not on the cache at all. Deleting the cache
    # cannot make a parent writable, and the parent is never ours to offer up.
    assert 'DENIED_SCOPE_MARKER = "denied-scope: "' in node
    assert 'DENIED_SCOPE_PARENT = "parent"' in node
    assert '$nodeOut -match "denied-scope: parent"' in caller
    assert "Exit-PathAccessDenied -Path $NodeParent" in caller
    assert "-OwnershipUnverified" in caller
