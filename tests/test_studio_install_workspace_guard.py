"""install.sh/install.ps1 must refuse to rm -rf an existing Unsloth venv in env-mode without a sentinel."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"

# Stub the rollback helper with a move. The directory predicate is extracted
# from install.sh because it controls whether the guard runs.
_INSTALL_GUARD_STUBS = (
    'substep() { :; }\n_start_studio_venv_replacement() {\n    mv -- "$1" "$1.replaced"\n}\n'
)


def _extract_install_sh_function(name: str) -> str:
    """Extract a top-level install.sh shell function, header line to closing brace."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?\n\}}\n", src, re.DOTALL | re.MULTILINE)
    assert m, f"install.sh function {name} not found"
    return m.group(0)


def _extract_install_sh_guard_block() -> str:
    """Extract install.sh's venv guard block (up to the first elif) as a self-contained snippet."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    m = re.search(
        r'(if \[ -x "\$VENV_DIR/bin/python" \] \|\| _dir_has_entries "\$VENV_DIR"; then\n.*?)'
        r'elif \[ "\$_STUDIO_HOME_REDIRECT" != "env"',
        src,
        re.DOTALL,
    )
    assert m, "install.sh venv guard block not found"
    return m.group(1) + "fi\n"


def _build_install_guard_script(
    studio_home: Path,
    redirect: str,
    block: str | None = None,
) -> str:
    """Build a self-contained bash script exercising the extracted guard block (with helper stubs)."""
    if block is None:
        block = _extract_install_sh_guard_block()
    return (
        _INSTALL_GUARD_STUBS
        + _extract_install_sh_function("_dir_has_entries")
        + f'STUDIO_HOME="{studio_home}"\n'
        + f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        + f'_STUDIO_HOME_REDIRECT="{redirect}"\n'
        + block
        + "echo RESULT=ok\n"
    )


def _run_install_guard(
    studio_home: Path,
    redirect: str,
    create_share_conf: bool = False,
    create_bin_shim: bool = False,
    create_venv_marker: bool = False,
) -> subprocess.CompletedProcess:
    venv_dir = studio_home / "unsloth_studio"
    (venv_dir / "bin").mkdir(parents = True, exist_ok = True)
    py = venv_dir / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    if create_share_conf:
        (studio_home / "share").mkdir(parents = True, exist_ok = True)
        (studio_home / "share" / "studio.conf").write_text("")
    if create_bin_shim:
        (studio_home / "bin").mkdir(parents = True, exist_ok = True)
        (studio_home / "bin" / "unsloth").write_text("")
    if create_venv_marker:
        (venv_dir / ".unsloth-studio-owned").write_text("")
    script = _build_install_guard_script(studio_home, redirect)
    return subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )


def test_env_mode_blocks_unsloth_studio_without_sentinels(tmp_path):
    studio_home = tmp_path / "ws"
    res = _run_install_guard(studio_home, redirect = "env")
    assert res.returncode != 0, (
        "env-mode without sentinels must refuse to rm -rf $VENV_DIR; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "does not look like an Unsloth Studio install" in res.stderr
    assert (studio_home / "unsloth_studio" / "bin" / "python").is_file()


def test_env_mode_passes_when_share_studio_conf_present(tmp_path):
    studio_home = tmp_path / "ws"
    res = _run_install_guard(studio_home, redirect = "env", create_share_conf = True)
    assert res.returncode == 0, (
        f"share/studio.conf sentinel must allow cleanup;"
        f" stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "RESULT=ok" in res.stdout
    assert not (studio_home / "unsloth_studio").exists()


def test_env_mode_passes_when_bin_unsloth_shim_present(tmp_path):
    studio_home = tmp_path / "ws"
    res = _run_install_guard(studio_home, redirect = "env", create_bin_shim = True)
    assert res.returncode == 0, res.stderr
    assert not (studio_home / "unsloth_studio").exists()


def test_default_mode_skips_sentinel_check(tmp_path):
    studio_home = tmp_path / "ws"
    res = _run_install_guard(studio_home, redirect = "default")
    assert res.returncode == 0, res.stderr
    assert "RESULT=ok" in res.stdout
    assert not (studio_home / "unsloth_studio").exists()


def test_install_ps1_has_matching_env_mode_guard():
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    block_start = src.index("# why: matching guard to the .venv branch below")
    block = src[block_start : block_start + 2000]
    assert (
        "$StudioRedirectMode -eq 'env'" in block
    ), "install.ps1 must gate Remove-Item $VenvDir on env-mode"
    assert "share\\studio.conf" in block, "install.ps1 guard must check share\\studio.conf sentinel"
    assert "bin\\unsloth.exe" in block, "install.ps1 guard must check bin\\unsloth.exe sentinel"
    assert "Refusing to delete non-Unsloth venv" in block


def test_setup_ps1_has_writability_probe():
    src = SETUP_PS1.read_text(encoding = "utf-8")
    idx = src.index("if (Test-Path -LiteralPath $_studioOverride -PathType Container)")
    block = src[idx : idx + 2000]
    assert (
        "WriteAllText" in block
    ), "setup.ps1 must write-probe UNSLOTH_STUDIO_HOME like setup.sh:417"
    assert (
        "is not writable" in block
    ), "setup.ps1 probe failure must produce a clear writable-error message"


def test_env_mode_blocks_when_bin_unsloth_is_a_directory(tmp_path):
    """A bare directory at bin/unsloth must NOT pass the sentinel (regression: `-e` accepted any type)."""
    studio_home = tmp_path / "ws"
    venv = studio_home / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    (venv / "important.txt").write_text("keep me")
    (studio_home / "bin" / "unsloth").mkdir(parents = True)
    script = _build_install_guard_script(studio_home, "env")
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert res.returncode != 0, (
        "directory at bin/unsloth must NOT satisfy the Unsloth sentinel; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert (venv / "important.txt").is_file(), "unrelated workspace data must survive"


def test_env_mode_passes_when_bin_unsloth_is_a_symlink(tmp_path):
    """A symlink at bin/unsloth (real installer artefact) must still satisfy the sentinel."""
    studio_home = tmp_path / "ws"
    venv = studio_home / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    (studio_home / "bin").mkdir(parents = True)
    target = studio_home / "bin" / "unsloth-real"
    target.write_text("#!/bin/sh\nexit 0\n")
    target.chmod(0o755)
    (studio_home / "bin" / "unsloth").symlink_to(target)
    script = _build_install_guard_script(studio_home, "env")
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert res.returncode == 0, res.stderr
    assert "RESULT=ok" in res.stdout
    assert not venv.exists()


def test_install_ps1_sentinel_uses_pathtype_leaf():
    """Remove-Item $VenvDir gate must use -PathType Leaf so a sentinel-path directory cannot satisfy it."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    block_start = src.index("# why: matching guard to the .venv branch below")
    block = src[block_start : block_start + 2000]
    assert (
        'share\\studio.conf") -PathType Leaf' in block
    ), "install.ps1 share\\studio.conf check must use -PathType Leaf"
    assert (
        'bin\\unsloth.exe") -PathType Leaf' in block
    ), "install.ps1 bin\\unsloth.exe check must use -PathType Leaf"


def test_setup_ps1_stale_venv_has_env_mode_guard():
    """setup.ps1 stale-venv branch must gate Remove-Item $VenvDir on a custom-root Unsloth sentinel."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    idx = src.index("Stale venv detected")
    block = src[idx : idx + 1500]
    assert (
        "$StudioHomeIsCustom" in block
    ), "setup.ps1 stale-venv branch must gate on $StudioHomeIsCustom"
    assert (
        'share\\studio.conf") -PathType Leaf' in block
    ), "setup.ps1 stale-venv guard must check share\\studio.conf with -PathType Leaf"
    assert (
        'bin\\unsloth.exe") -PathType Leaf' in block
    ), "setup.ps1 stale-venv guard must check bin\\unsloth.exe with -PathType Leaf"
    # The guard must fire BEFORE the destructive call.
    guard_idx = block.index("$StudioHomeIsCustom")
    rm_idx = block.index("Remove-Item -LiteralPath $VenvDir")
    assert guard_idx < rm_idx, "custom-root guard must precede Remove-Item -LiteralPath $VenvDir"


def test_setup_sh_prebuilt_llama_cpp_has_ownership_guard():
    """setup.sh prebuilt llama.cpp path must _assert_studio_owned_or_absent before install_llama_prebuilt.py."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    idx = src.index("installing prebuilt llama.cpp...")
    block = src[idx : idx + 2000]
    assert (
        '_assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"' in block
    ), "setup.sh must guard the prebuilt llama.cpp path with the ownership marker"
    guard_idx = block.index('_assert_studio_owned_or_absent "$LLAMA_CPP_DIR"')
    # Anchor on the actual command-array entry, not the why-comment mention.
    helper_idx = block.index('python "$SCRIPT_DIR/install_llama_prebuilt.py"')
    assert guard_idx < helper_idx, "ownership guard must precede the install_llama_prebuilt.py call"


def test_setup_ps1_prebuilt_llama_cpp_has_ownership_guard():
    """setup.ps1 prebuilt llama.cpp path must Assert-StudioOwnedOrAbsent before install_llama_prebuilt.py."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    idx = src.index("installing prebuilt llama.cpp bundle (preferred path)")
    block = src[idx : idx + 2000]
    assert (
        'Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"' in block
    ), "setup.ps1 must guard the prebuilt llama.cpp path with Assert-StudioOwnedOrAbsent"
    guard_idx = block.index("Assert-StudioOwnedOrAbsent -Path $LlamaCppDir")
    # Anchor on the actual command-array entry, not the why-comment mention.
    helper_idx = block.index('"$PSScriptRoot\\install_llama_prebuilt.py"')
    assert (
        guard_idx < helper_idx
    ), "Assert-StudioOwnedOrAbsent must precede the install_llama_prebuilt.py call"


def test_setup_ps1_adopts_existing_whisper_prebuilt_marker():
    text = SETUP_PS1.read_text(encoding = "utf-8")
    # The marker scan lives in Get-StudioAdoptableState; Test-StudioOwnedAdoptable
    # is the boolean view of it.
    helper_start = text.index("function Get-StudioAdoptableState")
    helper_end = text.index("function Assert-StudioOwnedOrAbsent", helper_start)
    helper = text[helper_start:helper_end]
    assert "UNSLOTH_WHISPER_PREBUILT_INFO.json" in helper


def test_env_mode_passes_when_venv_marker_present(tmp_path):
    """install.sh env-mode guard must accept the in-VENV .unsloth-studio-owned marker as a sentinel."""
    studio_home = tmp_path / "ws"
    res = _run_install_guard(studio_home, redirect = "env", create_venv_marker = True)
    assert (
        res.returncode == 0
    ), f"in-VENV marker must allow cleanup; stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "RESULT=ok" in res.stdout
    assert not (studio_home / "unsloth_studio").exists()


def test_env_mode_blocks_when_bin_unsloth_is_symlink_to_directory(tmp_path):
    """install.sh guard must reject a symlink-to-directory at bin/unsloth; only -f (file/symlink-to-file) counts."""
    studio_home = tmp_path / "ws"
    venv = studio_home / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    (venv / "important.txt").write_text("keep me")
    (studio_home / "bin").mkdir(parents = True)
    target_dir = studio_home / "bin" / "unsloth-target-dir"
    target_dir.mkdir()
    (studio_home / "bin" / "unsloth").symlink_to(target_dir)
    script = _build_install_guard_script(studio_home, "env")
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert res.returncode != 0, (
        "symlink-to-directory at bin/unsloth must NOT pass; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert (venv / "important.txt").is_file(), "unrelated workspace data must survive"


def test_env_mode_blocks_when_bin_unsloth_is_broken_symlink(tmp_path):
    """install.sh guard must reject a broken symlink at bin/unsloth."""
    studio_home = tmp_path / "ws"
    venv = studio_home / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    (venv / "important.txt").write_text("keep me")
    (studio_home / "bin").mkdir(parents = True)
    (studio_home / "bin" / "unsloth").symlink_to(studio_home / "bin" / "does-not-exist")
    script = _build_install_guard_script(studio_home, "env")
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert (
        res.returncode != 0
    ), f"broken symlink at bin/unsloth must NOT pass; stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (venv / "important.txt").is_file()


def test_install_sh_writes_venv_marker_after_uv_venv():
    """install.sh must write .unsloth-studio-owned into $VENV_DIR right after `uv venv` succeeds."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    create_idx = src.index('_uv_venv_requested "create venv"')
    tail = src[create_idx : create_idx + 600]
    assert (
        ".unsloth-studio-owned" in tail
    ), "install.sh must write .unsloth-studio-owned after uv venv create"


def test_install_ps1_writes_venv_marker_after_uv_venv():
    """install.ps1 must write .unsloth-studio-owned into $VenvDir after `uv venv` succeeds."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    # Anchored past the command token: uv is invoked as the resolved $script:UvExe.
    venv_create = src.index("venv $VenvDir --python")
    tail = src[venv_create : venv_create + 1500]
    assert (
        ".unsloth-studio-owned" in tail
    ), "install.ps1 must write .unsloth-studio-owned after uv venv create"


def test_install_ps1_guard_accepts_venv_marker():
    """install.ps1 env-mode guard must accept the in-VENV .unsloth-studio-owned marker as a sentinel."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    block_start = src.index("# why: matching guard to the .venv branch below")
    block = src[block_start : block_start + 2000]
    assert (
        '$VenvDir ".unsloth-studio-owned") -PathType Leaf' in block
    ), "install.ps1 guard must check the in-VENV marker with -PathType Leaf"


def test_setup_helpers_gate_on_canonical_custom_root():
    """setup.sh/setup.ps1 ownership guards must gate on a canonical custom-vs-legacy root comparison."""
    sh_src = SETUP_SH.read_text(encoding = "utf-8")
    sh_idx = sh_src.index("_assert_studio_owned_or_absent() {")
    sh_func = sh_src[sh_idx : sh_idx + 600]
    assert (
        '"$_STUDIO_HOME_IS_CUSTOM" = true' in sh_func
    ), "setup.sh _assert_studio_owned_or_absent must gate on _STUDIO_HOME_IS_CUSTOM"
    assert (
        "_LEGACY_STUDIO_HOME=" in sh_src
        and "_studio_home_canon=" in sh_src
        and "_STUDIO_HOME_IS_CUSTOM=" in sh_src
    ), "setup.sh must compute the canonical custom-root flag"

    ps_src = SETUP_PS1.read_text(encoding = "utf-8")
    ps_idx = ps_src.index("function Assert-StudioOwnedOrAbsent")
    # To the end of the function, not a fixed width, which a new parameter or
    # comment would push the assertions below out of.
    ps_func = ps_src[ps_idx:].split("\nfunction ", 1)[0]
    assert (
        "$StudioHomeIsCustom -and" in ps_func
    ), "setup.ps1 Assert-StudioOwnedOrAbsent must gate on $StudioHomeIsCustom"
    assert (
        "$StudioOwnedMarker) -PathType Leaf" in ps_func
    ), "setup.ps1 marker check must use -PathType Leaf so a directory cannot satisfy it"


def test_setup_ps1_inplace_git_sync_marks_studio_owned():
    """setup.ps1 in-place git-sync branch must Mark-StudioOwned after a successful sync."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    # Three-state probe so an ACL-denied tree stops instead of cloning over it.
    inplace_idx = src.index('if ($llamaGitState -eq "Present") {')
    # The in-place branch ends just before the temp-dir clone branch.
    clone_idx = src.index("Cloning llama.cpp @", inplace_idx)
    inplace_block = src[inplace_idx:clone_idx]
    assert (
        "Mark-StudioOwned -Path $LlamaCppDir" in inplace_block
    ), "in-place git-sync branch must call Mark-StudioOwned on success"
    assert (
        "$StudioHomeIsCustom" in inplace_block
    ), "in-place Mark-StudioOwned call should be gated on $StudioHomeIsCustom"


def test_setup_ps1_inplace_git_sync_asserts_studio_owned_before_mutation():
    """setup.ps1 in-place git-sync must Assert-StudioOwnedOrAbsent before any destructive git op."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    # Three-state probe so an ACL-denied tree stops instead of cloning over it.
    inplace_idx = src.index('if ($llamaGitState -eq "Present") {')
    clone_idx = src.index("Cloning llama.cpp @", inplace_idx)
    inplace_block = src[inplace_idx:clone_idx]
    assert (
        "Assert-StudioOwnedOrAbsent -Path $LlamaCppDir" in inplace_block
    ), "in-place git-sync must Assert-StudioOwnedOrAbsent before mutating $LlamaCppDir"
    guard_idx = inplace_block.index("Assert-StudioOwnedOrAbsent -Path $LlamaCppDir")
    git_idx = inplace_block.index("git -C $LlamaCppDir remote set-url")
    assert guard_idx < git_idx, "Assert-StudioOwnedOrAbsent must precede the first git mutation"


def _extract_check_health_function() -> str:
    src = INSTALL_SH.read_text(encoding = "utf-8")
    fn_start = src.index("_check_health() {")
    fn_end = src.index("\n}\n", fn_start) + 2
    return src[fn_start:fn_end]


def _run_check_health(expected_root_id: str, response_json: str) -> int:
    fn = _extract_check_health_function()
    script = (
        f"_EXPECTED_STUDIO_ROOT_ID={expected_root_id!r}\n"
        "_http_get() { printf '%s' \"$1\"; }\n"
        + fn.replace(
            '_resp=$(_http_get "http://127.0.0.1:$_port/api/health") || return 1',
            f"_resp={response_json!r}",
        )
        + "\n_check_health 8888\n"
        "echo rc=$?\n"
    )
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    rc_lines = [l for l in res.stdout.splitlines() if l.startswith("rc=")]
    return int(rc_lines[0].split("=")[1]) if rc_lines else res.returncode


def test_check_health_accepts_matching_studio_root_id():
    """Matching baked studio_root_id lets the launcher attach to its own backend."""
    expected_id = "a" * 64
    rc = _run_check_health(
        expected_id,
        f'{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{expected_id}"}}',
    )
    assert rc == 0, f"matching studio_root_id must allow attach (rc={rc})"


def test_check_health_rejects_mismatched_studio_root_id():
    """Mismatched studio_root_id rejects attach (workspace isolation across same-port Studios)."""
    expected_id = "a" * 64
    other_id = "b" * 64
    rc = _run_check_health(
        expected_id,
        f'{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{other_id}"}}',
    )
    assert rc != 0, "mismatched studio_root_id must reject attach (workspace isolation)"


def test_check_health_rejects_missing_studio_root_id_field():
    """A backend omitting studio_root_id must not be attached when an expected id is baked in."""
    expected_id = "a" * 64
    rc = _run_check_health(
        expected_id,
        '{"status":"healthy","service":"Unsloth UI Backend"}',
    )
    assert rc != 0, "missing studio_root_id field must reject attach"


def test_check_health_no_baked_id_accepts_any_healthy_backend():
    """Empty _EXPECTED_STUDIO_ROOT_ID falls back to legacy contract: accept any healthy Unsloth backend."""
    rc = _run_check_health(
        "",
        '{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"deadbeef"}',
    )
    assert rc == 0, "no baked id → accept any healthy Unsloth backend"


def test_check_health_rejects_non_unsloth_service():
    rc = _run_check_health(
        "",
        '{"status":"healthy","service":"Other UI Backend"}',
    )
    assert rc != 0, "non-Unsloth service must be rejected"


def test_check_health_handles_arbitrary_id_token():
    """A fully arbitrary 64-char hex install id must round-trip cleanly (hex-only, no JSON escapes)."""
    expected_id = "f0" + ("ed" * 31)  # 64 hex chars, not derived from any path
    rc = _run_check_health(
        expected_id,
        f'{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{expected_id}"}}',
    )
    assert rc == 0, "arbitrary 64-hex install id must round-trip cleanly (no JSON escape issue)"


def test_install_ps1_test_studio_health_verifies_studio_root_id():
    """install.ps1 Test-StudioHealth must compare studio_root_id against baked $_ExpectedStudioRootId."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    fn_start = src.index("function Test-StudioHealth")
    fn_end = src.index("\n}\n", fn_start) + 2
    fn = src[fn_start:fn_end]
    assert "studio_root_id" in fn, "Test-StudioHealth must inspect the studio_root_id field"
    assert (
        "$_ExpectedStudioRootId" in fn
    ), "Test-StudioHealth must compare against the install-time baked $_ExpectedStudioRootId"


def test_install_ps1_bakes_studio_root_id_into_launcher():
    """install.ps1 must persist a CSPRNG id at share/studio_install_id and bake it as $_ExpectedStudioRootId."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "$_studioRootId" in src, "install.ps1 must compute $_studioRootId for the launcher"
    assert (
        '"share"' in src and "studio_install_id" in src
    ), "install.ps1 must persist the id at $StudioHome\\share\\studio_install_id"
    assert (
        "RandomNumberGenerator" in src
    ), "install.ps1 must seed the id from a CSPRNG (RandomNumberGenerator)"
    assert (
        "$_ExpectedStudioRootId" in src
    ), "install.ps1 must bake $_ExpectedStudioRootId into the launcher"


def test_health_endpoint_exposes_studio_root_id_not_raw_path():
    """/api/health must expose studio_root_id (hex digest), NOT the raw path (info disclosure on -H 0.0.0.0)."""
    main_py = REPO_ROOT / "studio" / "backend" / "main.py"
    src = main_py.read_text(encoding = "utf-8")
    health_idx = src.index('@app.get("/api/health")')
    # Slice up to the next top-level @app. so a growing body stays in scope.
    next_app_idx = src.find("\n@app.", health_idx + 1)
    if next_app_idx == -1:
        next_app_idx = len(src)
    health_block = src[health_idx:next_app_idx]
    assert '"studio_root_id"' in health_block, "/api/health must expose studio_root_id (hex digest)"
    assert (
        '"studio_root":' not in health_block
    ), "/api/health must NOT expose the raw studio_root path (information disclosure)"
    assert "_studio_root_id()" in health_block, "/api/health must call the _studio_root_id helper"


def test_install_sh_bakes_studio_root_id_into_launcher():
    """install.sh must persist the id at share/studio_install_id and bake it into the launcher for ALL modes."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    assert (
        "_css_studio_root_id" in src
    ), "install.sh must compute _css_studio_root_id for the launcher"
    assert (
        '_css_id_file="$_css_id_dir/studio_install_id"' in src
    ), "install.sh must persist the id at $STUDIO_HOME/share/studio_install_id"
    assert (
        "od -An -N32 -tx1 /dev/urandom" in src
    ), "install.sh must seed new ids from /dev/urandom (CSPRNG)"
    assert (
        "@@STUDIO_ROOT_ID@@" in src
    ), "install.sh must use @@STUDIO_ROOT_ID@@ placeholder in the launcher heredoc"
    assert (
        "s|@@STUDIO_ROOT_ID@@|$_css_studio_root_id|g" in src
    ), "install.sh must sed-substitute @@STUDIO_ROOT_ID@@ unconditionally (not just env-mode)"


def test_tauri_preflight_scrubs_studio_home_env():
    """Tauri CLI-spawn sites must env_remove UNSLOTH_STUDIO_HOME and STUDIO_HOME."""
    # PR #5341 split preflight into a submodule dir; read whichever shape is on disk.
    preflight_root = REPO_ROOT / "studio" / "src-tauri" / "src"
    preflight_paths = [
        preflight_root / "preflight.rs",
        *(preflight_root / "preflight").glob("*.rs"),
    ]
    preflight = "\n".join(p.read_text(encoding = "utf-8") for p in preflight_paths if p.exists())
    commands = (REPO_ROOT / "studio" / "src-tauri" / "src" / "commands.rs").read_text(
        encoding = "utf-8"
    )
    # Expect 2 scrubs in preflight (run_cli_probe + probe_cli_capability), 1 in commands.
    assert (
        preflight.count('cmd.env_remove("UNSLOTH_STUDIO_HOME")') >= 2
    ), "preflight must scrub UNSLOTH_STUDIO_HOME in both run_cli_probe and probe_cli_capability"
    assert (
        preflight.count('cmd.env_remove("STUDIO_HOME")') >= 2
    ), "preflight must scrub STUDIO_HOME in both run_cli_probe and probe_cli_capability"
    assert (
        'cmd.env_remove("UNSLOTH_STUDIO_HOME")' in commands
    ), "commands.rs check_install_status must scrub UNSLOTH_STUDIO_HOME"
    assert (
        'cmd.env_remove("STUDIO_HOME")' in commands
    ), "commands.rs check_install_status must scrub STUDIO_HOME"


def test_install_sh_shim_uses_atomic_replace():
    """install.sh shim install must use ln -sfn for atomic replace (rm+ln left a missing-shim window)."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    shim_idx = src.index('_shim_path="$_LOCAL_BIN/unsloth"')
    block = src[shim_idx : shim_idx + 1500]
    assert (
        'ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path"' in block
    ), "install.sh must use ln -sfn for atomic shim replacement"
    assert (
        'rm -f -- "$_shim_path"' not in block
    ), "the explicit rm + ln pair must be replaced by atomic ln -sfn"


def test_install_sh_create_shortcuts_seeds_id_from_csprng_with_python_fallback(tmp_path):
    """_create_shortcuts seeds ids from /dev/urandom (python3 secrets fallback) and is re-run idempotent."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    fn_start = src.index('_css_data_dir="$DATA_DIR"')
    block = src[fn_start : fn_start + 3000]
    urandom_idx = block.index("od -An -N32 -tx1 /dev/urandom")
    py_fallback_idx = block.index("python3 -c 'import secrets;", urandom_idx)
    assert (
        urandom_idx < py_fallback_idx
    ), "/dev/urandom must be tried before the python3 secrets fallback"
    # Non-empty id file check before generation is what makes re-runs idempotent.
    assert (
        'if [ ! -s "$_css_id_file" ]; then' in block
    ), "install.sh must skip id generation when the file already has content"

    # Behavioral check: run the generation block twice to confirm idempotence.
    studio_home = tmp_path / "studio"
    (studio_home / "share").mkdir(parents = True)
    gen_script = (
        f'STUDIO_HOME="{studio_home}"\n'
        '_css_id_dir="$STUDIO_HOME/share"\n'
        '_css_id_file="$_css_id_dir/studio_install_id"\n'
        # Replicate the generation block narrowly so it fails loud on contract drift.
        "gen() {\n"
        '    if [ ! -s "$_css_id_file" ]; then\n'
        '        _css_new_id=$(od -An -N32 -tx1 /dev/urandom 2>/dev/null | tr -d " \\n")\n'
        '        _t="$_css_id_file.$$.tmp"\n'
        '        printf "%s" "$_css_new_id" > "$_t"\n'
        '        ln "$_t" "$_css_id_file" 2>/dev/null \\\n'
        '            || { [ -s "$_css_id_file" ] || mv "$_t" "$_css_id_file"; }\n'
        '        rm -f "$_t"\n'
        "    fi\n"
        '    cat "$_css_id_file"\n'
        "}\n"
        "a=$(gen); b=$(gen)\n"
        '[ "$a" = "$b" ] || { echo MISMATCH; exit 1; }\n'
        'echo "ID=$a"\n'
        'echo "LEN=${#a}"\n'
    )
    res = subprocess.run(["bash", "-c", gen_script], text = True, capture_output = True)
    assert res.returncode == 0, res.stderr
    out = dict(line.split("=", 1) for line in res.stdout.strip().splitlines() if "=" in line)
    assert out.get("LEN") == "64", f"id must be 64 hex chars, got LEN={out.get('LEN')!r}"
    assert all(
        c in "0123456789abcdef" for c in out.get("ID", "")
    ), f"id must be lowercase hex, got {out.get('ID')!r}"


def test_install_sh_publishes_the_id_without_clobbering():
    """install.sh must publish the id no-clobber, so it cannot replace one the desktop app minted."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    fn_start = src.index('_css_id_dir="$STUDIO_HOME/share"')
    block = src[fn_start : fn_start + 2400]
    assert (
        'ln "$_css_id_tmp" "$_css_id_file"' in block
    ), "install.sh must publish the id with ln (EEXIST on a race), not a clobbering mv"
    assert 'mv "$_css_id_tmp" "$_css_id_file"' not in block.replace(
        '[ -s "$_css_id_file" ] || mv "$_css_id_tmp" "$_css_id_file"', ""
    ), "the only remaining mv must be the no-hard-link fallback, guarded on the destination"
    assert (
        '[ -s "$_css_id_file" ] || mv' in block
    ), "the mv branch must refuse to replace a usable incumbent id"
    assert (
        'rm -f "$_css_id_file"' not in block
    ), "never unlink the id: an unlink opens a window where a valid id is deleted"
    assert 'rm -f "$_css_id_tmp"' in block, "the temp sibling must not be left behind"


def test_install_sh_id_publish_adopts_the_winner_of_a_race(tmp_path):
    """A second writer must adopt the id already on disk, never replace it."""
    studio_home = tmp_path / "studio"
    (studio_home / "share").mkdir(parents = True)
    id_file = studio_home / "share" / "studio_install_id"
    incumbent = "a" * 64
    id_file.write_text(incumbent, encoding = "utf-8")

    # Replicate the publish step with the guard removed, so only the publication
    # primitive decides the outcome: a clobbering mv would overwrite the incumbent.
    publish = (
        f'_css_id_file="{id_file}"\n'
        '_css_id_tmp="$_css_id_file.$$.tmp"\n'
        '_css_new_id="' + "b" * 64 + '"\n'
        'printf "%s" "$_css_new_id" > "$_css_id_tmp"\n'
        'if ! ln "$_css_id_tmp" "$_css_id_file" 2>/dev/null; then\n'
        '    [ -s "$_css_id_file" ] || mv "$_css_id_tmp" "$_css_id_file"\n'
        "fi\n"
        'rm -f "$_css_id_tmp"\n'
        'cat "$_css_id_file"\n'
    )
    res = subprocess.run(["sh", "-c", publish], text = True, capture_output = True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == incumbent, "the incumbent id must survive a concurrent publish"
    leftovers = [p.name for p in (studio_home / "share").iterdir() if p.name != "studio_install_id"]
    assert not leftovers, f"temp files must be cleaned up, found {leftovers}"


def test_install_sh_id_publish_replaces_a_blank_incumbent(tmp_path):
    """A zero-length id is an interrupted write: it must be replaced, never adopted.

    Adopting it would bake an empty $_ExpectedStudioRootId into the launcher, which
    permanently skips the ownership comparison in Test-StudioHealth.
    """
    studio_home = tmp_path / "studio"
    (studio_home / "share").mkdir(parents = True)
    id_file = studio_home / "share" / "studio_install_id"
    id_file.write_text("", encoding = "utf-8")
    fresh = "c" * 64

    publish = (
        f'_css_id_file="{id_file}"\n'
        f'_css_new_id="{fresh}"\n'
        '_css_id_tmp="$_css_id_file.$$.$(printf "%.8s" "$_css_new_id").tmp"\n'
        'printf "%s" "$_css_new_id" > "$_css_id_tmp"\n'
        'if ! ln "$_css_id_tmp" "$_css_id_file" 2>/dev/null; then\n'
        '    [ -s "$_css_id_file" ] || mv "$_css_id_tmp" "$_css_id_file"\n'
        "fi\n"
        'rm -f "$_css_id_tmp"\n'
        'cat "$_css_id_file"\n'
    )
    res = subprocess.run(["sh", "-c", publish], text = True, capture_output = True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == fresh, "a blank id must be replaced, not adopted"


def test_install_ps1_publishes_the_id_without_clobbering():
    """install.ps1 must publish no-clobber and adopt the winner, since it never re-reads the file."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    idx = src.index('$_studioIdFile = Join-Path $_studioIdDir "studio_install_id"')
    block = src[idx : idx + 3200]
    assert (
        "[System.IO.File]::Move($_idTmp, $_studioIdFile)" in block
    ), "install.ps1 must use the two-arg File.Move, which throws when the destination exists"
    # -Force may only appear AFTER the no-clobber attempt, as the branch that
    # replaces a blank incumbent. As the primary publish it would clobber an id
    # the desktop app may already have handed to a running backend.
    _force = "Move-Item -LiteralPath $_idTmp -Destination $_studioIdFile -Force"
    assert block.index("[System.IO.File]::Move($_idTmp, $_studioIdFile)") < block.index(
        _force
    ), "the no-clobber File.Move must be attempted before any -Force fallback"
    assert (
        "catch [System.IO.IOException]" in block
    ), "install.ps1 must catch the destination-exists IOException"
    assert (
        "$_adoptedRootId = ([System.IO.File]::ReadAllText($_studioIdFile)).Trim()"
        in block[block.index("catch [System.IO.IOException]") :]
    ), "on a lost race install.ps1 must adopt the winner's id, since it never re-reads it later"
    assert (
        "$_studioRootId = $_adoptedRootId" in block
    ), "the adopted id must become the value baked into the launcher"
    assert (
        "if ($_adoptedRootId)" in block
    ), "install.ps1 must only adopt a non-empty id, so a blank one cannot become the expected id"
    assert (
        "Remove-Item -LiteralPath $_studioIdFile" not in block
    ), "never unlink the id: an unlink opens a window where a valid id is deleted"


def test_install_sh_create_shortcuts_fails_fast_when_no_entropy():
    """With no entropy source, _create_shortcuts must `return 1` not bake an empty studio_root_id."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    fn_start = src.index('_css_data_dir="$DATA_DIR"')
    block = src[fn_start : fn_start + 4200]
    assert (
        "[WARN] Cannot create launcher: no entropy source for studio_install_id" in block
    ), "install.sh must warn when neither urandom nor python3 is available"
    assert (
        "[WARN] Cannot create launcher: failed to read" in block
    ), "install.sh must warn when the id file read produces no content"
    assert (
        block.count("return 1") >= 2
    ), "both the no-entropy branch and the empty-read branch must `return 1`"


def test_install_sh_bakes_installed_is_env_mode_flag_in_launcher():
    """install.sh must bake the install-time mode into the launcher so a sourced studio.conf can't flip it."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    assert (
        "_INSTALLED_IS_ENV_MODE='@@INSTALLED_IS_ENV_MODE@@'" in src
    ), "launcher heredoc must declare _INSTALLED_IS_ENV_MODE='@@INSTALLED_IS_ENV_MODE@@'"
    assert "_css_is_env_mode=false" in src, "install.sh must default _css_is_env_mode to false"
    assert (
        '[ "$_STUDIO_HOME_REDIRECT" = "env" ] && _css_is_env_mode=true' in src
    ), "install.sh must set _css_is_env_mode=true only when _STUDIO_HOME_REDIRECT=env"
    assert (
        "s|@@INSTALLED_IS_ENV_MODE@@|$_css_is_env_mode|g" in src
    ), "install.sh sed pipeline must substitute @@INSTALLED_IS_ENV_MODE@@"


def test_install_sh_launcher_gates_port_file_on_baked_flag_not_runtime_env():
    """Launcher PORT_FILE/LOCK_DIR must gate on baked $_INSTALLED_IS_ENV_MODE, not runtime $UNSLOTH_STUDIO_HOME."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    heredoc_start = src.index("cat > \"$_css_launcher\" << 'LAUNCHER_EOF'")
    heredoc_end = src.index("LAUNCHER_EOF\n", heredoc_start)
    heredoc = src[heredoc_start:heredoc_end]
    assert (
        'if [ "$_INSTALLED_IS_ENV_MODE" = "true" ]; then' in heredoc
    ), "launcher must gate PORT_FILE/LOCK_DIR on baked _INSTALLED_IS_ENV_MODE"
    port_block_start = heredoc.index('if [ "$_INSTALLED_IS_ENV_MODE" = "true" ]; then')
    port_block_end = heredoc.index("\nfi\n", port_block_start) + len("\nfi\n")
    port_block = heredoc[port_block_start:port_block_end]
    assert 'PORT_FILE="$DATA_DIR/studio.port"' in port_block
    assert (
        'if [ -n "${UNSLOTH_STUDIO_HOME:-}" ]; then\n    if command -v cksum' not in heredoc
    ), "launcher must NOT gate PORT_FILE on runtime UNSLOTH_STUDIO_HOME"

    def _run_launcher_gate(installed_flag: str, runtime_env: dict) -> str:
        # Run the LOCK_DIR/PORT_FILE init block in isolation.
        script = (
            f"_INSTALLED_IS_ENV_MODE={installed_flag!r}\n"
            "DATA_DIR=/tmp/test_data_dir\n"
            'LOCK_DIR="${XDG_RUNTIME_DIR:-/tmp}/unsloth-studio-launcher-$(id -u).lock"\n'
            'PORT_FILE=""\n' + port_block + '\necho "PORT_FILE=$PORT_FILE"\n'
        )
        env = {"PATH": "/usr/bin:/bin"}
        env.update(runtime_env)
        res = subprocess.run(
            ["bash", "-c", script],
            text = True,
            capture_output = True,
            env = env,
        )
        for line in res.stdout.splitlines():
            if line.startswith("PORT_FILE="):
                return line[len("PORT_FILE=") :]
        return ""

    # default-mode must keep PORT_FILE empty even if UNSLOTH_STUDIO_HOME leaks in.
    assert (
        _run_launcher_gate("false", {"UNSLOTH_STUDIO_HOME": "/tmp/leaked"}) == ""
    ), "default-mode launcher must keep PORT_FILE empty even with UNSLOTH_STUDIO_HOME in env"
    # env-mode must set PORT_FILE regardless of runtime env.
    assert (
        _run_launcher_gate("true", {}) == "/tmp/test_data_dir/studio.port"
    ), "env-mode launcher must set PORT_FILE based on baked DATA_DIR"


def test_main_py_studio_root_id_caches_at_module_load():
    """_studio_root_id() must read the id once at module load and reuse it (no per-poll FS/hash work)."""
    main_py = (REPO_ROOT / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    assert (
        "_STUDIO_ROOT_ID_CACHE: str = _read_studio_install_id()" in main_py
    ), "main.py must populate _STUDIO_ROOT_ID_CACHE from _read_studio_install_id() at module load"
    fn_idx = main_py.index("def _studio_root_id() -> str:")
    next_def_idx = main_py.index("\ndef ", fn_idx + 1)
    fn_block = main_py[fn_idx:next_def_idx]
    assert (
        "return _STUDIO_ROOT_ID_CACHE" in fn_block
    ), "_studio_root_id() body must return the cached value"
    assert (
        "read_text(" not in fn_block and "hashlib" not in fn_block
    ), "_studio_root_id() must NOT do filesystem or hash work on every call"


def test_main_py_read_studio_install_id_validates_hex_and_handles_missing(tmp_path, monkeypatch):
    """_read_studio_install_id returns "" for absent/empty/non-hex/wrong-length ids, else the token."""
    import re

    pattern = re.compile(r"^[0-9a-f]{64}$")

    def _read(root: Path) -> str:
        # Mirror the implementation to pin the exact accepted contract.
        try:
            token = (root / "share" / "studio_install_id").read_text().strip()
        except (OSError, ValueError):
            return ""
        return token if pattern.fullmatch(token) else ""

    root = tmp_path / "studio"
    (root / "share").mkdir(parents = True)

    # Missing file -> empty
    assert _read(root) == ""

    id_file = root / "share" / "studio_install_id"
    # Empty file -> empty
    id_file.write_text("")
    assert _read(root) == ""
    # Non-hex content -> empty
    id_file.write_text("not-a-hex-id-just-text-padded-to-64-chars-zzzzzzzzzzzzzzzzzzzzzz")
    assert _read(root) == ""
    # Uppercase hex -> empty (must be lowercase)
    id_file.write_text("F" * 64)
    assert _read(root) == ""
    # Wrong length -> empty (32 chars, not 64)
    id_file.write_text("a" * 32)
    assert _read(root) == ""
    # Valid 64-char lowercase hex with surrounding whitespace -> stripped+accepted
    valid = "0123456789abcdef" * 4
    id_file.write_text(f"\n  {valid}  \n")
    assert _read(root) == valid


def test_llama_cpp_search_roots_handles_studio_root_oserror():
    """Root resolution must catch (ImportError, OSError, ValueError) from studio_root().
    Discovery (_find_llama_server_binary) and cleanup (_kill_orphaned_servers) both
    delegate to the shared _resolved_studio_root_and_is_legacy() classifier, which
    holds the handler so the two never disagree on which root is legacy."""
    llama_cpp = (
        REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"
    ).read_text(encoding = "utf-8")

    def _method_body(name: str) -> str:
        # Whole method body (def to next sibling def) so the check survives growth.
        start = llama_cpp.index(f"def {name}")
        indent = " " * (start - llama_cpp.rfind("\n", 0, start) - 1)
        nxt = llama_cpp.find(f"\n{indent}def ", start + 1)
        return llama_cpp[start : nxt if nxt != -1 else len(llama_cpp)]

    assert (
        "except (ImportError, OSError, ValueError):"
        in _method_body("_resolved_studio_root_and_is_legacy")
    ), "_resolved_studio_root_and_is_legacy must catch (ImportError, OSError, ValueError) from studio_root()"
    # Both callers must route through the shared classifier so neither crashes.
    for caller in ("_find_llama_server_binary", "_kill_orphaned_servers"):
        assert "LlamaCppBackend._resolved_studio_root_and_is_legacy()" in _method_body(
            caller
        ), f"{caller} must resolve the install root via the shared classifier"


def test_install_sh_install_id_survives_symlinked_studio_home(tmp_path):
    """Regression: install id read from a file (not sha256(canonical_path)) agrees under a symlinked $STUDIO_HOME."""
    real = tmp_path / "realhome"
    real.mkdir()
    link = tmp_path / "linkhome"
    link.symlink_to(real)
    studio_home = real / ".unsloth" / "studio"
    (studio_home / "share").mkdir(parents = True)
    valid_id = "ab12" * 16
    (studio_home / "share" / "studio_install_id").write_text(valid_id)
    # Canonical and symlinked paths must see the SAME content (cat and read_text agree).
    raw_via_link = link / ".unsloth" / "studio" / "share" / "studio_install_id"
    raw_direct = studio_home / "share" / "studio_install_id"
    assert raw_via_link.read_text() == valid_id
    assert raw_direct.read_text() == valid_id
    # install.sh's `cat` sees the same.
    import subprocess as _sp

    res = _sp.run(["cat", str(raw_via_link)], capture_output = True, text = True)
    assert res.returncode == 0
    assert res.stdout == valid_id


def test_install_sh_substitutes_root_id_before_data_dir():
    """sed must bake the non-user-controlled placeholders before @@DATA_DIR@@ so a crafted $DATA_DIR isn't mutated."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    root_id_idx = src.index("s|@@STUDIO_ROOT_ID@@|$_css_studio_root_id|g")
    env_mode_idx = src.index("s|@@INSTALLED_IS_ENV_MODE@@|$_css_is_env_mode|g")
    data_dir_idx = src.index("s|@@DATA_DIR@@|$_sed_safe|g")
    assert root_id_idx < data_dir_idx, (
        "@@STUDIO_ROOT_ID@@ substitution must happen BEFORE @@DATA_DIR@@ "
        "(non-user-controlled placeholders first)"
    )
    assert (
        env_mode_idx < data_dir_idx
    ), "@@INSTALLED_IS_ENV_MODE@@ substitution must happen BEFORE @@DATA_DIR@@"


def test_install_sh_root_id_pass_does_not_mutate_user_data_dir(tmp_path):
    """A $DATA_DIR containing the literal @@STUDIO_ROOT_ID@@ must survive the placeholder-first sed passes."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    heredoc_start = src.index("cat > \"$_css_launcher\" << 'LAUNCHER_EOF'")
    heredoc_body_start = src.index("\n", heredoc_start) + 1
    heredoc_body_end = src.index("LAUNCHER_EOF\n", heredoc_start)
    template = src[heredoc_body_start:heredoc_body_end]
    launcher_path = tmp_path / "launch.sh"
    # template comes out of install.sh, so it carries whatever non-ASCII that
    # file holds and cp1252 cannot encode it back out.
    launcher_path.write_text(template, encoding = "utf-8")
    # sed order: root-id first, then data-dir.
    weird_data_dir = "/tmp/with-@@STUDIO_ROOT_ID@@/share"
    root_id = "deadbeef" * 8
    is_env = "true"
    script = f"""
sed -e "s|@@STUDIO_ROOT_ID@@|{root_id}|g" \\
    -e "s|@@INSTALLED_IS_ENV_MODE@@|{is_env}|g" \\
    "{launcher_path}" > "{launcher_path}.tmp" && mv "{launcher_path}.tmp" "{launcher_path}"
_sq_escaped=$(printf '%s' "{weird_data_dir}" | sed "s/'/'\\\\\\\\''/g")
_sed_safe=$(printf '%s' "$_sq_escaped" | sed 's/[\\\\&|]/\\\\&/g')
sed "s|@@DATA_DIR@@|$_sed_safe|g" "{launcher_path}" > "{launcher_path}.tmp" \\
    && mv "{launcher_path}.tmp" "{launcher_path}"
"""
    subprocess.run(["bash", "-c", script], check = True)
    # written as utf-8 just above, and the template carries U+2500.
    final = launcher_path.read_text(encoding = "utf-8")
    assert (
        f"DATA_DIR='{weird_data_dir}'" in final
    ), f"DATA_DIR must be preserved verbatim (no @@STUDIO_ROOT_ID@@ mutation); got: {final[:500]}"
    assert (
        f"_EXPECTED_STUDIO_ROOT_ID='{root_id}'" in final
    ), "STUDIO_ROOT_ID placeholder must still be substituted in the launcher heredoc"


def test_install_ps1_install_id_file_layout_matches_backend_read_path():
    """install.ps1 must write the id at share/studio_install_id where the backend reads it, idempotently."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    id_idx = src.index('$_studioIdDir = Join-Path $StudioHome "share"')
    context = src[id_idx : id_idx + 2400]
    assert (
        '$_studioIdFile = Join-Path $_studioIdDir "studio_install_id"' in context
    ), "install.ps1 must persist the id at $StudioHome\\share\\studio_install_id"
    assert (
        "Test-Path -LiteralPath $_studioIdFile" in context
    ), "install.ps1 must skip id generation when the file already has content (re-run idempotence)"
    assert (
        "RandomNumberGenerator" in context and "GetBytes($_idBytes)" in context
    ), "install.ps1 must seed new ids from a CSPRNG (RandomNumberGenerator)"
    assert (
        "[System.IO.File]::Move($_idTmp, $_studioIdFile)" in context
    ), "install.ps1 must atomic-rename the temp file into place to avoid half-written ids"


def _make_interpreterless_venv(studio_home):
    """A venv whose uv-managed CPython was deleted: pyvenv.cfg intact, bin/python dangling."""
    venv = studio_home / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    (venv / "pyvenv.cfg").write_text("home = /gone/bin\nversion_info = 3.13.14\n")
    (venv / "bin" / "python").symlink_to("/gone/bin/python3.13")
    return venv


def _run_guard_block(studio_home, redirect):
    return subprocess.run(
        ["bash", "-c", _build_install_guard_script(studio_home, redirect)],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )


def test_install_sh_replaces_venv_whose_interpreter_is_gone(tmp_path):
    """A venv with no usable bin/python must still be moved aside: uv 0.10 will not overwrite it."""
    studio_home = tmp_path / "ws"
    venv = _make_interpreterless_venv(studio_home)
    res = _run_guard_block(studio_home, "default")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "RESULT=ok" in res.stdout
    assert not venv.exists(), "install.sh must clear $VENV_DIR before `uv venv` runs"


def test_install_sh_replaces_venv_dir_holding_only_hidden_entries(tmp_path):
    """uv refuses any non-empty target, so a leftover holding only dotfiles must be cleared too."""
    studio_home = tmp_path / "ws"
    venv = studio_home / "unsloth_studio"
    venv.mkdir(parents = True)
    (venv / ".unsloth-studio-owned").write_text("")
    res = _run_guard_block(studio_home, "default")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert not venv.exists()


def test_install_sh_leaves_absent_and_empty_venv_dir_to_uv(tmp_path):
    """uv creates into a missing or empty directory, so neither may trigger a rollback move."""
    studio_home = tmp_path / "ws"
    studio_home.mkdir()
    res = _run_guard_block(studio_home, "default")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert not (studio_home / "unsloth_studio.replaced").exists()

    (studio_home / "unsloth_studio").mkdir()
    res = _run_guard_block(studio_home, "default")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (studio_home / "unsloth_studio").is_dir(), "an empty $VENV_DIR must be left in place"
    assert not (studio_home / "unsloth_studio.replaced").exists()


def test_env_mode_blocks_interpreterless_venv_without_sentinels(tmp_path):
    """The env-mode ownership guard must cover the interpreter-less case, not just the healthy one."""
    studio_home = tmp_path / "ws"
    venv = _make_interpreterless_venv(studio_home)
    (venv / "important.txt").write_text("keep me")
    res = _run_guard_block(studio_home, "env")
    assert res.returncode != 0, (
        "env-mode without sentinels must refuse to replace $VENV_DIR; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "does not look like an Unsloth Studio install" in res.stderr
    assert (venv / "important.txt").is_file(), "unrelated workspace data must survive"


def test_env_mode_replaces_interpreterless_venv_when_marker_present(tmp_path):
    """A partial install that left the marker must be replaceable on the next run."""
    studio_home = tmp_path / "ws"
    venv = _make_interpreterless_venv(studio_home)
    (venv / ".unsloth-studio-owned").write_text("")
    res = _run_guard_block(studio_home, "env")
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert not venv.exists()


def test_install_ps1_replacement_branch_covers_an_occupied_venv_dir():
    """install.ps1 must move a venv aside on directory content, not only on a present python.exe."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert (
        "if ((Test-Path -LiteralPath $VenvPython) -or (Test-DirectoryHasEntries -Path $VenvDir))"
        in src
    ), "install.ps1 must treat an occupied $VenvDir as an environment to replace"
    helper_start = src.index("function Test-DirectoryHasEntries")
    helper = src[helper_start : src.index("function Get-VenvBaseHome", helper_start)]
    assert (
        "[System.IO.Directory]::EnumerateFileSystemEntries($Path)" in helper
    ), "Test-DirectoryHasEntries must count hidden entries and not read the path as a wildcard"
    assert (
        "-PathType Container" in helper
    ), "Test-DirectoryHasEntries must answer false for a missing directory"


def _extract_install_sh_venv_chain() -> str:
    """Extract the venv if/elif chain past the legacy migration to its closing `fi`.

    _extract_install_sh_guard_block stops at the first elif, so it cannot see the two
    interacting.
    """
    src = INSTALL_SH.read_text(encoding = "utf-8")
    m = re.search(
        r'^(if \[ -x "\$VENV_DIR/bin/python" \] \|\| _dir_has_entries "\$VENV_DIR"; then\n.*?^fi$)',
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "install.sh venv chain not found"
    return m.group(1) + "\n"


def _run_venv_chain(studio_home, redirect = "default"):
    """Run the full chain, then report what `uv venv` would face at install.sh's create gate."""
    script = (
        _INSTALL_GUARD_STUBS
        + _extract_install_sh_function("_dir_has_entries")
        + f'STUDIO_HOME="{studio_home}"\n'
        + 'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        + f'_STUDIO_HOME_REDIRECT="{redirect}"\n'
        + 'SKIP_TORCH=true\n_MIGRATED=false\n_PREV_TORCH_VER=""\n'
        + _extract_install_sh_venv_chain()
        # Mirrors the `if [ ! -x "$VENV_DIR/bin/python" ]` create gate below the chain.
        + 'if [ -x "$VENV_DIR/bin/python" ]; then echo UV=skipped_migrated\n'
        + 'elif [ -d "$VENV_DIR" ] && [ -n "$(ls -A "$VENV_DIR" 2>/dev/null)" ]; then\n'
        + "    echo UV=would_fail_dir_not_empty\n"
        + "else echo UV=would_create_ok; fi\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )


def _make_legacy_venv(studio_home):
    """A healthy legacy ~/.unsloth/studio/.venv from before the unsloth_studio layout."""
    legacy = studio_home / ".venv"
    (legacy / "bin").mkdir(parents = True)
    py = legacy / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    (legacy / "marker.txt").write_text("legacy")
    return legacy


def test_legacy_migration_into_empty_venv_dir_does_not_nest(tmp_path):
    """An empty $VENV_DIR must not make `mv` nest the legacy env inside it (uv then fails)."""
    studio_home = tmp_path / "ws"
    studio_home.mkdir()
    _make_legacy_venv(studio_home)
    (studio_home / "unsloth_studio").mkdir()

    res = _run_venv_chain(studio_home)

    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    venv = studio_home / "unsloth_studio"
    assert not (venv / ".venv").exists(), (
        "legacy environment was nested at $VENV_DIR/.venv; `uv venv` would then refuse the "
        "occupied target with the same error as #9479"
    )
    assert (venv / "marker.txt").is_file(), "legacy environment must land directly in $VENV_DIR"
    assert "UV=skipped_migrated" in res.stdout


def test_legacy_migration_with_absent_venv_dir_still_migrates(tmp_path):
    """The ordinary migration must keep working once the empty-directory case is handled."""
    studio_home = tmp_path / "ws"
    studio_home.mkdir()
    _make_legacy_venv(studio_home)

    res = _run_venv_chain(studio_home)

    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (studio_home / "unsloth_studio" / "marker.txt").is_file()
    assert "UV=skipped_migrated" in res.stdout


def test_legacy_migration_clears_a_symlinked_empty_venv_dir_without_touching_target(tmp_path):
    """Unlinking $VENV_DIR must never remove the directory a symlink points at."""
    studio_home = tmp_path / "ws"
    studio_home.mkdir()
    _make_legacy_venv(studio_home)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (studio_home / "unsloth_studio").symlink_to(target)

    res = _run_venv_chain(studio_home)

    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert target.is_dir(), "the symlink target must survive"
    assert (studio_home / "unsloth_studio" / "marker.txt").is_file()


def test_occupied_venv_dir_still_wins_over_legacy_migration(tmp_path):
    """An occupied $VENV_DIR must be replaced rather than migrated into (the #9479 path)."""
    studio_home = tmp_path / "ws"
    studio_home.mkdir()
    legacy = _make_legacy_venv(studio_home)
    venv = _make_interpreterless_venv(studio_home)

    res = _run_venv_chain(studio_home)

    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "UV=would_create_ok" in res.stdout
    assert not venv.exists(), "$VENV_DIR must be cleared before `uv venv` runs"
    assert (legacy / "marker.txt").is_file(), "the legacy environment must be left intact"


def test_install_sh_reports_a_failed_venv_move(tmp_path):
    """A failed move must say so, matching install.ps1's Exit-InstallFailure on the same step."""
    src = INSTALL_SH.read_text(encoding = "utf-8")
    assert (
        'if ! _start_studio_venv_replacement "$VENV_DIR"; then' in src
    ), "install.sh must check the replacement helper rather than relying on bare set -e"
    assert "could not move $VENV_DIR aside to reinstall" in src


def _dir_has_entries_says(
    tmp_path,
    target,
    pre = "",
):
    """Run the real _dir_has_entries from install.sh against one directory."""
    script = (
        _extract_install_sh_function("_dir_has_entries")
        + f"{pre}\n"
        + f'if _dir_has_entries "{target}"; then echo yes; else echo no; fi\n'
    )
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert res.returncode == 0, f"stderr={res.stderr!r}"
    return res.stdout.strip()


def test_dir_has_entries_survives_noglob_in_the_caller(tmp_path):
    """The check is pure globbing, so `set -f` must not make an occupied directory look empty."""
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file.txt").write_text("x")

    assert _dir_has_entries_says(tmp_path, occupied, pre = "set -f") == "yes"
    assert _dir_has_entries_says(tmp_path, occupied) == "yes"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        _dir_has_entries_says(tmp_path, empty, pre = "set -f") == "no"
    ), "an empty directory must still be left for uv to create into"


def test_dir_has_entries_restores_the_callers_noglob_setting(tmp_path):
    """Saving and restoring `-f` matters because _path_has_dir depends on the flag."""
    empty = tmp_path / "empty"
    empty.mkdir()
    script = (
        _extract_install_sh_function("_dir_has_entries")
        + "set -f\n"
        + f'_dir_has_entries "{empty}" || true\n'
        + "case $- in *f*) echo NOGLOB_KEPT ;; *) echo NOGLOB_LOST ;; esac\n"
        + "set +f\n"
        + f'_dir_has_entries "{empty}" || true\n'
        + "case $- in *f*) echo GLOB_LOST ;; *) echo GLOB_KEPT ;; esac\n"
    )
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert "NOGLOB_KEPT" in res.stdout, "the caller's `set -f` must be restored"
    assert "GLOB_KEPT" in res.stdout, "a caller without `set -f` must not gain it"


@pytest.mark.parametrize("mode", [0o000, 0o111, 0o444])
def test_dir_has_entries_treats_an_unenumerable_directory_as_occupied(tmp_path, mode):
    """uv refuses these targets, so reporting them empty would wedge the repair.

    0o444 is readable but not searchable, 0o111 the mirror; both must answer as 0o000.
    """
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    blocked = tmp_path / f"blocked{mode:o}"
    blocked.mkdir()
    (blocked / "file.txt").write_text("x")
    blocked.chmod(mode)
    try:
        # install.ps1's catch returns $true here; the two must not disagree.
        assert _dir_has_entries_says(tmp_path, blocked) == "yes"
    finally:
        blocked.chmod(0o700)


def test_dir_has_entries_still_answers_no_for_a_searchable_empty_dir(tmp_path):
    """The fail-closed rule must not swallow the empty case uv creates into."""
    empty = tmp_path / "empty"
    empty.mkdir()
    empty.chmod(0o555)
    try:
        assert _dir_has_entries_says(tmp_path, empty) == "no"
    finally:
        empty.chmod(0o700)


# Measured against uv 0.12.1, the version install.sh pins: uv creates only into a
# path that is absent or an empty directory, every other shape is EEXIST. The
# predicate has to agree, or the repair loop continues for the shapes it misses.
_UV_REFUSES = [
    ("occupied real dir", "fulldir", True),
    ("regular file", "plainfile", True),
    ("dangling symlink", "dangling", True),
    ("symlink to a file", "link_to_file", True),
    ("symlink to an occupied dir", "link_to_full", True),
    ("absent path", "absent", False),
    ("empty real dir", "emptydir", False),
    ("symlink to an empty dir", "link_to_empty", False),
]


def _make_uv_shape(root, shape):
    if shape == "absent":
        return root / "absent"
    if shape == "emptydir":
        (root / "emptydir").mkdir()
        return root / "emptydir"
    if shape == "fulldir":
        (root / "fulldir").mkdir()
        (root / "fulldir" / "x").write_text("x")
        return root / "fulldir"
    if shape == "plainfile":
        (root / "plainfile").write_text("x")
        return root / "plainfile"
    if shape == "dangling":
        (root / "dangling").symlink_to(root / "gone")
        return root / "dangling"
    if shape == "link_to_file":
        (root / "target_file").write_text("x")
        (root / "link_to_file").symlink_to(root / "target_file")
        return root / "link_to_file"
    if shape == "link_to_full":
        (root / "target_full").mkdir()
        (root / "target_full" / "x").write_text("x")
        (root / "link_to_full").symlink_to(root / "target_full")
        return root / "link_to_full"
    if shape == "link_to_empty":
        (root / "target_empty").mkdir()
        (root / "link_to_empty").symlink_to(root / "target_empty")
        return root / "link_to_empty"
    raise AssertionError(shape)


@pytest.mark.parametrize(
    "label,shape,uv_refuses",
    _UV_REFUSES,
    ids = [row[1] for row in _UV_REFUSES],
)
def test_dir_has_entries_matches_what_uv_refuses(tmp_path, label, shape, uv_refuses):
    """The predicate must answer uv's question, not "is this a non-empty directory"."""
    target = _make_uv_shape(tmp_path, shape)
    answer = _dir_has_entries_says(tmp_path, target)
    assert answer == ("yes" if uv_refuses else "no"), (
        f"{label}: uv {'refuses' if uv_refuses else 'creates into'} this path, "
        f"so the replacement branch must {'run' if uv_refuses else 'be skipped'}"
    )


def test_install_ps1_helper_answers_on_the_link_itself():
    """install.ps1 must match: -PathType Container follows a link and misses a dangling one."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    helper_start = src.index("function Test-DirectoryHasEntries")
    helper = src[helper_start : src.index("function Clear-MigrationTargetDirectory", helper_start)]
    assert (
        "Get-Item -LiteralPath $Path -Force" in helper
    ), "a dangling link is invisible to -PathType Container but still blocks uv"


def _run_rollback_lifecycle(studio_home, shape):
    """Move $VENV_DIR aside, half-create a new venv, then fail and restore."""
    fns = [
        "_start_studio_venv_replacement",
        "_restore_studio_venv_replacement",
        "_commit_studio_venv_replacement",
    ]
    src = INSTALL_SH.read_text(encoding = "utf-8")
    helpers = ""
    for fn in fns:
        m = re.search(rf"^{re.escape(fn)}\(\) \{{.*?\n\}}\n", src, re.DOTALL | re.MULTILINE)
        assert m, fn
        helpers += m.group(0)
    venv = studio_home / "unsloth_studio"
    if shape == "realdir":
        venv.mkdir()
        (venv / "keep.txt").write_text("CANARY")
    elif shape == "regularfile":
        venv.write_text("CANARY")
    elif shape == "danglinglink":
        venv.symlink_to(studio_home / "gone")
    else:
        raise AssertionError(shape)
    script = (
        "substep() { :; }\nrollback_substep() { :; }\n"
        + helpers
        + f'STUDIO_HOME="{studio_home}"\n'
        + 'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        + '_VENV_ROLLBACK_TARGET="$VENV_DIR"\n_VENV_ROLLBACK_DIR=""\n_VENV_ROLLBACK_ACTIVE=false\n'
        + '_start_studio_venv_replacement "$VENV_DIR"\n'
        + 'mkdir -p "$VENV_DIR/bin"; echo partial > "$VENV_DIR/bin/python"\n'
        + "_restore_studio_venv_replacement\n"
    )
    return subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )


@pytest.mark.parametrize("shape", ["realdir", "regularfile", "danglinglink"])
def test_rollback_restores_every_shape_the_predicate_moves_aside(tmp_path, shape):
    """Whatever _dir_has_entries calls occupied has to be restorable on failure.

    Testing with -d dropped a regular file and a dangling link: the rollback
    deactivated itself and the half-built venv stayed at $VENV_DIR.
    """
    studio_home = tmp_path / "ws"
    studio_home.mkdir()
    res = _run_rollback_lifecycle(studio_home, shape)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"

    venv = studio_home / "unsloth_studio"
    assert venv.exists() or venv.is_symlink(), "the original must be back at $VENV_DIR"
    assert not (venv / "bin" / "python").is_file(), "the half-built venv must be gone"
    stranded = list(studio_home.glob("unsloth_studio.rollback.*"))
    assert not stranded, f"backup left stranded: {[p.name for p in stranded]}"


def test_install_ps1_rollback_tests_the_path_not_the_link_target():
    """Test-Path follows a link, so a dangling backup would read as absent."""
    src = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "function Test-StudioPathPresent" in src
    for fn, nxt in (
        ("Restore-StudioVenvRollback", "Complete-StudioVenvRollback"),
        ("Complete-StudioVenvRollback", None),
    ):
        start = src.index(f"function {fn} {{")
        end = src.index(f"function {nxt} {{", start) if nxt else start + 1200
        assert (
            "Test-StudioPathPresent" in src[start:end]
        ), f"{fn} must test the backup path itself, not the link target"
