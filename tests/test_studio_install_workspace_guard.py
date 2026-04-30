"""install.sh / install.ps1 must refuse to rm -rf an existing
$STUDIO_HOME/unsloth_studio in env-override mode unless the directory
carries a Studio sentinel (share/studio.conf or bin/unsloth). Also
asserts studio/setup.ps1 has the matching writability probe that
setup.sh:417 already performs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"


def _extract_install_sh_guard_block() -> str:
    """Pull the `if [ -x "$VENV_DIR/bin/python" ]; then ... fi` block out
    of install.sh as a self-contained snippet. Stops at the first elif so
    the block can be paired with a synthetic else and run in isolation."""
    src = INSTALL_SH.read_text()
    m = re.search(
        r'(if \[ -x "\$VENV_DIR/bin/python" \]; then\n.*?)elif \[ "\$_STUDIO_HOME_REDIRECT" != "env"',
        src,
        re.DOTALL,
    )
    assert m, "install.sh venv guard block not found"
    return m.group(1) + "fi\n"


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
    block = _extract_install_sh_guard_block()
    script = (
        f'STUDIO_HOME="{studio_home}"\n'
        f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        f'_STUDIO_HOME_REDIRECT="{redirect}"\n' + block + "echo RESULT=ok\n"
    )
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
    src = INSTALL_PS1.read_text()
    block_start = src.index("if (Test-Path -LiteralPath $VenvPython)")
    block = src[block_start : block_start + 2000]
    assert (
        "$StudioRedirectMode -eq 'env'" in block
    ), "install.ps1 must gate Remove-Item $VenvDir on env-mode"
    assert (
        "share\\studio.conf" in block
    ), "install.ps1 guard must check share\\studio.conf sentinel"
    assert (
        "bin\\unsloth.exe" in block
    ), "install.ps1 guard must check bin\\unsloth.exe sentinel"
    assert "Refusing to delete non-Studio venv" in block


def test_setup_ps1_has_writability_probe():
    src = SETUP_PS1.read_text()
    idx = src.index("if (Test-Path -LiteralPath $_studioOverride -PathType Container)")
    block = src[idx : idx + 2000]
    assert (
        "WriteAllText" in block
    ), "setup.ps1 must write-probe UNSLOTH_STUDIO_HOME like setup.sh:417"
    assert (
        "is not writable" in block
    ), "setup.ps1 probe failure must produce a clear writable-error message"


def test_env_mode_blocks_when_bin_unsloth_is_a_directory(tmp_path):
    """A bare directory at $STUDIO_HOME/bin/unsloth must NOT pass the
    sentinel. The previous `-e` test accepted any path type, allowing an
    unrelated workspace with sibling content under unsloth_studio plus
    a directory at bin/unsloth to be wiped."""
    studio_home = tmp_path / "ws"
    venv = studio_home / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    py = venv / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    (venv / "important.txt").write_text("keep me")
    (studio_home / "bin" / "unsloth").mkdir(parents = True)
    block = _extract_install_sh_guard_block()
    script = (
        f'STUDIO_HOME="{studio_home}"\n'
        f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        f'_STUDIO_HOME_REDIRECT="env"\n' + block + "echo RESULT=ok\n"
    )
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert res.returncode != 0, (
        "directory at bin/unsloth must NOT satisfy the Studio sentinel; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert (venv / "important.txt").is_file(), "unrelated workspace data must survive"


def test_env_mode_passes_when_bin_unsloth_is_a_symlink(tmp_path):
    """A symlink at $STUDIO_HOME/bin/unsloth (real installer artefact)
    must still satisfy the sentinel after the leaf-only tightening."""
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
    block = _extract_install_sh_guard_block()
    script = (
        f'STUDIO_HOME="{studio_home}"\n'
        f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        f'_STUDIO_HOME_REDIRECT="env"\n' + block + "echo RESULT=ok\n"
    )
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
    """The Test-Path checks that gate Remove-Item $VenvDir must use
    -PathType Leaf so a directory at the sentinel path cannot satisfy them."""
    src = INSTALL_PS1.read_text()
    block_start = src.index("if (Test-Path -LiteralPath $VenvPython)")
    block = src[block_start : block_start + 2000]
    assert (
        'share\\studio.conf") -PathType Leaf' in block
    ), "install.ps1 share\\studio.conf check must use -PathType Leaf"
    assert (
        'bin\\unsloth.exe") -PathType Leaf' in block
    ), "install.ps1 bin\\unsloth.exe check must use -PathType Leaf"


def test_setup_ps1_stale_venv_has_env_mode_guard():
    """studio/setup.ps1 stale-venv rebuild branch must mirror install.ps1:
    refuse to Remove-Item $VenvDir under custom-root mode unless the root
    carries a Studio sentinel (in-VENV marker, share\\studio.conf, or
    bin\\unsloth.exe leaf)."""
    src = SETUP_PS1.read_text()
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
    assert (
        guard_idx < rm_idx
    ), "custom-root guard must precede Remove-Item -LiteralPath $VenvDir"


def test_setup_sh_prebuilt_llama_cpp_has_ownership_guard():
    """studio/setup.sh prebuilt llama.cpp path must call
    _assert_studio_owned_or_absent before invoking install_llama_prebuilt.py
    so an unrelated $UNSLOTH_STUDIO_HOME/llama.cpp is not displaced by
    the helper's os.replace()."""
    src = SETUP_SH.read_text()
    idx = src.index("installing prebuilt llama.cpp...")
    block = src[idx : idx + 2000]
    assert (
        '_assert_studio_owned_or_absent "$LLAMA_CPP_DIR" "llama.cpp install"' in block
    ), "setup.sh must guard the prebuilt llama.cpp path with the ownership marker"
    guard_idx = block.index('_assert_studio_owned_or_absent "$LLAMA_CPP_DIR"')
    # Anchor on the actual command-array entry, not the why-comment mention.
    helper_idx = block.index('python "$SCRIPT_DIR/install_llama_prebuilt.py"')
    assert (
        guard_idx < helper_idx
    ), "ownership guard must precede the install_llama_prebuilt.py call"


def test_setup_ps1_prebuilt_llama_cpp_has_ownership_guard():
    """Mirror check for studio/setup.ps1: prebuilt llama.cpp path must
    call Assert-StudioOwnedOrAbsent before invoking install_llama_prebuilt.py."""
    src = SETUP_PS1.read_text()
    idx = src.index("installing prebuilt llama.cpp bundle (preferred path)")
    block = src[idx : idx + 2000]
    assert (
        'Assert-StudioOwnedOrAbsent -Path $LlamaCppDir -Label "llama.cpp install"'
        in block
    ), "setup.ps1 must guard the prebuilt llama.cpp path with Assert-StudioOwnedOrAbsent"
    guard_idx = block.index("Assert-StudioOwnedOrAbsent -Path $LlamaCppDir")
    # Anchor on the actual command-array entry, not the why-comment mention.
    helper_idx = block.index('"$PSScriptRoot\\install_llama_prebuilt.py"')
    assert (
        guard_idx < helper_idx
    ), "Assert-StudioOwnedOrAbsent must precede the install_llama_prebuilt.py call"


def test_env_mode_passes_when_venv_marker_present(tmp_path):
    """install.sh env-mode guard must accept the in-VENV
    .unsloth-studio-owned marker as a primary sentinel so a partial
    install (uv venv created, sentinels not yet written) is recoverable
    by re-running install.sh."""
    studio_home = tmp_path / "ws"
    res = _run_install_guard(studio_home, redirect = "env", create_venv_marker = True)
    assert res.returncode == 0, (
        f"in-VENV marker must allow cleanup; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "RESULT=ok" in res.stdout
    assert not (studio_home / "unsloth_studio").exists()


def test_env_mode_blocks_when_bin_unsloth_is_symlink_to_directory(tmp_path):
    """install.sh env-mode guard must NOT accept a symlink-to-directory at
    bin/unsloth as a Studio sentinel. Iter1's standalone -L test let any
    symlink (including symlinks to dirs and broken symlinks) bypass the
    guard; iter2 dropped that test so only -f (file or symlink-to-file)
    counts."""
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
    block = _extract_install_sh_guard_block()
    script = (
        f'STUDIO_HOME="{studio_home}"\n'
        f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        f'_STUDIO_HOME_REDIRECT="env"\n' + block + "echo RESULT=ok\n"
    )
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
    block = _extract_install_sh_guard_block()
    script = (
        f'STUDIO_HOME="{studio_home}"\n'
        f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        f'_STUDIO_HOME_REDIRECT="env"\n' + block + "echo RESULT=ok\n"
    )
    res = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        text = True,
        capture_output = True,
    )
    assert res.returncode != 0, (
        "broken symlink at bin/unsloth must NOT pass; "
        f"stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert (venv / "important.txt").is_file()


def test_install_sh_writes_venv_marker_after_uv_venv():
    """install.sh must write the .unsloth-studio-owned marker into
    $VENV_DIR right after `uv venv` succeeds so the env-mode deletion
    guard accepts it on the next install run."""
    src = INSTALL_SH.read_text()
    create_idx = src.index('run_install_cmd "create venv" uv venv "$VENV_DIR"')
    tail = src[create_idx : create_idx + 600]
    assert (
        ".unsloth-studio-owned" in tail
    ), "install.sh must write .unsloth-studio-owned after uv venv create"


def test_install_ps1_writes_venv_marker_after_uv_venv():
    """install.ps1 must write the .unsloth-studio-owned marker into
    $VenvDir after `uv venv` succeeds."""
    src = INSTALL_PS1.read_text()
    venv_create = src.index("uv venv $VenvDir --python")
    tail = src[venv_create : venv_create + 1500]
    assert (
        ".unsloth-studio-owned" in tail
    ), "install.ps1 must write .unsloth-studio-owned after uv venv create"


def test_install_ps1_guard_accepts_venv_marker():
    """install.ps1 env-mode guard must accept the in-VENV
    .unsloth-studio-owned marker as a primary sentinel."""
    src = INSTALL_PS1.read_text()
    block_start = src.index("if (Test-Path -LiteralPath $VenvPython)")
    block = src[block_start : block_start + 2000]
    assert (
        '$VenvDir ".unsloth-studio-owned") -PathType Leaf' in block
    ), "install.ps1 guard must check the in-VENV marker with -PathType Leaf"


def test_setup_helpers_gate_on_canonical_custom_root():
    """Both _assert_studio_owned_or_absent (setup.sh) and
    Assert-StudioOwnedOrAbsent (setup.ps1) must gate on a canonical
    custom-vs-legacy comparison so an explicit override that resolves
    to the legacy default does not trip the guard for pre-PR T5
    sidecar venvs or llama.cpp dirs."""
    sh_src = SETUP_SH.read_text()
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

    ps_src = SETUP_PS1.read_text()
    ps_idx = ps_src.index("function Assert-StudioOwnedOrAbsent")
    ps_func = ps_src[ps_idx : ps_idx + 800]
    assert (
        "$StudioHomeIsCustom -and" in ps_func
    ), "setup.ps1 Assert-StudioOwnedOrAbsent must gate on $StudioHomeIsCustom"
    assert (
        "$StudioOwnedMarker) -PathType Leaf" in ps_func
    ), "setup.ps1 marker check must use -PathType Leaf so a directory cannot satisfy it"


def test_setup_ps1_inplace_git_sync_marks_studio_owned():
    """setup.ps1 in-place git-sync branch (when $LlamaCppDir/.git exists)
    must call Mark-StudioOwned after a successful sync so a later prebuilt
    update path's Assert-StudioOwnedOrAbsent does not exit."""
    src = SETUP_PS1.read_text()
    inplace_idx = src.index('Test-Path -LiteralPath (Join-Path $LlamaCppDir ".git")')
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
    """setup.ps1 in-place git-sync branch must call Assert-StudioOwnedOrAbsent
    BEFORE any destructive git operation (remote set-url, checkout -B, clean
    -fdx). Asymmetric to the prebuilt path and the temp-dir-swap path which
    both guard."""
    src = SETUP_PS1.read_text()
    inplace_idx = src.index('Test-Path -LiteralPath (Join-Path $LlamaCppDir ".git")')
    clone_idx = src.index("Cloning llama.cpp @", inplace_idx)
    inplace_block = src[inplace_idx:clone_idx]
    assert (
        "Assert-StudioOwnedOrAbsent -Path $LlamaCppDir" in inplace_block
    ), "in-place git-sync must Assert-StudioOwnedOrAbsent before mutating $LlamaCppDir"
    guard_idx = inplace_block.index("Assert-StudioOwnedOrAbsent -Path $LlamaCppDir")
    git_idx = inplace_block.index("git -C $LlamaCppDir remote set-url")
    assert (
        guard_idx < git_idx
    ), "Assert-StudioOwnedOrAbsent must precede the first git mutation"


def _extract_check_health_function() -> str:
    src = INSTALL_SH.read_text()
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
    """Hex digest baked at install time matches the backend's
    /api/health studio_root_id -- launcher attaches to its own backend."""
    expected_id = "a" * 64
    rc = _run_check_health(
        expected_id,
        f'{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{expected_id}"}}',
    )
    assert rc == 0, f"matching studio_root_id must allow attach (rc={rc})"


def test_check_health_rejects_mismatched_studio_root_id():
    """Different install root → different sha256 → reject. Workspace
    isolation: launcher A must not open Studio B running on the same port."""
    expected_id = "a" * 64
    other_id = "b" * 64
    rc = _run_check_health(
        expected_id,
        f'{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{other_id}"}}',
    )
    assert rc != 0, "mismatched studio_root_id must reject attach (workspace isolation)"


def test_check_health_rejects_missing_studio_root_id_field():
    """A backend that omits studio_root_id (older or non-conforming) must
    not be attached to when an expected id is baked into the launcher."""
    expected_id = "a" * 64
    rc = _run_check_health(
        expected_id,
        '{"status":"healthy","service":"Unsloth UI Backend"}',
    )
    assert rc != 0, "missing studio_root_id field must reject attach"


def test_check_health_no_baked_id_accepts_any_healthy_backend():
    """If _EXPECTED_STUDIO_ROOT_ID is empty (e.g. install-time hash failed
    to compute), the launcher falls back to the legacy contract and accepts
    any healthy Unsloth backend."""
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


def test_check_health_handles_path_with_backslash_via_hash():
    """Iter3 used raw shell match against JSON-escaped studio_root, which
    failed for paths containing `\\` or `"` (FastAPI emits `\\\\` and `\\\"`).
    The hex digest baked at install time is escape-free, so paths with
    these characters round-trip correctly."""
    import hashlib

    weird_path = "/tmp/back\\slash"
    expected_id = hashlib.sha256(weird_path.encode("utf-8")).hexdigest()
    rc = _run_check_health(
        expected_id,
        f'{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{expected_id}"}}',
    )
    assert (
        rc == 0
    ), "path with backslash must round-trip via hash (no JSON escape issue)"


def test_install_ps1_test_studio_health_verifies_studio_root_id():
    """install.ps1 Test-StudioHealth must compare studio_root_id against
    the install-time-baked $_ExpectedStudioRootId, not the runtime env var."""
    src = INSTALL_PS1.read_text()
    fn_start = src.index("function Test-StudioHealth")
    fn_end = src.index("\n}\n", fn_start) + 2
    fn = src[fn_start:fn_end]
    assert (
        "studio_root_id" in fn
    ), "Test-StudioHealth must inspect the studio_root_id field"
    assert (
        "$_ExpectedStudioRootId" in fn
    ), "Test-StudioHealth must compare against the install-time baked $_ExpectedStudioRootId"


def test_install_ps1_bakes_studio_root_id_into_launcher():
    """install.ps1 must compute sha256($StudioHome) and bake it into the
    generated launcher as $_ExpectedStudioRootId so the launcher can
    verify the backend belongs to THIS install."""
    src = INSTALL_PS1.read_text()
    assert (
        "$_studioRootId" in src
    ), "install.ps1 must compute $_studioRootId from $StudioHome"
    assert (
        "SHA256" in src
        and ".HashData" in src
        or "SHA256" in src
        and ".ComputeHash" in src
    ), "install.ps1 must use SHA-256 hashing for the studio root id"
    assert (
        "$_ExpectedStudioRootId" in src
    ), "install.ps1 must bake $_ExpectedStudioRootId into the launcher"


def test_health_endpoint_exposes_studio_root_id_not_raw_path():
    """studio/backend/main.py /api/health must expose studio_root_id (a
    hex digest) and NOT the raw studio_root path. Studio supports
    `-H 0.0.0.0`; an unauthenticated /api/health that returns the raw
    install path leaks username, home dir, workspace name, etc."""
    main_py = REPO_ROOT / "studio" / "backend" / "main.py"
    src = main_py.read_text()
    health_idx = src.index('@app.get("/api/health")')
    health_block = src[health_idx : health_idx + 1500]
    assert (
        '"studio_root_id"' in health_block
    ), "/api/health must expose studio_root_id (hex digest)"
    assert (
        '"studio_root":' not in health_block
    ), "/api/health must NOT expose the raw studio_root path (information disclosure)"
    assert (
        "_studio_root_id()" in health_block
    ), "/api/health must call the _studio_root_id helper"


def test_install_sh_bakes_studio_root_id_into_launcher():
    """install.sh must compute sha256($STUDIO_HOME) and substitute it into
    the launcher heredoc placeholder for ALL modes (env / home / default)
    so the launcher's _check_health rejects sibling Studios on the same port."""
    src = INSTALL_SH.read_text()
    assert (
        "_css_studio_root_id" in src
    ), "install.sh must compute _css_studio_root_id from $STUDIO_HOME"
    assert (
        "@@STUDIO_ROOT_ID@@" in src
    ), "install.sh must use @@STUDIO_ROOT_ID@@ placeholder in the launcher heredoc"
    assert (
        "s|@@STUDIO_ROOT_ID@@|$_css_studio_root_id|g" in src
    ), "install.sh must sed-substitute @@STUDIO_ROOT_ID@@ unconditionally (not just env-mode)"


def test_tauri_preflight_scrubs_studio_home_env():
    """All three Tauri CLI-spawn sites that lacked the scrub must now
    env_remove UNSLOTH_STUDIO_HOME and STUDIO_HOME, mirroring
    process.rs / install.rs / desktop_auth.rs / update.rs."""
    preflight = (
        REPO_ROOT / "studio" / "src-tauri" / "src" / "preflight.rs"
    ).read_text()
    commands = (REPO_ROOT / "studio" / "src-tauri" / "src" / "commands.rs").read_text()
    # Both functions in preflight.rs (run_cli_probe + probe_cli_capability)
    # must scrub. Count occurrences -- expect 2 in preflight, 1 in commands.
    assert (
        preflight.count('cmd.env_remove("UNSLOTH_STUDIO_HOME")') >= 2
    ), "preflight.rs must scrub UNSLOTH_STUDIO_HOME in both run_cli_probe and probe_cli_capability"
    assert (
        preflight.count('cmd.env_remove("STUDIO_HOME")') >= 2
    ), "preflight.rs must scrub STUDIO_HOME in both run_cli_probe and probe_cli_capability"
    assert (
        'cmd.env_remove("UNSLOTH_STUDIO_HOME")' in commands
    ), "commands.rs check_install_status must scrub UNSLOTH_STUDIO_HOME"
    assert (
        'cmd.env_remove("STUDIO_HOME")' in commands
    ), "commands.rs check_install_status must scrub STUDIO_HOME"


def test_install_sh_shim_uses_atomic_replace():
    """install.sh shim install must use ln -sfn for atomic replace; the
    older `rm -f ...; ln -s ...` left a window where the shim was missing."""
    src = INSTALL_SH.read_text()
    shim_idx = src.index('_shim_path="$_LOCAL_BIN/unsloth"')
    block = src[shim_idx : shim_idx + 1500]
    assert (
        'ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path"' in block
    ), "install.sh must use ln -sfn for atomic shim replacement"
    assert (
        'rm -f -- "$_shim_path"' not in block
    ), "the explicit rm + ln pair must be replaced by atomic ln -sfn"


def test_install_sh_canonicalizes_studio_home_for_root_id_hash(tmp_path):
    """install.sh must canonicalize $STUDIO_HOME via cd -P / pwd -P before
    hashing so the digest matches the backend's Path(sys.prefix).resolve()
    in default and home-redirect modes (where $HOME may be a symlink or
    have a trailing slash)."""
    src = INSTALL_SH.read_text()
    assert (
        '_css_studio_root_input="$(CDPATH= cd -P -- "$STUDIO_HOME" 2>/dev/null && pwd -P)"'
        in src
    ), "install.sh must canonicalize STUDIO_HOME with cd -P/pwd -P before hashing"
    assert (
        '"$_css_python" - "$_css_studio_root_input"' in src
    ), "install.sh must hash the canonicalized input, not the raw STUDIO_HOME"
    real = tmp_path / "real_root"
    real.mkdir()
    sym = tmp_path / "sym_root"
    sym.symlink_to(real)
    for raw in (str(real) + "/", str(sym), str(real)):
        res = subprocess.run(
            ["bash", "-c", f'CDPATH= cd -P -- "{raw}" 2>/dev/null && pwd -P'],
            text = True,
            capture_output = True,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == str(
            real
        ), f"cd -P / pwd -P must canonicalize {raw!r} to {real}; got {res.stdout!r}"


def test_install_sh_create_shortcuts_uses_venv_python_first():
    """The studio_root_id hash command must prefer the venv Python so a host
    that runs uv-managed Python without a system python3 still produces a
    non-empty discriminator."""
    src = INSTALL_SH.read_text()
    fn_start = src.index('_css_data_dir="$DATA_DIR"')
    block = src[fn_start : fn_start + 2500]
    venv_idx = block.index('_css_python="$_css_exe_dir/python"')
    fallback_idx = block.index("command -v python3 2>/dev/null", venv_idx)
    hash_idx = block.index('_css_studio_root_id=$("$_css_python"', venv_idx)
    assert (
        venv_idx < fallback_idx < hash_idx
    ), "venv Python must be tried before system python3, and the hash must run after both lookups"


def test_install_sh_create_shortcuts_fails_fast_when_no_python():
    """If neither venv Python nor system python3/python is found, _create_shortcuts
    must `return 1` instead of silently baking an empty studio_root_id (which
    would disable the launcher's same-install discriminator)."""
    src = INSTALL_SH.read_text()
    fn_start = src.index('_css_data_dir="$DATA_DIR"')
    block = src[fn_start : fn_start + 2500]
    assert (
        "[WARN] Cannot create launcher: Python not found for studio_root_id" in block
    ), "install.sh must warn when no Python is available"
    assert (
        "[WARN] Cannot create launcher: failed to compute studio_root_id" in block
    ), "install.sh must warn when the python compute itself produces no output"
    assert (
        block.count("return 1") >= 2
    ), "both the no-Python branch and the empty-hash branch must `return 1`"
    assert (
        '|| echo ""' not in block.split("_css_studio_root_id=")[1].split("PY\n)")[0]
    ), 'install.sh must NOT silently swallow hash failures with || echo ""'


def test_install_sh_bakes_installed_is_env_mode_flag_in_launcher():
    """install.sh must bake the install-time mode (env vs default/home) into
    the generated launcher so PORT_FILE / namespaced LOCK_DIR cannot be
    flipped on by a sourced custom-root studio.conf in the user's shell."""
    src = INSTALL_SH.read_text()
    assert (
        "_INSTALLED_IS_ENV_MODE='@@INSTALLED_IS_ENV_MODE@@'" in src
    ), "launcher heredoc must declare _INSTALLED_IS_ENV_MODE='@@INSTALLED_IS_ENV_MODE@@'"
    assert (
        "_css_is_env_mode=false" in src
    ), "install.sh must default _css_is_env_mode to false"
    assert (
        '[ "$_STUDIO_HOME_REDIRECT" = "env" ] && _css_is_env_mode=true' in src
    ), "install.sh must set _css_is_env_mode=true only when _STUDIO_HOME_REDIRECT=env"
    assert (
        "s|@@INSTALLED_IS_ENV_MODE@@|$_css_is_env_mode|g" in src
    ), "install.sh sed pipeline must substitute @@INSTALLED_IS_ENV_MODE@@"


def test_install_sh_launcher_gates_port_file_on_baked_flag_not_runtime_env():
    """The launcher's PORT_FILE / namespaced LOCK_DIR must be gated on the
    baked $_INSTALLED_IS_ENV_MODE flag, not the runtime $UNSLOTH_STUDIO_HOME.
    Sourcing a custom-root studio.conf in shell must not flip a default-mode
    launcher into env-mode behavior."""
    src = INSTALL_SH.read_text()
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
        'if [ -n "${UNSLOTH_STUDIO_HOME:-}" ]; then\n    if command -v cksum'
        not in heredoc
    ), "launcher must NOT gate PORT_FILE on runtime UNSLOTH_STUDIO_HOME"

    def _run_launcher_gate(installed_flag: str, runtime_env: dict) -> str:
        # Reproduce just the LOCK_DIR/PORT_FILE init block in isolation.
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

    # default-mode install should NEVER set PORT_FILE, even if UNSLOTH_STUDIO_HOME leaks in.
    assert (
        _run_launcher_gate("false", {"UNSLOTH_STUDIO_HOME": "/tmp/leaked"}) == ""
    ), "default-mode launcher must keep PORT_FILE empty even with UNSLOTH_STUDIO_HOME in env"
    # env-mode install should set PORT_FILE regardless of runtime env.
    assert (
        _run_launcher_gate("true", {}) == "/tmp/test_data_dir/studio.port"
    ), "env-mode launcher must set PORT_FILE based on baked DATA_DIR"


def test_main_py_studio_root_id_caches_at_module_load():
    """_studio_root_id() is called on every /api/health poll; the digest is
    stable for the lifetime of the process so it must be computed once at
    module load and re-used (avoids a hot-path filesystem probe and protects
    against transient FS errors during health polling)."""
    main_py = (REPO_ROOT / "studio" / "backend" / "main.py").read_text()
    assert (
        "_STUDIO_ROOT_ID_CACHE: str = hashlib.sha256(" in main_py
    ), "main.py must compute _STUDIO_ROOT_ID_CACHE at module load"
    fn_idx = main_py.index("def _studio_root_id() -> str:")
    next_def_idx = main_py.index("\ndef ", fn_idx + 1)
    fn_block = main_py[fn_idx:next_def_idx]
    assert (
        "return _STUDIO_ROOT_ID_CACHE" in fn_block
    ), "_studio_root_id() body must return the cached value"
    assert (
        "hashlib.sha256(" not in fn_block
    ), "_studio_root_id() must NOT recompute sha256 on every call"


def test_llama_cpp_search_roots_handles_studio_root_oserror():
    """_find_llama_server_binary calls studio_root() which can raise
    OSError or ValueError from Path.expanduser().resolve() (broken symlink,
    null byte). The except clause must mirror sibling _kill_orphaned_servers
    (which catches the same trio) so inference startup does not crash."""
    llama_cpp = (
        REPO_ROOT / "studio" / "backend" / "core" / "inference" / "llama_cpp.py"
    ).read_text()
    find_block_start = llama_cpp.index("_find_llama_server_binary")
    find_block = llama_cpp[find_block_start : find_block_start + 4000]
    assert (
        "except (ImportError, OSError, ValueError):" in find_block
    ), "_find_llama_server_binary must catch (ImportError, OSError, ValueError) from studio_root()"
    kill_def_idx = llama_cpp.index("def _kill_orphaned_servers")
    kill_block = llama_cpp[kill_def_idx : kill_def_idx + 4000]
    assert (
        "except (ImportError, OSError, ValueError):" in kill_block
    ), "sibling _kill_orphaned_servers must keep its (ImportError, OSError, ValueError) handler"


def test_main_py_studio_root_id_hashes_resolved_root_not_unresolved():
    """The cached digest must hash _STUDIO_ROOT_RESOLVED (the canonicalized
    path) so it lines up with install.sh's `cd -P/pwd -P` digest input.
    Hashing str(_studio_root()) instead would diverge in default mode where
    the storage_roots fallback returns Path.home()/.unsloth/studio without
    .resolve(), breaking launchers on systems with a symlinked $HOME."""
    main_py = (REPO_ROOT / "studio" / "backend" / "main.py").read_text()
    cache_idx = main_py.index("_STUDIO_ROOT_ID_CACHE: str = hashlib.sha256(")
    cache_block = main_py[cache_idx : cache_idx + 200]
    assert (
        "str(_STUDIO_ROOT_RESOLVED)" in cache_block
    ), "_STUDIO_ROOT_ID_CACHE must hash the resolved root, not the raw _studio_root() return"
    assert (
        "str(_studio_root())" not in cache_block
    ), "_STUDIO_ROOT_ID_CACHE must NOT recompute studio_root() (would lose the .resolve())"


def test_install_sh_root_id_matches_backend_resolved_under_symlinked_home(tmp_path):
    """End-to-end behavioral check: install.sh's `cd -P/pwd -P` canonicalization
    must produce the same digest the backend computes via Path.resolve() on a
    symlinked $HOME. Reproduces the exact reviewer scenario."""
    real = tmp_path / "realhome"
    real.mkdir()
    link = tmp_path / "linkhome"
    link.symlink_to(real)
    studio_home = real / ".unsloth" / "studio"
    studio_home.mkdir(parents = True)
    raw_via_link = f"{link}/.unsloth/studio"
    res = subprocess.run(
        ["bash", "-c", f'CDPATH= cd -P -- "{raw_via_link}" 2>/dev/null && pwd -P'],
        capture_output = True,
        text = True,
    )
    install_canon = res.stdout.strip()
    backend_canon = str(Path(raw_via_link).resolve())
    assert (
        install_canon == backend_canon
    ), f"install.sh cd -P must equal Path.resolve(); got {install_canon!r} vs {backend_canon!r}"
    import hashlib

    install_id = hashlib.sha256(
        install_canon.encode("utf-8", "surrogatepass")
    ).hexdigest()
    backend_id = hashlib.sha256(
        backend_canon.encode("utf-8", "surrogatepass")
    ).hexdigest()
    assert (
        install_id == backend_id
    ), f"install vs backend digest must match for symlinked $HOME; got {install_id} vs {backend_id}"


def test_install_sh_substitutes_root_id_before_data_dir():
    """The two-stage sed substitution must bake @@STUDIO_ROOT_ID@@ /
    @@INSTALLED_IS_ENV_MODE@@ first (non-user-controlled), then @@DATA_DIR@@
    (user-controlled). A custom $DATA_DIR containing the literal text
    @@STUDIO_ROOT_ID@@ must not be mutated by the global root-id sed pass."""
    src = INSTALL_SH.read_text()
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
    """Behavioral subprocess test: a $DATA_DIR containing the literal text
    `@@STUDIO_ROOT_ID@@` must not be mutated when the placeholder pass runs
    first; only the actual placeholder occurrences in the launcher template
    are replaced."""
    src = INSTALL_SH.read_text()
    heredoc_start = src.index("cat > \"$_css_launcher\" << 'LAUNCHER_EOF'")
    heredoc_body_start = src.index("\n", heredoc_start) + 1
    heredoc_body_end = src.index("LAUNCHER_EOF\n", heredoc_start)
    template = src[heredoc_body_start:heredoc_body_end]
    launcher_path = tmp_path / "launch.sh"
    launcher_path.write_text(template)
    # Run the iter6 sed order: root-id first, then data-dir.
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
    final = launcher_path.read_text()
    assert (
        f"DATA_DIR='{weird_data_dir}'" in final
    ), f"DATA_DIR must be preserved verbatim (no @@STUDIO_ROOT_ID@@ mutation); got: {final[:500]}"
    assert (
        f"_EXPECTED_STUDIO_ROOT_ID='{root_id}'" in final
    ), "STUDIO_ROOT_ID placeholder must still be substituted in the launcher heredoc"


def test_install_ps1_canonicalizes_studio_home_before_root_id_hash():
    """install.ps1 must Resolve-Path the $StudioHome (after CreateDirectory)
    before computing $_studioRootId, so a junctioned USERPROFILE produces the
    same digest the backend computes via Path.resolve()."""
    src = INSTALL_PS1.read_text()
    bytes_idx = src.index("$_studioRootBytes = [Text.Encoding]::UTF8.GetBytes(")
    context = src[max(0, bytes_idx - 1000) : bytes_idx + 200]
    assert (
        "$_studioRootForId = $StudioHome" in context
    ), "install.ps1 must capture $StudioHome into $_studioRootForId for canonicalization"
    assert (
        "[System.IO.Directory]::CreateDirectory($_studioRootForId)" in context
    ), "install.ps1 must ensure the directory exists so Resolve-Path can succeed"
    assert (
        "Resolve-Path -LiteralPath $_studioRootForId" in context
    ), "install.ps1 must Resolve-Path before computing the SHA256 digest"
    assert (
        "GetBytes($_studioRootForId)" in context
    ), "install.ps1 must hash the canonicalized $_studioRootForId, not the raw $StudioHome"
