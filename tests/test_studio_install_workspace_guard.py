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
    block = _extract_install_sh_guard_block()
    script = (
        f'STUDIO_HOME="{studio_home}"\n'
        f'VENV_DIR="$STUDIO_HOME/unsloth_studio"\n'
        f'_STUDIO_HOME_REDIRECT="{redirect}"\n'
        + block
        + 'echo RESULT=ok\n'
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
    res = _run_install_guard(
        studio_home, redirect = "env", create_share_conf = True
    )
    assert res.returncode == 0, (
        f"share/studio.conf sentinel must allow cleanup;"
        f" stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "RESULT=ok" in res.stdout
    assert not (studio_home / "unsloth_studio").exists()


def test_env_mode_passes_when_bin_unsloth_shim_present(tmp_path):
    studio_home = tmp_path / "ws"
    res = _run_install_guard(
        studio_home, redirect = "env", create_bin_shim = True
    )
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
    assert "$StudioRedirectMode -eq 'env'" in block, (
        "install.ps1 must gate Remove-Item $VenvDir on env-mode"
    )
    assert 'share\\studio.conf' in block, (
        "install.ps1 guard must check share\\studio.conf sentinel"
    )
    assert 'bin\\unsloth.exe' in block, (
        "install.ps1 guard must check bin\\unsloth.exe sentinel"
    )
    assert "Refusing to delete non-Studio venv" in block


def test_setup_ps1_has_writability_probe():
    src = SETUP_PS1.read_text()
    idx = src.index("if (Test-Path -LiteralPath $_studioOverride -PathType Container)")
    block = src[idx : idx + 2000]
    assert "WriteAllText" in block, (
        "setup.ps1 must write-probe UNSLOTH_STUDIO_HOME like setup.sh:417"
    )
    assert "is not writable" in block, (
        "setup.ps1 probe failure must produce a clear writable-error message"
    )
