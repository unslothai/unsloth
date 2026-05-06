"""studio/setup.sh must mirror nvm v0.40.1's install-directory selection
when picking the NVM_DIR the parent shell sources from. nvm's own
nvm_install_dir / nvm_default_install_dir use the precedence
NVM_DIR -> $XDG_CONFIG_HOME/nvm -> $HOME/.nvm. If the script diverges
the installer writes nvm to one path and the script sources from
another, leaving `nvm install --lts` failing with exit 127 on
Distrobox / XDG-style setups."""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"


def _extract_nvm_dir_block() -> str:
    src = SETUP_SH.read_text()
    m = re.search(
        r'(    if \[ -n "\$\{NVM_DIR:-\}" \]; then\n.*?\n    fi\n)',
        src,
        re.DOTALL,
    )
    assert m, "Could not locate NVM_DIR selection block in studio/setup.sh"
    return textwrap.dedent(m.group(1))


def _run(block: str, env_setup: str) -> str:
    script = f"set -eu\n{env_setup}\n{block}\nprintf '%s' \"$NVM_DIR\"\n"
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def test_explicit_nvm_dir_is_preserved():
    block = _extract_nvm_dir_block()
    got = _run(block, 'export NVM_DIR="/opt/custom/nvm"\nexport HOME="/home/u"\nexport XDG_CONFIG_HOME="/home/u/.config"')
    assert got == "/opt/custom/nvm", got


def test_explicit_nvm_dir_wins_over_xdg():
    block = _extract_nvm_dir_block()
    got = _run(block, 'export NVM_DIR="/opt/custom/nvm"\nexport HOME="/home/u"\nunset XDG_CONFIG_HOME')
    assert got == "/opt/custom/nvm", got


def test_xdg_config_home_used_when_nvm_dir_unset():
    block = _extract_nvm_dir_block()
    got = _run(block, 'unset NVM_DIR\nexport HOME="/home/u"\nexport XDG_CONFIG_HOME="/home/u/.config"')
    assert got == "/home/u/.config/nvm", got


def test_xdg_config_home_used_when_nvm_dir_empty():
    block = _extract_nvm_dir_block()
    got = _run(block, 'export NVM_DIR=""\nexport HOME="/home/u"\nexport XDG_CONFIG_HOME="/home/u/.config"')
    assert got == "/home/u/.config/nvm", got


def test_home_nvm_fallback_when_both_unset():
    block = _extract_nvm_dir_block()
    got = _run(block, 'unset NVM_DIR\nunset XDG_CONFIG_HOME\nexport HOME="/home/u"')
    assert got == "/home/u/.nvm", got


def test_home_nvm_fallback_when_xdg_empty():
    block = _extract_nvm_dir_block()
    got = _run(block, 'unset NVM_DIR\nexport XDG_CONFIG_HOME=""\nexport HOME="/home/u"')
    assert got == "/home/u/.nvm", got


def test_block_runs_clean_under_set_u():
    block = _extract_nvm_dir_block()
    script = f'set -eu\nunset NVM_DIR\nunset XDG_CONFIG_HOME\nexport HOME="/home/u"\n{block}\nprintf "%s" "$NVM_DIR"\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout == "/home/u/.nvm", out.stdout


def test_setup_sh_does_not_unconditionally_overwrite_nvm_dir():
    """Regression guard: the unconditional, top-level
    `    export NVM_DIR="$HOME/.nvm"` (4-space indent, sibling of the
    surrounding `if [ "$NEED_NODE" = true ]` body) that broke
    Distrobox / XDG installations must not return."""
    src = SETUP_SH.read_text()
    pattern = re.compile(r'^    export NVM_DIR="\$HOME/\.nvm"$', re.MULTILINE)
    assert not pattern.search(src), (
        "studio/setup.sh contains an unconditional NVM_DIR override at "
        "the outer NEED_NODE scope; this regresses the precedence fix."
    )


def test_setup_sh_block_mentions_xdg_config_home():
    src = SETUP_SH.read_text()
    assert "XDG_CONFIG_HOME" in src, (
        "studio/setup.sh must honor XDG_CONFIG_HOME to match upstream "
        "nvm v0.40.1's nvm_default_install_dir."
    )
