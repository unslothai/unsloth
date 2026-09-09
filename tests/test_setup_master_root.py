# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh / setup.ps1 must install the managed runtimes at the same root the resolvers read.

`llama.cpp`, `node` and `whisper.cpp` are siblings of `studio/` under `UNSLOTH_HOME`, and
`storage_roots.unsloth_home()` sends every runtime resolver there. The CLI exports
`UNSLOTH_STUDIO_HOME=<root>/studio` before running setup, so a setup that derived the runtime
parent from the Studio home alone would install them one level too deep and no GGUF model,
managed Node or dictation engine would be found. Companion to
tests/test_managed_tools_master_root.py, which holds the resolvers to each other.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


def _slice(src: str, start: str, end: str) -> str:
    begin = src.index(start)
    return src[begin:src.index(end, begin)]


def _runtime_parent(env: dict[str, str]) -> tuple[str, str]:
    """Run the shipped setup.sh derivations and report (node parent, llama.cpp parent).

    The two blocks are executed rather than pattern-matched, so a later edit that keeps the
    words and changes the order still fails here.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    master = _slice(src, "_MASTER_ROOT=\"\"\n", "# Directory-local evidence")
    node = _slice(src, "# Mirror the llama.cpp UNSLOTH_HOME derivation", "NODE_DIR=")
    llama = _slice(src, "if [ -n \"$STAGE_ROOT\" ]; then\n    UNSLOTH_HOME=", "LLAMA_CPP_DIR=")
    script = "\n".join((
        "set -u",
        master,
        node,
        llama,
        'printf "%s\\n%s\\n" "$_NODE_PARENT" "$UNSLOTH_HOME"',
    ))
    completed = subprocess.run(
        ["bash", "-c", script],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    node_parent, llama_parent = completed.stdout.splitlines()
    return node_parent, llama_parent


def _env(home: Path, **overrides: str) -> dict[str, str]:
    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "STAGE_ROOT": "",
        "RUNTIME_ROOT": "",
        "STUDIO_HOME": str(home / ".unsloth" / "studio"),
        "_STUDIO_HOME_IS_CUSTOM": "false",
    }
    env.update(overrides)
    return env


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_master_root_puts_the_runtimes_beside_studio(tmp_path):
    root = tmp_path / "portable"
    (root / "studio").mkdir(parents = True)
    env = _env(
        tmp_path / "home",
        UNSLOTH_HOME = str(root),
        # What unsloth_cli/commands/studio.py exports before it runs setup.
        STUDIO_HOME = str(root / "studio"),
        _STUDIO_HOME_IS_CUSTOM = "true",
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == root
    assert Path(llama_parent) == root


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_no_master_root_keeps_the_studio_home_layout(tmp_path):
    """The pre-existing custom-root behaviour, which the new branch must not disturb."""
    studio = tmp_path / "elsewhere" / "studio"
    studio.mkdir(parents = True)
    env = _env(
        tmp_path / "home",
        STUDIO_HOME = str(studio),
        _STUDIO_HOME_IS_CUSTOM = "true",
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == studio
    assert Path(llama_parent) == studio


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_default_install_still_uses_the_legacy_root(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    node_parent, llama_parent = _runtime_parent(_env(home))
    assert Path(node_parent) == home / ".unsloth"
    assert Path(llama_parent) == home / ".unsloth"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_staging_root_still_outranks_the_master_root(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    env = _env(
        tmp_path / "home",
        UNSLOTH_HOME = str(tmp_path / "portable"),
        STAGE_ROOT = str(stage),
        RUNTIME_ROOT = str(stage),
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == stage
    assert Path(llama_parent) == stage


def test_setup_ps1_derives_both_runtimes_from_the_master_root():
    """No PowerShell on the Linux runners, so the Windows half is held structurally: both
    derivations must go through the one helper, and it must read UNSLOTH_HOME."""
    src = SETUP_PS1.read_text(encoding = "utf-8")
    assert "function Get-MasterRootOverride" in src
    helper = _slice(src, "function Get-MasterRootOverride", "function Get-ManagedLlamaCppDir")
    assert "$env:UNSLOTH_HOME" in helper
    llama = _slice(src, "function Get-ManagedLlamaCppDir", "\n# Failure reason when the managed")
    assert "Get-MasterRootOverride" in llama
    # The master root has to lose to an explicit staging root, as it does in setup.sh.
    assert llama.index("$StagingRoot") < llama.index("Get-MasterRootOverride")
    node = _slice(src, "    $_masterRoot = Get-MasterRootOverride", "    $NodeDir = Join-Path")
    assert "$NodeParent = $_masterRoot" in node
    assert node.index("$StageRoot") < node.index("$NodeParent = $_masterRoot")
