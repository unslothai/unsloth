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
    return src[begin : src.index(end, begin)]


def _runtime_parent(env: dict[str, str]) -> tuple[str, str]:
    """Run the shipped setup.sh derivations and report (node parent, llama.cpp parent).

    The two blocks are executed rather than pattern-matched, so a later edit that keeps the
    words and changes the order still fails here.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    master = _slice(src, "# Stripped before anything else", "# Directory-local evidence")
    node = _slice(src, "# Mirror the llama.cpp UNSLOTH_HOME derivation", "NODE_DIR=")
    llama = _slice(src, 'if [ -n "$STAGE_ROOT" ]; then\n    UNSLOTH_HOME=', "LLAMA_CPP_DIR=")
    script = "\n".join(
        (
            "set -u",
            master,
            node,
            llama,
            'printf "%s\\n%s\\n" "$_NODE_PARENT" "$UNSLOTH_HOME"',
        )
    )
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


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_padded_master_root_names_the_same_directory(tmp_path):
    """storage_roots.unsloth_home() and the CLI both .strip(), so setup has to as well: an
    unstripped value installs under a directory whose name carries the whitespace."""
    root = tmp_path / "portable"
    (root / "studio").mkdir(parents = True)
    env = _env(
        tmp_path / "home",
        UNSLOTH_HOME = f"  {root}  ",
        STUDIO_HOME = str(root / "studio"),
        _STUDIO_HOME_IS_CUSTOM = "true",
    )
    node_parent, llama_parent = _runtime_parent(env)
    assert Path(node_parent) == root
    assert Path(llama_parent) == root


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_blank_master_root_counts_as_unset(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    node_parent, llama_parent = _runtime_parent(_env(home, UNSLOTH_HOME = "   "))
    assert Path(node_parent) == home / ".unsloth"
    assert Path(llama_parent) == home / ".unsloth"


BUILD_WHISPER = REPO_ROOT / "scripts" / "build_whisper_cpp.sh"


def _whisper_root_block() -> str:
    """The shipped root selection, from the normalizer down to the INSTALL_DIR it decides."""
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    return _slice(src, "_root_value() {", "STUDIO_OWNED_MARKER=")


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_the_whisper_builder_installs_under_the_master_root(tmp_path):
    """setup.sh runs this with UNSLOTH_STUDIO_HOME=<root>/studio still inherited, so a builder
    that preferred it would install a level below _managed_whisper_cpp_dir()."""
    root = tmp_path / "portable"
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    block = _whisper_root_block()
    script = "\n".join(("set -eu", block, 'printf "%s\\n" "$INSTALL_DIR"'))
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "UNSLOTH_HOME": str(root),
            "UNSLOTH_STUDIO_HOME": str(root / "studio"),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == root / "whisper.cpp"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_the_whisper_builder_still_honours_a_studio_home_alone(tmp_path):
    studio = tmp_path / "elsewhere" / "studio"
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    block = _whisper_root_block()
    script = "\n".join(("set -eu", block, 'printf "%s\\n" "$INSTALL_DIR"'))
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "UNSLOTH_STUDIO_HOME": str(studio),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == studio / "whisper.cpp"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_blank_master_root_does_not_outrank_the_whisper_studio_home(tmp_path):
    """${VAR:-} only treats the EMPTY string as unset, so an unstripped whitespace value would
    win the new precedence and name a relative "   /whisper.cpp"."""
    studio = tmp_path / "elsewhere" / "studio"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -eu",
                    _whisper_root_block(),
                    'printf "%s\\n" "$INSTALL_DIR"',
                )
            ),
        ],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "UNSLOTH_HOME": "   ",
            "UNSLOTH_STUDIO_HOME": str(studio),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == studio / "whisper.cpp"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_tilde_master_root_expands_for_the_whisper_builder(tmp_path):
    home = tmp_path / "home"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                (
                    "set -eu",
                    _whisper_root_block(),
                    'printf "%s\\n" "$INSTALL_DIR"',
                )
            ),
        ],
        env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "UNSLOTH_HOME": "~/portable"},
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == home / "portable" / "whisper.cpp"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_relative_master_root_resolves_against_the_caller(tmp_path):
    """Node is chosen before setup.sh's first `cd "$SCRIPT_DIR"` and llama.cpp after it, so a
    value left relative names two different directories and the backend's neither."""
    caller = tmp_path / "caller"
    caller.mkdir()
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s\\n" "$_MASTER_ROOT"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd = str(caller),
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": "/usr/bin:/bin",
            "PWD": str(caller),
            "UNSLOTH_HOME": "not-created-yet",
            "_STUDIO_HOME_IS_CUSTOM": "false",
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == caller / "not-created-yet"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_master_root_alone_still_asserts_ownership_of_the_runtimes(tmp_path):
    """With only UNSLOTH_HOME set, STUDIO_HOME stays legacy and _STUDIO_HOME_IS_CUSTOM stays
    false, which is what licenses the installers to os.replace() and rm -rf without checking
    the Unsloth-owned marker. The runtimes moved, so the guard has to follow them."""
    root = tmp_path / "portable"
    root.mkdir()
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s %s\\n" "$_STUDIO_HOME_IS_CUSTOM" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
            "UNSLOTH_HOME": str(root),
            "_STUDIO_HOME_IS_CUSTOM": "false",
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["false", "true"]


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_no_master_root_leaves_the_ownership_flag_alone(tmp_path):
    src = SETUP_SH.read_text(encoding = "utf-8")
    script = "\n".join(
        (
            "set -u",
            _slice(src, "# Stripped before anything else", "# Directory-local evidence"),
            'printf "%s %s\\n" "$_STUDIO_HOME_IS_CUSTOM" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )
    for flag in ("false", "true"):
        completed = subprocess.run(
            ["bash", "-c", script],
            env = {
                "HOME": str(tmp_path / "home"),
                "PATH": "/usr/bin:/bin",
                "_STUDIO_HOME_IS_CUSTOM": flag,
            },
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.split() == [flag, flag]


def test_every_runtime_ownership_guard_uses_the_runtime_flag():
    """The Studio home and its venvs keep _STUDIO_HOME_IS_CUSTOM; every guard that names a
    runtime child has to move, or a master-root install loses the marker check on it."""
    src = SETUP_SH.read_text(encoding = "utf-8")
    for line in src.splitlines():
        if "_STUDIO_HOME_IS_CUSTOM" not in line:
            continue
        assert not any(
            name in line for name in ("$NODE_DIR", "$LLAMA_CPP_DIR", "$WHISPER_CPP_DIR")
        ), line
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    assert "$RuntimeRootIsCustom = $StudioHomeIsCustom -or [bool](Get-MasterRootOverride)" in ps
    for line in ps.splitlines():
        if "$StudioHomeIsCustom" not in line:
            continue
        assert not any(
            name in line for name in ("$LlamaCppDir", "$WhisperCppDir", "$NodeDir")
        ), line


UNINSTALL_SH = REPO_ROOT / "scripts" / "uninstall.sh"
UNINSTALL_PS1 = REPO_ROOT / "scripts" / "uninstall.ps1"


def test_both_uninstallers_clear_the_master_root_children():
    """setup installs llama.cpp, node and whisper.cpp as children of the master root, so an
    uninstaller that only knows the legacy siblings and the Studio root strands them. Behaviour
    is covered by tests/sh/test_uninstall_master_root.sh; this holds the PowerShell twin, which
    the Linux runners cannot execute, and pins the marker gate on both."""
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    assert "_master_root() {" in sh
    assert "function _MasterRoot" in ps
    for src, marker in ((sh, ".unsloth-studio-owned"), (ps, ".unsloth-studio-owned")):
        block = _slice(src, "master root's own children", "llama.cpp build + cache")
        assert marker in block, block
        for child in ("llama.cpp", "node", "whisper.cpp"):
            assert child in block, (child, block)
    # A user-chosen root reaches the deny list on both sides before anything is removed.
    assert '_is_unsafe_root "$_mr_root"' in sh
    assert "_IsUnsafeRoot $masterRoot" in ps
    # And it only contributes its studio child when neither exact override is set, as in
    # storage_roots.studio_root().
    assert '_emit "$(_master_root)/studio"' in sh
    assert '$envRoot = (Join-Path $master "studio")' in ps


def test_the_stop_pass_covers_the_master_root_runtimes():
    """Windows locks a loaded executable, so a runtime still running under the master root has
    to be stopped before its tree is removed or the delete exhausts its retries. _MasterRoot is
    resolved before the stop pass, and only marker-owned children join it, so an unmarked
    neighbour's process is never killed."""
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    stop_line = next(l for l in ps.splitlines() if l.strip().startswith("$stopRoots = "))
    assert "$masterChildrenToStop" in stop_line, stop_line
    block = _slice(ps, "$masterRootToStop = _MasterRoot", "$stopRoots = ")
    assert ".unsloth-studio-owned" in block, block
    assert "_IsUnsafeRoot $masterRootToStop" in block, block
    assert ps.index("$masterRootToStop = _MasterRoot") < ps.index(
        "_StopProcessesLockingRoots -Roots"
    )


def test_a_shared_staging_directory_is_pruned_not_deleted():
    """The prebuilt installers share <root>/.staging and prune it only when empty, so anything
    left in a user-chosen root is not ours to delete recursively."""
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    assert 'rmdir "$_mr_root/.staging"' in sh
    assert '_remove_path "$_mr_root/.staging"' not in sh
    staging = _slice(ps, "$masterStaging = Join-Path $masterRoot", "# Shared llama.cpp build")
    assert "Get-ChildItem -LiteralPath $masterStaging" in staging, staging
