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

import pathlib
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


@pytest.fixture
def whisper_path(tmp_path_factory):
    """PATH for the extracted whisper block, with git and cmake satisfied by stubs.

    The slice above is the shipped file verbatim, so it carries the builder's `command -v`
    preflight along with the root selection these tests are about. On a runner without cmake the
    block exits 1 before choosing anything and four tests fail for a reason that has nothing to
    do with what they assert. Stubbed rather than excised: cutting the preflight out of the slice
    would mean the tests no longer run the shipped text, which is the whole point of lifting it.
    """
    stub_bin = tmp_path_factory.mktemp("stubbin")
    for tool in ("git", "cmake"):
        path = stub_bin / tool
        path.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
        path.chmod(0o755)
    return f"{stub_bin}:/usr/bin:/bin"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_the_whisper_builder_installs_under_the_master_root(tmp_path, whisper_path):
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
            "PATH": whisper_path,
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
def test_the_whisper_builder_still_honours_a_studio_home_alone(tmp_path, whisper_path):
    studio = tmp_path / "elsewhere" / "studio"
    src = BUILD_WHISPER.read_text(encoding = "utf-8")
    block = _whisper_root_block()
    script = "\n".join(("set -eu", block, 'printf "%s\\n" "$INSTALL_DIR"'))
    completed = subprocess.run(
        ["bash", "-c", script],
        env = {
            "HOME": str(tmp_path / "home"),
            "PATH": whisper_path,
            "UNSLOTH_STUDIO_HOME": str(studio),
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == studio / "whisper.cpp"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_a_blank_master_root_does_not_outrank_the_whisper_studio_home(tmp_path, whisper_path):
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
            "PATH": whisper_path,
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
def test_a_tilde_master_root_expands_for_the_whisper_builder(tmp_path, whisper_path):
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
        env = {"HOME": str(home), "PATH": whisper_path, "UNSLOTH_HOME": "~/portable"},
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


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_the_legacy_root_named_explicitly_is_not_custom(tmp_path):
    """UNSLOTH_HOME=$HOME/.unsloth names the root a default install already uses.

    Nothing moves, so demanding an owner marker there rejects an update that used to reuse a
    legacy source-built ~/.unsloth/llama.cpp, which predates the marker entirely. Customness has
    to come from where the runtimes LAND, not from whether the variable was set.
    """
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    src = SETUP_SH.read_text(encoding = "utf-8")
    block = _slice(src, "# Stripped before anything else", "# Directory-local evidence")
    script = "\n".join(
        (
            "set -u",
            "_STUDIO_HOME_IS_CUSTOM=false",
            block,
            'printf "%s\\n" "$_RUNTIME_ROOT_IS_CUSTOM"',
        )
    )

    def flag(master: str) -> str:
        done = subprocess.run(
            ["bash", "-c", script],
            env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "UNSLOTH_HOME": master},
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    assert flag(str(home / ".unsloth")) == "false"
    # Non-vacuity: a root that really is elsewhere still takes the strict path.
    assert flag(str(tmp_path / "portable")) == "true"
    assert flag("") == "false"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_only_a_directory_can_be_adopted_at_a_runtime_path(tmp_path):
    """A dangling symlink and a regular file are both things the user put there.

    -d alone read either as "nothing is here", and the caller then rm -rf'd the path or let
    install_*_prebuilt.py os.replace() over it. A dangling link is the ordinary case: its target
    volume is simply not mounted, and resolving it later installs onto somebody's other disk.
    """
    src = SETUP_SH.read_text(encoding = "utf-8")
    block = _slice(src, "_studio_path_shape() {", "\n_packaged_frontend_available")
    script = "\n".join(
        (
            "set -u",
            "_STUDIO_OWNED_MARKER=.unsloth-studio-owned",
            "_studio_owned_adoptable() { return 1; }",
            "_studio_dir_unsearchable() { return 1; }",
            "_path_access_denied() { echo DENIED; exit 9; }",
            "setup_fail() { echo REFUSED; exit 1; }",
            block,
            '_assert_studio_owned_or_absent "$1" llama.cpp true',
            "echo ALLOWED",
        )
    )

    def verdict(path: pathlib.Path) -> str:
        done = subprocess.run(
            ["bash", "-c", script, "_", str(path)],
            env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
            capture_output = True,
            text = True,
            timeout = 60,
        )
        return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr[-120:]

    assert verdict(tmp_path / "absent") == "ALLOWED"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "no-such-volume" / "llama.cpp")
    assert verdict(dangling) == "REFUSED"

    plain = tmp_path / "afile"
    plain.write_text("mine")
    assert verdict(plain) == "REFUSED"

    owned = tmp_path / "owned"
    owned.mkdir()
    (owned / ".unsloth-studio-owned").touch()
    assert verdict(owned) == "ALLOWED"


def test_the_windows_inductor_cache_agrees_with_the_resolver():
    """setup.ps1 persists TORCHINDUCTOR_CACHE_DIR to the USER environment.

    Every later Studio process inherits it, so _setup_cache_env's fill-if-unset default never
    applies on Windows and the containment this branch is for does not happen there. It has to
    name the directory the resolver would have chosen. Two things still outrank that, and both
    are recorded here so a later edit cannot quietly drop them: long paths off keeps the short
    drive-root directory for MAX_PATH headroom, and a path containing a space is refused for the
    same reason storage_roots does, since the C++ builders paste it in unquoted.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(ps, "$TorchCacheDir = $null", "$env:TORCHINDUCTOR_CACHE_DIR = $TorchCacheDir")
    assert 'Join-Path (Join-Path $StudioHome "cache") "torchinductor"' in block
    assert "$LongPathsEnabled" in block
    assert "'\\s'" in block or '"\\s"' in block, "the whitespace refusal is gone"
    assert '"C:\\tc"' in block

    roots = (REPO_ROOT / "studio" / "backend" / "utils" / "paths" / "storage_roots.py").read_text(
        encoding = "utf-8"
    )
    # The same key, named by both sides, so the two cannot drift apart silently.
    assert '"TORCHINDUCTOR_CACHE_DIR",' in roots
    assert 'str(root / "torchinductor")' in roots


def test_the_windows_node_guard_covers_a_master_root():
    """setup.ps1's Node ownership guard and its marker both hung off $NodeOverride alone.

    $NodeOverride is set only in the UNSLOTH_STUDIO_HOME / STUDIO_HOME branch. The master-root
    branch sets $NodeParent and leaves it null, so <master>\\node reached the whole-directory
    os.replace() in install_node_prebuilt.py with no ownership evidence at all, and the tree the
    run then created stayed unmarked, which makes the uninstaller decline to remove it later.
    setup.sh had already moved these two sites to _RUNTIME_ROOT_IS_CUSTOM.

    The sibling test above only rejects $StudioHomeIsCustom beside $NodeDir, which this bug
    never wrote: it named a third variable. So the rule here is positive, not a denial.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    guards = [
        line
        for line in ps.splitlines()
        if "$NodeDir" in line
        and ".unsloth-studio-owned" not in line
        and ("$NodeOverride" in line or "$RuntimeRootIsCustom" in line)
    ]
    # The guard's `if`, and the marker's `if`. Fewer means the block was restructured and this
    # test would otherwise pass by finding nothing.
    assert len(guards) >= 2, guards
    for line in guards:
        assert "$RuntimeRootIsCustom" in line, line


UNINSTALL_SH = REPO_ROOT / "scripts" / "uninstall.sh"
UNINSTALL_PS1 = REPO_ROOT / "scripts" / "uninstall.ps1"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh not available")
def test_the_windows_uninstaller_resolves_a_relative_root_like_setup(tmp_path):
    """setup.ps1 resolves through PowerShell's own location; uninstall.ps1 used
    [IO.Path]::GetFullPath, which anchors at [Environment]::CurrentDirectory.

    PowerShell does not keep those two in step, so after a Set-Location a relative UNSLOTH_HOME
    named one install to setup and a different one to the uninstaller. The owner marker does not
    save you there: it spares trees that are not Unsloth's, and the wrongly resolved path is
    another Unsloth install, marker and all.

    Runs the shipped function rather than matching its text, so a rewrite that keeps the words
    and loses the behaviour still fails.
    """
    initial = tmp_path / "initial"
    chosen = tmp_path / "chosen"
    (chosen / "portable").mkdir(parents = True)
    initial.mkdir()
    script = tmp_path / "probe.ps1"
    script.write_text(
        f"""$txt = Get-Content -Raw "{UNINSTALL_PS1}"
foreach ($n in @("_ExpandTilde", "_MasterRoot")) {{
    $m = [regex]::Match($txt, "(?ms)^    function $n \\{{.*?^    \\}}")
    if (-not $m.Success) {{ Write-Output "EXTRACT-FAILED:$n"; exit 1 }}
    Invoke-Expression $m.Value
}}
[System.Environment]::CurrentDirectory = "{initial}"
Set-Location "{chosen}"
$env:UNSLOTH_HOME = "portable"
$env:USERPROFILE = "{tmp_path}/profile"
Write-Output (_MasterRoot)
""",
        encoding = "utf-8",
    )
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script)],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    assert "EXTRACT-FAILED" not in out, out
    assert out == str(chosen / "portable"), out


def test_neither_uninstaller_recurses_into_an_install_lock_path():
    """A lock is always a regular file, so a directory at one of those fixed names is the user's.

    prebuilt_core.install_lock creates it with os.open(O_CREAT | O_EXCL). Both uninstallers
    reached the lock names through their recursive remover, which in a user-chosen master root
    deletes a whole tree with none of the owner-marker proof the runtime children require.
    Behaviour is covered by tests/sh/test_uninstall_master_root.sh for the POSIX half; this
    holds the PowerShell twin, which has no runner here.
    """
    sh = UNINSTALL_SH.read_text(encoding = "utf-8")
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")

    assert "_remove_lock_file() {" in sh
    assert "function _RemoveLockFile" in ps

    # The call sites do not all spell the lock out: the PowerShell ones iterate $lockName over a
    # list built on the line above, and both sweep a $stale from a glob. Matching only on
    # "install.lock" let the master-root loop keep the recursive remover and still pass.
    lockish = ("install.lock", "lockName", "$stale", "_stale", "_mr_lock")
    for text, remover, shape in (
        (sh, "_remove_path ", "_remove_lock_file"),
        (ps, "_RemovePath ", "_RemoveLockFile"),
    ):
        hits = 0
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if remover not in line and shape not in line:
                continue
            if not any(token in line for token in lockish):
                continue
            hits += 1
            assert shape in line, line
        # Counts the call sites found, not the ones left wrong: once they are all correct the
        # remover no longer appears beside a lock at all, and a count of zero would mean the
        # tokens above had stopped matching and this loop proved nothing.
        assert hits >= 4, (remover, hits)

    # The rename install_node_prebuilt makes keeps the leading dot, so a glob without one also
    # matches names the user owns in their own root.
    assert '".*.install.lock.stale.*"' in ps
    assert '"*.install.lock.stale.*"' not in ps


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


def test_the_windows_uninstaller_finds_a_master_root_from_the_note(tmp_path):
    """`$env:UNSLOTH_HOME = 'D:\\portable'; unsloth studio update` names the root for one command.

    setup.ps1 puts node\\, llama.cpp\\ and whisper.cpp\\ under it and marks them, and a later
    uninstall run from an ordinary shell has no UNSLOTH_HOME at all. Reading only the current
    environment, _MasterRoot answered null there, so the run removed <master>\\studio and left
    multi-gigabyte runtimes beside it. setup.sh already wrote the note; setup.ps1 did not, and
    the PowerShell uninstaller did not read it.

    Runs the shipped function, so a rewrite that keeps the words and loses the behaviour fails.
    """
    master = tmp_path / "portable"
    (master / "studio" / "share").mkdir(parents = True)
    (master / "studio" / "share" / ".unsloth-master-root").write_text(
        f"{master}\n", encoding = "utf-8"
    )
    profile = tmp_path / "profile"
    (profile / ".unsloth" / "studio" / "share").mkdir(parents = True)

    script = tmp_path / "probe.ps1"
    script.write_text(
        f"""$txt = Get-Content -Raw "{UNINSTALL_PS1}"
foreach ($n in @("_ExpandTilde", "_MasterRoot")) {{
    $m = [regex]::Match($txt, "(?ms)^    function $n \\{{.*?^    \\}}")
    if (-not $m.Success) {{ Write-Output "EXTRACT-FAILED:$n"; exit 1 }}
    Invoke-Expression $m.Value
}}
$env:UNSLOTH_HOME = ""
$env:STUDIO_HOME = ""
$env:UNSLOTH_STUDIO_HOME = "{master}/studio"
$env:USERPROFILE = "{profile}"
Write-Output (_MasterRoot)
""",
        encoding = "utf-8",
    )
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(script)],
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    assert "EXTRACT-FAILED" not in out, out
    assert out == str(master), out


def test_the_windows_setup_records_the_master_root_for_the_uninstaller():
    """The note the test above reads has to be written, and only where it is true.

    A staging run installs to a throwaway root, and every non-master branch derives the root
    from paths the uninstaller already knows, so a note there would only ever be able to go
    stale. This is the same rule setup.sh applies.
    """
    ps = SETUP_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps, "# Record the master root inside the Studio tree", "$WithLlamaCppDir = $null"
    )
    assert "(Get-MasterRootOverride)" in block
    assert "-not $StageRoot" in block
    assert '".unsloth-master-root"' in block
    # Staged then renamed: a reader catching a half-written note would name a truncated path,
    # and this note licenses deletions.
    assert "$noteTmp" in block and "Move-Item" in block
    # The 3-argument overwrite overload is .NET Core only, and setup.ps1 runs under 5.1.
    assert "[System.IO.File]::Move(" not in block


def test_the_windows_uninstaller_clears_the_inductor_path_it_persisted():
    """setup.ps1 writes TORCHINDUCTOR_CACHE_DIR to the USER environment, so it outlives the
    install. Every later PyTorch process on the account inherits it, including ones unrelated to
    Unsloth, and they compile into the deleted tree and rebuild part of it.

    Only a value inside a root this run owned: a directory the user chose is theirs, and the
    shared C:\\tc fallback is not install specific and is not deleted here either.
    """
    ps = UNINSTALL_PS1.read_text(encoding = "utf-8")
    block = _slice(
        ps, "# Clear the persisted Inductor cache path", "# Remove HKCU\\Software\\Unsloth"
    )
    assert "GetEnvironmentVariable('TORCHINDUCTOR_CACHE_DIR', 'User')" in block
    assert "[NullString]::Value, 'User'" in block
    # Scoped to what this run owns. $knownRoots includes roots the gates refused to delete.
    assert "$ownedRoots" in block and "$knownRoots" not in block
    # Comments stripped first: this block explains why C:\tc is spared, and a comment saying so
    # is not the same thing as code naming it.
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    assert "C:\\tc" not in code
