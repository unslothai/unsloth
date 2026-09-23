# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Behavioural guards for the two in-container update helpers of the Docker image.

* `unsloth-studio-update` only WARNED when the new backend failed to import, so a
  release missing a `--no-deps` dependency replaced the healthy process with one that
  cannot start; supervisord then lands in FATAL and never leaves it.
* `unsloth-llama-update --check` reported "up to date" when it could not reach the
  release feed, and its in-place rollback left new-release-only shared objects beside
  the restored files, which ggml dlopen()s.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_UPDATE = REPO_ROOT / "docker" / "unsloth_studio_update.sh"
LLAMA_UPDATE = REPO_ROOT / "docker" / "unsloth_llama_update.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="needs bash",
)


def _stub(directory: Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _run(
    script: Path,
    args,
    env,
    cwd=None,
):
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=120,
    )


def _studio_env(
    tmp_path: Path,
    *,
    import_ok: bool = True,
    dist_ok: bool = True,
    status_exit: int = 0,
    git_ls_exit: int = 0,
    npm_build_ok: bool = True,
    src_link: bool = False,
    restart_exit: int = 0,
) -> dict:
    """A Studio home whose venv python is the real interpreter over a fake `studio`
    package, so the script's own import and frontend checks run for real. pip,
    supervisorctl, git and the bundled npm are recording stubs."""
    home = tmp_path / "studio"
    site = tmp_path / "site"
    (site / "studio" / "backend").mkdir(parents=True)
    (site / "studio" / "__init__.py").write_text("")
    (site / "studio" / "backend" / "__init__.py").write_text("")
    (site / "studio" / "backend" / "main.py").write_text(
        "" if import_ok else "raise ImportError('No module named structlog')\n"
    )
    # dist-info for both packages, so importlib.metadata answers from the fake site
    # and never from whatever the runner's own interpreter has installed
    for name, ver in (("unsloth", "2026.9.4"), ("unsloth_zoo", "2026.9.3")):
        info = site / f"{name}-{ver}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
    if dist_ok:
        (site / "studio" / "frontend" / "dist").mkdir(parents=True)
        (site / "studio" / "frontend" / "dist" / "index.html").write_text("<html></html>")
    # the Studio image presents src as a link into its own copy of Studio, so the home
    # can be a volume; a standalone image has it as a real directory
    src = (tmp_path / "app" / "src") if src_link else (home / "src")
    (src / "studio").mkdir(parents=True)
    (src / "OLD_TREE").write_text("previous source tree\n")
    if src_link:
        home.mkdir(parents=True, exist_ok=True)
        (home / "src").symlink_to(src)

    venv_bin = home / "unsloth_studio" / "bin"
    _stub(
        venv_bin,
        "python",
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  echo "STUB-PIP $*" >> "$STUB_LOG"\n'
        # a --with-deps run snapshots the dependency set first
        # after an install, a dependency the update pulled in shows up in the freeze
        # STUB_FREEZE_EXIT fails every freeze; STUB_FREEZE_EXIT_AFTER only the ones after an install
        '  if [ "$3" = "freeze" ]; then [ -n "${STUB_FREEZE_EXIT:-}" ] && exit "$STUB_FREEZE_EXIT"; [ -n "${STUB_FREEZE_EXIT_AFTER:-}" ] && [ -e "$STUB_LOG.installed" ] && exit "$STUB_FREEZE_EXIT_AFTER"; echo "transformers==4.0.0"; echo "torch==2.11.0+cu128"; [ -e "$STUB_LOG.installed" ] && echo "newdep==1.0"; exit 0; fi\n'
        '  case " $* " in *" -r "*) ;; *" install "*) : > "$STUB_LOG.installed" ;; esac\n'
        # the constraints file is deleted on exit, so record what it pinned
        '  _c=0; for _a in "$@"; do [ "$_c" = 1 ] && { echo "STUB-PIP-CONSTRAINTS $(tr "\\n" " " < "$_a")" >> "$STUB_LOG"; _c=0; }; [ "$_a" = "-c" ] && _c=1; done\n'
        # so is the requirements file a restore reinstalls from
        '  _r=0; for _a in "$@"; do [ "$_r" = 1 ] && { echo "STUB-PIP-REQ $(tr "\\n" " " < "$_a")" >> "$STUB_LOG"; _r=0; }; [ "$_a" = "-r" ] && _r=1; done\n'
        # an interrupted install: the updater is waiting on this child, so the signal
        # lands on it and its trap runs once we exit
        '  case " $* " in *" -e "*) [ -n "${STUB_PIP_INTERRUPT:-}" ] && kill -INT "$PPID" ;; esac\n'
        # the release path has no swap; an interrupt during its pip must restore too
        '  case " $* " in *" -U "*) [ -n "${STUB_PIP_INTERRUPT_RELEASE:-}" ] && kill -INT "$PPID" ;; esac\n'
        # a signal that lands while the restore itself runs must not cut the cleanup short
        '  case " $* " in *" -r "*) [ -n "${STUB_PIP_INTERRUPT_RESTORE:-}" ] && kill -TERM "$PPID" ;; esac\n'
        # reinstalling the recorded previous install brings the working tree back
        '  case " $* " in *" -r "*)\n'
        f'    : > "{site}/studio/backend/main.py"\n'
        f'    mkdir -p "{site}/studio/frontend/dist"; echo ok > "{site}/studio/frontend/dist/index.html" ;;\n'
        "  esac\n"
        # STUB_PIP_INSTALL_EXIT fails the update's own install and lets the restore's -r
        # installs through; STUB_PIP_EXIT fails every pip call, the restore included
        '  case " $* " in *" -r "*) ;; *) [ -n "${STUB_PIP_INSTALL_EXIT:-}" ] && exit "$STUB_PIP_INSTALL_EXIT" ;; esac\n'
        '  exit "${STUB_PIP_EXIT:-0}"\n'
        "fi\n"
        f'PYTHONPATH="{site}" exec "{shutil.which("python3")}" "$@"\n',
    )
    _stub(
        home / "node" / "bin",
        "npm",
        'echo "STUB-NPM $* in $PWD" >> "$STUB_LOG"\n'
        'if [ "$1" = "ci" ]; then exit "${STUB_NPM_CI_EXIT:-0}"; fi\n'
        'if [ "$*" = "run build" ]; then\n'
        + (
            "  mkdir -p dist && echo '<html></html>' > dist/index.html; exit 0\n"
            if npm_build_ok
            else "  exit 1\n"
        )
        + "fi\nexit 0\n",
    )
    bin_dir = tmp_path / "bin"
    _stub(
        bin_dir,
        "supervisorctl",
        'echo "STUB-SUPERVISORCTL $*" >> "$STUB_LOG"\n'
        # a program that was started reports RUNNING from then on
        f'if [ "$1" = "status" ]; then [ -e "$STUB_LOG.started" ] && exit 0; exit {status_exit}; fi\n'
        f'if [ "$1" = "restart" ] || [ "$1" = "start" ]; then [ {restart_exit} = 0 ] && : > "$STUB_LOG.started"; exit {restart_exit}; fi\nexit 0\n',
    )
    _stub(
        bin_dir,
        "git",
        'dir=""; if [ "$1" = "-C" ]; then dir="$2"; shift 2; fi\n'
        'case "$1" in\n'
        f"  ls-remote) exit {git_ls_exit} ;;\n"
        '  checkout) mkdir -p "$dir/studio/frontend/src" "$dir/.git"'
        ' "$dir/studio/backend/core/data_recipe/oxc-validator";'
        ' echo "{}" > "$dir/studio/frontend/package.json";'
        ' [ -n "${STUB_LOCKFILE:-}" ] && echo "{}" > "$dir/studio/frontend/package-lock.json";'
        ' echo "{}" > "$dir/studio/backend/core/data_recipe/oxc-validator/package.json";'
        ' [ -n "${STUB_OXC_LOCKFILE:-}" ] && echo "{}" > "$dir/studio/backend/core/data_recipe/oxc-validator/package-lock.json";'
        ' echo new > "$dir/NEW_TREE" ;;\n'
        "esac\nexit 0\n",
    )
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "0"
    env["STUB_LOG"] = str(tmp_path / "calls.log")
    return env


def _scratch(d: Path, pattern: str = ".src-*") -> list:
    """The updater's scratch entries, minus the lock file it keeps on purpose."""
    return [p for p in d.glob(pattern) if p.name != ".src-update.lock"]


def _calls(env) -> str:
    log = Path(env["STUB_LOG"])
    return log.read_text() if log.exists() else ""


def test_studio_update_restarts_when_the_backend_imports(tmp_path: Path):
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "STUB-SUPERVISORCTL restart studio" in _calls(env), _calls(env)


def test_studio_update_does_not_restart_into_a_backend_that_cannot_import(tmp_path: Path):
    env = _studio_env(tmp_path, import_ok=False)
    res = _run(STUDIO_UPDATE, [], env)
    calls = _calls(env)
    assert "STUB-SUPERVISORCTL restart studio" not in calls, (
        "restarting into code that cannot import kills a process that is serving "
        "fine and parks supervisord's studio program in FATAL:\n" + calls
    )
    assert res.returncode != 0, "a broken update must not report success"
    assert "--with-deps" in res.stdout, "the remedy must still be printed"
    # --force-reinstall: pip takes a same-version editable as already satisfying the
    # pin and would leave the new tree's metadata in place
    assert "install --no-deps --force-reinstall -r" in calls, (
        "the previous install must be put back:\n" + calls
    )


def test_studio_update_does_not_restart_into_a_tree_without_a_built_frontend(tmp_path: Path):
    """The --ref failure: a git build has no studio/frontend/dist, `unsloth studio`
    exits 1 on start, and three quick exits leave supervisord's program FATAL."""
    env = _studio_env(tmp_path, dist_ok=False)
    res = _run(STUDIO_UPDATE, [], env)
    calls = _calls(env)
    assert res.returncode != 0
    assert "no built frontend" in res.stdout
    assert "STUB-SUPERVISORCTL restart studio" not in calls, calls
    assert "install --no-deps --force-reinstall -r" in calls, calls


@pytest.mark.parametrize("status_exit, verb", [(0, "restart"), (3, "start")])
def test_studio_update_restarts_a_studio_that_is_not_running(tmp_path: Path, status_exit, verb):
    """`supervisorctl status` exits 3 for STOPPED/EXITED/FATAL. That is still a program
    supervisord manages, and FATAL is what a failed earlier update left behind. It gets
    `start`: `restart` stops it first, which supervisorctl reports as an error."""
    env = _studio_env(tmp_path, status_exit=status_exit)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert f"STUB-SUPERVISORCTL {verb} studio" in _calls(env), _calls(env)
    assert "not managing" not in res.stdout


def test_studio_update_fails_when_supervisor_cannot_restart_studio(tmp_path: Path):
    """A restart that fails used to leave the new install in place with Studio down and
    the previous tree already deleted. The previous tree is kept until the service is
    up, so a failed restart puts it back and starts that."""
    env = _studio_env(tmp_path, restart_exit=1)
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    calls = _calls(env)
    assert res.returncode != 0, res.stdout
    assert "ERROR: supervisorctl restart studio failed" in res.stdout, res.stdout
    assert "previous install is back in place" in res.stderr, res.stderr
    assert (home / "src" / "OLD_TREE").exists(), "the previous source tree was not put back"
    assert not (home / "src" / "NEW_TREE").exists()
    assert not _scratch(home)
    assert "install --no-deps --force-reinstall -r" in calls, calls
    sup = [l for l in calls.splitlines() if l.startswith("STUB-SUPERVISORCTL")]
    assert sup[-1] == "STUB-SUPERVISORCTL status studio", calls
    assert "STUB-SUPERVISORCTL restart studio" in sup[1:], (
        "the previous install must be started again:\n" + calls
    )
    # the release path has no tree to swap, but its pins go back the same way
    env = _studio_env(tmp_path / "release", restart_exit=1)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode != 0, res.stdout
    assert "install --no-deps --force-reinstall -r" in _calls(env), _calls(env)


def test_studio_update_refuses_to_run_beside_another_updater(tmp_path: Path):
    """Two updaters at once would take each other's staging and previous trees for
    leftovers and swap over each other's src."""
    fcntl = pytest.importorskip("fcntl")
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    lock = home / ".src-update.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
        assert res.returncode == 1, res.stderr + res.stdout
        assert "another unsloth-studio-update is running" in res.stderr, res.stderr
        assert "STUB-PIP" not in _calls(env) and "STUB-NPM" not in _calls(env), _calls(env)
        assert (home / "src" / "OLD_TREE").exists()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert (home / "src" / "NEW_TREE").exists()
    # the lock file shares the scratch prefix so the home linker never links it, and
    # the startup sweep must not take it for a leftover
    assert "left behind" not in res.stdout, res.stdout
    assert not _scratch(home), "the lock file must not outlive the run"


def test_studio_update_reports_an_unmanaged_studio(tmp_path: Path):
    env = _studio_env(tmp_path, status_exit=4)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "STUB-SUPERVISORCTL restart studio" not in _calls(env)
    assert "not managing 'studio'" in res.stdout


def test_studio_update_ref_builds_the_frontend_and_swaps_the_source_tree(tmp_path: Path):
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    calls = _calls(env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "STUB-NPM run build" in calls, calls
    assert (home / "src" / "NEW_TREE").exists(), "src was not replaced by the fetched ref"
    assert (home / "src" / "studio" / "frontend" / "dist" / "index.html").exists()
    assert not (home / "src" / ".git").exists()
    assert not _scratch(home, ".src-prev.*"), "the previous tree was not cleaned up"
    assert not _scratch(home, ".src-update.*"), "the staging tree was not cleaned up"
    assert f"install --no-deps -e {home / 'src'}" in calls, calls


def test_studio_update_ref_writes_through_a_linked_source_tree(tmp_path: Path):
    """Replacing that link with a directory would put the tree outside the image's copy,
    where the next container start relinks over it and the update is silently gone."""
    env = _studio_env(tmp_path, src_link=True)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    real_src = tmp_path / "app" / "src"
    assert res.returncode == 0, res.stderr + res.stdout
    assert (home / "src").is_symlink(), "the link was replaced by a directory"
    assert (real_src / "NEW_TREE").exists(), "the ref was not installed where the code lives"
    assert not _scratch(home), "staging trees must not land in the data volume"
    assert not _scratch(real_src.parent, ".src-prev.*")
    # pip records the home path, so the install keeps resolving through the link and
    # `unsloth-studio-home --restore` (back to a pre-split image) still finds it
    assert f"install --no-deps -e {home / 'src'}" in _calls(env), _calls(env)


def test_studio_update_ref_with_a_failed_frontend_build_changes_nothing(tmp_path: Path):
    env = _studio_env(tmp_path, npm_build_ok=False)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    calls = _calls(env)
    assert res.returncode != 0
    assert (home / "src" / "OLD_TREE").exists(), "the running source tree was touched"
    assert "STUB-PIP" not in calls, calls
    assert not _scratch(home, ".src-update.*")
    # errexit is off inside `( ... ) || return`, so the build step has to stop by itself
    assert "npm run build failed" in res.stdout, res.stdout
    assert "oxc-validator" not in calls, (
        "a failed build must not go on to the oxc install:\n" + calls
    )


def test_studio_update_ref_installs_the_oxc_runtime_after_a_good_build(tmp_path: Path):
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    oxc = [
        l
        for l in _calls(env).splitlines()
        if "STUB-NPM install" in l and l.endswith("oxc-validator")
    ]
    assert oxc, _calls(env)
    # the same lockfile rule as the frontend: a ref that ships one is installed from it
    env = _studio_env(tmp_path / "locked")
    env["STUB_OXC_LOCKFILE"] = "1"
    env["UNSLOTH_NPM_REGISTRY"] = "https://mirror.example/npm/"
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    oxc = [l for l in _calls(env).splitlines() if "STUB-NPM" in l and l.endswith("oxc-validator")]
    assert len(oxc) == 1 and oxc[0].startswith("STUB-NPM ci "), _calls(env)
    assert "--registry https://mirror.example/npm/" in oxc[0], oxc[0]
    env = _studio_env(tmp_path / "drift")
    env["STUB_OXC_LOCKFILE"] = "1"
    env["STUB_NPM_CI_EXIT"] = "1"
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert res.returncode != 0
    assert "npm ci failed for the oxc validator" in res.stdout, res.stdout
    assert not [
        l
        for l in _calls(env).splitlines()
        if "STUB-NPM install" in l and l.endswith("oxc-validator")
    ], "lockfile drift must not be re-resolved:\n" + _calls(env)
    assert (Path(env["UNSLOTH_STUDIO_HOME"]) / "src" / "OLD_TREE").exists()


def test_studio_update_rollback_keeps_the_editable_uri_pip_recorded(tmp_path: Path):
    """direct_url.json holds a file:// URI; a path with a space comes back percent
    encoded, and pip rejects that as a bare path but takes it as the URI."""
    env = _studio_env(tmp_path, import_ok=False)
    site = tmp_path / "site"
    info = site / "unsloth-2026.9.4.dist-info"  # seeded by _studio_env
    (info / "direct_url.json").write_text(
        '{"url": "file:///opt/my%20studio/src", "dir_info": {"editable": true}}'
    )
    res = _run(STUDIO_UPDATE, ["--no-restart"], env)
    assert res.returncode != 0
    assert "STUB-PIP-REQ -e file:///opt/my%20studio/src" in _calls(env), _calls(env)


def test_studio_update_ref_restores_the_tree_when_interrupted_after_the_swap(tmp_path: Path):
    """Ctrl-C or `docker stop` between the swap and the checks left the half-installed
    tree in place and the previous one beside it as .src-prev.*, which the next
    container start links into the Studio home."""
    env = _studio_env(tmp_path)
    env["STUB_PIP_INTERRUPT"] = "1"
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode == 130, res.stderr + res.stdout
    assert (home / "src" / "OLD_TREE").exists(), "the previous source tree was not put back"
    assert not (home / "src" / "NEW_TREE").exists()
    assert not _scratch(home, ".src-prev.*"), "the previous tree was left beside src"
    assert not _scratch(home, ".src-update.*")
    assert "interrupted" in res.stdout, res.stdout


def test_studio_update_release_restores_the_pins_when_interrupted_during_pip(tmp_path: Path):
    """The release path swaps nothing, but a pip interrupted half-way can have replaced
    the packages already; the recorded previous install goes back the same way."""
    env = _studio_env(tmp_path)
    env["STUB_PIP_INTERRUPT_RELEASE"] = "1"
    res = _run(STUDIO_UPDATE, ["--no-restart"], env)
    calls = _calls(env)
    assert res.returncode == 130, res.stderr + res.stdout
    assert "install --no-deps --force-reinstall -r" in calls, (
        "the previous install was not put back:\n" + calls
    )
    assert "interrupted" in res.stdout, res.stdout


def test_studio_update_restores_once_and_leaves_no_temp_files_when_signalled_mid_restore(
    tmp_path: Path,
):
    """After a failed tree check the script restores explicitly, then exits; the exit
    trap must not restore a second time, and a signal during the restore must not skip
    the temp-file cleanup."""
    env = _studio_env(tmp_path, dist_ok=False)
    env["STUB_PIP_INTERRUPT_RESTORE"] = "1"
    tmpd = tmp_path / "tmpd"
    tmpd.mkdir()
    env["TMPDIR"] = str(tmpd)
    res = _run(STUDIO_UPDATE, ["--no-restart"], env)
    calls = _calls(env)
    assert res.returncode != 0
    assert calls.count("--force-reinstall -r") == 1, "restore ran more than once:\n" + calls
    assert not list(tmpd.iterdir()), "temp files left behind: " + str(list(tmpd.iterdir()))


def test_studio_update_ref_refuses_a_venv_without_a_source_tree(tmp_path: Path):
    """A wheel-only install has no src; `mv` of a missing tree must not be the error."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    shutil.rmtree(home / "src")
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert res.returncode != 0
    assert "no source tree" in res.stderr, res.stderr
    assert "STUB-NPM" not in _calls(
        env
    ), "nothing should be built for a tree that cannot be swapped"


def test_studio_update_with_deps_puts_the_dependency_set_back(tmp_path: Path):
    """--with-deps lets pip move every dependency; restoring unsloth alone would leave
    the new dependency set under the old code."""
    env = _studio_env(tmp_path, import_ok=False)
    res = _run(STUDIO_UPDATE, ["--with-deps"], env)
    calls = _calls(env)
    assert res.returncode != 0
    assert "STUB-PIP -m pip freeze --exclude-editable" in calls, calls
    # the dependency snapshot goes back as pinned, the package identity by force
    assert calls.count("install --no-deps -r") == 1, "dependency snapshot:\n" + calls
    assert calls.count("install --no-deps --force-reinstall -r") == 1, "previous install:\n" + calls
    # the backend's requirement set is the `studio` extra, and torch/CUDA must not be
    # re-resolved (the venv's nvidia libs are linked into the base venv)
    assert "unsloth[studio]" in calls, calls
    assert "STUB-PIP-CONSTRAINTS torch==2.11.0+cu128" in calls, calls
    assert "transformers==4.0.0" not in [
        l for l in calls.splitlines() if l.startswith("STUB-PIP-CONSTRAINTS")
    ], "only the torch/CUDA stack is a constraint"
    env = _studio_env(tmp_path / "nodeps", import_ok=False)
    res = _run(STUDIO_UPDATE, [], env)
    assert "freeze" not in _calls(env), "a --no-deps update has nothing to snapshot"
    assert " -c " not in _calls(env)


def test_studio_update_ref_with_deps_installs_the_studio_extra(tmp_path: Path):
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--with-deps", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert f"-e {home / 'src'}[studio]" in _calls(env), _calls(env)
    assert "STUB-PIP-CONSTRAINTS torch==2.11.0+cu128" in _calls(env), _calls(env)


def test_studio_update_ref_puts_the_old_tree_back_when_the_new_one_cannot_start(tmp_path: Path):
    env = _studio_env(tmp_path, import_ok=False)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode != 0
    assert (home / "src" / "OLD_TREE").exists(), "the previous source tree was not restored"
    assert not (home / "src" / "NEW_TREE").exists()


def _zoo_spec(calls: str) -> str:
    for line in calls.splitlines():
        for token in line.split():
            if token.startswith("git+https://github.com/unslothai/unsloth-zoo.git@"):
                return token
    return ""


def test_studio_update_mirrors_the_ref_when_the_zoo_has_it(tmp_path: Path):
    env = _studio_env(tmp_path, git_ls_exit=0)
    res = _run(STUDIO_UPDATE, ["--ref", "v2026.7.5", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert _zoo_spec(_calls(env)).endswith("@v2026.7.5#egg=unsloth_zoo"), _calls(env)


def test_studio_update_falls_back_to_zoo_main_when_the_ref_is_absent(tmp_path: Path):
    env = _studio_env(tmp_path, git_ls_exit=2)
    res = _run(STUDIO_UPDATE, ["--ref", "v2026.7.5", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert _zoo_spec(_calls(env)).endswith("@main#egg=unsloth_zoo"), _calls(env)
    assert "has no ref" in res.stdout, res.stdout


def test_studio_update_aborts_when_the_zoo_lookup_never_reached_the_remote(tmp_path: Path):
    # treating 2 and 128 alike pairs the requested unsloth revision with an unrelated
    # zoo one once the network recovers, across a private API
    env = _studio_env(tmp_path, git_ls_exit=128)
    res = _run(STUDIO_UPDATE, ["--ref", "v2026.7.5", "--no-restart"], env)
    calls = _calls(env)
    assert "STUB-PIP" not in calls, "a transport failure must not install anything:\n" + calls
    assert res.returncode != 0, "an unresolvable zoo ref must not report success"
    assert "has no ref" not in res.stdout, (
        "an unreachable remote must not be reported as a missing ref:\n" + res.stdout
    )
    assert "--zoo-ref" in res.stderr, "the remedy must be printed"


def test_studio_update_puts_the_install_back_when_pip_itself_fails(tmp_path: Path):
    env = _studio_env(tmp_path)
    env["STUB_PIP_INSTALL_EXIT"] = "1"
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode != 0
    assert "previous install is back in place" in res.stderr, res.stderr
    assert (home / "src" / "OLD_TREE").exists(), "the previous source tree was not put back"
    assert not _scratch(home)
    assert "STUB-SUPERVISORCTL" not in _calls(env)


def test_a_signal_during_the_health_wait_puts_the_service_back_too(tmp_path: Path):
    """Ctrl-C while waiting for /api/health: the service is already running the new
    code, so restoring the files alone would leave it serving a mix of versions.
    The cleanup restarts it on the restored install, as back_out does."""
    env = _studio_env(tmp_path)
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "30"
    _stub(
        tmp_path / "bin",
        "curl",
        'echo "STUB-CURL $*" >> "$STUB_LOG"\nkill -TERM "$PPID"\nexit 22\n',
    )
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    calls = _calls(env).splitlines()
    assert res.returncode == 143, res.stdout + res.stderr
    assert "interrupted after the install started" in res.stdout, res.stdout
    assert (home / "src" / "OLD_TREE").exists() and not (home / "src" / "NEW_TREE").exists()
    reinst = [i for i, l in enumerate(calls) if "install --no-deps --force-reinstall -r" in l]
    assert reinst, calls
    after = calls[reinst[-1] + 1 :]
    assert "STUB-SUPERVISORCTL restart studio" in after, calls
    assert "the previous install is running again" in res.stdout, res.stdout
    assert not _scratch(home)


def test_a_health_wait_that_expires_before_its_first_check_still_asks_once(tmp_path: Path):
    """The deadline is whole seconds. With a short wait the clock can step past it before
    the loop first reads it, and a loop that checked the deadline first then rolled back a
    Studio it had never probed. Here every read of the clock is 5s after the last, so the
    deadline has always passed by the first check."""
    env = _studio_env(tmp_path)
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "1"
    clock = tmp_path / "clock"
    clock.write_text("1000000000", encoding="utf-8")
    _stub(
        tmp_path / "bin",
        "date",
        'if [ "$*" = "+%s" ]; then\n'
        f'  t=$(( $(cat "{clock}") + 5 )); echo "$t" > "{clock}"; echo "$t"; exit 0\n'
        "fi\n"
        'exec /bin/date "$@"\n',
    )
    _stub(tmp_path / "bin", "curl", 'echo "STUB-CURL $*" >> "$STUB_LOG"\nexit 0\n')
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    calls = _calls(env)
    assert "STUB-CURL" in calls, "the wait expired without probing Studio once:\n" + calls
    assert res.returncode == 0, res.stdout + res.stderr
    assert "answering on port 8000" in res.stdout, res.stdout
    assert (home / "src" / "NEW_TREE").exists(), "a healthy update was rolled back"


def test_studio_update_fails_when_studio_does_not_answer_after_the_restart(tmp_path: Path):
    """A backend that imports can still die at startup; the previous tree is kept
    until /api/health answers, and goes back when it does not."""
    env = _studio_env(tmp_path)
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "1"
    _stub(tmp_path / "bin", "curl", 'echo "STUB-CURL $*" >> "$STUB_LOG"\nexit 22\n')
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    calls = _calls(env)
    assert res.returncode != 0, res.stdout
    assert "did not answer on port 8000" in res.stdout, res.stdout
    assert "STUB-CURL" in calls
    # a container-wide HTTP_PROXY must not answer for the loopback probe
    assert "--noproxy * http://127.0.0.1:8000/api/health" in calls, calls
    assert "previous install is back in place" in res.stderr, res.stderr
    assert (home / "src" / "OLD_TREE").exists(), "the previous source tree was not put back"
    assert not (home / "src" / "NEW_TREE").exists()
    assert not _scratch(home)
    assert "install --no-deps --force-reinstall -r" in calls, calls
    assert calls.splitlines()[-1] == "STUB-SUPERVISORCTL status studio", calls
    env = _studio_env(tmp_path / "ok")
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "5"
    _stub(tmp_path / "ok" / "bin", "curl", "exit 0\n")
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode == 0, res.stdout
    assert "answering on port 8000" in res.stdout
    assert (home / "src" / "NEW_TREE").exists()
    assert not _scratch(home), "the previous tree must go once Studio is up"
    env = _studio_env(tmp_path / "junk")
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "soon"
    _stub(tmp_path / "junk" / "bin", "curl", "exit 0\n")
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stdout
    assert "not a number" in res.stdout


def test_studio_update_health_wait_zero_commits_once_the_restart_command_succeeds(tmp_path: Path):
    """0 asks for no validation: the previous tree goes as soon as the restart command
    returns, and nothing is fetched from port 8000."""
    env = _studio_env(tmp_path)
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "0"
    _stub(tmp_path / "bin", "curl", 'echo "STUB-CURL $*" >> "$STUB_LOG"\nexit 22\n')
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert "the update is committed" in res.stdout, res.stdout
    assert "STUB-CURL" not in _calls(env)
    assert (home / "src" / "NEW_TREE").exists()
    assert not _scratch(home)
    # with a restart that fails there is still nothing to commit
    env = _studio_env(tmp_path / "down", restart_exit=1)
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode != 0
    assert (home / "src" / "OLD_TREE").exists()


def test_studio_update_clears_leftovers_of_a_killed_run_before_it_starts(tmp_path: Path):
    """`docker stop` ends in SIGKILL, so no trap ran: a previous tree can sit beside src
    with a pid-derived name that a later run could reuse (mv would nest into it) and a
    staging tree can sit there with its node_modules."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / ".src-prev.4242" / "junk").mkdir(parents=True)
    (home / ".src-update.abc123" / "node_modules").mkdir(parents=True)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "left behind by an earlier update" in res.stdout, res.stdout
    assert (home / "src" / "NEW_TREE").exists()
    assert not (home / "src" / "junk").exists(), "the tree was nested into the stale dir"
    assert not _scratch(home)


def test_studio_update_recovers_a_source_tree_a_killed_run_moved_aside(tmp_path: Path):
    """SIGKILL between the two moves of the swap leaves no src at all, only .src-prev.*.
    A later plain update must put it back rather than fail on a missing tree."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / "src").rename(home / ".src-prev.abc123")
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "recovering the source tree" in res.stdout, res.stdout
    assert (home / "src" / "OLD_TREE").exists()
    assert not _scratch(home, ".src-prev.*")


def test_studio_update_puts_back_a_previous_tree_a_killed_run_never_committed(tmp_path: Path):
    """SIGKILL after the swap but before the health check committed it leaves the new
    tree in src and the previous one beside it. The new tree was never proven to serve,
    so the previous one goes back; treating it as stale would delete the only known-good
    tree."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / "src").rename(home / ".src-prev.abc123")
    (home / "src" / "studio").mkdir(parents=True)
    (home / "src" / "UNVERIFIED").write_text("never passed the health check\n")
    (home / ".src-update.rollback").write_text("-e file:///opt/prev-src\n")
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "putting it back over the unverified one" in res.stdout, res.stdout
    assert (home / "src" / "OLD_TREE").exists()
    assert not (home / "src" / "UNVERIFIED").exists()
    assert not _scratch(home), "the unverified tree was left behind"


def test_studio_update_reads_the_install_record_from_the_venv_not_the_cwd(tmp_path: Path):
    """`python -` searches the caller's cwd first, so a checkout there with its own
    dist-info would answer for the venv's installed unsloth and a rollback would
    reinstall the wrong thing."""
    env = _studio_env(tmp_path, import_ok=False)
    site = tmp_path / "site"
    info = site / "unsloth-2026.9.4.dist-info"  # seeded by _studio_env
    (info / "direct_url.json").write_text(
        '{"url": "file:///opt/venv-src", "dir_info": {"editable": true}}'
    )
    checkout = tmp_path / "checkout"
    decoy = checkout / "unsloth-0.0.1.dist-info"
    decoy.mkdir(parents=True)
    (decoy / "METADATA").write_text("Metadata-Version: 2.1\nName: unsloth\nVersion: 0.0.1\n")
    (decoy / "direct_url.json").write_text(
        '{"url": "file:///checkout", "dir_info": {"editable": true}}'
    )
    res = _run(STUDIO_UPDATE, ["--no-restart"], env, cwd=checkout)
    assert res.returncode != 0
    assert "STUB-PIP-REQ -e file:///opt/venv-src" in _calls(env), _calls(env)
    assert "checkout" not in _calls(env), _calls(env)
    assert "before: unsloth 2026.9.4" in res.stdout, res.stdout


def test_studio_update_records_every_packages_target_for_the_rollback(tmp_path: Path):
    """--packages can name more than unsloth and unsloth_zoo; a restore that put only
    those two back would leave the extra target upgraded, or newly installed, while
    reporting the previous install is back."""
    env = _studio_env(tmp_path, import_ok=False)
    site = tmp_path / "site"
    info = site / "bar-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: bar\nVersion: 1.0\n")
    res = _run(
        STUDIO_UPDATE, ["--no-restart", "--packages", "unsloth unsloth_zoo bar>=2 foo==2"], env
    )
    calls = _calls(env)
    assert res.returncode != 0
    req = [l for l in calls.splitlines() if l.startswith("STUB-PIP-REQ")][0]
    assert "bar==1.0" in req, req
    assert "# absent: foo" in req, req
    assert "STUB-PIP -m pip uninstall -y foo" in calls, calls
    assert "uninstall -y bar" not in calls


def test_studio_update_with_deps_stops_when_the_dependency_snapshot_fails(tmp_path: Path):
    """An empty snapshot would let the install run with nothing to pin back."""
    env = _studio_env(tmp_path, import_ok=False)
    env["STUB_FREEZE_EXIT"] = "1"
    res = _run(STUDIO_UPDATE, ["--with-deps"], env)
    assert res.returncode != 0
    assert "pip freeze failed" in res.stderr and "nothing was changed" in res.stderr, res.stderr
    assert "install" not in _calls(env), _calls(env)


def test_studio_update_finishes_the_package_restore_a_killed_run_left(tmp_path: Path):
    """SIGKILL after pip had replaced the packages but before the commit: the record
    kept beside the tree says what to put back, and the next run does that before it
    records anything as the previous install."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / ".src-update.rollback").write_text("-e file:///opt/prev-src\n# absent: foo\n")
    (home / ".src-update.freeze").write_text("transformers==3.9.0\n")
    res = _run(STUDIO_UPDATE, [], env)
    calls = _calls(env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "putting the previous packages back first" in res.stdout, res.stdout
    reqs = [l for l in calls.splitlines() if l.startswith("STUB-PIP-REQ")]
    assert reqs[0] == "STUB-PIP-REQ transformers==3.9.0 ", reqs
    assert "-e file:///opt/prev-src" in reqs[1], reqs
    assert "STUB-PIP -m pip uninstall -y foo" in calls, calls
    assert not _scratch(home, ".src-update.*")


@pytest.mark.parametrize("status_exit, verb", [(0, "restart"), (3, "start")])
def test_studio_update_recovery_puts_the_service_on_the_restored_install(
    tmp_path: Path, status_exit, verb
):
    """The killed run may have restarted Studio on the unverified code, or left it
    FATAL. After the packages are back the service is restarted on the restored
    install before this run does anything else, so a run that stops early (the ref
    cannot be fetched) still leaves Studio in a known state."""
    env = _studio_env(tmp_path, status_exit=status_exit, git_ls_exit=1)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / ".src-update.rollback").write_text("-e file:///opt/prev-src\n")
    res = _run(STUDIO_UPDATE, ["--ref", "main"], env)
    calls = _calls(env)
    assert res.returncode != 0, res.stdout
    assert "the previous packages are back" in res.stdout, res.stdout
    assert "running the restored install" in res.stdout, res.stdout
    sup = [l for l in calls.splitlines() if l.startswith("STUB-SUPERVISORCTL")]
    assert sup[1] == f"STUB-SUPERVISORCTL {verb} studio", sup
    assert not (home / ".src-update.rollback").exists()
    env = _studio_env(tmp_path / "noreset", git_ls_exit=1)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / ".src-update.rollback").write_text("-e file:///opt/prev-src\n")
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert "STUB-SUPERVISORCTL" not in _calls(env), _calls(env)
    assert "restart it to load the restored install" in res.stdout, res.stdout


def test_studio_update_with_deps_rollback_fails_when_pip_cannot_list_packages(tmp_path: Path):
    """The list of what the update pulled in comes from a second `pip freeze`; when
    that fails, nothing would be removed, so the restore is reported as unfinished
    and the record stays."""
    env = _studio_env(tmp_path, import_ok=False)
    env["STUB_FREEZE_EXIT_AFTER"] = "1"
    res = _run(STUDIO_UPDATE, ["--with-deps", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode != 0
    assert "could not list the installed packages" in res.stdout, res.stdout
    assert "could not be restored cleanly" in res.stdout, res.stdout
    assert (
        home / ".src-update.rollback"
    ).is_file(), "the record was dropped after a failed restore"


def test_recover_finishes_a_killed_update_and_does_nothing_else(tmp_path: Path):
    """The container start runs --recover before supervisord starts Studio: a previous
    tree beside src goes back over the unverified one, the recorded packages are
    reinstalled, and no update is attempted."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / "src" / "OLD_TREE").unlink()
    (home / "src" / "NEW_TREE").write_text("unverified\n")
    (home / ".src-prev.k9x2Qa" / "studio").mkdir(parents=True)
    (home / ".src-prev.k9x2Qa" / "OLD_TREE").write_text("previous\n")
    (home / ".src-update.rollback").write_text("-e file:///opt/prev-src\n")
    res = _run(STUDIO_UPDATE, ["--recover"], env)
    calls = _calls(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "recovery done" in res.stdout, res.stdout
    assert (home / "src" / "OLD_TREE").exists() and not (home / "src" / "NEW_TREE").exists()
    assert not (home / ".src-prev.k9x2Qa").exists()
    assert not (home / ".src-update.rollback").exists()
    assert "install --no-deps --force-reinstall -r" in calls, calls
    assert "install -U" not in calls and "STUB-GIT" not in calls, calls
    assert not _scratch(home)
    env = _studio_env(tmp_path / "clean")
    res = _run(STUDIO_UPDATE, ["--recover"], env)
    assert res.returncode == 0 and "nothing to recover" in res.stdout, res.stdout
    assert "STUB-PIP -m pip install" not in _calls(env)


def test_studio_update_says_when_the_restore_did_not_finish(tmp_path: Path):
    """pip failing during the restore itself must not be reported as the previous
    install being back; the record stays so the next run finishes it."""
    env = _studio_env(tmp_path, import_ok=False)
    env["STUB_PIP_EXIT"] = "1"
    res = _run(STUDIO_UPDATE, ["--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode != 0
    assert "back in place" not in res.stderr, res.stderr
    assert "could not put every previous package back" in res.stderr, res.stderr
    assert (
        home / ".src-update.rollback"
    ).is_file(), "the record was dropped after a failed restore"
    del env["STUB_PIP_EXIT"]
    res = _run(STUDIO_UPDATE, ["--no-restart"], env)
    assert "putting the previous packages back first" in res.stdout, res.stdout
    assert not (home / ".src-update.rollback").exists()


def test_studio_update_keeps_the_lock_file(tmp_path: Path):
    """Unlinking the lock would let a run that opened the old inode take the lock after
    this one exits while a later run locks a fresh file at the same path."""
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, ["--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert (Path(env["UNSLOTH_STUDIO_HOME"]) / ".src-update.lock").is_file()


def test_studio_update_treats_a_previous_tree_without_a_record_as_scratch(tmp_path: Path):
    """The record is unlinked before the previous tree is deleted, so a previous tree
    with no record is a committed update that was killed mid-delete, and src is the
    tree that passed the health check."""
    env = _studio_env(tmp_path)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    (home / ".src-prev.abc123" / "studio").mkdir(parents=True)
    (home / ".src-prev.abc123" / "COMMITTED_AWAY").write_text("")
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert (home / "src" / "OLD_TREE").exists()
    assert "putting it back" not in res.stdout, res.stdout
    assert not _scratch(home)


def test_studio_update_with_deps_adds_the_studio_extra_to_a_qualified_unsloth_spec(tmp_path: Path):
    """--packages can pin or qualify unsloth; the studio extra must still ride along,
    and unsloth_zoo must not be mistaken for it."""
    env = _studio_env(tmp_path)
    res = _run(
        STUDIO_UPDATE,
        ["--with-deps", "--no-restart", "--packages", "unsloth==2026.9.1 unsloth_zoo"],
        env,
    )
    assert res.returncode == 0, res.stderr + res.stdout
    calls = _calls(env)
    assert "unsloth[studio]==2026.9.1" in calls, calls
    assert "unsloth[studio]_zoo" not in calls and " unsloth_zoo" in calls, calls
    env = _studio_env(tmp_path / "extras")
    res = _run(
        STUDIO_UPDATE, ["--with-deps", "--no-restart", "--packages", "unsloth[cu128]>=2026.9"], env
    )
    assert "unsloth[studio,cu128]>=2026.9" in _calls(env), _calls(env)


def test_studio_update_with_deps_rollback_removes_what_the_update_pulled_in(tmp_path: Path):
    """Reinstalling the snapshot puts back what was there; a dependency the update
    introduced would otherwise stay and become part of the next snapshot."""
    env = _studio_env(tmp_path, import_ok=False)
    res = _run(STUDIO_UPDATE, ["--with-deps", "--no-restart"], env)
    assert res.returncode != 0
    calls = _calls(env)
    assert "STUB-PIP -m pip uninstall -y newdep" in calls, calls
    assert "uninstall -y transformers" not in calls and "uninstall -y torch" not in calls, calls


def test_studio_update_ref_uses_the_lockfile_and_does_not_fall_back_to_npm_install(tmp_path: Path):
    env = _studio_env(tmp_path)
    env["STUB_LOCKFILE"] = "1"
    env["STUB_NPM_CI_EXIT"] = "1"
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    calls = _calls(env)
    assert res.returncode != 0
    assert "npm ci failed" in res.stdout, res.stdout
    frontend_installs = [
        l for l in calls.splitlines() if "STUB-NPM install" in l and l.endswith("frontend")
    ]
    assert not frontend_installs, (
        "a lockfile that does not apply must not be re-resolved:\n" + calls
    )
    assert (home / "src" / "OLD_TREE").exists()
    env = _studio_env(tmp_path / "nolock")
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    calls = _calls(env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "STUB-NPM ci" not in calls, calls
    assert "no package-lock.json" in res.stdout, res.stdout


def test_studio_update_ref_threads_the_npm_registry_override(tmp_path: Path):
    env = _studio_env(tmp_path)
    env["STUB_LOCKFILE"] = "1"
    env["UNSLOTH_NPM_REGISTRY"] = "https://mirror.example/npm/"
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    for line in [
        l for l in _calls(env).splitlines() if "STUB-NPM ci" in l or "STUB-NPM install" in l
    ]:
        assert "--registry https://mirror.example/npm/" in line, line


def test_studio_update_help_prints_only_the_header(tmp_path: Path):
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, ["--help"], env)
    assert res.returncode == 0
    assert "set -euo pipefail" not in res.stdout
    assert "UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT" in res.stdout


def _llama_env(
    tmp_path: Path,
    *,
    latest: str | None,
    marker: str = '{"tag": "b1111-old"}',
) -> dict:
    install = tmp_path / "llama.cpp"
    install.mkdir(parents=True)
    (install / "UNSLOTH_PREBUILT_INFO.json").write_text(
        marker + "\n",
        encoding="utf-8",
    )
    fetcher = tmp_path / "fetch_llama_prebuilt.py"
    resolve = (
        "    raise RuntimeError('unreachable')\n" if latest is None else f"    return {latest!r}\n"
    )
    fetcher.write_text(
        "def resolve_latest_tag(repo):\n" + resolve,
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(install)
    env["UNSLOTH_LLAMA_FETCHER"] = str(fetcher)
    return env


def _llama_check(tmp_path: Path, latest, **kwargs):
    env = _llama_env(tmp_path, latest=latest, **kwargs)
    return _run(LLAMA_UPDATE, ["--check"], env)


def test_llama_check_reports_an_available_update(tmp_path: Path):
    res = _llama_check(tmp_path, "b2222-new")
    assert res.returncode == 0, res.stderr
    assert "an update is available" in res.stdout


def test_llama_check_reports_up_to_date(tmp_path: Path):
    res = _llama_check(tmp_path, "b1111-old")
    assert res.returncode == 0, res.stderr
    assert "up to date" in res.stdout


def test_llama_check_reads_the_full_release_tag_not_the_base_build(tmp_path: Path):
    # the latest pointer is always the full tag_name, so reading the normalized "tag"
    # first offers an update forever on an install that is already current
    res = _llama_check(
        tmp_path,
        "b10715-mix-86bd2d3",
        marker='{"tag": "b10715", "release_tag": "b10715-mix-86bd2d3"}',
    )
    assert res.returncode == 0, res.stderr
    assert "up to date" in res.stdout, (
        "the installed release IS the latest release; reporting an update here nags "
        "forever because applying it cannot change the comparison:\n" + res.stdout
    )
    assert "installed:   b10715-mix-86bd2d3" in res.stdout, (
        "the reported installed version must be the full release identity it is "
        "compared against:\n" + res.stdout
    )


def test_llama_check_still_offers_a_genuinely_newer_release(tmp_path: Path):
    res = _llama_check(
        tmp_path,
        "b10800-mix-aaaaaaa",
        marker='{"tag": "b10715", "release_tag": "b10715-mix-86bd2d3"}',
    )
    assert res.returncode == 0, res.stderr
    assert "an update is available" in res.stdout


def test_llama_check_does_not_claim_up_to_date_when_it_could_not_look(tmp_path: Path):
    res = _llama_check(tmp_path, None)
    assert "up to date" not in res.stdout, (
        "--check exists to report update status; saying 'up to date' for a lookup "
        "that never happened is the one answer it must never give:\n" + res.stdout
    )
    assert res.returncode != 0, "an unperformed check must not exit 0"
    assert "UNKNOWN" in res.stdout + res.stderr


def _llama_inplace_env(tmp_path: Path, old: list[str], new: list[str]) -> dict:
    """An in-place (volume-mounted) install whose activation fails part-way."""
    install = tmp_path / "llama.cpp"
    install.mkdir(parents=True)
    for name in old:
        (install / name).write_text("OLD\n", encoding="utf-8")
    (install / "UNSLOTH_PREBUILT_INFO.json").write_text(
        '{"tag": "b1111-old"}\n',
        encoding="utf-8",
    )
    fetcher = tmp_path / "fetch_llama_prebuilt.py"
    fetcher.write_text(
        "import os, sys\n"
        "def resolve_latest_tag(repo):\n"
        "    return 'b2222-new'\n"
        "if __name__ == '__main__':\n"
        "    dest = sys.argv[3]\n"
        "    os.makedirs(dest, exist_ok = True)\n"
        f"    for name in {new!r}:\n"
        "        open(os.path.join(dest, name), 'w').write('NEW\\n')\n"
        "    open(os.path.join(dest, 'UNSLOTH_PREBUILT_INFO.json'), 'w')"
        '.write(\'{"tag": "b2222-new"}\\n\')\n',
        encoding="utf-8",
    )
    # fail the ACTIVATION move AFTER it moved the files: the mid-swap abort the
    # rollback exists for
    bin_dir = tmp_path / "bin"
    _stub(
        bin_dir,
        "mv",
        'if [ "$1" = "-t" ] && [ "$2" = "$FAIL_MV_TARGET" ]; then\n'
        "  shift 2\n"
        '  for _s in "$@"; do /bin/mv "$_s" "$FAIL_MV_TARGET/"; done\n'
        "  exit 1\n"
        "fi\n"
        'exec /bin/mv "$@"\n',
    )
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(install)
    env["UNSLOTH_LLAMA_FETCHER"] = str(fetcher)
    env["UNSLOTH_LLAMA_UPDATE_IN_PLACE"] = "1"
    env["FAIL_MV_TARGET"] = str(install)
    return env


def test_llama_rollback_leaves_no_new_release_files_behind(tmp_path: Path):
    # only in the new release, so the rollback loop over the BACKUP's entries cannot
    # see it, and ggml would dlopen it against the restored older libggml-base.so
    old = ["libggml-base.so", "libggml-cpu-icelake.so", "llama-cli"]
    new = [
        "libggml-base.so",
        "libggml-cpu-icelake.so",
        "llama-cli",
        "libggml-hexagon.so",
        "llama-mtmd-cli",
    ]
    env = _llama_inplace_env(tmp_path, old, new)
    res = _run(LLAMA_UPDATE, [], env)
    assert res.returncode != 0, "a failed swap must not report success"
    install = tmp_path / "llama.cpp"
    present = sorted(p.name for p in install.iterdir())
    leftovers = [n for n in ("libggml-hexagon.so", "llama-mtmd-cli") if n in present]
    assert not leftovers, f"new-release-only files survived the rollback: {leftovers} in {present}"
    for name in old:
        assert (
            install / name
        ).read_text() == "OLD\n", f"{name} was not restored from the backup: {present}"


def test_llama_rollback_keeps_every_old_file_when_the_drain_is_interrupted(tmp_path: Path):
    # the mirror image: mid-drain, the entries left in the install dir are the only copy
    old = ["libggml-base.so", "libggml-cpu-icelake.so", "llama-cli", "llama-quantize"]
    env = _llama_inplace_env(tmp_path, old, old)
    install = tmp_path / "llama.cpp"
    # fail the DRAIN after one source, so half the old tree is still in the install dir
    _stub(
        tmp_path / "bin",
        "mv",
        'case "${1:-}:${2:-}" in\n'
        "  -t:*/.old.*)\n"
        '    _t="$2"; shift 2\n'
        '    [ $# -gt 0 ] && /bin/mv "$1" "$_t/"\n'
        "    exit 1;;\n"
        "esac\n"
        'exec /bin/mv "$@"\n',
    )
    res = _run(LLAMA_UPDATE, [], env)
    assert res.returncode != 0
    survivors = sorted(p.name for p in install.rglob("*") if p.is_file())
    for name in old:
        assert name in survivors, f"{name} was lost during an interrupted drain: {survivors}"


def _fetcher_module():
    """Import the build-time fetcher by path; stdlib-only, with a __main__ guard."""
    import importlib.util

    path = REPO_ROOT / "docker" / "fetch_llama_prebuilt.py"
    spec = importlib.util.spec_from_file_location("_fetch_llama_prebuilt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "release_tag, expected",
    [
        ("b10715-mix-86bd2d3", "b10715"),
        ("b10715", "b10715"),
        ("  b9596-mix-e6f2453  ", "b9596"),
        ("not-a-build-tag", "not-a-build-tag"),
    ],
)
def test_fetcher_normalizes_the_base_build_for_the_marker_tag(release_tag, expected):
    # the same split install_llama_prebuilt.py writes, or the two installers disagree
    assert _fetcher_module().base_build_tag(release_tag) == expected


def test_a_failed_studio_update_puts_the_previous_install_back(tmp_path: Path):
    """Not restarting protected only the code the running process had already
    imported; the venv on disk stayed replaced, so any lazy import or later restart
    failed the same way. The update now reinstalls exactly what it started from."""
    env = _studio_env(tmp_path, import_ok=False)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode != 0, "a broken update must not report success"
    assert "previous install was restored" in res.stdout, res.stdout
    assert "not restarted" in res.stdout, res.stdout
