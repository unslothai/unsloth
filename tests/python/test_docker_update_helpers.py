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
    reason = "needs bash",
)


def _stub(directory: Path, name: str, body: str) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / name
    path.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    path.chmod(0o755)


def _run(
    script: Path,
    args,
    env,
    cwd = None,
):
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output = True,
        text = True,
        env = env,
        cwd = cwd,
        timeout = 120,
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
) -> dict:
    """A Studio home whose venv python is the real interpreter over a fake `studio`
    package, so the script's own import and frontend checks run for real. pip,
    supervisorctl, git and the bundled npm are recording stubs."""
    home = tmp_path / "studio"
    site = tmp_path / "site"
    (site / "studio" / "backend").mkdir(parents = True)
    (site / "studio" / "__init__.py").write_text("")
    (site / "studio" / "backend" / "__init__.py").write_text("")
    (site / "studio" / "backend" / "main.py").write_text(
        "" if import_ok else "raise ImportError('No module named structlog')\n"
    )
    if dist_ok:
        (site / "studio" / "frontend" / "dist").mkdir(parents = True)
        (site / "studio" / "frontend" / "dist" / "index.html").write_text("<html></html>")
    # the Studio image presents src as a link into its own copy of Studio, so the home
    # can be a volume; a standalone image has it as a real directory
    src = (tmp_path / "app" / "src") if src_link else (home / "src")
    (src / "studio").mkdir(parents = True)
    (src / "OLD_TREE").write_text("previous source tree\n")
    if src_link:
        home.mkdir(parents = True, exist_ok = True)
        (home / "src").symlink_to(src)

    venv_bin = home / "unsloth_studio" / "bin"
    _stub(
        venv_bin,
        "python",
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  echo "STUB-PIP $*" >> "$STUB_LOG"\n'
        # reinstalling the recorded previous install brings the working tree back
        '  case " $* " in *" -r "*)\n'
        f'    : > "{site}/studio/backend/main.py"\n'
        f'    mkdir -p "{site}/studio/frontend/dist"; echo ok > "{site}/studio/frontend/dist/index.html" ;;\n'
        "  esac\n"
        '  exit "${STUB_PIP_EXIT:-0}"\n'
        "fi\n"
        f'PYTHONPATH="{site}" exec "{shutil.which("python3")}" "$@"\n',
    )
    _stub(
        home / "node" / "bin",
        "npm",
        'echo "STUB-NPM $* in $PWD" >> "$STUB_LOG"\n'
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
        f'if [ "$1" = "status" ]; then exit {status_exit}; fi\nexit 0\n',
    )
    _stub(
        bin_dir,
        "git",
        'dir=""; if [ "$1" = "-C" ]; then dir="$2"; shift 2; fi\n'
        'case "$1" in\n'
        f"  ls-remote) exit {git_ls_exit} ;;\n"
        '  checkout) mkdir -p "$dir/studio/frontend/src" "$dir/.git";'
        ' echo "{}" > "$dir/studio/frontend/package.json"; echo new > "$dir/NEW_TREE" ;;\n'
        "esac\nexit 0\n",
    )
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    env["UNSLOTH_STUDIO_UPDATE_HEALTH_WAIT"] = "0"
    env["STUB_LOG"] = str(tmp_path / "calls.log")
    return env


def _calls(env) -> str:
    log = Path(env["STUB_LOG"])
    return log.read_text() if log.exists() else ""


def test_studio_update_restarts_when_the_backend_imports(tmp_path: Path):
    env = _studio_env(tmp_path)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "STUB-SUPERVISORCTL restart studio" in _calls(env), _calls(env)


def test_studio_update_does_not_restart_into_a_backend_that_cannot_import(tmp_path: Path):
    env = _studio_env(tmp_path, import_ok = False)
    res = _run(STUDIO_UPDATE, [], env)
    calls = _calls(env)
    assert "STUB-SUPERVISORCTL restart studio" not in calls, (
        "restarting into code that cannot import kills a process that is serving "
        "fine and parks supervisord's studio program in FATAL:\n" + calls
    )
    assert res.returncode != 0, "a broken update must not report success"
    assert "--with-deps" in res.stdout, "the remedy must still be printed"
    assert "install --no-deps -r" in calls, "the previous install must be put back:\n" + calls


def test_studio_update_does_not_restart_into_a_tree_without_a_built_frontend(tmp_path: Path):
    """The --ref failure: a git build has no studio/frontend/dist, `unsloth studio`
    exits 1 on start, and three quick exits leave supervisord's program FATAL."""
    env = _studio_env(tmp_path, dist_ok = False)
    res = _run(STUDIO_UPDATE, [], env)
    calls = _calls(env)
    assert res.returncode != 0
    assert "no built frontend" in res.stdout
    assert "STUB-SUPERVISORCTL restart studio" not in calls, calls
    assert "install --no-deps -r" in calls, calls


@pytest.mark.parametrize("status_exit", [0, 3])
def test_studio_update_restarts_a_studio_that_is_not_running(tmp_path: Path, status_exit):
    """`supervisorctl status` exits 3 for STOPPED/EXITED/FATAL. That is still a program
    supervisord manages, and FATAL is what a failed earlier update left behind."""
    env = _studio_env(tmp_path, status_exit = status_exit)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "STUB-SUPERVISORCTL restart studio" in _calls(env)
    assert "not managing" not in res.stdout


def test_studio_update_reports_an_unmanaged_studio(tmp_path: Path):
    env = _studio_env(tmp_path, status_exit = 4)
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
    assert not list(home.glob(".src-prev.*")), "the previous tree was not cleaned up"
    assert not list(home.glob(".src-update.*")), "the staging tree was not cleaned up"
    assert f"install --no-deps -e {home / 'src'}" in calls, calls


def test_studio_update_ref_writes_through_a_linked_source_tree(tmp_path: Path):
    """Replacing that link with a directory would put the tree outside the image's copy,
    where the next container start relinks over it and the update is silently gone."""
    env = _studio_env(tmp_path, src_link = True)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    real_src = tmp_path / "app" / "src"
    assert res.returncode == 0, res.stderr + res.stdout
    assert (home / "src").is_symlink(), "the link was replaced by a directory"
    assert (real_src / "NEW_TREE").exists(), "the ref was not installed where the code lives"
    assert not list(home.glob(".src-*")), "staging trees must not land in the data volume"
    assert not list(real_src.parent.glob(".src-prev.*"))
    assert f"install --no-deps -e {real_src}" in _calls(env), _calls(env)


def test_studio_update_ref_with_a_failed_frontend_build_changes_nothing(tmp_path: Path):
    env = _studio_env(tmp_path, npm_build_ok = False)
    res = _run(STUDIO_UPDATE, ["--ref", "main", "--no-restart"], env)
    home = Path(env["UNSLOTH_STUDIO_HOME"])
    assert res.returncode != 0
    assert (home / "src" / "OLD_TREE").exists(), "the running source tree was touched"
    assert "STUB-PIP" not in _calls(env), _calls(env)
    assert not list(home.glob(".src-update.*"))


def test_studio_update_ref_puts_the_old_tree_back_when_the_new_one_cannot_start(tmp_path: Path):
    env = _studio_env(tmp_path, import_ok = False)
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
    env = _studio_env(tmp_path, git_ls_exit = 0)
    res = _run(STUDIO_UPDATE, ["--ref", "v2026.7.5", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert _zoo_spec(_calls(env)).endswith("@v2026.7.5#egg=unsloth_zoo"), _calls(env)


def test_studio_update_falls_back_to_zoo_main_when_the_ref_is_absent(tmp_path: Path):
    env = _studio_env(tmp_path, git_ls_exit = 2)
    res = _run(STUDIO_UPDATE, ["--ref", "v2026.7.5", "--no-restart"], env)
    assert res.returncode == 0, res.stderr + res.stdout
    assert _zoo_spec(_calls(env)).endswith("@main#egg=unsloth_zoo"), _calls(env)
    assert "has no ref" in res.stdout, res.stdout


def test_studio_update_aborts_when_the_zoo_lookup_never_reached_the_remote(tmp_path: Path):
    # treating 2 and 128 alike pairs the requested unsloth revision with an unrelated
    # zoo one once the network recovers, across a private API
    env = _studio_env(tmp_path, git_ls_exit = 128)
    res = _run(STUDIO_UPDATE, ["--ref", "v2026.7.5", "--no-restart"], env)
    calls = _calls(env)
    assert "STUB-PIP" not in calls, "a transport failure must not install anything:\n" + calls
    assert res.returncode != 0, "an unresolvable zoo ref must not report success"
    assert "has no ref" not in res.stdout, (
        "an unreachable remote must not be reported as a missing ref:\n" + res.stdout
    )
    assert "--zoo-ref" in res.stderr, "the remedy must be printed"


def _llama_env(
    tmp_path: Path,
    *,
    latest: str | None,
    marker: str = '{"tag": "b1111-old"}',
) -> dict:
    install = tmp_path / "llama.cpp"
    install.mkdir(parents = True)
    (install / "UNSLOTH_PREBUILT_INFO.json").write_text(
        marker + "\n",
        encoding = "utf-8",
    )
    fetcher = tmp_path / "fetch_llama_prebuilt.py"
    resolve = (
        "    raise RuntimeError('unreachable')\n" if latest is None else f"    return {latest!r}\n"
    )
    fetcher.write_text(
        "def resolve_latest_tag(repo):\n" + resolve,
        encoding = "utf-8",
    )
    env = dict(os.environ)
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(install)
    env["UNSLOTH_LLAMA_FETCHER"] = str(fetcher)
    return env


def _llama_check(tmp_path: Path, latest, **kwargs):
    env = _llama_env(tmp_path, latest = latest, **kwargs)
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
        marker = '{"tag": "b10715", "release_tag": "b10715-mix-86bd2d3"}',
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
        marker = '{"tag": "b10715", "release_tag": "b10715-mix-86bd2d3"}',
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
    install.mkdir(parents = True)
    for name in old:
        (install / name).write_text("OLD\n", encoding = "utf-8")
    (install / "UNSLOTH_PREBUILT_INFO.json").write_text(
        '{"tag": "b1111-old"}\n',
        encoding = "utf-8",
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
        encoding = "utf-8",
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
    env = _studio_env(tmp_path, import_ok = False)
    res = _run(STUDIO_UPDATE, [], env)
    assert res.returncode != 0, "a broken update must not report success"
    assert "previous install was restored" in res.stdout, res.stdout
    assert "not restarted" in res.stdout, res.stdout
