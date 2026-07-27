# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Behavioural guards for the two in-container update helpers of the Docker image.

Both are `docker exec` entry points that mutate a running container, so a wrong
answer costs an outage or a mixed-version install:

* `unsloth-studio-update` swaps the Studio Python packages and then restarts the
  service. It verifies the new backend imports first, but only warned -- so a
  release that pulls in a dependency `--no-deps` did not install got the healthy
  old process killed and replaced by one that cannot start. supervisord retries
  `startretries` times, lands in FATAL and never leaves it on its own, so the
  container serves nothing until someone exec's in.
* `unsloth-llama-update --check` reported "up to date" when it could not reach
  the release feed at all, and its in-place rollback only removed entries whose
  names the OLD tree also had, leaving new-release-only shared objects beside
  the restored files. ggml dlopen()s every `libggml-*.so` next to the binaries,
  so that mix is loaded on the next GGUF run.

These drive the real scripts with stub `pip` / `supervisorctl` / `python` /
`mv` on PATH. No docker, no GPU, no network.
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


# --- unsloth-studio-update ----------------------------------------------------


def _studio_env(tmp_path: Path, *, import_ok: bool) -> dict:
    home = tmp_path / "studio"
    venv_bin = home / "unsloth_studio" / "bin"
    venv_bin.mkdir(parents = True)
    _stub(
        venv_bin,
        "python",
        'if [ "$1" = "-c" ]; then\n'
        + (
            "  exit 0\n"
            if import_ok
            else '  case "$2" in *studio.backend.main*) exit 1;; esac\n  exit 0\n'
        )
        + "fi\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  if [ "$3" = "show" ]; then echo "Version: 2026.7.5"; exit 0; fi\n'
        '  echo "STUB-PIP $*" >> "$STUB_LOG"; exit 0\n'
        "fi\n"
        "exit 0\n",
    )
    bin_dir = tmp_path / "bin"
    _stub(
        bin_dir,
        "supervisorctl",
        'echo "STUB-SUPERVISORCTL $*" >> "$STUB_LOG"\n'
        'if [ "$1" = "status" ]; then exit 0; fi\nexit 0\n',
    )
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["UNSLOTH_STUDIO_HOME"] = str(home)
    env["STUB_LOG"] = str(tmp_path / "calls.log")
    return env


def test_studio_update_restarts_when_the_backend_imports(tmp_path: Path):
    env = _studio_env(tmp_path, import_ok = True)
    res = _run(STUDIO_UPDATE, [], env)
    calls = Path(env["STUB_LOG"]).read_text() if Path(env["STUB_LOG"]).exists() else ""
    assert res.returncode == 0, res.stderr
    assert "STUB-SUPERVISORCTL restart studio" in calls, calls


def test_studio_update_does_not_restart_into_a_backend_that_cannot_import(tmp_path: Path):
    env = _studio_env(tmp_path, import_ok = False)
    res = _run(STUDIO_UPDATE, [], env)
    calls = Path(env["STUB_LOG"]).read_text() if Path(env["STUB_LOG"]).exists() else ""
    assert "STUB-SUPERVISORCTL restart studio" not in calls, (
        "restarting into code that cannot import kills a process that is serving "
        "fine and parks supervisord's studio program in FATAL:\n" + calls
    )
    assert res.returncode != 0, "a broken update must not report success"
    assert "--with-deps" in res.stderr, "the remedy must still be printed"


# --- unsloth-llama-update -----------------------------------------------------


def _llama_env(tmp_path: Path, *, latest: str | None) -> dict:
    install = tmp_path / "llama.cpp"
    install.mkdir(parents = True)
    (install / "UNSLOTH_PREBUILT_INFO.json").write_text(
        '{"tag": "b1111-old"}\n',
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


def _llama_check(tmp_path: Path, latest):
    env = _llama_env(tmp_path, latest = latest)
    return _run(LLAMA_UPDATE, ["--check"], env)


def test_llama_check_reports_an_available_update(tmp_path: Path):
    res = _llama_check(tmp_path, "b2222-new")
    assert res.returncode == 0, res.stderr
    assert "an update is available" in res.stdout


def test_llama_check_reports_up_to_date(tmp_path: Path):
    res = _llama_check(tmp_path, "b1111-old")
    assert res.returncode == 0, res.stderr
    assert "up to date" in res.stdout


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
    # Fail the ACTIVATION move (-t <install dir>) AFTER it has moved the files, so
    # the install dir is populated with the new tree and `find` still reports the
    # failure -- the mid-swap abort the rollback exists for. The drain
    # (-t <backup>) and the rollback's own per-file moves must keep working, so
    # only that one invocation is broken.
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
    # "libggml-hexagon.so" exists only in the new release, so the rollback loop --
    # which iterates the BACKUP's entries -- cannot see it. ggml dlopen()s every
    # libggml-*.so sitting next to the binaries, so a leftover is loaded against
    # the restored older libggml-base.so.
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
    # The mirror image: abort while the OLD tree is still being moved into the
    # backup. The entries left in the install dir are then the only copy of those
    # old files, so clearing the directory before restoring would destroy them.
    old = ["libggml-base.so", "libggml-cpu-icelake.so", "llama-cli", "llama-quantize"]
    env = _llama_inplace_env(tmp_path, old, old)
    install = tmp_path / "llama.cpp"
    # Fail the DRAIN (-t <install dir>/.old.<pid>) after moving only the first
    # source, so half the old tree is still sitting in the install dir when the
    # rollback runs. Those entries are then the only copy there is.
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
