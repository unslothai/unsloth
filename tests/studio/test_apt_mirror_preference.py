"""retry-with-apt-lock.sh ranks the apt mirrors by measured speed and demotes the leader after a failure.

GitHub's Ubuntu images list azure.archive.ubuntu.com first in /etc/apt/apt-mirrors.txt, and apt only
fails over on errors, so a mirror trickling at a few KB/s held every retry (runner-images #14823).
These tests run the real helper against a temporary mirror file with `sudo`, `curl` and friends
stubbed on PATH, so nothing touches the host's apt.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".github" / "scripts" / "retry-with-apt-lock.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "bash helper")

AZURE = "http://azure.archive.ubuntu.com/ubuntu/"
GLOBAL = "http://archive.ubuntu.com/ubuntu/"
SECURITY = "http://security.ubuntu.com/ubuntu/"


def _stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body, encoding = "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path: Path, mirrors: str, speeds: dict, command: str, **env) -> tuple:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    mirrors_file = tmp_path / "apt-mirrors.txt"
    mirrors_file.write_text(mirrors, encoding = "utf-8")
    log = tmp_path / "orders.log"

    # sudo runs the command as is; tee writes where the helper points it, apt.conf.d is redirected.
    _stub(
        bin_dir,
        "sudo",
        'case "$1" in tee) [ "${2#/etc/apt/apt.conf.d}" != "$2" ] && exec cat > /dev/null ;; esac\nexec "$@"\n',
    )
    lines = "".join(f'  *"{host}"*) echo {bps} ;;\n' for host, bps in speeds.items())
    _stub(bin_dir, "curl", f'url="${{@: -1}}"\ncase "$url" in\n{lines}  *) echo 0 ;;\nesac\n')
    _stub(bin_dir, "pkill", "exit 1\n")
    _stub(bin_dir, "dpkg", '[ "$1" = --print-architecture ] && echo amd64\nexit 0\n')
    _stub(bin_dir, "fuser", "exit 1\n")
    _stub(bin_dir, "sleep", "exit 0\n")
    # Records the mirror order each attempt saw.
    _stub(
        bin_dir,
        "attempt",
        f'awk \'{{print $1}}\' "{mirrors_file}" | paste -sd" " >> "{log}"\n{command}\n',
    )

    run_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "APT_MIRRORS_FILE": str(mirrors_file),
        "RUNNER_TEMP": str(tmp_path),
        "RETRY_ATTEMPTS": "3",
        "RETRY_ATTEMPT_TIMEOUT": "30",
        **env,
    }
    proc = subprocess.run(
        ["bash", str(HELPER), "attempt"],
        env = run_env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    orders = log.read_text(encoding = "utf-8").splitlines() if log.exists() else []
    return proc, orders, mirrors_file.read_text(encoding = "utf-8")


def _has_os_release() -> bool:
    try:
        return "VERSION_CODENAME=" in Path("/etc/os-release").read_text(encoding = "utf-8")
    except OSError:
        return False


needs_codename = pytest.mark.skipif(not _has_os_release(), reason = "needs /etc/os-release")


@needs_codename
def test_the_fastest_mirror_goes_first(tmp_path):
    mirrors = f"{AZURE}\tpriority:1\n{SECURITY}\tpriority:2\n"
    proc, orders, written = _run(
        tmp_path,
        mirrors,
        {"azure.archive": 30_000, "//archive.ubuntu": 9_000_000, "security": 4_000_000},
        "exit 0",
    )
    assert proc.returncode == 0, proc.stderr
    # The global archive is added beside an *.archive.ubuntu.com mirror and wins on speed.
    assert orders == [f"{GLOBAL} {SECURITY} {AZURE}"], proc.stdout
    assert f"{GLOBAL}\tpriority:1" in written
    assert f"{AZURE}\tpriority:3" in written


@needs_codename
def test_a_fast_azure_mirror_keeps_its_place(tmp_path):
    mirrors = f"{AZURE}\tpriority:1\n{GLOBAL}\tpriority:2\n"
    proc, orders, _ = _run(
        tmp_path, mirrors, {"azure.archive": 9_000_000, "//archive.ubuntu": 1_000_000}, "exit 0"
    )
    assert proc.returncode == 0, proc.stderr
    assert orders == [f"{AZURE} {GLOBAL}"]


@needs_codename
def test_a_failed_attempt_moves_the_leader_last(tmp_path):
    mirrors = f"{AZURE}\tpriority:1\n{GLOBAL}\tpriority:2\n"
    # Fails the first time only.
    command = f'[ -e "{tmp_path}/done" ] && exit 0; touch "{tmp_path}/done"; exit 100'
    proc, orders, _ = _run(
        tmp_path, mirrors, {"azure.archive": 9_000_000, "//archive.ubuntu": 1_000_000}, command
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert orders == [f"{AZURE} {GLOBAL}", f"{GLOBAL} {AZURE}"], proc.stdout


@needs_codename
def test_mirrors_are_ranked_once_per_job(tmp_path):
    mirrors = f"{AZURE}\tpriority:1\n{GLOBAL}\tpriority:2\n"
    _run(tmp_path, mirrors, {"azure.archive": 1_000, "//archive.ubuntu": 9_000_000}, "exit 0")
    # A second helper call in the same job keeps the order it finds, even if speeds have flipped.
    proc, orders, _ = _run(
        tmp_path,
        f"{GLOBAL}\tpriority:1\n{AZURE}\tpriority:2\n",
        {"azure.archive": 9_000_000, "//archive.ubuntu": 1_000},
        "exit 0",
    )
    assert proc.returncode == 0
    assert orders[-1] == f"{GLOBAL} {AZURE}"


def test_no_mirror_file_runs_the_command_unchanged(tmp_path):
    proc, orders, _ = _run(
        tmp_path, "", {}, "exit 0", APT_MIRRORS_FILE = str(tmp_path / "absent.txt")
    )
    assert proc.returncode == 0, proc.stderr
    assert "apt mirror" not in proc.stdout


@needs_codename
def test_opt_out_leaves_the_mirror_file_alone(tmp_path):
    mirrors = f"{AZURE}\tpriority:1\n"
    proc, _, written = _run(
        tmp_path, mirrors, {"//archive.ubuntu": 9_000_000}, "exit 0", APT_PREFER_FAST_MIRROR = "0"
    )
    assert proc.returncode == 0
    assert written == mirrors


@needs_codename
def test_ports_mirrors_get_no_global_archive(tmp_path):
    ports = "http://ports.ubuntu.com/ubuntu-ports/"
    proc, orders, _ = _run(tmp_path, f"{ports}\tpriority:1\n", {"ports": 5_000_000}, "exit 0")
    assert proc.returncode == 0
    assert orders == [ports]
