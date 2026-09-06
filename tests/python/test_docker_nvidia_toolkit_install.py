# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""docker/install_nvidia_toolkit.sh sets up the one host-side piece the image cannot
carry: the NVIDIA Container Toolkit. Every package manager, restart and verification
path runs here against stubs, and docker/run.sh's offer to run it is checked too."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "docker" / "install_nvidia_toolkit.sh"
RUN_SH = REPO_ROOT / "docker" / "run.sh"

OS_RELEASE = {
    "ubuntu": 'ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\n',
    "debian": "ID=debian\nVERSION_ID=12\n",
    "rhel": 'ID="rhel"\nID_LIKE="fedora"\nVERSION_ID="9.4"\n',
    "fedora": "ID=fedora\nVERSION_ID=41\n",
    "amzn": 'ID="amzn"\nID_LIKE="fedora"\nVERSION_ID="2023"\n',
    "rocky": 'ID="rocky"\nID_LIKE="rhel centos fedora"\n',
    "suse": 'ID="opensuse-leap"\nID_LIKE="suse opensuse"\n',
    "alpine": "ID=alpine\n",
}


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _setup(
    tmp_path: Path,
    *,
    distro: str = "ubuntu",
    driver: bool = True,
    configured: bool = False,
    desktop: bool = False,
    systemctl: str = "ok",
    verify_ok: bool = True,
    uid: int = 0,
    wsl: bool = False,
) -> tuple[Path, Path, dict]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    log = tmp_path / "calls.log"
    marker = tmp_path / "configured"
    if configured:
        marker.write_text("", encoding = "utf-8")
    rec = f'echo "$(basename "$0") $*" >> {log}\n'
    _stub(bindir / "id", f"echo {uid}\n")
    _stub(bindir / "sudo", rec + "exit 0\n")
    _stub(
        bindir / "nvidia-smi",
        'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"\n' if driver else "exit 9\n",
    )
    _stub(bindir / "nvidia-ctk", rec + f"touch {marker}\n")
    _stub(
        bindir / "docker",
        rec
        + 'if [ "$1" = info ]; then\n'
        + ('  echo " Operating System: Docker Desktop"\n' if desktop else "")
        + f'  if [ -e {marker} ]; then echo " Runtimes: io.containerd.runc.v2 nvidia runc"; else echo " Runtimes: io.containerd.runc.v2 runc"; fi\n'
        + "  exit 0\nfi\n"
        + (
            "exit 0\n"
            if verify_ok
            else 'echo "docker: could not select device driver" >&2; exit 125\n'
        ),
    )
    for pm in ("apt-get", "dnf", "yum", "zypper", "service"):
        _stub(bindir / pm, rec + "exit 0\n")
    if systemctl == "ok":
        _stub(bindir / "systemctl", rec + "exit 0\n")
    elif systemctl == "fails":
        _stub(bindir / "systemctl", rec + "exit 1\n")
    _stub(
        bindir / "curl",
        rec + 'case "$*" in\n'
        '  *gpgkey) printf "FAKE-ARMORED-KEY\\n" ;;\n'
        '  *.list) printf "deb https://nvidia.github.io/libnvidia-container/stable/deb/\\$(ARCH) /\\n" ;;\n'
        '  *.repo) printf "[nvidia-container-toolkit]\\nbaseurl=https://nvidia.github.io/libnvidia-container/stable/rpm/\\$basearch\\n" ;;\n'
        "esac\n",
    )
    _stub(
        bindir / "gpg",
        rec + 'while [ $# -gt 0 ]; do if [ "$1" = -o ]; then out="$2"; shift; fi; shift; done\n'
        'printf "DEARMORED:"; cat > "$out"\n',
    )
    osr = tmp_path / "os-release"
    osr.write_text(OS_RELEASE[distro], encoding = "utf-8")
    procv = tmp_path / "proc-version"
    procv.write_text(
        "Linux version 5.15.167.4-microsoft-standard-WSL2\n"
        if wsl
        else "Linux version 6.8.0-45-generic\n",
        encoding = "utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}" + env["PATH"]
    env["UNSLOTH_DESTDIR"] = str(root)
    env["UNSLOTH_OS_RELEASE"] = str(osr)
    env["UNSLOTH_PROC_VERSION"] = str(procv)
    env.pop("UNSLOTH_TOOLKIT_VERIFY", None)
    return root, log, env


def _run(
    env: dict,
    script: Path = INSTALLER,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess:
    e = dict(env)
    e.update(extra_env or {})
    return subprocess.run(["bash", str(script)], capture_output = True, text = True, env = e, timeout = 60)


def _calls(log: Path) -> list[str]:
    return log.read_text(encoding = "utf-8").splitlines() if log.exists() else []


def test_ubuntu_gets_the_apt_recipe_then_docker_is_configured_restarted_and_verified(
    tmp_path: Path,
):
    root, log, env = _setup(tmp_path, distro = "ubuntu")
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(log)
    assert "apt-get update -qq" in calls
    assert any(
        c.startswith("apt-get install") and c.endswith("nvidia-container-toolkit") for c in calls
    )
    key = root / "usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    assert key.read_text(encoding = "utf-8") == "FAKE-ARMORED-KEY\n"
    lst = (root / "etc/apt/sources.list.d/nvidia-container-toolkit.list").read_text(
        encoding = "utf-8"
    )
    assert lst.startswith(
        "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://"
    )
    assert "nvidia-ctk runtime configure --runtime=docker" in calls
    assert "systemctl restart docker" in calls
    assert (
        calls[-1] == "docker run --rm --gpus all --platform linux/amd64 ubuntu:24.04 nvidia-smi -L"
    )
    assert not any(c.startswith(("dnf", "yum", "zypper")) for c in calls)


@pytest.mark.parametrize("distro", ["rhel", "fedora", "amzn", "rocky"])
def test_rpm_distributions_use_dnf_and_the_repo_file(tmp_path: Path, distro: str):
    root, log, env = _setup(tmp_path, distro = distro)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(log)
    assert "dnf install -y nvidia-container-toolkit" in calls
    repo = (root / "etc/yum.repos.d/nvidia-container-toolkit.repo").read_text(encoding = "utf-8")
    assert repo.startswith("[nvidia-container-toolkit]")
    assert not any(c.startswith(("apt-get", "yum", "zypper")) for c in calls)
    assert "nvidia-ctk runtime configure --runtime=docker" in calls


def test_yum_when_dnf_is_absent(tmp_path: Path):
    root, log, env = _setup(tmp_path, distro = "rhel")
    (tmp_path / "bin" / "dnf").unlink()
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "yum install -y nvidia-container-toolkit" in _calls(log)


def test_suse_uses_zypper(tmp_path: Path):
    _, log, env = _setup(tmp_path, distro = "suse")
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(log)
    assert any(c.startswith("zypper --non-interactive ar https://nvidia.github.io/") for c in calls)
    assert (
        "zypper --non-interactive --gpg-auto-import-keys install nvidia-container-toolkit" in calls
    )
    assert not any(c.startswith(("apt-get", "dnf", "yum")) for c in calls)


def test_an_unknown_distribution_stops_before_touching_anything(tmp_path: Path):
    _, log, env = _setup(tmp_path, distro = "alpine")
    res = _run(env)
    assert res.returncode == 2
    assert "unrecognised distribution" in res.stderr
    assert "install-guide" in res.stderr
    assert not any(
        c.startswith(("apt-get", "dnf", "yum", "zypper", "nvidia-ctk")) for c in _calls(log)
    )


def test_no_driver_means_stop_and_say_how_to_get_one(tmp_path: Path):
    _, log, env = _setup(tmp_path, driver = False)
    res = _run(env)
    assert res.returncode == 2
    assert "no NVIDIA driver found" in res.stderr
    assert "570.26" in res.stderr
    assert not any(c.startswith(("apt-get", "nvidia-ctk", "docker run")) for c in _calls(log))


def test_a_configured_host_only_verifies(tmp_path: Path):
    _, log, env = _setup(tmp_path, configured = True)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "already installed" in res.stdout
    calls = _calls(log)
    assert not any(c.startswith(("apt-get", "nvidia-ctk", "systemctl", "service")) for c in calls)
    assert "docker run --rm --gpus all --platform linux/amd64 ubuntu:24.04 nvidia-smi -L" in calls


def test_docker_desktop_needs_nothing(tmp_path: Path):
    _, log, env = _setup(tmp_path, desktop = True, driver = False)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Docker Desktop detected" in res.stdout
    assert _calls(log) == ["docker info"]


@pytest.mark.parametrize("systemctl", ["missing", "fails"])
def test_without_a_working_systemctl_the_service_command_restarts_docker(
    tmp_path: Path, systemctl: str
):
    _, log, env = _setup(tmp_path, systemctl = systemctl, wsl = True)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "service docker restart" in _calls(log)
    assert "WSL 2 host" in res.stdout


def test_a_failed_verification_is_an_error_with_a_pointer(tmp_path: Path):
    _, _, env = _setup(tmp_path, verify_ok = False)
    res = _run(env)
    assert res.returncode == 1
    assert "could not see the GPU" in res.stderr
    assert "troubleshooting" in res.stderr


def test_verification_can_be_skipped(tmp_path: Path):
    _, log, env = _setup(tmp_path, verify_ok = False)
    res = _run(env, extra_env = {"UNSLOTH_TOOLKIT_VERIFY": "0"})
    assert res.returncode == 0, res.stdout + res.stderr
    assert not any(c.startswith("docker run") for c in _calls(log))


def test_a_non_root_run_of_the_file_re_executes_itself_under_sudo(tmp_path: Path):
    _, log, env = _setup(tmp_path, uid = 1000)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert _calls(log) == [f"sudo -E bash {INSTALLER}"]


def test_a_non_root_pipe_cannot_re_execute_and_says_so(tmp_path: Path):
    _, log, env = _setup(tmp_path, uid = 1000)
    res = subprocess.run(
        ["bash", "-s"],
        input = INSTALLER.read_text(encoding = "utf-8"),
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )
    assert res.returncode == 2
    assert "sudo bash" in res.stderr
    assert _calls(log) == []


def _run_sh_env(tmp_path: Path, *, nvidia_runtime: bool) -> tuple[Path, Path, dict]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    argv = tmp_path / "argv"
    rec = f'echo "$(basename "$0") $*" >> {log}\n'
    runtimes = (
        "io.containerd.runc.v2 nvidia runc" if nvidia_runtime else "io.containerd.runc.v2 runc"
    )
    _stub(
        bindir / "docker",
        rec + f'if [ "$1" = info ]; then echo " Runtimes: {runtimes}"; exit 0; fi\n'
        f'printf "%s\\n" "$@" > {argv}\nexit 0\n',
    )
    _stub(bindir / "nvidia-smi", 'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"\n')
    _stub(bindir / "sudo", rec + "exit 0\n")
    _stub(bindir / "getent", "exit 2\n")
    dev_root = tmp_path / "root"
    (dev_root / "dev").mkdir(parents = True)
    (dev_root / "dev" / "nvidiactl").write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}" + env["PATH"]
    env["UNSLOTH_DEV_ROOT"] = str(dev_root)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    env.pop("UNSLOTH_INSTALL_TOOLKIT", None)
    return log, argv, env


def _run_run_sh(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RUN_SH), "true"],
        capture_output = True,
        text = True,
        env = env,
        stdin = subprocess.DEVNULL,
        timeout = 60,
    )


def test_run_sh_installs_the_toolkit_when_told_to_then_runs(tmp_path: Path):
    log, argv, env = _run_sh_env(tmp_path, nvidia_runtime = False)
    env["UNSLOTH_INSTALL_TOOLKIT"] = "1"
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"sudo bash {INSTALLER}" in _calls(log)
    assert "--gpus\nall\n" in argv.read_text(encoding = "utf-8")


def test_run_sh_without_a_terminal_prints_the_one_liner_and_still_runs(tmp_path: Path):
    log, argv, env = _run_sh_env(tmp_path, nvidia_runtime = False)
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "install_nvidia_toolkit.sh | sudo bash" in res.stderr
    assert not any(c.startswith("sudo") for c in _calls(log))
    assert argv.exists()


def test_run_sh_is_quiet_when_the_runtime_is_present(tmp_path: Path):
    _, _, env = _run_sh_env(tmp_path, nvidia_runtime = True)
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Container Toolkit" not in res.stderr


def test_the_docs_point_at_the_installer():
    for doc in (REPO_ROOT / "docker" / "DOCKERHUB.md", REPO_ROOT / "README.md"):
        text = doc.read_text(encoding = "utf-8")
        assert "docker/install_nvidia_toolkit.sh | sudo bash" in text, doc
    assert "Docker Desktop" in (REPO_ROOT / "docker" / "DOCKERHUB.md").read_text(encoding = "utf-8")
