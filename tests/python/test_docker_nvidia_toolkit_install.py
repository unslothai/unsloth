# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""docker/install_nvidia_toolkit.sh sets up the one host-side piece the image cannot
carry: the NVIDIA Container Toolkit. Every package manager, restart and verification
path runs here against stubs, and docker/run.sh's offer to run it is checked too."""

from __future__ import annotations

import os
import shutil
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


# The scripts under test are given ONLY these real tools, next to the stubs. A branch
# that reaches for an absent command (dnf gone, systemctl gone) must find nothing,
# never the runner's own package manager or init.
_REAL_TOOLS = (
    "bash",
    "grep",
    "sed",
    "head",
    "tr",
    "sort",
    "mkdir",
    "cat",
    "basename",
    "dirname",
    "touch",
    "mv",
    "cp",
    "chmod",
    "stat",
    "cut",
    "rm",
    "true",
)


def _isolated_path(tmp_path: Path, bindir: Path) -> str:
    sysbin = tmp_path / "sysbin"
    sysbin.mkdir(exist_ok = True)
    for name in _REAL_TOOLS:
        real = shutil.which(name)
        assert real, f"{name} not found on this host"
        (sysbin / name).symlink_to(real)
    return f"{bindir}{os.pathsep}{sysbin}"


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _setup(
    tmp_path: Path,
    *,
    distro: str = "ubuntu",
    driver: bool = True,
    configured: bool = False,
    toolkit_installed: bool | None = None,
    desktop: bool = False,
    systemctl: str = "ok",
    verify_ok: bool = True,
    uid: int = 0,
    wsl: bool = False,
    driver_version: str = "580.65.06",
    rootless: bool | str = False,
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
        (
            f'if [ "$1" = --query-gpu=driver_version ]; then echo " {driver_version}"; exit 0; fi\n'
            'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"\n'
        )
        if driver
        else "exit 9\n",
    )
    # the installer picks the verification platform from uname -m; pin it so the
    # suite means the same thing on an arm64 runner
    _stub(bindir / "uname", 'if [ "$1" = -s ]; then echo Linux; else echo x86_64; fi\n')
    ctk = bindir / "nvidia-ctk"
    ctk_body = rec + f"touch {marker}\n"
    if configured or toolkit_installed:
        _stub(ctk, ctk_body)
    (tmp_path / "ctk-stub-body").write_text("#!/usr/bin/env bash\n" + ctk_body, encoding = "utf-8")
    _stub(
        bindir / "docker",
        rec
        + 'if [ "$1" = context ]; then case "${DOCKER_CONTEXT:-}" in remote*) echo "tcp://gpu-box:2376" ;; *) echo "unix:///var/run/docker.sock" ;; esac; exit 0; fi\n'
        + 'if [ "$1" = info ]; then\n'
        + ('  echo " Operating System: Docker Desktop"\n' if desktop else "")
        + (
            '  if [ "$2" = "--format" ]; then\n'
            '    case "${DOCKER_HOST:-}" in *docker.sock) echo "name=rootless,name=seccomp" ;; *) echo "name=seccomp" ;; esac; exit 0\n'
            "  fi\n"
            if rootless == "via_sudo_uid"
            else (
                '  if [ "$2" = "--format" ]; then echo "name=rootless,name=seccomp"; exit 0; fi\n'
                if rootless
                else ""
            )
        )
        + f'  if [ -e {marker} ]; then echo " Runtimes: io.containerd.runc.v2 nvidia runc"; else echo " Runtimes: io.containerd.runc.v2 runc"; fi\n'
        + "  exit 0\nfi\n"
        + (
            "exit 0\n"
            if verify_ok
            else 'echo "docker: could not select device driver" >&2; exit 125\n'
        ),
    )
    # installing the package makes nvidia-ctk appear, as it does for real
    installs = f'case "$*" in *install*nvidia-container-toolkit*) cp {tmp_path / "ctk-stub-body"} {ctk}; chmod 755 {ctk} ;; esac\n'
    for pm in ("apt-get", "dnf", "yum", "zypper"):
        _stub(bindir / pm, rec + installs + "exit 0\n")
    _stub(bindir / "service", rec + "exit 0\n")
    if systemctl == "ok":
        _stub(bindir / "systemctl", rec + "exit 0\n")
    elif systemctl == "fails":
        _stub(bindir / "systemctl", rec + "exit 1\n")
    _stub(
        bindir / "curl",
        rec + 'case "$*" in\n'
        '  *gpgkey) printf "FAKE-ARMORED-KEY\\n" ;;\n'
        '  *.list) if [ -e "$LIST_DOWNLOAD_FAILS" ]; then echo "curl: (22) 503" >&2; exit 22; fi; printf "deb https://nvidia.github.io/libnvidia-container/stable/deb/\\$(ARCH) /\\n" ;;\n'
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
    env["PATH"] = _isolated_path(tmp_path, bindir)
    env["UNSLOTH_DESTDIR"] = str(root)
    env["LIST_DOWNLOAD_FAILS"] = str(tmp_path / "list-download-fails")
    env["UNSLOTH_OS_RELEASE"] = str(osr)
    env["UNSLOTH_PROC_VERSION"] = str(procv)
    env["UNSLOTH_RUN_USER_DIR"] = str(tmp_path / "run-user")
    env["UNSLOTH_WSL_LIB_DIR"] = str(tmp_path / "wsl-lib")
    env.pop("UNSLOTH_TOOLKIT_VERIFY", None)
    env.pop("SUDO_UID", None)
    env.pop("DOCKER_HOST", None)
    env.pop("DOCKER_CONTEXT", None)
    return root, log, env


def _run(
    env: dict,
    script: Path = INSTALLER,
    extra_env: dict | None = None,
    umask: str | None = None,
) -> subprocess.CompletedProcess:
    e = dict(env)
    e.update(extra_env or {})
    cmd = ["bash", str(script)]
    if umask:
        cmd = ["bash", "-c", f'umask {umask}; exec bash "$0"', str(script)]
    return subprocess.run(cmd, capture_output = True, text = True, env = e, timeout = 60)


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


def test_an_old_driver_is_reported_after_the_toolkit_is_in_place(tmp_path: Path):
    _, log, env = _setup(tmp_path, driver_version = "550.54.15")
    res = _run(env)
    assert res.returncode == 3
    assert "550.54.15 is below 570.26" in res.stderr
    calls = _calls(log)
    assert "nvidia-ctk runtime configure --runtime=docker" in calls
    assert any(c.startswith("docker run") for c in calls)


def test_a_current_driver_is_confirmed(tmp_path: Path):
    _, _, env = _setup(tmp_path, driver_version = "570.26")
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "570.26 meets the 570.26 minimum" in res.stdout


def test_rootless_docker_is_refused_before_anything_runs(tmp_path: Path):
    _, log, env = _setup(tmp_path, rootless = True, uid = 1000)
    res = _run(env)
    assert res.returncode == 2
    assert "rootless" in res.stderr and "#rootless-mode" in res.stderr
    assert not any(c.startswith(("sudo", "apt-get", "nvidia-ctk")) for c in _calls(log))


def test_rootless_docker_is_still_caught_when_piped_into_sudo_bash(tmp_path: Path):
    """`curl | sudo bash` arrives as root with the user's DOCKER_HOST stripped by
    env_reset; SUDO_UID survives, so the user's socket is probed through it."""
    import socket

    _, log, env = _setup(tmp_path, rootless = "via_sudo_uid", uid = 0)
    sock_dir = tmp_path / "run-user" / "1000"
    sock_dir.mkdir(parents = True)
    s = socket.socket(socket.AF_UNIX)
    cwd = os.getcwd()
    os.chdir(sock_dir)  # AF_UNIX paths are capped at 108 bytes; bind relative
    try:
        s.bind("docker.sock")
    finally:
        os.chdir(cwd)
    env["SUDO_UID"] = "1000"
    res = _run(env)
    s.close()
    assert res.returncode == 2
    assert "rootless" in res.stderr
    assert not res.stderr.rstrip().endswith(" 2"), "the exit code leaked into the message"
    assert not any(c.startswith(("apt-get", "nvidia-ctk")) for c in _calls(log))


def test_without_docker_the_script_says_so(tmp_path: Path):
    _, log, env = _setup(tmp_path)
    (tmp_path / "bin" / "docker").unlink()  # the PATH holds no real docker either
    res = _run(env)
    assert res.returncode == 2
    assert "docker is not installed" in res.stderr
    assert _calls(log) == []


def test_docker_context_wins_over_a_local_docker_host(tmp_path: Path):
    _, log, env = _setup(tmp_path, uid = 1000)
    env["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    env["DOCKER_CONTEXT"] = "remote-gpu"
    res = _run(env)
    assert res.returncode == 2
    assert "remote daemon (tcp://gpu-box:2376)" in res.stderr
    assert not any(c.startswith(("sudo", "apt-get")) for c in _calls(log))


def test_nvidia_smi_is_found_in_the_wsl_library_dir_after_sudo(tmp_path: Path):
    """secure_path drops /usr/lib/wsl/lib even with sudo -E, where the Windows
    driver keeps nvidia-smi on a WSL 2 distro running its own Docker Engine."""
    _, log, env = _setup(tmp_path, wsl = True)
    smi = tmp_path / "bin" / "nvidia-smi"
    wsl_lib = tmp_path / "wsl-lib"
    wsl_lib.mkdir()
    smi.rename(wsl_lib / "nvidia-smi")
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "nvidia-ctk runtime configure --runtime=docker" in _calls(log)
    assert "meets the 570.26 minimum" in res.stdout


def test_a_failed_source_list_download_keeps_the_existing_file(tmp_path: Path):
    root, log, env = _setup(tmp_path, distro = "ubuntu")
    lst = root / "etc/apt/sources.list.d/nvidia-container-toolkit.list"
    lst.parent.mkdir(parents = True)
    lst.write_text("deb [signed-by=/usr/share/keyrings/x.gpg] https://old /\n", encoding = "utf-8")
    (tmp_path / "list-download-fails").write_text("", encoding = "utf-8")
    res = _run(env)
    assert res.returncode != 0
    assert lst.read_text(encoding = "utf-8").startswith(
        "deb [signed-by=/usr/share/keyrings/x.gpg] https://old"
    )
    assert not any(c.startswith("nvidia-ctk") for c in _calls(log))


def test_an_unsupported_architecture_is_refused_before_any_install(tmp_path: Path):
    _, log, env = _setup(tmp_path)
    _stub(tmp_path / "bin" / "uname", 'if [ "$1" = -s ]; then echo Linux; else echo ppc64le; fi\n')
    res = _run(env)
    assert res.returncode == 2
    assert "unsupported architecture ppc64le" in res.stderr
    assert not any(c.startswith(("apt-get", "nvidia-ctk", "docker run")) for c in _calls(log))


def test_an_installed_toolkit_is_only_registered(tmp_path: Path):
    """nvidia-ctk present, runtime entry gone: no repository or package work, which
    is what an offline host needs."""
    _, log, env = _setup(tmp_path, toolkit_installed = True)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(log)
    assert not any(c.startswith(("apt-get", "dnf", "yum", "zypper", "curl")) for c in calls)
    assert "nvidia-ctk runtime configure --runtime=docker" in calls
    assert "systemctl restart docker" in calls


def test_the_keyring_is_world_readable_under_a_strict_umask(tmp_path: Path):
    root, _, env = _setup(tmp_path, distro = "ubuntu")
    res = _run(env, umask = "077")
    assert res.returncode == 0, res.stdout + res.stderr
    key = root / "usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    assert oct(key.stat().st_mode & 0o777) == "0o644"


def test_a_remote_docker_endpoint_is_refused(tmp_path: Path):
    _, log, env = _setup(tmp_path, uid = 1000)
    env["DOCKER_HOST"] = "tcp://gpu-box:2376"
    res = _run(env)
    assert res.returncode == 2
    assert "remote daemon (tcp://gpu-box:2376)" in res.stderr
    assert not any(c.startswith(("sudo", "apt-get", "nvidia-ctk")) for c in _calls(log))


def test_docker_desktop_on_native_linux_is_refused(tmp_path: Path):
    _, log, env = _setup(tmp_path, desktop = True, driver = False, uid = 1000)
    res = _run(env)
    assert res.returncode == 2
    assert "Docker Desktop for Linux has no NVIDIA GPU support" in res.stderr
    assert not any(c.startswith(("sudo", "apt-get")) for c in _calls(log))


def test_a_runtime_registered_without_nvidia_ctk_is_left_alone(tmp_path: Path):
    _, log, env = _setup(tmp_path, configured = True)
    (tmp_path / "bin" / "nvidia-ctk").unlink()
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = _calls(log)
    assert not any(c.startswith(("apt-get", "dnf", "curl", "systemctl")) for c in calls)
    assert "docker run --rm --gpus all --platform linux/amd64 ubuntu:24.04 nvidia-smi -L" in calls


def test_a_second_daemon_on_its_own_socket_is_refused(tmp_path: Path):
    _, log, env = _setup(tmp_path, uid = 1000)
    env["DOCKER_HOST"] = "unix:///run/dockerd-b/docker.sock"
    res = _run(env)
    assert res.returncode == 2
    assert "not the system daemon" in res.stderr
    assert not any(c.startswith(("sudo", "apt-get")) for c in _calls(log))


def test_the_default_sockets_are_accepted(tmp_path: Path):
    for i, sock in enumerate(("unix:///var/run/docker.sock", "unix:///run/docker.sock")):
        sub = tmp_path / f"case{i}"
        sub.mkdir()
        _, log, env = _setup(sub, uid = 1000)
        env["DOCKER_HOST"] = sock
        res = _run(env)
        assert "not the system daemon" not in res.stderr, sock


def test_docker_desktop_on_macos_says_cpu_only(tmp_path: Path):
    _, log, env = _setup(tmp_path, desktop = True, driver = False, uid = 1000)
    _stub(tmp_path / "bin" / "uname", 'if [ "$1" = -s ]; then echo Darwin; else echo arm64; fi\n')
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "no NVIDIA GPU" in res.stdout and "UNSLOTH_ALLOW_CPU=1" in res.stdout
    assert "GPU support comes with it" not in res.stdout


def test_the_old_driver_message_names_the_host_platform_even_without_verification(tmp_path: Path):
    _, _, env = _setup(tmp_path, driver_version = "550.54.15")
    _stub(tmp_path / "bin" / "uname", 'if [ "$1" = -s ]; then echo Linux; else echo aarch64; fi\n')
    res = _run(env, extra_env = {"UNSLOTH_TOOLKIT_VERIFY": "0"})
    assert res.returncode == 3
    assert "--platform linux/arm64" in res.stderr


def test_a_configured_host_only_verifies(tmp_path: Path):
    _, log, env = _setup(tmp_path, configured = True)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "already lists the nvidia runtime" in res.stdout
    calls = _calls(log)
    assert not any(c.startswith(("apt-get", "nvidia-ctk", "systemctl", "service")) for c in calls)
    assert "docker run --rm --gpus all --platform linux/amd64 ubuntu:24.04 nvidia-smi -L" in calls


def test_docker_desktop_on_wsl_needs_nothing(tmp_path: Path):
    _, log, env = _setup(tmp_path, desktop = True, driver = False, uid = 1000, wsl = True)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WSL 2 backend" in res.stdout and "nvidia.com" in res.stdout
    assert "wsl --update updates WSL itself" in res.stdout
    assert [c for c in _calls(log) if not c.startswith("docker ")] == []


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
    assert "wsl --update updates WSL itself" in res.stderr


def test_verification_can_be_skipped(tmp_path: Path):
    _, log, env = _setup(tmp_path, verify_ok = False)
    res = _run(env, extra_env = {"UNSLOTH_TOOLKIT_VERIFY": "0"})
    assert res.returncode == 0, res.stdout + res.stderr
    assert not any(c.startswith("docker run") for c in _calls(log))


def test_a_non_root_run_of_the_file_re_executes_itself_under_sudo(tmp_path: Path):
    _, log, env = _setup(tmp_path, uid = 1000)
    res = _run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert [c for c in _calls(log) if not c.startswith("docker ")] == [f"sudo -E bash {INSTALLER}"]


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
    assert "sudo -E bash" in res.stderr
    assert [c for c in _calls(log) if not c.startswith("docker ")] == []


def _run_sh_env(
    tmp_path: Path,
    *,
    nvidia_runtime: bool,
    docker_down: bool = False,
) -> tuple[Path, Path, dict]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    argv = tmp_path / "argv"
    rec = f'echo "$(basename "$0") $*" >> {log}\n'
    runtimes = (
        "io.containerd.runc.v2 nvidia runc" if nvidia_runtime else "io.containerd.runc.v2 runc"
    )
    info = (
        'echo "permission denied while trying to connect to the Docker daemon socket" >&2; exit 1'
        if docker_down
        else f'echo " Runtimes: {runtimes}"; exit 0'
    )
    _stub(
        bindir / "docker",
        rec + f'if [ "$1" = info ]; then {info}; fi\n' f'printf "%s\\n" "$@" > {argv}\nexit 0\n',
    )
    _stub(bindir / "nvidia-smi", 'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"\n')
    _stub(bindir / "sudo", rec + "exit 0\n")
    # never root here: as uid 0 run.sh would execute the REAL installer against the host
    _stub(bindir / "id", "echo 1000\n")
    _stub(bindir / "getent", "exit 2\n")
    dev_root = tmp_path / "root"
    (dev_root / "dev").mkdir(parents = True)
    (dev_root / "dev" / "nvidiactl").write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = _isolated_path(tmp_path, bindir)
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
    assert f"sudo -E bash {INSTALLER}" in _calls(log)
    assert "--gpus\nall\n" in argv.read_text(encoding = "utf-8")


def test_run_sh_still_runs_the_container_when_the_installer_fails(tmp_path: Path):
    """Exit 3 means the toolkit works and only the driver is old; a cancelled sudo is
    exit 1. Neither may stop the docker run the user asked for."""
    log, argv, env = _run_sh_env(tmp_path, nvidia_runtime = False)
    _stub(tmp_path / "bin" / "sudo", f'echo "$(basename "$0") $*" >> {log}\nexit 3\n')
    env["UNSLOTH_INSTALL_TOOLKIT"] = "1"
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"sudo -E bash {INSTALLER}" in _calls(log)
    assert argv.exists(), "docker run never happened"


def test_run_sh_without_a_terminal_prints_the_one_liner_and_still_runs(tmp_path: Path):
    log, argv, env = _run_sh_env(tmp_path, nvidia_runtime = False)
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "-o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh" in res.stderr
    assert not any(c.startswith("sudo") for c in _calls(log))
    assert argv.exists()


def test_run_sh_does_not_offer_an_install_when_docker_itself_is_unreachable(tmp_path: Path):
    log, argv, env = _run_sh_env(tmp_path, nvidia_runtime = False, docker_down = True)
    env["UNSLOTH_INSTALL_TOOLKIT"] = "1"
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "permission denied" in res.stderr and "docker group" in res.stderr
    assert "Container Toolkit is not set up" not in res.stderr
    assert not any(c.startswith("sudo") for c in _calls(log))
    assert argv.exists()


def test_run_sh_uses_no_scratch_file_for_the_daemon_probe():
    text = RUN_SH.read_text(encoding = "utf-8")
    assert "unsloth-docker-info" not in text and "mktemp" not in text
    assert 'DOCKER_INFO="$(docker info 2>&1)"' in text


def test_run_sh_is_quiet_when_the_runtime_is_present(tmp_path: Path):
    _, _, env = _run_sh_env(tmp_path, nvidia_runtime = True)
    res = _run_run_sh(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Container Toolkit" not in res.stderr


def test_the_docs_point_at_the_installer():
    for doc in (REPO_ROOT / "docker" / "DOCKERHUB.md", REPO_ROOT / "README.md"):
        text = doc.read_text(encoding = "utf-8")
        assert (
            "install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh"
            in text
        ), doc
        assert "| sudo" not in text, "a pipe into bash masks a failed download"

    assert "Docker Desktop" in (REPO_ROOT / "docker" / "DOCKERHUB.md").read_text(encoding = "utf-8")
