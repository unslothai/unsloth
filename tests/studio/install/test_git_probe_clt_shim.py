"""install_python_stack's git probe never runs Apple's Command Line Tools shim.

Without the CLT, Apple Silicon's /usr/bin/git is a shim, and running it (even `git --version`)
raises the "install the command line developer tools" dialog. install.sh's _has_working_git
answers from the path there. The Python probe ran `git --version` unconditionally, which was
harmless while only the triton step (skipped on macOS) asked. Since the pinned Diffusers main
build became the default, every macOS install asks, so a Mac without the CLT got the dialog in
the middle of the install.

Every call the probe makes goes through a stand-in `subprocess.run` here, so each row records
whether git itself would have been executed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STACK_PATH = REPO_ROOT / "studio" / "install_python_stack.py"


def _load():
    sys.path.insert(0, str(REPO_ROOT / "studio"))
    try:
        spec = importlib.util.spec_from_file_location("studio_stack_git_probe", STACK_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT / "studio"):
            sys.path.pop(0)
    return module


stack = _load()


def _probe(monkeypatch, *, macos, machine, arm64_sysctl, git, toolchain):
    """Run _has_working_git on a described host; return (answer, commands run)."""
    ran: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        ran.append(list(cmd))
        name = pathlib.PurePosixPath(cmd[0]).name
        if name == "sysctl":
            return types.SimpleNamespace(returncode = 0, stdout = arm64_sysctl + "\n")
        if name == "xcode-select":
            return types.SimpleNamespace(returncode = 0 if toolchain else 2, stdout = "")
        if name == "git":
            return types.SimpleNamespace(returncode = 0, stdout = "")
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(stack, "IS_MACOS", macos)
    monkeypatch.setattr(stack.platform, "machine", lambda: machine)
    monkeypatch.setattr(stack.shutil, "which", lambda name: git if name == "git" else None)
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    return stack._has_working_git(), ran


def _ran_git(ran):
    return any(pathlib.PurePosixPath(cmd[0]).name == "git" for cmd in ran)


@pytest.mark.parametrize(
    "machine, arm64_sysctl",
    [
        pytest.param("arm64", "1", id = "apple-silicon"),
        pytest.param("x86_64", "1", id = "apple-silicon-under-rosetta"),
    ],
)
def test_the_clt_shim_is_not_run_without_a_toolchain(monkeypatch, machine, arm64_sysctl):
    answer, ran = _probe(
        monkeypatch,
        macos = True,
        machine = machine,
        arm64_sysctl = arm64_sysctl,
        git = "/usr/bin/git",
        toolchain = False,
    )
    assert answer is False
    assert not _ran_git(ran), f"the probe ran Apple's git shim, which raises the dialog: {ran}"
    assert ["xcode-select", "-p"] in ran


@pytest.mark.parametrize(
    "macos, machine, arm64_sysctl, git, toolchain",
    [
        pytest.param(True, "arm64", "1", "/usr/bin/git", True, id = "shim-with-clt-installed"),
        pytest.param(True, "arm64", "1", "/opt/homebrew/bin/git", False, id = "homebrew-git"),
        pytest.param(True, "x86_64", "0", "/usr/bin/git", False, id = "intel-mac"),
        pytest.param(False, "x86_64", "0", "/usr/bin/git", False, id = "linux"),
    ],
)
def test_every_other_git_is_still_run(monkeypatch, macos, machine, arm64_sysctl, git, toolchain):
    answer, ran = _probe(
        monkeypatch,
        macos = macos,
        machine = machine,
        arm64_sysctl = arm64_sysctl,
        git = git,
        toolchain = toolchain,
    )
    assert answer is True
    assert [git, "--version"] in ran


def test_linux_never_asks_the_mac_questions(monkeypatch):
    _, ran = _probe(
        monkeypatch,
        macos = False,
        machine = "x86_64",
        arm64_sysctl = "0",
        git = "/usr/bin/git",
        toolchain = False,
    )
    assert ran == [["/usr/bin/git", "--version"]]


def test_no_git_on_path_runs_nothing(monkeypatch):
    answer, ran = _probe(
        monkeypatch,
        macos = True,
        machine = "arm64",
        arm64_sysctl = "1",
        git = None,
        toolchain = False,
    )
    assert answer is False
    assert ran == []


def test_the_shim_path_matches_install_sh():
    # Both probes have to agree on which git is the dialog shim, or one of them regresses alone.
    install_sh = (REPO_ROOT / "install.sh").read_text(encoding = "utf-8")
    assert '"${_CLT_GIT_SHIM:-/usr/bin/git}"' in install_sh
    assert stack._CLT_GIT_SHIM == "/usr/bin/git"
