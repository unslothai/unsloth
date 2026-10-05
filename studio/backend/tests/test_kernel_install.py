# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth install-kernels`: wheel-only, CUDA-matched, torch untouched."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import kernel_install  # noqa: E402

_REPO = _BACKEND.parent.parent
_PT = "https://download.pytorch.org/whl"
_CC1D = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4"


def _env(
    torch_version,
    cuda_version,
    python_tag = "cp313",
    platform_tag = "linux_x86_64",
):
    release = torch_version.split("+", 1)[0]
    return {
        "python_tag": python_tag,
        "torch_mm": ".".join(release.split(".")[:2]),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cuda_major": cuda_version.split(".", 1)[0] if cuda_version else "",
        "hip_version": "",
        "cxx11abi": "TRUE",
        "platform_tag": platform_tag,
    }


@pytest.fixture(autouse = True)
def _no_mirror(monkeypatch):
    monkeypatch.delenv("UNSLOTH_PYTORCH_MIRROR", raising = False)


@pytest.mark.parametrize(
    "torch_version, cuda_version, expected",
    [
        (
            "2.8.0+cu126",
            "12.6",
            f"{_PT}/cu126/xformers-0.0.32.post2-cp39-abi3-manylinux_2_28_x86_64.whl",
        ),
        (
            "2.9.0+cu128",
            "12.8",
            f"{_PT}/cu128/xformers-0.0.33.post1-cp39-abi3-manylinux_2_28_x86_64.whl",
        ),
        (
            "2.10.0+cu128",
            "12.8",
            f"{_PT}/cu128/xformers-0.0.34-cp39-abi3-manylinux_2_28_x86_64.whl",
        ),
        (
            "2.11.0+cu130",
            "13.0",
            f"{_PT}/cu130/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl",
        ),
        (
            "2.12.1+cu130",
            "13.0",
            f"{_PT}/cu130/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl",
        ),
        ("2.11.0+cpu", "", None),
    ],
)
def test_xformers_matches_torch_and_cuda(torch_version, cuda_version, expected):
    assert (
        kernel_install.resolve_wheel_url("xformers", _env(torch_version, cuda_version)) == expected
    )


def test_causal_conv1d_reuses_torch_2_10_wheel_on_2_11():
    url = kernel_install.resolve_wheel_url("causal_conv1d", _env("2.11.0+cu130", "13.0"))
    assert (
        url == f"{_CC1D}/causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    )


@pytest.mark.parametrize(
    "env",
    [
        None,
        _env("2.11.0+cpu", ""),
        _env("2.11.0+cu130", "13.0", platform_tag = "win_amd64"),
    ],
)
def test_causal_conv1d_has_no_wheel_off_linux_cuda(env):
    assert kernel_install.resolve_wheel_url("causal_conv1d", env) is None


class _Runner:
    def __init__(self, loads):
        self.loads = list(loads)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[1] == "-c":
            return SimpleNamespace(returncode = 0 if self.loads.pop(0) else 1, stdout = "")
        return SimpleNamespace(returncode = 0, stdout = "")

    @property
    def pip_calls(self):
        return [c[3:] for c in self.calls if c[1:3] == ["-m", "pip"]]


_COLAB = _env("2.11.0+cu130", "13.0")


def test_no_published_wheel_installs_nothing(capsys):
    run = _Runner([])
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: False)
        == 0
    )
    assert run.calls == []
    assert "using the torch fallback" in capsys.readouterr().out


def test_installs_the_matched_wheel_without_deps():
    run = _Runner([False, True])
    assert kernel_install.install_kernel("xformers", _COLAB, run = run, exists = lambda url: True) == 0
    url = f"{_PT}/cu130/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
    assert run.pip_calls == [["install", "--no-deps", "--force-reinstall", url]]


def test_working_install_is_left_alone():
    run = _Runner([True])
    assert kernel_install.install_kernel("xformers", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.pip_calls == []


def test_wheel_that_does_not_load_is_removed():
    run = _Runner([False, False])
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 1
    )
    assert run.pip_calls[-1] == ["uninstall", "-y", "causal-conv1d"]


def test_dry_run_prints_url_only(capsys):
    run = _Runner([])
    assert (
        kernel_install.install_kernel(
            "xformers", _COLAB, dry_run = True, run = run, exists = lambda url: True
        )
        == 0
    )
    assert run.calls == []
    assert (
        capsys.readouterr()
        .out.strip()
        .endswith("xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl")
    )


def test_console_entry_works_without_typer(tmp_path):
    (tmp_path / "typer").mkdir()
    (tmp_path / "typer" / "__init__.py").write_text(
        "raise ImportError('typer must not be imported')\n"
    )
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(tmp_path), str(_REPO)]))
    result = subprocess.run(
        [sys.executable, "-m", "unsloth_cli", "install-kernels", "--help"],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: unsloth install-kernels" in result.stdout
