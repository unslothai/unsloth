# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth install-kernels`: wheel-only, CUDA-matched, torch untouched."""

from __future__ import annotations

import os
import subprocess
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import kernel_install  # noqa: E402

_REAL_LOCK_PATH = kernel_install._kernel_lock_path

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
    # Forces the mamba_ssm decision below sm80; an inherited value would override the mocked Triton.
    monkeypatch.delenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", raising = False)


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
        _env("2.9.0+cu130", "13.0", platform_tag = "win_amd64"),
    ],
)
def test_causal_conv1d_has_no_wheel_off_linux_cuda(env):
    assert kernel_install.resolve_wheel_url("causal_conv1d", env) is None


class _Runner:
    def __init__(
        self,
        loads,
        capability = "9 0",
        triton = None,
    ):
        self.loads = list(loads)
        self.capability = capability
        self.triton = triton
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[1] == "-c" and "get_device_capability" in cmd[2]:
            return SimpleNamespace(returncode = 0, stdout = self.capability)
        if cmd[1] == "-c" and "import triton" in cmd[2]:
            if self.triton is None:
                return SimpleNamespace(returncode = 1, stdout = "")
            return SimpleNamespace(returncode = 0, stdout = self.triton + "\n")
        if cmd[1] == "-c":
            return SimpleNamespace(returncode = 0 if self.loads.pop(0) else 1, stdout = "")
        return SimpleNamespace(returncode = 0, stdout = "")

    @property
    def installer_calls(self):
        return [c for c in self.calls if c[1] != "-c"]


_COLAB = _env("2.11.0+cu130", "13.0")
_COLAB_XFORMERS = f"{_PT}/cu130/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"


@pytest.fixture(autouse = True)
def _nothing_installed_and_online(monkeypatch, tmp_path):
    monkeypatch.setattr(
        kernel_install, "_kernel_lock_path", lambda name: str(tmp_path / f"{name}.lock")
    )
    # Every copy of the module (and any child process) uses the test's folder too.
    monkeypatch.setenv("UNSLOTH_KERNEL_LOCK_DIR", str(tmp_path / "locks"))
    # Tests that need an installed distribution say so; the host's own packages and offline
    # switches must not decide them.
    monkeypatch.setattr(kernel_install, "_installed_version", lambda distribution: None)
    for name in ("UV_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising = False)


@pytest.fixture
def uv(monkeypatch):
    def use(available, outside_venv = False):
        monkeypatch.setattr(
            kernel_install.shutil, "which", lambda name: "/usr/bin/uv" if available else None
        )
        monkeypatch.setattr(kernel_install, "_outside_venv", lambda: outside_venv)

    return use


def test_no_published_wheel_installs_nothing(capsys):
    run = _Runner([])
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: False)
        == 0
    )
    assert run.calls == []
    assert "using the torch fallback" in capsys.readouterr().out


def test_pip_reinstalls_the_matched_wheel_without_deps(uv):
    uv(False)
    run = _Runner([False, True])
    assert kernel_install.install_kernel("xformers", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == [
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", _COLAB_XFORMERS]
    ]


def test_uv_reinstalls_into_the_system_interpreter_on_colab(uv):
    # The check failed, so the installed copy is broken: reinstall even at the same version.
    uv(True, outside_venv = True)
    run = _Runner([False, True])
    assert kernel_install.install_kernel("xformers", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == [
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--python",
            sys.executable,
            "--no-deps",
            "--reinstall",
            _COLAB_XFORMERS,
        ]
    ]


def test_working_install_is_left_alone():
    run = _Runner([True])
    assert kernel_install.install_kernel("xformers", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == []


@pytest.mark.parametrize(
    "has_uv, uninstall",
    [
        (False, [sys.executable, "-m", "pip", "uninstall", "-y", "causal-conv1d"]),
        (True, ["uv", "pip", "uninstall", "--python", sys.executable, "causal-conv1d"]),
    ],
)
def test_wheel_that_does_not_load_is_removed(uv, has_uv, uninstall):
    uv(has_uv)
    run = _Runner([False, False])
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 1
    )
    assert run.installer_calls[-1] == uninstall


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


def test_host_package_run_with_dash_m_is_not_intercepted(tmp_path):
    host = tmp_path / "hostapp"
    host.mkdir()
    (host / "__init__.py").write_text("import unsloth_cli\n")
    (host / "__main__.py").write_text("print('host main ran')\n")
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(tmp_path), str(_REPO)]))
    result = subprocess.run(
        [sys.executable, "-m", "hostapp", "install-kernels", "--help"],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr
    assert "host main ran" in result.stdout
    assert "usage: unsloth install-kernels" not in result.stdout


def test_mirror_credentials_are_not_printed(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_PYTORCH_MIRROR", "https://user:secret@mirror.example/whl?token=abc")
    for dry_run in (True, False):
        kernel_install.install_kernel(
            "xformers",
            _COLAB,
            dry_run = dry_run,
            run = _Runner([False, True]),
            exists = lambda url: True,
        )
    out = capsys.readouterr().out
    assert "mirror.example/whl/cu130/xformers-0.0.35" in out
    assert "secret" not in out and "token=abc" not in out


def test_failed_uninstall_is_not_reported_as_removed(uv, capsys):
    uv(False)

    def run(cmd, **kwargs):
        return SimpleNamespace(
            returncode = 1 if cmd[1] in ("-c",) or "uninstall" in cmd else 0, stdout = ""
        )

    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 1
    )
    out = capsys.readouterr().out
    assert "removed it" not in out and "pip uninstall causal-conv1d" in out


def test_unreachable_probe_does_not_log_mirror_credentials(caplog):
    from utils.wheel_utils import url_exists

    with caplog.at_level("WARNING"):
        assert url_exists("https://user:secret@mirror.invalid/whl/x.whl") is None
    assert "mirror.invalid/whl/x.whl" in caplog.text
    assert "secret" not in caplog.text


def test_console_entry_drops_an_unwritable_ssl_keylog_file(tmp_path):
    env = dict(
        os.environ,
        PYTHONPATH = str(_REPO),
        SSLKEYLOGFILE = str(tmp_path / "missing" / "keys.log"),
    )
    result = subprocess.run(
        [sys.executable, "-m", "unsloth_cli", "install-kernels", "--help"],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    assert "ignoring SSLKEYLOGFILE" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "env, expected",
    [
        (
            _env("2.11.0+cu130", "13.0"),
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/"
            "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        ),
        (
            _env("2.13.0+cu130", "13.0"),
            "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
            "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
        ),
        (
            _env("2.11.0+cu130", "13.0", platform_tag = "win_amd64"),
            "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
            "flash_attn-2.8.4+cu13torch2.11cxx11abiTRUE-cp313-cp313-win_amd64.whl",
        ),
        (
            _env("2.10.0+cu130", "13.0", platform_tag = "win_amd64"),
            "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
            "flash_attn-2.8.4+cu13torch2.10cxx11abiTRUE-cp313-cp313-win_amd64.whl",
        ),
        (
            _env("2.12.0+cu130", "13.0", platform_tag = "win_amd64"),
            "https://github.com/unslothai/unsloth/releases/download/prebuilt-wheels-cu13/"
            "flash_attn-2.8.4+cu13torch2.12cxx11abiTRUE-cp313-cp313-win_amd64.whl",
        ),
        (_env("2.9.0+cu130", "13.0", platform_tag = "win_amd64"), None),
        (_env("2.11.0+cpu", ""), None),
    ],
)
def test_flash_attn_resolution(env, expected):
    assert kernel_install.resolve_wheel_url("flash_attn", env) == expected


@pytest.mark.parametrize("name", ["flash_attn", "mamba_ssm"])
@pytest.mark.parametrize("capability", ["7 5", "None"])
def test_sm80_kernels_are_skipped_below_sm80(name, capability, capsys, monkeypatch):
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: (3, 3))
    run = _Runner([], capability = capability)
    assert kernel_install.install_kernel(name, _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == []
    assert f"skipping {name}, which needs sm80" in capsys.readouterr().out


# unsloth_zoo keeps mamba_ssm's fast path on sm75 with Triton 3.4+ (torch 2.8+).


@pytest.mark.parametrize("triton", [(3, 4), (3, 6), (4, 0)])
def test_mamba_ssm_installs_on_sm75_with_a_new_triton(uv, monkeypatch, triton):
    uv(False)
    monkeypatch.delenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", raising = False)
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: triton)
    run = _Runner([False, True], capability = "7 5")
    assert kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls[0][-1].endswith(
        "mamba_ssm-2.3.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    )


@pytest.mark.parametrize("triton", [(3, 3), (3, 2), None])
def test_mamba_ssm_is_skipped_on_sm75_with_an_old_triton(monkeypatch, capsys, triton):
    monkeypatch.delenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", raising = False)
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: triton)
    run = _Runner([], capability = "7 5")
    assert kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == []
    out = capsys.readouterr().out
    assert "sm75 with Triton 3.4+" in out
    assert ("missing" if triton is None else "%d.%d" % triton) in out


@pytest.mark.parametrize(
    "triton, installs",
    [("3.4.0", True), ("3.6.0+git1a2b3c", True), ("3.3.1", False), (None, False)],
)
def test_triton_is_read_from_the_module(uv, monkeypatch, triton, installs):
    """pytorch-triton and other providers ship the `triton` import under another
    distribution name, so the version comes from `triton.__version__`."""
    uv(False)
    monkeypatch.delenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", raising = False)
    run = _Runner([False, True], capability = "7 5", triton = triton)
    assert kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True) == 0
    assert bool(run.installer_calls) is installs


def test_mamba_ssm_is_skipped_below_sm75_whatever_the_triton(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", raising = False)
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: (3, 6))
    run = _Runner([], capability = "7 0")
    assert kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == []
    assert "the best GPU is sm70" in capsys.readouterr().out


@pytest.mark.parametrize(
    "forced, capability, installs",
    [
        ("1", "7 0", True),
        ("1", "7 5", True),
        ("0", "7 5", False),
    ],
)
def test_mamba_ssm_follows_the_unsloth_zoo_override(uv, monkeypatch, forced, capability, installs):
    uv(False)
    monkeypatch.setenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", forced)
    monkeypatch.setattr(
        kernel_install, "_triton_version", lambda run: (3, 6) if forced == "0" else (3, 3)
    )
    run = _Runner([False, True], capability = capability)
    assert kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True) == 0
    assert bool(run.installer_calls) is installs


def test_the_override_skip_message_reads_as_a_sentence(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", "0")
    run = _Runner([], capability = "7 5")
    kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True)
    assert (
        "Unsloth: skipping mamba_ssm, which UNSLOTH_MAMBA_PRE_AMPERE_FAST=0 turns off "
        "(the best GPU is sm75)." in capsys.readouterr().out
    )


def test_flash_attn_stays_sm80_only_with_a_new_triton(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MAMBA_PRE_AMPERE_FAST", "1")
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: (3, 6))
    run = _Runner([], capability = "7 5")
    assert (
        kernel_install.install_kernel("flash_attn", _COLAB, run = run, exists = lambda url: True) == 0
    )
    assert run.installer_calls == []


def test_causal_conv1d_still_installs_below_sm80(uv):
    uv(False)
    run = _Runner([False, True], capability = "7 5")
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 0
    )
    assert run.installer_calls[0][-1].startswith(f"{_CC1D}/causal_conv1d-1.6.1+cu13torch2.10")


@pytest.mark.parametrize(
    "capability, installs",
    [("None", False), ("7 0", False), ("9 0", True)],
)
def test_mamba_ssm_needs_a_visible_sm80_gpu(uv, monkeypatch, capability, installs):
    # No visible GPU or a pre-Ampere one skips. Linux, so Windows' triton-windows gate (its own
    # test) does not answer first.
    uv(False)
    monkeypatch.setattr(kernel_install.sys, "platform", "linux")
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: (3, 3))
    run = _Runner([False, True], capability = capability)
    assert kernel_install.install_kernel("mamba_ssm", _COLAB, run = run, exists = lambda url: True) == 0
    assert bool(run.installer_calls) is installs


def test_the_gpu_is_probed_once_for_every_sm80_kernel(capsys, monkeypatch):
    monkeypatch.setattr(kernel_install, "_triton_version", lambda run: (3, 3))
    run = _Runner([], capability = "7 5")
    for name in ("flash_attn", "mamba_ssm"):
        kernel_install.install_kernel(name, _COLAB, run = run, exists = lambda url: True)
    assert sum("get_device_capability" in " ".join(c) for c in run.calls) == 1


def test_flash_attn_installs_on_sm80_and_newer(uv):
    uv(False)
    run = _Runner([False, True], capability = "8 0")
    assert (
        kernel_install.install_kernel("flash_attn", _COLAB, run = run, exists = lambda url: True) == 0
    )
    assert run.installer_calls[0][-1].endswith(
        "flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    )


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], ["xformers", "flash_attn", "causal_conv1d", "mamba_ssm"]),
        (["mamba_ssm", "causal_conv1d", "mamba_ssm"], ["causal_conv1d", "mamba_ssm"]),
    ],
)
def test_main_installs_all_by_default_in_dependency_order(monkeypatch, argv, expected):
    order = []
    monkeypatch.setattr(kernel_install, "probe_torch_wheel_env", lambda **kwargs: _COLAB)
    monkeypatch.setattr(
        kernel_install, "install_kernel", lambda name, env, **kwargs: order.append(name) or 0
    )
    assert kernel_install.main(argv) == 0
    assert order == expected


# A GPU-less image build (Docker, CI) provisions for the GPU it will run on only when asked to.


@pytest.mark.parametrize("opt_in", ["flag", "env"])
def test_install_kernels_without_gpu_installs_mamba_ssm_headless(uv, monkeypatch, opt_in):
    uv(False)
    monkeypatch.setattr(kernel_install.sys, "platform", "linux")
    monkeypatch.setattr(kernel_install, "probe_torch_wheel_env", lambda **kwargs: _COLAB)
    monkeypatch.setattr(kernel_install, "url_exists", lambda url: True)
    run = _Runner([False, True], capability = "None")
    real = kernel_install.install_kernel
    monkeypatch.setattr(
        kernel_install,
        "install_kernel",
        lambda name, env, **kwargs: real(name, env, run = run, exists = lambda url: True, **kwargs),
    )
    argv = ["mamba_ssm"]
    monkeypatch.delenv("UNSLOTH_INSTALL_KERNELS_WITHOUT_GPU", raising = False)
    if opt_in == "flag":
        argv.append("--without-gpu")
    else:
        monkeypatch.setenv("UNSLOTH_INSTALL_KERNELS_WITHOUT_GPU", "1")
    assert kernel_install.main(argv) == 0
    assert run.installer_calls[0][-1].endswith(
        "mamba_ssm-2.3.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    )


@pytest.mark.parametrize("strict, code", [(False, 0), (True, 2)])
def test_a_requested_kernel_skipped_without_a_gpu_is_listed_and_fails_only_with_strict(
    monkeypatch, capsys, strict, code
):
    # 9f5b13325 and 4a6429a65 exit 0 here, and `set -e` install scripts rely on it.
    monkeypatch.delenv("UNSLOTH_INSTALL_KERNELS_WITHOUT_GPU", raising = False)
    monkeypatch.setattr(kernel_install, "probe_torch_wheel_env", lambda **kwargs: _COLAB)
    run = _Runner([], capability = "None")
    real = kernel_install.install_kernel
    monkeypatch.setattr(
        kernel_install,
        "install_kernel",
        lambda name, env, **kwargs: real(name, env, run = run, exists = lambda url: True, **kwargs),
    )
    assert kernel_install.main(["mamba_ssm"] + (["--strict"] if strict else [])) == code
    assert run.installer_calls == []
    out = capsys.readouterr().out
    assert "--without-gpu" in out
    assert "not installed on this machine: mamba_ssm" in out


@pytest.mark.parametrize(
    "name, env, capability",
    [
        # A ROCm source build: no CUDA wheel resolves.
        ("causal_conv1d", None, "9 0"),
        # A GPU-hidden builder with an earlier install.
        ("mamba_ssm", _COLAB, "None"),
        ("flash_attn", _COLAB, "None"),
    ],
)
def test_an_already_working_kernel_is_success_before_any_skip(
    monkeypatch, capsys, name, env, capability
):
    monkeypatch.setattr(
        kernel_install,
        "_installed_version",
        lambda distribution: "1.0" if distribution == kernel_install.KERNELS[name][0] else None,
    )
    monkeypatch.setattr(kernel_install, "probe_torch_wheel_env", lambda **kwargs: env)
    run = _Runner([True], capability = capability)
    real = kernel_install.install_kernel
    monkeypatch.setattr(
        kernel_install,
        "install_kernel",
        lambda n, e, **kwargs: real(n, e, run = run, exists = lambda url: True, **kwargs),
    )
    assert kernel_install.main([name, "--strict"]) == 0
    assert run.installer_calls == []
    assert not any("get_device_capability" in " ".join(c) for c in run.calls)
    out = capsys.readouterr().out
    assert f"{name} already installed and loads" in out
    assert "not installed on this machine" not in out


def test_an_installed_kernel_that_does_not_load_still_installs(uv, monkeypatch):
    uv(False)
    run = _Runner([False, False, True], capability = "9 0")
    monkeypatch.setattr(kernel_install, "_installed_version", lambda distribution: "1.0")
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 0
    )
    assert run.installer_calls


@pytest.mark.parametrize("variable", ["UV_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_the_cli_installs_nothing_offline(monkeypatch, capsys, variable):
    monkeypatch.setenv(variable, "1")
    run = _Runner([], capability = "9 0")
    url_checks = []
    assert (
        kernel_install.install_kernel(
            "causal_conv1d", _COLAB, run = run, exists = lambda url: url_checks.append(url)
        )
        == 0
    )
    assert run.calls == [] and url_checks == []
    assert "offline" in capsys.readouterr().out


def test_offline_an_already_working_kernel_still_succeeds(monkeypatch, capsys):
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setattr(kernel_install, "_installed_version", lambda distribution: "1.0")
    run = _Runner([True], capability = "9 0")
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 0
    )
    assert "already installed and loads" in capsys.readouterr().out


def test_the_default_run_lists_skipped_kernels_and_keeps_its_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(kernel_install, "probe_torch_wheel_env", lambda **kwargs: _COLAB)

    def _install(
        name,
        env,
        skipped = None,
        **kwargs,
    ):
        if name == "mamba_ssm":
            skipped.append(name)
        return 0

    monkeypatch.setattr(kernel_install, "install_kernel", _install)
    assert kernel_install.main([]) == 0
    assert "not installed on this machine: mamba_ssm" in capsys.readouterr().out


def test_flash_attn_without_a_wheel_never_probes_the_gpu():
    run = _Runner([])
    assert kernel_install.install_kernel("flash_attn", None, run = run, exists = lambda url: True) == 0
    assert run.calls == []


def test_capability_probe_timeout_counts_as_no_gpu():
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    assert kernel_install._gpu_capability(run) is None


def test_capability_probe_takes_the_best_visible_gpu():
    seen = {}

    def run(cmd, **kwargs):
        seen["check"], seen["timeout"] = cmd[2], kwargs.get("timeout")
        return SimpleNamespace(returncode = 0, stdout = "9 0\n")

    assert kernel_install._gpu_capability(run) == (9, 0)
    assert "max(torch.cuda.get_device_capability(i)" in seen["check"] and seen["timeout"]


def test_pinned_kernels_match_the_wheel_utils_pins():
    from utils import wheel_utils

    cc1d, mamba = kernel_install.CAUSAL_CONV1D, kernel_install.MAMBA_SSM
    assert (cc1d.package_version, cc1d.release_tag, cc1d.release_base_url) == (
        wheel_utils.CAUSAL_CONV1D_PACKAGE_VERSION,
        wheel_utils.CAUSAL_CONV1D_RELEASE_TAG,
        wheel_utils.CAUSAL_CONV1D_RELEASE_BASE_URL,
    )
    assert (mamba.package_version, mamba.release_tag, mamba.release_base_url) == (
        wheel_utils.MAMBA_SSM_PACKAGE_VERSION,
        wheel_utils.MAMBA_SSM_RELEASE_TAG,
        wheel_utils.MAMBA_SSM_RELEASE_BASE_URL,
    )
    assert (cc1d.import_name, cc1d.pypi_name, mamba.import_name, mamba.pypi_name) == (
        "causal_conv1d",
        "causal-conv1d",
        "mamba_ssm",
        "mamba-ssm",
    )
    url = cc1d.wheel_url(_env("2.10.0+cu128", "12.8"))
    assert url.startswith(f"{_CC1D}/causal_conv1d-1.6.1+cu12torch2.10")
    assert kernel_install.resolve_wheel_url("causal_conv1d", _env("2.10.0+cu128", "12.8")) == url


def _ok(code):
    return SimpleNamespace(returncode = code, stdout = f"out{code}")


@pytest.mark.parametrize(
    "attempts, verified, outcome, failed",
    [
        ([("uv", 0)], True, "installed", []),
        ([("uv", 1), ("pip", 0)], True, "installed", ["uv"]),
        ([("uv", 0)], False, "rejected", []),
        ([("uv", 1), ("pip", 1)], True, "failed", ["uv", "pip"]),
    ],
)
def test_install_prebuilt_outcomes(attempts, verified, outcome, failed):
    seen, failures, verifies = {}, [], []

    def install(url, **kwargs):
        seen["url"], seen["kwargs"] = url, kwargs
        return [(installer, _ok(code)) for installer, code in attempts]

    result = kernel_install.install_prebuilt(
        "https://example.invalid/k.whl",
        install = install,
        verify = lambda: verifies.append(1) or verified,
        on_failed = lambda installer, result: failures.append(installer),
        use_uv = True,
    )
    assert result == outcome
    assert failures == failed
    assert len(verifies) == (outcome != "failed")
    # Only what the caller passed is forwarded, so each caller's installer flags stay its own.
    assert seen == {
        "url": "https://example.invalid/k.whl",
        "kwargs": {"python_executable": sys.executable, "use_uv": True},
    }


@pytest.mark.parametrize(
    "use_uv, is_hip, reinstall, expected",
    [
        (
            True,
            False,
            False,
            ["uv", "pip", "install", "--python", "PY", "--no-build-isolation", "--no-deps", "k==1"],
        ),
        (
            True,
            True,
            True,
            [
                "uv",
                "pip",
                "install",
                "--python",
                "PY",
                "--no-build-isolation",
                "--no-deps",
                "--reinstall",
                "--no-cache",
                "k==1",
            ],
        ),
        (
            False,
            False,
            False,
            [
                "PY",
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "--no-cache-dir",
                "k==1",
            ],
        ),
        (
            False,
            True,
            True,
            [
                "PY",
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "--no-cache-dir",
                "--force-reinstall",
                "k==1",
            ],
        ),
    ],
)
def test_source_build_command(use_uv, is_hip, reinstall, expected):
    cmd = kernel_install.source_build_command(
        "k==1", use_uv = use_uv, is_hip = is_hip, reinstall = reinstall
    )
    assert cmd == [sys.executable if part == "PY" else part for part in expected]


def test_source_build_run_kwargs(monkeypatch):
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)
    kwargs, gcc = kernel_install.source_build_run_kwargs(
        is_hip = False, gcc_install_dir = lambda: pytest.fail("non-HIP never looks for gcc")
    )
    # CUDA builds get their own, far looser deadline (a slow aarch64 build must not be aborted).
    assert gcc is None and kwargs["timeout"] == kernel_install.SOURCE_BUILD_TIMEOUT_CUDA
    assert kwargs["encoding"] == "utf-8" and kwargs["stderr"] == subprocess.STDOUT

    kwargs, gcc = kernel_install.source_build_run_kwargs(
        is_hip = True, gcc_install_dir = lambda: "/usr/lib/gcc/x86_64-linux-gnu/13"
    )
    assert gcc == "/usr/lib/gcc/x86_64-linux-gnu/13" and kwargs["timeout"] == 1800
    assert kwargs["env"]["HIPCC_COMPILE_FLAGS_APPEND"] == f"--gcc-install-dir={gcc}"
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"

    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "--gcc-install-dir=/mine")
    kwargs, gcc = kernel_install.source_build_run_kwargs(
        is_hip = True, gcc_install_dir = lambda: pytest.fail("an explicit dir is respected")
    )
    assert gcc is None and kwargs["env"]["HIPCC_COMPILE_FLAGS_APPEND"] == "--gcc-install-dir=/mine"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 7200),
        ("", 7200),
        ("10800", 10800),
        ("3600.5", 3600),
        ("0", 7200),
        ("-5", 7200),
        ("x", 7200),
    ],
)
def test_the_cuda_source_build_timeout_is_7200_unless_overridden(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("UNSLOTH_SOURCE_BUILD_TIMEOUT", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_SOURCE_BUILD_TIMEOUT", raw)
    assert kernel_install._cuda_source_build_timeout() == expected


def test_hip_keeps_1800_and_the_lock_wait_outlasts_both_deadlines():
    assert kernel_install.SOURCE_BUILD_TIMEOUT == 1800
    assert kernel_install.SOURCE_BUILD_TIMEOUT_CUDA >= 1800
    assert kernel_install.KERNEL_LOCK_WAIT > kernel_install.SOURCE_BUILD_TIMEOUT
    assert kernel_install.KERNEL_LOCK_WAIT > kernel_install.SOURCE_BUILD_TIMEOUT_CUDA


@pytest.mark.parametrize("raw, cuda", [(None, 7200), ("14400", 14400), ("600", 600)])
def test_the_env_override_is_read_at_import_and_the_lock_wait_follows(raw, cuda):
    import json as _json

    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_SOURCE_BUILD_TIMEOUT"}
    if raw is not None:
        env["UNSLOTH_SOURCE_BUILD_TIMEOUT"] = raw
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); from utils import kernel_install as k; "
        "print(json.dumps([k.SOURCE_BUILD_TIMEOUT, k.SOURCE_BUILD_TIMEOUT_CUDA, k.KERNEL_LOCK_WAIT]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(_BACKEND)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
        check = True,
    ).stdout
    hip, cuda_seen, lock_wait = _json.loads(out.strip().splitlines()[-1])
    assert (hip, cuda_seen) == (1800, cuda)
    assert lock_wait == max(1800, cuda) + 600
    assert lock_wait > hip and lock_wait > cuda_seen


def test_a_dry_run_prints_the_url_even_offline(monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run = _Runner([])
    assert (
        kernel_install.install_kernel(
            "xformers", _COLAB, dry_run = True, run = run, exists = lambda url: True
        )
        == 0
    )
    assert run.calls == []
    out = capsys.readouterr().out
    assert "offline" not in out
    assert out.strip().endswith("xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl")


@pytest.mark.parametrize(
    "use_uv, system, expected",
    [
        (True, False, ["uv", "pip", "uninstall", "--python", "PY", "k"]),
        (True, True, ["uv", "pip", "uninstall", "--system", "--python", "PY", "k"]),
        (False, True, ["PY", "-m", "pip", "uninstall", "-y", "k"]),
    ],
)
def test_uninstall_command(use_uv, system, expected):
    cmd = kernel_install.uninstall_command("k", use_uv = use_uv, uv_needs_system = system)
    assert cmd == [sys.executable if part == "PY" else part for part in expected]


_WIN = {
    "platform_tag": "win_amd64",
    "torch_mm": "2.13",
    "torch_version": "2.13.0+cu130",
    "cuda_major": "13",
    "cuda_version": "13.0",
    "cxx11abi": "TRUE",
    "python_tag": "cp313",
    "abi_tag": "cp313",
    "hip_version": "",
}


def test_windows_mamba_ssm_waits_for_triton(uv, monkeypatch, capsys):
    """mamba_ssm imports triton at load; on Windows that is triton-windows, absent from its METADATA."""
    uv(False)
    monkeypatch.setattr(kernel_install.sys, "platform", "win32")
    run = _Runner([], triton = None)
    assert kernel_install.install_kernel("mamba_ssm", _WIN, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == []
    assert "needs Triton on Windows" in capsys.readouterr().out


def test_windows_mamba_ssm_installs_with_triton(uv, monkeypatch):
    uv(False)
    monkeypatch.setattr(kernel_install.sys, "platform", "win32")
    run = _Runner([False, True], triton = "3.7.1")
    assert kernel_install.install_kernel("mamba_ssm", _WIN, run = run, exists = lambda url: True) == 0
    assert run.installer_calls[0][-1].endswith(
        "mamba_ssm-2.3.2.post1+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"
    )


@pytest.mark.parametrize(
    "capability, below",
    [((7, 5), True), ((7, 0), True), ((8, 0), False), ((12, 0), False), (None, False)],
)
def test_flash_attn_floor_is_sm80_and_no_visible_gpu_is_not_below(monkeypatch, capability, below):
    monkeypatch.setattr(kernel_install, "_gpu_capability", lambda run: capability)
    monkeypatch.setattr(kernel_install, "_torch_is_hip", lambda run: False)
    assert kernel_install.gpu_below_flash_attn_floor() is below


def test_rocm_never_takes_the_sm80_floor(monkeypatch):
    # ROCm reports a gfx version as its capability; the sm80 floor is an NVIDIA rule.
    monkeypatch.setattr(kernel_install, "_gpu_capability", lambda run: (7, 5))
    monkeypatch.setattr(kernel_install, "_torch_is_hip", lambda run: True)
    assert kernel_install.gpu_below_flash_attn_floor() is False


def test_the_hip_probe_reads_torch_version_hip():
    def run(cmd, **kwargs):
        assert "torch.version.hip" in cmd[-1]
        return SimpleNamespace(returncode = 0, stdout = "True\n")

    assert kernel_install._torch_is_hip(run) is True


def test_install_wheel_never_falls_back_to_pip_under_uv_offline(monkeypatch):
    # uv refuses the index under UV_OFFLINE; pip has no such switch and would reach it.
    from utils import wheel_utils

    monkeypatch.setattr(wheel_utils.shutil, "which", lambda name: "/usr/bin/uv")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout = "offline")

    monkeypatch.setenv("UV_OFFLINE", "1")
    attempts = wheel_utils.install_wheel(
        "https://example.invalid/k.whl", python_executable = sys.executable, use_uv = True, run = run
    )
    assert [installer for installer, _ in attempts] == ["uv"]
    assert all(cmd[0] == "uv" for cmd in calls)

    monkeypatch.delenv("UV_OFFLINE")
    calls.clear()
    attempts = wheel_utils.install_wheel(
        "https://example.invalid/k.whl", python_executable = sys.executable, use_uv = True, run = run
    )
    assert [installer for installer, _ in attempts] == ["uv", "pip"]


@pytest.mark.skipif(os.name == "nt", reason = "POSIX process groups")
def test_a_timed_out_build_stops_its_whole_process_tree(tmp_path):
    # pip / uv start nvcc / hipcc / ninja; a timeout must stop those too, not just the leader.
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "time.sleep(60)"
    )
    # A generous timeout: on a loaded host the child can take seconds to start its grandchild.
    with pytest.raises(subprocess.TimeoutExpired):
        kernel_install.run_process_tree(
            [sys.executable, "-c", script], timeout = 15, stdout = subprocess.PIPE
        )
    assert pid_file.exists(), "the build never started its grandchild within the timeout"
    grandchild = int(pid_file.read_text())
    for _ in range(100):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        # Reaped by init once killed; a zombie still answers kill(0) briefly.
        try:
            with open(f"/proc/{grandchild}/stat") as stat:
                if stat.read().split()[2] == "Z":
                    break
        except FileNotFoundError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, 9)
        pytest.fail("the build's grandchild outlived the timeout")


def test_a_test_double_runner_is_called_as_given():
    calls = []

    def run(cmd, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0)

    kernel_install.run_process_tree(["x"], run = run, timeout = 5, text = True)
    assert calls == [{"timeout": 5, "text": True}]


def test_the_kernel_lock_is_exclusive_bounded_and_beats_while_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_install, "_kernel_lock_path", lambda name: str(tmp_path / name))
    beats = []
    with kernel_install.kernel_install_lock("causal_conv1d") as first:
        assert first is True
        with kernel_install.kernel_install_lock(
            "causal_conv1d", on_wait = lambda: beats.append(1), wait_seconds = 1.2, beat_seconds = 0.4
        ) as second:
            assert second is False  # held elsewhere past the bound: install nothing
        with kernel_install.kernel_install_lock("mamba_ssm", wait_seconds = 0) as other:
            assert other is True  # one lock per kernel
    assert len(beats) >= 2
    with kernel_install.kernel_install_lock("causal_conv1d", wait_seconds = 0) as again:
        assert again is True  # released


def test_the_kernel_lock_is_per_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_KERNEL_LOCK_DIR", raising = False)
    first_env, other_env = tmp_path / "a", tmp_path / "b"
    first_env.mkdir()
    other_env.mkdir()
    monkeypatch.setattr(kernel_install.sys, "prefix", str(first_env))
    first = _REAL_LOCK_PATH("causal_conv1d")
    monkeypatch.setattr(kernel_install.sys, "prefix", str(other_env))
    assert _REAL_LOCK_PATH("causal_conv1d") != first


def test_the_lock_lives_beside_the_environment_when_writable(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_KERNEL_LOCK_DIR", raising = False)
    env = tmp_path / "venv"
    env.mkdir()
    monkeypatch.setattr(kernel_install.sys, "prefix", str(env))
    assert _REAL_LOCK_PATH("mamba_ssm") == str(env / ".unsloth-kernel-locks" / "mamba_ssm.lock")


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason = "POSIX permissions, non-root")
def test_a_read_only_environment_falls_back_to_one_temp_lock(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_KERNEL_LOCK_DIR", raising = False)
    env = tmp_path / "venv"
    env.mkdir()
    env.chmod(0o555)
    try:
        monkeypatch.setattr(kernel_install.sys, "prefix", str(env))
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "tmp"))
        (tmp_path / "tmp").mkdir()
        first = _REAL_LOCK_PATH("mamba_ssm")
        assert first.startswith(str(tmp_path / "tmp")) and first == _REAL_LOCK_PATH("mamba_ssm")
    finally:
        env.chmod(0o755)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason = "POSIX permissions, non-root")
def test_a_lock_that_cannot_be_opened_installs_nothing(monkeypatch, tmp_path):
    lock = tmp_path / "ro.lock"
    lock.write_text("")
    lock.chmod(0o444)
    monkeypatch.setattr(kernel_install, "_kernel_lock_path", lambda name: str(lock))
    with kernel_install.kernel_install_lock("causal_conv1d", wait_seconds = 0) as held:
        assert held is False


def test_a_lock_the_filesystem_cannot_take_installs_nothing(monkeypatch, tmp_path):
    import errno

    monkeypatch.setattr(kernel_install, "_kernel_lock_path", lambda name: str(tmp_path / "x.lock"))

    def _no_locks(*args):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr("fcntl.flock", _no_locks)
    with kernel_install.kernel_install_lock("causal_conv1d", wait_seconds = 0) as held:
        assert held is False


def test_the_cli_installs_under_the_studio_lock(uv, monkeypatch, capsys):
    import contextlib

    uv(False)
    taken = []

    @contextlib.contextmanager
    def _busy(name, **kwargs):
        taken.append(name)
        yield False

    monkeypatch.setattr(kernel_install, "kernel_install_lock", _busy)
    run = _Runner([], capability = "9 0")
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 0
    )
    assert taken == ["causal_conv1d"] and run.installer_calls == []
    assert "install lock" in capsys.readouterr().out
