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
    def __init__(
        self,
        loads,
        capability = "9 0",
    ):
        self.loads = list(loads)
        self.capability = capability
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[1] == "-c" and "get_device_capability" in cmd[2]:
            return SimpleNamespace(returncode = 0, stdout = self.capability)
        if cmd[1] == "-c":
            return SimpleNamespace(returncode = 0 if self.loads.pop(0) else 1, stdout = "")
        return SimpleNamespace(returncode = 0, stdout = "")

    @property
    def installer_calls(self):
        return [c for c in self.calls if c[1] != "-c"]


_COLAB = _env("2.11.0+cu130", "13.0")
_COLAB_XFORMERS = f"{_PT}/cu130/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"


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
        (_env("2.11.0+cu130", "13.0", platform_tag = "win_amd64"), None),
        (_env("2.11.0+cpu", ""), None),
    ],
)
def test_flash_attn_resolution(env, expected):
    assert kernel_install.resolve_wheel_url("flash_attn", env) == expected


@pytest.mark.parametrize("name", ["flash_attn", "mamba_ssm"])
@pytest.mark.parametrize("capability", ["7 5", "None"])
def test_sm80_kernels_are_skipped_below_sm80(name, capability, capsys):
    run = _Runner([], capability = capability)
    assert kernel_install.install_kernel(name, _COLAB, run = run, exists = lambda url: True) == 0
    assert run.installer_calls == []
    assert f"skipping {name}, which needs sm80 or newer" in capsys.readouterr().out


def test_causal_conv1d_still_installs_below_sm80(uv):
    uv(False)
    run = _Runner([False, True], capability = "7 5")
    assert (
        kernel_install.install_kernel("causal_conv1d", _COLAB, run = run, exists = lambda url: True)
        == 0
    )
    assert run.installer_calls[0][-1].startswith(f"{_CC1D}/causal_conv1d-1.6.1+cu13torch2.10")


def test_the_gpu_is_probed_once_for_every_sm80_kernel(capsys):
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
        kernel_install, "install_kernel", lambda name, env, dry_run = False: order.append(name) or 0
    )
    assert kernel_install.main(argv) == 0
    assert order == expected


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
    assert gcc is None and "timeout" not in kwargs
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
