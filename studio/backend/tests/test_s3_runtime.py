# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-use S3 installation never contacts a package index in unit tests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from utils import s3_runtime as runtime


@pytest.fixture
def missing_sdk(monkeypatch, tmp_path):
    state = {"sdk": None, "calls": []}
    monkeypatch.setattr(runtime, "_import_sdk", lambda: state["sdk"])
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(runtime, "hf_env_offline", lambda: False)
    monkeypatch.setenv("UNSLOTH_S3_AUTO_INSTALL", "1")
    monkeypatch.delenv("UV_OFFLINE", raising = False)
    monkeypatch.delenv("UV_NO_INDEX", raising = False)
    monkeypatch.delenv("PIP_NO_INDEX", raising = False)
    monkeypatch.setattr(
        runtime,
        "distributions",
        lambda: [SimpleNamespace(metadata = {"Name": "urllib3"}, version = "2.5.0")],
    )

    def install(command, **kwargs):
        state["calls"].append((command, kwargs))
        constraints = Path(command[command.index("--constraint") + 1]).read_text()
        assert constraints == "urllib3==2.5.0\n"
        state["sdk"] = SimpleNamespace(__version__ = "test")

    monkeypatch.setattr(runtime.subprocess, "run", install)
    return state


def test_installed_sdk_is_reused_even_offline_and_disabled(missing_sdk, monkeypatch):
    missing_sdk["sdk"] = object()
    monkeypatch.setenv("UNSLOTH_S3_AUTO_INSTALL", "0")
    monkeypatch.setattr(runtime, "hf_env_offline", lambda: True)
    assert runtime.ensure_boto3() is missing_sdk["sdk"]
    assert not missing_sdk["calls"]


@pytest.mark.parametrize("installer", ["uv", "pip"])
def test_install_targets_this_python_and_serializes_threads(missing_sdk, monkeypatch, installer):
    monkeypatch.setattr(runtime, "_find_uv", lambda: "/tools/uv" if installer == "uv" else None)
    with ThreadPoolExecutor(max_workers = 4) as pool:
        results = list(pool.map(lambda _: runtime.ensure_boto3(), range(4)))
    assert all(result is missing_sdk["sdk"] for result in results)
    assert len(missing_sdk["calls"]) == 1
    command, kwargs = missing_sdk["calls"][0]
    assert command[-1] == "boto3>=1.34.0"
    assert command[command.index("--only-binary") + 1] == ":all:"
    if installer == "uv":
        assert command[0] == "/tools/uv"
        assert command[command.index("--python") + 1] == runtime.sys.executable
        assert "--no-config" in command
    else:
        assert command[:4] == [runtime.sys.executable, "-m", "pip", "install"]
    assert kwargs["check"] is True
    assert 0 < kwargs["timeout"] <= 40


def test_installer_does_not_receive_credentials_or_resolver_overrides(missing_sdk, monkeypatch):
    for name in (
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET",
        "UV_OVERRIDE",
        "PIP_TARGET",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "must-not-reach-child")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example")
    runtime.ensure_boto3()
    env = missing_sdk["calls"][0][1]["env"]
    assert "must-not-reach-child" not in env.values()
    assert env["HTTPS_PROXY"] == "https://proxy.example"


@pytest.mark.parametrize(
    "mode", ["disabled", "offline", "UV_OFFLINE", "UV_NO_INDEX", "PIP_NO_INDEX"]
)
def test_disabled_and_offline_install_never_runs(missing_sdk, monkeypatch, mode):
    if mode == "disabled":
        monkeypatch.setenv("UNSLOTH_S3_AUTO_INSTALL", "0")
    elif mode == "offline":
        monkeypatch.setattr(runtime, "hf_env_offline", lambda: True)
    else:
        monkeypatch.setenv(mode, "1")
    with pytest.raises(runtime.S3DependencyUnavailable, match = r"unsloth\[s3\]"):
        runtime.ensure_boto3()
    assert not missing_sdk["calls"]


@pytest.mark.parametrize("failure", ["resolver", "timeout", "missing-after-install"])
def test_failed_install_preserves_guidance_without_exposing_output(
    missing_sdk, monkeypatch, failure
):
    def fail(*args, **kwargs):
        if failure == "resolver":
            raise subprocess.CalledProcessError(1, "pip", stderr = "secret-proxy-password")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("pip", 120)

    monkeypatch.setattr(runtime.subprocess, "run", fail)
    with pytest.raises(runtime.S3DependencyUnavailable) as caught:
        runtime.ensure_boto3()
    assert runtime.INSTALL_COMMAND in str(caught.value)
    assert "secret-proxy-password" not in str(caught.value)


def test_broken_installed_sdk_is_not_reinstalled(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("missing botocore", name = "botocore")

    monkeypatch.setattr(runtime.importlib, "import_module", broken)
    with pytest.raises(runtime.S3DependencyUnavailable, match = r"unsloth\[s3\]"):
        runtime.ensure_boto3()


@pytest.mark.parametrize("elapsed", [25, 41])
def test_lock_wait_counts_against_install_timeout(missing_sdk, monkeypatch, elapsed):
    clock = iter((100, 100 + elapsed))
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic = lambda: next(clock)))
    if elapsed > 40:
        with pytest.raises(runtime.S3DependencyUnavailable):
            runtime.ensure_boto3()
        assert not missing_sdk["calls"]
    else:
        runtime.ensure_boto3()
        assert missing_sdk["calls"][0][1]["timeout"] == 15


@pytest.mark.parametrize("name", ["uv", "uv.exe"])
def test_uv_is_found_off_path(monkeypatch, tmp_path, name):
    from utils import mlx_repair

    uv = tmp_path / ".local" / "bin" / name
    uv.parent.mkdir(parents = True)
    uv.write_text("")
    uv.chmod(0o755)
    monkeypatch.setattr(mlx_repair, "_UV_NAME", name)
    monkeypatch.setattr(mlx_repair.shutil, "which", lambda name: None)
    monkeypatch.setattr(mlx_repair.Path, "home", lambda: tmp_path)
    assert runtime._find_uv() == str(uv)
