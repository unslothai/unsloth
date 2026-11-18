"""Focused video install uses the existing platform and package-source policy."""

import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studio"))
import install_python_stack as ips


@pytest.fixture
def installer(monkeypatch):
    monkeypatch.setattr(ips.install_manifest, "pass_lock", lambda: contextlib.nullcontext(True))
    monkeypatch.setattr(ips, "_video_runtime_constraints", contextlib.nullcontext)
    monkeypatch.setattr(ips, "IS_MACOS", True)
    monkeypatch.setattr(ips, "IS_MAC_ARM", True)
    monkeypatch.setattr(ips, "NO_TORCH", True)
    monkeypatch.setattr(ips, "USE_UV", False)
    monkeypatch.setattr(ips, "_bootstrap_uv", lambda: True)
    monkeypatch.setattr(ips, "_explicit_torch_index_url", lambda: None)
    monkeypatch.setattr(ips, "_is_win_arm64_interpreter", lambda: False)
    pip = Mock()
    monkeypatch.setattr(ips, "pip_install", pip)
    monkeypatch.setattr(ips.subprocess, "run", Mock(return_value = SimpleNamespace(returncode = 0)))
    for name in ("_ensure_cuda_torch", "_ensure_rocm_torch", "_ensure_xpu_torch", "_ensure_cpu_torch"):
        monkeypatch.setattr(ips, name, Mock())
    monkeypatch.setattr(ips, "_ensure_expected_torch_flavor", lambda: True)
    return pip


def test_apple_silicon_uses_standard_wheels_and_verifies_import(installer):
    assert ips.install_video_pytorch() == 0
    assert installer.call_args.args[1:] == ("torch", "torchvision", "torchaudio")
    assert installer.call_args.kwargs["constrain"] is True
    assert ips.subprocess.run.call_args.args[0][0] == sys.executable
    assert "import torch" in ips.subprocess.run.call_args.args[0][-1]
    ips._ensure_cuda_torch.assert_not_called()


def test_installer_refuses_a_concurrent_dependency_pass(installer, monkeypatch):
    monkeypatch.setattr(ips.install_manifest, "pass_lock", lambda: contextlib.nullcontext(False))
    assert ips.install_video_pytorch() == 1
    installer.assert_not_called()


def test_intel_or_rosetta_mac_is_rejected(installer, monkeypatch):
    monkeypatch.setattr(ips, "IS_MAC_ARM", False)
    assert ips.install_video_pytorch() == 1
    installer.assert_not_called()


def test_cuda_uses_selected_index_and_compatible_specs(installer, monkeypatch):
    monkeypatch.setattr(ips, "IS_MACOS", False)
    monkeypatch.setattr(ips, "_expected_torch_flavor_tag", lambda: "cu128")
    monkeypatch.setattr(ips, "_expected_torch_index_url", lambda tag: "https://example.test/" + tag)
    monkeypatch.setattr(ips, "_cuda_repair_torch_specs", lambda *a: ("torch==2.9.0", "torchvision", "torchaudio"))
    assert ips.install_video_pytorch() == 0
    assert installer.call_args.args[1:] == (
        "torch==2.9.0", "torchvision", "torchaudio", "--index-url", "https://example.test/cu128",
    )
    assert installer.call_args.kwargs["constrain"] is False


def test_unavailable_mirror_does_not_fall_back_to_public_index(installer, monkeypatch):
    monkeypatch.setattr(ips, "IS_MACOS", False)
    monkeypatch.setattr(ips, "_expected_torch_flavor_tag", lambda: "cu128")
    monkeypatch.setattr(ips, "_expected_torch_index_url", lambda tag: None)
    assert ips.install_video_pytorch() == 1
    installer.assert_not_called()


def test_failed_import_is_not_success(installer, monkeypatch):
    monkeypatch.setattr(ips.subprocess, "run", Mock(return_value = SimpleNamespace(returncode = 1)))
    assert ips.install_video_pytorch() == 1


def test_runtime_constraints_preserve_packages_and_active_requirements(monkeypatch):
    import importlib.metadata
    import os

    dist = SimpleNamespace(
        metadata = {"Name": "existing_app"},
        version = "1.2.3",
        requires = [
            "torch<2.13,>=2.4",
            "numpy>=1.26",
            'torch==1.0; extra == "obsolete"',
            'torch==1.1; python_version < "3.0"',
        ],
    )
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [dist])
    monkeypatch.setenv("UV_CONSTRAINT", "/existing/uv.txt")
    monkeypatch.setenv("PIP_CONSTRAINT", "/existing/pip.txt")
    with ips._video_runtime_constraints():
        path = Path(os.environ["UV_CONSTRAINT"].split()[-1])
        lines = path.read_text().splitlines()
        assert "existing-app==1.2.3" in lines
        assert "torch<2.13,>=2.4" in lines
        assert "numpy>=1.26" in lines
        assert not any("torch==1." in line for line in lines)
        assert os.environ["PIP_CONSTRAINT"].endswith(str(path))
    assert os.environ["UV_CONSTRAINT"] == "/existing/uv.txt"
    assert os.environ["PIP_CONSTRAINT"] == "/existing/pip.txt"
    assert not path.exists()


def test_runtime_constraints_restore_environment_after_failure(monkeypatch):
    import importlib.metadata
    import os

    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])
    monkeypatch.delenv("UV_CONSTRAINT", raising = False)
    monkeypatch.delenv("PIP_CONSTRAINT", raising = False)
    with pytest.raises(RuntimeError):
        with ips._video_runtime_constraints():
            raise RuntimeError("install failed")
    assert "UV_CONSTRAINT" not in os.environ
    assert "PIP_CONSTRAINT" not in os.environ
