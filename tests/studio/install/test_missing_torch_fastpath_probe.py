# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test the missing-torch CLI probe in isolated venvs with synthetic metadata."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import packaging
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_FLAG = "--missing-torch-needs-dependency-pass"


def _site_packages(root: Path) -> Path:
    if sys.platform == "win32":
        return root / "Lib" / "site-packages"
    return (
        root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )


def _python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _dist(root: Path, name: str, version: str, *requires: str) -> None:
    info = _site_packages(root) / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents = True)
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    lines += [f"Requires-Dist: {req}" for req in requires]
    (info / "METADATA").write_text("\n".join(lines) + "\n", encoding = "utf-8")


@pytest.fixture
def studio_venv(tmp_path):
    """Expose only packaging from the host, keeping its torch out of the test venv."""
    root = tmp_path / "venv"
    venv.EnvBuilder(with_pip = False).create(root)
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "packaging").symlink_to(Path(packaging.__file__).parent, target_is_directory = True)
    return root, shim


def _probe(studio_venv, **env) -> int:
    root, shim = studio_venv
    child_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_NO_TORCH", "STUDIO_PACKAGE_NAME", "PYTHONHOME")
    }
    child_env["PYTHONPATH"] = str(shim)
    child_env.update(env)
    result = subprocess.run(
        [str(_python(root)), str(_STACK_PATH), _FLAG],
        env = child_env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    # A crash also exits 1, which would read as "keep the fast path".
    assert "Traceback" not in result.stderr, result.stderr
    return result.returncode


TORCH_ON_EVERY_PLATFORM = "torch<2.13.0,>=2.4.0"


@pytest.mark.skipif(sys.platform == "win32", reason = "setup.sh is the only caller")
class TestMissingTorchProbe:
    def test_core_package_requires_torch_and_none_is_installed_forces_the_pass(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4", TORCH_ON_EVERY_PLATFORM)
        _dist(studio_venv[0], "unsloth-zoo", "2026.9.3")
        assert _probe(studio_venv) == 0

    def test_an_installed_torch_keeps_the_fast_path(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4", TORCH_ON_EVERY_PLATFORM)
        _dist(studio_venv[0], "torch", "2.12.1+cu130")
        assert _probe(studio_venv) == 1

    def test_the_requirement_through_unsloth_zoo_counts(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4")
        _dist(studio_venv[0], "unsloth-zoo", "2026.9.3", TORCH_ON_EVERY_PLATFORM)
        assert _probe(studio_venv) == 0

    def test_a_platform_the_core_packages_do_not_need_torch_on_keeps_the_fast_path(
        self, studio_venv
    ):
        _dist(studio_venv[0], "unsloth", "2026.9.4", 'torch>=2.4.0; sys_platform == "no-such-os"')
        assert _probe(studio_venv) == 1

    def test_a_torch_behind_an_extra_keeps_the_fast_path(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4", 'torch>=2.4.0; extra == "intelgpu"')
        assert _probe(studio_venv) == 1

    def test_a_recorded_no_torch_install_keeps_the_fast_path(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4", TORCH_ON_EVERY_PLATFORM)
        (studio_venv[0] / ".unsloth-no-torch").write_text("", encoding = "utf-8")
        assert _probe(studio_venv) == 1

    def test_unsloth_no_torch_keeps_the_fast_path(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4", TORCH_ON_EVERY_PLATFORM)
        assert _probe(studio_venv, UNSLOTH_NO_TORCH = "1") == 1

    def test_a_custom_package_is_read_instead_of_unsloth(self, studio_venv):
        _dist(studio_venv[0], "unsloth", "2026.9.4", TORCH_ON_EVERY_PLATFORM)
        _dist(studio_venv[0], "my-unsloth", "1.0", TORCH_ON_EVERY_PLATFORM)
        assert _probe(studio_venv, STUDIO_PACKAGE_NAME = "my-unsloth") == 0
        assert _probe(studio_venv, STUDIO_PACKAGE_NAME = "other-package") == 1

    def test_no_core_package_keeps_the_fast_path(self, studio_venv):
        assert _probe(studio_venv) == 1
