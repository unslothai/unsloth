# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native component controls, not LPAC or production qualification.

Build with build_activation_context.py and set UNSLOTH_ACTIVATION_DRIVER.
The fixture copies selected DLL bytes; installed packages remain unchanged.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox.activation_manifest import (
    inspect_activation_image,
    require_empty_activation_manifest,
)


@pytest.fixture
def activation_case(tmp_path):
    driver = os.environ.get("UNSLOTH_ACTIVATION_DRIVER")
    if sys.platform != "win32" or not driver:
        pytest.skip("Set UNSLOTH_ACTIVATION_DRIVER to the freshly built native component driver.")
    # Discover installed package files without importing PIL or a native image.
    paths = [Path(p) / "PIL" for p in sys.path if p]
    selected = next(
        (p for root in paths if root.is_dir() for p in root.glob("_imaging.cp*.pyd")), None
    )
    if selected is None:
        pytest.skip("Selected Python has no installed Pillow image for the native control.")
    copied = tmp_path / selected.name
    shutil.copyfile(selected, copied)
    # A standalone native driver does not already have CPython's DLL loaded.
    # Supply the selected interpreter's companion DLLs in the private fixture.
    for dependency in Path(sys.base_prefix).glob("*.dll"):
        shutil.copyfile(dependency, tmp_path / dependency.name)
    manifest = require_empty_activation_manifest(inspect_activation_image(copied))
    assert manifest is not None
    data = tmp_path / "approved.manifest"
    data.write_bytes(manifest.data)
    return Path(driver), copied, data, manifest


def run_case(case, mode):
    driver, image, data, manifest = case
    return subprocess.run(
        [str(driver), str(image), str(data), str(manifest.language), mode],
        capture_output = True,
        text = True,
        timeout = 20,
        check = False,
    )


@pytest.mark.parametrize("mode", ["prepare-only", "match", "reload"])
def test_preparation_module_identity_and_actual_loader_reloads(activation_case, mode):
    result = run_case(activation_case, mode)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    if mode == "match":
        assert "wrong_module_denied=1 unsupported_request_denied=1 references=128" in result.stdout
    elif mode == "reload":
        assert "reloads=32 substitutions=32" in result.stdout
    else:
        assert "resource_only=1" in result.stdout


def test_changed_manifest_bytes_never_create_a_context(activation_case):
    data = activation_case[2]
    data.write_bytes(data.read_bytes() + b" ")
    result = run_case(activation_case, "prepare-only")
    assert result.returncode == 5
    assert "prepare=13" in result.stdout  # ERROR_INVALID_DATA, not loader success.
