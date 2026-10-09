# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from importlib import metadata
from types import SimpleNamespace

import pytest

from test_windows_multiarch_all_11815 import stack_mod


@pytest.fixture
def installed(monkeypatch):
    packages = {}

    def add(
        name,
        version,
        requirements = (),
    ):
        packages[name] = SimpleNamespace(
            metadata = {"Name": name}, version = version, requires = list(requirements)
        )

    for package, version in (("torch", "2.11.0+rocm7.14.0"), ("torchvision", "0.26.0+rocm7.14.0")):
        leaf = f"amd-{package}-device-gfx1100"
        family = f"amd-{package}-device-gfx110x"
        add(
            package,
            version,
            [
                f'{leaf}=={version}; extra == "device-gfx1100"',
                f'{family}=={version}; extra == "device-gfx1100"',
                f'amd-{package}-device-gfx1151=={version}; extra == "device-gfx1151"',
            ],
        )
        add(leaf, version)
        add(family, version)

    def distribution(name):
        if name not in packages:
            raise metadata.PackageNotFoundError(name)
        return packages[name]

    monkeypatch.setattr(metadata, "distribution", distribution)
    return packages, add


def test_complete_selected_extra_is_accepted(installed):
    assert stack_mod._multiarch_device_pack_installed("gfx1100:xnack-")


@pytest.mark.parametrize("package", ["torch", "torchvision"])
@pytest.mark.parametrize("part", ["gfx1100", "gfx110x"])
@pytest.mark.parametrize("damage", ["missing", "version"])
def test_leaf_names_do_not_hide_missing_or_mismatched_packs(installed, package, part, damage):
    packages, _ = installed
    name = f"amd-{package}-device-{part}"
    if damage == "missing":
        del packages[name]
    else:
        packages[name].version = "2.12.0+rocm7.14.0"
    assert not stack_mod._multiarch_device_pack_installed("gfx1100")


def test_family_names_come_from_wheel_metadata_and_dependencies_are_recursive(installed):
    packages, add = installed
    version = packages["torch"].version
    packages["torch"].requires = [f'amd-torch-device-gfx1100=={version}; extra == "device-gfx1100"']
    packages["amd-torch-device-gfx1100"].requires = [f"amd-torch-device-gfx11=={version}"]
    assert not stack_mod._multiarch_device_pack_installed("gfx1100")
    add("amd-torch-device-gfx11", version, [f"amd-torch-device-gfx1100=={version}"])
    assert stack_mod._multiarch_device_pack_installed("gfx1100")


@pytest.mark.parametrize("requirements", [[], ["not a requirement"]])
def test_missing_or_invalid_extra_metadata_requires_repair(installed, requirements):
    packages, _ = installed
    packages["torch"].requires = requirements
    assert not stack_mod._multiarch_device_pack_installed("gfx1100")


@pytest.mark.parametrize("marker", ["0", "1"])
@pytest.mark.parametrize("damage", [None, "missing", "version"])
def test_update_repairs_the_selected_extra_even_with_setup_marker(
    installed, monkeypatch, marker, damage
):
    packages, _ = installed
    family = "amd-torch-device-gfx110x"
    if damage == "missing":
        del packages[family]
    elif damage == "version":
        packages[family].version = "2.12.0+rocm7.14.0"
    monkeypatch.setenv("UNSLOTH_ROCM_TORCH_INSTALLED", marker)
    monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MIRROR", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR", raising = False)
    for name, value in {
        "IS_WINDOWS": True,
        "IS_MACOS": False,
        "_TORCH_BACKEND": "rocm",
        "_rocm_windows_torch_installed": False,
    }.items():
        monkeypatch.setattr(stack_mod, name, value)
    for name, result in {
        "_explicit_unknown_family_torch_index_url": None,
        "_explicit_torch_index_family": "",
        "_explicit_torch_index_is_unusable": False,
        "_explicit_rocm_torch_index_url": None,
        "_has_usable_nvidia_gpu": False,
        "_detect_windows_gfx_arch": "gfx1100",
        "_probe_torch_runtime": (True, True, "2.11.0+rocm7.14.0", "7.14", None),
        "_install_bnb_windows_rocm": True,
        "_is_win_arm64_interpreter": False,
    }.items():
        monkeypatch.setattr(stack_mod, name, lambda result = result: result)
    calls = []

    def install(*args, **kwargs):
        calls.append(args)
        return True

    monkeypatch.setattr(stack_mod, "pip_install_try", install)
    stack_mod._ensure_rocm_torch()
    assert stack_mod._rocm_windows_torch_installed
    assert len(calls) == int(damage is not None)
    if calls:
        assert "--force-reinstall" in calls[0]
        assert any(arg.startswith("torch[device-gfx1100]==") for arg in calls[0])
        assert any(arg.startswith("torchvision[device-gfx1100]==") for arg in calls[0])
