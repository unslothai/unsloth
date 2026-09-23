# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF-only installs apply no-torch-runtime.txt with --no-deps, so Hub extras must be declared."""

from __future__ import annotations

import pathlib

import pytest
from packaging.requirements import Requirement

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
NO_TORCH_RUNTIME = REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
STUDIO_TXT = REPO_ROOT / "studio" / "backend" / "requirements" / "studio.txt"
CONSTRAINTS = REPO_ROOT / "studio" / "backend" / "requirements" / "single-env" / "constraints.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Machines hf-xet ships a wheel for, as of 1.6.0; off-list resolves the maturin sdist instead.
WHEELED_MACHINES = ("x86_64", "amd64", "AMD64", "arm64", "aarch64", "ARM64")
UNWHEELED_MACHINES = ("ppc64le", "s390x", "i686", "armv7l", "riscv64")

# huggingface-hub 1.32's own floor, which --no-deps leaves nothing else to enforce.
HUB_HF_XET_FLOOR = "1.5.2"

ALL_FILES = (NO_TORCH_RUNTIME, STUDIO_TXT, CONSTRAINTS)


def _requirement_names(path: pathlib.Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        names.add(Requirement(text).name.lower().replace("_", "-"))
    return names


def _hf_xet_requirements(path: pathlib.Path) -> list[Requirement]:
    out = []
    if path.suffix == ".toml":
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip().rstrip(",")
            if text.startswith('"hf-xet') or text.startswith("'hf-xet"):
                out.append(Requirement(text.strip("\"'")))
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        req = Requirement(text)
        if req.name.lower().replace("_", "-") == "hf-xet":
            out.append(req)
    return out


def _env(python_version: str, machine: str) -> dict[str, str]:
    """Full environment: an absent platform_machine makes the arch assertions read as the runner."""
    return {
        "python_version": python_version,
        "python_full_version": python_version + ".0",
        "platform_machine": machine,
        "platform_system": "Linux",
        "sys_platform": "linux",
        "os_name": "posix",
        "platform_release": "",
        "platform_version": "",
        "implementation_name": "cpython",
        "implementation_version": python_version + ".0",
        "platform_python_implementation": "CPython",
        "extra": "",
    }


def test_no_torch_runtime_declares_hf_xet_for_hub_large_downloads():
    names = _requirement_names(NO_TORCH_RUNTIME)
    assert "hf-xet" in names, (
        "no-torch-runtime.txt is installed --no-deps; without an explicit hf-xet pin, "
        "Desktop GGUF-only envs fall back to HTTP and huggingface_hub raises on large blobs."
    )


@pytest.mark.parametrize("path", ALL_FILES + (PYPROJECT,), ids=lambda p: p.name)
def test_every_hf_xet_pin_is_declared_once_and_marked(path: pathlib.Path):
    pins = _hf_xet_requirements(path)
    assert len(pins) == 1, f"expected one hf-xet line in {path.name}, got {pins!r}"
    assert pins[0].marker is not None, f"{path.name} hf-xet must be environment-marked"


@pytest.mark.parametrize("path", ALL_FILES + (PYPROJECT,), ids=lambda p: p.name)
def test_hf_xet_pins_are_python_310_plus_only(path: pathlib.Path):
    """Python < 3.10 stays on huggingface-hub 0.36.2, the same split the hub pins above use.

    Not a wheel question: hf-xet ships cp38-abi3 wheels and declares Requires-Python >= 3.8, so 3.9
    could install it. No Desktop or Studio path selects 3.9 (install.sh defaults 3.13/3.12,
    install.ps1 ranks 3.13/3.12/3.11), so the pre-Xet branch is left exactly as it was.
    """
    marker = _hf_xet_requirements(path)[0].marker
    assert marker.evaluate(_env("3.9", "x86_64")) is False
    assert marker.evaluate(_env("3.10", "x86_64")) is True
    assert marker.evaluate(_env("3.13", "x86_64")) is True


@pytest.mark.parametrize("path", ALL_FILES + (PYPROJECT,), ids=lambda p: p.name)
def test_hf_xet_pins_only_ask_for_machines_with_wheels(path: pathlib.Path):
    """A machine with no hf-xet wheel must not be asked for it: the fallback is a maturin build."""
    marker = _hf_xet_requirements(path)[0].marker
    for machine in WHEELED_MACHINES:
        assert marker.evaluate(_env("3.12", machine)) is True, machine
    for machine in UNWHEELED_MACHINES:
        assert marker.evaluate(_env("3.12", machine)) is False, (
            f"{path.name} would resolve hf-xet on {machine}, which has no wheel; under --no-deps "
            "that is a Rust source build on the user's machine."
        )


@pytest.mark.parametrize("path", ALL_FILES + (PYPROJECT,), ids=lambda p: p.name)
def test_hf_xet_floor_is_not_below_the_hub_requirement(path: pathlib.Path):
    specifier = _hf_xet_requirements(path)[0].specifier
    assert (
        HUB_HF_XET_FLOOR in specifier
    ), f"{path.name} pins hf-xet {specifier}, excluding the floor itself"
    below = ("1.5.1", "1.5.0", "1.1.3")
    allowed = [v for v in below if v in specifier]
    assert not allowed, (
        f"{path.name} pins hf-xet {specifier}, which still allows {allowed}, below the "
        f"{HUB_HF_XET_FLOOR} floor huggingface-hub declares; --no-deps means nothing else enforces it."
    )
    assert "2.0" not in specifier, f"{path.name} must keep the <2.0 ceiling"


def test_the_four_declarations_agree():
    pins = {p.name: str(_hf_xet_requirements(p)[0]) for p in ALL_FILES + (PYPROJECT,)}
    canonical = {str(Requirement(v)) for v in pins.values()}
    assert len(canonical) == 1, f"hf-xet pins disagree across files: {pins}"
