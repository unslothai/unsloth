# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The no-torch runtime picks exactly one Pillow per environment, and on Apple Silicon it stays
inside the MLX image engine's Pillow range (mflux breaks on Pillow 13)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[3]
NO_TORCH = REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
INSTALLER = REPO_ROOT / "studio" / "install_python_stack.py"


def _pillow_pins() -> list[Requirement]:
    reqs = []
    for line in NO_TORCH.read_text(encoding = "utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and Requirement(line).name.lower() == "pillow":
            reqs.append(Requirement(line))
    return reqs


def _mlx_pillow_range() -> Requirement:
    tree = ast.parse(INSTALLER.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "_MLX_DIFFUSION_DEPS"
        ):
            for spec in ast.literal_eval(node.value):
                if Requirement(spec).name.lower() == "pillow":
                    return Requirement(spec)
    raise AssertionError("_MLX_DIFFUSION_DEPS has no pillow entry")


def _env(python: str, sys_platform: str, machine: str) -> dict[str, str]:
    return {
        "python_version": python,
        "python_full_version": python + ".0",
        "sys_platform": sys_platform,
        "platform_machine": machine,
        "platform_system": {"darwin": "Darwin", "win32": "Windows"}.get(sys_platform, "Linux"),
        "os_name": "nt" if sys_platform == "win32" else "posix",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }


@pytest.mark.parametrize("python", ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"])
@pytest.mark.parametrize(
    "sys_platform, machine",
    [
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "AMD64"),
        ("darwin", "arm64"),
        ("darwin", "x86_64"),
    ],
)
def test_one_pillow_pin_applies(python, sys_platform, machine):
    env = _env(python, sys_platform, machine)
    applying = [r for r in _pillow_pins() if r.marker is None or r.marker.evaluate(env)]
    assert len(applying) == 1, (env, [str(r) for r in applying])


@pytest.mark.parametrize("python", ["3.11", "3.12", "3.13", "3.14"])
def test_apple_silicon_pillow_stays_inside_the_mlx_image_range(python):
    env = _env(python, "darwin", "arm64")
    (pin,) = [r for r in _pillow_pins() if r.marker is None or r.marker.evaluate(env)]
    version = next(iter(pin.specifier)).version
    assert _mlx_pillow_range().specifier.contains(version), (str(pin), str(_mlx_pillow_range()))
