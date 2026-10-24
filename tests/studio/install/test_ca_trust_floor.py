# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""certifi 2026.10.19 (and pip 26.2.2 on 3.10+) in the Studio venv and Core: neither keeps the emSign ECC G3 root."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from unittest import mock

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CERTIFI_FLOOR = (2026, 10, 19)

sys.path.insert(0, str(REPO_ROOT / "studio"))
import install_python_stack as ips  # noqa: E402


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.match(r"\d+(?:\.\d+)*", version).group(0).split("."))


def test_the_no_torch_runtime_pin_is_at_or_above_the_floor() -> None:
    text = (REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt").read_text(
        encoding = "utf-8"
    )
    (pin,) = re.findall(r"^certifi==(\S+)", text, re.MULTILINE)
    assert _release(pin) >= CERTIFI_FLOOR


def test_core_declares_the_certifi_floor() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8"))["project"][
        "dependencies"
    ]
    floors = [d for d in deps if re.match(r"certifi\s*>=", d)]
    assert len(floors) == 1
    assert _release(floors[0].split(">=", 1)[1]) >= CERTIFI_FLOOR


def test_the_installer_floor_matches() -> None:
    assert ips._TRUST_FLOOR_CERTIFI == "certifi>=2026.10.19"
    assert ips._TRUST_FLOOR_PIP == "pip>=26.2.2"


def _versions(**installed):
    return lambda name: installed.get(name)


@pytest.mark.parametrize(
    "installed, py, expected",
    [
        ({"certifi": "2026.7.22", "pip": "24.2"}, (3, 12), ["certifi>=2026.10.19", "pip>=26.2.2"]),
        ({"certifi": "2026.10.19", "pip": "26.2.2"}, (3, 12), []),
        # Newer than the floor is never touched.
        ({"certifi": "2027.1.1", "pip": "27.0"}, (3, 13), []),
        ({"certifi": "2026.7.22"}, (3, 12), ["certifi>=2026.10.19", "pip>=26.2.2"]),
        # pip 26.2.2 does not support 3.9: only certifi.
        ({"certifi": "2026.7.22", "pip": "24.2"}, (3, 9), ["certifi>=2026.10.19"]),
    ],
)
def test_only_unsatisfied_floors_are_installed(monkeypatch, installed, py, expected) -> None:
    monkeypatch.setattr(ips, "_installed_distribution_version", _versions(**installed))
    monkeypatch.setattr(ips.sys, "version_info", py)
    assert ips._trust_floor_specs() == expected


def test_a_satisfied_venv_runs_nothing(monkeypatch) -> None:
    monkeypatch.setattr(ips, "_trust_floor_specs", lambda: [])
    install = mock.Mock()
    monkeypatch.setattr(ips, "pip_install_try", install)
    ips._ensure_trust_floor()
    install.assert_not_called()


def test_an_offline_failure_warns_and_continues(monkeypatch) -> None:
    monkeypatch.setattr(ips, "_trust_floor_specs", lambda: ["certifi>=2026.10.19"])
    monkeypatch.setattr(ips, "pip_install_try", lambda *a, **k: False)
    notes: list[str] = []
    monkeypatch.setattr(ips, "_note", lambda message, *a: notes.append(message))
    ips._ensure_trust_floor()
    assert notes and "certifi>=2026.10.19" in notes[0]


def _load_no_torch_harness():
    path = REPO_ROOT / "tests" / "python" / "test_no_torch_filtering.py"
    spec = importlib.util.spec_from_file_location("_trust_floor_no_torch_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ips is ips
    return module.TestInstallPythonStackSubprocessMock()


@pytest.mark.parametrize("skip_base", [True, False], ids = ["installer-handoff", "update"])
@pytest.mark.parametrize("no_torch", [False, True], ids = ["torch", "no-torch"])
def test_the_full_pass_raises_the_floor_in_both_modes(no_torch, skip_base) -> None:
    """Reached on every path, before any requirements file is installed, through the uv installer."""
    harness = _load_no_torch_harness()
    with (
        mock.patch.object(
            ips, "_trust_floor_specs", return_value = ["certifi>=2026.10.19", "pip>=26.2.2"]
        ),
        mock.patch.object(ips, "_venv_pip_is_usable", return_value = True),
        mock.patch.object(ips, "_plan_pass", return_value = None),
    ):
        cmds = harness._capture_install(
            no_torch = no_torch, is_macos = False, is_windows = False, skip_base = skip_base
        )
    floor = [
        i
        for i, cmd in enumerate(cmds)
        if cmd.startswith("uv pip install")
        and "certifi>=2026.10.19" in cmd
        and "pip>=26.2.2" in cmd
    ]
    assert len(floor) == 1, cmds
    first_requirements_file = next(
        (i for i, cmd in enumerate(cmds) if " -r " in cmd and "uv pip install" in cmd), len(cmds)
    )
    assert floor[0] < first_requirements_file
