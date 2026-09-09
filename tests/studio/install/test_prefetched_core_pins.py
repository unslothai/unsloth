# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The swap after a background prefetch, with the index unreachable.

The core step asks the index which unsloth and unsloth-zoo are newest before it can
notice the wheels are already cached, so with PyPI denied it failed under uv and fell
through to pip. On a venv carrying a constraint conflict pip then started a resolution
the index it had not got could not finish (macOS, mlx-vlm against the transformers pin:
two minutes of 403 retries and exit 1). Given the pins the prefetch cached, the step is
retried from the cache with --offline before pip is tried at all.
"""

from __future__ import annotations

import types

import pytest

import install_python_stack as stack

PINS = ("unsloth==2026.9.5", "unsloth-zoo==2026.9.4", "numpy==2.5.3")


@pytest.fixture
def uv_only(monkeypatch):
    monkeypatch.setattr(stack, "USE_UV", True)
    monkeypatch.setattr(stack, "VERBOSE", False)
    monkeypatch.setattr(stack, "_invalidate_torch_runtime_probe", lambda: None)
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if "--offline" in cmd:
            return types.SimpleNamespace(returncode = 0, stdout = b"")
        return types.SimpleNamespace(
            returncode = 1, stdout = b"error: Failed to fetch: `https://pypi.org/simple/unsloth-zoo/`"
        )

    monkeypatch.setattr(stack.subprocess, "run", fake_run)

    def no_pip(*_a, **_k):
        raise AssertionError("pip must not run when the cached pins install")

    monkeypatch.setattr(stack, "run", no_pip)
    return calls


def test_the_cached_pins_are_installed_offline_before_pip_is_tried(uv_only):
    stack.pip_install(
        "Updating core packages",
        "--no-cache-dir",
        "--upgrade-package",
        "unsloth",
        "unsloth",
        "unsloth-zoo",
        offline_pins = PINS,
    )
    assert len(uv_only) == 2
    first, second = uv_only
    assert "--offline" not in first
    assert second[:3] == ["uv", "pip", "install"]
    assert "--offline" in second
    for pin in PINS:
        assert pin in second
    # The exact pins replace the upgrade request: nothing here asks the index anything.
    assert "--upgrade-package" not in second and "unsloth-zoo" not in second


def test_without_pins_the_pip_fallback_is_unchanged(monkeypatch, uv_only):
    ran = []
    monkeypatch.setattr(
        stack,
        "run",
        lambda label, cmd, **kw: ran.append(cmd) or types.SimpleNamespace(returncode = 0, stdout = b""),
    )
    stack.pip_install("Updating core packages", "--no-cache-dir", "unsloth", "unsloth-zoo")
    assert len(uv_only) == 1 and not any("--offline" in c for c in uv_only)
    assert len(ran) == 1


def test_a_failed_offline_install_still_falls_back_to_pip(monkeypatch, uv_only):
    monkeypatch.setattr(
        stack.subprocess,
        "run",
        lambda cmd, **kw: (
            uv_only.append(list(cmd)),
            types.SimpleNamespace(returncode = 1, stdout = b""),
        )[1],
    )
    ran = []
    monkeypatch.setattr(
        stack,
        "run",
        lambda label, cmd, **kw: ran.append(cmd) or types.SimpleNamespace(returncode = 0, stdout = b""),
    )
    stack.pip_install("Updating core packages", "unsloth", "unsloth-zoo", offline_pins = PINS)
    assert any("--offline" in c for c in uv_only)
    assert len(ran) == 1


@pytest.mark.parametrize(
    "value, expected",
    [
        ("unsloth==2026.9.5 unsloth-zoo==2026.9.4", ("unsloth==2026.9.5", "unsloth-zoo==2026.9.4")),
        ("  unsloth==2026.9.5\n", ("unsloth==2026.9.5",)),
        ("", ()),
        ("--offline unsloth unsloth==2026.9.5 a==b==c", ("unsloth==2026.9.5",)),
    ],
)
def test_only_well_formed_pins_are_read_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("UNSLOTH_PREFETCHED_CORE_PINS", value)
    assert stack._prefetched_core_pins() == expected


def test_the_core_steps_are_the_ones_that_get_the_pins():
    """A source guard: the default (PyPI) core step and the no-torch core step pass the
    pins, and only they do. A --local update overlays a checkout, which no prefetch
    prepared; a no-torch prefetch fetches the core packages with --no-deps, which is
    exactly what that step installs."""
    import inspect

    source = inspect.getsource(stack.install_python_stack)
    core = source[source.index('"[TAURI:DIAG] uv cache=') :]
    core = core[: core.index("if not skip_base:")]
    assert "offline_pins = _prefetched_core_pins()" in core
    no_torch = source[source.index("(no-torch mode)") :]
    no_torch = no_torch[: no_torch.index("pip_install(") ]
    assert "offline_pins = _prefetched_core_pins()" in no_torch
    local = source[source.index("# Local dev install:") :]
    local = local[: local.index("_overlay_local_core_packages(local_repo)")]
    assert "offline_pins" not in local
    assert source.count("offline_pins = _prefetched_core_pins()") == 2


def test_the_offline_retry_keeps_no_deps_for_the_no_torch_step(uv_only):
    stack.pip_install(
        "Updating unsloth + unsloth-zoo (no-torch mode)",
        "--no-cache-dir",
        "--no-deps",
        "--upgrade-package",
        "unsloth",
        "unsloth",
        "unsloth-zoo",
        offline_pins = PINS,
    )
    first, second = uv_only
    assert "--no-deps" in first and "--offline" in second and "--no-deps" in second
    # The default core step installs with dependencies, and so does its retry.
    uv_only.clear()
    stack.pip_install("Updating core packages", "--no-cache-dir", "unsloth", offline_pins = PINS)
    assert "--no-deps" not in uv_only[1]
