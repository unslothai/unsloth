# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the accelerator-stack health report (NVIDIA QA P0-1).

The managed Windows xformers was built for torch 2.10.0+cu128 and Python 3.10.11 while
the app ran cu130 and Python 3.13.2, so its CUDA extensions never loaded and
memory-efficient attention silently went missing. ``get_package_versions()`` reported
nothing at all about it -- it covered only unsloth/torch/transformers plus torch's CUDA
version, so a mismatched wheel looked identical to a healthy one.

These tests cover what the report must now answer: what Python is running, which
optimized kernels are installed, which of them actually load, and -- for xformers -- what
the wheel on disk was compiled against. Nothing here needs a GPU or a working xformers.
"""

import ast
import json
from pathlib import Path

import pytest

import utils.hardware.hardware as hw

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse = True)
def _clear_report_cache():
    # The report is cached for the process; a stale cache would leak between tests.
    hw._accelerator_report_cache = None
    yield
    hw._accelerator_report_cache = None


@pytest.fixture
def on_accelerator(monkeypatch):
    """Pin the host to CUDA so the probe runs, whatever the test machine is.

    Without this every probe assertion below silently passes on a CPU-only runner by
    never probing at all.
    """
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)


def _cpp_lib(torch = "2.10.0+cu128", cuda = 1208, python = "3.10.11", hip = None):
    """The real shape of an xformers wheel's cpp_lib.json (verified against 0.0.34)."""
    return {
        "version": {"cuda": cuda, "hip": hip, "torch": torch, "python": python},
        "env": {"XFORMERS_PACKAGE_FROM": "wheel-v0.0.34"},
    }


# -- get_package_versions stays additive ----------------------------------------------------------


def test_package_versions_keeps_its_existing_keys():
    # Every existing consumer (About tab, training method auto-selection, the ROCm and
    # XPU tests) reads these; the new entries must not disturb them.
    versions = hw.get_package_versions()
    for key in ("unsloth", "torch", "transformers", "cuda", "rocm"):
        assert key in versions
    # Flat str-or-None, as before: nothing nested crept into this dict.
    for key, value in versions.items():
        assert value is None or isinstance(value, str), f"{key} is {type(value)}"


def test_package_versions_reports_python_and_the_accelerators():
    versions = hw.get_package_versions()
    assert versions["python"] == __import__("platform").python_version()
    for key in ("xformers", "flash_attn", "torchao", "bitsandbytes"):
        assert key in versions, f"{key} missing from get_package_versions()"


def test_flash_attn_key_is_an_identifier():
    # The distribution is "flash-attn" but the JSON key must be usable as a property
    # name on the frontend without bracket access.
    assert "flash-attn" not in hw.get_package_versions()


# -- build metadata -------------------------------------------------------------------------------


def test_built_for_is_read_from_cpp_lib_json(tmp_path, monkeypatch):
    import importlib.util
    import types

    package = tmp_path / "xformers"
    package.mkdir()
    (package / "cpp_lib.json").write_text(json.dumps(_cpp_lib()), encoding = "utf-8")
    spec = types.SimpleNamespace(
        submodule_search_locations = [str(package)], origin = str(package / "__init__.py")
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: spec)

    assert hw._xformers_built_for() == {
        "torch": "2.10.0+cu128",
        # xformers stores major * 100 + minor; its own message prints the raw 1208.
        "cuda": "12.8",
        "hip": None,
        "python": "3.10.11",
    }


def test_built_for_is_none_without_xformers(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert hw._xformers_built_for() is None


def test_break_description_names_both_sides(monkeypatch):
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.10.0+cu130", "cuda": "13.0", "python": "3.13.2"},
    )
    reason = hw._describe_xformers_break(
        {"torch": "2.10.0+cu128", "cuda": "12.8", "hip": None, "python": "3.10.11"},
        OSError("[WinError 126] The specified module could not be found"),
    )
    # The whole P0 in one line: what it was built for, what is actually running.
    assert "2.10.0+cu128" in reason
    assert "3.10.11" in reason
    assert "2.10.0+cu130" in reason
    assert "3.13.2" in reason


def test_break_description_falls_back_to_the_raw_error():
    # No cpp_lib.json (source install): the exception is all we have, and saying nothing
    # would be worse.
    reason = hw._describe_xformers_break(None, ImportError("undefined symbol: _ZN3c10"))
    assert "undefined symbol" in reason


# -- the report -----------------------------------------------------------------------------------


def test_report_has_the_documented_shape(on_accelerator):
    report = hw.get_accelerator_report()
    assert report["python_version"] == __import__("platform").python_version()
    assert set(report["packages"]) == {"xformers", "flash_attn", "torchao", "bitsandbytes"}
    for name, entry in report["packages"].items():
        assert set(entry) >= {"version", "installed", "imports", "runs", "reason"}, name
        assert isinstance(entry["installed"], bool)
        assert isinstance(entry["imports"], bool)
        assert entry["runs"] in (True, False, None)
        assert entry["reason"] is None or isinstance(entry["reason"], str)
    assert isinstance(report["degraded"], list)
    # xformers is the one with a wheel-recorded build, so it carries built_for.
    assert "built_for" in report["packages"]["xformers"]


def test_report_flags_an_installed_but_dead_package(monkeypatch, on_accelerator):
    monkeypatch.setattr(hw, "pkg_version", lambda name: "0.0.34")
    monkeypatch.setattr(
        hw,
        "_probe_xformers",
        lambda: {
            "imports": True,
            "runs": False,
            "reason": "xformers was built for torch 2.10.0+cu128 ...",
            "built_for": _cpp_lib()["version"],
        },
    )
    monkeypatch.setattr(hw, "_probe_flash_attn", lambda: {"imports": True, "runs": True, "reason": None})
    monkeypatch.setattr(hw, "_probe_import", lambda name: {"imports": True, "runs": None, "reason": None})

    report = hw.get_accelerator_report(refresh = True)
    assert report["degraded"] == ["xformers"]
    assert report["packages"]["xformers"]["runs"] is False
    assert "2.10.0+cu128" in report["packages"]["xformers"]["reason"]


def test_absent_package_is_not_degraded(monkeypatch, on_accelerator):
    # "Not installed" is a normal configuration, not a broken one. Reporting it as
    # degraded would make the UI banner permanent on every machine without flash-attn.
    from importlib.metadata import PackageNotFoundError

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(hw, "pkg_version", missing)
    report = hw.get_accelerator_report(refresh = True)
    assert report["degraded"] == []
    for entry in report["packages"].values():
        assert entry["installed"] is False
        assert entry["reason"] is None


def test_probe_can_be_skipped(monkeypatch, on_accelerator):
    # Escape hatch for an install where importing a broken native wheel is worse than
    # not knowing. Versions still report; nothing is claimed to be broken.
    monkeypatch.setenv(hw._ACCELERATOR_PROBE_ENV, "1")
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")

    def must_not_run(*args, **kwargs):
        raise AssertionError("probe ran despite the skip flag")

    monkeypatch.setattr(hw, "_probe_xformers", must_not_run)
    monkeypatch.setattr(hw, "_probe_import", must_not_run)
    monkeypatch.setattr(hw, "_probe_flash_attn", must_not_run)

    report = hw.get_accelerator_report(refresh = True)
    assert report["probed"] is False
    assert report["degraded"] == []
    assert report["packages"]["xformers"]["reason"] == "not probed"


def test_report_survives_a_probe_that_raises(monkeypatch, on_accelerator):
    # A diagnostic must never be what 500s /api/system/hardware.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "0.0.34")

    def boom(*args, **kwargs):
        raise RuntimeError("native library blew up")

    monkeypatch.setattr(hw, "_probe_xformers", boom)
    monkeypatch.setattr(hw, "_probe_import", boom)
    monkeypatch.setattr(hw, "_probe_flash_attn", boom)

    report = hw.get_accelerator_report(refresh = True)
    assert "native library blew up" in report["packages"]["xformers"]["reason"]
    assert set(report["degraded"]) == {"xformers", "flash_attn", "torchao", "bitsandbytes"}


def test_report_is_cached(monkeypatch, on_accelerator):
    calls = {"n": 0}

    def counting_probe(name):
        calls["n"] += 1
        return {"imports": True, "runs": None, "reason": None}

    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    monkeypatch.setattr(hw, "_probe_import", counting_probe)
    monkeypatch.setattr(hw, "_probe_xformers", lambda: {"imports": True, "runs": True, "reason": None})
    monkeypatch.setattr(hw, "_probe_flash_attn", lambda: {"imports": True, "runs": True, "reason": None})

    hw.get_accelerator_report(refresh = True)
    first = calls["n"]
    hw.get_accelerator_report()
    hw.get_accelerator_report()
    assert calls["n"] == first, "the report re-imported on a cache hit"


def test_cached_report_is_not_shared_by_reference():
    # Callers mutate response bodies; the cache must not be editable through them.
    first = hw.get_accelerator_report()
    first["packages"]["xformers"]["version"] = "tampered"
    assert hw.get_accelerator_report()["packages"]["xformers"]["version"] != "tampered"


# -- endpoint wiring (ast, so it runs without starting the app) ------------------------------------


def test_hardware_endpoint_returns_the_report_only_with_details():
    source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "get_hardware_info"
    )
    body = ast.get_source_segment(source, node)
    assert "get_accelerator_report" in body
    # Detail-only: the default response is polled for training-method selection and must
    # not pay for a native import.
    detail_branch = body.split("if include_details:", 1)
    assert len(detail_branch) == 2, "include_details branch went missing"
    assert "get_accelerator_report()" in detail_branch[1]
    assert "get_accelerator_report()" not in detail_branch[0]
    assert '"accelerators"' in detail_branch[1]


def test_a_cpu_or_mac_host_is_never_reported_as_degraded(monkeypatch):
    # bitsandbytes is a dependency on every platform but only loads on CUDA/XPU, so on a
    # Mac or a CPU-only host its import failure is expected, not a broken acceleration
    # stack. Probing there would pin a permanent false banner to those installs.
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")

    def must_not_run(*args, **kwargs):
        raise AssertionError("probed a host that cannot use these kernels")

    monkeypatch.setattr(hw, "_probe_xformers", must_not_run)
    monkeypatch.setattr(hw, "_probe_import", must_not_run)
    monkeypatch.setattr(hw, "_probe_flash_attn", must_not_run)

    report = hw.get_accelerator_report(refresh = True)
    assert report["probed"] is False
    assert report["degraded"] == []
    assert report["packages"]["bitsandbytes"]["reason"] == "not used on this device"


def test_a_cpu_host_is_distinguishable_from_an_opted_out_one(monkeypatch):
    # Two different unknowns: "these kernels do not apply here" and "you told us not to
    # look". Collapsing them would make the About tab lie about one of them.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setenv(hw._ACCELERATOR_PROBE_ENV, "1")
    opted_out = hw.get_accelerator_report(refresh = True)

    monkeypatch.delenv(hw._ACCELERATOR_PROBE_ENV)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    no_gpu = hw.get_accelerator_report(refresh = True)

    assert opted_out["packages"]["torchao"]["reason"] == "not probed"
    assert no_gpu["packages"]["torchao"]["reason"] == "not used on this device"
