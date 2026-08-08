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
import types
from importlib.metadata import PackageNotFoundError
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
    """Pin the host to a plain CUDA box so the probe is considered applicable.

    Without this every probe assertion below silently passes on a CPU-only runner by
    never probing at all.
    """
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)


@pytest.fixture
def fake_probe(monkeypatch):
    """Replace the probe subprocess. Returns a setter for its result.

    Nothing in this file may run the real probe: it spawns an interpreter that imports
    the host's native wheels, which is slow, host-dependent, and on a genuinely broken
    wheel can abort the child -- the very case these tests describe.
    """
    calls = {"n": 0, "names": None}

    def install(results):
        def run(
            names,
            timeout = 180,
            status = None,
            keep_cuda = False,
        ):
            calls["n"] += 1
            calls["names"] = list(names)
            return results

        monkeypatch.setattr(hw, "_run_probe_subprocess", run)
        return calls

    return install


def _cpp_lib(
    torch = "2.10.0+cu128",
    cuda = 1208,
    python = "3.10.11",
    hip = None,
):
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
        "OSError: [WinError 126] The specified module could not be found",
    )
    # The whole P0 in one line: what it was built for, what is actually running.
    assert "2.10.0+cu128" in reason
    assert "3.10.11" in reason
    assert "2.10.0+cu130" in reason
    assert "3.13.2" in reason


def test_break_description_falls_back_to_the_raw_error():
    # No cpp_lib.json (source install): the exception is all we have, and saying nothing
    # would be worse.
    reason = hw._describe_xformers_break(None, "ImportError: undefined symbol: _ZN3c10")
    assert "undefined symbol" in reason


def test_a_matching_build_is_not_called_a_mismatch(monkeypatch):
    # [WinError 126] with a perfectly matching wheel is the most common Windows xformers
    # failure that is NOT a version mismatch (missing VC++ runtime or CUDA DLL). Claiming
    # a mismatch whenever build metadata merely exists misdiagnoses it AND throws the real
    # error away, leaving the user nothing to search for.
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.10.0+cu130", "cuda": "13.0", "python": "3.13.2"},
    )
    reason = hw._describe_xformers_break(
        {"torch": "2.10.0+cu130", "cuda": "13.0", "hip": None, "python": "3.10.11"},
        "OSError: [WinError 126] The specified module could not be found",
    )
    assert reason == "OSError: [WinError 126] The specified module could not be found"


def test_a_python_only_difference_is_not_a_mismatch(monkeypatch):
    # The wheels are abi3/none-tagged and _C loads through torch.ops.load_library, not the
    # CPython ABI: this very repo runs a 3.10-built xformers on 3.13 with working kernels.
    # Naming Python here sends people off to reinstall Python for nothing.
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.9.1+cu128", "cuda": "12.8", "python": "3.13.12"},
    )
    reason = hw._describe_xformers_break(
        {"torch": "2.9.1+cu128", "cuda": "12.8", "hip": None, "python": "3.10.19"},
        "OSError: boom",
    )
    assert reason == "OSError: boom"


def test_a_cuda_minor_difference_is_not_a_mismatch(monkeypatch):
    # CUDA minor version compatibility: a cu126-built wheel loads against a cu128 torch.
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.10.0", "cuda": "12.8", "python": "3.13.2"},
    )
    reason = hw._describe_xformers_break(
        {"torch": None, "cuda": "12.6", "hip": None, "python": None}, "OSError: boom"
    )
    assert reason == "OSError: boom"


# -- the report -----------------------------------------------------------------------------------


def test_report_has_the_documented_shape(on_accelerator, fake_probe):
    fake_probe(
        {
            name: {"imports": True, "runs": None, "error": None}
            for name, _ in hw._ACCELERATOR_PACKAGES
        }
    )
    report = hw.get_accelerator_report(refresh = True)
    assert report["python_version"] == __import__("platform").python_version()
    assert set(report["packages"]) == {"xformers", "flash_attn", "torchao", "bitsandbytes"}
    for name, entry in report["packages"].items():
        assert set(entry) >= {"version", "installed", "imports", "runs", "reason"}, name
        # installed must follow the metadata, not the probe: that split is the whole
        # point (a mismatched wheel is installed, imports, and does not work).
        assert entry["installed"] is (entry["version"] is not None), name
    assert isinstance(report["degraded"], list)
    assert report["torch_version"] == hw._running_torch()["torch"]
    # xformers is the one with a wheel-recorded build, so it carries built_for.
    assert "built_for" in report["packages"]["xformers"]


def test_report_flags_an_installed_but_dead_package(monkeypatch, on_accelerator, fake_probe):
    monkeypatch.setattr(hw, "pkg_version", lambda name: "0.0.34")
    monkeypatch.setattr(hw, "_xformers_built_for", lambda: _cpp_lib()["version"] | {"cuda": "12.8"})
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.10.0+cu130", "cuda": "13.0", "python": "3.13.2"},
    )
    fake_probe(
        {
            "xformers": {"imports": True, "runs": False, "error": "OSError: [WinError 126]"},
            "flash_attn": {"imports": True, "runs": True, "error": None},
            "torchao": {"imports": True, "runs": None, "error": None},
            "bitsandbytes": {"imports": True, "runs": None, "error": None},
        }
    )

    report = hw.get_accelerator_report(refresh = True)
    assert report["degraded"] == ["xformers"]
    assert report["packages"]["xformers"]["runs"] is False
    assert "2.10.0+cu128" in report["packages"]["xformers"]["reason"]


def test_only_the_installed_and_applicable_packages_are_probed(
    monkeypatch, on_accelerator, fake_probe
):
    # Probing something that is not installed wastes an import and can only produce a
    # ModuleNotFoundError the metadata already told us about.
    def only_xformers(name):
        if name == "xformers":
            return "0.0.34"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(hw, "pkg_version", only_xformers)
    calls = fake_probe({"xformers": {"imports": True, "runs": True, "error": None}})
    hw.get_accelerator_report(refresh = True)
    assert calls["names"] == ["xformers"]


def test_the_probe_is_one_child_per_visibility_group(monkeypatch, on_accelerator):
    """The interpreter start and the torch import dominate, so one child per PACKAGE would
    cost four times as much for the same answer. Two groups, though, not one: bitsandbytes
    picks its native library from torch.cuda.is_available(), so with the GPUs hidden a healthy
    CUDA install loads the CPU library and reports dead handles. It gets the app's own mask;
    everything else keeps the hidden one, which is what stops a diagnostic taking VRAM from a
    run in progress."""
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.0")
    seen: list[tuple] = []

    def run(
        names,
        timeout = 180,
        status = None,
        keep_cuda = False,
    ):
        seen.append((tuple(names), keep_cuda))
        return {name: {"imports": True, "runs": None, "error": None} for name in names}

    monkeypatch.setattr(hw, "_run_probe_subprocess", run)
    hw.get_accelerator_report(refresh = True)

    assert seen == [
        (("xformers", "flash_attn", "torchao"), False),
        (("bitsandbytes",), True),
    ]


def test_absent_package_is_not_degraded(monkeypatch, on_accelerator, fake_probe):
    # "Not installed" is a normal configuration, not a broken one. Reporting it as
    # degraded would make the UI banner permanent on every machine without flash-attn.
    from importlib.metadata import PackageNotFoundError

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(hw, "pkg_version", missing)
    fake_probe({})
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

    monkeypatch.setattr(hw, "_run_probe_subprocess", must_not_run)

    report = hw.get_accelerator_report(refresh = True)
    assert report["probed"] is False
    assert report["degraded"] == []
    assert report["packages"]["xformers"]["reason"] == "not probed"


def test_a_probe_that_cannot_answer_is_unknown_not_broken(monkeypatch, on_accelerator):
    # The child timed out, crashed, or printed nothing. That says nothing about the
    # packages, so it must not light the banner.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    monkeypatch.setattr(
        hw,
        "_run_probe_subprocess",
        lambda names, timeout = 180, status = None, keep_cuda = False: None,
    )

    report = hw.get_accelerator_report(refresh = True)
    assert report["probed"] is False
    assert report["degraded"] == []
    assert report["packages"]["torchao"]["reason"] == "could not be checked"


def test_a_long_reason_is_capped_before_it_reaches_the_ui(monkeypatch, on_accelerator, fake_probe):
    # `reason` lands in a settings row description AND in its aria-label, so an
    # unbounded traceback string would be read out in full by a screen reader.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    fake_probe({"torchao": {"imports": False, "runs": None, "error": "x" * 5000}})
    report = hw.get_accelerator_report(refresh = True)
    assert len(report["packages"]["torchao"]["reason"]) <= hw._MAX_REASON_CHARS


def test_report_is_cached(monkeypatch, on_accelerator, fake_probe):
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    calls = fake_probe(
        {
            name: {"imports": True, "runs": None, "error": None}
            for name, _ in hw._ACCELERATOR_PACKAGES
        }
    )

    hw.get_accelerator_report(refresh = True)
    first = calls["n"]
    hw.get_accelerator_report()
    hw.get_accelerator_report()
    assert calls["n"] == first, "the report spawned another probe on a cache hit"


def test_cached_report_is_not_shared_by_reference(on_accelerator, fake_probe):
    # Callers mutate response bodies; the cache must not be editable through them.
    fake_probe({"xformers": {"imports": True, "runs": True, "error": None}})
    first = hw.get_accelerator_report(refresh = True)
    first["packages"]["xformers"]["version"] = "tampered"
    assert hw.get_accelerator_report()["packages"]["xformers"]["version"] != "tampered"


# -- endpoint wiring (ast, so it runs without starting the app) ------------------------------------


def test_hardware_endpoint_gates_the_report_behind_its_own_flag():
    source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "get_hardware_info"
    )
    body = ast.get_source_segment(source, node)
    # Its own flag, not include_details: the detail path is also read by Export, Video and
    # onboarding, and this one spawns an interpreter.
    assert "include_accelerators" in body
    before, marker, after = body.partition("if include_accelerators:")
    assert marker, "include_accelerators branch went missing"
    assert "get_accelerator_report()" in after
    assert "get_accelerator_report()" not in before
    assert '"accelerators"' in after
    assert '"accelerators"' not in before


@pytest.mark.parametrize(
    "device, is_rocm, label",
    [
        (lambda: hw.DeviceType.MLX, False, "Apple Silicon"),
        (lambda: hw.DeviceType.CPU, False, "CPU-only"),
    ],
)
def test_a_host_these_kernels_cannot_run_on_is_never_degraded(monkeypatch, device, is_rocm, label):
    # bitsandbytes is a dependency on every platform but the stock build is CUDA-only, so
    # on these hosts its import failure is the design, not a broken acceleration stack.
    # Probing there pins a permanent false banner to those installs, which is how a
    # warning stops being read.
    monkeypatch.setattr(hw, "get_device", device)
    monkeypatch.setattr(hw, "IS_ROCM", is_rocm)
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")

    def must_not_run(*args, **kwargs):
        raise AssertionError(f"probed {label}, which cannot use these kernels")

    monkeypatch.setattr(hw, "_run_probe_subprocess", must_not_run)

    report = hw.get_accelerator_report(refresh = True)
    assert report["probed"] is False
    assert report["degraded"] == []
    assert report["packages"]["bitsandbytes"]["reason"] == "not used on this device"


def test_an_xpu_host_probes_the_one_package_that_runs_there(monkeypatch, fake_probe):
    """bitsandbytes has an XPU backend, and the loader gates 4-bit on
    native_kernels_ready(..., "xpu") with its own symbol set -- so a broken XPU wheel is
    something training will reach, and calling it "not used on this device" suppressed the
    warning until the first 4-bit load failed. flash-attn / xformers / torchao publish nothing
    for XPU, so they stay out."""
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.XPU)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    calls = fake_probe(
        {"bitsandbytes": {"imports": True, "runs": False, "error": "AttributeError: ..."}}
    )

    report = hw.get_accelerator_report(refresh = True)
    assert calls["names"] == ["bitsandbytes"]
    assert report["degraded"] == ["bitsandbytes"]
    assert report["packages"]["xformers"]["reason"] == "not used on this device"
    # ...and it is checked against the XPU symbol set, not the CUDA one.
    assert hw._probe_device_type() == "xpu"


def test_a_rocm_host_probes_only_what_it_can_actually_load(monkeypatch, fake_probe):
    """ROCm is the subtle one: those hosts report DeviceType.CUDA internally (there is
    deliberately no DeviceType.ROCM), so a plain device check waves them through -- but
    they are not empty either. Unsloth imports and enables flash-attn under
    DEVICE_TYPE == "hip", so a broken ROCm FlashAttention is something training will
    actually reach, and calling it "not used on this device" hides the one banner that
    explains the crash. The other three ship CUDA-only builds, where an import failure is
    the design."""
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    calls = fake_probe({"flash_attn": {"imports": False, "runs": None, "error": "OSError: boom"}})

    report = hw.get_accelerator_report(refresh = True)
    assert calls["names"] == ["flash_attn"]
    assert report["degraded"] == ["flash_attn"]
    assert report["packages"]["bitsandbytes"]["reason"] == "not used on this device"
    assert report["packages"]["xformers"]["reason"] == "not used on this device"
    assert report["packages"]["torchao"]["reason"] == "not used on this device"


_SMI_ROWS = (
    "0, GPU-aaaaaaaa-1111-2222-3333-444444444444, 9.0\n"
    "1, GPU-bbbbbbbb-5555-6666-7777-888888888888, 12.0\n"
)


def _fake_smi(
    monkeypatch,
    stdout = _SMI_ROWS,
    returncode = 0,
):
    monkeypatch.setattr(hw.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        hw.subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(returncode = returncode, stdout = stdout),
    )


@pytest.mark.parametrize(
    "mask, expected",
    [
        # No mask: the whole box is visible, and one uncovered card in it is still an
        # install this report has to call degraded.
        (None, "9.0,12.0"),
        ("1", "12.0"),
        # BOTH, in mask order: a rank landing on the second card falls back to SDPA
        # whatever the first can do, so a verdict from device 0 alone is a false all-clear.
        ("1,0", "12.0,9.0"),
        ("0,1", "9.0,12.0"),
        ("GPU-bbbbbbbb-5555-6666-7777-888888888888", "12.0"),
        ("GPU-bbbbbbbb", "12.0"),
        ("7", None),
        # CUDA stops at the first entry it cannot resolve, and so does this.
        ("1,nonsense", "12.0"),
    ],
)
def test_the_capability_follows_the_mask_the_app_runs_under(monkeypatch, mask, expected):
    """nvidia-smi lists PHYSICAL devices and ignores CUDA_VISIBLE_DEVICES. On a mixed host
    masked to the sm_120 card, reading row 0 evaluates the sm_90 op table and reports
    xFormers as working while the GPU the app can actually use has no kernel -- exactly the
    degraded state this report exists to surface. Resolved in the parent because the child
    is started with the mask cleared."""
    _fake_smi(monkeypatch)
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert hw._visible_compute_capability() == expected


def test_no_nvidia_smi_is_unknown_rather_than_a_guess(monkeypatch):
    monkeypatch.setattr(hw.shutil, "which", lambda name: None)
    assert hw._visible_compute_capability() is None
    _fake_smi(monkeypatch, stdout = "", returncode = 9)
    assert hw._visible_compute_capability() is None


def test_each_row_says_whether_it_was_probed(monkeypatch, fake_probe):
    """Report-wide "probed" cannot answer for a row. The probe set is per device, so on this
    ROCm host three of the four packages are installed and deliberately unprobed -- and with
    only the global flag the About table rendered all three "Not loading", a red badge for
    packages that are fine and that the banner does not even list."""
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    fake_probe({"flash_attn": {"imports": True, "runs": True, "error": None}})

    packages = hw.get_accelerator_report(refresh = True)["packages"]
    assert packages["flash_attn"]["probed"] is True
    for name in ("xformers", "torchao", "bitsandbytes"):
        assert packages[name]["installed"] is True
        assert packages[name]["probed"] is False


def test_a_cpu_host_is_distinguishable_from_an_opted_out_one(monkeypatch):
    # Two different unknowns: "these kernels do not apply here" and "you told us not to
    # look". Collapsing them would make the About tab lie about one of them.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setenv(hw._ACCELERATOR_PROBE_ENV, "1")
    opted_out = hw.get_accelerator_report(refresh = True)

    monkeypatch.delenv(hw._ACCELERATOR_PROBE_ENV)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    no_gpu = hw.get_accelerator_report(refresh = True)

    assert opted_out["packages"]["torchao"]["reason"] == "not probed"
    assert no_gpu["packages"]["torchao"]["reason"] == "not used on this device"


def test_a_child_that_dies_is_isolated_rather_than_erasing_the_report(monkeypatch, on_accelerator):
    # A native wheel that ABORTS the interpreter instead of raising is a failure mode this
    # probe is specifically built around -- and in the batch child it used to take the
    # whole answer down with it: every package read "could not be checked", degraded came
    # back empty, and Settings showed nothing at all for the one install that cannot load.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    seen = []

    def run(
        names,
        timeout = 180,
        status = None,
        keep_cuda = False,
    ):
        seen.append(list(names))
        if "bitsandbytes" in names:
            # bitsandbytes' child dies without answering.
            if status is not None:
                status["died"] = True
            return None
        return {name: {"imports": True, "runs": True, "error": None} for name in names}

    monkeypatch.setattr(hw, "_run_probe_subprocess", run)

    report = hw.get_accelerator_report(refresh = True)
    assert report["probed"] is True
    # The survivors keep their real answers...
    assert report["packages"]["xformers"]["imports"] is True
    assert report["packages"]["torchao"]["runs"] is True
    # ...and the one that kills its own child is reported broken, not unknown.
    assert report["degraded"] == ["bitsandbytes"]
    assert report["packages"]["bitsandbytes"]["runs"] is False
    assert "exited without answering" in report["packages"]["bitsandbytes"]["reason"]
    # The hidden-GPU group answers in one child; the bitsandbytes group is alone and its own
    # child already died, so that IS the diagnosis. No retry storm either way.
    assert seen == [["xformers", "flash_attn", "torchao"], ["bitsandbytes"]]


def test_a_probe_that_never_ran_is_still_unknown(monkeypatch, on_accelerator):
    # The other half of the same fence: a timeout or a missing script says nothing about
    # any package, and re-probing one at a time would only repeat it.
    monkeypatch.setattr(hw, "pkg_version", lambda name: "1.2.3")
    calls = {"n": 0}

    def run(
        names,
        timeout = 180,
        status = None,
        keep_cuda = False,
    ):
        calls["n"] += 1
        return None  # no status["died"]: the child never got to run

    monkeypatch.setattr(hw, "_run_probe_subprocess", run)

    report = hw.get_accelerator_report(refresh = True)
    # One child per visibility group, and neither retried.
    assert calls["n"] == 2
    assert report["probed"] is False and report["degraded"] == []
    assert report["packages"]["torchao"]["reason"] == "could not be checked"


def test_a_stable_abi_wheel_on_a_later_torch_is_not_called_a_mismatch(monkeypatch):
    # xFormers 0.0.34+ targets the PyTorch stable ABI, so a 2.10-built wheel on 2.12 is
    # the design. Diagnosing a torch mismatch there discards the REAL error (here a
    # missing DLL, the most common Windows failure that is not a version problem) and
    # blames a working pair.
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.12.1+cu128", "cuda": "12.8", "python": "3.13.2"},
    )
    error = "OSError: [WinError 126] The specified module could not be found"
    reason = hw._describe_xformers_break(
        {"torch": "2.10.0+cu128", "cuda": "12.8", "hip": None, "python": "3.10.11"}, error
    )
    assert reason == error


def test_a_compatible_local_tag_is_not_a_torch_mismatch(monkeypatch):
    # Same release, different CUDA MINOR: cu126 loads against cu128 fine, and the raw
    # string compare claimed a torch mismatch before the CUDA-major check could speak.
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.10.0+cu128", "cuda": "12.8", "python": "3.13.2"},
    )
    error = "OSError: [WinError 126] The specified module could not be found"
    reason = hw._describe_xformers_break(
        {"torch": "2.10.0+cu126", "cuda": "12.6", "hip": None, "python": "3.10.11"}, error
    )
    assert reason == error


def test_the_stable_abi_guarantee_does_not_cover_going_backwards(monkeypatch):
    monkeypatch.setattr(
        hw,
        "_running_torch",
        lambda: {"torch": "2.10.0+cu128", "cuda": "12.8", "python": "3.13.2"},
    )
    reason = hw._describe_xformers_break(
        {"torch": "2.12.0+cu128", "cuda": "12.8", "hip": None, "python": "3.10.11"}, None
    )
    assert reason is not None and "2.12.0+cu128" in reason
