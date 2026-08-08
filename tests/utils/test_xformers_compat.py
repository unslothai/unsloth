"""Tests for unsloth.xformers_compat -- the offline torch <-> xformers ABI check behind
the NVIDIA P0-1 report ("xFormers was built for PyTorch 2.10.0+cu128 and Python 3.10.11
while the app runs cu130 and Python 3.13.2, disabling its CUDA extensions").

Every version fact asserted here was read off the published wheels:
``pypi.org/pypi/xformers/<version>/json`` for the declared ``Requires-Dist``, and the
wheel's own ``xformers/cpp_lib.json`` for what it was actually compiled against.

Pure stdlib module under test -- no torch, no xformers, no GPU needed.
"""

import importlib.util
import json
import types

import pytest

from unsloth import xformers_compat as xc


# ---------------------------------------------------------------- version tables


@pytest.mark.parametrize(
    "xformers_version, torch_release",
    [
        # The old table stopped at torch 2.4; these are the releases it never covered.
        ("0.0.33", "2.9.0"),
        ("0.0.33.post1", "2.9.0"),
        ("0.0.33.post2", "2.9.1"),  # .post bumps the torch release, so it must be kept
        ("0.0.34", "2.10.0"),
        # 0.0.35 declares `torch>=2.10` but its _C is still one build against 2.10.0.
        ("0.0.35", "2.10.0"),
    ],
)
def test_expected_torch_covers_modern_releases(xformers_version, torch_release):
    assert xc.expected_torch_for_xformers(xformers_version) == torch_release


def test_declared_pin_is_absent_for_the_range_release():
    # 0.0.35 must not appear in the `==` pin table: it declares a range, so a lookup
    # there would report a pin the wheel never made.
    assert "0.0.35" not in xc.XFORMERS_TORCH_PINS
    assert xc.XFORMERS_BUILT_FOR_TORCH["0.0.35"] == "2.10.0"


@pytest.mark.parametrize(
    "torch_version, xformers_version",
    [
        ("2.9.0", "0.0.33.post1"),
        ("2.9.1", "0.0.33.post2"),
        ("2.9.1+cu128", "0.0.33.post2"),  # local tag must not defeat the lookup
        ("2.10.0", "0.0.34"),
        ("2.10.0+cu130", "0.0.34"),
    ],
)
def test_xformers_for_torch(torch_version, xformers_version):
    assert xc.xformers_for_torch(torch_version) == xformers_version


@pytest.mark.parametrize("torch_version", ["2.11.0", "2.12.0", "2.13.0"])
def test_no_xformers_release_is_built_for_torch_2_11_or_later(torch_version):
    # 0.0.35's `torch>=2.10` makes pip accept these pairings, but no wheel is compiled
    # for them. None is the honest answer; guessing 0.0.35 here would recommend the
    # exact install that produces the P0.
    assert xc.xformers_for_torch(torch_version) is None


def test_inverse_table_agrees_with_the_forward_tables():
    # TORCH_TO_XFORMERS is the inverse with posts winning; drift between the two is a
    # silent way to recommend a wheel built for a different torch.
    for torch_release, xformers_version in xc.TORCH_TO_XFORMERS.items():
        assert xc.expected_torch_for_xformers(xformers_version) == torch_release


# ---------------------------------------------------------------- version parsing


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2.10.0+cu130", "2.10.0"),
        ("2.10.0", "2.10.0"),
        ("2.11.0.dev20260101+cu130", "2.11.0"),
        ("2.10.0a0+gitabc123", "2.10.0"),
        ("", None),
        (None, None),
        ("not-a-version", None),
    ],
)
def test_normalize_release(raw, expected):
    assert xc.normalize_release(raw) == expected


def test_normalize_release_with_post_keeps_the_post():
    # 0.0.33 -> torch 2.9.0 but 0.0.33.post2 -> torch 2.9.1, so dropping the post here
    # would silently answer for the wrong wheel.
    assert xc.normalize_release_with_post("0.0.33.post2") == "0.0.33.post2"
    assert xc.normalize_release("0.0.33.post2") == "0.0.33"


@pytest.mark.parametrize(
    "torch_version, major",
    [
        ("2.10.0+cu130", 13),
        ("2.10.0+cu128", 12),
        ("2.4.1+cu118", 11),
        ("2.10.0", None),
        ("2.10.0+cpu", None),
        ("2.10.0+rocm6.4", None),
    ],
)
def test_cuda_major_from_torch_version(torch_version, major):
    assert xc.cuda_major_from_torch_version(torch_version) == major


@pytest.mark.parametrize(
    "raw, formatted",
    [(1208, "12.8"), (1300, "13.0"), (1126, "11.26"), (None, None), (True, None)],
)
def test_format_build_cuda(raw, formatted):
    # cpp_lib.json stores major * 100 + minor. xformers' own exception prints the raw
    # integer ("with CUDA 1208"), which is what makes its message unreadable.
    assert xc.format_build_cuda(raw) == formatted


# ---------------------------------------------------------------- build metadata


def _cpp_lib(torch = "2.10.0+cu128", cuda = 1208, python = "3.10.11", hip = None):
    """The real shape of an xformers wheel's cpp_lib.json (verified against 0.0.34)."""
    return {
        "version": {"cuda": cuda, "hip": hip, "torch": torch, "python": python},
        "env": {"XFORMERS_PACKAGE_FROM": "wheel-v0.0.34"},
    }


def test_build_metadata_is_read_from_disk_without_importing_xformers(tmp_path, monkeypatch):
    package = tmp_path / "xformers"
    package.mkdir()
    (package / "cpp_lib.json").write_text(json.dumps(_cpp_lib()), encoding = "utf-8")
    (package / "__init__.py").write_text("raise AssertionError('must not be imported')")

    spec = importlib.util.spec_from_file_location(
        "xformers", package / "__init__.py", submodule_search_locations = [str(package)]
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: spec if name == "xformers" else None
    )
    # The __init__ above blows up if executed; getting a result proves we only located it.
    assert xc.xformers_build_metadata() == _cpp_lib()
    assert xc.xformers_build_summary() == {
        "torch": "2.10.0+cu128",
        "cuda": "12.8",
        "hip": None,
        "python": "3.10.11",
    }


def test_build_metadata_is_none_when_xformers_is_absent(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert xc.xformers_build_metadata() is None
    assert xc.xformers_build_summary() is None


def test_build_metadata_survives_a_raising_find_spec(monkeypatch):
    def boom(name, *args, **kwargs):
        raise ValueError("half-removed distribution")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    # Diagnostics must never be the thing that takes the caller down.
    assert xc.xformers_build_metadata() is None


def test_build_metadata_ignores_a_malformed_cpp_lib(tmp_path, monkeypatch):
    package = tmp_path / "xformers"
    package.mkdir()
    (package / "cpp_lib.json").write_text("{ not json", encoding = "utf-8")
    spec = types.SimpleNamespace(
        submodule_search_locations = [str(package)], origin = str(package / "__init__.py")
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: spec)
    assert xc.xformers_build_metadata() is None


# ---------------------------------------------------------------- mismatch detection


def test_the_nvidia_p0_is_detected():
    # NVIDIA QA P0-1 exactly: the managed Windows wheel (xformers 0.0.34/0.0.35, built
    # for torch 2.10.0+cu128 on Python 3.10.11) beside a cu130 / Python 3.13.2 runtime.
    reason = xc.describe_xformers_mismatch(
        torch_version = "2.10.0+cu130",
        torch_cuda = "13.0",
        xformers_version = "0.0.34",
        build_metadata = _cpp_lib(),
        python_version = "3.13.2",
    )
    assert reason is not None
    assert "2.10.0+cu128" in reason
    assert "3.10.11" in reason
    assert "2.10.0+cu130" in reason
    assert "3.13.2" in reason


def test_torch_release_mismatch_is_detected():
    reason = xc.describe_xformers_mismatch(
        torch_version = "2.11.0+cu128",
        xformers_version = "0.0.35",
        build_metadata = _cpp_lib(torch = "2.10.0+cu128"),
    )
    assert reason is not None
    assert "2.10.0+cu128" in reason and "2.11.0+cu128" in reason


def test_matching_build_reports_nothing():
    assert (
        xc.describe_xformers_mismatch(
            torch_version = "2.10.0+cu128",
            torch_cuda = "12.8",
            xformers_version = "0.0.34",
            build_metadata = _cpp_lib(),
            python_version = "3.13.2",
        )
        is None
    )


def test_python_difference_alone_is_not_a_mismatch():
    # The wheels are abi3/none-tagged and _C is loaded via torch.ops.load_library, not
    # the CPython ABI, so 3.10-built kernels run fine on 3.13. Flagging this would send
    # every Studio user chasing the wrong thing.
    assert (
        xc.describe_xformers_mismatch(
            torch_version = "2.10.0+cu128",
            build_metadata = _cpp_lib(python = "3.10.11"),
            python_version = "3.13.2",
        )
        is None
    )


def test_cuda_minor_difference_alone_is_not_a_mismatch():
    # CUDA minor version compatibility: a cu126-built wheel loads against a cu128 torch.
    assert (
        xc.describe_xformers_mismatch(
            torch_version = "2.10.0+cu128",
            torch_cuda = "12.8",
            build_metadata = _cpp_lib(cuda = 1206, torch = "2.10.0+cu126"),
        )
        is None
    )


def test_cuda_major_mismatch_is_detected_without_a_local_tag():
    # Conda/source torch has no "+cuXXX", so torch.version.cuda is the only signal.
    reason = xc.describe_xformers_mismatch(
        torch_version = "2.10.0",
        torch_cuda = "13.0",
        build_metadata = _cpp_lib(torch = "2.10.0", cuda = 1208),
    )
    assert reason is not None
    assert "(CUDA 13.x)" in reason


def test_unknown_runtime_reports_nothing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert xc.describe_xformers_mismatch(torch_version = None) is None
    assert xc.describe_xformers_mismatch(torch_version = "garbage") is None


def test_pin_fallback_when_the_wheel_ships_no_build_metadata(monkeypatch):
    # Source/editable installs have no cpp_lib.json; the declared pin still catches a
    # wholesale torch-release mismatch.
    monkeypatch.setattr(xc, "xformers_build_metadata", lambda: None)
    monkeypatch.setattr(xc, "declared_torch_pin", lambda version = None: "2.9.1")
    reason = xc.describe_xformers_mismatch(
        torch_version = "2.10.0+cu128", xformers_version = "0.0.33.post2"
    )
    assert reason is not None
    assert "2.9.1" in reason


def test_pin_fallback_stays_quiet_when_the_pin_matches(monkeypatch):
    monkeypatch.setattr(xc, "xformers_build_metadata", lambda: None)
    monkeypatch.setattr(xc, "declared_torch_pin", lambda version = None: "2.10.0")
    assert (
        xc.describe_xformers_mismatch(
            torch_version = "2.10.0+cu128", xformers_version = "0.0.34"
        )
        is None
    )


def test_declared_pin_falls_back_to_the_table_for_a_different_version():
    # Asking about a version other than the resident one must use the table, not the
    # resident METADATA, which describes a different wheel entirely.
    assert xc.declared_torch_pin("0.0.30") == "2.7.0"
