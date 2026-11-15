# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""torchvision 0.29.1+ declares a torch range (stable ABI): 0.29.2 shipped with torch 2.15.0 and requires
torch>=2.14.0, with no torchvision 0.30. The check must accept that pair and still refuse real mismatches."""

import types

import pytest

from unsloth import import_fixes


def _env(monkeypatch, torch_version, torchvision_version, requires):
    versions = {"torch": torch_version, "torchvision": torchvision_version}
    monkeypatch.delenv("UNSLOTH_SKIP_TORCHVISION_CHECK", raising = False)
    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", lambda name, *a, **k: object())
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: versions[name])
    monkeypatch.setattr(
        import_fixes,
        "importlib_distribution",
        lambda name: types.SimpleNamespace(requires = requires),
    )
    probed = []
    monkeypatch.setattr(
        import_fixes, "_probe_torchvision_binary", lambda *args, **kwargs: probed.append(args)
    )
    return probed


@pytest.mark.parametrize("torch_version", ["2.15.0", "2.15.0+cu132", "2.15.1", "2.14.1"])
def test_stable_abi_torchvision_is_accepted(monkeypatch, torch_version):
    probed = _env(monkeypatch, torch_version, "0.29.2", ["torch>=2.14.0", "numpy"])
    import_fixes.torchvision_compatibility_check()
    assert probed and probed[0][2] == (0, 29, 2)


def test_range_that_excludes_torch_falls_back_to_the_formula(monkeypatch):
    # torch 2.16 against a torchvision that only declares >=2.14,<2.16: formula wants 0.31.
    _env(monkeypatch, "2.16.0", "0.29.2", ["torch<2.16,>=2.14.0"])
    with pytest.raises(ImportError, match = r"torchvision>=0\.31\.0"):
        import_fixes.torchvision_compatibility_check()


def test_exact_pin_mismatch_still_raises(monkeypatch):
    # Pre stable-ABI torchvision 0.28.0 pins torch==2.13.0; torch 2.14.1 needs 0.29.
    _env(monkeypatch, "2.14.1", "0.28.0", ["torch==2.13.0"])
    with pytest.raises(ImportError, match = r"torchvision>=0\.29\.0"):
        import_fixes.torchvision_compatibility_check()


def test_exact_pin_match_is_unchanged(monkeypatch):
    probed = _env(monkeypatch, "2.9.0", "0.24.0", ["torch==2.9.0"])
    import_fixes.torchvision_compatibility_check()
    assert probed


def test_no_metadata_requirement_uses_the_formula(monkeypatch):
    _env(monkeypatch, "2.15.0", "0.29.2", [])
    with pytest.raises(ImportError, match = r"torchvision>=0\.30\.0"):
        import_fixes.torchvision_compatibility_check()
