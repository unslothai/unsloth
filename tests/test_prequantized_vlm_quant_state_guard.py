# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the pre-quantized quant_state guard (unsloth #9867, #10010, #10017, #10276).

transformers 5.4.0 (PR #44300) merged a submodule's renamings into the parent without
re-scoping them, so a composite checkpoint's bitsandbytes sidecars renamed to keys the
model does not have and were dropped and every quantized Linear loads with
``quant_state = None``. Fixed upstream by PR #45567.

The guard asks the INSTALLED transformers whether it scopes submodule renamings, never a
version: main reported ``5.3.0.dev0`` at the commit that introduced the defect and
``5.6.0.dev0`` at the one that fixed it.

Loaded in isolation, no torch, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parent.parent
_IMPORT_FIXES_PATH = _ROOT / "unsloth" / "import_fixes.py"
_PYPROJECT_PATH = _ROOT / "pyproject.toml"

# Releases that carry the defect, and releases that do not. Used only to document the
# window in the warning text and to keep the pyproject specifier honest -- never as the
# detector's input.
BROKEN_RELEASES = ["5.4.0", "5.5.0", "5.5.1", "5.5.2", "5.5.3", "5.5.4"]
GOOD_RELEASES = ["4.57.6", "5.2.0", "5.3.0", "5.6.0", "5.6.2", "5.14.1", "5.17.0"]


def _load_import_fixes():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_quant_state_under_test", _IMPORT_FIXES_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def import_fixes():
    return _load_import_fixes()


# Fake transformers builds. Each one is a shape upstream really shipped.


def _install(
    monkeypatch,
    *,
    conversion_mapping = None,
    core_model_loading = None,
):
    """Put a fake `transformers` in sys.modules for the duration of one test."""
    fake = types.ModuleType("transformers")
    # A real transformers may already be imported, and `from transformers import X` falls
    # back to sys.modules["transformers.X"], so every submodule entry is pinned here --
    # None makes the import raise, which is what "this build does not have it" means.
    for name, submodule in (
        ("conversion_mapping", conversion_mapping),
        ("core_model_loading", core_model_loading),
    ):
        if submodule is not None:
            setattr(fake, name, submodule)
        monkeypatch.setitem(sys.modules, f"transformers.{name}", submodule)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _build_4x(monkeypatch):
    """transformers 4.x: no conversion_mapping module at all, so nothing to correct."""
    _install(monkeypatch)


def _build_pre_recursion(monkeypatch):
    """5.x before 5.4.0: a conversion_mapping, but no per-submodule extraction."""
    _install(
        monkeypatch,
        conversion_mapping = _module("transformers.conversion_mapping"),
        core_model_loading = _module("transformers.core_model_loading"),
    )


def _build_broken(monkeypatch):
    """5.4.0 to 5.5.4: recursion, and no way to say where a submodule's mapping belongs."""

    def extract_weight_conversions_for_model(model):
        return []

    class WeightTransform:
        pass

    class PrefixChange:
        pass

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = extract_weight_conversions_for_model,
        ),
        core_model_loading = _module(
            "transformers.core_model_loading",
            WeightTransform = WeightTransform,
            PrefixChange = PrefixChange,
        ),
    )


def _build_fixed_model_prefix(monkeypatch):
    """5.6 to 5.9 as PR #45567 first spelled it: a `model_prefix` argument."""

    def extract_weight_conversions_for_model(model, model_prefix = ""):
        return []

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = extract_weight_conversions_for_model,
        ),
        core_model_loading = _module("transformers.core_model_loading"),
    )


def _build_fixed_with_submodel_prefix(monkeypatch):
    """The same range, reached through `PrefixChange.with_submodel_prefix`."""

    def extract_weight_conversions_for_model(model):
        return []

    class PrefixChange:
        def with_submodel_prefix(self, prefix):
            return self

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = extract_weight_conversions_for_model,
        ),
        core_model_loading = _module("transformers.core_model_loading", PrefixChange = PrefixChange),
    )


def _build_fixed_scope_prefix(monkeypatch):
    """5.10 and later: every transform carries `scope_prefix`."""

    def extract_weight_conversions_for_model(model):
        return []

    class WeightTransform:
        scope_prefix = None

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = extract_weight_conversions_for_model,
        ),
        core_model_loading = _module(
            "transformers.core_model_loading", WeightTransform = WeightTransform
        ),
    )


def _build_broken_without_core_model_loading(monkeypatch):
    """The defect with `core_model_loading` unimportable: defensive, not observed.

    Measured on ten real releases both modules need torch and fail together, so the
    signature check is what would have to carry the answer if upstream ever split them.
    """

    def extract_weight_conversions_for_model(model):
        return []

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = extract_weight_conversions_for_model,
        ),
        core_model_loading = None,
    )


def _build_fixed_without_core_model_loading(monkeypatch):
    """The same split, fixed: `model_prefix` alone is enough to clear the build."""

    def extract_weight_conversions_for_model(model, model_prefix = ""):
        return []

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = extract_weight_conversions_for_model,
        ),
        core_model_loading = None,
    )


def _build_no_transformers(monkeypatch):
    """transformers is not installed."""
    for name in (
        "transformers",
        "transformers.conversion_mapping",
        "transformers.core_model_loading",
    ):
        monkeypatch.setitem(sys.modules, name, None)


DEFECTIVE_BUILDS = [_build_broken, _build_broken_without_core_model_loading]
HEALTHY_BUILDS = [
    _build_4x,
    _build_pre_recursion,
    _build_fixed_model_prefix,
    _build_fixed_with_submodel_prefix,
    _build_fixed_scope_prefix,
    _build_fixed_without_core_model_loading,
    _build_no_transformers,
]


# The probe


@pytest.mark.parametrize("build", DEFECTIVE_BUILDS, ids = lambda b: b.__name__)
def test_defective_build_is_detected(import_fixes, monkeypatch, build):
    build(monkeypatch)
    assert import_fixes._transformers_rescopes_submodule_prefix_renamings() is False
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state() is True


@pytest.mark.parametrize("build", HEALTHY_BUILDS, ids = lambda b: b.__name__)
def test_healthy_build_is_not_flagged(import_fixes, monkeypatch, build):
    build(monkeypatch)
    assert import_fixes._transformers_rescopes_submodule_prefix_renamings() is True
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state() is False


def test_an_unrecognisable_build_is_left_alone(import_fixes, monkeypatch):
    """Guessing wrong towards "broken" would warn every future build. Answer healthy."""

    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("no introspection here")

    _install(
        monkeypatch,
        conversion_mapping = _module(
            "transformers.conversion_mapping",
            extract_weight_conversions_for_model = Hostile(),
        ),
        core_model_loading = _module("transformers.core_model_loading"),
    )
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state() is False


def test_the_detector_never_reads_a_version(import_fixes, monkeypatch):
    """Main was 5.3.0.dev0 when broken and 5.6.0.dev0 when fixed: the answer must not
    move when the reported version does."""
    for reported in ("5.3.0.dev0", "5.6.0.dev0", "5.17.0", "4.57.6", "not-a-version"):
        monkeypatch.setattr(
            import_fixes, "importlib_version", lambda name, reported = reported: reported
        )
        _build_broken(monkeypatch)
        assert import_fixes._transformers_drops_prequantized_vlm_quant_state() is True
        _build_fixed_scope_prefix(monkeypatch)
        assert import_fixes._transformers_drops_prequantized_vlm_quant_state() is False


def test_a_git_main_build_carrying_the_defect_is_warned(import_fixes, monkeypatch, caplog):
    """5.3.0.dev0 with the defect: the old version window let this through silently."""
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.3.0.dev0")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    _build_broken(monkeypatch)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" in caplog.text


def test_a_fixed_nightly_is_not_told_it_is_broken(import_fixes, monkeypatch, caplog):
    """5.6.0.dev0 with the fix: the old version window warned this one falsely."""
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.6.0.dev0")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    _build_fixed_scope_prefix(monkeypatch)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text


# The warning


def _uninstall_runtime_repair(import_fixes, monkeypatch):
    """Put the live attribute back to an unpatched function.

    The check is silent once the runtime repair is installed, which is the point: the
    repair covers exactly these releases, so advising a downgrade would contradict it.
    These tests are about the message shown when the repair is NOT in effect, so they
    have to say so rather than depend on whether an earlier test installed it.

    A missing transformers is not an error here: nothing can have installed the repair,
    so there is nothing to undo, and the caller goes on to install a stand-in module.
    This file runs with pytest alone.
    """
    try:
        from transformers import conversion_mapping
    except Exception:
        return

    current = conversion_mapping.get_model_conversion_mapping
    while getattr(current, import_fixes._COMPOSITE_PREFIX_RENAMING_FLAG, False):
        unwrapped = getattr(current, "__wrapped__", None)
        if unwrapped is None:
            break
        current = unwrapped
    monkeypatch.setattr(conversion_mapping, "get_model_conversion_mapping", current)


def test_warning_names_the_cause_and_the_remedy(import_fixes, monkeypatch, caplog):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.5.4")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    # Uninstall first: `_build_broken` swaps in a stand-in conversion_mapping module that
    # has no `get_model_conversion_mapping` at all, and the uninstall reads that attribute.
    _uninstall_runtime_repair(import_fixes, monkeypatch)
    _build_broken(monkeypatch)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    text = caplog.text
    assert "5.5.4" in text
    assert "quant_state" in text
    # It must say the checkpoint is fine: the reported threads told users to regenerate it.
    assert "not be regenerated" in text.lower()
    assert "transformers>=5.6.0" in text
    # And it must not tell the user to do something this repository forbids without
    # saying how: unsloth still caps transformers at 5.5.0.
    assert "--no-deps" in text


def test_warning_silent_on_a_healthy_build(import_fixes, monkeypatch, caplog):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.17.0")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    _build_fixed_scope_prefix(monkeypatch)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text


def test_env_var_silences_the_warning(import_fixes, monkeypatch, caplog):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.5.4")
    monkeypatch.setenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", "1")
    _build_broken(monkeypatch)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text


def test_missing_transformers_never_raises(import_fixes, monkeypatch, caplog):
    def _boom(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(import_fixes, "importlib_version", _boom)
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    _build_no_transformers(monkeypatch)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text


def test_the_check_is_warn_only(import_fixes, monkeypatch):
    """A text-only or unquantized run must not be broken by this."""
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    for build in DEFECTIVE_BUILDS + HEALTHY_BUILDS:
        build(monkeypatch)
        assert import_fixes.check_transformers_prequantized_vlm_quant_state() is None


# pyproject


def _transformers_requirements():
    data = tomllib.loads(_PYPROJECT_PATH.read_text(encoding = "utf-8"))
    found = []
    for group in (data.get("project", {}).get("optional-dependencies", {}) or {}).values():
        for raw in group:
            try:
                req = Requirement(raw)
            except Exception:
                continue
            if req.name == "transformers":
                found.append(req)
    for raw in data.get("project", {}).get("dependencies", []) or []:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.name == "transformers":
            found.append(req)
    return found


def test_pyproject_still_allows_a_working_version():
    """unsloth_zoo caps transformers at 5.5.0 and `loader.py` needs >= 5.5.0 for Gemma 4,
    so excluding 5.5.0 here resolves DOWN to 5.3.0 and takes Gemma 4 with it."""
    reqs = _transformers_requirements()
    assert reqs, "no transformers requirement found in pyproject.toml"
    for req in reqs:
        assert any(
            req.specifier.contains(version, prereleases = True) for version in GOOD_RELEASES
        ), f"pyproject leaves no working transformers at all: {req}"
        assert req.specifier.contains("5.5.0", prereleases = True), (
            f"pyproject excludes 5.5.0 while unsloth_zoo still caps transformers at "
            f"5.5.0 and Gemma 4 requires >= 5.5.0, so this resolves backwards: {req}"
        )


# Where the check is called from


def test_the_check_runs_after_the_torchaudio_guard():
    """The probe imports transformers, and `disable_torchaudio_if_cuda_mismatched` exists
    because reaching torchaudio before it runs takes the whole `import unsloth` down."""
    import ast

    source = (_ROOT / "unsloth" / "_gpu_init.py").read_text(encoding = "utf-8")
    order = []
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            order.append(node.value.func.id)
    assert "disable_torchaudio_if_cuda_mismatched" in order, order
    assert "check_transformers_prequantized_vlm_quant_state" in order, order
    assert order.index("check_transformers_prequantized_vlm_quant_state") > order.index(
        "disable_torchaudio_if_cuda_mismatched"
    ), "the quant_state check imports transformers and must not precede the torchaudio guard"


def test_warning_is_silent_once_the_runtime_repair_is_installed(import_fixes, monkeypatch, caplog):
    """Telling a user to downgrade away from a version we just repaired is worse than silence."""
    import logging as _logging

    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.5.4")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    # This one really does need the live module: it installs the repair onto it. Skipping
    # keeps the rest of the file runnable with pytest alone.
    transformers = pytest.importorskip("transformers")
    conversion_mapping = pytest.importorskip("transformers.conversion_mapping")
    del transformers

    import_fixes.fix_transformers_composite_prefix_renaming()
    installed = getattr(
        conversion_mapping.get_model_conversion_mapping,
        import_fixes._COMPOSITE_PREFIX_RENAMING_FLAG,
        False,
    )
    if not installed:
        pytest.skip("this transformers is outside the defect window, so the repair declines")
    with caplog.at_level(_logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text
