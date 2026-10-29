# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fixtures copy trl/import_utils.py (0.26.2, 0.27.0) and transformers' helper (5.2.0, 5.3.0)."""

import types

import pytest

from unsloth.import_fixes import _coerce_trl_available_checks

INSTALLED = {"torch": "2.9.0"}


def _transformers_5_3_helper(pkg_name, return_version = False):
    exists = pkg_name in INSTALLED
    return (exists, INSTALLED.get(pkg_name, "N/A")) if return_version else (exists, None)


def _transformers_5_2_helper(pkg_name, return_version = False):
    exists = pkg_name in INSTALLED
    return (exists, INSTALLED.get(pkg_name, "N/A")) if return_version else exists


def _trl_0_27(helper):
    tiu = types.ModuleType("trl.import_utils")
    tiu._is_package_available = helper
    exec(
        "def is_weave_available():\n    return _is_package_available('weave')\n"
        "def is_torch_available():\n    return _is_package_available('torch')\n"
        "def vllm_version():\n    return _is_package_available('torch', return_version = True)\n",
        tiu.__dict__,
    )
    return tiu


def test_function_style_checks_return_bools():
    tiu = _trl_0_27(_transformers_5_3_helper)
    assert tiu.is_weave_available() == (False, None)
    is_weave_available = (
        tiu.is_weave_available
    )  # bound before the fix, like `from trl.import_utils import ...`
    _coerce_trl_available_checks(tiu)
    assert tiu.is_weave_available() is False
    assert is_weave_available() is False
    assert tiu.is_torch_available() is True
    assert tiu.vllm_version() == (True, "2.9.0")


def test_idempotent():
    tiu = _trl_0_27(_transformers_5_3_helper)
    _coerce_trl_available_checks(tiu)
    wrapped = tiu._is_package_available
    _coerce_trl_available_checks(tiu)
    assert tiu._is_package_available is wrapped
    assert tiu._is_package_available.__wrapped__ is _transformers_5_3_helper


@pytest.mark.parametrize("helper", [_transformers_5_2_helper, None])
def test_bool_helpers_are_left_alone(helper):
    # transformers <= 5.2 and TRL >= 0.29 (its own helper) already return bools
    tiu = _trl_0_27(helper or (lambda pkg_name, return_version = False: pkg_name in INSTALLED))
    before = tiu._is_package_available
    _coerce_trl_available_checks(tiu)
    assert tiu._is_package_available is before
    assert tiu.is_weave_available() is False


def test_cached_flags_still_coerced():
    tiu = types.ModuleType("trl.import_utils")
    tiu._weave_available = (False, None)
    tiu._torch_available = (True, None)
    _coerce_trl_available_checks(tiu)
    assert tiu._weave_available is False and tiu._torch_available is True
