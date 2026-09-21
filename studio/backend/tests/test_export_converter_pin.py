# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The GGUF converter pin is Unsloth's own routing, so it is scoped to one conversion.

Setting UNSLOTH_LLAMA_CPP_SCRIPTS_DIR and leaving it there cost two controls that live
in unsloth_zoo: the variable outranks UNSLOTH_LLAMA_CPP_CONVERTER_TAG, so the escape
hatch the unsupported-architecture message names did nothing for a Studio export, and
trust is read off the variable, so a converter Studio had just downloaded looked
user-pinned and UNSLOTH_CONVERTER_SCAN_STRICT stopped failing exports."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Reuse the absolute-paths stub harness: loads core/export/export.py without torch/unsloth.
from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)

_SCRIPTS_DIR = "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"
_CONVERTER_TAG = "UNSLOTH_LLAMA_CPP_CONVERTER_TAG"


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    monkeypatch.delenv(_SCRIPTS_DIR, raising = False)
    monkeypatch.delenv(_CONVERTER_TAG, raising = False)
    return _load_module(
        "test_core_export_backend_converter_pin", "core/export/export.py", monkeypatch
    )


def _zoo(
    monkeypatch,
    *,
    internal_pin = True,
    incomplete = False,
):
    """The installed unsloth_zoo, with the pieces the pin looks for."""
    llama_cpp = sys.modules["unsloth_zoo.llama_cpp"]
    calls = {"internal": [], "incomplete": []}

    @contextlib.contextmanager
    def _internal_scripts_dir_pin(folder):
        calls["internal"].append(folder)
        existing = os.environ.get(_SCRIPTS_DIR)
        if existing is None:
            os.environ[_SCRIPTS_DIR] = folder
        try:
            yield
        finally:
            if existing is None:
                os.environ.pop(_SCRIPTS_DIR, None)
            else:
                os.environ[_SCRIPTS_DIR] = existing

    if internal_pin:
        monkeypatch.setattr(
            llama_cpp, "internal_scripts_dir_pin", _internal_scripts_dir_pin, raising = False
        )
    else:
        monkeypatch.delattr(llama_cpp, "internal_scripts_dir_pin", raising = False)

    def _converter_dir_is_incomplete(folder):
        calls["incomplete"].append(folder)
        return incomplete

    monkeypatch.setattr(
        llama_cpp, "_converter_dir_is_incomplete", _converter_dir_is_incomplete, raising = False
    )
    return llama_cpp, calls


def test_pin_is_scoped_to_the_conversion(monkeypatch):
    mod = _export_mod(monkeypatch)
    llama_cpp, calls = _zoo(monkeypatch)

    with mod._llama_cpp_scripts_pin():
        assert os.environ[_SCRIPTS_DIR] == llama_cpp.LLAMA_CPP_DEFAULT_DIR
    # The leak is the bug: every later export, and the MLX path's own internal pin,
    # read a variable nobody set deliberately.
    assert _SCRIPTS_DIR not in os.environ
    assert calls["internal"] == [llama_cpp.LLAMA_CPP_DEFAULT_DIR]


def test_pin_goes_through_the_internal_helper(monkeypatch):
    """Not a raw os.environ write: unsloth_zoo grants the strict-scan exemption to a
    pin the USER set, and marking this one internal is what keeps
    UNSLOTH_CONVERTER_SCAN_STRICT=1 able to refuse a converter Studio downloaded."""
    mod = _export_mod(monkeypatch)
    llama_cpp, calls = _zoo(monkeypatch)

    with mod._llama_cpp_scripts_pin():
        pass
    assert calls["internal"] == [llama_cpp.LLAMA_CPP_DEFAULT_DIR]


def test_a_converter_tag_is_left_in_force(monkeypatch):
    """The pin outranks the tag, so pinning at all made `UNSLOTH_LLAMA_CPP_CONVERTER_TAG=b9999`
    a knob that changes nothing."""
    mod = _export_mod(monkeypatch)
    _llama_cpp, calls = _zoo(monkeypatch)
    monkeypatch.setenv(_CONVERTER_TAG, "b9999")

    with mod._llama_cpp_scripts_pin():
        assert _SCRIPTS_DIR not in os.environ
    assert calls["internal"] == []


def test_a_pin_the_user_set_is_untouched(monkeypatch):
    mod = _export_mod(monkeypatch)
    _llama_cpp, _calls = _zoo(monkeypatch)
    monkeypatch.setenv(_SCRIPTS_DIR, "/home/me/llama.cpp")

    with mod._llama_cpp_scripts_pin():
        assert os.environ[_SCRIPTS_DIR] == "/home/me/llama.cpp"
    assert os.environ[_SCRIPTS_DIR] == "/home/me/llama.cpp"


def test_an_incomplete_install_is_not_pinned(monkeypatch):
    """An entrypoint with no conversion/ beside it cannot run, and pinning it is what
    stops the staged resolver from fetching a co-versioned set that can."""
    mod = _export_mod(monkeypatch)
    _llama_cpp, calls = _zoo(monkeypatch, incomplete = True)

    with mod._llama_cpp_scripts_pin():
        assert _SCRIPTS_DIR not in os.environ
    assert calls["internal"] == []


def test_older_zoo_still_gets_a_scoped_pin(monkeypatch):
    """No internal-pin helper to call, so the variable is scoped by hand. The strict-scan
    exemption cannot be avoided on those builds, but the leak can."""
    mod = _export_mod(monkeypatch)
    llama_cpp, _calls = _zoo(monkeypatch, internal_pin = False)

    with mod._llama_cpp_scripts_pin():
        assert os.environ[_SCRIPTS_DIR] == llama_cpp.LLAMA_CPP_DEFAULT_DIR
    assert _SCRIPTS_DIR not in os.environ


def test_the_pin_unwinds_when_the_conversion_raises(monkeypatch):
    mod = _export_mod(monkeypatch)
    _llama_cpp, _calls = _zoo(monkeypatch)

    with contextlib.suppress(RuntimeError):
        with mod._llama_cpp_scripts_pin():
            raise RuntimeError("conversion failed")
    assert _SCRIPTS_DIR not in os.environ


def test_zoo_without_the_pin_warns_once_and_runs(monkeypatch):
    mod = _export_mod(monkeypatch)
    llama_cpp = sys.modules["unsloth_zoo.llama_cpp"]
    monkeypatch.delattr(llama_cpp, "_resolve_local_convert_script", raising = False)
    warnings = []
    monkeypatch.setattr(
        mod.logger, "warning", lambda message, *args, **kwargs: warnings.append(message)
    )
    monkeypatch.setattr(mod, "_LLAMA_CPP_SCRIPTS_WARNING_EMITTED", False)

    ran = 0
    for _ in range(2):
        with mod._llama_cpp_scripts_pin():
            ran += 1
        assert _SCRIPTS_DIR not in os.environ
    assert ran == 2
    assert len(warnings) == 1
    assert _SCRIPTS_DIR in warnings[0]
