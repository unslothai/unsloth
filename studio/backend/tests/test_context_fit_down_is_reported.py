# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A context the fit shortened must say so, naming the value that was asked for.

From a user report. They set Context Length to 512000, the load came up smaller, and
they hand-added ``--rope-scaling yarn --yarn-orig-ctx 32768`` to Extra Arguments --
what someone does when a setting looks ignored. Nothing in the backend said the
context had been reduced: the placement line reports ``context: <effective>`` only, so
the requested value appears nowhere and a fit-down is indistinguishable from a lost
setting.

Structural, like test_gpu_memory_mode.py's checks on the same region: load_model is
not callable in a unit test, so these read the source of the branch. They pin the
condition and the two values in the message, which is what makes the line useful.
"""

import inspect

import pytest

from core.inference import llama_cpp as llama_cpp_module

pytest.importorskip("fastapi")


def _load_model_source() -> str:
    return inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)


def _fit_down_block() -> str:
    src = _load_model_source()
    at = src.find("Context length was reduced from the requested")
    assert at != -1, "a shortened context must be reported; see the report in this file's docstring"
    return src[max(0, at - 700) : at + 500]


def test_the_report_is_guarded_by_an_actual_reduction():
    """Only when the fit really shortened it. requested_ctx is 0 for Auto, and
    effective_ctx is 0 where the context is not yet known, so an unguarded compare
    would warn on loads that reduced nothing."""
    assert "0 < effective_ctx < requested_ctx" in _fit_down_block()


def test_it_names_both_the_requested_and_the_effective_value():
    """The whole point: the placement line already prints the effective context. A
    message that repeats only that adds nothing a user could act on."""
    block = _fit_down_block()
    assert "requested_ctx," in block and "effective_ctx," in block


def test_it_is_a_warning_not_an_info():
    """The reporting user's entire log was info-level, which is why a silently
    shortened window read as a broken setting."""
    assert "logger.warning(" in _fit_down_block()


def test_it_tells_the_user_what_to_do():
    block = _fit_down_block()
    assert any(
        word in block for word in ("Lower the context", "free VRAM", "smaller")
    ), "a reduction the user cannot act on is only half the message"


def test_it_sits_with_the_placement_decision_it_explains():
    """Beside the GPUs-free line, so one read of the log shows the placement and the
    context it forced, rather than the two being pages apart."""
    src = _load_model_source()
    assert src.find("GPUs free:") < src.find("Context length was reduced from the requested")
