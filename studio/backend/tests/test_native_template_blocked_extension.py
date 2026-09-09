# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A native extension the loader refuses, arriving at the native chat template lookup.

Smart App Control blocked ``_sentencepiece.cp313-win_amd64.pyd`` on a reporting machine
(Code Integrity event 3077, publisher Unknown, signature type None). The field report
records that it "failed silently": no error surfaced and inference continued. That is
this catch. ``AutoTokenizer.from_pretrained`` raises the ImportError from inside the
tokenizer machinery, everything is caught, a warning is logged among many, None is
returned, and the model goes on generating under a substituted chat template. The
symptom is wrong prompt formatting with nothing naming the cause.

The template still falls back, because nothing here can invent a template that is not
loadable. What changes is that the line naming the blocked file is an error the operator
can act on rather than one warning in a stream of them.
"""

import logging
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import chat_template_helpers as helpers  # noqa: E402


def _blocked(winerror = 577):
    """What a refused module load looks like by the time it reaches this catch."""
    error = ImportError(
        "DLL load failed while importing _sentencepiece: "
        "Windows cannot verify the digital signature for this file."
    )
    error.winerror = winerror
    return error


@pytest.mark.parametrize("winerror", [225, 577, 1260])
def test_every_refusal_code_is_recognised(winerror: int):
    """577 is App Control and Smart App Control, 225 is an antivirus blocking on access,
    1260 is AppLocker or SRP. All three mean the same thing to a user: the file is there
    and Windows will not load it."""
    assert helpers._looks_like_a_blocked_import(_blocked(winerror)) is True


def test_a_wrapped_refusal_is_still_recognised():
    """transformers raises its own error from the original, so the winerror is a link or
    two down the chain rather than on the exception that arrives here."""
    original = _blocked()
    try:
        try:
            raise original
        except ImportError as exc:
            raise RuntimeError("could not build a tokenizer") from exc
    except RuntimeError as wrapper:
        assert helpers._looks_like_a_blocked_import(wrapper) is True


def test_an_ordinary_failure_is_not_mistaken_for_a_block():
    """The common case by far is a network error or a gated repository, and calling that
    a security block would send every operator looking at their antivirus."""
    assert helpers._looks_like_a_blocked_import(OSError("connection reset")) is False
    assert helpers._looks_like_a_blocked_import(ValueError("no such revision")) is False
    assert helpers._looks_like_a_blocked_import(None) is False


def test_a_missing_optional_package_is_not_a_block():
    """An ImportError naming a package nobody installed is not a refusal, and it already
    has a good message of its own."""
    assert (
        helpers._looks_like_a_blocked_import(ImportError("No module named 'flash_attn'")) is False
    )


def test_a_blocked_extension_is_reported_at_error_level(monkeypatch, caplog):
    """The behaviour that changes. Before this the line was a warning identical in shape
    to a failed fetch, and the model kept generating under a substituted template with
    nothing to point at."""
    model_info: dict = {}

    class _RaisingAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise _blocked()

    module = type(sys)("transformers")
    module.AutoTokenizer = _RaisingAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)

    with caplog.at_level(logging.WARNING):
        result = helpers.resolve_native_chat_template(
            model_info,
            "unsloth/Qwen3.5-2B",
        )

    assert result is None
    blocked = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert blocked, "a refused extension must not read as an ordinary fetch failure"
    assert "extension" in blocked[0].getMessage()
    # Still not cached: a caching of False would pin the substituted template for the
    # rest of the session, so allowing the file and restarting would not be enough.
    assert "native_chat_template" not in model_info


def test_an_ordinary_failure_still_logs_a_warning(monkeypatch, caplog):
    """The other side of the branch, so the escalation cannot swallow the common case."""
    model_info: dict = {}

    class _RaisingAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("connection reset by peer")

    module = type(sys)("transformers")
    module.AutoTokenizer = _RaisingAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", module)

    with caplog.at_level(logging.WARNING):
        assert helpers.resolve_native_chat_template(model_info, "unsloth/Qwen3.5-2B") is None

    assert [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
