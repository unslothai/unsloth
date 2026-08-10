# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One log record must not be able to grow without bound.

request_failed renders the whole traceback into an "exception" field. That is a
few KB for a normal failure, but an exception whose message embeds a request body
is not: a rejected binary upload produced a single 2.2 MB line. The head (raising
frame) and the tail (exception type and message) are what a reader needs, so the
middle is dropped with a count of what went missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from loggers import config as log_config  # noqa: E402


def test_short_tracebacks_are_untouched():
    ev = {"exception": "Traceback...\nValueError: nope"}
    assert log_config.truncate_exception(dict(ev)) == ev


def test_long_traceback_is_capped():
    text = "HEAD" + ("x" * 4_000_000) + "TAILValueError: nope"
    out = log_config.truncate_exception({"exception": text})["exception"]
    assert len(out) < log_config._MAX_EXC_CHARS + 200, len(out)


def test_head_and_tail_survive():
    text = "TRACEBACK_HEAD_MARKER" + ("x" * 4_000_000) + "EXC_TAIL_MARKER"
    out = log_config.truncate_exception({"exception": text})["exception"]
    assert out.startswith("TRACEBACK_HEAD_MARKER")
    assert out.endswith("EXC_TAIL_MARKER")
    assert "chars omitted" in out


def test_the_error_field_is_capped_too():
    # request_failed logs str(exc) under "error" as well as the rendered traceback,
    # so capping only the traceback still lets the same payload through.
    text = "ERR_HEAD" + ("x" * 4_000_000) + "ERR_TAIL"
    out = log_config.truncate_exception({"error": text})["error"]
    assert len(out) < log_config._MAX_ERROR_CHARS + 200, len(out)
    assert out.startswith("ERR_HEAD")
    assert out.endswith("ERR_TAIL")


def test_a_cap_smaller_than_the_tail_is_still_enforced(monkeypatch):
    # head = cap - tail went negative, so text[:-2048] kept nearly everything.
    monkeypatch.setattr(log_config, "_MAX_EXC_CHARS", 1024)
    text = "H" * 4_000_000
    out = log_config.truncate_exception({"exception": text})["exception"]
    assert len(out) < 1024 + 200, len(out)


def test_non_string_exception_field_is_ignored():
    ev = {"exception": None}
    assert log_config.truncate_exception(dict(ev)) == ev
    ev2 = {"event": "no exception here"}
    assert log_config.truncate_exception(dict(ev2)) == ev2


def test_cap_can_be_disabled(monkeypatch):
    monkeypatch.setattr(log_config, "_MAX_EXC_CHARS", 0)
    text = "y" * 100_000
    assert log_config.truncate_exception({"exception": text})["exception"] == text


def test_processor_signature_matches_structlog():
    text = "z" * 100_000
    out = log_config._truncate_exception_processor(None, "error", {"exception": text})
    assert len(out["exception"]) < len(text)


def test_redaction_runs_before_truncation():
    # redact_native_paths replaces exact strings, so truncating first could leave a
    # half path behind for it to miss.
    text = (_BACKEND / "loggers/config.py").read_text(encoding = "utf-8")
    order = text.index("filter_sensitive_data,\n"), text.index("_truncate_exception_processor,\n")
    assert order[0] < order[1], "filter_sensitive_data must come first in the chain"


def test_the_event_field_is_capped_too():
    # logger.error(f"failed: {e}", exc_info = True) puts the whole exception text in
    # the event, a third copy the first two caps never saw.
    text = "failed: " + ("q" * 4_000_000)
    out = log_config.truncate_exception({"event": text})["event"]
    assert len(out) < log_config._MAX_ERROR_CHARS + 200, len(out)
    assert out.startswith("failed: ")


def test_a_normal_event_name_is_untouched():
    ev = {"event": "request_failed", "error": "boom"}
    assert log_config.truncate_exception(dict(ev)) == ev


def test_positional_arguments_are_capped():
    # logger.error("stream error: %s", exc) keeps the exception under positional_args,
    # which the renderer stringifies with nothing in the chain to bound it.
    out = log_config.truncate_exception(
        {"event": "stream error: %s", "positional_args": (Exception("x" * 4_000_000),)}
    )
    assert len(str(out["positional_args"][0])) < log_config._MAX_ERROR_CHARS + 200
