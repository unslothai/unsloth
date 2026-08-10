# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The RAG embedder load must not write raw transformers output to the server log.

transformers >= 5 prints a multi-line, ANSI-coloured "<Model> LOAD REPORT" table
through logger.warning plus a "Loading weights" tqdm bar. bge-small-en-v1.5 always
trips it (legacy embeddings.position_ids key), so every Studio boot emitted ~7
unstructured lines into an otherwise JSON log. They are captured and re-emitted on
our own logger instead: debug when benign, warning when the report mentions
anything that could change the model.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.rag.embeddings import _quiet_transformers_load  # noqa: E402

_REPORT_LOGGER = "transformers.utils.loading_report"
_BENIGN = (
    "\x1b[1mBertModel LOAD REPORT\x1b[0m from: unsloth/bge-small-en-v1.5\n"
    "Key                     | Status\n"
    "embeddings.position_ids | UNEXPECTED"
)
_SERIOUS = "BertModel LOAD REPORT from: x\nencoder.layer.0.weight | MISSING"


class _Sink(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _attach_sink():
    log = logging.getLogger(_REPORT_LOGGER)
    sink = _Sink()
    log.addHandler(sink)
    log.propagate = False
    log.setLevel(logging.DEBUG)
    return log, sink


def test_load_report_is_swallowed_and_captured():
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load() as report:
            log.warning(_BENIGN)
        assert sink.messages == [], sink.messages
        assert len(report.reports) == 1
        assert "LOAD REPORT" in report.reports[0]
        assert report.is_serious() is False
    finally:
        log.removeHandler(sink)


def test_unrelated_transformers_warnings_still_pass_through():
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load():
            log.warning("something genuinely wrong happened")
        assert sink.messages == ["something genuinely wrong happened"]
    finally:
        log.removeHandler(sink)


def test_missing_keys_are_flagged_as_serious():
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load() as report:
            log.warning(_SERIOUS)
        assert report.is_serious() is True
    finally:
        log.removeHandler(sink)


def test_filter_is_removed_after_the_context():
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load():
            pass
        log.warning(_BENIGN)
        assert len(sink.messages) == 1
    finally:
        log.removeHandler(sink)


def test_progress_bar_state_is_restored():
    from transformers.utils import logging as hf_logging

    enabled_probe = getattr(hf_logging, "is_progress_bar_enabled", None)
    if enabled_probe is None:
        return  # nothing to assert on this transformers build

    hf_logging.enable_progress_bar()
    with _quiet_transformers_load():
        assert enabled_probe() is False
    assert enabled_probe() is True


def test_a_caller_that_already_disabled_bars_stays_disabled():
    from transformers.utils import logging as hf_logging

    enabled_probe = getattr(hf_logging, "is_progress_bar_enabled", None)
    if enabled_probe is None:
        return

    hf_logging.disable_progress_bar()
    try:
        with _quiet_transformers_load():
            pass
        assert enabled_probe() is False
    finally:
        hf_logging.enable_progress_bar()
