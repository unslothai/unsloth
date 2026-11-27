# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Re-log transformers' load report: debug when benign, warning if it could change the model."""

from __future__ import annotations

import logging
import sys

import pytest
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


_RESTORE: list = []


def _attach_sink(name: str = _REPORT_LOGGER):
    """Logger is process-global: restore propagate and level, or later tests silently lose records."""
    log = logging.getLogger(name)
    sink = _Sink()
    _RESTORE.append((log, sink, log.propagate, log.level))
    log.addHandler(sink)
    log.propagate = False
    log.setLevel(logging.DEBUG)
    return log, sink


@pytest.fixture(autouse = True)
def _restore_loggers():
    yield
    while _RESTORE:
        log, sink, propagate, level = _RESTORE.pop()
        log.removeHandler(sink)
        log.propagate = propagate
        log.setLevel(level)


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


def test_progress_bar_state_is_restored(_progress_bar_state):
    from transformers.utils import logging as hf_logging

    enabled_probe = getattr(hf_logging, "is_progress_bar_enabled", None)
    if enabled_probe is None:
        return

    hf_logging.enable_progress_bar()
    with _quiet_transformers_load():
        assert enabled_probe() is False
    assert enabled_probe() is True


def test_a_caller_that_already_disabled_bars_stays_disabled(_progress_bar_state):
    from transformers.utils import logging as hf_logging

    enabled_probe = getattr(hf_logging, "is_progress_bar_enabled", None)
    if enabled_probe is None:
        return

    hf_logging.disable_progress_bar()
    with _quiet_transformers_load():
        pass
    assert enabled_probe() is False


def test_a_concurrent_thread_is_not_captured():
    # the filters sit on process-global loggers, so concurrent loads must keep their own reports
    import threading
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load() as report:
            t = threading.Thread(target = lambda: log.warning(_SERIOUS))
            t.start()
            t.join()
        assert sink.messages == [_SERIOUS], sink.messages
        assert report.reports == []
    finally:
        log.removeHandler(sink)
        log.propagate = True


def test_reports_are_re_emitted_when_the_load_fails():
    from core.rag import embeddings as emb

    log, sink = _attach_sink()
    emitted = []
    real_warning = emb.logger.warning
    emb.logger.warning = lambda msg, *a, **k: emitted.append(msg % a if a else msg)
    try:
        try:
            with _quiet_transformers_load() as report:
                try:
                    log.warning(_SERIOUS)
                    raise RuntimeError("weight tying blew up")
                finally:
                    emb._emit_load_reports(report)
        except RuntimeError:
            pass
        assert any("MISSING" in m for m in emitted), emitted
    finally:
        emb.logger.warning = real_warning
        log.removeHandler(sink)
        log.propagate = True


@pytest.fixture
def _progress_bar_state():
    """Restore the progress-bar switches, which are process-global and make tests order-dependent."""
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )
    from transformers.utils import logging as hf_logging

    hub_was_off = bool(are_progress_bars_disabled())
    tf_was_on = bool(hf_logging.is_progress_bar_enabled())
    try:
        yield
    finally:
        if tf_was_on:
            hf_logging.enable_progress_bar()
        else:
            hf_logging.disable_progress_bar()
        if hub_was_off:
            disable_progress_bars()
        else:
            enable_progress_bars()


def test_a_hub_only_progress_disable_survives(_progress_bar_state):
    # transformers' enable_progress_bar() also enables Hub bars, undoing unsloth's xet disable
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars
    from transformers.utils import logging as hf_logging

    hf_logging.enable_progress_bar()
    disable_progress_bars()
    with _quiet_transformers_load():
        pass
    assert are_progress_bars_disabled() is True


def test_an_unexpected_key_other_than_the_legacy_one_stays_a_warning():
    # a discarded encoder weight can degrade retrieval; only bge position_ids is quiet
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load() as report:
            log.warning("BertModel LOAD REPORT from: x\nencoder.layer.0.dense | UNEXPECTED")
        assert report.is_serious() is True
    finally:
        log.removeHandler(sink)
        log.propagate = True


def test_the_legacy_position_ids_report_is_still_benign():
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load() as report:
            log.warning(_BENIGN)
        assert report.is_serious() is False
    finally:
        log.removeHandler(sink)
        log.propagate = True


def test_the_peft_integration_logger_is_covered():
    # transformers.integrations.peft is not a descendant of the other two loggers
    log, sink = _attach_sink("transformers.integrations.peft")
    try:
        with _quiet_transformers_load() as report:
            log.warning(_SERIOUS)
        assert sink.messages == [], sink.messages
        assert len(report.reports) == 1
    finally:
        log.removeHandler(sink)
        log.propagate = True


def test_a_mixed_report_is_serious():
    log, sink = _attach_sink()
    mixed = (
        "BertModel LOAD REPORT from: x\n"
        "embeddings.position_ids  | UNEXPECTED\n"
        "encoder.layer.0.dense    | UNEXPECTED"
    )
    try:
        with _quiet_transformers_load() as report:
            log.warning(mixed)
        assert report.is_serious() is True
    finally:
        log.removeHandler(sink)
        log.propagate = True


def test_the_notes_section_is_not_read_as_a_key_row():
    # transformers appends a "Notes:" footer to reports with unexpected keys; not a row
    log, sink = _attach_sink()
    with_notes = (
        "BertModel LOAD REPORT from: unsloth/bge-small-en-v1.5\n"
        "embeddings.position_ids | UNEXPECTED\n"
        "\nNotes:\n"
        "- UNEXPECTED:\tcan be ignored when loading from different task/architecture."
    )
    with _quiet_transformers_load() as report:
        log.warning(with_notes)
    assert report.is_serious() is False


def test_a_serious_report_is_flattened_to_one_plain_line():
    from core.rag import embeddings as emb

    emitted = []
    real_warning = emb.logger.warning
    emb.logger.warning = lambda msg, *a, **k: emitted.append(msg % a if a else msg)
    log, sink = _attach_sink()
    try:
        with _quiet_transformers_load() as report:
            log.warning("\x1b[1mBertModel LOAD REPORT\x1b[0m from: x\nencoder.0 | MISSING")
        emb._emit_load_reports(report)
    finally:
        emb.logger.warning = real_warning
    assert emitted, emitted
    assert "\x1b" not in emitted[0]
    assert "\n" not in emitted[0]
    assert "MISSING" in emitted[0]


def test_a_key_merely_containing_position_ids_is_serious():
    # a real discarded weight, not the legacy buffer: a substring match would hide it
    log, sink = _attach_sink()
    with _quiet_transformers_load() as report:
        log.warning(
            "BertModel LOAD REPORT from: x\nencoder.position_ids_projection.weight | UNEXPECTED"
        )
    assert report.is_serious() is True


def test_a_prefixed_legacy_buffer_is_still_benign():
    log, sink = _attach_sink()
    with _quiet_transformers_load() as report:
        log.warning(
            "BertModel LOAD REPORT from: x\n0_Transformer.embeddings.position_ids | UNEXPECTED"
        )
    assert report.is_serious() is False
