# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One definition of "a successful access record", in two languages.

The backend decides whether to emit an access line; the desktop shell decides whether to
mirror it into ``tauri.log``. Two implementations of the same idea, in two languages and
two processes, with nothing holding them together. The pair already drifted once: the
Python tee and the Rust ``collapse_progress_frames`` disagreed about CRLF, so a Windows
traceback survived on one sink and became blank lines on the other.

``tests/fixtures/access_log_records.json`` is the shared contract. This module checks the
Python side and, once the desktop-side filter exists, checks that the Rust side was written
against the same rule. The matching ``#[test]`` in ``process.rs`` consumes the same file, so
changing the rule on either side turns the other red.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "access_log_records.json"
_PROCESS_RS = Path(__file__).resolve().parents[2] / "src-tauri" / "src" / "process.rs"


def _cases():
    return json.loads(_FIXTURE.read_text(encoding = "utf-8"))["cases"]


def is_successful_access_record(line: str) -> bool:
    """The contract, in Python. ``keep`` in the fixture is the negation of this."""
    trimmed = line.lstrip()
    if not trimmed.startswith("{"):
        return False
    try:
        record = json.loads(trimmed)
    except ValueError:
        return False
    if not isinstance(record, dict):
        return False
    if record.get("event") != "request_completed":
        return False
    status = record.get("status_code")
    if not isinstance(status, int) or isinstance(status, bool):
        return False
    return 200 <= status <= 299


@pytest.mark.parametrize("case", _cases(), ids = lambda c: c["name"])
def test_python_side_matches_the_shared_fixture(case):
    dropped = is_successful_access_record(case["line"])
    assert dropped != case["keep"], (
        f"{case['name']}: the fixture says keep={case['keep']} ({case['why']}) but the "
        f"Python rule says this line is {'a' if dropped else 'not a'} successful access "
        "record. Update both sides together, or the two sinks stop agreeing."
    )


def test_the_fixture_still_covers_the_edges():
    """Guard the guard: a fixture quietly emptied of hard cases proves nothing."""
    names = {case["name"] for case in _cases()}
    required = {
        "199 is not success",
        "300 is not success",
        "request_completed with no status_code",
        "malformed JSON",
        "a different structured event",
    }
    missing = sorted(required - names)
    assert not missing, (
        "these boundary cases were removed from access_log_records.json:\n  "
        + "\n  ".join(missing)
        + "\nThey are the cases where the two implementations are most likely to diverge."
    )


class TestRustSide:
    """Checked from Python so a divergence fails even in a Python-only CI job."""

    def _source(self) -> str:
        if not _PROCESS_RS.is_file():
            pytest.skip(f"{_PROCESS_RS} not present")
        return _PROCESS_RS.read_text(encoding = "utf-8")

    def test_the_desktop_filter_uses_the_same_success_range(self):
        source = self._source()
        if "fn is_backend_access_log_line" not in source:
            pytest.skip(
                "the desktop shell does not filter successful access records on this "
                "revision; this check activates when that lands"
            )

        match = re.search(r"Some\((\d+)\.\.=(\d+)\)", source)
        assert match is not None, (
            "is_backend_access_log_line no longer matches a Some(lo..=hi) status range, "
            "which is how this test reads the rule. Update the test alongside it."
        )
        low, high = int(match.group(1)), int(match.group(2))
        assert (low, high) == (200, 299), (
            f"the desktop shell treats {low}..={high} as success while the backend and "
            "tests/fixtures/access_log_records.json use 200..=299. A line dropped on one "
            "sink and kept on the other is exactly the drift this fixture exists to catch."
        )

        assert (
            '"request_completed"' in source
        ), "is_backend_access_log_line no longer keys on the request_completed event"

    def test_the_rust_test_consumes_the_shared_fixture(self):
        source = self._source()
        if "fn is_backend_access_log_line" not in source:
            pytest.skip("desktop-side filter not present on this revision")
        assert "access_log_records.json" in source, (
            "process.rs implements the success rule but no Rust test reads "
            "tests/fixtures/access_log_records.json, so the two languages are only "
            "checked from the Python side. Add the shared-fixture #[test]."
        )
