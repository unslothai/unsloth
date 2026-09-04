# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import Optional

from .protocols import RecognizedFailure

_TRACEBACK = "Traceback (most recent call last)"
_EXIT = re.compile(r"(?:exit(?:ed)?(?: with)? code|returncode|exit_code)\s*[:=]?\s*(-?\d+)", re.I)

ENTER_SIM_TOOL_NAMES = frozenset({"rims_enter_sim", "rims.enter_sim"})

USER_FAIL_PHRASES = (
    "that failed",
    "that didn't work",
    "that did not work",
    "still broken",
    "still failing",
    "try in sim",
)

# Runner fingerprints
_PYTEST_FAILURES = re.compile(r"={3,}\s*FAILURES\b")
_PYTEST_FAILED_EQ = re.compile(r"(?i)\bfailed\s*=\s*[1-9]")
_PYTEST_N_FAILED = re.compile(r"(?i)\b[1-9]\d*\s+failed\b")
_FAILED_SPACE = re.compile(r"FAILED ")
_UNITTEST_FAILED_PAREN = re.compile(r"FAILED\s*\(")
_JEST_FAIL = re.compile(r"(?m)^FAIL ")
_JEST_TESTS_FAILED = re.compile(r"(?i)Tests:\s+(?:.*\b)?[1-9]\d*\s+failed")
_GO_FAIL_TAB = re.compile(r"FAIL\t")

_RUNNER_TOOL_NAMES = frozenset({"python", "terminal"})

RUN_ACTION_FAIL_PREFIXES = (
    "Execution timed out after ",
    "Execution cancelled.",
    "Blocked command(s) for safety:",
    "Execution error:",
    "Error: run_action supports",
    "No command provided.",
)


def user_declares_failure(text: str) -> bool:
    folded = (text or "").casefold().replace("\u2019", "'")
    return any(phrase in folded for phrase in USER_FAIL_PHRASES)


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def inspect_tool_result(
    name: str,
    result: str,
    *,
    contact: str = "world",
) -> Optional[RecognizedFailure]:
    if name in ENTER_SIM_TOOL_NAMES:
        return RecognizedFailure(summary = "enter_sim requested", source = "tool")
    text = result or ""
    head = text.lstrip()
    if name in _RUNNER_TOOL_NAMES:
        for prefix in RUN_ACTION_FAIL_PREFIXES:
            if head.startswith(prefix):
                return RecognizedFailure(
                    summary = head.splitlines()[0][:200] if head else f"{name} failed",
                    source = contact,
                )
    if _TRACEBACK in text:
        return RecognizedFailure(summary = f"{name} raised", source = contact)
    if text.startswith("Error:") or "\nError:" in text:
        first = text.strip().splitlines()[0][:200]
        return RecognizedFailure(summary = first, source = contact)
    if name in _RUNNER_TOOL_NAMES:
        last = _last_nonempty_line(text)
        if last and (
            _PYTEST_FAILED_EQ.search(last)
            or _PYTEST_N_FAILED.search(last)
            or _PYTEST_FAILURES.search(text)
            or _FAILED_SPACE.search(text)
            or _UNITTEST_FAILED_PAREN.search(text)
            or _JEST_FAIL.search(text)
            or _JEST_TESTS_FAILED.search(text)
            or _GO_FAIL_TAB.search(text)
        ):
            return RecognizedFailure(summary = f"{name} failed", source = contact)
    match = _EXIT.search(text)
    if match and match.group(1) not in {"0"}:
        return RecognizedFailure(
            summary = f"{name} exited {match.group(1)}",
            source = contact,
        )
    lowered = text.lower()
    if "command failed" in lowered or "returned non-zero" in lowered:
        return RecognizedFailure(summary = f"{name} failed", source = contact)
    return None


def grade_run_action(
    name: str,
    result: str | None,
    *,
    contact: str = "sim",
) -> Optional[RecognizedFailure]:
    """Grade a harness result. Timeout / cancel / block / empty are fail, never pass."""
    text = "" if result is None else str(result)
    if not text.strip():
        return RecognizedFailure(summary = f"{name} empty result", source = contact)
    head = text.lstrip()
    for prefix in RUN_ACTION_FAIL_PREFIXES:
        if head.startswith(prefix):
            return RecognizedFailure(summary = head.splitlines()[0][:200], source = contact)
    return inspect_tool_result(name, text, contact = contact)
