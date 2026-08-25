# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Log volume must not regress, and every polled path must be classified.

Studio's log-reduction work is several PRs deep and every round of it started with someone
noticing a log was huge. Nothing stopped the next chatty endpoint. These are the two guards
that do: an envelope on how much an idle app writes, and a closure check that makes it
impossible to add a poll without saying which suppression rule owns it.

The counts here are derived from a formula over the poll period and the de-duplication
window, not recorded from a run, so changing a poll interval moves the expectation with it
and only a genuine rule violation fails.

See ``test_log_signal_floor.py`` for the other half: these tests cap how much is written,
that one guarantees the important things still are.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

# Same idiom as test_server_disk_logging.py: make the import work regardless of which
# directory pytest was invoked from, rather than depending on the rootdir it picked.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from loggers import handlers as hmod  # noqa: E402
from log_budget import policy, replay, session  # noqa: E402


def _classes_present(paths) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in paths:
        grouped.setdefault(policy.classify(hmod, path), []).append(path)
    return grouped


class TestClassificationClosure:
    """Guard B. The registry and the middleware's own sets must describe the same world."""

    def test_every_classified_path_is_in_a_scenario(self):
        """A path cannot be quieted without saying how often it is polled.

        Otherwise a path joins ``_QUIET_POLL_PATHS`` for a reason nobody records, and the
        budget never sees it because no scenario asks for it.
        """
        classified = set()
        for attr in (
            "_QUIET_POLL_PATHS",
            "_LIVENESS_POLL_PATHS",
            "_WATCHDOG_POLL_PATHS",
            "_QUIET_SUCCESS_PATHS",
            "_CHAT_LIST_PATHS",
            "_SELF_READ_PATHS",
            "_EXCLUDED_PATHS",
        ):
            classified |= set(getattr(hmod, attr, ()) or ())

        missing = sorted(classified - set(session.ALL_POLLS))
        assert not missing, (
            "these paths are classified in loggers/handlers.py but no scenario in "
            "tests/log_budget/session.py polls them, so their log volume is unmeasured:\n  "
            + "\n  ".join(missing)
            + "\n\nAdd each to IDLE_POLLS (polled when nothing is happening) or BUSY_POLLS "
            "(polled only during an operation) with its interval."
        )

    def test_every_polled_path_has_exactly_one_class(self):
        """And a poll cannot be added without choosing a rule for it.

        ``classify`` returns ``normal`` for anything unlisted, which is a real class with a
        300 ms window, so the check is that the choice was deliberate: a path polled faster
        than a few seconds and left in ``normal`` logs on essentially every request.
        """
        offenders = {
            path
            for path, (period, _provenance) in session.ALL_POLLS.items()
            if policy.classify(hmod, path) == policy.NORMAL
            and period * 1000.0 > hmod._ACCESS_LOG_DEDUP_MS
        }

        new = sorted(offenders - session.KNOWN_UNCLASSIFIED_POLLS)
        assert not new, (
            "these paths are polled further apart than the `normal` window of "
            f"{hmod._ACCESS_LOG_DEDUP_MS} ms, so every single poll writes a line:\n  "
            + "\n  ".join(f"{path} every {session.ALL_POLLS[path][0]:g}s" for path in new)
            + f"\n\nPick a heartbeat class in loggers/handlers.py: "
            f"{', '.join(policy.ALL_CLASSES)}."
        )

        # The other direction, so the ledger cannot outlive the problem it records.
        fixed = sorted(session.KNOWN_UNCLASSIFIED_POLLS - offenders)
        assert not fixed, (
            "these paths are listed in KNOWN_UNCLASSIFIED_POLLS but now have a heartbeat "
            "class, so the entry is stale:\n  "
            + "\n  ".join(fixed)
            + "\n\nDelete them from tests/log_budget/session.py and tighten the envelopes."
        )

    def test_scenarios_do_not_overlap(self):
        both = sorted(set(session.IDLE_POLLS) & set(session.BUSY_POLLS))
        assert not both, (
            "a path must belong to exactly one scenario or its lines are counted twice:\n  "
            + "\n  ".join(both)
        )


class TestVolumeEnvelope:
    """Guard A. How much the app writes, and whether it honours its own windows."""

    @pytest.mark.parametrize(
        "label, polls, duration, envelope",
        [
            (
                "steady idle",
                session.IDLE_POLLS,
                session.STEADY_IDLE_SECONDS,
                session.STEADY_IDLE_LINE_ENVELOPE,
            ),
            (
                "operation in flight",
                session.BUSY_POLLS,
                session.BUSY_SECONDS,
                session.BUSY_LINE_ENVELOPE,
            ),
        ],
    )
    def test_scenario_stays_inside_its_envelope(
        self, label, polls, duration, envelope, monkeypatch
    ):
        result = replay.replay(hmod, monkeypatch, polls, duration)
        counts = Counter(result.capture.paths())

        if result.emitted > envelope:
            worst = "\n  ".join(
                f"{n:5d}  {path}  [{policy.classify(hmod, path)}]"
                for path, n in counts.most_common(8)
            )
            pytest.fail(
                f"{label}: {result.emitted} log lines over {duration / 60:.0f} virtual "
                f"minutes, envelope is {envelope}.\n"
                f"Biggest contributors:\n  {worst}\n\n"
                "If you added an endpoint, give it a heartbeat class in "
                "loggers/handlers.py rather than raising the envelope. Raising it is a "
                "decision about how much Studio is allowed to write when nobody is using "
                "it."
            )

    @pytest.mark.parametrize(
        "label, polls, duration",
        [
            ("steady idle", session.IDLE_POLLS, session.STEADY_IDLE_SECONDS),
            ("operation in flight", session.BUSY_POLLS, session.BUSY_SECONDS),
        ],
    )
    def test_each_path_matches_its_class_formula(self, label, polls, duration, monkeypatch):
        """The window is honoured exactly, not merely under a ceiling.

        A ceiling alone would pass if suppression stopped working and something else got
        quieter. Checking the derived count catches the rule itself breaking.
        """
        result = replay.replay(hmod, monkeypatch, polls, duration)
        counts = Counter(result.capture.paths())

        mismatches = []
        for path, (period, _provenance) in polls.items():
            # Shared buckets are asserted below; a member that is not the bucket owner
            # legitimately emits zero.
            if policy.bucket_of(hmod, path) != path:
                continue
            cls = policy.classify(hmod, path)
            expected = policy.expected_emissions(policy.window_ms(hmod, cls), period, duration)
            actual = counts.get(path, 0)
            if actual != expected:
                mismatches.append(
                    f"{path} [{cls}] polled every {period}s: emitted {actual}, "
                    f"the {policy.window_ms(hmod, cls)}ms window implies {expected}"
                )

        assert not mismatches, (
            f"{label}: these paths did not log the number of times their suppression class "
            "implies:\n  " + "\n  ".join(mismatches)
        )

    def test_the_liveness_burst_collapses_to_one_bucket(self, monkeypatch):
        """The five liveness paths answer one question and must cost one line, not five.

        Asserted on the bucket total rather than on which path won it: the SPA fires them
        together and whichever arrives first legitimately takes the line.
        """
        liveness = {
            path: value
            for path, value in session.IDLE_POLLS.items()
            if policy.classify(hmod, path) == policy.LIVENESS
        }
        if not liveness:
            pytest.skip("no liveness paths configured in this revision")

        result = replay.replay(hmod, monkeypatch, liveness, session.STEADY_IDLE_SECONDS)
        fastest = min(period for period, _ in liveness.values())
        expected = policy.expected_emissions(
            policy.window_ms(hmod, policy.LIVENESS), fastest, session.STEADY_IDLE_SECONDS
        )
        assert result.emitted == expected, (
            f"{len(liveness)} liveness paths polled together produced {result.emitted} "
            f"lines; sharing one bucket implies {expected}. If a path left "
            "_LIVENESS_POLL_PATHS it now heartbeats on its own and costs a line per window."
        )
