# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Dismissal of the integrated-GPU memory notice.

Once dismissed it must stay dismissed, but only for the allocation it was dismissed
at: a user who acts on the advice, raises the allocation and later runs short again
is in a new situation and worth telling.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import igpu_carveout_notice_settings as notice  # noqa: E402


class TestAFreshInstall:
    def test_nothing_is_dismissed(self):
        assert notice.get_dismissed_at_gb() is None
        assert notice.notice_already_dismissed(32.0) is False

    def test_an_unknown_allocation_is_not_silenced(self):
        # Nothing dismissed yet, so even an unreadable allocation may speak.
        assert notice.notice_already_dismissed(None) is False


class TestDismissal:
    def test_dismissing_silences_the_same_allocation(self):
        notice.dismiss_notice(32.0)
        assert notice.get_dismissed_at_gb() == 32.0
        assert notice.notice_already_dismissed(32.0) is True

    def test_it_stays_silent_at_a_smaller_allocation(self):
        # Lowering the allocation is not new information: they were already told.
        notice.dismiss_notice(64.0)
        assert notice.notice_already_dismissed(32.0) is True

    def test_it_speaks_again_after_the_user_raises_the_allocation(self):
        # Acted on the advice and hit the ceiling again: worth one more mention.
        notice.dismiss_notice(32.0)
        assert notice.notice_already_dismissed(64.0) is False

    def test_a_driver_rounding_difference_does_not_re_show_it(self):
        # A driver-reported byte count reads 95.83 against a 96.00 setting.
        notice.dismiss_notice(95.83)
        assert notice.notice_already_dismissed(95.9) is True

    def test_dismissal_only_ever_rises(self):
        # A stale client reporting an old, smaller allocation must not re-arm it.
        notice.dismiss_notice(64.0)
        notice.dismiss_notice(16.0)
        assert notice.get_dismissed_at_gb() == 64.0

    def test_an_unknown_allocation_is_silent_once_dismissed(self):
        notice.dismiss_notice(32.0)
        assert notice.notice_already_dismissed(None) is True


class TestCorruptRows:
    """A bad row must fail toward showing the notice, never toward hiding it."""

    def test_junk_reads_as_never_dismissed(self):
        from storage.studio_db import upsert_app_settings
        for junk in ("banana", "", -5, 0, True, False, None):
            upsert_app_settings({notice.IGPU_CARVEOUT_NOTICE_KEY: junk})
            assert notice.get_dismissed_at_gb() is None, junk
            assert notice.notice_already_dismissed(32.0) is False, junk

    def test_a_numeric_string_is_honoured(self):
        from storage.studio_db import upsert_app_settings
        upsert_app_settings({notice.IGPU_CARVEOUT_NOTICE_KEY: "48"})
        assert notice.get_dismissed_at_gb() == 48.0

    def test_dismissing_with_junk_does_not_crash_or_store(self):
        assert notice.dismiss_notice(None) is None
        assert notice.dismiss_notice(-1) is None
        assert notice.get_dismissed_at_gb() is None


class TestAHostileDismissalValue:
    """`current_gb` arrives in a client-controlled POST body, and Python's json accepts
    `Infinity`, so a value no machine will ever exceed really can reach this -- and
    storing it would silence the notice permanently."""

    @pytest.mark.parametrize(
        "value",
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            10**9,
            2**53,
            -1,
            0,
        ],
    )
    def test_it_cannot_silence_the_notice_forever(self, value):
        notice.dismiss_notice(value)
        assert notice.notice_already_dismissed(32.0) is False, value

    @pytest.mark.parametrize(
        "stored",
        [
            float("inf"),
            float("nan"),
            "Infinity",
            "1e999",
            "-inf",
        ],
    )
    def test_a_corrupt_row_reads_as_never_dismissed(self, stored):
        from storage.studio_db import upsert_app_settings
        upsert_app_settings({notice.IGPU_CARVEOUT_NOTICE_KEY: stored})
        assert notice.notice_already_dismissed(32.0) is False, stored


class TestTheToleranceBoundary:
    """The slack is a tenth of a GB, and both sides arrive rounded to a tenth."""

    def test_a_reading_exactly_one_tenth_above_stays_dismissed(self):
        # The client dismisses at the rounded value the advice showed (95.8), and a
        # later boot reading 95.9 is one tenth away. Binary floats put 95.8 + 0.1 at
        # 95.89999999999999, so the float comparison called it not dismissed.
        assert 95.8 + 0.1 == pytest.approx(95.9), "the float is only 1 ulp off"
        assert (95.9 <= 95.8 + 0.1) is False, "which is what the old comparison read"
        notice.dismiss_notice(95.8)
        assert notice.notice_already_dismissed(95.9) is True

    def test_two_tenths_above_still_speaks(self):
        # The slack is one tenth, not "anything close". A real change must be heard.
        notice.dismiss_notice(95.8)
        assert notice.notice_already_dismissed(96.0) is False

    def test_the_boundary_holds_across_the_ladder(self):
        # Every rung the ladder can suggest, dismissed at its rounded reading.
        for rung in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
            reading = round(rung - 0.2, 1)
            notice.dismiss_notice(reading)
            assert notice.notice_already_dismissed(round(reading + 0.1, 1)) is True, rung
            assert notice.notice_already_dismissed(round(reading + 0.2, 1)) is False, rung
