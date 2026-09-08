# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Dismissal of the integrated-GPU memory notice.

Once dismissed it must stay dismissed -- but only for the allocation it was
dismissed at. A user who acts on the advice and raises the allocation, then later
loads a model too big for the new one, is in a new situation and worth telling.
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
        # Dismissed at 32 GB, now running 64 GB and still short: they acted on the
        # advice and hit the ceiling again, which is worth one more mention.
        notice.dismiss_notice(32.0)
        assert notice.notice_already_dismissed(64.0) is False

    def test_a_driver_rounding_difference_does_not_re_show_it(self):
        # The allocation is a driver-reported byte count: 95.83 against a 96.00
        # setting on the development machine. Re-showing on that would look broken.
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
    """`current_gb` arrives in a POST body, so it is client-controlled. Python's
    json accepts `Infinity` even though the spec does not, so a value no machine
    will ever exceed really can reach this -- and storing it would silence the
    notice permanently, the opposite of the fail-toward-showing rule above."""

    @pytest.mark.parametrize("value", [
        float("inf"), float("-inf"), float("nan"), 10**9, 2**53, -1, 0,
    ])
    def test_it_cannot_silence_the_notice_forever(self, value):
        notice.dismiss_notice(value)
        assert notice.notice_already_dismissed(32.0) is False, value

    @pytest.mark.parametrize("stored", [
        float("inf"), float("nan"), "Infinity", "1e999", "-inf",
    ])
    def test_a_corrupt_row_reads_as_never_dismissed(self, stored):
        from storage.studio_db import upsert_app_settings
        upsert_app_settings({notice.IGPU_CARVEOUT_NOTICE_KEY: stored})
        assert notice.notice_already_dismissed(32.0) is False, stored
