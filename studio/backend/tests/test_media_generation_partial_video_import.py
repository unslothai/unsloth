# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load posted while another thread is still importing core.inference.video must not 500 (AttributeError)."""

import sys
import types

from hub.services.models import account_access


def test_a_video_module_still_importing_counts_as_no_video_job(monkeypatch):
    half_built = types.ModuleType("core.inference.video")
    monkeypatch.setitem(sys.modules, "core.inference.video", half_built)
    assert account_access.foreign_media_generations("someone") == 0


def test_a_loaded_video_module_still_reports_its_in_flight_account(monkeypatch):
    loaded = types.ModuleType("core.inference.video")
    loaded.generation_account_in_flight = lambda: "other-account"
    monkeypatch.setitem(sys.modules, "core.inference.video", loaded)
    assert account_access.foreign_media_generations("someone") == 1
    assert account_access.foreign_media_generations("other-account") == 0
    loaded.generation_account_in_flight = lambda: None
    assert account_access.foreign_media_generations("someone") == 0
