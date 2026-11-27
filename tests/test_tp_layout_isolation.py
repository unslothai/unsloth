# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tp_env caches the env layout per process; tests/conftest.py clears it before every test.

Without that, a test reading the layout with UNSLOTH_TP_SIZE unset left None cached, and a later test
in the same xdist worker that set UNSLOTH_TP_SIZE=2 / UNSLOTH_CP_SIZE=2 got the generic "does not
compose" refusal instead of its layout's (test_pre_init_query_reads_a_pending_peft_config). Run in
file order, the second test here fails without the fixture.
"""


def test_a_test_may_leave_the_layout_read(monkeypatch):
    from unsloth.models import tp_env

    monkeypatch.delenv("UNSLOTH_TP_SIZE", raising = False)
    monkeypatch.delenv("UNSLOTH_PARALLEL", raising = False)
    tp_env.env_layout()
    assert tp_env._LAYOUT is not tp_env._UNSET


def test_the_next_test_starts_with_the_layout_unread():
    from unsloth.models import tp_env
    assert tp_env._LAYOUT is tp_env._UNSET, "a layout read by an earlier test leaked into this one"
