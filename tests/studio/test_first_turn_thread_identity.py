# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First turn must carry its real thread id; otherwise runs file under the shared "__default" key."""

from __future__ import annotations

import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
PROVIDER = (WORKSPACE / "studio/frontend/src/features/chat/runtime-provider.tsx").read_text(
    encoding = "utf-8"
)


def test_the_tracked_promise_carries_the_assigned_thread_id():
    assert "Promise<string | undefined>\n>();" in PROVIDER
    assert re.search(
        r"trackRunStartReady\(\s*message\.id,\s*initializeThread\.then\(\(\{ remoteId \}\) => remoteId\),",
        PROVIDER,
    ), "append() must track the promise that resolves to the persisted thread id"


def test_wait_for_run_start_returns_the_id():
    assert re.search(
        r"async function waitForRunStartHistoryAppend\([^)]*\): Promise<string \| undefined>",
        PROVIDER,
        re.S,
    ), "the awaiter must hand back the id it waited for"
    assert "return adoptedThreadId;" in PROVIDER


def test_the_run_is_given_its_real_thread_id():
    block = re.search(
        r"async \*run\(options\) \{.*?const result = adapter\.run\(.*?\);",
        PROVIDER,
        re.S,
    )
    assert block, "createPersistedRunAdapter's run wrapper not found"
    body = block.group(0)
    assert "let adoptedThreadId: string | undefined;" in body
    assert re.search(
        r"adoptedThreadId\s*=\s*await waitForRunStartHistoryAppend\(",
        body,
    ), "the persisted-run preflight must retain the assigned thread id"
    assert (
        "!options.unstable_threadId && adoptedThreadId" in body
    ), "only fill in the id when assistant-ui had none"
    assert "unstable_threadId: adoptedThreadId" in body


def test_an_existing_thread_id_is_never_overwritten():
    # Replacing a resolved id would move a running chat's handles out from under the sidebar.
    block = re.search(
        r"const result = adapter\.run\((.*?)\);",
        PROVIDER,
        re.S,
    )
    assert block
    arg = block.group(1)
    assert "? { ...options, unstable_threadId: adoptedThreadId }" in arg
    assert ": options" in arg
