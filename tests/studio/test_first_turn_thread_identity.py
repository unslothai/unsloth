# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A first turn must reach the model adapter with its real thread id.

assistant-ui binds `unstable_threadId` before the thread is persisted, so a first
turn used to file every run handle under the shared "__default" key. Two of them
overlapping there is unresolvable after the fact: nothing links a run under that
key to the id its thread later receives, so the sidebar showed no spinner and Stop
could not reach either generation.

The link exists earlier. `append()` tracks `threadListItem.initialize()` by the
user message id, and `createPersistedRunAdapter` already awaits that promise before
invoking the adapter, so the id is known by the time the run starts. These tests pin
that the resolved id is carried through rather than discarded.
"""

from __future__ import annotations

import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
PROVIDER = (WORKSPACE / "studio/frontend/src/features/chat/runtime-provider.tsx").read_text(
    encoding = "utf-8"
)


def test_the_tracked_promise_carries_the_assigned_thread_id():
    # Resolving to void threw the id away, which is what forced the "__default" detour.
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
    # The whole point: the adapter must not start under the unresolved key when the id is already known by the
    # time the await above resolves.
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
    # A resolved thread already streams under its own id; replacing it would move a running chat's handles out from
    # under the sidebar row watching them.
    block = re.search(
        r"const result = adapter\.run\((.*?)\);",
        PROVIDER,
        re.S,
    )
    assert block
    arg = block.group(1)
    assert "? { ...options, unstable_threadId: adoptedThreadId }" in arg
    assert ": options" in arg
