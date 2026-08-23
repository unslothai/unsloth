# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The provider model backfill must be finished before the sync resolves (#7281).

``syncExternalProvidersFromBackend`` is what the credential bootstrap gate awaits before it
releases app content, so the backfill writes have to be complete when it returns. Two hops
carry that: the ``await`` on ``settleTasksIfCurrent`` at the call site, and the ``await`` on
``Promise.allSettled`` inside the helper. Drop either and the sync resolves while the writes
are still in flight, so an immediate close or a session transition loses them.

A string contract cannot hold this. ``await`` is one token in a source file; asserting it is
present is defeated by any reformat, and asserting the call is present says nothing about
whether it is awaited. So both hops are run for real instead: the helper and the call-site
tail are sliced VERBATIM out of the studio sources into a node harness (see
``_node_harness``) and driven with tasks that only finish on a timer. If either ``await``
goes, the tail resolves with the timers still pending and the recorded order is empty.

The same run pins the other half of the contract, that the batch SETTLES rather than
rejecting on the first failure: one task rejects immediately, and the two that resolve later
must still be recorded. Under ``Promise.all`` the tail would reject instead.
"""

from __future__ import annotations

import textwrap

import pytest

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

RECONCILIATION = source_path("studio/frontend/src/features/credentials/reconciliation.ts")
SYNC_PROVIDERS = source_path("studio/frontend/src/features/chat/sync-external-providers.ts")

SOURCES = (RECONCILIATION, SYNC_PROVIDERS)

TEMP = WORKDIR / "temp" / "provider_backfill_awaits_batch"

# The end of syncExternalProvidersFromBackend, which is where the backfill batch is awaited.
# Anchored on the unique return and walked BACK to the staleness guard, so the slice is taken
# without matching on the word being tested.
TAIL_END = "\n  return syncedProviders;\n}"
TAIL_START = "if (isCurrent && !isCurrent()) return existingProviders;"


def _helper_source() -> str:
    """settleTasksIfCurrent, verbatim."""
    text = read(RECONCILIATION)
    assert text.count("export async function settleTasksIfCurrent") == 1
    return slice_between(text, "export async function settleTasksIfCurrent", "\nexport ")


def _tail_source() -> str:
    """The awaiting tail of syncExternalProvidersFromBackend, verbatim."""
    text = read(SYNC_PROVIDERS)
    assert text.count(TAIL_END) == 1, "the sync no longer ends in a single return"
    end = text.index(TAIL_END) + len(TAIL_END) - len("\n}")
    start = text.rindex(TAIL_START, 0, end)
    return text[start:end]


def _harness_source() -> str:
    return (
        textwrap.dedent(
            """
        // @ts-nocheck
        // ---- PRELUDE: the sliced tail reads only through its parameters ----
        // ---- PRELUDE ENDS: verbatim studio source follows ----
        """
        )
        + _helper_source()
        + textwrap.dedent(
            """
        export async function syncBackfillTail(
          backfillTasks,
          isCurrent,
          existingProviders,
          syncedProviders,
        ) {
        """
        )
        + "  "
        + _tail_source()
        + "\n}\n"
    )


SCRIPT = """
// @ts-nocheck
import { settleTasksIfCurrent, syncBackfillTail } from "./harness.ts";

const finished = [];
const delayed = (name, ms) => () =>
  new Promise((resolve) => {
    setTimeout(() => {
      finished.push(name);
      resolve(null);
    }, ms);
  });

// One immediate rejection between two timer-backed writes: the tail must wait for both and
// must not be sunk by the failure in between.
const returned = await syncBackfillTail(
  [delayed("first", 40), () => Promise.reject(new Error("backfill failed")), delayed("last", 80)],
  () => true,
  ["existing"],
  ["synced"],
);
const finishedWhenSyncResolved = [...finished];

// A session that moved on skips the batch entirely, and must not run a task.
const stale = [];
await settleTasksIfCurrent(
  [
    () => {
      stale.push("ran");
      return Promise.resolve();
    },
  ],
  () => false,
);

console.log(JSON.stringify({ finishedWhenSyncResolved, returned, stale }));
"""


@pytest.fixture(scope = "module")
def result() -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), SCRIPT, sources = SOURCES)


def test_the_backfill_batch_is_complete_when_the_sync_resolves(result: dict):
    assert result["finishedWhenSyncResolved"] == ["first", "last"], (
        "the sync resolved with backfill writes still in flight, so a close or a session "
        "transition right after startup would lose them"
    )
    assert result["returned"] == ["synced"]


def test_a_stale_session_runs_no_backfill(result: dict):
    assert result["stale"] == []
