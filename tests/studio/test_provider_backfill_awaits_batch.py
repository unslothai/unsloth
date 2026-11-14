# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sync must await the backfill, and the batch must settle, not reject on the first failure."""

from __future__ import annotations

import re
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

# Walked back from the unique return so the slice does not match on the tested word.
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


# The sliced tail runs inside syncBackfillTail, so every local of the sync it reads must be one of these.
TAIL_PARAMS = (
    "backfillTasks",
    "isCurrent",
    "existingProviders",
    "syncedProviders",
    "registryRows",
)


def _sync_locals_read_by_tail() -> set[str]:
    """Locals (and parameters) of syncExternalProvidersFromBackend that the sliced tail reads."""
    text = read(SYNC_PROVIDERS)
    start = text.index("export async function syncExternalProvidersFromBackend(")
    tail = _tail_source()
    body = text[start : text.index(tail, start)]
    names = set()
    # Function-scope declarations only (two-space indent); nested blocks cannot leak into the tail.
    for group in re.findall(r"^  (?:const|let|var)\s+(\[[^\]]*\]|\{[^}]*\}|\w+)", body, re.M):
        names.update(re.findall(r"\b[A-Za-z_]\w*\b", group.split("=")[0]))
    signature = body[: body.index(")")]
    names.update(re.findall(r"\b([A-Za-z_]\w*)\??\s*:", signature))
    return {name for name in names if re.search(rf"\b{name}\b", tail)}


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
        """
        )
        + "".join(f"  {name},\n" for name in TAIL_PARAMS)
        + "){\n"
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
  [],
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


def test_the_tail_reads_no_sync_local_the_harness_does_not_pass():
    unpassed = sorted(_sync_locals_read_by_tail() - set(TAIL_PARAMS))
    assert (
        not unpassed
    ), f"the sync's tail now reads {unpassed}; add them to TAIL_PARAMS and pass them in SCRIPT"


def test_a_stale_session_runs_no_backfill(result: dict):
    assert result["stale"] == []
