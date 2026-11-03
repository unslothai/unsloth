// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { applyLogEntry, type LogState } from "../src/features/benchmark/lib/log-entry.ts";
import { deleteEach } from "../src/features/benchmark/lib/delete-runs.ts";

const empty: LogState = {
  logLines: [],
  lastSeq: null,
  stage: null,
  progressPercent: 0,
  progressDetail: null,
};

test("progress events move the bar and leave the log alone", () => {
  const line = JSON.stringify({ pct: 40, current: 4, total: 10, elapsed: "00:04", eta: "00:06" });
  const next = applyLogEntry(empty, { stream: "progress", line, ts: null });
  assert.equal(next.progressPercent, 40);
  assert.equal(next.progressDetail?.current, 4);
  assert.equal("logLines" in next, false);
});

test("log lines append, carry the seq and skip replays", () => {
  const next = { ...empty, ...applyLogEntry(empty, { stream: "status", line: "Loading", ts: 1 }, 7) };
  assert.equal(next.logLines.length, 1);
  assert.equal(next.lastSeq, 7);
  assert.equal(next.stage, "Loading");
  assert.equal(applyLogEntry(next, { stream: "stdout", line: "old", ts: 1 }, 7), next);
});

test("a failed delete keeps its run and reports the error", async () => {
  const { deleted, failed } = await deleteEach(["a", "b", "c"], async (id) => {
    if (id === "b") throw new Error("disk full");
  });
  assert.deepEqual([...deleted], ["a", "c"]);
  assert.equal(failed, "disk full");
  const ok = await deleteEach(["a"], async () => undefined);
  assert.equal(ok.failed, null);
});
