// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// The overlay .tsx pulls in motion, hugeicons and the router, so it cannot be imported
// here. Its new-traffic decision lives in a plain module, which this drives directly.
import {
  type WatchedEntry,
  type WatchedResponse,
  createWatch,
  observeResponse,
  rearmWatch,
  startWatching,
} from "../src/features/api-monitor/new-traffic.ts";

// The server's clock: entry timestamps are its time.time(), never a browser instant.
const SERVER_NOW = 1_000_000;
// performance.now() when the poll stood up.
const WATCH_AT = 1_000;

function entry(
  id: string,
  status: WatchedEntry["status"],
  startedAt: number,
  viaApiKey = true,
): WatchedEntry {
  // biome-ignore lint/style/useNamingConvention: API schema
  return { id, status, via_api_key: viaApiKey, started_at: startedAt };
}

function snapshot(
  entries: WatchedEntry[],
  serverTime: number | null = SERVER_NOW,
): WatchedResponse {
  // biome-ignore lint/style/useNamingConvention: API schema
  return { entries, server_time: serverTime };
}

function watchFrom(startedAtMs: number) {
  const watch = createWatch(0);
  startWatching(watch, startedAtMs);
  return watch;
}

test("a call that finished before the first snapshot arrived is new traffic", () => {
  // The tab was hidden for 4s, so poll() issued no fetch. The first curl ran 2s into
  // that gap and was done when the snapshot landed: terminal, but not history.
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([entry("apireq_new", "completed", SERVER_NOW - 2)]),
    WATCH_AT + 4_000,
  );
  assert.equal(opened, true);
});

test("traffic from before the watch began stays history", () => {
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([entry("apireq_old", "completed", SERVER_NOW - 90)]),
    WATCH_AT + 4_000,
  );
  assert.equal(opened, false);
});

test("a request still running at the first snapshot is live traffic", () => {
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([entry("apireq_live", "running", SERVER_NOW - 90)]),
    WATCH_AT + 10,
  );
  assert.equal(opened, true);
});

test("a fresh id in a later snapshot opens the panel", () => {
  const watch = watchFrom(WATCH_AT);
  const backlog = [entry("apireq_old", "completed", SERVER_NOW - 90)];
  assert.equal(observeResponse(watch, snapshot(backlog), WATCH_AT + 10), false);
  const opened = observeResponse(
    watch,
    snapshot(
      [entry("apireq_next", "completed", SERVER_NOW + 4), ...backlog],
      SERVER_NOW + 5,
    ),
    WATCH_AT + 5_010,
  );
  assert.equal(opened, true);
});

test("Unsloth's own chat never opens the panel", () => {
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([entry("uireq", "completed", SERVER_NOW - 2, false)]),
    WATCH_AT + 4_000,
  );
  assert.equal(opened, false);
});

test("a backend with no clock field keeps the old terminal-is-history seed", () => {
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([entry("apireq_new", "completed", SERVER_NOW - 2)], null),
    WATCH_AT + 4_000,
  );
  assert.equal(opened, false);
});

test("a browser clock disagreeing with the server's does not replay the backlog", () => {
  // The cutoff is the server's clock minus a browser DURATION, never minus a browser
  // timestamp, so a wall clock minutes off still dates the backlog correctly.
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([
      entry("apireq_a", "completed", SERVER_NOW - 300),
      entry("apireq_b", "completed", SERVER_NOW - 120),
    ]),
    WATCH_AT + 20,
  );
  assert.equal(opened, false);
});

test("coming back from the full page does not replay the rows it showed", () => {
  const watch = watchFrom(WATCH_AT);
  const backlog = [entry("apireq_old", "completed", SERVER_NOW - 90)];
  observeResponse(watch, snapshot(backlog), WATCH_AT + 10);
  // 60s on /api-monitor reading those rows, then back to chat.
  rearmWatch(watch);
  startWatching(watch, WATCH_AT + 60_000);
  const opened = observeResponse(
    watch,
    snapshot(backlog, SERVER_NOW + 60),
    WATCH_AT + 60_010,
  );
  assert.equal(opened, false);
});

test("a request still running when the full page is left does not reopen the overlay", () => {
  // /api-monitor was open on a long generation, then left for chat: that row was on
  // screen the whole time.
  const watch = watchFrom(WATCH_AT);
  const live = entry("apireq_live", "running", SERVER_NOW - 5);
  observeResponse(watch, snapshot([live]), WATCH_AT + 10);
  rearmWatch(watch);
  startWatching(watch, WATCH_AT + 60_000);
  const opened = observeResponse(
    watch,
    snapshot([live], SERVER_NOW + 60),
    WATCH_AT + 60_010,
  );
  assert.equal(opened, false);
});

test("a rearm writes off only the snapshot it comes back to", () => {
  // The write-off is one seed, not a mode: a later call is still new traffic.
  const watch = watchFrom(WATCH_AT);
  const live = entry("apireq_live", "running", SERVER_NOW - 5);
  observeResponse(watch, snapshot([live]), WATCH_AT + 10);
  rearmWatch(watch);
  startWatching(watch, WATCH_AT + 60_000);
  observeResponse(watch, snapshot([live], SERVER_NOW + 60), WATCH_AT + 60_010);
  const opened = observeResponse(
    watch,
    snapshot(
      [entry("apireq_next", "running", SERVER_NOW + 61), live],
      SERVER_NOW + 62,
    ),
    WATCH_AT + 62_010,
  );
  assert.equal(opened, true);
});

test("a fresh watch still reports a request that was already running", () => {
  // The rearm write-off must not seed a session that never saw the full page.
  const opened = observeResponse(
    watchFrom(WATCH_AT),
    snapshot([entry("apireq_live", "running", SERVER_NOW - 90)]),
    WATCH_AT + 10,
  );
  assert.equal(opened, true);
});
