// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// The overlay .tsx pulls in motion, hugeicons and the router, so it cannot be imported
// here. These drive its watch through the exact call order its effects use.
import {
  type ApiMonitorWatch,
  type WatchedEntry,
  type WatchedResponse,
  createWatch,
  observeResponse,
  standDownWatch,
  startWatching,
} from "../src/features/api-monitor/new-traffic.ts";

// The server's clock: entry timestamps are its time.time(), never a browser instant.
const SERVER_NOW = 1_000_000;
// performance.now() when the poll first stood up.
const WATCH_AT = 1_000;
// An hour spent with automatic opening switched off.
const OPT_OUT_MS = 3_600_000;

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
  serverTime: number = SERVER_NOW,
): WatchedResponse {
  // biome-ignore lint/style/useNamingConvention: API schema
  return { entries, server_time: serverTime };
}

/**
 * The overlay as its effects run it. `standDown` is the poll effect returning early for the
 * opt out; `standUp` is it running again; `observe` is the observer effect, which re-runs on
 * any store change and so sees the snapshot already in hand as well as each fresh one.
 */
function overlay(): {
  watch: ApiMonitorWatch;
  standDown: () => void;
  standUp: (nowMs: number) => void;
  observe: (response: WatchedResponse, nowMs: number) => boolean;
} {
  const watch = createWatch(0);
  return {
    watch,
    standDown: () => standDownWatch(watch),
    standUp: (nowMs) => startWatching(watch, nowMs),
    observe: (response, nowMs) => observeResponse(watch, response, nowMs),
  };
}

test("turning automatic opening back on does not pop the panel for the opt-out backlog", () => {
  const ui = overlay();
  const backlog = [entry("apireq_old", "completed", SERVER_NOW - 90)];
  const first = snapshot(backlog);
  ui.standUp(WATCH_AT);
  assert.equal(ui.observe(first, WATCH_AT + 10), false);

  // "Stop opening this automatically": the poll stands down with that snapshot in hand.
  ui.standDown();
  assert.equal(ui.observe(first, WATCH_AT + 20), false);

  // An hour of curl against the API key, none of it polled for.
  const during = [
    entry("apireq_curl_b", "completed", SERVER_NOW + 900),
    entry("apireq_curl_a", "completed", SERVER_NOW + 300),
    ...backlog,
  ];

  // The switch in settings goes back on: the poll stands up and the observer re-runs with
  // the stale snapshot before the first fetch of the new watch resolves.
  ui.standUp(WATCH_AT + OPT_OUT_MS);
  assert.equal(ui.observe(first, WATCH_AT + OPT_OUT_MS + 1), false);
  const opened = ui.observe(
    snapshot(during, SERVER_NOW + 3600),
    WATCH_AT + OPT_OUT_MS + 10,
  );
  assert.equal(opened, false);
});

test("the snapshot in hand at the stand down cannot spend the opt-out re-arm", () => {
  // Same run, one effect at a time: the re-arm has to survive the observer re-running on
  // the autoOpen change itself, both when it goes off and when it comes back.
  const ui = overlay();
  const first = snapshot([entry("apireq_old", "completed", SERVER_NOW - 90)]);
  ui.standUp(WATCH_AT);
  ui.observe(first, WATCH_AT + 10);
  ui.standDown();
  assert.equal(ui.watch.seeded, false);
  ui.observe(first, WATCH_AT + 20);
  assert.equal(ui.watch.seeded, false, "a re-fold must not seed the re-arm");
  ui.standUp(WATCH_AT + OPT_OUT_MS);
  ui.observe(first, WATCH_AT + OPT_OUT_MS + 1);
  assert.equal(
    ui.watch.seeded,
    false,
    "still owed to the first fresh snapshot",
  );
});

test("a call made after automatic opening is back on still opens the panel", () => {
  const ui = overlay();
  const backlog = [entry("apireq_old", "completed", SERVER_NOW - 90)];
  ui.standUp(WATCH_AT);
  ui.observe(snapshot(backlog), WATCH_AT + 10);
  ui.standDown();
  const during = [
    entry("apireq_curl", "completed", SERVER_NOW + 300),
    ...backlog,
  ];
  ui.standUp(WATCH_AT + OPT_OUT_MS);
  ui.observe(snapshot(during, SERVER_NOW + 3600), WATCH_AT + OPT_OUT_MS + 10);
  const opened = ui.observe(
    snapshot(
      [entry("apireq_next", "running", SERVER_NOW + 3605), ...during],
      SERVER_NOW + 3606,
    ),
    WATCH_AT + OPT_OUT_MS + 1_510,
  );
  assert.equal(opened, true);
});

test("a session that starts opted out still reports its first snapshot", () => {
  // Nothing has been folded in, so there is no backlog to write off; the persisted opt out
  // must not turn into a permanent silence once the switch goes back on.
  const ui = overlay();
  ui.standDown();
  assert.equal(ui.watch.resumed, false);
  ui.standUp(WATCH_AT);
  const opened = ui.observe(
    snapshot([entry("apireq_live", "running", SERVER_NOW - 5)]),
    WATCH_AT + 10,
  );
  assert.equal(opened, true);
});

test("an opt out with nothing behind it leaves the retained ids read", () => {
  // The ids held at the stand down were already diffed away, re-arm or not.
  const ui = overlay();
  const backlog = [entry("apireq_old", "completed", SERVER_NOW - 90)];
  ui.standUp(WATCH_AT);
  ui.observe(snapshot(backlog), WATCH_AT + 10);
  ui.standDown();
  ui.standUp(WATCH_AT + OPT_OUT_MS);
  const opened = ui.observe(
    snapshot(backlog, SERVER_NOW + 3600),
    WATCH_AT + OPT_OUT_MS + 10,
  );
  assert.equal(opened, false);
});
