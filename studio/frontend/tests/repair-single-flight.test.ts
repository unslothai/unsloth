// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One preflight and one repair at a time.
//
// After a backend crash the startup screen offers Retry, and Retry runs the preflight. Five
// clicks two seconds apart ran five preflights; each came back managed_stale with
// can_auto_repair, and each started its own repair. The Rust side then saw five
// start_managed_repair calls racing for one installer and surfaced "Installation is already
// running." over the progress of the one that won.
//
// The hook cannot be rendered here, so the guards are read off the shipped source the way
// desktop-stop-intent.test.ts does beside it.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const hook = await readSrcAsync("hooks/use-tauri-backend.ts");

function section(from: string, to: string): string {
  const start = hook.indexOf(from);
  const end = hook.indexOf(to, start);
  assert.ok(start >= 0 && end > start, `could not find ${from} .. ${to}`);
  return hook.slice(start, end);
}

test("a preflight already in flight is not run again", () => {
  const body = section(
    "async function checkInstallAndStart()",
    "async function startManagedServer()",
  );
  const guard = body.indexOf("if (preflightInFlightRef.current) return;");
  const arm = body.indexOf("preflightInFlightRef.current = true;");
  const preflight = body.indexOf('invoke<DesktopPreflightResult>("desktop_preflight")');
  assert.ok(guard > 0 && arm > guard && preflight > arm, "the guard must sit ahead of the preflight");
  // The persisted stop still wins: it must not arm a flag it never releases.
  assert.ok(
    body.indexOf("hasServerStopIntent()") < guard,
    "the stop-intent check runs before the in-flight guard",
  );
  assert.match(
    body,
    /finally \{\s*preflightInFlightRef\.current = false;\s*\}/,
    "a failed preflight must release the flag or Retry is dead for the session",
  );
  // The flag covers the PROBE, not what the probe leads to. managed_ready awaits
  // startManagedServer, which parks in a 500 ms port poll; server-start-timeout offers Retry from
  // inside that wait, and a click arriving before the poll next wakes would otherwise be swallowed
  // by this flag after clearing the error, leaving the screen with no attempt running.
  const release = body.indexOf("preflightInFlightRef.current = false;");
  const dispatch = body.indexOf("switch (preflight.disposition)");
  assert.ok(
    release > preflight && release < dispatch,
    "the flag must be released once the probe returns, before any disposition is acted on",
  );
  assert.ok(
    body.indexOf("await startManagedServer()") > release,
    "no long await may run while the flag is held",
  );
  assert.ok(body.indexOf("await startRepair()") > release);
});

test("a repair already in flight is not started again", () => {
  const body = section("async function startRepair(", "async function runRepair(");
  assert.ok(
    body.indexOf("if (repairInFlightRef.current) return;") > 0,
    "startRepair must refuse re-entry",
  );
  assert.match(
    body,
    /repairInFlightRef\.current = true;\s*try \{\s*await runRepair\(options\);\s*\} finally \{\s*repairInFlightRef\.current = false;\s*\}/,
    "the flag must cover the whole repair and be released on every exit",
  );
  // The refused call returns without touching state: the running repair owns the screen.
  assert.ok(
    body.indexOf("if (repairInFlightRef.current) return;") <
      body.indexOf("repairInFlightRef.current = true;"),
  );
});

test("the repair body itself is unchanged in what it invokes", () => {
  const body = section("async function runRepair(", "async function startServer()");
  assert.match(body, /invoke\("start_managed_repair", \{ forceInstaller \}\)/);
  assert.match(body, /forcedRepairRef\.current = forceInstaller;/);
});
