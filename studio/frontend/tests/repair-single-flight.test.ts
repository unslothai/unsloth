// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One preflight and one repair at a time. Five Retry clicks after a crash ran five preflights,
// each answering managed_stale with can_auto_repair, so five start_managed_repair calls raced
// for one installer and "Installation is already running." landed over the winner's progress.
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
    /finally \{\s*releasePreflight\(\);\s*\}/,
    "a failed preflight must release the flag or Retry is dead for the session",
  );
  // Held past the probe, the Retry that server-start-timeout offers from inside the port poll is
  // swallowed after clearing the error: error screen, nothing running.
  const release = body.indexOf("releasePreflight();");
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
  // Released twice, and by then a later call may hold the flag: an unowned clear would let a
  // third preflight through.
  assert.match(
    body,
    /const releasePreflight = \(\) => \{\s*if \(!ownsPreflight\) return;\s*ownsPreflight = false;\s*preflightInFlightRef\.current = false;\s*\};/,
    "the release must be a no-op once this call has handed the flag on",
  );
  assert.ok(
    !/preflightInFlightRef\.current = false;[\s\S]*preflightInFlightRef\.current = false;/.test(body),
    "only releasePreflight may clear the flag",
  );
});

test("a repair already in flight is not started again", () => {
  const body = section("async function startRepair(", "async function runRepair(");
  assert.ok(
    body.indexOf("if (repairInFlightRef.current) return;") > 0,
    "startRepair must refuse re-entry",
  );
  assert.match(
    body,
    /const releaseRepair = \(\) => \{\s*if \(!ownsRepair\) return;\s*ownsRepair = false;\s*repairInFlightRef\.current = false;\s*\};/,
    "the release must be a no-op once this call has handed the flag on",
  );
  assert.match(
    body,
    /try \{\s*await runRepair\(options, releaseRepair\);\s*\} finally \{\s*releaseRepair\(\);\s*\}/,
    "the flag must be released on every exit, including an early return",
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
  // The repair ends with the native call; holding the flag across the start that follows
  // swallows the Retry server-start-timeout offers.
  const native = body.indexOf('invoke("start_managed_repair"');
  const release = body.indexOf("releaseRepair();");
  const start = body.indexOf("await startManagedServer();");
  assert.ok(native < release && release < start, "release once the native repair returns");
});
