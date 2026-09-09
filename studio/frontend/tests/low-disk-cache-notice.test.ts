// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The low-disk toast rides the System tab's existing 3s poll, so the property that
// matters is not "does it fire" but "how often": once per crossing, and never again
// until free space climbs clear of the level that was announced.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import {
  CRITICAL_DISK_FREE_GB,
  INITIAL_LOW_DISK_STATE,
  LOW_DISK_FREE_GB,
  type LowDiskState,
  REARM_MARGIN_GB,
  diskPressure,
  nextLowDiskNotice,
  observeDiskPressure,
  resetLowDiskNotices,
} from "../src/features/settings/low-disk.ts";

/** Every .ts/.tsx file under a directory, so "mounted once" can be checked. */
function sourceFiles(directory: URL): URL[] {
  const found: URL[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const child = new URL(
      `${entry.name}${entry.isDirectory() ? "/" : ""}`,
      directory,
    );
    if (entry.isDirectory()) found.push(...sourceFiles(child));
    else if (/\.tsx?$/.test(entry.name)) found.push(child);
  }
  return found;
}

function disk(freeGb: number, totalGb = 500) {
  // biome-ignore lint/style/useNamingConvention: API schema
  return { free_gb: freeGb, total_gb: totalGb };
}

/** Feed a run of readings and collect only the levels that were announced. */
function announce(readings: number[]): string[] {
  let state: LowDiskState = INITIAL_LOW_DISK_STATE;
  const said: string[] = [];
  for (const free of readings) {
    const decision = nextLowDiskNotice(state, disk(free));
    state = decision.state;
    if (decision.notify) said.push(decision.notify);
  }
  return said;
}

test("pressure is read from free space, not from percent used", () => {
  assert.equal(diskPressure(disk(400, 4000)), "ok");
  assert.equal(diskPressure(disk(LOW_DISK_FREE_GB + 1)), "ok");
  assert.equal(diskPressure(disk(LOW_DISK_FREE_GB)), "low");
  assert.equal(diskPressure(disk(CRITICAL_DISK_FREE_GB + 1)), "low");
  assert.equal(diskPressure(disk(CRITICAL_DISK_FREE_GB)), "critical");
  assert.equal(diskPressure(disk(0)), "critical");
});

test("a disk the host could not read is not a warning", () => {
  assert.equal(diskPressure({ free_gb: null, total_gb: null }), null);
  // /api/system reports zeros when psutil raised, which is not a full disk.
  assert.equal(diskPressure(disk(0, 0)), null);
  assert.equal(diskPressure({ free_gb: Number.NaN, total_gb: 100 }), null);
});

test("crossing the low threshold warns exactly once", () => {
  // 3s poll: the same reading arrives twenty times a minute.
  const readings = [100, 30, 19, 18, 18, 17, 18, 19];
  assert.deepEqual(announce(readings), ["low"]);
});

test("falling from low to critical is worth saying again", () => {
  assert.deepEqual(announce([100, 15, 12, 4, 3]), ["low", "critical"]);
});

test("a critical disk does not re-warn as low when it recovers a little", () => {
  // Freeing 6 GB clears critical but is still below the low threshold, so the
  // toast that is owed has already been given.
  assert.deepEqual(announce([3, 11, 4]), ["critical", "critical"]);
  assert.deepEqual(announce([3, 8, 4]), ["critical"]);
});

test("recovering out of critical does not disarm the low warning too", () => {
  // 21 GB clears critical and its 5 GB margin, but not low's own re-arm point of
  // 25 GB. Storing the instantaneous pressure forgot low there and paid for it
  // with a second low toast on the next dip.
  assert.deepEqual(announce([3, 21, 19]), ["critical"]);
  assert.deepEqual(announce([3, LOW_DISK_FREE_GB + REARM_MARGIN_GB, 19]), [
    "critical",
    "low",
  ]);
});

test("hovering on the threshold does not toast on every reading", () => {
  const hovering = [
    LOW_DISK_FREE_GB - 0.1,
    LOW_DISK_FREE_GB + 0.1,
    LOW_DISK_FREE_GB - 0.1,
    LOW_DISK_FREE_GB + 0.2,
    LOW_DISK_FREE_GB - 0.2,
  ];
  assert.deepEqual(announce(hovering), ["low"]);
});

test("a real recovery re-arms the warning", () => {
  const recovered = LOW_DISK_FREE_GB + REARM_MARGIN_GB;
  assert.deepEqual(announce([10, recovered, 10]), ["low", "low"]);
  // One byte short of the margin is not a recovery.
  assert.deepEqual(announce([10, recovered - 0.1, 10]), ["low"]);
});

test("an unreadable reading in the middle does not re-arm anything", () => {
  let state = nextLowDiskNotice(INITIAL_LOW_DISK_STATE, disk(10)).state;
  const blind = nextLowDiskNotice(state, { free_gb: null, total_gb: null });
  assert.equal(blind.notify, null);
  state = blind.state;
  assert.equal(nextLowDiskNotice(state, disk(10)).notify, null);
});

test("the session state is shared across mounts, not reset by them", () => {
  resetLowDiskNotices();
  assert.equal(observeDiskPressure(disk(10)), "low");
  // Reopening Settings mounts the tab again; the warning was already given.
  assert.equal(observeDiskPressure(disk(10)), null);
  assert.equal(observeDiskPressure(disk(2)), "critical");
  resetLowDiskNotices();
  assert.equal(observeDiskPressure(disk(10)), "low");
});

test("the notice fetches its own readings rather than waiting to be told", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/settings/hooks/use-low-disk-notice.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // This assertion used to be its own inverse: it required subscribeSystemInfo
  // and BANNED useSystemInfo, on the belief that the app runs a system poll to
  // attach to. It does not. subscribeSystemInfo only adds a callback to a Set;
  // it never requests /api/system and never replays the cached reading, so a
  // bare subscriber hears something only when the floating monitor or the
  // resources tab happens to be open and fetching. Both are lazily mounted and
  // gated on being open, which left this notice silent for precisely the user
  // who never opens Settings. The test passed the whole time.
  assert.match(hook, /useSystemInfo\(\{ pollMs: LOW_DISK_POLL_MS \}\)/);
  assert.match(hook, /observeDiskPressure\(systemInfo\.disk\)/);
  assert.match(hook, /toast\.warning\(/);
  // Slow on purpose: a disk fills over hours, and this one runs on every route.
  assert.match(hook, /LOW_DISK_POLL_MS = 60_000/);
  // The toast has to lead somewhere: the Storage section it is about.
  assert.match(hook, /scrollTarget: "resources-caches"/);
});

test("the notice is mounted in the app shell, not on one route", () => {
  // It lived in the resources tab first, which warned only the people already
  // looking at the disk figure, and then on /studio, which warned only the
  // people who were training. A full disk belongs to whichever route the user
  // is on. The root layout wraps every signed-in route and, unlike a page,
  // stays mounted across navigation.
  const root = readFileSync(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  assert.match(root, /function LowDiskNoticeMount\(\)/);
  assert.match(root, /useLowDiskNotice\(\)/);
  // Not during the auth flow: there is no session to warn, and no Settings to
  // send the toast's action to.
  assert.match(root, /!isAuthFlowRoute && <LowDiskNoticeMount \/>/);

  // Exactly one mount in the whole app, or the toast arrives twice on the route
  // that also mounts it.
  const callers = sourceFiles(new URL("../src/", import.meta.url)).filter(
    // The declaration itself reads "function useLowDiskNotice(): void".
    (file) =>
      /(?<!function )useLowDiskNotice\(\)/.test(readFileSync(file, "utf8")),
  );
  assert.deepEqual(
    callers.map((file) => file.pathname.split("/src/")[1]),
    ["app/routes/__root.tsx"],
  );

  const tab = readFileSync(
    new URL("../src/features/settings/tabs/resources-tab.tsx", import.meta.url),
    "utf8",
  );
  // ...and the level is still not observed a second time inside Settings.
  assert.doesNotMatch(tab, /observeDiskPressure/);
});

test("the cache row is reachable from settings search", () => {
  const search = readFileSync(
    new URL("../src/features/settings/settings-search.ts", import.meta.url),
    "utf8",
  );
  assert.match(search, /"settings\.resources\.storage\.caches\.label"/);
  assert.match(search, /"settings\.resources\.storage\.caches\.keywords"/);
});
