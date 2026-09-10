// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The property that matters is not "does it fire" but "how often": once per
// crossing, and never again until free space climbs clear of that level.

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";

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
  // The same reading arrives on every poll.
  const readings = [100, 30, 19, 18, 18, 17, 18, 19];
  assert.deepEqual(announce(readings), ["low"]);
});

test("falling from low to critical is worth saying again", () => {
  assert.deepEqual(announce([100, 15, 12, 4, 3]), ["low", "critical"]);
});

test("a critical disk does not re-warn as low when it recovers a little", () => {
  // 6 GB clears critical and is still below low, which was already said.
  assert.deepEqual(announce([3, 11, 4]), ["critical", "critical"]);
  assert.deepEqual(announce([3, 8, 4]), ["critical"]);
});

test("recovering out of critical does not disarm the low warning too", () => {
  // 21 GB clears critical's margin but not low's own re-arm point of 25 GB, and
  // the instantaneous pressure forgot low there and re-toasted on the next dip.
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
  // subscribeSystemInfo only adds a callback to a Set: it never requests
  // /api/system, so a bare subscriber hears nothing unless the floating monitor
  // or the resources tab happens to be open, which is not this notice's user.
  // The notice asks for a reading itself, once, when the shell mounts.
  assert.match(hook, /checkDiskSpace\(\{ force: true \}\)/);
  assert.match(hook, /setLowDiskNotifier\(/);
  assert.match(hook, /toast\.warning\(/);
  // The description interpolates {free} and {total} bare, so the unit has to be
  // in the value or the toast reads "4.0 free of 500".
  assert.match(
    hook,
    /\$\{value >= 100 \? value\.toFixed\(0\) : value\.toFixed\(1\)\} GB/,
  );
  assert.match(
    en.settings.resources.storage.lowDisk.description,
    /\{free\} free of \{total\}/,
  );
  // The toast has to lead somewhere: the Storage section it is about.
  assert.match(hook, /scrollTarget: "resources-caches"/);
});

test("nothing polls the disk on a timer", () => {
  // The point of the redesign. An interval here runs in every open tab forever to answer a
  // question whose answer only changes when something writes to the disk, and the route it used
  // to poll, /api/system, enumerates GPUs and reads package metadata on the way.
  for (const file of [
    "../src/features/settings/hooks/use-low-disk-notice.ts",
    "../src/features/settings/low-disk-check.ts",
  ]) {
    const src = readFileSync(new URL(file, import.meta.url), "utf8");
    assert.doesNotMatch(src, /setInterval|pollMs|useSystemInfo/, file);
  }
});

test("a download asks the disk on the way past", () => {
  // requestStart is the single funnel every download goes through, which is what lets the
  // notice drop its timer without going silent for the user who is actually filling the disk.
  const funnel = readFileSync(
    new URL(
      "../src/features/hub/download-manager/transport-conflict.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(funnel, /import \{ checkDiskSpace \}/);
  // void, not await: a disk reading is advice and must never gate or delay a download.
  assert.match(funnel, /\n  void checkDiskSpace\(\);/);

  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  // One syscall, not the GPU-enumerating route.
  assert.match(check, /"\/api\/system\/disk"/);
  assert.doesNotMatch(check, /"\/api\/system"/);
  // A burst of downloads is still one disk.
  assert.match(check, /MIN_INTERVAL_MS/);
  // A host that cannot answer must not surface an error or stop anything.
  assert.match(check, /return null;/);
});

test("the disk route does one syscall and no directory walk", () => {
  const main = readFileSync(
    new URL("../../backend/main.py", import.meta.url),
    "utf8",
  );
  const route = main.slice(
    main.indexOf('@app.get("/api/system/disk")'),
    main.indexOf('@app.get("/api/system/gpu-visibility")'),
  );
  assert.ok(route.length > 0, "the disk route is gone");
  assert.match(route, /shutil\.disk_usage/);
  // Comments and the docstring stripped first, or a comment merely NAMING one of these names
  // fails the check below, and a comment explaining why psutil is absent is exactly the kind of
  // comment this route wants.
  const code = route
    .replace(/"""[\s\S]*?"""/g, "")
    .split("\n")
    .filter((line) => !line.trim().startsWith("#"))
    .join("\n");
  // Not os.walk, not scandir, not the cache inventory: this is asked for on the way into a
  // download and cannot afford to walk a multi-gigabyte cache.
  assert.doesNotMatch(code, /os\.walk|scandir|cache_inventory|psutil/);
  // Measured where the bytes land, which is a different volume whenever the model cache is on
  // another disk from the filesystem root.
  assert.match(route, /hf_default_cache_dir/);
});

test("the notice is mounted in the app shell, not on one route", () => {
  // A full disk belongs to whichever route the user is on, and the root layout
  // is the only thing that stays mounted across navigation.
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
