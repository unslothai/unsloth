// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  // /api/system reports zeros when psutil raised.
  assert.equal(diskPressure(disk(0, 0)), null);
  assert.equal(diskPressure({ free_gb: Number.NaN, total_gb: 100 }), null);
});

test("crossing the low threshold warns exactly once", () => {
  const readings = [100, 30, 19, 18, 18, 17, 18, 19];
  assert.deepEqual(announce(readings), ["low"]);
});

test("falling from low to critical is worth saying again", () => {
  assert.deepEqual(announce([100, 15, 12, 4, 3]), ["low", "critical"]);
});

test("a critical disk does not re-warn as low when it recovers a little", () => {
  assert.deepEqual(announce([3, 11, 4]), ["critical", "critical"]);
  assert.deepEqual(announce([3, 8, 4]), ["critical"]);
});

test("recovering out of critical does not disarm the low warning too", () => {
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
  // subscribeSystemInfo never fetches, so the notice requests one reading on mount.
  assert.match(hook, /checkDiskSpace\(\{ force: true \}\)/);
  assert.match(hook, /setLowDiskNotifier\(/);
  assert.match(hook, /toast\.warning\(/);
  // The description interpolates {free}/{total} bare, so values must carry units.
  assert.match(
    hook,
    /\$\{value >= 100 \? value\.toFixed\(0\) : value\.toFixed\(1\)\} GB/,
  );
  assert.match(
    en.settings.resources.storage.lowDisk.description,
    /\{free\} free of \{total\}/,
  );
  assert.match(hook, /scrollTarget: "resources-caches"/);
});

test("nothing polls the disk on a timer", () => {
  // No interval: disk space only changes when something writes.
  for (const file of [
    "../src/features/settings/hooks/use-low-disk-notice.ts",
    "../src/features/settings/low-disk-check.ts",
  ]) {
    const src = readFileSync(new URL(file, import.meta.url), "utf8");
    assert.doesNotMatch(src, /setInterval|pollMs|useSystemInfo/, file);
  }
});

test("a download asks the disk on the way past", () => {
  const funnel = readFileSync(
    new URL(
      "../src/features/hub/download-manager/transport-conflict.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(funnel, /import \{ checkDiskSpace \}/);
  // void, not await: a disk reading must never delay a download.
  assert.match(funnel, /\n  void checkDiskSpace\(\);/);

  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  assert.match(check, /"\/api\/system\/disk"/);
  assert.doesNotMatch(check, /"\/api\/system"/);
  assert.match(check, /MIN_INTERVAL_MS/);
  assert.match(check, /return null;/);
});

test("a reading that never settles cannot silence the feature for the session", () => {
  // inFlight is the only slot, so a hung fetch would silence warnings; the read is bounded.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  assert.match(check, /disposableTimeoutSignal/);
  assert.match(check, /READ_TIMEOUT_MS/);
  assert.match(check, /authFetch\("\/api\/system\/disk", \{ signal: timeout\.signal \}\)/);
  assert.match(check, /finally \{\s*\n\s*\/\/[^\n]*\n\s*timeout\.dispose\(\);/);
  const read = check.slice(check.indexOf("async function readDisk"));
  assert.match(read.slice(0, read.indexOf("\n}")), /return null;/);
});

test("a notifier that throws loses one toast, not the promise and not the session", () => {
  // The crossing is recorded before the notifier runs, so a throw must be caught.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  const body = check.slice(check.indexOf("function runCheck"));
  const observe = body.indexOf("observeDiskPressure(disk)");
  const guarded = body.indexOf("try {\n      notifier(level, disk);");
  assert.ok(guarded !== -1, "the notifier call is not guarded");
  assert.ok(observe < guarded, "the crossing is spent before the notifier is even attempted");
  assert.match(body, /\.finally\(\(\) => \{\s*\n\s*inFlight = null;/);
});

test("a model load asks the disk too, because the backend downloads inside it", () => {
  // loadModel downloads uncached models server-side without a download job.
  const api = readFileSync(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /import \{ checkDiskSpace \}/);

  const load = api.slice(api.indexOf("export async function loadModel"));
  const body = load.slice(0, load.indexOf("\nexport "));
  assert.match(body, /void checkDiskSpace\(\);/);
  // Forced after the load so the reading reflects the write.
  assert.match(body, /void checkDiskSpace\(\{ force: true \}\);/);
  assert.ok(
    body.indexOf("} finally {") < body.indexOf("void checkDiskSpace({ force: true })"),
    "the post-load reading must be in the finally, so a failed load still reports",
  );
  assert.doesNotMatch(body, /await checkDiskSpace/);
});

test("a training run asks the disk on both sides of the worker's download", () => {
  // Training pulls models via the worker, bypassing requestStart and loadModel.
  const api = readFileSync(
    new URL("../src/features/training/api/train-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /import \{ checkDiskSpace \}/);
  const start = api.slice(api.indexOf("export async function startTraining"));
  const body = start.slice(0, start.indexOf("\nexport "));
  assert.match(body, /void checkDiskSpace\(\);/);
  // Not forced here: the worker downloads long after this request returns.
  assert.doesNotMatch(body, /checkDiskSpace\(\{ force: true \}\)/);

  const watch = readFileSync(
    new URL(
      "../src/features/training/hooks/use-training-completion-watch.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(watch, /import \{ checkDiskSpace \}/);
  assert.match(watch, /void checkDiskSpace\(\{ force: true \}\);/);
  const cleanup = watch.slice(watch.indexOf("return () => {"));
  assert.ok(
    cleanup.indexOf("void checkDiskSpace({ force: true })") !== -1,
    "the post-run reading must be in the watch cleanup, not on every tick",
  );
  assert.doesNotMatch(watch, /await checkDiskSpace/);
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
  // Strip comments first so a comment naming these does not fail the check.
  const code = route
    .replace(/"""[\s\S]*?"""/g, "")
    .split("\n")
    .filter((line) => !line.trim().startsWith("#"))
    .join("\n");
  // Must not walk a multi-gigabyte cache.
  assert.doesNotMatch(code, /os\.walk|scandir|cache_inventory|psutil/);
  // The model cache may be on a different volume from root.
  assert.match(route, /hf_default_cache_dir/);
});

test("the notice is mounted in the app shell, not on one route", () => {
  const root = readFileSync(
    new URL("../src/app/routes/__root.tsx", import.meta.url),
    "utf8",
  );
  assert.match(root, /function LowDiskNoticeMount\(\)/);
  assert.match(root, /useLowDiskNotice\(\)/);
  assert.match(root, /!isAuthFlowRoute && <LowDiskNoticeMount \/>/);

  // Exactly one mount, or the toast fires twice.
  const callers = sourceFiles(new URL("../src/", import.meta.url)).filter(
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

test("a forced reading is taken after the request that is already in flight", () => {
  // A completion reading must not be handed the pre-download in-flight result.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  assert.match(check, /if \(!options\.force\) return inFlight;/);
  assert.match(check, /queued = inFlight/);
  assert.match(check, /if \(!queued\)/);
});

test("the download that used the space is the one that reports it", () => {
  // finalize is the single terminal path for complete, cancelled and error.
  const loop = readFileSync(
    new URL("../src/features/hub/download-manager/poll-loop.ts", import.meta.url),
    "utf8",
  );
  assert.match(loop, /import \{ checkDiskSpace \}/);
  const finalize = loop.slice(loop.indexOf("export function finalize"));
  const body = finalize.slice(0, finalize.indexOf("\nexport "));
  // Forced, or the throttle returns the stale pre-download reading.
  assert.match(body, /\n  void checkDiskSpace\(\{ force: true \}\);/);
  assert.ok(
    body.indexOf("TERMINAL_DISPLAY_STATES") < body.indexOf("void checkDiskSpace("),
    "the reading has to sit after the already-terminal guard, or a job that has already "
      + "finished asks again on every poll",
  );
});

test("the notice is owner only, because its action is", () => {
  // resources is owner-only, so managed accounts must not get a toast pointing there.
  const hook = readFileSync(
    new URL(
      "../src/features/settings/hooks/use-low-disk-notice.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /useIsAccountOwner\(\)/,
    "the low-disk notice does not ask whether the account owns this installation",
  );
  assert.match(
    hook,
    /if \(!isOwner\) return;/,
    "the notifier is registered before ownership is checked, so a managed account still gets the toast",
  );

  const visibility = readFileSync(
    new URL(
      "../src/features/settings/settings-tab-visibility.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    visibility,
    /OWNER_ONLY_SETTINGS_TABS[\s\S]*"resources"/,
    "resources stopped being owner-only, so this gate may no longer be the right one",
  );
});

test("a reading with no notifier registered does not spend the crossing", () => {
  // observeDiskPressure records the level, so running it without a listener swallows
  // the owner's later warning.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  const guard = check.indexOf("if (!notifier) return;");
  const observe = check.indexOf("observeDiskPressure(disk)");
  assert.ok(guard !== -1, "runCheck does not bail out when no notifier is registered");
  assert.ok(observe !== -1, "runCheck no longer observes disk pressure at all");
  assert.ok(
    guard < observe,
    "observeDiskPressure runs before the notifier check, so a crossing is spent unheard",
  );
});
