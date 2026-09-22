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

test("a reading that never settles cannot silence the feature for the session", () => {
  // inFlight is the only slot. A fetch that never resolves, or a body that never finishes
  // reading, would hold it for the life of the page: every later check either queues behind it
  // or is handed it, so the disk warning goes quiet for the rest of the session. That is the
  // one failure this feature cannot report on its own, so the read is bounded.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  assert.match(check, /disposableTimeoutSignal/);
  assert.match(check, /READ_TIMEOUT_MS/);
  // The signal has to reach the request, not merely be constructed.
  assert.match(check, /authFetch\("\/api\/system\/disk", \{ signal: timeout\.signal \}\)/);
  // The helper's documented contract: dispose once settled, or abort listeners accumulate.
  assert.match(check, /finally \{\s*\n\s*\/\/[^\n]*\n\s*timeout\.dispose\(\);/);
  // A timeout must read as "could not tell", never as a full disk.
  const read = check.slice(check.indexOf("async function readDisk"));
  assert.match(read.slice(0, read.indexOf("\n}")), /return null;/);
});

test("a notifier that throws loses one toast, not the promise and not the session", () => {
  // observeDiskPressure records the crossing BEFORE the notifier runs, so a throwing notifier
  // would lose the warning and reject a detached promise nobody awaits.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  const body = check.slice(check.indexOf("function runCheck"));
  const observe = body.indexOf("observeDiskPressure(disk)");
  const guarded = body.indexOf("try {\n      notifier(level, disk);");
  assert.ok(guarded !== -1, "the notifier call is not guarded");
  assert.ok(observe < guarded, "the crossing is spent before the notifier is even attempted");
  // and inFlight is still cleared, so one bad toast cannot wedge the slot either
  assert.match(body, /\.finally\(\(\) => \{\s*\n\s*inFlight = null;/);
});

test("a model load asks the disk too, because the backend downloads inside it", () => {
  // The download manager is not the only way bytes reach the cache, so requestStart is not the
  // whole funnel. Selecting an uncached model in Chat calls loadModel, and the BACKEND fetches
  // the repo inside that one request (_maybe_auto_download_model in routes/inference.py): no
  // download job is created, so neither requestStart nor the poll loop's finalize ever runs.
  // Left to the mount reading alone, a load that fills the disk warns nobody until the next
  // Hub operation, which is exactly the user this notice is for.
  const api = readFileSync(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /import \{ checkDiskSpace \}/);

  const load = api.slice(api.indexOf("export async function loadModel"));
  const body = load.slice(0, load.indexOf("\nexport "));
  // Before the request, throttled: picking through several models costs one reading.
  assert.match(body, /void checkDiskSpace\(\);/);
  // And after it, forced, for the reason finalize is forced: the reading has to be taken after
  // the write, and unforced it is swallowed by the interval or handed the pre-load figure.
  assert.match(body, /void checkDiskSpace\(\{ force: true \}\);/);
  // In a finally: a load that FAILED is the likeliest one to have filled the disk doing it.
  assert.ok(
    body.indexOf("} finally {") < body.indexOf("void checkDiskSpace({ force: true })"),
    "the post-load reading must be in the finally, so a failed load still reports",
  );
  // void, never awaited: a disk reading must not gate or delay a model load.
  assert.doesNotMatch(body, /await checkDiskSpace/);
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

test("a forced reading is taken after the request that is already in flight", () => {
  // The pre-start reading and the completion reading are different questions about the same
  // disk, and returning the first as the answer to the second reports the space as it was
  // BEFORE the download wrote anything.
  const check = readFileSync(
    new URL("../src/features/settings/low-disk-check.ts", import.meta.url),
    "utf8",
  );
  assert.match(check, /if \(!options\.force\) return inFlight;/);
  // Chained behind it, and coalesced: one waiting, not one per caller.
  assert.match(check, /queued = inFlight/);
  assert.match(check, /if \(!queued\)/);
});

test("the download that used the space is the one that reports it", () => {
  // requestStart reads the disk BEFORE a download, which is the right moment to refuse one. A
  // download that starts with room and then eats it crosses the threshold with nobody looking:
  // there is no interval, so without a completion-side reading the warning waits for the next
  // download attempt. finalize is the single terminal path for complete, cancelled and error.
  const loop = readFileSync(
    new URL("../src/features/hub/download-manager/poll-loop.ts", import.meta.url),
    "utf8",
  );
  assert.match(loop, /import \{ checkDiskSpace \}/);
  const finalize = loop.slice(loop.indexOf("export function finalize"));
  const body = finalize.slice(0, finalize.indexOf("\nexport "));
  // void, not await: a reading must never delay the teardown of a finished job. Forced, or the
  // interval swallows it for any download shorter than 30 s and the in-flight pre-download
  // reading is handed back in its place, which is the figure this call exists to correct.
  assert.match(body, /\n  void checkDiskSpace\(\{ force: true \}\);/);
  // After the early returns, or a job that was already terminal asks again on every poll.
  assert.ok(
    body.indexOf("TERMINAL_DISPLAY_STATES") < body.indexOf("void checkDiskSpace("),
    "the reading has to sit after the already-terminal guard, or a job that has already "
      + "finished asks again on every poll",
  );
});

test("the notice is owner only, because its action is", () => {
  // The shell mounts this hook for every authenticated user, but "resources" is in
  // OWNER_ONLY_SETTINGS_TABS, so resolveSettingsTab sends a managed account to General.
  // Warning someone about a disk and handing them a button that lands somewhere else is
  // worse than not warning them: the caches are install-wide state they cannot clear.
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
  // The notice is owner-only, but the download path calls checkDiskSpace for everyone.
  // observeDiskPressure RECORDS the level it returns, so running it with nobody listening
  // marks the disk as already warned about: a managed account's download, or a reading that
  // lands after logout, would leave an owner signing in later in the same SPA session hearing
  // nothing until free space recovered past the re-arm margin.
  //
  // Asserted on the source rather than by driving the module, because low-disk-check.ts
  // imports through the "@/" alias and cannot be loaded by the bare node test runner.
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
