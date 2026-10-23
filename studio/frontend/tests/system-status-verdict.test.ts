// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The verdict and preservation rule live in .tsx files that cannot be imported, so lift and
// evaluate them. DEFAULT_SYSTEM looks CPU-only, so rendering it misreports GPU hosts.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const tabSrc = await readSrcAsync("features/settings/tabs/resources-tab.tsx");
const hookSrc = await readSrcAsync("hooks/use-system.ts");

function lift(src: string, pattern: RegExp, what: string, where: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in ${where}`);
  return found[0];
}

const verdict = [
  lift(tabSrc, /const hasGpu =\n?[\s\S]*?;/, "hasGpu", "resources-tab.tsx"),
  lift(tabSrc, /const hostUnread = [\s\S]*?;/, "hostUnread", "resources-tab.tsx"),
  lift(tabSrc, /const gpuUnknown = [\s\S]*?;/, "gpuUnknown", "resources-tab.tsx"),
  lift(tabSrc, /const gpuUnknownLabel =\n?[\s\S]*?;/, "gpuUnknownLabel", "resources-tab.tsx"),
  lift(tabSrc, /const backendLabel = \([\s\S]*?\)\.toUpperCase\(\);/, "backendLabel", "resources-tab.tsx"),
  lift(tabSrc, /const hostReading = [\s\S]*?;/, "hostReading", "resources-tab.tsx").replace(
    "(value: string)",
    "(value)",
  ),
].join("\n");

const gpuSection = lift(
  tabSrc,
  /gpuUnknown \? gpuUnknownLabel : t\("settings\.resources\.gpu\.noGpu"\)/,
  "the GPU section fallback",
  "resources-tab.tsx",
);

const t = (key: string) => key;

interface Snapshot {
  status: string;
  device_backend: string;
  gpu: { available: boolean; backend?: string; devices: unknown[] };
  ml_packages?: { torch?: string };
}

function verdictFor(systemInfo: Snapshot) {
  const run = new Function(
    "systemInfo",
    "displayedGpu",
    "metrics",
    "t",
    "unknownLabel",
    `${verdict}
     return {
       gpuUnknown,
       hasGpu,
       gpuSection: hasGpu ? "<device rows>" : (${gpuSection}),
       backendRow: hostReading(backendLabel),
       torchRow: hostReading(
         systemInfo.ml_packages?.torch ?? "settings.resources.environment.notInstalled",
       ),
     };`,
  );
  return run(
    systemInfo,
    systemInfo.gpu,
    { devices: systemInfo.gpu.devices },
    t,
    "settings.resources.environment.unknown",
  ) as {
    gpuUnknown: boolean;
    hasGpu: boolean;
    gpuSection: string;
    backendRow: string;
    torchRow: string;
  };
}

const NO_GPU = "settings.resources.gpu.noGpu";
const DETECTING = "settings.resources.gpu.detecting";
const UNREADABLE = "settings.resources.gpu.unreadable";
const UNKNOWN = "settings.resources.environment.unknown";
const NOT_INSTALLED = "settings.resources.environment.notInstalled";

const cpuShaped = (status: string): Snapshot => ({
  status,
  device_backend: "cpu",
  gpu: { available: false, devices: [] },
});

test("the placeholder is shaped exactly like a CPU-only host, which is why this gate exists", () => {
  const placeholder = lift(
    hookSrc,
    /const DEFAULT_SYSTEM: SystemInfoResponse = \{[\s\S]*?\n\};/,
    "DEFAULT_SYSTEM",
    "use-system.ts",
  );
  assert.match(placeholder, /status: "pending"/);
  assert.match(placeholder, /device_backend: "cpu"/);
  assert.match(placeholder, /gpu: \{ available: false, devices: \[\] \}/);
});

test("before the host has been read, the tab says it is checking, not that there is no GPU", () => {
  const out = verdictFor(cpuShaped("pending"));
  assert.equal(out.gpuUnknown, true);
  assert.equal(out.gpuSection, DETECTING);
  assert.notEqual(out.gpuSection, NO_GPU);
});

test("after the read settles empty, the tab says so, and still does not claim there is no GPU", () => {
  const out = verdictFor(cpuShaped("unavailable"));
  assert.equal(out.gpuUnknown, true);
  assert.equal(out.gpuSection, UNREADABLE);
  assert.notEqual(out.gpuSection, NO_GPU);
  assert.notEqual(out.gpuSection, DETECTING);
});

test("the Environment rows wait for the same read", () => {
  for (const status of ["pending", "unavailable"]) {
    const out = verdictFor(cpuShaped(status));
    assert.equal(out.backendRow, UNKNOWN, `backend row on a ${status} host`);
    assert.equal(out.torchRow, UNKNOWN, `torch row on a ${status} host`);
  }
});

test("no tile fabricates a usage percentage for a host it has not read", () => {
  // An unread host is not an idle one, so no tile may pass 0.
  const tiles = tabSrc.match(/<MetricTile\b[\s\S]*?\/>/g) ?? [];
  // Count opening tags too: the optional models-disk tile may be missed by the match above.
  const opened = (tabSrc.match(/<MetricTile\b/g) ?? []).length;
  assert.ok(tiles.length >= 4, `the Live monitor's tiles: found ${tiles.length}`);
  assert.equal(tiles.length, opened, "every <MetricTile> is captured whole");
  for (const tile of tiles) {
    const label = /label=\{t\("([^"]+)"\)\}/.exec(tile)?.[1] ?? "?";
    const percent = /percent=\{([^}]*)\}/.exec(tile)?.[1] ?? "";
    assert.match(
      percent,
      /hostUnread \? null :|vramUsageKnown \? metrics\.vramPercent : null/,
      `${label} gates its percentage`,
    );
  }
});

test("the two non-answers are worded for the whole host, not just its GPUs", () => {
  const detail = lift(tabSrc, /const hostUnreadDetail =\n?[\s\S]*?;/, "hostUnreadDetail", "resources-tab.tsx");
  const forStatus = (status: string) =>
    new Function("systemInfo", "t", `${detail}\nreturn hostUnreadDetail;`)({ status }, t);
  assert.equal(forStatus("unavailable"), UNREADABLE);
  assert.equal(forStatus("pending"), "common.loading");
  assert.notEqual(forStatus("pending"), DETECTING, "which is about GPUs specifically");
});

test("a host that really has no GPU still gets the CPU-only verdict, and its real rows", () => {
  const out = verdictFor(cpuShaped("ready"));
  assert.equal(out.gpuUnknown, false);
  assert.equal(out.gpuSection, NO_GPU);
  assert.equal(out.backendRow, "CPU");
  assert.equal(out.torchRow, NOT_INSTALLED);
});

test("a host with a GPU gets none of the three fallbacks", () => {
  const out = verdictFor({
    status: "ready",
    device_backend: "rocm",
    gpu: { available: true, backend: "rocm", devices: [{ name: "Radeon RX 9060 XT" }] },
    ml_packages: { torch: "2.11.0+rocm7.13.0" },
  });
  assert.equal(out.gpuUnknown, false);
  assert.equal(out.hasGpu, true);
  assert.equal(out.gpuSection, "<device rows>");
  assert.equal(out.backendRow, "ROCM");
  assert.equal(out.torchRow, "2.11.0+rocm7.13.0");
  for (const forbidden of [NO_GPU, DETECTING, UNREADABLE, UNKNOWN]) {
    assert.notEqual(out.gpuSection, forbidden);
    assert.notEqual(out.backendRow, forbidden);
  }
});

test("a failed poll preserves the whole reading on screen, not just its status", () => {
  // Only the identity short-circuit keeps the reading; the branch rebuilds from DEFAULT_SYSTEM.
  const branch = lift(
    hookSrc,
    /\(previous\) => \{\n\s*const status = settledFailureStatus\([\s\S]*?\n\s*\}/,
    "the setSystemInfo failure callback",
    "use-system.ts",
  );
  const DEFAULT_SYSTEM = { status: "pending", device_backend: "cpu" };
  const settledFailureStatus = (previous: string) =>
    previous === "ready" ? "ready" : "unavailable";
  const onFailure = new Function(
    "DEFAULT_SYSTEM",
    "settledFailureStatus",
    `return ${branch};`,
  )(DEFAULT_SYSTEM, settledFailureStatus) as (previous: unknown) => { status: string };

  const reading = { status: "ready", device_backend: "rocm" };
  assert.equal(onFailure(reading), reading, "the same object, so React does not re-render");

  const settled = onFailure({ ...DEFAULT_SYSTEM });
  assert.equal(settled.status, "unavailable", "a placeholder does move, once, off pending");
  assert.equal(onFailure(settled), settled, "and does not churn on every later failure");
});

test("a success published by another consumer clears a placeholder but not a reading", () => {
  // Nothing retries a settled placeholder while polling is off.
  const rule = lift(
    hookSrc,
    /\(previous\) => \(previous\.status === "ready" \? previous : info\)/,
    "the subscribeSystemInfo callback",
    "use-system.ts",
  );
  const info = { status: "ready", device_backend: "rocm" };
  const onPublish = new Function("info", `return ${rule};`)(info) as (
    previous: unknown,
  ) => unknown;

  for (const status of ["pending", "unavailable"]) {
    assert.equal(onPublish({ status }), info, `a ${status} placeholder takes the snapshot`);
  }
  const mine = { status: "ready", device_backend: "cuda" };
  assert.equal(onPublish(mine), mine, "a reading already on screen is left to its own poll");
});
