// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the UI says about a host whose GPUs PyTorch cannot use.
//
// A Windows in-app update that resolved torch from PyPI left a 2.11.0+cpu wheel beside two
// working RTX A4000s (#8473, HF discussion 87). The backend now reports those cards in
// gpu.physical_devices with a gpu.mismatch reason instead of going silent, and three places
// have to stop saying the opposite: the System tab's VRAM tile and GPU section, the sidebar's
// Train hint, and videoNavHint -- all of which told that host it needed to get a GPU.
//
// The derivations live in resources-tab.tsx and app-sidebar.tsx, which are .tsx and pull in
// the whole app, so they are lifted by regex and evaluated, exactly as
// system-status-verdict.test.ts lifts the host verdict beside them. hardware-verdict.ts is
// import-free by design and is imported outright.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { videoNavHint } = await import("../src/config/hardware-verdict.ts");
const { en } = await import("../src/i18n/locales/en.ts");

const tabSrc = await readFile(
  new URL("../src/features/settings/tabs/resources-tab.tsx", import.meta.url),
  "utf8",
);
const sidebarSrc = await readFile(
  new URL("../src/components/app-sidebar.tsx", import.meta.url),
  "utf8",
);

function lift(src: string, pattern: RegExp, what: string, where: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in ${where}`);
  return found[0];
}

// t() returns its key, so a failure names the string the user would have seen.
const t = (key: string) => key;

const CPU_BUILD = "settings.resources.gpu.mismatchCpuBuild";
const UNAVAILABLE = "settings.resources.gpu.mismatchUnavailable";
const NO_USABLE_GPU = "settings.resources.gpu.noUsableGpu";
const NO_GPU = "settings.resources.gpu.noGpu";
const UNKNOWN = "settings.resources.environment.unknown";

// gpuInventory itself carries a TS cast, so it is asserted on as source below and supplied
// here as an input instead. Everything downstream of it is run.
const derivation = [
  lift(tabSrc, /const gpuMismatch = [\s\S]*?;/, "gpuMismatch", "resources-tab.tsx"),
  lift(tabSrc, /const physicalDevices = [\s\S]*?;/, "physicalDevices", "resources-tab.tsx"),
  lift(
    tabSrc,
    /const gpuMismatchMessage = [\s\S]*?;/,
    "gpuMismatchMessage",
    "resources-tab.tsx",
  ),
].join("\n");

interface Inventory {
  mismatch?: { reason?: string; torch_version?: string | null } | null;
  physical_devices?: { name?: string }[];
}

function mismatchFor(gpuInventory: Inventory | null) {
  const run = new Function(
    "gpuInventory",
    "t",
    "unknownLabel",
    `${derivation}
     return { gpuMismatch, physicalDevices, gpuMismatchMessage };`,
  );
  return run(gpuInventory, t, UNKNOWN) as {
    gpuMismatch: { reason?: string } | null;
    physicalDevices: { name?: string }[];
    gpuMismatchMessage: string | null;
  };
}

test("a CPU-only wheel and a dead accelerator wheel get different sentences", () => {
  const cpuBuild = mismatchFor({
    mismatch: { reason: "torch_cpu_build", torch_version: "2.11.0+cpu" },
    physical_devices: [{ name: "NVIDIA RTX A4000" }, { name: "NVIDIA RTX A4000" }],
  });
  assert.equal(cpuBuild.gpuMismatchMessage, CPU_BUILD);
  assert.equal(cpuBuild.physicalDevices.length, 2);

  // Reinstalling torch is the wrong advice for a healthy cu124 wheel whose runtime will
  // not start, so the two reasons must not collapse into one string.
  const dead = mismatchFor({
    mismatch: { reason: "torch_cuda_unavailable", torch_version: "2.6.0+cu124" },
    physical_devices: [{ name: "NVIDIA RTX A4000" }],
  });
  assert.equal(dead.gpuMismatchMessage, UNAVAILABLE);
});

test("a healthy host, and one that really has no GPU, get no banner at all", () => {
  for (const inventory of [null, {}, { mismatch: null }] as (Inventory | null)[]) {
    const out = mismatchFor(inventory);
    assert.equal(out.gpuMismatch, null);
    assert.equal(out.gpuMismatchMessage, null);
    // No banner means no rows either: physical_devices present without a mismatch
    // would otherwise render an unexplained list of greyed-out cards.
    assert.deepEqual(out.physicalDevices, []);
  }
  const strayRows = mismatchFor({ physical_devices: [{ name: "NVIDIA RTX A4000" }] });
  assert.deepEqual(strayRows.physicalDevices, []);
});

test("the verdict is taken from a settled read only, and from the training view", () => {
  // The placeholder useSystemInfo starts from is shaped like a CPU-only host, which is the
  // whole reason system-status-verdict.test.ts exists; a banner derived from it would
  // accuse a host nobody has measured yet.
  const inventory = lift(
    tabSrc,
    /const gpuInventory = [\s\S]*?;\n/,
    "gpuInventory",
    "resources-tab.tsx",
  );
  assert.match(inventory, /hostUnread\s*\n?\s*\?\s*null/, "gated on the read having settled");
  // systemInfo.gpu, NOT displayedGpu: a Vulkan llama.cpp makes displayedGpu fall back to the
  // inference inventory, and a card Vulkan enumerates while torch cannot is exactly the host
  // that must be told (the second report in #8473).
  assert.match(inventory, /systemInfo\.gpu/);
  assert.doesNotMatch(inventory, /displayedGpu/);
});

test("the GPU section stops telling this host there is no GPU", () => {
  // The CPU-only host's own line has to survive untouched, so assert both branches.
  assert.match(
    tabSrc,
    new RegExp(`\\) : gpuMismatch \\? \\([\\s\\S]*?t\\("${NO_USABLE_GPU}"\\)`),
    "a host with unusable cards gets its own line",
  );
  assert.match(
    tabSrc,
    /gpuUnknown \? gpuUnknownLabel : t\("settings\.resources\.gpu\.noGpu"\)/,
    "and a host that really has no GPU still gets the CPU-only one",
  );
});

test("the VRAM tile stops reading as a CPU-only host", () => {
  const tiles = tabSrc.match(/<MetricTile\b[\s\S]*?\/>/g) ?? [];
  const vram = tiles.find((tile) => tile.includes("liveMonitor.vram"));
  assert.ok(vram, "the VRAM tile");
  // The mismatch branch must come BEFORE the noGpu fallback, or it never runs.
  const mismatchAt = vram.indexOf("liveMonitor.gpuUnusable");
  const noGpuAt = vram.indexOf("liveMonitor.noGpu");
  assert.ok(mismatchAt > -1, "the tile has a mismatch state");
  assert.ok(noGpuAt > -1, "and still has the CPU-only state");
  assert.ok(mismatchAt < noGpuAt, "the mismatch state is reached first");
});

test("the physically detected cards are shown, and never offered as devices", () => {
  // The banner renders physicalDevices; the selectable rows render metrics.devices, which
  // comes from the backend's `devices`. If the banner ever read metrics.devices the two
  // would merge, which is the failure mode the backend field split exists to prevent.
  const banner = lift(
    tabSrc,
    /\{gpuMismatch \? \(\n[\s\S]*?\n\s*\) : null\}/,
    "the mismatch banner",
    "resources-tab.tsx",
  );
  assert.match(banner, /physicalDevices\.map/);
  assert.doesNotMatch(banner, /metrics\.devices/);
  assert.match(banner, /settings\.resources\.gpu\.unusableDevice/);
});

// ========== The two navigation hints ==========

test("videoNavHint stops telling a two-GPU host to get a GPU", () => {
  for (const reason of ["torch_cpu_build", "torch_cuda_unavailable"]) {
    const hint = videoNavHint(true, reason);
    assert.ok(hint, `${reason} explains the disabled Video row`);
    assert.doesNotMatch(
      hint,
      /needs an NVIDIA or AMD GPU/,
      `${reason} is not a missing-GPU host`,
    );
    assert.match(hint, /PyTorch/, `${reason} names what is actually wrong`);
    // Unmeasured hosts keep their pass, as every other reason does.
    assert.equal(videoNavHint(false, reason), undefined);
  }
  // And the genuine no-GPU host keeps the sentence that is true for it.
  assert.equal(videoNavHint(true, "no_gpu"), "Video generation needs an NVIDIA or AMD GPU.");
});

test("the sidebar's Train hint stops doing the same", () => {
  const hint = lift(
    sidebarSrc,
    /const trainDisabledHint: string \| undefined = [\s\S]*?\n\s*: undefined;/,
    "trainDisabledHint",
    "app-sidebar.tsx",
  );
  const forReason = (chatOnlyReason: string, chatOnlyDetail: string | null) =>
    new Function(
      "chatOnlyMeasured",
      "chatOnlyReason",
      "chatOnlyDetail",
      `${hint.replace(": string | undefined", "")}\nreturn trainDisabledHint;`,
    )(true, chatOnlyReason, chatOnlyDetail) as string | undefined;

  for (const reason of ["torch_cpu_build", "torch_cuda_unavailable"]) {
    const withDetail = forReason(reason, "2.11.0+cpu");
    assert.ok(withDetail);
    assert.doesNotMatch(withDetail, /needs an NVIDIA or AMD GPU/);
    // The installed build is the one thing that makes this actionable to someone whose
    // update has already run, so it is named when the backend supplies it.
    assert.match(withDetail, /2\.11\.0\+cpu/);
    const withoutDetail = forReason(reason, null);
    assert.ok(withoutDetail);
    assert.doesNotMatch(withoutDetail, /needs an NVIDIA or AMD GPU/);
  }
  assert.equal(forReason("no_gpu", null), "Training needs an NVIDIA or AMD GPU.");
  assert.equal(forReason("detection_failed", null), undefined);
});

// ========== Strings ==========

test("every string the banner reaches for exists", () => {
  const gpu = en.settings.resources.gpu as Record<string, string>;
  const liveMonitor = en.settings.resources.liveMonitor as Record<string, string>;
  for (const key of ["noUsableGpu", "mismatchCpuBuild", "mismatchUnavailable", "unusableDevice"]) {
    assert.equal(typeof gpu[key], "string", `settings.resources.gpu.${key}`);
  }
  for (const key of ["gpuUnusable", "gpuUnusableDetail"]) {
    assert.equal(typeof liveMonitor[key], "string", `settings.resources.liveMonitor.${key}`);
  }
  // The version is what turns "PyTorch is CPU-only" into something a user can check
  // against their own install, so both sentences have to carry it.
  assert.match(gpu.mismatchCpuBuild, /\{version\}/);
  assert.match(gpu.mismatchUnavailable, /\{version\}/);
  // And the CPU-only host's line is still the one it always was.
  assert.equal(t(NO_GPU), NO_GPU);
  assert.match(gpu.noGpu, /No visible GPU detected/);
});
