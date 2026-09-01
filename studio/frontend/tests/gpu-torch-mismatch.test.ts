// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the UI says about a host whose GPUs PyTorch cannot use.
//
// A Windows in-app update that resolved torch from PyPI left a 2.11.0+cpu wheel beside two
// working RTX A4000s (#8473). The backend now reports those cards in gpu.physical_devices
// with a gpu.mismatch reason, and three places have to stop saying the opposite: the System
// tab's VRAM tile and GPU section, the sidebar's Train hint, and videoNavHint.
//
// The derivations live in .tsx files that pull in the whole app, so they are lifted by
// regex and evaluated, as system-status-verdict.test.ts does beside them.

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

function lift(
  src: string,
  pattern: RegExp,
  what: string,
  where: string,
): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in ${where}`);
  return found[0];
}

const t = (key: string) => key;

const CPU_BUILD = "settings.resources.gpu.mismatchCpuBuild";
const UNAVAILABLE = "settings.resources.gpu.mismatchUnavailable";
const NO_USABLE_GPU = "settings.resources.gpu.noUsableGpu";
const NO_GPU = "settings.resources.gpu.noGpu";
const UNKNOWN = "settings.resources.environment.unknown";

// gpuInventory carries a TS cast, so it is asserted on as source below and supplied here
// as an input instead.
const derivation = [
  lift(
    tabSrc,
    /const gpuMismatch = [\s\S]*?;/,
    "gpuMismatch",
    "resources-tab.tsx",
  ),
  lift(
    tabSrc,
    /const physicalDevices = [\s\S]*?;/,
    "physicalDevices",
    "resources-tab.tsx",
  ),
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
    physical_devices: [
      { name: "NVIDIA RTX A4000" },
      { name: "NVIDIA RTX A4000" },
    ],
  });
  assert.equal(cpuBuild.gpuMismatchMessage, CPU_BUILD);
  assert.equal(cpuBuild.physicalDevices.length, 2);

  // Reinstalling torch is the wrong advice for a healthy wheel whose runtime will not
  // start, so the two reasons must not collapse into one string.
  const dead = mismatchFor({
    mismatch: {
      reason: "torch_cuda_unavailable",
      torch_version: "2.6.0+cu124",
    },
    physical_devices: [{ name: "NVIDIA RTX A4000" }],
  });
  assert.equal(dead.gpuMismatchMessage, UNAVAILABLE);
});

test("a healthy host, and one that really has no GPU, get no banner at all", () => {
  for (const inventory of [
    null,
    {},
    { mismatch: null },
  ] as (Inventory | null)[]) {
    const out = mismatchFor(inventory);
    assert.equal(out.gpuMismatch, null);
    assert.equal(out.gpuMismatchMessage, null);
    assert.deepEqual(out.physicalDevices, []);
  }
  const strayRows = mismatchFor({
    physical_devices: [{ name: "NVIDIA RTX A4000" }],
  });
  assert.deepEqual(strayRows.physicalDevices, []);
});

test("the verdict is taken from a settled read only, and from the training view", () => {
  // The placeholder useSystemInfo starts from is shaped like a CPU-only host, so a banner
  // derived from it would accuse a host nobody has measured yet.
  const inventory = lift(
    tabSrc,
    /const gpuInventory = [\s\S]*?;\n/,
    "gpuInventory",
    "resources-tab.tsx",
  );
  assert.match(
    inventory,
    /hostUnread\s*\n?\s*\?\s*null/,
    "gated on the read having settled",
  );
  // systemInfo.gpu, NOT displayedGpu: a Vulkan llama.cpp makes displayedGpu fall back to the
  // inference inventory, and that host is exactly the second report in #8473.
  assert.match(inventory, /systemInfo\.gpu/);
  assert.doesNotMatch(inventory, /displayedGpu/);
});

test("the GPU section stops telling this host there is no GPU", () => {
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
  const mismatchAt = vram.indexOf("liveMonitor.gpuUnusable");
  const noGpuAt = vram.indexOf("liveMonitor.noGpu");
  assert.ok(mismatchAt > -1, "the tile has a mismatch state");
  assert.ok(noGpuAt > -1, "and still has the CPU-only state");
  assert.ok(mismatchAt < noGpuAt, "the mismatch state is reached first");
});

test("the physically detected cards are shown, and never offered as devices", () => {
  // The banner renders physicalDevices and the selectable rows metrics.devices. If the
  // banner ever read metrics.devices the two would merge, which the field split prevents.
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
    assert.equal(videoNavHint(false, reason), undefined);
  }
  // And the genuine no-GPU host keeps the sentence that is true for it.
  assert.equal(
    videoNavHint(true, "no_gpu"),
    "Video generation needs an NVIDIA or AMD GPU.",
  );
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
    // The installed build is what makes this actionable to someone whose update already ran.
    assert.match(withDetail, /2\.11\.0\+cpu/);
    const withoutDetail = forReason(reason, null);
    assert.ok(withoutDetail);
    assert.doesNotMatch(withoutDetail, /needs an NVIDIA or AMD GPU/);
  }
  assert.equal(
    forReason("no_gpu", null),
    "Training needs an NVIDIA or AMD GPU.",
  );
  assert.equal(forReason("detection_failed", null), undefined);
});


test("every string the banner reaches for exists", () => {
  const gpu = en.settings.resources.gpu as Record<string, string>;
  const liveMonitor = en.settings.resources.liveMonitor as Record<
    string,
    string
  >;
  for (const key of [
    "noUsableGpu",
    "mismatchCpuBuild",
    "mismatchUnavailable",
    "unusableDevice",
  ]) {
    assert.equal(typeof gpu[key], "string", `settings.resources.gpu.${key}`);
  }
  for (const key of ["gpuUnusable", "gpuUnusableDetail"]) {
    assert.equal(
      typeof liveMonitor[key],
      "string",
      `settings.resources.liveMonitor.${key}`,
    );
  }
  // The version is what a user can check against their own install, so both sentences
  // have to carry it.
  assert.match(gpu.mismatchCpuBuild, /\{version\}/);
  assert.match(gpu.mismatchUnavailable, /\{version\}/);
  // And the CPU-only host's line is still the one it always was.
  assert.equal(t(NO_GPU), NO_GPU);
  assert.match(gpu.noGpu, /No visible GPU detected/);
});

// The repair row must not be offered for a backend the desktop does not manage.
//
// start_managed_repair rejects that mutation, but only after startRepair has cleared
// isExternalServer, stopped the external-server poll and swapped the shell to the repairing
// screen, so a connected user lands on the repair-error screen instead of on their server.
test("the repair row hides itself for an externally started backend", async () => {
  const source = await readFile(
    new URL(
      "../src/features/settings/components/desktop-repair-control.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /if\s*\(!repair\s*\|\|\s*repair\.isExternalServer\)\s*return null;/,
    "the control must bail out on an external server as well as outside Tauri",
  );

  const context = await readFile(
    new URL("../src/hooks/tauri-repair-context.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    context,
    /isExternalServer:\s*boolean;/,
    "the controller has to carry the flag for the control to read it",
  );

  const provider = await readFile(
    new URL("../src/app/provider.tsx", import.meta.url),
    "utf8",
  );
  const memo = provider.slice(provider.indexOf("const repairController"));
  assert.match(
    memo.slice(0, 300),
    /isExternalServer,/,
    "the provider has to publish the flag",
  );
  assert.match(
    memo.slice(0, 300),
    /\[isExternalServer\]/,
    "and list it as a dependency, or the context freezes on the first render's value",
  );
});

// The verdict can change without a restart, so the sidebar has to keep asking.
//
// The backend refreshes its physical inventory on a 60s TTL: attach an eGPU to a CPU-torch
// machine and no_gpu becomes torch_cpu_build. The polling effect stopped at the first
// settled verdict, so the new hint was unreachable for the rest of the session.
test("the sidebar keeps polling while the inventory can still change the verdict", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );

  assert.match(
    source,
    /INVENTORY_SENSITIVE_REASONS = new Set\(\[[^\]]*"no_gpu"[^\]]*"torch_cpu_build"[^\]]*"torch_cuda_unavailable"/s,
    "the three verdicts the inventory can move must all keep the poll alive",
  );

  const set = source.slice(
    source.indexOf("INVENTORY_SENSITIVE_REASONS = new Set(["),
  );
  const listed = set.slice(0, set.indexOf("]"));
  for (const settled of ["mlx_unavailable", "intel_mac"]) {
    assert.ok(
      !listed.includes(settled),
      `${settled} cannot change on a probe and must not keep polling`,
    );
  }
  // detection_failed IS listed. current_chat_only_verdict() can replace it once the
  // inventory recovers, for the host whose torch will not import but whose wheel was
  // classified from disk, so treating it as settled froze the sidebar on the failure.
  assert.ok(listed.includes("detection_failed"));

  assert.match(
    source,
    /if \(selfHealSettled && !capabilitiesUnknown && !inventorySensitive\) return;/,
    "the early return has to consider the inventory-sensitive case",
  );
  assert.match(source, /const INVENTORY_POLL_MS = 60000;/);
  assert.match(
    source,
    /selfHealSettled\s*\?\s*INVENTORY_POLL_MS\s*:\s*SELF_HEAL_POLL_MS/,
    "a settled host polls at the inventory cadence, not the self-heal one",
  );
});

// The poll decision itself, evaluated rather than pattern-matched.
//
// The test above pins the shape of the early return; this one runs it. A regression that
// keeps the guard's text but inverts its sense would give every working install a forced
// /api/system read a minute for the life of the session, which is the opposite of what
// this change is for.
test("only a host the inventory can still reclassify keeps polling", () => {
  const guard = lift(
    sidebarSrc,
    /const inventorySensitive =[\s\S]*?if \(selfHealSettled && !capabilitiesUnknown && !inventorySensitive\) return;/,
    "the polling guard",
    "app-sidebar.tsx",
  );
  const reasons = lift(
    sidebarSrc,
    /const INVENTORY_SENSITIVE_REASONS = new Set\(\[[\s\S]*?\]\);/,
    "INVENTORY_SENSITIVE_REASONS",
    "app-sidebar.tsx",
  );
  const polls = (
    chatOnly: boolean,
    chatOnlyReason: string | null,
    selfHealSettled = true,
    capabilitiesUnknown = false,
  ) =>
    new Function(
      "chatOnly",
      "chatOnlyReason",
      "selfHealSettled",
      "capabilitiesUnknown",
      `${reasons}
       ${guard}
       return true;`,
    )(chatOnly, chatOnlyReason, selfHealSettled, capabilitiesUnknown) === true;

  // The hosts this change exists for. Their verdict moves on the next inventory refresh.
  assert.ok(polls(true, "torch_cpu_build"));
  assert.ok(polls(true, "torch_cuda_unavailable"));
  assert.ok(polls(true, "no_gpu"), "an eGPU can arrive on a CPU-only box");

  // And the hosts that were working before this PR and must keep working the same way.
  assert.ok(
    !polls(false, null),
    "a healthy GPU host must not gain a forced read a minute",
  );
  assert.ok(!polls(true, "intel_mac"), "an Intel Mac stays an Intel Mac");
  // detection_failed is NOT settled when torch is the thing that failed: the backend
  // classifies the wheel from disk and swaps in the mismatch once the inventory recovers,
  // so stopping the poll froze the sidebar on the failure for the session.
  assert.ok(
    polls(true, "detection_failed"),
    "the backend can still replace this one, so the read has to keep happening",
  );

  // The two pre-existing polls are untouched.
  assert.ok(polls(true, "mlx_unavailable", false), "the MLX self-heal poll");
  assert.ok(polls(false, null, true, true), "the unknown-verdict poll");
});

test("a settled inventory poll collects the refresh it triggered", () => {
  // The backend's caches carry their own 60 second TTL and the health path reads them
  // non-blocking, so the read that finds them expired only SCHEDULES the refresh and
  // returns the stale entry. Polling on the TTL alone puts the read that collects the
  // new answer a whole interval later, leaving an attached eGPU invisible for close to
  // two minutes.
  assert.match(sidebarSrc, /const INVENTORY_FOLLOW_UP_MS = (\d+);/);
  const followUp = Number(
    /const INVENTORY_FOLLOW_UP_MS = (\d+);/.exec(sidebarSrc)![1],
  );
  const interval = Number(/const INVENTORY_POLL_MS = (\d+);/.exec(sidebarSrc)![1]);
  assert.ok(
    followUp > 0 && followUp < interval,
    `the follow-up (${followUp}ms) has to land inside the interval (${interval}ms)`,
  );

  // Scoped to the settled inventory case: the unknown poll is already fast, and the
  // self-heal poll is not waiting on a TTL.
  assert.match(sidebarSrc, /if \(!selfHealSettled \|\| capabilitiesUnknown\) return;/);
  // And torn down with the interval, or it would outlive the verdict it was scheduled for.
  assert.match(
    sidebarSrc,
    /window\.clearInterval\(id\);\s*\n\s*if \(followUp\) window\.clearTimeout\(followUp\);/,
  );
});
