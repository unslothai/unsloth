// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Derivations live in .tsx files, so they are lifted by regex and evaluated.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { videoNavHint } = await import("../src/config/hardware-verdict.ts");
const { en } = await import("../src/i18n/locales/en.ts");

const tabSrc = await readSrcAsync("features/settings/tabs/resources-tab.tsx");
const sidebarSrc = await readSrcAsync("components/app-sidebar.tsx");

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
  // systemInfo.gpu, NOT displayedGpu: Vulkan llama.cpp makes displayedGpu fall back.
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
  assert.match(gpu.mismatchCpuBuild, /\{version\}/);
  assert.match(gpu.mismatchUnavailable, /\{version\}/);
  assert.equal(t(NO_GPU), NO_GPU);
  assert.match(gpu.noGpu, /No visible GPU detected/);
});

// start_managed_repair rejects external backends only after the shell already swapped screens.
test("the repair row hides itself for an externally started backend", async () => {
  const source = await readSrcAsync("features/settings/components/desktop-repair-control.tsx");
  assert.match(
    source,
    /if\s*\(!repair\s*\|\|\s*repair\.isExternalServer\)\s*return null;/,
    "the control must bail out on an external server as well as outside Tauri",
  );

  const context = await readSrcAsync("hooks/tauri-repair-context.ts");
  assert.match(
    context,
    /isExternalServer:\s*boolean;/,
    "the controller has to carry the flag for the control to read it",
  );

  const provider = await readSrcAsync("app/provider.tsx");
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

test("the sidebar keeps polling while the inventory can still change the verdict", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");

  assert.match(
    source,
    /INVENTORY_SENSITIVE_REASONS = new Set\(\[[^\]]*"no_gpu"[^\]]*"torch_cpu_build"[^\]]*"torch_cuda_unavailable"/s,
    "the three verdicts the inventory can move must all keep the poll alive",
  );

  const set = source.slice(
    source.indexOf("INVENTORY_SENSITIVE_REASONS = new Set(["),
  );
  const listed = set.slice(0, set.indexOf("]"));
  for (const settled of ["mlx_unavailable", "no_torch", "intel_mac"]) {
    assert.ok(
      !listed.includes(settled),
      `${settled} cannot change on a probe and must not keep polling`,
    );
  }
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

  assert.ok(polls(true, "torch_cpu_build"));
  assert.ok(polls(true, "torch_cuda_unavailable"));
  assert.ok(polls(true, "no_gpu"), "an eGPU can arrive on a CPU-only box");

  assert.ok(
    !polls(false, null),
    "a healthy GPU host must not gain a forced read a minute",
  );
  assert.ok(!polls(true, "intel_mac"), "an Intel Mac stays an Intel Mac");
  assert.ok(
    !polls(true, "no_torch"),
    "a --no-torch install declined the training stack; nothing is coming to change it",
  );
  assert.ok(
    polls(true, "detection_failed"),
    "the backend can still replace this one, so the read has to keep happening",
  );

  assert.ok(polls(true, "mlx_unavailable", false), "the MLX self-heal poll");
  assert.ok(polls(false, null, true, true), "the unknown-verdict poll");
});

test("a settled inventory poll collects the refresh it triggered", () => {
  // The backend's 60s-TTL read only schedules the refresh, so a follow-up read is needed.
  assert.match(sidebarSrc, /const INVENTORY_FOLLOW_UP_MS = (\d+);/);
  const followUp = Number(
    /const INVENTORY_FOLLOW_UP_MS = (\d+);/.exec(sidebarSrc)![1],
  );
  const interval = Number(/const INVENTORY_POLL_MS = (\d+);/.exec(sidebarSrc)![1]);
  assert.ok(
    followUp > 0 && followUp < interval,
    `the follow-up (${followUp}ms) has to land inside the interval (${interval}ms)`,
  );

  assert.match(sidebarSrc, /if \(!selfHealSettled \|\| capabilitiesUnknown\) return;/);
  assert.match(
    sidebarSrc,
    /window\.clearInterval\(id\);\s*\n\s*if \(followUp\) window\.clearTimeout\(followUp\);/,
  );
});
