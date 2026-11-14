// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { HugeiconsIcon } from "@hugeicons/react";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import type * as MemoryEstimateModule from "../src/features/model-picker/components/memory-estimate-row.tsx";
import * as chevronIcons from "../src/lib/chevron-icons.ts";
import * as memoryFit from "../src/features/model-picker/model-config/memory-fit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { glueNoteItems, resolveDraftCacheNote, resolveKvNote } = memoryFit;
const { MemoryEstimateRow } = loadWithStubs<typeof MemoryEstimateModule>(
  new URL(
    "../src/features/model-picker/components/memory-estimate-row.tsx",
    import.meta.url,
  ),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@hugeicons/react": { HugeiconsIcon },
    "@/lib/chevron-icons": chevronIcons,
    "../model-config/memory-fit": memoryFit,
    "@/components/ui/tooltip": {
      Tooltip: ({ children }: { children: React.ReactNode }) => children,
      TooltipTrigger: ({ children }: { children: React.ReactNode }) => children,
      TooltipContent: () => null,
    },
  },
);

type Props = React.ComponentProps<typeof MemoryEstimateRow>;
const GIB = 1024 ** 3;
const props: Props = {
  estimate: {
    available: true,
    reason: null,
    weightsBytes: 3.25 * GIB,
    kvBytes: 1.5 * GIB,
    kvCheckpointBytes: 0,
    computeBytes: 0.75 * GIB,
    drafterRuntimeBytes: 0,
    drafterRuntimeGpuBytes: 0,
    projectorRuntimeBytes: 0,
    drafterKvUnsized: false,
    adaptersUnsized: false,
    totalBytes: 5.5 * GIB,
    contextFitted: null,
    contextIsPinned: true,
    gpuFloorBytes: null,
    floorCanOffload: false,
    gpuBytes: 4.75 * GIB,
    perGpu: [],
    perGpuBasis: null,
    kvEstimable: true,
    kvOnGpu: true,
    nCtx: 262144,
    cacheTypeKv: "f16",
    nParallel: 4,
    layerCount: 27,
    gpuLayers: 12,
    moeOffloadUnmodelled: false,
    mlxPleOffload: null,
    mlxExpertOffload: null,
    mlxOffloadable: [],
  },
  loading: false,
  stale: false,
  gpuCapacityGb: 24,
  totalCapacityGb: 88,
  systemRamCapacityGb: 64,
  freeGpuCapacityGb: 24,
  usableSystemRamGb: 60,
  isUnifiedMemory: false,
  singleMemoryPool: false,
  expanded: false,
  onExpandedChange: () => {},
};

function render(overrides: Partial<Props> = {}): string {
  return renderToStaticMarkup(
    React.createElement(MemoryEstimateRow, { ...props, ...overrides }),
  );
}

const NBSP = " ";


test("per-GPU subtotals follow load order and warn when one card overflows", () => {
  const devices = [0, 1].map((index) => ({
    index, name: index === 0 ? "Small" : "Large", indexKind: "physical" as const,
    memoryFreeGb: index === 0 ? 0 : 18, memoryFreeKnown: true,
    memoryTotalGb: index === 0 ? 16 : 24, sharedMemory: false,
    diffusionPinnable: true, pinnable: true,
  }));
  const estimate = {
    ...props.estimate!,
    perGpu: [
      { gpuId: 1, layerCount: 5, outputLayer: false, weightsBytes: 8 * GIB, cacheBytes: null },
      { gpuId: 0, layerCount: 1, outputLayer: true, weightsBytes: 2 * GIB, cacheBytes: GIB },
    ],
  };
  const placement = (overrides: Partial<Props> = {}) => render({
    estimate, gpuDevices: devices, expanded: true, ...overrides,
  }).replaceAll(NBSP, " ");
  const html = placement();
  assert.match(html, /6 repeating layers and the output layer on GPU/);
  assert.doesNotMatch(html, /12 of 28 layers on GPU/);
  const [rows, details] = html.split("<details");
  assert.match(rows, /Weights \+ cache per GPU[\s\S]*GPU 1 · 24 GiB[\s\S]*≥ 8\.00 GiB[\s\S]*GPU 0 · 16 GiB[\s\S]*text-red-500[\s\S]*3\.00 GiB/);
  assert.doesNotMatch(rows, /cache unknown|0\.00 GiB free/);
  assert.match(details, /^[^>]*>(?:<summary[^>]*>)Layer and memory details<\/summary>/);
  assert.doesNotMatch(details.split(">")[0], /\bopen\b/);
  assert.match(details, /GPU 1 · Large[\s\S]*5 repeating layers · 8\.00 GiB weights · cache unknown/);
  assert.match(details, /GPU 0 · Small[\s\S]*1 repeating layer · output layer · 2\.00 GiB weights · 1\.00 GiB cache/);
  assert.match(html, /GPU 0 needs more than its currently free VRAM/);
  const reload = placement({ residentGpuIds: [0], reclaimableGpuBytes: 3 * GIB });
  assert.doesNotMatch(reload, /GPU 0 needs more than its currently free VRAM|text-red-500/);
  assert.match(reload, /0\.00 GiB free/);
  assert.doesNotMatch(placement({ residentGpuIds: null }), /needs more than its currently free VRAM/);
  assert.match(placement({ residentGpuIds: [1] }), /GPU 0 needs more than its currently free VRAM/);
  assert.match(placement({ residentGpuIds: null, gpuDevices: devices.map((gpu) => ({ ...gpu, memoryTotalGb: 1 })) }), /GPU 1 needs more than its total VRAM/);
  assert.match(html, /Compute buffers and other runtime allocations need additional memory/);
  assert.doesNotMatch(html, /can change when a model unloads/);
  assert.match(placement({ estimate: { ...estimate, perGpuBasis: "free_memory" } }), /can change when a model unloads/);
  assert.doesNotMatch(placement({ stale: true }), /Weights \+ cache per GPU|GPU 0 · Small|needs more than/);
  assert.doesNotMatch(placement({ gpuDevices: [] }), /needs more than|0.00 GiB free/);
  const aggregatePressure = placement({ gpuDevices: [], freeGpuCapacityGb: 2 });
  assert.match(aggregatePressure, /Little VRAM is free across the selected GPUs/);
  assert.doesNotMatch(aggregatePressure, /Fits this GPU/);
  assert.doesNotMatch(placement({ gpuDevices: devices.map((gpu) => ({ ...gpu, unifiedMemory: true })) }), /needs more than|GPU 0 · 16 GiB/);
  assert.match(placement({ gpuDevices: devices.map((gpu) => ({ ...gpu, memoryTotalGb: 1 })) }), /GPU 1 needs more than its total VRAM/);
});

test("automatic multi-GPU placement explains when the split is decided", () => {
  assert.match(render({ expanded: true, placementMode: "auto" }), /Per-GPU placement is determined at load time/);
  assert.match(render({ expanded: true, placementMode: "manual" }), /Per-GPU placement is unavailable/);
  assert.doesNotMatch(render({ expanded: true }), /Per-GPU placement/);
  assert.match(render({ estimate: { ...props.estimate!, gpuLayers: 999 } }), /28 of 28 layers on GPU/);
  assert.doesNotMatch(render({ estimate: { ...props.estimate!, gpuLayers: -1 } }), /of 28 layers on GPU/);
});

test("the title and figures occupy separate rows", () => {
  const html = render();
  const button = html.match(/<button\b[^>]*>([\s\S]*?)<\/button>/)?.[1];
  assert.ok(button);
  assert.match(button, /Estimated Memory Usage/);
  assert.doesNotMatch(button, /GiB/);
  assert.match(html, />GPU<\/span>/);
  assert.match(html, />Total<\/span>/);
  assert.match(html, /4\.75 GiB/);
  assert.match(html, /5\.50 GiB/);
});

test("the disclosure controls an existing breakdown in both states", () => {
  for (const expanded of [false, true]) {
    const html = render({ expanded });
    assert.ok(html.includes(`aria-expanded="${expanded}"`));
    const contentId = html.match(/aria-controls="([^"]+)"/)?.[1];
    assert.ok(contentId);
    assert.ok(html.includes(`id="${contentId}"`));
    assert.equal(html.includes(`id="${contentId}" hidden=""`), !expanded);
    assert.match(html, /Weights/);
    assert.match(html, /KV cache/);
  }
});

test("context checkpoints get their own host RAM row instead of inflating the KV cache", () => {
  // Qwen3.8-27B: 4 slots x 10 checkpoints x 149.625 MiB of recurrent state.
  const checkpoints = 40 * 149.625 * 1024 ** 2;
  const html = render({
    expanded: true,
    estimate: {
      ...props.estimate!,
      kvBytes: 16.584 * GIB + checkpoints,
      kvCheckpointBytes: checkpoints,
    },
  });
  assert.match(html, /KV cache[\s\S]*?16\.58 GiB/);
  assert.doesNotMatch(html, /22\.43 GiB/);
  assert.match(
    html,
    /Context checkpoints[\s\S]*?5\.84 GiB[\s\S]*?host RAM, filled as conversations grow/,
  );
  assert.doesNotMatch(render({ expanded: true }), /Context checkpoints/);
});

test("MLX offload is noted on the Weights line", () => {
  const html = render({ expanded: true, estimate: { ...props.estimate!, mlxExpertOffload: 0.5 } });
  assert.match(html, /Weights[\s\S]*?experts\u00a050%\u00a0on\u00a0disk/);
  assert.doesNotMatch(render({ expanded: true }), /on\u00a0disk/);
});

test("a shared pool shows the total without a duplicate GPU figure", () => {
  const html = render({ singleMemoryPool: true, isUnifiedMemory: true });
  assert.match(html, />Unified<\/span>/);
  assert.match(html, /5\.50 GiB/);
  assert.doesNotMatch(html, />GPU<\/span>|>Total<\/span>|4\.75 GiB/);
});

test("an unavailable estimate stays hidden", () => {
  assert.equal(render({ estimate: null }), "");
  assert.equal(
    render({ estimate: { ...props.estimate!, available: false } }),
    "",
  );
});

test("a RAM-only load shows one figure with CPU-appropriate guidance", () => {
  const html = render({
    estimate: { ...props.estimate!, gpuBytes: 0, gpuLayers: 0, kvOnGpu: false },
    usableSystemRamGb: 2,
  });
  assert.match(html, />RAM<\/span>/);
  assert.match(html, /aria-label="RAM: 5\.50 GiB"/);
  assert.match(html, /Fits system RAM, but little is free right now/);
  assert.doesNotMatch(html, />GPU<\/span>|>Total<\/span>|fewer CPU layers/);
});

test("zero free VRAM keeps the GPU figure and its warning", () => {
  const html = render({ freeGpuCapacityGb: 0, freeGpuCapacityKnown: true });
  assert.match(html, />GPU<\/span>/);
  assert.match(html, />Total<\/span>/);
  assert.match(html, /little VRAM is free right now/);
});

test("memory figures are keyboard targets with the full value as their name", () => {
  const html = render();
  assert.match(
    html,
    /<button[^>]*type="button"[^>]*aria-label="GPU: 4\.75 GiB"/,
  );
  assert.match(
    html,
    /<button[^>]*type="button"[^>]*aria-label="Total: 5\.50 GiB"/,
  );
});

test("breakdown captions preserve word groups", () => {
  assert.ok(
    render({ expanded: true }).includes(
      glueNoteItems("f16 · 262,144 tokens · 4 slots"),
    ),
  );
});

test("an item's own spaces do not break", () => {
  const glued = glueNoteItems("f16 · 262,144 tokens · 4 slots");
  assert.equal(glued, `f16 ·${NBSP}262,144${NBSP}tokens ·${NBSP}4${NBSP}slots`);
  assert.equal(glued.split(" ").length - 1, 2);
});

test("the bullet leads its item, so a break cannot orphan it", () => {
  for (const item of glueNoteItems("f16 · 4 slots").split(" ").slice(1)) {
    assert.ok(item.startsWith(`·${NBSP}`), `bullet detached from ${item}`);
  }
});

test("a note with no separator keeps every break opportunity it had", () => {
  for (const note of [
    "256 of 257 layers on GPU",
    "2.14 GB on GPU",
    "host RAM",
    "f16",
  ]) {
    assert.equal(glueNoteItems(note), note);
    assert.doesNotMatch(glueNoteItems(note), new RegExp(NBSP));
  }
});

test("the notes the row actually builds are left breakable", () => {
  assert.match(render({ expanded: true }), /12 of 28 layers on GPU/);
  const hostNote = resolveDraftCacheNote(0, 1e9);
  assert.equal(hostNote, "host RAM");
  for (const note of ["256 of 257 layers on GPU", hostNote ?? ""]) {
    const spaces = (glueNoteItems(note).match(/ /g) || []).length;
    assert.ok(spaces > 0, `${note} has no break opportunity left`);
  }
});

test("gluing round-trips the note the row actually builds", () => {
  const note = resolveKvNote({
    cacheTypeKv: "q8_0",
    nCtx: 262144,
    nParallel: 4,
    kvOnGpu: false,
  });
  assert.equal(glueNoteItems(note).replace(new RegExp(NBSP, "g"), " "), note);
});
