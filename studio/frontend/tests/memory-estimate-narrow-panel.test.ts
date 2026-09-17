// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Verify the memory row's disclosure markup and wrapping captions.

import assert from "node:assert/strict";
import test from "node:test";
import { ChevronDown } from "lucide-react";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import type * as MemoryEstimateModule from "../src/features/model-picker/components/memory-estimate-row.tsx";
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
    "lucide-react": { ChevronDown },
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
    computeBytes: 0.75 * GIB,
    drafterRuntimeBytes: 0,
    drafterRuntimeGpuBytes: 0,
    projectorRuntimeBytes: 0,
    drafterKvUnsized: false,
    adaptersUnsized: false,
    totalBytes: 5.5 * GIB,
    gpuBytes: 4.75 * GIB,
    kvEstimable: true,
    kvOnGpu: true,
    nCtx: 262144,
    cacheTypeKv: "f16",
    nParallel: 4,
    layerCount: 27,
    gpuLayers: 12,
    moeOffloadUnmodelled: false,
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
  // One breakable space remains per bullet.
  assert.equal(glued.split(" ").length - 1, 2);
});

test("the bullet leads its item, so a break cannot orphan it", () => {
  for (const item of glueNoteItems("f16 · 4 slots").split(" ").slice(1)) {
    assert.ok(item.startsWith(`·${NBSP}`), `bullet detached from ${item}`);
  }
});

test("a note with no separator keeps every break opportunity it had", () => {
  // Prose captions must still wrap.
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
  // Only whitespace changes.
  assert.equal(glueNoteItems(note).replace(new RegExp(NBSP, "g"), " "), note);
});
