// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Hub detail pills reuse the sidebar's Video and Library icons, Base chips draw
// Hugeicons' Layer, and a GGUF's Memory stat reads the selected quant's size.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { ggufMemoryLabel } = await import(
  "../src/features/hub/hooks/use-hub-model-vram.ts"
);

function sidebarIcon(row: string): string {
  const sidebar = readSrc("components/app-sidebar.tsx");
  const match = sidebar.match(new RegExp(`\\n    ${row}: \\{\\n      icon: (\\w+),`));
  assert.ok(match, `sidebar ${row} row`);
  return match[1];
}

test("Video gen draws the sidebar's Video icon", () => {
  const shared = readSrc("features/hub/catalog/shared.tsx");
  const icon = sidebarIcon("video");
  assert.match(
    shared,
    new RegExp(`key === "diffusion" && label === "Video gen"\\) return ${icon};`),
  );
  assert.match(shared, /icon=\{capabilityIcon\(capability\.key, capability\.label\)\}/);
});

test("the Library stat draws the sidebar's Library icon", () => {
  const inspector = readSrc("features/hub/catalog/model-inspector.tsx");
  const icon = sidebarIcon("library");
  assert.match(inspector, new RegExp(`label="Library"[\\s\\S]{0,80}icon=\\{${icon}\\}`));
});

test("every Base chip draws Layer", () => {
  const inspector = readSrc("features/hub/catalog/model-inspector.tsx");
  assert.match(inspector, /icon=\{LayerIcon\}[\s\S]{0,160}>Base<\/span>/);
  const local = readSrc("features/hub/catalog/local-on-device-card.tsx");
  const reference = local.slice(local.indexOf("function BaseModelReference"));
  assert.match(reference.slice(0, 1500), /icon=\{LayerIcon\}/);
});

test("a GGUF's memory is the selected quant, else the earlier estimate", () => {
  assert.equal(ggufMemoryLabel(null, 45_500_000_000), "46 GB");
  assert.equal(ggufMemoryLabel("120 GB", 4_500_000_000), "4.5 GB");
  assert.equal(ggufMemoryLabel("3 GB", null), "3 GB");
  assert.equal(ggufMemoryLabel(null, 0), null);
});

test("both GGUF cards report their selected quant to the Memory stat", () => {
  const inspector = readSrc("features/hub/catalog/model-inspector.tsx");
  assert.match(inspector, /<LocalOnDeviceCard[\s\S]*?onSelectedBytesChange=\{onSelectedGgufBytesChange\}/);
  assert.match(inspector, /<DownloadSection[\s\S]*?onSelectedGgufBytesChange=\{onSelectedGgufBytesChange\}/);
  const local = readSrc("features/hub/catalog/local-on-device-card.tsx");
  assert.match(local, /onSelectedBytesChange\?\.\(modelId, selectedBytes\)/);
});
