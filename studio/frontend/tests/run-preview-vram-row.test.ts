// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * GPU preview rows are rendered by the real presentation component.
 */

import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import test, { after } from "node:test";
import { createServer } from "vite";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const server = await createServer({
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const { TrainingGpuPreview } = await server.ssrLoadModule(
  "/src/features/studio/wizard/training-gpu-preview.tsx",
);
after(() => server.close());

const device = {
  index: 1,
  indexKind: "physical" as const,
  name: "One GPU",
  memoryTotalGb: 32.4,
  memoryFreeGb: 24,
  sharedMemory: false,
  pinnable: true,
  diffusionPinnable: true,
};
const labels = {
  hardware: "Hardware",
  vram: "VRAM",
  automatic: "Automatic",
  unavailable: "No GPU",
  device: (index: number, name: string) => `GPU ${index}: ${name}`,
  total: (total: string) => `Total: ${total} GiB`,
};

test("single-device hardware row does not fold VRAM into the device label", () => {
  const html = renderToStaticMarkup(
    createElement(TrainingGpuPreview, {
      mode: "single",
      devices: [device],
      gpuAvailable: true,
      totalMemoryGb: device.memoryTotalGb,
      labels,
    }),
  );
  assert.match(html, /GPU 1: One GPU/);
  assert.match(html, />32\.4 GiB</);
  assert.match(html, /title="32\.4 GiB"/);
});

test("multi-device preview adds per-device VRAM and aggregates only model sharding", () => {
  const devices = [device, { ...device, index: 3, name: "Second GPU", memoryTotalGb: 16 }];
  const ddpHtml = renderToStaticMarkup(
    createElement(TrainingGpuPreview, {
      mode: "ddp",
      devices,
      gpuAvailable: true,
      totalMemoryGb: 48.4,
      labels,
    }),
  );
  const shardingHtml = renderToStaticMarkup(
    createElement(TrainingGpuPreview, {
      mode: "model_parallel",
      devices,
      gpuAvailable: true,
      totalMemoryGb: 48.4,
      labels,
    }),
  );
  assert.match(ddpHtml, /32\.4 GiB/);
  assert.match(ddpHtml, /16\.0 GiB/);
  assert.doesNotMatch(ddpHtml, /Total:/);
  assert.match(shardingHtml, /Total: 48\.4 GiB/);
});
