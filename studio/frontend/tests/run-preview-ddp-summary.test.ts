// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import test, { after } from "node:test";
import { createServer } from "vite";
import {
  selectedTrainingPreviewDevices,
  trainingPreviewGpuCount,
} from "../src/features/training/lib/training-gpu-selection.ts";
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

test("run preview selects DDP physical devices and derives global batch from them", () => {
  const allDevices = [
    { index: 0, name: "Unselected", memoryTotalGb: 8 },
    { index: 2, name: "Selected A", memoryTotalGb: 16 },
    { index: 5, name: "Selected B", memoryTotalGb: 24 },
  ].map((device) => ({
    ...device,
    indexKind: "physical" as const,
    memoryFreeGb: device.memoryTotalGb,
    sharedMemory: false,
    pinnable: true,
    diffusionPinnable: true,
  }));
  const selectedIds = [2, 5];
  const selectedDevices = selectedTrainingPreviewDevices(
    "ddp",
    selectedIds,
    allDevices,
  );
  assert.equal(trainingPreviewGpuCount("ddp", selectedIds), 2);
  assert.equal(trainingPreviewGpuCount("single", selectedIds), 1);
  const html = renderToStaticMarkup(
    createElement(TrainingGpuPreview, {
      mode: "ddp",
      devices: selectedDevices,
      gpuAvailable: true,
      totalMemoryGb: 999,
      labels: {
        hardware: "Hardware",
        vram: "VRAM",
        automatic: "Automatic",
        unavailable: "No GPU",
        device: (index: number, name: string) => `GPU ${index}: ${name}`,
        total: (total: string) => `Total: ${total} GiB`,
      },
    }),
  );

  assert.match(html, /GPU 2: Selected A/);
  assert.match(html, /GPU 5: Selected B/);
  assert.doesNotMatch(html, /Unselected/);
  assert.doesNotMatch(html, /Total:/);
  assert.doesNotMatch(html, />VRAM</);
});