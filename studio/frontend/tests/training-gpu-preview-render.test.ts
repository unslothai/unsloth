// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
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

const labels = {
  hardware: "Hardware",
  vram: "VRAM",
  automatic: "Automatic device selection at launch",
  unavailable: "No GPU available",
  noSelection: "No GPU selected",
  device: (index: number, name: string) => `GPU ${index}: ${name}`,
  total: (memoryGb: string) => `Total: ${memoryGb} GiB`,
};

function device(index: number, name: string, memoryTotalGb: number) {
  return {
    index,
    indexKind: "physical" as const,
    name,
    memoryTotalGb,
    memoryFreeGb: memoryTotalGb,
    sharedMemory: false,
    pinnable: true,
    diffusionPinnable: true,
  };
}

function render(
  mode: "auto" | "single" | "ddp" | "model_parallel",
  devices: ReturnType<typeof device>[],
  gpuAvailable = true,
  totalMemoryOverride?: number,
) {
  return renderToStaticMarkup(
    createElement(TrainingGpuPreview, {
      mode,
      devices,
      gpuAvailable,
      totalMemoryGb:
        totalMemoryOverride ??
        (mode === "model_parallel"
          ? devices.reduce((total, item) => total + item.memoryTotalGb, 0)
          : (devices[0]?.memoryTotalGb ?? 0)),
      labels,
    }),
  );
}

test("Automatic preview does not present inventory order as the selected device", () => {
  const html = render("auto", []);

  assert.match(html, /Automatic device selection at launch/);
  assert.doesNotMatch(html, /GPU \d+:/);
});

test("empty manual selection is not misreported as missing GPU hardware", () => {
  const html = render("ddp", [], true);

  assert.match(html, /No GPU selected/);
  assert.doesNotMatch(html, /Automatic device selection at launch/);
});

test("single-device preview reports the selected device and its VRAM", () => {
  const html = render("single", [device(3, "Selected adapter", 12.25)]);

  assert.match(html, /GPU 3: Selected adapter/);
  assert.match(html, /12\.3 GiB/);
  assert.match(html, /title="12\.3 GiB"/);
  assert.doesNotMatch(html, /Total:/);
});

test("in-progress single-device sharding preview does not duplicate VRAM", () => {
  const html = render("model_parallel", [device(3, "Selected adapter", 12.25)]);

  assert.match(html, /GPU 3: Selected adapter/);
  assert.match(html, />VRAM</);
  assert.doesNotMatch(html, /Total:/);
});

test("DDP preview reports selected devices separately without an aggregate", () => {
  const devices = [
    device(2, "First selected", 16),
    device(5, "Second selected", 24),
  ];
  const html = render("ddp", devices);
  const contradictoryAggregateHtml = render("ddp", devices, true, 999);

  assert.match(html, /GPU 2: First selected/);
  assert.match(html, /GPU 5: Second selected/);
  assert.match(html, /16\.0 GiB/);
  assert.match(html, /24\.0 GiB/);
  assert.doesNotMatch(html, /Total:/);
  assert.doesNotMatch(html, />VRAM</);
  assert.equal(contradictoryAggregateHtml, html);
});

test("model-sharding preview reports per-device VRAM and the selected-device total", () => {
  const html = render("model_parallel", [
    device(2, "First selected", 16),
    device(5, "Second selected", 24),
  ]);

  assert.match(html, /GPU 2: First selected/);
  assert.match(html, /GPU 5: Second selected/);
  assert.match(html, /Total: 40\.0 GiB/);
});
