// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const resourcesTab = readSrc("features/settings/tabs/resources-tab.tsx");

test("Resources separates dedicated VRAM from shared GPU memory", () => {
  assert.match(resourcesTab, /gpuMemoryTotalsGb\(usageDevices\)/);
  assert.match(resourcesTab, /metrics\.vramShared > 0/);
  assert.match(resourcesTab, /environment\.vramWithShared/);
  assert.match(resourcesTab, /vram: formatGiB\(metrics\.vramDedicated\)/);
  assert.match(resourcesTab, /shared: formatGiB\(metrics\.vramShared\)/);
  assert.match(resourcesTab, /\$\{vramCapacityLabel\}/);
});

const floatingMonitor = readSrc("components/floating-monitor.tsx");

test("a pool with no dedicated VRAM shows the shared total alone, with its headroom", () => {
  // "7.03 GiB / 0.00 GiB VRAM + 64.0 GiB shared" on an M3 Max read as a broken probe.
  assert.match(
    resourcesTab,
    /metrics\.vramShared > 0 && metrics\.vramDedicated === 0/,
  );
  assert.match(
    resourcesTab,
    /formatGiB\(unifiedPool \? metrics\.vramShared : metrics\.vramTotal\)/,
  );
  assert.match(resourcesTab, /memoryDisplay\.unifiedAvailableGb !== null/);
  assert.match(
    floatingMonitor,
    /hasSharedPool && memoryTotals\.dedicated === 0/,
  );
  assert.match(floatingMonitor, /hasSharedPool && !unifiedPool/);
  assert.match(
    floatingMonitor,
    /formatGiB\(memoryDisplay\.unifiedAvailableGb\)/,
  );
});

test("the Live Monitor shows -- for RAM until the host has been read", () => {
  assert.match(
    floatingMonitor,
    /const ramKnown = systemInfo\.status === "ready"/,
  );
  assert.match(
    floatingMonitor,
    /ramKnown \? `\$\{Math\.round\(ramPercent\)\}%` : "--"/,
  );
});
