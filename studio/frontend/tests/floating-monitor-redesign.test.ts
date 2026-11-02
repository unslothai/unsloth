// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_MONITOR_REFRESH_MS,
  MONITOR_REFRESH_OPTIONS_MS,
  monitorRefreshMs,
} from "../src/components/monitor-refresh.ts";
import { readSrc } from "./helpers/kit.ts";

const floatingMonitor = readSrc("components/floating-monitor.tsx");

test("refresh rate keeps the old 5 s default and only accepts listed intervals", () => {
  assert.equal(DEFAULT_MONITOR_REFRESH_MS, 5000);
  for (const ms of MONITOR_REFRESH_OPTIONS_MS) {
    assert.equal(monitorRefreshMs(ms), ms);
  }
  // A 0 or junk value from storage must not turn the poll into a busy loop.
  for (const junk of [0, -1, 1, 250, Number.NaN, "1000", null, undefined]) {
    assert.equal(monitorRefreshMs(junk), DEFAULT_MONITOR_REFRESH_MS);
  }
});

test("the floating monitor polls at the stored rate, not a hard-coded 5 s", () => {
  assert.match(floatingMonitor, /pollMs: refreshMs/);
  assert.doesNotMatch(floatingMonitor, /pollMs: 5000/);
});

test("the panel has no native resize grip: it sizes to its content", () => {
  assert.match(floatingMonitor, /layout \? "top-0 left-0" : "right-0 bottom-0"/);
});

test("unified memory is one Memory row, not a RAM bar and a VRAM bar of the same pool", () => {
  assert.match(floatingMonitor, /unifiedPool &&\s+memoryDisplay\.sharedDevices\.length === 0/);
  assert.match(floatingMonitor, /liveMonitor\.memory"/);
  assert.match(floatingMonitor, /liveMonitor\.gpuInUse"/);
  assert.match(floatingMonitor, /liveMonitor\.availableForModels"/);
});

test("the card drags from its background, and controls and resize edges keep their pointer", () => {
  assert.match(floatingMonitor, /event\.target\.closest\(BACKGROUND_DRAG_EXCLUDED\)/);
  assert.match(floatingMonitor, /\[role="button"\], \[data-testid="floating-monitor-resize"\]/);
  // The grip stays.
  assert.match(floatingMonitor, /data-testid="floating-monitor-drag-handle"/);
});

test("at a fast refresh the GPU rows follow it instead of the backend's 10 s GPU cache", () => {
  assert.match(floatingMonitor, /gpuFollowsPoll: true/);
  const hook = readSrc("hooks/use-system.ts");
  assert.match(hook, /params\.set\("gpu_max_age", String\(gpuMaxAgeMs \/ 1000\)\)/);
  assert.match(hook, /gpuMaxAgeMs: gpuFollowsPoll \? pollMs : undefined/);
  // Settings keeps the default cache: only the monitor opts in.
  assert.doesNotMatch(readSrc("features/settings/tabs/resources-tab.tsx"), /gpuFollowsPoll/);
});
