// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const resourcesTab = readSrc("features/settings/tabs/resources-tab.tsx");

test("Resources separates dedicated VRAM from shared GPU memory", () => {
  assert.match(resourcesTab, /gpuMemoryTotalsGb\(devices\)/);
  assert.match(resourcesTab, /metrics\.vramShared > 0/);
  assert.match(resourcesTab, /environment\.vramWithShared/);
  assert.match(resourcesTab, /vram: formatGiB\(metrics\.vramDedicated\)/);
  assert.match(resourcesTab, /shared: formatGiB\(metrics\.vramShared\)/);
  assert.match(resourcesTab, /\$\{vramCapacityLabel\}/);
});
