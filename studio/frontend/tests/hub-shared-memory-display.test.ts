// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const hubPage = readSrc("features/hub/hub-page.tsx");
const modelsHeader = readSrc("features/hub/catalog/models-header.tsx");

test("Model Hub separates dedicated VRAM from shared GPU memory", () => {
  assert.match(hubPage, /gpu\.dedicatedMemoryTotalGb/);
  assert.match(hubPage, /gpu\.memorySharedGb/);
  assert.match(hubPage, /gpuSharedLabel=\{gpuSharedLabel\}/);
  assert.match(modelsHeader, /`\$\{gpuLabel\} VRAM \+ \$\{gpuSharedLabel\}`/);
  assert.match(modelsHeader, /label=\{gpuSharedLabel \? "shared" : "VRAM"\}/);
});
