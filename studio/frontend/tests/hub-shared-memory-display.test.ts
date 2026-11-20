// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const hubPage = readSrc("features/hub/hub-page.tsx");
const modelsHeader = readSrc("features/hub/catalog/models-header.tsx");

test("Model Hub separates dedicated VRAM from shared GPU memory", () => {
  assert.match(hubPage, /headerGpu\.dedicatedMemoryTotalGb/);
  assert.match(hubPage, /headerGpu\.memorySharedGb/);
  assert.match(hubPage, /gpuSharedLabel=\{gpuSharedLabel\}/);
  assert.match(modelsHeader, /`\$\{gpuLabel\} VRAM \+ \$\{gpuSharedLabel\}`/);
  assert.match(modelsHeader, /label=\{gpuSharedLabel \? "shared" : "VRAM"\}/);
});

test("the Hub GPU chip falls back to the GGUF inventory when training sees no GPU", () => {
  // A GGUF-only Mac runs llama.cpp on Metal; reading only the training inventory showed "Unavailable".
  assert.match(
    hubPage,
    /const headerGpu = gpu\.available \? gpu : inferenceGpu;/,
  );
});
