// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loraFuseLabel } from "../src/features/images/lib/lora-fuse.ts";
import { readSrc } from "./helpers/kit.ts";

test("a video load's baked LoRA set reads as fused at load", () => {
  const baked = loraFuseLabel("baked");
  assert.equal(baked?.label, "Fused at load");
  assert.match(baked?.detail ?? "", /merged into the bf16 denoiser weights at load/);
});

test("fused rows say fused and unfused output can differ slightly", () => {
  for (const status of ["fused", "baked"]) {
    assert.match(loraFuseLabel(status)?.detail ?? "", /Fused and unfused output can differ slightly/);
  }
  // unfused rows do not carry the note
  assert.doesNotMatch(loraFuseLabel("offload")?.detail ?? "", /differ slightly/);
});

test("the video Loaded build shows the LoRA fuse row from status", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /loraFuseLabel\(status\.lora_fuse\)/);
  const start = page.search(/function LoadedBuildSummary/);
  const panel = page.slice(start, start + 4000);
  assert.match(panel, /<BuildRow\s+label="LoRA"/);
  assert.match(readSrc("features/video/api.ts"), /lora_fuse\?: string \| null;/);
});
