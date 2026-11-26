// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Users saw LoRA fusion silently off under CPU/GPU offload (NVIDIA feedback p.14, p.41): the LoRA slot now says why.

import assert from "node:assert/strict";
import test from "node:test";

import { loraFusionNote } from "../src/features/images/lib/engine-choice.ts";
import { readSrc } from "./helpers/kit.ts";

const OFFLOAD = "LoRA fusion skipped: the denoiser is offloaded, and folded weights must stay on the GPU";

test("an unfused LoRA set names the reason; a fused one, no LoRAs or an older backend say nothing", () => {
  const base = { loaded: true, supports_lora: true };
  assert.equal(loraFusionNote({ ...base, lora_fusion: { fused: false, skipped_reason: OFFLOAD } }), OFFLOAD);
  assert.equal(loraFusionNote({ ...base, lora_fusion: { fused: true, skipped_reason: null } }), null);
  assert.equal(loraFusionNote({ ...base, lora_fusion: null }), null);
  assert.equal(loraFusionNote(base), null);
  assert.equal(loraFusionNote({ loaded: false, lora_fusion: { fused: false, skipped_reason: OFFLOAD } }), null);
  // LoRA off entirely already shows lora_unavailable_reason instead.
  assert.equal(loraFusionNote({ loaded: true, supports_lora: false, lora_fusion: { fused: false, skipped_reason: OFFLOAD } }), null);
});

test("the Images LoRA field renders the note", () => {
  const src = readSrc("features/images/images-page.tsx");
  assert.match(src, /const loraUnfusedNote = external \? null : loraFusionNote\(status\);/);
  const start = src.indexOf('{loraCapable && (\n              <Field\n                label="LoRAs"');
  assert.ok(start > 0, "the LoRAs field");
  const field = src.slice(start);
  assert.match(field.slice(0, field.indexOf("{controlnetCapable &&")), /\{loraUnfusedNote && \(/);
});
