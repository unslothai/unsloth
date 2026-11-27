// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { loraFuseLabel } from "../src/features/images/lib/lora-fuse.ts";

test("loraFuseLabel names each backend fuse status", () => {
  assert.equal(loraFuseLabel("fused")?.label, "Fused");
  assert.equal(loraFuseLabel("offload")?.label, "Not fused (offload)");
  assert.equal(loraFuseLabel("bit_exact")?.label, "Not fused (bit-exact)");
  assert.equal(loraFuseLabel("quantized")?.label, "Not fused (quantized)");
  assert.equal(loraFuseLabel("disabled")?.label, "Not fused (fusion off)");
});

test("loraFuseLabel hides the row without LoRAs or for an unknown status", () => {
  assert.equal(loraFuseLabel(null), null);
  assert.equal(loraFuseLabel(undefined), null);
  assert.equal(loraFuseLabel("something-new"), null);
});

test("Loaded build shows the LoRA fuse row", () => {
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  assert.match(page, /loraFuseLabel\(status\.lora_fuse\)/);
  const api = readFileSync(new URL("../src/features/images/api.ts", import.meta.url), "utf8");
  assert.match(api, /lora_fuse\?: string \| null;/);
});
