// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  LORA_FUSE_CONTROL_HINT,
  LORA_FUSE_CONTROL_LABEL,
  LORA_FUSE_OPTIONS,
  loraFuseSelection,
} from "../src/features/images/lib/lora-fuse.ts";
import type { ResolvedControl } from "../src/lib/resolved-precision.ts";

test("the Merge LoRA control offers Auto, On and Off with an accuracy hint", () => {
  assert.equal(LORA_FUSE_CONTROL_LABEL, "Merge LoRA for speed");
  assert.deepEqual(LORA_FUSE_OPTIONS, [
    ["auto", "Auto"],
    ["on", "On"],
    ["off", "Off"],
  ]);
  assert.match(LORA_FUSE_CONTROL_HINT, /Faster, but slightly less exact/);
  assert.ok(!LORA_FUSE_CONTROL_HINT.includes("\u2014"));
});

test("the select reseeds from resolved.lora_fuse", () => {
  const auto: ResolvedControl = { value: "on", requested: null, source: "auto", status: "applied", reason: "env" };
  assert.equal(loraFuseSelection(auto), "auto");
  const on: ResolvedControl = { value: "on", requested: "on", source: "explicit", status: "applied", reason: "" };
  assert.equal(loraFuseSelection(on), "on");
  const off: ResolvedControl = { value: "off", requested: "off", source: "explicit", status: "applied", reason: "" };
  assert.equal(loraFuseSelection(off), "off");
  assert.equal(loraFuseSelection(undefined), null);
});

test("the page sends lora_fuse only when not auto, and badges it", () => {
  const page = readFileSync(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  assert.match(page, /lora_fuse: loraFuse === "auto" \? undefined : loraFuse,/);
  assert.match(page, /lora_fuse: advanced\.lora_fuse,/);
  assert.match(page, /controlKey="lora_fuse"/);
  const api = readFileSync(new URL("../src/features/images/api.ts", import.meta.url), "utf8");
  assert.match(api, /lora_fuse\?: "on" \| "off";/);
});

test("a reload with another lora_fuse choice changes the reseed key", async () => {
  const { resolvedSeedKey } = await import("../src/lib/resolved-precision.ts");
  const quant: ResolvedControl = { value: "off", requested: null, source: "auto", status: "applied", reason: "" };
  const at = (lora_fuse: ResolvedControl) => resolvedSeedKey({ transformer_quant: quant, lora_fuse });
  const auto: ResolvedControl = { value: "off", requested: null, source: "auto", status: "applied", reason: "default" };
  const on: ResolvedControl = { value: "on", requested: "on", source: "explicit", status: "applied", reason: "" };
  const off: ResolvedControl = { value: "off", requested: "off", source: "explicit", status: "applied", reason: "" };
  assert.notEqual(at(auto), at(on));
  assert.notEqual(at(on), at(off));
  assert.notEqual(at(auto), at(off));
  assert.equal(at(on), at({ ...on, reason: "chosen for this load" }));
});
