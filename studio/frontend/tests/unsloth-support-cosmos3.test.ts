// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";
import {
  VIDEO_CATALOG,
  catalogToModelOptions,
  curatedRowLabelFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

// nvidia/Cosmos3-* cards name no pipeline task (library "cosmos") and tag diffusers.
const HUB_CARD = {
  pipelineTag: null,
  libraryName: "cosmos",
  tags: ["diffusers", "safetensors", "cosmos3", "cosmos3_omni", "license:other"],
};
const UNSUPPORTED = { status: "unsupported", reason: "Library: diffusers." };

test("nvidia's Cosmos3 Nano and Edge are routed to the Video page; siblings and copies stay unsupported", () => {
  const cases: Array<[string, object]> = [
    ["nvidia/Cosmos3-Edge", { ...UNSUPPORTED, supportedIn: "video" }],
    ["NVIDIA/cosmos3-edge", { ...UNSUPPORTED, supportedIn: "video" }],
    ["nvidia/Cosmos3-Nano", { ...UNSUPPORTED, supportedIn: "video" }],
    ["NVIDIA/cosmos3-nano", { ...UNSUPPORTED, supportedIn: "video" }],
    ["someone/Cosmos3-Nano", UNSUPPORTED],
    ["nvidia/Cosmos3-Nano-FP8", UNSUPPORTED],
    ["nvidia/Cosmos3-Edge-Policy-DROID", UNSUPPORTED],
    ["nvidia/Cosmos3-Nano-Reasoner", UNSUPPORTED],
    ["someone/Cosmos3-Edge", UNSUPPORTED],
  ];
  for (const [modelId, expected] of cases) {
    assert.deepEqual(classifyUnslothSupport({ modelId, ...HUB_CARD }), expected, modelId);
  }
});

test("the Video picker lists Cosmos3 Nano beside Edge, and the Nano row names the guardrail it needs", () => {
  const rows = catalogToModelOptions(VIDEO_CATALOG);
  const nano = rows.find((row) => row.id === "nvidia/Cosmos3-Nano");
  assert.ok(nano, "nvidia/Cosmos3-Nano missing from the Video picker");
  assert.ok(rows.some((row) => row.id === "nvidia/Cosmos3-Edge"));
  assert.equal(nano.name, "Cosmos3 Nano");
  assert.match(nano.description ?? "", /cosmos-guardrail/);
});

test("the Nano row carries a visible cosmos-guardrail chip; Edge carries none", () => {
  // The picker draws these tags beside the name; the description is only searchable.
  assert.deepEqual(curatedRowLabelFor("nvidia/Cosmos3-Nano", VIDEO_CATALOG)?.tags, ["needs cosmos-guardrail"]);
  assert.deepEqual(curatedRowLabelFor("nvidia/Cosmos3-Edge", VIDEO_CATALOG)?.tags, []);
});
