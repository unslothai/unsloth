// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

// nvidia/Cosmos3-* cards name no pipeline task (library "cosmos") and tag diffusers.
const HUB_CARD = {
  pipelineTag: null,
  libraryName: "cosmos",
  tags: ["diffusers", "safetensors", "cosmos3", "cosmos3_omni", "license:other"],
};
const UNSUPPORTED = { status: "unsupported", reason: "Library: diffusers." };

test("only nvidia's Cosmos3 Edge is routed to the Video page; Nano, siblings and copies stay unsupported", () => {
  const cases: Array<[string, object]> = [
    ["nvidia/Cosmos3-Edge", { ...UNSUPPORTED, supportedIn: "video" }],
    ["NVIDIA/cosmos3-edge", { ...UNSUPPORTED, supportedIn: "video" }],
    // Nano's checkpoint enables the Cosmos guardrail, whose package Studio does not install.
    ["nvidia/Cosmos3-Nano", UNSUPPORTED],
    ["nvidia/Cosmos3-Nano-FP8", UNSUPPORTED],
    ["nvidia/Cosmos3-Edge-Policy-DROID", UNSUPPORTED],
    ["nvidia/Cosmos3-Nano-Reasoner", UNSUPPORTED],
    ["someone/Cosmos3-Edge", UNSUPPORTED],
  ];
  for (const [modelId, expected] of cases) {
    assert.deepEqual(classifyUnslothSupport({ modelId, ...HUB_CARD }), expected, modelId);
  }
});
