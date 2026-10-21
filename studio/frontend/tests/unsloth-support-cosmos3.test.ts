// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";

// nvidia/Cosmos3-Edge and -Nano name no pipeline task on the Hub (library "cosmos") and tag diffusers, so the library
// check alone called the Video page's own model unsupported.
const HUB_CARD = {
  pipelineTag: null,
  libraryName: "cosmos",
  tags: [
    "diffusers",
    "safetensors",
    "cosmos3",
    "cosmos3_omni",
    "license:other",
  ],
};

test("Cosmos3 Edge runs on the Video page, not in chat", () => {
  for (const modelId of ["nvidia/Cosmos3-Edge", "NVIDIA/cosmos3-edge"]) {
    assert.deepEqual(
      classifyUnslothSupport({ modelId, ...HUB_CARD }),
      {
        status: "unsupported",
        reason: "Library: diffusers.",
        supportedIn: "video",
      },
      modelId,
    );
  }
});

test("other repos with the same card stay unsupported: Nano, siblings and copies the backend does not trust", () => {
  for (const modelId of [
    // A default install refuses it: its checkpoint enables the Cosmos guardrail, whose package Studio does not install.
    "nvidia/Cosmos3-Nano",
    "nvidia/Cosmos3-Nano-FP8",
    "nvidia/Cosmos3-Edge-Policy-DROID",
    "nvidia/Cosmos3-Nano-Reasoner",
    "someone/Cosmos3-Edge",
  ]) {
    assert.deepEqual(
      classifyUnslothSupport({ modelId, ...HUB_CARD }),
      { status: "unsupported", reason: "Library: diffusers." },
      modelId,
    );
  }
});
