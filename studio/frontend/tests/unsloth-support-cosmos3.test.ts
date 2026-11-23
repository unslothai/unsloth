// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";
import {
  VIDEO_CATALOG,
  catalogToModelOptions,
  curatedRowLabelFor,
  groupForRepoId,
  loadSpecFor,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

// nvidia/Cosmos3-* cards name no pipeline task (library "cosmos") and tag diffusers.
const HUB_CARD = {
  pipelineTag: null,
  libraryName: "cosmos",
  tags: ["diffusers", "safetensors", "cosmos3", "cosmos3_omni", "license:other"],
};
const UNSUPPORTED = { status: "unsupported", reason: "Library: diffusers." };

test("nvidia's Cosmos3 Nano and Edge (and their FP8 builds) are routed to the Video page; siblings and copies stay unsupported", () => {
  const cases: Array<[string, object]> = [
    ["nvidia/Cosmos3-Edge", { ...UNSUPPORTED, supportedIn: "video" }],
    ["NVIDIA/cosmos3-edge", { ...UNSUPPORTED, supportedIn: "video" }],
    ["nvidia/Cosmos3-Nano", { ...UNSUPPORTED, supportedIn: "video" }],
    ["NVIDIA/cosmos3-nano", { ...UNSUPPORTED, supportedIn: "video" }],
    ["nvidia/Cosmos3-Nano-FP8", { ...UNSUPPORTED, supportedIn: "video" }],
    ["NVIDIA/cosmos3-nano-fp8", { ...UNSUPPORTED, supportedIn: "video" }],
    ["nvidia/Cosmos3-Edge-FP8", { ...UNSUPPORTED, supportedIn: "video" }],
    ["someone/Cosmos3-Nano", UNSUPPORTED],
    ["someone/Cosmos3-Nano-FP8", UNSUPPORTED],
    ["nvidia/Cosmos3-Super-FP8", UNSUPPORTED],
    ["nvidia/Cosmos3-Edge-Policy-DROID", UNSUPPORTED],
    ["nvidia/Cosmos3-Nano-Reasoner", UNSUPPORTED],
    ["someone/Cosmos3-Edge", UNSUPPORTED],
  ];
  for (const [modelId, expected] of cases) {
    assert.deepEqual(classifyUnslothSupport({ modelId, ...HUB_CARD }), expected, modelId);
  }
});

test("the Video picker lists Cosmos3 Nano beside Edge, with no guardrail requirement", () => {
  const rows = catalogToModelOptions(VIDEO_CATALOG);
  const nano = rows.find((row) => row.id === "nvidia/Cosmos3-Nano");
  assert.ok(nano, "nvidia/Cosmos3-Nano missing from the Video picker");
  assert.ok(rows.some((row) => row.id === "nvidia/Cosmos3-Edge"));
  // A BF16 and an FP8 build share the group, so each row names its precision.
  assert.equal(nano.name, "Cosmos3 Nano (BF16)");
  // The safety checker is off by default, so Nano needs nothing beyond Studio.
  assert.doesNotMatch(nano.description ?? "", /cosmos-guardrail/);
});

test("each Cosmos3 picker group offers its FP8 build beside the BF16 pipeline", () => {
  const rows = catalogToModelOptions(VIDEO_CATALOG);
  for (const [bf16, fp8, name] of [
    ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Nano-FP8", "Cosmos3 Nano (FP8)"],
    ["nvidia/Cosmos3-Edge", "nvidia/Cosmos3-Edge-FP8", "Cosmos3 Edge (FP8)"],
  ]) {
    const dense = rows.find((row) => row.id === bf16);
    const quant = rows.find((row) => row.id === fp8);
    assert.ok(dense && quant, `${fp8} missing from the Video picker`);
    assert.equal(quant.name, name);
    assert.equal(dense.descriptionSuffix, "BF16");
    assert.equal(quant.descriptionSuffix, "FP8");
    assert.equal(loadSpecFor(fp8, VIDEO_CATALOG)?.kind, "pipeline");
    assert.equal(groupForRepoId(fp8, VIDEO_CATALOG), groupForRepoId(bf16, VIDEO_CATALOG));
  }
});

test("no Cosmos3 row carries a needs chip, only its stored precision", () => {
  // The guardrail is opt-in, not required.
  assert.deepEqual(curatedRowLabelFor("nvidia/Cosmos3-Nano", VIDEO_CATALOG)?.tags, ["BF16"]);
  assert.deepEqual(curatedRowLabelFor("nvidia/Cosmos3-Edge", VIDEO_CATALOG)?.tags, ["BF16"]);
  assert.deepEqual(curatedRowLabelFor("nvidia/Cosmos3-Nano-FP8", VIDEO_CATALOG)?.tags, ["FP8"]);
  assert.deepEqual(curatedRowLabelFor("nvidia/Cosmos3-Edge-FP8", VIDEO_CATALOG)?.tags, ["FP8"]);
});

test("a video-to-video pipeline tag is not routed to the Video page (Nano's continuation is gated on status)", () => {
  const result = classifyUnslothSupport({
    modelId: "someone/any-video-to-video",
    pipelineTag: "video-to-video",
    libraryName: "diffusers",
    tags: ["diffusers"],
  }) as { supportedIn?: string };
  assert.notEqual(result.supportedIn, "video");
});
