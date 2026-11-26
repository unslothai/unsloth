// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } =
  await import("../src/features/training/api/mappers.ts");
const { initialTrainingConfigState } =
  await import("../src/features/training/stores/training-config-policy.ts");

function payloadFor(patch: Partial<TrainingConfigState>) {
  const config: TrainingConfigState = {
    ...initialTrainingConfigState,
    modelType: "text",
    selectedModel: "unsloth/Qwen3.5-0.8B-Base",
    datasetSource: "huggingface",
    dataset: "unsloth/test",
    ...patch,
  };
  return buildTrainingStartPayload(config, null);
}

const TEMPLATE = "{% for message in messages %}{{ message.content }}{% endfor %}\n";

test("a custom chat template is sent exactly as entered", () => {
  const payload = payloadFor({ chatTemplate: TEMPLATE });

  assert.equal(payload.chat_template, TEMPLATE);
});

test("an empty or whitespace template keeps the model and dataset default", () => {
  assert.equal(payloadFor({ chatTemplate: "" }).chat_template, null);
  assert.equal(payloadFor({ chatTemplate: "  \n" }).chat_template, null);
});

test("raw text, CPT, decision and embedding runs do not send a chat template", () => {
  for (const patch of [
    { datasetFormat: "raw" },
    { trainingMethod: "cpt" },
    { modelType: "decision" },
    { modelType: "embeddings" },
  ] as const) {
    assert.equal(payloadFor({ ...patch, chatTemplate: TEMPLATE }).chat_template, null);
  }
});
