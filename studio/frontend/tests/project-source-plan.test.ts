// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { planChatItemSources } from "../src/features/chat/utils/project-source-plan.ts";

test("a single chat is saved once, under its own title", () => {
  assert.deepEqual(
    planChatItemSources({ id: "t1", title: "Fix my regex", type: "single" }, []),
    [{ id: "t1", title: "Fix my regex" }],
  );
});

test("each half of a pair is named after the model that answered", () => {
  assert.deepEqual(
    planChatItemSources({ id: "p1", title: "Fix my regex", type: "pair" }, [
      { id: "t1", modelId: "unsloth/Qwen3-8B-GGUF:Q4_K_M" },
      { id: "t2", modelId: "openai/gpt-5" },
    ]),
    [
      { id: "t1", title: "Fix my regex - Qwen3-8B-GGUF" },
      { id: "t2", title: "Fix my regex - gpt-5" },
    ],
  );
});

test("an unnamed model falls back to its position, so the two never collide", () => {
  const plans = planChatItemSources(
    { id: "p1", title: "Fix my regex", type: "pair" },
    [{ id: "t1" }, { id: "t2", modelId: "  " }],
  );
  assert.deepEqual(
    plans.map((plan) => plan.title),
    ["Fix my regex - 1", "Fix my regex - 2"],
  );
  assert.equal(new Set(plans.map((plan) => plan.title)).size, 2);
});

test("a LoRA compare names the adapter halves apart, not twice the same", () => {
  // The base/lora compare toggles the adapter on one loaded checkpoint, so both
  // threads record the same modelId; the model name alone cannot tell them apart.
  const plans = planChatItemSources(
    { id: "p1", title: "Fix my regex", type: "pair" },
    [
      { id: "t1", modelId: "unsloth/Qwen3-8B", modelType: "base" },
      { id: "t2", modelId: "unsloth/Qwen3-8B", modelType: "lora" },
    ],
  );
  assert.deepEqual(plans, [
    { id: "t1", title: "Fix my regex - Qwen3-8B - base" },
    { id: "t2", title: "Fix my regex - Qwen3-8B - fine-tuned" },
  ]);
});

test("two panes on the same checkpoint fall back to their position", () => {
  // Same repo, different quant: the variant is not part of modelId, and the
  // colon suffix is stripped anyway, so both labels read "Qwen3-8B-GGUF".
  const plans = planChatItemSources(
    { id: "p1", title: "Fix my regex", type: "pair" },
    [
      { id: "t1", modelId: "unsloth/Qwen3-8B-GGUF:Q4_K_M", modelType: "model1" },
      { id: "t2", modelId: "unsloth/Qwen3-8B-GGUF:Q8_0", modelType: "model2" },
    ],
  );
  assert.deepEqual(
    plans.map((plan) => plan.title),
    ["Fix my regex - Qwen3-8B-GGUF - 1", "Fix my regex - Qwen3-8B-GGUF - 2"],
  );
  assert.equal(new Set(plans.map((plan) => plan.title)).size, 2);
});

test("a pair with one surviving half keeps the plain title", () => {
  assert.deepEqual(
    planChatItemSources({ id: "p1", title: "Fix my regex", type: "pair" }, [
      { id: "t1", modelId: "unsloth/Qwen3-8B" },
    ]),
    [{ id: "t1", title: "Fix my regex" }],
  );
  assert.deepEqual(
    planChatItemSources({ id: "p1", title: "Fix my regex", type: "pair" }, []),
    [],
  );
});
