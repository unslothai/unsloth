// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { resolveDecisionQuestionTypes } = await import(
  "../src/features/chat/lib/decision-capabilities.ts"
);
const { decisionNativeText, formatDecisionResult } = await import(
  "../src/features/chat/lib/decision-result.ts"
);
const {
  decisionQuestionReady,
  decisionQuestionToWire,
  defaultDraft,
  newDecisionQuestion,
  useDecisionConfigStore,
} = await import("../src/features/chat/stores/decision-config-store.ts");
const { loadedContextFields } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { scoreLevels } = await import(
  "../src/features/settings/lib/decision-request.ts"
);

test("resolveDecisionQuestionTypes respects decision flag, backend list, and fallback", () => {
  assert.deepEqual(
    resolveDecisionQuestionTypes({
      isDecision: false,
      decisionType: "lfm3-d1",
    }),
    [],
  );
  assert.deepEqual(
    resolveDecisionQuestionTypes({
      isDecision: true,
      decisionType: "lfm3-d1",
    }),
    ["noul", "choice", "score"],
  );
  assert.deepEqual(
    resolveDecisionQuestionTypes({
      isDecision: true,
      decisionType: "future-arch",
    }),
    ["noul", "choice", "score"],
  );
  assert.deepEqual(
    resolveDecisionQuestionTypes({
      isDecision: true,
      decisionType: "lfm3-d1",
      declared: ["score", "noul"],
    }),
    ["noul", "score"],
  );
});

test("decisionQuestionReady validates required fields per question type", () => {
  const noul = newDecisionQuestion("noul");
  assert.equal(decisionQuestionReady(noul), true);
  assert.equal(
    decisionQuestionReady({
      ...noul,
      draft: { ...noul.draft, instructions: "   " },
    }),
    false,
  );

  const choice = newDecisionQuestion("choice");
  assert.equal(decisionQuestionReady(choice), false);
  assert.equal(
    decisionQuestionReady({
      ...choice,
      draft: {
        ...choice.draft,
        options: [
          { name: "Alpha", description: "" },
          { name: "Beta", description: "" },
        ],
      },
    }),
    true,
  );

  const score = newDecisionQuestion("score");
  assert.equal(decisionQuestionReady(score), true);
  assert.equal(
    decisionQuestionReady({
      ...score,
      draft: { ...score.draft, levels: ["  ", ""] },
    }),
    false,
  );
});

test("decisionQuestionToWire serializes choice options under 1-based keys", () => {
  const q = newDecisionQuestion("choice");
  const wire = decisionQuestionToWire({
    ...q,
    draft: {
      ...q.draft,
      instructions: " Pick one ",
      options: [
        { name: "  First ", description: "" },
        { name: "", description: "" },
        { name: " Third ", description: "" },
      ],
    },
  });
  assert.equal(wire.name, "answer");
  assert.deepEqual(wire.body, {
    type: "choice",
    instructions: "Pick one",
    criteria: {
      "1": "First",
      "3": "Third",
    },
  });
});

test("useDecisionConfigStore preserves custom instructions across type switches", () => {
  const store = useDecisionConfigStore.getState();
  store.reset();

  // Untouched default instructions are replaced on type switch.
  store.update({ type: "score" });
  assert.equal(
    useDecisionConfigStore.getState().question.draft.instructions,
    defaultDraft("score").instructions,
  );

  // User-edited instructions survive type switch.
  store.update({ draft: { instructions: "Rate code clarity" } });
  store.update({ type: "noul" });
  assert.equal(
    useDecisionConfigStore.getState().question.draft.instructions,
    "Rate code clarity",
  );
  store.reset();
});

test("decisionNativeText and scoreLevels handle undefined and missing legend safely", () => {
  assert.equal(decisionNativeText(undefined), "");
  assert.equal(decisionNativeText("raw"), "raw");
  assert.deepEqual(
    scoreLevels({
      type: "score",
      score: 1.5,
      confidence: 0.7,
      legend: undefined as unknown as Record<string, string>,
      probabilities: { "0": 0.2, "1": 0.8 },
    }),
    { keys: [], top: "0", max: 0 },
  );

  const formatted = formatDecisionResult({
    model: "AtomicChat/d1-2B-GGUF",
    latency_ms: 42.3,
    usage: { input_tokens: 18, output_tokens: 0 },
    answers: {
      answer: { type: "noul", noul: 0.9 },
    },
  });
  assert.match(formatted, /\*\*Yes\*\* · 90\.0%/);
  assert.match(formatted, /42 ms · 18 input tokens/);
});

test("loadedContextFields sets loadedIsDecision and loadedDecisionType consistently", () => {
  const decisionFields = loadedContextFields({
    is_gguf: true,
    is_decision: true,
    decision_type: "lfm3-d1",
    context_length: 4096,
  });
  assert.equal(decisionFields.loadedIsDecision, true);
  assert.equal(decisionFields.loadedDecisionType, "lfm3-d1");

  const nonDecisionFields = loadedContextFields({
    is_gguf: true,
    is_decision: false,
    decision_type: "granite-switch",
    context_length: 4096,
  });
  assert.equal(nonDecisionFields.loadedIsDecision, false);
  assert.equal(nonDecisionFields.loadedDecisionType, null);
});
