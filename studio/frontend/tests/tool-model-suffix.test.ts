// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  toolModelNoteTarget,
  toolModelSuffixExtraContent,
  withToolModelSuffix,
} from "../src/features/chat/tool-model-suffix.ts";

test("a model note targets the latest card for a reused backend id", () => {
  const parts = [
    { toolCallId: "call_0:a", backendToolCallId: "call_0" },
    { toolCallId: "call_1:b", backendToolCallId: "call_1" },
    { toolCallId: "call_0:c", backendToolCallId: "call_0" },
  ];
  assert.equal(toolModelNoteTarget(parts, "call_0"), 2);
  assert.equal(toolModelNoteTarget(parts, "call_1:b"), 1);
  assert.equal(toolModelNoteTarget(parts, "missing"), -1);
  assert.equal(toolModelNoteTarget(parts, ""), -1);
});

test("suffixes accumulate in order and keep the rest of the provenance", () => {
  const once = withToolModelSuffix(
    { source: "local", round_id: 3 },
    "\n\nfirst",
  );
  const twice = withToolModelSuffix(once, "\n\nsecond");
  assert.deepEqual(twice, {
    source: "local",
    round_id: 3,
    model_suffix: "\n\nfirst\n\nsecond",
  });
});

test("only a noted card replays extra_content", () => {
  assert.deepEqual(toolModelSuffixExtraContent({ source: "local" }), {});
  assert.deepEqual(toolModelSuffixExtraContent(null), {});
  assert.deepEqual(
    toolModelSuffixExtraContent({ model_suffix: "\n\nUse what you have." }),
    {
      extra_content: { studio_model_suffix: "\n\nUse what you have." },
    },
  );
});

test("empty and non-string suffixes do not create replay metadata", () => {
  for (const model_suffix of ["", null, undefined, false, 0, [], {}]) {
    assert.deepEqual(toolModelSuffixExtraContent({ model_suffix }), {});
  }
});

test("a new suffix replaces invalid legacy metadata without mutating provenance", () => {
  const original = Object.freeze({
    source: "local",
    round_id: 3,
    model_suffix: 7,
  });
  const suffix = "\n\nUse what you have.";
  const updated = withToolModelSuffix(original, suffix);
  assert.deepEqual(updated, { source: "local", round_id: 3, model_suffix: suffix });
  assert.deepEqual(original, { source: "local", round_id: 3, model_suffix: 7 });
  assert.deepEqual(toolModelSuffixExtraContent(updated), {
    extra_content: { studio_model_suffix: suffix },
  });
});
