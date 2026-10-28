// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  toolModelNoteTarget,
  toolModelSuffixExtraContent,
  withToolModelSuffix,
} from "../src/features/chat/tool-model-suffix.ts";

import { readSrc } from "./helpers/kit.ts";

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
  // A reconnect that replays the same note keeps it once.
  assert.deepEqual(withToolModelSuffix(twice, "\n\nsecond"), twice);
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

test("the adapter publishes a model note as soon as it lands", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  const start = src.indexOf('toolEvent.type === "tool_model_note"');
  assert.notEqual(start, -1);
  const branch = src.slice(start, src.indexOf("continue;", start));
  // A stop before the next chunk saves only the last yield, so the note must reach one.
  assert.match(branch, /yield \{\s*content: liveAssistantContent\(\)/);
});
