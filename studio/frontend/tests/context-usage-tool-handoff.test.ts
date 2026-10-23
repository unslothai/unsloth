// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #12624: a tool loop sends its context as it hands off to a tool, so the meter moves mid-turn.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeChatGenerationChunkPayload } = await import(
  "../src/features/chat/api/chat-generation-api.ts"
);
const { generationChunkCountsTowardTiming } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

const frame = {
  type: "context_usage",
  prompt_tokens: 100,
  completion_tokens: 30,
  context_tokens: 130,
  cached_tokens: 64,
};

test("a persisted context_usage frame is re-tagged like the live stream", () => {
  assert.deepEqual(normalizeChatGenerationChunkPayload(frame), { _contextUsage: frame });
});

test("a context_usage frame is not a decoded chunk for recovered timing", () => {
  assert.equal(generationChunkCountsTowardTiming({ _contextUsage: frame }), false);
});

test("the live stream routes context_usage off the transcript", () => {
  const src = readSrc("features/chat/api/chat-api.ts");
  assert.match(
    src,
    /parsed\.type === "context_usage"\) \{\s*yield \{ _contextUsage: parsed \}/,
  );
});

test("the adapter prices the handoff through the same gated meter write as the turn end", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  const handler = src.slice(src.indexOf(")._contextUsage;"));
  assert.ok(handler.length > 0, "the _contextUsage handler disappeared");
  assert.match(
    handler.slice(0, 1500),
    /publishContextUsage\(\{[\s\S]*?totalTokens: context_tokens,\s*contextTokens: context_tokens,/,
  );
  // The turn-end write shares the helper, so both obey the thread and checkpoint gate.
  assert.equal(src.match(/publishContextUsage\(\{/g)?.length, 2);
  assert.match(
    src,
    /store\.activeThreadId === \(usageThreadKey \?\? activeThreadIdAtRunStart\) &&\s*store\.params\.checkpoint === params\.checkpoint/,
  );
});
