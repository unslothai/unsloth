// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeChatGenerationChunkPayload } = await import(
  "../src/features/chat/api/chat-generation-api.ts"
);

const waiting = {
  type: "admission_status",
  state: "waiting",
  reason: "context_capacity",
  free: 3,
  committed: 29000,
  budget: 35000,
};

test("a persisted admission status frame is re-tagged like the live stream", () => {
  assert.deepEqual(normalizeChatGenerationChunkPayload(waiting), {
    _admissionStatus: waiting,
  });
});

test("the live stream routes admission status off the transcript", () => {
  const src = readSrc("features/chat/api/chat-api.ts");
  assert.match(
    src,
    /parsed\.type === "admission_status"\) \{\s*yield \{ _admissionStatus: parsed \}/,
  );
});

test("the adapter explains a context wait and clears the badge on admission", () => {
  const src = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    src,
    /admissionStatus\.reason === "context_capacity"[\s\S]*?admissionContextCapacity[\s\S]*?admissionServingSlot/,
  );
  assert.match(
    src,
    /runtime\.setToolStatus\(threadKey, null, serverCancel\)/,
  );
});
