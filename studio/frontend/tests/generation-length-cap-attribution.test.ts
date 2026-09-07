// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { maxTokensIsTheLimit } from "../src/features/chat/api/generation-length.ts";

// Hoisted: biome's useTopLevelRegex flags a literal recompiled per call.
const LOCAL_WINDOW_ARGUMENT =
  /isExternalRequest\s*\n\s*\? null\s*\n\s*: \(runtime\.loadedCustomContextLength \?\?\s*\n\s*runtime\.loadedContextLength \?\?\s*\n\s*\(params\.maxSeqLength \|\| null\)\)/;
// The EDITABLE field must not be what a stop is judged against: the store defines a
// pending context edit as exactly `customContextLength !== loadedCustomContextLength`.
const PENDING_FIELD = /: \(runtime\.customContextLength \?\?/;

test("a cap the prompt left no room for is not the limit that was hit", () => {
  // 4096 window, 3000-token prompt, Max Tokens 2048: generation stops at roughly 1096,
  // well short of the cap. Blaming Max Tokens sends the user to raise a setting that
  // cannot create any room.
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: 4096, promptTokens: 3000 }),
    false,
  );
});

test("a cap the prompt left room for is the limit that was hit", () => {
  assert.equal(
    maxTokensIsTheLimit({ cap: 512, contextLength: 4096, promptTokens: 3000 }),
    true,
  );
});

test("hitting the cap and the context wall together is context-bound", () => {
  // Retargeted. This asserted the cap wins at equality, which is exactly backwards: when
  // promptTokens + cap equals the window, both limits are reached in the same token, so
  // raising Max Tokens creates no room at all and the Context Length remedy is the only
  // one that can work.
  assert.equal(
    maxTokensIsTheLimit({ cap: 1096, contextLength: 4096, promptTokens: 3000 }),
    false,
  );
  // One token of headroom and the cap really is what stopped it.
  assert.equal(
    maxTokensIsTheLimit({ cap: 1095, contextLength: 4096, promptTokens: 3000 }),
    true,
  );
});

test("Max Tokens on Max is never the limit", () => {
  // The backend substitutes the whole context length, so a cap equal to it is
  // indistinguishable from unset, and raising it is impossible either way.
  assert.equal(
    maxTokensIsTheLimit({ cap: 4096, contextLength: 4096, promptTokens: 10 }),
    false,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: null, contextLength: 4096, promptTokens: 10 }),
    false,
  );
});

test("without a prompt count the cap alone decides", () => {
  // What this did before the server's count was read: the safe fallback, not a refusal.
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: 4096, promptTokens: null }),
    true,
  );
});

test("an unknown context length cannot make a cap the limit", () => {
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: null, promptTokens: null }),
    true,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: null, contextLength: null, promptTokens: null }),
    false,
  );
});

test("a local model with no GGUF window still reports one", () => {
  // A safetensors or MLX request on the legacy stream path has neither
  // customContextLength nor loadedContextLength, while params.maxSeqLength IS its
  // effective window and is also where the default Max Tokens comes from. Passing
  // null there makes the window infinite below, so every context-length stop is
  // reported as a Max Tokens stop and the user is told to raise a setting that is
  // already at the model's maximum.
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: null, promptTokens: 3000 }),
    true,
  );
  assert.equal(
    maxTokensIsTheLimit({ cap: 2048, contextLength: 4096, promptTokens: 3000 }),
    false,
  );

  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );
  assert.match(adapter, LOCAL_WINDOW_ARGUMENT);
});

test("a pending Context Length edit does not decide what stopped the generation", () => {
  // Typing 8192 into the field while the model still serves at 4096 would make the
  // 4096 stop look user-imposed, and the advice would be to raise Max Tokens rather
  // than to reload at the larger context.
  const adapter = readFileSync(
    fileURLToPath(
      new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    ),
    "utf8",
  );

  assert.match(adapter, LOCAL_WINDOW_ARGUMENT);
  assert.doesNotMatch(adapter, PENDING_FIELD);
});
