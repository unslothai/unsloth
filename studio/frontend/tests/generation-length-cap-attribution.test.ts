// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { maxTokensIsTheLimit } from "../src/features/chat/api/generation-length.ts";

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
